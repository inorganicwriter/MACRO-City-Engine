from __future__ import annotations

"""Build independent city-level observation panels (road + VIIRS)."""

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import requests
from requests import RequestException

from .city_catalog import load_city_catalog
from .config import TimeRange, load_config
from .gee_city_observed import import_gee_viirs_city_monthly
from .historical_viirs import import_historical_viirs_to_raw
from .utils import DATA_RAW, dump_json, haversine_km

LOGGER = logging.getLogger(__name__)

OVERPASS_MIRRORS: List[str] = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

MAJOR_ROAD_CLASSES = {"motorway", "trunk", "primary", "secondary"}

VIIRS_BASE = "https://gis.ngdc.noaa.gov/arcgis/rest/services/NPP_VIIRS_DNB/Nightly_Radiance/ImageServer"
VIIRS_QUERY = f"{VIIRS_BASE}/query"
VIIRS_SAMPLES = f"{VIIRS_BASE}/getSamples"


@dataclass(frozen=True)
class BuildSummary:
    road_path: str
    viirs_monthly_path: str
    viirs_daily_path: str
    road_rows: int
    viirs_monthly_rows: int
    viirs_daily_rows: int
    viirs_historical_rows: int
    road_source_counts: Dict[str, int]
    viirs_source_counts: Dict[str, int]


def _request_json(
    method: str,
    url: str,
    *,
    timeout: int = 60,
    retries: int = 4,
    backoff: float = 1.0,
    **kwargs: Any,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.request(method=method, url=url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except (RequestException, ValueError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            sleep_s = backoff * (2 ** (attempt - 1))
            time.sleep(sleep_s)
    raise RuntimeError(f"Request failed after {retries} attempts for {url}: {last_error}")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_road_query(lat: float, lon: float, radius_m: int) -> str:
    return f"""
    [out:json][timeout:60];
    (
      way(around:{radius_m},{lat},{lon})["highway"];
    );
    (._;>;);
    out body;
    """.strip()


def _road_metrics_from_overpass_payload(payload: dict, radius_km: float) -> Dict[str, float]:
    elements = payload.get("elements", [])
    node_coords: Dict[int, tuple[float, float]] = {}
    ways: List[dict] = []

    for elem in elements:
        et = elem.get("type")
        if et == "node":
            node_id = elem.get("id")
            lat = elem.get("lat")
            lon = elem.get("lon")
            if node_id is None or lat is None or lon is None:
                continue
            node_coords[int(node_id)] = (float(lat), float(lon))
        elif et == "way":
            ways.append(elem)

    total_len_km = 0.0
    major_len_km = 0.0
    node_degree: Dict[int, int] = {}

    for way in ways:
        node_ids = [int(n) for n in way.get("nodes", []) if n in node_coords]
        if len(node_ids) < 2:
            continue

        length_km = 0.0
        for a, b in zip(node_ids[:-1], node_ids[1:]):
            lat1, lon1 = node_coords[a]
            lat2, lon2 = node_coords[b]
            length_km += haversine_km(lat1, lon1, lat2, lon2)

        total_len_km += length_km
        road_class = str((way.get("tags") or {}).get("highway", "")).strip().lower()
        if road_class in MAJOR_ROAD_CLASSES:
            major_len_km += length_km

        for nid in set(node_ids):
            node_degree[nid] = node_degree.get(nid, 0) + 1

    # Degree>=3 reduces false intersections from segmented two-way chains.
    intersection_count = float(sum(1 for deg in node_degree.values() if deg >= 3))
    area_km2 = math.pi * float(radius_km) * float(radius_km)
    inter_density = intersection_count / max(area_km2, 1e-9)

    return {
        "road_length_km_total": float(total_len_km),
        "arterial_share": float(major_len_km / max(total_len_km, 1e-9)),
        "intersection_density": float(inter_density),
    }


def _city_overpass_endpoints(config_overpass_url: str, city: pd.Series) -> List[str]:
    del city
    endpoints: List[str] = []
    preferred = [
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        str(config_overpass_url).strip(),
    ]
    for url in preferred:
        if url and url not in endpoints:
            endpoints.append(url)
    return endpoints


def _build_road_snapshot(
    cities: pd.DataFrame,
    config_overpass_url: str,
    *,
    radius_m: int = 2000,
    use_cache: bool = True,
    sleep_s: float = 0.25,
    endpoint_timeout_s: int = 12,
) -> pd.DataFrame:
    snapshot_path = DATA_RAW / "city_road_network_snapshot.csv"
    cached = pd.DataFrame()
    if use_cache and snapshot_path.exists():
        try:
            cached = pd.read_csv(snapshot_path)
        except Exception:  # noqa: BLE001
            cached = pd.DataFrame()

    done_ids: set[str] = set()
    if not cached.empty and "city_id" in cached.columns:
        if "road_source" in cached.columns:
            done_mask = cached["road_source"].astype(str) != "missing"
            done_ids = set(cached.loc[done_mask, "city_id"].astype(str).unique().tolist())
        else:
            done_ids = set(cached["city_id"].astype(str).unique().tolist())

    rows: List[dict] = []
    total = int(len(cities))

    for idx, city in enumerate(cities.to_dict(orient="records"), start=1):
        city_id = str(city["city_id"])
        if city_id in done_ids:
            continue

        if idx > 1:
            time.sleep(max(0.0, float(sleep_s)))

        query = _build_road_query(float(city["latitude"]), float(city["longitude"]), int(radius_m))
        endpoints = _city_overpass_endpoints(config_overpass_url, pd.Series(city))
        endpoints = endpoints[:3]
        metrics: Dict[str, float] | None = None
        error_msg: str | None = None
        radius_candidates: List[int] = []
        for rr in [int(radius_m), int(round(radius_m * 0.75)), int(round(radius_m * 0.5))]:
            rr = max(800, int(rr))
            if rr not in radius_candidates:
                radius_candidates.append(rr)
        used_radius = int(radius_candidates[0])

        for rr in radius_candidates:
            query = _build_road_query(float(city["latitude"]), float(city["longitude"]), int(rr))
            for ep in endpoints:
                try:
                    payload = _request_json(
                        "POST",
                        ep,
                        timeout=max(8, int(endpoint_timeout_s)),
                        retries=1,
                        backoff=1.0,
                        data={"data": query},
                    )
                    metrics = _road_metrics_from_overpass_payload(payload, radius_km=float(rr) / 1000.0)
                    error_msg = None
                    used_radius = int(rr)
                    break
                except Exception as exc:  # noqa: BLE001
                    error_msg = str(exc)
                    time.sleep(0.25)
                    continue
            if metrics is not None:
                break

        if metrics is None:
            row = {
                "city_id": city_id,
                "city_name": city["city_name"],
                "road_length_km_total": np.nan,
                "arterial_share": np.nan,
                "intersection_density": np.nan,
                "road_source": "missing",
                "road_error": (error_msg or "")[:300],
                "snapshot_utc": _utc_now_iso(),
                "radius_m": int(used_radius),
            }
            LOGGER.warning("Road fetch failed for %s (%s): %s", city["city_name"], city_id, error_msg)
        else:
            row = {
                "city_id": city_id,
                "city_name": city["city_name"],
                **metrics,
                "road_source": "osm_overpass_snapshot",
                "road_error": "",
                "snapshot_utc": _utc_now_iso(),
                "radius_m": int(used_radius),
            }

        rows.append(row)
        LOGGER.info("Road city %s/%s done: %s (%s)", idx, total, city["city_name"], row["road_source"])

        if idx % 1 == 0 or idx == total:
            LOGGER.info("Road snapshot progress: %s/%s", idx, total)
            merged = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
            merged = merged.drop_duplicates(subset=["city_id"], keep="last")
            merged.to_csv(snapshot_path, index=False)

    out = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
    out = out.drop_duplicates(subset=["city_id"], keep="last")
    out.to_csv(snapshot_path, index=False)
    return out


def _expand_road_snapshot_yearly(snapshot: pd.DataFrame, years: Iterable[int]) -> pd.DataFrame:
    rows: List[dict] = []
    for r in snapshot.to_dict(orient="records"):
        for y in years:
            rows.append(
                {
                    "city_id": r["city_id"],
                    "year": int(y),
                    "road_length_km_total": r.get("road_length_km_total"),
                    "arterial_share": r.get("arterial_share"),
                    "intersection_density": r.get("intersection_density"),
                    "road_source": r.get("road_source", "missing"),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["city_id", "year"]).reset_index(drop=True)


def _fetch_viirs_catalog(max_records: int = 1000) -> pd.DataFrame:
    payload = _request_json(
        "GET",
        VIIRS_QUERY,
        timeout=30,
        retries=3,
        params={
            "f": "pjson",
            "where": "1=1",
            "outFields": "OBJECTID,Date,Name",
            "orderByFields": "Date ASC",
            "resultRecordCount": int(max_records),
            "returnGeometry": "false",
        },
    )
    feats = payload.get("features", [])
    rows: List[dict] = []
    for f in feats:
        a = f.get("attributes", {})
        oid = a.get("OBJECTID")
        date_ms = a.get("Date")
        if oid is None or date_ms is None:
            continue
        rows.append(
            {
                "raster_id": int(oid),
                "date_ms": int(date_ms),
                "name": str(a.get("Name", "")),
            }
        )

    out = pd.DataFrame(rows).drop_duplicates(subset=["raster_id"]).sort_values("date_ms")
    if out.empty:
        raise RuntimeError("VIIRS catalog is empty.")
    LOGGER.info("VIIRS catalog loaded: %s rasters", len(out))
    return out


def _city_sample_points(cities: pd.DataFrame, step_deg: float = 0.01) -> List[dict]:
    # 5-point cross per city reduces one-pixel geolocation instability.
    offsets = [(0.0, 0.0), (step_deg, 0.0), (-step_deg, 0.0), (0.0, step_deg), (0.0, -step_deg)]
    points: List[dict] = []
    for row in cities.to_dict(orient="records"):
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        for i, (dlat, dlon) in enumerate(offsets):
            points.append(
                {
                    "city_id": str(row["city_id"]),
                    "city_name": str(row["city_name"]),
                    "sample_id": int(i),
                    "lat": lat + dlat,
                    "lon": lon + dlon,
                }
            )
    return points


def _chunk(seq: List[dict], size: int) -> Iterable[List[dict]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _sample_viirs_one_raster(
    raster_id: int,
    point_records: List[dict],
    *,
    batch_points: int = 450,
) -> Dict[str, float]:
    city_vals: Dict[str, List[float]] = {}
    mosaic_rule = {"mosaicMethod": "esriMosaicLockRaster", "lockRasterIds": [int(raster_id)]}

    for batch in _chunk(point_records, size=int(batch_points)):
        geometry = {
            "points": [[float(p["lon"]), float(p["lat"])] for p in batch],
            "spatialReference": {"wkid": 4326},
        }
        payload = _request_json(
            "POST",
            VIIRS_SAMPLES,
            timeout=30,
            retries=2,
            backoff=1.0,
            data={
                "f": "pjson",
                "geometryType": "esriGeometryMultipoint",
                "geometry": json.dumps(geometry, separators=(",", ":")),
                "returnGeometry": "false",
                "mosaicRule": json.dumps(mosaic_rule, separators=(",", ":")),
            },
        )
        samples = payload.get("samples", [])
        for s in samples:
            loc_id = s.get("locationId")
            if loc_id is None:
                continue
            i = int(loc_id)
            if i < 0 or i >= len(batch):
                continue
            city_id = str(batch[i]["city_id"])
            try:
                v = float(s.get("value"))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v) or v <= -900.0:
                continue
            city_vals.setdefault(city_id, []).append(v)

    # Per-raster city robust aggregation.
    out: Dict[str, float] = {}
    for city_id, vals in city_vals.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        out[city_id] = float(np.median(arr))
    return out


def _build_viirs_daily_panel(cities: pd.DataFrame, *, use_cache: bool = True) -> pd.DataFrame:
    daily_path = DATA_RAW / "viirs_city_daily_observed.csv"
    cached = pd.DataFrame()
    if use_cache and daily_path.exists():
        try:
            cached = pd.read_csv(daily_path)
        except Exception:  # noqa: BLE001
            cached = pd.DataFrame()

    done_pairs: set[tuple[str, int]] = set()
    if not cached.empty and {"city_id", "raster_id"}.issubset(cached.columns):
        tmp = cached[["city_id", "raster_id"]].copy()
        tmp["city_id"] = tmp["city_id"].astype(str)
        tmp["raster_id"] = pd.to_numeric(tmp["raster_id"], errors="coerce").fillna(-1).astype(int)
        done_pairs = set(zip(tmp["city_id"], tmp["raster_id"]))

    catalog = _fetch_viirs_catalog()
    points = _city_sample_points(cities)
    LOGGER.info("VIIRS sampling points prepared: %s points (%s cities)", len(points), cities['city_id'].nunique())
    city_ids = cities["city_id"].astype(str).tolist()
    new_rows: List[dict] = []

    total_r = int(len(catalog))
    for i, rec in enumerate(catalog.to_dict(orient="records"), start=1):
        rid = int(rec["raster_id"])
        date_ms = int(rec["date_ms"])
        date_utc = datetime.fromtimestamp(date_ms / 1000.0, tz=timezone.utc)

        missing_cities = [cid for cid in city_ids if (cid, rid) not in done_pairs]
        if not missing_cities:
            continue

        # Keep only points from cities missing this raster.
        points_subset = [p for p in points if p["city_id"] in set(missing_cities)]
        LOGGER.info(
            "VIIRS raster %s/%s sampling: raster_id=%s date=%s cities=%s",
            i,
            total_r,
            rid,
            date_utc.date(),
            len(missing_cities),
        )
        try:
            sampled = _sample_viirs_one_raster(rid, points_subset)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("VIIRS sample failed for raster=%s date=%s: %s", rid, date_utc.date(), exc)
            sampled = {}

        for city_id in missing_cities:
            val = sampled.get(city_id, np.nan)
            new_rows.append(
                {
                    "city_id": city_id,
                    "raster_id": rid,
                    "date": date_utc.date().isoformat(),
                    "year": int(date_utc.year),
                    "month": int(date_utc.month),
                    "radiance": float(val) if np.isfinite(val) else np.nan,
                    "viirs_source": "noaa_viirs_nightly_radiance",
                }
            )

        if i % 8 == 0 or i == total_r:
            LOGGER.info("VIIRS sampling progress: raster %s/%s", i, total_r)
            merged = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
            merged = merged.drop_duplicates(subset=["city_id", "raster_id"], keep="last")
            merged.to_csv(daily_path, index=False)

        time.sleep(0.05)

    out = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
    out = out.drop_duplicates(subset=["city_id", "raster_id"], keep="last")
    out.to_csv(daily_path, index=False)
    return out


def _daily_to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy()
    work["city_id"] = work["city_id"].astype(str)
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["month"] = pd.to_numeric(work["month"], errors="coerce")
    work["radiance"] = pd.to_numeric(work["radiance"], errors="coerce")
    work = work.dropna(subset=["year", "month"]).copy()
    work["year"] = work["year"].astype(int)
    work["month"] = work["month"].astype(int)
    out = (
        work.groupby(["city_id", "year", "month"], as_index=False)
        .agg(
            radiance=("radiance", "mean"),
            viirs_source=("viirs_source", "first"),
        )
        .sort_values(["city_id", "year", "month"])
    )
    # Optional column kept for compatibility with downstream aliases.
    out["lit_area_km2"] = np.nan
    return out


def build_observed_city_signals(
    *,
    max_cities: int = 295,
    start_year: int = 2015,
    end_year: int = 2025,
    use_cache: bool = True,
    road_radius_m: int = 2000,
    build_road: bool = True,
    build_viirs: bool = True,
    historical_viirs_root: str | None = None,
    gee_viirs_csv: str | None = None,
) -> BuildSummary:
    cfg = load_config()
    cities = load_city_catalog(max_cities=int(max_cities)).copy()
    years = list(range(int(start_year), int(end_year) + 1))

    road_path = DATA_RAW / "city_road_network_yearly.csv"
    if build_road:
        road_snapshot = _build_road_snapshot(
            cities,
            cfg.api.overpass_url,
            radius_m=int(road_radius_m),
            use_cache=bool(use_cache),
        )
        road_yearly = _expand_road_snapshot_yearly(road_snapshot, years=years)
        road_yearly.to_csv(road_path, index=False)
    else:
        road_yearly = pd.read_csv(road_path) if road_path.exists() else pd.DataFrame(columns=["road_source"])

    viirs_monthly_path = DATA_RAW / "viirs_city_monthly.csv"
    viirs_daily_path = DATA_RAW / "viirs_city_daily_observed.csv"
    if build_viirs:
        viirs_daily = _build_viirs_daily_panel(cities, use_cache=bool(use_cache))
        viirs_monthly = _daily_to_monthly(viirs_daily)
        viirs_monthly.to_csv(viirs_monthly_path, index=False)
    else:
        viirs_daily = pd.read_csv(viirs_daily_path) if viirs_daily_path.exists() else pd.DataFrame(columns=["viirs_source"])
        viirs_monthly = pd.read_csv(viirs_monthly_path) if viirs_monthly_path.exists() else pd.DataFrame(columns=["viirs_source"])

    hist_rows = 0
    if historical_viirs_root:
        hist_summary = import_historical_viirs_to_raw(
            source_root=historical_viirs_root,
            max_cities=int(max_cities),
            start_year=int(start_year),
            end_year=int(end_year),
            merge_existing=True,
            output_path=viirs_monthly_path,
        )
        hist_rows = int(hist_summary.get("rows", 0))
        viirs_monthly = pd.read_csv(viirs_monthly_path) if viirs_monthly_path.exists() else viirs_monthly
    if gee_viirs_csv:
        gee_summary = import_gee_viirs_city_monthly(
            source_path=gee_viirs_csv,
            output_path=viirs_monthly_path,
            merge_existing=True,
            max_cities=int(max_cities),
        )
        hist_rows += int(gee_summary.get("rows_imported", 0))
        viirs_monthly = pd.read_csv(viirs_monthly_path) if viirs_monthly_path.exists() else viirs_monthly

    summary = BuildSummary(
        road_path=str(road_path),
        viirs_monthly_path=str(viirs_monthly_path),
        viirs_daily_path=str(viirs_daily_path),
        road_rows=int(len(road_yearly)),
        viirs_monthly_rows=int(len(viirs_monthly)),
        viirs_daily_rows=int(len(viirs_daily)),
        viirs_historical_rows=int(hist_rows),
        road_source_counts={
            str(k): int(v)
            for k, v in road_yearly.get("road_source", pd.Series(dtype=str)).value_counts(dropna=False).items()
        },
        viirs_source_counts={
            str(k): int(v)
            for k, v in viirs_monthly.get("viirs_source", pd.Series(dtype=str)).value_counts(dropna=False).items()
        },
    )

    summary_path = DATA_RAW / "observed_city_signals_summary.json"
    dump_json(
        summary_path,
        {
            "generated_at_utc": _utc_now_iso(),
            "max_cities": int(max_cities),
            "time_range": [int(start_year), int(end_year)],
            "road_radius_m": int(road_radius_m),
            "build_road": bool(build_road),
            "build_viirs": bool(build_viirs),
            "road_path": summary.road_path,
            "viirs_monthly_path": summary.viirs_monthly_path,
            "viirs_daily_path": summary.viirs_daily_path,
            "road_rows": summary.road_rows,
            "viirs_monthly_rows": summary.viirs_monthly_rows,
            "viirs_daily_rows": summary.viirs_daily_rows,
            "viirs_historical_rows": summary.viirs_historical_rows,
            "road_source_counts": summary.road_source_counts,
            "viirs_source_counts": summary.viirs_source_counts,
            "historical_viirs_root": str(historical_viirs_root) if historical_viirs_root else None,
            "gee_viirs_csv": str(gee_viirs_csv) if gee_viirs_csv else None,
        },
    )
    return summary


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build observed road + VIIRS city panels.")
    parser.add_argument("--max-cities", type=int, default=295)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--road-radius-m", type=int, default=2000)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--skip-road", action="store_true")
    parser.add_argument("--skip-viirs", action="store_true")
    parser.add_argument("--historical-viirs-root", type=str, default=None)
    parser.add_argument("--gee-viirs-csv", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    summary = build_observed_city_signals(
        max_cities=int(args.max_cities),
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        use_cache=not bool(args.no_cache),
        road_radius_m=int(args.road_radius_m),
        build_road=not bool(args.skip_road),
        build_viirs=not bool(args.skip_viirs),
        historical_viirs_root=args.historical_viirs_root,
        gee_viirs_csv=args.gee_viirs_csv,
    )
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
