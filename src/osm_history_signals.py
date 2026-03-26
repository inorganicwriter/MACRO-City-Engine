from __future__ import annotations

"""Build city-year OSM historical signals from ohsome API."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import requests
from requests import RequestException

from .city_catalog import load_city_catalog
from .utils import DATA_RAW, dump_json

LOGGER = logging.getLogger(__name__)

OHSOME_BASE = "https://api.ohsome.org/v1"

OSM_HIST_LEVEL_TO_YOY: Dict[str, str] = {
    "osm_hist_road_length_m": "osm_hist_road_yoy",
    "osm_hist_building_count": "osm_hist_building_yoy",
    "osm_hist_poi_count": "osm_hist_poi_yoy",
    "osm_hist_poi_food_count": "osm_hist_poi_food_yoy",
    "osm_hist_poi_retail_count": "osm_hist_poi_retail_yoy",
    "osm_hist_poi_nightlife_count": "osm_hist_poi_nightlife_yoy",
}

POI_YEAR_LEVEL_COLS: List[str] = [
    "amenity_count",
    "shop_count",
    "office_count",
    "leisure_count",
    "transport_count",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _request_json(
    endpoint: str,
    *,
    data: Dict[str, str],
    timeout: int = 30,
    retries: int = 3,
    backoff: float = 1.0,
) -> dict:
    url = f"{OHSOME_BASE}/{endpoint.lstrip('/')}"
    last_err: Exception | None = None
    for i in range(1, retries + 1):
        try:
            r = requests.post(url, data=data, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (RequestException, ValueError) as exc:
            last_err = exc
            if i >= retries:
                break
            time.sleep(backoff * (2 ** (i - 1)))
    raise RuntimeError(f"ohsome request failed ({endpoint}): {last_err}")


def _bbox_str(lat: float, lon: float, half_size_deg: float) -> str:
    west = float(lon) - float(half_size_deg)
    east = float(lon) + float(half_size_deg)
    south = float(lat) - float(half_size_deg)
    north = float(lat) + float(half_size_deg)
    return f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}"


def _time_range_str(start_year: int, end_year: int) -> str:
    # Inclusive yearly stamps at Jan-01 for each year (e.g., 2015..2025).
    return f"{int(start_year)}-01-01/{int(end_year)}-01-01/P1Y"


def _year_value_map(payload: dict) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for row in payload.get("result", []):
        ts = str(row.get("timestamp", ""))
        val = row.get("value")
        if len(ts) < 4:
            continue
        try:
            year = int(ts[:4])
            out[year] = float(val) if val is not None else np.nan
        except Exception:  # noqa: BLE001
            continue
    return out


def _multi_tag_filter(*, key: str, values: Iterable[str], element_types: Iterable[str] = ("node", "way")) -> str:
    terms: List[str] = []
    vals = [str(v).strip() for v in values if str(v).strip()]
    for et in element_types:
        etag = str(et).strip()
        if not etag:
            continue
        for v in vals:
            terms.append(f"(type:{etag} and {key}={v})")
    return " or ".join(terms)


def _fetch_city_history_series(
    *,
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    half_size_deg: float,
) -> Dict[str, Dict[int, float]]:
    bboxes = _bbox_str(lat, lon, half_size_deg)
    tparam = _time_range_str(start_year, end_year)
    queries: Dict[str, tuple[str, str]] = {
        "osm_hist_road_length_m": ("elements/length", "type:way and highway=*"),
        "osm_hist_building_count": ("elements/count", "type:way and building=*"),
        "osm_hist_poi_count": (
            "elements/count",
            (
                "(type:node and amenity=*) or (type:way and amenity=*) or "
                "(type:node and shop=*) or (type:way and shop=*) or "
                "(type:node and office=*) or (type:way and office=*) or "
                "(type:node and leisure=*) or (type:way and leisure=*)"
            ),
        ),
        "osm_hist_poi_food_count": (
            "elements/count",
            _multi_tag_filter(
                key="amenity",
                values=["restaurant", "fast_food", "cafe", "food_court", "ice_cream"],
            ),
        ),
        "osm_hist_poi_retail_count": ("elements/count", "(type:node and shop=*) or (type:way and shop=*)"),
        "osm_hist_poi_nightlife_count": (
            "elements/count",
            _multi_tag_filter(
                key="amenity",
                values=["bar", "pub", "nightclub"],
            ),
        ),
    }

    def _run(endpoint: str, filt: str) -> Dict[int, float]:
        payload = _request_json(
            endpoint,
            data={
                "bboxes": bboxes,
                "time": tparam,
                "filter": filt,
            },
        )
        return _year_value_map(payload)

    out: Dict[str, Dict[int, float]] = {}
    with ThreadPoolExecutor(max_workers=min(6, len(queries))) as pool:
        future_map = {
            pool.submit(_run, endpoint, filt): key for key, (endpoint, filt) in queries.items()
        }
        for future in as_completed(future_map):
            key = future_map[future]
            out[key] = future.result()
    return out


def _full_years(start_year: int, end_year: int) -> List[int]:
    return list(range(int(start_year), int(end_year) + 1))


def _done_city_ids(cached: pd.DataFrame, years: Iterable[int]) -> set[str]:
    if cached.empty:
        return set()
    required_cols = list(OSM_HIST_LEVEL_TO_YOY.keys())
    if not set(required_cols).issubset(set(cached.columns)):
        # New schema rollout: force refresh when old cache lacks new observed signals.
        return set()
    req = set(int(y) for y in years)
    out: set[str] = set()
    for city_id, grp in cached.groupby("city_id"):
        yset = set(pd.to_numeric(grp["year"], errors="coerce").dropna().astype(int).tolist())
        source = grp.get("osm_hist_source", pd.Series([], dtype=str)).astype(str)
        ok_source = bool((source == "ohsome_api").all())
        has_signal = True
        for col in required_cols:
            if not pd.to_numeric(grp[col], errors="coerce").notna().any():
                has_signal = False
                break
        if req.issubset(yset) and ok_source and has_signal:
            out.add(str(city_id))
    return out


def _done_city_ids_poi_yearly(cached: pd.DataFrame, years: Iterable[int]) -> set[str]:
    if cached.empty:
        return set()
    required_cols = POI_YEAR_LEVEL_COLS + ["poi_total", "poi_diversity", "poi_temporal_source"]
    if not set(required_cols).issubset(set(cached.columns)):
        return set()
    req = set(int(y) for y in years)
    out: set[str] = set()
    for city_id, grp in cached.groupby("city_id"):
        yset = set(pd.to_numeric(grp["year"], errors="coerce").dropna().astype(int).tolist())
        src = grp["poi_temporal_source"].fillna("missing").astype(str).str.lower()
        ok_source = bool(src.str.contains("ohsome").all())
        has_signal = True
        for col in POI_YEAR_LEVEL_COLS + ["poi_total"]:
            if not pd.to_numeric(grp[col], errors="coerce").notna().any():
                has_signal = False
                break
        if req.issubset(yset) and ok_source and has_signal:
            out.add(str(city_id))
    return out


def _fetch_city_poi_yearly_series(
    *,
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    half_size_deg: float,
) -> Dict[str, Dict[int, float]]:
    bboxes = _bbox_str(lat, lon, half_size_deg)
    tparam = _time_range_str(start_year, end_year)
    queries: Dict[str, tuple[str, str]] = {
        "amenity_count": ("elements/count", "(type:node and amenity=*) or (type:way and amenity=*)"),
        "shop_count": ("elements/count", "(type:node and shop=*) or (type:way and shop=*)"),
        "office_count": ("elements/count", "(type:node and office=*) or (type:way and office=*)"),
        "leisure_count": ("elements/count", "(type:node and leisure=*) or (type:way and leisure=*)"),
        "transport_count": (
            "elements/count",
            (
                "(type:node and public_transport=*) or (type:way and public_transport=*) or "
                "(type:node and railway=station) or (type:way and railway=station)"
            ),
        ),
    }

    def _run(endpoint: str, filt: str) -> Dict[int, float]:
        payload = _request_json(
            endpoint,
            data={
                "bboxes": bboxes,
                "time": tparam,
                "filter": filt,
            },
        )
        return _year_value_map(payload)

    out: Dict[str, Dict[int, float]] = {}
    with ThreadPoolExecutor(max_workers=min(5, len(queries))) as pool:
        future_map = {pool.submit(_run, endpoint, filt): key for key, (endpoint, filt) in queries.items()}
        for future in as_completed(future_map):
            key = future_map[future]
            out[key] = future.result()
    return out


def build_city_osm_history_yearly(
    *,
    max_cities: int = 295,
    start_year: int = 2015,
    end_year: int = 2025,
    use_cache: bool = True,
    half_size_deg: float = 0.03,
    sleep_s: float = 0.1,
) -> pd.DataFrame:
    cities = load_city_catalog(max_cities=int(max_cities)).copy()
    years = _full_years(start_year, end_year)
    out_path = DATA_RAW / "city_osm_history_yearly.csv"

    cached = pd.DataFrame()
    if use_cache and out_path.exists():
        try:
            cached = pd.read_csv(out_path)
        except Exception:  # noqa: BLE001
            cached = pd.DataFrame()

    done_ids = _done_city_ids(cached, years)
    new_rows: List[dict] = []
    total = int(len(cities))

    for i, city in enumerate(cities.to_dict(orient="records"), start=1):
        city_id = str(city["city_id"])
        if city_id in done_ids:
            continue

        try:
            series = _fetch_city_history_series(
                lat=float(city["latitude"]),
                lon=float(city["longitude"]),
                start_year=int(start_year),
                end_year=int(end_year),
                half_size_deg=float(half_size_deg),
            )
            src = "ohsome_api"
            err = ""
        except Exception as exc:  # noqa: BLE001
            series = {k: {} for k in OSM_HIST_LEVEL_TO_YOY}
            src = "missing"
            err = str(exc)[:260]
            LOGGER.warning("ohsome history failed for %s (%s): %s", city["city_name"], city_id, exc)

        for y in years:
            new_rows.append(
                {
                    "city_id": city_id,
                    "year": int(y),
                    "osm_hist_road_length_m": series["osm_hist_road_length_m"].get(int(y), np.nan),
                    "osm_hist_building_count": series["osm_hist_building_count"].get(int(y), np.nan),
                    "osm_hist_poi_count": series["osm_hist_poi_count"].get(int(y), np.nan),
                    "osm_hist_poi_food_count": series["osm_hist_poi_food_count"].get(int(y), np.nan),
                    "osm_hist_poi_retail_count": series["osm_hist_poi_retail_count"].get(int(y), np.nan),
                    "osm_hist_poi_nightlife_count": series["osm_hist_poi_nightlife_count"].get(int(y), np.nan),
                    "osm_hist_source": src,
                    "osm_hist_error": err,
                    "snapshot_utc": _utc_now_iso(),
                }
            )

        LOGGER.info("OSM-history city %s/%s done: %s (%s)", i, total, city["city_name"], src)

        merged = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
        merged = merged.sort_values(["city_id", "year", "snapshot_utc"]).drop_duplicates(["city_id", "year"], keep="last")
        # Growth fields computed after dedupe.
        for level_col, yoy_col in OSM_HIST_LEVEL_TO_YOY.items():
            merged[yoy_col] = pd.to_numeric(merged[level_col], errors="coerce")
            merged[yoy_col] = merged.groupby("city_id")[yoy_col].diff()
        merged.to_csv(out_path, index=False)

        if float(sleep_s) > 0:
            time.sleep(float(sleep_s))

    out = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    summary = {
        "generated_at_utc": _utc_now_iso(),
        "path": str(out_path),
        "rows": int(len(out)),
        "cities": int(out["city_id"].nunique()) if "city_id" in out.columns else 0,
        "year_range": [int(start_year), int(end_year)],
        "source_counts": out.get("osm_hist_source", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
    }
    dump_json(DATA_RAW / "city_osm_history_summary.json", summary)
    return out


def build_city_poi_yearly_from_ohsome(
    *,
    max_cities: int = 295,
    start_year: int = 2015,
    end_year: int = 2025,
    use_cache: bool = True,
    half_size_deg: float = 0.03,
    sleep_s: float = 0.1,
) -> pd.DataFrame:
    cities = load_city_catalog(max_cities=int(max_cities)).copy()
    years = _full_years(start_year, end_year)
    out_path = DATA_RAW / "city_poi_features_yearly.csv"

    cached = pd.DataFrame()
    if use_cache and out_path.exists():
        try:
            cached = pd.read_csv(out_path)
        except Exception:  # noqa: BLE001
            cached = pd.DataFrame()

    done_ids = _done_city_ids_poi_yearly(cached, years)
    new_rows: List[dict] = []
    total = int(len(cities))

    for i, city in enumerate(cities.to_dict(orient="records"), start=1):
        city_id = str(city["city_id"])
        if city_id in done_ids:
            continue

        try:
            series = _fetch_city_poi_yearly_series(
                lat=float(city["latitude"]),
                lon=float(city["longitude"]),
                start_year=int(start_year),
                end_year=int(end_year),
                half_size_deg=float(half_size_deg),
            )
            src = "osm"
            temporal_src = "ohsome_yearly_panel"
        except Exception as exc:  # noqa: BLE001
            series = {k: {} for k in POI_YEAR_LEVEL_COLS}
            src = "missing"
            temporal_src = "missing"
            LOGGER.warning("ohsome poi-yearly failed for %s (%s): %s", city["city_name"], city_id, exc)

        for y in years:
            counts = {k: float(series[k].get(int(y), np.nan)) for k in POI_YEAR_LEVEL_COLS}
            total_count = float(
                np.nansum([counts["amenity_count"], counts["shop_count"], counts["office_count"], counts["leisure_count"], counts["transport_count"]])
            ) if any(np.isfinite(v) for v in counts.values()) else np.nan
            probs = np.array(
                [counts["amenity_count"], counts["shop_count"], counts["office_count"], counts["leisure_count"]],
                dtype=float,
            )
            probs = np.where(np.isfinite(probs), np.maximum(probs, 0.0), np.nan)
            if np.isfinite(probs).sum() >= 1 and np.nansum(probs) > 0:
                probs = probs / np.nansum(probs)
                shannon = float(-(probs[np.isfinite(probs) & (probs > 0)] * np.log(probs[np.isfinite(probs) & (probs > 0)])).sum())
            else:
                shannon = float("nan")

            new_rows.append(
                {
                    "city_id": city_id,
                    "city_name": city["city_name"],
                    "year": int(y),
                    "amenity_count": counts["amenity_count"],
                    "shop_count": counts["shop_count"],
                    "office_count": counts["office_count"],
                    "leisure_count": counts["leisure_count"],
                    "transport_count": counts["transport_count"],
                    "poi_total": total_count,
                    "poi_diversity": shannon,
                    "poi_source": src,
                    "poi_temporal_source": temporal_src,
                    "poi_backcast_scale": 1.0 if src == "osm" else np.nan,
                    "has_poi_historical_support": 1 if src == "osm" else 0,
                    "has_poi_observation": 1 if src == "osm" else 0,
                }
            )

        LOGGER.info("POI-yearly city %s/%s done: %s (%s)", i, total, city["city_name"], temporal_src)

        merged = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
        merged = merged.sort_values(["city_id", "year"]).drop_duplicates(["city_id", "year"], keep="last")
        merged.to_csv(out_path, index=False)

        if float(sleep_s) > 0:
            time.sleep(float(sleep_s))

    out = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    summary = {
        "generated_at_utc": _utc_now_iso(),
        "path": str(out_path),
        "rows": int(len(out)),
        "cities": int(out["city_id"].nunique()) if "city_id" in out.columns else 0,
        "year_range": [int(start_year), int(end_year)],
        "poi_source_counts": out.get("poi_source", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "poi_temporal_source_counts": out.get("poi_temporal_source", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
    }
    dump_json(DATA_RAW / "city_poi_yearly_summary.json", summary)
    return out


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build city-year OSM history panel via ohsome.")
    parser.add_argument("--max-cities", type=int, default=295)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--half-size-deg", type=float, default=0.03, help="Half-size of city bbox in degrees.")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    out = build_city_osm_history_yearly(
        max_cities=int(args.max_cities),
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        use_cache=not bool(args.no_cache),
        half_size_deg=float(args.half_size_deg),
    )
    print(
        json.dumps(
            {
                "rows": int(len(out)),
                "cities": int(out["city_id"].nunique()) if "city_id" in out.columns else 0,
                "source_counts": out.get("osm_hist_source", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
                "path": str(DATA_RAW / "city_osm_history_yearly.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    cli()
