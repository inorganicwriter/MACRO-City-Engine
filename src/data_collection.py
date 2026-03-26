from __future__ import annotations

"""Data ingestion, fallback synthesis, and panel construction."""

import logging
import math
from dataclasses import asdict
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from requests import RequestException

from .config import ProjectConfig
from .utils import DATA_PROCESSED, DATA_RAW, dump_json, minmax_scale, point_line_distance_km

LOGGER = logging.getLogger(__name__)
POI_CATEGORIES = ("amenity", "shop", "office", "leisure")


def build_grid(config: ProjectConfig) -> pd.DataFrame:
    """Build fixed-size spatial cells over the configured bounding box."""
    bbox = config.bbox
    cell_km = config.cell_size_km

    lat_step = cell_km / 111.0
    mean_lat = (bbox.south + bbox.north) / 2.0
    lon_step = cell_km / (111.0 * math.cos(math.radians(mean_lat)))

    if lat_step <= 0 or lon_step <= 0:
        msg = "Invalid cell size produced non-positive coordinate step."
        raise ValueError(msg)

    rows: List[dict] = []
    lat_vals = np.arange(bbox.south, bbox.north, lat_step)
    lon_vals = np.arange(bbox.west, bbox.east, lon_step)

    for i, lat0 in enumerate(lat_vals):
        for j, lon0 in enumerate(lon_vals):
            lat1 = min(lat0 + lat_step, bbox.north)
            lon1 = min(lon0 + lon_step, bbox.east)
            rows.append(
                {
                    "cell_id": f"C{i:03d}_{j:03d}",
                    "row": i,
                    "col": j,
                    "lat_min": lat0,
                    "lat_max": lat1,
                    "lon_min": lon0,
                    "lon_max": lon1,
                    "lat_center": (lat0 + lat1) / 2.0,
                    "lon_center": (lon0 + lon1) / 2.0,
                }
            )

    grid = pd.DataFrame(rows)
    grid.to_csv(DATA_RAW / "grid_cells.csv", index=False)
    LOGGER.info("Grid built: %s cells", len(grid))
    return grid


def _overpass_query(bbox: Tuple[float, float, float, float]) -> str:
    south, west, north, east = bbox
    return f"""
    [out:json][timeout:120];
    (
      node["amenity"]({south},{west},{north},{east});
      node["shop"]({south},{west},{north},{east});
      node["office"]({south},{west},{north},{east});
      node["leisure"]({south},{west},{north},{east});
    );
    out body;
    """.strip()


def _classify_poi(tags: Dict[str, str]) -> str:
    if "shop" in tags:
        return "shop"
    if "office" in tags:
        return "office"
    if "leisure" in tags:
        return "leisure"
    return "amenity"


def fetch_osm_poi(config: ProjectConfig) -> pd.DataFrame:
    """Download POI points from Overpass API."""
    bbox = (config.bbox.south, config.bbox.west, config.bbox.north, config.bbox.east)
    query = _overpass_query(bbox)

    try:
        response = requests.post(config.api.overpass_url, data={"data": query}, timeout=180)
        response.raise_for_status()
        payload = response.json()
    except RequestException as exc:
        msg = f"OSM request failed: {exc}"
        raise RuntimeError(msg) from exc
    except ValueError as exc:
        msg = "OSM response is not valid JSON."
        raise RuntimeError(msg) from exc

    rows: List[dict] = []
    for elem in payload.get("elements", []):
        lat = elem.get("lat")
        lon = elem.get("lon")
        tags = elem.get("tags", {})
        if lat is None or lon is None:
            continue
        rows.append(
            {
                "lat": float(lat),
                "lon": float(lon),
                "category": _classify_poi(tags),
                "name": tags.get("name", ""),
            }
        )

    poi = pd.DataFrame(rows)
    if poi.empty:
        raise RuntimeError("OSM POI query returned empty result.")

    poi.to_csv(DATA_RAW / "poi_points.csv", index=False)
    LOGGER.info("OSM POI collected: %s rows", len(poi))
    return poi


def fetch_open_meteo_daily(config: ProjectConfig) -> pd.DataFrame:
    """Download daily weather records from Open-Meteo archive API."""
    center_lat = (config.bbox.south + config.bbox.north) / 2.0
    center_lon = (config.bbox.west + config.bbox.east) / 2.0
    start = date(config.time_range.start_year, 1, 1).isoformat()
    end = date(config.time_range.end_year, 12, 31).isoformat()

    params = {
        "latitude": center_lat,
        "longitude": center_lon,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "Asia/Shanghai",
    }

    try:
        response = requests.get(config.api.open_meteo_url, params=params, timeout=120)
        response.raise_for_status()
        payload = response.json()
    except RequestException as exc:
        msg = f"Open-Meteo request failed: {exc}"
        raise RuntimeError(msg) from exc
    except ValueError as exc:
        msg = "Open-Meteo response is not valid JSON."
        raise RuntimeError(msg) from exc

    daily = payload.get("daily", {})
    times = daily.get("time", [])
    temp = daily.get("temperature_2m_mean", [])
    rain = daily.get("precipitation_sum", [])

    if not times:
        raise RuntimeError("Open-Meteo daily series is empty.")

    weather = pd.DataFrame(
        {
            "date": pd.to_datetime(times),
            "temperature_mean": pd.to_numeric(temp),
            "precipitation_sum": pd.to_numeric(rain),
        }
    )
    weather.to_csv(DATA_RAW / "weather_daily.csv", index=False)
    LOGGER.info("Weather records collected: %s daily rows", len(weather))
    return weather


def _synthetic_poi(config: ProjectConfig, grid: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic POI records when online sources are unavailable."""
    rng = np.random.default_rng(config.random_seed)
    center_lat = (config.bbox.south + config.bbox.north) / 2.0
    center_lon = (config.bbox.west + config.bbox.east) / 2.0

    n_points = int(max(5000, len(grid) * 10))
    probs = np.array([0.45, 0.25, 0.20, 0.10])

    rows: List[dict] = []
    for _ in range(n_points):
        lat = rng.uniform(config.bbox.south, config.bbox.north)
        lon = rng.uniform(config.bbox.west, config.bbox.east)

        distance = math.sqrt((lat - center_lat) ** 2 + (lon - center_lon) ** 2)
        keep_prob = max(0.15, 1.0 - 2.2 * distance)
        if rng.random() > keep_prob:
            continue

        rows.append(
            {
                "lat": float(lat),
                "lon": float(lon),
                "category": str(rng.choice(POI_CATEGORIES, p=probs)),
                "name": "",
            }
        )

    poi = pd.DataFrame(rows)
    poi.to_csv(DATA_RAW / "poi_points.csv", index=False)
    LOGGER.warning("Using synthetic POI data: %s rows", len(poi))
    return poi


def _synthetic_weather(config: ProjectConfig) -> pd.DataFrame:
    """Generate yearly climate proxies when weather API is unavailable."""
    rng = np.random.default_rng(config.random_seed + 7)
    records: List[dict] = []

    for year in range(config.time_range.start_year, config.time_range.end_year + 1):
        temp_year = 13.0 + 0.03 * (year - config.time_range.start_year) + rng.normal(0, 0.25)
        rain_year = 560 + 2.0 * (year - config.time_range.start_year) + rng.normal(0, 15)
        records.append(
            {
                "date": pd.Timestamp(year=year, month=7, day=1),
                "temperature_mean": temp_year,
                "precipitation_sum": rain_year,
            }
        )

    weather = pd.DataFrame(records)
    weather.to_csv(DATA_RAW / "weather_daily.csv", index=False)
    LOGGER.warning("Using synthetic weather data: %s yearly rows", len(weather))
    return weather


def collect_raw_data(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect base datasets and persist source metadata."""
    grid = build_grid(config)

    poi_source = "osm"
    weather_source = "open-meteo"

    try:
        poi = fetch_osm_poi(config)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("POI download failed, fallback to synthetic data: %s", exc)
        poi = _synthetic_poi(config, grid)
        poi_source = "synthetic"

    try:
        weather = fetch_open_meteo_daily(config)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Weather download failed, fallback to synthetic data: %s", exc)
        weather = _synthetic_weather(config)
        weather_source = "synthetic"

    source = {
        "poi_source": poi_source,
        "weather_source": weather_source,
        "city": config.city,
        "bbox": asdict(config.bbox),
    }
    dump_json(DATA_RAW / "data_sources.json", source)

    return grid, poi, weather


def _assign_points_to_cells(grid: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """Spatially aggregate point observations to grid cells."""
    if points.empty:
        return pd.DataFrame(columns=["cell_id", "category", "count"])

    south = float(grid["lat_min"].min())
    west = float(grid["lon_min"].min())
    lat_step = float(grid.iloc[0]["lat_max"] - grid.iloc[0]["lat_min"])
    lon_step = float(grid.iloc[0]["lon_max"] - grid.iloc[0]["lon_min"])

    n_rows = int(grid["row"].max()) + 1
    n_cols = int(grid["col"].max()) + 1

    lat_idx = np.floor((points["lat"].to_numpy() - south) / lat_step).astype(int)
    lon_idx = np.floor((points["lon"].to_numpy() - west) / lon_step).astype(int)

    mask = (lat_idx >= 0) & (lat_idx < n_rows) & (lon_idx >= 0) & (lon_idx < n_cols)
    valid = points.loc[mask].copy()
    valid["row"] = lat_idx[mask]
    valid["col"] = lon_idx[mask]

    key_map = grid[["row", "col", "cell_id"]]
    valid = valid.merge(key_map, on=["row", "col"], how="left")
    return valid.groupby(["cell_id", "category"], as_index=False).size().rename(columns={"size": "count"})


def _normalize_within_year(panel: pd.DataFrame, source_col: str, target_col: str) -> None:
    panel[target_col] = panel.groupby("year")[source_col].transform(lambda x: 100.0 * minmax_scale(x.to_numpy()))


def build_panel_dataset(
    config: ProjectConfig,
    grid: pd.DataFrame,
    poi: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Create a cell-year panel with engineered features and target indices."""
    rng = np.random.default_rng(config.random_seed)

    poi_counts = _assign_points_to_cells(grid, poi)
    if poi_counts.empty:
        raise RuntimeError("No POI available to build panel dataset.")

    pivot = (
        poi_counts.pivot_table(index="cell_id", columns="category", values="count", fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for cat in POI_CATEGORIES:
        if cat not in pivot:
            pivot[cat] = 0

    base = grid.merge(pivot, on="cell_id", how="left").fillna(0)

    center_lat = (config.bbox.south + config.bbox.north) / 2.0
    center_lon = (config.bbox.west + config.bbox.east) / 2.0
    dist_center = np.sqrt((base["lat_center"] - center_lat) ** 2 + (base["lon_center"] - center_lon) ** 2)
    base["centrality"] = 1.0 / (1.0 + 20.0 * dist_center)

    line_a = config.intervention.line_start
    line_b = config.intervention.line_end
    base["line_dist_km"] = base.apply(
        lambda r: point_line_distance_km((float(r["lat_center"]), float(r["lon_center"])), line_a, line_b),
        axis=1,
    )
    base["treated"] = (base["line_dist_km"] <= config.intervention.treatment_radius_km).astype(int)

    weather2 = weather.copy()
    weather2["date"] = pd.to_datetime(weather2["date"])
    weather2["year"] = weather2["date"].dt.year
    weather_year = (
        weather2.groupby("year", as_index=False)
        .agg(
            temperature_mean=("temperature_mean", "mean"),
            precipitation_sum=("precipitation_sum", "sum"),
        )
        .sort_values("year")
    )
    weather_year["temperature_norm"] = minmax_scale(weather_year["temperature_mean"].to_numpy())
    weather_year["precipitation_norm"] = minmax_scale(weather_year["precipitation_sum"].to_numpy())

    records: List[dict] = []
    year_span = max(1, config.time_range.end_year - config.time_range.start_year)

    for year in range(config.time_range.start_year, config.time_range.end_year + 1):
        year_fraction = (year - config.time_range.start_year) / year_span
        weather_this_year = weather_year.loc[weather_year["year"] == year]
        if weather_this_year.empty:
            temp_norm = 0.5
            rain_norm = 0.5
        else:
            temp_norm = float(weather_this_year["temperature_norm"].iloc[0])
            rain_norm = float(weather_this_year["precipitation_norm"].iloc[0])

        for row in base.itertuples(index=False):
            commercial = math.log1p(row.shop + row.office)
            public_service = math.log1p(row.amenity)
            leisure = math.log1p(row.leisure)
            density = math.log1p(row.amenity + row.shop + row.office + row.leisure)

            metro_base = math.exp(-row.line_dist_km / 2.5)
            metro_gain = 0.28 * math.exp(-row.line_dist_km / 1.2) if year >= config.intervention.start_year else 0.0
            metro_access = metro_base + metro_gain

            digital_attention = 0.4 * commercial + 0.3 * row.centrality + 0.2 * year_fraction + rng.normal(0, 0.03)
            crowding = 0.6 * density + 0.3 * row.centrality + rng.normal(0, 0.04)

            econ_raw = (
                0.44 * commercial
                + 0.20 * metro_access
                + 0.15 * digital_attention
                + 0.12 * row.centrality
                + 0.09 * year_fraction
                + rng.normal(0, 0.05)
            )
            livability_raw = (
                0.30 * public_service
                + 0.23 * leisure
                + 0.16 * rain_norm
                + 0.10 * temp_norm
                - 0.16 * crowding
                + 0.12 * row.centrality
                + rng.normal(0, 0.05)
            )
            innovation_raw = (
                0.40 * math.log1p(row.office)
                + 0.25 * metro_access
                + 0.18 * digital_attention
                + 0.10 * row.centrality
                + 0.07 * year_fraction
                + rng.normal(0, 0.05)
            )

            records.append(
                {
                    "year": year,
                    "cell_id": row.cell_id,
                    "lat_center": row.lat_center,
                    "lon_center": row.lon_center,
                    "lat_min": row.lat_min,
                    "lat_max": row.lat_max,
                    "lon_min": row.lon_min,
                    "lon_max": row.lon_max,
                    "treated": int(row.treated),
                    "line_dist_km": row.line_dist_km,
                    "amenity": row.amenity,
                    "shop": row.shop,
                    "office": row.office,
                    "leisure": row.leisure,
                    "centrality": row.centrality,
                    "temperature_norm": temp_norm,
                    "precipitation_norm": rain_norm,
                    "metro_access": metro_access,
                    "digital_attention": digital_attention,
                    "crowding": crowding,
                    "economic_raw": econ_raw,
                    "livability_raw": livability_raw,
                    "innovation_raw": innovation_raw,
                }
            )

    panel = pd.DataFrame(records)

    _normalize_within_year(panel, "economic_raw", "economic_vitality")
    _normalize_within_year(panel, "livability_raw", "livability")
    _normalize_within_year(panel, "innovation_raw", "innovation")

    panel["composite_index"] = (
        0.45 * panel["economic_vitality"]
        + 0.35 * panel["livability"]
        + 0.20 * panel["innovation"]
    )

    panel.to_csv(DATA_PROCESSED / "city_panel.csv", index=False)
    LOGGER.info("Panel dataset created: %s rows, %s columns", panel.shape[0], panel.shape[1])
    return panel
