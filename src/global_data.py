from __future__ import annotations

"""Global city data ingestion and panel feature engineering."""

import logging
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
from requests import Response
from requests import RequestException
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer

from .city_catalog import load_city_catalog
from .config import ProjectConfig
from .policy_taxonomy import (
    classify_policy_record,
    policy_direct_core_evidence_eligible as shared_policy_direct_core_evidence_eligible,
    policy_evidence_track as shared_policy_evidence_track,
    policy_subtype as shared_policy_subtype,
    policy_treatment_bucket as shared_policy_treatment_bucket,
    policy_type_coarse as shared_policy_type_coarse,
)
from .social_sentiment import aggregate_social_posts
from .theory_metrics import add_spatial_structure_features, entropy_weighted_score, fit_cobb_douglas_vitality
from .utils import DATA_PROCESSED, DATA_RAW, dump_json, haversine_km, minmax_scale

LOGGER = logging.getLogger(__name__)

WB_INDICATORS: Dict[str, str] = {
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "population": "SP.POP.TOTL",
    "unemployment": "SL.UEM.TOTL.ZS",
    "internet_users": "IT.NET.USER.ZS",
    "capital_formation": "NE.GDI.FTOT.ZS",
    "inflation": "FP.CPI.TOTL.ZG",
}

WB_EXTRA_INDICATORS: Dict[str, str] = {
    "patent_residents": "IP.PAT.RESD",
    "researchers_per_million": "SP.POP.SCIE.RD.P6",
    "high_tech_exports_share": "TX.VAL.TECH.MF.ZS",
    "employment_rate": "SL.EMP.TOTL.SP.ZS",
    "urban_population_share": "SP.URB.TOTL.IN.ZS",
    "electricity_access": "EG.ELC.ACCS.ZS",
    "fixed_broadband_subscriptions": "IT.NET.BBND.P2",
    "pm25_exposure": "EN.ATM.PM25.MC.M3",
}

MACRO_BASE_COLUMNS: List[str] = [
    "gdp_per_capita",
    "population",
    "unemployment",
    "internet_users",
    "capital_formation",
    "inflation",
]

CITY_MACRO_EXTRA_NUMERIC_COLUMNS: List[str] = [
    "gdp_total_ppp_observed",
    "gdp_total_local_observed",
    "gdp_share_national_observed",
]

CITY_MACRO_EXTRA_STRING_COLUMNS: List[str] = [
    "macro_geo_code_observed",
    "macro_geo_name_observed",
    "macro_currency_code",
    "oecd_fua_code",
    "oecd_city_code",
    "oecd_fua_name",
    "oecd_city_name",
]

CITY_MACRO_COLUMN_ALIASES: Dict[str, List[str]] = {
    "gdp_per_capita": ["gdp_per_capita", "gdp_pc", "city_gdp_per_capita"],
    "population": ["population", "city_population", "pop"],
    "unemployment": ["unemployment", "unemployment_rate", "city_unemployment"],
    "internet_users": ["internet_users", "internet_penetration", "city_internet_users"],
    "capital_formation": ["capital_formation", "gross_capital_formation", "city_capital_formation"],
    "inflation": ["inflation", "city_inflation", "cpi_inflation"],
}

CITY_MACRO_EXTRA_COLUMN_ALIASES: Dict[str, List[str]] = {
    "gdp_total_ppp_observed": [
        "gdp_total_ppp_observed",
        "gdp_total_ppp",
        "gdp_total",
        "city_gdp_total",
        "gdp_ppp",
    ],
    "gdp_total_local_observed": [
        "gdp_total_local_observed",
        "gdp_total_local",
        "gdp_total_nominal_observed",
        "city_gdp_total_local",
        "gdp_local",
    ],
    "gdp_share_national_observed": [
        "gdp_share_national_observed",
        "gdp_share_national",
        "gdp_pct_national",
        "gdp_national_share",
    ],
}

ROAD_COLUMN_ALIASES: Dict[str, List[str]] = {
    "road_length_km_total": [
        "road_length_km_total",
        "road_length_km",
        "road_total_km",
        "total_road_length_km",
    ],
    "arterial_share": [
        "arterial_share",
        "main_road_share",
        "trunk_primary_share",
    ],
    "intersection_density": [
        "intersection_density",
        "intersections_per_km2",
        "junction_density",
    ],
}

VIIRS_RADIANCE_ALIASES: List[str] = [
    "radiance",
    "radiance_mean",
    "avg_rad",
    "ntl_mean",
    "dnb_mean",
    "mean_ntl",
]

VIIRS_LIT_AREA_ALIASES: List[str] = [
    "lit_area_km2",
    "lit_pixels",
    "lit_area",
]

VIIRS_SUM_ALIASES: List[str] = [
    "viirs_ntl_sum",
    "radiance_sum",
    "avg_rad_sum",
    "ntl_sum",
    "sum_ntl",
]

NO2_ALIASES: Dict[str, List[str]] = {
    "no2_trop_mean": [
        "no2_trop_mean",
        "no2_mean",
        "tropospheric_no2_column_number_density",
        "ColumnAmountNO2TropCloudScreened",
        "no2_column",
    ],
    "no2_valid_obs_count": [
        "no2_valid_obs_count",
        "valid_obs_count",
        "count",
        "no2_count",
    ],
}

CONNECTIVITY_ALIASES: Dict[str, List[str]] = {
    "flight_connectivity_total": [
        "flight_connectivity_total",
        "flight_total",
        "flight_count",
        "flight_movements",
    ],
    "flight_degree_centrality": [
        "flight_degree_centrality",
        "degree_centrality",
        "airport_degree_centrality",
    ],
    "airport_count_mapped": [
        "airport_count_mapped",
        "mapped_airport_count",
        "airport_count_active",
    ],
    "international_route_share": [
        "international_route_share",
        "intl_route_share",
        "international_flight_share",
    ],
    "shipping_connectivity_total": [
        "shipping_connectivity_total",
        "shipping_total",
        "shipping_movements",
        "maritime_connectivity_total",
    ],
}

GHSL_ALIASES: Dict[str, List[str]] = {
    "ghsl_built_surface_km2": [
        "ghsl_built_surface_km2",
        "built_surface_km2",
        "built_surface",
    ],
    "ghsl_built_surface_nres_km2": [
        "ghsl_built_surface_nres_km2",
        "built_surface_nres_km2",
        "built_surface_nres",
    ],
    "ghsl_built_volume_m3": [
        "ghsl_built_volume_m3",
        "built_volume_m3",
        "built_volume_total",
        "built_volume",
    ],
    "ghsl_built_volume_nres_m3": [
        "ghsl_built_volume_nres_m3",
        "built_volume_nres_m3",
        "built_volume_nres",
    ],
}

OSM_HISTORY_ALIASES: Dict[str, List[str]] = {
    "osm_hist_road_length_m": ["osm_hist_road_length_m", "road_length_m", "road_len_m"],
    "osm_hist_building_count": ["osm_hist_building_count", "building_count"],
    "osm_hist_poi_count": ["osm_hist_poi_count", "poi_count"],
    "osm_hist_poi_food_count": ["osm_hist_poi_food_count", "poi_food_count", "food_count"],
    "osm_hist_poi_retail_count": ["osm_hist_poi_retail_count", "poi_retail_count", "retail_count"],
    "osm_hist_poi_nightlife_count": [
        "osm_hist_poi_nightlife_count",
        "poi_nightlife_count",
        "nightlife_count",
    ],
    "osm_hist_road_yoy": ["osm_hist_road_yoy", "road_yoy"],
    "osm_hist_building_yoy": ["osm_hist_building_yoy", "building_yoy"],
    "osm_hist_poi_yoy": ["osm_hist_poi_yoy", "poi_yoy"],
    "osm_hist_poi_food_yoy": ["osm_hist_poi_food_yoy", "food_yoy"],
    "osm_hist_poi_retail_yoy": ["osm_hist_poi_retail_yoy", "retail_yoy"],
    "osm_hist_poi_nightlife_yoy": ["osm_hist_poi_nightlife_yoy", "nightlife_yoy"],
}

OSM_HISTORY_LEVEL_TO_YOY: Dict[str, str] = {
    "osm_hist_road_length_m": "osm_hist_road_yoy",
    "osm_hist_building_count": "osm_hist_building_yoy",
    "osm_hist_poi_count": "osm_hist_poi_yoy",
    "osm_hist_poi_food_count": "osm_hist_poi_food_yoy",
    "osm_hist_poi_retail_count": "osm_hist_poi_retail_yoy",
    "osm_hist_poi_nightlife_count": "osm_hist_poi_nightlife_yoy",
}

SOCIAL_SENTIMENT_ALIASES: Dict[str, List[str]] = {
    "social_sentiment_score": [
        "social_sentiment_score",
        "sentiment_score",
        "sentiment_mean",
        "city_sentiment_score",
        "avg_tone",
        "average_tone",
        "gdelt_avg_tone",
        "tone",
    ],
    "social_sentiment_volatility": [
        "social_sentiment_volatility",
        "sentiment_volatility",
        "sentiment_std",
    ],
    "social_sentiment_positive_share": [
        "social_sentiment_positive_share",
        "positive_share",
        "sentiment_positive_share",
        "positive_tone_share",
    ],
    "social_sentiment_negative_share": [
        "social_sentiment_negative_share",
        "negative_share",
        "sentiment_negative_share",
        "negative_tone_share",
    ],
    "social_sentiment_volume": [
        "social_sentiment_volume",
        "sentiment_volume",
        "post_count",
        "n_posts",
        "num_articles",
        "article_count",
        "gdelt_num_articles",
        "mention_count",
    ],
    "social_sentiment_platform_count": [
        "social_sentiment_platform_count",
        "platform_count",
        "source_count",
    ],
    "social_sentiment_buzz": [
        "social_sentiment_buzz",
        "sentiment_buzz",
        "buzz",
    ],
}


def _request_json(
    method: str,
    url: str,
    *,
    timeout: int,
    retries: int = 3,
    backoff_seconds: float = 1.0,
    **kwargs,
) -> dict:
    """HTTP JSON helper with retry and exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response: Response = requests.request(method, url, timeout=timeout, **kwargs)
            if response.status_code in {429, 500, 502, 503, 504}:
                response.raise_for_status()
            response.raise_for_status()
            return response.json()
        except (RequestException, ValueError) as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep_s = backoff_seconds * (2 ** (attempt - 1))
            LOGGER.warning("Request failed (%s), retry %s/%s in %.1fs", exc, attempt, retries, sleep_s)
            time.sleep(sleep_s)

    msg = f"Request failed after {retries} attempts for {url}: {last_error}"
    raise RuntimeError(msg)


def _world_bank_url(indicator_code: str, start_year: int, end_year: int) -> str:
    return (
        "https://api.worldbank.org/v2/country/all/indicator/"
        f"{indicator_code}?format=json&date={start_year}:{end_year}&per_page=20000"
    )


def fetch_world_bank_indicator(indicator_code: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch one World Bank indicator for all countries and years.

    Handles pagination: the WB API may split results across multiple pages.
    """
    page = 1
    all_rows: List[dict] = []

    while True:
        url = _world_bank_url(indicator_code, start_year, end_year) + f"&page={page}"
        payload = _request_json("GET", url, timeout=60, retries=4)

        if not isinstance(payload, list) or len(payload) < 2:
            if page == 1:
                msg = f"Unexpected World Bank payload for {indicator_code}."
                raise RuntimeError(msg)
            break  # No more data on subsequent pages.

        meta = payload[0]
        total_pages = int(meta.get("pages", 1))

        for row in payload[1]:
            iso3 = row.get("countryiso3code")
            year_raw = row.get("date")
            value = row.get("value")
            if not iso3 or value is None:
                continue
            try:
                year = int(year_raw)
                value_num = float(value)
            except (TypeError, ValueError):
                continue
            all_rows.append({"iso3": iso3, "year": year, "value": value_num})

        if page >= total_pages:
            break
        page += 1

    out = pd.DataFrame(all_rows)
    if out.empty:
        msg = f"World Bank indicator {indicator_code} has no usable rows."
        raise RuntimeError(msg)
    LOGGER.debug("WB indicator %s: fetched %d rows across %d page(s)", indicator_code, len(out), page)
    return out


def _synthetic_world_bank(cities: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Generate synthetic macro panel if remote data is unavailable."""
    if getattr(config, "strict_no_synthetic", False):
        msg = "Synthetic World Bank data generation is disabled (strict_no_synthetic=True)."
        raise RuntimeError(msg)
    rng = np.random.default_rng(config.random_seed + 101)
    records: List[dict] = []

    continent_base_gdp = {
        "North America": 52000,
        "Europe": 43000,
        "Asia": 20000,
        "South America": 14000,
        "Africa": 7000,
        "Oceania": 47000,
    }

    for row in cities.itertuples(index=False):
        base = continent_base_gdp.get(row.continent, 18000)
        city_factor = 0.85 + 0.30 * rng.random()
        gdp0 = base * city_factor

        for year in range(config.time_range.start_year, config.time_range.end_year + 1):
            trend = 1.0 + 0.018 * (year - config.time_range.start_year)
            shock = rng.normal(0, 0.03)
            gdp = max(1200.0, gdp0 * trend * (1.0 + shock))
            population = 2_000_000 + 15_000_000 * rng.random() + 120_000 * (year - config.time_range.start_year)
            unemployment = max(2.0, min(20.0, 7.0 + rng.normal(0, 1.5)))
            internet = max(25.0, min(98.0, 55.0 + 3.2 * (year - config.time_range.start_year) + rng.normal(0, 4.0)))
            capital = max(10.0, min(45.0, 24.0 + rng.normal(0, 3.0)))
            inflation = max(-1.0, min(20.0, 3.5 + rng.normal(0, 2.0)))

            records.append(
                {
                    "iso3": row.iso3,
                    "year": year,
                    "gdp_per_capita": gdp,
                    "population": population,
                    "unemployment": unemployment,
                    "internet_users": internet,
                    "capital_formation": capital,
                    "inflation": inflation,
                }
            )

    panel = pd.DataFrame(records)
    panel["macro_source"] = "synthetic"
    return panel


def _is_macro_cache_valid(df: pd.DataFrame, cities: pd.DataFrame, config: ProjectConfig) -> bool:
    city_iso = set(cities["iso3"].unique())
    years = set(range(config.time_range.start_year, config.time_range.end_year + 1))
    cached_iso = set(df["iso3"].unique()) if "iso3" in df else set()
    cached_years = set(df["year"].unique()) if "year" in df else set()
    required_cols = {"iso3", "year", *WB_INDICATORS.keys(), "macro_source"}
    return required_cols.issubset(set(df.columns)) and city_iso.issubset(cached_iso) and years.issubset(cached_years)


def collect_world_bank_panel(
    cities: pd.DataFrame,
    config: ProjectConfig,
    use_cache: bool = True,
    strict_real_data: bool = False,
) -> pd.DataFrame:
    """Collect and merge all required World Bank indicators."""
    cache_path = DATA_RAW / "wb_macro_panel.csv"
    if use_cache and cache_path.exists():
        cached = pd.read_csv(cache_path)
        if strict_real_data:
            if "macro_source" in cached.columns:
                cached = cached[cached["macro_source"] == "world_bank"].copy()
        if _is_macro_cache_valid(cached, cities, config):
            LOGGER.info("Using cached World Bank macro panel.")
            return cached[cached["iso3"].isin(cities["iso3"].unique())].copy()

    start_year = config.time_range.start_year
    end_year = config.time_range.end_year

    frames: List[pd.DataFrame] = []
    used_synthetic = False

    for name, code in WB_INDICATORS.items():
        try:
            indicator = fetch_world_bank_indicator(code, start_year, end_year)
            indicator = indicator.rename(columns={"value": name})
            frames.append(indicator)
            indicator.to_csv(DATA_RAW / f"wb_{name}.csv", index=False)
            LOGGER.info("World Bank indicator loaded: %s (%s rows)", name, len(indicator))
        except Exception as exc:  # noqa: BLE001
            if strict_real_data:
                msg = f"Strict mode: failed to fetch World Bank indicator {name}: {exc}"
                raise RuntimeError(msg) from exc
            LOGGER.warning("Failed to fetch %s (%s), switching to synthetic macro panel.", name, exc)
            LOGGER.warning(
                "WARNING: Synthetic macro data will be generated. "
                "Results using synthetic data are NOT valid for publication. "
                "Re-run with strict_real_data=True or ensure network access to World Bank API."
            )
            used_synthetic = True
            break

    if used_synthetic:
        macro = _synthetic_world_bank(cities, config)
        macro.to_csv(cache_path, index=False)
        return macro

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["iso3", "year"], how="outer")

    merged = merged.sort_values(["iso3", "year"]).reset_index(drop=True)
    target_iso = sorted(cities["iso3"].unique())
    target_years = list(range(start_year, end_year + 1))
    full_index = pd.MultiIndex.from_product([target_iso, target_years], names=["iso3", "year"]).to_frame(index=False)
    merged = full_index.merge(merged, on=["iso3", "year"], how="left")
    merged["macro_source"] = "world_bank"
    merged = _impute_country_year_indicator_panel(
        merged,
        list(WB_INDICATORS.keys()),
        cities,
        strategy_col="macro_imputation_strategy",
    )
    merged = merged.drop(columns=["continent"], errors="ignore")

    merged.to_csv(cache_path, index=False)
    return merged


def _is_extra_wb_cache_valid(df: pd.DataFrame, cities: pd.DataFrame, config: ProjectConfig) -> bool:
    iso_set = set(cities["iso3"].unique())
    years = set(range(config.time_range.start_year, config.time_range.end_year + 1))
    required = {"iso3", "year", *WB_EXTRA_INDICATORS.keys()}
    has_source = ("extra_wb_source" in df.columns) or ("external_source" in df.columns)
    cached_iso = set(df["iso3"].unique()) if "iso3" in df.columns else set()
    cached_years = set(df["year"].unique()) if "year" in df.columns else set()
    return required.issubset(set(df.columns)) and has_source and iso_set.issubset(cached_iso) and years.issubset(cached_years)


def collect_world_bank_extra_panel(
    cities: pd.DataFrame,
    config: ProjectConfig,
    use_cache: bool = True,
    strict_real_data: bool = False,
) -> pd.DataFrame:
    """Collect additional objective World Bank indicators used for AI-economic modeling."""
    cache_path = DATA_RAW / "wb_extra_panel.csv"
    start_year = config.time_range.start_year
    end_year = config.time_range.end_year
    target_iso = sorted(cities["iso3"].unique().tolist())
    target_years = list(range(start_year, end_year + 1))

    if use_cache and cache_path.exists():
        cached = pd.read_csv(cache_path)
        if "extra_wb_source" not in cached.columns and "external_source" in cached.columns:
            cached["extra_wb_source"] = cached["external_source"].astype(str)
        if "external_source" not in cached.columns and "extra_wb_source" in cached.columns:
            cached["external_source"] = cached["extra_wb_source"].astype(str)
        if strict_real_data and "extra_wb_source" in cached.columns:
            cached = cached[cached["extra_wb_source"] == "world_bank"].copy()
        if _is_extra_wb_cache_valid(cached, cities, config):
            out = cached[(cached["iso3"].isin(target_iso)) & (cached["year"].between(start_year, end_year))].copy()
            LOGGER.info("Using cached World Bank extra panel.")
            return out

    frames: List[pd.DataFrame] = []
    for name, code in WB_EXTRA_INDICATORS.items():
        try:
            indicator = fetch_world_bank_indicator(code, start_year, end_year)
            indicator = indicator.rename(columns={"value": name})
            frames.append(indicator)
            indicator.to_csv(DATA_RAW / f"wb_extra_{name}.csv", index=False)
            LOGGER.info("World Bank extra indicator loaded: %s (%s rows)", name, len(indicator))
        except Exception as exc:  # noqa: BLE001
            if strict_real_data:
                msg = f"Strict mode: failed to fetch World Bank extra indicator {name}: {exc}"
                raise RuntimeError(msg) from exc
            LOGGER.warning("Failed to fetch extra indicator %s (%s), will impute from available signals.", name, exc)

    full_index = pd.MultiIndex.from_product([target_iso, target_years], names=["iso3", "year"]).to_frame(index=False)
    if not frames:
        out = full_index.copy()
        for col in WB_EXTRA_INDICATORS:
            out[col] = np.nan
    else:
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=["iso3", "year"], how="outer")
        out = full_index.merge(merged, on=["iso3", "year"], how="left")
    for col in WB_EXTRA_INDICATORS:
        if col not in out.columns:
            out[col] = np.nan
    out = _impute_country_year_indicator_panel(
        out,
        list(WB_EXTRA_INDICATORS.keys()),
        cities,
        strategy_col="extra_wb_imputation_strategy",
    )
    out = out.drop(columns=["continent"], errors="ignore")

    if len(frames) == len(WB_EXTRA_INDICATORS):
        source_tag = "world_bank"
    elif len(frames) > 0:
        source_tag = "world_bank_partial"
    else:
        source_tag = "unavailable"
    out["extra_wb_source"] = source_tag
    out["external_source"] = source_tag

    out.to_csv(cache_path, index=False)
    return out


def _fetch_city_weather(
    city_lat: float,
    city_lon: float,
    config: ProjectConfig,
    strict_real_data: bool = False,
) -> pd.DataFrame:
    """Fetch yearly weather aggregates for a city from Open-Meteo."""
    start_date = f"{config.time_range.start_year}-01-01"
    end_date = f"{config.time_range.end_year}-12-31"

    params = {
        "latitude": city_lat,
        "longitude": city_lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "auto",
    }

    retries = 1 if strict_real_data else 3
    payload = _request_json("GET", config.api.open_meteo_url, timeout=45, retries=retries, params=params)

    daily = payload.get("daily", {})
    times = pd.to_datetime(daily.get("time", []))
    if len(times) == 0:
        msg = "Open-Meteo returned empty daily time series."
        raise RuntimeError(msg)

    df = pd.DataFrame(
        {
            "date": times,
            "temperature_mean": pd.to_numeric(daily.get("temperature_2m_mean", [])),
            "precipitation_sum": pd.to_numeric(daily.get("precipitation_sum", [])),
        }
    )
    df["year"] = df["date"].dt.year
    return (
        df.groupby("year", as_index=False)
        .agg(temperature_mean=("temperature_mean", "mean"), precipitation_sum=("precipitation_sum", "sum"))
        .sort_values("year")
    )


def _fetch_city_weather_nasa(
    city_lat: float,
    city_lon: float,
    config: ProjectConfig,
    strict_real_data: bool = False,
) -> pd.DataFrame:
    """Fetch yearly weather aggregates from NASA POWER as a real-data fallback."""
    start = f"{config.time_range.start_year}0101"
    end = f"{config.time_range.end_year}1231"
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,PRECTOTCORR&community=RE&longitude={city_lon:.4f}"
        f"&latitude={city_lat:.4f}&start={start}&end={end}&format=JSON"
    )

    payload = _request_json(
        "GET",
        url,
        timeout=20 if strict_real_data else 60,
        retries=1 if strict_real_data else 5,
        backoff_seconds=0.2 if strict_real_data else 1.0,
    )
    parameters = payload.get("properties", {}).get("parameter", {})
    t2m = parameters.get("T2M", {})
    prec = parameters.get("PRECTOTCORR", {})
    if not t2m or not prec:
        msg = "NASA POWER returned empty weather payload."
        raise RuntimeError(msg)

    rows: List[dict] = []
    for day_str, temp in t2m.items():
        if day_str not in prec:
            continue
        try:
            year = int(day_str[:4])
            temp_val = float(temp)
            rain_val = float(prec[day_str])
        except (TypeError, ValueError):
            continue
        if year < config.time_range.start_year or year > config.time_range.end_year:
            continue
        rows.append({"year": year, "temperature_mean": temp_val, "precipitation_sum": rain_val})

    if not rows:
        msg = "NASA POWER has no usable weather rows."
        raise RuntimeError(msg)

    df = pd.DataFrame(rows)
    return (
        df.groupby("year", as_index=False)
        .agg(temperature_mean=("temperature_mean", "mean"), precipitation_sum=("precipitation_sum", "sum"))
        .sort_values("year")
    )


def _synthetic_city_weather(config: ProjectConfig, city_name: str, seed_offset: int) -> pd.DataFrame:
    """Generate fallback climate features when API fetch fails."""
    if getattr(config, "strict_no_synthetic", False):
        msg = f"Synthetic weather generation for {city_name} is disabled (strict_no_synthetic=True)."
        raise RuntimeError(msg)
    rng = np.random.default_rng(config.random_seed + seed_offset)
    rows: List[dict] = []

    base_temp = 11.0 + 10.0 * rng.random()
    base_rain = 350.0 + 1200.0 * rng.random()

    for year in range(config.time_range.start_year, config.time_range.end_year + 1):
        rows.append(
            {
                "year": year,
                "temperature_mean": base_temp + 0.02 * (year - config.time_range.start_year) + rng.normal(0, 0.3),
                "precipitation_sum": base_rain + rng.normal(0, 40.0),
                "weather_source": "synthetic",
                "city_name": city_name,
            }
        )

    return pd.DataFrame(rows)


def _is_weather_cache_valid(df: pd.DataFrame, cities: pd.DataFrame, config: ProjectConfig) -> bool:
    if not {"city_id", "year", "temperature_mean", "precipitation_sum", "weather_source"}.issubset(df.columns):
        return False
    city_set = set(cities["city_id"])
    year_set = set(range(config.time_range.start_year, config.time_range.end_year + 1))
    return city_set.issubset(set(df["city_id"])) and year_set.issubset(set(df["year"]))


def _weather_complete_city_ids(df: pd.DataFrame, config: ProjectConfig) -> set[str]:
    """Return city ids that already have a complete year series in cache."""
    required_years = config.time_range.end_year - config.time_range.start_year + 1
    if df.empty:
        return set()
    counts = df.groupby("city_id")["year"].nunique()
    return set(counts[counts >= required_years].index.tolist())


def _truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(value: str | None, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:  # noqa: BLE001
        return int(default)


def _env_float(value: str | None, default: float) -> float:
    try:
        return float(str(value).strip())
    except Exception:  # noqa: BLE001
        return float(default)


def _impute_weather_from_real_pool(
    missing_cities: pd.DataFrame,
    real_pool: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Impute missing weather by continent-year medians from real observed weather pool."""
    if missing_cities.empty:
        return pd.DataFrame()
    if real_pool.empty:
        return pd.DataFrame()

    pool = real_pool.copy()
    if "continent" not in pool.columns:
        pool["continent"] = "__unknown__"
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").round().astype(int)
    pool["temperature_mean"] = pd.to_numeric(pool["temperature_mean"], errors="coerce")
    pool["precipitation_sum"] = pd.to_numeric(pool["precipitation_sum"], errors="coerce")
    pool = pool.dropna(subset=["year", "temperature_mean", "precipitation_sum"]).copy()
    pool["continent"] = pool["continent"].fillna("__unknown__").astype(str)
    if pool.empty:
        return pd.DataFrame()

    cont_year = (
        pool.groupby(["continent", "year"], as_index=False)
        .agg(
            temperature_mean=("temperature_mean", "median"),
            precipitation_sum=("precipitation_sum", "median"),
        )
    )
    global_year = (
        pool.groupby("year", as_index=False)
        .agg(
            temperature_mean=("temperature_mean", "median"),
            precipitation_sum=("precipitation_sum", "median"),
        )
    )
    global_med_temp = float(pool["temperature_mean"].median())
    global_med_prec = float(pool["precipitation_sum"].median())

    years = list(range(int(config.time_range.start_year), int(config.time_range.end_year) + 1))
    rows: List[dict] = []
    cont_year_key = {
        (str(r.continent), int(r.year)): (float(r.temperature_mean), float(r.precipitation_sum))
        for r in cont_year.itertuples(index=False)
    }
    global_year_key = {
        int(r.year): (float(r.temperature_mean), float(r.precipitation_sum))
        for r in global_year.itertuples(index=False)
    }
    for city in missing_cities.itertuples(index=False):
        for year in years:
            t, p = cont_year_key.get(
                (str(city.continent), int(year)),
                global_year_key.get(int(year), (global_med_temp, global_med_prec)),
            )
            rows.append(
                {
                    "city_id": city.city_id,
                    "city_name": city.city_name,
                    "year": int(year),
                    "temperature_mean": float(t),
                    "precipitation_sum": float(p),
                    "weather_source": "imputed_from_weather_pool",
                }
            )
    return pd.DataFrame(rows)


def collect_city_weather(
    cities: pd.DataFrame,
    config: ProjectConfig,
    use_cache: bool = True,
    strict_real_data: bool = False,
) -> pd.DataFrame:
    """Collect weather series for all cities."""
    cache_path = DATA_RAW / "city_weather_yearly.csv"
    cached_other = pd.DataFrame()
    cached_subset = pd.DataFrame()
    cached_ids: set[str] = set()

    if use_cache and cache_path.exists():
        cached = pd.read_csv(cache_path)
        required = {"city_id", "city_name", "year", "temperature_mean", "precipitation_sum", "weather_source"}
        if required.issubset(cached.columns):
            cached_other = cached[~cached["city_id"].isin(cities["city_id"])].copy()
            cached_subset = cached[cached["city_id"].isin(cities["city_id"])].copy()
            if strict_real_data:
                cached_subset = cached_subset[cached_subset["weather_source"].isin(["open-meteo", "nasa-power"])].copy()
            cached_ids = _weather_complete_city_ids(cached_subset, config)
            cached_subset = cached_subset[cached_subset["city_id"].isin(cached_ids)].copy()
            if cached_ids:
                LOGGER.info("Using cached weather for %s/%s cities.", len(cached_ids), len(cities))

    missing_cities = cities[~cities["city_id"].isin(cached_ids)].copy()
    records: List[pd.DataFrame] = [cached_subset] if not cached_subset.empty else []
    total_missing = len(missing_cities)
    failures: List[str] = []
    consecutive_failures = 0
    allow_pool_impute = _truthy_env(os.environ.get("URBAN_PULSE_STRICT_ALLOW_WEATHER_POOL_IMPUTE", "1"))
    strict_circuit = max(1, _env_int(os.environ.get("URBAN_PULSE_STRICT_WEATHER_CIRCUIT"), 8))

    for idx, city in enumerate(missing_cities.itertuples(index=False), start=1):
        if idx > 1:
            time.sleep(0.20 if strict_real_data else 0.25)

        try:
            weather = _fetch_city_weather(city.latitude, city.longitude, config, strict_real_data=strict_real_data)
            weather["weather_source"] = "open-meteo"
            consecutive_failures = 0
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Open-Meteo failed for %s, trying NASA POWER: %s", city.city_name, exc)
            try:
                weather = _fetch_city_weather_nasa(
                    city.latitude,
                    city.longitude,
                    config,
                    strict_real_data=strict_real_data,
                )
                weather["weather_source"] = "nasa-power"
                consecutive_failures = 0
            except Exception as exc2:  # noqa: BLE001
                if strict_real_data:
                    LOGGER.error("Strict mode weather failure for %s: %s", city.city_name, exc2)
                    failures.append(city.city_name)
                    consecutive_failures += 1
                    if consecutive_failures >= strict_circuit:
                        LOGGER.warning(
                            "Weather strict-mode circuit breaker triggered (consecutive_failures=%s). "
                            "Remaining cities will be handled via objective weather-pool imputation.",
                            consecutive_failures,
                        )
                        break
                    continue
                LOGGER.warning("Weather fallback for %s: %s", city.city_name, exc2)
                weather = _synthetic_city_weather(config, city.city_name, seed_offset=1000 + idx)

        weather["city_id"] = city.city_id
        weather["city_name"] = city.city_name
        records.append(weather)

        if idx % 10 == 0 or idx == total_missing:
            LOGGER.info("Weather incremental progress: %s/%s cities", idx, total_missing)
            checkpoint = pd.concat(records, ignore_index=True).sort_values(["city_id", "year"])
            checkpoint = checkpoint.drop_duplicates(subset=["city_id", "year"], keep="last")
            if not cached_other.empty:
                checkpoint = pd.concat([cached_other, checkpoint], ignore_index=True)
                checkpoint = checkpoint.sort_values(["city_id", "year"]).drop_duplicates(subset=["city_id", "year"], keep="last")
            checkpoint.to_csv(cache_path, index=False)

    if not records:
        out = pd.DataFrame(columns=["year", "temperature_mean", "precipitation_sum", "weather_source", "city_id", "city_name"])
    else:
        out = pd.concat(records, ignore_index=True)

    if strict_real_data and failures:
        out_ids = set(out["city_id"].astype(str).unique().tolist()) if not out.empty else set()
        missing_ids = [cid for cid in cities["city_id"].astype(str).tolist() if cid not in out_ids]
        if allow_pool_impute and missing_ids:
            real_pool = (
                out[out["weather_source"].isin(["open-meteo", "nasa-power"])].copy()
                if not out.empty
                else pd.DataFrame()
            )
            if not real_pool.empty:
                city_cont = cities[["city_id", "continent"]].drop_duplicates("city_id")
                real_pool = real_pool.merge(city_cont, on="city_id", how="left")
            miss_df = cities[cities["city_id"].astype(str).isin(missing_ids)].copy()
            imputed = _impute_weather_from_real_pool(miss_df, real_pool, config)
            if not imputed.empty:
                LOGGER.warning(
                    "Strict mode weather imputation applied from real-weather pool: imputed_cities=%s, real_pool_cities=%s",
                    imputed["city_id"].nunique(),
                    real_pool["city_id"].nunique() if not real_pool.empty else 0,
                )
                records2 = [out, imputed] if not out.empty else [imputed]
                out = pd.concat(records2, ignore_index=True)
            else:
                msg = (
                    "Strict real-data mode failed during weather collection and no real weather pool is available "
                    "for objective imputation."
                )
                raise RuntimeError(msg)
        elif not allow_pool_impute:
            msg = (
                "Strict real-data mode failed during weather collection. "
                f"Failed cities ({len(failures)}): {', '.join(failures[:20])}"
                + (" ..." if len(failures) > 20 else "")
            )
            raise RuntimeError(msg)

    out = out.sort_values(["city_id", "year"]).drop_duplicates(subset=["city_id", "year"], keep="last")
    cache_out = out
    if not cached_other.empty:
        cache_out = pd.concat([cached_other, out], ignore_index=True)
        cache_out = cache_out.sort_values(["city_id", "year"]).drop_duplicates(subset=["city_id", "year"], keep="last")
    cache_out.to_csv(cache_path, index=False)

    LOGGER.info("City weather panel generated: %s rows", len(out))
    return out[out["city_id"].isin(cities["city_id"])].copy()


def _build_poi_query(lat: float, lon: float, radius_m: int = 7000) -> str:
    return f"""
    [out:json][timeout:45];
    (
      node(around:{radius_m},{lat},{lon})["amenity"];
      node(around:{radius_m},{lat},{lon})["shop"];
      node(around:{radius_m},{lat},{lon})["office"];
      node(around:{radius_m},{lat},{lon})["leisure"];
      node(around:{radius_m},{lat},{lon})["public_transport"];
      node(around:{radius_m},{lat},{lon})["railway"="station"];
    );
    out body;
    """.strip()


def _overpass_endpoints(primary_url: str) -> List[str]:
    """Return prioritized Overpass mirrors for resilience."""
    mirrors = [
        primary_url,
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
    ]
    dedup: List[str] = []
    for url in mirrors:
        if url not in dedup:
            dedup.append(url)
    return dedup


def _fetch_city_poi(
    city_lat: float,
    city_lon: float,
    overpass_url: str,
    strict_real_data: bool = False,
) -> Dict[str, float]:
    # Use consistent 7km radius in both modes for comparable results.
    query = _build_poi_query(city_lat, city_lon, radius_m=7000)
    last_error: Exception | None = None
    payload: dict | None = None
    endpoints = _overpass_endpoints(overpass_url)
    strict_all_endpoints = _truthy_env(os.getenv("URBAN_PULSE_STRICT_POI_ALL_ENDPOINTS", "1"))
    if strict_real_data and not strict_all_endpoints:
        endpoints = endpoints[:2]
    if endpoints:
        shift = int((abs(float(city_lat)) + abs(float(city_lon))) * 1000) % len(endpoints)
        endpoints = endpoints[shift:] + endpoints[:shift]
    strict_timeout = max(8, _env_int(os.getenv("URBAN_PULSE_STRICT_POI_TIMEOUT"), 14))
    strict_retries = max(1, _env_int(os.getenv("URBAN_PULSE_STRICT_POI_RETRIES"), 2))
    strict_backoff = max(0.2, _env_float(os.getenv("URBAN_PULSE_STRICT_POI_BACKOFF"), 1.0))
    for idx, endpoint in enumerate(endpoints, start=1):
        try:
            payload = _request_json(
                "POST",
                endpoint,
                timeout=strict_timeout if strict_real_data else 60,
                retries=strict_retries if strict_real_data else 4,
                backoff_seconds=strict_backoff if strict_real_data else 1.0,
                data={"data": query},
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if strict_real_data:
                # In strict mode, avoid hammering mirrors when rate-limited.
                err_text = str(exc).lower()
                wait_scale = 1.0 if ("429" in err_text or "rate limit" in err_text) else 0.35
                time.sleep(wait_scale * float(idx))
            continue
    if payload is None:
        msg = f"All Overpass mirrors failed: {last_error}"
        raise RuntimeError(msg)

    counts = {
        "amenity": 0,
        "shop": 0,
        "office": 0,
        "leisure": 0,
        "transport": 0,
    }

    for elem in payload.get("elements", []):
        tags = elem.get("tags", {})
        if "amenity" in tags:
            counts["amenity"] += 1
        if "shop" in tags:
            counts["shop"] += 1
        if "office" in tags:
            counts["office"] += 1
        if "leisure" in tags:
            counts["leisure"] += 1
        if "public_transport" in tags or tags.get("railway") == "station":
            counts["transport"] += 1

    return counts


def _synthetic_city_poi(city: pd.Series, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    base = {
        "Asia": 2200,
        "Europe": 1800,
        "North America": 1600,
        "South America": 1300,
        "Africa": 900,
        "Oceania": 1100,
    }.get(str(city["continent"]), 1200)

    amenity = max(200, int(base * (0.8 + 0.6 * rng.random())))
    shop = max(120, int(base * (0.5 + 0.7 * rng.random())))
    office = max(100, int(base * (0.4 + 0.8 * rng.random())))
    leisure = max(60, int(base * (0.2 + 0.5 * rng.random())))
    transport = max(20, int(base * (0.05 + 0.15 * rng.random())))
    return {
        "amenity": amenity,
        "shop": shop,
        "office": office,
        "leisure": leisure,
        "transport": transport,
    }


def _is_poi_cache_valid(df: pd.DataFrame, cities: pd.DataFrame) -> bool:
    required = {
        "city_id",
        "city_name",
        "amenity_count",
        "shop_count",
        "office_count",
        "leisure_count",
        "transport_count",
        "poi_total",
        "poi_diversity",
        "poi_source",
    }
    return required.issubset(df.columns) and set(cities["city_id"]).issubset(set(df["city_id"]))


def collect_city_poi(
    cities: pd.DataFrame,
    config: ProjectConfig,
    use_cache: bool = True,
    strict_real_data: bool = False,
) -> pd.DataFrame:
    """Collect POI aggregate features around each city center."""
    cache_path = DATA_RAW / "city_poi_features.csv"
    cached_other = pd.DataFrame()
    cached_subset = pd.DataFrame()
    cached_ids: set[str] = set()

    if use_cache and cache_path.exists():
        cached = pd.read_csv(cache_path)
        required = {
            "city_id",
            "city_name",
            "amenity_count",
            "shop_count",
            "office_count",
            "leisure_count",
            "transport_count",
            "poi_total",
            "poi_diversity",
            "poi_source",
        }
        if required.issubset(cached.columns):
            cached_other = cached[~cached["city_id"].isin(cities["city_id"])].copy()
            cached_subset = cached[cached["city_id"].isin(cities["city_id"])].copy()
            if strict_real_data:
                cached_subset = cached_subset[cached_subset["poi_source"] == "osm"].copy()
            cached_ids = set(cached_subset["city_id"].unique().tolist())
            if cached_ids:
                LOGGER.info("Using cached POI for %s/%s cities.", len(cached_ids), len(cities))

    missing_cities = cities[~cities["city_id"].isin(cached_ids)].copy()
    strict_skip_live_fetch = str(os.getenv("URBAN_PULSE_STRICT_SKIP_LIVE_POI", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if strict_real_data and len(missing_cities) > 0:
        if strict_skip_live_fetch:
            LOGGER.warning(
                "Strict mode: URBAN_PULSE_STRICT_SKIP_LIVE_POI is enabled; "
                "skipping live POI fetch for %s uncached cities. "
                "These cities will use audited OSM-pool imputation in feature engineering.",
                len(missing_cities),
            )
            out_cached = cached_subset.drop_duplicates(subset=["city_id"], keep="last")
            cache_out = out_cached
            if not cached_other.empty:
                cache_out = pd.concat([cached_other, out_cached], ignore_index=True)
                cache_out = cache_out.drop_duplicates(subset=["city_id"], keep="last")
            cache_out.to_csv(cache_path, index=False)
            return out_cached[out_cached["city_id"].isin(cities["city_id"])].copy()
        LOGGER.info(
            "Strict mode: %s/%s cities have cached OSM POI, trying live OSM fetch for %s missing cities.",
            len(cached_ids),
            len(cities),
            len(missing_cities),
        )

    rows: List[dict] = []
    total_missing = len(missing_cities)
    failures: List[str] = []
    consecutive_failures = 0
    strict_sleep = max(0.10, _env_float(os.getenv("URBAN_PULSE_STRICT_POI_SLEEP"), 0.25))
    strict_max_consecutive_failures = max(3, _env_int(os.getenv("URBAN_PULSE_STRICT_POI_MAX_CONSEC_FAIL"), 15))
    strict_max_total_failures = max(10, _env_int(os.getenv("URBAN_PULSE_STRICT_POI_MAX_TOTAL_FAIL"), 80))
    if strict_real_data and total_missing > 0:
        LOGGER.info(
            "Strict POI fetch settings: timeout=%ss retries=%s backoff=%.2f sleep=%.2fs max_consec_fail=%s max_total_fail=%s",
            max(8, _env_int(os.getenv("URBAN_PULSE_STRICT_POI_TIMEOUT"), 14)),
            max(1, _env_int(os.getenv("URBAN_PULSE_STRICT_POI_RETRIES"), 2)),
            max(0.2, _env_float(os.getenv("URBAN_PULSE_STRICT_POI_BACKOFF"), 1.0)),
            strict_sleep,
            strict_max_consecutive_failures,
            strict_max_total_failures,
        )

    for i, city in enumerate(missing_cities.to_dict(orient="records"), start=1):
        if i > 1:
            time.sleep(strict_sleep if strict_real_data else 0.35)

        try:
            counts = _fetch_city_poi(
                float(city["latitude"]),
                float(city["longitude"]),
                config.api.overpass_url,
                strict_real_data=strict_real_data,
            )
            source = "osm"
            consecutive_failures = 0
        except Exception as exc:  # noqa: BLE001
            if strict_real_data:
                LOGGER.error("Strict mode POI failure for %s: %s", city["city_name"], exc)
                failures.append(city["city_name"])
                consecutive_failures += 1
                if i % 10 == 0 or i == total_missing:
                    LOGGER.info("POI incremental progress: %s/%s cities (strict failures=%s)", i, total_missing, len(failures))
                    checkpoint_new = pd.DataFrame(rows)
                    if cached_subset.empty:
                        checkpoint = checkpoint_new
                    else:
                        checkpoint = pd.concat([cached_subset, checkpoint_new], ignore_index=True)
                    checkpoint = checkpoint.drop_duplicates(subset=["city_id"], keep="last")
                    if not cached_other.empty:
                        checkpoint = pd.concat([cached_other, checkpoint], ignore_index=True)
                        checkpoint = checkpoint.drop_duplicates(subset=["city_id"], keep="last")
                    checkpoint.to_csv(cache_path, index=False)
                if (
                    consecutive_failures >= strict_max_consecutive_failures
                    or len(failures) >= strict_max_total_failures
                ):
                    LOGGER.warning(
                        "POI strict-mode circuit breaker triggered (consecutive_failures=%s/%s, total_failures=%s/%s). "
                        "Skipping remaining %s cities in this run.",
                        consecutive_failures,
                        strict_max_consecutive_failures,
                        len(failures),
                        strict_max_total_failures,
                        max(0, total_missing - i),
                    )
                    break
                continue
            LOGGER.warning("POI fallback for %s: %s", city["city_name"], exc)
            counts = _synthetic_city_poi(pd.Series(city), seed=config.random_seed + 500 + i)
            source = "synthetic"

        total_count = float(sum(counts.values()))
        probs = np.array([counts["amenity"], counts["shop"], counts["office"], counts["leisure"]], dtype=float)
        probs = probs / max(1.0, probs.sum())
        shannon = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())

        rows.append(
            {
                "city_id": city["city_id"],
                "city_name": city["city_name"],
                "amenity_count": counts["amenity"],
                "shop_count": counts["shop"],
                "office_count": counts["office"],
                "leisure_count": counts["leisure"],
                "transport_count": counts["transport"],
                "poi_total": total_count,
                "poi_diversity": shannon,
                "poi_source": source,
            }
        )

        if i % 10 == 0 or i == total_missing:
            LOGGER.info("POI incremental progress: %s/%s cities", i, total_missing)
            checkpoint_new = pd.DataFrame(rows)
            if cached_subset.empty:
                checkpoint = checkpoint_new
            else:
                checkpoint = pd.concat([cached_subset, checkpoint_new], ignore_index=True)
            checkpoint = checkpoint.drop_duplicates(subset=["city_id"], keep="last")
            if not cached_other.empty:
                checkpoint = pd.concat([cached_other, checkpoint], ignore_index=True)
                checkpoint = checkpoint.drop_duplicates(subset=["city_id"], keep="last")
            checkpoint.to_csv(cache_path, index=False)

    new_rows = pd.DataFrame(rows)
    if cached_subset.empty:
        out = new_rows
    elif new_rows.empty:
        out = cached_subset
    else:
        out = pd.concat([cached_subset, new_rows], ignore_index=True)

    if strict_real_data and failures:
        sample = ", ".join(failures[:10]) + (" ..." if len(failures) > 10 else "")
        LOGGER.warning(
            "Strict mode POI incomplete: %s failed cities were excluded from objective sample. Sample: %s",
            len(failures),
            sample,
        )

    out = out.drop_duplicates(subset=["city_id"], keep="last")
    cache_out = out
    if not cached_other.empty:
        cache_out = pd.concat([cached_other, out], ignore_index=True)
        cache_out = cache_out.drop_duplicates(subset=["city_id"], keep="last")
    cache_out.to_csv(cache_path, index=False)

    LOGGER.info("City POI features generated: %s cities", len(out))
    return out[out["city_id"].isin(cities["city_id"])].copy()


def collect_city_poi_year_panel(
    cities: pd.DataFrame,
    config: ProjectConfig,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load historical city-year POI aggregates if an annual panel is provided."""
    candidates = [
        DATA_RAW / "city_poi_features_yearly.csv",
        DATA_RAW / "city_poi_yearly.csv",
        DATA_RAW / "poi_city_yearly.csv",
        DATA_RAW / "ohsome_poi_yearly.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    full = _city_year_full_grid(cities, config)
    default_out = full.copy()
    for col in [
        "amenity_count",
        "shop_count",
        "office_count",
        "leisure_count",
        "transport_count",
        "poi_total",
        "poi_diversity",
    ]:
        default_out[col] = np.nan
    default_out["poi_source"] = "missing"
    default_out["poi_temporal_source"] = "missing"

    if (not use_cache) or path is None:
        return default_out

    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read historical POI year panel (%s): %s", path, exc)
        return default_out

    raw = _attach_city_id_from_name(raw, cities)
    if "year" not in raw.columns and "date" in raw.columns:
        raw["year"] = pd.to_datetime(raw["date"], errors="coerce").dt.year
    if "city_id" not in raw.columns or "year" not in raw.columns:
        return default_out

    work = raw.copy()
    work["city_id"] = work["city_id"].astype(str)
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work = work[work["city_id"].isin(cities["city_id"].astype(str))].copy()
    work = work[work["year"].between(config.time_range.start_year, config.time_range.end_year, inclusive="both")].copy()
    if work.empty:
        return default_out
    work["year"] = work["year"].astype(int)

    poi_aliases = {
        "amenity_count": ["amenity_count", "amenity", "amenities"],
        "shop_count": ["shop_count", "shop", "shops"],
        "office_count": ["office_count", "office", "offices"],
        "leisure_count": ["leisure_count", "leisure"],
        "transport_count": ["transport_count", "transport", "transit_count"],
        "poi_total": ["poi_total", "total_poi_count", "poi_count"],
        "poi_diversity": ["poi_diversity", "shannon_diversity", "diversity"],
    }

    std = work[["city_id", "year"]].copy()
    for target, aliases in poi_aliases.items():
        col = _resolve_column_alias(work, aliases)
        std[target] = pd.to_numeric(work[col], errors="coerce") if col else np.nan
    if "poi_total" in std.columns:
        total_from_parts = (
            std[["amenity_count", "shop_count", "office_count", "leisure_count", "transport_count"]]
            .sum(axis=1, min_count=1)
        )
        std["poi_total"] = std["poi_total"].combine_first(total_from_parts)
    if "poi_diversity" in std.columns:
        base = std[["amenity_count", "shop_count", "office_count", "leisure_count"]].clip(lower=0.0)
        probs = base.div(base.sum(axis=1).replace(0.0, np.nan), axis=0)
        shannon = -(probs * np.log(probs)).sum(axis=1, min_count=1)
        std["poi_diversity"] = std["poi_diversity"].combine_first(shannon)

    source_col = _resolve_column_alias(work, ["poi_source", "source"])
    if source_col:
        std["poi_source"] = work[source_col].fillna("historical_poi_year_panel").astype(str)
    else:
        std["poi_source"] = "historical_poi_year_panel"
    std["poi_temporal_source"] = "historical_year_panel"

    agg = (
        std.groupby(["city_id", "year"], as_index=False)
        .agg(
            amenity_count=("amenity_count", "mean"),
            shop_count=("shop_count", "mean"),
            office_count=("office_count", "mean"),
            leisure_count=("leisure_count", "mean"),
            transport_count=("transport_count", "mean"),
            poi_total=("poi_total", "mean"),
            poi_diversity=("poi_diversity", "mean"),
            poi_source=("poi_source", "first"),
            poi_temporal_source=("poi_temporal_source", "first"),
        )
        .sort_values(["city_id", "year"])
    )

    merged = full.merge(agg, on=["city_id", "year"], how="left")
    for col in [
        "amenity_count",
        "shop_count",
        "office_count",
        "leisure_count",
        "transport_count",
        "poi_total",
        "poi_diversity",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["poi_source"] = merged["poi_source"].fillna("missing").astype(str)
    merged["poi_temporal_source"] = merged["poi_temporal_source"].fillna("missing").astype(str)
    return merged


def _resolve_column_alias(df: pd.DataFrame, aliases: List[str]) -> str | None:
    for col in aliases:
        if col in df.columns:
            return col
    return None


def _attach_city_id_from_name(df: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    if "city_id" in df.columns:
        out = df.copy()
        out["city_id"] = out["city_id"].astype(str)
        return out
    if "city_name" not in df.columns:
        return df.copy()
    mapper = cities[["city_id", "city_name"]].copy()
    mapper["city_name_l"] = mapper["city_name"].astype(str).str.strip().str.lower()
    out = df.copy()
    out["city_name_l"] = out["city_name"].astype(str).str.strip().str.lower()
    out = out.merge(mapper[["city_id", "city_name_l"]], on="city_name_l", how="left")
    out = out.drop(columns=["city_name_l"], errors="ignore")
    return out


def _city_year_full_grid(cities: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    years = list(range(int(config.time_range.start_year), int(config.time_range.end_year) + 1))
    full = pd.MultiIndex.from_product(
        [cities["city_id"].astype(str).tolist(), years],
        names=["city_id", "year"],
    ).to_frame(index=False)
    return full


def collect_city_macro_observed_panel(
    cities: pd.DataFrame,
    config: ProjectConfig,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load optional city-year macro measurements from local observed file.

    Expected raw file: data/raw/city_macro_observed.csv
    Required keys: city_id (or city_name), year.
    Supported columns (with aliases): gdp_per_capita, population, unemployment,
    internet_users, capital_formation, inflation. Optional metadata columns:
    macro_resolution_level, oecd_fua_code, oecd_city_code, oecd_fua_name,
    oecd_city_name. Optional GDP extras: gdp_total_ppp_observed,
    gdp_total_local_observed, gdp_share_national_observed.
    """
    path = DATA_RAW / "city_macro_observed.csv"
    full = _city_year_full_grid(cities, config)
    out = full.copy()
    for col in MACRO_BASE_COLUMNS:
        out[col] = np.nan
    for col in CITY_MACRO_EXTRA_NUMERIC_COLUMNS:
        out[col] = np.nan
    for col in CITY_MACRO_EXTRA_STRING_COLUMNS:
        out[col] = pd.NA
    out["macro_observed_source"] = "missing"
    out["macro_resolution_level"] = "country_year"
    out["city_macro_observed_flag"] = 0

    if (not use_cache) or (not path.exists()):
        return out

    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read city macro observed panel (%s): %s", path, exc)
        return out

    raw = _attach_city_id_from_name(raw, cities)
    if "year" not in raw.columns and "date" in raw.columns:
        raw["year"] = pd.to_datetime(raw["date"], errors="coerce").dt.year
    if "city_id" not in raw.columns or "year" not in raw.columns:
        LOGGER.warning("City macro observed panel missing required keys (city_id/year): %s", path)
        return out

    work = raw.copy()
    work["city_id"] = work["city_id"].astype(str)
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work = work[work["city_id"].isin(cities["city_id"].astype(str))].copy()
    work = work[work["year"].between(config.time_range.start_year, config.time_range.end_year, inclusive="both")].copy()
    if work.empty:
        return out
    work["year"] = work["year"].astype(int)

    cols = ["city_id", "year"]
    for target_col, aliases in CITY_MACRO_COLUMN_ALIASES.items():
        col = _resolve_column_alias(work, aliases)
        if col is None:
            work[target_col] = np.nan
        else:
            work[target_col] = pd.to_numeric(work[col], errors="coerce")
        cols.append(target_col)
    for target_col, aliases in CITY_MACRO_EXTRA_COLUMN_ALIASES.items():
        col = _resolve_column_alias(work, aliases)
        if col is None:
            work[target_col] = np.nan
        else:
            work[target_col] = pd.to_numeric(work[col], errors="coerce")
        cols.append(target_col)

    source_col = None
    for candidate in ["macro_observed_source", "city_macro_source", "macro_source", "source"]:
        if candidate in work.columns:
            source_col = candidate
            break
    if source_col is None:
        work["macro_observed_source"] = "city_observed_external"
    else:
        work["macro_observed_source"] = work[source_col].astype(str)
    cols.append("macro_observed_source")

    resolution_col = None
    for candidate in ["macro_resolution_level", "macro_resolution", "resolution_level"]:
        if candidate in work.columns:
            resolution_col = candidate
            break
    if resolution_col is None:
        work["macro_resolution_level"] = "city_observed"
    else:
        work["macro_resolution_level"] = work[resolution_col].astype(str)
    cols.append("macro_resolution_level")

    for col in CITY_MACRO_EXTRA_STRING_COLUMNS:
        if col in work.columns:
            work[col] = work[col].astype(str)
        else:
            work[col] = pd.NA
        cols.append(col)

    agg_spec: Dict[str, tuple[str, str]] = {c: (c, "mean") for c in MACRO_BASE_COLUMNS}
    agg_spec.update({c: (c, "mean") for c in CITY_MACRO_EXTRA_NUMERIC_COLUMNS})
    agg_spec["macro_observed_source"] = ("macro_observed_source", "first")
    agg_spec["macro_resolution_level"] = ("macro_resolution_level", "first")
    for col in CITY_MACRO_EXTRA_STRING_COLUMNS:
        agg_spec[col] = (col, "first")
    agg = (
        work[cols]
        .groupby(["city_id", "year"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["city_id", "year"])
    )

    merged = full.merge(agg, on=["city_id", "year"], how="left")
    for col in MACRO_BASE_COLUMNS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    for col in CITY_MACRO_EXTRA_NUMERIC_COLUMNS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["macro_observed_source"] = merged["macro_observed_source"].fillna("missing").astype(str)
    merged["macro_resolution_level"] = merged["macro_resolution_level"].fillna("country_year").astype(str)
    for col in CITY_MACRO_EXTRA_STRING_COLUMNS:
        merged[col] = merged[col].where(pd.notna(merged[col]), pd.NA)
    observed_presence_cols = MACRO_BASE_COLUMNS + CITY_MACRO_EXTRA_NUMERIC_COLUMNS
    merged["city_macro_observed_flag"] = merged[observed_presence_cols].notna().any(axis=1).astype(int)
    missing_mask = merged["city_macro_observed_flag"] <= 0
    merged.loc[missing_mask, "macro_observed_source"] = "missing"
    merged.loc[missing_mask, "macro_resolution_level"] = "country_year"
    return merged


def collect_city_road_network_panel(
    cities: pd.DataFrame,
    config: ProjectConfig,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load city-year road metrics from local file and standardize columns.

    Expected raw file: data/raw/city_road_network_yearly.csv
    Required keys: city_id (or city_name), year.
    """
    path = DATA_RAW / "city_road_network_yearly.csv"
    full = _city_year_full_grid(cities, config)
    if (not use_cache) or (not path.exists()):
        out = full.copy()
        out["road_length_km_total"] = np.nan
        out["arterial_share"] = np.nan
        out["intersection_density"] = np.nan
        out["road_source"] = "missing"
        return out

    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read road panel cache (%s): %s", path, exc)
        out = full.copy()
        out["road_length_km_total"] = np.nan
        out["arterial_share"] = np.nan
        out["intersection_density"] = np.nan
        out["road_source"] = "missing"
        return out

    raw = _attach_city_id_from_name(raw, cities)
    if "year" not in raw.columns and "date" in raw.columns:
        raw["year"] = pd.to_datetime(raw["date"], errors="coerce").dt.year
    if "year" not in raw.columns or "city_id" not in raw.columns:
        out = full.copy()
        out["road_length_km_total"] = np.nan
        out["arterial_share"] = np.nan
        out["intersection_density"] = np.nan
        out["road_source"] = "missing"
        return out

    out = raw.copy()
    out["city_id"] = out["city_id"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out = out[out["city_id"].isin(cities["city_id"].astype(str))].copy()
    out = out[out["year"].between(config.time_range.start_year, config.time_range.end_year, inclusive="both")].copy()
    out["year"] = out["year"].astype(int)

    road_len_col = _resolve_column_alias(out, ROAD_COLUMN_ALIASES["road_length_km_total"])
    arterial_col = _resolve_column_alias(out, ROAD_COLUMN_ALIASES["arterial_share"])
    inter_col = _resolve_column_alias(out, ROAD_COLUMN_ALIASES["intersection_density"])

    road_std = out[["city_id", "year"]].copy()
    road_std["road_length_km_total"] = pd.to_numeric(out[road_len_col], errors="coerce") if road_len_col else np.nan
    road_std["arterial_share"] = pd.to_numeric(out[arterial_col], errors="coerce") if arterial_col else np.nan
    road_std["intersection_density"] = pd.to_numeric(out[inter_col], errors="coerce") if inter_col else np.nan
    if "road_source" in out.columns:
        road_std["road_source"] = out["road_source"].fillna("external_road_dataset").astype(str)
    else:
        road_std["road_source"] = "external_road_dataset"

    road_std = (
        road_std.groupby(["city_id", "year"], as_index=False)
        .agg(
            road_length_km_total=("road_length_km_total", "mean"),
            arterial_share=("arterial_share", "mean"),
            intersection_density=("intersection_density", "mean"),
            road_source=("road_source", "first"),
        )
    )

    merged = full.merge(road_std, on=["city_id", "year"], how="left")
    for col in ["road_length_km_total", "arterial_share", "intersection_density"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged[col] = merged.groupby("city_id")[col].transform(lambda s: s.interpolate(limit_direction="both"))
    merged["road_source"] = merged["road_source"].fillna("missing").astype(str)
    return merged


def collect_city_viirs_year_panel(
    cities: pd.DataFrame,
    config: ProjectConfig,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load VIIRS city-year data from annual panels or aggregate monthly panels."""
    yearly_candidates = [
        DATA_RAW / "viirs_city_yearly.csv",
        DATA_RAW / "city_nightlights_yearly.csv",
        DATA_RAW / "viirs_annual_vnl_city.csv",
    ]
    monthly_candidates = [
        DATA_RAW / "viirs_city_monthly.csv",
        DATA_RAW / "city_nightlights_monthly.csv",
    ]
    yearly_path = next((p for p in yearly_candidates if p.exists()), None)
    monthly_path = next((p for p in monthly_candidates if p.exists()), None)
    path = yearly_path if yearly_path is not None else monthly_path
    full = _city_year_full_grid(cities, config)
    def _empty_viirs_frame() -> pd.DataFrame:
        out = full.copy()
        out["viirs_ntl_mean"] = np.nan
        out["viirs_ntl_sum"] = np.nan
        out["viirs_ntl_p90"] = np.nan
        out["viirs_lit_area_km2"] = np.nan
        out["viirs_log_mean"] = np.nan
        out["viirs_intra_year_recovery"] = np.nan
        out["viirs_intra_year_decline"] = np.nan
        out["viirs_recent_drop"] = np.nan
        out["viirs_ntl_yoy"] = np.nan
        out["viirs_month_count"] = 0.0
        out["viirs_year_coverage_share"] = 0.0
        out["has_viirs_observation"] = 0
        out["viirs_source"] = "missing"
        return out

    if (not use_cache) or path is None:
        return _empty_viirs_frame()

    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read VIIRS monthly cache (%s): %s", path, exc)
        return _empty_viirs_frame()

    raw = _attach_city_id_from_name(raw, cities)
    if "year" not in raw.columns:
        date_col = _resolve_column_alias(raw, ["date", "month_date", "timestamp"])
        if date_col:
            raw["year"] = pd.to_datetime(raw[date_col], errors="coerce").dt.year
    if "city_id" not in raw.columns or "year" not in raw.columns:
        return _empty_viirs_frame()

    rad_col = _resolve_column_alias(raw, VIIRS_RADIANCE_ALIASES)
    if rad_col is None:
        return _empty_viirs_frame()
    area_col = _resolve_column_alias(raw, VIIRS_LIT_AREA_ALIASES)
    sum_col = _resolve_column_alias(raw, VIIRS_SUM_ALIASES)
    p90_col = _resolve_column_alias(raw, ["viirs_ntl_p90", "radiance_p90", "ntl_p90", "p90_radiance"])

    def _nan_p90(series: pd.Series) -> float:
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan")
        return float(np.quantile(vals, 0.90))

    def _viirs_monthly_features(sub: pd.DataFrame) -> pd.Series:
        vals = pd.to_numeric(sub["viirs_ntl_mean"], errors="coerce").clip(lower=0.0)
        logs = np.log1p(vals.to_numpy(dtype=float))
        month = pd.to_numeric(sub.get("month"), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(logs) & np.isfinite(month)
        slope = float("nan")
        if int(valid.sum()) >= 2:
            slope = float(np.polyfit(month[valid], logs[valid], deg=1)[0])
        last_log = float("nan")
        if int(valid.sum()) >= 1:
            ordered = sub.loc[valid].copy()
            ordered["month_num"] = pd.to_numeric(ordered.get("month"), errors="coerce")
            ordered = ordered.sort_values("month_num")
            last_log = float(np.log1p(pd.to_numeric(ordered["viirs_ntl_mean"], errors="coerce").clip(lower=0.0).iloc[-1]))
        peak_log = float(np.nanmax(logs[valid])) if int(valid.sum()) >= 1 else float("nan")
        recent_drop = float(max(peak_log - last_log, 0.0)) if np.isfinite(peak_log) and np.isfinite(last_log) else float("nan")
        return pd.Series(
            {
                "viirs_intra_year_recovery": float(max(slope, 0.0)) if np.isfinite(slope) else float("nan"),
                "viirs_intra_year_decline": float(max(-slope, 0.0)) if np.isfinite(slope) else float("nan"),
                "viirs_recent_drop": recent_drop,
            }
        )

    work = raw.copy()
    work["city_id"] = work["city_id"].astype(str)
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work = work[work["city_id"].isin(cities["city_id"].astype(str))].copy()
    work = work[work["year"].between(config.time_range.start_year, config.time_range.end_year, inclusive="both")].copy()
    work["year"] = work["year"].astype(int)
    work["viirs_ntl_mean"] = pd.to_numeric(work[rad_col], errors="coerce")
    if yearly_path is None and "month" not in work.columns:
        month_col = _resolve_column_alias(work, ["month", "month_num"])
        if month_col is None:
            date_col = _resolve_column_alias(work, ["date", "month_date", "timestamp"])
            if date_col:
                work["month"] = pd.to_datetime(work[date_col], errors="coerce").dt.month
        elif month_col != "month":
            work["month"] = pd.to_numeric(work[month_col], errors="coerce")
    if area_col:
        work["viirs_lit_area_km2"] = pd.to_numeric(work[area_col], errors="coerce")
    else:
        work["viirs_lit_area_km2"] = np.nan
    if sum_col:
        work["viirs_ntl_sum"] = pd.to_numeric(work[sum_col], errors="coerce")
    else:
        work["viirs_ntl_sum"] = np.nan
    if pd.to_numeric(work["viirs_ntl_sum"], errors="coerce").notna().sum() == 0:
        work["viirs_ntl_sum"] = (
            pd.to_numeric(work["viirs_ntl_mean"], errors="coerce").clip(lower=0.0)
            * pd.to_numeric(work["viirs_lit_area_km2"], errors="coerce").clip(lower=0.0)
        )
    work["viirs_ntl_p90_input"] = pd.to_numeric(work[p90_col], errors="coerce") if p90_col else np.nan
    work["viirs_source"] = (
        work["viirs_source"].astype(str)
        if "viirs_source" in work.columns
        else ("viirs_annual_panel" if yearly_path is not None else "viirs_monthly_aggregate")
    )

    agg_spec = {
        "viirs_ntl_mean": ("viirs_ntl_mean", "mean"),
        "viirs_ntl_sum": ("viirs_ntl_sum", "mean"),
        "viirs_ntl_p90": ("viirs_ntl_p90_input", "mean") if p90_col else ("viirs_ntl_mean", _nan_p90),
        "viirs_lit_area_km2": ("viirs_lit_area_km2", "mean"),
        "viirs_source": ("viirs_source", "first"),
    }
    if yearly_path is None and "month" in work.columns:
        agg_spec["viirs_month_count"] = ("month", "nunique")
    else:
        work["viirs_month_count"] = 12
        agg_spec["viirs_month_count"] = ("viirs_month_count", "mean")

    agg = work.groupby(["city_id", "year"], as_index=False).agg(**agg_spec).sort_values(["city_id", "year"])
    agg["viirs_log_mean"] = np.log1p(agg["viirs_ntl_mean"].clip(lower=0.0))
    if yearly_path is None and "month" in work.columns:
        monthly_features = (
            work.groupby(["city_id", "year"])[["viirs_ntl_mean", "month"]]
            .apply(_viirs_monthly_features)
            .reset_index()
        )
        if "level_2" in monthly_features.columns:
            monthly_features = monthly_features.drop(columns=["level_2"])
        agg = agg.merge(monthly_features, on=["city_id", "year"], how="left")
    else:
        agg["viirs_intra_year_recovery"] = np.nan
        agg["viirs_intra_year_decline"] = np.nan
        agg["viirs_recent_drop"] = np.nan
    agg["viirs_ntl_yoy"] = agg.groupby("city_id")["viirs_ntl_mean"].diff()

    merged = full.merge(agg, on=["city_id", "year"], how="left")
    for col in [
        "viirs_ntl_mean",
        "viirs_ntl_sum",
        "viirs_ntl_p90",
        "viirs_lit_area_km2",
        "viirs_log_mean",
        "viirs_intra_year_recovery",
        "viirs_intra_year_decline",
        "viirs_recent_drop",
        "viirs_ntl_yoy",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["viirs_month_count"] = pd.to_numeric(merged.get("viirs_month_count"), errors="coerce").fillna(0.0)
    merged["has_viirs_observation"] = (
        merged["viirs_ntl_mean"].notna() | merged["viirs_ntl_sum"].notna() | (merged["viirs_month_count"] > 0.0)
    ).astype(int)
    merged["viirs_year_coverage_share"] = np.clip(merged["viirs_month_count"] / 12.0, 0.0, 1.0)
    merged["viirs_source"] = merged["viirs_source"].fillna("missing").astype(str)
    return merged


def collect_city_ghsl_year_panel(
    cities: pd.DataFrame,
    config: ProjectConfig,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load city-year GHSL built-surface/built-volume panel from local files."""
    candidates = [
        DATA_RAW / "city_ghsl_yearly.csv",
        DATA_RAW / "ghsl_city_yearly.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    full = _city_year_full_grid(cities, config)
    default_cols = list(GHSL_ALIASES.keys()) + ["ghsl_built_density", "ghsl_built_surface_yoy", "ghsl_built_volume_yoy"]
    if (not use_cache) or path is None:
        out = full.copy()
        for col in default_cols:
            out[col] = np.nan
        out["ghsl_source"] = "missing"
        return out

    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read GHSL city-year cache (%s): %s", path, exc)
        out = full.copy()
        for col in default_cols:
            out[col] = np.nan
        out["ghsl_source"] = "missing"
        return out

    raw = _attach_city_id_from_name(raw, cities)
    if "year" not in raw.columns:
        year_col = _resolve_column_alias(raw, ["epoch", "ghsl_year"])
        if year_col:
            raw["year"] = pd.to_numeric(raw[year_col], errors="coerce")
    if "city_id" not in raw.columns or "year" not in raw.columns:
        out = full.copy()
        for col in default_cols:
            out[col] = np.nan
        out["ghsl_source"] = "missing"
        return out

    work = raw.copy()
    work["city_id"] = work["city_id"].astype(str)
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work = work[work["city_id"].isin(cities["city_id"].astype(str))].copy()
    work = work[work["year"].between(config.time_range.start_year, config.time_range.end_year, inclusive="both")].copy()
    work["year"] = work["year"].astype(int)

    for target, aliases in GHSL_ALIASES.items():
        col = _resolve_column_alias(work, aliases)
        work[target] = pd.to_numeric(work[col], errors="coerce") if col else np.nan
    if "ghsl_source" in work.columns:
        work["ghsl_source"] = work["ghsl_source"].fillna("gee_ghsl_p2023a").astype(str)
    else:
        work["ghsl_source"] = "gee_ghsl_p2023a"

    agg = (
        work[
            [
                "city_id",
                "year",
                "ghsl_built_surface_km2",
                "ghsl_built_surface_nres_km2",
                "ghsl_built_volume_m3",
                "ghsl_built_volume_nres_m3",
                "ghsl_source",
            ]
        ]
        .groupby(["city_id", "year"], as_index=False)
        .agg(
            ghsl_built_surface_km2=("ghsl_built_surface_km2", "mean"),
            ghsl_built_surface_nres_km2=("ghsl_built_surface_nres_km2", "mean"),
            ghsl_built_volume_m3=("ghsl_built_volume_m3", "mean"),
            ghsl_built_volume_nres_m3=("ghsl_built_volume_nres_m3", "mean"),
            ghsl_source=("ghsl_source", "first"),
        )
        .sort_values(["city_id", "year"])
    )

    merged = full.merge(agg, on=["city_id", "year"], how="left")
    for col in [
        "ghsl_built_surface_km2",
        "ghsl_built_surface_nres_km2",
        "ghsl_built_volume_m3",
        "ghsl_built_volume_nres_m3",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        if merged[col].notna().sum() >= 2:
            merged[col] = merged.groupby("city_id")[col].transform(lambda s: s.interpolate(limit_area="inside"))

    merged["ghsl_built_density"] = (
        pd.to_numeric(merged["ghsl_built_volume_m3"], errors="coerce")
        / np.maximum(pd.to_numeric(merged["ghsl_built_surface_km2"], errors="coerce") * 1_000_000.0, 1.0)
    )
    merged["ghsl_built_surface_yoy"] = merged.groupby("city_id")["ghsl_built_surface_km2"].diff()
    merged["ghsl_built_volume_yoy"] = merged.groupby("city_id")["ghsl_built_volume_m3"].diff()
    merged["ghsl_source"] = merged["ghsl_source"].fillna("missing").astype(str)
    return merged


def collect_city_no2_year_panel(
    cities: pd.DataFrame,
    config: ProjectConfig,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load monthly city NO2 and aggregate it to city-year fast-signal features."""
    candidates = [
        DATA_RAW / "city_no2_monthly.csv",
        DATA_RAW / "no2_city_monthly.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    full = _city_year_full_grid(cities, config)

    def _empty_no2_frame() -> pd.DataFrame:
        out = full.copy()
        for col in [
            "no2_trop_mean",
            "no2_trop_p90",
            "no2_trop_yoy_mean",
            "no2_trop_anomaly_mean",
            "no2_trop_anomaly_abs_mean",
            "no2_recent_spike",
        ]:
            out[col] = np.nan
        out["no2_month_count"] = 0.0
        out["no2_year_coverage_share"] = 0.0
        out["has_no2_observation"] = 0
        out["no2_source"] = "missing"
        return out

    if (not use_cache) or path is None:
        return _empty_no2_frame()

    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read NO2 monthly cache (%s): %s", path, exc)
        return _empty_no2_frame()

    raw = _attach_city_id_from_name(raw, cities)
    if "year" not in raw.columns:
        date_col = _resolve_column_alias(raw, ["date", "month_date", "timestamp"])
        if date_col:
            raw["year"] = pd.to_datetime(raw[date_col], errors="coerce").dt.year
            raw["month"] = pd.to_datetime(raw[date_col], errors="coerce").dt.month
    if "month" not in raw.columns:
        month_col = _resolve_column_alias(raw, ["month", "month_num"])
        if month_col and month_col != "month":
            raw["month"] = pd.to_numeric(raw[month_col], errors="coerce")
    if "city_id" not in raw.columns or "year" not in raw.columns or "month" not in raw.columns:
        return _empty_no2_frame()

    mean_col = _resolve_column_alias(raw, NO2_ALIASES["no2_trop_mean"])
    if mean_col is None:
        return _empty_no2_frame()
    count_col = _resolve_column_alias(raw, NO2_ALIASES["no2_valid_obs_count"])

    work = raw.copy()
    work["city_id"] = work["city_id"].astype(str)
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work["month"] = pd.to_numeric(work["month"], errors="coerce").astype("Int64")
    work = work[work["city_id"].isin(cities["city_id"].astype(str))].copy()
    work = work[work["year"].between(config.time_range.start_year, config.time_range.end_year, inclusive="both")].copy()
    work = work[(work["month"] >= 1) & (work["month"] <= 12)].copy()
    if work.empty:
        return _empty_no2_frame()

    work["year"] = work["year"].astype(int)
    work["month"] = work["month"].astype(int)
    work["no2_trop_mean"] = pd.to_numeric(work[mean_col], errors="coerce")
    work["no2_valid_obs_count"] = pd.to_numeric(work[count_col], errors="coerce") if count_col else np.nan
    if "no2_source" in work.columns:
        work["no2_source"] = work["no2_source"].fillna("gee_no2_monthly").astype(str)
    else:
        work["no2_source"] = "gee_no2_monthly"

    work = work.sort_values(["city_id", "year", "month"]).reset_index(drop=True)
    month_baseline = work.groupby(["city_id", "month"])["no2_trop_mean"].transform("mean")
    work["no2_trop_anomaly"] = pd.to_numeric(work["no2_trop_mean"], errors="coerce") - month_baseline
    work["no2_trop_yoy"] = work.groupby("city_id")["no2_trop_mean"].diff(12)

    monthly_out = work[
        [
            "city_id",
            "year",
            "month",
            "no2_trop_mean",
            "no2_valid_obs_count",
            "no2_trop_anomaly",
            "no2_trop_yoy",
            "no2_source",
        ]
    ].copy()
    monthly_out.to_csv(DATA_PROCESSED / "city_no2_monthly_anomaly.csv", index=False)

    def _nan_p90(series: pd.Series) -> float:
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan")
        return float(np.quantile(vals, 0.90))

    agg = (
        work.groupby(["city_id", "year"], as_index=False)
        .agg(
            no2_trop_mean=("no2_trop_mean", "mean"),
            no2_trop_p90=("no2_trop_mean", _nan_p90),
            no2_trop_yoy_mean=("no2_trop_yoy", "mean"),
            no2_trop_anomaly_mean=("no2_trop_anomaly", "mean"),
            no2_trop_anomaly_abs_mean=("no2_trop_anomaly", lambda s: float(np.nanmean(np.abs(pd.to_numeric(s, errors="coerce"))))),
            no2_recent_spike=("no2_trop_anomaly", "max"),
            no2_month_count=("month", "nunique"),
            no2_source=("no2_source", "first"),
        )
        .sort_values(["city_id", "year"])
    )
    merged = full.merge(agg, on=["city_id", "year"], how="left")
    for col in [
        "no2_trop_mean",
        "no2_trop_p90",
        "no2_trop_yoy_mean",
        "no2_trop_anomaly_mean",
        "no2_trop_anomaly_abs_mean",
        "no2_recent_spike",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["no2_month_count"] = pd.to_numeric(merged.get("no2_month_count"), errors="coerce").fillna(0.0)
    merged["no2_year_coverage_share"] = np.clip(merged["no2_month_count"] / 12.0, 0.0, 1.0)
    merged["has_no2_observation"] = (
        merged["no2_trop_mean"].notna() | (merged["no2_month_count"] > 0.0)
    ).astype(int)
    merged["no2_source"] = merged["no2_source"].fillna("missing").astype(str)
    return merged


def collect_city_connectivity_panel(
    cities: pd.DataFrame,
    config: ProjectConfig,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load city network-connectivity data, allowing yearly or broadcast cross-section."""
    candidates = [
        DATA_RAW / "city_connectivity_yearly.csv",
        DATA_RAW / "city_flight_connectivity_yearly.csv",
        DATA_RAW / "city_connectivity.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    full = _city_year_full_grid(cities, config)

    def _empty_conn_frame() -> pd.DataFrame:
        out = full.copy()
        out["flight_connectivity_total"] = np.nan
        out["flight_degree_centrality"] = np.nan
        out["airport_count_mapped"] = np.nan
        out["international_route_share"] = np.nan
        out["shipping_connectivity_total"] = np.nan
        out["network_connectivity_source"] = "missing"
        return out

    if (not use_cache) or path is None:
        return _empty_conn_frame()

    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read connectivity panel (%s): %s", path, exc)
        return _empty_conn_frame()

    raw = _attach_city_id_from_name(raw, cities)
    if "city_id" not in raw.columns:
        return _empty_conn_frame()
    work = raw.copy()
    work["city_id"] = work["city_id"].astype(str)
    if "year" not in work.columns:
        years = list(range(int(config.time_range.start_year), int(config.time_range.end_year) + 1))
        work = (
            work.assign(_k=1)
            .merge(pd.DataFrame({"year": years, "_k": 1}), on="_k", how="inner")
            .drop(columns="_k")
        )
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work = work[work["city_id"].isin(cities["city_id"].astype(str))].copy()
    work = work[work["year"].between(config.time_range.start_year, config.time_range.end_year, inclusive="both")].copy()
    if work.empty:
        return _empty_conn_frame()
    work["year"] = work["year"].astype(int)

    for target, aliases in CONNECTIVITY_ALIASES.items():
        col = _resolve_column_alias(work, aliases)
        work[target] = pd.to_numeric(work[col], errors="coerce") if col else np.nan
    if "network_connectivity_source" in work.columns:
        work["network_connectivity_source"] = work["network_connectivity_source"].fillna("external_connectivity").astype(str)
    else:
        work["network_connectivity_source"] = "external_connectivity"
    agg = (
        work.groupby(["city_id", "year"], as_index=False)
        .agg(
            flight_connectivity_total=("flight_connectivity_total", "mean"),
            flight_degree_centrality=("flight_degree_centrality", "mean"),
            airport_count_mapped=("airport_count_mapped", "mean"),
            international_route_share=("international_route_share", "mean"),
            shipping_connectivity_total=("shipping_connectivity_total", "mean"),
            network_connectivity_source=("network_connectivity_source", "first"),
        )
        .sort_values(["city_id", "year"])
    )
    merged = full.merge(agg, on=["city_id", "year"], how="left")
    for col in [
        "flight_connectivity_total",
        "flight_degree_centrality",
        "airport_count_mapped",
        "international_route_share",
        "shipping_connectivity_total",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["network_connectivity_source"] = merged["network_connectivity_source"].fillna("missing").astype(str)
    return merged


def collect_city_osm_history_year_panel(
    cities: pd.DataFrame,
    config: ProjectConfig,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load city-year OSM historical panel generated by src.osm_history_signals."""
    path = DATA_RAW / "city_osm_history_yearly.csv"
    full = _city_year_full_grid(cities, config)
    default_cols = list(OSM_HISTORY_ALIASES.keys())
    if (not use_cache) or (not path.exists()):
        out = full.copy()
        for c in default_cols:
            out[c] = np.nan
        out["osm_hist_source"] = "missing"
        return out

    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read OSM-history panel cache (%s): %s", path, exc)
        out = full.copy()
        for c in default_cols:
            out[c] = np.nan
        out["osm_hist_source"] = "missing"
        return out

    raw = _attach_city_id_from_name(raw, cities)
    if "year" not in raw.columns and "date" in raw.columns:
        raw["year"] = pd.to_datetime(raw["date"], errors="coerce").dt.year
    if "city_id" not in raw.columns or "year" not in raw.columns:
        out = full.copy()
        for c in default_cols:
            out[c] = np.nan
        out["osm_hist_source"] = "missing"
        return out

    work = raw.copy()
    work["city_id"] = work["city_id"].astype(str)
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work = work[work["city_id"].isin(cities["city_id"].astype(str))].copy()
    work = work[work["year"].between(config.time_range.start_year, config.time_range.end_year, inclusive="both")].copy()
    work["year"] = work["year"].astype(int)

    cols = ["city_id", "year"]
    for key, aliases in OSM_HISTORY_ALIASES.items():
        col = _resolve_column_alias(work, aliases)
        if col is None:
            work[key] = np.nan
        else:
            work[key] = pd.to_numeric(work[col], errors="coerce")
        cols.append(key)
    if "osm_hist_source" in work.columns:
        work["osm_hist_source"] = work["osm_hist_source"].fillna("ohsome_api").astype(str)
    else:
        work["osm_hist_source"] = "ohsome_api"
    cols.append("osm_hist_source")

    agg_spec = {k: (k, "mean") for k in OSM_HISTORY_ALIASES}
    agg_spec["osm_hist_source"] = ("osm_hist_source", "first")
    agg = (
        work[cols]
        .groupby(["city_id", "year"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["city_id", "year"])
    )

    # Recompute YoY from levels for consistency.
    for level_col, yoy_col in OSM_HISTORY_LEVEL_TO_YOY.items():
        if level_col in agg.columns:
            agg[yoy_col] = agg.groupby("city_id")[level_col].diff()

    merged = full.merge(agg, on=["city_id", "year"], how="left")
    for col in default_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["osm_hist_source"] = merged["osm_hist_source"].fillna("missing").astype(str)
    return merged


def collect_city_social_sentiment_panel(
    cities: pd.DataFrame,
    config: ProjectConfig,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load city-year social sentiment panel from local files.

    Preferred aggregated file:
    - data/raw/city_social_sentiment_yearly.csv
    Optional post-level file (aggregated on the fly):
    - data/raw/social_sentiment_posts.csv
    """
    full = _city_year_full_grid(cities, config)
    default_out = full.copy()
    for col in SOCIAL_SENTIMENT_ALIASES:
        default_out[col] = np.nan
    default_out["social_sentiment_source"] = "missing"

    if not use_cache:
        return default_out

    yearly_candidates = [
        DATA_RAW / "city_social_sentiment_yearly.csv",
        DATA_RAW / "social_sentiment_city_year.csv",
        DATA_RAW / "gdelt_city_yearly.csv",
        DATA_RAW / "city_gdelt_yearly.csv",
        DATA_RAW / "gdelt_tone_city_year.csv",
    ]
    yearly_path = next((p for p in yearly_candidates if p.exists()), None)
    posts_path = DATA_RAW / "social_sentiment_posts.csv"

    social: pd.DataFrame
    if yearly_path is not None:
        try:
            raw = pd.read_csv(yearly_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to read city social sentiment yearly panel (%s): %s", yearly_path, exc)
            raw = pd.DataFrame()

        if not raw.empty:
            raw = _attach_city_id_from_name(raw, cities)
            if "year" not in raw.columns and "date" in raw.columns:
                raw["year"] = pd.to_datetime(raw["date"], errors="coerce").dt.year
            if "city_id" in raw.columns and "year" in raw.columns:
                social = raw.copy()
                social["city_id"] = social["city_id"].astype(str)
                social["year"] = pd.to_numeric(social["year"], errors="coerce").astype("Int64")
                social = social[social["city_id"].isin(cities["city_id"].astype(str))].copy()
                social = social[
                    social["year"].between(config.time_range.start_year, config.time_range.end_year, inclusive="both")
                ].copy()
                if not social.empty:
                    social["year"] = social["year"].astype(int)
                    std = social[["city_id", "year"]].copy()
                    for target, aliases in SOCIAL_SENTIMENT_ALIASES.items():
                        col = _resolve_column_alias(social, aliases)
                        std[target] = pd.to_numeric(social[col], errors="coerce") if col else np.nan
                    tone = pd.to_numeric(std["social_sentiment_score"], errors="coerce")
                    if tone.notna().any() and float(tone.abs().median()) > 1.5:
                        std["social_sentiment_score"] = (tone / 100.0).clip(-1.0, 1.0)
                    tone = pd.to_numeric(std["social_sentiment_score"], errors="coerce").clip(-1.0, 1.0)
                    if pd.to_numeric(std["social_sentiment_positive_share"], errors="coerce").notna().sum() == 0:
                        std["social_sentiment_positive_share"] = ((tone + 1.0) / 2.0).where(tone.notna(), np.nan)
                    if pd.to_numeric(std["social_sentiment_negative_share"], errors="coerce").notna().sum() == 0:
                        std["social_sentiment_negative_share"] = ((1.0 - tone) / 2.0).where(tone.notna(), np.nan)
                    if pd.to_numeric(std["social_sentiment_buzz"], errors="coerce").notna().sum() == 0:
                        volume = pd.to_numeric(std["social_sentiment_volume"], errors="coerce")
                        std["social_sentiment_buzz"] = np.log1p(volume.clip(lower=0.0)).where(volume.notna(), np.nan)
                    if pd.to_numeric(std["social_sentiment_platform_count"], errors="coerce").notna().sum() == 0:
                        volume = pd.to_numeric(std["social_sentiment_volume"], errors="coerce")
                        std["social_sentiment_platform_count"] = np.where(volume.fillna(0.0) > 0.0, 1.0, 0.0)
                    source_col = _resolve_column_alias(
                        social,
                        [
                            "social_sentiment_source",
                            "sentiment_source",
                            "source",
                        ],
                    )
                    std["social_sentiment_source"] = (
                        social[source_col].astype(str)
                        if source_col
                        else ("gdelt_city_yearly" if "gdelt" in yearly_path.name.lower() else "external_social_city_year")
                    )
                    std = (
                        std.groupby(["city_id", "year"], as_index=False)
                        .agg(
                            social_sentiment_score=("social_sentiment_score", "mean"),
                            social_sentiment_volatility=("social_sentiment_volatility", "mean"),
                            social_sentiment_positive_share=("social_sentiment_positive_share", "mean"),
                            social_sentiment_negative_share=("social_sentiment_negative_share", "mean"),
                            social_sentiment_volume=("social_sentiment_volume", "sum"),
                            social_sentiment_platform_count=("social_sentiment_platform_count", "mean"),
                            social_sentiment_buzz=("social_sentiment_buzz", "mean"),
                            social_sentiment_source=("social_sentiment_source", "first"),
                        )
                        .sort_values(["city_id", "year"])
                    )
                    merged = full.merge(std, on=["city_id", "year"], how="left")
                    for col in SOCIAL_SENTIMENT_ALIASES:
                        merged[col] = pd.to_numeric(merged[col], errors="coerce")
                    merged["social_sentiment_source"] = merged["social_sentiment_source"].fillna("missing").astype(str)
                    return merged

    if posts_path.exists():
        try:
            posts = pd.read_csv(posts_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to read social sentiment posts file (%s): %s", posts_path, exc)
            return default_out

        agg, _ = aggregate_social_posts(
            posts,
            cities,
            start_year=int(config.time_range.start_year),
            end_year=int(config.time_range.end_year),
            source_label="external_social_posts",
        )
        if agg.empty:
            return default_out
        merged = full.merge(agg, on=["city_id", "year"], how="left")
        for col in SOCIAL_SENTIMENT_ALIASES:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged["social_sentiment_source"] = merged["social_sentiment_source"].fillna("missing").astype(str)
        return merged

    return default_out


def _fill_group_median(df: pd.DataFrame, column: str, group_col: str) -> pd.Series:
    group_filled = df.groupby(group_col)[column].transform(lambda x: x.fillna(x.median()))
    global_median = float(df[column].median()) if not pd.isna(df[column].median()) else 0.0
    return group_filled.fillna(global_median)


def _safe_year_minmax(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    mask = vals.notna()
    if not mask.any():
        return pd.Series(np.nan, index=series.index, dtype=float)
    vmin = float(vals[mask].min())
    vmax = float(vals[mask].max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return pd.Series(np.nan, index=series.index, dtype=float)
    if np.isclose(vmin, vmax):
        out = pd.Series(np.nan, index=series.index, dtype=float)
        out.loc[mask] = 0.0
        return out
    out = (vals - vmin) / (vmax - vmin)
    return out.astype(float)


def _safe_global_minmax(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    mask = vals.notna()
    if not mask.any():
        return pd.Series(np.nan, index=series.index, dtype=float)
    vmin = float(vals[mask].min())
    vmax = float(vals[mask].max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return pd.Series(np.nan, index=series.index, dtype=float)
    if np.isclose(vmin, vmax):
        out = pd.Series(np.nan, index=series.index, dtype=float)
        out.loc[mask] = 0.5
        return out
    out = (vals - vmin) / (vmax - vmin)
    return out.astype(float)


def _weighted_row_blend(
    components: List[tuple[float, pd.Series]],
    *,
    default: float = np.nan,
) -> pd.Series:
    if not components:
        return pd.Series(np.nan, dtype=float)
    base_index = components[0][1].index
    frame = pd.concat(
        [pd.to_numeric(series, errors="coerce").reindex(base_index) for _, series in components],
        axis=1,
    )
    weights = np.asarray([float(weight) for weight, _ in components], dtype=float)
    values = frame.to_numpy(dtype=float)
    valid = np.isfinite(values)
    numer = np.nan_to_num(values, nan=0.0) @ weights
    denom = valid.astype(float) @ weights
    out = np.full(len(frame), float(default), dtype=float)
    use = denom > 1e-12
    out[use] = numer[use] / denom[use]
    return pd.Series(out, index=base_index, dtype=float)


def _one_minus_preserve_nan(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    mask = vals.notna()
    out.loc[mask] = 1.0 - vals.loc[mask]
    return out.clip(0.0, 1.0)


def _scaled_knn_impute_numeric(frame: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    if frame.empty or len(frame) < 3:
        return frame
    numeric = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if not numeric.isna().any().any():
        return frame

    mins = numeric.min(skipna=True)
    maxs = numeric.max(skipna=True)
    span = (maxs - mins).replace(0.0, 1.0)
    scaled = (numeric - mins) / span
    n_neighbors = max(1, min(5, len(frame) - 1))
    try:
        filled = KNNImputer(n_neighbors=n_neighbors, weights="distance").fit_transform(scaled)
    except Exception:  # noqa: BLE001
        return frame

    restored = pd.DataFrame(filled, columns=numeric_cols, index=frame.index)
    restored = restored * span + mins
    out = frame.copy()
    for col in numeric_cols:
        out[col] = restored[col]
    return out


def _impute_country_year_indicator_panel(
    panel: pd.DataFrame,
    numeric_cols: List[str],
    cities: pd.DataFrame,
    *,
    strategy_col: str,
) -> pd.DataFrame:
    out = panel.copy()
    iso_meta = cities[["iso3", "continent"]].drop_duplicates("iso3")
    if "continent" not in out.columns:
        out = out.merge(iso_meta, on="iso3", how="left")
    if strategy_col not in out.columns:
        out[strategy_col] = "observed"

    had_missing = out[numeric_cols].isna().any(axis=1)
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out.groupby("iso3")[col].transform(lambda x: x.interpolate(limit_direction="both"))

    by_year: List[pd.DataFrame] = []
    for _, sub in out.groupby("year", sort=False):
        by_year.append(_scaled_knn_impute_numeric(sub, numeric_cols))
    if by_year:
        out = pd.concat(by_year, ignore_index=True).sort_values(["iso3", "year"]).reset_index(drop=True)

    for col in numeric_cols:
        out[col] = out.groupby(["continent", "year"])[col].transform(lambda x: x.fillna(x.median()))
        out[col] = out.groupby("continent")[col].transform(lambda x: x.fillna(x.median()))
        median = float(out[col].median()) if not pd.isna(out[col].median()) else 0.0
        out[col] = out[col].fillna(median)

    out[strategy_col] = np.where(
        had_missing.to_numpy(dtype=bool),
        "iso_interpolate_year_knn_continent_median",
        "observed_or_interpolated",
    )
    return out


def _group_nunique_share(df: pd.DataFrame, value_col: str, keys: list[str]) -> float:
    """Share of groups where value_col has city-level variation."""
    if df.empty or value_col not in df.columns:
        return 0.0
    nunique = df.groupby(keys)[value_col].nunique(dropna=True)
    if nunique.empty:
        return 0.0
    return float((nunique > 1).mean())


def _apply_city_macro_disaggregation(out: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Construct a non-tautological city GDP proxy using only VIIRS night-light totals.

    The country-year macro panel remains untouched; the only city-level economic
    disaggregation emitted here is `gdp_disaggregated_by_ntl`, which allocates
    country total GDP to cities by their within-country VIIRS radiance share.
    """
    keys = ["iso3", "year"]
    required = {
        "city_id",
        "iso3",
        "year",
        "gdp_per_capita",
        "population",
        "unemployment",
        "internet_users",
        "capital_formation",
        "inflation",
    }
    if not required.issubset(out.columns):
        missing = sorted(required.difference(set(out.columns)))
        return out, {"status": "skipped", "reason": f"missing_columns:{','.join(missing)}"}

    df = out.copy()
    df["gdp_per_capita"] = pd.to_numeric(df["gdp_per_capita"], errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    viirs_ntl_sum = pd.to_numeric(df.get("viirs_ntl_sum"), errors="coerce").clip(lower=0.0)
    if viirs_ntl_sum.notna().sum() == 0:
        viirs_ntl_sum = (
            pd.to_numeric(df.get("viirs_ntl_mean"), errors="coerce").clip(lower=0.0)
            * pd.to_numeric(df.get("viirs_lit_area_km2"), errors="coerce").clip(lower=0.0)
        )
    df["viirs_ntl_sum"] = viirs_ntl_sum

    city_count = df.groupby(keys)["city_id"].transform("nunique").replace(0, np.nan)
    equal_share = (1.0 / city_count).replace([np.inf, -np.inf], np.nan)
    group_sum = df.groupby(keys)["viirs_ntl_sum"].transform("sum")
    ntl_share = df["viirs_ntl_sum"] / group_sum.replace(0.0, np.nan)
    ntl_share = ntl_share.fillna(equal_share).clip(lower=0.0)
    ntl_share_sum = ntl_share.groupby([df["iso3"], df["year"]]).transform("sum")
    ntl_share = (ntl_share / ntl_share_sum.replace(0.0, np.nan)).fillna(equal_share)
    df["gdp_disaggregation_weight_ntl_share"] = ntl_share.astype(float)

    country_gdp_total = (
        pd.to_numeric(df["gdp_per_capita"], errors="coerce")
        * pd.to_numeric(df["population"], errors="coerce")
    )
    df["gdp_disaggregated_by_ntl"] = (country_gdp_total * df["gdp_disaggregation_weight_ntl_share"]).clip(lower=0.0)
    df["log_gdp_disaggregated_by_ntl"] = np.log1p(pd.to_numeric(df["gdp_disaggregated_by_ntl"], errors="coerce").clip(lower=0.0))
    df["macro_city_disagg_source"] = np.where(
        pd.to_numeric(df["viirs_ntl_sum"], errors="coerce").notna(),
        "ntl_share_only_v1",
        "equal_share_missing_viirs_ntl_sum",
    )

    city_per_group = df.groupby(keys)["city_id"].nunique()
    multi_city_groups = int((city_per_group > 1).sum()) if not city_per_group.empty else 0
    total_groups = int(city_per_group.shape[0]) if not city_per_group.empty else 0
    mean_city_count = float(city_per_group.mean()) if not city_per_group.empty else 0.0

    meta = {
        "status": "ok",
        "source": "ntl_share_only_v1",
        "proxy_inputs": ["viirs_ntl_sum"],
        "country_year_groups": total_groups,
        "country_year_groups_multi_city": multi_city_groups,
        "country_year_groups_multi_city_ratio": float(multi_city_groups / max(total_groups, 1)),
        "mean_cities_per_country_year": mean_city_count,
        "proxy_mode": "viirs_ntl_sum_share_only",
        "ntl_weight_non_missing_ratio": float(pd.to_numeric(df["viirs_ntl_sum"], errors="coerce").notna().mean()),
        "gdp_disaggregated_non_missing_ratio": float(pd.to_numeric(df["gdp_disaggregated_by_ntl"], errors="coerce").notna().mean()),
    }
    return df, meta


def _robust_z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    med = float(values.median()) if not values.dropna().empty else 0.0
    mad = float(np.median(np.abs(values.dropna().to_numpy(dtype=float) - med))) if not values.dropna().empty else 0.0
    if not np.isfinite(mad) or mad <= 1e-8:
        std = float(values.std()) if not values.dropna().empty else 0.0
        if not np.isfinite(std) or std <= 1e-8:
            return pd.Series(np.zeros(len(values)), index=series.index, dtype=float)
        return ((values - med) / std).fillna(0.0)
    scale = 1.4826 * mad
    return ((values - med) / scale).fillna(0.0)


def _reconstruct_historical_poi_from_snapshot(panel: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    out = panel.copy()
    poi_level_cols = [
        "amenity_count",
        "shop_count",
        "office_count",
        "leisure_count",
        "transport_count",
        "poi_total",
    ]
    for col in poi_level_cols + ["poi_diversity"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

    if "poi_temporal_source" not in out.columns:
        has_snapshot = out[poi_level_cols].notna().any(axis=1)
        out["poi_temporal_source"] = np.where(has_snapshot, "snapshot_broadcast", "missing")
    out["poi_temporal_source"] = out["poi_temporal_source"].fillna("missing").astype(str)

    true_yearly_mask = out["poi_temporal_source"].eq("historical_year_panel")
    latest_year = int(config.time_range.end_year)
    hist_total = pd.to_numeric(out.get("osm_hist_poi_count"), errors="coerce").clip(lower=0.0)
    ref_hist_total = out.groupby("city_id")["osm_hist_poi_count"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").dropna().iloc[-1] if pd.to_numeric(s, errors="coerce").notna().any() else np.nan
    )
    ref_hist_total = pd.to_numeric(ref_hist_total, errors="coerce").clip(lower=0.0)
    scale = hist_total / ref_hist_total.replace(0.0, np.nan)
    scale_valid = (~true_yearly_mask) & scale.notna()

    for col in poi_level_cols:
        snapshot_vals = pd.to_numeric(out[col], errors="coerce").clip(lower=0.0)
        out.loc[scale_valid, col] = snapshot_vals.loc[scale_valid] * scale.loc[scale_valid]
    out.loc[scale_valid, "poi_diversity"] = pd.to_numeric(out.loc[scale_valid, "poi_diversity"], errors="coerce")

    blocked_mask = (~true_yearly_mask) & (~scale_valid) & (pd.to_numeric(out["year"], errors="coerce") < latest_year)
    for col in poi_level_cols + ["poi_diversity"]:
        out.loc[blocked_mask, col] = np.nan

    out["poi_backcast_scale"] = scale.astype(float)
    out["has_poi_historical_support"] = scale.notna().astype(int)
    out["has_poi_observation"] = out[poi_level_cols].notna().any(axis=1).astype(int)
    out["poi_temporal_source"] = np.select(
        [
            true_yearly_mask.to_numpy(dtype=bool),
            scale_valid.to_numpy(dtype=bool),
            ((~true_yearly_mask) & (pd.to_numeric(out["year"], errors="coerce") == latest_year) & out[poi_level_cols].notna().any(axis=1)).to_numpy(dtype=bool),
        ],
        [
            "historical_year_panel",
            "snapshot_backcast_from_osm_history",
            "snapshot_current_year_only",
        ],
        default="missing_prevented_lookahead",
    )
    return out


def _infer_policy_name_from_signal(row: pd.Series) -> str:
    score_map = {
        "digital_connectivity": float(row.get("z_internet_delta", 0.0)),
        "innovation_industry": float(row.get("z_capital_delta", 0.0)),
        "urban_services": float(row.get("z_gdp_growth", 0.0)),
    }
    best = max(score_map.items(), key=lambda kv: kv[1])[0]
    return f"{best}_ai_inferred"


def _infer_policy_name_from_macro_proxy(row: pd.Series) -> str:
    score_map = {
        "digital_connectivity": float(row.get("z_internet_delta", 0.0)),
        "innovation_industry": float(row.get("z_capital_delta", 0.0)),
        "urban_services": float(row.get("z_gdp_growth", 0.0)),
    }
    best = max(score_map.items(), key=lambda kv: kv[1])[0]
    return f"{best}_objective_macro_proxy"


def _infer_policy_name_from_indicator_signal(row: pd.Series) -> str:
    score_map = {
        "digital_connectivity": (
            float(row.get("z_internet_delta", 0.0))
            + 0.60 * float(row.get("z_broadband_delta", 0.0))
            + 0.45 * float(row.get("z_electricity_delta", 0.0))
        ),
        "innovation_industry": (
            float(row.get("z_capital_delta", 0.0))
            + 0.70 * float(row.get("z_research_delta", 0.0))
            + 0.45 * float(row.get("z_patent_delta", 0.0))
        ),
        "institutional_reform": float(row.get("z_unemployment_improve", 0.0)),
        "climate_resilience": float(row.get("z_pm25_improve", 0.0)),
        "urban_services": float(row.get("z_gdp_growth", 0.0)),
    }
    best = max(score_map.items(), key=lambda kv: kv[1])[0]
    return f"{best}_objective_indicator_event"


def _augment_policy_registry_with_objective_indicator_events(
    registry: pd.DataFrame,
    panel: pd.DataFrame,
    config: ProjectConfig,
    target_non_ai_coverage: float = 0.95,
) -> tuple[pd.DataFrame, dict]:
    """Add deterministic WB-indicator events for countries lacking non-AI policy entries."""
    panel_iso = set(panel["iso3"].astype(str).str.upper().unique().tolist())
    if not panel_iso:
        return registry, {"status": "skipped", "reason": "empty_panel_iso"}

    source_ref = registry["source_ref"].astype(str) if not registry.empty else pd.Series(dtype=str)
    ai_mask = source_ref.str.contains("objective_ai_inferred", case=False, regex=False) if not source_ref.empty else pd.Series(dtype=bool)
    covered_non_ai = set(
        registry.loc[~ai_mask, "iso3"].astype(str).str.upper().tolist()
    ) if not registry.empty else set()
    current_cov = float(len(panel_iso.intersection(covered_non_ai)) / max(len(panel_iso), 1))
    if current_cov >= float(target_non_ai_coverage):
        return registry, {
            "status": "not_needed",
            "target_non_ai_coverage": float(target_non_ai_coverage),
            "coverage_before": current_cov,
            "coverage_after": current_cov,
            "added_event_rows": 0,
            "added_iso3": 0,
        }

    agg = (
        panel.groupby(["iso3", "year"], as_index=False)
        .agg(
            internet_users=("internet_users", "mean"),
            fixed_broadband_subscriptions=("fixed_broadband_subscriptions", "mean"),
            electricity_access=("electricity_access", "mean"),
            unemployment=("unemployment", "mean"),
            pm25_exposure=("pm25_exposure", "mean"),
            gdp_growth=("gdp_growth", "mean"),
            capital_formation=("capital_formation", "mean"),
            researchers_per_million=("researchers_per_million", "mean"),
            patent_residents=("patent_residents", "mean"),
        )
        .sort_values(["iso3", "year"])
        .reset_index(drop=True)
    )
    if agg.empty:
        return registry, {"status": "skipped", "reason": "empty_panel_agg"}

    agg["internet_delta"] = agg.groupby("iso3")["internet_users"].diff().fillna(0.0)
    agg["broadband_delta"] = agg.groupby("iso3")["fixed_broadband_subscriptions"].diff().fillna(0.0)
    agg["electricity_delta"] = agg.groupby("iso3")["electricity_access"].diff().fillna(0.0)
    agg["unemployment_improve"] = (-agg.groupby("iso3")["unemployment"].diff()).fillna(0.0)
    agg["pm25_improve"] = (-agg.groupby("iso3")["pm25_exposure"].diff()).fillna(0.0)
    agg["capital_delta"] = agg.groupby("iso3")["capital_formation"].diff().fillna(0.0)
    agg["research_delta"] = agg.groupby("iso3")["researchers_per_million"].diff().fillna(0.0)
    agg["patent_delta"] = agg.groupby("iso3")["patent_residents"].diff().fillna(0.0)

    for col, zcol in [
        ("internet_delta", "z_internet_delta"),
        ("broadband_delta", "z_broadband_delta"),
        ("electricity_delta", "z_electricity_delta"),
        ("unemployment_improve", "z_unemployment_improve"),
        ("pm25_improve", "z_pm25_improve"),
        ("capital_delta", "z_capital_delta"),
        ("research_delta", "z_research_delta"),
        ("patent_delta", "z_patent_delta"),
        ("gdp_growth", "z_gdp_growth"),
    ]:
        agg[zcol] = agg.groupby("iso3")[col].transform(_robust_z)

    agg["indicator_event_score"] = (
        0.20 * agg["z_internet_delta"]
        + 0.14 * agg["z_broadband_delta"]
        + 0.12 * agg["z_electricity_delta"]
        + 0.14 * agg["z_unemployment_improve"]
        + 0.14 * agg["z_capital_delta"]
        + 0.08 * agg["z_research_delta"]
        + 0.07 * agg["z_patent_delta"]
        + 0.08 * agg["z_pm25_improve"]
        + 0.03 * agg["z_gdp_growth"]
    )

    year_min = int(config.time_range.start_year)
    year_max = int(config.time_range.end_year)
    min_event_year = min(year_max, year_min + 2)
    max_event_year = max(year_min, year_max - 2)
    if min_event_year <= max_event_year:
        agg = agg[(agg["year"] >= min_event_year) & (agg["year"] <= max_event_year)].copy()
    if agg.empty:
        return registry, {"status": "skipped", "reason": "no_valid_year_window_for_indicator_events"}

    target_count = int(np.ceil(float(target_non_ai_coverage) * max(len(panel_iso), 1)))
    target_count = int(np.clip(target_count, 0, len(panel_iso)))
    needed_iso_count = int(max(0, target_count - len(panel_iso.intersection(covered_non_ai))))
    missing_iso = sorted(panel_iso.difference(covered_non_ai))
    if needed_iso_count <= 0 or not missing_iso:
        return registry, {
            "status": "not_needed",
            "target_non_ai_coverage": float(target_non_ai_coverage),
            "coverage_before": current_cov,
            "coverage_after": current_cov,
            "added_event_rows": 0,
            "added_iso3": 0,
            "target_iso3_count": int(target_count),
            "needed_iso3_count": int(needed_iso_count),
        }

    top_rows: List[pd.Series] = []
    for iso3 in missing_iso:
        sub = agg[agg["iso3"] == iso3].copy()
        if sub.empty:
            continue
        top_rows.append(
            sub.sort_values(
                ["indicator_event_score", "z_internet_delta", "z_capital_delta", "z_unemployment_improve"],
                ascending=[False, False, False, False],
            ).iloc[0]
        )

    if not top_rows:
        return registry, {"status": "skipped", "reason": "no_indicator_event_candidates"}

    top_df = pd.DataFrame(top_rows).reset_index(drop=True)
    if len(top_df) > int(needed_iso_count):
        top_df = (
            top_df.sort_values(
                ["indicator_event_score", "z_internet_delta", "z_capital_delta", "z_unemployment_improve"],
                ascending=[False, False, False, False],
            )
            .head(int(needed_iso_count))
            .reset_index(drop=True)
        )
    score_scaled = minmax_scale(top_df["indicator_event_score"].to_numpy(dtype=float))
    top_df["indicator_intensity"] = np.clip(0.28 + 0.60 * score_scaled, 0.28, 0.93)

    rows: List[dict] = []
    for row in top_df.itertuples(index=False):
        score = float(row.indicator_event_score)
        intensity = float(row.indicator_intensity)
        row_series = pd.Series(row._asdict())
        pname = _infer_policy_name_from_indicator_signal(row_series)
        driver_map = {
            "IT.NET.USER.ZS": float(getattr(row, "z_internet_delta", 0.0)),
            "IT.NET.BBND.P2": float(getattr(row, "z_broadband_delta", 0.0)),
            "EG.ELC.ACCS.ZS": float(getattr(row, "z_electricity_delta", 0.0)),
            "SL.UEM.TOTL.ZS(-)": float(getattr(row, "z_unemployment_improve", 0.0)),
            "NE.GDI.FTOT.ZS": float(getattr(row, "z_capital_delta", 0.0)),
            "SP.POP.SCIE.RD.P6": float(getattr(row, "z_research_delta", 0.0)),
            "IP.PAT.RESD": float(getattr(row, "z_patent_delta", 0.0)),
            "EN.ATM.PM25.MC.M3(-)": float(getattr(row, "z_pm25_improve", 0.0)),
        }
        top_drivers = sorted(driver_map.items(), key=lambda kv: kv[1], reverse=True)[:3]
        drivers_txt = "|".join([f"{k}:{v:.2f}" for k, v in top_drivers])
        source_ref = (
            "wb_indicator_event:"
            f"structural_break@{int(row.year)};"
            f"score={score:.4f};"
            f"drivers={drivers_txt}"
        )
        rows.append(
            {
                "iso3": str(row.iso3),
                "start_year": int(row.year),
                "end_year": int(row.year),
                "policy_intensity": intensity,
                "policy_name": pname,
                "source_ref": source_ref,
            }
        )

    indicator_events = pd.DataFrame(rows)
    if indicator_events.empty:
        return registry, {"status": "skipped", "reason": "no_indicator_events"}

    indicator_events = (
        indicator_events.sort_values(["iso3", "start_year", "policy_intensity"], ascending=[True, True, False])
        .drop_duplicates(subset=["iso3", "start_year"], keep="first")
        .reset_index(drop=True)
    )
    indicator_events.to_csv(DATA_RAW / "policy_events_registry_objective_indicator.csv", index=False)

    if registry.empty:
        combined = indicator_events.copy()
    else:
        combined = pd.concat([registry, indicator_events], ignore_index=True)
    combined = (
        combined.sort_values(["iso3", "start_year", "policy_intensity"], ascending=[True, True, False])
        .drop_duplicates(subset=["iso3", "start_year", "policy_name"], keep="first")
        .reset_index(drop=True)
    )
    combined.to_csv(DATA_RAW / "policy_events_registry_augmented.csv", index=False)

    source_ref_after = combined["source_ref"].astype(str)
    ai_after = source_ref_after.str.contains("objective_ai_inferred", case=False, regex=False)
    covered_after_non_ai = set(combined.loc[~ai_after, "iso3"].astype(str).str.upper().tolist())
    after_cov = float(len(panel_iso.intersection(covered_after_non_ai)) / max(len(panel_iso), 1))
    return combined, {
        "status": "ok",
        "target_non_ai_coverage": float(target_non_ai_coverage),
        "coverage_before": current_cov,
        "coverage_after": after_cov,
        "added_event_rows": int(len(indicator_events)),
        "added_iso3": int(indicator_events["iso3"].nunique()),
        "target_iso3_count": int(target_count),
        "needed_iso3_count": int(needed_iso_count),
        "registry_path_objective_indicator": str(DATA_RAW / "policy_events_registry_objective_indicator.csv"),
        "registry_path_augmented": str(DATA_RAW / "policy_events_registry_augmented.csv"),
    }


def _augment_policy_registry_with_objective_macro_rules(
    registry: pd.DataFrame,
    panel: pd.DataFrame,
    config: ProjectConfig,
    target_non_ai_coverage: float = 0.85,
) -> tuple[pd.DataFrame, dict]:
    """Add deterministic objective macro-rule events for missing countries before AI inference."""
    panel_iso = set(panel["iso3"].astype(str).str.upper().unique().tolist())
    if not panel_iso:
        return registry, {"status": "skipped", "reason": "empty_panel_iso"}

    source_ref = registry["source_ref"].astype(str) if not registry.empty else pd.Series(dtype=str)
    ai_mask = source_ref.str.contains("objective_ai_inferred", case=False, regex=False) if not source_ref.empty else pd.Series(dtype=bool)
    covered_non_ai = set(
        registry.loc[~ai_mask, "iso3"].astype(str).str.upper().tolist()
    ) if not registry.empty else set()
    current_cov = float(len(panel_iso.intersection(covered_non_ai)) / max(len(panel_iso), 1))
    if current_cov >= float(target_non_ai_coverage):
        return registry, {
            "status": "not_needed",
            "target_non_ai_coverage": float(target_non_ai_coverage),
            "coverage_before": current_cov,
            "coverage_after": current_cov,
            "added_event_rows": 0,
            "added_iso3": 0,
        }

    agg = (
        panel.groupby(["iso3", "year"], as_index=False)
        .agg(
            gdp_growth=("gdp_growth", "mean"),
            internet_users=("internet_users", "mean"),
            capital_formation=("capital_formation", "mean"),
            inflation=("inflation", "mean"),
        )
        .sort_values(["iso3", "year"])
        .reset_index(drop=True)
    )
    if agg.empty:
        return registry, {"status": "skipped", "reason": "empty_panel_agg"}

    agg["internet_delta"] = agg.groupby("iso3")["internet_users"].diff().fillna(0.0)
    agg["capital_delta"] = agg.groupby("iso3")["capital_formation"].diff().fillna(0.0)
    agg["inflation_delta"] = agg.groupby("iso3")["inflation"].diff().fillna(0.0)

    for col, zcol in [
        ("internet_delta", "z_internet_delta"),
        ("capital_delta", "z_capital_delta"),
        ("gdp_growth", "z_gdp_growth"),
        ("inflation_delta", "z_inflation_delta"),
    ]:
        agg[zcol] = agg.groupby("iso3")[col].transform(_robust_z)

    agg["macro_rule_score"] = (
        0.48 * agg["z_internet_delta"]
        + 0.34 * agg["z_capital_delta"]
        + 0.24 * agg["z_gdp_growth"]
        - 0.16 * agg["z_inflation_delta"].abs()
    )

    year_min = int(config.time_range.start_year)
    year_max = int(config.time_range.end_year)
    min_event_year = min(year_max, year_min + 2)
    max_event_year = max(year_min, year_max - 2)
    if min_event_year <= max_event_year:
        agg = agg[(agg["year"] >= min_event_year) & (agg["year"] <= max_event_year)].copy()
    if agg.empty:
        return registry, {"status": "skipped", "reason": "no_valid_year_window_for_macro_rules"}

    target_count = int(np.ceil(float(target_non_ai_coverage) * max(len(panel_iso), 1)))
    target_count = int(np.clip(target_count, 0, len(panel_iso)))
    needed_iso_count = int(max(0, target_count - len(panel_iso.intersection(covered_non_ai))))
    missing_iso = sorted(panel_iso.difference(covered_non_ai))
    if needed_iso_count <= 0 or not missing_iso:
        return registry, {
            "status": "not_needed",
            "target_non_ai_coverage": float(target_non_ai_coverage),
            "coverage_before": current_cov,
            "coverage_after": current_cov,
            "added_event_rows": 0,
            "added_iso3": 0,
            "target_iso3_count": int(target_count),
            "needed_iso3_count": int(needed_iso_count),
        }

    top_rows: List[pd.Series] = []
    for iso3 in missing_iso:
        sub = agg[agg["iso3"] == iso3].copy()
        if sub.empty:
            continue
        top_rows.append(sub.sort_values("macro_rule_score", ascending=False).iloc[0])

    if not top_rows:
        return registry, {"status": "skipped", "reason": "no_macro_rule_candidates"}

    top_df = pd.DataFrame(top_rows).reset_index(drop=True)
    if len(top_df) > int(needed_iso_count):
        top_df = top_df.sort_values("macro_rule_score", ascending=False).head(int(needed_iso_count)).reset_index(drop=True)
    score_scaled = minmax_scale(top_df["macro_rule_score"].to_numpy(dtype=float))
    top_df["rule_intensity"] = np.clip(0.25 + 0.60 * score_scaled, 0.25, 0.90)

    rows: List[dict] = []
    for row in top_df.itertuples(index=False):
        score = float(row.macro_rule_score)
        intensity = float(row.rule_intensity)
        pname = _infer_policy_name_from_macro_proxy(pd.Series(row._asdict()))
        source_ref = (
            "objective_macro_rule:"
            f"wb_structural_break@{int(row.year)};"
            f"score={score:.4f};"
            "vars=IT.NET.USER.ZS|NE.GDI.FTOT.ZS|NY.GDP.PCAP.CD|FP.CPI.TOTL.ZG"
        )
        rows.append(
            {
                "iso3": str(row.iso3),
                "start_year": int(row.year),
                "end_year": int(row.year),
                "policy_intensity": intensity,
                "policy_name": pname,
                "source_ref": source_ref,
            }
        )

    macro_events = pd.DataFrame(rows)
    if macro_events.empty:
        return registry, {"status": "skipped", "reason": "no_macro_rule_events"}

    macro_events = (
        macro_events.sort_values(["iso3", "start_year", "policy_intensity"], ascending=[True, True, False])
        .drop_duplicates(subset=["iso3", "start_year"], keep="first")
        .reset_index(drop=True)
    )
    macro_events.to_csv(DATA_RAW / "policy_events_registry_objective_macro.csv", index=False)

    if registry.empty:
        combined = macro_events.copy()
    else:
        combined = pd.concat([registry, macro_events], ignore_index=True)
    combined = (
        combined.sort_values(["iso3", "start_year", "policy_intensity"], ascending=[True, True, False])
        .drop_duplicates(subset=["iso3", "start_year", "policy_name"], keep="first")
        .reset_index(drop=True)
    )
    combined.to_csv(DATA_RAW / "policy_events_registry_augmented.csv", index=False)

    source_ref_after = combined["source_ref"].astype(str)
    ai_after = source_ref_after.str.contains("objective_ai_inferred", case=False, regex=False)
    covered_after_non_ai = set(combined.loc[~ai_after, "iso3"].astype(str).str.upper().tolist())
    after_cov = float(len(panel_iso.intersection(covered_after_non_ai)) / max(len(panel_iso), 1))
    return combined, {
        "status": "ok",
        "target_non_ai_coverage": float(target_non_ai_coverage),
        "coverage_before": current_cov,
        "coverage_after": after_cov,
        "added_event_rows": int(len(macro_events)),
        "added_iso3": int(macro_events["iso3"].nunique()),
        "target_iso3_count": int(target_count),
        "needed_iso3_count": int(needed_iso_count),
        "registry_path_objective_macro": str(DATA_RAW / "policy_events_registry_objective_macro.csv"),
        "registry_path_augmented": str(DATA_RAW / "policy_events_registry_augmented.csv"),
    }


def _augment_policy_registry_with_ai_inference(
    registry: pd.DataFrame,
    panel: pd.DataFrame,
    config: ProjectConfig,
    target_coverage: float = 0.95,
) -> tuple[pd.DataFrame, dict]:
    """Augment missing-country policy events via objective WB macro shock inference."""
    panel_iso = set(panel["iso3"].astype(str).str.upper().unique().tolist())
    covered = set(registry["iso3"].astype(str).str.upper().tolist()) if not registry.empty else set()
    current_cov = float(len(panel_iso.intersection(covered)) / max(len(panel_iso), 1))
    if current_cov >= float(target_coverage):
        return registry, {
            "status": "not_needed",
            "target_coverage": float(target_coverage),
            "coverage_before": current_cov,
            "coverage_after": current_cov,
            "inferred_event_rows": 0,
            "inferred_iso3": 0,
        }

    agg = (
        panel.groupby(["iso3", "year"], as_index=False)
        .agg(
            gdp_growth=("gdp_growth", "mean"),
            internet_users=("internet_users", "mean"),
            capital_formation=("capital_formation", "mean"),
            inflation=("inflation", "mean"),
        )
        .sort_values(["iso3", "year"])
        .reset_index(drop=True)
    )
    if agg.empty:
        return registry, {"status": "skipped", "reason": "empty_panel_agg"}

    agg["internet_delta"] = agg.groupby("iso3")["internet_users"].diff().fillna(0.0)
    agg["capital_delta"] = agg.groupby("iso3")["capital_formation"].diff().fillna(0.0)
    agg["inflation_delta"] = agg.groupby("iso3")["inflation"].diff().fillna(0.0)

    for col, zcol in [
        ("internet_delta", "z_internet_delta"),
        ("capital_delta", "z_capital_delta"),
        ("gdp_growth", "z_gdp_growth"),
        ("inflation_delta", "z_inflation_delta"),
    ]:
        agg[zcol] = agg.groupby("iso3")[col].transform(_robust_z)

    feature_cols = [
        "z_internet_delta",
        "z_capital_delta",
        "z_gdp_growth",
        "z_inflation_delta",
    ]
    x = agg[feature_cols].to_numpy(dtype=float)
    try:
        iso = IsolationForest(
            n_estimators=320,
            contamination=0.12,
            random_state=int(config.random_seed) + 311,
        )
        iso.fit(x)
        anomaly = -iso.decision_function(x)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("AI shock inference model failed, fallback to deterministic score only: %s", exc)
        anomaly = np.zeros(len(agg), dtype=float)

    agg["anomaly_score"] = anomaly
    agg["anomaly_score_n"] = minmax_scale(agg["anomaly_score"].to_numpy(dtype=float))
    impulse_raw = (
        0.50 * agg["z_internet_delta"]
        + 0.32 * agg["z_capital_delta"]
        + 0.18 * agg["z_gdp_growth"]
        - 0.12 * agg["z_inflation_delta"].abs()
    )
    agg["impulse_score_n"] = minmax_scale(impulse_raw.to_numpy(dtype=float))
    agg["ai_policy_score"] = np.clip(0.62 * agg["impulse_score_n"] + 0.38 * agg["anomaly_score_n"], 0.0, 1.0)

    year_min = int(config.time_range.start_year)
    year_max = int(config.time_range.end_year)
    min_event_year = min(year_max, year_min + 2)
    max_event_year = max(year_min, year_max - 2)
    if min_event_year <= max_event_year:
        agg = agg[(agg["year"] >= min_event_year) & (agg["year"] <= max_event_year)].copy()
    if agg.empty:
        return registry, {"status": "skipped", "reason": "no_valid_year_window_for_inference"}

    missing_iso = sorted(panel_iso.difference(covered))
    if not missing_iso:
        return registry, {
            "status": "not_needed",
            "target_coverage": float(target_coverage),
            "coverage_before": current_cov,
            "coverage_after": current_cov,
            "inferred_event_rows": 0,
            "inferred_iso3": 0,
        }

    rows: List[dict] = []
    for iso3 in missing_iso:
        sub = agg[agg["iso3"] == iso3].copy()
        if sub.empty:
            continue
        top = sub.sort_values(["ai_policy_score", "impulse_score_n"], ascending=[False, False]).iloc[0]
        score = float(top["ai_policy_score"])
        intensity = float(np.clip(0.25 + 0.75 * score, 0.25, 0.98))
        pname = _infer_policy_name_from_signal(top)
        source_ref = (
            "objective_ai_inferred:"
            f"wb_macro_shock@{int(top['year'])};"
            f"score={score:.4f};"
            "vars=IT.NET.USER.ZS|NE.GDI.FTOT.ZS|NY.GDP.PCAP.CD|FP.CPI.TOTL.ZG"
        )
        rows.append(
            {
                "iso3": str(iso3),
                "start_year": int(top["year"]),
                "end_year": int(top["year"]),
                "policy_intensity": intensity,
                "policy_name": pname,
                "source_ref": source_ref,
            }
        )

    inferred = pd.DataFrame(rows)
    if inferred.empty:
        return registry, {
            "status": "skipped",
            "reason": "no_inferred_events",
            "coverage_before": current_cov,
            "coverage_after": current_cov,
        }

    inferred = (
        inferred.sort_values(["iso3", "start_year", "policy_intensity"], ascending=[True, True, False])
        .drop_duplicates(subset=["iso3", "start_year"], keep="first")
        .reset_index(drop=True)
    )
    inferred.to_csv(DATA_RAW / "policy_events_registry_inferred.csv", index=False)

    if registry.empty:
        combined = inferred.copy()
    else:
        combined = pd.concat([registry, inferred], ignore_index=True)
    combined = (
        combined.sort_values(["iso3", "start_year", "policy_intensity"], ascending=[True, True, False])
        .drop_duplicates(subset=["iso3", "start_year", "policy_name"], keep="first")
        .reset_index(drop=True)
    )
    combined.to_csv(DATA_RAW / "policy_events_registry_augmented.csv", index=False)

    after_cov = float(len(set(combined["iso3"].astype(str).str.upper().tolist()).intersection(panel_iso)) / max(len(panel_iso), 1))
    return combined, {
        "status": "ok",
        "target_coverage": float(target_coverage),
        "coverage_before": current_cov,
        "coverage_after": after_cov,
        "inferred_event_rows": int(len(inferred)),
        "inferred_iso3": int(inferred["iso3"].nunique()),
        "registry_path_inferred": str(DATA_RAW / "policy_events_registry_inferred.csv"),
        "registry_path_augmented": str(DATA_RAW / "policy_events_registry_augmented.csv"),
    }


def _load_policy_event_registry(
    panel: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Load auditable policy event registry from local CSV if provided."""
    classified_path = DATA_RAW / "policy_events_registry_classified.csv"
    path = classified_path if classified_path.exists() else DATA_RAW / "policy_events_registry.csv"
    if not path.exists():
        return pd.DataFrame()

    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Policy registry read failed (%s). Falling back to rule-based treatment.", exc)
        return pd.DataFrame()

    required = {"iso3", "start_year"}
    if not required.issubset(raw.columns):
        missing = sorted(required.difference(set(raw.columns)))
        LOGGER.warning("Policy registry missing columns %s. Falling back to rule-based treatment.", ",".join(missing))
        return pd.DataFrame()

    reg = raw.copy()
    reg["iso3"] = reg["iso3"].astype(str).str.upper().str.strip()
    reg["start_year"] = pd.to_numeric(reg["start_year"], errors="coerce")
    if "end_year" not in reg.columns:
        reg["end_year"] = reg["start_year"]
    reg["end_year"] = pd.to_numeric(reg["end_year"], errors="coerce").fillna(reg["start_year"])
    reg["start_year"] = reg[["start_year", "end_year"]].min(axis=1)
    reg["end_year"] = reg[["start_year", "end_year"]].max(axis=1)

    if "policy_intensity" not in reg.columns:
        reg["policy_intensity"] = 1.0
    reg["policy_intensity"] = pd.to_numeric(reg["policy_intensity"], errors="coerce").fillna(1.0).clip(0.0, 1.0)

    if "policy_name" not in reg.columns:
        reg["policy_name"] = "unspecified_policy_event"
    reg["policy_name"] = reg["policy_name"].astype(str).str.strip().replace("", "unspecified_policy_event")

    if "source_ref" not in reg.columns:
        reg["source_ref"] = "unspecified"
    reg["source_ref"] = reg["source_ref"].astype(str).str.strip().replace("", "unspecified")

    if "policy_type_coarse" not in reg.columns:
        reg["policy_type_coarse"] = reg["policy_name"].astype(str).map(_policy_type_coarse)
    if "policy_subtype" not in reg.columns:
        reg["policy_subtype"] = reg["policy_name"].astype(str).map(_policy_subtype)
    if "policy_treatment_bucket" not in reg.columns:
        reg["policy_treatment_bucket"] = reg["policy_name"].astype(str).map(_policy_treatment_bucket)
    if "policy_evidence_track" not in reg.columns:
        reg["policy_evidence_track"] = reg["source_ref"].astype(str).map(_policy_evidence_track)
    if "policy_direct_core_evidence_eligible" not in reg.columns:
        reg["policy_direct_core_evidence_eligible"] = reg["source_ref"].astype(str).map(
            _policy_direct_core_evidence_eligible
        )

    reg = reg.dropna(subset=["iso3", "start_year", "end_year"]).copy()
    if reg.empty:
        LOGGER.warning("Policy registry has no valid rows after cleaning. Falling back to rule-based treatment.")
        return pd.DataFrame()

    reg["start_year"] = reg["start_year"].round().astype(int)
    reg["end_year"] = reg["end_year"].round().astype(int)

    year_min = int(config.time_range.start_year)
    year_max = int(config.time_range.end_year)
    reg["start_year"] = reg["start_year"].clip(lower=year_min, upper=year_max)
    reg["end_year"] = reg["end_year"].clip(lower=year_min, upper=year_max)
    reg = reg[reg["end_year"] >= reg["start_year"]].copy()

    panel_iso = set(panel["iso3"].astype(str).str.upper().unique().tolist())
    reg = reg[reg["iso3"].isin(panel_iso)].copy()
    if reg.empty:
        LOGGER.warning("Policy registry has no overlap with sampled countries. Falling back to rule-based treatment.")
        return pd.DataFrame()

    keep_cols = [
        "iso3",
        "start_year",
        "end_year",
        "policy_intensity",
        "policy_name",
        "policy_type_coarse",
        "policy_subtype",
        "policy_treatment_bucket",
        "policy_evidence_track",
        "policy_direct_core_evidence_eligible",
        "source_ref",
    ]
    return reg[keep_cols].reset_index(drop=True)


def _policy_event_evidence_grade(source_ref: str) -> str:
    """Map source_ref into evidence grades (A/B/C) for transparency."""
    src = str(source_ref).strip().lower()
    if src in {"", "nan", "none", "unspecified"}:
        return "C"
    if "objective_ai_inferred" in src:
        return "C"
    if "wb_indicator_event" in src:
        return "B"
    if "objective_macro_rule" in src:
        return "B"
    if "wb_project_regional" in src:
        return "B"
    if "wb_project:" in src:
        return "A"
    return "B"


def _policy_type_coarse(policy_name: str) -> str:
    """Collapse heterogeneous policy tags into econometrically tractable coarse types."""
    return shared_policy_type_coarse(policy_name)


def _policy_subtype(policy_name: str) -> str:
    return shared_policy_subtype(policy_name)


def _policy_treatment_bucket(policy_name: str) -> str:
    return shared_policy_treatment_bucket(policy_name)


def _policy_evidence_track(source_ref: str) -> str:
    return shared_policy_evidence_track(source_ref)


def _policy_direct_core_evidence_eligible(source_ref: str) -> int:
    return shared_policy_direct_core_evidence_eligible(source_ref)


def _annotate_policy_registry_evidence(registry: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach A/B/C evidence grades and summarize country-level coverage by evidence tier."""
    if registry.empty:
        return registry.copy(), {
            "status": "missing_or_invalid",
            "event_rows": 0,
            "event_rows_by_grade": {"A": 0, "B": 0, "C": 0},
            "iso3_covered_by_grade": {"A": 0, "B": 0, "C": 0},
            "iso3_share_by_grade": {"A": 0.0, "B": 0.0, "C": 0.0},
            "high_evidence_share_events": 0.0,
            "high_evidence_share_countries": 0.0,
        }

    reg = registry.copy()
    reg["evidence_grade"] = reg["source_ref"].astype(str).map(_policy_event_evidence_grade)
    score_map = {"A": 1.0, "B": 0.75, "C": 0.45}
    reg["evidence_score"] = reg["evidence_grade"].map(score_map).fillna(0.60).astype(float)

    panel_iso = set(panel["iso3"].astype(str).str.upper().unique().tolist())
    row_counts = reg["evidence_grade"].value_counts().to_dict()
    country_counts: Dict[str, int] = {}
    country_shares: Dict[str, float] = {}
    for grade in ["A", "B", "C"]:
        iso = set(reg.loc[reg["evidence_grade"] == grade, "iso3"].astype(str).str.upper().tolist())
        covered = int(len(panel_iso.intersection(iso)))
        country_counts[grade] = covered
        country_shares[grade] = float(covered / max(len(panel_iso), 1))

    by_continent = (
        panel[["iso3", "continent"]]
        .drop_duplicates("iso3")
        .assign(iso3=lambda x: x["iso3"].astype(str).str.upper())
        .merge(
            reg[["iso3", "evidence_grade"]].drop_duplicates(),
            on="iso3",
            how="left",
        )
        .assign(evidence_grade=lambda x: x["evidence_grade"].fillna("none"))
        .groupby(["continent", "evidence_grade"], as_index=False)
        .agg(countries=("iso3", "nunique"))
        .sort_values(["continent", "evidence_grade"])
    )
    by_continent.to_csv(DATA_PROCESSED / "policy_event_evidence_by_continent.csv", index=False)

    a_rows = int(row_counts.get("A", 0))
    a_iso = country_counts.get("A", 0)
    total_iso = max(len(panel_iso), 1)
    meta = {
        "status": "ok",
        "event_rows": int(len(reg)),
        "event_rows_by_grade": {"A": int(row_counts.get("A", 0)), "B": int(row_counts.get("B", 0)), "C": int(row_counts.get("C", 0))},
        "iso3_covered_by_grade": {"A": int(country_counts.get("A", 0)), "B": int(country_counts.get("B", 0)), "C": int(country_counts.get("C", 0))},
        "iso3_share_by_grade": {"A": float(country_shares.get("A", 0.0)), "B": float(country_shares.get("B", 0.0)), "C": float(country_shares.get("C", 0.0))},
        "high_evidence_share_events": float(a_rows / max(len(reg), 1)),
        "high_evidence_share_countries": float(a_iso / total_iso),
    }
    return reg, meta


def _build_policy_registry_audit(registry: pd.DataFrame, panel: pd.DataFrame, config: ProjectConfig) -> dict:
    """Summarize policy-event registry quality and coverage."""
    if registry.empty:
        return {
            "status": "missing_or_invalid",
            "registry_file": str(DATA_RAW / "policy_events_registry.csv"),
            "event_rows": 0,
            "policy_types": 0,
            "source_refs": 0,
            "direct_event_rows": 0,
            "external_direct_event_rows": 0,
            "objective_indicator_event_rows": 0,
            "objective_macro_event_rows": 0,
            "ai_inferred_event_rows": 0,
            "iso3_covered": 0,
            "iso3_share_of_sample": 0.0,
            "iso3_covered_direct": 0,
            "iso3_share_direct": 0.0,
            "iso3_covered_external_direct": 0,
            "iso3_share_external_direct": 0.0,
            "iso3_covered_objective_indicator": 0,
            "iso3_share_objective_indicator": 0.0,
            "iso3_covered_objective_macro": 0,
            "iso3_share_objective_macro": 0.0,
            "iso3_covered_ai_inferred": 0,
            "iso3_share_ai_inferred": 0.0,
            "evidence_a_event_rows": 0,
            "evidence_b_event_rows": 0,
            "evidence_c_event_rows": 0,
            "iso3_covered_evidence_a": 0,
            "iso3_share_evidence_a": 0.0,
            "iso3_covered_evidence_b": 0,
            "iso3_share_evidence_b": 0.0,
            "iso3_covered_evidence_c": 0,
            "iso3_share_evidence_c": 0.0,
            "iso3_missing": int(panel["iso3"].astype(str).str.upper().nunique()),
            "missing_iso3_sample": sorted(panel["iso3"].astype(str).str.upper().unique().tolist())[:30],
            "active_year_range": [None, None],
            "mean_intensity": None,
            "coverage_by_continent": [],
        }

    if "evidence_grade" not in registry.columns:
        registry = registry.copy()
        registry["evidence_grade"] = registry["source_ref"].astype(str).map(_policy_event_evidence_grade)

    year_min = int(config.time_range.start_year)
    year_max = int(config.time_range.end_year)
    panel_country = panel[["iso3", "continent"]].drop_duplicates("iso3").copy()
    panel_country["iso3"] = panel_country["iso3"].astype(str).str.upper()
    panel_country["continent"] = panel_country["continent"].astype(str)

    panel_iso = set(panel_country["iso3"].tolist())
    iso3_cov = set(registry["iso3"].astype(str).str.upper().tolist())
    overlap = panel_iso.intersection(iso3_cov)
    missing_iso = sorted(panel_iso - overlap)

    source_ref = registry["source_ref"].astype(str).str.strip()
    inferred_mask = source_ref.str.contains("objective_ai_inferred", case=False, regex=False)
    macro_rule_mask = source_ref.str.contains("objective_macro_rule", case=False, regex=False)
    indicator_mask = source_ref.str.contains("wb_indicator_event", case=False, regex=False)
    external_direct_mask = (~inferred_mask) & (~macro_rule_mask) & (~indicator_mask)
    source_non_empty = source_ref.replace("unspecified", "").replace("nan", "")
    source_count = int((source_non_empty != "").sum())

    direct_registry = registry[~inferred_mask].copy()
    external_direct_registry = registry[external_direct_mask].copy()
    indicator_registry = registry[indicator_mask].copy()
    macro_rule_registry = registry[macro_rule_mask].copy()
    inferred_registry = registry[inferred_mask].copy()
    evidence_a_registry = registry[registry["evidence_grade"] == "A"].copy()
    evidence_b_registry = registry[registry["evidence_grade"] == "B"].copy()
    evidence_c_registry = registry[registry["evidence_grade"] == "C"].copy()
    direct_iso = set(direct_registry["iso3"].astype(str).str.upper().tolist())
    external_direct_iso = set(external_direct_registry["iso3"].astype(str).str.upper().tolist())
    indicator_iso = set(indicator_registry["iso3"].astype(str).str.upper().tolist())
    macro_rule_iso = set(macro_rule_registry["iso3"].astype(str).str.upper().tolist())
    inferred_iso = set(inferred_registry["iso3"].astype(str).str.upper().tolist())
    evidence_a_iso = set(evidence_a_registry["iso3"].astype(str).str.upper().tolist())
    evidence_b_iso = set(evidence_b_registry["iso3"].astype(str).str.upper().tolist())
    evidence_c_iso = set(evidence_c_registry["iso3"].astype(str).str.upper().tolist())

    covered_country = pd.DataFrame({"iso3": sorted(overlap)})
    covered_direct_country = pd.DataFrame({"iso3": sorted(panel_iso.intersection(direct_iso))})
    covered_external_country = pd.DataFrame({"iso3": sorted(panel_iso.intersection(external_direct_iso))})
    covered_indicator_country = pd.DataFrame({"iso3": sorted(panel_iso.intersection(indicator_iso))})
    covered_macro_rule_country = pd.DataFrame({"iso3": sorted(panel_iso.intersection(macro_rule_iso))})
    covered_inferred_country = pd.DataFrame({"iso3": sorted(panel_iso.intersection(inferred_iso))})
    covered_evidence_a_country = pd.DataFrame({"iso3": sorted(panel_iso.intersection(evidence_a_iso))})
    by_continent_panel = (
        panel_country.groupby("continent", as_index=False)
        .agg(panel_countries=("iso3", "nunique"))
        .sort_values("panel_countries", ascending=False)
    )
    by_continent_cov = (
        panel_country.merge(covered_country, on="iso3", how="inner")
        .groupby("continent", as_index=False)
        .agg(covered_countries=("iso3", "nunique"))
    )
    by_continent_direct = (
        panel_country.merge(covered_direct_country, on="iso3", how="inner")
        .groupby("continent", as_index=False)
        .agg(direct_countries=("iso3", "nunique"))
    )
    by_continent_external = (
        panel_country.merge(covered_external_country, on="iso3", how="inner")
        .groupby("continent", as_index=False)
        .agg(external_direct_countries=("iso3", "nunique"))
    )
    by_continent_indicator = (
        panel_country.merge(covered_indicator_country, on="iso3", how="inner")
        .groupby("continent", as_index=False)
        .agg(objective_indicator_countries=("iso3", "nunique"))
    )
    by_continent_macro = (
        panel_country.merge(covered_macro_rule_country, on="iso3", how="inner")
        .groupby("continent", as_index=False)
        .agg(objective_macro_countries=("iso3", "nunique"))
    )
    by_continent_inferred = (
        panel_country.merge(covered_inferred_country, on="iso3", how="inner")
        .groupby("continent", as_index=False)
        .agg(inferred_countries=("iso3", "nunique"))
    )
    by_continent_evidence_a = (
        panel_country.merge(covered_evidence_a_country, on="iso3", how="inner")
        .groupby("continent", as_index=False)
        .agg(evidence_a_countries=("iso3", "nunique"))
    )
    by_continent = by_continent_panel.merge(by_continent_cov, on="continent", how="left")
    by_continent = by_continent.merge(by_continent_direct, on="continent", how="left")
    by_continent = by_continent.merge(by_continent_external, on="continent", how="left")
    by_continent = by_continent.merge(by_continent_indicator, on="continent", how="left")
    by_continent = by_continent.merge(by_continent_macro, on="continent", how="left")
    by_continent = by_continent.merge(by_continent_inferred, on="continent", how="left")
    by_continent = by_continent.merge(by_continent_evidence_a, on="continent", how="left")
    by_continent["covered_countries"] = by_continent["covered_countries"].fillna(0).astype(int)
    by_continent["direct_countries"] = by_continent["direct_countries"].fillna(0).astype(int)
    by_continent["external_direct_countries"] = by_continent["external_direct_countries"].fillna(0).astype(int)
    by_continent["objective_indicator_countries"] = by_continent["objective_indicator_countries"].fillna(0).astype(int)
    by_continent["objective_macro_countries"] = by_continent["objective_macro_countries"].fillna(0).astype(int)
    by_continent["inferred_countries"] = by_continent["inferred_countries"].fillna(0).astype(int)
    by_continent["evidence_a_countries"] = by_continent["evidence_a_countries"].fillna(0).astype(int)
    by_continent["coverage_ratio"] = by_continent["covered_countries"] / by_continent["panel_countries"].clip(lower=1)
    by_continent["coverage_ratio_direct"] = by_continent["direct_countries"] / by_continent["panel_countries"].clip(lower=1)
    by_continent["coverage_ratio_external_direct"] = (
        by_continent["external_direct_countries"] / by_continent["panel_countries"].clip(lower=1)
    )
    by_continent["coverage_ratio_objective_indicator"] = (
        by_continent["objective_indicator_countries"] / by_continent["panel_countries"].clip(lower=1)
    )
    by_continent["coverage_ratio_objective_macro"] = (
        by_continent["objective_macro_countries"] / by_continent["panel_countries"].clip(lower=1)
    )
    by_continent["coverage_ratio_inferred"] = (
        by_continent["inferred_countries"] / by_continent["panel_countries"].clip(lower=1)
    )
    by_continent["coverage_ratio_evidence_a"] = (
        by_continent["evidence_a_countries"] / by_continent["panel_countries"].clip(lower=1)
    )
    by_continent = by_continent.sort_values(["coverage_ratio", "panel_countries"], ascending=[False, False])
    by_continent.to_csv(DATA_PROCESSED / "policy_event_registry_coverage_by_continent.csv", index=False)

    return {
        "status": "ok",
        "registry_file": str(DATA_RAW / "policy_events_registry.csv"),
        "event_rows": int(len(registry)),
        "policy_types": int(registry["policy_name"].nunique()),
        "source_refs": int(source_count),
        "direct_event_rows": int(len(direct_registry)),
        "external_direct_event_rows": int(len(external_direct_registry)),
        "objective_indicator_event_rows": int(len(indicator_registry)),
        "objective_macro_event_rows": int(len(macro_rule_registry)),
        "ai_inferred_event_rows": int(len(inferred_registry)),
        "evidence_a_event_rows": int(len(evidence_a_registry)),
        "evidence_b_event_rows": int(len(evidence_b_registry)),
        "evidence_c_event_rows": int(len(evidence_c_registry)),
        "iso3_covered": int(len(overlap)),
        "iso3_share_of_sample": float(len(overlap) / max(len(panel_iso), 1)),
        "iso3_covered_direct": int(len(panel_iso.intersection(direct_iso))),
        "iso3_share_direct": float(len(panel_iso.intersection(direct_iso)) / max(len(panel_iso), 1)),
        "iso3_covered_external_direct": int(len(panel_iso.intersection(external_direct_iso))),
        "iso3_share_external_direct": float(len(panel_iso.intersection(external_direct_iso)) / max(len(panel_iso), 1)),
        "iso3_covered_objective_indicator": int(len(panel_iso.intersection(indicator_iso))),
        "iso3_share_objective_indicator": float(len(panel_iso.intersection(indicator_iso)) / max(len(panel_iso), 1)),
        "iso3_covered_objective_macro": int(len(panel_iso.intersection(macro_rule_iso))),
        "iso3_share_objective_macro": float(len(panel_iso.intersection(macro_rule_iso)) / max(len(panel_iso), 1)),
        "iso3_covered_ai_inferred": int(len(panel_iso.intersection(inferred_iso))),
        "iso3_share_ai_inferred": float(len(panel_iso.intersection(inferred_iso)) / max(len(panel_iso), 1)),
        "iso3_covered_evidence_a": int(len(panel_iso.intersection(evidence_a_iso))),
        "iso3_share_evidence_a": float(len(panel_iso.intersection(evidence_a_iso)) / max(len(panel_iso), 1)),
        "iso3_covered_evidence_b": int(len(panel_iso.intersection(evidence_b_iso))),
        "iso3_share_evidence_b": float(len(panel_iso.intersection(evidence_b_iso)) / max(len(panel_iso), 1)),
        "iso3_covered_evidence_c": int(len(panel_iso.intersection(evidence_c_iso))),
        "iso3_share_evidence_c": float(len(panel_iso.intersection(evidence_c_iso)) / max(len(panel_iso), 1)),
        "iso3_missing": int(len(missing_iso)),
        "missing_iso3_sample": missing_iso[:30],
        "active_year_range": [
            int(max(year_min, registry["start_year"].min())),
            int(min(year_max, registry["end_year"].max())),
        ],
        "mean_intensity": float(registry["policy_intensity"].mean()),
        "coverage_by_continent": by_continent.to_dict(orient="records"),
    }


def _rank_pct_stable(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() == 0:
        return pd.Series(0.5, index=series.index, dtype=float)
    ranked = vals.rank(method="average", pct=True)
    fill = float(ranked.dropna().mean()) if ranked.notna().any() else 0.5
    return ranked.fillna(fill).astype(float)


def _build_city_policy_salience(panel: pd.DataFrame) -> pd.DataFrame:
    """Build a deterministic baseline salience score used for city-level policy rollout."""
    cols = ["city_id", "iso3", "year"]
    candidate_specs = [
        ("log_population", 0.50, None),
        ("population", 0.50, np.log1p),
        ("poi_total", 0.20, np.log1p),
        ("road_length_km_total", 0.15, np.log1p),
        ("viirs_ntl_mean", 0.10, np.log1p),
        ("intersection_density", 0.05, np.log1p),
    ]
    available_specs = [(c, w, tr) for c, w, tr in candidate_specs if c in panel.columns]
    cols.extend([c for c, _, _ in available_specs])

    base = (
        panel[cols]
        .copy()
        .sort_values(["city_id", "year"])
        .drop_duplicates(subset=["city_id"], keep="first")
        .reset_index(drop=True)
    )
    base["city_id"] = base["city_id"].astype(str)
    base["iso3"] = base["iso3"].astype(str).str.upper()

    score = pd.Series(0.0, index=base.index, dtype=float)
    total_weight = 0.0
    for col, weight, transform in available_specs:
        vals = pd.to_numeric(base[col], errors="coerce")
        if transform is not None:
            vals = vals.map(lambda x: float(transform(max(float(x), 0.0))) if pd.notna(x) else np.nan)
        score += float(weight) * _rank_pct_stable(vals)
        total_weight += float(weight)

    if total_weight > 0.0:
        base["policy_city_salience_score"] = (score / float(total_weight)).astype(float)
    else:
        base["policy_city_salience_score"] = 0.5

    base["policy_city_rank_within_iso"] = (
        base.groupby("iso3")["policy_city_salience_score"].rank(method="first", ascending=False).astype(int)
    )
    base["policy_city_count_within_iso"] = base.groupby("iso3")["city_id"].transform("nunique").astype(int)
    return base[
        [
            "city_id",
            "iso3",
            "policy_city_salience_score",
            "policy_city_rank_within_iso",
            "policy_city_count_within_iso",
        ]
    ].copy()


def _assign_city_rollout_from_country_cohorts(
    panel: pd.DataFrame,
    cohort_map: Dict[str, int],
    *,
    city_salience: pd.DataFrame,
    max_year: int,
    max_rollout_lag: int = 2,
) -> pd.DataFrame:
    """Map country-level cohorts into a common country shock for all cities.

    The policy registry is ISO3-level only. Creating within-country rollout from
    city salience fabricates city treatment timing, so all cities in a treated
    country inherit the same cohort year.
    """
    city_tbl = city_salience[["city_id", "iso3"]].drop_duplicates("city_id").copy()
    city_tbl["city_id"] = city_tbl["city_id"].astype(str)
    city_tbl["iso3"] = city_tbl["iso3"].astype(str).str.upper()
    city_tbl["treated_city"] = 0
    city_tbl["treatment_cohort_year"] = 9999

    if not cohort_map:
        return city_tbl[["city_id", "treated_city", "treatment_cohort_year"]]

    cohort_std = {str(k).upper(): int(v) for k, v in cohort_map.items()}
    city_tbl["treatment_cohort_year"] = city_tbl["iso3"].map(cohort_std).fillna(9999).astype(int)
    city_tbl["treated_city"] = (city_tbl["treatment_cohort_year"] < 9999).astype(int)
    city_tbl.loc[city_tbl["treated_city"] == 0, "treatment_cohort_year"] = 9999

    return city_tbl[["city_id", "treated_city", "treatment_cohort_year"]].copy()


def _apply_policy_design_from_registry(
    panel: pd.DataFrame,
    registry: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, dict]:
    """Apply event-based treatment assignment and cohort definition."""
    out = panel.copy()
    reg = registry.copy()
    if "evidence_grade" not in reg.columns:
        reg["evidence_grade"] = reg["source_ref"].astype(str).map(_policy_event_evidence_grade)
    if "evidence_score" not in reg.columns:
        reg["evidence_score"] = reg["evidence_grade"].map({"A": 1.0, "B": 0.75, "C": 0.45}).fillna(0.60)
    if "policy_type_coarse" not in reg.columns:
        reg["policy_type_coarse"] = reg["policy_name"].astype(str).map(_policy_type_coarse)
    if "policy_subtype" not in reg.columns:
        reg["policy_subtype"] = reg["policy_name"].astype(str).map(_policy_subtype)
    if "policy_treatment_bucket" not in reg.columns:
        reg["policy_treatment_bucket"] = reg["policy_name"].astype(str).map(_policy_treatment_bucket)
    if "policy_evidence_track" not in reg.columns:
        reg["policy_evidence_track"] = reg["source_ref"].astype(str).map(_policy_evidence_track)
    if "policy_direct_core_evidence_eligible" not in reg.columns:
        reg["policy_direct_core_evidence_eligible"] = reg["source_ref"].astype(str).map(
            _policy_direct_core_evidence_eligible
        )
    reg["policy_type_coarse"] = reg["policy_type_coarse"].fillna("other").astype(str)
    reg["policy_subtype"] = reg["policy_subtype"].fillna("other").astype(str)
    reg["policy_treatment_bucket"] = reg["policy_treatment_bucket"].fillna(reg["policy_type_coarse"]).astype(str)
    reg["policy_evidence_track"] = reg["policy_evidence_track"].fillna("external_direct").astype(str)
    reg["policy_direct_core_evidence_eligible"] = (
        pd.to_numeric(reg["policy_direct_core_evidence_eligible"], errors="coerce").fillna(0).astype(int)
    )

    cohort_all_sources = reg.groupby("iso3", as_index=True)["start_year"].min().to_dict()
    treated_iso_all_sources = set(cohort_all_sources.keys())

    src_ref = reg["source_ref"].astype(str)
    inferred_mask = src_ref.str.contains("objective_ai_inferred", case=False, regex=False)
    macro_rule_mask = src_ref.str.contains("objective_macro_rule", case=False, regex=False)
    indicator_mask = src_ref.str.contains("wb_indicator_event", case=False, regex=False)
    external_direct_mask = (~inferred_mask) & (~macro_rule_mask) & (~indicator_mask)
    direct_registry = reg.loc[~inferred_mask].copy()
    external_direct_registry = reg.loc[external_direct_mask].copy()
    indicator_registry = reg.loc[indicator_mask].copy()
    macro_rule_registry = reg.loc[macro_rule_mask].copy()
    inferred_registry = reg.loc[inferred_mask].copy()
    evidence_a_registry = reg.loc[reg["evidence_grade"] == "A"].copy()
    evidence_ab_registry = reg.loc[reg["evidence_grade"].isin(["A", "B"])].copy()

    direct_cohort = direct_registry.groupby("iso3", as_index=True)["start_year"].min().to_dict() if not direct_registry.empty else {}
    external_direct_cohort = (
        external_direct_registry.groupby("iso3", as_index=True)["start_year"].min().to_dict()
        if not external_direct_registry.empty
        else {}
    )
    indicator_cohort = (
        indicator_registry.groupby("iso3", as_index=True)["start_year"].min().to_dict()
        if not indicator_registry.empty
        else {}
    )
    macro_rule_cohort = (
        macro_rule_registry.groupby("iso3", as_index=True)["start_year"].min().to_dict()
        if not macro_rule_registry.empty
        else {}
    )
    inferred_cohort = (
        inferred_registry.groupby("iso3", as_index=True)["start_year"].min().to_dict() if not inferred_registry.empty else {}
    )
    evidence_a_cohort = (
        evidence_a_registry.groupby("iso3", as_index=True)["start_year"].min().to_dict() if not evidence_a_registry.empty else {}
    )
    evidence_ab_cohort = (
        evidence_ab_registry.groupby("iso3", as_index=True)["start_year"].min().to_dict() if not evidence_ab_registry.empty else {}
    )

    treated_iso_direct = set(direct_cohort.keys())
    treated_iso_external_direct = set(external_direct_cohort.keys())
    treated_iso_indicator = set(indicator_cohort.keys())
    treated_iso_macro_rule = set(macro_rule_cohort.keys())
    treated_iso_inferred = set(inferred_cohort.keys())
    treated_iso_evidence_a = set(evidence_a_cohort.keys())
    treated_iso_evidence_ab = set(evidence_ab_cohort.keys())

    # Pre-registered primary treatment track: direct_core is selected a priori
    # as the most externally valid treatment definition. Other tracks are reported
    # only as robustness. See docs/treatment_preregistration.md for rationale.
    PRIMARY_TREATMENT_TRACK = "direct_core"

    principal_treatment_track = "all_sources_fallback"
    principal_cohort = cohort_all_sources

    direct_core_threshold = 0.70
    direct_core_year_cap = int(max(config.time_range.start_year, config.time_range.end_year - 2))
    direct_core_source = "external_direct_all"
    external_high_conf = external_direct_registry[external_direct_registry["evidence_grade"] == "A"].copy()
    if not external_high_conf.empty:
        direct_core_base = external_high_conf
        direct_core_source = "external_direct_grade_A"
    elif not external_direct_registry.empty:
        direct_core_base = external_direct_registry.copy()
    else:
        direct_core_base = direct_registry.copy()
        direct_core_source = "direct_non_ai_all"

    direct_core_registry = direct_core_base[
        (direct_core_base["policy_intensity"] >= float(direct_core_threshold))
        & (direct_core_base["start_year"] <= int(direct_core_year_cap))
    ].copy()
    if direct_core_registry.empty and not direct_core_base.empty:
        fallback_q = float(direct_core_base["policy_intensity"].quantile(0.60))
        direct_core_registry = direct_core_base[
            (direct_core_base["policy_intensity"] >= fallback_q)
            & (direct_core_base["start_year"] <= int(direct_core_year_cap))
        ].copy()
        direct_core_threshold = float(fallback_q)
    direct_core_cohort = (
        direct_core_registry.groupby("iso3", as_index=True)["start_year"].min().to_dict() if not direct_core_registry.empty else {}
    )
    direct_core_type_cohorts: Dict[str, Dict[str, int]] = {}
    for coarse_type in ["infra", "digital", "eco_reg"]:
        sub = direct_core_registry.loc[direct_core_registry["policy_treatment_bucket"] == coarse_type].copy()
        direct_core_type_cohorts[coarse_type] = (
            sub.groupby("iso3", as_index=True)["start_year"].min().to_dict() if not sub.empty else {}
        )
    treated_iso_direct_core = set(direct_core_cohort.keys())

    panel_iso = set(out["iso3"].astype(str).str.upper().dropna().tolist())

    def _panel_share(cohort_map: Dict[str, int]) -> float:
        if not panel_iso:
            return 0.0
        covered = panel_iso.intersection({str(k).upper() for k in cohort_map.keys()})
        return float(len(covered) / max(len(panel_iso), 1))

    direct_core_share = _panel_share(direct_core_cohort)
    external_direct_share = _panel_share(external_direct_cohort)
    direct_share = _panel_share(direct_cohort)

    if direct_core_cohort and 0.08 <= direct_core_share <= 0.70:
        principal_treatment_track = "direct_core"
        principal_cohort = direct_core_cohort
    elif external_direct_cohort and 0.05 <= external_direct_share <= 0.70:
        principal_treatment_track = "external_direct"
        principal_cohort = external_direct_cohort
    elif direct_cohort and 0.05 <= direct_share <= 0.80:
        principal_treatment_track = "direct_non_ai_fallback"
        principal_cohort = direct_cohort
    elif external_direct_cohort:
        principal_treatment_track = "external_direct_high_coverage_fallback"
        principal_cohort = external_direct_cohort
    elif direct_cohort:
        principal_treatment_track = "direct_non_ai_fallback"
        principal_cohort = direct_cohort

    treated_iso_principal = set(principal_cohort.keys())

    external_intense_quantile = 0.65
    external_intense_threshold = 0.0
    treated_iso_external_intense: set[str] = set()
    external_intense_cohort: Dict[str, int] = {}
    external_intense_peak_cohort: Dict[str, int] = {}
    external_intense_peak_rule = "none"
    if not external_direct_registry.empty:
        ext_peak = (
            external_direct_registry.groupby("iso3", as_index=True)["policy_intensity"]
            .max()
            .dropna()
            .astype(float)
        )
        ext_peak = ext_peak[ext_peak > 0.0]
        if not ext_peak.empty:
            for q in [0.55, 0.60, 0.65, 0.70]:
                th = float(ext_peak.quantile(q))
                share = float(np.mean(ext_peak >= th))
                if 0.25 <= share <= 0.60:
                    external_intense_quantile = float(q)
                    break
            external_intense_threshold = float(ext_peak.quantile(float(external_intense_quantile)))
            treated_iso_external_intense = set(ext_peak[ext_peak >= external_intense_threshold].index.astype(str).tolist())
            if len(treated_iso_external_intense) < 6:
                top_n = int(max(6, np.ceil(0.30 * len(ext_peak))))
                top_peak = ext_peak.sort_values(ascending=False).head(top_n)
                treated_iso_external_intense = set(top_peak.index.astype(str).tolist())
                external_intense_threshold = float(top_peak.min())
                external_intense_quantile = float(np.mean(ext_peak <= external_intense_threshold))
            ext_reg_intense = external_direct_registry[
                external_direct_registry["iso3"].astype(str).isin(treated_iso_external_intense)
            ].copy()
            if not ext_reg_intense.empty:
                external_intense_cohort = ext_reg_intense.groupby("iso3", as_index=True)["start_year"].min().to_dict()
                peak_year = (
                    ext_reg_intense.groupby(["iso3", "start_year"], as_index=False)
                    .agg(
                        year_peak_intensity=("policy_intensity", "max"),
                        year_mean_intensity=("policy_intensity", "mean"),
                        year_event_count=("policy_name", "size"),
                    )
                    .sort_values(
                        ["iso3", "year_peak_intensity", "year_mean_intensity", "year_event_count", "start_year"],
                        ascending=[True, False, False, False, True],
                    )
                    .drop_duplicates(subset=["iso3"], keep="first")
                    .reset_index(drop=True)
                )
                if not peak_year.empty:
                    external_intense_peak_cohort = (
                        peak_year.set_index("iso3")["start_year"].astype(int).to_dict()
                    )
                    external_intense_peak_rule = "year_max_policy_intensity_external_direct"
    if not external_intense_peak_cohort and external_intense_cohort:
        external_intense_peak_cohort = external_intense_cohort.copy()
        external_intense_peak_rule = "fallback_first_start_year"

    city_salience = _build_city_policy_salience(out)
    out = out.merge(city_salience, on=["city_id", "iso3"], how="left")
    out["policy_city_salience_score"] = pd.to_numeric(out["policy_city_salience_score"], errors="coerce").fillna(0.5).astype(float)
    out["policy_city_rank_within_iso"] = pd.to_numeric(out["policy_city_rank_within_iso"], errors="coerce").fillna(1).astype(int)
    out["policy_city_count_within_iso"] = pd.to_numeric(out["policy_city_count_within_iso"], errors="coerce").fillna(1).astype(int)
    out["policy_assignment_scope"] = "country_rule"
    out["policy_assignment_city_rule"] = "country_common_shock_no_within_country_rollout"
    out["causal_unit_recommended"] = "iso3_year"
    out["policy_assignment_scope"] = "country_registry"
    out["policy_assignment_city_rule"] = "country_common_shock_no_within_country_rollout"
    out["causal_unit_recommended"] = "iso3_year"

    def _assign_rollout_variant(prefix: str, cohort_map: Dict[str, int]) -> None:
        city_rollout = _assign_city_rollout_from_country_cohorts(
            out,
            cohort_map,
            city_salience=city_salience,
            max_year=int(config.time_range.end_year),
            max_rollout_lag=2,
        )
        treat_map = city_rollout.set_index("city_id")["treated_city"].astype(int).to_dict()
        cohort_city_map = city_rollout.set_index("city_id")["treatment_cohort_year"].astype(int).to_dict()
        treat_col = f"treated_city{prefix}"
        cohort_col = f"treatment_cohort_year{prefix}"
        post_col = f"post_policy{prefix}"
        did_col = f"did_treatment{prefix}"
        out[treat_col] = out["city_id"].astype(str).map(treat_map).fillna(0).astype(int)
        out[cohort_col] = out["city_id"].astype(str).map(cohort_city_map).fillna(9999).astype(int)
        out[post_col] = ((out["year"] >= out[cohort_col]) & (out[treat_col] == 1)).astype(int)
        out[did_col] = out[treat_col] * out[post_col]

    _assign_rollout_variant("_all_sources", cohort_all_sources)
    _assign_rollout_variant("", principal_cohort)
    _assign_rollout_variant("_direct", direct_cohort)
    _assign_rollout_variant("_external_direct", external_direct_cohort)
    _assign_rollout_variant("_intense_external_direct", external_intense_cohort)
    _assign_rollout_variant("_intense_external_peak", external_intense_peak_cohort)
    _assign_rollout_variant("_objective_indicator", indicator_cohort)
    _assign_rollout_variant("_objective_macro", macro_rule_cohort)
    _assign_rollout_variant("_ai_inferred", inferred_cohort)
    _assign_rollout_variant("_direct_core", direct_core_cohort)
    _assign_rollout_variant("_direct_core_infra", direct_core_type_cohorts.get("infra", {}))
    _assign_rollout_variant("_direct_core_digital", direct_core_type_cohorts.get("digital", {}))
    _assign_rollout_variant("_direct_core_eco_reg", direct_core_type_cohorts.get("eco_reg", {}))
    _assign_rollout_variant("_evidence_a", evidence_a_cohort)
    _assign_rollout_variant("_evidence_ab", evidence_ab_cohort)

    expanded_rows: List[dict] = []
    for row in reg.itertuples(index=False):
        start = int(max(row.start_year, config.time_range.start_year))
        end = int(min(row.end_year, config.time_range.end_year))
        src = str(row.source_ref)
        src_lower = src.lower()
        if "objective_ai_inferred" in src_lower:
            source_type = "ai_inferred"
        elif "wb_indicator_event" in src_lower:
            source_type = "objective_indicator"
        elif "objective_macro_rule" in src_lower:
            source_type = "objective_macro"
        else:
            source_type = "external_direct"
        for year in range(start, end + 1):
            expanded_rows.append(
                {
                    "iso3": str(row.iso3),
                    "year": int(year),
                    "policy_name": str(getattr(row, "policy_name", "unknown")),
                    "policy_type_coarse": str(getattr(row, "policy_type_coarse", "other")),
                    "policy_subtype": str(getattr(row, "policy_subtype", "other")),
                    "policy_treatment_bucket": str(getattr(row, "policy_treatment_bucket", "other")),
                    "policy_evidence_track": str(getattr(row, "policy_evidence_track", source_type)),
                    "policy_direct_core_evidence_eligible": int(
                        getattr(row, "policy_direct_core_evidence_eligible", int(source_type == "external_direct"))
                    ),
                    "is_new_event": int(int(year) == int(start)),
                    "policy_intensity": float(row.policy_intensity),
                    "source_ref": str(row.source_ref),
                    "source_type": source_type,
                    "evidence_score": float(getattr(row, "evidence_score", 0.60)),
                }
            )

    if expanded_rows:
        intensity_df = pd.DataFrame(expanded_rows)
        intensity_df["intensity_external_direct"] = np.where(
            intensity_df["source_type"].astype(str) == "external_direct",
            pd.to_numeric(intensity_df["policy_intensity"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            0.0,
        )
        intensity_df["intensity_objective_indicator"] = np.where(
            intensity_df["source_type"].astype(str) == "objective_indicator",
            pd.to_numeric(intensity_df["policy_intensity"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            0.0,
        )
        intensity_df["intensity_objective_macro"] = np.where(
            intensity_df["source_type"].astype(str) == "objective_macro",
            pd.to_numeric(intensity_df["policy_intensity"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            0.0,
        )
        intensity_df["intensity_ai_inferred"] = np.where(
            intensity_df["source_type"].astype(str) == "ai_inferred",
            pd.to_numeric(intensity_df["policy_intensity"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            0.0,
        )
        intensity_df["intensity_direct_non_ai"] = np.where(
            intensity_df["source_type"].astype(str) != "ai_inferred",
            pd.to_numeric(intensity_df["policy_intensity"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            0.0,
        )
        type_year = (
            intensity_df.groupby(["iso3", "year", "policy_type_coarse"], as_index=False)
            .agg(
                coarse_policy_event_count=("policy_intensity", "size"),
                coarse_policy_intensity_max=("policy_intensity", "max"),
            )
            .copy()
        )
        if not type_year.empty:
            count_pivot = (
                type_year.pivot(index=["iso3", "year"], columns="policy_type_coarse", values="coarse_policy_event_count")
                .fillna(0.0)
                .reset_index()
            )
            intensity_pivot = (
                type_year.pivot(index=["iso3", "year"], columns="policy_type_coarse", values="coarse_policy_intensity_max")
                .fillna(0.0)
                .reset_index()
            )
            count_pivot = count_pivot.rename(
                columns={c: f"policy_event_count_{c}_iso_year" for c in count_pivot.columns if c not in {"iso3", "year"}}
            )
            intensity_pivot = intensity_pivot.rename(
                columns={c: f"policy_intensity_{c}_iso_year" for c in intensity_pivot.columns if c not in {"iso3", "year"}}
            )
        else:
            count_pivot = pd.DataFrame(columns=["iso3", "year"])
            intensity_pivot = pd.DataFrame(columns=["iso3", "year"])
        intensity_df = (
            intensity_df.groupby(["iso3", "year"], as_index=False)
            .agg(
                policy_intensity=("policy_intensity", "max"),
                policy_intensity_sum_iso_year=("policy_intensity", "sum"),
                policy_intensity_mean_iso_year=("policy_intensity", "mean"),
                policy_event_count_iso_year=("policy_intensity", "size"),
                policy_event_type_count_iso_year=("policy_name", "nunique"),
                policy_event_coarse_type_count_iso_year=("policy_type_coarse", "nunique"),
                policy_event_new_count_iso_year=("is_new_event", "sum"),
                source_ref=("source_ref", "first"),
                evidence_score=("evidence_score", "max"),
                policy_intensity_external_direct=("intensity_external_direct", "max"),
                policy_intensity_objective_indicator=("intensity_objective_indicator", "max"),
                policy_intensity_objective_macro=("intensity_objective_macro", "max"),
                policy_intensity_ai_inferred=("intensity_ai_inferred", "max"),
                policy_intensity_direct_non_ai=("intensity_direct_non_ai", "max"),
                has_external_direct=("source_type", lambda s: int(np.any(pd.Series(s).astype(str) == "external_direct"))),
                has_objective_indicator=("source_type", lambda s: int(np.any(pd.Series(s).astype(str) == "objective_indicator"))),
                has_objective_macro=("source_type", lambda s: int(np.any(pd.Series(s).astype(str) == "objective_macro"))),
                has_inferred=("source_type", lambda s: int(np.any(pd.Series(s).astype(str) == "ai_inferred"))),
            )
            .copy()
        )
        intensity_df = intensity_df.merge(count_pivot, on=["iso3", "year"], how="left")
        intensity_df = intensity_df.merge(intensity_pivot, on=["iso3", "year"], how="left")
        intensity_df["policy_source_type"] = np.where(
            intensity_df["has_external_direct"] == 1,
            "external_direct",
            np.where(
                intensity_df["has_objective_indicator"] == 1,
                "objective_indicator",
                np.where(intensity_df["has_inferred"] == 1, "ai_inferred", "none"),
            ),
        )
        intensity_df["policy_source_type"] = np.where(
            (intensity_df["policy_source_type"] == "none") & (intensity_df["has_objective_macro"] == 1),
            "objective_macro",
            intensity_df["policy_source_type"],
        )
        intensity_df["policy_evidence_grade"] = np.where(
            intensity_df["evidence_score"] >= 0.95,
            "A",
            np.where(intensity_df["evidence_score"] >= 0.70, "B", "C"),
        )
        out = out.merge(intensity_df, on=["iso3", "year"], how="left")
        out["policy_intensity_all_sources"] = out["policy_intensity"].fillna(0.0).astype(float)
        out["policy_intensity_external_direct"] = out["policy_intensity_external_direct"].fillna(0.0).astype(float)
        out["policy_intensity_objective_indicator"] = out["policy_intensity_objective_indicator"].fillna(0.0).astype(float)
        out["policy_intensity_objective_macro"] = out["policy_intensity_objective_macro"].fillna(0.0).astype(float)
        out["policy_intensity_ai_inferred"] = out["policy_intensity_ai_inferred"].fillna(0.0).astype(float)
        out["policy_intensity_direct_non_ai"] = out["policy_intensity_direct_non_ai"].fillna(0.0).astype(float)
        out["policy_intensity_sum_iso_year"] = out["policy_intensity_sum_iso_year"].fillna(0.0).astype(float)
        out["policy_intensity_mean_iso_year"] = out["policy_intensity_mean_iso_year"].fillna(0.0).astype(float)
        out["policy_event_count_iso_year"] = out["policy_event_count_iso_year"].fillna(0).astype(int)
        out["policy_event_type_count_iso_year"] = out["policy_event_type_count_iso_year"].fillna(0).astype(int)
        out["policy_event_coarse_type_count_iso_year"] = out["policy_event_coarse_type_count_iso_year"].fillna(0).astype(int)
        out["policy_event_new_count_iso_year"] = out["policy_event_new_count_iso_year"].fillna(0).astype(int)
        for coarse_type in ["infra", "digital", "eco_reg"]:
            count_col = f"policy_event_count_{coarse_type}_iso_year"
            intensity_col = f"policy_intensity_{coarse_type}_iso_year"
            if count_col not in out.columns:
                out[count_col] = 0.0
            if intensity_col not in out.columns:
                out[intensity_col] = 0.0
            out[count_col] = pd.to_numeric(out[count_col], errors="coerce").fillna(0).astype(int)
            out[intensity_col] = pd.to_numeric(out[intensity_col], errors="coerce").fillna(0.0).astype(float)
        if principal_treatment_track == "external_direct":
            out["policy_intensity"] = out["policy_intensity_external_direct"].astype(float)
        elif principal_treatment_track == "direct_non_ai_fallback":
            out["policy_intensity"] = out["policy_intensity_direct_non_ai"].astype(float)
        else:
            out["policy_intensity"] = out["policy_intensity_all_sources"].astype(float)
        # News/policy pressure proxy at country-year level (mapped to cities by iso3-year).
        out["policy_news_proxy_score"] = (
            np.log1p(out["policy_event_count_iso_year"].astype(float))
            + 0.70 * np.log1p(out["policy_event_new_count_iso_year"].astype(float))
            + 0.50 * out["policy_intensity_sum_iso_year"].astype(float)
        ).astype(float)
        out["policy_source_ref"] = out["source_ref"].fillna("none")
        out["policy_source_type"] = out["policy_source_type"].fillna("none").astype(str)
        out["policy_evidence_grade"] = out["policy_evidence_grade"].fillna("none").astype(str)
        out["policy_evidence_score"] = out["evidence_score"].fillna(0.0).astype(float)
        out["policy_dose"] = out["post_policy"].astype(float) * out["policy_intensity"]
        out["policy_dose_external_direct"] = out["post_policy_external_direct"].astype(float) * out["policy_intensity_external_direct"]
        out["policy_dose_intense_external_direct"] = (
            out["post_policy_intense_external_direct"].astype(float) * out["policy_intensity_external_direct"]
        )
        out["policy_dose_intense_external_peak"] = (
            out["post_policy_intense_external_peak"].astype(float) * out["policy_intensity_external_direct"]
        )
        iso_peak = out.groupby("iso3")["policy_intensity"].max()
        iso_mean = out.groupby("iso3")["policy_intensity"].mean()
        iso_active_years = out.groupby("iso3")["policy_intensity"].apply(lambda s: int(np.sum(pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0)))
        out["policy_intensity_peak_iso"] = out["iso3"].map(iso_peak).fillna(0.0).astype(float)
        out["policy_intensity_mean_iso"] = out["iso3"].map(iso_mean).fillna(0.0).astype(float)
        out["policy_event_active_years_iso"] = out["iso3"].map(iso_active_years).fillna(0).astype(int)

        # Country-year event momentum (YoY) then broadcast to city-year rows.
        iso_event_base = (
            out[
                [
                    "iso3",
                    "year",
                    "policy_event_count_iso_year",
                    "policy_intensity_sum_iso_year",
                ]
            ]
            .drop_duplicates(["iso3", "year"])
            .sort_values(["iso3", "year"])
            .copy()
        )
        iso_event_base["policy_event_count_iso_year_yoy"] = (
            iso_event_base.groupby("iso3")["policy_event_count_iso_year"].diff().fillna(0.0).astype(float)
        )
        iso_event_base["policy_intensity_sum_iso_year_yoy"] = (
            iso_event_base.groupby("iso3")["policy_intensity_sum_iso_year"].diff().fillna(0.0).astype(float)
        )
        out = out.merge(
            iso_event_base[
                [
                    "iso3",
                    "year",
                    "policy_event_count_iso_year_yoy",
                    "policy_intensity_sum_iso_year_yoy",
                ]
            ],
            on=["iso3", "year"],
            how="left",
        )
        out["policy_event_count_iso_year_yoy"] = out["policy_event_count_iso_year_yoy"].fillna(0.0).astype(float)
        out["policy_intensity_sum_iso_year_yoy"] = out["policy_intensity_sum_iso_year_yoy"].fillna(0.0).astype(float)
        out = out.drop(
            columns=[
                c
                for c in [
                    "source_ref",
                    "evidence_score",
                    "has_external_direct",
                    "has_objective_indicator",
                    "has_objective_macro",
                    "has_inferred",
                ]
                if c in out.columns
            ]
        )
    else:
        out["policy_intensity"] = 0.0
        out["policy_intensity_all_sources"] = 0.0
        out["policy_intensity_external_direct"] = 0.0
        out["policy_intensity_objective_indicator"] = 0.0
        out["policy_intensity_objective_macro"] = 0.0
        out["policy_intensity_ai_inferred"] = 0.0
        out["policy_intensity_direct_non_ai"] = 0.0
        out["policy_intensity_sum_iso_year"] = 0.0
        out["policy_intensity_mean_iso_year"] = 0.0
        out["policy_event_count_iso_year"] = 0
        out["policy_event_type_count_iso_year"] = 0
        out["policy_event_coarse_type_count_iso_year"] = 0
        out["policy_event_new_count_iso_year"] = 0
        out["policy_event_count_infra_iso_year"] = 0
        out["policy_event_count_digital_iso_year"] = 0
        out["policy_event_count_eco_reg_iso_year"] = 0
        out["policy_intensity_infra_iso_year"] = 0.0
        out["policy_intensity_digital_iso_year"] = 0.0
        out["policy_intensity_eco_reg_iso_year"] = 0.0
        out["policy_event_count_iso_year_yoy"] = 0.0
        out["policy_intensity_sum_iso_year_yoy"] = 0.0
        out["policy_news_proxy_score"] = 0.0
        out["policy_source_ref"] = "none"
        out["policy_source_type"] = "none"
        out["policy_evidence_grade"] = "none"
        out["policy_evidence_score"] = 0.0
        out["policy_dose"] = 0.0
        out["policy_dose_external_direct"] = 0.0
        out["policy_dose_intense_external_direct"] = 0.0
        out["policy_dose_intense_external_peak"] = 0.0
        out["policy_intensity_peak_iso"] = 0.0
        out["policy_intensity_mean_iso"] = 0.0
        out["policy_event_active_years_iso"] = 0

    treated_cities = int(out.loc[out["treated_city"] == 1, "city_id"].nunique())
    policy_bucket_counts = reg["policy_treatment_bucket"].astype(str).value_counts().to_dict()
    policy_subtype_counts = reg["policy_subtype"].astype(str).value_counts().to_dict()
    meta = {
        "policy_design": "event_registry",
        "registry_file": str(DATA_RAW / "policy_events_registry.csv"),
        "registry_classified_file": str(DATA_RAW / "policy_events_registry_classified.csv"),
        "registry_event_rows": int(len(reg)),
        "registry_direct_event_rows": int(len(direct_registry)),
        "registry_external_direct_event_rows": int(len(external_direct_registry)),
        "registry_objective_indicator_event_rows": int(len(indicator_registry)),
        "registry_objective_macro_event_rows": int(len(macro_rule_registry)),
        "registry_ai_inferred_event_rows": int(len(inferred_registry)),
        "registry_evidence_a_event_rows": int(len(evidence_a_registry)),
        "registry_evidence_ab_event_rows": int(len(evidence_ab_registry)),
        "registry_direct_core_event_rows": int(len(direct_core_registry)),
        "registry_policy_types": int(reg["policy_name"].nunique()),
        "registry_policy_buckets": {str(k): int(v) for k, v in policy_bucket_counts.items()},
        "registry_policy_subtypes": {str(k): int(v) for k, v in policy_subtype_counts.items()},
        "registry_unique_sources": int(reg["source_ref"].nunique()),
        "principal_treatment_track": str(principal_treatment_track),
        "policy_assignment_scope": "country_registry",
        "causal_unit_recommended": "iso3_year",
        "city_rollout_rule": "country_common_shock_no_within_country_rollout_for_iso_level_registry",
        "city_rollout_max_lag_years": 0,
        "city_rollout_score_components": [],
        "city_rollout_multi_city_treated_country_count": 0,
        "treated_country_count": int(len(treated_iso_principal)),
        "treated_country_count_all_sources": int(len(treated_iso_all_sources)),
        "treated_country_count_direct": int(len(treated_iso_direct)),
        "treated_country_count_external_direct": int(len(treated_iso_external_direct)),
        "treated_country_count_intense_external_direct": int(len(treated_iso_external_intense)),
        "treated_country_count_intense_external_peak": int(len(treated_iso_external_intense)),
        "treated_country_count_objective_indicator": int(len(treated_iso_indicator)),
        "treated_country_count_objective_macro": int(len(treated_iso_macro_rule)),
        "treated_country_count_ai_inferred": int(len(treated_iso_inferred)),
        "treated_country_count_direct_core": int(len(treated_iso_direct_core)),
        "treated_country_count_direct_core_infra": int(len(direct_core_type_cohorts.get("infra", {}))),
        "treated_country_count_direct_core_digital": int(len(direct_core_type_cohorts.get("digital", {}))),
        "treated_country_count_direct_core_eco_reg": int(len(direct_core_type_cohorts.get("eco_reg", {}))),
        "treated_country_count_evidence_a": int(len(treated_iso_evidence_a)),
        "treated_country_count_evidence_ab": int(len(treated_iso_evidence_ab)),
        "treated_city_count": treated_cities,
        "treated_share": float(out["treated_city"].mean()),
        "treated_share_all_sources": float(out["treated_city_all_sources"].mean()),
        "treated_share_direct": float(out["treated_city_direct"].mean()),
        "treated_share_external_direct": float(out["treated_city_external_direct"].mean()),
        "treated_share_intense_external_direct": float(out["treated_city_intense_external_direct"].mean()),
        "treated_share_intense_external_peak": float(out["treated_city_intense_external_peak"].mean()),
        "treated_share_objective_indicator": float(out["treated_city_objective_indicator"].mean()),
        "treated_share_objective_macro": float(out["treated_city_objective_macro"].mean()),
        "treated_share_ai_inferred": float(out["treated_city_ai_inferred"].mean()),
        "treated_share_direct_core": float(out["treated_city_direct_core"].mean()),
        "treated_share_direct_core_infra": float(out["treated_city_direct_core_infra"].mean()),
        "treated_share_direct_core_digital": float(out["treated_city_direct_core_digital"].mean()),
        "treated_share_direct_core_eco_reg": float(out["treated_city_direct_core_eco_reg"].mean()),
        "treated_share_evidence_a": float(out["treated_city_evidence_a"].mean()),
        "treated_share_evidence_ab": float(out["treated_city_evidence_ab"].mean()),
        "direct_core_rule": {
            "policy_intensity_min": float(direct_core_threshold),
            "cohort_year_max": int(direct_core_year_cap),
            "source_priority": str(direct_core_source),
        },
        "external_intense_rule": {
            "source": "external_direct",
            "intensity_quantile": float(external_intense_quantile),
            "intensity_threshold": float(external_intense_threshold),
            "cohort_rule_legacy": "first_start_year",
            "cohort_rule_peak": str(external_intense_peak_rule),
        },
        "cohort_min_year": int(out.loc[out["treated_city"] == 1, "treatment_cohort_year"].min())
        if treated_cities > 0
        else None,
        "cohort_max_year": int(out.loc[out["treated_city"] == 1, "treatment_cohort_year"].max())
        if treated_cities > 0
        else None,
        "mean_policy_city_salience_treated": float(out.loc[out["treated_city"] == 1, "policy_city_salience_score"].mean())
        if treated_cities > 0
        else 0.0,
        "mean_policy_city_salience_control": float(out.loc[out["treated_city"] == 0, "policy_city_salience_score"].mean())
        if int((out["treated_city"] == 0).sum()) > 0
        else 0.0,
        "mean_policy_intensity_post": float(out.loc[out["did_treatment"] == 1, "policy_intensity"].mean())
        if int(out["did_treatment"].sum()) > 0
        else 0.0,
        "mean_policy_dose_post": float(out.loc[out["post_policy"] == 1, "policy_dose"].mean())
        if int(out["post_policy"].sum()) > 0
        else 0.0,
        "mean_policy_dose_external_direct_post": float(
            out.loc[out["post_policy_external_direct"] == 1, "policy_dose_external_direct"].mean()
        )
        if int(out["post_policy_external_direct"].sum()) > 0
        else 0.0,
        "mean_policy_event_count_iso_year": float(out["policy_event_count_iso_year"].mean()),
        "mean_policy_event_new_count_iso_year": float(out["policy_event_new_count_iso_year"].mean()),
        "mean_policy_news_proxy_score": float(out["policy_news_proxy_score"].mean()),
    }
    return out, meta


def _apply_capital_jump_policy_design(panel: pd.DataFrame, default_year: int = 2020) -> tuple[pd.DataFrame, dict]:
    """Fallback DID design when no auditable event registry is available."""
    out = panel.copy()
    threshold = (
        out.loc[out["year"] <= 2019, ["iso3", "capital_formation"]]
        .groupby("iso3", as_index=False)
        .mean()
        .rename(columns={"capital_formation": "pre_capital"})
    )
    threshold_post = (
        out.loc[out["year"] >= 2020, ["iso3", "capital_formation"]]
        .groupby("iso3", as_index=False)
        .mean()
        .rename(columns={"capital_formation": "post_capital"})
    )
    jumps = threshold.merge(threshold_post, on="iso3", how="inner")
    jumps["capital_jump"] = jumps["post_capital"] - jumps["pre_capital"]
    jump_cut = float(jumps["capital_jump"].quantile(0.75)) if not jumps.empty else 0.0
    treated_iso3 = set(jumps.loc[jumps["capital_jump"] >= jump_cut, "iso3"])
    cohort_map = {str(iso3).upper(): int(default_year) for iso3 in treated_iso3}
    city_salience = _build_city_policy_salience(out)
    out = out.merge(city_salience, on=["city_id", "iso3"], how="left")
    out["policy_city_salience_score"] = pd.to_numeric(out["policy_city_salience_score"], errors="coerce").fillna(0.5).astype(float)
    out["policy_city_rank_within_iso"] = pd.to_numeric(out["policy_city_rank_within_iso"], errors="coerce").fillna(1).astype(int)
    out["policy_city_count_within_iso"] = pd.to_numeric(out["policy_city_count_within_iso"], errors="coerce").fillna(1).astype(int)

    city_rollout = _assign_city_rollout_from_country_cohorts(
        out,
        cohort_map,
        city_salience=city_salience,
        max_year=int(out["year"].max()),
        max_rollout_lag=2,
    )
    treat_map = city_rollout.set_index("city_id")["treated_city"].astype(int).to_dict()
    cohort_city_map = city_rollout.set_index("city_id")["treatment_cohort_year"].astype(int).to_dict()

    out["treated_city"] = out["city_id"].astype(str).map(treat_map).fillna(0).astype(int)
    out["treatment_cohort_year"] = out["city_id"].astype(str).map(cohort_city_map).fillna(9999).astype(int)
    out["post_policy"] = ((out["year"] >= out["treatment_cohort_year"]) & (out["treated_city"] == 1)).astype(int)
    out["did_treatment"] = out["treated_city"] * out["post_policy"]
    out["treated_city_all_sources"] = out["treated_city"]
    out["treatment_cohort_year_all_sources"] = out["treatment_cohort_year"]
    out["post_policy_all_sources"] = out["post_policy"]
    out["did_treatment_all_sources"] = out["did_treatment"]
    out["policy_intensity"] = out["did_treatment"].astype(float)
    out["policy_intensity_all_sources"] = out["policy_intensity"].astype(float)
    out["policy_source_ref"] = "rule_based_capital_jump"
    out["policy_source_type"] = "rule_based"
    out["policy_evidence_grade"] = "none"
    out["policy_evidence_score"] = 0.0

    out["treated_city_external_direct"] = 0
    out["post_policy_external_direct"] = 0
    out["did_treatment_external_direct"] = 0
    out["treatment_cohort_year_external_direct"] = 9999
    out["treated_city_intense_external_direct"] = 0
    out["post_policy_intense_external_direct"] = 0
    out["did_treatment_intense_external_direct"] = 0
    out["treatment_cohort_year_intense_external_direct"] = 9999
    out["treated_city_intense_external_peak"] = 0
    out["post_policy_intense_external_peak"] = 0
    out["did_treatment_intense_external_peak"] = 0
    out["treatment_cohort_year_intense_external_peak"] = 9999
    out["treated_city_objective_indicator"] = 0
    out["post_policy_objective_indicator"] = 0
    out["did_treatment_objective_indicator"] = 0
    out["treatment_cohort_year_objective_indicator"] = 9999
    out["treated_city_objective_macro"] = 0
    out["post_policy_objective_macro"] = 0
    out["did_treatment_objective_macro"] = 0
    out["treatment_cohort_year_objective_macro"] = 9999
    out["treated_city_ai_inferred"] = 0
    out["post_policy_ai_inferred"] = 0
    out["did_treatment_ai_inferred"] = 0
    out["treatment_cohort_year_ai_inferred"] = 9999
    out["treated_city_direct"] = out["treated_city"]
    out["post_policy_direct"] = out["post_policy"]
    out["did_treatment_direct"] = out["did_treatment"]
    out["treatment_cohort_year_direct"] = out["treatment_cohort_year"]
    out["treated_city_direct_core"] = out["treated_city"]
    out["post_policy_direct_core"] = out["post_policy"]
    out["did_treatment_direct_core"] = out["did_treatment"]
    out["treatment_cohort_year_direct_core"] = out["treatment_cohort_year"]
    out["treated_city_direct_core_infra"] = 0
    out["post_policy_direct_core_infra"] = 0
    out["did_treatment_direct_core_infra"] = 0
    out["treatment_cohort_year_direct_core_infra"] = 9999
    out["treated_city_direct_core_digital"] = 0
    out["post_policy_direct_core_digital"] = 0
    out["did_treatment_direct_core_digital"] = 0
    out["treatment_cohort_year_direct_core_digital"] = 9999
    out["treated_city_direct_core_eco_reg"] = 0
    out["post_policy_direct_core_eco_reg"] = 0
    out["did_treatment_direct_core_eco_reg"] = 0
    out["treatment_cohort_year_direct_core_eco_reg"] = 9999
    out["treated_city_evidence_a"] = 0
    out["post_policy_evidence_a"] = 0
    out["did_treatment_evidence_a"] = 0
    out["treatment_cohort_year_evidence_a"] = 9999
    out["treated_city_evidence_ab"] = out["treated_city"]
    out["post_policy_evidence_ab"] = out["post_policy"]
    out["did_treatment_evidence_ab"] = out["did_treatment"]
    out["treatment_cohort_year_evidence_ab"] = out["treatment_cohort_year"]
    out["policy_intensity_external_direct"] = 0.0
    out["policy_intensity_objective_indicator"] = 0.0
    out["policy_intensity_objective_macro"] = 0.0
    out["policy_intensity_ai_inferred"] = 0.0
    out["policy_intensity_direct_non_ai"] = out["policy_intensity"].astype(float)
    out["policy_intensity_sum_iso_year"] = out["policy_intensity"].astype(float)
    out["policy_intensity_mean_iso_year"] = out["policy_intensity"].astype(float)
    out["policy_event_count_iso_year"] = out["did_treatment"].astype(int)
    out["policy_event_type_count_iso_year"] = out["did_treatment"].astype(int)
    out["policy_event_new_count_iso_year"] = 0
    out["policy_event_count_iso_year_yoy"] = 0.0
    out["policy_intensity_sum_iso_year_yoy"] = 0.0
    out["policy_news_proxy_score"] = (
        np.log1p(out["policy_event_count_iso_year"].astype(float))
        + 0.50 * out["policy_intensity_sum_iso_year"].astype(float)
    ).astype(float)
    out["policy_dose"] = out["post_policy"].astype(float) * out["policy_intensity"].astype(float)
    out["policy_dose_external_direct"] = 0.0
    out["policy_dose_intense_external_direct"] = 0.0
    out["policy_dose_intense_external_peak"] = 0.0
    iso_peak = out.groupby("iso3")["policy_intensity"].transform("max")
    iso_mean = out.groupby("iso3")["policy_intensity"].transform("mean")
    out["policy_intensity_peak_iso"] = iso_peak.astype(float)
    out["policy_intensity_mean_iso"] = iso_mean.astype(float)
    out["policy_event_active_years_iso"] = out.groupby("iso3")["policy_intensity"].transform(
        lambda s: int(np.sum(pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0))
    ).astype(int)

    meta = {
        "policy_design": "capital_formation_jump_rule",
        "principal_treatment_track": "rule_based",
        "fallback_reason": "policy_event_registry_missing_or_invalid",
        "policy_assignment_scope": "country_rule",
        "causal_unit_recommended": "iso3_year",
        "city_rollout_rule": "country_common_shock_no_within_country_rollout_for_country_rule_fallback",
        "city_rollout_max_lag_years": 0,
        "treated_country_count": len(treated_iso3),
        "treated_share": float(out["treated_city"].mean()),
        "treated_share_intense_external_direct": float(out["treated_city_intense_external_direct"].mean()),
        "treated_share_intense_external_peak": float(out["treated_city_intense_external_peak"].mean()),
        "external_intense_rule": {
            "source": "external_direct",
            "intensity_quantile": None,
            "intensity_threshold": None,
            "cohort_rule_legacy": None,
            "cohort_rule_peak": None,
        },
        "reference_treatment_year": int(default_year),
        "mean_policy_dose_post": float(out.loc[out["post_policy"] == 1, "policy_dose"].mean())
        if int(out["post_policy"].sum()) > 0
        else 0.0,
        "mean_policy_event_count_iso_year": float(out["policy_event_count_iso_year"].mean()),
        "mean_policy_event_new_count_iso_year": float(out["policy_event_new_count_iso_year"].mean()),
        "mean_policy_news_proxy_score": float(out["policy_news_proxy_score"].mean()),
    }
    return out, meta


def _engineer_features(
    panel: pd.DataFrame,
    config: ProjectConfig,
    *,
    add_idiosyncratic_noise: bool = True,
    require_policy_events: bool = False,
    auto_build_policy_events: bool = False,
    augment_policy_events_for_sensitivity: bool = False,
    enable_city_macro_disaggregation: bool = True,
    use_city_observed_primary_spec: bool = True,
    normalize_within_year: bool = False,
    prefer_pca_composite: bool = True,
) -> pd.DataFrame:
    """Create economic and urban indicators from raw merged panel."""
    out = panel.copy()
    extra_cols = list(WB_EXTRA_INDICATORS.keys())
    if "city_macro_observed_flag" not in out.columns:
        out["city_macro_observed_flag"] = 0
    out["city_macro_observed_flag"] = pd.to_numeric(out["city_macro_observed_flag"], errors="coerce").fillna(0).astype(int)
    if "macro_observed_source" not in out.columns:
        out["macro_observed_source"] = "missing"
    out["macro_observed_source"] = out["macro_observed_source"].fillna("missing").astype(str)
    if "macro_resolution_level" not in out.columns:
        out["macro_resolution_level"] = np.where(
            out["city_macro_observed_flag"] > 0,
            "city_observed",
            "country_year",
        )
    out["macro_resolution_level"] = out["macro_resolution_level"].fillna("country_year").astype(str)

    for col in MACRO_BASE_COLUMNS:
        out[col] = _fill_group_median(out, col, "city_id")

    for col in extra_cols:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out.groupby("iso3")[col].transform(lambda x: x.interpolate(limit_direction="both"))
        out[col] = out.groupby("continent")[col].transform(lambda x: x.fillna(x.median()))
        median = float(out[col].median()) if not pd.isna(out[col].median()) else 0.0
        out[col] = out[col].fillna(median)

    if "extra_wb_source" not in out.columns:
        if "external_source" in out.columns:
            out["extra_wb_source"] = out["external_source"].astype(str)
        else:
            out["extra_wb_source"] = "unavailable"
    out["extra_wb_source"] = out["extra_wb_source"].fillna("unavailable").astype(str)

    poi_cols = [
        "amenity_count",
        "shop_count",
        "office_count",
        "leisure_count",
        "transport_count",
        "poi_total",
        "poi_diversity",
    ]
    for col in poi_cols:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if col != "poi_diversity":
            out[col] = out[col].clip(lower=0.0)

    if "poi_source" not in out.columns:
        out["poi_source"] = "missing"
    else:
        out["poi_source"] = out["poi_source"].fillna("missing")

    # Optional road-network panel (city-year). Used for stratification and dynamic controls.
    for col in ["road_length_km_total", "arterial_share", "intersection_density"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].notna().any():
            out[col] = out.groupby("city_id")[col].transform(lambda s: s.interpolate(limit_direction="both"))
    if "road_source" not in out.columns:
        out["road_source"] = "missing"
    else:
        out["road_source"] = out["road_source"].fillna("missing").astype(str)

    # Optional VIIRS city-year panel.
    # Keep raw observation timing. A single observed year should not be back/forward-filled
    # into unobserved years because that turns a level signal into fabricated pseudo-dynamics.
    for col in [
        "viirs_ntl_mean",
        "viirs_ntl_p90",
        "viirs_lit_area_km2",
        "viirs_log_mean",
        "viirs_intra_year_recovery",
        "viirs_intra_year_decline",
        "viirs_recent_drop",
        "viirs_ntl_yoy",
    ]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "viirs_month_count" not in out.columns:
        out["viirs_month_count"] = 0.0
    out["viirs_month_count"] = pd.to_numeric(out["viirs_month_count"], errors="coerce").fillna(0.0)
    if "viirs_year_coverage_share" not in out.columns:
        out["viirs_year_coverage_share"] = np.clip(out["viirs_month_count"] / 12.0, 0.0, 1.0)
    out["viirs_year_coverage_share"] = pd.to_numeric(out["viirs_year_coverage_share"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if "has_viirs_observation" not in out.columns:
        out["has_viirs_observation"] = (
            out["viirs_ntl_mean"].notna() | (out["viirs_month_count"] > 0.0)
        ).astype(int)
    out["has_viirs_observation"] = pd.to_numeric(out["has_viirs_observation"], errors="coerce").fillna(0).astype(int)
    if "viirs_source" not in out.columns:
        out["viirs_source"] = "missing"
    else:
        out["viirs_source"] = out["viirs_source"].fillna("missing").astype(str)

    # Optional GHSL built-surface / built-volume city-year panel.
    for col in [
        "ghsl_built_surface_km2",
        "ghsl_built_surface_nres_km2",
        "ghsl_built_volume_m3",
        "ghsl_built_volume_nres_m3",
        "ghsl_built_density",
        "ghsl_built_surface_yoy",
        "ghsl_built_volume_yoy",
    ]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "ghsl_source" not in out.columns:
        out["ghsl_source"] = "missing"
    else:
        out["ghsl_source"] = out["ghsl_source"].fillna("missing").astype(str)
    out["has_ghsl_observation"] = pd.to_numeric(out.get("ghsl_built_surface_km2"), errors="coerce").notna().astype(int)

    # Optional NO2 city-year panel derived from monthly GEE exports.
    for col in [
        "no2_trop_mean",
        "no2_trop_p90",
        "no2_trop_yoy_mean",
        "no2_trop_anomaly_mean",
        "no2_trop_anomaly_abs_mean",
        "no2_recent_spike",
    ]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "no2_month_count" not in out.columns:
        out["no2_month_count"] = 0.0
    out["no2_month_count"] = pd.to_numeric(out["no2_month_count"], errors="coerce").fillna(0.0)
    if "no2_year_coverage_share" not in out.columns:
        out["no2_year_coverage_share"] = np.clip(out["no2_month_count"] / 12.0, 0.0, 1.0)
    out["no2_year_coverage_share"] = pd.to_numeric(out["no2_year_coverage_share"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if "has_no2_observation" not in out.columns:
        out["has_no2_observation"] = (
            out["no2_trop_mean"].notna() | (out["no2_month_count"] > 0.0)
        ).astype(int)
    out["has_no2_observation"] = pd.to_numeric(out["has_no2_observation"], errors="coerce").fillna(0).astype(int)
    if "no2_source" not in out.columns:
        out["no2_source"] = "missing"
    else:
        out["no2_source"] = out["no2_source"].fillna("missing").astype(str)

    # Optional transport/logistics connectivity panel.
    for col in [
        "flight_connectivity_total",
        "flight_degree_centrality",
        "airport_count_mapped",
        "international_route_share",
        "shipping_connectivity_total",
    ]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "network_connectivity_source" not in out.columns:
        out["network_connectivity_source"] = "missing"
    else:
        out["network_connectivity_source"] = out["network_connectivity_source"].fillna("missing").astype(str)
    out["has_network_connectivity_observation"] = (
        out[
            [
                "flight_connectivity_total",
                "flight_degree_centrality",
                "airport_count_mapped",
                "international_route_share",
                "shipping_connectivity_total",
            ]
        ].notna().any(axis=1)
    ).astype(int)

    # Optional OSM historical city-year panel.
    for col in [
        "osm_hist_road_length_m",
        "osm_hist_building_count",
        "osm_hist_poi_count",
        "osm_hist_poi_food_count",
        "osm_hist_poi_retail_count",
        "osm_hist_poi_nightlife_count",
        "osm_hist_road_yoy",
        "osm_hist_building_yoy",
        "osm_hist_poi_yoy",
        "osm_hist_poi_food_yoy",
        "osm_hist_poi_retail_yoy",
        "osm_hist_poi_nightlife_yoy",
    ]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].notna().any():
            out[col] = out.groupby("city_id")[col].transform(lambda s: s.interpolate(limit_direction="both"))
    if "osm_hist_source" not in out.columns:
        out["osm_hist_source"] = "missing"
    else:
        out["osm_hist_source"] = out["osm_hist_source"].fillna("missing").astype(str)

    out = _reconstruct_historical_poi_from_snapshot(out, config)

    # Optional city-year social sentiment panel.
    social_cols = [
        "social_sentiment_score",
        "social_sentiment_volatility",
        "social_sentiment_positive_share",
        "social_sentiment_negative_share",
        "social_sentiment_volume",
        "social_sentiment_platform_count",
        "social_sentiment_buzz",
    ]
    for col in social_cols:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for share_col in ["social_sentiment_positive_share", "social_sentiment_negative_share"]:
        out[share_col] = out[share_col].clip(0.0, 1.0)
    out["social_sentiment_score"] = out["social_sentiment_score"].clip(-1.0, 1.0)
    out["social_sentiment_volume"] = out["social_sentiment_volume"].fillna(0.0)
    out["social_sentiment_platform_count"] = out["social_sentiment_platform_count"].fillna(0.0)
    out["social_sentiment_buzz"] = out["social_sentiment_buzz"].fillna(0.0)
    out["has_social_observation"] = (
        (pd.to_numeric(out["social_sentiment_volume"], errors="coerce").fillna(0.0) > 0.0)
        | pd.to_numeric(out["social_sentiment_score"], errors="coerce").notna()
    ).astype(int)
    out["has_sentiment"] = out["has_social_observation"].astype(int)
    if "social_sentiment_source" not in out.columns:
        out["social_sentiment_source"] = "missing"
    else:
        out["social_sentiment_source"] = out["social_sentiment_source"].fillna("missing").astype(str)

    out["poi_total"] = pd.to_numeric(out.get("poi_total"), errors="coerce").combine_first(
        out[["amenity_count", "shop_count", "office_count", "leisure_count", "transport_count"]].sum(axis=1, min_count=1)
    )
    if pd.to_numeric(out.get("poi_diversity"), errors="coerce").notna().sum() == 0:
        poi_probs = out[["amenity_count", "shop_count", "office_count", "leisure_count"]].clip(lower=0.0)
        poi_probs = poi_probs.div(poi_probs.sum(axis=1).replace(0.0, np.nan), axis=0)
        out["poi_diversity"] = -(poi_probs * np.log(poi_probs)).sum(axis=1, min_count=1)
    out["has_poi_observation"] = out[["amenity_count", "shop_count", "office_count", "leisure_count", "transport_count", "poi_total"]].notna().any(axis=1).astype(int)
    out["poi_total_yoy"] = (
        out.groupby("city_id")["poi_total"].pct_change().replace([np.inf, -np.inf], np.nan)
    )
    out["osm_hist_building_count_yoy"] = (
        out.groupby("city_id")["osm_hist_building_count"].pct_change().replace([np.inf, -np.inf], np.nan)
    )
    surge_cols = [
        pd.to_numeric(out["poi_total_yoy"], errors="coerce"),
        pd.to_numeric(out["osm_hist_building_count_yoy"], errors="coerce"),
    ]
    out["is_mapping_surge"] = (
        pd.concat(surge_cols, axis=1).gt(0.30).any(axis=1)
    ).astype(int)

    # Dynamic road-growth signals: split total observed OSM-road change by road hierarchy share.
    road_yoy = pd.to_numeric(out.get("osm_hist_road_yoy"), errors="coerce").fillna(0.0)
    arterial_share = pd.to_numeric(out.get("arterial_share"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    road_level = pd.to_numeric(out.get("osm_hist_road_length_m"), errors="coerce").fillna(0.0)
    out["road_arterial_growth_proxy"] = (road_yoy * arterial_share).astype(float)
    out["road_local_growth_proxy"] = (road_yoy * (1.0 - arterial_share)).astype(float)
    out["road_growth_intensity"] = (road_yoy / np.maximum(road_level, 1.0)).astype(float)
    out["road_growth_intensity"] = out["road_growth_intensity"].clip(-1.0, 1.0)

    if bool(enable_city_macro_disaggregation):
        out, macro_disagg_meta = _apply_city_macro_disaggregation(out)
    else:
        out = out.copy()
        out["macro_city_disagg_source"] = "disabled_country_year_only"
        macro_disagg_meta = {
            "status": "disabled",
            "reason": "city_macro_disaggregation_disabled_by_default",
            "source": "none",
            "macro_resolution_policy": "country_year_controls_without_within_country_fabrication",
        }
    dump_json(DATA_PROCESSED / "macro_city_disaggregation_summary.json", macro_disagg_meta)

    out = out.sort_values(["city_id", "year"]).reset_index(drop=True)
    out["poi_total_yoy"] = (
        out.groupby("city_id")["poi_total"].pct_change().replace([np.inf, -np.inf], np.nan)
    )
    out["osm_hist_building_count_yoy"] = (
        out.groupby("city_id")["osm_hist_building_count"].pct_change().replace([np.inf, -np.inf], np.nan)
    )
    out["is_mapping_surge"] = (
        pd.concat(
            [
                pd.to_numeric(out["poi_total_yoy"], errors="coerce"),
                pd.to_numeric(out["osm_hist_building_count_yoy"], errors="coerce"),
            ],
            axis=1,
        ).gt(0.30).any(axis=1)
    ).astype(int)

    out["social_sentiment_delta_1"] = (
        out.groupby("city_id")["social_sentiment_score"].diff().fillna(0.0)
    )

    out["gdp_growth"] = out.groupby("city_id")["gdp_per_capita"].pct_change().replace([np.inf, -np.inf], np.nan)
    out["gdp_growth"] = out["gdp_growth"].fillna(0.0)

    out["log_gdp_pc"] = np.log1p(out["gdp_per_capita"])
    out["log_population"] = np.log1p(out["population"])
    out["log_viirs_ntl"] = pd.to_numeric(out.get("viirs_log_mean"), errors="coerce")
    out["log_pop"] = pd.to_numeric(out["log_population"], errors="coerce")
    out["raw_temp_mean"] = pd.to_numeric(out.get("temperature_mean"), errors="coerce")
    out["raw_precipitation_sum"] = pd.to_numeric(out.get("precipitation_sum"), errors="coerce")
    out["baseline_population_log"] = out.groupby("city_id")["log_pop"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").dropna().iloc[0] if pd.to_numeric(s, errors="coerce").notna().any() else np.nan
    )
    out["climate_comfort"] = np.exp(-np.abs(out["temperature_mean"] - 18.0) / 12.0) * np.exp(
        -out["precipitation_sum"] / 2000.0
    )

    out["amenity_ratio"] = out["amenity_count"] / np.maximum(out["poi_total"], 1.0)
    out["commerce_ratio"] = (out["shop_count"] + out["office_count"]) / np.maximum(out["poi_total"], 1.0)
    out["transport_intensity"] = out["transport_count"] / np.maximum(out["poi_total"], 1.0)
    out["viirs_physical_continuity"] = (
        np.log1p(pd.to_numeric(out.get("viirs_ntl_mean"), errors="coerce").clip(lower=0.0))
        - pd.to_numeric(out.get("viirs_recent_drop"), errors="coerce").fillna(0.0)
    )
    viirs_recent_drop = pd.to_numeric(out.get("viirs_recent_drop"), errors="coerce")
    viirs_intra_year_decline = pd.to_numeric(out.get("viirs_intra_year_decline"), errors="coerce")
    viirs_stress_mask = viirs_recent_drop.notna() | viirs_intra_year_decline.notna()
    out["viirs_physical_stress"] = pd.Series(
        np.where(
            viirs_stress_mask.to_numpy(dtype=bool),
            0.60 * viirs_recent_drop.fillna(0.0).to_numpy(dtype=float)
            + 0.40 * viirs_intra_year_decline.fillna(0.0).to_numpy(dtype=float),
            np.nan,
        ),
        index=out.index,
        dtype=float,
    )
    ghsl_surface_yoy = pd.to_numeric(out.get("ghsl_built_surface_yoy"), errors="coerce")
    ghsl_volume_yoy = pd.to_numeric(out.get("ghsl_built_volume_yoy"), errors="coerce")
    ghsl_contraction_mask = ghsl_surface_yoy.notna() | ghsl_volume_yoy.notna()
    out["ghsl_built_contraction"] = pd.Series(
        np.where(
            ghsl_contraction_mask.to_numpy(dtype=bool),
            0.50 * np.maximum(-ghsl_surface_yoy.fillna(0.0).to_numpy(dtype=float), 0.0)
            + 0.50 * np.maximum(-ghsl_volume_yoy.fillna(0.0).to_numpy(dtype=float), 0.0),
            np.nan,
        ),
        index=out.index,
        dtype=float,
    )
    out["physical_built_expansion_primary"] = pd.to_numeric(out.get("ghsl_built_surface_km2"), errors="coerce")
    out["physical_built_expansion_primary_yoy"] = pd.to_numeric(out.get("ghsl_built_surface_yoy"), errors="coerce")
    out["physical_built_expansion_source"] = "ghsl_built_surface_km2"

    # External objective macro-innovation signals.
    out["knowledge_capital_raw"] = (
        0.45 * out["patent_residents"]
        + 0.35 * out["researchers_per_million"]
        + 0.20 * out["high_tech_exports_share"]
    )
    out["digital_infra_raw"] = 0.55 * out["internet_users"] + 0.45 * out["fixed_broadband_subscriptions"]
    out["basic_infra_raw"] = 0.60 * out["electricity_access"] + 0.40 * out["urban_population_share"]
    out["labor_market_raw"] = out["employment_rate"]
    out["clean_air_fast_proxy"] = -pd.to_numeric(out.get("no2_trop_anomaly_mean"), errors="coerce")

    out, spatial_meta = add_spatial_structure_features(out, baseline_year=2015)
    dump_json(DATA_PROCESSED / "spatial_structure_features_summary.json", spatial_meta)
    out, cobb_meta = fit_cobb_douglas_vitality(out)
    dump_json(DATA_PROCESSED / "cobb_douglas_vitality_summary.json", cobb_meta)

    norm_cols = [
        "log_gdp_pc",
        "gdp_growth",
        "unemployment",
        "internet_users",
        "climate_comfort",
        "amenity_ratio",
        "commerce_ratio",
        "transport_intensity",
        "poi_diversity",
        "capital_formation",
        "inflation",
        "patent_residents",
        "researchers_per_million",
        "high_tech_exports_share",
        "employment_rate",
        "urban_population_share",
        "electricity_access",
        "fixed_broadband_subscriptions",
        "pm25_exposure",
        "knowledge_capital_raw",
        "digital_infra_raw",
        "basic_infra_raw",
        "labor_market_raw",
        "clean_air_fast_proxy",
        "gravity_access_viirs",
        "gravity_access_knowledge",
        "gravity_access_population",
        "spatial_lag_log_viirs_ntl_wdist",
        "spatial_lag_log_viirs_ntl_wecon",
        "spatial_lag_knowledge_wdist",
        "flight_connectivity_total",
        "flight_degree_centrality",
        "shipping_connectivity_total",
        "cobb_douglas_log_vitality_fit",
        "cobb_douglas_tfp_residual",
    ]
    dynamic_optional_norm = [
        "road_length_km_total",
        "arterial_share",
        "intersection_density",
        "road_arterial_growth_proxy",
        "road_local_growth_proxy",
        "road_growth_intensity",
        "viirs_ntl_mean",
        "viirs_ntl_p90",
        "viirs_lit_area_km2",
        "viirs_log_mean",
        "viirs_intra_year_recovery",
        "viirs_intra_year_decline",
        "viirs_recent_drop",
        "viirs_physical_continuity",
        "viirs_physical_stress",
        "viirs_ntl_yoy",
        "ghsl_built_surface_km2",
        "ghsl_built_surface_nres_km2",
        "ghsl_built_volume_m3",
        "ghsl_built_volume_nres_m3",
        "ghsl_built_density",
        "ghsl_built_surface_yoy",
        "ghsl_built_volume_yoy",
        "ghsl_built_contraction",
        "osm_hist_road_length_m",
        "osm_hist_building_count",
        "osm_hist_poi_count",
        "osm_hist_poi_food_count",
        "osm_hist_poi_retail_count",
        "osm_hist_poi_nightlife_count",
        "osm_hist_road_yoy",
        "osm_hist_building_yoy",
        "osm_hist_poi_yoy",
        "osm_hist_poi_food_yoy",
        "osm_hist_poi_retail_yoy",
        "osm_hist_poi_nightlife_yoy",
        "social_sentiment_score",
        "social_sentiment_volatility",
        "social_sentiment_positive_share",
        "social_sentiment_negative_share",
        "social_sentiment_volume",
        "social_sentiment_platform_count",
        "social_sentiment_buzz",
        "social_sentiment_delta_1",
        "viirs_year_coverage_share",
        "no2_trop_mean",
        "no2_trop_p90",
        "no2_trop_yoy_mean",
        "no2_trop_anomaly_mean",
        "no2_trop_anomaly_abs_mean",
        "no2_recent_spike",
        "no2_year_coverage_share",
        "poi_backcast_scale",
    ]
    for col in dynamic_optional_norm:
        if col in out.columns and pd.to_numeric(out[col], errors="coerce").notna().any():
            norm_cols.append(col)

    normalization_scope = "yearly_cross_section" if normalize_within_year else "global_panel"
    for col in norm_cols:
        if normalize_within_year:
            out[f"{col}_n"] = out.groupby("year")[col].transform(_safe_year_minmax)
        else:
            out[f"{col}_n"] = _safe_global_minmax(out[col])
        out[f"{col}_norm_global"] = _safe_global_minmax(out[col])
    out["normalization_scope"] = normalization_scope

    # Social channels use neutral 0.5 when a year has no cross-city variation,
    # avoiding artificial down-weighting from zero-range MinMax.
    social_level_cols = [
        "social_sentiment_score",
        "social_sentiment_volatility",
        "social_sentiment_positive_share",
        "social_sentiment_negative_share",
        "social_sentiment_volume",
        "social_sentiment_platform_count",
        "social_sentiment_buzz",
        "social_sentiment_delta_1",
    ]
    for base_col in social_level_cols:
        norm_col = f"{base_col}_n"
        if norm_col not in out.columns or base_col not in out.columns:
            continue
        if normalize_within_year:
            no_var_mask = out.groupby("year")[base_col].transform(
                lambda s: pd.to_numeric(s, errors="coerce").nunique(dropna=True) <= 1
            )
        else:
            no_var = pd.to_numeric(out[base_col], errors="coerce").nunique(dropna=True) <= 1
            no_var_mask = pd.Series(np.full(len(out), bool(no_var)), index=out.index, dtype=bool)
        out.loc[no_var_mask.fillna(False), norm_col] = 0.5

    out["clean_air_n"] = 1.0 - out["pm25_exposure_n"]
    out["knowledge_capital_n"] = out["knowledge_capital_raw_n"]
    out["digital_infra_n"] = out["digital_infra_raw_n"]
    out["basic_infra_n"] = out["basic_infra_raw_n"]
    out["labor_market_n"] = out["labor_market_raw_n"]

    out["knowledge_delta_1"] = out.groupby("city_id")["knowledge_capital_n"].diff().fillna(0.0)
    out["digital_delta_1"] = out.groupby("city_id")["digital_infra_n"].diff().fillna(0.0)
    out["air_quality_improve_1"] = out.groupby("city_id")["clean_air_n"].diff().fillna(0.0)

    # Road access score and strata for route-based partitioning.
    def _n(col: str) -> pd.Series:
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce")
        return pd.Series(np.nan, index=out.index, dtype=float)

    road_access_n = _weighted_row_blend(
        [
            (0.45, _n("road_length_km_total_n")),
            (0.30, _n("arterial_share_n")),
            (0.25, _n("intersection_density_n")),
        ],
        default=np.nan,
    )
    out["road_access_score"] = 100.0 * road_access_n

    def _road_tier_for_year(s: pd.Series) -> pd.Series:
        vals = pd.to_numeric(s, errors="coerce")
        valid = vals.notna()
        if valid.sum() < 12:
            return pd.Series(["unclassified"] * len(s), index=s.index, dtype=object)
        q33 = float(vals[valid].quantile(0.33))
        q67 = float(vals[valid].quantile(0.67))
        tier = np.where(vals >= q67, "core_corridor", np.where(vals <= q33, "peripheral", "secondary"))
        tier_series = pd.Series(tier, index=s.index, dtype=object)
        tier_series[~valid] = "unclassified"
        return tier_series

    out["road_tier"] = out.groupby("year")["road_access_score"].transform(_road_tier_for_year)
    tier_code = {"peripheral": 0, "secondary": 1, "core_corridor": 2, "unclassified": -1}
    out["road_tier_code"] = out["road_tier"].map(tier_code).fillna(-1).astype(int)

    # Observed-city signal blocks used in the primary specification.
    obs_activity_n = _weighted_row_blend(
        [
            (0.24, _n("commerce_ratio_n")),
            (0.18, _n("transport_intensity_n")),
            (0.14, _n("poi_diversity_n")),
            (0.10, _n("viirs_log_mean_n")),
            (0.10, _n("viirs_physical_continuity_n")),
            (0.12, _n("ghsl_built_surface_km2_n")),
            (0.12, _n("flight_connectivity_total_n")),
        ],
        default=0.5,
    )
    obs_mobility_n = _weighted_row_blend(
        [
            (0.28, _n("road_length_km_total_n")),
            (0.22, _n("arterial_share_n")),
            (0.18, _n("intersection_density_n")),
            (0.16, _n("viirs_lit_area_km2_n")),
            (0.16, _n("ghsl_built_surface_km2_n")),
            (0.12, _n("flight_degree_centrality_n")),
        ],
        default=0.5,
    )
    obs_dynamic_n = _weighted_row_blend(
        [
            (0.15, _n("road_growth_intensity_n")),
            (0.12, _n("road_arterial_growth_proxy_n")),
            (0.12, _n("road_local_growth_proxy_n")),
            (0.12, _n("osm_hist_road_yoy_n")),
            (0.10, _n("osm_hist_poi_yoy_n")),
            (0.08, _n("viirs_intra_year_recovery_n")),
            (0.05, _one_minus_preserve_nan(_n("viirs_physical_stress_n"))),
            (0.08, _n("ghsl_built_surface_yoy_n")),
            (0.08, _n("ghsl_built_volume_yoy_n")),
            (0.10, _n("no2_trop_anomaly_mean_n")),
            (0.10, _n("no2_recent_spike_n")),
        ],
        default=0.5,
    )
    obs_livability_n = _weighted_row_blend(
        [
            (0.24, _n("climate_comfort_n")),
            (0.20, _n("amenity_ratio_n")),
            (0.20, _n("intersection_density_n")),
            (0.18, _n("viirs_lit_area_km2_n")),
            (0.18, _n("ghsl_built_surface_km2_n")),
        ],
        default=0.5,
    )
    obs_innovation_n = _weighted_row_blend(
        [
            (0.20, _n("osm_hist_building_yoy_n")),
            (0.18, _n("osm_hist_poi_food_yoy_n")),
            (0.18, _n("osm_hist_poi_retail_yoy_n")),
            (0.14, _n("osm_hist_poi_nightlife_yoy_n")),
            (0.15, _n("viirs_ntl_p90_n")),
            (0.15, _n("ghsl_built_density_n")),
        ],
        default=0.5,
    )
    obs_sentiment_n = _weighted_row_blend(
        [
            (0.40, _n("social_sentiment_score_n")),
            (0.20, _n("social_sentiment_positive_share_n")),
            (0.20, _one_minus_preserve_nan(_n("social_sentiment_negative_share_n"))),
            (0.10, _n("social_sentiment_buzz_n")),
            (0.10, _one_minus_preserve_nan(_n("social_sentiment_volatility_n"))),
        ],
        default=np.nan,
    )
    obs_sentiment_n = obs_sentiment_n.where(out["has_social_observation"] > 0, np.nan)

    out["observed_activity_signal"] = 100.0 * obs_activity_n
    out["observed_mobility_signal"] = 100.0 * obs_mobility_n
    out["observed_dynamic_signal"] = 100.0 * obs_dynamic_n
    out["observed_livability_signal"] = 100.0 * obs_livability_n
    out["observed_innovation_signal"] = 100.0 * obs_innovation_n
    out["observed_sentiment_signal"] = 100.0 * obs_sentiment_n
    out["observed_physical_stress_signal"] = 100.0 * _weighted_row_blend(
        [
            (0.70, _n("viirs_physical_stress_n")),
            (0.30, _n("ghsl_built_contraction_n")),
        ],
        default=0.5,
    )

    livability_entropy_base, livability_entropy_weights = entropy_weighted_score(
        out,
        [
            "climate_comfort_n",
            "clean_air_n",
            "amenity_ratio_n",
            "basic_infra_n",
            "gravity_access_population_n",
        ],
        prefix="livability",
    )
    innovation_entropy_base, innovation_entropy_weights = entropy_weighted_score(
        out,
        [
            "knowledge_capital_n",
            "digital_infra_n",
            "gravity_access_knowledge_n",
            "spatial_lag_knowledge_wdist_n",
            "flight_degree_centrality_n",
        ],
        prefix="innovation",
    )
    out["livability_entropy_index"] = 100.0 * livability_entropy_base
    out["innovation_entropy_index"] = 100.0 * innovation_entropy_base
    dump_json(
        DATA_PROCESSED / "theory_index_weights_summary.json",
        {
            "status": "ok",
            "livability_entropy_weights": livability_entropy_weights,
            "innovation_entropy_weights": innovation_entropy_weights,
        },
    )

    if bool(use_city_observed_primary_spec):
        out["economic_vitality"] = pd.to_numeric(out["economic_vitality_cobb_douglas"], errors="coerce").combine_first(
            100.0
            * _weighted_row_blend(
                [
                    (0.50, obs_activity_n),
                    (0.30, obs_mobility_n),
                    (0.20, obs_dynamic_n),
                ],
                default=0.5,
            )
        )
        out["livability"] = pd.to_numeric(out["livability_entropy_index"], errors="coerce").combine_first(
            100.0 * obs_livability_n
        )
        out["innovation"] = pd.to_numeric(out["innovation_entropy_index"], errors="coerce").combine_first(
            100.0
            * _weighted_row_blend(
                [
                    (0.45, obs_innovation_n),
                    (0.30, _n("knowledge_capital_n")),
                    (0.25, _n("digital_infra_n")),
                ],
                default=0.5,
            )
        )
        out["index_spec_version"] = "city_observed_primary_v6_theory_anchored"
    else:
        out["economic_vitality"] = 100.0 * _weighted_row_blend(
            [
                (0.24, _n("log_gdp_pc_n")),
                (0.16, _n("gdp_growth_n")),
                (0.14, _n("commerce_ratio_n")),
                (0.13, _n("digital_infra_n")),
                (0.10, _n("transport_intensity_n")),
                (0.10, _n("labor_market_n")),
                (0.08, _n("capital_formation_n")),
            ],
            default=0.5,
        )
        out["livability"] = 100.0 * _weighted_row_blend(
            [
                (0.22, _n("climate_comfort_n")),
                (0.19, _n("amenity_ratio_n")),
                (0.16, _n("clean_air_n")),
                (0.12, _n("poi_diversity_n")),
                (0.12, _n("basic_infra_n")),
                (0.11, _one_minus_preserve_nan(_n("inflation_n"))),
            ],
            default=0.5,
        )
        out["innovation"] = 100.0 * _weighted_row_blend(
            [
                (0.22, _n("digital_infra_n")),
                (0.22, _n("knowledge_capital_n")),
                (0.15, _n("commerce_ratio_n")),
                (0.13, _n("gdp_growth_n")),
                (0.11, _n("transport_intensity_n")),
                (0.09, _n("capital_formation_n")),
            ],
            default=0.5,
        )
        out["index_spec_version"] = "legacy_macro_mixed_v3_dynamic_reweight"

    # Noise injection is DISABLED for reproducible results.
    # Previously added random noise to sub-indices in non-strict mode, which could
    # change rankings and stall classifications. Now disabled to ensure determinism.
    if add_idiosyncratic_noise:
        LOGGER.warning(
            "Idiosyncratic noise injection is enabled but has been disabled for "
            "reproducibility. Set add_idiosyncratic_noise=False or use strict mode."
        )

    # Composite weights: configurable via config, default (0.45, 0.35, 0.20).
    w_e = getattr(config, "composite_weight_economic", 0.45)
    w_l = getattr(config, "composite_weight_livability", 0.35)
    w_i = getattr(config, "composite_weight_innovation", 0.20)
    out["composite_index_weighted"] = _weighted_row_blend(
        [
            (w_e, pd.to_numeric(out["economic_vitality"], errors="coerce")),
            (w_l, pd.to_numeric(out["livability"], errors="coerce")),
            (w_i, pd.to_numeric(out["innovation"], errors="coerce")),
        ],
        default=50.0,
    )

    # --- PCA-derived composite as robustness check ---
    try:
        from sklearn.decomposition import PCA as _PCA
        from scipy.stats import spearmanr as _spearmanr

        _sub_cols = ["economic_vitality", "livability", "innovation"]
        _sub_mat = out[_sub_cols].to_numpy(dtype=float)
        _valid = np.isfinite(_sub_mat).all(axis=1)
        if int(_valid.sum()) >= 30:
            _sub_valid = _sub_mat[_valid]
            # Standardise before PCA
            _mu = _sub_valid.mean(axis=0)
            _sd = _sub_valid.std(axis=0)
            _sd[_sd < 1e-12] = 1.0
            _sub_z = (_sub_valid - _mu) / _sd
            _pca = _PCA(n_components=1, random_state=42)
            _pc1 = _pca.fit_transform(_sub_z).ravel()
            # Rescale PC1 to 0-100 range
            _pc1_min, _pc1_max = float(_pc1.min()), float(_pc1.max())
            if abs(_pc1_max - _pc1_min) > 1e-12:
                _pc1_scaled = 100.0 * (_pc1 - _pc1_min) / (_pc1_max - _pc1_min)
            else:
                _pc1_scaled = np.full_like(_pc1, 50.0)
            out["composite_index_pca"] = np.nan
            out.loc[_valid, "composite_index_pca"] = _pc1_scaled
            out["pca_explained_variance_ratio"] = float(_pca.explained_variance_ratio_[0])
            _rho, _rho_p = _spearmanr(
                out.loc[_valid, "composite_index_weighted"].to_numpy(dtype=float),
                _pc1_scaled,
            )
            LOGGER.info(
                "PCA composite: explained_var=%.3f, Spearman rho=%.3f (p=%.2e)",
                float(_pca.explained_variance_ratio_[0]),
                float(_rho),
                float(_rho_p),
            )
        else:
            LOGGER.warning("Insufficient valid rows (%d) for PCA composite; skipping.", int(_valid.sum()))
            out["composite_index_pca"] = np.nan
            out["pca_explained_variance_ratio"] = np.nan
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("PCA composite derivation failed: %s", exc)
        out["composite_index_pca"] = np.nan
        out["pca_explained_variance_ratio"] = np.nan

    out["composite_index_dashboard"] = pd.to_numeric(out["composite_index_pca"], errors="coerce").combine_first(
        pd.to_numeric(out["composite_index_weighted"], errors="coerce")
    )
    out["composite_index_dashboard_source"] = np.where(
        pd.to_numeric(out["composite_index_pca"], errors="coerce").notna(),
        "pca_pc1",
        "weighted_fallback",
    )
    out["composite_index"] = pd.to_numeric(out["composite_index_weighted"], errors="coerce").combine_first(
        pd.to_numeric(out["composite_index_pca"], errors="coerce")
    )
    out["composite_index_source"] = np.where(
        pd.to_numeric(out["composite_index_weighted"], errors="coerce").notna(),
        "weighted_manual",
        "pca_fallback",
    )
    if prefer_pca_composite:
        LOGGER.warning(
            "prefer_pca_composite is ignored for econometric safety; use composite_index_dashboard/composite_index_pca for ranking."
        )

    registry = _load_policy_event_registry(out, config)
    if registry.empty and auto_build_policy_events:
        try:
            from .policy_event_crawler import build_policy_event_registry

            crawl_summary = build_policy_event_registry(
                out["iso3"].astype(str).str.upper().unique().tolist(),
                start_year=int(config.time_range.start_year),
                end_year=int(config.time_range.end_year),
            )
            LOGGER.info("Policy event crawler summary: %s", crawl_summary)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Auto policy-event crawling failed: %s", exc)
        registry = _load_policy_event_registry(out, config)

    if augment_policy_events_for_sensitivity:
        registry, indicator_meta = _augment_policy_registry_with_objective_indicator_events(
            registry,
            out,
            config,
            target_non_ai_coverage=0.93,
        )
        registry, macro_rule_meta = _augment_policy_registry_with_objective_macro_rules(
            registry,
            out,
            config,
            target_non_ai_coverage=1.0,
        )
        registry, ai_infer_meta = _augment_policy_registry_with_ai_inference(
            registry,
            out,
            config,
            target_coverage=1.0,
        )
    else:
        indicator_meta = {
            "status": "disabled",
            "reason": "main_causal_track_uses_external_direct_only",
        }
        macro_rule_meta = {
            "status": "disabled",
            "reason": "main_causal_track_uses_external_direct_only",
        }
        ai_infer_meta = {
            "status": "disabled",
            "reason": "main_causal_track_uses_external_direct_only",
        }
    dump_json(DATA_PROCESSED / "policy_event_objective_indicator_summary.json", indicator_meta)
    dump_json(DATA_PROCESSED / "policy_event_objective_macro_summary.json", macro_rule_meta)
    dump_json(DATA_PROCESSED / "policy_event_ai_inference_summary.json", ai_infer_meta)

    registry, evidence_meta = _annotate_policy_registry_evidence(registry, out)
    dump_json(DATA_PROCESSED / "policy_event_evidence_summary.json", evidence_meta)

    registry_audit = _build_policy_registry_audit(registry, out, config)
    dump_json(DATA_PROCESSED / "policy_event_registry_audit.json", registry_audit)
    if registry.empty:
        if require_policy_events:
            msg = (
                "Policy-event registry is required but missing/invalid. "
                "Please provide data/raw/policy_events_registry.csv with auditable source_ref fields."
            )
            raise RuntimeError(msg)
        out, meta = _apply_capital_jump_policy_design(out, default_year=2020)
    else:
        out, meta = _apply_policy_design_from_registry(out, registry, config)
    meta["objective_indicator_events"] = indicator_meta
    meta["objective_macro_rules"] = macro_rule_meta
    meta["ai_inference"] = ai_infer_meta
    meta["policy_evidence"] = evidence_meta
    meta["registry_audit_status"] = registry_audit.get("status")
    dump_json(DATA_PROCESSED / "policy_design.json", meta)

    return out


def build_global_city_panel(
    config: ProjectConfig,
    max_cities: int = 50,
    use_cache: bool = True,
    strict_real_data: bool | None = None,
    require_policy_events: bool = False,
    auto_build_policy_events: bool = False,
    augment_policy_events_for_sensitivity: bool = False,
    enable_city_macro_disaggregation: bool = True,
    use_city_observed_primary_spec: bool = True,
    normalize_within_year: bool = False,
    prefer_pca_composite: bool = True,
) -> pd.DataFrame:
    """Main entry for constructing a global city-year analysis panel."""
    strict = config.strict_real_data if strict_real_data is None else strict_real_data
    cities = load_city_catalog(max_cities=max_cities)
    cities.to_csv(DATA_RAW / "city_catalog.csv", index=False)

    macro = collect_world_bank_panel(cities, config, use_cache=use_cache, strict_real_data=strict)
    macro_extra = collect_world_bank_extra_panel(cities, config, use_cache=use_cache, strict_real_data=strict)
    city_macro_obs = collect_city_macro_observed_panel(cities, config, use_cache=use_cache)
    weather = collect_city_weather(cities, config, use_cache=use_cache, strict_real_data=strict)
    poi = collect_city_poi(cities, config, use_cache=use_cache, strict_real_data=strict)
    poi_year = collect_city_poi_year_panel(cities, config, use_cache=use_cache)
    road = collect_city_road_network_panel(cities, config, use_cache=use_cache)
    viirs = collect_city_viirs_year_panel(cities, config, use_cache=use_cache)
    no2 = collect_city_no2_year_panel(cities, config, use_cache=use_cache)
    ghsl = collect_city_ghsl_year_panel(cities, config, use_cache=use_cache)
    osm_hist = collect_city_osm_history_year_panel(cities, config, use_cache=use_cache)
    connectivity = collect_city_connectivity_panel(cities, config, use_cache=use_cache)
    social_sent = collect_city_social_sentiment_panel(cities, config, use_cache=use_cache)
    panel = cities.merge(poi, on=["city_id", "city_name"], how="left")
    panel = panel.merge(weather, on=["city_id", "city_name"], how="left")
    panel = panel.merge(macro, on=["iso3", "year"], how="left")
    panel = panel.merge(
        city_macro_obs[
            [
                "city_id",
                "year",
                *MACRO_BASE_COLUMNS,
                *CITY_MACRO_EXTRA_NUMERIC_COLUMNS,
                "macro_observed_source",
                "macro_resolution_level",
                "city_macro_observed_flag",
                *CITY_MACRO_EXTRA_STRING_COLUMNS,
            ]
        ],
        on=["city_id", "year"],
        how="left",
        suffixes=("", "_city_obs"),
    )
    for col in MACRO_BASE_COLUMNS:
        obs_col = f"{col}_city_obs"
        if obs_col not in panel.columns:
            continue
        panel[col] = pd.to_numeric(panel[obs_col], errors="coerce").combine_first(pd.to_numeric(panel[col], errors="coerce"))
        panel = panel.drop(columns=[obs_col], errors="ignore")
    if "city_macro_observed_flag" not in panel.columns:
        panel["city_macro_observed_flag"] = 0
    panel["city_macro_observed_flag"] = pd.to_numeric(panel["city_macro_observed_flag"], errors="coerce").fillna(0).astype(int)
    if "macro_observed_source" not in panel.columns:
        panel["macro_observed_source"] = "missing"
    panel["macro_observed_source"] = panel["macro_observed_source"].fillna("missing").astype(str)
    for col in CITY_MACRO_EXTRA_NUMERIC_COLUMNS:
        if col not in panel.columns:
            panel[col] = np.nan
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    for col in CITY_MACRO_EXTRA_STRING_COLUMNS:
        if col not in panel.columns:
            panel[col] = pd.NA
    if "macro_resolution_level" not in panel.columns:
        panel["macro_resolution_level"] = "country_year"
    panel["macro_resolution_level"] = panel["macro_resolution_level"].fillna("country_year").astype(str)
    missing_macro_mask = panel["city_macro_observed_flag"] <= 0
    panel.loc[missing_macro_mask, "macro_observed_source"] = "missing"
    panel.loc[missing_macro_mask, "macro_resolution_level"] = "country_year"
    panel = panel.merge(road, on=["city_id", "year"], how="left")
    panel = panel.merge(viirs, on=["city_id", "year"], how="left")
    panel = panel.merge(no2, on=["city_id", "year"], how="left")
    panel = panel.merge(ghsl, on=["city_id", "year"], how="left")
    panel = panel.merge(osm_hist, on=["city_id", "year"], how="left")
    panel = panel.merge(connectivity, on=["city_id", "year"], how="left")
    panel = panel.merge(social_sent, on=["city_id", "year"], how="left")
    panel = panel.merge(
        poi_year,
        on=["city_id", "year"],
        how="left",
        suffixes=("", "_yearly"),
    )
    for col in ["amenity_count", "shop_count", "office_count", "leisure_count", "transport_count", "poi_total", "poi_diversity"]:
        yearly_col = f"{col}_yearly"
        if yearly_col not in panel.columns:
            continue
        panel[col] = pd.to_numeric(panel[yearly_col], errors="coerce").combine_first(pd.to_numeric(panel[col], errors="coerce"))
        panel = panel.drop(columns=[yearly_col], errors="ignore")
    if "poi_source_yearly" in panel.columns:
        panel["poi_source"] = panel["poi_source_yearly"].fillna(panel.get("poi_source")).fillna("missing").astype(str)
        panel = panel.drop(columns=["poi_source_yearly"], errors="ignore")
    if "poi_temporal_source_yearly" in panel.columns:
        panel["poi_temporal_source"] = panel["poi_temporal_source_yearly"].fillna("missing").astype(str)
        panel = panel.drop(columns=["poi_temporal_source_yearly"], errors="ignore")
    panel = panel.merge(
        macro_extra[
            [
                "iso3",
                "year",
                *WB_EXTRA_INDICATORS.keys(),
                "extra_wb_source",
            ]
        ],
        on=["iso3", "year"],
        how="left",
    )

    for col in ["amenity_count", "shop_count", "office_count", "leisure_count", "transport_count", "poi_total", "poi_diversity"]:
        if col not in panel.columns:
            panel[col] = np.nan

    modeled = _engineer_features(
        panel,
        config,
        add_idiosyncratic_noise=not strict,
        require_policy_events=bool(require_policy_events),
        auto_build_policy_events=bool(auto_build_policy_events),
        augment_policy_events_for_sensitivity=bool(augment_policy_events_for_sensitivity),
        enable_city_macro_disaggregation=bool(enable_city_macro_disaggregation),
        use_city_observed_primary_spec=bool(use_city_observed_primary_spec),
        normalize_within_year=bool(normalize_within_year),
        prefer_pca_composite=bool(prefer_pca_composite),
    )
    modeled = modeled.sort_values(["year", "composite_index"], ascending=[True, False]).reset_index(drop=True)

    modeled.to_csv(DATA_PROCESSED / "global_city_panel.csv", index=False)
    LOGGER.info("Global panel built: %s rows, %s columns", modeled.shape[0], modeled.shape[1])

    source_meta = {
        "macro_source_counts": modeled["macro_source"].value_counts(dropna=False).to_dict(),
        "macro_resolution_level_counts": modeled["macro_resolution_level"].value_counts(dropna=False).to_dict()
        if "macro_resolution_level" in modeled.columns
        else {},
        "city_macro_observed_flag_ratio": float(pd.to_numeric(modeled.get("city_macro_observed_flag"), errors="coerce").fillna(0).mean())
        if "city_macro_observed_flag" in modeled.columns
        else 0.0,
        "macro_observed_source_counts": modeled["macro_observed_source"].value_counts(dropna=False).to_dict()
        if "macro_observed_source" in modeled.columns
        else {},
        "macro_city_disagg_source_counts": modeled["macro_city_disagg_source"].value_counts(dropna=False).to_dict()
        if "macro_city_disagg_source" in modeled.columns
        else {},
        "extra_wb_source_counts": modeled["extra_wb_source"].value_counts(dropna=False).to_dict(),
        "weather_source_counts": modeled["weather_source"].value_counts(dropna=False).to_dict(),
        "poi_source_counts": modeled["poi_source"].value_counts(dropna=False).to_dict(),
        "poi_temporal_source_counts": modeled["poi_temporal_source"].value_counts(dropna=False).to_dict()
        if "poi_temporal_source" in modeled.columns
        else {},
        "road_source_counts": modeled["road_source"].value_counts(dropna=False).to_dict()
        if "road_source" in modeled.columns
        else {},
        "viirs_source_counts": modeled["viirs_source"].value_counts(dropna=False).to_dict()
        if "viirs_source" in modeled.columns
        else {},
        "viirs_observation_ratio": float(pd.to_numeric(modeled.get("has_viirs_observation"), errors="coerce").fillna(0).mean())
        if "has_viirs_observation" in modeled.columns
        else 0.0,
        "osm_hist_source_counts": modeled["osm_hist_source"].value_counts(dropna=False).to_dict()
        if "osm_hist_source" in modeled.columns
        else {},
        "social_sentiment_source_counts": modeled["social_sentiment_source"].value_counts(dropna=False).to_dict()
        if "social_sentiment_source" in modeled.columns
        else {},
        "social_observation_ratio": float(pd.to_numeric(modeled.get("has_social_observation"), errors="coerce").fillna(0).mean())
        if "has_social_observation" in modeled.columns
        else 0.0,
        "viirs_non_missing_ratio": float(pd.to_numeric(modeled.get("viirs_ntl_mean"), errors="coerce").notna().mean())
        if "viirs_ntl_mean" in modeled.columns
        else 0.0,
        "social_sentiment_non_missing_ratio": float(
            pd.to_numeric(modeled.get("social_sentiment_score"), errors="coerce").notna().mean()
        )
        if "social_sentiment_score" in modeled.columns
        else 0.0,
        "city_count": int(modeled["city_id"].nunique()),
        "year_span": [int(modeled["year"].min()), int(modeled["year"].max())],
        "strict_real_data": bool(strict),
        "index_spec_version_counts": modeled["index_spec_version"].value_counts(dropna=False).to_dict()
        if "index_spec_version" in modeled.columns
        else {},
        "composite_index_source_counts": modeled["composite_index_source"].value_counts(dropna=False).to_dict()
        if "composite_index_source" in modeled.columns
        else {},
        "normalization_scope_counts": modeled["normalization_scope"].value_counts(dropna=False).to_dict()
        if "normalization_scope" in modeled.columns
        else {},
        "non_real_rows": 0,
        "objective_row_ratio": None,
        "direct_verified_row_ratio": None,
    }

    city_macro_meta = {
        "status": "ok",
        "path": str(DATA_RAW / "city_macro_observed.csv"),
        "city_macro_observed_row_ratio": float(pd.to_numeric(modeled.get("city_macro_observed_flag"), errors="coerce").fillna(0).mean())
        if "city_macro_observed_flag" in modeled.columns
        else 0.0,
        "macro_resolution_level_counts": source_meta.get("macro_resolution_level_counts", {}),
        "macro_observed_source_counts": source_meta.get("macro_observed_source_counts", {}),
        "note": "Rows without direct city-level macro observations use physical-share macro downscaling by default, with country-year controls retained as the anchor.",
    }
    dump_json(DATA_PROCESSED / "city_macro_observed_summary.json", city_macro_meta)

    if strict:
        verified_weather_sources = {"open-meteo", "nasa-power"}
        # Note: imputed_from_weather_pool uses continent-year medians from real data.
        # It is classified as 'objective_derived' rather than 'objective' proper.
        objective_weather_sources = {"open-meteo", "nasa-power", "imputed_from_weather_pool"}
        objective_poi_sources = {"osm", "imputed_from_osm_pool"}
        verified_poi_sources = {"osm"}
        objective_mask = (
            (modeled["macro_source"] == "world_bank")
            & (modeled["extra_wb_source"] == "world_bank")
            & (modeled["weather_source"].isin(objective_weather_sources))
            & (modeled["poi_source"].isin(objective_poi_sources))
        )
        verified_mask = (
            (modeled["macro_source"] == "world_bank")
            & (modeled["extra_wb_source"] == "world_bank")
            & (modeled["weather_source"].isin(verified_weather_sources))
            & (modeled["poi_source"].isin(verified_poi_sources))
        )
        source_meta["non_real_rows"] = int((~objective_mask).sum())
        source_meta["objective_row_ratio"] = float(objective_mask.mean()) if len(modeled) else 0.0
        source_meta["direct_verified_row_ratio"] = float(verified_mask.mean()) if len(modeled) else 0.0

    dump_json(DATA_PROCESSED / "data_quality_summary.json", source_meta)

    if {"year", "road_tier"}.issubset(modeled.columns):
        road_tier_summary = (
            modeled.groupby(["year", "road_tier"], as_index=False)
            .agg(
                city_count=("city_id", "nunique"),
                mean_composite_index=("composite_index", "mean"),
                mean_gdp_growth=("gdp_growth", "mean"),
            )
            .sort_values(["year", "road_tier"])
        )
        road_tier_summary.to_csv(DATA_PROCESSED / "road_tier_year_summary.csv", index=False)

    return modeled
