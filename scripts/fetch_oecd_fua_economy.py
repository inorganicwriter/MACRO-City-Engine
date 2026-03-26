from __future__ import annotations

import io
import json
import re
import sys
import unicodedata
import zipfile
from io import StringIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

CITY_CATALOG_PATH = DATA_RAW / "city_catalog.csv"
OECD_CITIES_PATH = DATA_RAW / "boundaries" / "oecd_downloads" / "cities (4).zip"
OECD_FUAS_PATH = DATA_RAW / "boundaries" / "oecd_downloads" / "fuas (1).zip"
OUTPUT_PATH = DATA_RAW / "city_macro_observed.csv"
MAPPING_PATH = DATA_RAW / "city_macro_observed_mapping.csv"
STATCAN_MAPPING_PATH = DATA_RAW / "city_macro_observed_statcan_mapping.csv"
IBGE_MAPPING_PATH = DATA_RAW / "city_macro_observed_ibge_mapping.csv"
CHINA_OFFICIAL_GDP_PATH = DATA_RAW / "city_macro_observed_china_official.csv"
CHINA_MAPPING_PATH = DATA_RAW / "city_macro_observed_china_mapping.csv"
MANUAL_SUPPLEMENT_PATH = DATA_RAW / "city_macro_observed_manual_supplement.csv"
MANUAL_SUPPLEMENT_MAPPING_PATH = DATA_RAW / "city_macro_observed_manual_supplement_mapping.csv"
MANUAL_PER_CAPITA_SUPPLEMENT_PATH = DATA_RAW / "city_macro_observed_manual_per_capita_supplement.csv"
MANUAL_PER_CAPITA_MAPPING_PATH = DATA_RAW / "city_macro_observed_manual_per_capita_mapping.csv"
SUMMARY_PATH = DATA_PROCESSED / "city_macro_observed_collection_summary.json"
LEGACY_SUMMARY_PATH = DATA_PROCESSED / "oecd_fua_economy_collection_summary.json"

OECD_FUA_ECONOMY_URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "DSD_FUA_ECO@DF_ECONOMY/....?format=csvfile"
)
STATCAN_CMA_GDP_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/36100468-eng.zip"
IBGE_MUNICIPAL_GDP_URL = "https://apisidra.ibge.gov.br/values/t/5938/n6/{codes}/v/37,496/p/{periods}?formato=json"

EXPECTED_OUTPUT_COLUMNS = [
    "city_id",
    "city_name",
    "iso3",
    "year",
    "gdp_per_capita",
    "gdp_total_ppp_observed",
    "gdp_total_local_observed",
    "gdp_share_national_observed",
    "macro_observed_source",
    "macro_resolution_level",
    "macro_geo_code_observed",
    "macro_geo_name_observed",
    "macro_currency_code",
    "oecd_fua_code",
    "oecd_city_code",
    "oecd_fua_name",
    "oecd_city_name",
]

SOURCE_PRIORITY = {
    "china_city_official_gdp": 0,
    "hong_kong_censtatd_gdp": 1,
    "singapore_data_gov_gdp": 1,
    "buenos_aires_estadisticaciudad_gdp": 1,
    "mexico_city_inegi_gdp": 1,
    "dubai_dsc_gdp": 1,
    "abu_dhabi_scad_gdp": 1,
    "kenya_knbs_gcp": 1,
    "kazakhstan_stat_grdp_per_capita": 1,
    "delhi_des_state_domestic_product": 1,
    "jakarta_bps_grdp": 1,
    "thailand_nesdc_gpp": 1,
    "malaysia_dosm_state_gdp": 1,
    "hanoi_yearbook_grdp": 1,
    "hochiminhcity_gso_grdp": 1,
    "oecd_fua_economy": 2,
    "statcan_cma_gdp": 3,
    "ibge_municipal_gdp": 4,
}

DROP_TOKENS = {
    "city",
    "greater",
    "metropolitan",
    "metro",
    "urban",
    "region",
    "province",
    "prefecture",
    "municipality",
}

STATCAN_CITY_ALIASES = {
    "ottawa_ca": ["ottawa gatineau", "ottawa c gatineau"],
}

IBGE_BRAZIL_CITY_CODES = {
    "sao_paulo_br": {"municipality_code": "3550308", "municipality_name": "Sao Paulo"},
    "rio_de_janeiro_br": {"municipality_code": "3304557", "municipality_name": "Rio de Janeiro"},
    "brasilia_br": {"municipality_code": "5300108", "municipality_name": "Brasilia"},
    "belo_horizonte_br": {"municipality_code": "3106200", "municipality_name": "Belo Horizonte"},
    "porto_alegre_br": {"municipality_code": "4314902", "municipality_name": "Porto Alegre"},
    "recife_br": {"municipality_code": "2611606", "municipality_name": "Recife"},
    "curitiba_br": {"municipality_code": "4106902", "municipality_name": "Curitiba"},
    "salvador_br": {"municipality_code": "2927408", "municipality_name": "Salvador"},
    "fortaleza_br": {"municipality_code": "2304400", "municipality_name": "Fortaleza"},
    "manaus_br": {"municipality_code": "1302603", "municipality_name": "Manaus"},
    "belem_br": {"municipality_code": "1501402", "municipality_name": "Belem"},
}


def _normalize_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token and token not in DROP_TOKENS]
    return " ".join(tokens)


def _build_unique_match_table(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name_col: str,
    right_name_cols: list[str],
    match_kind: str,
) -> pd.DataFrame:
    candidates = left.copy()
    right_cols = ["iso3", *right_name_cols]
    for meta_col in ["citycode", "cityname_e", "fuacode", "fuaname_en"]:
        if meta_col in right.columns and meta_col not in right_cols:
            right_cols.append(meta_col)
    merged = candidates.merge(right[right_cols], on="iso3", how="left")
    match_mask = pd.Series(False, index=merged.index)
    for col in right_name_cols:
        match_mask = match_mask | (merged[left_name_col] == merged[col])
    matched = merged.loc[match_mask].copy()
    if matched.empty:
        return matched
    matched["match_kind"] = match_kind
    matched["match_rank"] = 0 if match_kind == "oecd_city_exact" else 1
    matched = matched.sort_values(["city_id", "match_rank"])
    matched = matched.groupby("city_id", as_index=False).first()
    return matched


def _load_oecd_boundaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not OECD_CITIES_PATH.exists():
        raise FileNotFoundError(f"Missing OECD cities boundary zip: {OECD_CITIES_PATH}")
    if not OECD_FUAS_PATH.exists():
        raise FileNotFoundError(f"Missing OECD FUA boundary zip: {OECD_FUAS_PATH}")

    oecd_cities = gpd.read_file(OECD_CITIES_PATH)[["iso3", "citycode", "cityname", "cityname_e", "fuacode"]].copy()
    oecd_fuas = gpd.read_file(OECD_FUAS_PATH)[["iso3", "fuacode", "fuaname", "fuaname_en", "geometry"]].copy()
    for col in ["cityname", "cityname_e"]:
        oecd_cities[f"{col}_n"] = oecd_cities[col].map(_normalize_name)
    for col in ["fuaname", "fuaname_en"]:
        oecd_fuas[f"{col}_n"] = oecd_fuas[col].map(_normalize_name)
    return oecd_cities, oecd_fuas


def _match_catalog_to_oecd(catalog: pd.DataFrame, oecd_cities: pd.DataFrame, oecd_fuas: pd.DataFrame) -> pd.DataFrame:
    left = catalog.copy()
    left["city_name_n"] = left["city_name"].map(_normalize_name)

    city_match = _build_unique_match_table(
        left,
        oecd_cities,
        left_name_col="city_name_n",
        right_name_cols=["cityname_n", "cityname_e_n"],
        match_kind="oecd_city_exact",
    )
    matched_ids = set(city_match["city_id"].tolist())
    fua_match = _build_unique_match_table(
        left.loc[~left["city_id"].isin(matched_ids)].copy(),
        oecd_fuas,
        left_name_col="city_name_n",
        right_name_cols=["fuaname_n", "fuaname_en_n"],
        match_kind="oecd_fua_exact",
    )

    matched = pd.concat([city_match, fua_match], ignore_index=True, sort=False)
    matched = matched.merge(
        oecd_fuas[["iso3", "fuacode", "fuaname_en"]].rename(columns={"fuaname_en": "oecd_fua_name"}),
        on=["iso3", "fuacode"],
        how="left",
    )
    matched = matched.rename(
        columns={
            "fuacode": "oecd_fua_code",
            "citycode": "oecd_city_code",
            "cityname_e": "oecd_city_name",
        }
    )
    if "oecd_city_name" not in matched.columns:
        matched["oecd_city_name"] = pd.NA
    if "cityname" in matched.columns:
        matched["oecd_city_name"] = matched["oecd_city_name"].where(pd.notna(matched["oecd_city_name"]), matched["cityname"])
    keep_cols = [
        "city_id",
        "city_name",
        "country",
        "iso3",
        "oecd_fua_code",
        "oecd_city_code",
        "oecd_fua_name",
        "oecd_city_name",
        "match_kind",
    ]
    return matched[keep_cols].drop_duplicates().sort_values(["iso3", "city_name"])


def _match_catalog_to_oecd_spatial_fallback(
    catalog: pd.DataFrame,
    existing_mapping: pd.DataFrame,
    oecd_fuas: pd.DataFrame,
) -> pd.DataFrame:
    required_cols = {"city_id", "city_name", "country", "iso3", "latitude", "longitude"}
    if not required_cols.issubset(catalog.columns):
        return pd.DataFrame(columns=existing_mapping.columns)
    if "geometry" not in oecd_fuas.columns:
        return pd.DataFrame(columns=existing_mapping.columns)

    unmatched = catalog.loc[~catalog["city_id"].isin(existing_mapping["city_id"])].copy()
    unmatched["latitude"] = pd.to_numeric(unmatched["latitude"], errors="coerce")
    unmatched["longitude"] = pd.to_numeric(unmatched["longitude"], errors="coerce")
    unmatched = unmatched.dropna(subset=["latitude", "longitude"]).copy()
    if unmatched.empty:
        return pd.DataFrame(columns=existing_mapping.columns)

    used_fuas = set(existing_mapping["oecd_fua_code"].dropna().astype(str).tolist())
    candidate_fuas = oecd_fuas.loc[~oecd_fuas["fuacode"].astype(str).isin(used_fuas)].copy()
    if candidate_fuas.empty:
        return pd.DataFrame(columns=existing_mapping.columns)

    points = gpd.GeoDataFrame(
        unmatched[["city_id", "city_name", "country", "iso3", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(unmatched["longitude"], unmatched["latitude"]),
        crs="EPSG:4326",
    )
    hits = gpd.sjoin(points, candidate_fuas[["iso3", "fuacode", "fuaname_en", "geometry"]], how="inner", predicate="within")
    if hits.empty:
        return pd.DataFrame(columns=existing_mapping.columns)

    hits = hits.rename(columns={"iso3_left": "iso3", "iso3_right": "iso3_fua", "fuaname_en": "oecd_fua_name"})
    hits = hits[hits["iso3"] == hits["iso3_fua"]].copy()
    if hits.empty:
        return pd.DataFrame(columns=existing_mapping.columns)

    # Keep only one city per fallback FUA to avoid duplicating the same metro GDP across multiple cities.
    hits = hits.sort_values(["fuacode", "city_id"]).drop_duplicates(subset=["fuacode"], keep="first")
    hits = hits.rename(columns={"fuacode": "oecd_fua_code"})
    hits["oecd_city_code"] = pd.NA
    hits["oecd_city_name"] = pd.NA
    hits["match_kind"] = "oecd_fua_spatial_within_unique"
    keep_cols = [
        "city_id",
        "city_name",
        "country",
        "iso3",
        "oecd_fua_code",
        "oecd_city_code",
        "oecd_fua_name",
        "oecd_city_name",
        "match_kind",
    ]
    return hits[keep_cols].drop_duplicates().sort_values(["iso3", "city_name"])


def _download_oecd_fua_economy() -> pd.DataFrame:
    response = requests.get(OECD_FUA_ECONOMY_URL, timeout=120)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def _reshape_oecd_gdp(raw: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    data = raw.copy()
    data["TIME_PERIOD"] = pd.to_numeric(data["TIME_PERIOD"], errors="coerce")
    data["OBS_VALUE"] = pd.to_numeric(data["OBS_VALUE"], errors="coerce")
    data = data[data["TIME_PERIOD"].between(start_year, end_year, inclusive="both")].copy()
    data = data[data["MEASURE"] == "GDP"].copy()
    data = data[data["UNIT_MEASURE"].isin(["USD_PPP", "USD_PPP_PS", "PT_NAT"])].copy()
    if data.empty:
        return pd.DataFrame(columns=["oecd_fua_code", "year", "gdp_total_ppp_observed", "gdp_per_capita", "gdp_share_national_observed"])
    pivot = (
        data.pivot_table(
            index=["REF_AREA", "TIME_PERIOD"],
            columns="UNIT_MEASURE",
            values="OBS_VALUE",
            aggfunc="mean",
        )
        .reset_index()
        .rename(
            columns={
                "REF_AREA": "oecd_fua_code",
                "TIME_PERIOD": "year",
                "USD_PPP": "gdp_total_ppp_observed",
                "USD_PPP_PS": "gdp_per_capita",
                "PT_NAT": "gdp_share_national_observed",
            }
        )
    )
    pivot["year"] = pivot["year"].astype(int)
    return pivot


def _build_oecd_observed(mapping: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    oecd_raw = _download_oecd_fua_economy()
    gdp_panel = _reshape_oecd_gdp(oecd_raw, start_year=start_year, end_year=end_year)
    observed = mapping.merge(gdp_panel, on="oecd_fua_code", how="inner")
    observed["gdp_total_local_observed"] = np.nan
    observed["macro_observed_source"] = "oecd_fua_economy"
    observed["macro_resolution_level"] = "fua_year"
    observed["macro_geo_code_observed"] = observed["oecd_fua_code"]
    observed["macro_geo_name_observed"] = observed["oecd_fua_name"]
    observed["macro_currency_code"] = "USD_PPP"
    return observed[EXPECTED_OUTPUT_COLUMNS].sort_values(["city_id", "year"])


def _load_statcan_table() -> pd.DataFrame:
    response = requests.get(STATCAN_CMA_GDP_URL, timeout=180)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    with archive.open("36100468.csv") as handle:
        table = pd.read_csv(handle)
    return table


def _match_catalog_to_statcan_cma(catalog: pd.DataFrame, oecd_mapping: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    canada = catalog.loc[catalog["iso3"] == "CAN", ["city_id", "city_name", "country", "iso3"]].copy()
    if canada.empty:
        return pd.DataFrame(columns=["city_id", "city_name", "iso3", "statcan_dguid", "statcan_geo_name"])

    statcan = raw.copy()
    statcan["DGUID"] = statcan["DGUID"].astype(str)
    statcan = statcan[statcan["DGUID"].str.contains(r"S050[35]", na=False, regex=True)].copy()
    statcan["geo_core_n"] = statcan["GEO"].astype(str).str.split(",", n=1).str[0].map(_normalize_name)
    statcan = statcan[["DGUID", "GEO", "geo_core_n"]].drop_duplicates().copy()

    rows: list[dict] = []
    for city in canada.to_dict(orient="records"):
        aliases = {_normalize_name(city["city_name"])}
        aliases.update(STATCAN_CITY_ALIASES.get(str(city["city_id"]), []))
        aliases = {alias for alias in aliases if alias}
        hit = statcan[statcan["geo_core_n"].isin(aliases)].copy()
        if hit.empty:
            continue
        hit = hit.sort_values(["geo_core_n", "GEO"]).drop_duplicates(subset=["DGUID"])
        rows.append(
            {
                "city_id": city["city_id"],
                "city_name": city["city_name"],
                "country": city["country"],
                "iso3": city["iso3"],
                "statcan_dguid": ";".join(sorted(hit["DGUID"].astype(str).unique().tolist())),
                "statcan_geo_name": " | ".join(sorted(hit["GEO"].astype(str).unique().tolist())),
                "match_kind": "statcan_cma_exact",
            }
        )

    mapping = pd.DataFrame(rows)
    if mapping.empty:
        return mapping
    keep_cols = ["city_id", "oecd_fua_code", "oecd_city_code", "oecd_fua_name", "oecd_city_name"]
    mapping = mapping.merge(oecd_mapping[keep_cols], on="city_id", how="left")
    return mapping.sort_values(["city_id"])


def _build_statcan_observed(
    catalog: pd.DataFrame,
    oecd_mapping: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _load_statcan_table()
    mapping = _match_catalog_to_statcan_cma(catalog, oecd_mapping, raw)
    if mapping.empty:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), mapping

    data = raw.copy()
    data["REF_DATE"] = pd.to_numeric(data["REF_DATE"], errors="coerce")
    data["VALUE"] = pd.to_numeric(data["VALUE"], errors="coerce")
    data["DGUID"] = data["DGUID"].astype(str)
    data = data[data["REF_DATE"].between(start_year, end_year, inclusive="both")].copy()

    national = (
        data.loc[data["GEO"].astype(str).str.strip().eq("Canada"), ["REF_DATE", "VALUE"]]
        .rename(columns={"REF_DATE": "year", "VALUE": "national_gdp_total_local"})
        .dropna(subset=["year"])
    )
    national["year"] = national["year"].astype(int)

    metros = data[data["DGUID"].str.contains(r"S050[35]", na=False, regex=True)].copy()
    metros = metros.rename(columns={"REF_DATE": "year", "VALUE": "gdp_total_local_observed"})
    metros["year"] = metros["year"].astype(int)

    mapping_expanded = mapping.copy()
    mapping_expanded["statcan_dguid"] = mapping_expanded["statcan_dguid"].astype(str).str.split(";")
    mapping_expanded = mapping_expanded.explode("statcan_dguid")
    mapping_expanded["statcan_dguid"] = mapping_expanded["statcan_dguid"].astype(str).str.strip()

    observed = mapping_expanded.merge(
        metros[["DGUID", "GEO", "year", "gdp_total_local_observed"]],
        left_on="statcan_dguid",
        right_on="DGUID",
        how="inner",
    )
    observed = (
        observed.groupby(
            [
                "city_id",
                "city_name",
                "iso3",
                "year",
                "oecd_fua_code",
                "oecd_city_code",
                "oecd_fua_name",
                "oecd_city_name",
            ],
            as_index=False,
        )
        .agg(
            gdp_total_local_observed=("gdp_total_local_observed", "sum"),
            macro_geo_code_observed=("DGUID", lambda s: ";".join(sorted({str(v) for v in s.dropna()}))),
            macro_geo_name_observed=("GEO", lambda s: " | ".join(sorted({str(v) for v in s.dropna()}))),
        )
    )
    observed = observed.merge(national, on="year", how="left")
    observed["gdp_total_local_observed"] = pd.to_numeric(observed["gdp_total_local_observed"], errors="coerce") * 1_000_000.0
    observed["gdp_share_national_observed"] = (
        100.0 * pd.to_numeric(observed["gdp_total_local_observed"], errors="coerce")
        / (pd.to_numeric(observed["national_gdp_total_local"], errors="coerce") * 1_000_000.0)
    )
    observed["gdp_per_capita"] = np.nan
    observed["gdp_total_ppp_observed"] = np.nan
    observed["macro_observed_source"] = "statcan_cma_gdp"
    observed["macro_resolution_level"] = "cma_year"
    observed["macro_currency_code"] = "CAD"
    observed = observed[EXPECTED_OUTPUT_COLUMNS].sort_values(["city_id", "year"])
    return observed, mapping


def _build_ibge_brazil_observed(catalog: pd.DataFrame, start_year: int, end_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    brazil = catalog.loc[catalog["iso3"] == "BRA", ["city_id", "city_name", "country", "iso3"]].copy()
    if brazil.empty:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), pd.DataFrame()

    rows: list[dict] = []
    for city in brazil.to_dict(orient="records"):
        meta = IBGE_BRAZIL_CITY_CODES.get(str(city["city_id"]))
        if meta is None:
            continue
        rows.append(
            {
                "city_id": city["city_id"],
                "city_name": city["city_name"],
                "country": city["country"],
                "iso3": city["iso3"],
                "macro_geo_code_observed": meta["municipality_code"],
                "macro_geo_name_observed": meta["municipality_name"],
                "match_kind": "ibge_municipality_code_manual",
            }
        )
    mapping = pd.DataFrame(rows)
    if mapping.empty:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), mapping

    periods = [str(year) for year in range(start_year, min(end_year, 2021) + 1)]
    if not periods:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), mapping
    codes = ",".join(mapping["macro_geo_code_observed"].astype(str).tolist())
    url = IBGE_MUNICIPAL_GDP_URL.format(codes=codes, periods=",".join(periods))
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    payload = response.json()
    raw = pd.DataFrame(payload[1:] if payload else [])
    if raw.empty:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), mapping

    raw["D1C"] = raw["D1C"].astype(str)
    raw["D3C"] = pd.to_numeric(raw["D3C"], errors="coerce").astype("Int64")
    raw["V"] = pd.to_numeric(raw["V"], errors="coerce")
    raw = raw[raw["D3C"].between(start_year, end_year, inclusive="both")].copy()
    if raw.empty:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), mapping

    gdp_rows = raw[raw["D2C"].astype(str) == "37"][["D1C", "D3C", "V"]].rename(
        columns={"D1C": "macro_geo_code_observed", "D3C": "year", "V": "gdp_total_local_observed"}
    )
    share_rows = raw[raw["D2C"].astype(str) == "496"][["D1C", "D3C", "V"]].rename(
        columns={"D1C": "macro_geo_code_observed", "D3C": "year", "V": "gdp_share_national_observed"}
    )
    geo_names = (
        raw[["D1C", "D1N"]]
        .drop_duplicates()
        .rename(columns={"D1C": "macro_geo_code_observed", "D1N": "macro_geo_name_observed_raw"})
    )
    observed = mapping.merge(gdp_rows, on="macro_geo_code_observed", how="inner")
    observed = observed.merge(share_rows, on=["macro_geo_code_observed", "year"], how="left")
    observed = observed.merge(geo_names, on="macro_geo_code_observed", how="left")
    observed["year"] = observed["year"].astype(int)
    observed["gdp_total_local_observed"] = pd.to_numeric(observed["gdp_total_local_observed"], errors="coerce") * 1_000.0
    observed["gdp_per_capita"] = np.nan
    observed["gdp_total_ppp_observed"] = np.nan
    observed["macro_observed_source"] = "ibge_municipal_gdp"
    observed["macro_resolution_level"] = "municipality_year"
    observed["macro_currency_code"] = "BRL"
    observed["macro_geo_name_observed"] = observed["macro_geo_name_observed_raw"].where(
        observed["macro_geo_name_observed_raw"].notna(),
        observed["macro_geo_name_observed"],
    )
    for col in ["oecd_fua_code", "oecd_city_code", "oecd_fua_name", "oecd_city_name"]:
        observed[col] = pd.NA
    observed = observed[EXPECTED_OUTPUT_COLUMNS].sort_values(["city_id", "year"])
    return observed, mapping


def _build_china_official_observed(
    catalog: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CHINA_OFFICIAL_GDP_PATH.exists():
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), pd.DataFrame()

    raw = pd.read_csv(CHINA_OFFICIAL_GDP_PATH)
    if raw.empty:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), pd.DataFrame()

    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    raw["gdp_total_local_billion_cny"] = pd.to_numeric(raw["gdp_total_local_billion_cny"], errors="coerce")
    raw = raw[raw["year"].between(start_year, end_year, inclusive="both")].copy()
    if raw.empty:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), pd.DataFrame()

    china_catalog = catalog.loc[catalog["iso3"] == "CHN", ["city_id", "city_name", "country", "iso3"]].copy()
    observed = raw.merge(china_catalog, on="city_id", how="inner")
    if observed.empty:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), pd.DataFrame()

    observed["year"] = observed["year"].astype(int)
    observed["gdp_total_local_observed"] = observed["gdp_total_local_billion_cny"] * 100_000_000.0
    observed["gdp_per_capita"] = np.nan
    observed["gdp_total_ppp_observed"] = np.nan
    observed["gdp_share_national_observed"] = np.nan
    observed["macro_observed_source"] = "china_city_official_gdp"
    observed["macro_resolution_level"] = "municipality_year"
    observed["macro_currency_code"] = "CNY"
    for col in ["oecd_fua_code", "oecd_city_code", "oecd_fua_name", "oecd_city_name"]:
        observed[col] = pd.NA

    mapping = observed[
        [
            "city_id",
            "city_name",
            "country",
            "iso3",
            "macro_geo_code_observed",
            "macro_geo_name_observed",
            "source_family",
        ]
    ].drop_duplicates().sort_values(["city_id"])
    mapping["match_kind"] = "china_official_manual"

    observed = observed[EXPECTED_OUTPUT_COLUMNS].sort_values(["city_id", "year"])
    return observed, mapping


def _build_manual_supplement_observed(
    catalog: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MANUAL_SUPPLEMENT_PATH.exists():
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), pd.DataFrame()

    raw = pd.read_csv(MANUAL_SUPPLEMENT_PATH)
    if raw.empty:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), pd.DataFrame()

    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    raw["gdp_local_value"] = pd.to_numeric(raw["gdp_local_value"], errors="coerce")
    raw["gdp_local_unit_multiplier"] = pd.to_numeric(raw["gdp_local_unit_multiplier"], errors="coerce")
    raw = raw[raw["year"].between(start_year, end_year, inclusive="both")].copy()
    if raw.empty:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), pd.DataFrame()

    lookup = catalog[["city_id", "city_name", "country", "iso3"]].copy()
    observed = raw.merge(lookup, on="city_id", how="inner")
    if observed.empty:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), pd.DataFrame()

    observed["year"] = observed["year"].astype(int)
    observed["gdp_total_local_observed"] = observed["gdp_local_value"] * observed["gdp_local_unit_multiplier"]
    observed["gdp_per_capita"] = np.nan
    observed["gdp_total_ppp_observed"] = np.nan
    observed["gdp_share_national_observed"] = np.nan
    for col in ["oecd_fua_code", "oecd_city_code", "oecd_fua_name", "oecd_city_name"]:
        observed[col] = pd.NA

    mapping = observed[
        [
            "city_id",
            "city_name",
            "country",
            "iso3",
            "macro_geo_code_observed",
            "macro_geo_name_observed",
            "macro_currency_code",
            "macro_resolution_level",
            "macro_observed_source",
            "source_family",
        ]
    ].drop_duplicates().sort_values(["city_id"])
    mapping["match_kind"] = "manual_official_supplement"

    observed = observed[EXPECTED_OUTPUT_COLUMNS].sort_values(["city_id", "year"])
    return observed, mapping


def _build_manual_per_capita_supplement_observed(
    catalog: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MANUAL_PER_CAPITA_SUPPLEMENT_PATH.exists():
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), pd.DataFrame()

    raw = pd.read_csv(MANUAL_PER_CAPITA_SUPPLEMENT_PATH)
    if raw.empty:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), pd.DataFrame()

    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    raw["gdp_per_capita_value"] = pd.to_numeric(raw["gdp_per_capita_value"], errors="coerce")
    raw["gdp_per_capita_unit_multiplier"] = pd.to_numeric(raw["gdp_per_capita_unit_multiplier"], errors="coerce")
    raw = raw[raw["year"].between(start_year, end_year, inclusive="both")].copy()
    if raw.empty:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), pd.DataFrame()

    lookup = catalog[["city_id", "city_name", "country", "iso3"]].copy()
    observed = raw.merge(lookup, on="city_id", how="inner")
    if observed.empty:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS), pd.DataFrame()

    observed["year"] = observed["year"].astype(int)
    observed["gdp_per_capita"] = observed["gdp_per_capita_value"] * observed["gdp_per_capita_unit_multiplier"]
    observed["gdp_total_local_observed"] = np.nan
    observed["gdp_total_ppp_observed"] = np.nan
    observed["gdp_share_national_observed"] = np.nan
    for col in ["oecd_fua_code", "oecd_city_code", "oecd_fua_name", "oecd_city_name"]:
        observed[col] = pd.NA

    mapping = observed[
        [
            "city_id",
            "city_name",
            "country",
            "iso3",
            "macro_geo_code_observed",
            "macro_geo_name_observed",
            "macro_currency_code",
            "macro_resolution_level",
            "macro_observed_source",
            "source_family",
        ]
    ].drop_duplicates().sort_values(["city_id"])
    mapping["match_kind"] = "manual_official_per_capita_supplement"

    observed = observed[EXPECTED_OUTPUT_COLUMNS].sort_values(["city_id", "year"])
    return observed, mapping


def _collapse_multi_source_rows(observed: pd.DataFrame) -> pd.DataFrame:
    if observed.empty:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS)
    work = observed.copy()
    work["source_priority"] = work["macro_observed_source"].map(SOURCE_PRIORITY).fillna(999).astype(int)
    work["non_null_score"] = work[
        ["gdp_per_capita", "gdp_total_ppp_observed", "gdp_total_local_observed", "gdp_share_national_observed"]
    ].notna().sum(axis=1)
    work = work.sort_values(
        ["city_id", "year", "source_priority", "non_null_score"],
        ascending=[True, True, True, False],
    )
    return work.drop_duplicates(subset=["city_id", "year"], keep="first")[EXPECTED_OUTPUT_COLUMNS].reset_index(drop=True)


def _load_existing_observed_source(source_name: str) -> pd.DataFrame:
    if not OUTPUT_PATH.exists():
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS)
    existing = pd.read_csv(OUTPUT_PATH)
    if "macro_observed_source" not in existing.columns:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS)
    existing = existing.loc[existing["macro_observed_source"].astype(str) == str(source_name)].copy()
    if existing.empty:
        return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS)
    for col in EXPECTED_OUTPUT_COLUMNS:
        if col not in existing.columns:
            existing[col] = pd.NA
    return existing[EXPECTED_OUTPUT_COLUMNS].copy()


def _load_existing_mapping(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    existing = pd.read_csv(path)
    for col in columns:
        if col not in existing.columns:
            existing[col] = pd.NA
    return existing[columns].copy()


def main() -> None:
    config = load_config()
    catalog = pd.read_csv(CITY_CATALOG_PATH)[["city_id", "city_name", "country", "iso3", "latitude", "longitude"]].copy()
    start_year = int(config.time_range.start_year)
    end_year = int(config.time_range.end_year)

    oecd_cities, oecd_fuas = _load_oecd_boundaries()
    oecd_mapping = _match_catalog_to_oecd(catalog, oecd_cities, oecd_fuas)
    oecd_spatial_fallback = _match_catalog_to_oecd_spatial_fallback(catalog, oecd_mapping, oecd_fuas)
    if not oecd_spatial_fallback.empty:
        oecd_mapping = (
            pd.concat([oecd_mapping, oecd_spatial_fallback], ignore_index=True, sort=False)
            .drop_duplicates(subset=["city_id"], keep="first")
            .sort_values(["iso3", "city_name"])
            .reset_index(drop=True)
        )
    try:
        oecd_observed = _build_oecd_observed(oecd_mapping, start_year=start_year, end_year=end_year)
    except Exception as exc:
        print(f"[warn] OECD FUA fetch failed, keeping cached rows: {exc}", file=sys.stderr)
        oecd_observed = _load_existing_observed_source("oecd_fua_economy")

    statcan_mapping_columns = [
        "city_id",
        "city_name",
        "country",
        "iso3",
        "statcan_dguid",
        "statcan_geo_name",
        "match_kind",
        "oecd_fua_code",
        "oecd_city_code",
        "oecd_fua_name",
        "oecd_city_name",
    ]
    try:
        statcan_observed, statcan_mapping = _build_statcan_observed(
            catalog,
            oecd_mapping,
            start_year=start_year,
            end_year=end_year,
        )
    except Exception as exc:
        print(f"[warn] Statistics Canada fetch failed, keeping cached rows: {exc}", file=sys.stderr)
        statcan_observed = _load_existing_observed_source("statcan_cma_gdp")
        statcan_mapping = _load_existing_mapping(STATCAN_MAPPING_PATH, statcan_mapping_columns)

    ibge_mapping_columns = [
        "city_id",
        "city_name",
        "country",
        "iso3",
        "macro_geo_code_observed",
        "macro_geo_name_observed",
        "match_kind",
    ]
    try:
        ibge_observed, ibge_mapping = _build_ibge_brazil_observed(
            catalog,
            start_year=start_year,
            end_year=end_year,
        )
    except Exception as exc:
        print(f"[warn] IBGE fetch failed, keeping cached rows: {exc}", file=sys.stderr)
        ibge_observed = _load_existing_observed_source("ibge_municipal_gdp")
        ibge_mapping = _load_existing_mapping(IBGE_MAPPING_PATH, ibge_mapping_columns)
    china_observed, china_mapping = _build_china_official_observed(
        catalog,
        start_year=start_year,
        end_year=end_year,
    )
    manual_observed, manual_mapping = _build_manual_supplement_observed(
        catalog,
        start_year=start_year,
        end_year=end_year,
    )
    manual_per_capita_observed, manual_per_capita_mapping = _build_manual_per_capita_supplement_observed(
        catalog,
        start_year=start_year,
        end_year=end_year,
    )

    observed = _collapse_multi_source_rows(
        pd.concat(
            [
                oecd_observed,
                statcan_observed,
                ibge_observed,
                china_observed,
                manual_observed,
                manual_per_capita_observed,
            ],
            ignore_index=True,
            sort=False,
        )
    )
    observed = observed.sort_values(["city_id", "year"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    observed.to_csv(OUTPUT_PATH, index=False)
    oecd_mapping.to_csv(MAPPING_PATH, index=False)
    statcan_mapping.to_csv(STATCAN_MAPPING_PATH, index=False)
    ibge_mapping.to_csv(IBGE_MAPPING_PATH, index=False)
    china_mapping.to_csv(CHINA_MAPPING_PATH, index=False)
    manual_mapping.to_csv(MANUAL_SUPPLEMENT_MAPPING_PATH, index=False)
    manual_per_capita_mapping.to_csv(MANUAL_PER_CAPITA_MAPPING_PATH, index=False)

    matched_ids = sorted(oecd_mapping["city_id"].unique().tolist())
    covered_city_ids = sorted(observed["city_id"].unique().tolist())
    source_row_counts = observed["macro_observed_source"].value_counts(dropna=False).to_dict()
    source_city_counts = observed.groupby("macro_observed_source")["city_id"].nunique().to_dict()
    summary = {
        "status": "ok",
        "sources": {
            "china_city_official_gdp": str(CHINA_OFFICIAL_GDP_PATH),
            "manual_supplement": str(MANUAL_SUPPLEMENT_PATH),
            "manual_per_capita_supplement": str(MANUAL_PER_CAPITA_SUPPLEMENT_PATH),
            "oecd_fua_economy": OECD_FUA_ECONOMY_URL,
            "statcan_cma_gdp": STATCAN_CMA_GDP_URL,
            "ibge_municipal_gdp": IBGE_MUNICIPAL_GDP_URL,
        },
        "cities_in_catalog": int(catalog["city_id"].nunique()),
        "cities_matched_to_oecd_fua": int(len(matched_ids)),
        "cities_with_gdp_rows_written": int(observed["city_id"].nunique()),
        "rows_written": int(len(observed)),
        "year_range": [
            int(observed["year"].min()) if not observed.empty else None,
            int(observed["year"].max()) if not observed.empty else None,
        ],
        "gdp_per_capita_non_null_rows": int(pd.to_numeric(observed["gdp_per_capita"], errors="coerce").notna().sum()),
        "gdp_total_ppp_non_null_rows": int(pd.to_numeric(observed["gdp_total_ppp_observed"], errors="coerce").notna().sum()),
        "gdp_total_local_non_null_rows": int(pd.to_numeric(observed["gdp_total_local_observed"], errors="coerce").notna().sum()),
        "source_row_counts": source_row_counts,
        "source_city_counts": source_city_counts,
        "cities_with_china_rows_written": int(china_observed["city_id"].nunique()),
        "rows_written_china": int(len(china_observed)),
        "cities_with_manual_supplement_rows_written": int(manual_observed["city_id"].nunique()),
        "rows_written_manual_supplement": int(len(manual_observed)),
        "cities_with_manual_per_capita_rows_written": int(manual_per_capita_observed["city_id"].nunique()),
        "rows_written_manual_per_capita": int(len(manual_per_capita_observed)),
        "cities_with_statcan_rows_written": int(statcan_observed["city_id"].nunique()),
        "rows_written_statcan": int(len(statcan_observed)),
        "cities_with_ibge_rows_written": int(ibge_observed["city_id"].nunique()),
        "rows_written_ibge": int(len(ibge_observed)),
        "matched_city_ids": covered_city_ids,
        "unmatched_city_ids_sample": sorted(catalog.loc[~catalog["city_id"].isin(covered_city_ids), "city_id"].tolist())[:50],
    }
    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    SUMMARY_PATH.write_text(payload, encoding="utf-8")
    LEGACY_SUMMARY_PATH.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
