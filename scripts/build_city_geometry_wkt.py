from __future__ import annotations

import argparse
import math
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CITY_CATALOG = ROOT / "data" / "raw" / "city_catalog.csv"
DEFAULT_UCDB_GPKG = (
    ROOT
    / "data"
    / "raw"
    / "boundaries"
    / "ghsl_ucdb_r2024a"
    / "GHS_UCDB_GLOBE_R2024A.gpkg"
)
DEFAULT_UCDB_LAYER = "GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"
DEFAULT_FUA_GPKG = (
    ROOT
    / "data"
    / "raw"
    / "boundaries"
    / "ghsl_fua_r2019a"
    / "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"
)
DEFAULT_FUA_LAYER = "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0"
DEFAULT_OECD_CITIES_ZIP = (
    ROOT / "data" / "raw" / "boundaries" / "oecd_downloads" / "cities (4).zip"
)
DEFAULT_MAPPING_OUTPUT = ROOT / "data" / "raw" / "city_boundary_mapping.csv"
DEFAULT_UNMATCHED_OUTPUT = ROOT / "data" / "raw" / "city_boundary_unmatched.csv"
DEFAULT_MANUAL_OVERRIDES = ROOT / "data" / "raw" / "city_boundary_manual_overrides.csv"
MAPPING_OUTPUT_COLUMNS = [
    "city_id",
    "city_name",
    "country",
    "iso3",
    "boundary_source",
    "source_layer",
    "source_id",
    "source_city_name",
    "source_country_name",
    "match_type",
    "match_score",
]
UNMATCHED_OUTPUT_COLUMNS = [
    "city_id",
    "city_name",
    "country",
    "iso3",
    "boundary_source_attempted",
    "match_type",
    "match_score",
    "suggested_source_id",
    "suggested_source_city_name",
    "suggested_source_country_name",
]

REQUIRED_CITY_COLUMNS = [
    "city_id",
    "city_name",
    "country",
    "iso3",
    "latitude",
    "longitude",
]
REQUIRED_UCDB_COLUMNS = [
    "ID_UC_G0",
    "GC_UCN_MAI_2025",
    "GC_UCN_LIS_2025",
    "GC_CNT_GAD_2025",
    "geometry",
]
REQUIRED_FUA_COLUMNS = [
    "eFUA_ID",
    "eFUA_name",
    "Cntry_name",
    "geometry",
]
REQUIRED_OECD_CITY_COLUMNS = [
    "iso3",
    "citycode",
    "cityname",
    "cityname_e",
    "geometry",
]

COUNTRY_ALIASES = {
    "bolivia": "bolivia plurinational state of",
    "cape verde": "cabo verde",
    "cote d ivoire": "cote divoire",
    "czech republic": "czechia",
    "democratic republic of the congo": "congo democratic republic of the",
    "hong kong": "hong kong sar china",
    "iran": "iran islamic republic of",
    "laos": "lao peoples democratic republic",
    "macau": "macao sar china",
    "moldova": "moldova republic of",
    "palestine": "palestine state of",
    "russia": "russian federation",
    "south korea": "korea republic of",
    "syria": "syrian arab republic",
    "taiwan": "taiwan province of china",
    "tanzania": "tanzania united republic of",
    "turkey": "turkiye",
    "venezuela": "venezuela bolivarian republic of",
    "vietnam": "viet nam",
}

CITY_ALIASES = {
    "canberra": {"canberra", "north canberra", "north canberra canberra"},
    "darwin": {"darwin", "greater darwin"},
    "delhi": {"delhi", "new delhi"},
    "djibouti city": {"djibouti city", "djibouti"},
    "ho chi minh city": {"ho chi minh city", "thanh pho ho chi minh", "saigon"},
    "hong kong": {"hong kong", "hong kong sar"},
    "mexico city": {"mexico city", "ciudad de mexico"},
    "quebec city": {"quebec city", "quebec"},
    "sao paulo": {"sao paulo", "sao paulo"},
    "santa cruz": {"santa cruz", "santa cruz de la sierra"},
    "st petersburg": {"st petersburg", "saint petersburg"},
    "washington": {"washington", "washington dc", "district of columbia"},
}


def _normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("&", " and ")
    text = re.sub(r"[\u2019']", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _expand_country_aliases(country: str) -> set[str]:
    country_norm = _normalize_text(country)
    values = {country_norm}
    if country_norm in COUNTRY_ALIASES:
        values.add(_normalize_text(COUNTRY_ALIASES[country_norm]))
    reverse_matches = {
        left for left, right in COUNTRY_ALIASES.items() if _normalize_text(right) == country_norm
    }
    values.update(_normalize_text(value) for value in reverse_matches)
    return {value for value in values if value}


def _expand_city_aliases(city: str) -> set[str]:
    city_norm = _normalize_text(city)
    values = {city_norm}
    if city_norm in CITY_ALIASES:
        values.update(_normalize_text(value) for value in CITY_ALIASES[city_norm])
    reverse_matches = {
        left for left, aliases in CITY_ALIASES.items() if city_norm in {_normalize_text(v) for v in aliases}
    }
    values.update(_normalize_text(value) for value in reverse_matches)
    return {value for value in values if value}


def _split_aliases(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r"[;|/]", text)
    return [part.strip() for part in parts if part.strip()]


def _distance_key(candidate: pd.Series, lon: float, lat: float) -> float:
    dx = float(candidate["centroid_lon"]) - lon
    dy = float(candidate["centroid_lat"]) - lat
    return dx * dx + dy * dy


def _select_best_candidate(candidates: pd.DataFrame, lon: float, lat: float) -> pd.Series | None:
    if candidates.empty:
        return None
    if len(candidates) == 1:
        return candidates.iloc[0]
    return min(
        (row for _, row in candidates.iterrows()),
        key=lambda row: _distance_key(row, lon, lat),
    )


def _best_fuzzy_candidate(
    city_names: set[str],
    country_names: set[str],
    ucdb: pd.DataFrame,
) -> tuple[pd.Series | None, float, float]:
    search_space = ucdb[ucdb["country_norm"].isin(country_names)]
    if search_space.empty:
        search_space = ucdb

    best_row: pd.Series | None = None
    best_score = -1.0
    second_score = -1.0

    for _, row in search_space.iterrows():
        score = max(
            (
                SequenceMatcher(None, catalog_name, ghsl_name).ratio()
                for catalog_name in city_names
                for ghsl_name in row["all_name_norms"]
            ),
            default=0.0,
        )
        if score > best_score:
            second_score = best_score
            best_score = score
            best_row = row
        elif score > second_score:
            second_score = score

    gap = best_score - max(second_score, 0.0)
    return best_row, best_score, gap


def _strip_bom_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns=lambda col: col.lstrip("\ufeff") if isinstance(col, str) else col)


def _load_ucdb(path: Path, layer: str) -> gpd.GeoDataFrame:
    ucdb = gpd.read_file(path, layer=layer)
    ucdb = _strip_bom_columns(ucdb)
    missing = [column for column in REQUIRED_UCDB_COLUMNS if column not in ucdb.columns]
    if missing:
        raise ValueError(f"UCDB layer missing required columns: {missing}")

    ucdb = ucdb[REQUIRED_UCDB_COLUMNS].copy()
    centroids = gpd.GeoSeries(ucdb.geometry.centroid, crs=ucdb.crs).to_crs(epsg=4326)
    ucdb = ucdb.to_crs(epsg=4326)
    ucdb["centroid_lon"] = centroids.x
    ucdb["centroid_lat"] = centroids.y
    ucdb["source_id"] = ucdb["ID_UC_G0"].astype("Int64")
    ucdb["source_city_name"] = ucdb["GC_UCN_MAI_2025"].astype(str)
    ucdb["source_country_name"] = ucdb["GC_CNT_GAD_2025"].astype(str)
    ucdb["country_norm"] = ucdb["source_country_name"].map(_normalize_text)

    def build_name_norms(row: pd.Series) -> list[str]:
        names = [row["source_city_name"], *_split_aliases(row["GC_UCN_LIS_2025"])]
        normalized = {_normalize_text(name) for name in names if _normalize_text(name)}
        return sorted(normalized)

    ucdb["all_name_norms"] = ucdb.apply(build_name_norms, axis=1)
    ucdb["boundary_source"] = "ghsl_ucdb_r2024a"
    ucdb["source_layer"] = layer
    return ucdb


def _load_fua(path: Path, layer: str) -> gpd.GeoDataFrame:
    fua = gpd.read_file(path, layer=layer)
    fua = _strip_bom_columns(fua)
    missing = [column for column in REQUIRED_FUA_COLUMNS if column not in fua.columns]
    if missing:
        raise ValueError(f"FUA layer missing required columns: {missing}")

    fua = fua[REQUIRED_FUA_COLUMNS].copy()
    centroids = gpd.GeoSeries(fua.to_crs(epsg=3857).geometry.centroid, crs=3857).to_crs(epsg=4326)
    fua = fua.to_crs(epsg=4326)
    fua["centroid_lon"] = centroids.x
    fua["centroid_lat"] = centroids.y
    fua["source_id"] = pd.to_numeric(fua["eFUA_ID"], errors="coerce").astype("Int64")
    fua["source_city_name"] = fua["eFUA_name"].astype(str)
    fua["source_country_name"] = fua["Cntry_name"].astype(str)
    fua["country_norm"] = fua["source_country_name"].map(_normalize_text)
    fua["all_name_norms"] = fua["source_city_name"].map(lambda name: [_normalize_text(name)])
    fua["boundary_source"] = "ghsl_fua_r2019a"
    fua["source_layer"] = layer
    return fua


def _load_oecd_cities(path: Path) -> gpd.GeoDataFrame:
    cities = gpd.read_file(path)
    cities = _strip_bom_columns(cities)
    missing = [column for column in REQUIRED_OECD_CITY_COLUMNS if column not in cities.columns]
    if missing:
        raise ValueError(f"OECD cities source missing required columns: {missing}")

    cities = cities[REQUIRED_OECD_CITY_COLUMNS].copy()
    centroids = gpd.GeoSeries(cities.to_crs(epsg=3857).geometry.centroid, crs=3857).to_crs(epsg=4326)
    cities = cities.to_crs(epsg=4326)
    cities["centroid_lon"] = centroids.x
    cities["centroid_lat"] = centroids.y
    cities["source_id"] = cities["citycode"].astype(str)
    cities["source_city_name"] = cities["cityname_e"].fillna(cities["cityname"]).astype(str)
    cities["source_country_name"] = cities["iso3"].astype(str)
    cities["country_norm"] = cities["iso3"].map(_normalize_text)
    cities["all_name_norms"] = cities.apply(
        lambda row: sorted(
            {
                _normalize_text(row["cityname"]),
                _normalize_text(row["cityname_e"]),
            }
            - {""}
        ),
        axis=1,
    )
    cities["boundary_source"] = "oecd_cities"
    cities["source_layer"] = path.name
    return cities


def _build_name_index(boundaries: pd.DataFrame) -> dict[tuple[str, str], list[int]]:
    name_index: dict[tuple[str, str], list[int]] = {}
    for idx, row in boundaries.iterrows():
        for city_name in row["all_name_norms"]:
            key = (row["country_norm"], city_name)
            name_index.setdefault(key, []).append(idx)
    return name_index


def _load_manual_overrides(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["city_id", "source_id"])
    overrides = pd.read_csv(path)
    if "city_id" not in overrides.columns:
        raise ValueError("Manual overrides must contain a city_id column")
    return overrides


def _match_city(
    city_row: pd.Series,
    ucdb: pd.DataFrame,
    name_index: dict[tuple[str, str], list[int]],
    manual_overrides: pd.DataFrame,
) -> tuple[pd.Series | None, dict[str, Any]]:
    city_names = _expand_city_aliases(city_row["city_name"])
    country_names = _expand_country_aliases(city_row["country"])
    country_names.add(_normalize_text(city_row.get("iso3")))
    lon = float(city_row["longitude"])
    lat = float(city_row["latitude"])

    override = manual_overrides[manual_overrides["city_id"] == city_row["city_id"]]
    if not override.empty:
        source_id = override.iloc[0].get("source_id")
        if pd.notna(source_id):
            matched = ucdb[ucdb["source_id"].astype(str) == str(source_id).strip()]
            row = _select_best_candidate(matched, lon, lat)
            if row is not None:
                return row, {"match_type": "manual_source_id", "match_score": 1.0}

    exact_matches: list[int] = []
    for country_name in country_names:
        for city_name in city_names:
            exact_matches.extend(name_index.get((country_name, city_name), []))

    if exact_matches:
        matched = ucdb.iloc[sorted(set(exact_matches))]
        row = _select_best_candidate(matched, lon, lat)
        if row is not None:
            match_type = "exact_name_country"
            if len(matched) > 1:
                match_type = "exact_name_country_nearest"
            return row, {"match_type": match_type, "match_score": 1.0}

    unique_global_matches: list[int] = []
    for city_name in city_names:
        unique_global_matches.extend(
            ucdb.index[ucdb["all_name_norms"].map(lambda names: city_name in names)].tolist()
        )
    unique_global_matches = sorted(set(unique_global_matches))
    if len(unique_global_matches) == 1:
        row = ucdb.loc[unique_global_matches[0]]
        return row, {"match_type": "exact_name_global_unique", "match_score": 1.0}

    fuzzy_row, fuzzy_score, fuzzy_gap = _best_fuzzy_candidate(city_names, country_names, ucdb)
    if fuzzy_row is not None and fuzzy_score >= 0.94 and fuzzy_gap >= 0.05:
        return fuzzy_row, {
            "match_type": "fuzzy_name_country_high_confidence",
            "match_score": round(float(fuzzy_score), 4),
        }

    return None, {
        "match_type": "unmatched",
        "match_score": round(float(max(fuzzy_score, 0.0)), 4),
        "suggested_source_id": str(fuzzy_row["source_id"]) if fuzzy_row is not None else pd.NA,
        "suggested_source_city_name": fuzzy_row["source_city_name"] if fuzzy_row is not None else pd.NA,
        "suggested_source_country_name": (
            fuzzy_row["source_country_name"] if fuzzy_row is not None else pd.NA
        ),
    }


def build_geometry_wkt(
    city_catalog_path: Path,
    ucdb_gpkg_path: Path,
    ucdb_layer: str,
    fua_gpkg_path: Path,
    fua_layer: str,
    oecd_cities_zip_path: Path,
    output_city_catalog_path: Path,
    mapping_output_path: Path,
    unmatched_output_path: Path,
    manual_overrides_path: Path,
) -> None:
    cities = pd.read_csv(city_catalog_path)
    missing = [column for column in REQUIRED_CITY_COLUMNS if column not in cities.columns]
    if missing:
        raise ValueError(f"City catalog missing required columns: {missing}")

    if "geometry_wkt" not in cities.columns:
        cities["geometry_wkt"] = pd.NA
    cities["geometry_wkt"] = cities["geometry_wkt"].astype("object")

    ucdb = _load_ucdb(ucdb_gpkg_path, ucdb_layer)
    fua = _load_fua(fua_gpkg_path, fua_layer) if fua_gpkg_path.exists() else None
    oecd_cities = _load_oecd_cities(oecd_cities_zip_path) if oecd_cities_zip_path.exists() else None
    manual_overrides = _load_manual_overrides(manual_overrides_path)
    boundary_datasets: list[tuple[str, pd.DataFrame, dict[tuple[str, str], list[int]]]] = [
        ("ghsl_ucdb_r2024a", ucdb, _build_name_index(ucdb))
    ]
    if fua is not None:
        boundary_datasets.append(("ghsl_fua_r2019a", fua, _build_name_index(fua)))
    if oecd_cities is not None:
        boundary_datasets.append(("oecd_cities", oecd_cities, _build_name_index(oecd_cities)))

    mapping_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []

    for idx, city_row in cities.iterrows():
        matched_row = None
        meta: dict[str, Any] = {}
        boundary_source = pd.NA
        for dataset_name, boundaries, name_index in boundary_datasets:
            candidate_row, candidate_meta = _match_city(
                city_row,
                boundaries,
                name_index,
                manual_overrides,
            )
            if candidate_row is not None:
                matched_row = candidate_row
                meta = candidate_meta
                boundary_source = dataset_name
                break
            if not meta:
                meta = candidate_meta

        if matched_row is None:
            unmatched_rows.append(
                {
                    "city_id": city_row["city_id"],
                    "city_name": city_row["city_name"],
                    "country": city_row["country"],
                    "iso3": city_row["iso3"],
                    "boundary_source_attempted": ",".join(name for name, _, _ in boundary_datasets),
                    **meta,
                }
            )
            continue

        geometry = matched_row.geometry
        if geometry is None or geometry.is_empty:
            unmatched_rows.append(
                {
                    "city_id": city_row["city_id"],
                    "city_name": city_row["city_name"],
                    "country": city_row["country"],
                    "iso3": city_row["iso3"],
                    "match_type": "matched_but_empty_geometry",
                    "source_id": matched_row["source_id"],
                    "source_city_name": matched_row["source_city_name"],
                    "source_country_name": matched_row["source_country_name"],
                }
            )
            continue

        cities.at[idx, "geometry_wkt"] = geometry.wkt
        mapping_rows.append(
            {
                "city_id": city_row["city_id"],
                "city_name": city_row["city_name"],
                "country": city_row["country"],
                "iso3": city_row["iso3"],
                "boundary_source": boundary_source,
                "source_layer": matched_row["source_layer"],
                "source_id": matched_row["source_id"],
                "source_city_name": matched_row["source_city_name"],
                "source_country_name": matched_row["source_country_name"],
                "match_type": meta["match_type"],
                "match_score": meta["match_score"],
            }
        )

    output_city_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_output_path.parent.mkdir(parents=True, exist_ok=True)
    unmatched_output_path.parent.mkdir(parents=True, exist_ok=True)

    mapping_frame = pd.DataFrame(mapping_rows, columns=MAPPING_OUTPUT_COLUMNS)
    unmatched_frame = pd.DataFrame(unmatched_rows, columns=UNMATCHED_OUTPUT_COLUMNS)
    if not mapping_frame.empty:
        mapping_frame = mapping_frame.sort_values("city_id")
    if not unmatched_frame.empty:
        unmatched_frame = unmatched_frame.sort_values("city_id")

    cities.to_csv(output_city_catalog_path, index=False)
    mapping_frame.to_csv(mapping_output_path, index=False)
    unmatched_frame.to_csv(unmatched_output_path, index=False)

    matched_count = int(cities["geometry_wkt"].notna().sum())
    unmatched_count = int(cities["geometry_wkt"].isna().sum())
    print(f"Matched cities: {matched_count}")
    print(f"Unmatched cities: {unmatched_count}")
    print(f"City catalog written to: {output_city_catalog_path}")
    print(f"Mapping report written to: {mapping_output_path}")
    print(f"Unmatched report written to: {unmatched_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build geometry_wkt for city_catalog.csv from GHSL UCDB polygons."
    )
    parser.add_argument(
        "--city-catalog",
        type=Path,
        default=DEFAULT_CITY_CATALOG,
        help="Input city catalog CSV.",
    )
    parser.add_argument(
        "--ucdb-gpkg",
        type=Path,
        default=DEFAULT_UCDB_GPKG,
        help="Path to the GHSL UCDB GeoPackage.",
    )
    parser.add_argument(
        "--ucdb-layer",
        default=DEFAULT_UCDB_LAYER,
        help="UCDB polygon layer to use.",
    )
    parser.add_argument(
        "--fua-gpkg",
        type=Path,
        default=DEFAULT_FUA_GPKG,
        help="Optional GHSL FUA GeoPackage used as a fallback source.",
    )
    parser.add_argument(
        "--fua-layer",
        default=DEFAULT_FUA_LAYER,
        help="GHSL FUA polygon layer to use.",
    )
    parser.add_argument(
        "--oecd-cities-zip",
        type=Path,
        default=DEFAULT_OECD_CITIES_ZIP,
        help="Optional OECD cities shapefile zip used as an additional fallback source.",
    )
    parser.add_argument(
        "--output-city-catalog",
        type=Path,
        default=DEFAULT_CITY_CATALOG,
        help="Output city catalog CSV. Defaults to in-place overwrite.",
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=DEFAULT_MAPPING_OUTPUT,
        help="Output CSV for matched boundary metadata.",
    )
    parser.add_argument(
        "--unmatched-output",
        type=Path,
        default=DEFAULT_UNMATCHED_OUTPUT,
        help="Output CSV for unmatched cities and suggested candidates.",
    )
    parser.add_argument(
        "--manual-overrides",
        type=Path,
        default=DEFAULT_MANUAL_OVERRIDES,
        help="Optional CSV with manual overrides containing city_id and source_id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_geometry_wkt(
        city_catalog_path=args.city_catalog,
        ucdb_gpkg_path=args.ucdb_gpkg,
        ucdb_layer=args.ucdb_layer,
        fua_gpkg_path=args.fua_gpkg,
        fua_layer=args.fua_layer,
        oecd_cities_zip_path=args.oecd_cities_zip,
        output_city_catalog_path=args.output_city_catalog,
        mapping_output_path=args.mapping_output,
        unmatched_output_path=args.unmatched_output,
        manual_overrides_path=args.manual_overrides,
    )


if __name__ == "__main__":
    main()
