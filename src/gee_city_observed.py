from __future__ import annotations

"""Google Earth Engine prep and CSV import helpers for city-observed signals."""

import logging
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .city_catalog import load_city_catalog
from .historical_viirs import merge_viirs_monthly_panels
from .utils import DATA_RAW, dump_json

LOGGER = logging.getLogger(__name__)

GEE_VIIRS_RADIANCE_ALIASES: List[str] = [
    "radiance",
    "avg_rad",
    "avg_rad_mean",
    "avg_radiance",
    "ntl_mean",
    "mean_ntl",
]

GEE_VIIRS_COVERAGE_ALIASES: List[str] = [
    "cf_cvg",
    "cf_cvg_mean",
    "cloud_free_coverage",
]

GEE_VIIRS_LIT_AREA_ALIASES: List[str] = [
    "lit_area_km2",
    "lit_area",
]

GEE_VIIRS_SUM_ALIASES: List[str] = [
    "radiance_sum",
    "viirs_ntl_sum",
    "ntl_sum",
]

GEE_NO2_MEAN_ALIASES: List[str] = [
    "no2_trop_mean",
    "no2_mean",
    "tropospheric_no2_column_number_density",
    "ColumnAmountNO2TropCloudScreened",
    "no2_column",
]

GEE_NO2_COUNT_ALIASES: List[str] = [
    "no2_valid_obs_count",
    "valid_obs_count",
    "count",
    "no2_count",
]

GEE_GHSL_ALIASES: Dict[str, List[str]] = {
    "ghsl_built_surface_km2": [
        "ghsl_built_surface_km2",
        "built_surface_km2",
        "built_surface",
        "built_surface_mean",
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
    return out.drop(columns=["city_name_l"], errors="ignore")


def _city_points_frame(max_cities: int, buffer_m: int) -> pd.DataFrame:
    cities = load_city_catalog(max_cities=int(max_cities)).copy()
    cities["buffer_m"] = int(buffer_m)
    if "geometry_wkt" not in cities.columns:
        cities["geometry_wkt"] = np.nan
    return cities[
        [
            "city_id",
            "city_name",
            "country",
            "iso3",
            "continent",
            "latitude",
            "longitude",
            "buffer_m",
            "geometry_wkt",
        ]
    ].copy()


def _viirs_script_template(asset_id: str, start_year: int, end_year: int) -> str:
    return dedent(
        f"""
        // Export city-level monthly VIIRS means from Google Earth Engine.
        // Dataset: NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG
        // Official docs: https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG
        //
        // Upload the polygon city table (with geometry_wkt) as a table asset and update ASSET_ID below.

        var ASSET_ID = '{asset_id}';
        var START = ee.Date('{int(start_year)}-01-01');
        var END = ee.Date('{int(end_year + 1)}-01-01');

        var cities = ee.FeatureCollection(ASSET_ID)
          .map(function(f) {{
            var geom = f.geometry();
            var hasBoundary = ee.List(['Polygon', 'MultiPolygon']).contains(ee.String(geom.type()));
            return ee.Feature(geom, f.toDictionary()).set('has_boundary_geometry', hasBoundary);
          }})
          .filter(ee.Filter.eq('has_boundary_geometry', true));

        var collection = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
          .filterDate(START, END)
          .select(['avg_rad', 'cf_cvg']);

        var rows = collection.map(function(img) {{
          var date = ee.Date(img.get('system:time_start'));
          var clipped = img.clipToCollection(cities);
          var reduced = clipped.reduceRegions({{
            collection: cities,
            reducer: ee.Reducer.mean().combine({{
              reducer2: ee.Reducer.sum(),
              sharedInputs: true
            }}),
            scale: 500,
          }});
          return reduced.map(function(f) {{
            return ee.Feature(null, {{
              city_id: f.get('city_id'),
              city_name: f.get('city_name'),
              iso3: f.get('iso3'),
              year: date.get('year'),
              month: date.get('month'),
              radiance: f.get('avg_rad_mean'),
              radiance_sum: f.get('avg_rad_sum'),
              cf_cvg: f.get('cf_cvg_mean'),
              viirs_source: 'gee_viirs_monthly_vcmslcfg_polygon'
            }});
          }});
        }}).flatten();

        Export.table.toDrive({{
          collection: rows,
          description: 'urban_pulse_viirs_monthly',
          fileNamePrefix: 'urban_pulse_viirs_monthly',
          fileFormat: 'CSV',
          selectors: ['city_id', 'city_name', 'iso3', 'year', 'month', 'radiance', 'radiance_sum', 'cf_cvg', 'viirs_source']
        }});
        """
    ).strip() + "\n"


def _ghsl_script_template(asset_id: str) -> str:
    return dedent(
        f"""
        // Export city-level GHSL built-surface and built-volume summaries from Google Earth Engine.
        // Datasets:
        //   JRC/GHSL/P2023A/GHS_BUILT_S
        //   JRC/GHSL/P2023A/GHS_BUILT_V
        // Official docs:
        //   https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_S
        //   https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_V
        //
        // Upload the polygon city table (with geometry_wkt) as a table asset and update ASSET_ID below.

        var ASSET_ID = '{asset_id}';
        var YEARS = [2015, 2020, 2025];

        var cities = ee.FeatureCollection(ASSET_ID)
          .map(function(f) {{
            var geom = f.geometry();
            var hasBoundary = ee.List(['Polygon', 'MultiPolygon']).contains(ee.String(geom.type()));
            return ee.Feature(geom, f.toDictionary()).set('has_boundary_geometry', hasBoundary);
          }})
          .filter(ee.Filter.eq('has_boundary_geometry', true));

        var rows = ee.FeatureCollection(YEARS.map(function(year) {{
          var builtSurface = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/' + year)
            .select(['built_surface', 'built_surface_nres']);
          var builtVolume = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_V/' + year)
            .select(['built_volume_total', 'built_volume_nres']);
          var image = builtSurface.addBands(builtVolume).clipToCollection(cities);
          var reduced = image.reduceRegions({{
            collection: cities,
            reducer: ee.Reducer.sum(),
            scale: 100,
          }});
          return reduced.map(function(f) {{
            return ee.Feature(null, {{
              city_id: f.get('city_id'),
              city_name: f.get('city_name'),
              iso3: f.get('iso3'),
              year: year,
              ghsl_built_surface_km2: ee.Number(f.get('built_surface')).divide(1e6),
              ghsl_built_surface_nres_km2: ee.Number(f.get('built_surface_nres')).divide(1e6),
              ghsl_built_volume_m3: f.get('built_volume_total'),
              ghsl_built_volume_nres_m3: f.get('built_volume_nres'),
              ghsl_source: 'gee_ghsl_p2023a'
            }});
          }});
        }})).flatten();

        Export.table.toDrive({{
          collection: rows,
          description: 'urban_pulse_ghsl_yearly',
          fileNamePrefix: 'urban_pulse_ghsl_yearly',
          fileFormat: 'CSV',
          selectors: [
            'city_id',
            'city_name',
            'iso3',
            'year',
            'ghsl_built_surface_km2',
            'ghsl_built_surface_nres_km2',
            'ghsl_built_volume_m3',
            'ghsl_built_volume_nres_m3',
            'ghsl_source'
          ]
        }});
        """
    ).strip() + "\n"


def _no2_script_template(asset_id: str, start_year: int, end_year: int) -> str:
    effective_start = "2018-07-01" if int(start_year) <= 2018 else f"{int(start_year)}-01-01"
    return dedent(
        f"""
        // Export city-level monthly NO2 means from Google Earth Engine.
        // Datasets:
        //   COPERNICUS/S5P/OFFL/L3_NO2 (2018-07+)
        //
        // Official S5P OFFL docs:
        //   https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2

        var ASSET_ID = '{asset_id}';
        var START = ee.Date('{effective_start}');
        var END = ee.Date('{int(end_year + 1)}-01-01');

        var cities = ee.FeatureCollection(ASSET_ID)
          .map(function(f) {{
            var geom = f.geometry();
            var hasBoundary = ee.List(['Polygon', 'MultiPolygon']).contains(ee.String(geom.type()));
            return ee.Feature(geom, f.toDictionary()).set('has_boundary_geometry', hasBoundary);
          }})
          .filter(ee.Filter.eq('has_boundary_geometry', true));

        function monthStarts(start, end) {{
          var nMonths = end.difference(start, 'month').round();
          return ee.List.sequence(0, nMonths.subtract(1)).map(function(i) {{
            return start.advance(i, 'month');
          }});
        }}

        function monthlyNo2Image(monthStart) {{
          monthStart = ee.Date(monthStart);
          var monthEnd = monthStart.advance(1, 'month');
          var s5p = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
            .filterDate(monthStart, monthEnd)
            .select('tropospheric_NO2_column_number_density')
            .mean()
            .rename('no2_trop');
          return s5p
            .set('year', monthStart.get('year'))
            .set('month', monthStart.get('month'))
            .set('no2_source', 's5p_offl_no2');
        }}

        var rows = ee.FeatureCollection(monthStarts(START, END).map(function(monthStart) {{
          monthStart = ee.Date(monthStart);
          var img = monthlyNo2Image(monthStart);
          var valid = ee.Image.constant(1).updateMask(img.mask()).rename('valid_obs');
          return cities.map(function(f) {{
            var meanStats = img.reduceRegion({{
              reducer: ee.Reducer.mean(),
              geometry: f.geometry(),
              scale: 10000,
              bestEffort: true,
              maxPixels: 1e9,
              tileScale: 4
            }});
            var countStats = valid.reduceRegion({{
              reducer: ee.Reducer.sum(),
              geometry: f.geometry(),
              scale: 10000,
              bestEffort: true,
              maxPixels: 1e9,
              tileScale: 4
            }});
            return ee.Feature(null, {{
              city_id: f.get('city_id'),
              city_name: f.get('city_name'),
              iso3: f.get('iso3'),
              year: monthStart.get('year'),
              month: monthStart.get('month'),
              no2_trop_mean: meanStats.get('no2_trop'),
              no2_valid_obs_count: countStats.get('valid_obs'),
              no2_source: img.get('no2_source')
            }});
          }});
        }})).flatten();

        Export.table.toDrive({{
          collection: rows,
          description: 'urban_pulse_no2_monthly',
          fileNamePrefix: 'urban_pulse_no2_monthly',
          fileFormat: 'CSV',
          selectors: ['city_id', 'city_name', 'iso3', 'year', 'month', 'no2_trop_mean', 'no2_valid_obs_count', 'no2_source']
        }});
        """
    ).strip() + "\n"


def _read_csv_or_raise(path: str | Path) -> pd.DataFrame:
    src = Path(path).expanduser().resolve()
    if not src.exists():
        msg = f"Source file does not exist: {src}"
        raise FileNotFoundError(msg)
    return pd.read_csv(src)


def prepare_gee_city_bundle(
    *,
    max_cities: int = 295,
    buffer_m: int = 5000,
    output_dir: str | Path | None = None,
    asset_id: str = "users/your_username/gee_city_points",
    start_year: int = 2014,
    end_year: int = 2025,
) -> Dict[str, Any]:
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else DATA_RAW
    out_dir.mkdir(parents=True, exist_ok=True)

    city_points = _city_points_frame(max_cities=int(max_cities), buffer_m=int(buffer_m))
    city_points_path = out_dir / "gee_city_points.csv"
    city_points.to_csv(city_points_path, index=False)

    viirs_script = out_dir / "gee_export_viirs_monthly.js"
    ghsl_script = out_dir / "gee_export_ghsl_yearly.js"
    no2_script = out_dir / "gee_export_no2_monthly.js"
    readme = out_dir / "gee_export_README.md"

    viirs_script.write_text(
        _viirs_script_template(asset_id=str(asset_id), start_year=int(start_year), end_year=int(end_year)),
        encoding="utf-8",
    )
    ghsl_script.write_text(_ghsl_script_template(asset_id=str(asset_id)), encoding="utf-8")
    no2_script.write_text(
        _no2_script_template(asset_id=str(asset_id), start_year=max(int(start_year), 2015), end_year=int(end_year)),
        encoding="utf-8",
    )
    readme.write_text(
        dedent(
            f"""
            # Google Earth Engine Export Bundle

            1. Upload the polygon city table (`gee_city_points.csv` or `city_catalog.csv` with `geometry_wkt`) to Earth Engine as a table asset.
            2. Update `ASSET_ID` inside the JS scripts if needed.
            3. Run `gee_export_viirs_monthly.js` in the Earth Engine Code Editor and export the CSV.
            4. Run `gee_export_ghsl_yearly.js` in the Earth Engine Code Editor and export the CSV.
            5. Run `gee_export_no2_monthly.js` in the Earth Engine Code Editor and export the CSV.
            6. Import the exported CSV files locally with:

               `python3 run_gee_bridge.py --import-viirs <viirs_csv> --import-ghsl <ghsl_csv> --import-no2 <no2_csv>`

            Current defaults:
            - city count: {int(max_cities)}
            - buffer radius: {int(buffer_m)} m
            - VIIRS range: {int(start_year)}-{int(end_year)}
            - NO2 range: {max(int(start_year), 2015)}-{int(end_year)}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "status": "ok",
        "city_points_csv": str(city_points_path),
        "viirs_script": str(viirs_script),
        "ghsl_script": str(ghsl_script),
        "no2_script": str(no2_script),
        "readme": str(readme),
        "city_count": int(len(city_points)),
        "buffer_m": int(buffer_m),
        "asset_id_template": str(asset_id),
    }
    dump_json(out_dir / "gee_export_bundle_summary.json", summary)
    return summary


def import_gee_viirs_city_monthly(
    *,
    source_path: str | Path,
    output_path: str | Path | None = None,
    merge_existing: bool = True,
    max_cities: int = 295,
) -> Dict[str, Any]:
    cities = load_city_catalog(max_cities=int(max_cities)).copy()
    raw = _attach_city_id_from_name(_read_csv_or_raise(source_path), cities)

    if "year" not in raw.columns:
        date_col = _resolve_column_alias(raw, ["date", "month_date", "timestamp"])
        if date_col:
            raw["year"] = pd.to_datetime(raw[date_col], errors="coerce").dt.year
            raw["month"] = pd.to_datetime(raw[date_col], errors="coerce").dt.month
    if "month" not in raw.columns:
        month_col = _resolve_column_alias(raw, ["month", "month_num"])
        if month_col and month_col != "month":
            raw["month"] = pd.to_numeric(raw[month_col], errors="coerce")

    rad_col = _resolve_column_alias(raw, GEE_VIIRS_RADIANCE_ALIASES)
    sum_col = _resolve_column_alias(raw, GEE_VIIRS_SUM_ALIASES)
    cov_col = _resolve_column_alias(raw, GEE_VIIRS_COVERAGE_ALIASES)
    lit_col = _resolve_column_alias(raw, GEE_VIIRS_LIT_AREA_ALIASES)
    if rad_col is None and cov_col is None:
        msg = "GEE VIIRS CSV is missing radiance/coverage columns."
        raise ValueError(msg)

    work = raw.copy()
    work["city_id"] = work["city_id"].astype(str)
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    if "month" not in work.columns:
        work["month"] = np.nan
    work["month"] = pd.to_numeric(work["month"], errors="coerce").astype("Int64")
    work = work[work["city_id"].isin(cities["city_id"].astype(str))].copy()
    work = work.dropna(subset=["city_id", "year", "month"]).copy()
    work["year"] = work["year"].astype(int)
    work["month"] = work["month"].astype(int)
    work = work[(work["month"] >= 1) & (work["month"] <= 12)].copy()
    work["radiance"] = pd.to_numeric(work[rad_col], errors="coerce") if rad_col else np.nan
    work["radiance_sum"] = pd.to_numeric(work[sum_col], errors="coerce") if sum_col else np.nan
    work["cf_cvg"] = pd.to_numeric(work[cov_col], errors="coerce") if cov_col else np.nan
    work["lit_area_km2"] = pd.to_numeric(work[lit_col], errors="coerce") if lit_col else np.nan
    source_col = _resolve_column_alias(raw, ["viirs_source", "source"])
    work["viirs_source"] = work[source_col].astype(str) if source_col else "gee_viirs_monthly_vcmslcfg"
    out = work[["city_id", "year", "month", "radiance", "radiance_sum", "cf_cvg", "lit_area_km2", "viirs_source"]].copy()

    dest = Path(output_path).expanduser().resolve() if output_path else (DATA_RAW / "viirs_city_monthly.csv")
    existing = pd.DataFrame()
    if merge_existing and dest.exists():
        try:
            existing = pd.read_csv(dest)
        except Exception:  # noqa: BLE001
            existing = pd.DataFrame()
    merged = merge_viirs_monthly_panels(existing, out)
    merged.to_csv(dest, index=False)

    summary = {
        "status": "ok",
        "source_path": str(Path(source_path).expanduser().resolve()),
        "output_path": str(dest),
        "rows_imported": int(len(out)),
        "rows_after_merge": int(len(merged)),
        "city_count": int(out["city_id"].nunique()) if not out.empty else 0,
        "year_min": int(out["year"].min()) if not out.empty else None,
        "year_max": int(out["year"].max()) if not out.empty else None,
        "source_counts": out["viirs_source"].value_counts(dropna=False).to_dict() if not out.empty else {},
    }
    dump_json(dest.parent / "gee_viirs_import_summary.json", summary)
    return summary


def import_gee_ghsl_city_yearly(
    *,
    source_path: str | Path,
    output_path: str | Path | None = None,
    merge_existing: bool = True,
    max_cities: int = 295,
) -> Dict[str, Any]:
    cities = load_city_catalog(max_cities=int(max_cities)).copy()
    raw = _attach_city_id_from_name(_read_csv_or_raise(source_path), cities)
    year_col = _resolve_column_alias(raw, ["year", "epoch", "ghsl_year"])
    if year_col is None:
        msg = "GEE GHSL CSV is missing a year/epoch column."
        raise ValueError(msg)

    work = raw.copy()
    work["city_id"] = work["city_id"].astype(str)
    work["year"] = pd.to_numeric(work[year_col], errors="coerce").astype("Int64")
    work = work[work["city_id"].isin(cities["city_id"].astype(str))].copy()
    work = work.dropna(subset=["city_id", "year"]).copy()
    work["year"] = work["year"].astype(int)

    for target, aliases in GEE_GHSL_ALIASES.items():
        col = _resolve_column_alias(work, aliases)
        work[target] = pd.to_numeric(work[col], errors="coerce") if col else np.nan
    source_col = _resolve_column_alias(work, ["ghsl_source", "source"])
    work["ghsl_source"] = work[source_col].astype(str) if source_col else "gee_ghsl_p2023a"

    out = (
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

    dest = Path(output_path).expanduser().resolve() if output_path else (DATA_RAW / "city_ghsl_yearly.csv")
    if merge_existing and dest.exists():
        try:
            existing = pd.read_csv(dest)
        except Exception:  # noqa: BLE001
            existing = pd.DataFrame()
        out = pd.concat([existing, out], ignore_index=True, sort=False)
        out = (
            out.groupby(["city_id", "year"], as_index=False)
            .agg(
                ghsl_built_surface_km2=("ghsl_built_surface_km2", "mean"),
                ghsl_built_surface_nres_km2=("ghsl_built_surface_nres_km2", "mean"),
                ghsl_built_volume_m3=("ghsl_built_volume_m3", "mean"),
                ghsl_built_volume_nres_m3=("ghsl_built_volume_nres_m3", "mean"),
                ghsl_source=("ghsl_source", "last"),
            )
            .sort_values(["city_id", "year"])
        )
    out.to_csv(dest, index=False)

    summary = {
        "status": "ok",
        "source_path": str(Path(source_path).expanduser().resolve()),
        "output_path": str(dest),
        "rows_after_merge": int(len(out)),
        "city_count": int(out["city_id"].nunique()) if not out.empty else 0,
        "year_min": int(out["year"].min()) if not out.empty else None,
        "year_max": int(out["year"].max()) if not out.empty else None,
        "source_counts": out["ghsl_source"].value_counts(dropna=False).to_dict() if not out.empty else {},
    }
    dump_json(dest.parent / "gee_ghsl_import_summary.json", summary)
    return summary


def import_gee_no2_city_monthly(
    *,
    source_path: str | Path,
    output_path: str | Path | None = None,
    merge_existing: bool = True,
    max_cities: int = 295,
) -> Dict[str, Any]:
    cities = load_city_catalog(max_cities=int(max_cities)).copy()
    raw = _attach_city_id_from_name(_read_csv_or_raise(source_path), cities)

    if "year" not in raw.columns:
        date_col = _resolve_column_alias(raw, ["date", "month_date", "timestamp"])
        if date_col:
            raw["year"] = pd.to_datetime(raw[date_col], errors="coerce").dt.year
            raw["month"] = pd.to_datetime(raw[date_col], errors="coerce").dt.month
    if "month" not in raw.columns:
        month_col = _resolve_column_alias(raw, ["month", "month_num"])
        if month_col and month_col != "month":
            raw["month"] = pd.to_numeric(raw[month_col], errors="coerce")

    mean_col = _resolve_column_alias(raw, GEE_NO2_MEAN_ALIASES)
    count_col = _resolve_column_alias(raw, GEE_NO2_COUNT_ALIASES)
    if mean_col is None:
        msg = "GEE NO2 CSV is missing an NO2 mean column."
        raise ValueError(msg)

    work = raw.copy()
    work["city_id"] = work["city_id"].astype(str)
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work["month"] = pd.to_numeric(work.get("month"), errors="coerce").astype("Int64")
    work = work[work["city_id"].isin(cities["city_id"].astype(str))].copy()
    work = work.dropna(subset=["city_id", "year", "month"]).copy()
    work["year"] = work["year"].astype(int)
    work["month"] = work["month"].astype(int)
    work = work[(work["month"] >= 1) & (work["month"] <= 12)].copy()
    work["no2_trop_mean"] = pd.to_numeric(work[mean_col], errors="coerce")
    work["no2_valid_obs_count"] = pd.to_numeric(work[count_col], errors="coerce") if count_col else np.nan
    source_col = _resolve_column_alias(raw, ["no2_source", "source"])
    work["no2_source"] = work[source_col].astype(str) if source_col else "gee_no2_monthly"
    out = work[
        ["city_id", "year", "month", "no2_trop_mean", "no2_valid_obs_count", "no2_source"]
    ].copy()

    dest = Path(output_path).expanduser().resolve() if output_path else (DATA_RAW / "city_no2_monthly.csv")
    if merge_existing and dest.exists():
        try:
            existing = pd.read_csv(dest)
        except Exception:  # noqa: BLE001
            existing = pd.DataFrame()
        out = pd.concat([existing, out], ignore_index=True, sort=False)
        out = (
            out.sort_values(["city_id", "year", "month"])
            .drop_duplicates(subset=["city_id", "year", "month"], keep="last")
            .reset_index(drop=True)
        )
    out.to_csv(dest, index=False)

    summary = {
        "status": "ok",
        "source_path": str(Path(source_path).expanduser().resolve()),
        "output_path": str(dest),
        "rows_after_merge": int(len(out)),
        "city_count": int(out["city_id"].nunique()) if not out.empty else 0,
        "year_min": int(out["year"].min()) if not out.empty else None,
        "year_max": int(out["year"].max()) if not out.empty else None,
        "source_counts": out["no2_source"].value_counts(dropna=False).to_dict() if not out.empty else {},
    }
    dump_json(dest.parent / "gee_no2_import_summary.json", summary)
    return summary
