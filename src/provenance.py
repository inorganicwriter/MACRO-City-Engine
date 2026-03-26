from __future__ import annotations

"""Data provenance audit and strict-source filtering."""

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from .utils import DATA_PROCESSED, dump_json


VERIFIED_WEATHER_SOURCES = {"open-meteo", "nasa-power"}
OBJECTIVE_WEATHER_SOURCES = VERIFIED_WEATHER_SOURCES | {"imputed_from_weather_pool"}


@dataclass(frozen=True)
class ProvenancePolicy:
    macro_source: str = "world_bank"
    extra_wb_source: str = "world_bank"
    poi_sources: tuple[str, ...] = ("osm", "imputed_from_osm_pool")
    verified_poi_sources: tuple[str, ...] = ("osm",)
    require_city_complete: bool = True


def _row_objective_mask(panel: pd.DataFrame, policy: ProvenancePolicy) -> pd.Series:
    poi_ok = panel["poi_source"].isin(set(policy.poi_sources))
    extra_ok = panel["extra_wb_source"] == policy.extra_wb_source if "extra_wb_source" in panel.columns else True
    return (
        (panel["macro_source"] == policy.macro_source)
        & extra_ok
        & (panel["weather_source"].isin(OBJECTIVE_WEATHER_SOURCES))
        & poi_ok
    )


def _row_verified_mask(panel: pd.DataFrame, policy: ProvenancePolicy) -> pd.Series:
    poi_ok = panel["poi_source"].isin(set(policy.verified_poi_sources))
    extra_ok = panel["extra_wb_source"] == policy.extra_wb_source if "extra_wb_source" in panel.columns else True
    return (
        (panel["macro_source"] == policy.macro_source)
        & extra_ok
        & (panel["weather_source"].isin(VERIFIED_WEATHER_SOURCES))
        & poi_ok
    )


def audit_and_filter_objective_sources(
    panel: pd.DataFrame,
    *,
    strict_mode: bool,
    enforce_verified: bool = True,
    policy: ProvenancePolicy | None = None,
    min_verified_city_retention_for_verified_filter: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Audit source composition and optionally filter to strict objective rows."""
    p = policy or ProvenancePolicy()
    df = panel.copy()
    df["objective_row"] = _row_objective_mask(df, p).astype(int)
    df["verified_row"] = _row_verified_mask(df, p).astype(int)

    city_stats = (
        df.groupby("city_id", as_index=False)
        .agg(
            years=("year", "nunique"),
            objective_rows=("objective_row", "sum"),
            objective_ratio=("objective_row", "mean"),
            verified_rows=("verified_row", "sum"),
            verified_ratio=("verified_row", "mean"),
            continent=("continent", "first"),
            country=("country", "first"),
            city_name=("city_name", "first"),
        )
        .sort_values(["verified_ratio", "objective_ratio", "city_id"], ascending=[False, False, True])
    )
    city_stats["is_objective_complete_city"] = (
        (city_stats["objective_rows"] >= city_stats["years"]).astype(int)
    )
    city_stats["is_verified_complete_city"] = (
        (city_stats["verified_rows"] >= city_stats["years"]).astype(int)
    )

    filter_basis = "none"
    if strict_mode:
        if enforce_verified:
            if p.require_city_complete:
                keep_ids_verified = set(city_stats.loc[city_stats["is_verified_complete_city"] == 1, "city_id"].tolist())
                verified_retention = float(len(keep_ids_verified) / max(city_stats["city_id"].nunique(), 1))
                threshold = float(max(0.0, min_verified_city_retention_for_verified_filter))
                if verified_retention + 1e-12 >= threshold:
                    filtered = df[df["city_id"].isin(keep_ids_verified)].copy()
                    filter_basis = "verified_city_complete"
                else:
                    keep_ids_objective = set(
                        city_stats.loc[city_stats["is_objective_complete_city"] == 1, "city_id"].tolist()
                    )
                    filtered = df[df["city_id"].isin(keep_ids_objective)].copy()
                    filter_basis = "objective_city_complete_fallback"
            else:
                verified_row_ratio = float(df["verified_row"].mean()) if len(df) else 0.0
                threshold = float(max(0.0, min_verified_city_retention_for_verified_filter))
                if verified_row_ratio + 1e-12 >= threshold:
                    filtered = df[df["verified_row"] == 1].copy()
                    filter_basis = "verified_row"
                else:
                    filtered = df[df["objective_row"] == 1].copy()
                    filter_basis = "objective_row_fallback"
        else:
            if p.require_city_complete:
                keep_ids = set(city_stats.loc[city_stats["is_objective_complete_city"] == 1, "city_id"].tolist())
                filtered = df[df["city_id"].isin(keep_ids)].copy()
                filter_basis = "objective_city_complete"
            else:
                filtered = df[df["objective_row"] == 1].copy()
                filter_basis = "objective_row"
    else:
        filtered = df.copy()

    combo_cols = ["macro_source", "weather_source", "poi_source"]
    if "extra_wb_source" in df.columns:
        combo_cols.insert(1, "extra_wb_source")
    source_combo = df.groupby(combo_cols, dropna=False).size().reset_index(name="rows").sort_values("rows", ascending=False)
    source_combo["macro_source"] = source_combo["macro_source"].fillna("missing")
    if "extra_wb_source" in source_combo.columns:
        source_combo["extra_wb_source"] = source_combo["extra_wb_source"].fillna("missing")
    source_combo["weather_source"] = source_combo["weather_source"].fillna("missing")
    source_combo["poi_source"] = source_combo["poi_source"].fillna("missing")

    audit = {
        "strict_mode": bool(strict_mode),
        "enforce_verified": bool(enforce_verified),
        "filter_basis": filter_basis,
        "policy": {
            "macro_source": p.macro_source,
            "extra_wb_source": p.extra_wb_source,
            "weather_sources_objective": sorted(OBJECTIVE_WEATHER_SOURCES),
            "weather_sources_verified": sorted(VERIFIED_WEATHER_SOURCES),
            "poi_sources": list(p.poi_sources),
            "verified_poi_sources": list(p.verified_poi_sources),
            "require_city_complete": bool(p.require_city_complete),
            "min_verified_city_retention_for_verified_filter": float(
                min_verified_city_retention_for_verified_filter
            ),
        },
        "before": {
            "rows": int(len(df)),
            "cities": int(df["city_id"].nunique()),
        },
        "after": {
            "rows": int(len(filtered)),
            "cities": int(filtered["city_id"].nunique()),
        },
        "dropped": {
            "rows": int(len(df) - len(filtered)),
            "cities": int(df["city_id"].nunique() - filtered["city_id"].nunique()),
        },
        "objective_row_ratio": float(df["objective_row"].mean()) if len(df) else 0.0,
        "objective_complete_city_ratio": float(city_stats["is_objective_complete_city"].mean())
        if len(city_stats)
        else 0.0,
        "verified_row_ratio": float(df["verified_row"].mean()) if len(df) else 0.0,
        "verified_complete_city_ratio": float(city_stats["is_verified_complete_city"].mean())
        if len(city_stats)
        else 0.0,
        "city_retention_ratio": (
            float(filtered["city_id"].nunique() / max(df["city_id"].nunique(), 1))
            if len(df)
            else 0.0
        ),
        "after_verified_row_ratio": float(filtered["verified_row"].mean()) if len(filtered) else 0.0,
    }

    city_stats.to_csv(DATA_PROCESSED / "source_audit_city.csv", index=False)
    source_combo.to_csv(DATA_PROCESSED / "source_audit_combo.csv", index=False)
    dump_json(DATA_PROCESSED / "source_audit_summary.json", audit)

    filtered = filtered.drop(columns=["objective_row", "verified_row"], errors="ignore")
    return filtered, audit


def build_global_coverage_report(
    panel: pd.DataFrame,
    *,
    strict_mode: bool,
    policy: ProvenancePolicy | None = None,
) -> Dict[str, object]:
    """Build country/continent/year coverage diagnostics for transparency."""
    p = policy or ProvenancePolicy()
    if panel.empty:
        out = {"status": "skipped", "reason": "empty_panel", "strict_mode": bool(strict_mode)}
        dump_json(DATA_PROCESSED / "coverage_summary.json", out)
        pd.DataFrame().to_csv(DATA_PROCESSED / "coverage_by_country.csv", index=False)
        pd.DataFrame().to_csv(DATA_PROCESSED / "coverage_by_continent.csv", index=False)
        return out

    df = panel.copy()
    if {"macro_source", "weather_source", "poi_source"}.issubset(df.columns):
        df["objective_row"] = _row_objective_mask(df, p).astype(int)
        df["verified_row"] = _row_verified_mask(df, p).astype(int)
    else:
        df["objective_row"] = 0
        df["verified_row"] = 0

    years = sorted(df["year"].dropna().astype(int).unique().tolist())
    n_years = int(len(years))
    year_min = int(min(years)) if years else None
    year_max = int(max(years)) if years else None

    by_country = (
        df.groupby(["country", "continent"], as_index=False)
        .agg(
            city_count=("city_id", "nunique"),
            observed_rows=("year", "size"),
            objective_rows=("objective_row", "sum"),
            verified_rows=("verified_row", "sum"),
            years_observed=("year", "nunique"),
        )
        .sort_values(["continent", "country"], ascending=[True, True])
    )
    by_country["expected_rows"] = by_country["city_count"] * max(1, n_years)
    by_country["row_coverage"] = by_country["observed_rows"] / by_country["expected_rows"].replace(0, 1)
    by_country["objective_row_ratio"] = by_country["objective_rows"] / by_country["observed_rows"].replace(0, 1)
    by_country["verified_row_ratio"] = by_country["verified_rows"] / by_country["observed_rows"].replace(0, 1)
    by_country["year_coverage"] = by_country["years_observed"] / max(1, n_years)
    by_country = by_country.sort_values(
        ["row_coverage", "verified_row_ratio", "objective_row_ratio", "country"],
        ascending=[True, True, True, True],
    )

    by_continent = (
        df.groupby("continent", as_index=False)
        .agg(
            country_count=("country", "nunique"),
            city_count=("city_id", "nunique"),
            observed_rows=("year", "size"),
            objective_rows=("objective_row", "sum"),
            verified_rows=("verified_row", "sum"),
            years_observed=("year", "nunique"),
        )
        .sort_values("continent")
    )
    by_continent["expected_rows"] = by_continent["city_count"] * max(1, n_years)
    by_continent["row_coverage"] = by_continent["observed_rows"] / by_continent["expected_rows"].replace(0, 1)
    by_continent["objective_row_ratio"] = by_continent["objective_rows"] / by_continent["observed_rows"].replace(0, 1)
    by_continent["verified_row_ratio"] = by_continent["verified_rows"] / by_continent["observed_rows"].replace(0, 1)
    by_continent["year_coverage"] = by_continent["years_observed"] / max(1, n_years)

    by_country.to_csv(DATA_PROCESSED / "coverage_by_country.csv", index=False)
    by_continent.to_csv(DATA_PROCESSED / "coverage_by_continent.csv", index=False)

    low_cov = (
        by_country.sort_values(["row_coverage", "verified_row_ratio", "objective_row_ratio"])
        .head(15)[["country", "continent", "row_coverage", "verified_row_ratio", "objective_row_ratio", "city_count"]]
        .to_dict(orient="records")
    )
    summary = {
        "status": "ok",
        "strict_mode": bool(strict_mode),
        "year_range": [year_min, year_max],
        "n_years": n_years,
        "n_rows": int(len(df)),
        "n_cities": int(df["city_id"].nunique()),
        "n_countries": int(df["country"].nunique()),
        "n_continents": int(df["continent"].nunique()),
        "mean_country_row_coverage": float(by_country["row_coverage"].mean()) if not by_country.empty else None,
        "mean_country_objective_ratio": float(by_country["objective_row_ratio"].mean()) if not by_country.empty else None,
        "mean_country_verified_ratio": float(by_country["verified_row_ratio"].mean()) if not by_country.empty else None,
        "mean_continent_row_coverage": float(by_continent["row_coverage"].mean()) if not by_continent.empty else None,
        "mean_continent_objective_ratio": float(by_continent["objective_row_ratio"].mean())
        if not by_continent.empty
        else None,
        "mean_continent_verified_ratio": float(by_continent["verified_row_ratio"].mean())
        if not by_continent.empty
        else None,
        "lowest_coverage_countries": low_cov,
    }
    dump_json(DATA_PROCESSED / "coverage_summary.json", summary)
    return summary
