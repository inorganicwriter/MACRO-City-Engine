from __future__ import annotations

"""Observed-evidence diagnostics for measurement credibility and innovation claims."""

import logging
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .utils import DATA_OUTPUTS, dump_json

LOGGER = logging.getLogger(__name__)

MACRO_CONTEXT_FEATURES: List[str] = [
    "gdp_per_capita",
    "population",
    "unemployment",
    "internet_users",
    "capital_formation",
    "inflation",
    "employment_rate",
    "urban_population_share",
    "electricity_access",
    "fixed_broadband_subscriptions",
    "pm25_exposure",
]

OBSERVED_STATIC_FEATURES: List[str] = [
    "latitude",
    "longitude",
    "temperature_mean",
    "precipitation_sum",
    "climate_comfort",
    "amenity_ratio",
    "commerce_ratio",
    "transport_intensity",
    "poi_diversity",
    "observed_activity_signal",
    "observed_mobility_signal",
    "observed_livability_signal",
    "observed_innovation_signal",
    "observed_physical_stress_signal",
    "road_access_score",
    "road_tier_code",
    "road_length_km_total",
    "arterial_share",
    "intersection_density",
    "viirs_ntl_mean",
    "viirs_log_mean",
    "viirs_ntl_p90",
    "viirs_lit_area_km2",
    "viirs_physical_continuity",
    "ghsl_built_surface_km2",
    "ghsl_built_volume_m3",
    "ghsl_built_density",
    "osm_hist_road_length_m",
    "osm_hist_building_count",
    "osm_hist_poi_count",
]

OBSERVED_DYNAMIC_FEATURES: List[str] = [
    "observed_dynamic_signal",
    "observed_sentiment_signal",
    "road_arterial_growth_proxy",
    "road_local_growth_proxy",
    "road_growth_intensity",
    "viirs_ntl_yoy",
    "viirs_intra_year_recovery",
    "viirs_intra_year_decline",
    "viirs_recent_drop",
    "viirs_physical_stress",
    "ghsl_built_surface_yoy",
    "ghsl_built_volume_yoy",
    "ghsl_built_contraction",
    "osm_hist_road_yoy",
    "osm_hist_building_yoy",
    "osm_hist_poi_yoy",
    "osm_hist_poi_food_yoy",
    "osm_hist_poi_retail_yoy",
    "osm_hist_poi_nightlife_yoy",
    "social_sentiment_score",
    "social_sentiment_buzz",
    "social_sentiment_delta_1",
]

POLICY_CONTEXT_FEATURES: List[str] = [
    "policy_event_count_iso_year",
    "policy_event_type_count_iso_year",
    "policy_event_new_count_iso_year",
    "policy_intensity_sum_iso_year",
    "policy_intensity_mean_iso_year",
    "policy_event_count_iso_year_yoy",
    "policy_intensity_sum_iso_year_yoy",
    "policy_news_proxy_score",
]

TARGETS: List[str] = ["economic_vitality", "livability", "innovation", "composite_index"]

CONSISTENCY_SIGNAL_COLUMNS: Dict[str, str] = {
    "road_growth_intensity": "road_growth_intensity",
    "viirs_ntl_yoy": "viirs_ntl_yoy",
    "viirs_recent_drop": "viirs_recent_drop",
    "ghsl_built_surface_yoy": "ghsl_built_surface_yoy",
    "ghsl_built_contraction": "ghsl_built_contraction",
    "osm_hist_road_yoy": "osm_hist_road_yoy",
    "osm_hist_poi_yoy": "osm_hist_poi_yoy",
    "social_sentiment_delta_1": "social_sentiment_delta_1",
}


def _metric(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _train_test_split_by_year(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(panel["year"].dropna().unique().tolist())
    if len(years) < 4:
        raise RuntimeError("Need at least 4 years for observed-evidence diagnostics.")
    split_year = years[-3] if len(years) >= 5 else years[-2]
    train = panel[panel["year"] <= split_year].copy()
    test = panel[panel["year"] > split_year].copy()
    if test.empty:
        test = panel[panel["year"] == years[-1]].copy()
    return train, test


def _resolve_feature_columns(train: pd.DataFrame, test: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    cols: List[str] = []
    for col in candidates:
        if col not in train.columns or col not in test.columns:
            continue
        s_tr = pd.to_numeric(train[col], errors="coerce")
        s_te = pd.to_numeric(test[col], errors="coerce")
        if float(s_tr.notna().mean()) < 0.55:
            continue
        if float(s_te.notna().mean()) <= 0.0:
            continue
        cols.append(col)
    return cols


def _build_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]] | None:
    tr = train.dropna(subset=[target_col]).copy()
    te = test.dropna(subset=[target_col]).copy()
    if len(tr) < 40 or len(te) < 15:
        return None

    usable = _resolve_feature_columns(tr, te, feature_cols)
    if len(usable) < 4:
        return None

    medians: Dict[str, float] = {}
    for col in usable:
        s_tr = pd.to_numeric(tr[col], errors="coerce")
        medians[col] = float(s_tr.median()) if s_tr.notna().any() else 0.0

    x_tr = np.column_stack(
        [pd.to_numeric(tr[col], errors="coerce").fillna(medians[col]).to_numpy(dtype=float) for col in usable]
    )
    x_te = np.column_stack(
        [pd.to_numeric(te[col], errors="coerce").fillna(medians[col]).to_numpy(dtype=float) for col in usable]
    )
    y_tr = pd.to_numeric(tr[target_col], errors="coerce").to_numpy(dtype=float)
    y_te = pd.to_numeric(te[target_col], errors="coerce").to_numpy(dtype=float)
    return x_tr, y_tr, x_te, y_te, usable


def _fit_group_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> List[Dict[str, Any]]:
    mats = _build_matrices(train, test, target_col, feature_cols)
    if mats is None:
        return []
    x_tr, y_tr, x_te, y_te, usable = mats

    models: Dict[str, Any] = {
        "linear": LinearRegression(),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            random_state=42,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=0.02,
        ),
    }
    rows: List[Dict[str, Any]] = []
    for name, model in models.items():
        try:
            model.fit(x_tr, y_tr)
            pred = np.asarray(model.predict(x_te), dtype=float)
            met = _metric(y_te, pred)
            rows.append(
                {
                    "model": name,
                    "feature_count": int(len(usable)),
                    "n_train": int(len(y_tr)),
                    "n_test": int(len(y_te)),
                    "rmse": float(met["rmse"]),
                    "mae": float(met["mae"]),
                    "r2": float(met["r2"]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Observed-evidence fit failed for %s/%s: %s", target_col, name, exc)
    return rows


def _has_numeric_signal(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    out = pd.Series(False, index=df.index, dtype=bool)
    for col in columns:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        out = out | vals.notna()
    return out


def _has_source_signal(
    df: pd.DataFrame,
    *,
    value_columns: Iterable[str],
    source_column: str | None = None,
    extra_presence: pd.Series | None = None,
) -> pd.Series:
    mask = _has_numeric_signal(df, value_columns)
    if extra_presence is not None:
        mask = mask | extra_presence.fillna(False).astype(bool)
    if source_column and source_column in df.columns:
        src = df[source_column].fillna("missing").astype(str).str.strip().str.lower()
        mask = mask & (~src.isin({"", "missing", "unavailable", "none"}))
    return mask.astype(int)


def _build_measurement_audit(panel: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    audit = panel[
        [c for c in ["city_id", "city_name", "country", "continent", "iso3", "year", "index_spec_version", "macro_resolution_level"] if c in panel.columns]
    ].copy()
    audit["road_present"] = _has_source_signal(
        panel,
        value_columns=["road_length_km_total", "arterial_share", "intersection_density"],
        source_column="road_source",
    )
    audit["viirs_present"] = _has_source_signal(
        panel,
        value_columns=[
            "viirs_ntl_mean",
            "viirs_ntl_p90",
            "viirs_lit_area_km2",
            "viirs_recent_drop",
            "viirs_intra_year_decline",
            "viirs_intra_year_recovery",
        ],
        source_column="viirs_source",
    )
    audit["osm_history_present"] = _has_source_signal(
        panel,
        value_columns=[
            "osm_hist_road_length_m",
            "osm_hist_building_count",
            "osm_hist_poi_count",
            "osm_hist_road_yoy",
            "osm_hist_poi_yoy",
        ],
        source_column="osm_hist_source",
    )
    social_presence = (
        pd.to_numeric(panel.get("social_sentiment_volume"), errors="coerce").fillna(0.0) > 0.0
        if "social_sentiment_volume" in panel.columns
        else pd.Series(False, index=panel.index, dtype=bool)
    )
    audit["social_present"] = _has_source_signal(
        panel,
        value_columns=["social_sentiment_score", "social_sentiment_delta_1"],
        source_column="social_sentiment_source",
        extra_presence=social_presence,
    )
    audit["poi_present"] = _has_source_signal(
        panel,
        value_columns=["poi_total", "poi_diversity", "commerce_ratio"],
        source_column="poi_source",
    )
    audit["weather_present"] = _has_source_signal(
        panel,
        value_columns=["temperature_mean", "precipitation_sum", "climate_comfort"],
        source_column="weather_source",
    )
    audit["city_macro_present"] = (
        pd.to_numeric(panel.get("city_macro_observed_flag"), errors="coerce").fillna(0).astype(int)
        if "city_macro_observed_flag" in panel.columns
        else 0
    )
    direct_cols = [
        "road_present",
        "viirs_present",
        "osm_history_present",
        "social_present",
        "poi_present",
        "weather_present",
        "city_macro_present",
    ]
    dynamic_cols = ["road_present", "viirs_present", "osm_history_present", "social_present"]
    audit["observed_channel_count"] = audit[direct_cols].sum(axis=1).astype(int)
    audit["observed_dynamic_channel_count"] = audit[dynamic_cols].sum(axis=1).astype(int)
    audit["observed_channel_coverage_ratio"] = audit["observed_channel_count"] / float(len(direct_cols))
    audit["observed_dynamic_coverage_ratio"] = audit["observed_dynamic_channel_count"] / float(len(dynamic_cols))
    audit["target_defined_by_observed_channels"] = (
        audit.get("index_spec_version", pd.Series("", index=audit.index))
        .astype(str)
        .str.contains("city_observed_primary", regex=False)
        .astype(int)
    )
    audit["macro_context_only_row"] = (
        (audit.get("macro_resolution_level", pd.Series("", index=audit.index)).astype(str) == "country_year")
        & (audit["city_macro_present"] <= 0)
    ).astype(int)

    tier = np.where(
        audit["observed_channel_count"] >= 5,
        "high_observed_coverage",
        np.where(audit["observed_channel_count"] >= 3, "moderate_observed_coverage", "limited_observed_coverage"),
    )
    audit["measurement_tier"] = pd.Series(tier, index=audit.index, dtype=object)

    out_csv = DATA_OUTPUTS / "observed_measurement_audit.csv"
    audit.to_csv(out_csv, index=False)

    summary = {
        "status": "ok",
        "rows": int(len(audit)),
        "cities": int(audit["city_id"].nunique()) if "city_id" in audit.columns else 0,
        "year_range": [int(audit["year"].min()), int(audit["year"].max())] if "year" in audit.columns and not audit.empty else None,
        "observed_primary_spec_share": float(audit["target_defined_by_observed_channels"].mean()) if not audit.empty else 0.0,
        "macro_context_only_row_ratio": float(audit["macro_context_only_row"].mean()) if not audit.empty else 0.0,
        "city_macro_observed_row_ratio": float(audit["city_macro_present"].mean()) if not audit.empty else 0.0,
        "mean_observed_channel_count": float(audit["observed_channel_count"].mean()) if not audit.empty else 0.0,
        "share_rows_observed_channels_ge_3": float((audit["observed_channel_count"] >= 3).mean()) if not audit.empty else 0.0,
        "share_rows_dynamic_channels_ge_2": float((audit["observed_dynamic_channel_count"] >= 2).mean()) if not audit.empty else 0.0,
        "channel_coverage": {
            "road": float(audit["road_present"].mean()) if not audit.empty else 0.0,
            "viirs": float(audit["viirs_present"].mean()) if not audit.empty else 0.0,
            "osm_history": float(audit["osm_history_present"].mean()) if not audit.empty else 0.0,
            "social": float(audit["social_present"].mean()) if not audit.empty else 0.0,
            "poi": float(audit["poi_present"].mean()) if not audit.empty else 0.0,
            "weather": float(audit["weather_present"].mean()) if not audit.empty else 0.0,
            "city_macro": float(audit["city_macro_present"].mean()) if not audit.empty else 0.0,
        },
        "measurement_tier_counts": audit["measurement_tier"].value_counts(dropna=False).to_dict() if not audit.empty else {},
        "output_file": str(out_csv),
        "note": "Country-year macro variables act as contextual controls, while the index specification is audited separately for direct city-observed signal dependence.",
    }
    dump_json(DATA_OUTPUTS / "observed_measurement_summary.json", summary)
    return audit, summary


def _best_rows_by_group(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    best = (
        detail.sort_values(["target", "feature_group", "rmse", "r2"], ascending=[True, True, True, False])
        .drop_duplicates(["target", "feature_group"], keep="first")
        .reset_index(drop=True)
    )
    return best


def _run_feature_group_ablation(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df = panel.sort_values(["city_id", "year"]).copy()
    for target in TARGETS:
        df[f"{target}_t1"] = df.groupby("city_id")[target].shift(-1)
    train, test = _train_test_split_by_year(df)

    feature_groups: Dict[str, List[str]] = {
        "macro_context_only": MACRO_CONTEXT_FEATURES,
        "observed_core": list(dict.fromkeys(OBSERVED_STATIC_FEATURES + OBSERVED_DYNAMIC_FEATURES)),
        "observed_plus_policy": list(dict.fromkeys(OBSERVED_STATIC_FEATURES + OBSERVED_DYNAMIC_FEATURES + POLICY_CONTEXT_FEATURES)),
        "mixed_full": list(
            dict.fromkeys(MACRO_CONTEXT_FEATURES + OBSERVED_STATIC_FEATURES + OBSERVED_DYNAMIC_FEATURES + POLICY_CONTEXT_FEATURES)
        ),
    }

    rows: List[Dict[str, Any]] = []
    for target in TARGETS:
        target_col = f"{target}_t1"
        for group_name, features in feature_groups.items():
            fit_rows = _fit_group_models(train, test, target_col=target_col, feature_cols=features)
            for row in fit_rows:
                rows.append({"target": target, "feature_group": group_name, **row})

    detail = pd.DataFrame(rows)
    detail_path = DATA_OUTPUTS / "observed_feature_group_ablation.csv"
    detail.to_csv(detail_path, index=False)

    best = _best_rows_by_group(detail)
    if best.empty:
        summary = {
            "status": "skipped",
            "reason": "no_valid_feature_group_runs",
            "detail_file": str(detail_path),
        }
        dump_json(DATA_OUTPUTS / "observed_feature_group_summary.json", summary)
        return detail, pd.DataFrame(), summary

    summary_rows: List[Dict[str, Any]] = []
    for target in TARGETS:
        sub = best[best["target"] == target].copy()
        if sub.empty:
            continue
        row: Dict[str, Any] = {"target": target}
        for group_name in feature_groups:
            grp = sub[sub["feature_group"] == group_name]
            if grp.empty:
                row[f"{group_name}_rmse"] = np.nan
                row[f"{group_name}_r2"] = np.nan
                row[f"{group_name}_feature_count"] = np.nan
                continue
            rec = grp.iloc[0]
            row[f"{group_name}_rmse"] = float(rec["rmse"])
            row[f"{group_name}_r2"] = float(rec["r2"])
            row[f"{group_name}_feature_count"] = int(rec["feature_count"])
        macro_rmse = pd.to_numeric(pd.Series([row.get("macro_context_only_rmse")]), errors="coerce").iloc[0]
        observed_rmse = pd.to_numeric(pd.Series([row.get("observed_core_rmse")]), errors="coerce").iloc[0]
        policy_rmse = pd.to_numeric(pd.Series([row.get("observed_plus_policy_rmse")]), errors="coerce").iloc[0]
        full_rmse = pd.to_numeric(pd.Series([row.get("mixed_full_rmse")]), errors="coerce").iloc[0]
        macro_r2 = pd.to_numeric(pd.Series([row.get("macro_context_only_r2")]), errors="coerce").iloc[0]
        observed_r2 = pd.to_numeric(pd.Series([row.get("observed_core_r2")]), errors="coerce").iloc[0]
        policy_r2 = pd.to_numeric(pd.Series([row.get("observed_plus_policy_r2")]), errors="coerce").iloc[0]
        full_r2 = pd.to_numeric(pd.Series([row.get("mixed_full_r2")]), errors="coerce").iloc[0]
        row["delta_rmse_observed_vs_macro"] = float(observed_rmse - macro_rmse) if np.isfinite(observed_rmse) and np.isfinite(macro_rmse) else np.nan
        row["delta_rmse_policy_vs_observed"] = float(policy_rmse - observed_rmse) if np.isfinite(policy_rmse) and np.isfinite(observed_rmse) else np.nan
        row["delta_rmse_full_vs_observed"] = float(full_rmse - observed_rmse) if np.isfinite(full_rmse) and np.isfinite(observed_rmse) else np.nan
        row["delta_r2_observed_vs_macro"] = float(observed_r2 - macro_r2) if np.isfinite(observed_r2) and np.isfinite(macro_r2) else np.nan
        row["delta_r2_policy_vs_observed"] = float(policy_r2 - observed_r2) if np.isfinite(policy_r2) and np.isfinite(observed_r2) else np.nan
        row["delta_r2_full_vs_observed"] = float(full_r2 - observed_r2) if np.isfinite(full_r2) and np.isfinite(observed_r2) else np.nan
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = DATA_OUTPUTS / "observed_feature_group_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    summary = {
        "status": "ok",
        "targets_evaluated": int(summary_df["target"].nunique()) if not summary_df.empty else 0,
        "detail_file": str(detail_path),
        "summary_file": str(summary_path),
        "mean_delta_rmse_observed_vs_macro": float(pd.to_numeric(summary_df["delta_rmse_observed_vs_macro"], errors="coerce").mean())
        if ("delta_rmse_observed_vs_macro" in summary_df.columns and pd.to_numeric(summary_df["delta_rmse_observed_vs_macro"], errors="coerce").notna().any())
        else None,
        "mean_delta_rmse_policy_vs_observed": float(pd.to_numeric(summary_df["delta_rmse_policy_vs_observed"], errors="coerce").mean())
        if ("delta_rmse_policy_vs_observed" in summary_df.columns and pd.to_numeric(summary_df["delta_rmse_policy_vs_observed"], errors="coerce").notna().any())
        else None,
        "mean_delta_rmse_full_vs_observed": float(pd.to_numeric(summary_df["delta_rmse_full_vs_observed"], errors="coerce").mean())
        if ("delta_rmse_full_vs_observed" in summary_df.columns and pd.to_numeric(summary_df["delta_rmse_full_vs_observed"], errors="coerce").notna().any())
        else None,
        "mean_delta_r2_observed_vs_macro": float(pd.to_numeric(summary_df["delta_r2_observed_vs_macro"], errors="coerce").mean())
        if ("delta_r2_observed_vs_macro" in summary_df.columns and pd.to_numeric(summary_df["delta_r2_observed_vs_macro"], errors="coerce").notna().any())
        else None,
        "mean_delta_r2_policy_vs_observed": float(pd.to_numeric(summary_df["delta_r2_policy_vs_observed"], errors="coerce").mean())
        if ("delta_r2_policy_vs_observed" in summary_df.columns and pd.to_numeric(summary_df["delta_r2_policy_vs_observed"], errors="coerce").notna().any())
        else None,
        "mean_delta_r2_full_vs_observed": float(pd.to_numeric(summary_df["delta_r2_full_vs_observed"], errors="coerce").mean())
        if ("delta_r2_full_vs_observed" in summary_df.columns and pd.to_numeric(summary_df["delta_r2_full_vs_observed"], errors="coerce").notna().any())
        else None,
        "summary": summary_df.to_dict(orient="records"),
    }
    dump_json(DATA_OUTPUTS / "observed_feature_group_summary.json", summary)
    return detail, summary_df, summary


def _zscore_within_year(values: pd.Series, years: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")

    def _one_year(x: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce")
        if x.notna().sum() < 2:
            return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
        std = float(x.std(ddof=0))
        if (not np.isfinite(std)) or std <= 1e-12:
            return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
        return ((x - float(x.mean())) / std).astype(float)

    return vals.groupby(years).transform(_one_year)


def _pairwise_sign_agreement(arr: np.ndarray) -> float:
    valid = arr[np.isfinite(arr)]
    if valid.size < 2:
        return float("nan")
    signs = np.sign(valid)
    pairs = 0
    agree = 0
    for i in range(len(signs)):
        for j in range(i + 1, len(signs)):
            pairs += 1
            if signs[i] == signs[j]:
                agree += 1
    if pairs == 0:
        return float("nan")
    return float(agree / pairs)


def _signal_coverage_mask(df: pd.DataFrame, signal_name: str, column: str) -> pd.Series:
    mask = pd.to_numeric(df.get(column), errors="coerce").notna()
    if signal_name.startswith("road_") and "road_source" in df.columns:
        src = df["road_source"].fillna("missing").astype(str).str.strip().str.lower()
        mask = mask & (~src.isin({"", "missing", "unavailable", "none"}))
    elif signal_name.startswith("viirs_") and "viirs_source" in df.columns:
        src = df["viirs_source"].fillna("missing").astype(str).str.strip().str.lower()
        mask = mask & (~src.isin({"", "missing", "unavailable", "none"}))
    elif signal_name.startswith("osm_hist_") and "osm_hist_source" in df.columns:
        src = df["osm_hist_source"].fillna("missing").astype(str).str.strip().str.lower()
        mask = mask & (~src.isin({"", "missing", "unavailable", "none"}))
    elif signal_name.startswith("social_"):
        src = (
            df["social_sentiment_source"].fillna("missing").astype(str).str.strip().str.lower()
            if "social_sentiment_source" in df.columns
            else pd.Series("missing", index=df.index, dtype=object)
        )
        volume = (
            pd.to_numeric(df["social_sentiment_volume"], errors="coerce").fillna(0.0)
            if "social_sentiment_volume" in df.columns
            else pd.Series(0.0, index=df.index, dtype=float)
        )
        mask = mask & (~src.isin({"", "missing", "unavailable", "none"})) & (volume > 0.0)
    return mask.astype(bool)


def _run_cross_source_consistency(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df = panel.sort_values(["city_id", "year"]).copy()
    base_cols = [c for c in ["city_id", "city_name", "country", "continent", "year", "composite_index"] if c in df.columns]
    out = df[base_cols].copy()
    available_signals: List[str] = []
    for name, col in CONSISTENCY_SIGNAL_COLUMNS.items():
        if col not in df.columns:
            continue
        coverage = _signal_coverage_mask(df, name, col)
        if float(coverage.mean()) < 0.05:
            continue
        s = pd.to_numeric(df[col], errors="coerce").where(coverage)
        valid = s.dropna()
        if valid.empty:
            continue
        std = float(valid.std(ddof=0)) if len(valid) > 1 else 0.0
        if (not np.isfinite(std)) or std <= 1e-12:
            continue
        out[f"{name}_z"] = _zscore_within_year(s, df["year"]).where(coverage)
        available_signals.append(name)

    if not available_signals:
        empty_city = pd.DataFrame(columns=base_cols + ["available_signal_count", "consistency_score"])
        empty_pair = pd.DataFrame(columns=["signal_a", "signal_b", "n_overlap", "corr", "sign_agreement"])
        summary = {"status": "skipped", "reason": "no_dynamic_signals_available"}
        empty_city.to_csv(DATA_OUTPUTS / "observed_cross_source_city_year.csv", index=False)
        empty_pair.to_csv(DATA_OUTPUTS / "observed_cross_source_pairs.csv", index=False)
        dump_json(DATA_OUTPUTS / "observed_cross_source_summary.json", summary)
        return empty_city, empty_pair, summary

    z_cols = [f"{name}_z" for name in available_signals]
    z_mat = out[z_cols].to_numpy(dtype=float)
    out["available_signal_count"] = np.sum(np.isfinite(z_mat), axis=1).astype(int)
    out["observed_net_signal"] = np.nanmean(z_mat, axis=1)
    out.loc[out["available_signal_count"] <= 0, "observed_net_signal"] = np.nan
    out["observed_signal_dispersion"] = np.nanstd(z_mat, axis=1)
    out.loc[out["available_signal_count"] <= 1, "observed_signal_dispersion"] = np.nan
    out["sign_agreement_share"] = [
        _pairwise_sign_agreement(np.asarray(z_mat[i, :], dtype=float)) for i in range(z_mat.shape[0])
    ]
    out["consistency_score"] = (
        pd.to_numeric(out["sign_agreement_share"], errors="coerce").fillna(0.0)
        / (1.0 + pd.to_numeric(out["observed_signal_dispersion"], errors="coerce").fillna(0.0))
    )
    out["consistency_band"] = np.where(
        out["consistency_score"] >= 0.55,
        "high",
        np.where(out["consistency_score"] >= 0.30, "moderate", "low"),
    )

    city_year_path = DATA_OUTPUTS / "observed_cross_source_city_year.csv"
    out.to_csv(city_year_path, index=False)

    pair_rows: List[Dict[str, Any]] = []
    for idx, a in enumerate(available_signals):
        for b in available_signals[idx + 1 :]:
            s_a = pd.to_numeric(df[CONSISTENCY_SIGNAL_COLUMNS[a]], errors="coerce")
            s_b = pd.to_numeric(df[CONSISTENCY_SIGNAL_COLUMNS[b]], errors="coerce")
            mask = _signal_coverage_mask(df, a, CONSISTENCY_SIGNAL_COLUMNS[a]) & _signal_coverage_mask(df, b, CONSISTENCY_SIGNAL_COLUMNS[b])
            mask = mask & s_a.notna() & s_b.notna()
            n_overlap = int(mask.sum())
            if n_overlap <= 20:
                continue
            std_a = float(s_a[mask].std(ddof=0)) if n_overlap > 1 else 0.0
            std_b = float(s_b[mask].std(ddof=0)) if n_overlap > 1 else 0.0
            corr = float(s_a[mask].corr(s_b[mask])) if (n_overlap > 1 and std_a > 1e-12 and std_b > 1e-12) else float("nan")
            sign_agree = float(np.mean(np.sign(s_a[mask]) == np.sign(s_b[mask]))) if n_overlap > 0 else float("nan")
            pair_rows.append(
                {
                    "signal_a": a,
                    "signal_b": b,
                    "n_overlap": n_overlap,
                    "corr": corr,
                    "sign_agreement": sign_agree,
                }
            )
    pair_df = pd.DataFrame(pair_rows)
    pair_path = DATA_OUTPUTS / "observed_cross_source_pairs.csv"
    pair_df.to_csv(pair_path, index=False)

    summary = {
        "status": "ok",
        "signals_used": available_signals,
        "city_year_file": str(city_year_path),
        "pair_file": str(pair_path),
        "share_rows_with_2plus_signals": float((out["available_signal_count"] >= 2).mean()) if not out.empty else 0.0,
        "mean_consistency_score": float(pd.to_numeric(out["consistency_score"], errors="coerce").mean()) if not out.empty else 0.0,
        "median_sign_agreement_share": float(pd.to_numeric(out["sign_agreement_share"], errors="coerce").median())
        if pd.to_numeric(out["sign_agreement_share"], errors="coerce").notna().any()
        else None,
        "high_consistency_row_share": float(
            ((pd.to_numeric(out["consistency_score"], errors="coerce") >= 0.55) & (out["available_signal_count"] >= 2)).mean()
        )
        if not out.empty
        else 0.0,
        "pairwise_summary": pair_df.to_dict(orient="records"),
    }
    dump_json(DATA_OUTPUTS / "observed_cross_source_summary.json", summary)
    return out, pair_df, summary


def run_observed_evidence_suite(panel: pd.DataFrame) -> Dict[str, Any]:
    """Generate auditable outputs that emphasize direct city-observed evidence."""
    audit_df, audit_summary = _build_measurement_audit(panel)
    ablation_detail, ablation_summary_df, ablation_summary = _run_feature_group_ablation(panel)
    consistency_df, consistency_pairs, consistency_summary = _run_cross_source_consistency(panel)

    summary = {
        "status": "ok",
        "measurement_audit": audit_summary,
        "feature_group_ablation": ablation_summary,
        "cross_source_consistency": consistency_summary,
        "outputs": {
            "measurement_audit_csv": str(DATA_OUTPUTS / "observed_measurement_audit.csv"),
            "measurement_audit_json": str(DATA_OUTPUTS / "observed_measurement_summary.json"),
            "feature_group_ablation_csv": str(DATA_OUTPUTS / "observed_feature_group_ablation.csv"),
            "feature_group_summary_csv": str(DATA_OUTPUTS / "observed_feature_group_summary.csv"),
            "feature_group_summary_json": str(DATA_OUTPUTS / "observed_feature_group_summary.json"),
            "cross_source_city_year_csv": str(DATA_OUTPUTS / "observed_cross_source_city_year.csv"),
            "cross_source_pairs_csv": str(DATA_OUTPUTS / "observed_cross_source_pairs.csv"),
            "cross_source_summary_json": str(DATA_OUTPUTS / "observed_cross_source_summary.json"),
        },
        "row_counts": {
            "measurement_audit": int(len(audit_df)),
            "feature_group_ablation": int(len(ablation_detail)),
            "feature_group_summary": int(len(ablation_summary_df)),
            "cross_source_city_year": int(len(consistency_df)),
            "cross_source_pairs": int(len(consistency_pairs)),
        },
    }
    dump_json(DATA_OUTPUTS / "observed_evidence_summary.json", summary)
    return summary


__all__ = ["run_observed_evidence_suite"]
