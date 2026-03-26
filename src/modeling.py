from __future__ import annotations

"""Predictive modeling for urban multi-dimensional indices."""

import logging
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .feature_backfill import add_no2_backcast_features
from .utils import DATA_OUTPUTS, MODELS_DIR, dump_json

LOGGER = logging.getLogger(__name__)

OOT_TRAIN_END_YEAR = 2022
OOT_TEST_START_YEAR = 2023

TARGETS = ["economic_vitality", "livability", "innovation", "composite_index"]
FEATURES = [
    # NOTE: 'year' excluded — it leaks temporal information in t->t+1 holdout.
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
    "observed_dynamic_signal",
    "observed_livability_signal",
    "observed_innovation_signal",
    "observed_physical_stress_signal",
    "gravity_access_viirs",
    "gravity_access_knowledge",
    "gravity_access_population",
    "spatial_lag_log_viirs_ntl_wdist",
    "spatial_lag_log_viirs_ntl_wecon",
    "spatial_lag_knowledge_wdist",
    "road_access_score",
    "road_tier_code",
    "road_length_km_total",
    "arterial_share",
    "intersection_density",
    "road_arterial_growth_proxy",
    "road_local_growth_proxy",
    "road_growth_intensity",
    "has_poi_observation",
    "poi_backcast_scale",
    "viirs_ntl_mean",
    "viirs_log_mean",
    "viirs_ntl_p90",
    "viirs_intra_year_recovery",
    "viirs_intra_year_decline",
    "viirs_recent_drop",
    "viirs_physical_continuity",
    "viirs_physical_stress",
    "viirs_ntl_yoy",
    "viirs_lit_area_km2",
    "has_viirs_observation",
    "viirs_year_coverage_share",
    "no2_trop_mean_filled",
    "no2_trop_p90_filled",
    "no2_trop_yoy_mean_filled",
    "no2_trop_anomaly_mean_filled",
    "no2_trop_anomaly_abs_mean_filled",
    "no2_recent_spike_filled",
    "no2_year_coverage_share",
    "has_no2_observation",
    "has_no2_observation_or_backcast",
    "no2_backcast_flag",
    "ghsl_built_surface_km2",
    "ghsl_built_volume_m3",
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
    "policy_event_count_iso_year",
    "policy_event_type_count_iso_year",
    "policy_event_coarse_type_count_iso_year",
    "policy_event_count_infra_iso_year",
    "policy_event_count_digital_iso_year",
    "policy_event_count_eco_reg_iso_year",
    "policy_intensity_infra_iso_year",
    "policy_intensity_digital_iso_year",
    "policy_intensity_eco_reg_iso_year",
    "policy_event_new_count_iso_year",
    "policy_intensity_sum_iso_year",
    "policy_intensity_mean_iso_year",
    "policy_event_count_iso_year_yoy",
    "policy_intensity_sum_iso_year_yoy",
    "policy_news_proxy_score",
    "flight_connectivity_total",
    "flight_degree_centrality",
    "airport_count_mapped",
    "international_route_share",
    "shipping_connectivity_total",
    "has_network_connectivity_observation",
]

STRUCTURAL_FEATURES = [
    # NOTE: 'year' excluded — same reason as FEATURES.
    "latitude",
    "longitude",
    "temperature_mean",
    "precipitation_sum",
    "climate_comfort",
    "amenity_ratio",
    "commerce_ratio",
    "transport_intensity",
    "poi_diversity",
    "road_access_score",
    "road_tier_code",
    "road_length_km_total",
    "arterial_share",
    "intersection_density",
    "has_poi_observation",
    "viirs_log_mean",
    "viirs_physical_continuity",
    "has_viirs_observation",
    "no2_trop_anomaly_mean_filled",
    "observed_physical_stress_signal",
    "viirs_lit_area_km2",
    "ghsl_built_surface_km2",
    "ghsl_built_density",
    "gravity_access_viirs",
    "gravity_access_knowledge",
    "osm_hist_poi_count",
    "flight_degree_centrality",
    "airport_count_mapped",
    "international_route_share",
]

PULSE_AI_FEATURE_CANDIDATES = [
    "acceleration_score",
    "stall_probability",
    "stall_risk_score",
    "critical_transition_score",
    "critical_transition_risk_score",
    "kinetic_energy_score",
    "turning_point_risk",
    "damping_pressure",
    "dynamic_hazard_fused_score",
    "graph_diffusion_fused_score",
    "dynamic_pulse_index",
    "dynamic_pulse_delta_1y",
    "dynamic_pulse_trend_3y",
    "regime_forward_risk",
    "regime_transition_entropy",
]


def _train_test_split_by_year(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(pd.to_numeric(panel["year"], errors="coerce").dropna().astype(int).unique().tolist())
    if len(years) < 4:
        msg = "Need at least 4 distinct years for train/test split."
        raise ValueError(msg)

    train = panel[pd.to_numeric(panel["year"], errors="coerce") <= int(OOT_TRAIN_END_YEAR)].copy()
    test = panel[pd.to_numeric(panel["year"], errors="coerce") >= int(OOT_TEST_START_YEAR)].copy()
    if train.empty or test.empty:
        msg = (
            f"Out-of-time split requires train years <= {OOT_TRAIN_END_YEAR} and "
            f"test years >= {OOT_TEST_START_YEAR}."
        )
        raise ValueError(msg)
    return train, test


def _leakage_safe_feature_columns(target: str, feature_cols: List[str]) -> List[str]:
    cols = list(dict.fromkeys(feature_cols))
    target_l = str(target).strip().lower()
    if target_l == "economic_vitality":
        cols = [c for c in cols if ("viirs" not in c.lower()) and ("gdp" not in c.lower())]
    if target_l == "composite_index":
        cols = [c for c in cols if c not in {"economic_vitality", "livability", "innovation", "composite_index"}]
    return cols


def _resolve_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in FEATURES:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if float(s.notna().mean()) < 0.55:
            continue
        cols.append(c)
    if len(cols) < 8:
        msg = f"Insufficient modeling features after resolution: {cols}"
        raise RuntimeError(msg)
    return cols


def _load_pulse_ai_features() -> tuple[pd.DataFrame, List[str]]:
    """Load AI dynamic features derived from pulse engine for incrementality diagnostics."""
    path = DATA_OUTPUTS / "pulse_ai_scores.csv"
    if not path.exists():
        return pd.DataFrame(), []
    try:
        raw = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(), []
    if raw.empty or ("city_id" not in raw.columns) or ("year" not in raw.columns):
        return pd.DataFrame(), []

    ai_cols = [c for c in PULSE_AI_FEATURE_CANDIDATES if c in raw.columns]
    if len(ai_cols) < 4:
        return pd.DataFrame(), []
    out = raw[["city_id", "year"] + ai_cols].copy()
    for col in ai_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.sort_values(["city_id", "year"]).drop_duplicates(["city_id", "year"], keep="last")
    return out, ai_cols


def _fit_feature_set(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> Dict[str, object]:
    cols = [c for c in feature_cols if c in train.columns and c in test.columns]
    if len(cols) < 4:
        return {"status": "skipped", "reason": "insufficient_feature_count"}

    tr = train.dropna(subset=[target_col]).copy()
    te = test.dropna(subset=[target_col]).copy()
    if len(tr) < 40 or len(te) < 15:
        return {"status": "skipped", "reason": "insufficient_rows_after_na_filter"}

    usable_cols: List[str] = []
    fill_values: Dict[str, float] = {}
    for col in cols:
        s_tr = pd.to_numeric(tr[col], errors="coerce")
        s_te = pd.to_numeric(te[col], errors="coerce")
        if float(s_tr.notna().mean()) < 0.55:
            continue
        if float(s_te.notna().mean()) <= 0.0:
            continue
        median = float(s_tr.median()) if s_tr.notna().any() else 0.0
        fill_values[col] = median
        usable_cols.append(col)
    if len(usable_cols) < 4:
        return {"status": "skipped", "reason": "insufficient_usable_feature_count"}

    x_train = np.column_stack(
        [
            pd.to_numeric(tr[col], errors="coerce").fillna(fill_values[col]).to_numpy(dtype=float)
            for col in usable_cols
        ]
    )
    y_train = tr[target_col].to_numpy(dtype=float)
    x_test = np.column_stack(
        [
            pd.to_numeric(te[col], errors="coerce").fillna(fill_values[col]).to_numpy(dtype=float)
            for col in usable_cols
        ]
    )
    y_test = te[target_col].to_numpy(dtype=float)

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    pred_lr = lr.predict(x_test)

    hgb = HistGradientBoostingRegressor(
        random_state=42,
        learning_rate=0.04,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.02,
        max_bins=255,
    )
    hgb.fit(x_train, y_train)
    pred_hgb = hgb.predict(x_test)

    metrics = {
        "linear": _eval(y_test, pred_lr),
        "hist_gradient_boosting": _eval(y_test, pred_hgb),
    }
    best_name = min(metrics.keys(), key=lambda k: float(metrics[k]["rmse"]))
    best_metrics = metrics[best_name]
    return {
        "status": "ok",
        "feature_count": int(len(usable_cols)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "metrics": metrics,
        "best_model": str(best_name),
        "best_metrics": best_metrics,
    }


def _compute_ai_incrementality(
    train: pd.DataFrame,
    test: pd.DataFrame,
    targets: List[str],
    baseline_feature_cols: List[str],
    engineered_feature_cols: List[str],
    pulse_ai_cols: List[str],
) -> Dict[str, object]:
    """Evaluate whether AI dynamic features provide incremental predictive value."""
    rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for target in targets:
        target_col = f"{target}_t1"
        safe_baseline = _leakage_safe_feature_columns(target, baseline_feature_cols)
        safe_engineered = _leakage_safe_feature_columns(target, engineered_feature_cols)
        safe_pulse = _leakage_safe_feature_columns(target, pulse_ai_cols)
        set_defs: List[tuple[str, List[str]]] = [
            ("structural_only", list(dict.fromkeys([c for c in safe_baseline if c in train.columns]))),
            (
                "structural_plus_engineered",
                list(
                    dict.fromkeys(
                        [c for c in safe_baseline if c in train.columns]
                        + [c for c in safe_engineered if c in train.columns]
                    )
                ),
            ),
        ]
        if safe_pulse:
            set_defs.append(
                (
                    "full_plus_pulse_ai",
                    list(
                        dict.fromkeys(
                            [c for c in safe_baseline if c in train.columns]
                            + [c for c in safe_engineered if c in train.columns]
                            + [c for c in safe_pulse if c in train.columns]
                        )
                    ),
                )
            )
            set_defs.append(("pulse_ai_only", list(dict.fromkeys([c for c in safe_pulse if c in train.columns]))))

        per_set: Dict[str, Dict[str, object]] = {}
        for set_name, cols in set_defs:
            fit = _fit_feature_set(train, test, target_col=target_col, feature_cols=cols)
            per_set[set_name] = fit
            if fit.get("status") != "ok":
                continue
            metrics = fit.get("metrics", {})
            for model_name, met in metrics.items():
                rows.append(
                    {
                        "target": target,
                        "feature_set": set_name,
                        "model": str(model_name),
                        "feature_count": int(fit.get("feature_count", 0)),
                        "n_train": int(fit.get("n_train", 0)),
                        "n_test": int(fit.get("n_test", 0)),
                        "rmse": float(met["rmse"]),
                        "mae": float(met["mae"]),
                        "r2": float(met["r2"]),
                    }
                )

        base_best = per_set.get("structural_only", {}).get("best_metrics", {})
        eng_best = per_set.get("structural_plus_engineered", {}).get("best_metrics", {})
        full_best = per_set.get("full_plus_pulse_ai", {}).get("best_metrics", {}) if pulse_ai_cols else {}

        base_rmse = float(base_best.get("rmse")) if base_best and ("rmse" in base_best) else None
        eng_rmse = float(eng_best.get("rmse")) if eng_best and ("rmse" in eng_best) else None
        full_rmse = float(full_best.get("rmse")) if full_best and ("rmse" in full_best) else None
        base_r2 = float(base_best.get("r2")) if base_best and ("r2" in base_best) else None
        eng_r2 = float(eng_best.get("r2")) if eng_best and ("r2" in eng_best) else None
        full_r2 = float(full_best.get("r2")) if full_best and ("r2" in full_best) else None

        summary_rows.append(
            {
                "target": target,
                "structural_rmse": base_rmse,
                "structural_plus_engineered_rmse": eng_rmse,
                "full_plus_pulse_ai_rmse": full_rmse,
                "delta_rmse_engineered_vs_structural": (eng_rmse - base_rmse)
                if (base_rmse is not None and eng_rmse is not None)
                else None,
                "delta_rmse_full_ai_vs_structural": (full_rmse - base_rmse)
                if (base_rmse is not None and full_rmse is not None)
                else None,
                "structural_r2": base_r2,
                "structural_plus_engineered_r2": eng_r2,
                "full_plus_pulse_ai_r2": full_r2,
                "delta_r2_engineered_vs_structural": (eng_r2 - base_r2)
                if (base_r2 is not None and eng_r2 is not None)
                else None,
                "delta_r2_full_ai_vs_structural": (full_r2 - base_r2)
                if (base_r2 is not None and full_r2 is not None)
                else None,
                "best_set": min(
                    [
                        ("structural_only", base_rmse),
                        ("structural_plus_engineered", eng_rmse),
                        ("full_plus_pulse_ai", full_rmse),
                    ],
                    key=lambda kv: (float("inf") if kv[1] is None else kv[1]),
                )[0],
            }
        )

    detail_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)
    detail_df.to_csv(DATA_OUTPUTS / "model_ai_incrementality.csv", index=False)
    summary_df.to_csv(DATA_OUTPUTS / "model_ai_incrementality_summary.csv", index=False)

    if summary_df.empty:
        return {"status": "skipped", "reason": "ai_incrementality_no_valid_runs"}

    rmse_gain = pd.to_numeric(summary_df["delta_rmse_full_ai_vs_structural"], errors="coerce")
    r2_gain = pd.to_numeric(summary_df["delta_r2_full_ai_vs_structural"], errors="coerce")
    return {
        "status": "ok",
        "targets_evaluated": int(summary_df["target"].nunique()),
        "detail_file": str(DATA_OUTPUTS / "model_ai_incrementality.csv"),
        "summary_file": str(DATA_OUTPUTS / "model_ai_incrementality_summary.csv"),
        "mean_delta_rmse_full_ai_vs_structural": float(rmse_gain.mean()) if rmse_gain.notna().any() else None,
        "share_targets_rmse_improved_full_ai": float(np.mean(rmse_gain < 0.0)) if rmse_gain.notna().any() else None,
        "mean_delta_r2_full_ai_vs_structural": float(r2_gain.mean()) if r2_gain.notna().any() else None,
        "share_targets_r2_improved_full_ai": float(np.mean(r2_gain > 0.0)) if r2_gain.notna().any() else None,
        "summary": summary_df.to_dict(orient="records"),
    }


def _eval(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _fit_single_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    feature_cols: List[str],
) -> Tuple[Dict[str, Dict[str, float]], object, str, pd.DataFrame, pd.DataFrame]:
    target_col = f"{target}_t1"
    train = train.dropna(subset=[target_col]).copy()
    test = test.dropna(subset=[target_col]).copy()
    if len(train) < 30 or len(test) < 10:
        msg = f"Insufficient rows for target {target} in t+1 setup."
        raise RuntimeError(msg)

    usable_cols: List[str] = []
    fill_values: Dict[str, float] = {}
    for col in feature_cols:
        if col not in train.columns or col not in test.columns:
            continue
        s_train = pd.to_numeric(train[col], errors="coerce")
        s_test = pd.to_numeric(test[col], errors="coerce")
        if float(s_train.notna().mean()) < 0.55:
            continue
        if float(s_test.notna().mean()) <= 0.0:
            continue
        fill_values[col] = float(s_train.median()) if s_train.notna().any() else 0.0
        usable_cols.append(col)

    if len(usable_cols) < 8:
        msg = f"Insufficient usable features for target {target}: {usable_cols}"
        raise RuntimeError(msg)

    X_train = np.column_stack(
        [
            pd.to_numeric(train[col], errors="coerce").fillna(fill_values[col]).to_numpy(dtype=float)
            for col in usable_cols
        ]
    )
    X_test = np.column_stack(
        [
            pd.to_numeric(test[col], errors="coerce").fillna(fill_values[col]).to_numpy(dtype=float)
            for col in usable_cols
        ]
    )
    y_train = train[target_col].to_numpy(dtype=float)
    y_test = test[target_col].to_numpy(dtype=float)

    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    pred_base = baseline.predict(X_test)

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=10,  # Raised from 2 to prevent overfitting on ~3k rows
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    gbr = GradientBoostingRegressor(
        random_state=42,
        n_estimators=320,
        learning_rate=0.04,
        max_depth=3,
        min_samples_leaf=2,
        subsample=0.9,
    )
    gbr.fit(X_train, y_train)
    pred_gbr = gbr.predict(X_test)

    hgb = HistGradientBoostingRegressor(
        random_state=42,
        learning_rate=0.04,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.02,
        max_bins=255,
    )
    hgb.fit(X_train, y_train)
    pred_hgb = hgb.predict(X_test)

    metrics = {
        "linear": _eval(y_test, pred_base),
        "random_forest": _eval(y_test, pred_rf),
        "gradient_boosting": _eval(y_test, pred_gbr),
        "hist_gradient_boosting": _eval(y_test, pred_hgb),
    }

    model_lookup = {
        "linear": baseline,
        "random_forest": rf,
        "gradient_boosting": gbr,
        "hist_gradient_boosting": hgb,
    }
    best_name = min(metrics.keys(), key=lambda k: float(metrics[k]["rmse"]))
    best_model = model_lookup[best_name]

    pred = pd.DataFrame(
        {
            "city_id": test["city_id"].to_numpy(),
            "year": test["year"].to_numpy(),
            "target": target,
            "horizon": "t_plus_1",
            "actual": y_test,
            "pred_linear": pred_base,
            "pred_random_forest": pred_rf,
            "pred_gradient_boosting": pred_gbr,
            "pred_hist_gradient_boosting": pred_hgb,
            "pred_best": best_model.predict(X_test),
            "best_model": best_name,
        }
    )

    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame({"feature": usable_cols, "importance": best_model.feature_importances_}).sort_values(
            "importance", ascending=False
        )
    elif hasattr(rf, "feature_importances_"):
        fi = pd.DataFrame({"feature": usable_cols, "importance": rf.feature_importances_}).sort_values(
            "importance", ascending=False
        )
    else:
        fi = pd.DataFrame({"feature": usable_cols, "importance": np.nan})

    return metrics, best_model, best_name, fi, pred


def train_all_targets(panel: pd.DataFrame) -> dict:
    """Train baseline and tree models for all targets and persist outputs."""
    panel = panel.sort_values(["city_id", "year"]).copy()
    panel, no2_backcast = add_no2_backcast_features(panel, fit_end_year=OOT_TRAIN_END_YEAR, output_stub="modeling")
    pulse_ai_df, pulse_ai_cols = _load_pulse_ai_features()
    if (not pulse_ai_df.empty) and pulse_ai_cols:
        panel = panel.merge(pulse_ai_df, on=["city_id", "year"], how="left")
        LOGGER.info("Predictive modeling: merged pulse-AI features: %s", len(pulse_ai_cols))
    for target in TARGETS:
        panel[f"{target}_t1"] = panel.groupby("city_id")[target].shift(-1)

    feature_cols = _resolve_feature_columns(panel)
    baseline_feature_cols = [c for c in STRUCTURAL_FEATURES if c in panel.columns]
    engineered_feature_cols = [c for c in feature_cols if c not in baseline_feature_cols]
    train, test = _train_test_split_by_year(panel)
    LOGGER.info(
        "Predictive modeling: out-of-time split ready: train<=%s rows=%s, test>=%s rows=%s, targets=%s",
        OOT_TRAIN_END_YEAR,
        len(train),
        OOT_TEST_START_YEAR,
        len(test),
        len(TARGETS),
    )

    all_metrics: Dict[str, object] = {}
    all_preds: List[pd.DataFrame] = []
    all_fi: List[pd.DataFrame] = []

    for target in TARGETS:
        LOGGER.info("Predictive modeling: fitting target=%s", target)
        safe_feature_cols = _leakage_safe_feature_columns(target, feature_cols)
        metrics, model, best_name, fi, pred = _fit_single_target(train, test, target, feature_cols=safe_feature_cols)
        all_metrics[target] = metrics
        all_fi.append(fi.assign(target=target))
        all_preds.append(pred)

        model_path = MODELS_DIR / f"{target}_best_model.joblib"
        joblib.dump(model, model_path)

        LOGGER.info("Model trained for %s, best=%s", target, best_name)

    ai_incrementality = _compute_ai_incrementality(
        train=train,
        test=test,
        targets=TARGETS,
        baseline_feature_cols=baseline_feature_cols,
        engineered_feature_cols=engineered_feature_cols,
        pulse_ai_cols=[c for c in pulse_ai_cols if c in panel.columns],
    )

    pred_all = pd.concat(all_preds, ignore_index=True)
    pred_all.to_csv(DATA_OUTPUTS / "predictions.csv", index=False)

    fi_all = pd.concat(all_fi, ignore_index=True)
    fi_all.to_csv(DATA_OUTPUTS / "feature_importance.csv", index=False)

    all_metrics["train_test_split"] = {
        "strategy": "out_of_time_fixed_cutoff",
        "train_end_year": int(OOT_TRAIN_END_YEAR),
        "test_start_year": int(OOT_TEST_START_YEAR),
    }
    all_metrics["no2_backcast"] = no2_backcast
    all_metrics["ai_incrementality_summary"] = ai_incrementality
    dump_json(DATA_OUTPUTS / "model_metrics.json", all_metrics)
    dump_json(DATA_OUTPUTS / "model_ai_incrementality.json", ai_incrementality)
    return all_metrics
