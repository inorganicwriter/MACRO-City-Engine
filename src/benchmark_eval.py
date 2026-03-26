from __future__ import annotations

"""Benchmark protocol for temporal and spatial OOD evaluation."""

import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .econometrics import _assert_valid_causal_outcome, _resolve_primary_causal_outcome
from .feature_backfill import add_no2_backcast_features
from .utils import DATA_OUTPUTS, dump_json, haversine_km

LOGGER = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}

try:  # Optional strong baseline
    from xgboost import XGBRegressor
except Exception:  # noqa: BLE001
    XGBRegressor = None  # type: ignore[assignment]

try:  # Optional strong baseline
    from lightgbm import LGBMRegressor
except Exception:  # noqa: BLE001
    LGBMRegressor = None  # type: ignore[assignment]


FEATURES: List[str] = [
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
    "graph_neighbor_composite_mean",
    "graph_neighbor_growth_mean",
    "graph_neighbor_internet_mean",
    "graph_spillover_gap",
    "graph_neighbor_composite_delta1",
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


def _metric(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _resolve_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in FEATURES:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        non_null_ratio = float(s.notna().mean()) if len(s) else 0.0
        if non_null_ratio < 0.55:
            continue
        cols.append(c)
    if len(cols) < 10:
        msg = f"Insufficient benchmark features after resolution: {cols}"
        raise RuntimeError(msg)
    return cols


def _resolve_benchmark_target(panel: pd.DataFrame, target: str | None = None) -> str:
    requested = str(target).strip() if target is not None else ""
    if requested:
        try:
            _assert_valid_causal_outcome(requested)
            if requested in panel.columns:
                values = pd.to_numeric(panel[requested], errors="coerce")
                if values.notna().sum() >= 40:
                    return requested
        except Exception:  # noqa: BLE001
            LOGGER.warning(
                "Benchmark target %r is not admissible for core analysis; falling back to a raw observed target.",
                requested,
            )
    return _resolve_primary_causal_outcome(panel)


def _build_neighbor_map(df: pd.DataFrame, k_neighbors: int = 5) -> Dict[str, List[str]]:
    city = df[["city_id", "latitude", "longitude"]].drop_duplicates("city_id").copy()
    city = city.dropna(subset=["latitude", "longitude"])
    out: Dict[str, List[str]] = {}
    if city.empty:
        return out

    ids = city["city_id"].astype(str).tolist()
    lat = city["latitude"].to_numpy(dtype=float)
    lon = city["longitude"].to_numpy(dtype=float)
    k = max(1, int(k_neighbors))

    for i, cid in enumerate(ids):
        dists: List[tuple[float, str]] = []
        for j, other in enumerate(ids):
            if i == j:
                continue
            d = haversine_km(float(lat[i]), float(lon[i]), float(lat[j]), float(lon[j]))
            dists.append((d, other))
        dists.sort(key=lambda x: x[0])
        out[str(cid)] = [str(x[1]) for x in dists[:k]]
    return out


def _add_graph_spillover_features(df: pd.DataFrame, k_neighbors: int = 5) -> pd.DataFrame:
    out = df.copy()
    required = {"city_id", "year", "latitude", "longitude"}
    if not required.issubset(set(out.columns)):
        return out

    neighbors = _build_neighbor_map(out, k_neighbors=k_neighbors)
    if not neighbors:
        return out

    value_map: Dict[str, Dict[tuple[str, int], float]] = {}
    # Neighbor signals are built from city-observed dynamic channels.
    target_proxy = "log_viirs_ntl" if "log_viirs_ntl" in out.columns else "composite_index"
    for col in [target_proxy, "road_growth_intensity", "viirs_ntl_yoy"]:
        if col in out.columns:
            value_map[col] = (
                out[["city_id", "year", col]]
                .dropna(subset=[col])
                .assign(city_id=lambda x: x["city_id"].astype(str), year=lambda x: x["year"].astype(int))
                .set_index(["city_id", "year"])[col]
                .to_dict()
            )

    n = len(out)
    neigh_comp = np.full(n, np.nan, dtype=float)
    neigh_growth = np.full(n, np.nan, dtype=float)
    neigh_net = np.full(n, np.nan, dtype=float)
    self_comp = pd.to_numeric(out.get(target_proxy), errors="coerce").to_numpy(dtype=float)

    def _fallback_series(col_name: str, default: float = 0.0) -> pd.Series:
        if col_name in out.columns:
            return pd.to_numeric(out[col_name], errors="coerce").fillna(default)
        return pd.Series(np.full(n, default, dtype=float))

    for i, row in enumerate(out.itertuples(index=False)):
        cid = str(row.city_id)
        yr = int(row.year)
        neigh_ids = neighbors.get(cid, [])
        if not neigh_ids:
            continue

        def _mean_from_map(col_name: str) -> float:
            src = value_map.get(col_name, {})
            # Use lagged (yr-1) neighbor values to avoid look-ahead bias.
            vals = [src.get((nid, yr - 1), np.nan) for nid in neigh_ids]
            vals = [float(v) for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")

        neigh_comp[i] = _mean_from_map(target_proxy)
        neigh_growth[i] = _mean_from_map("road_growth_intensity")
        neigh_net[i] = _mean_from_map("viirs_ntl_yoy")

    out["graph_neighbor_composite_mean"] = pd.Series(neigh_comp).fillna(pd.Series(self_comp)).astype(float)
    out["graph_neighbor_growth_mean"] = (
        pd.Series(neigh_growth)
        .fillna(_fallback_series("road_growth_intensity", default=0.0))
        .astype(float)
    )
    out["graph_neighbor_internet_mean"] = (
        pd.Series(neigh_net)
        .fillna(_fallback_series("viirs_ntl_yoy", default=0.0))
        .astype(float)
    )
    out["graph_spillover_gap"] = (
        out["graph_neighbor_composite_mean"]
        - pd.to_numeric(out.get(target_proxy), errors="coerce").fillna(out["graph_neighbor_composite_mean"])
    ).astype(float)
    out = out.sort_values(["city_id", "year"]).copy()
    out["graph_neighbor_composite_delta1"] = (
        out.groupby("city_id")["graph_neighbor_composite_mean"].diff().fillna(0.0).astype(float)
    )
    return out


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    random_seed: int = 42,
) -> Dict[str, Any]:
    fast_mode = _env_flag("BENCHMARK_FAST", False)
    prep = _prepare_feature_matrices(train, test, target, feature_cols, min_train_rows=40, min_test_rows=15)
    if prep is None:
        return {
            "metrics": {},
            "model_status": {"all": "skipped:insufficient_rows_or_features"},
            "feature_count": 0,
        }
    x_train, y_train, x_test, y_test, usable_cols = prep

    metrics: Dict[str, Dict[str, float]] = {}
    model_status: Dict[str, str] = {}

    def _fit_one(name: str, model: Any) -> None:
        try:
            model.fit(x_train, y_train)
            pred = np.asarray(model.predict(x_test), dtype=float)
            metrics[name] = _metric(y_test, pred)
            model_status[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            model_status[name] = f"failed:{type(exc).__name__}"

    _fit_one("linear", LinearRegression())
    _fit_one(
        "elastic_net",
        make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=0.03,
                l1_ratio=0.4,
                random_state=random_seed,
                max_iter=5000,
            ),
        ),
    )
    _fit_one(
        "random_forest",
        RandomForestRegressor(
            n_estimators=80 if fast_mode else 160,
            max_depth=10,
            min_samples_leaf=10,  # Raised from 2 to prevent overfitting
            random_state=random_seed,
            n_jobs=-1,
        ),
    )
    _fit_one(
        "hist_gradient_boosting",
        HistGradientBoostingRegressor(
            random_state=random_seed,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=0.02,
        ),
    )
    _fit_one(
        "extra_trees",
        ExtraTreesRegressor(
            n_estimators=80 if fast_mode else 200,
            max_depth=12,  # Capped from None to prevent memorization
            min_samples_leaf=10,  # Raised from 2 to prevent overfitting
            random_state=random_seed,
            n_jobs=-1,
        ),
    )

    graph_cols = [c for c in usable_cols if c.startswith("graph_")]
    if graph_cols:
        _fit_one(
            "graph_spillover_rf",
            RandomForestRegressor(
                n_estimators=80 if fast_mode else 160,
                max_depth=9,
                min_samples_leaf=10,  # Consistent with other RF models
                random_state=random_seed + 9,
                n_jobs=-1,
            ),
        )
    else:
        model_status["graph_spillover_rf"] = "skipped:no_graph_features"

    if fast_mode:
        model_status["xgboost"] = "skipped:fast_mode"
    elif XGBRegressor is not None:
        _fit_one(
            "xgboost",
            XGBRegressor(
                n_estimators=420,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                random_state=random_seed,
                n_jobs=4,
                objective="reg:squarederror",
            ),
        )
    else:
        model_status["xgboost"] = "skipped:dependency_missing"

    if fast_mode:
        model_status["lightgbm"] = "skipped:fast_mode"
    elif LGBMRegressor is not None:
        _fit_one(
            "lightgbm",
            LGBMRegressor(
                n_estimators=480,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_seed,
                n_jobs=4,
                objective="regression",
                verbosity=-1,
            ),
        )
    else:
        model_status["lightgbm"] = "skipped:dependency_missing"

    return {
        "metrics": metrics,
        "model_status": model_status,
        "feature_count": int(len(usable_cols)),
    }


def _prepare_feature_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    min_train_rows: int,
    min_test_rows: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, List[str]] | None:
    tr = train.dropna(subset=[target]).copy()
    te = test.dropna(subset=[target]).copy()
    if len(tr) < min_train_rows or len(te) < min_test_rows:
        return None

    usable_cols: List[str] = []
    fill_values: Dict[str, float] = {}
    for col in feature_cols:
        if col not in tr.columns or col not in te.columns:
            continue
        s_tr = pd.to_numeric(tr[col], errors="coerce")
        s_te = pd.to_numeric(te[col], errors="coerce")
        if float(s_tr.notna().mean()) < 0.55:
            continue
        if float(s_te.notna().mean()) <= 0.0:
            continue
        fill_values[col] = float(s_tr.median()) if s_tr.notna().any() else 0.0
        usable_cols.append(col)
    if len(usable_cols) < 4:
        return None

    x_train = pd.DataFrame(
        {col: pd.to_numeric(tr[col], errors="coerce").fillna(fill_values[col]).astype(float) for col in usable_cols}
    )
    y_train = tr[target].to_numpy(dtype=float)
    x_test = pd.DataFrame(
        {col: pd.to_numeric(te[col], errors="coerce").fillna(fill_values[col]).astype(float) for col in usable_cols}
    )
    y_test = te[target].to_numpy(dtype=float)
    return x_train, y_train, x_test, y_test, usable_cols


def _summarize_spatial(spatial_df: pd.DataFrame) -> Dict[str, Any]:
    if spatial_df.empty:
        return {"status": "insufficient_spatial_groups"}

    out: Dict[str, Any] = {
        "continents_evaluated": int(spatial_df["continent"].nunique()),
    }
    for model in [
        "linear",
        "elastic_net",
        "random_forest",
        "hist_gradient_boosting",
        "extra_trees",
        "graph_spillover_rf",
        "xgboost",
        "lightgbm",
    ]:
        rmse_col = f"{model}_rmse"
        r2_col = f"{model}_r2"
        if rmse_col in spatial_df.columns and spatial_df[rmse_col].notna().any():
            out[f"mean_{model}_rmse"] = float(spatial_df[rmse_col].mean())
        if r2_col in spatial_df.columns and spatial_df[r2_col].notna().any():
            out[f"mean_{model}_r2"] = float(spatial_df[r2_col].mean())

    # Keep backward compatibility for manuscript sync scripts.
    out["mean_rf_rmse"] = out.get("mean_random_forest_rmse")
    out["mean_rf_r2"] = out.get("mean_random_forest_r2")
    out["mean_hgb_rmse"] = out.get("mean_hist_gradient_boosting_rmse")
    out["mean_hgb_r2"] = out.get("mean_hist_gradient_boosting_r2")
    out["mean_et_rmse"] = out.get("mean_extra_trees_rmse")
    out["mean_et_r2"] = out.get("mean_extra_trees_r2")
    return out


def _run_expanding_window_cv(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    min_train_years: int = 4,
) -> Dict[str, Any]:
    """Expanding-window time series CV: train on [start..y], test on [y+1].

    This provides more robust evaluation than a single temporal split by
    averaging over multiple train/test boundaries.
    """
    years = sorted(df["year"].unique().tolist())
    if len(years) < min_train_years + 1:
        return {"status": "skipped", "reason": "too_few_years_for_cv"}

    fold_results: List[Dict[str, Any]] = []
    for fold_idx, test_year in enumerate(years[min_train_years:], start=1):
        train_fold = df[df["year"] < test_year].copy()
        test_fold = df[df["year"] == test_year].copy()
        if len(train_fold) < 50 or len(test_fold) < 10:
            continue

        prep = _prepare_feature_matrices(
            train_fold,
            test_fold,
            target_col,
            feature_cols,
            min_train_rows=50,
            min_test_rows=10,
        )
        if prep is None:
            continue
        x_tr, y_tr, x_te, y_te, usable_cols = prep

        from sklearn.linear_model import LinearRegression as LR
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB

        lr = LR()
        lr.fit(x_tr, y_tr)
        hgb = HGB(random_state=42, learning_rate=0.05, max_depth=6, min_samples_leaf=20)
        hgb.fit(x_tr, y_tr)

        fold_results.append({
            "fold": fold_idx,
            "test_year": int(test_year),
            "n_train": len(train_fold),
            "n_test": len(test_fold),
            "feature_count": int(len(usable_cols)),
            "linear_rmse": float(np.sqrt(mean_squared_error(y_te, lr.predict(x_te)))),
            "linear_r2": float(r2_score(y_te, lr.predict(x_te))),
            "hgb_rmse": float(np.sqrt(mean_squared_error(y_te, hgb.predict(x_te)))),
            "hgb_r2": float(r2_score(y_te, hgb.predict(x_te))),
        })

    if not fold_results:
        return {"status": "skipped", "reason": "no_valid_folds"}

    cv_df = pd.DataFrame(fold_results)
    cv_df.to_csv(DATA_OUTPUTS / "benchmark_expanding_cv.csv", index=False)

    return {
        "status": "ok",
        "n_folds": len(fold_results),
        "mean_linear_rmse": float(cv_df["linear_rmse"].mean()),
        "std_linear_rmse": float(cv_df["linear_rmse"].std()),
        "mean_linear_r2": float(cv_df["linear_r2"].mean()),
        "mean_hgb_rmse": float(cv_df["hgb_rmse"].mean()),
        "std_hgb_rmse": float(cv_df["hgb_rmse"].std()),
        "mean_hgb_r2": float(cv_df["hgb_r2"].mean()),
        "per_fold": fold_results,
        "output_file": str(DATA_OUTPUTS / "benchmark_expanding_cv.csv"),
    }


def _run_prospective_governance_eval(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    min_train_years: int = 4,
) -> Dict[str, Any]:
    """Prospective rolling-origin evaluation with frozen yearly retraining.

    At feature-year t, models are trained on years < t and predict target_{t+1}.
    Baselines include persistence (predict y_{t+1}=y_t) and train-mean.
    """
    years = sorted(df["year"].unique().tolist())
    if len(years) < min_train_years + 2:
        return {"status": "skipped", "reason": "too_few_years_for_prospective_eval"}

    folds: List[Dict[str, Any]] = []
    for eval_year in years[min_train_years:-1]:
        train_fold = df[df["year"] < eval_year].copy()
        test_fold = df[df["year"] == eval_year].copy()
        if len(train_fold) < 80 or len(test_fold) < 20:
            continue

        prep = _prepare_feature_matrices(
            train_fold,
            test_fold,
            target_col,
            feature_cols,
            min_train_rows=80,
            min_test_rows=20,
        )
        if prep is None:
            continue
        x_tr, y_tr, x_te, y_te, usable_cols = prep
        y_level_te = pd.to_numeric(test_fold.get("_target_current"), errors="coerce").to_numpy(dtype=float)

        models: Dict[str, Any] = {
            "linear": LinearRegression(),
            "hist_gradient_boosting": HistGradientBoostingRegressor(
                random_state=42,
                learning_rate=0.05,
                max_depth=6,
                min_samples_leaf=20,
            ),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=220,
                max_depth=12,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            ),
        }

        row: Dict[str, Any] = {
            "feature_year": int(eval_year),
            "target_year": int(eval_year + 1),
            "n_train": int(len(train_fold)),
            "n_test": int(len(test_fold)),
            "feature_count": int(len(usable_cols)),
        }
        model_preds: Dict[str, np.ndarray] = {}
        for name, model in models.items():
            try:
                model.fit(x_tr, y_tr)
                pred = np.asarray(model.predict(x_te), dtype=float)
                model_preds[name] = pred
                met = _metric(y_te, pred)
                row[f"{name}_rmse"] = float(met["rmse"])
                row[f"{name}_mae"] = float(met["mae"])
                row[f"{name}_r2"] = float(met["r2"])
            except Exception:  # noqa: BLE001
                row[f"{name}_rmse"] = np.nan
                row[f"{name}_mae"] = np.nan
                row[f"{name}_r2"] = np.nan

        mean_pred = np.repeat(float(np.nanmean(y_tr)), len(y_te))
        persist_pred = np.where(np.isfinite(y_level_te), y_level_te, float(np.nanmean(y_tr)))
        for base_name, pred in [("baseline_mean", mean_pred), ("baseline_persistence", persist_pred)]:
            met = _metric(y_te, pred)
            row[f"{base_name}_rmse"] = float(met["rmse"])
            row[f"{base_name}_mae"] = float(met["mae"])
            row[f"{base_name}_r2"] = float(met["r2"])

        # Directional utility for policy monitoring: sign of one-year change.
        delta_true = y_te - y_level_te
        delta_base = persist_pred - y_level_te
        if np.isfinite(delta_true).any():
            row["baseline_persistence_directional_acc"] = float(np.mean(np.sign(delta_true) == np.sign(delta_base)))
        for name, pred in model_preds.items():
            delta_pred = pred - y_level_te
            if np.isfinite(delta_true).any():
                row[f"{name}_directional_acc"] = float(np.mean(np.sign(delta_true) == np.sign(delta_pred)))
            else:
                row[f"{name}_directional_acc"] = np.nan
        folds.append(row)

    if not folds:
        return {"status": "skipped", "reason": "no_valid_prospective_folds"}

    fold_df = pd.DataFrame(folds).sort_values("feature_year")
    out_csv = DATA_OUTPUTS / "benchmark_prospective_governance_eval.csv"
    fold_df.to_csv(out_csv, index=False)

    model_names = ["linear", "hist_gradient_boosting", "extra_trees"]
    model_rmse = {
        m: float(pd.to_numeric(fold_df[f"{m}_rmse"], errors="coerce").mean())
        for m in model_names
        if f"{m}_rmse" in fold_df.columns
    }
    model_rmse = {k: v for k, v in model_rmse.items() if np.isfinite(v)}
    if not model_rmse:
        return {"status": "skipped", "reason": "all_models_failed_in_prospective_eval", "output_file": str(out_csv)}
    best_model = min(model_rmse, key=model_rmse.get)
    best_rmse = float(model_rmse[best_model])
    base_rmse = float(pd.to_numeric(fold_df["baseline_persistence_rmse"], errors="coerce").mean())
    best_r2 = float(pd.to_numeric(fold_df[f"{best_model}_r2"], errors="coerce").mean())
    base_r2 = float(pd.to_numeric(fold_df["baseline_persistence_r2"], errors="coerce").mean())
    best_dir = float(pd.to_numeric(fold_df.get(f"{best_model}_directional_acc"), errors="coerce").mean())
    base_dir = float(pd.to_numeric(fold_df.get("baseline_persistence_directional_acc"), errors="coerce").mean())

    return {
        "status": "ok",
        "n_folds": int(len(fold_df)),
        "best_model_by_rmse": str(best_model),
        "best_model_mean_rmse": best_rmse,
        "best_model_mean_r2": best_r2,
        "baseline_persistence_mean_rmse": base_rmse,
        "baseline_persistence_mean_r2": base_r2,
        "rmse_skill_vs_persistence": float(1.0 - best_rmse / max(base_rmse, 1e-12)),
        "r2_gain_vs_persistence": float(best_r2 - base_r2),
        "best_model_directional_acc": best_dir,
        "baseline_persistence_directional_acc": base_dir,
        "directional_acc_gain": float(best_dir - base_dir),
        "per_model_mean_rmse": {k: float(v) for k, v in sorted(model_rmse.items())},
        "output_file": str(out_csv),
    }


def run_benchmark_suite(panel: pd.DataFrame, target: str | None = None) -> Dict[str, object]:
    """Evaluate temporal and spatial OOD performance for benchmark reporting."""
    fast_mode = _env_flag("BENCHMARK_FAST", False)
    resolved_target = _resolve_benchmark_target(panel, target)
    df = panel.sort_values(["city_id", "year"]).copy()
    df = _add_graph_spillover_features(df, k_neighbors=5)
    df["_target_current"] = pd.to_numeric(df[resolved_target], errors="coerce")
    df["target_t1"] = df.groupby("city_id")[resolved_target].shift(-1)
    years = sorted(df["year"].unique().tolist())
    split_year = years[-3] if len(years) >= 4 else years[-2]
    df, no2_backcast = add_no2_backcast_features(df, fit_end_year=split_year, output_stub="benchmark")
    feature_cols = _resolve_feature_columns(df)
    df = df.dropna(subset=["target_t1"]).copy()
    if len(df) < 200:
        out = {
            "status": "skipped",
            "reason": "too_few_rows_for_benchmark",
            "requested_target": target,
            "resolved_target": resolved_target,
            "no2_backcast": no2_backcast,
        }
        dump_json(DATA_OUTPUTS / "benchmark_scores.json", out)
        return out

    train_t = df[df["year"] <= split_year].copy()
    test_t = df[df["year"] > split_year].copy()
    temporal_fit = _fit_predict(train_t, test_t, "target_t1", feature_cols=feature_cols, random_seed=42)
    temporal = temporal_fit["metrics"]

    temporal_rows: List[Dict[str, Any]] = []
    for model, met in temporal.items():
        temporal_rows.append(
            {
                "model": model,
                "rmse": float(met["rmse"]),
                "mae": float(met["mae"]),
                "r2": float(met["r2"]),
                "split_year": int(split_year),
                "n_train": int(len(train_t)),
                "n_test": int(len(test_t)),
            }
        )
    temporal_board = pd.DataFrame(temporal_rows).sort_values("rmse")
    temporal_board.to_csv(DATA_OUTPUTS / "benchmark_temporal_model_board.csv", index=False)

    continents = sorted(df["continent"].dropna().unique().tolist())
    spatial_rows: List[Dict[str, object]] = []
    for idx, cont in enumerate(continents):
        train_s = df[df["continent"] != cont].copy()
        test_s = df[df["continent"] == cont].copy()
        if len(train_s) < 100 or len(test_s) < 30:
            continue
        fit = _fit_predict(train_s, test_s, "target_t1", feature_cols=feature_cols, random_seed=300 + idx)
        metrics = fit["metrics"]
        row: Dict[str, object] = {"continent": cont, "n_test": int(len(test_s))}
        for name, met in metrics.items():
            row[f"{name}_rmse"] = float(met["rmse"])
            row[f"{name}_mae"] = float(met["mae"])
            row[f"{name}_r2"] = float(met["r2"])
        spatial_rows.append(row)

    spatial_df = pd.DataFrame(spatial_rows)
    if not spatial_df.empty:
        spatial_df.to_csv(DATA_OUTPUTS / "benchmark_spatial_ood.csv", index=False)
    spatial_summary = _summarize_spatial(spatial_df)

    # --- First-difference benchmark (Review Issue #5) ---
    if fast_mode:
        fd_holdout = {"status": "skipped", "reason": "fast_mode"}
        fd_holdout_common_shift_adj = {"status": "skipped", "reason": "fast_mode"}
        stall_sensitivity = {"status": "skipped", "reason": "fast_mode"}
        expanding_cv = {"status": "skipped", "reason": "fast_mode"}
        prospective_eval = {"status": "skipped", "reason": "fast_mode"}
    else:
        fd_holdout = _run_first_difference_benchmark(panel, resolved_target, feature_cols, split_year, output_tag="raw")
        # Additional regime-adjusted track:
        # Remove global year-level common shift in the level target first, then benchmark differences.
        # This isolates city-specific dynamics and prevents measurement-regime shifts (e.g., new data layer
        # available only in late years) from dominating raw-difference scores.
        panel_adj = panel.copy()
        tgt_vals = pd.to_numeric(panel_adj[resolved_target], errors="coerce")
        panel_adj["_fd_target_common_shift_adj"] = tgt_vals - tgt_vals.groupby(panel_adj["year"]).transform("mean")
        fd_holdout_common_shift_adj = _run_first_difference_benchmark(
            panel_adj,
            "_fd_target_common_shift_adj",
            feature_cols,
            split_year,
            output_tag="common_shift_adjusted",
        )

        # --- Stall threshold sensitivity (Review Issue #5) ---
        stall_sensitivity = _run_stall_threshold_sensitivity(panel, resolved_target)

        # --- Expanding-window time series CV for robustness ---
        expanding_cv = _run_expanding_window_cv(df, "target_t1", feature_cols, min_train_years=4)
        prospective_eval = _run_prospective_governance_eval(df, "target_t1", feature_cols, min_train_years=4)

    out = {
        "target": f"{resolved_target}_t_plus_1",
        "requested_target": target,
        "resolved_target": resolved_target,
        "temporal_holdout": temporal,
        "temporal_model_status": temporal_fit["model_status"],
        "spatial_ood": spatial_summary,
        "first_difference_holdout": fd_holdout,
        "first_difference_holdout_common_shift_adjusted": fd_holdout_common_shift_adj,
        "stall_threshold_sensitivity": stall_sensitivity,
        "expanding_window_cv": expanding_cv,
        "prospective_governance_eval": prospective_eval,
        "split_year": int(split_year),
        "n_rows": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "graph_feature_count": int(len([c for c in feature_cols if c.startswith("graph_")])),
        "no2_backcast": no2_backcast,
        "fast_mode": bool(fast_mode),
        "strong_baselines": {
            "xgboost_available": bool((XGBRegressor is not None) and not fast_mode),
            "lightgbm_available": bool((LGBMRegressor is not None) and not fast_mode),
            "graph_spillover_baseline_enabled": bool(len([c for c in feature_cols if c.startswith("graph_")]) > 0),
            "temporal_leaderboard_file": str(DATA_OUTPUTS / "benchmark_temporal_model_board.csv"),
        },
    }
    dump_json(DATA_OUTPUTS / "benchmark_scores.json", out)

    ref_model = "hist_gradient_boosting" if "hist_gradient_boosting" in temporal else "linear"
    LOGGER.info(
        "Benchmark suite done: target=%s, temporal %s rmse=%.3f, spatial status=%s",
        resolved_target,
        ref_model,
        float(temporal[ref_model]["rmse"]),
        "ok" if "mean_random_forest_rmse" in out["spatial_ood"] else "insufficient",
    )
    return out


def _run_first_difference_benchmark(
    panel: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    split_year: int,
    output_tag: str = "raw",
) -> Dict[str, Any]:
    """Predict dynamic changes with two tracks: raw differences and idiosyncratic residuals.

    Track A (raw) keeps the classic first-difference benchmark.
    Track B (idiosyncratic) removes country-year common shocks from differences:
    ΔY_idio = ΔY - E[ΔY | country, year].

    This addresses cross-level mismatch (city target vs country-level macro channels)
    and reports gain over a persistence baseline.
    """
    df = panel.sort_values(["city_id", "year"]).copy()
    df["_target_level"] = pd.to_numeric(df[target], errors="coerce")
    df["_target_delta"] = df.groupby("city_id")["_target_level"].diff()
    df["_target_delta_t1"] = df.groupby("city_id")["_target_delta"].shift(-1)
    cols = [c for c in feature_cols if c in df.columns]
    if "iso3" in df.columns:
        df["_target_delta_country_year_mean"] = df.groupby(["iso3", "year"])["_target_delta"].transform("mean")
    else:
        df["_target_delta_country_year_mean"] = df.groupby("year")["_target_delta"].transform("mean")
    df["_target_delta_cyres"] = df["_target_delta"] - df["_target_delta_country_year_mean"]
    df["_target_delta_cyres_t1"] = df.groupby("city_id")["_target_delta_cyres"].shift(-1)

    # Lag and city-history features for idiosyncratic dynamics (all shifted to avoid leakage).
    df["_target_level_l1"] = df.groupby("city_id")["_target_level"].shift(1)
    df["_target_delta_l1"] = df.groupby("city_id")["_target_delta"].shift(1)
    df["_target_delta_cyres_l1"] = df.groupby("city_id")["_target_delta_cyres"].shift(1)
    df["_target_delta_hist_mean"] = (
        df.groupby("city_id")["_target_delta"].expanding().mean().reset_index(level=0, drop=True).shift(1)
    )
    df["_target_delta_cyres_hist_mean"] = (
        df.groupby("city_id")["_target_delta_cyres"].expanding().mean().reset_index(level=0, drop=True).shift(1)
    )
    df["_target_delta_cyres_hist_std"] = (
        df.groupby("city_id")["_target_delta_cyres"].expanding().std(ddof=0).reset_index(level=0, drop=True).shift(1)
    )
    df["_target_delta_cyres_hist_std"] = pd.to_numeric(df["_target_delta_cyres_hist_std"], errors="coerce").fillna(0.0)

    # Detect near-static features for difference forecasting (e.g., static POI structure, lat/lon).
    variation_rows: List[Dict[str, Any]] = []
    dynamic_cols: List[str] = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        by_city_var = df.assign(_tmp=s).groupby("city_id")["_tmp"].var(ddof=0)
        var_arr = pd.to_numeric(by_city_var, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        share_zero = float(np.mean(np.isclose(var_arr, 0.0)))
        med_var = float(np.median(var_arr)) if len(var_arr) > 0 else float("nan")
        is_near_static = bool(share_zero >= 0.95 and med_var <= 1e-10)
        variation_rows.append(
            {
                "feature": col,
                "median_within_city_var": med_var,
                "share_city_zero_var": share_zero,
                "is_near_static_for_fd": int(is_near_static),
            }
        )
        if not is_near_static:
            dynamic_cols.append(col)

    if len(dynamic_cols) < 8:
        # Fall back to all resolved features when the dynamic subset is too small.
        dynamic_cols = cols.copy()
    tag = str(output_tag).strip().lower().replace(" ", "_")
    if tag in {"", "raw"}:
        variation_file = DATA_OUTPUTS / "benchmark_fd_feature_variation.csv"
        leaderboard_file = DATA_OUTPUTS / "benchmark_first_difference_board.csv"
    else:
        variation_file = DATA_OUTPUTS / f"benchmark_fd_feature_variation_{tag}.csv"
        leaderboard_file = DATA_OUTPUTS / f"benchmark_first_difference_board_{tag}.csv"

    variation_df = pd.DataFrame(variation_rows).sort_values(["is_near_static_for_fd", "share_city_zero_var"], ascending=[False, False])
    variation_df.to_csv(variation_file, index=False)

    # Feature sets per track.
    raw_features = [c for c in list(dict.fromkeys(dynamic_cols + ["_target_delta"])) if c in df.columns]
    idio_features = [
        c
        for c in list(
            dict.fromkeys(
                dynamic_cols
                + [
                    "_target_delta_cyres",
                    "_target_delta_country_year_mean",
                    "_target_delta_cyres_l1",
                    "_target_delta_l1",
                    "_target_level_l1",
                    "_target_delta_hist_mean",
                    "_target_delta_cyres_hist_mean",
                    "_target_delta_cyres_hist_std",
                ]
            )
        )
        if c in df.columns
    ]

    raw_df = df.dropna(subset=["_target_delta_t1"]).copy()
    idio_df = df.dropna(subset=["_target_delta_cyres_t1"]).copy()
    if len(raw_df) < 100 or len(idio_df) < 100:
        return {"status": "skipped", "reason": "too_few_rows_for_fd_benchmark"}
    train_raw = raw_df[raw_df["year"] <= split_year].copy()
    test_raw = raw_df[raw_df["year"] > split_year].copy()
    train_idio = idio_df[idio_df["year"] <= split_year].copy()
    test_idio = idio_df[idio_df["year"] > split_year].copy()
    if len(train_raw) < 50 or len(test_raw) < 20 or len(train_idio) < 50 or len(test_idio) < 20:
        return {"status": "skipped", "reason": "insufficient_train_test_for_fd"}

    fit_raw = _fit_predict(train_raw, test_raw, "_target_delta_t1", feature_cols=raw_features, random_seed=99)
    fit_idio = _fit_predict(
        train_idio,
        test_idio,
        "_target_delta_cyres_t1",
        feature_cols=idio_features,
        random_seed=123,
    )

    def _baseline_metrics(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, persistence_col: str) -> Dict[str, Dict[str, float]]:
        y_test = test_df[target_col].to_numpy(dtype=float)
        y_train = train_df[target_col].to_numpy(dtype=float)
        train_mean = float(np.nanmean(y_train)) if len(y_train) > 0 else 0.0
        pred_mean = np.repeat(train_mean, len(y_test))
        pred_persist = pd.to_numeric(test_df.get(persistence_col), errors="coerce").fillna(train_mean).to_numpy(dtype=float)
        return {
            "train_mean": _metric(y_test, pred_mean),
            "persistence": _metric(y_test, pred_persist),
        }

    raw_baselines = _baseline_metrics(train_raw, test_raw, "_target_delta_t1", "_target_delta")
    idio_baselines = _baseline_metrics(train_idio, test_idio, "_target_delta_cyres_t1", "_target_delta_cyres")

    fd_rows: List[Dict[str, Any]] = []
    for model, met in fit_raw["metrics"].items():
        fd_rows.append(
            {
                "task": "first_difference_holdout_raw",
                "model": model,
                "rmse": float(met["rmse"]),
                "mae": float(met["mae"]),
                "r2": float(met["r2"]),
                "n_train": int(len(train_raw)),
                "n_test": int(len(test_raw)),
                "feature_count": int(fit_raw.get("feature_count", len(raw_features))),
            }
        )
    for model, met in fit_idio["metrics"].items():
        fd_rows.append(
            {
                "task": "first_difference_holdout_idiosyncratic",
                "model": model,
                "rmse": float(met["rmse"]),
                "mae": float(met["mae"]),
                "r2": float(met["r2"]),
                "n_train": int(len(train_idio)),
                "n_test": int(len(test_idio)),
                "feature_count": int(fit_idio.get("feature_count", len(idio_features))),
            }
        )
    for bname, met in raw_baselines.items():
        fd_rows.append(
            {
                "task": "first_difference_holdout_raw",
                "model": f"baseline_{bname}",
                "rmse": float(met["rmse"]),
                "mae": float(met["mae"]),
                "r2": float(met["r2"]),
                "n_train": int(len(train_raw)),
                "n_test": int(len(test_raw)),
                "feature_count": 0,
            }
        )
    for bname, met in idio_baselines.items():
        fd_rows.append(
            {
                "task": "first_difference_holdout_idiosyncratic",
                "model": f"baseline_{bname}",
                "rmse": float(met["rmse"]),
                "mae": float(met["mae"]),
                "r2": float(met["r2"]),
                "n_train": int(len(train_idio)),
                "n_test": int(len(test_idio)),
                "feature_count": 0,
            }
        )
    fd_board = pd.DataFrame(fd_rows).sort_values(["task", "rmse", "model"])
    fd_board.to_csv(leaderboard_file, index=False)

    raw_best_r2 = max([float(v.get("r2", float("-inf"))) for v in fit_raw["metrics"].values()] or [float("nan")])
    idio_best_r2 = max([float(v.get("r2", float("-inf"))) for v in fit_idio["metrics"].values()] or [float("nan")])
    idio_persist_r2 = float((idio_baselines.get("persistence") or {}).get("r2", np.nan))

    return {
        "status": "ok",
        # Backward-compatible keys point to the raw-difference track.
        "metrics": fit_raw["metrics"],
        "model_status": fit_raw["model_status"],
        "n_train": int(len(train_raw)),
        "n_test": int(len(test_raw)),
        # New explicit two-track disclosure.
        "raw_track": {
            "metrics": fit_raw["metrics"],
            "model_status": fit_raw["model_status"],
            "baseline_metrics": raw_baselines,
            "n_train": int(len(train_raw)),
            "n_test": int(len(test_raw)),
            "feature_count": int(len(raw_features)),
        },
        "idiosyncratic_track": {
            "metrics": fit_idio["metrics"],
            "model_status": fit_idio["model_status"],
            "baseline_metrics": idio_baselines,
            "n_train": int(len(train_idio)),
            "n_test": int(len(test_idio)),
            "feature_count": int(len(idio_features)),
        },
        "best_r2_raw": raw_best_r2,
        "best_r2_idiosyncratic": idio_best_r2,
        "delta_best_r2_idio_minus_raw": float(idio_best_r2 - raw_best_r2),
        "delta_best_r2_idio_minus_persistence": float(idio_best_r2 - idio_persist_r2)
        if np.isfinite(idio_persist_r2)
        else None,
        "feature_variation_file": str(variation_file),
        "leaderboard_file": str(leaderboard_file),
    }


def _run_stall_threshold_sensitivity(
    panel: pd.DataFrame,
    target: str,
    quantiles: List[float] | None = None,
) -> Dict[str, Any]:
    """Evaluate stall-risk classifier performance across different quantile thresholds."""
    if quantiles is None:
        quantiles = [0.20, 0.25, 0.30, 0.35, 0.40]

    df = panel.sort_values(["city_id", "year"]).copy()
    df["_level"] = pd.to_numeric(df[target], errors="coerce")
    df["_delta"] = df.groupby("city_id")["_level"].diff()
    df["_delta_next"] = df.groupby("city_id")["_delta"].shift(-1)
    df = df.dropna(subset=["_delta", "_delta_next"])
    if len(df) < 100:
        return {"status": "skipped", "reason": "too_few_rows"}

    rows: List[Dict[str, Any]] = []
    for q in quantiles:
        threshold = float(df["_delta"].quantile(q))
        stall_label = (df["_delta_next"] <= threshold).astype(int).to_numpy()
        stall_rate = float(stall_label.mean())
        # Use negative delta as a simple predictor score
        scores = -df["_delta"].to_numpy()
        n_pos = int(stall_label.sum())
        n_neg = int(len(stall_label) - n_pos)
        if n_pos < 5 or n_neg < 5:
            auc = float("nan")
        else:
            correct = 0.0
            pos_scores = scores[stall_label == 1]
            neg_scores = scores[stall_label == 0]
            for sp in pos_scores:
                correct += float((neg_scores < sp).sum()) + 0.5 * float((neg_scores == sp).sum())
            auc = correct / (n_pos * n_neg)
        rows.append({
            "quantile": q,
            "threshold_value": threshold,
            "stall_rate": stall_rate,
            "n_stall": n_pos,
            "n_total": len(stall_label),
            "naive_delta_auc": auc,
        })

    sens_df = pd.DataFrame(rows)
    sens_df.to_csv(DATA_OUTPUTS / "stall_threshold_sensitivity.csv", index=False)

    return {
        "status": "ok",
        "quantiles_evaluated": len(quantiles),
        "results": rows,
        "output_file": str(DATA_OUTPUTS / "stall_threshold_sensitivity.csv"),
    }
