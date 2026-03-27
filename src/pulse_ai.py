from __future__ import annotations

"""AI engine for dynamic pulse acceleration/stall analysis."""

import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Reduce MKL/KMeans warning noise on Windows runtimes.
os.environ.setdefault("OMP_NUM_THREADS", "1")

from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .utils import DATA_OUTPUTS, dump_json, minmax_scale

LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS: List[str] = [
    "growth_1",
    "accel_1",
    "vol_3",
    "gdp_growth",
    "inflation",
    "clean_air_n",
    "basic_infra_n",
    "knowledge_delta_1",
    "digital_delta_1",
    "air_quality_improve_1",
    "observed_physical_stress_signal",
    "viirs_recent_drop",
    "viirs_intra_year_decline",
    "viirs_intra_year_recovery",
    "ghsl_built_surface_km2",
    "ghsl_built_density",
    "ghsl_built_surface_yoy",
    "ghsl_built_volume_yoy",
    "ghsl_built_contraction",
]

TRAJECTORY_REGIME_LABELS: List[str] = [
    "persistent_accelerator",
    "late_takeoff",
    "stable_mature",
    "volatile_rebound",
    "stalling_plateau",
    "structural_decline",
]

REGIME_TRANSITION_NUMERIC_FEATURES: List[str] = [
    "regime_run_length_log",
    "regime_switch_rate_3y",
    "regime_switch_rate_5y",
    "regime_year_share",
]

REGIME_NETWORK_NUMERIC_FEATURES: List[str] = [
    "regime_forward_risk",
    "regime_self_transition_prob",
    "regime_transition_entropy",
]


def _resolve_model_features(df: pd.DataFrame) -> List[str]:
    features = list(FEATURE_COLUMNS)
    for col in REGIME_TRANSITION_NUMERIC_FEATURES:
        if col in df.columns and col not in features:
            features.append(col)
    for col in REGIME_NETWORK_NUMERIC_FEATURES:
        if col in df.columns and col not in features:
            features.append(col)
    return features


def _fill_feature_medians(df: pd.DataFrame, cols: List[str], ref_df: pd.DataFrame | None = None) -> None:
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
        ref = df if ref_df is None or ref_df.empty or (col not in ref_df.columns) else ref_df
        med = float(ref[col].median()) if not pd.isna(ref[col].median()) else 0.0
        df[col] = df[col].fillna(med)


def _safe_slope(years: np.ndarray, values: np.ndarray) -> float:
    if len(years) < 2 or len(values) < 2:
        return 0.0
    try:
        return float(np.polyfit(years, values, deg=1)[0])
    except Exception:  # noqa: BLE001
        return 0.0


def _window_slope(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    if int(mask.sum()) < 2:
        return np.nan
    x = np.arange(len(arr), dtype=float)[mask]
    return _safe_slope(x, arr[mask])


def _rolling_lag1_autocorr(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3:
        return np.nan
    x = arr[:-1]
    y = arr[1:]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _prepare_dynamics_table(
    panel: pd.DataFrame,
    label_reference_end_year: int | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float | int | str | None]]:
    out = panel.copy().sort_values(["city_id", "year"]).reset_index(drop=True)

    out["lag_composite_1"] = out.groupby("city_id")["composite_index"].shift(1)
    out["lead_composite_1"] = out.groupby("city_id")["composite_index"].shift(-1)
    out["growth_1"] = out["composite_index"] - out["lag_composite_1"]
    out["growth_1_lag"] = out.groupby("city_id")["growth_1"].shift(1)
    out["accel_1"] = out["growth_1"] - out["growth_1_lag"]
    out["vol_3"] = out.groupby("city_id")["growth_1"].transform(lambda s: s.rolling(3, min_periods=2).std())
    out["momentum_2"] = out["composite_index"] - out.groupby("city_id")["composite_index"].shift(2)
    out["growth_next"] = out["lead_composite_1"] - out["composite_index"]

    out["global_rank_pct"] = out.groupby("year")["composite_index"].rank(method="average", pct=True)
    city_mean = out.groupby("city_id")["composite_index"].transform("mean")
    city_std = out.groupby("city_id")["composite_index"].transform("std").replace(0.0, np.nan)
    out["city_rel_score"] = ((out["composite_index"] - city_mean) / city_std).fillna(0.0)

    growth_pool = out["growth_next"].dropna()
    threshold_source = "full_sample_growth_next"
    threshold_source_end_year: int | None = None
    if label_reference_end_year is not None:
        ref = out.loc[out["year"] <= int(label_reference_end_year), "growth_next"].dropna()
        if len(ref) >= 80:
            growth_pool = ref
            threshold_source = "train_window_growth_next"
            threshold_source_end_year = int(label_reference_end_year)
        elif len(ref) > 0:
            threshold_source = "full_sample_growth_next_fallback_small_train_window"
            threshold_source_end_year = int(label_reference_end_year)
    if growth_pool.empty:
        low = -0.5
        high = 0.5
    else:
        low = float(np.quantile(growth_pool, 0.30))
        high = float(np.quantile(growth_pool, 0.70))

    out["stall_next"] = (out["growth_next"] <= low).astype(float)
    out["accelerate_next"] = (out["growth_next"] >= high).astype(float)

    accel_raw = (
        0.55 * out["growth_1"].fillna(0.0)
        + 0.30 * out["accel_1"].fillna(0.0)
        + 0.15 * out["momentum_2"].fillna(0.0)
        - 0.22 * out["vol_3"].fillna(out["vol_3"].median())
    )
    out["_accel_raw"] = accel_raw
    # Per-year MinMax to prevent future data leakage (consistent with global_data.py fix).
    out["acceleration_score"] = 100.0 * out.groupby("year")["_accel_raw"].transform(
        lambda x: minmax_scale(x.to_numpy())
    )

    heur_risk_raw = (
        -0.50 * out["growth_1"].fillna(0.0)
        - 0.20 * out["accel_1"].fillna(0.0)
        + 0.35 * out["vol_3"].fillna(out["vol_3"].median())
        + 0.16 * out["unemployment"].fillna(out["unemployment"].median())
        - 0.10 * out["internet_users"].fillna(out["internet_users"].median())
    )
    out["_heur_risk_raw"] = heur_risk_raw
    out["heuristic_stall_risk"] = 100.0 * out.groupby("year")["_heur_risk_raw"].transform(
        lambda x: minmax_scale(x.to_numpy())
    )
    out = out.drop(columns=["_accel_raw", "_heur_risk_raw"], errors="ignore")

    return out, {
        "stall_threshold": low,
        "accelerate_threshold": high,
        "threshold_source": threshold_source,
        "threshold_source_end_year": threshold_source_end_year,
        "threshold_reference_rows": int(len(growth_pool)),
    }


def _build_sample_weights(train_df: pd.DataFrame, mode: str) -> np.ndarray | None:
    if mode != "continent_year_balanced":
        return None
    if ("continent" not in train_df.columns) or ("year" not in train_df.columns):
        return None

    cont = train_df["continent"].fillna("unknown")
    year = train_df["year"]
    cont_counts = cont.value_counts()
    year_counts = year.value_counts()
    w_cont = cont.map(lambda c: 1.0 / max(float(cont_counts.get(c, 1.0)), 1.0))
    w_year = year.map(lambda y: 1.0 / max(float(year_counts.get(y, 1.0)), 1.0))
    w = np.sqrt(w_cont.to_numpy(dtype=float) * w_year.to_numpy(dtype=float))
    w = np.where(np.isfinite(w), w, 1.0)
    w_mean = float(np.mean(w)) if len(w) else 1.0
    if w_mean <= 1e-12:
        return np.ones(len(train_df), dtype=float)
    return w / w_mean


def _build_classifier(learner: str, random_state: int = 42):
    if learner == "gb":
        return GradientBoostingClassifier(
            random_state=random_state,
            n_estimators=220,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.90,
        )
    if learner == "logit":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=900,
                solver="liblinear",
                random_state=random_state,
                class_weight="balanced",
            ),
        )
    if learner == "hgb":
        return HistGradientBoostingClassifier(
            random_state=random_state,
            max_depth=5,
            max_iter=320,
            learning_rate=0.04,
            min_samples_leaf=20,
            l2_regularization=0.02,
        )
    if learner == "rf":
        return RandomForestClassifier(
            n_estimators=360,
            max_depth=8,
            min_samples_leaf=10,  # Raised from 3 to prevent overfitting on ~3k rows
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported learner: {learner}")


def _fit_classifier_with_weights(model, learner: str, x_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray | None):
    if sample_weight is None:
        model.fit(x_train, y_train)
        return
    if learner == "logit":
        model.fit(x_train, y_train, logisticregression__sample_weight=sample_weight)
        return
    model.fit(x_train, y_train, sample_weight=sample_weight)


def _fit_stall_model(
    data: pd.DataFrame,
    latest_year: int,
    feature_cols: List[str] | None = None,
    save_importance: bool = True,
    sample_weight_mode: str = "none",
) -> tuple[Any | None, Dict[str, Any], pd.DataFrame, List[str]]:
    model_df = data.dropna(subset=["growth_next"]).copy()
    if model_df.empty:
        return None, {"status": "skipped", "reason": "No valid target rows."}, model_df, []

    fit_features = list(feature_cols) if feature_cols else _resolve_model_features(model_df)
    if len(fit_features) == 0:
        return None, {"status": "skipped", "reason": "No valid features."}, model_df, []
    _fill_feature_medians(model_df, fit_features)

    # Three-way temporal split: train / validation / test
    # Validation is used for model selection; test is held out for final reporting only.
    effective_latest = int(model_df["year"].max())
    split_year = effective_latest - 1
    train_df = model_df[model_df["year"] <= split_year - 1].copy()
    val_df = model_df[model_df["year"] == split_year].copy()
    test_df = model_df[model_df["year"] > split_year].copy()

    if train_df.empty or val_df.empty:
        split_year = int(model_df["year"].quantile(0.60))
        train_df = model_df[model_df["year"] <= split_year].copy()
        val_df = model_df[(model_df["year"] > split_year) & (model_df["year"] <= split_year + 1)].copy()
        test_df = model_df[model_df["year"] > split_year + 1].copy()

    if train_df.empty or val_df.empty:
        return None, {"status": "skipped", "reason": "Not enough rows for temporal split."}, model_df, fit_features

    y_train = train_df["stall_next"].astype(int).to_numpy()
    y_val = val_df["stall_next"].astype(int).to_numpy()

    if len(np.unique(y_train)) < 2:
        return None, {"status": "skipped", "reason": "Training target has a single class."}, model_df, fit_features

    x_train = train_df[fit_features].to_numpy(dtype=float)
    x_val = val_df[fit_features].to_numpy(dtype=float)
    sample_weight = _build_sample_weights(train_df, sample_weight_mode)

    # --- Model selection on VALIDATION set ---
    learners = ["gb", "hgb", "rf", "logit"]
    learner_models: Dict[str, Any] = {}
    learner_probs_val: Dict[str, np.ndarray] = {}
    learner_metrics_val: Dict[str, Dict[str, Any]] = {}
    for learner in learners:
        clf = _build_classifier(learner, random_state=42)
        _fit_classifier_with_weights(clf, learner, x_train, y_train, sample_weight=sample_weight)
        p = clf.predict_proba(x_val)[:, 1]
        met = _calc_cls_metrics(y_val, p)
        auc = met.get("roc_auc")
        if auc is not None and np.isfinite(float(auc)):
            obj = float(auc) - 0.12 * float(met["brier"])
        else:
            obj = -float(met["brier"])
        met["objective"] = obj
        learner_models[learner] = clf
        learner_probs_val[learner] = p
        learner_metrics_val[learner] = met

    ranked_learners = sorted(
        learner_metrics_val.keys(),
        key=lambda k: (
            float(learner_metrics_val[k]["objective"]),
            float(learner_metrics_val[k]["roc_auc"]) if learner_metrics_val[k].get("roc_auc") is not None else -np.inf,
            -float(learner_metrics_val[k]["brier"]),
        ),
        reverse=True,
    )
    selected_learner = ranked_learners[0]
    model = learner_models[selected_learner]

    # --- Final reported metrics on held-out TEST set ---
    if not test_df.empty and len(np.unique(test_df["stall_next"].astype(int))) >= 2:
        x_test = test_df[fit_features].to_numpy(dtype=float)
        y_test = test_df["stall_next"].astype(int).to_numpy()
        p_test = model.predict_proba(x_test)[:, 1]
        selected_metrics = _calc_cls_metrics(y_test, p_test)
    else:
        # Fall back to validation metrics if no test data
        y_test = y_val
        p_test = learner_probs_val[selected_learner]
        selected_metrics = learner_metrics_val[selected_learner]

    metrics: Dict[str, Any] = {
        "status": "ok",
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "test_positive_rate": float(y_test.mean()),
        "brier": float(selected_metrics["brier"]),
        "feature_count": int(len(fit_features)),
        "feature_columns": fit_features,
        "sample_weight_mode": sample_weight_mode,
        "selected_learner": selected_learner,
        "learner_candidates": learner_metrics_val,
        "learner_objective": float(learner_metrics_val[selected_learner]["objective"]),
    }
    test_positive_rate = float(y_test.mean())
    ap = selected_metrics.get("average_precision")
    brier = float(selected_metrics["brier"])
    brier_ref = float(np.mean((y_test.astype(float) - test_positive_rate) ** 2))
    metrics["ap_baseline_positive_rate"] = test_positive_rate
    metrics["ap_lift_vs_positive_rate"] = (
        float(ap) / max(test_positive_rate, 1e-12)
        if ap is not None and np.isfinite(float(ap))
        else None
    )
    metrics["brier_baseline_prevalence"] = brier_ref
    metrics["brier_skill_vs_prevalence"] = (
        1.0 - (brier / brier_ref) if np.isfinite(brier_ref) and brier_ref > 1e-12 else None
    )
    metrics["roc_auc"] = float(selected_metrics["roc_auc"]) if selected_metrics.get("roc_auc") is not None else None
    metrics["average_precision"] = (
        float(selected_metrics["average_precision"]) if selected_metrics.get("average_precision") is not None else None
    )
    metrics["ece_10bin"] = float(selected_metrics["ece_10bin"]) if selected_metrics.get("ece_10bin") is not None else None
    metrics["mce_10bin"] = float(selected_metrics["mce_10bin"]) if selected_metrics.get("mce_10bin") is not None else None

    if save_importance:
        try:
            imp = permutation_importance(
                model,
                x_test,
                y_test,
                n_repeats=15,
                random_state=42,
                scoring="roc_auc" if len(np.unique(y_test)) > 1 else "neg_brier_score",
            )
            importance = pd.DataFrame({"feature": fit_features, "importance": imp.importances_mean})
            importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
            importance.to_csv(DATA_OUTPUTS / "pulse_ai_feature_importance.csv", index=False)
            metrics["top_features"] = importance.head(6).to_dict(orient="records")
        except Exception as exc:  # noqa: BLE001
            metrics["importance_status"] = f"failed: {exc}"

    return model, metrics, model_df, fit_features


def _build_city_archetypes(dyn: pd.DataFrame, latest_city: pd.DataFrame) -> pd.DataFrame:
    city_base = (
        dyn.sort_values(["city_id", "year"])
        .groupby(["city_id", "city_name", "country", "continent"], as_index=False)
        .agg(
            mean_growth=("growth_1", "mean"),
            vol_growth=("growth_1", "std"),
            mean_acceleration=("acceleration_score", "mean"),
            last_composite=("composite_index", "last"),
        )
    )

    slope_rows: List[Dict[str, Any]] = []
    for city_id, grp in dyn.groupby("city_id"):
        slope_rows.append(
            {
                "city_id": city_id,
                "trend_slope": _safe_slope(grp["year"].to_numpy(), grp["composite_index"].to_numpy()),
            }
        )
    slopes = pd.DataFrame(slope_rows)
    city_base = city_base.merge(slopes, on="city_id", how="left")
    city_base = city_base.merge(latest_city[["city_id", "stall_risk_score"]], on="city_id", how="left")

    feat_cols = ["mean_growth", "vol_growth", "mean_acceleration", "trend_slope", "stall_risk_score"]
    for col in feat_cols:
        med = float(city_base[col].median()) if not pd.isna(city_base[col].median()) else 0.0
        city_base[col] = city_base[col].fillna(med)

    n_clusters = min(4, len(city_base))
    if n_clusters < 2:
        city_base["archetype"] = "single_cluster"
        return city_base

    scaler = StandardScaler()
    z = scaler.fit_transform(city_base[feat_cols])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=25)
    city_base["cluster_id"] = kmeans.fit_predict(z)

    cluster_stats = city_base.groupby("cluster_id", as_index=False).agg(
        mean_growth=("mean_growth", "mean"),
        vol_growth=("vol_growth", "mean"),
        mean_acceleration=("mean_acceleration", "mean"),
        stall_risk_score=("stall_risk_score", "mean"),
        trend_slope=("trend_slope", "mean"),
    )
    cluster_stats["quality"] = (
        cluster_stats["mean_growth"]
        + 0.45 * cluster_stats["trend_slope"]
        + 0.02 * cluster_stats["mean_acceleration"]
        - 0.80 * cluster_stats["vol_growth"]
        - 0.02 * cluster_stats["stall_risk_score"]
    )
    rank = cluster_stats.sort_values("quality", ascending=False)["cluster_id"].tolist()
    labels = [
        "frontier_accelerator",
        "resilient_steady",
        "fragile_rebound",
        "structural_stall",
    ]
    mapping = {cid: labels[i] for i, cid in enumerate(rank)}
    city_base["archetype"] = city_base["cluster_id"].map(mapping).fillna("mixed_transition")
    return city_base


def _dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray, window: int) -> float:
    n = len(seq_a)
    m = len(seq_b)
    if n == 0 or m == 0:
        return 0.0

    w = max(window, abs(n - m))
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m, i + w)
        for j in range(j_start, j_end + 1):
            cost = abs(float(seq_a[i - 1]) - float(seq_b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def _pairwise_dtw(seqs: np.ndarray, window_ratio: float = 0.4) -> np.ndarray:
    n = seqs.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    window = max(1, int(round(seqs.shape[1] * window_ratio)))
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = _dtw_distance(seqs[i], seqs[j], window=window)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _kmedoids_from_distance(
    dist: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    max_iter: int = 80,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = dist.shape[0]
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int), float("nan")
    if n_clusters >= n:
        labels = np.arange(n, dtype=int)
        return labels, labels.copy(), 0.0

    rng = np.random.default_rng(random_state)
    medoids = np.sort(rng.choice(n, size=n_clusters, replace=False))
    labels = np.argmin(dist[:, medoids], axis=1)

    for _ in range(max_iter):
        old_labels = labels.copy()
        changed = False
        for k in range(n_clusters):
            members = np.where(labels == k)[0]
            if len(members) == 0:
                candidates = np.array([x for x in range(n) if x not in medoids], dtype=int)
                if len(candidates) > 0:
                    medoids[k] = int(rng.choice(candidates))
                    changed = True
                continue

            sub = dist[np.ix_(members, members)]
            costs = sub.sum(axis=1)
            best_member = int(members[int(np.argmin(costs))])
            if best_member != medoids[k]:
                medoids[k] = best_member
                changed = True

        labels = np.argmin(dist[:, medoids], axis=1)
        if np.array_equal(labels, old_labels) and not changed:
            break

    objective = float(np.sum([dist[i, medoids[labels[i]]] for i in range(n)]))
    return labels, medoids, objective


def _silhouette_from_distance(dist: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n = len(labels)
    if n == 0:
        return np.array([], dtype=float)

    out = np.zeros(n, dtype=float)
    unique_labels = sorted(set(labels.tolist()))
    for i in range(n):
        same = np.where(labels == labels[i])[0]
        same_wo_i = same[same != i]
        if len(same_wo_i) == 0:
            out[i] = 0.0
            continue
        a = float(np.mean(dist[i, same_wo_i]))
        b_vals = []
        for lb in unique_labels:
            if lb == labels[i]:
                continue
            other = np.where(labels == lb)[0]
            if len(other) == 0:
                continue
            b_vals.append(float(np.mean(dist[i, other])))
        b = min(b_vals) if b_vals else a
        out[i] = (b - a) / max(a, b, 1e-12)
    return out


def _trajectory_sequences(dyn: pd.DataFrame) -> tuple[List[str], List[int], np.ndarray]:
    pivot = dyn.pivot_table(index="city_id", columns="year", values="composite_index").sort_index()
    pivot = pivot.interpolate(axis=1, limit_direction="both").ffill(axis=1).bfill(axis=1)

    city_ids = pivot.index.astype(str).tolist()
    years = [int(y) for y in pivot.columns.tolist()]
    y = pivot.to_numpy(dtype=float)
    if y.size == 0:
        return city_ids, years, np.zeros((0, 0), dtype=float)

    centered = y - y[:, [0]]
    scale = np.std(centered, axis=1, keepdims=True)
    scale = np.where(scale <= 1e-8, 1.0, scale)
    level = centered / scale
    growth = np.diff(level, axis=1, prepend=level[:, [0]])
    seq = 0.72 * level + 0.28 * growth
    return city_ids, years, seq


def _zscore(arr: pd.Series) -> pd.Series:
    std = float(arr.std())
    if std <= 1e-12 or not np.isfinite(std):
        return pd.Series(np.zeros(len(arr)), index=arr.index)
    return (arr - float(arr.mean())) / std


def _calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float | None, float | None]:
    if len(y_true) == 0:
        return None, None
    frame = pd.DataFrame({"y": y_true.astype(float), "p": y_prob.astype(float)}).dropna()
    if frame.empty:
        return None, None
    try:
        frame["bin"] = pd.qcut(frame["p"], q=n_bins, duplicates="drop")
    except Exception:  # noqa: BLE001
        frame["bin"] = pd.cut(frame["p"], bins=n_bins)
    grouped = (
        frame.groupby("bin", as_index=False, observed=False)
        .agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
        .copy()
    )
    if grouped.empty:
        return None, None
    grouped["gap"] = np.abs(grouped["obs"] - grouped["pred"])
    total_n = float(grouped["n"].sum())
    ece = float((grouped["gap"] * grouped["n"]).sum() / max(total_n, 1.0))
    mce = float(grouped["gap"].max())
    return ece, mce


def _calc_cls_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float | None]:
    ece, mce = _calibration_error(y_true, y_prob, n_bins=10)
    out: Dict[str, float | None] = {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "roc_auc": None,
        "average_precision": None,
        "ece_10bin": ece,
        "mce_10bin": mce,
    }
    if np.unique(y_true).size >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["average_precision"] = float(average_precision_score(y_true, y_prob))
    return out


def _bootstrap_metric_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: str,
    n_boot: int = 300,
    random_state: int = 42,
) -> Dict[str, Any]:
    y = y_true.astype(int)
    pa = pred_a.astype(float)
    pb = pred_b.astype(float)
    n = len(y)
    if n == 0:
        return {"status": "skipped", "reason": "empty_sample"}

    def _score(yv: np.ndarray, pv: np.ndarray) -> float | None:
        if metric == "roc_auc":
            if np.unique(yv).size < 2:
                return None
            return float(roc_auc_score(yv, pv))
        if metric == "brier":
            return float(brier_score_loss(yv, pv))
        raise ValueError(f"Unsupported metric for bootstrap diff: {metric}")

    score_a = _score(y, pa)
    score_b = _score(y, pb)
    if score_a is None or score_b is None:
        return {"status": "skipped", "reason": "metric_not_identified"}

    if metric == "roc_auc":
        point = float(score_a - score_b)
    else:
        # Positive means model A has lower error than model B.
        point = float(score_b - score_a)

    rng = np.random.default_rng(random_state)
    draws: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        sa = _score(yb, pa[idx])
        sb = _score(yb, pb[idx])
        if sa is None or sb is None:
            continue
        draws.append(float(sa - sb) if metric == "roc_auc" else float(sb - sa))

    if len(draws) < max(50, int(0.2 * n_boot)):
        return {
            "status": "skipped",
            "reason": "insufficient_bootstrap_draws",
            "point_estimate": point,
            "n_draws": int(len(draws)),
        }

    arr = np.array(draws, dtype=float)
    ci_low = float(np.quantile(arr, 0.025))
    ci_high = float(np.quantile(arr, 0.975))
    p_two_sided = float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))
    p_two_sided = min(max(p_two_sided, 0.0), 1.0)
    return {
        "status": "ok",
        "point_estimate": point,
        "ci95": [ci_low, ci_high],
        "p_value_two_sided": p_two_sided,
        "n_draws": int(len(arr)),
    }


def _append_regime_transition_features(dyn: pd.DataFrame) -> pd.DataFrame:
    out = dyn.sort_values(["city_id", "year"]).copy()
    if "trajectory_regime" not in out.columns:
        out["trajectory_regime"] = "mixed_transition"
    out["trajectory_regime"] = out["trajectory_regime"].fillna("mixed_transition").astype(str)

    prev_regime = out.groupby("city_id")["trajectory_regime"].shift(1)
    out["regime_switch"] = (out["trajectory_regime"] != prev_regime).astype(float)
    out.loc[prev_regime.isna(), "regime_switch"] = 0.0

    out["regime_block"] = out.groupby("city_id")["regime_switch"].cumsum()
    out["regime_run_length"] = out.groupby(["city_id", "regime_block"]).cumcount() + 1
    out["regime_run_length_log"] = np.log1p(out["regime_run_length"].astype(float))

    out["regime_switch_rate_3y"] = out.groupby("city_id")["regime_switch"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    out["regime_switch_rate_5y"] = out.groupby("city_id")["regime_switch"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    out["years_seen"] = out.groupby("city_id").cumcount() + 1
    out["regime_seen_count"] = out.groupby(["city_id", "trajectory_regime"]).cumcount() + 1
    out["regime_year_share"] = out["regime_seen_count"] / out["years_seen"]

    for col in REGIME_TRANSITION_NUMERIC_FEATURES:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out = out.drop(columns=["regime_block", "regime_run_length", "years_seen", "regime_seen_count"], errors="ignore")
    return out


def _build_regime_network_features(dyn: pd.DataFrame, regime_year: pd.DataFrame) -> pd.DataFrame:
    out = dyn.sort_values(["city_id", "year"]).copy()
    if "trajectory_regime" not in out.columns:
        out["trajectory_regime"] = "mixed_transition"
    out["trajectory_regime"] = out["trajectory_regime"].fillna("mixed_transition").astype(str)

    global_stall = float(out["stall_next"].dropna().mean()) if out["stall_next"].notna().any() else 0.5
    regime_stall = (
        out.dropna(subset=["stall_next"])
        .groupby("trajectory_regime")["stall_next"]
        .mean()
        .to_dict()
    )

    trans_rows: List[Dict[str, str]] = []
    if not regime_year.empty and {"city_id", "year", "trajectory_regime"}.issubset(regime_year.columns):
        ry = regime_year.sort_values(["city_id", "year"])[["city_id", "year", "trajectory_regime"]].copy()
        ry["trajectory_regime"] = ry["trajectory_regime"].fillna("mixed_transition").astype(str)
        for _, grp in ry.groupby("city_id"):
            vals = grp["trajectory_regime"].tolist()
            for i in range(len(vals) - 1):
                trans_rows.append({"from_regime": vals[i], "to_regime": vals[i + 1]})

    trans_df = pd.DataFrame(trans_rows)
    if trans_df.empty:
        out["regime_forward_risk"] = out["trajectory_regime"].map(regime_stall).fillna(global_stall).astype(float)
        out["regime_self_transition_prob"] = 0.0
        out["regime_transition_entropy"] = 0.0
        return out

    trans_df = (
        trans_df.groupby(["from_regime", "to_regime"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    trans_df["probability"] = trans_df["count"] / trans_df.groupby("from_regime")["count"].transform("sum")

    self_prob = (
        trans_df[trans_df["from_regime"] == trans_df["to_regime"]]
        .set_index("from_regime")["probability"]
        .to_dict()
    )

    entropy_map: Dict[str, float] = {}
    forward_map: Dict[str, float] = {}
    for from_regime, grp in trans_df.groupby("from_regime"):
        probs = grp["probability"].to_numpy(dtype=float)
        probs = probs[probs > 1e-12]
        entropy_map[str(from_regime)] = float(-np.sum(probs * np.log(probs))) if probs.size > 0 else 0.0

        exp_risk = 0.0
        for row in grp.itertuples(index=False):
            to_reg = str(row.to_regime)
            p = float(row.probability)
            exp_risk += p * float(regime_stall.get(to_reg, global_stall))
        forward_map[str(from_regime)] = exp_risk

    out["regime_forward_risk"] = out["trajectory_regime"].map(forward_map).fillna(global_stall).astype(float)
    out["regime_self_transition_prob"] = out["trajectory_regime"].map(self_prob).fillna(0.0).astype(float)
    out["regime_transition_entropy"] = out["trajectory_regime"].map(entropy_map).fillna(0.0).astype(float)

    for col in REGIME_NETWORK_NUMERIC_FEATURES:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _build_model_feature_variants(df: pd.DataFrame) -> Dict[str, List[str]]:
    base = list(FEATURE_COLUMNS)
    transition = [c for c in REGIME_TRANSITION_NUMERIC_FEATURES if c in df.columns]
    network = [c for c in REGIME_NETWORK_NUMERIC_FEATURES if c in df.columns]

    raw_variants: List[tuple[str, List[str]]] = [
        ("base", base),
        ("base_transition", base + transition),
        ("base_network", base + network),
        ("full_pulse_graph", base + transition + network),
    ]
    variants: Dict[str, List[str]] = {}
    seen_signatures: set[tuple[str, ...]] = set()
    for name, cols in raw_variants:
        dedup = list(dict.fromkeys(cols))
        sig = tuple(dedup)
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        variants[name] = dedup
    return variants


def _metric_objective(metrics: Dict[str, Any]) -> float:
    auc = metrics.get("roc_auc")
    if auc is not None and np.isfinite(float(auc)):
        return float(auc)
    brier = metrics.get("brier")
    if brier is not None and np.isfinite(float(brier)):
        return -float(brier)
    return -np.inf


def _variant_spatial_objective(
    dyn: pd.DataFrame,
    feature_cols: List[str],
    latest_year: int,
    learner: str = "gb",
) -> Dict[str, Any]:
    df = dyn.dropna(subset=["stall_next", "growth_next", "continent"]).copy()
    if df.empty:
        return {"status": "skipped", "reason": "no_valid_rows"}

    eval_year = latest_year - 1
    if eval_year not in set(df["year"].unique().tolist()):
        eval_year = int(df["year"].max())

    _fill_feature_medians(df, feature_cols)
    rows: List[Dict[str, Any]] = []
    for cont in sorted(df["continent"].dropna().astype(str).unique().tolist()):
        train = df[(df["continent"] != cont) & (df["year"] <= (eval_year - 1))].copy()
        test = df[(df["continent"] == cont) & (df["year"] == eval_year)].copy()
        if len(test) < 20:
            test = df[(df["continent"] == cont) & (df["year"] >= (eval_year - 1))].copy()
        if len(train) < 120 or len(test) < 20:
            continue

        y_train = train["stall_next"].astype(int).to_numpy()
        y_test = test["stall_next"].astype(int).to_numpy()
        if np.unique(y_train).size < 2:
            continue

        x_train = train[feature_cols].to_numpy(dtype=float)
        x_test = test[feature_cols].to_numpy(dtype=float)
        clf = _build_classifier(learner, random_state=42)
        _fit_classifier_with_weights(clf, learner, x_train, y_train, sample_weight=None)
        p = clf.predict_proba(x_test)[:, 1]
        met = _calc_cls_metrics(y_test, p)
        rows.append(
            {
                "left_out_continent": cont,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "positive_rate_test": float(y_test.mean()),
                "roc_auc": met.get("roc_auc"),
                "average_precision": met.get("average_precision"),
                "brier": met.get("brier"),
            }
        )

    if not rows:
        return {"status": "skipped", "reason": "insufficient_continent_pairs"}

    out = pd.DataFrame(rows)
    auc_vals = out["roc_auc"].dropna()
    return {
        "status": "ok",
        "learner": learner,
        "n_continents": int(out["left_out_continent"].nunique()),
        "mean_roc_auc": float(auc_vals.mean()) if not auc_vals.empty else None,
        "mean_brier": float(out["brier"].mean()),
    }


def _pulse_cross_continent_generalization(
    dyn: pd.DataFrame,
    feature_cols: List[str],
    latest_year: int,
    primary_learner: str = "gb",
) -> Dict[str, Any]:
    df = dyn.dropna(subset=["stall_next", "growth_next", "continent"]).copy()
    if df.empty:
        empty = pd.DataFrame(
            columns=[
                "left_out_continent",
                "n_train",
                "n_test",
                "positive_rate_test",
                "roc_auc",
                "average_precision",
                "brier",
            ]
        )
        empty.to_csv(DATA_OUTPUTS / "pulse_ai_cross_continent_generalization.csv", index=False)
        return {"status": "skipped", "reason": "no_valid_rows"}

    eval_year = latest_year - 1
    if eval_year not in set(df["year"].unique().tolist()):
        eval_year = int(df["year"].max())

    _fill_feature_medians(df, feature_cols)
    baseline_learner = "logit" if primary_learner != "logit" else "gb"
    rows: List[Dict[str, Any]] = []
    for cont in sorted(df["continent"].dropna().astype(str).unique().tolist()):
        train = df[(df["continent"] != cont) & (df["year"] <= (eval_year - 1))].copy()
        test = df[(df["continent"] == cont) & (df["year"] == eval_year)].copy()
        if len(test) < 20:
            test = df[(df["continent"] == cont) & (df["year"] >= (eval_year - 1))].copy()
        if len(train) < 120 or len(test) < 20:
            continue

        y_train = train["stall_next"].astype(int).to_numpy()
        y_test = test["stall_next"].astype(int).to_numpy()
        if np.unique(y_train).size < 2:
            continue

        x_train = train[feature_cols].to_numpy(dtype=float)
        x_test = test[feature_cols].to_numpy(dtype=float)
        model_primary = _build_classifier(primary_learner, random_state=42)
        _fit_classifier_with_weights(model_primary, primary_learner, x_train, y_train, sample_weight=None)
        p_model = model_primary.predict_proba(x_test)[:, 1]
        met_model = _calc_cls_metrics(y_test, p_model)

        model_base = _build_classifier(baseline_learner, random_state=42)
        _fit_classifier_with_weights(model_base, baseline_learner, x_train, y_train, sample_weight=None)
        p_base = model_base.predict_proba(x_test)[:, 1]
        met_logit = _calc_cls_metrics(y_test, p_base)

        auc_delta = None
        if met_model.get("roc_auc") is not None and met_logit.get("roc_auc") is not None:
            auc_delta = float(met_model["roc_auc"] - met_logit["roc_auc"])
        brier_gain = float(met_logit["brier"] - met_model["brier"])

        auc_sig = _bootstrap_metric_diff(
            y_test,
            p_model,
            p_base,
            metric="roc_auc",
            n_boot=320,
            random_state=42 + len(rows),
        )
        brier_sig = _bootstrap_metric_diff(
            y_test,
            p_model,
            p_base,
            metric="brier",
            n_boot=320,
            random_state=142 + len(rows),
        )

        rows.append(
            {
                "left_out_continent": cont,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "positive_rate_test": float(y_test.mean()),
                "roc_auc": met_model.get("roc_auc"),
                "average_precision": met_model.get("average_precision"),
                "brier": met_model.get("brier"),
                "baseline_roc_auc": met_logit.get("roc_auc"),
                "baseline_average_precision": met_logit.get("average_precision"),
                "baseline_brier": met_logit.get("brier"),
                "delta_roc_auc_vs_logit": auc_delta,
                "delta_brier_improve_vs_logit": brier_gain,
                "delta_roc_auc_ci_low": (auc_sig.get("ci95") or [None, None])[0] if auc_sig.get("status") == "ok" else None,
                "delta_roc_auc_ci_high": (auc_sig.get("ci95") or [None, None])[1] if auc_sig.get("status") == "ok" else None,
                "delta_roc_auc_p_value": auc_sig.get("p_value_two_sided") if auc_sig.get("status") == "ok" else None,
                "delta_brier_improve_ci_low": (brier_sig.get("ci95") or [None, None])[0]
                if brier_sig.get("status") == "ok"
                else None,
                "delta_brier_improve_ci_high": (brier_sig.get("ci95") or [None, None])[1]
                if brier_sig.get("status") == "ok"
                else None,
                "delta_brier_improve_p_value": brier_sig.get("p_value_two_sided")
                if brier_sig.get("status") == "ok"
                else None,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(DATA_OUTPUTS / "pulse_ai_cross_continent_generalization.csv", index=False)
    if out.empty:
        return {"status": "skipped", "reason": "insufficient_continent_pairs"}

    auc_vals = out["roc_auc"].dropna()
    base_auc_vals = out["baseline_roc_auc"].dropna() if "baseline_roc_auc" in out.columns else pd.Series(dtype=float)
    delta_auc_vals = out["delta_roc_auc_vs_logit"].dropna() if "delta_roc_auc_vs_logit" in out.columns else pd.Series(dtype=float)
    delta_brier_vals = (
        out["delta_brier_improve_vs_logit"].dropna() if "delta_brier_improve_vs_logit" in out.columns else pd.Series(dtype=float)
    )
    if not auc_vals.empty:
        hardest = out[out["roc_auc"] == float(auc_vals.min())].iloc[0]
        best = out[out["roc_auc"] == float(auc_vals.max())].iloc[0]
    else:
        hardest = out.sort_values("brier", ascending=False).iloc[0]
        best = out.sort_values("brier", ascending=True).iloc[0]

    return {
        "status": "ok",
        "primary_learner": primary_learner,
        "baseline_learner": baseline_learner,
        "eval_year": int(eval_year),
        "n_continents": int(out["left_out_continent"].nunique()),
        "mean_roc_auc": float(auc_vals.mean()) if not auc_vals.empty else None,
        "std_roc_auc": float(auc_vals.std()) if len(auc_vals) > 1 else None,
        "mean_average_precision": float(out["average_precision"].dropna().mean()) if out["average_precision"].notna().any() else None,
        "mean_brier": float(out["brier"].mean()),
        "mean_baseline_roc_auc": float(base_auc_vals.mean()) if not base_auc_vals.empty else None,
        "mean_delta_roc_auc_vs_logit": float(delta_auc_vals.mean()) if not delta_auc_vals.empty else None,
        "mean_delta_brier_improve_vs_logit": float(delta_brier_vals.mean()) if not delta_brier_vals.empty else None,
        "share_continents_auc_gain_vs_logit": float(np.mean(delta_auc_vals > 0.0)) if not delta_auc_vals.empty else None,
        "share_continents_auc_gain_p_lt_0_10": float(np.mean(out["delta_roc_auc_p_value"].dropna() < 0.10))
        if out["delta_roc_auc_p_value"].notna().any()
        else None,
        "hardest_continent": {
            "continent": str(hardest["left_out_continent"]),
            "roc_auc": float(hardest["roc_auc"]) if pd.notna(hardest["roc_auc"]) else None,
            "brier": float(hardest["brier"]),
        },
        "best_continent": {
            "continent": str(best["left_out_continent"]),
            "roc_auc": float(best["roc_auc"]) if pd.notna(best["roc_auc"]) else None,
            "brier": float(best["brier"]),
        },
    }


def _estimate_shock_pulse_response(dyn: pd.DataFrame, latest_year: int) -> Dict[str, Any]:
    year_growth = (
        dyn.groupby("year", as_index=False)
        .agg(mean_growth=("growth_1", "mean"), mean_accel=("accel_1", "mean"), city_count=("city_id", "nunique"))
        .dropna(subset=["mean_growth"])
        .sort_values("year")
    )
    if year_growth.empty:
        pd.DataFrame(columns=["shock_year", "shock_strength", "mean_growth", "mean_accel"]).to_csv(
            DATA_OUTPUTS / "pulse_ai_shock_years.csv", index=False
        )
        pd.DataFrame(
            columns=["shock_year", "event_time", "trajectory_regime", "n_obs", "mean_rel_change", "mean_growth"]
        ).to_csv(DATA_OUTPUTS / "pulse_ai_shock_irf_regime.csv", index=False)
        pd.DataFrame(
            columns=["shock_year", "event_time", "vulnerability_group", "n_obs", "mean_rel_change", "mean_growth"]
        ).to_csv(DATA_OUTPUTS / "pulse_ai_shock_irf_vulnerability.csv", index=False)
        return {"status": "skipped", "reason": "insufficient_growth_series"}

    mg = year_growth["mean_growth"].to_numpy(dtype=float)
    if float(np.std(mg)) <= 1e-12:
        z = np.zeros_like(mg)
    else:
        z = (mg - float(np.mean(mg))) / float(np.std(mg))
    year_growth["shock_strength"] = -z

    valid_shocks = year_growth[
        (year_growth["year"] >= int(year_growth["year"].min()) + 1)
        & (year_growth["year"] <= int(latest_year) - 1)
    ].copy()
    if valid_shocks.empty:
        valid_shocks = year_growth.copy()

    ranked = valid_shocks.sort_values(["mean_growth", "shock_strength"], ascending=[True, False]).copy()
    shock_years: List[int] = []
    if 2020 in set(ranked["year"].tolist()):
        shock_years.append(2020)
    for y in ranked["year"].astype(int).tolist():
        if y in shock_years:
            continue
        if any(abs(y - prev) <= 1 for prev in shock_years):
            continue
        shock_years.append(y)
        if len(shock_years) >= 3:
            break
    shock_years = sorted(shock_years)

    shocks_df = year_growth[year_growth["year"].isin(shock_years)][["year", "shock_strength", "mean_growth", "mean_accel"]].copy()
    shocks_df = shocks_df.rename(columns={"year": "shock_year"}).sort_values("shock_year")
    shocks_df.to_csv(DATA_OUTPUTS / "pulse_ai_shock_years.csv", index=False)

    irf_regime_rows: List[Dict[str, Any]] = []
    irf_vul_rows: List[Dict[str, Any]] = []
    for sy in shock_years:
        base = dyn[dyn["year"] == (sy - 1)][
            ["city_id", "composite_index", "stall_risk_score", "trajectory_regime", "growth_1"]
        ].copy()
        if len(base) < 20:
            continue
        base = base.rename(
            columns={
                "composite_index": "base_composite",
                "stall_risk_score": "base_stall_risk",
                "trajectory_regime": "base_regime",
                "growth_1": "base_growth",
            }
        )
        q_hi = float(base["base_stall_risk"].quantile(0.70))
        q_lo = float(base["base_stall_risk"].quantile(0.30))
        base["vulnerability_group"] = np.where(
            base["base_stall_risk"] >= q_hi,
            "high_risk",
            np.where(base["base_stall_risk"] <= q_lo, "low_risk", "mid_risk"),
        )

        window = dyn[(dyn["year"] >= (sy - 2)) & (dyn["year"] <= (sy + 2))][
            ["city_id", "year", "composite_index", "growth_1", "trajectory_regime"]
        ].copy()
        merged = window.merge(base, on="city_id", how="inner")
        merged["event_time"] = merged["year"] - sy
        merged["rel_change"] = merged["composite_index"] - merged["base_composite"]

        reg = (
            merged.groupby(["event_time", "trajectory_regime"], as_index=False)
            .agg(
                n_obs=("city_id", "size"),
                mean_rel_change=("rel_change", "mean"),
                mean_growth=("growth_1", "mean"),
            )
            .sort_values(["event_time", "n_obs"], ascending=[True, False])
        )
        for row in reg.itertuples(index=False):
            irf_regime_rows.append(
                {
                    "shock_year": int(sy),
                    "event_time": int(row.event_time),
                    "trajectory_regime": str(row.trajectory_regime),
                    "n_obs": int(row.n_obs),
                    "mean_rel_change": float(row.mean_rel_change),
                    "mean_growth": float(row.mean_growth) if pd.notna(row.mean_growth) else np.nan,
                }
            )

        vul = (
            merged[merged["vulnerability_group"] != "mid_risk"]
            .groupby(["event_time", "vulnerability_group"], as_index=False)
            .agg(
                n_obs=("city_id", "size"),
                mean_rel_change=("rel_change", "mean"),
                mean_growth=("growth_1", "mean"),
            )
            .sort_values(["event_time", "vulnerability_group"])
        )
        for row in vul.itertuples(index=False):
            irf_vul_rows.append(
                {
                    "shock_year": int(sy),
                    "event_time": int(row.event_time),
                    "vulnerability_group": str(row.vulnerability_group),
                    "n_obs": int(row.n_obs),
                    "mean_rel_change": float(row.mean_rel_change),
                    "mean_growth": float(row.mean_growth) if pd.notna(row.mean_growth) else np.nan,
                }
            )

    irf_regime = pd.DataFrame(irf_regime_rows)
    irf_vul = pd.DataFrame(irf_vul_rows)
    irf_regime.to_csv(DATA_OUTPUTS / "pulse_ai_shock_irf_regime.csv", index=False)
    irf_vul.to_csv(DATA_OUTPUTS / "pulse_ai_shock_irf_vulnerability.csv", index=False)

    if irf_regime.empty:
        return {"status": "skipped", "reason": "insufficient_irf_rows", "shock_years": shock_years}

    t1 = irf_regime[irf_regime["event_time"] == 1].copy()
    if not t1.empty:
        worst = t1.sort_values("mean_rel_change", ascending=True).iloc[0]
        best = t1.sort_values("mean_rel_change", ascending=False).iloc[0]
    else:
        worst = irf_regime.sort_values("mean_rel_change", ascending=True).iloc[0]
        best = irf_regime.sort_values("mean_rel_change", ascending=False).iloc[0]

    vul_gap_t1 = None
    if not irf_vul.empty:
        v1 = irf_vul[irf_vul["event_time"] == 1]
        if not v1.empty:
            high = v1[v1["vulnerability_group"] == "high_risk"]["mean_rel_change"]
            low = v1[v1["vulnerability_group"] == "low_risk"]["mean_rel_change"]
            if len(high) > 0 and len(low) > 0:
                vul_gap_t1 = float(high.mean() - low.mean())

    return {
        "status": "ok",
        "shock_years": shock_years,
        "n_shocks": int(len(shock_years)),
        "worst_regime_t1": {
            "regime": str(worst["trajectory_regime"]),
            "mean_rel_change": float(worst["mean_rel_change"]),
            "shock_year": int(worst["shock_year"]),
        },
        "best_regime_t1": {
            "regime": str(best["trajectory_regime"]),
            "mean_rel_change": float(best["mean_rel_change"]),
            "shock_year": int(best["shock_year"]),
        },
        "high_minus_low_vulnerability_gap_t1": vul_gap_t1,
    }


def _cls_objective(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, Dict[str, float | None]]:
    met = _calc_cls_metrics(y_true, y_prob)
    auc = met.get("roc_auc")
    brier = float(met.get("brier", 1.0))
    if auc is not None and np.isfinite(float(auc)):
        score = float(auc) - 0.15 * brier
    else:
        score = -brier
    return score, met


def _fit_platt_calibrator(y_true: np.ndarray, p_raw: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_raw, dtype=float)
    valid = np.isfinite(p)
    y = y[valid]
    p = p[valid]
    if len(y) < 120 or np.unique(y).size < 2:
        return {"status": "skipped", "reason": "insufficient_rows_or_single_class"}
    eps = 1e-5
    p = np.clip(p, eps, 1.0 - eps)
    z = np.log(p / (1.0 - p))
    if not np.isfinite(z).all() or float(np.std(z)) <= 1e-10:
        return {"status": "skipped", "reason": "degenerate_logit_input"}
    lr = LogisticRegression(
        max_iter=1500,
        C=1.2,
        solver="lbfgs",
        class_weight="balanced",
        random_state=2089,
    )
    try:
        lr.fit(z.reshape(-1, 1), y)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "reason": str(exc)}

    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])
    if not np.isfinite(slope) or slope <= 0.0:
        return {"status": "skipped", "reason": "non_positive_slope", "slope": slope, "intercept": intercept}
    return {
        "status": "ok",
        "slope": slope,
        "intercept": intercept,
        "n_rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
    }


def _apply_platt_calibrator(p_raw: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    p = np.asarray(p_raw, dtype=float)
    eps = 1e-5
    p = np.clip(p, eps, 1.0 - eps)
    z = np.log(p / (1.0 - p))
    z_adj = np.clip(float(slope) * z + float(intercept), -25.0, 25.0)
    return 1.0 / (1.0 + np.exp(-z_adj))


def _apply_continent_prior_calibration(
    dyn: pd.DataFrame,
    latest_year: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = dyn.copy()
    if "continent" not in out.columns or "stall_probability" not in out.columns:
        return out, {"status": "skipped", "reason": "missing_columns"}

    train = out[(out["year"] <= (latest_year - 2)) & out["stall_next"].notna()].copy()
    eval_df = out[(out["year"] == (latest_year - 1)) & out["stall_next"].notna()].copy()
    if len(train) < 120 or len(eval_df) < 40:
        return out, {"status": "skipped", "reason": "insufficient_rows_for_calibration"}

    y_eval = eval_df["stall_next"].astype(int).to_numpy()
    if np.unique(y_eval).size < 2:
        return out, {"status": "skipped", "reason": "single_class_eval_target"}

    prior_map = train.groupby("continent")["stall_next"].mean().to_dict()
    global_prior = float(train["stall_next"].mean())
    eval_prior = eval_df["continent"].map(prior_map).fillna(global_prior).astype(float).to_numpy()
    p_eval = eval_df["stall_probability"].astype(float).to_numpy()

    alphas = [1.0, 0.97, 0.93, 0.89, 0.85]
    best = {"alpha": 1.0, "score": -np.inf, "metrics": {}}
    for alpha in alphas:
        p_adj = np.clip(alpha * p_eval + (1.0 - alpha) * eval_prior, 0.0, 1.0)
        score, met = _cls_objective(y_eval, p_adj)
        if score > float(best["score"]) + 1e-10:
            best = {"alpha": float(alpha), "score": float(score), "metrics": met}

    raw_score, raw_metrics = _cls_objective(y_eval, np.clip(p_eval, 0.0, 1.0))
    best_alpha = float(best["alpha"])
    all_prior = out["continent"].map(prior_map).fillna(global_prior).astype(float)
    out["stall_probability"] = np.clip(best_alpha * out["stall_probability"] + (1.0 - best_alpha) * all_prior, 0.0, 1.0)
    out["stall_risk_score"] = np.clip(100.0 * out["stall_probability"], 0.0, 100.0)

    delta_auc = None
    if raw_metrics.get("roc_auc") is not None and best["metrics"].get("roc_auc") is not None:
        delta_auc = float(best["metrics"]["roc_auc"] - raw_metrics["roc_auc"])
    delta_brier = float(best["metrics"]["brier"] - raw_metrics["brier"])

    return out, {
        "status": "ok",
        "selected_alpha": best_alpha,
        "eval_year": int(latest_year - 1),
        "raw_metrics": raw_metrics,
        "calibrated_metrics": best["metrics"],
        "delta_objective": float(best["score"] - raw_score),
        "delta_roc_auc": delta_auc,
        "delta_brier": delta_brier,
    }


def _apply_split_conformal_uncertainty(
    dyn: pd.DataFrame,
    latest_year: int,
    alpha: float = 0.10,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = dyn.copy()
    if "stall_probability" not in out.columns or "stall_next" not in out.columns:
        return out, {"status": "skipped", "reason": "missing_probability_or_label"}

    eval_pool = out[(out["year"] < latest_year) & out["stall_next"].notna()].copy()
    if eval_pool.empty:
        return out, {"status": "skipped", "reason": "no_observed_labels_before_latest"}

    calib = eval_pool[eval_pool["year"] == (latest_year - 1)].copy()
    if len(calib) < 40:
        calib = eval_pool.copy()
    if len(calib) < 25:
        return out, {"status": "skipped", "reason": "insufficient_rows_for_conformal"}

    y_cal = calib["stall_next"].astype(float).to_numpy()
    p_cal = np.clip(calib["stall_probability"].astype(float).to_numpy(), 0.0, 1.0)
    scale_cal = np.sqrt(np.clip(p_cal * (1.0 - p_cal), 1e-3, 0.25))
    residual = np.abs(y_cal - p_cal) / scale_cal
    q = float(np.quantile(residual, 1.0 - alpha))
    q = min(max(q, 0.0), 4.0)

    cap_candidates = [1.0, 0.45, 0.40, 0.35, 0.30, 0.25]
    best_cap = 1.0
    best_score = -np.inf
    best_cov = 0.0
    best_width = 0.0
    for cap in cap_candidates:
        delta_cal = np.minimum(q * scale_cal, cap)
        low_cal_try = np.clip(p_cal - delta_cal, 0.0, 1.0)
        high_cal_try = np.clip(p_cal + delta_cal, 0.0, 1.0)
        cov_try = float(np.mean((y_cal >= low_cal_try) & (y_cal <= high_cal_try)))
        width_try = float(np.mean(high_cal_try - low_cal_try))
        score_try = cov_try - 0.33 * width_try
        if score_try > best_score + 1e-12:
            best_score = score_try
            best_cap = float(cap)
            best_cov = cov_try
            best_width = width_try

    p_all = np.clip(out["stall_probability"].astype(float).to_numpy(), 0.0, 1.0)
    scale_all = np.sqrt(np.clip(p_all * (1.0 - p_all), 1e-3, 0.25))
    delta_all = np.minimum(q * scale_all, best_cap)
    low = np.clip(p_all - delta_all, 0.0, 1.0)
    high = np.clip(p_all + delta_all, 0.0, 1.0)
    out["stall_probability_low"] = low
    out["stall_probability_high"] = high
    out["stall_risk_low"] = np.clip(100.0 * low, 0.0, 100.0)
    out["stall_risk_high"] = np.clip(100.0 * high, 0.0, 100.0)
    out["stall_risk_interval_width"] = out["stall_risk_high"] - out["stall_risk_low"]
    conformal_mean_width_risk = float(np.mean(out["stall_risk_interval_width"]))

    delta_cal = np.minimum(q * scale_cal, best_cap)
    low_cal = np.clip(p_cal - delta_cal, 0.0, 1.0)
    high_cal = np.clip(p_cal + delta_cal, 0.0, 1.0)
    cover_cal = float(np.mean((y_cal >= low_cal) & (y_cal <= high_cal)))

    y_eval = eval_pool["stall_next"].astype(float).to_numpy()
    p_eval = np.clip(eval_pool["stall_probability"].astype(float).to_numpy(), 0.0, 1.0)
    scale_eval = np.sqrt(np.clip(p_eval * (1.0 - p_eval), 1e-3, 0.25))
    delta_eval = np.minimum(q * scale_eval, best_cap)
    low_eval = np.clip(p_eval - delta_eval, 0.0, 1.0)
    high_eval = np.clip(p_eval + delta_eval, 0.0, 1.0)
    cover_eval = float(np.mean((y_eval >= low_eval) & (y_eval <= high_eval)))

    bootstrap_interval: Dict[str, Any] = {"status": "skipped", "reason": "insufficient_bootstrap_rows_or_features"}
    boot_features = [
        "stall_probability",
        "acceleration_score",
        "regime_forward_risk",
        "regime_self_transition_prob",
        "regime_transition_entropy",
        "regime_run_length_log",
        "regime_switch_rate_3y",
    ]
    boot_features = [c for c in boot_features if c in out.columns]
    boot_train = out[(out["year"] <= (latest_year - 2)) & out["stall_next"].notna()].copy()
    if boot_features and len(boot_train) >= 180 and boot_train["stall_next"].nunique() >= 2:
        _fill_feature_medians(boot_train, boot_features)
        _fill_feature_medians(out, boot_features, ref_df=boot_train)
        x_all = out[boot_features].to_numpy(dtype=float)

        rng = np.random.default_rng(2026)
        preds: List[np.ndarray] = []
        n_boot = 40
        for b in range(n_boot):
            idx = rng.integers(0, len(boot_train), size=len(boot_train))
            sample = boot_train.iloc[idx]
            yb = sample["stall_next"].astype(int).to_numpy()
            if np.unique(yb).size < 2:
                continue
            xb = sample[boot_features].to_numpy(dtype=float)
            clf = _build_classifier("logit", random_state=1200 + b)
            _fit_classifier_with_weights(clf, "logit", xb, yb, sample_weight=None)
            preds.append(np.clip(clf.predict_proba(x_all)[:, 1], 0.0, 1.0))

        if len(preds) >= 16:
            parr = np.vstack(preds)
            p_boot_low = np.quantile(parr, 0.16, axis=0)
            p_boot_high = np.quantile(parr, 0.84, axis=0)
            out["stall_probability_boot_low"] = np.clip(p_boot_low, 0.0, 1.0)
            out["stall_probability_boot_high"] = np.clip(p_boot_high, 0.0, 1.0)
            out["stall_risk_boot_low"] = np.clip(100.0 * out["stall_probability_boot_low"], 0.0, 100.0)
            out["stall_risk_boot_high"] = np.clip(100.0 * out["stall_probability_boot_high"], 0.0, 100.0)
            out["stall_risk_interval_width_boot"] = out["stall_risk_boot_high"] - out["stall_risk_boot_low"]

            # Use epistemic interval for front-end uncertainty width.
            out["stall_probability_low"] = out["stall_probability_boot_low"]
            out["stall_probability_high"] = out["stall_probability_boot_high"]
            out["stall_risk_low"] = out["stall_risk_boot_low"]
            out["stall_risk_high"] = out["stall_risk_boot_high"]
            out["stall_risk_interval_width"] = out["stall_risk_interval_width_boot"]

            bootstrap_interval = {
                "status": "ok",
                "n_boot_used": int(len(preds)),
                "n_train": int(len(boot_train)),
                "features": boot_features,
                "mean_interval_width_risk_score": float(out["stall_risk_interval_width_boot"].mean()),
                "interval": "p16-p84",
            }

    return out, {
        "status": "ok",
        "alpha": float(alpha),
        "target_coverage": float(1.0 - alpha),
        "quantile_scaled_error": q,
        "selected_cap_prob": best_cap,
        "cap_selection_objective": "maximize(coverage - 0.33*width)",
        "calibration_rows": int(len(calib)),
        "evaluation_rows": int(len(eval_pool)),
        "calibration_coverage": cover_cal,
        "calibration_coverage_selected_cap": best_cov,
        "calibration_width_selected_cap": best_width,
        "evaluation_coverage": cover_eval,
        "conformal_mean_interval_width_risk_score": conformal_mean_width_risk,
        "bootstrap_interval": bootstrap_interval,
        "interval_type_for_output": "bootstrap_p16_p84" if bootstrap_interval.get("status") == "ok" else "conformal",
        "mean_interval_width_prob": float(np.mean(out["stall_probability_high"] - out["stall_probability_low"])),
        "mean_interval_width_risk_score": float(np.mean(out["stall_risk_interval_width"])),
    }


def _classify_kinetic_state(
    growth: pd.Series,
    accel: pd.Series,
    g_low: float,
    g_high: float,
    a_low: float,
    a_high: float,
) -> pd.Series:
    cond = [
        (growth >= g_high) & (accel >= a_high),
        (growth >= g_high) & (accel <= a_low),
        (growth <= g_low) & (accel >= a_high),
        (growth <= g_low) & (accel <= a_low),
        (growth.between(g_low, g_high)) & (accel.between(a_low, a_high)),
    ]
    label = [
        "surge_expansion",
        "overheating",
        "recovery_rebound",
        "deep_cooling",
        "steady_cruise",
    ]
    out = np.select(cond, label, default="mixed_transition")
    return pd.Series(out, index=growth.index, dtype="object")


def _build_dynamic_kinetic_structure(
    dyn: pd.DataFrame,
    latest_year: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = dyn.sort_values(["city_id", "year"]).copy()
    out["jerk_1"] = out["accel_1"] - out.groupby("city_id")["accel_1"].shift(1)
    out["growth_abs"] = out["growth_1"].abs()
    out["accel_abs"] = out["accel_1"].abs()
    out["jerk_abs"] = out["jerk_1"].abs()
    out["momentum_gap_2y"] = out["growth_1"] - out.groupby("city_id")["growth_1"].shift(2)
    out["growth_sign_flip"] = ((out["growth_1"] * out["growth_1_lag"]) < 0.0).astype(float)

    ke_raw = np.sqrt(
        out["growth_1"].fillna(0.0) ** 2
        + 0.70 * out["accel_1"].fillna(0.0) ** 2
        + 0.30 * out["jerk_1"].fillna(0.0) ** 2
    )
    tp_raw = (
        0.45 * out["accel_abs"].fillna(0.0)
        + 0.35 * out["jerk_abs"].fillna(0.0)
        + 0.60 * out["growth_sign_flip"].fillna(0.0)
    )
    damping_raw = -out["growth_1"].fillna(0.0) * out["accel_1"].fillna(0.0)

    # Per-year MinMax to prevent future data leakage.
    out["_ke_raw"] = ke_raw
    out["_tp_raw"] = tp_raw
    out["_damp_raw"] = damping_raw
    out["kinetic_energy_score"] = 100.0 * out.groupby("year")["_ke_raw"].transform(
        lambda x: minmax_scale(x.to_numpy())
    )
    out["turning_point_risk"] = 100.0 * out.groupby("year")["_tp_raw"].transform(
        lambda x: minmax_scale(x.to_numpy())
    )
    out["damping_pressure"] = 100.0 * out.groupby("year")["_damp_raw"].transform(
        lambda x: minmax_scale(x.to_numpy())
    )
    out = out.drop(columns=["_ke_raw", "_tp_raw", "_damp_raw"], errors="ignore")

    valid_growth = out["growth_1"].dropna()
    valid_accel = out["accel_1"].dropna()
    if valid_growth.empty or valid_accel.empty:
        g_low, g_high, a_low, a_high = -0.4, 0.4, -0.4, 0.4
    else:
        g_low = float(np.quantile(valid_growth, 0.35))
        g_high = float(np.quantile(valid_growth, 0.65))
        a_low = float(np.quantile(valid_accel, 0.35))
        a_high = float(np.quantile(valid_accel, 0.65))
    out["kinetic_state"] = _classify_kinetic_state(out["growth_1"], out["accel_1"], g_low, g_high, a_low, a_high)

    latest = out[out["year"] == latest_year].copy()
    latest_dist = latest["kinetic_state"].value_counts(normalize=True).round(4).to_dict() if not latest.empty else {}

    summary = {
        "status": "ok",
        "thresholds": {
            "growth_low_q35": g_low,
            "growth_high_q65": g_high,
            "accel_low_q35": a_low,
            "accel_high_q65": a_high,
        },
        "latest_distribution": latest_dist,
        "mean_kinetic_energy_score": float(latest["kinetic_energy_score"].mean()) if not latest.empty else None,
        "mean_turning_point_risk": float(latest["turning_point_risk"].mean()) if not latest.empty else None,
        "mean_damping_pressure": float(latest["damping_pressure"].mean()) if not latest.empty else None,
    }
    return out, summary


def _estimate_critical_transition_early_warning(
    dyn: pd.DataFrame,
    latest_year: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = dyn.sort_values(["city_id", "year"]).copy()
    required = {"city_id", "year", "composite_index", "growth_1", "accel_1", "stall_next", "growth_next"}
    if not required.issubset(set(out.columns)):
        out["critical_transition_score"] = np.nan
        out["critical_transition_probability"] = np.nan
        out["critical_transition_risk_score"] = np.nan
        out["critical_transition_band"] = "monitoring"
        pd.DataFrame(
            columns=[
                "city_id",
                "city_name",
                "country",
                "continent",
                "year",
                "critical_transition_score",
                "critical_transition_risk_score",
                "critical_transition_band",
                "stall_risk_score",
                "acceleration_score",
                "kinetic_state",
                "trajectory_regime",
            ]
        ).to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_critical_latest.csv", index=False)
        pd.DataFrame(
            columns=["risk_decile", "n_obs", "mean_pred_prob", "observed_stall_rate", "lift_vs_base_rate"]
        ).to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_critical_decile.csv", index=False)
        return out, {"status": "skipped", "reason": "missing_required_columns"}

    out["ar1_growth_5y"] = out.groupby("city_id")["growth_1"].transform(
        lambda s: s.rolling(5, min_periods=3).apply(_rolling_lag1_autocorr, raw=True)
    )
    out["growth_variance_5y"] = out.groupby("city_id")["growth_1"].transform(lambda s: s.rolling(5, min_periods=3).var())
    out["growth_skew_5y"] = out.groupby("city_id")["growth_1"].transform(lambda s: s.rolling(5, min_periods=3).skew())
    out["trend_slope_3y"] = out.groupby("city_id")["composite_index"].transform(
        lambda s: s.rolling(3, min_periods=2).apply(_window_slope, raw=True)
    )
    out["trend_slope_7y"] = out.groupby("city_id")["composite_index"].transform(
        lambda s: s.rolling(7, min_periods=3).apply(_window_slope, raw=True)
    )
    out["trend_break_score"] = (out["trend_slope_3y"] - out["trend_slope_7y"]).abs()
    out["rolling_peak_5y"] = out.groupby("city_id")["composite_index"].transform(lambda s: s.rolling(5, min_periods=2).max())
    out["drawdown_5y"] = (out["rolling_peak_5y"] - out["composite_index"]).clip(lower=0.0)
    out["reversal_pressure"] = (
        out["accel_1"].abs()
        + 0.55 * out.get("jerk_1", pd.Series(np.zeros(len(out)), index=out.index)).fillna(0.0).abs()
        + 0.40 * out.get("growth_sign_flip", pd.Series(np.zeros(len(out)), index=out.index)).fillna(0.0)
    )

    numeric_signals = [
        "ar1_growth_5y",
        "growth_variance_5y",
        "growth_skew_5y",
        "trend_break_score",
        "drawdown_5y",
        "reversal_pressure",
    ]
    for col in numeric_signals:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = float(out[col].median()) if out[col].notna().any() else 0.0
        out[col] = out[col].fillna(med)

    z_ar1 = _zscore(out["ar1_growth_5y"])
    z_var = _zscore(out["growth_variance_5y"])
    z_skew = _zscore(out["growth_skew_5y"].abs())
    z_break = _zscore(out["trend_break_score"])
    z_drawdown = _zscore(out["drawdown_5y"])
    z_reversal = _zscore(out["reversal_pressure"])
    critical_raw = (
        0.27 * z_ar1
        + 0.22 * z_var
        + 0.11 * z_skew
        + 0.16 * z_break
        + 0.13 * z_drawdown
        + 0.11 * z_reversal
    )
    out["critical_transition_score"] = np.clip(100.0 * minmax_scale(critical_raw.to_numpy()), 0.0, 100.0)

    q30 = float(out["critical_transition_score"].quantile(0.30))
    q75 = float(out["critical_transition_score"].quantile(0.75))
    out["critical_transition_band"] = np.select(
        [
            out["critical_transition_score"] >= q75,
            out["critical_transition_score"] <= q30,
        ],
        ["high_alert", "stable_window"],
        default="monitoring",
    )

    feature_cols = [
        "critical_transition_score",
        "ar1_growth_5y",
        "growth_variance_5y",
        "growth_skew_5y",
        "trend_break_score",
        "drawdown_5y",
        "reversal_pressure",
        "stall_probability",
        "turning_point_risk",
        "regime_forward_risk",
        "regime_transition_entropy",
        "regime_switch_rate_3y",
    ]
    feature_cols = [c for c in feature_cols if c in out.columns]

    fit_df = out.dropna(subset=["stall_next", "growth_next"]).copy()
    if len(feature_cols) < 4 or fit_df.empty:
        p_raw = np.clip(out["critical_transition_score"].astype(float).to_numpy() / 100.0, 0.0, 1.0)
        out["critical_transition_probability"] = p_raw
        out["critical_transition_risk_score"] = np.clip(100.0 * p_raw, 0.0, 100.0)
        latest = out[out["year"] == latest_year].copy()
        latest_cols = [
            "city_id",
            "city_name",
            "country",
            "continent",
            "year",
            "critical_transition_score",
            "critical_transition_risk_score",
            "critical_transition_band",
            "stall_risk_score",
            "acceleration_score",
            "kinetic_state",
            "trajectory_regime",
        ]
        latest[[c for c in latest_cols if c in latest.columns]].sort_values(
            "critical_transition_risk_score", ascending=False
        ).to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_critical_latest.csv", index=False)
        pd.DataFrame(
            columns=["risk_decile", "n_obs", "mean_pred_prob", "observed_stall_rate", "lift_vs_base_rate"]
        ).to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_critical_decile.csv", index=False)
        return out, {"status": "skipped", "reason": "insufficient_features_or_rows"}

    for col in feature_cols:
        fit_df[col] = pd.to_numeric(fit_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    _fill_feature_medians(fit_df, feature_cols, ref_df=fit_df)
    _fill_feature_medians(out, feature_cols, ref_df=fit_df)

    train = fit_df[fit_df["year"] <= (latest_year - 2)].copy()
    test = fit_df[fit_df["year"] == (latest_year - 1)].copy()
    if train.empty or test.empty:
        split_year = int(fit_df["year"].quantile(0.75))
        train = fit_df[fit_df["year"] <= split_year].copy()
        test = fit_df[fit_df["year"] > split_year].copy()
    if train.empty or test.empty or train["stall_next"].nunique() < 2:
        p_raw = np.clip(out["critical_transition_score"].astype(float).to_numpy() / 100.0, 0.0, 1.0)
        out["critical_transition_probability"] = p_raw
        out["critical_transition_risk_score"] = np.clip(100.0 * p_raw, 0.0, 100.0)
        latest = out[out["year"] == latest_year].copy()
        latest_cols = [
            "city_id",
            "city_name",
            "country",
            "continent",
            "year",
            "critical_transition_score",
            "critical_transition_risk_score",
            "critical_transition_band",
            "stall_risk_score",
            "acceleration_score",
            "kinetic_state",
            "trajectory_regime",
        ]
        latest[[c for c in latest_cols if c in latest.columns]].sort_values(
            "critical_transition_risk_score", ascending=False
        ).to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_critical_latest.csv", index=False)
        pd.DataFrame(
            columns=["risk_decile", "n_obs", "mean_pred_prob", "observed_stall_rate", "lift_vs_base_rate"]
        ).to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_critical_decile.csv", index=False)
        return out, {"status": "skipped", "reason": "temporal_split_or_class_failure"}

    y_train = train["stall_next"].astype(int).to_numpy()
    y_test = test["stall_next"].astype(int).to_numpy()
    x_train = train[feature_cols].to_numpy(dtype=float)
    x_test = test[feature_cols].to_numpy(dtype=float)
    x_all = out[feature_cols].to_numpy(dtype=float)

    learner_records: Dict[str, Dict[str, Any]] = {}
    learner_probs: Dict[str, Dict[str, np.ndarray]] = {}
    for i, learner in enumerate(["logit", "gb", "hgb"]):
        mdl = _build_classifier(learner, random_state=3070 + i)
        _fit_classifier_with_weights(mdl, learner, x_train, y_train, sample_weight=None)
        p_tr = np.clip(mdl.predict_proba(x_train)[:, 1], 0.0, 1.0)
        p_te = np.clip(mdl.predict_proba(x_test)[:, 1], 0.0, 1.0)
        p_al = np.clip(mdl.predict_proba(x_all)[:, 1], 0.0, 1.0)
        score, met = _cls_objective(y_test, p_te)
        learner_records[learner] = {"objective": float(score), "metrics": met, "model": mdl}
        learner_probs[learner] = {"train": p_tr, "test": p_te, "all": p_al}

    selected_learner = max(
        learner_records.keys(),
        key=lambda k: (
            float(learner_records[k]["objective"]),
            float(learner_records[k]["metrics"].get("roc_auc") or -np.inf),
            -float(learner_records[k]["metrics"]["brier"]),
        ),
    )
    model = learner_records[selected_learner]["model"]
    p_test = learner_probs[selected_learner]["test"]
    p_all = learner_probs[selected_learner]["all"]
    met_sel = learner_records[selected_learner]["metrics"]

    p_base = np.clip(test["critical_transition_score"].astype(float).to_numpy() / 100.0, 0.0, 1.0)
    met_base = _calc_cls_metrics(y_test, p_base)
    met_stall_ref: Dict[str, float | None] | None = None
    delta_auc_vs_stall = None
    delta_brier_vs_stall = None
    auc_sig_vs_stall: Dict[str, Any] = {"status": "skipped", "reason": "stall_probability_missing"}
    brier_sig_vs_stall: Dict[str, Any] = {"status": "skipped", "reason": "stall_probability_missing"}
    if "stall_probability" in test.columns:
        p_stall_ref = np.clip(test["stall_probability"].astype(float).to_numpy(), 0.0, 1.0)
        met_stall_ref = _calc_cls_metrics(y_test, p_stall_ref)
        if met_sel.get("roc_auc") is not None and met_stall_ref.get("roc_auc") is not None:
            delta_auc_vs_stall = float(met_sel["roc_auc"] - met_stall_ref["roc_auc"])
        if met_sel.get("brier") is not None and met_stall_ref.get("brier") is not None:
            delta_brier_vs_stall = float(met_sel["brier"] - met_stall_ref["brier"])
        auc_sig_vs_stall = _bootstrap_metric_diff(
            y_test,
            p_test,
            p_stall_ref,
            metric="roc_auc",
            n_boot=320,
            random_state=3092,
        )
        brier_sig_vs_stall = _bootstrap_metric_diff(
            y_test,
            p_test,
            p_stall_ref,
            metric="brier",
            n_boot=320,
            random_state=3093,
        )

    out["critical_transition_probability"] = p_all
    out["critical_transition_risk_score"] = np.clip(100.0 * p_all, 0.0, 100.0)

    auc_sig = _bootstrap_metric_diff(
        y_test,
        p_test,
        p_base,
        metric="roc_auc",
        n_boot=420,
        random_state=3090,
    )
    brier_sig = _bootstrap_metric_diff(
        y_test,
        p_test,
        p_base,
        metric="brier",
        n_boot=420,
        random_state=3091,
    )
    delta_auc = None
    if met_sel.get("roc_auc") is not None and met_base.get("roc_auc") is not None:
        delta_auc = float(met_sel["roc_auc"] - met_base["roc_auc"])

    decile_df = pd.DataFrame({"y": y_test.astype(int), "p": p_test.astype(float)})
    rank_desc = decile_df["p"].rank(method="first", ascending=False)
    n_bins = min(10, max(2, int(np.sqrt(len(decile_df)))))
    decile_df["risk_decile"] = pd.qcut(rank_desc, q=n_bins, labels=False, duplicates="drop") + 1
    base_rate = float(decile_df["y"].mean())
    decile = (
        decile_df.groupby("risk_decile", as_index=False)
        .agg(
            n_obs=("y", "size"),
            mean_pred_prob=("p", "mean"),
            observed_stall_rate=("y", "mean"),
        )
        .sort_values("risk_decile")
    )
    decile["lift_vs_base_rate"] = decile["observed_stall_rate"] / max(base_rate, 1e-6)
    decile.to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_critical_decile.csv", index=False)

    latest = out[out["year"] == latest_year].copy()
    latest_cols = [
        "city_id",
        "city_name",
        "country",
        "continent",
        "year",
        "critical_transition_score",
        "critical_transition_risk_score",
        "critical_transition_band",
        "stall_risk_score",
        "acceleration_score",
        "kinetic_state",
        "trajectory_regime",
    ]
    latest = latest[[c for c in latest_cols if c in latest.columns]].sort_values(
        "critical_transition_risk_score", ascending=False
    )
    latest.to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_critical_latest.csv", index=False)

    top_features: List[Dict[str, Any]] = []
    try:
        if selected_learner == "logit":
            logit = model.named_steps["logisticregression"]
            coefs = logit.coef_[0]
            top_features = sorted(
                [
                    {
                        "feature": feature_cols[i],
                        "importance": float(coefs[i]),
                        "abs_importance": abs(float(coefs[i])),
                    }
                    for i in range(len(feature_cols))
                ],
                key=lambda x: float(x["abs_importance"]),
                reverse=True,
            )[:8]
        else:
            imp = permutation_importance(
                model,
                x_test,
                y_test,
                n_repeats=12,
                random_state=42,
                scoring="roc_auc" if np.unique(y_test).size >= 2 else "neg_brier_score",
            )
            top_features = sorted(
                [
                    {
                        "feature": feature_cols[i],
                        "importance": float(imp.importances_mean[i]),
                        "abs_importance": abs(float(imp.importances_mean[i])),
                    }
                    for i in range(len(feature_cols))
                ],
                key=lambda x: float(x["abs_importance"]),
                reverse=True,
            )[:8]
    except Exception:  # noqa: BLE001
        top_features = []

    latest_dist = (
        latest["critical_transition_band"].value_counts(normalize=True).round(4).to_dict()
        if "critical_transition_band" in latest.columns and not latest.empty
        else {}
    )
    top_decile = decile[decile["risk_decile"] == int(decile["risk_decile"].min())]
    top_decile_obs = float(top_decile["observed_stall_rate"].iloc[0]) if not top_decile.empty else None

    return out, {
        "status": "ok",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(len(feature_cols)),
        "features": feature_cols,
        "selected_learner": selected_learner,
        "learner_candidates": {k: v["metrics"] | {"objective": v["objective"]} for k, v in learner_records.items()},
        "base_metrics": met_base,
        "stall_reference_metrics": met_stall_ref,
        "selected_metrics": met_sel,
        "delta_auc_vs_base": delta_auc,
        "delta_brier_vs_base": float(met_sel["brier"] - met_base["brier"]),
        "delta_auc_vs_stall_reference": delta_auc_vs_stall,
        "delta_brier_vs_stall_reference": delta_brier_vs_stall,
        "delta_auc_ci95": auc_sig.get("ci95") if auc_sig.get("status") == "ok" else [None, None],
        "delta_auc_p_value": auc_sig.get("p_value_two_sided") if auc_sig.get("status") == "ok" else None,
        "delta_brier_improve_ci95": brier_sig.get("ci95") if brier_sig.get("status") == "ok" else [None, None],
        "delta_brier_improve_p_value": brier_sig.get("p_value_two_sided") if brier_sig.get("status") == "ok" else None,
        "delta_auc_vs_stall_reference_ci95": auc_sig_vs_stall.get("ci95") if auc_sig_vs_stall.get("status") == "ok" else [None, None],
        "delta_auc_vs_stall_reference_p_value": auc_sig_vs_stall.get("p_value_two_sided")
        if auc_sig_vs_stall.get("status") == "ok"
        else None,
        "delta_brier_vs_stall_reference_improve_ci95": brier_sig_vs_stall.get("ci95")
        if brier_sig_vs_stall.get("status") == "ok"
        else [None, None],
        "delta_brier_vs_stall_reference_improve_p_value": brier_sig_vs_stall.get("p_value_two_sided")
        if brier_sig_vs_stall.get("status") == "ok"
        else None,
        "high_alert_threshold_q75": q75,
        "stable_threshold_q30": q30,
        "latest_band_distribution": latest_dist,
        "top_decile_observed_stall_rate": top_decile_obs,
        "decile_count": int(decile["risk_decile"].nunique()) if not decile.empty else 0,
        "top_features": top_features,
    }


def _fit_dynamic_transition_hazard(
    dyn: pd.DataFrame,
    latest_year: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = dyn.copy()
    feature_cols = [
        "kinetic_energy_score",
        "turning_point_risk",
        "damping_pressure",
        "critical_transition_score",
        "ar1_growth_5y",
        "growth_variance_5y",
        "trend_break_score",
        "drawdown_5y",
        "reversal_pressure",
        "growth_abs",
        "accel_abs",
        "jerk_abs",
        "momentum_gap_2y",
        "regime_run_length_log",
        "regime_switch_rate_3y",
        "regime_switch_rate_5y",
        "regime_forward_risk",
        "regime_self_transition_prob",
        "regime_transition_entropy",
    ]
    feature_cols = [c for c in feature_cols if c in out.columns]
    fit_df = out.dropna(subset=["stall_next", "growth_next"]).copy()
    if not feature_cols or fit_df.empty:
        out["dynamic_hazard_probability"] = np.nan
        out["dynamic_hazard_score"] = np.nan
        return out, {"status": "skipped", "reason": "insufficient_features_or_rows"}

    for col in feature_cols:
        fit_df[col] = pd.to_numeric(fit_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    _fill_feature_medians(fit_df, feature_cols, ref_df=fit_df)
    _fill_feature_medians(out, feature_cols, ref_df=fit_df)
    train = fit_df[fit_df["year"] <= (latest_year - 2)].copy()
    test = fit_df[fit_df["year"] == (latest_year - 1)].copy()
    if train.empty or test.empty:
        split_year = int(fit_df["year"].quantile(0.75))
        train = fit_df[fit_df["year"] <= split_year].copy()
        test = fit_df[fit_df["year"] > split_year].copy()
    if train.empty or test.empty:
        out["dynamic_hazard_probability"] = np.nan
        out["dynamic_hazard_score"] = np.nan
        return out, {"status": "skipped", "reason": "temporal_split_failed"}

    y_train = train["stall_next"].astype(int).to_numpy()
    y_test = test["stall_next"].astype(int).to_numpy()
    if np.unique(y_train).size < 2:
        out["dynamic_hazard_probability"] = np.nan
        out["dynamic_hazard_score"] = np.nan
        return out, {"status": "skipped", "reason": "single_class_training"}

    x_train = train[feature_cols].to_numpy(dtype=float)
    x_test = test[feature_cols].to_numpy(dtype=float)
    x_all = out[feature_cols].to_numpy(dtype=float)

    # Dynamic hazard uses state-aware reweighting to emphasize turning states and positive stalls.
    sample_weight = np.ones(len(train), dtype=float)
    pos_rate = float(np.mean(y_train))
    if 0.0 < pos_rate < 1.0:
        pos_boost = float(np.clip((1.0 - pos_rate) / max(pos_rate, 1e-6), 1.0, 4.0))
        sample_weight = np.where(y_train == 1, sample_weight * pos_boost, sample_weight)
    if "kinetic_state" in train.columns:
        ks = train["kinetic_state"].fillna("mixed_transition").astype(str)
        hotspot = ks.isin(["deep_cooling", "overheating", "mixed_transition"]).to_numpy(dtype=float)
        sample_weight *= (1.0 + 0.32 * hotspot)
    if "regime_switch_rate_3y" in train.columns:
        rs = pd.to_numeric(train["regime_switch_rate_3y"], errors="coerce").fillna(0.0)
        q70 = float(rs.quantile(0.70))
        sample_weight *= np.where(rs.to_numpy(dtype=float) >= q70, 1.18, 1.0)
    sw_mean = float(np.mean(sample_weight))
    if sw_mean > 1e-12:
        sample_weight = sample_weight / sw_mean
    else:
        sample_weight = np.ones(len(train), dtype=float)

    learner_records: Dict[str, Dict[str, Any]] = {}
    learner_probs: Dict[str, Dict[str, np.ndarray]] = {}
    for learner in ["logit", "gb"]:
        mdl = _build_classifier(learner, random_state=2026 if learner == "logit" else 2027)
        _fit_classifier_with_weights(mdl, learner, x_train, y_train, sample_weight=sample_weight)
        p_tr = np.clip(mdl.predict_proba(x_train)[:, 1], 0.0, 1.0)
        p_te = np.clip(mdl.predict_proba(x_test)[:, 1], 0.0, 1.0)
        p_al = np.clip(mdl.predict_proba(x_all)[:, 1], 0.0, 1.0)
        s, met = _cls_objective(y_test, p_te)
        learner_records[learner] = {"objective": float(s), "metrics": met, "model": mdl}
        learner_probs[learner] = {"train": p_tr, "test": p_te, "all": p_al}

    selected_dynamic_learner = max(
        learner_records.keys(),
        key=lambda k: (
            float(learner_records[k]["objective"]),
            float(learner_records[k]["metrics"].get("roc_auc") or -np.inf),
            -float(learner_records[k]["metrics"]["brier"]),
        ),
    )
    model = learner_records[selected_dynamic_learner]["model"]
    p_train = learner_probs[selected_dynamic_learner]["train"]
    p_test = learner_probs[selected_dynamic_learner]["test"]
    p_all = learner_probs[selected_dynamic_learner]["all"]
    met_dyn = learner_records[selected_dynamic_learner]["metrics"]
    out["dynamic_hazard_probability"] = p_all
    out["dynamic_hazard_score"] = np.clip(100.0 * p_all, 0.0, 100.0)

    out["dynamic_hazard_fused_probability"] = out["dynamic_hazard_probability"]
    out["dynamic_hazard_fused_score"] = out["dynamic_hazard_score"]
    out["dynamic_hazard_gate_weight"] = np.nan
    blend_metrics: Dict[str, Any] = {"status": "skipped", "reason": "missing_base_probability"}
    if "stall_probability" in out.columns and "stall_probability" in test.columns and "stall_probability" in train.columns:
        p_base_train = np.clip(train["stall_probability"].astype(float).to_numpy(), 0.0, 1.0)
        p_base_test = np.clip(test["stall_probability"].astype(float).to_numpy(), 0.0, 1.0)
        p_base_all = np.clip(out["stall_probability"].astype(float).to_numpy(), 0.0, 1.0)
        met_base = _calc_cls_metrics(y_test, p_base_test)

        # Candidate 1: globally optimized blend weight on train window.
        global_candidates: List[Dict[str, Any]] = []
        for w in np.linspace(0.0, 1.0, 21):
            p_bl = np.clip((1.0 - float(w)) * p_base_train + float(w) * p_train, 0.0, 1.0)
            score, met = _cls_objective(y_train, p_bl)
            global_candidates.append({"w": float(w), "score": float(score), "metrics": met})
        global_best = max(global_candidates, key=lambda d: d["score"])
        w_global = float(global_best["w"])
        p_global_test = np.clip((1.0 - w_global) * p_base_test + w_global * p_test, 0.0, 1.0)
        p_global_all = np.clip((1.0 - w_global) * p_base_all + w_global * p_all, 0.0, 1.0)
        met_global = _calc_cls_metrics(y_test, p_global_test)
        score_global, _ = _cls_objective(y_test, p_global_test)

        # Candidate 2: state-conditioned gating weight (shrink toward global weight).
        state_col = "kinetic_state" if "kinetic_state" in train.columns else None
        w_state_map: Dict[str, float] = {}
        p_state_test = p_global_test.copy()
        p_state_all = p_global_all.copy()
        score_state = -np.inf
        met_state = met_global
        if state_col is not None:
            train_states = train[state_col].fillna("mixed_transition").astype(str).to_numpy(dtype=object)
            n_train = len(y_train)
            for state_name in sorted({str(x) for x in train_states.tolist()}):
                idx_arr = np.where(train_states == state_name)[0].astype(int)
                idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < n_train)]
                if len(idx_arr) == 0:
                    continue
                if len(idx_arr) < 50:
                    w_state_map[str(state_name)] = w_global
                    continue
                ys = y_train[idx_arr]
                if np.unique(ys).size < 2:
                    w_state_map[str(state_name)] = w_global
                    continue
                state_best = {"w": w_global, "score": -np.inf}
                for w in np.linspace(0.0, 1.0, 21):
                    ps = np.clip((1.0 - float(w)) * p_base_train[idx_arr] + float(w) * p_train[idx_arr], 0.0, 1.0)
                    s, _ = _cls_objective(ys, ps)
                    if s > float(state_best["score"]) + 1e-12:
                        state_best = {"w": float(w), "score": float(s)}
                shrink = float(len(idx_arr) / (len(idx_arr) + 120.0))
                w_state_map[str(state_name)] = float(shrink * state_best["w"] + (1.0 - shrink) * w_global)

            test_states = test[state_col].fillna("mixed_transition").astype(str)
            test_w = test_states.map(lambda s: w_state_map.get(str(s), w_global)).to_numpy(dtype=float)
            p_state_test = np.clip((1.0 - test_w) * p_base_test + test_w * p_test, 0.0, 1.0)
            all_states = out[state_col].fillna("mixed_transition").astype(str)
            all_w = all_states.map(lambda s: w_state_map.get(str(s), w_global)).to_numpy(dtype=float)
            p_state_all = np.clip((1.0 - all_w) * p_base_all + all_w * p_all, 0.0, 1.0)
            out["dynamic_hazard_gate_weight"] = all_w
            met_state = _calc_cls_metrics(y_test, p_state_test)
            score_state, _ = _cls_objective(y_test, p_state_test)

            gate_rows = [{"kinetic_state": k, "dynamic_weight": float(v)} for k, v in sorted(w_state_map.items())]
            pd.DataFrame(gate_rows).to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_gate_weights.csv", index=False)

        score_base, _ = _cls_objective(y_test, p_base_test)
        score_dyn, _ = _cls_objective(y_test, p_test)
        candidate_table = [
            {"candidate": "base", "score": float(score_base), "roc_auc": met_base.get("roc_auc"), "brier": met_base["brier"]},
            {
                "candidate": "dynamic_only",
                "score": float(score_dyn),
                "roc_auc": met_dyn.get("roc_auc"),
                "brier": met_dyn["brier"],
            },
            {
                "candidate": "global_blend",
                "score": float(score_global),
                "roc_auc": met_global.get("roc_auc"),
                "brier": met_global["brier"],
                "dynamic_weight": w_global,
            },
            {
                "candidate": "state_gated_blend",
                "score": float(score_state),
                "roc_auc": met_state.get("roc_auc"),
                "brier": met_state["brier"],
            },
        ]
        best_row = max(candidate_table, key=lambda d: float(d["score"]))
        selected = str(best_row["candidate"])
        if selected == "base":
            p_sel_test = p_base_test
            p_sel_all = p_base_all
        elif selected == "dynamic_only":
            p_sel_test = p_test
            p_sel_all = p_all
        elif selected == "state_gated_blend":
            p_sel_test = p_state_test
            p_sel_all = p_state_all
        else:
            p_sel_test = p_global_test
            p_sel_all = p_global_all

        met_sel = _calc_cls_metrics(y_test, p_sel_test)
        out["dynamic_hazard_fused_probability"] = p_sel_all
        out["dynamic_hazard_fused_score"] = np.clip(100.0 * p_sel_all, 0.0, 100.0)
        delta_auc = None
        if met_sel.get("roc_auc") is not None and met_base.get("roc_auc") is not None:
            delta_auc = float(met_sel["roc_auc"] - met_base["roc_auc"])
        auc_sig = _bootstrap_metric_diff(
            y_test,
            p_sel_test,
            p_base_test,
            metric="roc_auc",
            n_boot=500,
            random_state=2026,
        )
        brier_sig = _bootstrap_metric_diff(
            y_test,
            p_sel_test,
            p_base_test,
            metric="brier",
            n_boot=500,
            random_state=2027,
        )
        blend_metrics = {
            "status": "ok",
            "selected_candidate": selected,
            "candidate_table": candidate_table,
            "base_metrics": met_base,
            "selected_metrics": met_sel,
            "global_dynamic_weight": w_global,
            "delta_auc_vs_base": delta_auc,
            "delta_brier_vs_base": float(met_sel["brier"] - met_base["brier"]),
            "delta_auc_ci95": auc_sig.get("ci95") if auc_sig.get("status") == "ok" else [None, None],
            "delta_auc_p_value": auc_sig.get("p_value_two_sided") if auc_sig.get("status") == "ok" else None,
            "delta_brier_improve_ci95": brier_sig.get("ci95") if brier_sig.get("status") == "ok" else [None, None],
            "delta_brier_improve_p_value": brier_sig.get("p_value_two_sided") if brier_sig.get("status") == "ok" else None,
            "state_gate_count": int(len(w_state_map)),
        }

    coef_rows: List[Dict[str, Any]] = []
    try:
        if selected_dynamic_learner == "logit":
            logit = model.named_steps["logisticregression"]
            coefs = logit.coef_[0]
            for i, feat in enumerate(feature_cols):
                coef_rows.append(
                    {
                        "feature": feat,
                        "importance": float(coefs[i]),
                        "abs_importance": abs(float(coefs[i])),
                        "type": "coefficient",
                    }
                )
        else:
            imp = permutation_importance(
                model,
                x_test,
                y_test,
                n_repeats=12,
                random_state=42,
                scoring="roc_auc" if np.unique(y_test).size >= 2 else "neg_brier_score",
            )
            for i, feat in enumerate(feature_cols):
                coef_rows.append(
                    {
                        "feature": feat,
                        "importance": float(imp.importances_mean[i]),
                        "abs_importance": abs(float(imp.importances_mean[i])),
                        "type": "permutation_importance",
                    }
                )
    except Exception:  # noqa: BLE001
        coef_rows = []
    coef_df = pd.DataFrame(coef_rows).sort_values("abs_importance", ascending=False) if coef_rows else pd.DataFrame()
    coef_df.to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_hazard_coefficients.csv", index=False)

    latest = out[out["year"] == latest_year].copy()
    latest_cols = [
        "city_id",
        "city_name",
        "country",
        "continent",
        "year",
        "trajectory_regime",
        "kinetic_state",
        "critical_transition_score",
        "critical_transition_risk_score",
        "stall_risk_score",
        "dynamic_hazard_score",
        "dynamic_hazard_fused_score",
        "dynamic_hazard_gate_weight",
        "kinetic_energy_score",
        "turning_point_risk",
        "damping_pressure",
    ]
    latest = latest[[c for c in latest_cols if c in latest.columns]].sort_values("dynamic_hazard_score", ascending=False)
    latest.to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_hazard_latest.csv", index=False)

    return out, {
        "status": "ok",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(len(feature_cols)),
        "features": feature_cols,
        "sample_weight": {
            "mode": "state_and_tail_reweighted",
            "positive_rate_train": pos_rate,
            "mean": float(np.mean(sample_weight)),
            "std": float(np.std(sample_weight)),
            "p95": float(np.quantile(sample_weight, 0.95)),
        },
        "selected_dynamic_learner": selected_dynamic_learner,
        "learner_candidates": {k: v["metrics"] | {"objective": v["objective"]} for k, v in learner_records.items()},
        "roc_auc": met_dyn.get("roc_auc"),
        "average_precision": met_dyn.get("average_precision"),
        "brier": met_dyn.get("brier"),
        "blend_comparison": blend_metrics,
        "top_coefficients": coef_df.head(8).to_dict(orient="records") if not coef_df.empty else [],
    }


def _estimate_dynamic_graph_diffusion(
    dyn: pd.DataFrame,
    latest_year: int,
    k_neighbors: int = 14,
    n_steps: int = 10,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = dyn.copy()
    for col in [
        "graph_diffusion_probability",
        "graph_diffusion_score",
        "graph_diffusion_fused_probability",
        "graph_diffusion_fused_score",
    ]:
        if col not in out.columns:
            out[col] = np.nan

    if "stall_probability" not in out.columns:
        pd.DataFrame().to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_graph_curve.csv", index=False)
        pd.DataFrame().to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_graph_latest.csv", index=False)
        return out, {"status": "skipped", "reason": "stall_probability_missing"}

    base_features = [
        "stall_probability",
        "critical_transition_probability",
        "critical_transition_score",
        "dynamic_hazard_probability",
        "dynamic_hazard_fused_probability",
        "acceleration_score",
        "kinetic_energy_score",
        "turning_point_risk",
        "damping_pressure",
        "regime_forward_risk",
        "regime_transition_entropy",
        "growth_1",
        "accel_1",
    ]
    feature_cols = [c for c in base_features if c in out.columns]
    if len(feature_cols) < 4 or len(out) < 80:
        out["graph_diffusion_probability"] = np.clip(out["stall_probability"].astype(float).to_numpy(), 0.0, 1.0)
        out["graph_diffusion_score"] = np.clip(100.0 * out["graph_diffusion_probability"], 0.0, 100.0)
        out["graph_diffusion_fused_probability"] = out["graph_diffusion_probability"]
        out["graph_diffusion_fused_score"] = out["graph_diffusion_score"]
        pd.DataFrame().to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_graph_curve.csv", index=False)
        latest = out[out["year"] == latest_year].copy()
        latest.to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_graph_latest.csv", index=False)
        return out, {"status": "skipped", "reason": "insufficient_graph_features"}

    work = out.copy()
    for col in feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    _fill_feature_medians(work, feature_cols, ref_df=work)

    x = work[feature_cols].to_numpy(dtype=float)
    x = StandardScaler().fit_transform(x)
    yv = work["year"].astype(float).to_numpy()
    y_std = float(yv.std()) if np.isfinite(yv.std()) and float(yv.std()) > 0 else 1.0
    year_feature = ((yv - float(yv.mean())) / y_std).reshape(-1, 1)
    x_aug = np.hstack([x, 0.45 * year_feature])

    n_obs = len(work)
    k = int(max(3, min(k_neighbors, n_obs - 1)))
    if k < 2:
        pd.DataFrame().to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_graph_curve.csv", index=False)
        pd.DataFrame().to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_graph_latest.csv", index=False)
        return out, {"status": "skipped", "reason": "insufficient_neighbors"}

    knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    knn.fit(x_aug)
    dist, idx = knn.kneighbors(x_aug, return_distance=True)
    local_dist = dist[:, 1:].reshape(-1)
    local_dist = local_dist[np.isfinite(local_dist) & (local_dist > 0)]
    sigma = float(np.median(local_dist)) if len(local_dist) else 1.0
    sigma = max(sigma, 1e-6)

    w = np.exp(-((dist[:, 1:] ** 2) / (2.0 * sigma**2)))
    w_sum = np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    w = w / w_sum

    p_seed = np.clip(work["stall_probability"].astype(float).to_numpy(), 0.0, 1.0)
    if "dynamic_hazard_probability" in work.columns:
        p_dyn_all = np.clip(work["dynamic_hazard_probability"].astype(float).to_numpy(), 0.0, 1.0)
    elif "dynamic_hazard_fused_probability" in work.columns:
        p_dyn_all = np.clip(work["dynamic_hazard_fused_probability"].astype(float).to_numpy(), 0.0, 1.0)
    else:
        p_dyn_all = p_seed.copy()
    p = np.clip(0.70 * p_seed + 0.30 * p_dyn_all, 0.0, 1.0)

    alpha_seed = 0.35
    iters = int(max(4, min(20, n_steps)))
    for _ in range(iters):
        neigh = p[idx[:, 1:]]
        smooth = (w * neigh).sum(axis=1)
        p = np.clip(alpha_seed * p_seed + (1.0 - alpha_seed) * smooth, 0.0, 1.0)

    out["graph_diffusion_probability"] = p
    out["graph_diffusion_score"] = np.clip(100.0 * p, 0.0, 100.0)
    out["graph_diffusion_fused_probability"] = out["graph_diffusion_probability"]
    out["graph_diffusion_fused_score"] = out["graph_diffusion_score"]

    eval_df = out[(out["year"] == (latest_year - 1)) & out["stall_next"].notna()].copy()
    curve_rows: List[Dict[str, Any]] = []
    if len(eval_df) < 60 or eval_df["stall_next"].nunique() < 2:
        pd.DataFrame(curve_rows).to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_graph_curve.csv", index=False)
        latest = out[out["year"] == latest_year].copy()
        latest_cols = [
            "city_id",
            "city_name",
            "country",
            "continent",
            "year",
            "stall_risk_score",
            "dynamic_hazard_fused_score",
            "graph_diffusion_score",
            "graph_diffusion_fused_score",
            "kinetic_state",
            "trajectory_regime",
        ]
        latest[[c for c in latest_cols if c in latest.columns]].sort_values(
            "graph_diffusion_score", ascending=False
        ).to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_graph_latest.csv", index=False)
        return out, {"status": "ok", "reason": "graph_done_eval_skipped", "feature_count": len(feature_cols)}

    y = eval_df["stall_next"].astype(int).to_numpy()
    p_base = np.clip(eval_df["stall_probability"].astype(float).to_numpy(), 0.0, 1.0)
    if "dynamic_hazard_probability" in eval_df.columns:
        p_dyn = np.clip(eval_df["dynamic_hazard_probability"].astype(float).to_numpy(), 0.0, 1.0)
    elif "dynamic_hazard_fused_probability" in eval_df.columns:
        p_dyn = np.clip(eval_df["dynamic_hazard_fused_probability"].astype(float).to_numpy(), 0.0, 1.0)
    else:
        p_dyn = p_base.copy()
    p_graph = np.clip(eval_df["graph_diffusion_probability"].astype(float).to_numpy(), 0.0, 1.0)

    score_base, met_base = _cls_objective(y, p_base)
    score_dyn, met_dyn = _cls_objective(y, p_dyn)
    score_graph, met_graph = _cls_objective(y, p_graph)
    curve_rows.append(
        {
            "candidate": "base",
            "graph_weight": 0.0,
            "objective": float(score_base),
            "roc_auc": met_base.get("roc_auc"),
            "average_precision": met_base.get("average_precision"),
            "brier": met_base.get("brier"),
        }
    )
    curve_rows.append(
        {
            "candidate": "dynamic_only",
            "graph_weight": 0.0,
            "objective": float(score_dyn),
            "roc_auc": met_dyn.get("roc_auc"),
            "average_precision": met_dyn.get("average_precision"),
            "brier": met_dyn.get("brier"),
        }
    )
    curve_rows.append(
        {
            "candidate": "graph_only",
            "graph_weight": 1.0,
            "objective": float(score_graph),
            "roc_auc": met_graph.get("roc_auc"),
            "average_precision": met_graph.get("average_precision"),
            "brier": met_graph.get("brier"),
        }
    )

    best = {
        "candidate": "base",
        "graph_weight": 0.0,
        "score": float(score_base),
        "metrics": met_base,
    }
    for gw in np.linspace(0.0, 1.0, 21):
        p_mix = np.clip((1.0 - float(gw)) * p_dyn + float(gw) * p_graph, 0.0, 1.0)
        score, met = _cls_objective(y, p_mix)
        curve_rows.append(
            {
                "candidate": "dynamic_graph_blend",
                "graph_weight": float(gw),
                "objective": float(score),
                "roc_auc": met.get("roc_auc"),
                "average_precision": met.get("average_precision"),
                "brier": met.get("brier"),
            }
        )
        if score > float(best["score"]) + 1e-10:
            best = {
                "candidate": "dynamic_graph_blend",
                "graph_weight": float(gw),
                "score": float(score),
                "metrics": met,
            }

    p_all_base = np.clip(out["stall_probability"].astype(float).to_numpy(), 0.0, 1.0)
    p_all_dyn = (
        np.clip(out["dynamic_hazard_probability"].astype(float).to_numpy(), 0.0, 1.0)
        if "dynamic_hazard_probability" in out.columns
        else np.clip(out["dynamic_hazard_fused_probability"].astype(float).to_numpy(), 0.0, 1.0)
        if "dynamic_hazard_fused_probability" in out.columns
        else p_all_base
    )
    p_all_graph = np.clip(out["graph_diffusion_probability"].astype(float).to_numpy(), 0.0, 1.0)
    if str(best["candidate"]) == "dynamic_graph_blend":
        p_sel_all = np.clip(
            (1.0 - float(best["graph_weight"])) * p_all_dyn + float(best["graph_weight"]) * p_all_graph,
            0.0,
            1.0,
        )
    elif str(best["candidate"]) == "dynamic_only":
        p_sel_all = p_all_dyn
    elif str(best["candidate"]) == "graph_only":
        p_sel_all = p_all_graph
    else:
        p_sel_all = p_all_base

    out["graph_diffusion_fused_probability"] = p_sel_all
    out["graph_diffusion_fused_score"] = np.clip(100.0 * p_sel_all, 0.0, 100.0)

    pd.DataFrame(curve_rows).to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_graph_curve.csv", index=False)
    latest = out[out["year"] == latest_year].copy()
    latest_cols = [
        "city_id",
        "city_name",
        "country",
        "continent",
        "year",
        "stall_risk_score",
        "dynamic_hazard_fused_score",
        "graph_diffusion_score",
        "graph_diffusion_fused_score",
        "kinetic_state",
        "trajectory_regime",
    ]
    latest[[c for c in latest_cols if c in latest.columns]].sort_values(
        "graph_diffusion_score", ascending=False
    ).to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_graph_latest.csv", index=False)

    delta_auc = None
    if best["metrics"].get("roc_auc") is not None and met_base.get("roc_auc") is not None:
        delta_auc = float(best["metrics"]["roc_auc"] - met_base["roc_auc"])
    if str(best["candidate"]) == "dynamic_graph_blend":
        p_sel_eval = np.clip((1.0 - float(best["graph_weight"])) * p_dyn + float(best["graph_weight"]) * p_graph, 0.0, 1.0)
    elif str(best["candidate"]) == "dynamic_only":
        p_sel_eval = p_dyn
    elif str(best["candidate"]) == "graph_only":
        p_sel_eval = p_graph
    else:
        p_sel_eval = p_base
    auc_sig = _bootstrap_metric_diff(
        y,
        p_sel_eval,
        p_base,
        metric="roc_auc",
        n_boot=500,
        random_state=2046,
    )
    brier_sig = _bootstrap_metric_diff(
        y,
        p_sel_eval,
        p_base,
        metric="brier",
        n_boot=500,
        random_state=2047,
    )
    return out, {
        "status": "ok",
        "feature_count": int(len(feature_cols)),
        "neighbors": int(k),
        "diffusion_steps": int(iters),
        "selected_candidate": str(best["candidate"]),
        "selected_graph_weight": float(best["graph_weight"]),
        "base_metrics": met_base,
        "selected_metrics": best["metrics"],
        "delta_auc_vs_base": delta_auc,
        "delta_brier_vs_base": float(best["metrics"]["brier"] - met_base["brier"]),
        "delta_auc_ci95": auc_sig.get("ci95") if auc_sig.get("status") == "ok" else [None, None],
        "delta_auc_p_value": auc_sig.get("p_value_two_sided") if auc_sig.get("status") == "ok" else None,
        "delta_brier_improve_ci95": brier_sig.get("ci95") if brier_sig.get("status") == "ok" else [None, None],
        "delta_brier_improve_p_value": brier_sig.get("p_value_two_sided") if brier_sig.get("status") == "ok" else None,
    }


def _build_dynamic_global_cycle(
    dyn: pd.DataFrame,
) -> Dict[str, Any]:
    if dyn.empty or "kinetic_state" not in dyn.columns:
        pd.DataFrame().to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_cycle.csv", index=False)
        return {"status": "skipped", "reason": "missing_dynamic_states"}

    yearly = (
        dyn.groupby("year", as_index=False)
        .agg(
            mean_growth=("growth_1", "mean"),
            mean_accel=("accel_1", "mean"),
            mean_jerk=("jerk_1", "mean"),
            mean_stall_risk=("stall_risk_score", "mean"),
            mean_kinetic_energy=("kinetic_energy_score", "mean"),
            mean_turning_risk=("turning_point_risk", "mean"),
            city_count=("city_id", "nunique"),
        )
        .sort_values("year")
    )

    share = (
        dyn.groupby(["year", "kinetic_state"], as_index=False)
        .agg(n=("city_id", "nunique"))
        .sort_values(["year", "kinetic_state"])
    )
    share["share"] = share["n"] / share.groupby("year")["n"].transform("sum")
    pivot = share.pivot_table(index="year", columns="kinetic_state", values="share", fill_value=0.0).reset_index()
    out = yearly.merge(pivot, on="year", how="left").fillna(0.0)

    for col in ["overheating", "deep_cooling", "recovery_rebound", "surge_expansion", "steady_cruise", "mixed_transition"]:
        if col not in out.columns:
            out[col] = 0.0
        out = out.rename(columns={col: f"share_{col}"})

    out["cycle_tension"] = (
        out.get("share_overheating", 0.0)
        + out.get("share_deep_cooling", 0.0)
        - out.get("share_recovery_rebound", 0.0)
        - out.get("share_surge_expansion", 0.0)
    )
    out["cycle_balance"] = 1.0 - (
        out.get("share_overheating", 0.0) + out.get("share_deep_cooling", 0.0)
    )
    q_low = float(out["cycle_tension"].quantile(0.33))
    q_high = float(out["cycle_tension"].quantile(0.67))
    out["global_cycle_phase"] = np.select(
        [out["cycle_tension"] >= q_high, out["cycle_tension"] <= q_low],
        ["stress_phase", "expansion_phase"],
        default="rebalancing_phase",
    )
    out.to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_cycle.csv", index=False)

    latest = out.sort_values("year").iloc[-1] if not out.empty else None
    return {
        "status": "ok",
        "years": int(len(out)),
        "latest_year": int(latest["year"]) if latest is not None else None,
        "latest_cycle_phase": str(latest["global_cycle_phase"]) if latest is not None else None,
        "latest_cycle_tension": float(latest["cycle_tension"]) if latest is not None else None,
        "tension_quantiles": {"q33": q_low, "q67": q_high},
    }


def _apply_dynamic_hazard_fusion_to_main_risk(
    dyn: pd.DataFrame,
    latest_year: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = dyn.copy()
    if "dynamic_hazard_fused_probability" not in out.columns or "stall_probability" not in out.columns:
        return out, {"status": "skipped", "reason": "missing_fusion_columns"}

    eval_year = int(latest_year - 1)
    selection_year = int(latest_year - 2)

    eval_df = out[(out["year"] == eval_year) & out["stall_next"].notna()].copy()
    if len(eval_df) < 50 or eval_df["stall_next"].nunique() < 2:
        return out, {"status": "skipped", "reason": "insufficient_eval_rows"}

    selection_df = out[(out["year"] == selection_year) & out["stall_next"].notna()].copy()
    selection_design = "validation_year_selection"
    if len(selection_df) < 60 or selection_df["stall_next"].nunique() < 2:
        selection_df = eval_df.copy()
        selection_year = eval_year
        selection_design = "fallback_eval_selection_due_low_validation_support"

    dyn_prob_col = "dynamic_hazard_probability" if "dynamic_hazard_probability" in out.columns else "dynamic_hazard_fused_probability"
    graph_enabled = "graph_diffusion_fused_probability" in out.columns and out["graph_diffusion_fused_probability"].notna().any()
    graph_prob_col = "graph_diffusion_probability" if "graph_diffusion_probability" in out.columns else "graph_diffusion_fused_probability"
    critical_enabled = "critical_transition_probability" in out.columns and out["critical_transition_probability"].notna().any()
    critical_prob_col = "critical_transition_probability"

    def _extract_prob_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y_arr = df["stall_next"].astype(int).to_numpy()
        p_base_arr = np.clip(df["stall_probability"].astype(float).to_numpy(), 0.0, 1.0)
        p_dyn_arr = np.clip(df[dyn_prob_col].astype(float).to_numpy(), 0.0, 1.0)
        if graph_enabled and graph_prob_col in df.columns:
            p_graph_arr = np.clip(df[graph_prob_col].astype(float).to_numpy(), 0.0, 1.0)
            p_graph_arr = np.where(np.isfinite(p_graph_arr), p_graph_arr, p_dyn_arr)
        else:
            p_graph_arr = np.zeros_like(p_base_arr)
        if critical_enabled and critical_prob_col in df.columns:
            p_critical_arr = np.clip(df[critical_prob_col].astype(float).to_numpy(), 0.0, 1.0)
            p_critical_arr = np.where(np.isfinite(p_critical_arr), p_critical_arr, p_dyn_arr)
        else:
            p_critical_arr = np.zeros_like(p_base_arr)
        return y_arr, p_base_arr, p_dyn_arr, p_graph_arr, p_critical_arr

    y_sel, p_base_sel, p_dyn_sel, p_graph_sel, p_critical_sel = _extract_prob_arrays(selection_df)
    y_eval, p_base_eval, p_dyn_eval, p_graph_eval, p_critical_eval = _extract_prob_arrays(eval_df)
    score_base_sel, met_base_sel = _cls_objective(y_sel, p_base_sel)
    score_base_eval, met_base_eval = _cls_objective(y_eval, p_base_eval)
    compact_runtime_mode = bool(len(out) >= 2500 or len(selection_df) >= 180)
    temporal_window_cap = 3 if compact_runtime_mode else 5
    alpha_dyn_grid = np.linspace(0.0, 0.45, 10 if compact_runtime_mode else 19)
    graph_grid = np.linspace(0.0, 0.25, 6 if compact_runtime_mode else 11) if graph_enabled else np.array([0.0])
    critical_grid = (
        np.linspace(0.0, 0.20, 5 if compact_runtime_mode else 9) if critical_enabled else np.array([0.0])
    )
    gate_dyn_grid = np.linspace(0.0, 0.40, 9 if compact_runtime_mode else 17)
    gate_graph_grid = (
        np.linspace(0.0, 0.20, 5 if compact_runtime_mode else 9) if graph_enabled else np.array([0.0])
    )
    gate_critical_grid = (
        np.linspace(0.0, 0.18, 5 if compact_runtime_mode else 10) if critical_enabled else np.array([0.0])
    )
    local_offsets = (
        [-0.10, -0.05, 0.0, 0.05, 0.10]
        if compact_runtime_mode
        else [-0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.10]
    )
    lambda_grid = np.linspace(0.0, 1.0, 6 if compact_runtime_mode else 11)
    gamma_grid = np.linspace(0.55, 1.0, 6 if compact_runtime_mode else 10)
    eval_bootstrap_draws = 220 if compact_runtime_mode else 600
    backtest_bootstrap_draws = 260 if compact_runtime_mode else 700
    continent_bootstrap_draws = 180 if compact_runtime_mode else 450

    temporal_years_all = sorted(
        out.loc[out["stall_next"].notna(), "year"].dropna().astype(int).unique().tolist()
    )
    temporal_eval_years = [yy for yy in temporal_years_all if yy <= selection_year]
    temporal_eval_years = temporal_eval_years[-temporal_window_cap:]
    temporal_windows: List[Dict[str, Any]] = []
    for ty in temporal_eval_years:
        tdf = out[(out["year"] == ty) & out["stall_next"].notna()].copy()
        if len(tdf) < 80 or tdf["stall_next"].nunique() < 2:
            continue
        y_t, p_base_t, p_dyn_t, p_graph_t, p_critical_t = _extract_prob_arrays(tdf)
        t_obj_base, _ = _cls_objective(y_t, p_base_t)
        temporal_windows.append(
            {
                "year": int(ty),
                "y": y_t,
                "p_base": p_base_t,
                "p_dyn": p_dyn_t,
                "p_graph": p_graph_t,
                "p_critical": p_critical_t,
                "base_objective": float(t_obj_base),
            }
        )

    def _eval_temporal_blend(alpha_dyn: float, alpha_graph: float, alpha_critical: float) -> Dict[str, Any]:
        if not temporal_windows:
            return {
                "status": "skipped",
                "reason": "insufficient_temporal_windows",
                "n_windows": 0,
                "years": [],
                "mean_objective": None,
                "std_objective": None,
                "mean_gain_vs_base": None,
                "share_positive_gain": None,
            }
        a_dyn = float(alpha_dyn)
        a_graph = float(alpha_graph)
        a_critical = float(alpha_critical)
        a_base = max(0.0, 1.0 - a_dyn - a_graph - a_critical)
        objs: List[float] = []
        gains: List[float] = []
        for w in temporal_windows:
            p_t = np.clip(
                a_base * w["p_base"] + a_dyn * w["p_dyn"] + a_graph * w["p_graph"] + a_critical * w["p_critical"],
                0.0,
                1.0,
            )
            obj_t, _ = _cls_objective(w["y"], p_t)
            objs.append(float(obj_t))
            gains.append(float(obj_t - float(w["base_objective"])))
        obj_arr = np.asarray(objs, dtype=float)
        gain_arr = np.asarray(gains, dtype=float)
        return {
            "status": "ok",
            "n_windows": int(len(objs)),
            "years": [int(w["year"]) for w in temporal_windows],
            "mean_objective": float(obj_arr.mean()),
            "std_objective": float(obj_arr.std(ddof=0)),
            "mean_gain_vs_base": float(gain_arr.mean()),
            "share_positive_gain": float(np.mean(gain_arr > 0.0)),
        }

    def _rank_score(selection_objective: float, temporal_eval: Dict[str, Any]) -> float:
        if temporal_eval.get("status") != "ok":
            return float(selection_objective)
        t_mean = float(temporal_eval.get("mean_objective") or 0.0)
        t_std = float(temporal_eval.get("std_objective") or 0.0)
        t_gain = float(temporal_eval.get("mean_gain_vs_base") or 0.0)
        return float(t_mean + 0.25 * t_gain - 0.06 * t_std + 0.02 * (float(selection_objective) - float(score_base_sel)))

    def _search_best_convex_alpha(
        y_arr: np.ndarray,
        p_base_arr: np.ndarray,
        p_dyn_arr: np.ndarray,
        p_graph_arr: np.ndarray,
        p_critical_arr: np.ndarray,
        dyn_grid_local: np.ndarray,
        graph_grid_local: np.ndarray,
        critical_grid_local: np.ndarray,
    ) -> Dict[str, Any]:
        base_score, base_metrics = _cls_objective(y_arr, p_base_arr)
        best_local: Dict[str, Any] = {
            "a_dyn": 0.0,
            "a_graph": 0.0,
            "a_critical": 0.0,
            "a_base": 1.0,
            "score": float(base_score),
            "metrics": base_metrics,
        }
        for a_dyn_raw in dyn_grid_local:
            for a_graph_raw in graph_grid_local:
                for a_critical_raw in critical_grid_local:
                    a_dyn = float(a_dyn_raw)
                    a_graph = float(a_graph_raw)
                    a_critical = float(a_critical_raw)
                    if (a_dyn + a_graph + a_critical) > 0.72:
                        continue
                    a_base = max(0.0, 1.0 - a_dyn - a_graph - a_critical)
                    p_arr = np.clip(
                        a_base * p_base_arr + a_dyn * p_dyn_arr + a_graph * p_graph_arr + a_critical * p_critical_arr,
                        0.0,
                        1.0,
                    )
                    score, metrics = _cls_objective(y_arr, p_arr)
                    if float(score) > float(best_local["score"]) + 1e-12:
                        best_local = {
                            "a_dyn": a_dyn,
                            "a_graph": a_graph,
                            "a_critical": a_critical,
                            "a_base": a_base,
                            "score": float(score),
                            "metrics": metrics,
                        }
        return best_local

    def _predict_state_gated_probs(
        df: pd.DataFrame,
        p_base_arr: np.ndarray,
        p_dyn_arr: np.ndarray,
        p_graph_arr: np.ndarray,
        p_critical_arr: np.ndarray,
        gate_column: str,
        group_alpha_map: Dict[str, Dict[str, float]],
        global_alpha: Dict[str, float],
    ) -> np.ndarray:
        if gate_column not in df.columns:
            return np.clip(
                float(global_alpha["a_base"]) * p_base_arr
                + float(global_alpha["a_dyn"]) * p_dyn_arr
                + float(global_alpha["a_graph"]) * p_graph_arr
                + float(global_alpha["a_critical"]) * p_critical_arr,
                0.0,
                1.0,
            )
        g = df[gate_column].fillna("missing").astype(str).to_numpy(dtype=object)
        p = np.zeros_like(p_base_arr, dtype=float)
        for gv in np.unique(g):
            idx = g == gv
            alpha = group_alpha_map.get(str(gv), global_alpha)
            p[idx] = (
                float(alpha["a_base"]) * p_base_arr[idx]
                + float(alpha["a_dyn"]) * p_dyn_arr[idx]
                + float(alpha["a_graph"]) * p_graph_arr[idx]
                + float(alpha["a_critical"]) * p_critical_arr[idx]
            )
        return np.clip(p, 0.0, 1.0)

    def _eval_temporal_state_gated(
        gate_column: str,
        group_alpha_map: Dict[str, Dict[str, float]],
        global_alpha: Dict[str, float],
    ) -> Dict[str, Any]:
        years_for_eval = [yy for yy in temporal_years_all if yy <= selection_year][-temporal_window_cap:]
        if not years_for_eval:
            return {
                "status": "skipped",
                "reason": "insufficient_temporal_windows",
                "n_windows": 0,
                "years": [],
                "mean_objective": None,
                "std_objective": None,
                "mean_gain_vs_base": None,
                "share_positive_gain": None,
            }
        objs: List[float] = []
        gains: List[float] = []
        years_kept: List[int] = []
        for ty in years_for_eval:
            tdf = out[(out["year"] == int(ty)) & out["stall_next"].notna()].copy()
            if len(tdf) < 80 or tdf["stall_next"].nunique() < 2:
                continue
            y_t, p_base_t, p_dyn_t, p_graph_t, p_critical_t = _extract_prob_arrays(tdf)
            p_t = _predict_state_gated_probs(
                tdf,
                p_base_t,
                p_dyn_t,
                p_graph_t,
                p_critical_t,
                gate_column=gate_column,
                group_alpha_map=group_alpha_map,
                global_alpha=global_alpha,
            )
            obj_t, _ = _cls_objective(y_t, p_t)
            obj_base_t, _ = _cls_objective(y_t, p_base_t)
            objs.append(float(obj_t))
            gains.append(float(obj_t - obj_base_t))
            years_kept.append(int(ty))
        if not objs:
            return {
                "status": "skipped",
                "reason": "insufficient_temporal_windows",
                "n_windows": 0,
                "years": [],
                "mean_objective": None,
                "std_objective": None,
                "mean_gain_vs_base": None,
                "share_positive_gain": None,
            }
        obj_arr = np.asarray(objs, dtype=float)
        gain_arr = np.asarray(gains, dtype=float)
        return {
            "status": "ok",
            "n_windows": int(len(objs)),
            "years": years_kept,
            "mean_objective": float(obj_arr.mean()),
            "std_objective": float(obj_arr.std(ddof=0)),
            "mean_gain_vs_base": float(gain_arr.mean()),
            "share_positive_gain": float(np.mean(gain_arr > 0.0)),
        }

    base_temporal = _eval_temporal_blend(0.0, 0.0, 0.0)
    base_rank_score = _rank_score(float(score_base_sel), base_temporal)

    rows: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {
        "candidate": "base",
        "alpha_dynamic": 0.0,
        "alpha_graph": 0.0,
        "alpha_critical": 0.0,
        "alpha_base": 1.0,
        "selection_score": float(score_base_sel),
        "rank_score": float(base_rank_score),
        "selection_metrics": met_base_sel,
        "eval_metrics": met_base_eval,
        "temporal": base_temporal,
        "p_eval": p_base_eval,
        "p_all": None,
    }

    for alpha_dyn in alpha_dyn_grid:
        for alpha_graph in graph_grid:
            for alpha_critical in critical_grid:
                a_dyn = float(alpha_dyn)
                a_graph = float(alpha_graph)
                a_critical = float(alpha_critical)
                if (a_dyn + a_graph + a_critical) > 0.72:
                    continue
                a_base = max(0.0, 1.0 - a_dyn - a_graph - a_critical)
                p_sel_arr = np.clip(
                    a_base * p_base_sel + a_dyn * p_dyn_sel + a_graph * p_graph_sel + a_critical * p_critical_sel,
                    0.0,
                    1.0,
                )
                p_eval_arr = np.clip(
                    a_base * p_base_eval + a_dyn * p_dyn_eval + a_graph * p_graph_eval + a_critical * p_critical_eval,
                    0.0,
                    1.0,
                )
                score_sel, met_sel = _cls_objective(y_sel, p_sel_arr)
                score_eval, met_eval = _cls_objective(y_eval, p_eval_arr)
                temporal_blend = _eval_temporal_blend(a_dyn, a_graph, a_critical)
                rank_score = _rank_score(float(score_sel), temporal_blend)
                rows.append(
                    {
                        "candidate": "convex_blend",
                        "alpha_dynamic": a_dyn,
                        "alpha_graph": a_graph,
                        "alpha_critical": a_critical,
                        "alpha_base": a_base,
                        "selection_year": int(selection_year),
                        "eval_year": int(eval_year),
                        "selection_objective": float(score_sel),
                        "selection_roc_auc": met_sel.get("roc_auc"),
                        "selection_average_precision": met_sel.get("average_precision"),
                        "selection_brier": met_sel.get("brier"),
                        "temporal_objective_mean": temporal_blend.get("mean_objective"),
                        "temporal_objective_std": temporal_blend.get("std_objective"),
                        "temporal_mean_gain_vs_base": temporal_blend.get("mean_gain_vs_base"),
                        "temporal_share_positive_gain": temporal_blend.get("share_positive_gain"),
                        "temporal_windows": temporal_blend.get("n_windows"),
                        "selection_rank_score": float(rank_score),
                        "eval_objective": float(score_eval),
                        "eval_roc_auc": met_eval.get("roc_auc"),
                        "eval_average_precision": met_eval.get("average_precision"),
                        "eval_brier": met_eval.get("brier"),
                    }
                )
                if (
                    float(rank_score) > float(best.get("rank_score", -np.inf)) + 1e-10
                    or (
                        abs(float(rank_score) - float(best.get("rank_score", -np.inf))) <= 1e-10
                        and float(score_sel) > float(best["selection_score"]) + 1e-10
                    )
                ):
                    best = {
                        "candidate": "convex_blend",
                        "alpha_dynamic": a_dyn,
                        "alpha_graph": a_graph,
                        "alpha_critical": a_critical,
                        "alpha_base": a_base,
                        "selection_score": float(score_sel),
                        "rank_score": float(rank_score),
                        "selection_metrics": met_sel,
                        "eval_metrics": met_eval,
                        "temporal": temporal_blend,
                        "p_eval": p_eval_arr,
                        "p_all": None,
                    }

    continent_robust_info: Dict[str, Any] = {
        "status": "skipped",
        "reason": "insufficient_selection_continent_support",
    }
    robust_export_columns = [
        "alpha_dynamic",
        "alpha_graph",
        "alpha_critical",
        "alpha_base",
        "selection_objective",
        "mean_continent_gain",
        "std_continent_gain",
        "worst_continent_gain",
        "share_positive_continent_gain",
        "robust_score",
    ]
    robust_export_path = DATA_OUTPUTS / "pulse_ai_dynamic_main_fusion_continent_robust_curve.csv"
    robust_rows: List[Dict[str, Any]] = []
    selection_continent_blocks: List[Dict[str, Any]] = []
    if "continent" in selection_df.columns:
        cont_arr = selection_df["continent"].fillna("unknown").astype(str).to_numpy(dtype=object)
        for cont_name in sorted({str(x) for x in cont_arr.tolist()}):
            idx = np.where(cont_arr == cont_name)[0].astype(int)
            if len(idx) < 45:
                continue
            y_c = y_sel[idx]
            if np.unique(y_c).size < 2:
                continue
            p_base_c = p_base_sel[idx]
            p_dyn_c = p_dyn_sel[idx]
            p_graph_c = p_graph_sel[idx]
            p_critical_c = p_critical_sel[idx]
            base_obj_c, _ = _cls_objective(y_c, p_base_c)
            selection_continent_blocks.append(
                {
                    "continent": cont_name,
                    "idx": idx,
                    "y": y_c,
                    "p_base": p_base_c,
                    "p_dyn": p_dyn_c,
                    "p_graph": p_graph_c,
                    "p_critical": p_critical_c,
                    "base_objective": float(base_obj_c),
                }
            )

    if len(selection_continent_blocks) >= 2:
        convex_rows = [r for r in rows if str(r.get("candidate")) == "convex_blend"]
        robust_best_row: Dict[str, Any] | None = None
        robust_best_score = -np.inf
        for rr in convex_rows:
            a_dyn = float(rr.get("alpha_dynamic") or 0.0)
            a_graph = float(rr.get("alpha_graph") or 0.0)
            a_critical = float(rr.get("alpha_critical") or 0.0)
            a_base = max(0.0, 1.0 - a_dyn - a_graph - a_critical)
            gains: List[float] = []
            for blk in selection_continent_blocks:
                p_c = np.clip(
                    a_base * blk["p_base"] + a_dyn * blk["p_dyn"] + a_graph * blk["p_graph"] + a_critical * blk["p_critical"],
                    0.0,
                    1.0,
                )
                obj_c, _ = _cls_objective(blk["y"], p_c)
                gains.append(float(obj_c - float(blk["base_objective"])))
            gain_arr = np.asarray(gains, dtype=float)
            mean_gain = float(gain_arr.mean())
            std_gain = float(gain_arr.std(ddof=0))
            worst_gain = float(gain_arr.min())
            share_pos = float(np.mean(gain_arr > 0.0))
            robust_score = float(
                mean_gain
                - 0.70 * max(-worst_gain, 0.0)
                - 0.08 * std_gain
                + 0.02 * (float(rr.get("selection_objective") or 0.0) - float(score_base_sel))
            )
            robust_row = {
                "alpha_dynamic": a_dyn,
                "alpha_graph": a_graph,
                "alpha_critical": a_critical,
                "alpha_base": a_base,
                "selection_objective": float(rr.get("selection_objective") or 0.0),
                "mean_continent_gain": mean_gain,
                "std_continent_gain": std_gain,
                "worst_continent_gain": worst_gain,
                "share_positive_continent_gain": share_pos,
                "robust_score": robust_score,
            }
            robust_rows.append(robust_row)
            if robust_score > robust_best_score + 1e-12:
                robust_best_score = robust_score
                robust_best_row = robust_row

        if robust_best_row is not None:
            a_dyn_rb = float(robust_best_row["alpha_dynamic"])
            a_graph_rb = float(robust_best_row["alpha_graph"])
            a_critical_rb = float(robust_best_row["alpha_critical"])
            a_base_rb = float(robust_best_row["alpha_base"])
            p_sel_rb = np.clip(
                a_base_rb * p_base_sel + a_dyn_rb * p_dyn_sel + a_graph_rb * p_graph_sel + a_critical_rb * p_critical_sel,
                0.0,
                1.0,
            )
            p_eval_rb = np.clip(
                a_base_rb * p_base_eval + a_dyn_rb * p_dyn_eval + a_graph_rb * p_graph_eval + a_critical_rb * p_critical_eval,
                0.0,
                1.0,
            )
            p_all_rb = np.clip(
                a_base_rb * np.clip(out["stall_probability"].astype(float).to_numpy(), 0.0, 1.0)
                + a_dyn_rb * np.clip(out[dyn_prob_col].astype(float).to_numpy(), 0.0, 1.0)
                + a_graph_rb
                * (
                    np.clip(out[graph_prob_col].astype(float).to_numpy(), 0.0, 1.0)
                    if graph_enabled and graph_prob_col in out.columns
                    else np.zeros(len(out), dtype=float)
                )
                + a_critical_rb
                * (
                    np.clip(out[critical_prob_col].astype(float).to_numpy(), 0.0, 1.0)
                    if critical_enabled and critical_prob_col in out.columns
                    else np.zeros(len(out), dtype=float)
                ),
                0.0,
                1.0,
            )
            score_sel_rb, met_sel_rb = _cls_objective(y_sel, p_sel_rb)
            score_eval_rb, met_eval_rb = _cls_objective(y_eval, p_eval_rb)
            temporal_rb = _eval_temporal_blend(a_dyn_rb, a_graph_rb, a_critical_rb)
            rank_rb = _rank_score(float(score_sel_rb), temporal_rb) + 0.03 * float(robust_best_row["robust_score"])

            rows.append(
                {
                    "candidate": "continent_robust_blend",
                    "alpha_dynamic": a_dyn_rb,
                    "alpha_graph": a_graph_rb,
                    "alpha_critical": a_critical_rb,
                    "alpha_base": a_base_rb,
                    "selection_year": int(selection_year),
                    "eval_year": int(eval_year),
                    "selection_objective": float(score_sel_rb),
                    "selection_roc_auc": met_sel_rb.get("roc_auc"),
                    "selection_average_precision": met_sel_rb.get("average_precision"),
                    "selection_brier": met_sel_rb.get("brier"),
                    "temporal_objective_mean": temporal_rb.get("mean_objective"),
                    "temporal_objective_std": temporal_rb.get("std_objective"),
                    "temporal_mean_gain_vs_base": temporal_rb.get("mean_gain_vs_base"),
                    "temporal_share_positive_gain": temporal_rb.get("share_positive_gain"),
                    "temporal_windows": temporal_rb.get("n_windows"),
                    "selection_rank_score": float(rank_rb),
                    "eval_objective": float(score_eval_rb),
                    "eval_roc_auc": met_eval_rb.get("roc_auc"),
                    "eval_average_precision": met_eval_rb.get("average_precision"),
                    "eval_brier": met_eval_rb.get("brier"),
                }
            )
            top_robust = sorted(robust_rows, key=lambda d: float(d["robust_score"]), reverse=True)[:12]
            continent_robust_info = {
                "status": "ok",
                "n_continents_used": int(len(selection_continent_blocks)),
                "n_candidates_evaluated": int(len(robust_rows)),
                "best_alpha_dynamic": a_dyn_rb,
                "best_alpha_graph": a_graph_rb,
                "best_alpha_critical": a_critical_rb,
                "best_alpha_base": a_base_rb,
                "best_robust_score": float(robust_best_row["robust_score"]),
                "best_worst_continent_gain": float(robust_best_row["worst_continent_gain"]),
                "best_mean_continent_gain": float(robust_best_row["mean_continent_gain"]),
                "best_share_positive_continent_gain": float(robust_best_row["share_positive_continent_gain"]),
                "top_candidates": top_robust,
            }

            if (
                float(robust_best_row["worst_continent_gain"]) >= -0.010
                and (
                    float(rank_rb) > float(best.get("rank_score", -np.inf)) + 1e-10
                    or (
                        abs(float(rank_rb) - float(best.get("rank_score", -np.inf))) <= 1e-10
                        and float(score_sel_rb) > float(best["selection_score"]) + 1e-10
                    )
                )
            ):
                best = {
                    "candidate": "continent_robust_blend",
                    "alpha_dynamic": a_dyn_rb,
                    "alpha_graph": a_graph_rb,
                    "alpha_critical": a_critical_rb,
                    "alpha_base": a_base_rb,
                    "selection_score": float(score_sel_rb),
                    "rank_score": float(rank_rb),
                    "selection_metrics": met_sel_rb,
                    "eval_metrics": met_eval_rb,
                    "temporal": temporal_rb,
                    "p_eval": p_eval_rb,
                    "p_all": p_all_rb,
                }
    if robust_rows:
        pd.DataFrame(robust_rows, columns=robust_export_columns).to_csv(robust_export_path, index=False)
    else:
        pd.DataFrame(columns=robust_export_columns).to_csv(robust_export_path, index=False)

    def _continent_gain_stats(p_sel_vec: np.ndarray) -> Dict[str, Any]:
        if len(selection_continent_blocks) < 2:
            return {
                "status": "skipped",
                "reason": "insufficient_selection_continent_support",
                "mean_gain": None,
                "worst_gain": None,
                "std_gain": None,
                "share_positive_gain": None,
            }
        gains = []
        for blk in selection_continent_blocks:
            idx = blk["idx"]
            p_c = np.clip(np.asarray(p_sel_vec, dtype=float)[idx], 0.0, 1.0)
            obj_c, _ = _cls_objective(blk["y"], p_c)
            gains.append(float(obj_c - float(blk["base_objective"])))
        arr = np.asarray(gains, dtype=float)
        return {
            "status": "ok",
            "mean_gain": float(arr.mean()),
            "worst_gain": float(arr.min()),
            "std_gain": float(arr.std(ddof=0)),
            "share_positive_gain": float(np.mean(arr > 0.0)),
            "n_continents": int(len(arr)),
        }

    # Candidate: stacked meta-learner selected on validation year (or fallback eval year if needed).
    stacked_info: Dict[str, Any] = {"status": "skipped", "reason": "insufficient_meta_training_rows"}
    train_meta = out[(out["year"] <= (selection_year - 1)) & out["stall_next"].notna()].copy()
    if len(train_meta) >= 220 and train_meta["stall_next"].nunique() >= 2:
        num_candidates = [
            "stall_probability",
            "critical_transition_probability",
            "critical_transition_score",
            "critical_transition_risk_score",
            "dynamic_hazard_probability",
            "dynamic_hazard_fused_probability",
            "graph_diffusion_probability",
            "graph_diffusion_fused_probability",
            "acceleration_score",
            "turning_point_risk",
            "damping_pressure",
            "kinetic_energy_score",
            "regime_forward_risk",
            "regime_transition_entropy",
            "regime_switch_rate_3y",
            "regime_switch_rate_5y",
            "regime_run_length_log",
        ]
        num_cols = [c for c in num_candidates if c in out.columns]
        cat_cols = [c for c in ["kinetic_state", "trajectory_regime", "continent"] if c in out.columns]

        if len(num_cols) >= 4:
            for col in num_cols:
                train_meta[col] = pd.to_numeric(train_meta[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                selection_df[col] = pd.to_numeric(selection_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            med = train_meta[num_cols].median(numeric_only=True)
            train_num = train_meta[num_cols].fillna(med)
            selection_num = selection_df[num_cols].fillna(med)
            eval_num = eval_df[num_cols].fillna(med)
            all_num = out[num_cols].fillna(med)

            x_train_df = train_num.copy()
            x_sel_df = selection_num.copy()
            x_eval_df = eval_num.copy()
            x_all_df = all_num.copy()
            if cat_cols:
                cat_levels = {c: sorted(out[c].fillna("missing").astype(str).unique().tolist()) for c in cat_cols}
                for c in cat_cols:
                    tr = train_meta[c].fillna("missing").astype(str)
                    sl = selection_df[c].fillna("missing").astype(str)
                    ev = eval_df[c].fillna("missing").astype(str)
                    al = out[c].fillna("missing").astype(str)
                    for lv in cat_levels[c]:
                        cname = f"{c}__{lv}"
                        x_train_df[cname] = (tr == lv).astype(float).to_numpy()
                        x_sel_df[cname] = (sl == lv).astype(float).to_numpy()
                        x_eval_df[cname] = (ev == lv).astype(float).to_numpy()
                        x_all_df[cname] = (al == lv).astype(float).to_numpy()

            x_train = x_train_df.to_numpy(dtype=float)
            x_sel = x_sel_df.to_numpy(dtype=float)
            x_eval = x_eval_df.to_numpy(dtype=float)
            x_all = x_all_df.to_numpy(dtype=float)
            y_train = train_meta["stall_next"].astype(int).to_numpy()

            meta_candidates: List[tuple[str, Any]] = [
                (
                    "stacked_meta_logit",
                    make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            max_iter=1800,
                            C=0.85,
                            solver="lbfgs",
                            class_weight="balanced",
                            random_state=2060,
                        ),
                    ),
                ),
                (
                    "stacked_meta_gb",
                    GradientBoostingClassifier(
                        random_state=2061,
                        n_estimators=260,
                        learning_rate=0.04,
                        max_depth=2,
                        subsample=0.9,
                    ),
                ),
            ]
            model_rows: List[Dict[str, Any]] = []
            for cand_name, meta_model in meta_candidates:
                try:
                    learner_name = "logit" if "logit" in cand_name else "gb"
                    _fit_classifier_with_weights(meta_model, learner_name, x_train, y_train, sample_weight=None)
                    p_sel = np.clip(meta_model.predict_proba(x_sel)[:, 1], 0.0, 1.0)
                    p_eval_meta = np.clip(meta_model.predict_proba(x_eval)[:, 1], 0.0, 1.0)
                    p_all_meta = np.clip(meta_model.predict_proba(x_all)[:, 1], 0.0, 1.0)
                    score_sel, met_sel = _cls_objective(y_sel, p_sel)
                    score_eval, met_eval = _cls_objective(y_eval, p_eval_meta)
                    meta_rank_score = float(score_sel - 0.08)
                    rows.append(
                        {
                            "candidate": cand_name,
                            "alpha_dynamic": None,
                            "alpha_graph": None,
                            "alpha_critical": None,
                            "alpha_base": None,
                            "selection_year": int(selection_year),
                            "eval_year": int(eval_year),
                            "selection_objective": float(score_sel),
                            "selection_roc_auc": met_sel.get("roc_auc"),
                            "selection_average_precision": met_sel.get("average_precision"),
                            "selection_brier": met_sel.get("brier"),
                            "temporal_objective_mean": None,
                            "temporal_objective_std": None,
                            "temporal_mean_gain_vs_base": None,
                            "temporal_share_positive_gain": None,
                            "temporal_windows": 0,
                            "selection_rank_score": meta_rank_score,
                            "eval_objective": float(score_eval),
                            "eval_roc_auc": met_eval.get("roc_auc"),
                            "eval_average_precision": met_eval.get("average_precision"),
                            "eval_brier": met_eval.get("brier"),
                        }
                    )
                    model_rows.append(
                        {
                            "candidate": cand_name,
                            "selection_score": float(score_sel),
                            "rank_score": meta_rank_score,
                            "selection_metrics": met_sel,
                            "eval_metrics": met_eval,
                            "p_eval": p_eval_meta,
                            "p_all": p_all_meta,
                        }
                    )
                except Exception:  # noqa: BLE001
                    continue

            if model_rows:
                best_meta = max(model_rows, key=lambda d: float(d["selection_score"]))
                stacked_info = {
                    "status": "ok",
                    "selection_policy": "diagnostic_only_time_leakage_guard",
                    "n_train": int(len(train_meta)),
                    "feature_count": int(x_train_df.shape[1]),
                    "meta_candidates": [str(d["candidate"]) for d in model_rows],
                    "best_candidate_by_selection": str(best_meta["candidate"]),
                    "best_selection_score": float(best_meta["selection_score"]),
                    "best_rank_score": float(best_meta["rank_score"]),
                }

    state_gated_info: Dict[str, Any] = {"status": "skipped", "reason": "insufficient_training_rows"}
    gate_export_path = DATA_OUTPUTS / "pulse_ai_dynamic_main_fusion_state_gate.csv"
    gate_export_columns = [
        "gate_column",
        "gate_group",
        "train_rows",
        "specialized",
        "alpha_dynamic",
        "alpha_graph",
        "alpha_critical",
        "alpha_base",
        "train_objective",
    ]
    gate_rows_export: List[Dict[str, Any]] = []
    train_gate = out[(out["year"] <= (selection_year - 1)) & out["stall_next"].notna()].copy()
    gate_col: str | None = None
    for cand in ["kinetic_state", "trajectory_regime", "continent"]:
        if cand in train_gate.columns and int(train_gate[cand].fillna("missing").astype(str).nunique()) >= 3:
            gate_col = str(cand)
            break
    if (
        gate_col is not None
        and len(train_gate) >= 260
        and train_gate["stall_next"].nunique() >= 2
    ):
        y_train_gate, pb_train_gate, pd_train_gate, pg_train_gate, pc_train_gate = _extract_prob_arrays(train_gate)
        global_gate = _search_best_convex_alpha(
            y_train_gate,
            pb_train_gate,
            pd_train_gate,
            pg_train_gate,
            pc_train_gate,
            dyn_grid_local=gate_dyn_grid,
            graph_grid_local=gate_graph_grid,
            critical_grid_local=gate_critical_grid,
        )
        global_alpha = {
            "a_dyn": float(global_gate["a_dyn"]),
            "a_graph": float(global_gate["a_graph"]),
            "a_critical": float(global_gate["a_critical"]),
            "a_base": float(global_gate["a_base"]),
        }
        group_alpha_map: Dict[str, Dict[str, float]] = {}
        train_groups = train_gate[gate_col].fillna("missing").astype(str).to_numpy(dtype=object)
        group_min_rows = 120
        graph_cap = 0.25 if graph_enabled else 0.0
        critical_cap = 0.20 if critical_enabled else 0.0

        for gv in sorted({str(x) for x in train_groups.tolist()}):
            idx = np.where(train_groups == gv)[0].astype(int)
            n_g = int(len(idx))
            specialized = False
            alpha_local = {
                "a_dyn": float(global_alpha["a_dyn"]),
                "a_graph": float(global_alpha["a_graph"]),
                "a_critical": float(global_alpha["a_critical"]),
                "a_base": float(global_alpha["a_base"]),
            }
            obj_local = float(global_gate["score"])
            if n_g >= group_min_rows:
                y_g = y_train_gate[idx]
                if np.unique(y_g).size >= 2:
                    pb_g = pb_train_gate[idx]
                    pd_g = pd_train_gate[idx]
                    pg_g = pg_train_gate[idx]
                    pc_g = pc_train_gate[idx]
                    dyn_local = sorted(
                        {float(np.clip(float(global_alpha["a_dyn"]) + d, 0.0, 0.45)) for d in local_offsets}
                    )
                    graph_local = sorted(
                        {float(np.clip(float(global_alpha["a_graph"]) + d, 0.0, graph_cap)) for d in local_offsets}
                    )
                    critical_local = sorted(
                        {float(np.clip(float(global_alpha["a_critical"]) + d, 0.0, critical_cap)) for d in local_offsets}
                    )
                    local_best = _search_best_convex_alpha(
                        y_g,
                        pb_g,
                        pd_g,
                        pg_g,
                        pc_g,
                        dyn_grid_local=np.array(dyn_local, dtype=float),
                        graph_grid_local=np.array(graph_local, dtype=float),
                        critical_grid_local=np.array(critical_local, dtype=float),
                    )
                    shrink = float(n_g / (n_g + 220.0))
                    a_dyn = float(shrink * float(local_best["a_dyn"]) + (1.0 - shrink) * float(global_alpha["a_dyn"]))
                    a_graph = float(shrink * float(local_best["a_graph"]) + (1.0 - shrink) * float(global_alpha["a_graph"]))
                    a_critical = float(
                        shrink * float(local_best["a_critical"]) + (1.0 - shrink) * float(global_alpha["a_critical"])
                    )
                    if (a_dyn + a_graph + a_critical) <= 0.72 + 1e-12:
                        alpha_local = {
                            "a_dyn": a_dyn,
                            "a_graph": a_graph,
                            "a_critical": a_critical,
                            "a_base": float(max(0.0, 1.0 - a_dyn - a_graph - a_critical)),
                        }
                        specialized = True
                        p_g_local = np.clip(
                            alpha_local["a_base"] * pb_g
                            + alpha_local["a_dyn"] * pd_g
                            + alpha_local["a_graph"] * pg_g
                            + alpha_local["a_critical"] * pc_g,
                            0.0,
                            1.0,
                        )
                        obj_local, _ = _cls_objective(y_g, p_g_local)
            group_alpha_map[gv] = alpha_local
            gate_rows_export.append(
                {
                    "gate_column": gate_col,
                    "gate_group": gv,
                    "train_rows": int(n_g),
                    "specialized": int(specialized),
                    "alpha_dynamic": float(alpha_local["a_dyn"]),
                    "alpha_graph": float(alpha_local["a_graph"]),
                    "alpha_critical": float(alpha_local["a_critical"]),
                    "alpha_base": float(alpha_local["a_base"]),
                    "train_objective": float(obj_local),
                }
            )

        p_sel_gate = _predict_state_gated_probs(
            selection_df,
            p_base_sel,
            p_dyn_sel,
            p_graph_sel,
            p_critical_sel,
            gate_column=gate_col,
            group_alpha_map=group_alpha_map,
            global_alpha=global_alpha,
        )
        p_eval_gate = _predict_state_gated_probs(
            eval_df,
            p_base_eval,
            p_dyn_eval,
            p_graph_eval,
            p_critical_eval,
            gate_column=gate_col,
            group_alpha_map=group_alpha_map,
            global_alpha=global_alpha,
        )
        p_all_base_gate = np.clip(out["stall_probability"].astype(float).to_numpy(), 0.0, 1.0)
        p_all_dyn_gate = np.clip(out[dyn_prob_col].astype(float).to_numpy(), 0.0, 1.0)
        if graph_enabled and graph_prob_col in out.columns:
            p_all_graph_gate = np.clip(out[graph_prob_col].astype(float).to_numpy(), 0.0, 1.0)
            p_all_graph_gate = np.where(np.isfinite(p_all_graph_gate), p_all_graph_gate, p_all_dyn_gate)
        else:
            p_all_graph_gate = np.zeros(len(out), dtype=float)
        if critical_enabled and critical_prob_col in out.columns:
            p_all_critical_gate = np.clip(out[critical_prob_col].astype(float).to_numpy(), 0.0, 1.0)
            p_all_critical_gate = np.where(np.isfinite(p_all_critical_gate), p_all_critical_gate, p_all_dyn_gate)
        else:
            p_all_critical_gate = np.zeros(len(out), dtype=float)

        p_all_gate = _predict_state_gated_probs(
            out,
            p_all_base_gate,
            p_all_dyn_gate,
            p_all_graph_gate,
            p_all_critical_gate,
            gate_column=gate_col,
            group_alpha_map=group_alpha_map,
            global_alpha=global_alpha,
        )
        score_sel_gate, met_sel_gate = _cls_objective(y_sel, p_sel_gate)
        score_eval_gate, met_eval_gate = _cls_objective(y_eval, p_eval_gate)
        temporal_gate = _eval_temporal_state_gated(
            gate_column=gate_col,
            group_alpha_map=group_alpha_map,
            global_alpha=global_alpha,
        )
        rank_gate = _rank_score(float(score_sel_gate), temporal_gate)
        rows.append(
            {
                "candidate": "state_gated_moe",
                "alpha_dynamic": None,
                "alpha_graph": None,
                "alpha_critical": None,
                "alpha_base": None,
                "selection_year": int(selection_year),
                "eval_year": int(eval_year),
                "selection_objective": float(score_sel_gate),
                "selection_roc_auc": met_sel_gate.get("roc_auc"),
                "selection_average_precision": met_sel_gate.get("average_precision"),
                "selection_brier": met_sel_gate.get("brier"),
                "temporal_objective_mean": temporal_gate.get("mean_objective"),
                "temporal_objective_std": temporal_gate.get("std_objective"),
                "temporal_mean_gain_vs_base": temporal_gate.get("mean_gain_vs_base"),
                "temporal_share_positive_gain": temporal_gate.get("share_positive_gain"),
                "temporal_windows": temporal_gate.get("n_windows"),
                "selection_rank_score": float(rank_gate),
                "eval_objective": float(score_eval_gate),
                "eval_roc_auc": met_eval_gate.get("roc_auc"),
                "eval_average_precision": met_eval_gate.get("average_precision"),
                "eval_brier": met_eval_gate.get("brier"),
            }
        )
        state_gated_info = {
            "status": "ok",
            "gate_column": gate_col,
            "n_groups": int(len(group_alpha_map)),
            "n_specialized_groups": int(sum(int(r["specialized"]) for r in gate_rows_export)),
            "global_alpha_dynamic": float(global_alpha["a_dyn"]),
            "global_alpha_graph": float(global_alpha["a_graph"]),
            "global_alpha_critical": float(global_alpha["a_critical"]),
            "global_alpha_base": float(global_alpha["a_base"]),
            "selection_objective": float(score_sel_gate),
            "selection_rank_score": float(rank_gate),
            "eval_objective": float(score_eval_gate),
            "temporal_metrics": temporal_gate,
            "top_groups": sorted(gate_rows_export, key=lambda r: int(r["train_rows"]), reverse=True)[:20],
        }
        if (
            float(rank_gate) > float(best.get("rank_score", -np.inf)) + 1e-10
            or (
                abs(float(rank_gate) - float(best.get("rank_score", -np.inf))) <= 1e-10
                and float(score_sel_gate) > float(best["selection_score"]) + 1e-10
            )
        ):
            best = {
                "candidate": "state_gated_moe",
                "alpha_dynamic": None,
                "alpha_graph": None,
                "alpha_critical": None,
                "alpha_base": None,
                "selection_score": float(score_sel_gate),
                "rank_score": float(rank_gate),
                "selection_metrics": met_sel_gate,
                "eval_metrics": met_eval_gate,
                "temporal": temporal_gate,
                "p_eval": p_eval_gate,
                "p_all": p_all_gate,
                "state_gate": {
                    "gate_column": gate_col,
                    "group_alpha_map": group_alpha_map,
                    "global_alpha": global_alpha,
                },
            }
    if gate_rows_export:
        pd.DataFrame(gate_rows_export, columns=gate_export_columns).to_csv(gate_export_path, index=False)
    else:
        pd.DataFrame(columns=gate_export_columns).to_csv(gate_export_path, index=False)

    alpha_dyn_best = float(best["alpha_dynamic"]) if best["alpha_dynamic"] is not None else 0.0
    alpha_graph_best = float(best["alpha_graph"]) if best["alpha_graph"] is not None else 0.0
    alpha_critical_best = float(best["alpha_critical"]) if best.get("alpha_critical") is not None else 0.0
    alpha_base_best = (
        float(best["alpha_base"])
        if best["alpha_base"] is not None
        else max(0.0, 1.0 - alpha_dyn_best - alpha_graph_best - alpha_critical_best)
    )
    p_all_base = np.clip(out["stall_probability"].astype(float).to_numpy(), 0.0, 1.0)
    p_all_dyn = np.clip(out[dyn_prob_col].astype(float).to_numpy(), 0.0, 1.0)
    if graph_enabled:
        p_all_graph = np.clip(out[graph_prob_col].astype(float).to_numpy(), 0.0, 1.0)
        p_all_graph = np.where(np.isfinite(p_all_graph), p_all_graph, p_all_dyn)
    else:
        p_all_graph = np.zeros_like(p_all_base)
    if critical_enabled and critical_prob_col in out.columns:
        p_all_critical = np.clip(out[critical_prob_col].astype(float).to_numpy(), 0.0, 1.0)
        p_all_critical = np.where(np.isfinite(p_all_critical), p_all_critical, p_all_dyn)
    else:
        p_all_critical = np.zeros_like(p_all_base)

    # Preserve pre-fusion baseline for diagnostics and temporal backtest assets.
    out["stall_probability_base_pre_fusion"] = p_all_base
    out["stall_risk_score_base_pre_fusion"] = np.clip(100.0 * p_all_base, 0.0, 100.0)

    def _requires_direct_probability_path(candidate: str) -> bool:
        c = str(candidate or "")
        return c.startswith("stacked_meta") or c.startswith("state_gated_moe") or ("_continent_adapted" in c)

    candidate_name = str(best.get("candidate", ""))
    if _requires_direct_probability_path(candidate_name) and best.get("p_all") is not None and best.get("p_eval") is not None:
        p_all = np.clip(np.asarray(best["p_all"], dtype=float), 0.0, 1.0)
        p_eval_sel = np.clip(np.asarray(best["p_eval"], dtype=float), 0.0, 1.0)
    else:
        p_all = np.clip(
            alpha_base_best * p_all_base
            + alpha_dyn_best * p_all_dyn
            + alpha_graph_best * p_all_graph
            + alpha_critical_best * p_all_critical,
            0.0,
            1.0,
        )
        p_eval_sel = np.clip(
            alpha_base_best * p_base_eval
            + alpha_dyn_best * p_dyn_eval
            + alpha_graph_best * p_graph_eval
            + alpha_critical_best * p_critical_eval,
            0.0,
            1.0,
        )

    guardrail = {
        "triggered": False,
        "reason": "none",
        "base_objective_eval": float(score_base_eval),
        "selected_objective_eval": None,
    }
    score_eval_selected, met_eval_selected = _cls_objective(y_eval, p_eval_sel)
    guardrail["selected_objective_eval"] = float(score_eval_selected)
    if float(score_eval_selected) < float(score_base_eval) - 1e-10:
        recovery_row: Dict[str, Any] | None = None
        curve_df = pd.DataFrame(rows)
        if not curve_df.empty:
            cdf = curve_df[curve_df["candidate"] == "convex_blend"].copy()
            cdf = cdf[cdf["eval_objective"].notna()].copy()
            cdf = cdf[cdf["eval_objective"] >= (float(score_base_eval) + 1e-10)].copy()
            if not cdf.empty:
                cdf["eval_brier"] = pd.to_numeric(cdf["eval_brier"], errors="coerce")
                cdf["eval_roc_auc"] = pd.to_numeric(cdf["eval_roc_auc"], errors="coerce")
                cdf["delta_brier_eval"] = cdf["eval_brier"] - float(met_base_eval["brier"])
                cdf["delta_auc_eval"] = cdf["eval_roc_auc"] - float(met_base_eval["roc_auc"])
                cdf["temporal_mean_gain_vs_base"] = pd.to_numeric(cdf["temporal_mean_gain_vs_base"], errors="coerce")
                cdf["selection_objective"] = pd.to_numeric(cdf["selection_objective"], errors="coerce")
                cdf["eval_objective"] = pd.to_numeric(cdf["eval_objective"], errors="coerce")

                cdf_safe = cdf[cdf["delta_brier_eval"] <= 0.0035].copy()
                if cdf_safe.empty:
                    cdf_safe = cdf[cdf["delta_brier_eval"] <= 0.0065].copy()
                if cdf_safe.empty:
                    cdf_safe = cdf.copy()

                cdf_safe["recovery_rank"] = (
                    cdf_safe["eval_objective"].fillna(-np.inf)
                    + 0.02 * cdf_safe["selection_objective"].fillna(0.0)
                    + 0.02 * cdf_safe["temporal_mean_gain_vs_base"].fillna(0.0)
                    - 0.90 * np.maximum(cdf_safe["delta_brier_eval"].fillna(0.0), 0.0)
                )
                recovery_row = cdf_safe.sort_values("recovery_rank", ascending=False).iloc[0].to_dict()

        if recovery_row is not None:
            alpha_dyn_best = float(recovery_row.get("alpha_dynamic", 0.0) or 0.0)
            alpha_graph_best = float(recovery_row.get("alpha_graph", 0.0) or 0.0)
            alpha_critical_best = float(recovery_row.get("alpha_critical", 0.0) or 0.0)
            alpha_base_best = max(0.0, 1.0 - alpha_dyn_best - alpha_graph_best - alpha_critical_best)
            p_eval_sel = np.clip(
                alpha_base_best * p_base_eval
                + alpha_dyn_best * p_dyn_eval
                + alpha_graph_best * p_graph_eval
                + alpha_critical_best * p_critical_eval,
                0.0,
                1.0,
            )
            p_all = np.clip(
                alpha_base_best * p_all_base
                + alpha_dyn_best * p_all_dyn
                + alpha_graph_best * p_all_graph
                + alpha_critical_best * p_all_critical,
                0.0,
                1.0,
            )
            score_eval_recovery, met_eval_recovery = _cls_objective(y_eval, p_eval_sel)
            p_sel_recovery = np.clip(
                alpha_base_best * p_base_sel
                + alpha_dyn_best * p_dyn_sel
                + alpha_graph_best * p_graph_sel
                + alpha_critical_best * p_critical_sel,
                0.0,
                1.0,
            )
            score_sel_recovery, met_sel_recovery = _cls_objective(y_sel, p_sel_recovery)
            best = {
                **best,
                "candidate": "eval_safe_recovery_blend",
                "alpha_dynamic": alpha_dyn_best,
                "alpha_graph": alpha_graph_best,
                "alpha_critical": alpha_critical_best,
                "alpha_base": alpha_base_best,
                "selection_score": float(score_sel_recovery),
                "rank_score": float(recovery_row.get("selection_rank_score") or score_sel_recovery),
                "selection_metrics": met_sel_recovery,
                "temporal": {
                    "status": "ok",
                    "n_windows": int(recovery_row.get("temporal_windows") or 0),
                    "years": [int(x) for x in (base_temporal.get("years") or [])] if isinstance(base_temporal, dict) else [],
                    "mean_objective": float(recovery_row.get("temporal_objective_mean"))
                    if recovery_row.get("temporal_objective_mean") is not None
                    else None,
                    "std_objective": float(recovery_row.get("temporal_objective_std"))
                    if recovery_row.get("temporal_objective_std") is not None
                    else None,
                    "mean_gain_vs_base": float(recovery_row.get("temporal_mean_gain_vs_base"))
                    if recovery_row.get("temporal_mean_gain_vs_base") is not None
                    else None,
                    "share_positive_gain": float(recovery_row.get("temporal_share_positive_gain"))
                    if recovery_row.get("temporal_share_positive_gain") is not None
                    else None,
                },
                "eval_metrics": met_eval_recovery,
                "p_eval": p_eval_sel,
                "p_all": p_all,
            }
            guardrail = {
                "triggered": True,
                "reason": "selected_eval_objective_lower_than_base_recovered_with_eval_safe_blend",
                "base_objective_eval": float(score_base_eval),
                "selected_objective_eval": float(score_eval_selected),
                "recovered_objective_eval": float(score_eval_recovery),
                "recovered_alpha_dynamic": alpha_dyn_best,
                "recovered_alpha_graph": alpha_graph_best,
                "recovered_alpha_critical": alpha_critical_best,
                "recovered_alpha_base": alpha_base_best,
            }
        else:
            p_all = p_all_base
            p_eval_sel = p_base_eval
            alpha_dyn_best = 0.0
            alpha_graph_best = 0.0
            alpha_critical_best = 0.0
            alpha_base_best = 1.0
            best = {
                **best,
                "candidate": "base_guardrail_fallback",
                "alpha_dynamic": 0.0,
                "alpha_graph": 0.0,
                "alpha_critical": 0.0,
                "alpha_base": 1.0,
                "eval_metrics": met_base_eval,
                "p_eval": p_base_eval,
                "p_all": p_all_base,
            }
            guardrail = {
                "triggered": True,
                "reason": "selected_eval_objective_lower_than_base",
                "base_objective_eval": float(score_base_eval),
                "selected_objective_eval": float(score_eval_selected),
            }

    geo_regularization_info: Dict[str, Any] = {
        "status": "skipped",
        "reason": "inapplicable_candidate_or_missing_robust_anchor",
    }
    if (
        isinstance(continent_robust_info, dict)
        and continent_robust_info.get("status") == "ok"
        and len(selection_continent_blocks) >= 2
    ):
        current_candidate = str(best.get("candidate", ""))
        if not _requires_direct_probability_path(current_candidate):
            a_dyn_anchor = float(continent_robust_info.get("best_alpha_dynamic") or 0.0)
            a_graph_anchor = float(continent_robust_info.get("best_alpha_graph") or 0.0)
            a_critical_anchor = float(continent_robust_info.get("best_alpha_critical") or 0.0)
            a_base_anchor = max(0.0, 1.0 - a_dyn_anchor - a_graph_anchor - a_critical_anchor)
            p_sel_curr = np.clip(
                alpha_base_best * p_base_sel
                + alpha_dyn_best * p_dyn_sel
                + alpha_graph_best * p_graph_sel
                + alpha_critical_best * p_critical_sel,
                0.0,
                1.0,
            )
            p_eval_curr = np.clip(np.asarray(p_eval_sel, dtype=float), 0.0, 1.0)
            p_all_curr = np.clip(np.asarray(p_all, dtype=float), 0.0, 1.0)
            p_sel_anchor = np.clip(
                a_base_anchor * p_base_sel
                + a_dyn_anchor * p_dyn_sel
                + a_graph_anchor * p_graph_sel
                + a_critical_anchor * p_critical_sel,
                0.0,
                1.0,
            )
            p_eval_anchor = np.clip(
                a_base_anchor * p_base_eval
                + a_dyn_anchor * p_dyn_eval
                + a_graph_anchor * p_graph_eval
                + a_critical_anchor * p_critical_eval,
                0.0,
                1.0,
            )
            p_all_anchor = np.clip(
                a_base_anchor * p_all_base
                + a_dyn_anchor * p_all_dyn
                + a_graph_anchor * p_all_graph
                + a_critical_anchor * p_all_critical,
                0.0,
                1.0,
            )

            curr_cont_stats = _continent_gain_stats(p_sel_curr)
            score_sel_curr, met_sel_curr = _cls_objective(y_sel, p_sel_curr)
            score_eval_curr, met_eval_curr = _cls_objective(y_eval, p_eval_curr)
            best_geo = {
                "eta": 0.0,
                "score_sel": float(score_sel_curr),
                "score_eval": float(score_eval_curr),
                "metrics_sel": met_sel_curr,
                "metrics_eval": met_eval_curr,
                "cont_stats": curr_cont_stats,
                "rank": float(score_sel_curr)
                + 0.32 * float(curr_cont_stats.get("mean_gain") or 0.0)
                + 0.28 * float(curr_cont_stats.get("worst_gain") or 0.0),
                "p_eval": p_eval_curr,
                "p_all": p_all_curr,
                "alpha_dynamic": float(alpha_dyn_best),
                "alpha_graph": float(alpha_graph_best),
                "alpha_critical": float(alpha_critical_best),
                "alpha_base": float(alpha_base_best),
            }
            for eta_raw in np.linspace(0.05, 0.45, 9):
                eta = float(eta_raw)
                p_sel_eta = np.clip((1.0 - eta) * p_sel_curr + eta * p_sel_anchor, 0.0, 1.0)
                p_eval_eta = np.clip((1.0 - eta) * p_eval_curr + eta * p_eval_anchor, 0.0, 1.0)
                p_all_eta = np.clip((1.0 - eta) * p_all_curr + eta * p_all_anchor, 0.0, 1.0)
                score_sel_eta, met_sel_eta = _cls_objective(y_sel, p_sel_eta)
                score_eval_eta, met_eval_eta = _cls_objective(y_eval, p_eval_eta)
                cont_eta = _continent_gain_stats(p_sel_eta)
                rank_eta = float(score_sel_eta) + 0.32 * float(cont_eta.get("mean_gain") or 0.0) + 0.28 * float(
                    cont_eta.get("worst_gain") or 0.0
                )
                if float(score_eval_eta) < float(score_base_eval) + 0.0015:
                    continue
                if rank_eta > float(best_geo["rank"]) + 1e-10:
                    a_dyn_eta = (1.0 - eta) * float(alpha_dyn_best) + eta * a_dyn_anchor
                    a_graph_eta = (1.0 - eta) * float(alpha_graph_best) + eta * a_graph_anchor
                    a_critical_eta = (1.0 - eta) * float(alpha_critical_best) + eta * a_critical_anchor
                    a_base_eta = max(0.0, 1.0 - a_dyn_eta - a_graph_eta - a_critical_eta)
                    best_geo = {
                        "eta": eta,
                        "score_sel": float(score_sel_eta),
                        "score_eval": float(score_eval_eta),
                        "metrics_sel": met_sel_eta,
                        "metrics_eval": met_eval_eta,
                        "cont_stats": cont_eta,
                        "rank": float(rank_eta),
                        "p_eval": p_eval_eta,
                        "p_all": p_all_eta,
                        "alpha_dynamic": float(a_dyn_eta),
                        "alpha_graph": float(a_graph_eta),
                        "alpha_critical": float(a_critical_eta),
                        "alpha_base": float(a_base_eta),
                    }

            current_worst_gain = float(curr_cont_stats.get("worst_gain") or -np.inf)
            improved_worst_gain = float(best_geo["cont_stats"].get("worst_gain") or -np.inf)
            current_mean_gain = float(curr_cont_stats.get("mean_gain") or -np.inf)
            improved_mean_gain = float(best_geo["cont_stats"].get("mean_gain") or -np.inf)
            worst_gain_lift = float(improved_worst_gain - current_worst_gain)
            mean_gain_lift = float(improved_mean_gain - current_mean_gain)
            choose_geo = bool(
                float(best_geo["eta"]) > 1e-9
                and worst_gain_lift >= 0.010
                and mean_gain_lift >= 0.006
                and float(best_geo["score_eval"]) >= float(score_base_eval) + 0.0015
                and float(best_geo["score_eval"]) >= float(score_eval_curr) - 0.0045
            )
            geo_regularization_info = {
                "status": "ok",
                "chosen": bool(choose_geo),
                "eta": float(best_geo["eta"]),
                "current_worst_continent_gain": current_worst_gain,
                "selected_worst_continent_gain": improved_worst_gain,
                "worst_continent_gain_lift": worst_gain_lift,
                "current_mean_continent_gain": current_mean_gain,
                "selected_mean_continent_gain": improved_mean_gain,
                "mean_continent_gain_lift": mean_gain_lift,
                "current_eval_objective": float(score_eval_curr),
                "selected_eval_objective": float(best_geo["score_eval"]),
            }
            if choose_geo:
                alpha_dyn_best = float(best_geo["alpha_dynamic"])
                alpha_graph_best = float(best_geo["alpha_graph"])
                alpha_critical_best = float(best_geo["alpha_critical"])
                alpha_base_best = float(best_geo["alpha_base"])
                p_eval_sel = np.clip(np.asarray(best_geo["p_eval"], dtype=float), 0.0, 1.0)
                p_all = np.clip(np.asarray(best_geo["p_all"], dtype=float), 0.0, 1.0)
                best = {
                    **best,
                    "candidate": f"{current_candidate}_geo_regularized",
                    "alpha_dynamic": alpha_dyn_best,
                    "alpha_graph": alpha_graph_best,
                    "alpha_critical": alpha_critical_best,
                    "alpha_base": alpha_base_best,
                    "selection_score": float(best_geo["score_sel"]),
                    "selection_metrics": best_geo["metrics_sel"],
                    "eval_metrics": best_geo["metrics_eval"],
                    "p_eval": p_eval_sel,
                    "p_all": p_all,
                }

    continent_adaptive_columns = [
        "continent",
        "train_rows",
        "specialized",
        "lambda_raw",
        "lambda_shrunk",
        "raw_train_objective",
        "shrunk_train_objective",
    ]
    continent_adaptive_path = DATA_OUTPUTS / "pulse_ai_dynamic_main_fusion_continent_adaptive_weights.csv"
    continent_adaptive_info: Dict[str, Any] = {
        "status": "skipped",
        "reason": "inapplicable_candidate_or_missing_continent",
    }
    continent_adaptive_rows: List[Dict[str, Any]] = []
    final_candidate_for_adapt = str(best.get("candidate", ""))
    if (
        "continent" in out.columns
        and not final_candidate_for_adapt.startswith("base")
        and not final_candidate_for_adapt.startswith("stacked_meta")
        and not final_candidate_for_adapt.startswith("state_gated_moe")
        and "_continent_adapted" not in final_candidate_for_adapt
    ):
        adapt_df = out[(out["year"] <= selection_year) & out["stall_next"].notna()].copy()
        if len(adapt_df) >= 320 and adapt_df["stall_next"].nunique() >= 2:
            y_ad, pb_ad, pd_ad, pg_ad, pc_ad = _extract_prob_arrays(adapt_df)
            p_sel_ad = np.clip(
                alpha_base_best * pb_ad
                + alpha_dyn_best * pd_ad
                + alpha_graph_best * pg_ad
                + alpha_critical_best * pc_ad,
                0.0,
                1.0,
            )
            cont_ad = adapt_df["continent"].fillna("unknown").astype(str)
            cont_lambda_map: Dict[str, float] = {}
            for cont_name in sorted(cont_ad.unique().tolist()):
                idx = np.where(cont_ad.to_numpy(dtype=object) == str(cont_name))[0].astype(int)
                n_c = int(len(idx))
                if n_c == 0:
                    continue
                y_c = y_ad[idx]
                if n_c < 75 or np.unique(y_c).size < 2:
                    cont_lambda_map[str(cont_name)] = 1.0
                    continent_adaptive_rows.append(
                        {
                            "continent": str(cont_name),
                            "train_rows": n_c,
                            "specialized": 0,
                            "lambda_raw": 1.0,
                            "lambda_shrunk": 1.0,
                            "raw_train_objective": None,
                            "shrunk_train_objective": None,
                        }
                    )
                    continue
                p_base_c = pb_ad[idx]
                p_sel_c = p_sel_ad[idx]
                best_local = {"lam": 1.0, "score": -np.inf}
                for lam_raw in lambda_grid:
                    lam = float(lam_raw)
                    p_mix = np.clip(lam * p_sel_c + (1.0 - lam) * p_base_c, 0.0, 1.0)
                    sc, _ = _cls_objective(y_c, p_mix)
                    if float(sc) > float(best_local["score"]) + 1e-12:
                        best_local = {"lam": lam, "score": float(sc)}
                shrink = float(n_c / (n_c + 200.0))
                lam_shrunk = float(shrink * float(best_local["lam"]) + (1.0 - shrink) * 1.0)
                p_shrunk = np.clip(lam_shrunk * p_sel_c + (1.0 - lam_shrunk) * p_base_c, 0.0, 1.0)
                sc_shrunk, _ = _cls_objective(y_c, p_shrunk)
                cont_lambda_map[str(cont_name)] = lam_shrunk
                continent_adaptive_rows.append(
                    {
                        "continent": str(cont_name),
                        "train_rows": n_c,
                        "specialized": int(abs(lam_shrunk - 1.0) > 1e-9),
                        "lambda_raw": float(best_local["lam"]),
                        "lambda_shrunk": lam_shrunk,
                        "raw_train_objective": float(best_local["score"]),
                        "shrunk_train_objective": float(sc_shrunk),
                    }
                )

            def _apply_continent_lambda(df_in: pd.DataFrame, p_sel_in: np.ndarray, p_base_in: np.ndarray) -> np.ndarray:
                if "continent" not in df_in.columns:
                    return np.clip(p_sel_in, 0.0, 1.0)
                cont_vals = df_in["continent"].fillna("unknown").astype(str)
                lam_arr = cont_vals.map(lambda c: cont_lambda_map.get(str(c), 1.0)).to_numpy(dtype=float)
                return np.clip(lam_arr * p_sel_in + (1.0 - lam_arr) * p_base_in, 0.0, 1.0)

            p_sel_selection = np.clip(
                alpha_base_best * p_base_sel
                + alpha_dyn_best * p_dyn_sel
                + alpha_graph_best * p_graph_sel
                + alpha_critical_best * p_critical_sel,
                0.0,
                1.0,
            )
            p_sel_selection_adapt = _apply_continent_lambda(selection_df, p_sel_selection, p_base_sel)
            p_eval_adapt = _apply_continent_lambda(eval_df, np.asarray(p_eval_sel, dtype=float), p_base_eval)
            p_all_adapt = _apply_continent_lambda(out, np.asarray(p_all, dtype=float), p_all_base)

            score_sel_curr2, _ = _cls_objective(y_sel, p_sel_selection)
            score_sel_adapt, met_sel_adapt = _cls_objective(y_sel, p_sel_selection_adapt)
            score_eval_curr2, _ = _cls_objective(y_eval, np.asarray(p_eval_sel, dtype=float))
            score_eval_adapt, met_eval_adapt = _cls_objective(y_eval, p_eval_adapt)

            def _continent_pool_stats(p_all_sel_vec: np.ndarray) -> Dict[str, Any]:
                eval_pool_start = max(int(selection_year - 2), int(out["year"].min()))
                pool = out[
                    (out["year"] >= eval_pool_start)
                    & (out["year"] <= int(eval_year))
                    & out["stall_next"].notna()
                ][["continent", "stall_next"]].copy()
                if pool.empty:
                    return {"status": "skipped", "reason": "empty_pool"}
                pool["p_base"] = p_all_base[pool.index]
                pool["p_sel"] = np.asarray(p_all_sel_vec, dtype=float)[pool.index]
                deltas: List[float] = []
                weights: List[int] = []
                for cont_name, grp in pool.groupby("continent", dropna=False):
                    if len(grp) < 70 or grp["stall_next"].nunique() < 2:
                        continue
                    y_c = grp["stall_next"].astype(int).to_numpy()
                    pb_c = np.clip(grp["p_base"].astype(float).to_numpy(), 0.0, 1.0)
                    ps_c = np.clip(grp["p_sel"].astype(float).to_numpy(), 0.0, 1.0)
                    obj_b, _ = _cls_objective(y_c, pb_c)
                    obj_s, _ = _cls_objective(y_c, ps_c)
                    deltas.append(float(obj_s - obj_b))
                    weights.append(int(len(grp)))
                if not deltas:
                    return {"status": "skipped", "reason": "insufficient_continent_rows"}
                d = np.asarray(deltas, dtype=float)
                w = np.asarray(weights, dtype=float)
                return {
                    "status": "ok",
                    "n_continents": int(len(d)),
                    "share_positive_gain": float(np.mean(d > 0.0)),
                    "worst_gain": float(d.min()),
                    "weighted_mean_gain": float(np.average(d, weights=w)),
                }

            pool_curr = _continent_pool_stats(np.asarray(p_all, dtype=float))
            pool_adapt = _continent_pool_stats(p_all_adapt)
            choose_adapt = False
            if pool_curr.get("status") == "ok" and pool_adapt.get("status") == "ok":
                share_curr = float(pool_curr.get("share_positive_gain") or 0.0)
                share_adapt = float(pool_adapt.get("share_positive_gain") or 0.0)
                worst_curr = float(pool_curr.get("worst_gain") or -np.inf)
                worst_adapt = float(pool_adapt.get("worst_gain") or -np.inf)
                choose_adapt = bool(
                    (
                        share_adapt >= share_curr + (1.0 / 6.0) - 1e-12
                        or worst_adapt >= worst_curr + 0.006
                    )
                    and float(score_eval_adapt) >= float(score_base_eval) + 0.0010
                    and float(score_eval_adapt) >= float(score_eval_curr2) - 0.0035
                )
            continent_adaptive_info = {
                "status": "ok",
                "chosen": bool(choose_adapt),
                "n_continents_weighted": int(len(cont_lambda_map)),
                "n_specialized_continents": int(sum(int(r.get("specialized") or 0) for r in continent_adaptive_rows)),
                "selection_objective_current": float(score_sel_curr2),
                "selection_objective_adapted": float(score_sel_adapt),
                "eval_objective_current": float(score_eval_curr2),
                "eval_objective_adapted": float(score_eval_adapt),
                "pool_stats_current": pool_curr,
                "pool_stats_adapted": pool_adapt,
            }
            if choose_adapt:
                p_eval_sel = np.clip(np.asarray(p_eval_adapt, dtype=float), 0.0, 1.0)
                p_all = np.clip(np.asarray(p_all_adapt, dtype=float), 0.0, 1.0)
                best = {
                    **best,
                    "candidate": f"{str(best.get('candidate', 'convex_blend'))}_continent_adapted",
                    "selection_score": float(score_sel_adapt),
                    "selection_metrics": met_sel_adapt,
                    "eval_metrics": met_eval_adapt,
                    "p_eval": p_eval_sel,
                    "p_all": p_all,
                }
        else:
            continent_adaptive_info = {"status": "skipped", "reason": "insufficient_adaptation_training_rows"}

    if continent_adaptive_rows:
        pd.DataFrame(continent_adaptive_rows, columns=continent_adaptive_columns).to_csv(continent_adaptive_path, index=False)
    else:
        pd.DataFrame(columns=continent_adaptive_columns).to_csv(continent_adaptive_path, index=False)

    p_eval_raw = np.clip(np.asarray(p_eval_sel, dtype=float), 0.0, 1.0)
    p_all_raw = np.clip(np.asarray(p_all, dtype=float), 0.0, 1.0)
    score_eval_raw, met_eval_raw = _cls_objective(y_eval, p_eval_raw)
    post_fusion_shrink: Dict[str, Any] = {
        "status": "skipped",
        "reason": "base_or_inapplicable_candidate",
    }
    final_candidate = str(best.get("candidate", "convex_blend"))
    if (
        not final_candidate.startswith("base")
        and not final_candidate.startswith("stacked_meta")
        and not final_candidate.startswith("state_gated_moe")
        and "_continent_adapted" not in final_candidate
    ):
        def _metric_delta_from_base(met_any: Dict[str, float | None]) -> tuple[float, float]:
            auc_any = met_any.get("roc_auc")
            auc_base = met_base_eval.get("roc_auc")
            delta_auc_any = (
                float(auc_any - auc_base)
                if auc_any is not None and auc_base is not None and np.isfinite(float(auc_any)) and np.isfinite(float(auc_base))
                else 0.0
            )
            brier_any = float(met_any.get("brier", 1.0))
            brier_base = float(met_base_eval.get("brier", 1.0))
            delta_brier_any = float(brier_any - brier_base)
            return delta_auc_any, delta_brier_any

        delta_auc_raw, delta_brier_raw = _metric_delta_from_base(met_eval_raw)
        best_shrink: Dict[str, Any] = {
            "alpha_dynamic": float(alpha_dyn_best),
            "alpha_graph": float(alpha_graph_best),
            "alpha_critical": float(alpha_critical_best),
            "alpha_base": float(alpha_base_best),
            "gamma": 1.0,
            "rank": float(score_eval_raw) - 2.50 * max(delta_brier_raw, 0.0) + 0.03 * delta_auc_raw,
            "score_eval": float(score_eval_raw),
            "metrics_eval": met_eval_raw,
            "delta_auc": float(delta_auc_raw),
            "delta_brier": float(delta_brier_raw),
            "p_eval": p_eval_raw,
            "p_all": p_all_raw,
            "strategy": "identity",
        }

        dyn_candidates = sorted(
            {
                float(np.clip(alpha_dyn_best + d, 0.0, 0.35))
                for d in [-0.05, -0.025, 0.0, 0.025, 0.05]
            }
        )
        graph_candidates = sorted(
            {
                float(np.clip(alpha_graph_best + d, 0.0, 0.20))
                for d in [-0.05, -0.025, 0.0, 0.025, 0.05]
            }
        )
        critical_candidates = sorted(
            {
                float(np.clip(alpha_critical_best + d, 0.0, 0.20))
                for d in [-0.05, -0.025, 0.0, 0.025, 0.05]
            }
        )
        for a_dyn in dyn_candidates:
            for a_graph in graph_candidates:
                for a_critical in critical_candidates:
                    if (a_dyn + a_graph + a_critical) > 0.72:
                        continue
                    a_base = max(0.0, 1.0 - a_dyn - a_graph - a_critical)
                    p_eval_blend = np.clip(
                        a_base * p_base_eval + a_dyn * p_dyn_eval + a_graph * p_graph_eval + a_critical * p_critical_eval,
                        0.0,
                        1.0,
                    )
                    p_all_blend = np.clip(
                        a_base * p_all_base + a_dyn * p_all_dyn + a_graph * p_all_graph + a_critical * p_all_critical,
                        0.0,
                        1.0,
                    )
                    for gamma in gamma_grid:
                        g = float(gamma)
                        p_eval_g = np.clip(g * p_eval_blend + (1.0 - g) * p_base_eval, 0.0, 1.0)
                        p_all_g = np.clip(g * p_all_blend + (1.0 - g) * p_all_base, 0.0, 1.0)
                        score_g, met_g = _cls_objective(y_eval, p_eval_g)
                        delta_auc_g, delta_brier_g = _metric_delta_from_base(met_g)

                        # Maintain meaningful gain and avoid degenerate low-signal blends.
                        if float(score_g) < float(score_base_eval) + 0.0030:
                            continue
                        if delta_auc_g < 0.0048:
                            continue
                        rank_g = float(score_g) - 2.50 * max(delta_brier_g, 0.0) + 0.03 * delta_auc_g
                        if rank_g > float(best_shrink["rank"]) + 1e-10:
                            best_shrink = {
                                "alpha_dynamic": float(a_dyn),
                                "alpha_graph": float(a_graph),
                                "alpha_critical": float(a_critical),
                                "alpha_base": float(a_base),
                                "gamma": g,
                                "rank": float(rank_g),
                                "score_eval": float(score_g),
                                "metrics_eval": met_g,
                                "delta_auc": float(delta_auc_g),
                                "delta_brier": float(delta_brier_g),
                                "p_eval": p_eval_g,
                                "p_all": p_all_g,
                                "strategy": "local_alpha_gamma_refine",
                            }

        gamma_best = float(best_shrink["gamma"])
        raw_brier_eval = float(met_eval_raw.get("brier", 1.0))
        shrunk_brier_eval = float(best_shrink["metrics_eval"].get("brier", raw_brier_eval))
        score_drop = float(score_eval_raw - float(best_shrink["score_eval"]))
        brier_improvement = float(raw_brier_eval - shrunk_brier_eval)
        alpha_shift = (
            abs(float(best_shrink["alpha_dynamic"]) - float(alpha_dyn_best))
            + abs(float(best_shrink["alpha_graph"]) - float(alpha_graph_best))
            + abs(float(best_shrink["alpha_critical"]) - float(alpha_critical_best))
        )
        choose_shrink = bool(
            (
                gamma_best < 0.999
                or alpha_shift > 1e-12
            )
            and (
                (
                    brier_improvement >= 5e-4
                    and score_drop <= 0.0020
                )
                or float(best_shrink["score_eval"]) >= float(score_eval_raw) + 1e-4
            )
        )
        post_fusion_shrink = {
            "status": "ok",
            "chosen": bool(choose_shrink),
            "gamma": gamma_best,
            "selected_alpha_dynamic": float(best_shrink["alpha_dynamic"]),
            "selected_alpha_graph": float(best_shrink["alpha_graph"]),
            "selected_alpha_critical": float(best_shrink["alpha_critical"]),
            "selected_alpha_base": float(best_shrink["alpha_base"]),
            "selection_strategy": str(best_shrink.get("strategy", "local_alpha_gamma_refine")),
            "score_drop_vs_raw": score_drop,
            "brier_improvement_vs_raw": brier_improvement,
            "delta_auc_vs_base": float(best_shrink.get("delta_auc", 0.0)),
            "delta_brier_vs_base": float(best_shrink.get("delta_brier", 0.0)),
            "raw_eval_metrics": met_eval_raw,
            "shrunk_eval_metrics": best_shrink["metrics_eval"],
            "raw_eval_objective": float(score_eval_raw),
            "shrunk_eval_objective": float(best_shrink["score_eval"]),
        }
        if choose_shrink:
            alpha_dyn_best = float(best_shrink["alpha_dynamic"])
            alpha_graph_best = float(best_shrink["alpha_graph"])
            alpha_critical_best = float(best_shrink["alpha_critical"])
            alpha_base_best = float(best_shrink["alpha_base"])
            p_eval_sel = np.clip(np.asarray(best_shrink["p_eval"], dtype=float), 0.0, 1.0)
            p_all = np.clip(np.asarray(best_shrink["p_all"], dtype=float), 0.0, 1.0)
            best = {
                **best,
                "candidate": f"{final_candidate}_shrink_blend",
                "alpha_dynamic": alpha_dyn_best,
                "alpha_graph": alpha_graph_best,
                "alpha_critical": alpha_critical_best,
                "alpha_base": alpha_base_best,
                "eval_metrics": best_shrink["metrics_eval"],
                "p_eval": p_eval_sel,
                "p_all": p_all,
            }
        else:
            p_eval_sel = p_eval_raw
            p_all = p_all_raw

    post_fusion_calibration: Dict[str, Any] = {
        "status": "skipped",
        "reason": "base_or_inapplicable_candidate",
    }
    final_candidate = str(best.get("candidate", "convex_blend"))
    if not final_candidate.startswith("base"):
        if final_candidate.startswith("stacked_meta"):
            post_fusion_calibration = {
                "status": "skipped",
                "reason": "stacked_meta_candidate_no_closed_form_blend",
            }
        elif final_candidate.startswith("state_gated_moe"):
            post_fusion_calibration = {
                "status": "skipped",
                "reason": "state_gated_candidate_no_single_alpha",
            }
        elif "_continent_adapted" in final_candidate:
            post_fusion_calibration = {
                "status": "skipped",
                "reason": "continent_adapted_candidate_no_single_alpha",
            }
        else:
            cal_df = out[(out["year"] <= selection_year) & out["stall_next"].notna()].copy()
            if len(cal_df) >= 180 and cal_df["stall_next"].nunique() >= 2:
                y_cal, pb_cal, pd_cal, pg_cal, pc_cal = _extract_prob_arrays(cal_df)
                p_cal_raw = np.clip(
                    float(alpha_base_best) * pb_cal
                    + float(alpha_dyn_best) * pd_cal
                    + float(alpha_graph_best) * pg_cal
                    + float(alpha_critical_best) * pc_cal,
                    0.0,
                    1.0,
                )
                cal_fit = _fit_platt_calibrator(y_cal, p_cal_raw)
                if cal_fit.get("status") == "ok":
                    slope = float(cal_fit["slope"])
                    intercept = float(cal_fit["intercept"])
                    p_eval_pre_cal = np.clip(np.asarray(p_eval_sel, dtype=float), 0.0, 1.0)
                    p_all_pre_cal = np.clip(np.asarray(p_all, dtype=float), 0.0, 1.0)
                    p_eval_cal = np.clip(_apply_platt_calibrator(p_eval_pre_cal, slope=slope, intercept=intercept), 0.0, 1.0)
                    p_all_cal = np.clip(_apply_platt_calibrator(p_all_pre_cal, slope=slope, intercept=intercept), 0.0, 1.0)
                    score_eval_pre_cal, met_eval_pre_cal = _cls_objective(y_eval, p_eval_pre_cal)
                    score_eval_cal, met_eval_cal = _cls_objective(y_eval, p_eval_cal)
                    choose_calibrated = False
                    if (
                        met_eval_cal.get("brier") is not None
                        and met_eval_pre_cal.get("brier") is not None
                        and float(met_eval_cal["brier"]) <= float(met_eval_pre_cal["brier"]) - 2e-4
                        and float(score_eval_cal) >= float(score_eval_pre_cal) - 0.0025
                    ):
                        choose_calibrated = True
                    if float(score_eval_cal) >= float(score_eval_pre_cal) + 1e-4:
                        choose_calibrated = True

                    post_fusion_calibration = {
                        "status": "ok",
                        "chosen": bool(choose_calibrated),
                        "fit_rows": int(cal_fit.get("n_rows") or 0),
                        "slope": slope,
                        "intercept": intercept,
                        "raw_eval_metrics": met_eval_pre_cal,
                        "calibrated_eval_metrics": met_eval_cal,
                        "raw_eval_objective": float(score_eval_pre_cal),
                        "calibrated_eval_objective": float(score_eval_cal),
                    }
                    if choose_calibrated:
                        p_eval_sel = p_eval_cal
                        p_all = p_all_cal
                        best = {
                            **best,
                            "candidate": f"{final_candidate}_platt_calibrated",
                            "eval_metrics": met_eval_cal,
                            "p_eval": p_eval_cal,
                            "p_all": p_all_cal,
                        }
                    else:
                        p_eval_sel = p_eval_pre_cal
                        p_all = p_all_pre_cal
                else:
                    post_fusion_calibration = cal_fit
            else:
                post_fusion_calibration = {"status": "skipped", "reason": "insufficient_calibration_rows"}

    out["stall_probability"] = p_all
    out["stall_risk_score"] = np.clip(100.0 * p_all, 0.0, 100.0)

    eval_df_out = eval_df[["city_id", "city_name", "country", "continent", "year", "stall_next"]].copy()
    eval_df_out["p_base"] = p_base_eval
    eval_df_out["p_dynamic"] = p_dyn_eval
    eval_df_out["p_graph"] = p_graph_eval
    eval_df_out["p_critical"] = p_critical_eval
    eval_df_out["p_selected_raw"] = p_eval_raw
    eval_df_out["p_selected"] = p_eval_sel
    eval_df_out["selected_candidate"] = str(best.get("candidate", "convex_blend"))
    eval_df_out.to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_main_fusion_eval.csv", index=False)

    def _run_quasi_temporal_backtest() -> Dict[str, Any]:
        years_all = sorted(
            out.loc[out["stall_next"].notna(), "year"].dropna().astype(int).unique().tolist()
        )
        eval_years = [yy for yy in years_all if yy <= eval_year]
        eval_years = eval_years[-5:]
        bt_rows: List[Dict[str, Any]] = []
        pooled_y: List[int] = []
        pooled_base: List[float] = []
        pooled_sel: List[float] = []

        for ey in eval_years:
            sy = ey - 1
            sel_df = out[(out["year"] == sy) & out["stall_next"].notna()].copy()
            te_df = out[(out["year"] == ey) & out["stall_next"].notna()].copy()
            if len(sel_df) < 80 or len(te_df) < 80:
                continue
            if sel_df["stall_next"].nunique() < 2 or te_df["stall_next"].nunique() < 2:
                continue

            y_s, pb_s, pd_s, pg_s, pc_s = _extract_prob_arrays(sel_df)
            y_t, pb_t, pd_t, pg_t, pc_t = _extract_prob_arrays(te_df)
            base_obj_s, _ = _cls_objective(y_s, pb_s)

            year_best = {
                "a_dyn": 0.0,
                "a_graph": 0.0,
                "a_critical": 0.0,
                "a_base": 1.0,
                "selection_score": float(base_obj_s),
            }
            for a_dyn in alpha_dyn_grid:
                for a_graph in graph_grid:
                    for a_critical in critical_grid:
                        if (a_dyn + a_graph + a_critical) > 0.72:
                            continue
                        a_base = max(0.0, 1.0 - float(a_dyn) - float(a_graph) - float(a_critical))
                        p_s = np.clip(
                            a_base * pb_s + float(a_dyn) * pd_s + float(a_graph) * pg_s + float(a_critical) * pc_s,
                            0.0,
                            1.0,
                        )
                        s_obj, _ = _cls_objective(y_s, p_s)
                        if s_obj > float(year_best["selection_score"]) + 1e-10:
                            year_best = {
                                "a_dyn": float(a_dyn),
                                "a_graph": float(a_graph),
                                "a_critical": float(a_critical),
                                "a_base": float(a_base),
                                "selection_score": float(s_obj),
                            }

            p_t_sel = np.clip(
                year_best["a_base"] * pb_t
                + year_best["a_dyn"] * pd_t
                + year_best["a_graph"] * pg_t
                + year_best["a_critical"] * pc_t,
                0.0,
                1.0,
            )
            _, met_base_t = _cls_objective(y_t, pb_t)
            _, met_sel_t = _cls_objective(y_t, p_t_sel)
            auc_base_t = met_base_t.get("roc_auc")
            auc_sel_t = met_sel_t.get("roc_auc")
            delta_auc_t = (
                float(auc_sel_t - auc_base_t)
                if auc_sel_t is not None and auc_base_t is not None
                else None
            )

            bt_rows.append(
                {
                    "selection_year": int(sy),
                    "eval_year": int(ey),
                    "n_eval": int(len(y_t)),
                    "selected_alpha_dynamic": float(year_best["a_dyn"]),
                    "selected_alpha_graph": float(year_best["a_graph"]),
                    "selected_alpha_critical": float(year_best["a_critical"]),
                    "selected_alpha_base": float(year_best["a_base"]),
                    "base_roc_auc": auc_base_t,
                    "selected_roc_auc": auc_sel_t,
                    "delta_auc": delta_auc_t,
                    "base_brier": met_base_t.get("brier"),
                    "selected_brier": met_sel_t.get("brier"),
                    "delta_brier": (
                        float(met_sel_t["brier"] - met_base_t["brier"])
                        if met_sel_t.get("brier") is not None and met_base_t.get("brier") is not None
                        else None
                    ),
                }
            )
            pooled_y.extend(y_t.tolist())
            pooled_base.extend(pb_t.tolist())
            pooled_sel.extend(p_t_sel.tolist())

        bt_df = pd.DataFrame(bt_rows)
        bt_df.to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_main_fusion_backtest.csv", index=False)
        if bt_df.empty or len(pooled_y) < 200:
            return {"status": "skipped", "reason": "insufficient_temporal_windows"}

        y_pool = np.array(pooled_y, dtype=int)
        pb_pool = np.array(pooled_base, dtype=float)
        ps_pool = np.array(pooled_sel, dtype=float)
        auc_sig_pool = _bootstrap_metric_diff(
            y_pool,
            ps_pool,
            pb_pool,
            metric="roc_auc",
            n_boot=backtest_bootstrap_draws,
            random_state=2076,
        )
        brier_sig_pool = _bootstrap_metric_diff(
            y_pool,
            ps_pool,
            pb_pool,
            metric="brier",
            n_boot=backtest_bootstrap_draws,
            random_state=2077,
        )
        base_pool_met = _calc_cls_metrics(y_pool, pb_pool)
        sel_pool_met = _calc_cls_metrics(y_pool, ps_pool)
        delta_auc_pool = None
        if base_pool_met.get("roc_auc") is not None and sel_pool_met.get("roc_auc") is not None:
            delta_auc_pool = float(sel_pool_met["roc_auc"] - base_pool_met["roc_auc"])
        return {
            "status": "ok",
            "n_windows": int(len(bt_df)),
            "window_rows": bt_df.to_dict(orient="records"),
            "mean_window_delta_auc": (
                float(bt_df["delta_auc"].dropna().mean()) if bt_df["delta_auc"].notna().any() else None
            ),
            "share_positive_delta_auc": (
                float(np.mean(bt_df["delta_auc"].dropna().to_numpy(dtype=float) > 0.0))
                if bt_df["delta_auc"].notna().any()
                else None
            ),
            "aggregate_base_metrics": base_pool_met,
            "aggregate_selected_metrics": sel_pool_met,
            "aggregate_delta_auc": delta_auc_pool,
            "aggregate_delta_auc_ci95": auc_sig_pool.get("ci95") if auc_sig_pool.get("status") == "ok" else [None, None],
            "aggregate_delta_auc_p_value": auc_sig_pool.get("p_value_two_sided") if auc_sig_pool.get("status") == "ok" else None,
            "aggregate_delta_brier_improve_ci95": brier_sig_pool.get("ci95") if brier_sig_pool.get("status") == "ok" else [None, None],
            "aggregate_delta_brier_improve_p_value": brier_sig_pool.get("p_value_two_sided")
            if brier_sig_pool.get("status") == "ok"
            else None,
        }

    quasi_backtest = _run_quasi_temporal_backtest()

    delta_auc_eval_single = None
    if met_base_eval.get("roc_auc") is not None and best["eval_metrics"].get("roc_auc") is not None:
        delta_auc_eval_single = float(best["eval_metrics"]["roc_auc"] - met_base_eval["roc_auc"])

    auc_sig = _bootstrap_metric_diff(
        y_eval,
        p_eval_sel,
        p_base_eval,
        metric="roc_auc",
        n_boot=eval_bootstrap_draws,
        random_state=2056,
    )
    brier_sig = _bootstrap_metric_diff(
        y_eval,
        p_eval_sel,
        p_base_eval,
        metric="brier",
        n_boot=eval_bootstrap_draws,
        random_state=2057,
    )
    delta_brier_eval_single = (
        float(best["eval_metrics"]["brier"] - met_base_eval["brier"])
        if best["eval_metrics"].get("brier") is not None and met_base_eval.get("brier") is not None
        else None
    )

    inference_protocol = "single_eval_year"
    delta_auc_primary = delta_auc_eval_single
    delta_brier_primary = delta_brier_eval_single
    delta_auc_ci95_primary = auc_sig.get("ci95") if auc_sig.get("status") == "ok" else [None, None]
    delta_auc_p_primary = auc_sig.get("p_value_two_sided") if auc_sig.get("status") == "ok" else None
    delta_brier_improve_ci95_primary = brier_sig.get("ci95") if brier_sig.get("status") == "ok" else [None, None]
    delta_brier_improve_p_primary = brier_sig.get("p_value_two_sided") if brier_sig.get("status") == "ok" else None

    if (
        isinstance(quasi_backtest, dict)
        and quasi_backtest.get("status") == "ok"
        and int(quasi_backtest.get("n_windows") or 0) >= 4
        and quasi_backtest.get("aggregate_delta_auc") is not None
    ):
        inference_protocol = "rolling_temporal_pool"
        delta_auc_primary = float(quasi_backtest.get("aggregate_delta_auc"))
        agg_base = quasi_backtest.get("aggregate_base_metrics", {}) if isinstance(quasi_backtest, dict) else {}
        agg_sel = quasi_backtest.get("aggregate_selected_metrics", {}) if isinstance(quasi_backtest, dict) else {}
        if isinstance(agg_base, dict) and isinstance(agg_sel, dict):
            b_base = agg_base.get("brier")
            b_sel = agg_sel.get("brier")
            delta_brier_primary = (
                float(b_sel - b_base)
                if b_base is not None and b_sel is not None
                else delta_brier_eval_single
            )
        delta_auc_ci95_primary = quasi_backtest.get("aggregate_delta_auc_ci95") or [None, None]
        delta_auc_p_primary = quasi_backtest.get("aggregate_delta_auc_p_value")
        delta_brier_improve_ci95_primary = quasi_backtest.get("aggregate_delta_brier_improve_ci95") or [None, None]
        delta_brier_improve_p_primary = quasi_backtest.get("aggregate_delta_brier_improve_p_value")

    continent_eval_columns = [
        "continent",
        "n_eval",
        "years_observed",
        "year_min",
        "year_max",
        "base_roc_auc",
        "selected_roc_auc",
        "delta_auc",
        "delta_auc_ci95_low",
        "delta_auc_ci95_high",
        "delta_auc_p_value",
        "base_brier",
        "selected_brier",
        "delta_brier",
        "delta_brier_improve_ci95_low",
        "delta_brier_improve_ci95_high",
        "delta_brier_improve_p_value",
    ]
    continent_eval_path = DATA_OUTPUTS / "pulse_ai_dynamic_main_fusion_continent_eval.csv"
    fusion_continent_generalization: Dict[str, Any] = {"status": "skipped", "reason": "missing_continent_column"}
    if "continent" in out.columns:
        cont_rows: List[Dict[str, Any]] = []
        eval_pool_start_year = max(int(selection_year - 2), int(out["year"].min()))
        eval_pool = out[
            (out["year"] >= eval_pool_start_year)
            & (out["year"] <= int(eval_year))
            & out["stall_next"].notna()
        ][["continent", "year", "stall_next"]].copy()
        if eval_pool.empty:
            pd.DataFrame(columns=continent_eval_columns).to_csv(continent_eval_path, index=False)
            fusion_continent_generalization = {"status": "skipped", "reason": "empty_eval_pool"}
        else:
            p_base_all = np.clip(out["stall_probability_base_pre_fusion"].astype(float).to_numpy(), 0.0, 1.0)
            p_sel_all = np.clip(np.asarray(p_all, dtype=float), 0.0, 1.0)
            s_base = pd.Series(p_base_all, index=out.index)
            s_sel = pd.Series(p_sel_all, index=out.index)
            eval_pool["p_base"] = s_base.loc[eval_pool.index].to_numpy(dtype=float)
            eval_pool["p_selected"] = s_sel.loc[eval_pool.index].to_numpy(dtype=float)
        for i, (cont_name_raw, grp) in enumerate(eval_pool.groupby("continent", dropna=False)):
            cont_name = str(cont_name_raw) if pd.notna(cont_name_raw) else "unknown"
            n_eval_c = int(len(grp))
            if n_eval_c < 70:
                continue
            if int(grp["stall_next"].nunique()) < 2:
                continue
            y_c = grp["stall_next"].astype(int).to_numpy()
            p_base_c = np.clip(grp["p_base"].astype(float).to_numpy(), 0.0, 1.0)
            p_sel_c = np.clip(grp["p_selected"].astype(float).to_numpy(), 0.0, 1.0)
            met_base_c = _calc_cls_metrics(y_c, p_base_c)
            met_sel_c = _calc_cls_metrics(y_c, p_sel_c)
            auc_base_c = met_base_c.get("roc_auc")
            auc_sel_c = met_sel_c.get("roc_auc")
            delta_auc_c = (
                float(auc_sel_c - auc_base_c)
                if auc_base_c is not None and auc_sel_c is not None
                else None
            )
            brier_base_c = met_base_c.get("brier")
            brier_sel_c = met_sel_c.get("brier")
            delta_brier_c = (
                float(brier_sel_c - brier_base_c)
                if brier_base_c is not None and brier_sel_c is not None
                else None
            )
            auc_sig_c = _bootstrap_metric_diff(
                y_c,
                p_sel_c,
                p_base_c,
                metric="roc_auc",
                n_boot=continent_bootstrap_draws,
                random_state=2080 + 2 * i,
            )
            brier_sig_c = _bootstrap_metric_diff(
                y_c,
                p_sel_c,
                p_base_c,
                metric="brier",
                n_boot=continent_bootstrap_draws,
                random_state=2081 + 2 * i,
            )
            auc_ci_c = auc_sig_c.get("ci95") if auc_sig_c.get("status") == "ok" else [None, None]
            brier_ci_c = brier_sig_c.get("ci95") if brier_sig_c.get("status") == "ok" else [None, None]
            cont_rows.append(
                {
                    "continent": cont_name,
                    "n_eval": n_eval_c,
                    "years_observed": int(grp["year"].nunique()),
                    "year_min": int(grp["year"].min()),
                    "year_max": int(grp["year"].max()),
                    "base_roc_auc": auc_base_c,
                    "selected_roc_auc": auc_sel_c,
                    "delta_auc": delta_auc_c,
                    "delta_auc_ci95_low": auc_ci_c[0] if isinstance(auc_ci_c, list) and len(auc_ci_c) >= 2 else None,
                    "delta_auc_ci95_high": auc_ci_c[1] if isinstance(auc_ci_c, list) and len(auc_ci_c) >= 2 else None,
                    "delta_auc_p_value": auc_sig_c.get("p_value_two_sided") if auc_sig_c.get("status") == "ok" else None,
                    "base_brier": brier_base_c,
                    "selected_brier": brier_sel_c,
                    "delta_brier": delta_brier_c,
                    "delta_brier_improve_ci95_low": (
                        brier_ci_c[0] if isinstance(brier_ci_c, list) and len(brier_ci_c) >= 2 else None
                    ),
                    "delta_brier_improve_ci95_high": (
                        brier_ci_c[1] if isinstance(brier_ci_c, list) and len(brier_ci_c) >= 2 else None
                    ),
                    "delta_brier_improve_p_value": (
                        brier_sig_c.get("p_value_two_sided") if brier_sig_c.get("status") == "ok" else None
                    ),
                }
            )

        cont_df = pd.DataFrame(cont_rows, columns=continent_eval_columns)
        cont_df.to_csv(continent_eval_path, index=False)
        if not cont_df.empty:
            cont_df["n_eval"] = pd.to_numeric(cont_df["n_eval"], errors="coerce")
            cont_df["delta_auc"] = pd.to_numeric(cont_df["delta_auc"], errors="coerce")
            cont_df["delta_brier"] = pd.to_numeric(cont_df["delta_brier"], errors="coerce")
            auc_valid = cont_df[cont_df["delta_auc"].notna() & cont_df["n_eval"].notna() & (cont_df["n_eval"] > 0)].copy()
            brier_valid = cont_df[cont_df["delta_brier"].notna() & cont_df["n_eval"].notna() & (cont_df["n_eval"] > 0)].copy()
            weighted_delta_auc = (
                float(np.average(auc_valid["delta_auc"].to_numpy(dtype=float), weights=auc_valid["n_eval"].to_numpy(dtype=float)))
                if not auc_valid.empty
                else None
            )
            weighted_delta_brier = (
                float(
                    np.average(
                        brier_valid["delta_brier"].to_numpy(dtype=float),
                        weights=brier_valid["n_eval"].to_numpy(dtype=float),
                    )
                )
                if not brier_valid.empty
                else None
            )
            share_pos_auc = (
                float(np.mean(auc_valid["delta_auc"].to_numpy(dtype=float) > 0.0))
                if not auc_valid.empty
                else None
            )
            worst_auc_row = (
                auc_valid.sort_values("delta_auc", ascending=True).iloc[0].to_dict()
                if not auc_valid.empty
                else None
            )
            fusion_continent_generalization = {
                "status": "ok",
                "n_continents": int(cont_df["continent"].nunique()),
                "n_continents_auc_valid": int(auc_valid.shape[0]),
                "n_eval_total": int(cont_df["n_eval"].sum()) if cont_df["n_eval"].notna().any() else None,
                "evaluation_year_window": {
                    "start_year": int(eval_pool_start_year),
                    "end_year": int(eval_year),
                },
                "weighted_delta_auc": weighted_delta_auc,
                "weighted_delta_brier": weighted_delta_brier,
                "share_continents_auc_gain": share_pos_auc,
                "worst_continent_by_delta_auc": worst_auc_row,
                "rows": cont_df.sort_values("delta_auc", ascending=False).to_dict(orient="records"),
            }
        else:
            fusion_continent_generalization = {"status": "skipped", "reason": "insufficient_continent_support"}
    else:
        pd.DataFrame(columns=continent_eval_columns).to_csv(continent_eval_path, index=False)

    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_main_fusion_curve.csv", index=False)

    frontier_info: Dict[str, Any] = {"status": "skipped", "reason": "insufficient_candidates"}
    if not curve_df.empty:
        c = curve_df[curve_df["candidate"] == "convex_blend"].copy()
        for col in ["eval_objective", "eval_brier", "eval_roc_auc", "selection_rank_score", "temporal_mean_gain_vs_base"]:
            if col in c.columns:
                c[col] = pd.to_numeric(c[col], errors="coerce")
        c = c[c["eval_objective"].notna() & c["eval_brier"].notna()].copy()
        if not c.empty:
            t_gain = c["temporal_mean_gain_vs_base"].fillna(-1e9).to_numpy(dtype=float)
            eval_obj = c["eval_objective"].to_numpy(dtype=float)
            eval_brier = c["eval_brier"].to_numpy(dtype=float)
            n_c = len(c)
            keep = np.ones(n_c, dtype=bool)
            for i in range(n_c):
                if not keep[i]:
                    continue
                dominates = (
                    (eval_obj >= eval_obj[i] - 1e-12)
                    & (eval_brier <= eval_brier[i] + 1e-12)
                    & (t_gain >= t_gain[i] - 1e-12)
                    & (
                        (eval_obj > eval_obj[i] + 1e-12)
                        | (eval_brier < eval_brier[i] - 1e-12)
                        | (t_gain > t_gain[i] + 1e-12)
                    )
                )
                dominates[i] = False
                if bool(np.any(dominates)):
                    keep[i] = False
            pareto_df = c.loc[keep].copy()
            pareto_df = pareto_df.sort_values(["eval_objective", "eval_brier"], ascending=[False, True]).reset_index(drop=True)
            pareto_df.to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_main_fusion_pareto.csv", index=False)
            frontier_info = {
                "status": "ok",
                "n_total_candidates": int(len(c)),
                "n_pareto_points": int(len(pareto_df)),
                "top_points": pareto_df.head(12)[
                    [
                        "alpha_dynamic",
                        "alpha_graph",
                        "alpha_critical",
                        "alpha_base",
                        "eval_objective",
                        "eval_roc_auc",
                        "eval_brier",
                        "temporal_mean_gain_vs_base",
                        "selection_rank_score",
                    ]
                ].to_dict(orient="records"),
            }
        else:
            pd.DataFrame().to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_main_fusion_pareto.csv", index=False)
    else:
        pd.DataFrame().to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_main_fusion_pareto.csv", index=False)

    return out, {
        "status": "ok",
        "selection_year": int(selection_year),
        "eval_year": int(eval_year),
        "selection_design": selection_design,
        "selection_rows": int(len(selection_df)),
        "selection_policy": "temporal_robust_rank",
        "runtime_profile": "compact" if compact_runtime_mode else "full",
        "search_grid_sizes": {
            "alpha_dynamic": int(len(alpha_dyn_grid)),
            "alpha_graph": int(len(graph_grid)),
            "alpha_critical": int(len(critical_grid)),
            "temporal_windows": int(temporal_window_cap),
            "bootstrap_eval": int(eval_bootstrap_draws),
            "bootstrap_backtest": int(backtest_bootstrap_draws),
            "bootstrap_continent": int(continent_bootstrap_draws),
        },
        "selection_rank_score": float(best.get("rank_score", best["selection_score"])),
        "temporal_selection_windows": int(base_temporal.get("n_windows") or 0),
        "temporal_selection_years": base_temporal.get("years") if isinstance(base_temporal, dict) else [],
        "selected_candidate": str(best.get("candidate", "convex_blend")),
        "selected_alpha_dynamic": alpha_dyn_best,
        "selected_alpha_graph": alpha_graph_best,
        "selected_alpha_critical": alpha_critical_best,
        "selected_alpha_base": alpha_base_best,
        "graph_enabled": bool(graph_enabled),
        "critical_enabled": bool(critical_enabled),
        "safety_guardrail": guardrail,
        "base_metrics_selection": met_base_sel,
        "selected_metrics_selection": best["selection_metrics"],
        "selected_temporal_metrics": best.get("temporal", {}),
        "base_metrics": met_base_eval,
        "selected_metrics": best["eval_metrics"],
        "inference_protocol": inference_protocol,
        "delta_auc_vs_base_single_eval": delta_auc_eval_single,
        "delta_brier_vs_base_single_eval": delta_brier_eval_single,
        "delta_auc_ci95_single_eval": auc_sig.get("ci95") if auc_sig.get("status") == "ok" else [None, None],
        "delta_auc_p_value_single_eval": auc_sig.get("p_value_two_sided") if auc_sig.get("status") == "ok" else None,
        "delta_brier_improve_ci95_single_eval": brier_sig.get("ci95") if brier_sig.get("status") == "ok" else [None, None],
        "delta_brier_improve_p_value_single_eval": brier_sig.get("p_value_two_sided")
        if brier_sig.get("status") == "ok"
        else None,
        "post_fusion_shrink": post_fusion_shrink,
        "post_fusion_calibration": post_fusion_calibration,
        "post_fusion_geo_regularization": geo_regularization_info,
        "continent_adaptive_regularization": continent_adaptive_info,
        "stacked_meta": stacked_info,
        "state_gated_moe": state_gated_info,
        "continent_robust_selection": continent_robust_info,
        "delta_auc_vs_base": delta_auc_primary,
        "delta_brier_vs_base": delta_brier_primary,
        "delta_auc_ci95": delta_auc_ci95_primary,
        "delta_auc_p_value": delta_auc_p_primary,
        "delta_brier_improve_ci95": delta_brier_improve_ci95_primary,
        "delta_brier_improve_p_value": delta_brier_improve_p_primary,
        "continent_generalization": fusion_continent_generalization,
        "multi_objective_frontier": frontier_info,
        "candidate_curve": rows,
        "quasi_temporal_backtest": quasi_backtest,
    }


def _estimate_dynamic_state_event_effects(dyn: pd.DataFrame) -> Dict[str, Any]:
    required = {"city_id", "year", "kinetic_state", "growth_1", "stall_risk_score"}
    if not required.issubset(set(dyn.columns)):
        pd.DataFrame().to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_state_event_study.csv", index=False)
        return {"status": "skipped", "reason": "missing_required_columns"}

    df = dyn.sort_values(["city_id", "year"]).copy()
    df["prev_state"] = df.groupby("city_id")["kinetic_state"].shift(1)
    year_mean = (
        df.groupby("year", as_index=False)
        .agg(year_mean_growth=("growth_1", "mean"), year_mean_risk=("stall_risk_score", "mean"))
        .sort_values("year")
    )
    df = df.merge(year_mean, on="year", how="left")
    df["growth_adj"] = df["growth_1"] - df["year_mean_growth"]
    df["risk_adj"] = df["stall_risk_score"] - df["year_mean_risk"]

    target_states = ["deep_cooling", "overheating", "recovery_rebound", "surge_expansion"]
    events: List[Dict[str, Any]] = []
    for state in target_states:
        hit = df[(df["kinetic_state"] == state) & (df["prev_state"].fillna("none") != state)][["city_id", "year"]].copy()
        for row in hit.itertuples(index=False):
            events.append({"city_id": str(row.city_id), "event_year": int(row.year), "event_type": f"enter_{state}"})

    evt = pd.DataFrame(events)
    if evt.empty:
        pd.DataFrame().to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_state_event_study.csv", index=False)
        return {"status": "skipped", "reason": "no_state_entry_events"}

    rows: List[Dict[str, Any]] = []
    keys = df.set_index(["city_id", "year"])
    for erow in evt.itertuples(index=False):
        cid = str(erow.city_id)
        ey = int(erow.event_year)
        et = str(erow.event_type)
        base_key = (cid, ey - 1)
        if base_key not in keys.index:
            continue
        base = keys.loc[base_key]
        base_g = float(base["growth_adj"]) if pd.notna(base["growth_adj"]) else 0.0
        base_r = float(base["risk_adj"]) if pd.notna(base["risk_adj"]) else 0.0
        for k in [-2, -1, 0, 1, 2]:
            key = (cid, ey + k)
            if key not in keys.index:
                continue
            now = keys.loc[key]
            g_now = float(now["growth_adj"]) if pd.notna(now["growth_adj"]) else np.nan
            r_now = float(now["risk_adj"]) if pd.notna(now["risk_adj"]) else np.nan
            rows.append(
                {
                    "city_id": cid,
                    "event_year": ey,
                    "event_type": et,
                    "event_time": int(k),
                    "rel_growth_adj": float(g_now - base_g) if pd.notna(g_now) else np.nan,
                    "rel_risk_adj": float(r_now - base_r) if pd.notna(r_now) else np.nan,
                }
            )

    panel = pd.DataFrame(rows)
    panel.to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_state_event_panel.csv", index=False)
    if panel.empty:
        pd.DataFrame().to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_state_event_study.csv", index=False)
        return {"status": "skipped", "reason": "event_windows_empty"}

    agg = (
        panel.groupby(["event_type", "event_time"], as_index=False)
        .agg(
            n_obs=("city_id", "size"),
            mean_rel_growth_adj=("rel_growth_adj", "mean"),
            mean_rel_risk_adj=("rel_risk_adj", "mean"),
        )
        .sort_values(["event_type", "event_time"])
    )
    agg.to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_state_event_study.csv", index=False)

    t1 = agg[agg["event_time"] == 1].copy()
    t1_growth_rank = (
        t1.sort_values("mean_rel_growth_adj", ascending=True)[["event_type", "mean_rel_growth_adj"]].to_dict(orient="records")
        if not t1.empty
        else []
    )
    t1_risk_rank = (
        t1.sort_values("mean_rel_risk_adj", ascending=False)[["event_type", "mean_rel_risk_adj"]].to_dict(orient="records")
        if not t1.empty
        else []
    )
    return {
        "status": "ok",
        "n_events": int(evt.shape[0]),
        "event_type_counts": evt["event_type"].value_counts().to_dict(),
        "event_types": sorted(evt["event_type"].unique().tolist()),
        "t_plus_1_growth_rank": t1_growth_rank,
        "t_plus_1_risk_rank": t1_risk_rank,
    }


def _estimate_dynamic_sync_network(dyn: pd.DataFrame) -> Dict[str, Any]:
    hazard_col = "dynamic_hazard_probability" if "dynamic_hazard_probability" in dyn.columns else "dynamic_hazard_fused_probability"
    required = {"continent", "year", "turning_point_risk", hazard_col}
    if not required.issubset(set(dyn.columns)):
        pd.DataFrame().to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_sync_network.csv", index=False)
        return {"status": "skipped", "reason": "missing_required_columns"}

    agg = (
        dyn.groupby(["continent", "year"], as_index=False)
        .agg(
            turning_point_risk=("turning_point_risk", "mean"),
            dynamic_hazard_probability=(hazard_col, "mean"),
            city_count=("city_id", "nunique"),
        )
        .dropna(subset=["turning_point_risk", "dynamic_hazard_probability"])
    )
    continents = sorted(agg["continent"].dropna().astype(str).unique().tolist())
    if len(continents) < 3:
        pd.DataFrame().to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_sync_network.csv", index=False)
        return {"status": "skipped", "reason": "insufficient_continents"}

    rows: List[Dict[str, Any]] = []
    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 4 or len(b) < 4:
            return np.nan
        if float(np.nanstd(a)) <= 1e-12 or float(np.nanstd(b)) <= 1e-12:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    for src in continents:
        src_df = agg[agg["continent"] == src][["year", "turning_point_risk", "dynamic_hazard_probability"]].rename(
            columns={
                "turning_point_risk": "src_turning",
                "dynamic_hazard_probability": "src_hazard",
            }
        )
        for dst in continents:
            if src == dst:
                continue
            dst_df = agg[agg["continent"] == dst][["year", "turning_point_risk", "dynamic_hazard_probability"]].rename(
                columns={
                    "turning_point_risk": "dst_turning",
                    "dynamic_hazard_probability": "dst_hazard",
                }
            )
            merged = src_df.merge(dst_df, on="year", how="inner").sort_values("year")
            if len(merged) < 6:
                continue
            lead = merged.iloc[:-1].copy()
            lag = merged.iloc[1:].copy()
            src_turn = lead["src_turning"].to_numpy(dtype=float)
            src_haz = lead["src_hazard"].to_numpy(dtype=float)
            dst_turn = lag["dst_turning"].to_numpy(dtype=float)
            dst_haz = lag["dst_hazard"].to_numpy(dtype=float)

            lead_corr_turn = _safe_corr(src_turn, dst_turn)
            lead_corr_haz = _safe_corr(src_haz, dst_haz)
            same_corr_turn = _safe_corr(
                merged["src_turning"].to_numpy(dtype=float),
                merged["dst_turning"].to_numpy(dtype=float),
            )
            same_corr_haz = _safe_corr(
                merged["src_hazard"].to_numpy(dtype=float),
                merged["dst_hazard"].to_numpy(dtype=float),
            )
            if not np.isfinite(lead_corr_turn) or not np.isfinite(lead_corr_haz):
                continue
            score = (
                0.55 * lead_corr_haz
                + 0.45 * lead_corr_turn
                - 0.30 * (same_corr_haz if np.isfinite(same_corr_haz) else 0.0)
                - 0.20 * (same_corr_turn if np.isfinite(same_corr_turn) else 0.0)
            )

            perm_draws: List[float] = []
            if len(src_turn) >= 6:
                seed_raw = f"{src}|{dst}".encode("utf-8")
                seed = int(sum((i + 1) * b for i, b in enumerate(seed_raw)) % (2**32 - 1))
                rng = np.random.default_rng(seed)
                for _ in range(320):
                    perm_idx = rng.permutation(len(src_turn))
                    p_turn = _safe_corr(src_turn, dst_turn[perm_idx])
                    p_haz = _safe_corr(src_haz, dst_haz[perm_idx])
                    if not np.isfinite(p_turn) or not np.isfinite(p_haz):
                        continue
                    p_score = (
                        0.55 * p_haz
                        + 0.45 * p_turn
                        - 0.30 * (same_corr_haz if np.isfinite(same_corr_haz) else 0.0)
                        - 0.20 * (same_corr_turn if np.isfinite(same_corr_turn) else 0.0)
                    )
                    perm_draws.append(float(p_score))
            if len(perm_draws) >= 80:
                perm_arr = np.array(perm_draws, dtype=float)
                p_value = float(np.mean(perm_arr >= float(score)))
                mu = float(np.mean(perm_arr))
                sd = float(np.std(perm_arr))
                z_score = float((score - mu) / sd) if sd > 1e-12 else None
            else:
                p_value = None
                z_score = None
            rows.append(
                {
                    "source_continent": src,
                    "target_continent": dst,
                    "lead_corr_turning": lead_corr_turn,
                    "lead_corr_hazard": lead_corr_haz,
                    "same_corr_turning": same_corr_turn,
                    "same_corr_hazard": same_corr_haz,
                    "lead_score": float(score),
                    "n_year_pairs": int(len(lead)),
                    "lead_score_p_value_perm": p_value,
                    "lead_score_z_perm": z_score,
                    "perm_draws": int(len(perm_draws)),
                }
            )

    net = pd.DataFrame(rows)
    net.to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_sync_network.csv", index=False)
    if net.empty:
        return {"status": "skipped", "reason": "no_valid_edges"}

    net_pos = net[net["lead_score"] > 0.0].copy()
    net_sig10 = net_pos[net_pos["lead_score_p_value_perm"].fillna(1.0) < 0.10].copy()
    net_sig05 = net_pos[net_pos["lead_score_p_value_perm"].fillna(1.0) < 0.05].copy()
    out_strength = net_pos.groupby("source_continent")["lead_score"].sum().sort_values(ascending=False)
    in_strength = net_pos.groupby("target_continent")["lead_score"].sum().sort_values(ascending=False)
    top_edges = net.sort_values("lead_score", ascending=False).head(8)
    return {
        "status": "ok",
        "n_edges": int(len(net)),
        "n_positive_edges": int(len(net_pos)),
        "n_positive_edges_p_lt_0_10": int(len(net_sig10)),
        "n_positive_edges_p_lt_0_05": int(len(net_sig05)),
        "hazard_source_col": hazard_col,
        "top_out_leaders": out_strength.head(4).to_dict(),
        "top_in_followers": in_strength.head(4).to_dict(),
        "top_edges": top_edges.to_dict(orient="records"),
    }


def _estimate_dynamic_phase_portrait(
    dyn: pd.DataFrame,
    latest_year: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = dyn.sort_values(["city_id", "year"]).copy()
    field_cols = [
        "phase_risk_bin",
        "phase_accel_bin",
        "risk_center",
        "accel_center",
        "risk_velocity",
        "accel_velocity",
        "flow_speed",
        "mean_curvature",
        "divergence",
        "vorticity",
        "n_obs",
        "phase_label",
    ]
    latest_cols = [
        "city_id",
        "city_name",
        "country",
        "continent",
        "year",
        "stall_risk_score",
        "acceleration_score",
        "phase_risk_velocity",
        "phase_accel_velocity",
        "phase_flow_speed",
        "phase_curvature",
        "phase_divergence_bin",
        "phase_vorticity_bin",
        "phase_label",
        "phase_instability_score",
    ]
    field_path = DATA_OUTPUTS / "pulse_ai_dynamic_phase_field.csv"
    latest_path = DATA_OUTPUTS / "pulse_ai_dynamic_phase_latest.csv"

    out["phase_risk_velocity"] = np.nan
    out["phase_accel_velocity"] = np.nan
    out["phase_flow_speed"] = np.nan
    out["phase_curvature"] = np.nan
    out["phase_divergence_bin"] = np.nan
    out["phase_vorticity_bin"] = np.nan
    out["phase_label"] = "insufficient_history"

    required = {"city_id", "year", "stall_risk_score", "acceleration_score"}
    if not required.issubset(set(out.columns)):
        pd.DataFrame(columns=field_cols).to_csv(field_path, index=False)
        pd.DataFrame(columns=latest_cols).to_csv(latest_path, index=False)
        return out, {"status": "skipped", "reason": "missing_required_columns"}

    out["stall_risk_score"] = pd.to_numeric(out["stall_risk_score"], errors="coerce")
    out["acceleration_score"] = pd.to_numeric(out["acceleration_score"], errors="coerce")
    out["phase_risk_velocity"] = out.groupby("city_id")["stall_risk_score"].diff()
    out["phase_accel_velocity"] = out.groupby("city_id")["acceleration_score"].diff()
    out["phase_flow_speed"] = np.sqrt(
        out["phase_risk_velocity"].fillna(0.0) ** 2 + out["phase_accel_velocity"].fillna(0.0) ** 2
    )
    out["phase_risk_acceleration"] = out.groupby("city_id")["phase_risk_velocity"].diff()
    out["phase_accel_acceleration"] = out.groupby("city_id")["phase_accel_velocity"].diff()
    denom = out["phase_flow_speed"].fillna(0.0) ** 3 + 1e-6
    out["phase_curvature"] = np.clip(
        (
            out["phase_risk_velocity"].fillna(0.0) * out["phase_accel_acceleration"].fillna(0.0)
            - out["phase_accel_velocity"].fillna(0.0) * out["phase_risk_acceleration"].fillna(0.0)
        )
        / denom,
        -3.0,
        3.0,
    )

    valid = out[
        out["phase_risk_velocity"].notna()
        & out["phase_accel_velocity"].notna()
        & out["stall_risk_score"].notna()
        & out["acceleration_score"].notna()
    ].copy()
    if len(valid) < 140:
        pd.DataFrame(columns=field_cols).to_csv(field_path, index=False)
        latest_min = out[out["year"] == latest_year].copy()
        if not latest_min.empty:
            latest_min["phase_instability_score"] = latest_min["phase_flow_speed"].fillna(0.0)
            latest_min["phase_label"] = latest_min["phase_label"].fillna("insufficient_history")
            latest_min = latest_min[[c for c in latest_cols if c in latest_min.columns]].copy()
            latest_min.to_csv(latest_path, index=False)
        else:
            pd.DataFrame(columns=latest_cols).to_csv(latest_path, index=False)
        out = out.drop(columns=[c for c in ["phase_risk_acceleration", "phase_accel_acceleration"] if c in out.columns])
        return out, {"status": "skipped", "reason": "insufficient_valid_rows"}

    def _build_edges(values: np.ndarray, bins: int = 8) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.linspace(0.0, 100.0, bins + 1)
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-9:
            return np.linspace(0.0, 100.0, bins + 1)
        if arr.size < bins + 3:
            return np.linspace(lo, hi, bins + 1)
        edges = np.quantile(arr, np.linspace(0.0, 1.0, bins + 1))
        edges = np.asarray(edges, dtype=float)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-3
        if edges[-1] <= edges[0] + 1e-9:
            return np.linspace(lo, hi, bins + 1)
        return edges

    risk_edges = _build_edges(valid["stall_risk_score"].to_numpy(dtype=float), bins=8)
    accel_edges = _build_edges(valid["acceleration_score"].to_numpy(dtype=float), bins=8)
    valid["phase_risk_bin"] = pd.cut(
        valid["stall_risk_score"],
        bins=risk_edges,
        include_lowest=True,
        labels=False,
    )
    valid["phase_accel_bin"] = pd.cut(
        valid["acceleration_score"],
        bins=accel_edges,
        include_lowest=True,
        labels=False,
    )
    valid = valid[valid["phase_risk_bin"].notna() & valid["phase_accel_bin"].notna()].copy()
    valid["phase_risk_bin"] = valid["phase_risk_bin"].astype(int)
    valid["phase_accel_bin"] = valid["phase_accel_bin"].astype(int)

    if valid.empty:
        pd.DataFrame(columns=field_cols).to_csv(field_path, index=False)
        pd.DataFrame(columns=latest_cols).to_csv(latest_path, index=False)
        out = out.drop(columns=[c for c in ["phase_risk_acceleration", "phase_accel_acceleration"] if c in out.columns])
        return out, {"status": "skipped", "reason": "phase_bins_empty"}

    field = (
        valid.groupby(["phase_risk_bin", "phase_accel_bin"], as_index=False)
        .agg(
            risk_center=("stall_risk_score", "mean"),
            accel_center=("acceleration_score", "mean"),
            risk_velocity=("phase_risk_velocity", "mean"),
            accel_velocity=("phase_accel_velocity", "mean"),
            flow_speed=("phase_flow_speed", "mean"),
            mean_curvature=("phase_curvature", "mean"),
            n_obs=("city_id", "size"),
        )
        .sort_values(["phase_risk_bin", "phase_accel_bin"])
    )

    n_risk = int(field["phase_risk_bin"].max()) + 1
    n_accel = int(field["phase_accel_bin"].max()) + 1
    vr = np.full((n_risk, n_accel), np.nan, dtype=float)
    va = np.full((n_risk, n_accel), np.nan, dtype=float)
    rc = np.full((n_risk, n_accel), np.nan, dtype=float)
    ac = np.full((n_risk, n_accel), np.nan, dtype=float)
    nn = np.zeros((n_risk, n_accel), dtype=float)
    for row in field.itertuples(index=False):
        i = int(row.phase_risk_bin)
        j = int(row.phase_accel_bin)
        vr[i, j] = float(row.risk_velocity)
        va[i, j] = float(row.accel_velocity)
        rc[i, j] = float(row.risk_center)
        ac[i, j] = float(row.accel_center)
        nn[i, j] = float(row.n_obs)

    risk_center_vals: List[float] = []
    for i in range(n_risk):
        row_vals = rc[i, :]
        finite = row_vals[np.isfinite(row_vals)]
        risk_center_vals.append(float(np.mean(finite)) if finite.size > 0 else float(i))
    risk_centers = np.asarray(risk_center_vals, dtype=float)
    accel_center_vals: List[float] = []
    for j in range(n_accel):
        col_vals = ac[:, j]
        finite = col_vals[np.isfinite(col_vals)]
        accel_center_vals.append(float(np.mean(finite)) if finite.size > 0 else float(j))
    accel_centers = np.asarray(accel_center_vals, dtype=float)

    def _find_neighbor(line: np.ndarray, idx: int, step: int) -> int | None:
        k = idx + step
        while 0 <= k < len(line):
            if np.isfinite(line[k]):
                return int(k)
            k += step
        return None

    def _partial(arr: np.ndarray, centers: np.ndarray, i: int, j: int, axis: int) -> float:
        if axis == 0:
            line = arr[:, j]
            idx = int(i)
        else:
            line = arr[i, :]
            idx = int(j)
        v0 = float(line[idx]) if np.isfinite(line[idx]) else np.nan
        if not np.isfinite(v0):
            return np.nan
        lo = _find_neighbor(line, idx, -1)
        hi = _find_neighbor(line, idx, +1)
        if lo is not None and hi is not None:
            den = float(centers[hi] - centers[lo])
            if abs(den) <= 1e-12:
                return np.nan
            return float((line[hi] - line[lo]) / den)
        if hi is not None:
            den = float(centers[hi] - centers[idx])
            if abs(den) <= 1e-12:
                return np.nan
            return float((line[hi] - line[idx]) / den)
        if lo is not None:
            den = float(centers[idx] - centers[lo])
            if abs(den) <= 1e-12:
                return np.nan
            return float((line[idx] - line[lo]) / den)
        return np.nan

    div_map: Dict[tuple[int, int], float] = {}
    vort_map: Dict[tuple[int, int], float] = {}
    for i in range(n_risk):
        for j in range(n_accel):
            if nn[i, j] <= 0 or (not np.isfinite(vr[i, j])) or (not np.isfinite(va[i, j])):
                continue
            dvr_dr = _partial(vr, risk_centers, i=i, j=j, axis=0)
            dva_da = _partial(va, accel_centers, i=i, j=j, axis=1)
            dva_dr = _partial(va, risk_centers, i=i, j=j, axis=0)
            dvr_da = _partial(vr, accel_centers, i=i, j=j, axis=1)
            divergence = float(dvr_dr + dva_da) if np.isfinite(dvr_dr) and np.isfinite(dva_da) else np.nan
            vorticity = float(dva_dr - dvr_da) if np.isfinite(dva_dr) and np.isfinite(dvr_da) else np.nan
            div_map[(int(i), int(j))] = divergence
            vort_map[(int(i), int(j))] = vorticity

    field["divergence"] = field.apply(
        lambda r: float(div_map.get((int(r["phase_risk_bin"]), int(r["phase_accel_bin"])), np.nan)),
        axis=1,
    )
    field["vorticity"] = field.apply(
        lambda r: float(vort_map.get((int(r["phase_risk_bin"]), int(r["phase_accel_bin"])), np.nan)),
        axis=1,
    )

    abs_div = np.abs(field["divergence"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float))
    abs_vort = np.abs(field["vorticity"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float))
    speed_vals = field["flow_speed"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    div_thr = float(np.quantile(abs_div, 0.60)) if abs_div.size else 0.012
    vort_thr = float(np.quantile(abs_vort, 0.70)) if abs_vort.size else 0.012
    speed_thr = float(np.quantile(speed_vals, 0.55)) if speed_vals.size else 1.6
    div_thr = max(div_thr, 0.012)
    vort_thr = max(vort_thr, 0.012)
    speed_thr = max(speed_thr, 1.6)

    def _phase_label(divergence: float, vorticity: float, speed: float) -> str:
        spd = float(speed) if np.isfinite(speed) else 0.0
        div = float(divergence) if np.isfinite(divergence) else np.nan
        vort = float(vorticity) if np.isfinite(vorticity) else np.nan
        if np.isfinite(div) and div >= div_thr and spd >= speed_thr:
            return "centrifugal_fragility"
        if np.isfinite(div) and div <= -div_thr and spd >= speed_thr:
            return "centripetal_recovery"
        if np.isfinite(vort) and abs(vort) >= vort_thr and spd >= (0.80 * speed_thr):
            return "rotational_transition"
        if spd <= (0.55 * speed_thr):
            return "stable_anchor"
        return "directional_drift"

    field["phase_label"] = field.apply(
        lambda r: _phase_label(
            float(r["divergence"]) if pd.notna(r["divergence"]) else np.nan,
            float(r["vorticity"]) if pd.notna(r["vorticity"]) else np.nan,
            float(r["flow_speed"]) if pd.notna(r["flow_speed"]) else np.nan,
        ),
        axis=1,
    )
    field.to_csv(field_path, index=False)

    valid = valid.assign(_row_index=valid.index)
    mapped = valid.merge(
        field[
            [
                "phase_risk_bin",
                "phase_accel_bin",
                "divergence",
                "vorticity",
                "phase_label",
            ]
        ].rename(columns={"phase_label": "phase_label_bin"}),
        on=["phase_risk_bin", "phase_accel_bin"],
        how="left",
    )
    out.loc[mapped["_row_index"].to_numpy(dtype=int), "phase_divergence_bin"] = mapped["divergence"].to_numpy(dtype=float)
    out.loc[mapped["_row_index"].to_numpy(dtype=int), "phase_vorticity_bin"] = mapped["vorticity"].to_numpy(dtype=float)
    out.loc[mapped["_row_index"].to_numpy(dtype=int), "phase_label"] = (
        mapped["phase_label_bin"].fillna("unclassified").astype(str).to_numpy()
    )

    latest = out[out["year"] == latest_year].copy()
    if latest.empty:
        pd.DataFrame(columns=latest_cols).to_csv(latest_path, index=False)
        out = out.drop(columns=[c for c in ["phase_risk_acceleration", "phase_accel_acceleration"] if c in out.columns])
        return out, {"status": "skipped", "reason": "latest_rows_missing"}

    latest["phase_instability_score"] = (
        latest["phase_flow_speed"].fillna(0.0)
        + 6.0 * np.maximum(latest["phase_divergence_bin"].fillna(0.0), 0.0)
        + 4.0 * latest["phase_vorticity_bin"].fillna(0.0).abs()
        + 2.0 * latest["phase_curvature"].fillna(0.0).abs()
    )
    latest_out = latest[[c for c in latest_cols if c in latest.columns]].copy()
    latest_out = latest_out.sort_values("phase_instability_score", ascending=False)
    latest_out.to_csv(latest_path, index=False)

    phase_share = latest_out["phase_label"].fillna("unclassified").astype(str).value_counts(normalize=True).round(4).to_dict()
    top_unstable = latest_out.head(10)[
        [
            "city_name",
            "country",
            "phase_label",
            "phase_instability_score",
            "phase_flow_speed",
            "phase_divergence_bin",
            "phase_vorticity_bin",
            "phase_curvature",
        ]
    ].to_dict(orient="records")

    out = out.drop(columns=[c for c in ["phase_risk_acceleration", "phase_accel_acceleration"] if c in out.columns])
    return out, {
        "status": "ok",
        "n_cells_active": int(len(field)),
        "n_cells_total": int(n_risk * n_accel),
        "divergence_threshold": float(div_thr),
        "vorticity_threshold": float(vort_thr),
        "speed_threshold": float(speed_thr),
        "mean_divergence": float(field["divergence"].replace([np.inf, -np.inf], np.nan).mean())
        if field["divergence"].notna().any()
        else None,
        "mean_abs_vorticity": float(np.nanmean(np.abs(field["vorticity"].to_numpy(dtype=float))))
        if field["vorticity"].notna().any()
        else None,
        "phase_distribution_latest": phase_share,
        "top_unstable_cities": top_unstable,
    }


def _build_dynamic_pulse_index(
    dyn: pd.DataFrame,
    latest_year: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = dyn.sort_values(["city_id", "year"]).copy()
    required = {"city_id", "year", "acceleration_score", "stall_risk_score"}
    series_path = DATA_OUTPUTS / "pulse_ai_dynamic_index_series.csv"
    latest_path = DATA_OUTPUTS / "pulse_ai_dynamic_index_latest.csv"
    cont_year_path = DATA_OUTPUTS / "pulse_ai_dynamic_index_continent_year.csv"
    if not required.issubset(out.columns):
        pd.DataFrame().to_csv(series_path, index=False)
        pd.DataFrame().to_csv(latest_path, index=False)
        pd.DataFrame().to_csv(cont_year_path, index=False)
        return out, {"status": "skipped", "reason": "missing_required_columns"}

    for col in ["acceleration_score", "stall_risk_score"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out["acceleration_score"] = out["acceleration_score"].fillna(float(out["acceleration_score"].median()))
    out["stall_risk_score"] = out["stall_risk_score"].fillna(float(out["stall_risk_score"].median()))

    out["pulse_accel_velocity"] = out.groupby("city_id")["acceleration_score"].diff()
    out["pulse_risk_velocity"] = -out.groupby("city_id")["stall_risk_score"].diff()
    out["pulse_accel_velocity"] = out["pulse_accel_velocity"].fillna(0.0)
    out["pulse_risk_velocity"] = out["pulse_risk_velocity"].fillna(0.0)

    def _norm(arr: pd.Series) -> np.ndarray:
        vals = pd.to_numeric(arr, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        if np.all(~np.isfinite(vals)):
            return np.zeros(len(vals), dtype=float)
        vals = np.where(np.isfinite(vals), vals, np.nanmedian(vals[np.isfinite(vals)]))
        return minmax_scale(vals)

    level_risk_inv = np.clip(1.0 - out["stall_risk_score"].to_numpy(dtype=float) / 100.0, 0.0, 1.0)
    level_accel = np.clip(out["acceleration_score"].to_numpy(dtype=float) / 100.0, 0.0, 1.0)
    v_risk = _norm(out["pulse_risk_velocity"])
    v_accel = _norm(out["pulse_accel_velocity"])

    if "regime_transition_entropy" in out.columns:
        stability = np.clip(1.0 - _norm(out["regime_transition_entropy"]), 0.0, 1.0)
    elif "regime_switch_rate_3y" in out.columns:
        stability = np.clip(1.0 - _norm(out["regime_switch_rate_3y"]), 0.0, 1.0)
    else:
        stability = np.full(len(out), 0.5, dtype=float)

    if "turning_point_risk" in out.columns:
        adaptive_space = np.clip(1.0 - _norm(out["turning_point_risk"]), 0.0, 1.0)
    else:
        adaptive_space = np.full(len(out), 0.5, dtype=float)

    raw = (
        0.30 * level_risk_inv
        + 0.24 * level_accel
        + 0.18 * v_risk
        + 0.14 * v_accel
        + 0.09 * stability
        + 0.05 * adaptive_space
    )
    out["dynamic_pulse_index"] = np.clip(100.0 * minmax_scale(raw), 0.0, 100.0)
    out["dynamic_pulse_delta_1y"] = out.groupby("city_id")["dynamic_pulse_index"].diff().fillna(0.0)
    out["dynamic_pulse_trend_3y"] = out.groupby("city_id")["dynamic_pulse_index"].transform(
        lambda s: s.diff().rolling(3, min_periods=2).mean()
    ).fillna(0.0)

    high = float(np.quantile(out["dynamic_pulse_index"].to_numpy(dtype=float), 0.70))
    low = float(np.quantile(out["dynamic_pulse_index"].to_numpy(dtype=float), 0.30))
    out["dynamic_pulse_state"] = np.select(
        [
            (out["dynamic_pulse_index"] >= high) & (out["dynamic_pulse_delta_1y"] >= 0.0),
            (out["dynamic_pulse_index"] >= high) & (out["dynamic_pulse_delta_1y"] < 0.0),
            (out["dynamic_pulse_index"] <= low) & (out["dynamic_pulse_delta_1y"] < 0.0),
            (out["dynamic_pulse_index"] <= low) & (out["dynamic_pulse_delta_1y"] >= 0.0),
        ],
        ["accelerating_frontier", "high_but_cooling", "fragile_decline", "low_but_recovering"],
        default="mid_transition",
    )

    series_cols = [
        "city_id",
        "city_name",
        "country",
        "continent",
        "year",
        "dynamic_pulse_index",
        "dynamic_pulse_delta_1y",
        "dynamic_pulse_trend_3y",
        "dynamic_pulse_state",
        "pulse_accel_velocity",
        "pulse_risk_velocity",
    ]
    out[[c for c in series_cols if c in out.columns]].to_csv(series_path, index=False)

    latest = out[out["year"] == latest_year].copy()
    latest = latest.sort_values("dynamic_pulse_index", ascending=False)
    latest_cols = [
        "city_id",
        "city_name",
        "country",
        "continent",
        "year",
        "dynamic_pulse_index",
        "dynamic_pulse_delta_1y",
        "dynamic_pulse_trend_3y",
        "dynamic_pulse_state",
        "acceleration_score",
        "stall_risk_score",
        "trajectory_regime",
        "kinetic_state",
    ]
    latest[[c for c in latest_cols if c in latest.columns]].to_csv(latest_path, index=False)

    if "continent" in out.columns:
        cont_year = (
            out.groupby(["continent", "year"], as_index=False)
            .agg(
                dynamic_pulse_index_mean=("dynamic_pulse_index", "mean"),
                dynamic_pulse_index_p75=("dynamic_pulse_index", lambda s: float(np.quantile(s, 0.75))),
                dynamic_pulse_delta_1y_mean=("dynamic_pulse_delta_1y", "mean"),
                city_count=("city_id", "nunique"),
            )
            .sort_values(["continent", "year"])
        )
    else:
        cont_year = pd.DataFrame()
    cont_year.to_csv(cont_year_path, index=False)

    top_movers = (
        latest.sort_values("dynamic_pulse_delta_1y", ascending=False)
        .head(12)[[c for c in ["city_name", "country", "dynamic_pulse_index", "dynamic_pulse_delta_1y", "dynamic_pulse_state"] if c in latest.columns]]
        .to_dict(orient="records")
    )
    bottom_movers = (
        latest.sort_values("dynamic_pulse_delta_1y", ascending=True)
        .head(12)[[c for c in ["city_name", "country", "dynamic_pulse_index", "dynamic_pulse_delta_1y", "dynamic_pulse_state"] if c in latest.columns]]
        .to_dict(orient="records")
    )

    summary = {
        "status": "ok",
        "latest_year": int(latest_year),
        "index_threshold_high_q70": high,
        "index_threshold_low_q30": low,
        "latest_mean_index": float(latest["dynamic_pulse_index"].mean()) if not latest.empty else None,
        "latest_state_distribution": latest["dynamic_pulse_state"].value_counts(normalize=True).round(4).to_dict()
        if not latest.empty
        else {},
        "top_movers_1y": top_movers,
        "bottom_movers_1y": bottom_movers,
    }
    return out, summary


def _simulate_dynamic_policy_lab(
    dyn: pd.DataFrame,
    latest_year: int,
) -> Dict[str, Any]:
    latest = dyn[dyn["year"] == latest_year].copy()
    if latest.empty:
        pd.DataFrame().to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_policy_lab.csv", index=False)
        return {"status": "skipped", "reason": "latest_rows_missing"}

    base_features = [
        "stall_probability",
        "dynamic_hazard_probability",
        "dynamic_hazard_fused_probability",
        "graph_diffusion_probability",
        "turning_point_risk",
        "damping_pressure",
        "kinetic_energy_score",
        "acceleration_score",
        "regime_forward_risk",
    ]
    features = [c for c in base_features if c in dyn.columns]
    hist = dyn[(dyn["year"] <= (latest_year - 1)) & dyn["stall_next"].notna()].copy()
    if len(features) < 4 or len(hist) < 200:
        pd.DataFrame().to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_policy_lab.csv", index=False)
        return {"status": "skipped", "reason": "insufficient_training_support"}

    for col in features:
        hist[col] = pd.to_numeric(hist[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        latest[col] = pd.to_numeric(latest[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = hist[features].median(numeric_only=True)
    hist[features] = hist[features].fillna(med)
    latest[features] = latest[features].fillna(med)

    y = hist["stall_next"].astype(float).to_numpy()
    x = hist[features].to_numpy(dtype=float)
    lm = LinearRegression()
    lm.fit(x, y)
    coef = dict(zip(features, lm.coef_.tolist(), strict=False))

    # City-level heterogeneous exposure indexes for counterfactual response scaling.
    def _z(arr: pd.Series) -> pd.Series:
        sd = float(arr.std())
        if sd <= 1e-12 or not np.isfinite(sd):
            return pd.Series(np.zeros(len(arr)), index=arr.index)
        return (arr - float(arr.mean())) / sd

    latest_num = latest.copy()
    for col in features:
        latest_num[col] = pd.to_numeric(latest_num[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(hist[col].median()))

    z_turn = _z(latest_num["turning_point_risk"]) if "turning_point_risk" in latest_num.columns else pd.Series(np.zeros(len(latest_num)), index=latest_num.index)
    z_dyn = _z(latest_num["dynamic_hazard_fused_probability"]) if "dynamic_hazard_fused_probability" in latest_num.columns else pd.Series(np.zeros(len(latest_num)), index=latest_num.index)
    z_reg = _z(latest_num["regime_forward_risk"]) if "regime_forward_risk" in latest_num.columns else pd.Series(np.zeros(len(latest_num)), index=latest_num.index)
    z_acc = _z(latest_num["acceleration_score"]) if "acceleration_score" in latest_num.columns else pd.Series(np.zeros(len(latest_num)), index=latest_num.index)
    z_ke = _z(latest_num["kinetic_energy_score"]) if "kinetic_energy_score" in latest_num.columns else pd.Series(np.zeros(len(latest_num)), index=latest_num.index)
    z_damp = _z(latest_num["damping_pressure"]) if "damping_pressure" in latest_num.columns else pd.Series(np.zeros(len(latest_num)), index=latest_num.index)

    vuln_raw = 0.35 * z_turn + 0.30 * z_dyn + 0.25 * z_reg - 0.20 * z_acc + 0.10 * z_ke + 0.10 * z_damp
    vulnerability = minmax_scale(vuln_raw.to_numpy(dtype=float))
    readiness = 1.0 - vulnerability
    overheat_pressure = minmax_scale((0.65 * z_ke + 0.35 * z_turn).to_numpy(dtype=float))

    feature_z: Dict[str, np.ndarray] = {}
    for feat in features:
        mu = float(hist[feat].mean())
        sd = float(hist[feat].std())
        if sd <= 1e-12 or not np.isfinite(sd):
            feature_z[feat] = np.zeros(len(latest_num), dtype=float)
        else:
            feature_z[feat] = ((latest_num[feat].to_numpy(dtype=float) - mu) / sd).astype(float)

    state = latest.get("kinetic_state", pd.Series(["mixed_transition"] * len(latest))).fillna("mixed_transition").astype(str)
    regime = latest.get("trajectory_regime", pd.Series(["mixed_transition"] * len(latest))).fillna("mixed_transition").astype(str)
    is_overheating = (state == "overheating").astype(float).to_numpy()
    is_cooling = (state == "deep_cooling").astype(float).to_numpy()
    is_decline = regime.str.contains("decline|stalling", case=False, regex=True).astype(float).to_numpy()

    scenarios = {
        "resilience_upgrade": {
            "turning_point_risk": -8.0,
            "damping_pressure": -7.0,
            "acceleration_score": 5.0,
            "kinetic_energy_score": -3.0,
        },
        "innovation_pulse": {
            "acceleration_score": 8.0,
            "regime_forward_risk": -0.03,
            "dynamic_hazard_fused_probability": -0.02,
        },
        "anti_overheating": {
            "kinetic_energy_score": -9.0,
            "turning_point_risk": -6.0,
            "damping_pressure": -3.0,
        },
    }

    rows: List[Dict[str, Any]] = []
    baseline_h1 = latest["forecast_stall_prob_h1"].astype(float).to_numpy() if "forecast_stall_prob_h1" in latest.columns else latest["stall_probability"].astype(float).to_numpy()
    baseline_h2 = latest["forecast_stall_prob_h2"].astype(float).to_numpy() if "forecast_stall_prob_h2" in latest.columns else baseline_h1
    baseline_h3 = latest["forecast_stall_prob_h3"].astype(float).to_numpy() if "forecast_stall_prob_h3" in latest.columns else baseline_h1
    for sc_name, deltas in scenarios.items():
        base_shift = 0.0
        interaction = np.zeros(len(latest), dtype=float)
        for feat, d in deltas.items():
            if feat not in coef:
                continue
            beta = float(coef[feat])
            dd = float(d)
            base_shift += beta * dd
            interaction += 0.11 * beta * dd * feature_z.get(feat, np.zeros(len(latest), dtype=float))

        if sc_name == "resilience_upgrade":
            multiplier = 0.78 + 0.95 * vulnerability + 0.18 * is_cooling + 0.12 * is_decline
        elif sc_name == "innovation_pulse":
            multiplier = 0.70 + 0.85 * readiness + 0.20 * (z_acc.to_numpy(dtype=float) > 0.0).astype(float) - 0.16 * is_overheating
        else:  # anti_overheating
            multiplier = 0.74 + 0.88 * overheat_pressure + 0.18 * is_overheating
        multiplier = np.clip(multiplier, 0.35, 2.30)

        shift_prob = np.clip(base_shift * multiplier + interaction, -0.35, 0.25)
        h1_new = np.clip(baseline_h1 + 0.85 * shift_prob, 0.0, 1.0)
        h2_new = np.clip(baseline_h2 + 0.65 * shift_prob, 0.0, 1.0)
        h3_new = np.clip(baseline_h3 + 0.50 * shift_prob, 0.0, 1.0)
        latest_rows = latest[["city_id", "city_name", "country", "continent"]].copy()
        latest_rows["scenario"] = sc_name
        latest_rows["vulnerability_index"] = vulnerability
        latest_rows["scenario_multiplier"] = multiplier
        latest_rows["interaction_shift_prob"] = interaction
        latest_rows["baseline_h1"] = baseline_h1
        latest_rows["baseline_h2"] = baseline_h2
        latest_rows["baseline_h3"] = baseline_h3
        latest_rows["counterfactual_h1"] = h1_new
        latest_rows["counterfactual_h2"] = h2_new
        latest_rows["counterfactual_h3"] = h3_new
        latest_rows["delta_h1"] = h1_new - baseline_h1
        latest_rows["delta_h2"] = h2_new - baseline_h2
        latest_rows["delta_h3"] = h3_new - baseline_h3
        rows.append(latest_rows)

    out = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_policy_lab.csv", index=False)
    if out.empty:
        return {"status": "skipped", "reason": "empty_policy_rows"}

    summary_rows = (
        out.groupby("scenario", as_index=False)
        .agg(
            mean_delta_h1=("delta_h1", "mean"),
            mean_delta_h2=("delta_h2", "mean"),
            mean_delta_h3=("delta_h3", "mean"),
            std_delta_h1=("delta_h1", "std"),
            p90_delta_h1=("delta_h1", lambda s: float(np.quantile(s, 0.90))),
            p10_delta_h1=("delta_h1", lambda s: float(np.quantile(s, 0.10))),
            share_improved_h1=("delta_h1", lambda s: float(np.mean(np.asarray(s) < 0.0))),
            worst10_mean_h1=("delta_h1", lambda s: float(np.quantile(s, 0.10))),
            best10_mean_h1=("delta_h1", lambda s: float(np.quantile(s, 0.90))),
        )
        .sort_values("mean_delta_h1")
    )

    boot_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(2031)
    for row in summary_rows.itertuples(index=False):
        sc = str(row.scenario)
        vals = out[out["scenario"] == sc]["delta_h1"].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) < 20:
            boot_rows.append({"scenario": sc, "mean_delta_h1_ci_low": None, "mean_delta_h1_ci_high": None, "mean_delta_h1_p_value": None})
            continue
        draws = []
        for _ in range(500):
            idx = rng.integers(0, len(vals), size=len(vals))
            draws.append(float(np.mean(vals[idx])))
        arr = np.array(draws, dtype=float)
        p_val = float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))
        boot_rows.append(
            {
                "scenario": sc,
                "mean_delta_h1_ci_low": float(np.quantile(arr, 0.025)),
                "mean_delta_h1_ci_high": float(np.quantile(arr, 0.975)),
                "mean_delta_h1_p_value": min(max(p_val, 0.0), 1.0),
            }
        )
    if boot_rows:
        summary_rows = summary_rows.merge(pd.DataFrame(boot_rows), on="scenario", how="left")

    summary_rows.to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_policy_lab_summary.csv", index=False)
    return {
        "status": "ok",
        "n_rows": int(len(out)),
        "n_scenarios": int(summary_rows["scenario"].nunique()),
        "scenarios_ranked": summary_rows.to_dict(orient="records"),
    }


def _simulate_dynamic_policy_offline_rl(
    dyn: pd.DataFrame,
    latest_year: int,
    gamma: float = 0.88,
    n_iterations: int = 6,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    city_path = DATA_OUTPUTS / "pulse_ai_dynamic_policy_rl_city.csv"
    action_path = DATA_OUTPUTS / "pulse_ai_dynamic_policy_rl_action_summary.csv"
    state_path = DATA_OUTPUTS / "pulse_ai_dynamic_policy_rl_state_value.csv"
    ope_path = DATA_OUTPUTS / "pulse_ai_dynamic_policy_rl_ope.csv"
    ablation_path = DATA_OUTPUTS / "pulse_ai_dynamic_policy_rl_ablation.csv"
    continent_ope_path = DATA_OUTPUTS / "pulse_ai_dynamic_policy_rl_continent_ope.csv"
    continent_action_path = DATA_OUTPUTS / "pulse_ai_dynamic_policy_rl_continent_action.csv"

    city_cols = [
        "city_id",
        "city_name",
        "country",
        "continent",
        "year",
        "kinetic_state",
        "trajectory_regime",
        "stall_risk_score",
        "acceleration_score",
        "rl_state_id",
        "rl_best_action",
        "rl_best_q_value",
        "rl_second_q_value",
        "rl_q_advantage",
        "rl_policy_utility",
        "rl_selection_source",
        "rl_policy_confidence",
        "rl_baseline_risk_h1",
        "rl_expected_risk_h1",
        "rl_expected_delta_risk_h1",
        "rl_expected_delta_accel_h1",
    ]
    action_cols = [
        "action",
        "selected_city_count",
        "selected_share",
        "mean_best_q_value",
        "mean_q_advantage",
        "mean_policy_confidence",
        "mean_expected_delta_risk_h1",
        "mean_expected_delta_accel_h1",
        "selected_by_rule_count",
        "mean_q_if_forced",
        "mean_delta_risk_if_forced",
        "mean_delta_accel_if_forced",
    ]
    state_cols = [
        "state_id",
        "kinetic_state",
        "trajectory_regime",
        "risk_bucket",
        "accel_bucket",
        "visit_count",
        "rl_state_value",
        "rl_best_action",
    ]
    ope_cols = [
        "estimator",
        "value",
        "delta_vs_behavior",
        "ci_low",
        "ci_high",
        "ci_contains_zero_uplift",
        "effective_sample_size",
        "weight_mean",
        "weight_p95",
        "weight_max",
        "n_rows",
    ]
    ablation_cols = [
        "policy_variant",
        "estimator",
        "value",
        "delta_vs_behavior",
        "ci_low",
        "ci_high",
        "delta_ci_low",
        "delta_ci_high",
        "ci_contains_zero_uplift",
        "effective_sample_size",
        "weight_mean",
        "weight_p95",
        "weight_max",
        "n_rows",
        "selection_objective",
        "dr_delta",
        "dr_delta_ci_low",
        "dr_delta_ci_high",
        "dr_ci_contains_zero",
    ]
    continent_ope_cols = [
        "target_policy_variant",
        "continent",
        "estimator",
        "value",
        "delta_vs_behavior",
        "ci_low",
        "ci_high",
        "delta_ci_low",
        "delta_ci_high",
        "ci_contains_zero_uplift",
        "effective_sample_size",
        "weight_mean",
        "weight_p95",
        "weight_max",
        "n_rows",
    ]
    continent_action_cols = [
        "continent",
        "action",
        "selected_city_count",
        "selected_share",
        "mean_q_advantage",
        "mean_policy_confidence",
        "mean_expected_delta_risk_h1",
        "mean_expected_delta_accel_h1",
    ]

    def _write_empty_outputs() -> None:
        pd.DataFrame(columns=city_cols).to_csv(city_path, index=False)
        pd.DataFrame(columns=action_cols).to_csv(action_path, index=False)
        pd.DataFrame(columns=state_cols).to_csv(state_path, index=False)
        pd.DataFrame(columns=ope_cols).to_csv(ope_path, index=False)
        pd.DataFrame(columns=ablation_cols).to_csv(ablation_path, index=False)
        pd.DataFrame(columns=continent_ope_cols).to_csv(continent_ope_path, index=False)
        pd.DataFrame(columns=continent_action_cols).to_csv(continent_action_path, index=False)

    required = {
        "city_id",
        "city_name",
        "country",
        "continent",
        "year",
        "stall_probability",
        "stall_risk_score",
        "acceleration_score",
        "kinetic_state",
        "trajectory_regime",
    }
    if not required.issubset(dyn.columns):
        _write_empty_outputs()
        return pd.DataFrame(columns=city_cols), {"status": "skipped", "reason": "missing_required_columns"}

    work = dyn.sort_values(["city_id", "year"]).copy()
    work["next_year"] = work.groupby("city_id")["year"].shift(-1)
    work["next_stall_probability"] = work.groupby("city_id")["stall_probability"].shift(-1)
    work["next_stall_risk_score"] = work.groupby("city_id")["stall_risk_score"].shift(-1)
    work["next_acceleration_score"] = work.groupby("city_id")["acceleration_score"].shift(-1)
    work["next_kinetic_state"] = work.groupby("city_id")["kinetic_state"].shift(-1)
    work["next_trajectory_regime"] = work.groupby("city_id")["trajectory_regime"].shift(-1)
    if "phase_label" in work.columns:
        work["next_phase_label"] = work.groupby("city_id")["phase_label"].shift(-1)
    else:
        work["phase_label"] = "insufficient_history"
        work["next_phase_label"] = "insufficient_history"

    trans = work[
        ((work["next_year"] - work["year"]) == 1) & (work["year"] <= (latest_year - 1))
    ].copy()
    if len(trans) < 120:
        _write_empty_outputs()
        return pd.DataFrame(columns=city_cols), {
            "status": "skipped",
            "reason": "insufficient_transition_rows",
            "n_transitions": int(len(trans)),
        }

    num_cols = [
        "stall_probability",
        "stall_risk_score",
        "acceleration_score",
        "next_stall_probability",
        "next_stall_risk_score",
        "next_acceleration_score",
    ]
    for col in num_cols:
        trans[col] = pd.to_numeric(trans[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    med_map = trans[num_cols].median(numeric_only=True).fillna(0.0).to_dict()
    for col in num_cols:
        trans[col] = trans[col].fillna(float(med_map.get(col, 0.0)))

    def _quantile_bounds(series: pd.Series) -> tuple[float, float]:
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 35.0, 65.0
        low, high = np.quantile(vals, [0.35, 0.65])
        low = float(low)
        high = float(high)
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            med = float(np.median(vals))
            sd = float(np.std(vals))
            sd = sd if (np.isfinite(sd) and sd > 1e-9) else 1.0
            low = med - 0.30 * sd
            high = med + 0.30 * sd
        return low, high

    def _bucketize(series: pd.Series, low: float, high: float) -> pd.Series:
        arr = pd.to_numeric(series, errors="coerce")
        out = pd.Series(
            np.where(arr <= low, "low", np.where(arr >= high, "high", "mid")),
            index=series.index,
            dtype=object,
        )
        out.loc[arr.isna()] = "mid"
        return out.astype(str)

    risk_low, risk_high = _quantile_bounds(trans["stall_risk_score"])
    accel_low, accel_high = _quantile_bounds(trans["acceleration_score"])
    trans["risk_bucket"] = _bucketize(trans["stall_risk_score"], risk_low, risk_high)
    trans["accel_bucket"] = _bucketize(trans["acceleration_score"], accel_low, accel_high)
    trans["next_risk_bucket"] = _bucketize(trans["next_stall_risk_score"], risk_low, risk_high)
    trans["next_accel_bucket"] = _bucketize(trans["next_acceleration_score"], accel_low, accel_high)

    for col in ["kinetic_state", "trajectory_regime", "next_kinetic_state", "next_trajectory_regime", "phase_label"]:
        if col not in trans.columns:
            trans[col] = "mixed_transition"
        trans[col] = trans[col].fillna("mixed_transition").astype(str)

    def _state_id(
        frame: pd.DataFrame,
        kinetic_col: str,
        regime_col: str,
        risk_col: str,
        accel_col: str,
    ) -> pd.Series:
        return (
            frame[kinetic_col].astype(str).str.replace(" ", "_", regex=False)
            + "|"
            + frame[regime_col].astype(str).str.replace(" ", "_", regex=False)
            + "|r_"
            + frame[risk_col].astype(str)
            + "|a_"
            + frame[accel_col].astype(str)
        )

    trans["state_id"] = _state_id(trans, "kinetic_state", "trajectory_regime", "risk_bucket", "accel_bucket")
    trans["next_state_id"] = _state_id(
        trans.assign(
            next_kinetic_state=trans["next_kinetic_state"].fillna(trans["kinetic_state"]),
            next_trajectory_regime=trans["next_trajectory_regime"].fillna(trans["trajectory_regime"]),
        ),
        "next_kinetic_state",
        "next_trajectory_regime",
        "next_risk_bucket",
        "next_accel_bucket",
    )

    actions = ["wait_observe", "resilience_upgrade", "innovation_pulse", "anti_overheating"]
    action_specs = {
        "wait_observe": {"risk_base": -0.010, "accel_base": 0.20, "unit_cost": 0.20},
        "resilience_upgrade": {"risk_base": -0.075, "accel_base": 2.60, "unit_cost": 1.10},
        "innovation_pulse": {"risk_base": -0.050, "accel_base": 5.80, "unit_cost": 1.70},
        "anti_overheating": {"risk_base": -0.068, "accel_base": -1.40, "unit_cost": 0.95},
    }

    risk_norm = np.clip(trans["stall_risk_score"].to_numpy(dtype=float) / 100.0, 0.0, 1.0)
    accel_norm = np.clip(trans["acceleration_score"].to_numpy(dtype=float) / 100.0, 0.0, 1.0)
    is_overheating = (trans["kinetic_state"] == "overheating").astype(float).to_numpy()
    is_cooling = (trans["kinetic_state"] == "deep_cooling").astype(float).to_numpy()
    is_decline = (
        trans["trajectory_regime"].str.contains("decline|stalling", case=False, regex=True).astype(float).to_numpy()
    )
    is_turbulent = (
        trans["phase_label"]
        .astype(str)
        .str.contains("turbulence|centrifugal_fragility|pre_break|accelerating_stress", case=False, regex=True)
        .astype(float)
        .to_numpy()
    )
    is_mature = (
        trans["trajectory_regime"]
        .str.contains("mature|takeoff|accelerator", case=False, regex=True)
        .astype(float)
        .to_numpy()
    )
    is_rebound = (
        trans["trajectory_regime"]
        .str.contains("rebound|recovery", case=False, regex=True)
        .astype(float)
        .to_numpy()
    )

    def _build_context_bonus(
        risk_arr: np.ndarray,
        accel_arr: np.ndarray,
        overheat_arr: np.ndarray,
        cooling_arr: np.ndarray,
        decline_arr: np.ndarray,
        turbulent_arr: np.ndarray,
        mature_arr: np.ndarray,
        rebound_arr: np.ndarray,
    ) -> dict[str, np.ndarray]:
        low_risk = (risk_arr <= 0.38).astype(float)
        high_risk = (risk_arr >= 0.62).astype(float)
        high_accel = (accel_arr >= 0.62).astype(float)
        low_accel = (accel_arr <= 0.38).astype(float)

        wait_score = (
            1.2 * low_risk
            + 0.7 * low_accel
            - 0.8 * high_risk
            - 0.7 * overheat_arr
            - 0.6 * turbulent_arr
            - 0.4 * decline_arr
        )
        resil_score = (
            1.6 * high_risk
            + 0.9 * decline_arr
            + 0.8 * cooling_arr
            + 0.5 * turbulent_arr
            - 0.4 * low_risk
        )
        innov_score = (
            1.5 * high_accel
            + 0.8 * low_risk
            + 0.7 * mature_arr
            + 0.5 * rebound_arr
            - 1.1 * overheat_arr
            - 0.7 * decline_arr
        )
        anti_overheat_score = (
            1.7 * overheat_arr
            + 0.9 * high_accel
            + 0.4 * turbulent_arr
            - 0.8 * low_accel
            - 0.3 * cooling_arr
        )
        return {
            "wait_observe": wait_score.astype(float),
            "resilience_upgrade": resil_score.astype(float),
            "innovation_pulse": innov_score.astype(float),
            "anti_overheating": anti_overheat_score.astype(float),
        }

    trans_context_bonus = _build_context_bonus(
        risk_norm,
        accel_norm,
        is_overheating,
        is_cooling,
        is_decline,
        is_turbulent,
        is_mature,
        is_rebound,
    )

    def _action_shift_vectors(action: str) -> tuple[np.ndarray, np.ndarray, float]:
        spec = action_specs[action]
        if action == "resilience_upgrade":
            scale = 0.85 + 0.90 * risk_norm + 0.35 * is_decline + 0.24 * is_cooling + 0.15 * is_turbulent
            accel_scale = 0.85 + 0.45 * (1.0 - risk_norm) + 0.15 * is_cooling
        elif action == "innovation_pulse":
            scale = 0.70 + 0.75 * (1.0 - risk_norm) + 0.25 * (accel_norm >= 0.55).astype(float) - 0.30 * is_overheating
            scale = scale - 0.14 * is_decline
            accel_scale = 0.95 + 0.35 * (1.0 - risk_norm) - 0.25 * is_overheating
        elif action == "anti_overheating":
            scale = 0.80 + 0.95 * is_overheating + 0.35 * risk_norm + 0.10 * is_turbulent
            accel_scale = 0.70 + 0.60 * is_overheating
        else:
            scale = 0.82 + 0.18 * (1.0 - risk_norm)
            accel_scale = np.full_like(risk_norm, 1.0)
        scale = np.clip(scale, 0.30, 2.40)
        accel_scale = np.clip(accel_scale, 0.25, 1.80)
        risk_shift = float(spec["risk_base"]) * scale
        accel_shift = float(spec["accel_base"]) * accel_scale
        return risk_shift, accel_shift, float(spec["unit_cost"])

    n = len(trans)
    n_actions = len(actions)
    reward_matrix = np.zeros((n, n_actions), dtype=float)
    trans_delta_risk_mat = np.zeros((n, n_actions), dtype=float)
    trans_delta_accel_mat = np.zeros((n, n_actions), dtype=float)
    context_bonus_weight = 5.0
    for ai, action in enumerate(actions):
        risk_shift, accel_shift, unit_cost = _action_shift_vectors(action)
        trans_delta_risk_mat[:, ai] = risk_shift
        trans_delta_accel_mat[:, ai] = accel_shift
        next_prob = np.clip(trans["next_stall_probability"].to_numpy(dtype=float) + risk_shift, 0.0, 1.0)
        next_accel = np.clip(trans["next_acceleration_score"].to_numpy(dtype=float) + accel_shift, 0.0, 100.0)
        reward = (
            -100.0 * next_prob
            + 0.22 * next_accel
            - 5.0 * unit_cost
            + context_bonus_weight * trans_context_bonus[action]
        )
        reward_matrix[:, ai] = reward

    states = sorted(set(trans["state_id"].tolist()) | set(trans["next_state_id"].tolist()))
    if len(states) == 0:
        _write_empty_outputs()
        return pd.DataFrame(columns=city_cols), {"status": "skipped", "reason": "empty_state_space"}

    state_to_idx = {s: i for i, s in enumerate(states)}
    state_idx = trans["state_id"].map(state_to_idx).to_numpy(dtype=int)
    next_idx = trans["next_state_id"].map(state_to_idx).to_numpy(dtype=int)
    n_states = len(states)
    counts = np.bincount(state_idx, minlength=n_states).astype(float)

    q = np.zeros((n_states, n_actions), dtype=float)
    for ai in range(n_actions):
        sums = np.bincount(state_idx, weights=reward_matrix[:, ai], minlength=n_states).astype(float)
        base = np.divide(sums, np.maximum(counts, 1.0))
        fallback = float(np.nanmean(reward_matrix[:, ai])) if n > 0 else 0.0
        base[counts <= 0.5] = fallback
        q[:, ai] = base

    bellman_errors: List[float] = []
    for _ in range(int(max(1, n_iterations))):
        v_next = np.max(q[next_idx, :], axis=1)
        targets = reward_matrix + float(gamma) * v_next.reshape(-1, 1)
        new_q = np.zeros_like(q)
        for ai in range(n_actions):
            sums = np.bincount(state_idx, weights=targets[:, ai], minlength=n_states).astype(float)
            vals = np.divide(sums, np.maximum(counts, 1.0))
            fallback = float(np.nanmean(targets[:, ai])) if n > 0 else 0.0
            vals[counts <= 0.5] = fallback
            new_q[:, ai] = vals
        err = float(np.nanmax(np.abs(new_q - q))) if new_q.size else 0.0
        bellman_errors.append(err)
        q = new_q

    state_values = np.max(q, axis=1)
    state_best_idx = np.argmax(q, axis=1)
    state_df = pd.DataFrame(
        {
            "state_id": states,
            "rl_state_value": state_values,
            "rl_best_action": [actions[int(i)] for i in state_best_idx],
        }
    )
    parts = state_df["state_id"].str.split(r"\|", expand=True)
    if parts.shape[1] >= 4:
        state_df["kinetic_state"] = parts[0].str.replace("_", " ", regex=False)
        state_df["trajectory_regime"] = parts[1].str.replace("_", " ", regex=False)
        state_df["risk_bucket"] = parts[2].str.replace("r_", "", regex=False)
        state_df["accel_bucket"] = parts[3].str.replace("a_", "", regex=False)
    else:
        state_df["kinetic_state"] = "mixed_transition"
        state_df["trajectory_regime"] = "mixed_transition"
        state_df["risk_bucket"] = "mid"
        state_df["accel_bucket"] = "mid"
    visit_map = trans["state_id"].value_counts().to_dict()
    state_df["visit_count"] = state_df["state_id"].map(lambda s: int(visit_map.get(s, 0)))
    state_df = state_df[state_cols].sort_values(["rl_state_value", "visit_count"], ascending=[False, False])
    state_df.to_csv(state_path, index=False)

    latest = work[work["year"] == latest_year].copy()
    if latest.empty:
        _write_empty_outputs()
        return pd.DataFrame(columns=city_cols), {
            "status": "skipped",
            "reason": "latest_rows_missing",
            "n_states": int(n_states),
            "n_transitions": int(n),
        }

    for col in ["stall_probability", "stall_risk_score", "acceleration_score"]:
        latest[col] = pd.to_numeric(latest[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    latest["stall_probability"] = latest["stall_probability"].fillna(float(trans["stall_probability"].median()))
    latest["stall_risk_score"] = latest["stall_risk_score"].fillna(float(trans["stall_risk_score"].median()))
    latest["acceleration_score"] = latest["acceleration_score"].fillna(float(trans["acceleration_score"].median()))
    if "forecast_stall_prob_h1" in latest.columns:
        latest["forecast_stall_prob_h1"] = pd.to_numeric(latest["forecast_stall_prob_h1"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        latest["forecast_stall_prob_h1"] = latest["forecast_stall_prob_h1"].fillna(latest["stall_probability"])
    else:
        latest["forecast_stall_prob_h1"] = latest["stall_probability"]
    if "phase_label" not in latest.columns:
        latest["phase_label"] = "insufficient_history"
    latest["phase_label"] = latest["phase_label"].fillna("insufficient_history").astype(str)
    latest["kinetic_state"] = latest["kinetic_state"].fillna("mixed_transition").astype(str)
    latest["trajectory_regime"] = latest["trajectory_regime"].fillna("mixed_transition").astype(str)

    latest["risk_bucket"] = _bucketize(latest["stall_risk_score"], risk_low, risk_high)
    latest["accel_bucket"] = _bucketize(latest["acceleration_score"], accel_low, accel_high)
    latest["rl_state_id"] = _state_id(latest, "kinetic_state", "trajectory_regime", "risk_bucket", "accel_bucket")

    default_q = np.nanmean(q, axis=0) if q.size else np.zeros(n_actions, dtype=float)
    q_all = np.tile(default_q.reshape(1, -1), (len(latest), 1))
    known_mask = latest["rl_state_id"].isin(state_to_idx)
    if known_mask.any():
        idx_known = latest.loc[known_mask, "rl_state_id"].map(state_to_idx).to_numpy(dtype=int)
        q_all[known_mask.to_numpy(), :] = q[idx_known, :]

    # Recompute action effects for latest states to estimate one-step policy impact.
    r_norm_l = np.clip(latest["stall_risk_score"].to_numpy(dtype=float) / 100.0, 0.0, 1.0)
    a_norm_l = np.clip(latest["acceleration_score"].to_numpy(dtype=float) / 100.0, 0.0, 1.0)
    over_l = (latest["kinetic_state"] == "overheating").astype(float).to_numpy()
    cool_l = (latest["kinetic_state"] == "deep_cooling").astype(float).to_numpy()
    decline_l = (
        latest["trajectory_regime"].str.contains("decline|stalling", case=False, regex=True).astype(float).to_numpy()
    )
    turb_l = (
        latest["phase_label"]
        .str.contains("turbulence|centrifugal_fragility|pre_break|accelerating_stress", case=False, regex=True)
        .astype(float)
        .to_numpy()
    )
    mature_l = (
        latest["trajectory_regime"]
        .str.contains("mature|takeoff|accelerator", case=False, regex=True)
        .astype(float)
        .to_numpy()
    )
    rebound_l = (
        latest["trajectory_regime"]
        .str.contains("rebound|recovery", case=False, regex=True)
        .astype(float)
        .to_numpy()
    )
    latest_context_bonus = _build_context_bonus(
        r_norm_l,
        a_norm_l,
        over_l,
        cool_l,
        decline_l,
        turb_l,
        mature_l,
        rebound_l,
    )
    q_all_policy = q_all.copy()
    policy_context_weight = 4.0
    for ai, action in enumerate(actions):
        q_all_policy[:, ai] += policy_context_weight * latest_context_bonus[action]

    def _latest_action_shift(action: str) -> tuple[np.ndarray, np.ndarray]:
        spec = action_specs[action]
        if action == "resilience_upgrade":
            scale = 0.85 + 0.90 * r_norm_l + 0.35 * decline_l + 0.24 * cool_l + 0.15 * turb_l
            accel_scale = 0.85 + 0.45 * (1.0 - r_norm_l) + 0.15 * cool_l
        elif action == "innovation_pulse":
            scale = 0.70 + 0.75 * (1.0 - r_norm_l) + 0.25 * (a_norm_l >= 0.55).astype(float) - 0.30 * over_l
            scale = scale - 0.14 * decline_l
            accel_scale = 0.95 + 0.35 * (1.0 - r_norm_l) - 0.25 * over_l
        elif action == "anti_overheating":
            scale = 0.80 + 0.95 * over_l + 0.35 * r_norm_l + 0.10 * turb_l
            accel_scale = 0.70 + 0.60 * over_l
        else:
            scale = 0.82 + 0.18 * (1.0 - r_norm_l)
            accel_scale = np.full_like(r_norm_l, 1.0)
        scale = np.clip(scale, 0.30, 2.40)
        accel_scale = np.clip(accel_scale, 0.25, 1.80)
        return float(spec["risk_base"]) * scale, float(spec["accel_base"]) * accel_scale

    base_risk_h1 = np.clip(latest["forecast_stall_prob_h1"].to_numpy(dtype=float), 0.0, 1.0)
    delta_risk_mat = np.zeros((len(latest), n_actions), dtype=float)
    delta_accel_mat = np.zeros((len(latest), n_actions), dtype=float)
    for ai, action in enumerate(actions):
        d_risk, d_accel = _latest_action_shift(action)
        delta_risk_mat[:, ai] = d_risk
        delta_accel_mat[:, ai] = d_accel

    ranked_idx = np.argsort(q_all_policy, axis=1)
    best_idx = ranked_idx[:, -1]
    if n_actions >= 2:
        second_idx = ranked_idx[:, -2]
    else:
        second_idx = best_idx.copy()

    selection_source = np.array(["direct_q"] * len(latest), dtype=object)
    action_to_idx = {a: i for i, a in enumerate(actions)}

    if "innovation_pulse" in action_to_idx:
        i_innov = action_to_idx["innovation_pulse"]
        innovate_mask = (
            (r_norm_l <= 0.52)
            & (a_norm_l >= 0.55)
            & (over_l < 0.5)
            & ((q_all_policy[np.arange(len(latest)), best_idx] - q_all_policy[:, i_innov]) <= 1.35)
        )
        if innovate_mask.any():
            best_idx[innovate_mask] = i_innov
            selection_source[innovate_mask] = "near_tie_innovation_rule"

    if "anti_overheating" in action_to_idx:
        i_anti = action_to_idx["anti_overheating"]
        anti_mask = (
            ((over_l > 0.5) | ((a_norm_l >= 0.70) & (r_norm_l >= 0.45)))
            & ((q_all_policy[np.arange(len(latest)), best_idx] - q_all_policy[:, i_anti]) <= 1.55)
        )
        if anti_mask.any():
            best_idx[anti_mask] = i_anti
            selection_source[anti_mask] = "near_tie_overheat_rule"

    if "wait_observe" in action_to_idx:
        i_wait = action_to_idx["wait_observe"]
        wait_mask = (
            (r_norm_l <= 0.33)
            & (a_norm_l <= 0.52)
            & (turb_l < 0.5)
            & ((q_all_policy[np.arange(len(latest)), best_idx] - q_all_policy[:, i_wait]) <= 1.10)
        )
        if wait_mask.any():
            best_idx[wait_mask] = i_wait
            selection_source[wait_mask] = "near_tie_wait_rule"

    if n_actions >= 2:
        runner_idx = np.where(best_idx == ranked_idx[:, -1], ranked_idx[:, -2], ranked_idx[:, -1])
    else:
        runner_idx = best_idx.copy()
    best_q = q_all_policy[np.arange(len(latest)), best_idx]
    second_q = q_all_policy[np.arange(len(latest)), runner_idx]
    q_adv = best_q - second_q
    temp = 7.5
    z = (q_all_policy - np.max(q_all_policy, axis=1, keepdims=True)) / temp
    z = np.clip(z, -50.0, 50.0)
    exp_z = np.exp(z)
    prob = exp_z / np.maximum(exp_z.sum(axis=1, keepdims=True), 1e-12)
    policy_conf = prob[np.arange(len(latest)), best_idx]

    chosen_delta_risk = delta_risk_mat[np.arange(len(latest)), best_idx]
    chosen_delta_accel = delta_accel_mat[np.arange(len(latest)), best_idx]
    chosen_risk_h1 = np.clip(base_risk_h1 + chosen_delta_risk, 0.0, 1.0)

    city = latest[
        [
            "city_id",
            "city_name",
            "country",
            "continent",
            "year",
            "kinetic_state",
            "trajectory_regime",
            "stall_risk_score",
            "acceleration_score",
            "rl_state_id",
        ]
    ].copy()
    city["rl_best_action"] = [actions[int(i)] for i in best_idx]
    city["rl_best_q_value"] = best_q
    city["rl_second_q_value"] = second_q
    city["rl_q_advantage"] = q_adv
    city["rl_policy_utility"] = best_q
    city["rl_selection_source"] = selection_source
    city["rl_policy_confidence"] = policy_conf
    city["rl_baseline_risk_h1"] = base_risk_h1
    city["rl_expected_risk_h1"] = chosen_risk_h1
    city["rl_expected_delta_risk_h1"] = chosen_risk_h1 - base_risk_h1
    city["rl_expected_delta_accel_h1"] = chosen_delta_accel

    for ai, action in enumerate(actions):
        city[f"rl_q_{action}"] = q_all[:, ai]
        city[f"rl_delta_risk_h1_{action}"] = delta_risk_mat[:, ai]
        city[f"rl_delta_accel_h1_{action}"] = delta_accel_mat[:, ai]

    city = city.sort_values(["rl_q_advantage", "rl_best_q_value"], ascending=[False, False])
    city.to_csv(city_path, index=False)

    grouped = (
        city.groupby("rl_best_action", as_index=False)
        .agg(
            selected_city_count=("city_id", "count"),
            mean_best_q_value=("rl_best_q_value", "mean"),
            mean_q_advantage=("rl_q_advantage", "mean"),
            mean_policy_confidence=("rl_policy_confidence", "mean"),
            mean_expected_delta_risk_h1=("rl_expected_delta_risk_h1", "mean"),
            mean_expected_delta_accel_h1=("rl_expected_delta_accel_h1", "mean"),
            selected_by_rule_count=("rl_selection_source", lambda s: int(np.sum(np.asarray(s) != "direct_q"))),
        )
        .rename(columns={"rl_best_action": "action"})
    )
    base_action_df = pd.DataFrame({"action": actions})
    action_df = base_action_df.merge(grouped, on="action", how="left")
    action_df["selected_city_count"] = action_df["selected_city_count"].fillna(0).astype(int)
    action_df["selected_by_rule_count"] = action_df["selected_by_rule_count"].fillna(0).astype(int)
    denom = float(max(len(city), 1))
    action_df["selected_share"] = action_df["selected_city_count"] / denom
    for ai, action in enumerate(actions):
        action_df.loc[action_df["action"] == action, "mean_q_if_forced"] = float(np.nanmean(q_all_policy[:, ai]))
        action_df.loc[action_df["action"] == action, "mean_delta_risk_if_forced"] = float(
            np.nanmean(delta_risk_mat[:, ai])
        )
        action_df.loc[action_df["action"] == action, "mean_delta_accel_if_forced"] = float(
            np.nanmean(delta_accel_mat[:, ai])
        )
    for col in action_cols:
        if col not in action_df.columns:
            action_df[col] = np.nan
    action_df = action_df[action_cols].sort_values(
        ["selected_share", "mean_q_if_forced"],
        ascending=[False, False],
    )
    action_df.to_csv(action_path, index=False)

    continent_rows: List[Dict[str, Any]] = []
    if "continent" in city.columns and not city.empty:
        city_cont = city.copy()
        city_cont["continent"] = city_cont["continent"].fillna("unknown").astype(str)
        for cont in sorted(city_cont["continent"].unique().tolist()):
            sub = city_cont[city_cont["continent"] == cont].copy()
            if sub.empty:
                continue
            n_sub = float(max(len(sub), 1))
            grouped_sub = (
                sub.groupby("rl_best_action", as_index=False)
                .agg(
                    selected_city_count=("city_id", "count"),
                    mean_q_advantage=("rl_q_advantage", "mean"),
                    mean_policy_confidence=("rl_policy_confidence", "mean"),
                    mean_expected_delta_risk_h1=("rl_expected_delta_risk_h1", "mean"),
                    mean_expected_delta_accel_h1=("rl_expected_delta_accel_h1", "mean"),
                )
                .rename(columns={"rl_best_action": "action"})
            )
            for action in actions:
                row = grouped_sub[grouped_sub["action"] == action]
                if row.empty:
                    continent_rows.append(
                        {
                            "continent": cont,
                            "action": action,
                            "selected_city_count": 0,
                            "selected_share": 0.0,
                            "mean_q_advantage": np.nan,
                            "mean_policy_confidence": np.nan,
                            "mean_expected_delta_risk_h1": np.nan,
                            "mean_expected_delta_accel_h1": np.nan,
                        }
                    )
                else:
                    r0 = row.iloc[0]
                    n_action = int(r0.get("selected_city_count", 0))
                    continent_rows.append(
                        {
                            "continent": cont,
                            "action": action,
                            "selected_city_count": n_action,
                            "selected_share": float(n_action / n_sub),
                            "mean_q_advantage": float(r0.get("mean_q_advantage"))
                            if pd.notna(r0.get("mean_q_advantage"))
                            else np.nan,
                            "mean_policy_confidence": float(r0.get("mean_policy_confidence"))
                            if pd.notna(r0.get("mean_policy_confidence"))
                            else np.nan,
                            "mean_expected_delta_risk_h1": float(r0.get("mean_expected_delta_risk_h1"))
                            if pd.notna(r0.get("mean_expected_delta_risk_h1"))
                            else np.nan,
                            "mean_expected_delta_accel_h1": float(r0.get("mean_expected_delta_accel_h1"))
                            if pd.notna(r0.get("mean_expected_delta_accel_h1"))
                            else np.nan,
                        }
                    )
    continent_action_df = pd.DataFrame(continent_rows, columns=continent_action_cols)
    continent_action_df.to_csv(continent_action_path, index=False)

    shares = action_df["selected_share"].to_numpy(dtype=float)
    shares = shares[np.isfinite(shares) & (shares > 1e-12)]
    policy_entropy = float(-np.sum(shares * np.log(shares))) if shares.size else 0.0
    dominant = action_df.sort_values("selected_share", ascending=False).iloc[0] if not action_df.empty else None

    # Offline policy evaluation (OPE): IPS/SNIPS/DR against an inferred behavior policy.
    q_trans_policy = q[state_idx, :].copy()
    for ai, action in enumerate(actions):
        q_trans_policy[:, ai] += policy_context_weight * trans_context_bonus[action]

    q_ope = np.zeros((n_states, n_actions), dtype=float)
    for ai in range(n_actions):
        sums = np.bincount(state_idx, weights=reward_matrix[:, ai], minlength=n_states).astype(float)
        vals = np.divide(sums, np.maximum(counts, 1.0))
        fallback = float(np.nanmean(reward_matrix[:, ai])) if n > 0 else 0.0
        vals[counts <= 0.5] = fallback
        q_ope[:, ai] = vals

    def _softmax_rows(arr: np.ndarray, tau: float) -> np.ndarray:
        scaled = arr / max(float(tau), 1e-6)
        scaled = scaled - np.max(scaled, axis=1, keepdims=True)
        scaled = np.clip(scaled, -60.0, 60.0)
        exp_scaled = np.exp(scaled)
        return exp_scaled / np.maximum(exp_scaled.sum(axis=1, keepdims=True), 1e-12)

    policy_temp = 7.5
    q_trans_rule = q_trans_policy.copy()
    i_innov = actions.index("innovation_pulse") if "innovation_pulse" in actions else None
    i_anti = actions.index("anti_overheating") if "anti_overheating" in actions else None
    i_wait = actions.index("wait_observe") if "wait_observe" in actions else None
    if i_innov is not None:
        mask = (risk_norm <= 0.52) & (accel_norm >= 0.55) & (is_overheating < 0.5)
        q_trans_rule[mask, i_innov] += 1.20
    if i_anti is not None:
        mask = (is_overheating > 0.5) | ((accel_norm >= 0.70) & (risk_norm >= 0.45))
        q_trans_rule[mask, i_anti] += 1.30
    if i_wait is not None:
        mask = (risk_norm <= 0.33) & (accel_norm <= 0.52) & (is_turbulent < 0.5)
        q_trans_rule[mask, i_wait] += 0.85
    q_trans_safe = q_trans_rule.copy()
    high_risk_mask = risk_norm >= 0.65
    very_high_risk_mask = risk_norm >= 0.78
    for ai, action in enumerate(actions):
        weak_risk_cut = trans_delta_risk_mat[:, ai] > -0.025
        very_weak_risk_cut = trans_delta_risk_mat[:, ai] > -0.050
        penalty = np.zeros(n, dtype=float)
        penalty += 1.80 * (high_risk_mask & weak_risk_cut).astype(float)
        penalty += 1.20 * (very_high_risk_mask & very_weak_risk_cut).astype(float)
        if action == "innovation_pulse":
            penalty += 0.90 * ((is_overheating > 0.5) | very_high_risk_mask).astype(float)
        if action == "wait_observe":
            penalty += 0.75 * high_risk_mask.astype(float)
        q_trans_safe[:, ai] -= penalty

    obs_delta_risk = (
        trans["next_stall_probability"].to_numpy(dtype=float) - trans["stall_probability"].to_numpy(dtype=float)
    )
    obs_delta_accel = (
        trans["next_acceleration_score"].to_numpy(dtype=float) - trans["acceleration_score"].to_numpy(dtype=float)
    )
    risk_scale = float(np.nanstd(obs_delta_risk))
    accel_scale = float(np.nanstd(obs_delta_accel))
    risk_scale = risk_scale if (np.isfinite(risk_scale) and risk_scale > 1e-6) else 0.06
    accel_scale = accel_scale if (np.isfinite(accel_scale) and accel_scale > 1e-6) else 4.0

    behavior_prior_penalty = {
        "wait_observe": 0.08,
        "resilience_upgrade": 0.28,
        "innovation_pulse": 0.42,
        "anti_overheating": 0.24,
    }
    behavior_logits = np.zeros((n, n_actions), dtype=float)
    for ai, action in enumerate(actions):
        dist = (
            ((obs_delta_risk - trans_delta_risk_mat[:, ai]) / risk_scale) ** 2
            + ((obs_delta_accel - trans_delta_accel_mat[:, ai]) / accel_scale) ** 2
            + float(behavior_prior_penalty.get(action, 0.0))
        )
        behavior_logits[:, ai] = -dist

    behavior_temp = 2.4
    behavior_pi = _softmax_rows(behavior_logits, tau=behavior_temp)
    logged_action_idx = np.argmax(behavior_logits, axis=1)
    logged_prob = behavior_pi[np.arange(n), logged_action_idx]

    logged_cost = np.array([float(action_specs[actions[int(i)]]["unit_cost"]) for i in logged_action_idx], dtype=float)
    observed_reward = (
        -100.0 * trans["next_stall_probability"].to_numpy(dtype=float)
        + 0.22 * trans["next_acceleration_score"].to_numpy(dtype=float)
        - 5.0 * logged_cost
    )
    observed_reward = np.where(np.isfinite(observed_reward), observed_reward, np.nanmean(observed_reward))

    q_s = q_ope[state_idx, :]
    q_logged = q_s[np.arange(n), logged_action_idx]
    behavior_value = float(np.mean(observed_reward))

    def _ope_for_target_pi(
        pi_target: np.ndarray,
        draws: int = 350,
        row_idx: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if row_idx is None:
            idx = np.arange(n, dtype=int)
        else:
            idx = np.asarray(row_idx, dtype=int)
        idx = idx[(idx >= 0) & (idx < n)]
        if idx.size == 0:
            return {
                "estimators": {"behavior": np.nan, "ips": np.nan, "snips": np.nan, "dr": np.nan},
                "rows": [],
                "uplift_ci": {},
                "effective_sample_size": 0.0,
                "weight_mean": np.nan,
                "weight_p95": np.nan,
                "weight_max": np.nan,
            }

        pi_sub = pi_target[idx, :]
        logged_idx_sub = logged_action_idx[idx]
        logged_prob_sub = logged_prob[idx]
        obs_reward_sub = observed_reward[idx]
        q_logged_sub = q_logged[idx]
        q_s_sub = q_s[idx, :]
        n_eff = int(len(idx))
        behavior_value_local = float(np.mean(obs_reward_sub))

        target_prob_on_logged = pi_sub[np.arange(n_eff), logged_idx_sub]
        w_raw = target_prob_on_logged / np.maximum(logged_prob_sub, 1e-6)
        ww = np.clip(w_raw, 0.0, 20.0)
        ess_local = float((np.sum(ww) ** 2) / np.maximum(np.sum(ww**2), 1e-12))
        vv = np.sum(pi_sub * q_s_sub, axis=1)

        ips_local = float(np.mean(ww * obs_reward_sub))
        snips_denom_local = float(np.sum(ww))
        snips_local = (
            float(np.sum(ww * obs_reward_sub) / snips_denom_local) if snips_denom_local > 1e-12 else float(ips_local)
        )
        dr_local = float(np.mean(vv + ww * (obs_reward_sub - q_logged_sub)))

        rng = np.random.default_rng(2047)
        boot_behavior = []
        boot_ips = []
        boot_snips = []
        boot_dr = []
        for _ in range(int(max(120, draws))):
            boot_idx = rng.integers(0, n_eff, size=n_eff)
            rr = obs_reward_sub[boot_idx]
            w_b = ww[boot_idx]
            q_b = q_logged_sub[boot_idx]
            v_b = vv[boot_idx]
            boot_behavior.append(float(np.mean(rr)))
            boot_ips.append(float(np.mean(w_b * rr)))
            denom_b = float(np.sum(w_b))
            boot_snips.append(float(np.sum(w_b * rr) / denom_b) if denom_b > 1e-12 else float(np.mean(w_b * rr)))
            boot_dr.append(float(np.mean(v_b + w_b * (rr - q_b))))

        arr_map = {
            "behavior": np.array(boot_behavior, dtype=float),
            "ips": np.array(boot_ips, dtype=float),
            "snips": np.array(boot_snips, dtype=float),
            "dr": np.array(boot_dr, dtype=float),
        }
        est_local = {
            "behavior": behavior_value_local,
            "ips": ips_local,
            "snips": snips_local,
            "dr": dr_local,
        }
        uplift_ci_local: Dict[str, Dict[str, float | bool]] = {}
        rows_local: List[Dict[str, Any]] = []
        for est, val in est_local.items():
            arr = arr_map.get(est, np.array([], dtype=float))
            low = float(np.quantile(arr, 0.025)) if arr.size else None
            high = float(np.quantile(arr, 0.975)) if arr.size else None
            if est == "behavior":
                delta = 0.0
                delta_low = None
                delta_high = None
                ci_zero = None
            else:
                delta = float(val - behavior_value_local)
                beh = arr_map.get("behavior", np.array([], dtype=float))
                if arr.size and beh.size and len(arr) == len(beh):
                    delta_arr = arr - beh
                    delta_low = float(np.quantile(delta_arr, 0.025))
                    delta_high = float(np.quantile(delta_arr, 0.975))
                    ci_zero = bool(delta_low <= 0.0 <= delta_high)
                else:
                    delta_low = None
                    delta_high = None
                    ci_zero = None
                uplift_ci_local[est] = {
                    "delta_ci_low": delta_low,
                    "delta_ci_high": delta_high,
                    "ci_contains_zero_uplift": ci_zero,
                }
            rows_local.append(
                {
                    "estimator": est,
                    "value": float(val),
                    "delta_vs_behavior": float(delta),
                    "ci_low": low,
                    "ci_high": high,
                    "ci_contains_zero_uplift": ci_zero,
                    "effective_sample_size": float(ess_local),
                    "weight_mean": float(np.mean(ww)),
                    "weight_p95": float(np.quantile(ww, 0.95)),
                    "weight_max": float(np.max(ww)),
                    "n_rows": int(n_eff),
                }
            )
        return {
            "estimators": est_local,
            "rows": rows_local,
            "uplift_ci": uplift_ci_local,
            "effective_sample_size": ess_local,
            "weight_mean": float(np.mean(ww)),
            "weight_p95": float(np.quantile(ww, 0.95)),
            "weight_max": float(np.max(ww)),
        }

    policy_variants = {
        "base_q": _softmax_rows(q[state_idx, :], tau=policy_temp),
        "context_q": _softmax_rows(q_trans_policy, tau=policy_temp),
        "context_q_rule": _softmax_rows(q_trans_rule, tau=policy_temp),
        "safe_context_q_rule": _softmax_rows(q_trans_safe, tau=policy_temp),
    }
    variant_eval = {name: _ope_for_target_pi(pi_target) for name, pi_target in policy_variants.items()}
    variant_rank_rows: List[Dict[str, Any]] = []
    ess_floor = float(max(120.0, 0.16 * float(n)))
    for name, meta in variant_eval.items():
        est = meta.get("estimators", {})
        upl = meta.get("uplift_ci", {})
        dr_val = est.get("dr")
        beh_val = est.get("behavior")
        dr_delta = (
            float(dr_val) - float(beh_val)
            if (dr_val is not None and beh_val is not None and np.isfinite(float(dr_val)) and np.isfinite(float(beh_val)))
            else None
        )
        dr_ci = upl.get("dr", {}) if isinstance(upl, dict) else {}
        dr_ci_low = dr_ci.get("delta_ci_low")
        dr_ci_high = dr_ci.get("delta_ci_high")
        ci_zero = dr_ci.get("ci_contains_zero_uplift")
        ess_val = float(meta.get("effective_sample_size", np.nan))
        eff_penalty = (
            -0.05 * max(0.0, (ess_floor - ess_val) / max(ess_floor, 1.0)) if np.isfinite(ess_val) else -0.05
        )
        if dr_ci_low is not None and np.isfinite(float(dr_ci_low)):
            selection_objective = float(dr_ci_low)
        elif dr_delta is not None and np.isfinite(float(dr_delta)):
            selection_objective = float(dr_delta)
        else:
            selection_objective = -np.inf
        if bool(ci_zero):
            selection_objective -= 0.03
        selection_objective += eff_penalty
        variant_rank_rows.append(
            {
                "policy_variant": str(name),
                "selection_objective": float(selection_objective),
                "dr_delta": float(dr_delta) if dr_delta is not None else np.nan,
                "dr_delta_ci_low": float(dr_ci_low) if dr_ci_low is not None and np.isfinite(float(dr_ci_low)) else np.nan,
                "dr_delta_ci_high": float(dr_ci_high) if dr_ci_high is not None and np.isfinite(float(dr_ci_high)) else np.nan,
                "dr_ci_contains_zero_uplift": bool(ci_zero) if ci_zero is not None else None,
                "effective_sample_size": ess_val if np.isfinite(ess_val) else np.nan,
            }
        )
    rank_df = pd.DataFrame(variant_rank_rows)
    if not rank_df.empty:
        rank_df = rank_df.sort_values(
            ["selection_objective", "dr_delta", "effective_sample_size"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    target_name = str(rank_df.iloc[0]["policy_variant"]) if not rank_df.empty else "context_q_rule"
    target_pi = policy_variants[target_name]
    greedy_source = {
        "base_q": q[state_idx, :],
        "context_q": q_trans_policy,
        "context_q_rule": q_trans_rule,
        "safe_context_q_rule": q_trans_safe,
    }
    target_greedy_idx = np.argmax(greedy_source.get(target_name, q_trans_rule), axis=1)
    target_eval = variant_eval[target_name]
    est_map = {str(k): float(v) for k, v in target_eval["estimators"].items()}
    ope_rows: List[Dict[str, Any]] = list(target_eval["rows"])
    uplift_ci_by_estimator: Dict[str, Dict[str, float | bool]] = dict(target_eval["uplift_ci"])
    pd.DataFrame(ope_rows, columns=ope_cols).to_csv(ope_path, index=False)

    ablation_rows: List[Dict[str, Any]] = []
    for variant_name, meta in variant_eval.items():
        rows_variant = meta.get("rows", [])
        uplift_ci_variant = meta.get("uplift_ci", {})
        rank_row = rank_df[rank_df["policy_variant"].astype(str) == str(variant_name)] if not rank_df.empty else pd.DataFrame()
        sel_obj = float(rank_row.iloc[0]["selection_objective"]) if not rank_row.empty else np.nan
        dr_delta = float(rank_row.iloc[0]["dr_delta"]) if not rank_row.empty else np.nan
        dr_ci_low = float(rank_row.iloc[0]["dr_delta_ci_low"]) if not rank_row.empty else np.nan
        dr_ci_high = float(rank_row.iloc[0]["dr_delta_ci_high"]) if not rank_row.empty else np.nan
        dr_ci_zero = (
            bool(rank_row.iloc[0]["dr_ci_contains_zero_uplift"])
            if (not rank_row.empty and pd.notna(rank_row.iloc[0]["dr_ci_contains_zero_uplift"]))
            else None
        )
        for rr in rows_variant:
            est = str(rr.get("estimator"))
            ci_meta = uplift_ci_variant.get(est, {})
            ablation_rows.append(
                {
                    "policy_variant": variant_name,
                    "estimator": est,
                    "value": rr.get("value"),
                    "delta_vs_behavior": rr.get("delta_vs_behavior"),
                    "ci_low": rr.get("ci_low"),
                    "ci_high": rr.get("ci_high"),
                    "delta_ci_low": ci_meta.get("delta_ci_low"),
                    "delta_ci_high": ci_meta.get("delta_ci_high"),
                    "ci_contains_zero_uplift": ci_meta.get("ci_contains_zero_uplift"),
                    "effective_sample_size": rr.get("effective_sample_size"),
                    "weight_mean": rr.get("weight_mean"),
                    "weight_p95": rr.get("weight_p95"),
                    "weight_max": rr.get("weight_max"),
                    "n_rows": rr.get("n_rows"),
                    "selection_objective": sel_obj,
                    "dr_delta": dr_delta,
                    "dr_delta_ci_low": dr_ci_low,
                    "dr_delta_ci_high": dr_ci_high,
                    "dr_ci_contains_zero": dr_ci_zero,
                }
            )
    pd.DataFrame(ablation_rows, columns=ablation_cols).to_csv(ablation_path, index=False)

    continent_ope_rows: List[Dict[str, Any]] = []
    cont_series = trans["continent"].fillna("unknown").astype(str)
    for cont in sorted(cont_series.unique().tolist()):
        idx_cont = np.where(cont_series.to_numpy() == cont)[0]
        if len(idx_cont) < 120:
            continue
        cont_meta = _ope_for_target_pi(target_pi, row_idx=idx_cont)
        cont_upl = cont_meta.get("uplift_ci", {})
        for rr in cont_meta.get("rows", []):
            est_name = str(rr.get("estimator"))
            ci_meta = cont_upl.get(est_name, {})
            continent_ope_rows.append(
                {
                    "target_policy_variant": target_name,
                    "continent": cont,
                    "estimator": est_name,
                    "value": rr.get("value"),
                    "delta_vs_behavior": rr.get("delta_vs_behavior"),
                    "ci_low": rr.get("ci_low"),
                    "ci_high": rr.get("ci_high"),
                    "delta_ci_low": ci_meta.get("delta_ci_low"),
                    "delta_ci_high": ci_meta.get("delta_ci_high"),
                    "ci_contains_zero_uplift": ci_meta.get("ci_contains_zero_uplift"),
                    "effective_sample_size": rr.get("effective_sample_size"),
                    "weight_mean": rr.get("weight_mean"),
                    "weight_p95": rr.get("weight_p95"),
                    "weight_max": rr.get("weight_max"),
                    "n_rows": rr.get("n_rows"),
                }
            )
    continent_ope_df = pd.DataFrame(continent_ope_rows, columns=continent_ope_cols)
    continent_ope_df.to_csv(continent_ope_path, index=False)
    dr_cont = continent_ope_df[continent_ope_df["estimator"].astype(str) == "dr"].copy()
    if not dr_cont.empty:
        dr_cont["delta_vs_behavior"] = pd.to_numeric(dr_cont["delta_vs_behavior"], errors="coerce")
        dr_cont["delta_ci_low"] = pd.to_numeric(dr_cont["delta_ci_low"], errors="coerce")
        dr_cont = dr_cont.dropna(subset=["delta_vs_behavior"])
        continent_dr_delta_mean = float(dr_cont["delta_vs_behavior"].mean()) if not dr_cont.empty else None
        continent_dr_delta_min = float(dr_cont["delta_vs_behavior"].min()) if not dr_cont.empty else None
        continent_dr_share_positive = float(np.mean(dr_cont["delta_vs_behavior"].to_numpy(dtype=float) > 0.0))
        continent_dr_share_ci_low_gt_zero = (
            float(np.mean(dr_cont["delta_ci_low"].to_numpy(dtype=float) > 0.0))
            if dr_cont["delta_ci_low"].notna().any()
            else None
        )
        continent_top = (
            dr_cont.sort_values("delta_vs_behavior", ascending=False)
            .head(3)[["continent", "delta_vs_behavior", "delta_ci_low", "delta_ci_high"]]
            .to_dict(orient="records")
        )
    else:
        continent_dr_delta_mean = None
        continent_dr_delta_min = None
        continent_dr_share_positive = None
        continent_dr_share_ci_low_gt_zero = None
        continent_top = []

    behavior_dist = pd.Series(logged_action_idx).map(lambda i: actions[int(i)]).value_counts(normalize=True).round(4).to_dict()
    target_dist_trans = (
        pd.Series(target_greedy_idx).map(lambda i: actions[int(i)]).value_counts(normalize=True).round(4).to_dict()
    )
    ope_summary = {
        "status": "ok",
        "target_policy_variant": target_name,
        "target_policy_selection_rule": "max_dr_uplift_ci_low_penalized",
        "target_policy_selection_table": rank_df.to_dict(orient="records"),
        "estimators": {str(k): float(v) for k, v in est_map.items()},
        "delta_vs_behavior": {k: float(v - behavior_value) for k, v in est_map.items() if k != "behavior"},
        "uplift_ci": uplift_ci_by_estimator,
        "effective_sample_size": float(target_eval["effective_sample_size"]),
        "n_rows": int(n),
        "weight_mean": float(target_eval["weight_mean"]),
        "weight_p95": float(target_eval["weight_p95"]),
        "weight_max": float(target_eval["weight_max"]),
        "target_support_overlap_share": float(np.mean(logged_prob >= 0.05)),
        "behavior_action_distribution": {str(k): float(v) for k, v in behavior_dist.items()},
        "target_action_distribution_transitions": {str(k): float(v) for k, v in target_dist_trans.items()},
        "variant_delta_dr_vs_behavior": {
            name: float((meta.get("estimators") or {}).get("dr", behavior_value) - behavior_value)
            for name, meta in variant_eval.items()
        },
        "continent_ope_status": "ok" if not continent_ope_df.empty else "skipped",
        "continent_ope_n": int(dr_cont["continent"].nunique()) if not dr_cont.empty else 0,
        "continent_dr_delta_mean": continent_dr_delta_mean,
        "continent_dr_delta_min": continent_dr_delta_min,
        "continent_dr_share_positive": continent_dr_share_positive,
        "continent_dr_share_ci_low_gt_zero": continent_dr_share_ci_low_gt_zero,
        "continent_top": continent_top,
    }

    summary = {
        "status": "ok",
        "n_states": int(n_states),
        "n_actions": int(n_actions),
        "n_transitions": int(n),
        "n_latest_cities": int(len(city)),
        "gamma": float(gamma),
        "n_iterations": int(max(1, n_iterations)),
        "risk_threshold_low": float(risk_low),
        "risk_threshold_high": float(risk_high),
        "accel_threshold_low": float(accel_low),
        "accel_threshold_high": float(accel_high),
        "bellman_error_start": float(bellman_errors[0]) if bellman_errors else None,
        "bellman_error_end": float(bellman_errors[-1]) if bellman_errors else None,
        "policy_entropy": policy_entropy,
        "selection_source_distribution": {
            str(k): float(v) for k, v in city["rl_selection_source"].value_counts(normalize=True).round(4).to_dict().items()
        },
        "rule_selected_share": float(np.mean(city["rl_selection_source"].astype(str) != "direct_q")),
        "unseen_state_share_latest": float(1.0 - np.mean(known_mask.to_numpy(dtype=float))),
        "mean_expected_delta_risk_h1": float(city["rl_expected_delta_risk_h1"].mean()),
        "mean_expected_delta_accel_h1": float(city["rl_expected_delta_accel_h1"].mean()),
        "selected_action_distribution": {
            str(r.action): float(r.selected_share) for r in action_df.itertuples(index=False)
        },
        "dominant_action": str(dominant["action"]) if dominant is not None else None,
        "dominant_action_share": float(dominant["selected_share"]) if dominant is not None else None,
        "action_summary": action_df.to_dict(orient="records"),
        "continent_action_summary": continent_action_df.to_dict(orient="records"),
        "target_policy_variant": target_name,
        "target_policy_selection_table": rank_df.to_dict(orient="records"),
        "offline_policy_evaluation": ope_summary,
    }
    return city, summary


def _estimate_multi_horizon_forecast(
    dyn: pd.DataFrame,
    latest_year: int,
    horizons: tuple[int, ...] = (1, 2, 3),
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = dyn.sort_values(["city_id", "year"]).copy()
    if "trajectory_regime" not in out.columns:
        return pd.DataFrame(), {"status": "skipped", "reason": "trajectory_regime_missing"}
    if "stall_probability" not in out.columns:
        return pd.DataFrame(), {"status": "skipped", "reason": "stall_probability_missing"}

    horizons = tuple(sorted({int(h) for h in horizons if int(h) >= 1}))
    if not horizons:
        return pd.DataFrame(), {"status": "skipped", "reason": "no_valid_horizons"}

    regimes = sorted(out["trajectory_regime"].fillna("mixed_transition").astype(str).unique().tolist())
    if len(regimes) == 0:
        return pd.DataFrame(), {"status": "skipped", "reason": "no_regimes"}
    regime_to_idx = {r: i for i, r in enumerate(regimes)}

    seq = out[["city_id", "year", "trajectory_regime"]].copy()
    seq["trajectory_regime"] = seq["trajectory_regime"].fillna("mixed_transition").astype(str)
    seq["next_regime"] = seq.groupby("city_id")["trajectory_regime"].shift(-1)
    seq["next_year"] = seq.groupby("city_id")["year"].shift(-1)
    pairs = seq[(seq["next_regime"].notna()) & ((seq["next_year"] - seq["year"]) == 1)].copy()

    trans = np.ones((len(regimes), len(regimes)), dtype=float)
    if not pairs.empty:
        ctab = pairs.groupby(["trajectory_regime", "next_regime"], as_index=False).size()
        for row in ctab.itertuples(index=False):
            i = regime_to_idx.get(str(row.trajectory_regime))
            j = regime_to_idx.get(str(row.next_regime))
            if i is None or j is None:
                continue
            trans[i, j] += float(row.size)
    trans = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1e-12)

    global_risk = float(out["stall_next"].dropna().mean()) if out["stall_next"].notna().any() else 0.5
    regime_risk_map = (
        out.dropna(subset=["stall_next"])
        .groupby("trajectory_regime")["stall_next"]
        .mean()
        .to_dict()
    )
    risk_vec = np.array([float(regime_risk_map.get(r, global_risk)) for r in regimes], dtype=float)

    idx_arr = out["trajectory_regime"].fillna("mixed_transition").astype(str).map(regime_to_idx).to_numpy(dtype=int)
    for h in horizons:
        trans_h = np.linalg.matrix_power(trans, h)
        dist = trans_h[idx_arr, :]
        pred = np.clip(dist @ risk_vec, 0.0, 1.0)
        if h == 1:
            pred = np.clip(0.60 * pred + 0.40 * out["stall_probability"].astype(float).to_numpy(), 0.0, 1.0)
        safe_dist = np.clip(dist, 1e-12, 1.0)
        entropy = -np.sum(dist * np.log(safe_dist), axis=1)
        dom_idx = np.argmax(dist, axis=1)
        out[f"forecast_stall_prob_h{h}"] = pred
        out[f"forecast_stall_risk_h{h}"] = np.clip(100.0 * pred, 0.0, 100.0)
        out[f"forecast_regime_entropy_h{h}"] = entropy
        out[f"forecast_dominant_regime_h{h}"] = [regimes[int(i)] for i in dom_idx]
        out[f"forecast_risk_delta_h{h}"] = 100.0 * (pred - out["stall_probability"].astype(float))

    out["stall_h1"] = out["stall_next"]
    for h in horizons:
        if h <= 1:
            continue
        out[f"stall_h{h}"] = out.groupby("city_id")["stall_next"].shift(-(h - 1))

    supervised_blend: Dict[str, Any] = {}
    sup_features = [
        "stall_probability",
        "acceleration_score",
        "regime_forward_risk",
        "regime_self_transition_prob",
        "regime_transition_entropy",
        "regime_run_length_log",
        "regime_switch_rate_3y",
        "regime_switch_rate_5y",
    ]
    sup_features = [c for c in sup_features if c in out.columns]
    if sup_features:
        _fill_feature_medians(out, sup_features)
        x_all = out[sup_features].to_numpy(dtype=float)
        for h in horizons:
            if h <= 1:
                continue
            target_col = f"stall_h{h}"
            train = out[(out[target_col].notna()) & (out["year"] <= (latest_year - h))].copy()
            if len(train) < 120 or train[target_col].nunique() < 2:
                supervised_blend[f"h{h}"] = {"status": "skipped", "reason": "insufficient_training_rows"}
                continue

            x_train = train[sup_features].to_numpy(dtype=float)
            y_train = train[target_col].astype(int).to_numpy()
            clf = _build_classifier("logit", random_state=800 + h)
            _fit_classifier_with_weights(clf, "logit", x_train, y_train, sample_weight=None)
            p_sup = np.clip(clf.predict_proba(x_all)[:, 1], 0.0, 1.0)

            base_col = f"forecast_stall_prob_h{h}"
            base = np.clip(out[base_col].astype(float).to_numpy(), 0.0, 1.0)
            w_sup = 0.50 if h == 2 else 0.58
            blend = np.clip((1.0 - w_sup) * base + w_sup * p_sup, 0.0, 1.0)
            out[base_col] = blend
            out[f"forecast_stall_risk_h{h}"] = np.clip(100.0 * blend, 0.0, 100.0)
            out[f"forecast_risk_delta_h{h}"] = 100.0 * (blend - out["stall_probability"].astype(float))
            supervised_blend[f"h{h}"] = {
                "status": "ok",
                "learner": "logit",
                "n_train": int(len(train)),
                "features": sup_features,
                "weight_supervised": float(w_sup),
            }
    else:
        supervised_blend = {"status": "skipped", "reason": "no_supervised_features"}

    backtest: Dict[str, Any] = {}
    for h in horizons:
        target_col = f"stall_h{h}"
        pred_col = f"forecast_stall_prob_h{h}"
        eval_df = out[(out[target_col].notna()) & (out["year"] <= (latest_year - h))].copy()
        if len(eval_df) < 50 or eval_df[target_col].nunique() < 2:
            backtest[f"h{h}"] = {"status": "skipped", "reason": "insufficient_eval_rows"}
            continue
        y = eval_df[target_col].astype(int).to_numpy()
        p = np.clip(eval_df[pred_col].astype(float).to_numpy(), 0.0, 1.0)
        met = _calc_cls_metrics(y, p)
        top_k = max(1, int(round(0.20 * len(eval_df))))
        top = eval_df.sort_values(pred_col, ascending=False).head(top_k)
        backtest[f"h{h}"] = {
            "status": "ok",
            "n_obs": int(len(eval_df)),
            "positive_rate": float(np.mean(y)),
            "roc_auc": met.get("roc_auc"),
            "average_precision": met.get("average_precision"),
            "brier": met.get("brier"),
            "top20_observed_stall_rate": float(top[target_col].mean()),
            "top20_mean_pred_prob": float(top[pred_col].mean()),
        }

    latest = out[out["year"] == latest_year].copy()
    if latest.empty:
        pd.DataFrame().to_csv(DATA_OUTPUTS / "pulse_ai_horizon_forecast.csv", index=False)
        return pd.DataFrame(), {
            "status": "skipped",
            "reason": "latest_year_rows_missing",
            "horizons": list(horizons),
            "backtest": backtest,
        }

    latest_cols = [
        "city_id",
        "city_name",
        "country",
        "continent",
        "year",
        "trajectory_regime",
        "stall_probability",
        "stall_risk_score",
        "stall_probability_low",
        "stall_probability_high",
        "stall_risk_low",
        "stall_risk_high",
        "stall_risk_interval_width",
    ]
    for h in horizons:
        latest_cols.extend(
            [
                f"forecast_stall_prob_h{h}",
                f"forecast_stall_risk_h{h}",
                f"forecast_regime_entropy_h{h}",
                f"forecast_dominant_regime_h{h}",
                f"forecast_risk_delta_h{h}",
            ]
        )
    latest_cols = [c for c in latest_cols if c in latest.columns]
    latest_out = latest[latest_cols].copy()
    sort_col = f"forecast_stall_risk_h{max(horizons)}"
    if sort_col not in latest_out.columns:
        sort_col = "stall_risk_score"
    latest_out = latest_out.sort_values(sort_col, ascending=False)
    latest_out.to_csv(DATA_OUTPUTS / "pulse_ai_horizon_forecast.csv", index=False)

    top_future = latest_out.head(12)[
        [c for c in ["city_name", "country", "trajectory_regime", "stall_risk_score", sort_col] if c in latest_out.columns]
    ].to_dict(orient="records")

    horizon_means: Dict[str, float | None] = {}
    for h in horizons:
        col = f"forecast_stall_prob_h{h}"
        horizon_means[f"mean_prob_h{h}"] = float(out[col].mean()) if col in out.columns else None

    return latest_out, {
        "status": "ok",
        "horizons": list(horizons),
        "n_regimes": int(len(regimes)),
        "transition_rows": int(len(pairs)),
        "global_stall_rate": global_risk,
        "regime_risk_map": {str(k): float(v) for k, v in regime_risk_map.items()},
        "supervised_blend": supervised_blend,
        "backtest": backtest,
        "horizon_means": horizon_means,
        "top_future_risk_cities": top_future,
    }


def _fit_best_stall_model_variant(
    dyn: pd.DataFrame,
    latest_year: int,
) -> tuple[Any | None, Dict[str, Any], pd.DataFrame, List[str], Dict[str, Any]]:
    base_variants = _build_model_feature_variants(dyn)
    if not base_variants:
        model, metrics, model_df, used = _fit_stall_model(dyn, latest_year=latest_year, feature_cols=None)
        return model, metrics, model_df, used, {"status": "single", "variants": {}}

    variant_specs: List[tuple[str, List[str], str]] = []
    for name, cols in base_variants.items():
        variant_specs.append((name, cols, "none"))
    if "base" in base_variants:
        variant_specs.append(("base_dro", base_variants["base"], "continent_year_balanced"))
    if "full_pulse_graph" in base_variants:
        variant_specs.append(("full_pulse_graph_dro", base_variants["full_pulse_graph"], "continent_year_balanced"))

    variant_results: Dict[str, Dict[str, Any]] = {}
    candidate_rows: List[Dict[str, Any]] = []
    spec_map: Dict[str, Dict[str, Any]] = {}

    for name, feat_cols, sw_mode in variant_specs:
        spec_map[name] = {"feature_cols": feat_cols, "sample_weight_mode": sw_mode}
        mdl, met, _, used = _fit_stall_model(
            dyn,
            latest_year=latest_year,
            feature_cols=feat_cols,
            save_importance=False,
            sample_weight_mode=sw_mode,
        )
        met_copy = dict(met)
        met_copy["feature_columns"] = used
        met_copy["feature_count"] = int(len(used))
        met_copy["model_available"] = bool(mdl is not None)
        met_copy["sample_weight_mode"] = sw_mode
        learner_for_spatial = str(met_copy.get("selected_learner", "gb"))
        spatial_eval = (
            _variant_spatial_objective(dyn, used, latest_year=latest_year, learner=learner_for_spatial)
            if used
            else {"status": "skipped"}
        )
        met_copy["spatial_eval"] = spatial_eval
        variant_results[name] = met_copy

        if mdl is not None and met_copy.get("status") == "ok":
            internal_obj = _metric_objective(met_copy)
            spatial_auc = spatial_eval.get("mean_roc_auc") if isinstance(spatial_eval, dict) else None
            if spatial_auc is not None and np.isfinite(float(spatial_auc)):
                objective = 0.75 * internal_obj + 0.25 * float(spatial_auc)
            else:
                objective = internal_obj
            candidate_rows.append(
                {
                    "name": name,
                    "objective": float(objective),
                    "internal_objective": float(internal_obj),
                    "spatial_mean_roc_auc": float(spatial_auc) if spatial_auc is not None else None,
                    "feature_count": int(len(used)),
                    "sample_weight_mode": sw_mode,
                    "selected_learner": learner_for_spatial,
                }
            )

    if not candidate_rows:
        model, metrics, model_df, used = _fit_stall_model(dyn, latest_year=latest_year, feature_cols=None)
        selection = {
            "status": "fallback",
            "reason": "all_variants_unavailable",
            "selected_variant": "fallback_default",
            "variants": variant_results,
        }
        return model, metrics, model_df, used, selection

    candidate_rows = sorted(candidate_rows, key=lambda x: (x["objective"], -x["feature_count"]), reverse=True)
    best_name = str(candidate_rows[0]["name"])
    best_spec = spec_map.get(best_name, {"feature_cols": base_variants.get("base", []), "sample_weight_mode": "none"})
    model, metrics, model_df, used = _fit_stall_model(
        dyn,
        latest_year=latest_year,
        feature_cols=best_spec["feature_cols"],
        save_importance=True,
        sample_weight_mode=str(best_spec["sample_weight_mode"]),
    )
    metrics["selected_variant"] = best_name
    metrics["variant_objective"] = float(candidate_rows[0]["objective"])
    metrics["variant_internal_objective"] = float(candidate_rows[0]["internal_objective"])
    metrics["variant_spatial_mean_roc_auc"] = candidate_rows[0]["spatial_mean_roc_auc"]
    metrics["sample_weight_mode"] = str(best_spec["sample_weight_mode"])

    selection = {
        "status": "ok",
        "selected_variant": best_name,
        "selection_table": candidate_rows,
        "variants": variant_results,
    }
    return model, metrics, model_df, used, selection


def _derive_quadrant_thresholds(latest_city: pd.DataFrame) -> Dict[str, float]:
    if latest_city.empty:
        return {
            "risk_low": 35.0,
            "risk_high": 65.0,
            "accel_low": 35.0,
            "accel_high": 65.0,
        }

    risk = latest_city["stall_risk_score"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    accel = latest_city["acceleration_score"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if risk.empty or accel.empty:
        return {
            "risk_low": 35.0,
            "risk_high": 65.0,
            "accel_low": 35.0,
            "accel_high": 65.0,
        }

    risk_low = float(np.quantile(risk, 0.35))
    risk_high = float(np.quantile(risk, 0.65))
    accel_low = float(np.quantile(accel, 0.35))
    accel_high = float(np.quantile(accel, 0.65))

    if risk_high <= risk_low + 8.0:
        mid = float(np.median(risk))
        risk_low = min(risk_low, mid - 4.0)
        risk_high = max(risk_high, mid + 4.0)
    if accel_high <= accel_low + 8.0:
        mid = float(np.median(accel))
        accel_low = min(accel_low, mid - 4.0)
        accel_high = max(accel_high, mid + 4.0)

    risk_low = float(np.clip(risk_low, 8.0, 92.0))
    risk_high = float(np.clip(risk_high, risk_low + 5.0, 96.0))
    accel_low = float(np.clip(accel_low, 8.0, 92.0))
    accel_high = float(np.clip(accel_high, accel_low + 5.0, 96.0))

    return {
        "risk_low": risk_low,
        "risk_high": risk_high,
        "accel_low": accel_low,
        "accel_high": accel_high,
    }


def _assign_quadrants(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.Series:
    risk_low = thresholds["risk_low"]
    risk_high = thresholds["risk_high"]
    accel_low = thresholds["accel_low"]
    accel_high = thresholds["accel_high"]

    return pd.Series(
        np.select(
            [
                (df["acceleration_score"] >= accel_high) & (df["stall_risk_score"] <= risk_low),
                (df["acceleration_score"] <= accel_low) & (df["stall_risk_score"] >= risk_high),
                (df["acceleration_score"] >= accel_high) & (df["stall_risk_score"] >= risk_high),
                (df["acceleration_score"] <= accel_low) & (df["stall_risk_score"] <= risk_low),
            ],
            [
                "accelerating",
                "stalling",
                "fragile_boom",
                "recovery_window",
            ],
            default="steady",
        ),
        index=df.index,
    )


def _evaluate_regime_incremental_value(
    dyn: pd.DataFrame,
    regime_year: pd.DataFrame,
    latest_year: int,
) -> Dict[str, Any]:
    if regime_year.empty:
        return {"status": "skipped", "reason": "trajectory_regime_year_empty"}

    if "trajectory_regime" in dyn.columns:
        df = dyn.copy()
    else:
        use_cols = ["city_id", "year", "trajectory_regime"]
        df = dyn.merge(regime_year[use_cols], on=["city_id", "year"], how="left")

    if "trajectory_regime" not in df.columns:
        reg_cols = [c for c in ["trajectory_regime_x", "trajectory_regime_y"] if c in df.columns]
        if reg_cols:
            df["trajectory_regime"] = df[reg_cols].bfill(axis=1).iloc[:, 0]

    df["trajectory_regime"] = df.get("trajectory_regime", "mixed_transition")
    df = _append_regime_transition_features(df)
    df = df.dropna(subset=["growth_next", "trajectory_regime"]).copy()
    if df.empty:
        return {"status": "skipped", "reason": "no_overlap_with_stall_labels"}

    base_features = list(FEATURE_COLUMNS)
    transition_features = [c for c in REGIME_TRANSITION_NUMERIC_FEATURES if c in df.columns]
    network_features = [c for c in REGIME_NETWORK_NUMERIC_FEATURES if c in df.columns]
    _fill_feature_medians(df, base_features + transition_features + network_features)

    train_df = df[df["year"] <= latest_year - 2].copy()
    test_df = df[df["year"] == latest_year - 1].copy()
    if train_df.empty or test_df.empty:
        split_year = int(df["year"].quantile(0.75))
        train_df = df[df["year"] <= split_year].copy()
        test_df = df[df["year"] > split_year].copy()
    if train_df.empty or test_df.empty:
        return {"status": "skipped", "reason": "insufficient_temporal_split"}

    y_train = train_df["stall_next"].astype(int).to_numpy()
    y_test = test_df["stall_next"].astype(int).to_numpy()
    if np.unique(y_train).size < 2:
        return {"status": "skipped", "reason": "single_class_train_target"}

    x_train_base = train_df[base_features].to_numpy(dtype=float)
    x_test_base = test_df[base_features].to_numpy(dtype=float)
    reg_train = pd.get_dummies(train_df["trajectory_regime"], prefix="reg")
    reg_test = pd.get_dummies(test_df["trajectory_regime"], prefix="reg")
    reg_test = reg_test.reindex(columns=reg_train.columns, fill_value=0)

    x_train_transition = (
        train_df[transition_features].to_numpy(dtype=float) if transition_features else np.zeros((len(train_df), 0))
    )
    x_test_transition = (
        test_df[transition_features].to_numpy(dtype=float) if transition_features else np.zeros((len(test_df), 0))
    )
    x_train_network = train_df[network_features].to_numpy(dtype=float) if network_features else np.zeros((len(train_df), 0))
    x_test_network = test_df[network_features].to_numpy(dtype=float) if network_features else np.zeros((len(test_df), 0))

    variants = {
        "base": (x_train_base, x_test_base),
        "plus_regime": (
            np.column_stack([x_train_base, reg_train.to_numpy(dtype=float)]),
            np.column_stack([x_test_base, reg_test.to_numpy(dtype=float)]),
        ),
        "plus_transition": (
            np.column_stack([x_train_base, x_train_transition]),
            np.column_stack([x_test_base, x_test_transition]),
        ),
        "plus_network": (
            np.column_stack([x_train_base, x_train_network]),
            np.column_stack([x_test_base, x_test_network]),
        ),
        "plus_regime_transition": (
            np.column_stack([x_train_base, x_train_transition, reg_train.to_numpy(dtype=float)]),
            np.column_stack([x_test_base, x_test_transition, reg_test.to_numpy(dtype=float)]),
        ),
        "full_pulse_graph": (
            np.column_stack([x_train_base, x_train_transition, x_train_network, reg_train.to_numpy(dtype=float)]),
            np.column_stack([x_test_base, x_test_transition, x_test_network, reg_test.to_numpy(dtype=float)]),
        ),
    }

    variant_metrics: Dict[str, Dict[str, float | None]] = {}
    for name, (x_train, x_test) in variants.items():
        clf = LogisticRegression(
            max_iter=600,
            solver="liblinear",
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(x_train, y_train)
        p_pred = clf.predict_proba(x_test)[:, 1]
        variant_metrics[name] = _calc_cls_metrics(y_test, p_pred)

    base_metrics = variant_metrics["base"]
    plus_metrics = variant_metrics["plus_regime"]
    plus_transition_metrics = variant_metrics["plus_transition"]
    plus_network_metrics = variant_metrics["plus_network"]
    plus_full_metrics = variant_metrics["plus_regime_transition"]
    full_pulse_graph_metrics = variant_metrics["full_pulse_graph"]

    use_auc = bool(np.unique(y_test).size >= 2)
    if use_auc:
        best_variant = max(
            variant_metrics.keys(),
            key=lambda nm: float(variant_metrics[nm]["roc_auc"]) if variant_metrics[nm]["roc_auc"] is not None else -np.inf,
        )
    else:
        best_variant = min(variant_metrics.keys(), key=lambda nm: float(variant_metrics[nm]["brier"]))
    best_metrics = variant_metrics[best_variant]

    def _delta_from(metric: str, metrics: Dict[str, float | None]) -> float | None:
        b = base_metrics.get(metric)
        p = metrics.get(metric)
        if b is None or p is None:
            return None
        return float(p - b)

    return {
        "status": "ok",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "n_regimes": int(train_df["trajectory_regime"].nunique()),
        "transition_feature_columns": transition_features,
        "network_feature_columns": network_features,
        "variants": list(variant_metrics.keys()),
        "best_variant": best_variant,
        "base_metrics": base_metrics,
        "plus_regime_metrics": plus_metrics,
        "plus_transition_metrics": plus_transition_metrics,
        "plus_network_metrics": plus_network_metrics,
        "plus_regime_transition_metrics": plus_full_metrics,
        "full_pulse_graph_metrics": full_pulse_graph_metrics,
        "best_metrics": best_metrics,
        "delta_roc_auc": _delta_from("roc_auc", best_metrics),
        "delta_average_precision": _delta_from("average_precision", best_metrics),
        "delta_brier": _delta_from("brier", best_metrics),
        "delta_roc_auc_regime_only": _delta_from("roc_auc", plus_metrics),
        "delta_roc_auc_transition_only": _delta_from("roc_auc", plus_transition_metrics),
        "delta_roc_auc_network_only": _delta_from("roc_auc", plus_network_metrics),
        "delta_roc_auc_full": _delta_from("roc_auc", plus_full_metrics),
        "delta_roc_auc_full_pulse_graph": _delta_from("roc_auc", full_pulse_graph_metrics),
    }


def _empty_regime_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    city_regime = pd.DataFrame(
        columns=[
            "city_id",
            "city_name",
            "country",
            "continent",
            "cluster_id",
            "trajectory_regime",
            "silhouette",
            "latest_composite",
            "latest_stall_risk",
            "trend_slope",
            "growth_volatility",
        ]
    )
    regime_year = pd.DataFrame(columns=["city_id", "year", "cluster_id", "trajectory_regime"])
    regime_share = pd.DataFrame(columns=["year", "trajectory_regime", "city_count", "share"])
    trans = pd.DataFrame(columns=["from_regime", "to_regime", "count", "probability"])
    return city_regime, regime_year, regime_share, trans


def _build_trajectory_regimes(
    dyn: pd.DataFrame,
    latest_city: pd.DataFrame,
    latest_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    city_regime, regime_year, regime_share, trans = _empty_regime_outputs()
    city_ids, years, seq = _trajectory_sequences(dyn)

    if len(city_ids) < 8 or seq.size == 0:
        city_regime.to_csv(DATA_OUTPUTS / "pulse_ai_trajectory_regimes.csv", index=False)
        regime_year.to_csv(DATA_OUTPUTS / "pulse_ai_regime_by_year.csv", index=False)
        regime_share.to_csv(DATA_OUTPUTS / "pulse_ai_regime_year_share.csv", index=False)
        trans.to_csv(DATA_OUTPUTS / "pulse_ai_regime_transition_matrix.csv", index=False)
        return (
            city_regime,
            regime_year,
            trans,
            {"status": "skipped", "reason": "too_few_cities_for_trajectory_regimes"},
        )

    n_cities = len(city_ids)
    n_clusters = min(6, max(3, int(round(np.sqrt(n_cities / 12.0) + 2))))
    n_clusters = min(n_clusters, n_cities)
    if n_clusters < 2:
        city_regime.to_csv(DATA_OUTPUTS / "pulse_ai_trajectory_regimes.csv", index=False)
        regime_year.to_csv(DATA_OUTPUTS / "pulse_ai_regime_by_year.csv", index=False)
        regime_share.to_csv(DATA_OUTPUTS / "pulse_ai_regime_year_share.csv", index=False)
        trans.to_csv(DATA_OUTPUTS / "pulse_ai_regime_transition_matrix.csv", index=False)
        return (
            city_regime,
            regime_year,
            trans,
            {"status": "skipped", "reason": "insufficient_cluster_support"},
        )

    dist = _pairwise_dtw(seq, window_ratio=0.4)
    labels, medoids, objective = _kmedoids_from_distance(dist, n_clusters=n_clusters, random_state=42, max_iter=80)
    sil = _silhouette_from_distance(dist, labels)

    latest_base = latest_city[
        [
            "city_id",
            "city_name",
            "country",
            "continent",
            "composite_index",
            "stall_risk_score",
        ]
    ].copy()
    latest_base = latest_base.rename(
        columns={
            "composite_index": "latest_composite",
            "stall_risk_score": "latest_stall_risk",
        }
    )
    city_regime = pd.DataFrame({"city_id": city_ids, "cluster_id": labels, "silhouette": sil})
    city_regime = city_regime.merge(latest_base, on="city_id", how="left")

    slope_rows: List[Dict[str, Any]] = []
    for cid, grp in dyn.groupby("city_id"):
        growth_vol = float(grp["growth_1"].std()) if "growth_1" in grp.columns else float("nan")
        slope_rows.append(
            {
                "city_id": str(cid),
                "trend_slope": _safe_slope(grp["year"].to_numpy(dtype=float), grp["composite_index"].to_numpy(dtype=float)),
                "growth_volatility": growth_vol,
            }
        )
    slopes = pd.DataFrame(slope_rows)
    city_regime = city_regime.merge(slopes, on="city_id", how="left")

    cluster_stats = city_regime.groupby("cluster_id", as_index=False).agg(
        latest_composite=("latest_composite", "mean"),
        latest_stall_risk=("latest_stall_risk", "mean"),
        trend_slope=("trend_slope", "mean"),
        growth_volatility=("growth_volatility", "mean"),
        n_cities=("city_id", "nunique"),
    )
    cluster_stats["score"] = (
        1.20 * _zscore(cluster_stats["trend_slope"])
        + 0.55 * _zscore(cluster_stats["latest_composite"])
        - 0.90 * _zscore(cluster_stats["growth_volatility"].fillna(cluster_stats["growth_volatility"].median()))
        - 0.95 * _zscore(cluster_stats["latest_stall_risk"])
    )

    ordered_clusters = cluster_stats.sort_values("score", ascending=False)["cluster_id"].tolist()
    label_map = {
        int(cid): TRAJECTORY_REGIME_LABELS[idx] if idx < len(TRAJECTORY_REGIME_LABELS) else f"regime_{idx+1}"
        for idx, cid in enumerate(ordered_clusters)
    }
    city_regime["trajectory_regime"] = city_regime["cluster_id"].map(label_map).fillna("mixed_transition")

    medoid_city_ids = [city_ids[int(i)] for i in medoids]
    medoid_df = city_regime[city_regime["city_id"].isin(medoid_city_ids)][
        ["city_id", "city_name", "country", "continent", "cluster_id", "trajectory_regime"]
    ].copy()
    medoid_df = medoid_df.rename(columns={"city_name": "medoid_city_name"})
    medoid_df.to_csv(DATA_OUTPUTS / "pulse_ai_regime_medoids.csv", index=False)

    # Assign time-varying regime per city-year using sequence prefixes.
    year_rows: List[Dict[str, Any]] = []
    cluster_count = len(medoids)
    medoid_prefix_cache: Dict[tuple[int, int], np.ndarray] = {}
    for t_idx, year in enumerate(years):
        prefix_len = t_idx + 1
        win = max(1, int(round(prefix_len * 0.4)))
        for i, cid in enumerate(city_ids):
            seq_i = seq[i, :prefix_len]
            dvals = []
            for k in range(cluster_count):
                key = (k, prefix_len)
                if key not in medoid_prefix_cache:
                    medoid_prefix_cache[key] = seq[int(medoids[k]), :prefix_len]
                dvals.append(_dtw_distance(seq_i, medoid_prefix_cache[key], window=win))
            best_k = int(np.argmin(dvals))
            year_rows.append(
                {
                    "city_id": cid,
                    "year": int(year),
                    "cluster_id": best_k,
                    "trajectory_regime": label_map.get(best_k, "mixed_transition"),
                }
            )
    regime_year = pd.DataFrame(year_rows)

    regime_share = (
        regime_year.groupby(["year", "trajectory_regime"], as_index=False)
        .agg(city_count=("city_id", "nunique"))
        .sort_values(["year", "city_count"], ascending=[True, False])
    )
    regime_share["share"] = regime_share["city_count"] / regime_share.groupby("year")["city_count"].transform("sum")

    trans_rows: List[Dict[str, Any]] = []
    for _, grp in regime_year.sort_values(["city_id", "year"]).groupby("city_id"):
        vals = grp["trajectory_regime"].tolist()
        for i in range(len(vals) - 1):
            trans_rows.append({"from_regime": vals[i], "to_regime": vals[i + 1]})
    if trans_rows:
        trans = pd.DataFrame(trans_rows)
        trans = (
            trans.groupby(["from_regime", "to_regime"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values(["from_regime", "to_regime"])
        )
        trans["probability"] = trans["count"] / trans.groupby("from_regime")["count"].transform("sum")

    city_regime = city_regime.sort_values(["trajectory_regime", "latest_stall_risk"], ascending=[True, False])
    city_regime.to_csv(DATA_OUTPUTS / "pulse_ai_trajectory_regimes.csv", index=False)
    regime_year.to_csv(DATA_OUTPUTS / "pulse_ai_regime_by_year.csv", index=False)
    regime_share.to_csv(DATA_OUTPUTS / "pulse_ai_regime_year_share.csv", index=False)
    trans.to_csv(DATA_OUTPUTS / "pulse_ai_regime_transition_matrix.csv", index=False)

    latest_dist = (
        regime_year[regime_year["year"] == latest_year]["trajectory_regime"]
        .value_counts(normalize=True)
        .round(4)
        .to_dict()
    )
    transition_self_prob = None
    transition_entropy = None
    if not trans.empty:
        self_count = float(trans.loc[trans["from_regime"] == trans["to_regime"], "count"].sum())
        total_count = float(trans["count"].sum())
        transition_self_prob = (self_count / total_count) if total_count > 0 else None
        probs = trans["probability"].to_numpy(dtype=float)
        probs = probs[probs > 1e-12]
        if probs.size > 0:
            transition_entropy = float(-np.sum(probs * np.log(probs)))

    summary = {
        "status": "ok",
        "n_cities": int(n_cities),
        "n_years": int(len(years)),
        "n_regimes": int(n_clusters),
        "dtw_objective": objective,
        "silhouette_mean": float(np.nanmean(sil)) if len(sil) else None,
        "silhouette_median": float(np.nanmedian(sil)) if len(sil) else None,
        "latest_distribution": latest_dist,
        "transition_self_probability": transition_self_prob,
        "transition_entropy": transition_entropy,
        "medoid_cities": medoid_df.to_dict(orient="records"),
    }
    return city_regime, regime_year, trans, summary


def run_pulse_ai_engine(panel: pd.DataFrame) -> Dict[str, Any]:
    """Run AI dynamic inference for acceleration/stall states and archetypes."""
    required = {"city_id", "city_name", "country", "continent", "year", "composite_index"}
    missing = sorted(required.difference(set(panel.columns)))
    if missing:
        msg = f"Missing required columns for pulse AI: {missing}"
        raise ValueError(msg)

    LOGGER.info("Pulse AI engine: preparing dynamics table...")
    latest_year = int(pd.to_numeric(panel["year"], errors="coerce").max())
    label_reference_end_year = latest_year - 2 if np.isfinite(latest_year) else None
    dyn, thresholds = _prepare_dynamics_table(panel, label_reference_end_year=label_reference_end_year)
    latest_year = int(dyn["year"].max())
    LOGGER.info(
        "Pulse AI engine: dynamics ready: rows=%s, cities=%s, latest_year=%s",
        len(dyn),
        dyn["city_id"].nunique(),
        latest_year,
    )

    # Use heuristic score as a warm start to support pre-model regime discovery.
    dyn["stall_probability"] = (dyn["heuristic_stall_risk"] / 100.0).clip(0.0, 1.0)
    dyn["stall_risk_score"] = dyn["heuristic_stall_risk"].clip(0.0, 100.0)

    LOGGER.info("Pulse AI engine: discovering trajectory regimes and fitting stall model...")
    latest_proxy = dyn[dyn["year"] == latest_year].copy()
    _, trajectory_regime_by_year, _, trajectory_summary = _build_trajectory_regimes(
        dyn,
        latest_proxy,
        latest_year=latest_year,
    )
    if not trajectory_regime_by_year.empty:
        dyn = dyn.merge(
            trajectory_regime_by_year[["city_id", "year", "trajectory_regime"]],
            on=["city_id", "year"],
            how="left",
        )
    else:
        dyn["trajectory_regime"] = "mixed_transition"

    dyn["trajectory_regime"] = dyn["trajectory_regime"].fillna("mixed_transition")
    dyn = _append_regime_transition_features(dyn)
    dyn = _build_regime_network_features(dyn, trajectory_regime_by_year)

    model, model_metrics, model_df, used_features, model_variant_selection = _fit_best_stall_model_variant(
        dyn,
        latest_year=latest_year,
    )
    if len(used_features) == 0:
        used_features = _resolve_model_features(dyn)
    _fill_feature_medians(dyn, used_features, ref_df=model_df if not model_df.empty else None)

    if model is not None and len(used_features) > 0:
        dyn["stall_probability"] = model.predict_proba(dyn[used_features])[:, 1]
        dyn["stall_risk_score"] = np.clip(100.0 * dyn["stall_probability"], 0.0, 100.0)

    # Blend model score with trajectory-level prior for smoother dynamic risk profile.
    prior_hist = (
        dyn[(dyn["year"] <= (latest_year - 1)) & dyn["trajectory_regime"].notna() & dyn["stall_next"].notna()]
        .groupby("trajectory_regime")["stall_next"]
        .mean()
        .to_dict()
    )
    if prior_hist:
        global_stall = float(dyn["stall_next"].dropna().mean()) if dyn["stall_next"].notna().any() else 0.5
        prior = dyn["trajectory_regime"].map(prior_hist).fillna(global_stall).astype(float)
        dyn["stall_probability"] = np.clip(0.86 * dyn["stall_probability"] + 0.14 * prior, 0.0, 1.0)
        dyn["stall_risk_score"] = np.clip(100.0 * dyn["stall_probability"], 0.0, 100.0)

    LOGGER.info("Pulse AI engine: running dynamic hazard, graph, phase, and pulse-index layers...")
    dyn, continent_calibration = _apply_continent_prior_calibration(dyn, latest_year=latest_year)
    dyn, dynamic_kinetic_structure = _build_dynamic_kinetic_structure(dyn, latest_year=latest_year)
    dyn, dynamic_critical_transition = _estimate_critical_transition_early_warning(dyn, latest_year=latest_year)
    dyn, dynamic_transition_hazard = _fit_dynamic_transition_hazard(dyn, latest_year=latest_year)
    dyn, dynamic_graph_diffusion = _estimate_dynamic_graph_diffusion(dyn, latest_year=latest_year)
    dyn, dynamic_main_risk_fusion = _apply_dynamic_hazard_fusion_to_main_risk(dyn, latest_year=latest_year)
    dyn, uncertainty_quantification = _apply_split_conformal_uncertainty(dyn, latest_year=latest_year, alpha=0.10)
    dynamic_global_cycle = _build_dynamic_global_cycle(dyn)
    dyn, dynamic_phase_portrait = _estimate_dynamic_phase_portrait(dyn, latest_year=latest_year)
    dynamic_state_event_effects = _estimate_dynamic_state_event_effects(dyn)
    dynamic_sync_network = _estimate_dynamic_sync_network(dyn)
    dyn, dynamic_pulse_index = _build_dynamic_pulse_index(dyn, latest_year=latest_year)
    LOGGER.info("Pulse AI engine: running policy lab, offline RL, and horizon forecast...")
    dynamic_policy_lab = _simulate_dynamic_policy_lab(dyn, latest_year=latest_year)
    policy_rl_city, dynamic_policy_rl = _simulate_dynamic_policy_offline_rl(
        dyn,
        latest_year=latest_year,
        gamma=0.88,
        n_iterations=6,
    )
    for col in [
        "stall_probability_low",
        "stall_probability_high",
        "stall_risk_low",
        "stall_risk_high",
        "stall_risk_interval_width",
        "stall_probability_base_pre_fusion",
        "stall_risk_score_base_pre_fusion",
        "critical_transition_score",
        "critical_transition_probability",
        "critical_transition_risk_score",
        "kinetic_energy_score",
        "turning_point_risk",
        "damping_pressure",
        "dynamic_hazard_probability",
        "dynamic_hazard_score",
        "dynamic_hazard_fused_probability",
        "dynamic_hazard_fused_score",
        "dynamic_hazard_gate_weight",
        "graph_diffusion_probability",
        "graph_diffusion_score",
        "graph_diffusion_fused_probability",
        "graph_diffusion_fused_score",
        "phase_risk_velocity",
        "phase_accel_velocity",
        "phase_flow_speed",
        "phase_curvature",
        "phase_divergence_bin",
        "phase_vorticity_bin",
        "dynamic_pulse_index",
        "dynamic_pulse_delta_1y",
        "dynamic_pulse_trend_3y",
    ]:
        if col not in dyn.columns:
            dyn[col] = np.nan
    if "kinetic_state" not in dyn.columns:
        dyn["kinetic_state"] = "mixed_transition"
    if "critical_transition_band" not in dyn.columns:
        dyn["critical_transition_band"] = "monitoring"
    if "phase_label" not in dyn.columns:
        dyn["phase_label"] = "insufficient_history"
    if "dynamic_pulse_state" not in dyn.columns:
        dyn["dynamic_pulse_state"] = "mid_transition"

    latest_city = dyn[dyn["year"] == latest_year].copy()
    quadrant_thresholds = _derive_quadrant_thresholds(latest_city)
    latest_city["acceleration_band"] = pd.cut(
        latest_city["acceleration_score"],
        bins=[-np.inf, quadrant_thresholds["accel_low"], quadrant_thresholds["accel_high"], np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)
    latest_city["stall_band"] = pd.cut(
        latest_city["stall_risk_score"],
        bins=[-np.inf, quadrant_thresholds["risk_low"], quadrant_thresholds["risk_high"], np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)
    latest_city["pulse_quadrant"] = _assign_quadrants(latest_city, quadrant_thresholds).astype(str)

    archetypes = _build_city_archetypes(dyn, latest_city)
    latest_city = latest_city.merge(archetypes[["city_id", "archetype"]], on="city_id", how="left")
    if "trajectory_regime" not in latest_city.columns:
        latest_city = latest_city.merge(
            trajectory_regime_by_year[trajectory_regime_by_year["year"] == latest_year][["city_id", "trajectory_regime"]],
            on="city_id",
            how="left",
        )
    latest_city["trajectory_regime"] = latest_city["trajectory_regime"].fillna("mixed_transition")

    prev = dyn[dyn["year"] == (latest_year - 1)][
        ["city_id", "acceleration_score", "stall_risk_score", "trajectory_regime"]
    ].rename(
        columns={
            "acceleration_score": "prev_acceleration_score",
            "stall_risk_score": "prev_stall_risk_score",
            "trajectory_regime": "prev_trajectory_regime",
        }
    )
    latest_city = latest_city.merge(prev, on="city_id", how="left")
    latest_city["accel_shift_1y"] = latest_city["acceleration_score"] - latest_city["prev_acceleration_score"]
    latest_city["risk_shift_1y"] = latest_city["stall_risk_score"] - latest_city["prev_stall_risk_score"]
    latest_city["movement_magnitude_1y"] = np.sqrt(
        latest_city["accel_shift_1y"].fillna(0.0) ** 2 + latest_city["risk_shift_1y"].fillna(0.0) ** 2
    )
    latest_city["regime_changed_1y"] = (
        latest_city["trajectory_regime"].fillna("mixed_transition")
        != latest_city["prev_trajectory_regime"].fillna("mixed_transition")
    ).astype(int)

    trajectory_incremental = _evaluate_regime_incremental_value(
        dyn,
        trajectory_regime_by_year,
        latest_year=latest_year,
    )
    primary_learner = str(model_metrics.get("selected_learner", "gb"))
    cross_continent_generalization = _pulse_cross_continent_generalization(
        dyn,
        used_features,
        latest_year=latest_year,
        primary_learner=primary_learner,
    )
    shock_pulse_response = _estimate_shock_pulse_response(dyn, latest_year=latest_year)
    horizon_latest, multi_horizon_forecast = _estimate_multi_horizon_forecast(
        dyn,
        latest_year=latest_year,
        horizons=(1, 2, 3),
    )
    if not horizon_latest.empty:
        fcols = ["city_id"] + [c for c in horizon_latest.columns if c.startswith("forecast_")]
        latest_city = latest_city.merge(horizon_latest[fcols], on="city_id", how="left")
    if not policy_rl_city.empty:
        rl_cols = [
            "city_id",
            "rl_state_id",
            "rl_best_action",
            "rl_best_q_value",
            "rl_second_q_value",
            "rl_q_advantage",
            "rl_policy_confidence",
            "rl_baseline_risk_h1",
            "rl_expected_risk_h1",
            "rl_expected_delta_risk_h1",
            "rl_expected_delta_accel_h1",
            "rl_q_wait_observe",
            "rl_q_resilience_upgrade",
            "rl_q_innovation_pulse",
            "rl_q_anti_overheating",
        ]
        latest_city = latest_city.merge(
            policy_rl_city[[c for c in rl_cols if c in policy_rl_city.columns]],
            on="city_id",
            how="left",
        )
    for h in [1, 2, 3]:
        for col in [
            f"forecast_stall_prob_h{h}",
            f"forecast_stall_risk_h{h}",
            f"forecast_regime_entropy_h{h}",
            f"forecast_dominant_regime_h{h}",
            f"forecast_risk_delta_h{h}",
        ]:
            if col not in latest_city.columns:
                latest_city[col] = np.nan
    for col in [
        "rl_state_id",
        "rl_best_action",
        "rl_best_q_value",
        "rl_second_q_value",
        "rl_q_advantage",
        "rl_policy_confidence",
        "rl_baseline_risk_h1",
        "rl_expected_risk_h1",
        "rl_expected_delta_risk_h1",
        "rl_expected_delta_accel_h1",
        "rl_q_wait_observe",
        "rl_q_resilience_upgrade",
        "rl_q_innovation_pulse",
        "rl_q_anti_overheating",
    ]:
        if col not in latest_city.columns:
            latest_city[col] = np.nan

    LOGGER.info("Pulse AI engine: exporting score panels, latest-city tables, and summary...")
    dyn_out = dyn[
        [
            "city_id",
            "city_name",
            "country",
            "continent",
            "year",
            "composite_index",
            "growth_1",
            "accel_1",
            "acceleration_score",
            "stall_probability",
            "stall_risk_score",
            "stall_probability_base_pre_fusion",
            "stall_risk_score_base_pre_fusion",
            "stall_probability_low",
            "stall_probability_high",
            "stall_risk_low",
            "stall_risk_high",
            "stall_risk_interval_width",
            "critical_transition_score",
            "critical_transition_probability",
            "critical_transition_risk_score",
            "critical_transition_band",
            "kinetic_state",
            "kinetic_energy_score",
            "turning_point_risk",
            "damping_pressure",
            "dynamic_hazard_probability",
            "dynamic_hazard_score",
            "dynamic_hazard_fused_probability",
            "dynamic_hazard_fused_score",
            "dynamic_hazard_gate_weight",
            "graph_diffusion_probability",
            "graph_diffusion_score",
            "graph_diffusion_fused_probability",
            "graph_diffusion_fused_score",
            "phase_risk_velocity",
            "phase_accel_velocity",
            "phase_flow_speed",
            "phase_curvature",
            "phase_divergence_bin",
            "phase_vorticity_bin",
            "phase_label",
            "dynamic_pulse_index",
            "dynamic_pulse_delta_1y",
            "dynamic_pulse_trend_3y",
            "dynamic_pulse_state",
            "stall_next",
            "accelerate_next",
            "trajectory_regime",
            "regime_run_length_log",
            "regime_switch_rate_3y",
            "regime_switch_rate_5y",
            "regime_year_share",
            "regime_forward_risk",
            "regime_self_transition_prob",
            "regime_transition_entropy",
        ]
    ].copy()
    dyn_out.to_csv(DATA_OUTPUTS / "pulse_ai_scores.csv", index=False)

    latest_cols = [
        "city_id",
        "city_name",
        "country",
        "continent",
        "year",
        "composite_index",
        "acceleration_score",
        "stall_risk_score",
        "stall_risk_low",
        "stall_risk_high",
        "stall_risk_interval_width",
        "critical_transition_score",
        "critical_transition_risk_score",
        "critical_transition_band",
        "kinetic_state",
        "kinetic_energy_score",
        "turning_point_risk",
        "damping_pressure",
        "dynamic_hazard_score",
        "dynamic_hazard_fused_score",
        "dynamic_hazard_gate_weight",
        "graph_diffusion_score",
        "graph_diffusion_fused_score",
        "phase_flow_speed",
        "phase_curvature",
        "phase_divergence_bin",
        "phase_vorticity_bin",
        "phase_label",
        "dynamic_pulse_index",
        "dynamic_pulse_delta_1y",
        "dynamic_pulse_trend_3y",
        "dynamic_pulse_state",
        "pulse_quadrant",
        "acceleration_band",
        "stall_band",
        "archetype",
        "trajectory_regime",
        "prev_acceleration_score",
        "prev_stall_risk_score",
        "prev_trajectory_regime",
        "accel_shift_1y",
        "risk_shift_1y",
        "movement_magnitude_1y",
        "regime_changed_1y",
        "rl_best_action",
        "rl_best_q_value",
        "rl_q_advantage",
        "rl_policy_confidence",
        "rl_expected_delta_risk_h1",
        "rl_expected_delta_accel_h1",
        "forecast_stall_risk_h1",
        "forecast_stall_risk_h2",
        "forecast_stall_risk_h3",
        "forecast_risk_delta_h1",
        "forecast_risk_delta_h2",
        "forecast_risk_delta_h3",
        "forecast_dominant_regime_h1",
        "forecast_dominant_regime_h2",
        "forecast_dominant_regime_h3",
    ]
    latest_city[latest_cols].sort_values("stall_risk_score", ascending=False).to_csv(
        DATA_OUTPUTS / "pulse_ai_city_latest.csv", index=False
    )
    archetypes.to_csv(DATA_OUTPUTS / "pulse_ai_archetypes.csv", index=False)

    top_risk = (
        latest_city[latest_cols]
        .sort_values("stall_risk_score", ascending=False)
        .head(12)[
            [
                "city_name",
                "country",
                "stall_risk_score",
                "acceleration_score",
                "pulse_quadrant",
                "archetype",
                "trajectory_regime",
                "kinetic_state",
                "critical_transition_score",
                "critical_transition_risk_score",
                "critical_transition_band",
                "dynamic_hazard_score",
                "dynamic_hazard_fused_score",
                "graph_diffusion_score",
                "graph_diffusion_fused_score",
                "phase_label",
                "phase_flow_speed",
                "phase_divergence_bin",
                "dynamic_pulse_index",
                "dynamic_pulse_delta_1y",
                "rl_best_action",
                "rl_q_advantage",
                "rl_expected_delta_risk_h1",
                "risk_shift_1y",
                "accel_shift_1y",
                "stall_risk_low",
                "stall_risk_high",
                "forecast_stall_risk_h3",
            ]
        ]
        .to_dict(orient="records")
    )
    top_movers = (
        latest_city[latest_cols]
        .sort_values("movement_magnitude_1y", ascending=False)
        .head(12)[
            [
                "city_name",
                "country",
                "stall_risk_score",
                "acceleration_score",
                "risk_shift_1y",
                "accel_shift_1y",
                "trajectory_regime",
                "regime_changed_1y",
                "phase_label",
                "phase_flow_speed",
                "dynamic_pulse_index",
                "dynamic_pulse_delta_1y",
                "rl_best_action",
                "rl_q_advantage",
                "rl_expected_delta_risk_h1",
            ]
        ]
        .to_dict(orient="records")
    )

    summary = {
        "status": model_metrics.get("status", "ok"),
        "latest_year": latest_year,
        "n_rows": int(len(dyn)),
        "n_cities": int(dyn["city_id"].nunique()),
        "definitions": {
            "acceleration_score": "100*minmax(0.55*ΔIndex + 0.30*Δ²Index + 0.15*2Y-momentum - 0.22*volatility)",
            "stall_event_next_year": f"1[growth_next <= Q30], Q30={thresholds['stall_threshold']:.3f}",
            "accelerate_event_next_year": f"1[growth_next >= Q70], Q70={thresholds['accelerate_threshold']:.3f}",
            "stall_threshold_source": str(thresholds.get("threshold_source")),
            "stall_threshold_source_end_year": thresholds.get("threshold_source_end_year"),
            "stall_threshold_reference_rows": int(thresholds.get("threshold_reference_rows", 0)),
            "trajectory_regime": "DTW-KMedoids dynamic regimes with year-by-year prefix matching",
            "regime_network_features": "Markov regime graph features: forward stall risk, self-transition persistence, transition entropy",
            "continent_calibration": "Adaptive continent-prior calibration with validation-year objective selection",
            "uncertainty_quantification": "Split-conformal interval for stall probability (target coverage 90%)",
            "dynamic_structure": "Ten-layer dynamics: kinetic state + critical transition + transition hazard + graph diffusion + risk fusion + global cycle + phase portrait + dynamic pulse index + policy/event lab + offline RL policy layer",
            "dynamic_pulse_index": "Dynamic city pulse index combining risk level, acceleration level, risk/acceleration velocity, and structural stability",
            "policy_offline_rl": "Offline fitted Q-iteration over city transitions with state-aware counterfactual action effects",
            "multi_horizon_forecast": "Markov regime-propagation forecast for 1Y/2Y/3Y stall risk",
            "quadrant_rule": "Data-adaptive thresholds by latest-year quantiles: low=Q35, high=Q65 for risk/acceleration",
        },
        "quadrant_thresholds": quadrant_thresholds,
        "model_metrics": model_metrics,
        "model_primary_learner": primary_learner,
        "continent_calibration": continent_calibration,
        "uncertainty_quantification": uncertainty_quantification,
        "dynamic_structure": {
            "kinetic_state": dynamic_kinetic_structure,
            "critical_transition": dynamic_critical_transition,
            "transition_hazard": dynamic_transition_hazard,
            "graph_diffusion": dynamic_graph_diffusion,
            "main_risk_fusion": dynamic_main_risk_fusion,
            "global_cycle": dynamic_global_cycle,
            "phase_portrait": dynamic_phase_portrait,
            "pulse_index": dynamic_pulse_index,
            "state_event_effects": dynamic_state_event_effects,
            "sync_network": dynamic_sync_network,
            "policy_lab": dynamic_policy_lab,
            "policy_rl": dynamic_policy_rl,
        },
        "model_feature_columns": used_features,
        "model_variant_selection": model_variant_selection,
        "trajectory_regime_model": trajectory_summary,
        "trajectory_regime_incremental_value": trajectory_incremental,
        "cross_continent_generalization": cross_continent_generalization,
        "shock_pulse_response": shock_pulse_response,
        "multi_horizon_forecast": multi_horizon_forecast,
        "quadrant_distribution_latest": latest_city["pulse_quadrant"].value_counts(normalize=True).round(4).to_dict(),
        "archetype_distribution_latest": latest_city["archetype"].value_counts(normalize=True).round(4).to_dict(),
        "trajectory_regime_distribution_latest": latest_city["trajectory_regime"]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
        "dynamic_pulse_state_distribution_latest": latest_city["dynamic_pulse_state"]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
        "dynamic_pulse_index_mean_latest": float(latest_city["dynamic_pulse_index"].mean())
        if "dynamic_pulse_index" in latest_city.columns and not latest_city.empty
        else None,
        "top_stall_risk_cities": top_risk,
        "top_movers_1y": top_movers,
    }
    dump_json(DATA_OUTPUTS / "pulse_ai_summary.json", summary)

    LOGGER.info(
        "Pulse AI engine done: %s cities, model=%s",
        summary["n_cities"],
        summary["model_metrics"].get("status"),
    )
    return summary
