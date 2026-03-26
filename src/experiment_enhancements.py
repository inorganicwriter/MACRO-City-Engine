from __future__ import annotations

"""Additional robustness and uncertainty analyses for publication-quality experiments."""

import logging
import math
import json
from contextlib import contextmanager
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from .benchmark_eval import FEATURES
from .causal_st import CausalSTConfig, _run_single
from .econometrics import run_did_two_way_fe, run_staggered_did
from .utils import DATA_OUTPUTS, dump_json

LOGGER = logging.getLogger(__name__)


@contextmanager
def _temporary_log_level(logger_name: str, level: int):
    logger = logging.getLogger(logger_name)
    old = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _ols_hc1(y: np.ndarray, x: np.ndarray) -> Dict[str, np.ndarray | int]:
    """Small OLS helper with HC1 robust SE for sensitivity specs."""
    xtx = x.T @ x
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta

    n, k = x.shape
    meat = np.zeros((k, k))
    for i in range(n):
        xi = x[i : i + 1, :]
        ui2 = float(resid[i] ** 2)
        meat += ui2 * (xi.T @ xi)

    scale = n / max(1, n - k)
    cov = scale * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    t_val = beta / se
    return {"coef": beta, "stderr": se, "t_value": t_val, "n_obs": n}


def _two_way_demean(df: pd.DataFrame, col: str, city_col: str = "city_id", year_col: str = "year") -> pd.Series:
    city_mean = df.groupby(city_col)[col].transform("mean")
    year_mean = df.groupby(year_col)[col].transform("mean")
    grand_mean = float(df[col].mean())
    return df[col] - city_mean - year_mean + grand_mean


def _ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    cdf = pd.DataFrame({"y": y_true.astype(float), "p": y_prob.astype(float)})
    try:
        cdf["bin"] = pd.qcut(cdf["p"], q=n_bins, duplicates="drop")
    except Exception:  # noqa: BLE001
        cdf["bin"] = pd.cut(cdf["p"], bins=n_bins)
    grouped = (
        cdf.groupby("bin", as_index=False, observed=False)
        .agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
        .copy()
    )
    if grouped.empty:
        return float("nan")
    grouped["gap"] = np.abs(grouped["obs"] - grouped["pred"])
    return float((grouped["gap"] * grouped["n"]).sum() / max(float(grouped["n"].sum()), 1.0))


def _metric(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _resolve_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in FEATURES if c in df.columns]
    if len(cols) < 8:
        msg = f"Insufficient enhancement features after resolution: {cols}"
        raise RuntimeError(msg)
    return cols


def _load_main_design_variant_from_econometrics() -> str | None:
    path = DATA_OUTPUTS / "econometric_summary.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    v = payload.get("main_design_variant")
    return str(v) if v is not None else None


def _resolve_did_design_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Select a treatment design with real treated-control variation for robustness diagnostics."""
    candidates: List[Dict[str, str]] = [
        {
            "design_variant": "intense_external_peak",
            "treated_col": "treated_city_intense_external_peak",
            "post_col": "post_policy_intense_external_peak",
            "did_col": "did_treatment_intense_external_peak",
            "cohort_col": "treatment_cohort_year_intense_external_peak",
        },
        {
            "design_variant": "intense_external_direct",
            "treated_col": "treated_city_intense_external_direct",
            "post_col": "post_policy_intense_external_direct",
            "did_col": "did_treatment_intense_external_direct",
            "cohort_col": "treatment_cohort_year_intense_external_direct",
        },
        {
            "design_variant": "evidence_a",
            "treated_col": "treated_city_evidence_a",
            "post_col": "post_policy_evidence_a",
            "did_col": "did_treatment_evidence_a",
            "cohort_col": "treatment_cohort_year_evidence_a",
        },
        {
            "design_variant": "evidence_ab",
            "treated_col": "treated_city_evidence_ab",
            "post_col": "post_policy_evidence_ab",
            "did_col": "did_treatment_evidence_ab",
            "cohort_col": "treatment_cohort_year_evidence_ab",
        },
        {
            "design_variant": "external_direct",
            "treated_col": "treated_city_external_direct",
            "post_col": "post_policy_external_direct",
            "did_col": "did_treatment_external_direct",
            "cohort_col": "treatment_cohort_year_external_direct",
        },
        {
            "design_variant": "direct_core",
            "treated_col": "treated_city_direct_core",
            "post_col": "post_policy_direct_core",
            "did_col": "did_treatment_direct_core",
            "cohort_col": "treatment_cohort_year_direct_core",
        },
        {
            "design_variant": "direct",
            "treated_col": "treated_city_direct",
            "post_col": "post_policy_direct",
            "did_col": "did_treatment_direct",
            "cohort_col": "treatment_cohort_year_direct",
        },
        {
            "design_variant": "all_sources_full_treated",
            "treated_col": "treated_city",
            "post_col": "post_policy",
            "did_col": "did_treatment",
            "cohort_col": "treatment_cohort_year",
        },
    ]

    preferred = _load_main_design_variant_from_econometrics()
    if preferred is not None:
        candidates = sorted(candidates, key=lambda x: 0 if x["design_variant"] == preferred else 1)

    def _try_candidate(spec: Dict[str, str]) -> Dict[str, Any] | None:
        needed = {spec["treated_col"], spec["post_col"], spec["did_col"]}
        if not needed.issubset(set(panel.columns)):
            return None
        tmp = panel.copy()
        tmp["treated_city"] = pd.to_numeric(tmp[spec["treated_col"]], errors="coerce").fillna(0).astype(int)
        tmp["post_policy"] = pd.to_numeric(tmp[spec["post_col"]], errors="coerce").fillna(0).astype(int)
        tmp["did_treatment"] = pd.to_numeric(tmp[spec["did_col"]], errors="coerce").fillna(0).astype(int)
        cohort_col = spec.get("cohort_col")
        if cohort_col and cohort_col in tmp.columns:
            tmp["treatment_cohort_year"] = pd.to_numeric(tmp[cohort_col], errors="coerce").fillna(9999).astype(int)

        did_unique = int(pd.to_numeric(tmp["did_treatment"], errors="coerce").fillna(0).nunique())
        treated_share = float(pd.to_numeric(tmp["treated_city"], errors="coerce").fillna(0).mean())
        treated_rows = int(pd.to_numeric(tmp["treated_city"], errors="coerce").fillna(0).sum())
        strictly_mixed = bool(treated_rows > 0 and treated_rows < len(tmp))
        balanced_share = bool(0.03 <= treated_share <= 0.97)
        variation_ok = bool(did_unique > 1 and strictly_mixed)
        quality = int(variation_ok) * 2 + int(balanced_share) + int(did_unique > 1)
        return {
            "panel": tmp,
            "design_variant": str(spec["design_variant"]),
            "treated_share": treated_share,
            "did_variation": did_unique,
            "treated_rows": treated_rows,
            "strictly_mixed_treatment": strictly_mixed,
            "variation_ok": variation_ok,
            "balanced_share": balanced_share,
            "quality_score": quality,
            "source_columns": {
                "treated_col": spec["treated_col"],
                "post_col": spec["post_col"],
                "did_col": spec["did_col"],
                "cohort_col": spec.get("cohort_col"),
            },
        }

    attempts: List[Dict[str, Any]] = []
    for cand in candidates:
        out = _try_candidate(cand)
        if out is not None:
            attempts.append(out)

    if not attempts:
        return panel.copy(), {
            "status": "fallback_raw",
            "reason": "no_candidate_design_columns_found",
            "design_variant": None,
            "treated_share": float(pd.to_numeric(panel.get("treated_city", 0), errors="coerce").fillna(0).mean())
            if "treated_city" in panel.columns
            else None,
        }

    strong = [a for a in attempts if a["variation_ok"] and a["balanced_share"]]
    selected = strong[0] if strong else max(attempts, key=lambda x: x["quality_score"])
    return selected["panel"], {
        "status": "ok",
        "design_variant": selected["design_variant"],
        "treated_share": float(selected["treated_share"]),
        "did_variation": int(selected["did_variation"]),
        "treated_rows": int(selected["treated_rows"]),
        "strictly_mixed_treatment": bool(selected["strictly_mixed_treatment"]),
        "balanced_share": bool(selected["balanced_share"]),
        "preferred_from_econometrics": preferred,
        "source_columns": selected["source_columns"],
        "candidate_count": int(len(attempts)),
    }


def _resolve_treatment_reference_year(panel: pd.DataFrame, fallback: int = 2020) -> int:
    years = pd.to_numeric(panel.get("year"), errors="coerce").dropna().astype(int)
    if years.empty:
        return int(fallback)
    min_year = int(years.min())
    max_year = int(years.max())

    if "treatment_cohort_year" in panel.columns:
        cohort = pd.to_numeric(panel["treatment_cohort_year"], errors="coerce")
        cohort = cohort[(cohort >= min_year) & (cohort <= max_year) & (cohort < 9999)]
        if not cohort.empty:
            return int(np.median(cohort.to_numpy(dtype=float)))

    if {"treated_city", "post_policy", "year"}.issubset(set(panel.columns)):
        onset = pd.to_numeric(
            panel.loc[
                (pd.to_numeric(panel["treated_city"], errors="coerce").fillna(0) > 0)
                & (pd.to_numeric(panel["post_policy"], errors="coerce").fillna(0) > 0),
                "year",
            ],
            errors="coerce",
        ).dropna()
        if not onset.empty:
            return int(onset.min())

    return int(np.clip(fallback, min_year, max_year))


def _fit_lr_rf(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    x_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train[target].to_numpy(dtype=float)
    x_test = test[feature_cols].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    lr_pred = lr.predict(x_test)

    rf = RandomForestRegressor(
        n_estimators=280,
        max_depth=10,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)

    return {
        "linear": _metric(y_test, lr_pred),
        "random_forest": _metric(y_test, rf_pred),
    }


def _prepare_panel_for_benchmark(panel: pd.DataFrame, target: str = "composite_index") -> pd.DataFrame:
    df = panel.sort_values(["city_id", "year"]).copy()
    df["target_t1"] = df.groupby("city_id")[target].shift(-1)
    feature_cols = _resolve_feature_columns(df)
    df = df.dropna(subset=["target_t1"] + feature_cols).copy()
    df.attrs["feature_cols"] = feature_cols
    return df


def _temporal_split_sensitivity(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    years = sorted(df["year"].unique().tolist())
    rows: List[Dict[str, Any]] = []
    # Keep enough training window and enough test horizon.
    candidate_splits = [y for y in years if y >= years[3] and y <= years[-2]]
    for split_year in candidate_splits:
        train = df[df["year"] <= split_year].copy()
        test = df[df["year"] > split_year].copy()
        if len(train) < 160 or len(test) < 80:
            continue
        met = _fit_lr_rf(train, test, target="target_t1", feature_cols=feature_cols, seed=42 + int(split_year))
        rows.append(
            {
                "split_year": int(split_year),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "linear_rmse": met["linear"]["rmse"],
                "linear_mae": met["linear"]["mae"],
                "linear_r2": met["linear"]["r2"],
                "rf_rmse": met["random_forest"]["rmse"],
                "rf_mae": met["random_forest"]["mae"],
                "rf_r2": met["random_forest"]["r2"],
            }
        )
    return pd.DataFrame(rows)


def _spatial_dispersion(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    continents = sorted(df["continent"].dropna().unique().tolist())
    for cont in continents:
        train = df[df["continent"] != cont].copy()
        test = df[df["continent"] == cont].copy()
        if len(train) < 160 or len(test) < 40:
            continue
        met = _fit_lr_rf(train, test, target="target_t1", feature_cols=feature_cols, seed=700 + len(rows))
        rows.append(
            {
                "left_out_continent": cont,
                "n_test": int(len(test)),
                "linear_rmse": met["linear"]["rmse"],
                "linear_r2": met["linear"]["r2"],
                "rf_rmse": met["random_forest"]["rmse"],
                "rf_r2": met["random_forest"]["r2"],
            }
        )
    return pd.DataFrame(rows)


def _pulse_uncertainty_and_calibration(
    n_boot: int = 500,
    random_state: int = 42,
) -> Dict[str, Any]:
    scores_path = DATA_OUTPUTS / "pulse_ai_scores.csv"
    if not scores_path.exists():
        return {"status": "skipped", "reason": "pulse_ai_scores_missing"}

    df = pd.read_csv(scores_path)
    df = df.dropna(subset=["stall_probability", "stall_next"]).copy()
    if df.empty or df["stall_next"].nunique() < 2:
        return {"status": "skipped", "reason": "invalid_pulse_labels"}

    # Evaluate only where next-period outcomes are observable (exclude terminal year by default).
    max_year = int(df["year"].max())
    eval_df = df[df["year"] < max_year].copy()
    if eval_df.empty or eval_df["stall_next"].nunique() < 2:
        eval_df = df.copy()

    y = eval_df["stall_next"].astype(int).to_numpy()
    p = eval_df["stall_probability"].astype(float).to_numpy()

    auc0 = float(roc_auc_score(y, p))
    ap0 = float(average_precision_score(y, p))
    brier0 = float(brier_score_loss(y, p))

    # Calibration bins.
    cal_df = eval_df.copy()
    cal_df["bin"] = pd.qcut(cal_df["stall_probability"], q=10, duplicates="drop")
    bins = (
        cal_df.groupby("bin", as_index=False, observed=False)
        .agg(pred=("stall_probability", "mean"), obs=("stall_next", "mean"), n=("stall_next", "size"))
        .copy()
    )
    total_n = float(bins["n"].sum())
    bins["abs_gap"] = np.abs(bins["obs"] - bins["pred"])
    ece = float((bins["abs_gap"] * bins["n"]).sum() / max(total_n, 1.0))
    mce = float(bins["abs_gap"].max())
    bins.to_csv(DATA_OUTPUTS / "experiment_pulse_calibration_bins.csv", index=False)

    # Precision@k on latest evaluable year with observed positives.
    year_stats = (
        eval_df.groupby("year", as_index=False)
        .agg(pos_rate=("stall_next", "mean"), n=("stall_next", "size"))
        .sort_values("year")
    )
    valid_years = year_stats[(year_stats["pos_rate"] > 0.0) & (year_stats["pos_rate"] < 1.0)]
    topk_eval_year = int(valid_years["year"].max()) if not valid_years.empty else int(year_stats["year"].max())
    latest = eval_df[eval_df["year"] == topk_eval_year].copy().sort_values("stall_probability", ascending=False)
    topk_rows: List[Dict[str, Any]] = []
    for frac in [0.10, 0.20, 0.30, 0.40]:
        k = max(1, int(round(len(latest) * frac)))
        picked = latest.head(k)
        topk_rows.append(
            {
                "top_fraction": frac,
                "k": int(k),
                "mean_pred_prob": float(picked["stall_probability"].mean()),
                "observed_stall_rate": float(picked["stall_next"].mean()),
            }
        )
    topk_df = pd.DataFrame(topk_rows)
    topk_df.to_csv(DATA_OUTPUTS / "experiment_pulse_topk.csv", index=False)

    # Bootstrap CI for AUC/AP.
    rng = np.random.default_rng(random_state)
    n = len(eval_df)
    boot_rows: List[Dict[str, float]] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        pb = p[idx]
        if np.unique(yb).size < 2:
            continue
        boot_rows.append(
            {
                "auc": float(roc_auc_score(yb, pb)),
                "ap": float(average_precision_score(yb, pb)),
                "brier": float(brier_score_loss(yb, pb)),
            }
        )
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(DATA_OUTPUTS / "experiment_pulse_bootstrap_metrics.csv", index=False)
    if boot.empty:
        ci = {"status": "insufficient_bootstrap_variation"}
    else:
        ci = {
            "auc_ci_95": [float(boot["auc"].quantile(0.025)), float(boot["auc"].quantile(0.975))],
            "ap_ci_95": [float(boot["ap"].quantile(0.025)), float(boot["ap"].quantile(0.975))],
            "brier_ci_95": [float(boot["brier"].quantile(0.025)), float(boot["brier"].quantile(0.975))],
            "n_boot_used": int(len(boot)),
        }

    return {
        "auc": auc0,
        "average_precision": ap0,
        "brier": brier0,
        "ece_10bin": ece,
        "mce_10bin": mce,
        "bootstrap_ci": ci,
        "evaluation_year_max": int(eval_df["year"].max()),
        "topk_eval_year": int(topk_eval_year),
    }


def _did_placebo_years(
    panel: pd.DataFrame,
    stacked_placebo: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    base = run_did_two_way_fe(panel)
    base_p = float(2.0 * (1.0 - _normal_cdf(abs(base["t_value"]))))
    rows.append(
        {
            "spec": "actual_2020",
            "coef": float(base["coef"]),
            "stderr": float(base["stderr"]),
            "t_value": float(base["t_value"]),
            "p_value_norm": base_p,
            "n_obs": int(base["n_obs"]),
        }
    )

    if isinstance(stacked_placebo, dict) and stacked_placebo.get("status") == "ok":
        stacked_rows = stacked_placebo.get("rows", [])
        if isinstance(stacked_rows, list) and len(stacked_rows) > 0:
            for r in stacked_rows:
                lead = int(r.get("lead_years", -1))
                coef = float(r.get("coef", 0.0))
                stderr = float(r.get("stderr", 1e-6))
                t_val = float(r.get("t_value", 0.0))
                p_val = float(r.get("p_value", 1.0))
                n_obs = int(round(float(r.get("n_obs_weighted", len(panel)))))
                rows.append(
                    {
                        "spec": f"stacked_lead_{lead}",
                        "coef": coef,
                        "stderr": stderr,
                        "t_value": t_val,
                        "p_value_norm": p_val,
                        "n_obs": n_obs,
                    }
                )

    for placebo_year in [2017, 2018, 2019]:
        df = panel.copy()
        df["did_treatment"] = df["treated_city"].astype(int) * (df["year"] >= placebo_year).astype(int)
        res = run_did_two_way_fe(df)
        p_val = float(2.0 * (1.0 - _normal_cdf(abs(res["t_value"]))))
        rows.append(
            {
                "spec": f"placebo_{placebo_year}",
                "coef": float(res["coef"]),
                "stderr": float(res["stderr"]),
                "t_value": float(res["t_value"]),
                "p_value_norm": p_val,
                "n_obs": int(res["n_obs"]),
            }
        )

    return pd.DataFrame(rows)


def _did_permutation_test(panel: pd.DataFrame, n_perm: int = 400, random_state: int = 42) -> Dict[str, Any]:
    actual = run_did_two_way_fe(panel)
    actual_coef = float(actual["coef"])

    rng = np.random.default_rng(random_state)
    city_flags = panel[["city_id", "treated_city"]].drop_duplicates("city_id").copy()
    city_ids = city_flags["city_id"].tolist()
    base_flags = city_flags["treated_city"].astype(int).to_numpy()

    perm_coefs: List[float] = []
    for _ in range(n_perm):
        shuffled = rng.permutation(base_flags)
        mapping = dict(zip(city_ids, shuffled, strict=False))
        df = panel.copy()
        df["treated_city"] = df["city_id"].map(mapping).fillna(0).astype(int)
        df["did_treatment"] = df["treated_city"] * df["post_policy"].astype(int)
        try:
            res = run_did_two_way_fe(df)
            perm_coefs.append(float(res["coef"]))
        except Exception:  # noqa: BLE001
            continue

    dist = pd.DataFrame({"perm_coef": perm_coefs})
    dist.to_csv(DATA_OUTPUTS / "experiment_did_permutation_distribution.csv", index=False)
    if dist.empty:
        return {"status": "skipped", "reason": "no_permutation_draws"}

    arr = dist["perm_coef"].to_numpy(dtype=float)
    p_val = float((1.0 + np.sum(np.abs(arr) >= abs(actual_coef))) / (1.0 + len(arr)))
    return {
        "actual_coef": actual_coef,
        "actual_t_value": float(actual["t_value"]),
        "actual_stderr": float(actual["stderr"]),
        "perm_mean": float(np.mean(arr)),
        "perm_std": float(np.std(arr)),
        "perm_p_value_two_sided": p_val,
        "n_permutations_used": int(len(arr)),
    }


def _run_did_fe_spec(
    panel: pd.DataFrame,
    outcome: str,
    controls: List[str],
    *,
    include_treated_trend: bool,
    weight_col: str | None,
    year_min: int | None,
    year_max: int | None,
    drop_year: int | None,
) -> Dict[str, float] | None:
    required = ["city_id", "year", "did_treatment", outcome] + controls
    if include_treated_trend:
        required.append("treated_city")
    if weight_col is not None:
        required.append(weight_col)

    for col in required:
        if col not in panel.columns:
            return None

    df = panel[required].copy()
    if year_min is not None:
        df = df[df["year"] >= year_min]
    if year_max is not None:
        df = df[df["year"] <= year_max]
    if drop_year is not None:
        df = df[df["year"] != drop_year]
    df = df.dropna().copy()

    if len(df) < 80:
        return None
    if df["city_id"].nunique() < 12 or df["year"].nunique() < 5:
        return None
    if df["did_treatment"].nunique() < 2:
        return None

    regressors = ["did_treatment"] + list(controls)
    if include_treated_trend:
        base_year = int(df["year"].min())
        df["treated_trend"] = df["treated_city"].astype(float) * (df["year"].astype(float) - float(base_year))
        regressors.append("treated_trend")

    tw_cols = [outcome] + regressors
    tw = df[["city_id", "year"] + tw_cols].copy()
    for col in tw_cols:
        tw[f"{col}_tw"] = _two_way_demean(tw, col)

    y = tw[f"{outcome}_tw"].to_numpy(dtype=float)
    x = tw[[f"{c}_tw" for c in regressors]].to_numpy(dtype=float)

    if weight_col is not None:
        w = df[weight_col].to_numpy(dtype=float)
        valid = np.isfinite(w) & (w > 0.0)
        if valid.sum() < 80:
            return None
        y = y[valid]
        x = x[valid, :]
        w = w[valid]
        w_scale = np.sqrt(w / max(float(np.mean(w)), 1e-8))
        y = y * w_scale
        x = x * w_scale[:, None]

    if len(y) < 80:
        return None
    if np.linalg.matrix_rank(x) < x.shape[1]:
        return None

    fit = _ols_hc1(y, x)
    coef = float(fit["coef"][0])
    stderr = float(fit["stderr"][0])
    t_val = float(fit["t_value"][0])
    p_val = float(2.0 * (1.0 - _normal_cdf(abs(t_val))))
    return {
        "coef": coef,
        "stderr": stderr,
        "t_value": t_val,
        "p_value_norm": p_val,
        "n_obs": int(fit["n_obs"]),
    }


def _did_specification_curve(panel: pd.DataFrame, fast_mode: bool = False) -> Dict[str, Any]:
    outcomes = [c for c in ["composite_index", "economic_vitality", "livability", "innovation"] if c in panel.columns]
    if not outcomes:
        return {"status": "skipped", "reason": "no_available_outcomes"}

    control_sets: List[tuple[str, List[str]]] = [
        ("none", []),
        ("baseline", ["log_gdp_pc", "internet_users", "unemployment"]),
        ("digital_labor", ["internet_users", "unemployment"]),
        ("macro", ["log_gdp_pc", "gdp_growth", "capital_formation", "inflation"]),
        (
            "full",
            [
                "log_gdp_pc",
                "internet_users",
                "unemployment",
                "gdp_growth",
                "log_population",
                "capital_formation",
                "inflation",
            ],
        ),
    ]
    if fast_mode:
        control_sets = [control_sets[0], control_sets[1], control_sets[-1]]

    min_year = int(panel["year"].min())
    max_year = int(panel["year"].max())
    window_specs: List[tuple[str, int | None, int | None, int | None]] = [
        ("full_window", None, None, None),
        ("late_start", min_year + 2, None, None),
        ("trim_tail", None, max_year - 1, None),
        ("exclude_2020", None, None, 2020),
    ]
    if fast_mode:
        window_specs = [window_specs[0], window_specs[-1]]

    trend_specs = [False] if fast_mode else [False, True]
    weight_specs: List[tuple[str, str | None]] = [("unweighted", None)]
    if (not fast_mode) and ("population" in panel.columns):
        weight_specs.append(("population_weighted", "population"))

    rows: List[Dict[str, Any]] = []
    spec_id = 0
    for outcome in outcomes:
        for control_name, controls in control_sets:
            use_controls = [c for c in controls if c in panel.columns]
            for window_name, year_min, year_max, drop_year in window_specs:
                for trend in trend_specs:
                    for weight_name, weight_col in weight_specs:
                        res = _run_did_fe_spec(
                            panel,
                            outcome=outcome,
                            controls=use_controls,
                            include_treated_trend=trend,
                            weight_col=weight_col,
                            year_min=year_min,
                            year_max=year_max,
                            drop_year=drop_year,
                        )
                        if res is None:
                            continue
                        spec_id += 1
                        rows.append(
                            {
                                "spec_id": spec_id,
                                "outcome": outcome,
                                "control_set": control_name,
                                "controls_count": int(len(use_controls)),
                                "window": window_name,
                                "with_treated_trend": int(trend),
                                "weighting": weight_name,
                                **res,
                            }
                        )

    spec_df = pd.DataFrame(rows)
    spec_df.to_csv(DATA_OUTPUTS / "experiment_did_specification_curve.csv", index=False)
    if spec_df.empty:
        return {"status": "skipped", "reason": "no_valid_specifications"}

    by_outcome_rows = (
        spec_df.groupby("outcome", as_index=False)
        .agg(
            n_specs=("coef", "size"),
            coef_median=("coef", "median"),
            coef_mean=("coef", "mean"),
            share_p_lt_0_1=("p_value_norm", lambda s: float(np.mean(s < 0.10))),
        )
        .sort_values("outcome")
    )

    return {
        "n_specs": int(len(spec_df)),
        "share_negative_coef": float(np.mean(spec_df["coef"] < 0.0)),
        "share_positive_coef": float(np.mean(spec_df["coef"] > 0.0)),
        "share_p_lt_0_10": float(np.mean(spec_df["p_value_norm"] < 0.10)),
        "share_p_lt_0_05": float(np.mean(spec_df["p_value_norm"] < 0.05)),
        "coef_median": float(spec_df["coef"].median()),
        "coef_iqr": [float(spec_df["coef"].quantile(0.25)), float(spec_df["coef"].quantile(0.75))],
        "t_abs_median": float(np.median(np.abs(spec_df["t_value"]))),
        "by_outcome": by_outcome_rows.to_dict(orient="records"),
    }


def _did_negative_controls(panel: pd.DataFrame, treatment_year: int = 2020) -> Dict[str, Any]:
    outcomes = [c for c in ["temperature_mean", "precipitation_sum", "climate_comfort"] if c in panel.columns]
    if not outcomes:
        return {"status": "skipped", "reason": "negative_control_outcomes_missing"}

    controls = [c for c in ["log_gdp_pc", "internet_users", "unemployment"] if c in panel.columns]
    rows: List[Dict[str, Any]] = []
    for outcome in outcomes:
        twfe = _run_did_fe_spec(
            panel,
            outcome=outcome,
            controls=controls,
            include_treated_trend=True,
            weight_col=None,
            year_min=None,
            year_max=None,
            drop_year=None,
        )
        if twfe is not None:
            rows.append(
                {
                    "outcome": outcome,
                    "method": "twfe_did",
                    "status": "ok",
                    "coef": float(twfe["coef"]),
                    "stderr": float(twfe["stderr"]),
                    "t_value": float(twfe["t_value"]),
                    "p_value_norm": float(twfe["p_value_norm"]),
                    "n_obs": int(twfe["n_obs"]),
                }
            )
        else:
            rows.append({"outcome": outcome, "method": "twfe_did", "status": "skipped"})

        try:
            st = run_staggered_did(panel, outcome=outcome, treatment_year=int(treatment_year))
        except Exception as exc:  # noqa: BLE001
            st = {"status": "failed", "reason": str(exc)}
        if isinstance(st, dict) and st.get("status") == "ok":
            t_val = float(st.get("post_avg_t_value", np.nan))
            p_val = float(2.0 * (1.0 - _normal_cdf(abs(t_val)))) if np.isfinite(t_val) else np.nan
            rows.append(
                {
                    "outcome": outcome,
                    "method": "staggered_did",
                    "status": "ok",
                    "coef": float(st.get("post_avg_att", np.nan)),
                    "stderr": np.nan,
                    "t_value": t_val,
                    "p_value_norm": p_val,
                    "n_obs": int(st.get("treated_city_count", 0)),
                    "pretrend_max_abs_t": float(st.get("pretrend_max_abs_t", np.nan)),
                    "pretrend_share_p_lt_0_10": float(st.get("pretrend_share_p_lt_0_10", np.nan)),
                    "pretrend_joint_p_value": float(st.get("pretrend_joint_p_value", np.nan)),
                }
            )
        else:
            rows.append(
                {
                    "outcome": outcome,
                    "method": "staggered_did",
                    "status": str(st.get("status", "skipped")) if isinstance(st, dict) else "skipped",
                    "reason": st.get("reason") if isinstance(st, dict) else "unknown",
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(DATA_OUTPUTS / "experiment_did_negative_controls.csv", index=False)
    if out.empty:
        return {"status": "skipped", "reason": "negative_control_estimation_failed"}

    valid = out[(out["status"].astype(str) == "ok")].copy()
    valid["t_value"] = pd.to_numeric(valid.get("t_value"), errors="coerce")
    valid["p_value_norm"] = pd.to_numeric(valid.get("p_value_norm"), errors="coerce")
    if valid.empty or not valid["p_value_norm"].notna().any():
        return {"status": "skipped", "reason": "negative_control_estimation_failed"}

    def _method_summary(method: str) -> Dict[str, Any]:
        sub = valid[valid["method"].astype(str) == method].copy()
        if sub.empty:
            return {"n_outcomes": 0}
        out_summary: Dict[str, Any] = {
            "n_outcomes": int(sub["outcome"].nunique()),
            "max_abs_t_value": float(np.nanmax(np.abs(sub["t_value"].to_numpy(dtype=float)))),
            "share_p_lt_0_10": float(np.nanmean(sub["p_value_norm"].to_numpy(dtype=float) < 0.10)),
            "share_p_lt_0_05": float(np.nanmean(sub["p_value_norm"].to_numpy(dtype=float) < 0.05)),
        }
        if "pretrend_joint_p_value" in sub.columns:
            pj = pd.to_numeric(sub["pretrend_joint_p_value"], errors="coerce")
            if pj.notna().any():
                out_summary["share_pretrend_joint_p_gt_0_10"] = float(np.mean(pj.dropna().to_numpy(dtype=float) > 0.10))
        if "pretrend_share_p_lt_0_10" in sub.columns:
            ps = pd.to_numeric(sub["pretrend_share_p_lt_0_10"], errors="coerce")
            if ps.notna().any():
                out_summary["share_pretrend_clean"] = float(np.mean(ps.dropna().to_numpy(dtype=float) <= 0.0))
        return out_summary

    twfe_summary = _method_summary("twfe_did")
    staggered_summary = _method_summary("staggered_did")
    primary_method = "staggered_did" if int(staggered_summary.get("n_outcomes", 0)) >= 2 else "twfe_did"
    primary = staggered_summary if primary_method == "staggered_did" else twfe_summary

    return {
        "status": "ok",
        "methods_available": sorted(valid["method"].astype(str).unique().tolist()),
        "primary_method": primary_method,
        "n_outcomes": int(primary.get("n_outcomes", 0)),
        "max_abs_t_value": float(primary.get("max_abs_t_value", np.nan)),
        "share_p_lt_0_10": float(primary.get("share_p_lt_0_10", np.nan)),
        "share_p_lt_0_05": float(primary.get("share_p_lt_0_05", np.nan)),
        "twfe_did": twfe_summary,
        "staggered_did": staggered_summary,
    }


def _load_pulse_eval_df() -> pd.DataFrame:
    scores_path = DATA_OUTPUTS / "pulse_ai_scores.csv"
    if not scores_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(scores_path)
    df = df.dropna(subset=["stall_probability", "stall_next"]).copy()
    if df.empty:
        return pd.DataFrame()

    max_year = int(df["year"].max())
    eval_df = df[df["year"] < max_year].copy()
    if eval_df.empty or eval_df["stall_next"].nunique() < 2:
        eval_df = df.copy()
    return eval_df


def _pulse_group_fairness(panel: pd.DataFrame) -> Dict[str, Any]:
    eval_df = _load_pulse_eval_df()
    if eval_df.empty:
        return {"status": "skipped", "reason": "pulse_scores_missing_or_invalid"}

    if "gdp_per_capita" in panel.columns:
        first_year = int(panel["year"].min())
        income_base = panel.loc[panel["year"] == first_year, ["city_id", "gdp_per_capita"]].drop_duplicates("city_id")
        valid_income = income_base["gdp_per_capita"].notna().sum()
        if len(income_base) >= 8 and valid_income >= 8:
            income_base["income_group"] = pd.qcut(
                income_base["gdp_per_capita"].rank(method="first"),
                q=4,
                labels=["low", "lower_middle", "upper_middle", "high"],
            ).astype(str)
        else:
            income_base["income_group"] = "mixed"
    else:
        income_base = panel[["city_id"]].drop_duplicates("city_id")
        income_base["income_group"] = "mixed"

    eval_df = eval_df.merge(income_base[["city_id", "income_group"]], on="city_id", how="left")
    eval_df["income_group"] = eval_df["income_group"].fillna("mixed")

    group_specs = [("continent", "continent"), ("income_group", "income_group")]
    rows: List[Dict[str, Any]] = []
    for group_type, col in group_specs:
        if col not in eval_df.columns:
            continue
        for group, sub in eval_df.groupby(col, dropna=False):
            y = sub["stall_next"].astype(int).to_numpy()
            p = sub["stall_probability"].astype(float).to_numpy()
            if len(sub) < 20:
                continue
            auc = float(roc_auc_score(y, p)) if np.unique(y).size >= 2 else float("nan")
            ap = float(average_precision_score(y, p)) if np.unique(y).size >= 2 else float("nan")
            rows.append(
                {
                    "group_type": group_type,
                    "group": str(group),
                    "n_obs": int(len(sub)),
                    "positive_rate": float(np.mean(y)),
                    "auc": auc,
                    "average_precision": ap,
                    "brier": float(brier_score_loss(y, p)),
                    "ece_10bin": _ece_score(y, p, n_bins=10),
                }
            )

    fairness_df = pd.DataFrame(rows)
    fairness_df.to_csv(DATA_OUTPUTS / "experiment_pulse_group_fairness.csv", index=False)
    if fairness_df.empty:
        return {"status": "skipped", "reason": "no_valid_groups"}

    disparity_rows: List[Dict[str, Any]] = []
    for gtype, sub in fairness_df.groupby("group_type"):
        auc_vals = sub["auc"].dropna()
        brier_vals = sub["brier"].dropna()
        ece_vals = sub["ece_10bin"].dropna()
        disparity_rows.append(
            {
                "group_type": gtype,
                "n_groups": int(sub["group"].nunique()),
                "auc_gap_max_min": float(auc_vals.max() - auc_vals.min()) if len(auc_vals) >= 2 else None,
                "brier_gap_max_min": float(brier_vals.max() - brier_vals.min()) if len(brier_vals) >= 2 else None,
                "ece_gap_max_min": float(ece_vals.max() - ece_vals.min()) if len(ece_vals) >= 2 else None,
            }
        )

    disparity_df = pd.DataFrame(disparity_rows)
    disparity_df.to_csv(DATA_OUTPUTS / "experiment_pulse_group_fairness_summary.csv", index=False)
    return {
        "n_rows": int(len(fairness_df)),
        "group_types": sorted(fairness_df["group_type"].unique().tolist()),
        "disparities": disparity_df.to_dict(orient="records"),
    }


def _pulse_decision_curve() -> Dict[str, Any]:
    eval_df = _load_pulse_eval_df()
    if eval_df.empty:
        return {"status": "skipped", "reason": "pulse_scores_missing_or_invalid"}

    y = eval_df["stall_next"].astype(int).to_numpy()
    p = eval_df["stall_probability"].astype(float).to_numpy()
    n = len(eval_df)
    prevalence = float(np.mean(y))

    rows: List[Dict[str, Any]] = []
    for threshold in np.arange(0.05, 0.96, 0.05):
        pred = p >= threshold
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        odds = float(threshold / max(1.0 - threshold, 1e-8))
        nb_model = float((tp / n) - (fp / n) * odds)
        nb_all = float(prevalence - (1.0 - prevalence) * odds)
        rows.append(
            {
                "threshold": float(threshold),
                "n_selected": int(pred.sum()),
                "selected_rate": float(pred.mean()),
                "tp": tp,
                "fp": fp,
                "net_benefit_model": nb_model,
                "net_benefit_treat_all": nb_all,
                "net_benefit_treat_none": 0.0,
                "nb_gain_vs_best_default": float(nb_model - max(nb_all, 0.0)),
            }
        )

    curve = pd.DataFrame(rows)
    curve.to_csv(DATA_OUTPUTS / "experiment_pulse_decision_curve.csv", index=False)
    if curve.empty:
        return {"status": "skipped", "reason": "empty_decision_curve"}

    best = curve.sort_values("net_benefit_model", ascending=False).iloc[0]
    better = curve["net_benefit_model"] > np.maximum(curve["net_benefit_treat_all"], 0.0)
    return {
        "best_threshold": float(best["threshold"]),
        "best_net_benefit": float(best["net_benefit_model"]),
        "best_selected_rate": float(best["selected_rate"]),
        "share_thresholds_model_beats_defaults": float(np.mean(better)),
        "prevalence": prevalence,
    }


def _continent_transfer_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    continents = sorted(df["continent"].dropna().unique().tolist())
    feature_cols = _resolve_feature_columns(df)
    rows: List[Dict[str, Any]] = []

    for train_cont in continents:
        train = df[df["continent"] == train_cont].copy()
        if len(train) < 100:
            continue
        x_train = train[feature_cols].to_numpy(dtype=float)
        y_train = train["target_t1"].to_numpy(dtype=float)
        model = LinearRegression()
        model.fit(x_train, y_train)

        for test_cont in continents:
            if test_cont == train_cont:
                continue
            test = df[df["continent"] == test_cont].copy()
            if len(test) < 40:
                continue
            pred = model.predict(test[feature_cols].to_numpy(dtype=float))
            met = _metric(test["target_t1"].to_numpy(dtype=float), pred)
            rows.append(
                {
                    "train_continent": train_cont,
                    "test_continent": test_cont,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "rmse": met["rmse"],
                    "mae": met["mae"],
                    "r2": met["r2"],
                }
            )

    mat = pd.DataFrame(rows)
    mat.to_csv(DATA_OUTPUTS / "experiment_continent_transfer_matrix.csv", index=False)
    if mat.empty:
        return {"status": "skipped", "reason": "insufficient_pairs"}

    by_test = (
        mat.groupby("test_continent", as_index=False)
        .agg(mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"), n_pairs=("rmse", "size"))
        .sort_values("mean_rmse", ascending=False)
    )
    by_test.to_csv(DATA_OUTPUTS / "experiment_continent_transfer_summary.csv", index=False)
    best_pair = mat.sort_values("rmse", ascending=True).iloc[0]
    worst_pair = mat.sort_values("rmse", ascending=False).iloc[0]
    return {
        "n_pairs": int(len(mat)),
        "mean_rmse": float(mat["rmse"].mean()),
        "std_rmse": float(mat["rmse"].std()),
        "hardest_test_continent": str(by_test.iloc[0]["test_continent"]),
        "best_pair": {
            "train": str(best_pair["train_continent"]),
            "test": str(best_pair["test_continent"]),
            "rmse": float(best_pair["rmse"]),
            "r2": float(best_pair["r2"]),
        },
        "worst_pair": {
            "train": str(worst_pair["train_continent"]),
            "test": str(worst_pair["test_continent"]),
            "rmse": float(worst_pair["rmse"]),
            "r2": float(worst_pair["r2"]),
        },
    }


def _leave_one_continent_out_stability(panel: pd.DataFrame, include_causal: bool = True) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    continents = sorted(panel["continent"].dropna().unique().tolist())
    cfg = CausalSTConfig(use_spatial=True, use_temporal=True, use_dr=True)

    for cont in continents:
        sub = panel[panel["continent"] != cont].copy()
        if len(sub) < 180 or sub["city_id"].nunique() < 15:
            continue
        did = run_did_two_way_fe(sub)
        row: Dict[str, Any] = {
            "excluded_continent": cont,
            "n_obs": int(len(sub)),
            "n_cities": int(sub["city_id"].nunique()),
            "did_coef": float(did["coef"]),
            "did_t": float(did["t_value"]),
        }
        if include_causal:
            cst = _run_single(sub, cfg)
            if "summary" in cst:
                row["causal_st_att"] = float(cst["summary"]["att_post"])
                row["causal_st_t"] = float(cst["summary"]["t_value"])
            else:
                row["causal_st_att"] = np.nan
                row["causal_st_t"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def _event_pretrend_diagnostics() -> Dict[str, Any]:
    path = DATA_OUTPUTS / "econometric_summary.json"
    if not path.exists():
        return {"status": "skipped", "reason": "econometric_summary_missing"}
    with path.open("r", encoding="utf-8") as f:
        econ = json.load(f)

    points = econ.get("event_study_fe", {}).get("points", [])
    if not points:
        return {"status": "skipped", "reason": "event_study_points_missing"}

    pre = [p for p in points if float(p.get("rel_year", 0)) < 0]
    if not pre:
        return {"status": "skipped", "reason": "no_pre_period_points"}

    pre_coef = np.array([float(p["coef"]) for p in pre], dtype=float)
    pre_t = np.array([float(p["t_value"]) for p in pre], dtype=float)
    return {
        "n_pre_points": int(len(pre)),
        "mean_pre_coef": float(pre_coef.mean()),
        "mean_abs_pre_coef": float(np.abs(pre_coef).mean()),
        "max_abs_pre_t": float(np.abs(pre_t).max()),
        "share_pre_abs_t_gt_1_96": float(np.mean(np.abs(pre_t) > 1.96)),
    }


def _matched_did_diagnostics() -> Dict[str, Any]:
    path = DATA_OUTPUTS / "econometric_summary.json"
    if not path.exists():
        return {"status": "skipped", "reason": "econometric_summary_missing"}
    with path.open("r", encoding="utf-8") as f:
        econ = json.load(f)

    matched = econ.get("did_matched_trend", {})
    if not matched or matched.get("status") != "ok":
        return {"status": "skipped", "reason": "did_matched_trend_missing_or_invalid"}

    placebo = matched.get("placebo", [])
    plc_df = pd.DataFrame(placebo)
    if plc_df.empty:
        return {
            "status": "ok",
            "coef": float(matched.get("coef", 0.0)),
            "t_value": float(matched.get("t_value", 0.0)),
            "p_value": float(matched.get("p_value", 1.0)),
            "matched_pairs": int(matched.get("matched_pairs", 0)),
            "avg_abs_smd_before": float(matched.get("avg_abs_smd_before", 0.0)),
            "avg_abs_smd_after": float(matched.get("avg_abs_smd_after", 0.0)),
            "placebo_count": 0,
        }

    return {
        "status": "ok",
        "coef": float(matched.get("coef", 0.0)),
        "t_value": float(matched.get("t_value", 0.0)),
        "p_value": float(matched.get("p_value", 1.0)),
        "matched_pairs": int(matched.get("matched_pairs", 0)),
        "avg_abs_smd_before": float(matched.get("avg_abs_smd_before", 0.0)),
        "avg_abs_smd_after": float(matched.get("avg_abs_smd_after", 0.0)),
        "placebo_count": int(len(plc_df)),
        "placebo_max_abs_t": float(np.abs(plc_df["t_value"]).max()),
        "placebo_share_p_lt_0_10": float(np.mean(plc_df["p_value"] < 0.10)),
        "placebo_share_p_lt_0_05": float(np.mean(plc_df["p_value"] < 0.05)),
    }


def _staggered_did_diagnostics() -> Dict[str, Any]:
    path = DATA_OUTPUTS / "econometric_summary.json"
    if not path.exists():
        return {"status": "skipped", "reason": "econometric_summary_missing"}
    with path.open("r", encoding="utf-8") as f:
        econ = json.load(f)

    staggered = econ.get("staggered_did", {})
    if not staggered or staggered.get("status") != "ok":
        return {"status": "skipped", "reason": "staggered_did_missing_or_invalid"}

    return {
        "status": "ok",
        "cohort_count": int(staggered.get("cohort_count", 0)),
        "treated_city_count": int(staggered.get("treated_city_count", 0)),
        "post_avg_att": float(staggered.get("post_avg_att", 0.0)),
        "post_avg_t_value": float(staggered.get("post_avg_t_value", 0.0)),
        "pretrend_max_abs_t": float(staggered.get("pretrend_max_abs_t", 0.0)),
        "pretrend_share_p_lt_0_10": float(staggered.get("pretrend_share_p_lt_0_10", 0.0)),
    }


def _not_yet_treated_did_diagnostics() -> Dict[str, Any]:
    path = DATA_OUTPUTS / "econometric_summary.json"
    if not path.exists():
        return {"status": "skipped", "reason": "econometric_summary_missing"}
    with path.open("r", encoding="utf-8") as f:
        econ = json.load(f)

    nyt = econ.get("not_yet_treated_did", {})
    if not nyt or nyt.get("status") != "ok":
        return {"status": "skipped", "reason": "not_yet_treated_did_missing_or_invalid"}

    return {
        "status": "ok",
        "cohort_count": int(nyt.get("cohort_count", 0)),
        "att_weighted": float(nyt.get("att_weighted", 0.0)),
        "t_value_weighted": float(nyt.get("t_value_weighted", 0.0)),
        "p_value_weighted": float(nyt.get("p_value_weighted", 1.0)),
        "ci95_weighted": nyt.get("ci95_weighted", [None, None]),
        "placebo_share_p_lt_0_10": float(nyt.get("placebo_share_p_lt_0_10", 0.0)),
        "placebo_max_abs_t": float(nyt.get("placebo_max_abs_t", 0.0)),
        "robust_placebo_pass_cohorts": int(nyt.get("robust_placebo_pass_cohorts", 0)),
        "robust_att_weighted": float(nyt.get("robust_att_weighted", 0.0))
        if nyt.get("robust_att_weighted") is not None
        else None,
        "robust_t_value_weighted": float(nyt.get("robust_t_value_weighted", 0.0))
        if nyt.get("robust_t_value_weighted") is not None
        else None,
        "robust_p_value_weighted": float(nyt.get("robust_p_value_weighted", 1.0))
        if nyt.get("robust_p_value_weighted") is not None
        else None,
        "robust_ci95_weighted": nyt.get("robust_ci95_weighted", [None, None]),
    }


def _identification_scorecard_diagnostics() -> Dict[str, Any]:
    path = DATA_OUTPUTS / "econometric_summary.json"
    if not path.exists():
        return {"status": "skipped", "reason": "econometric_summary_missing"}
    with path.open("r", encoding="utf-8") as f:
        econ = json.load(f)

    card = econ.get("identification_scorecard", {})
    if not card or card.get("status") != "ok":
        return {"status": "skipped", "reason": "identification_scorecard_missing_or_invalid"}

    pref = card.get("preferred", {})
    return {
        "status": "ok",
        "preferred_estimator": pref.get("estimator"),
        "preferred_effect": pref.get("effect"),
        "preferred_p_value": pref.get("p_value"),
        "preferred_credibility_score": pref.get("credibility_score"),
        "ranking": card.get("ranking", []),
    }


def _policy_source_sensitivity_diagnostics() -> Dict[str, Any]:
    path = DATA_OUTPUTS / "econometric_summary.json"
    if not path.exists():
        return {"status": "skipped", "reason": "econometric_summary_missing"}
    with path.open("r", encoding="utf-8") as f:
        econ = json.load(f)

    sens = econ.get("policy_source_sensitivity", {})
    if not isinstance(sens, dict) or sens.get("status") != "ok":
        return {"status": "skipped", "reason": "policy_source_sensitivity_missing_or_invalid"}

    table = pd.DataFrame(sens.get("table", []))
    if table.empty:
        table_file = sens.get("table_file")
        if table_file:
            try:
                table = pd.read_csv(table_file)
            except Exception:  # noqa: BLE001
                table = pd.DataFrame()
    if table.empty:
        return {"status": "skipped", "reason": "policy_source_sensitivity_table_empty"}

    if "status" in table.columns:
        ok = table[table["status"].astype(str) == "ok"].copy()
    else:
        ok = table.copy()
    if ok.empty:
        return {"status": "skipped", "reason": "no_ok_variants_in_policy_source_sensitivity"}

    ok = ok.copy()
    if "identification_strength" in ok.columns:
        ok["identification_strength"] = pd.to_numeric(ok["identification_strength"], errors="coerce")
        ok = ok.sort_values("identification_strength", ascending=False)
    best = ok.iloc[0]

    def _pick_variant(name: str) -> Dict[str, Any]:
        sub = ok[ok["variant"].astype(str) == name].copy()
        if sub.empty:
            return {}
        row = sub.iloc[0]
        return {
            "variant": str(row.get("variant", name)),
            "effect": float(row["preferred_effect"]) if pd.notna(row.get("preferred_effect")) else None,
            "estimator": None if pd.isna(row.get("preferred_estimator")) else str(row.get("preferred_estimator")),
            "strength": float(row["identification_strength"]) if pd.notna(row.get("identification_strength")) else None,
        }

    sign_col = "effect_sign_consistent_with_main"
    sign_share_all = None
    sign_share_source = None
    if sign_col in ok.columns:
        sign_vals = pd.to_numeric(ok[sign_col], errors="coerce")
        sign_vals = sign_vals[sign_vals.notna()]
        if len(sign_vals) > 0:
            sign_share_all = float(sign_vals.mean())
        src_mask = ok["variant"].astype(str).str.startswith("source_")
        src_vals = pd.to_numeric(ok.loc[src_mask, sign_col], errors="coerce")
        src_vals = src_vals[src_vals.notna()]
        if len(src_vals) > 0:
            sign_share_source = float(src_vals.mean())
    if sign_share_source is None:
        sign_share_source = sens.get("sign_consistency_with_main_share")
    if sign_share_all is None:
        sign_share_all = sens.get("sign_consistency_with_main_share_all_variants")

    return {
        "status": "ok",
        "variant_count_ok": int(len(ok)),
        "best_variant": str(best.get("variant")),
        "best_estimator": None if pd.isna(best.get("preferred_estimator")) else str(best.get("preferred_estimator")),
        "best_effect": float(best["preferred_effect"]) if pd.notna(best.get("preferred_effect")) else None,
        "best_identification_strength": float(best["identification_strength"])
        if pd.notna(best.get("identification_strength"))
        else None,
        "mean_identification_strength": float(pd.to_numeric(ok["identification_strength"], errors="coerce").mean())
        if "identification_strength" in ok.columns
        else None,
        "sign_consistency_with_main_share": sign_share_source,
        "sign_consistency_with_main_share_all_variants": sign_share_all,
        "external_direct": _pick_variant("source_external_direct"),
        "objective_indicator": _pick_variant("source_objective_indicator"),
        "objective_macro": _pick_variant("source_objective_macro"),
        "ai_inferred": _pick_variant("source_ai_inferred"),
    }


def _ai_incrementality_diagnostics() -> Dict[str, Any]:
    path = DATA_OUTPUTS / "model_ai_incrementality.json"
    if not path.exists():
        return {"status": "skipped", "reason": "model_ai_incrementality_missing"}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:  # noqa: BLE001
        return {"status": "skipped", "reason": f"model_ai_incrementality_unreadable:{exc}"}

    if not isinstance(payload, dict) or payload.get("status") != "ok":
        return {
            "status": "skipped",
            "reason": "model_ai_incrementality_invalid",
            "raw_status": payload.get("status") if isinstance(payload, dict) else None,
        }
    summary_rows = pd.DataFrame(payload.get("summary", []))
    if summary_rows.empty:
        return {"status": "skipped", "reason": "model_ai_incrementality_summary_empty"}

    d_rmse = pd.to_numeric(summary_rows["delta_rmse_full_ai_vs_structural"], errors="coerce")
    d_r2 = pd.to_numeric(summary_rows["delta_r2_full_ai_vs_structural"], errors="coerce")
    best_sets = summary_rows["best_set"].astype(str).value_counts().to_dict() if "best_set" in summary_rows.columns else {}
    return {
        "status": "ok",
        "targets_evaluated": int(summary_rows["target"].nunique()) if "target" in summary_rows.columns else int(len(summary_rows)),
        "mean_delta_rmse_full_ai_vs_structural": float(d_rmse.mean()) if d_rmse.notna().any() else None,
        "share_targets_rmse_improved_full_ai": float(np.mean(d_rmse < 0.0)) if d_rmse.notna().any() else None,
        "mean_delta_r2_full_ai_vs_structural": float(d_r2.mean()) if d_r2.notna().any() else None,
        "share_targets_r2_improved_full_ai": float(np.mean(d_r2 > 0.0)) if d_r2.notna().any() else None,
        "best_set_distribution": best_sets,
    }


def _mechanism_decomposition_diagnostics() -> Dict[str, Any]:
    path = DATA_OUTPUTS / "econometric_summary.json"
    if not path.exists():
        return {"status": "skipped", "reason": "econometric_summary_missing"}
    with path.open("r", encoding="utf-8") as f:
        econ = json.load(f)

    mech = econ.get("mechanism_decomposition", {})
    if not isinstance(mech, dict) or mech.get("status") != "ok":
        return {"status": "skipped", "reason": "mechanism_decomposition_missing_or_invalid"}

    rows = pd.DataFrame(mech.get("rows", []))
    if rows.empty:
        return {"status": "skipped", "reason": "mechanism_rows_empty"}
    ok = rows[rows["status"].astype(str) == "ok"].copy() if "status" in rows.columns else rows.copy()
    if ok.empty:
        return {"status": "skipped", "reason": "no_valid_mechanism_channels"}

    if "weighted_contribution" not in ok.columns:
        return {"status": "skipped", "reason": "weighted_contribution_missing"}
    if "p_value" not in ok.columns:
        ok["p_value"] = np.nan
    ok["weighted_contribution"] = pd.to_numeric(ok["weighted_contribution"], errors="coerce")
    ok["p_value"] = pd.to_numeric(ok["p_value"], errors="coerce")
    dominant = ok.iloc[ok["weighted_contribution"].abs().argmax()]
    return {
        "status": "ok",
        "channels_ok": int(len(ok)),
        "implied_composite_from_channels": mech.get("implied_composite_from_channels"),
        "decomposition_gap_vs_reference": mech.get("decomposition_gap_vs_reference"),
        "share_channels_p_lt_0_10": float(np.mean(ok["p_value"] < 0.10)) if ok["p_value"].notna().any() else None,
        "dominant_channel": str(dominant.get("channel")),
        "dominant_weighted_contribution": float(dominant.get("weighted_contribution")),
    }


def _dynamic_phase_heterogeneity_diagnostics() -> Dict[str, Any]:
    path = DATA_OUTPUTS / "econometric_summary.json"
    if not path.exists():
        return {"status": "skipped", "reason": "econometric_summary_missing"}
    with path.open("r", encoding="utf-8") as f:
        econ = json.load(f)

    dyn = econ.get("dynamic_phase_heterogeneity", {})
    if not isinstance(dyn, dict) or dyn.get("status") != "ok":
        return {"status": "skipped", "reason": "dynamic_phase_heterogeneity_missing_or_invalid"}

    interactions = dyn.get("interaction_effects", {}) if isinstance(dyn.get("interaction_effects"), dict) else {}
    high = interactions.get("high_phase_instability", {}) if isinstance(interactions.get("high_phase_instability"), dict) else {}
    fragile = interactions.get("fragile_phase", {}) if isinstance(interactions.get("fragile_phase"), dict) else {}
    top = pd.DataFrame(dyn.get("phase_label_interactions_top", []))
    sig_count = 0
    if not top.empty and "p_value_interaction" in top.columns:
        pvals = pd.to_numeric(top["p_value_interaction"], errors="coerce")
        sig_count = int(np.sum(pvals < 0.10))

    return {
        "status": "ok",
        "baseline_year": dyn.get("baseline_year"),
        "high_phase_instability_share": dyn.get("high_phase_instability_share"),
        "fragile_phase_share": dyn.get("fragile_phase_share"),
        "fragile_phase_rule": dyn.get("fragile_phase_rule"),
        "high_phase_interaction_coef": high.get("coef_interaction"),
        "high_phase_interaction_p_value": high.get("p_value_interaction"),
        "fragile_phase_interaction_coef": fragile.get("coef_interaction"),
        "fragile_phase_interaction_p_value": fragile.get("p_value_interaction"),
        "phase_label_interaction_top_count": int(len(top)),
        "phase_label_interaction_sig_count_p_lt_0_10": int(sig_count),
        "table_file": dyn.get("table_file"),
    }


def _dynamic_phase_rule_sensitivity_diagnostics() -> Dict[str, Any]:
    path = DATA_OUTPUTS / "econometric_summary.json"
    if not path.exists():
        return {"status": "skipped", "reason": "econometric_summary_missing"}
    with path.open("r", encoding="utf-8") as f:
        econ = json.load(f)

    sens = econ.get("dynamic_phase_rule_sensitivity", {})
    if not isinstance(sens, dict) or sens.get("status") != "ok":
        return {"status": "skipped", "reason": "dynamic_phase_rule_sensitivity_missing_or_invalid"}

    rows = pd.DataFrame(sens.get("rows", []))
    if rows.empty:
        table_file = sens.get("table_file")
        if table_file:
            try:
                rows = pd.read_csv(table_file)
            except Exception:  # noqa: BLE001
                rows = pd.DataFrame()
    if rows.empty:
        return {"status": "skipped", "reason": "dynamic_phase_rule_sensitivity_rows_empty"}

    if "status" in rows.columns:
        ok = rows[rows["status"].astype(str) == "ok"].copy()
    else:
        ok = rows.copy()
    if ok.empty:
        return {"status": "skipped", "reason": "dynamic_phase_rule_sensitivity_no_ok_rows"}

    ok["coef_interaction"] = pd.to_numeric(ok.get("coef_interaction"), errors="coerce")
    ok["p_value_interaction"] = pd.to_numeric(ok.get("p_value_interaction"), errors="coerce")
    ok["abs_t_interaction"] = np.abs(pd.to_numeric(ok.get("t_value_interaction"), errors="coerce"))
    ok = ok.sort_values(["p_value_interaction", "abs_t_interaction"], ascending=[True, False])
    best = ok.iloc[0]

    return {
        "status": "ok",
        "rules_ok": int(len(ok)),
        "rules_considered": int(sens.get("rules_considered", len(rows))),
        "best_rule": str(best.get("rule_name")),
        "best_interaction_coef": float(best["coef_interaction"]) if pd.notna(best.get("coef_interaction")) else None,
        "best_interaction_p_value": float(best["p_value_interaction"]) if pd.notna(best.get("p_value_interaction")) else None,
        "share_rules_p_lt_0_10": float(np.mean(ok["p_value_interaction"] < 0.10)) if ok["p_value_interaction"].notna().any() else None,
        "share_rules_p_lt_0_05": float(np.mean(ok["p_value_interaction"] < 0.05)) if ok["p_value_interaction"].notna().any() else None,
        "sign_consistency_share_vs_reference": sens.get("sign_consistency_share_vs_reference"),
        "reference_rule": sens.get("reference_rule"),
        "table_file": sens.get("table_file"),
    }


def _pulse_trajectory_regime_diagnostics() -> Dict[str, Any]:
    path = DATA_OUTPUTS / "pulse_ai_summary.json"
    if not path.exists():
        return {"status": "skipped", "reason": "pulse_ai_summary_missing"}
    with path.open("r", encoding="utf-8") as f:
        pulse = json.load(f)

    model = pulse.get("trajectory_regime_model", {})
    inc = pulse.get("trajectory_regime_incremental_value", {})
    cross = pulse.get("cross_continent_generalization", {})
    shock = pulse.get("shock_pulse_response", {})
    calib = pulse.get("continent_calibration", {})
    uncertainty = pulse.get("uncertainty_quantification", {})
    horizon = pulse.get("multi_horizon_forecast", {})
    dynamic = pulse.get("dynamic_structure", {})
    dynamic_hazard = dynamic.get("transition_hazard", {}) if isinstance(dynamic, dict) else {}
    dynamic_graph = dynamic.get("graph_diffusion", {}) if isinstance(dynamic, dict) else {}
    dynamic_cycle = dynamic.get("global_cycle", {}) if isinstance(dynamic, dict) else {}
    dynamic_fusion = dynamic.get("main_risk_fusion", {}) if isinstance(dynamic, dict) else {}
    dynamic_event = dynamic.get("state_event_effects", {}) if isinstance(dynamic, dict) else {}
    dynamic_sync = dynamic.get("sync_network", {}) if isinstance(dynamic, dict) else {}
    dynamic_policy = dynamic.get("policy_lab", {}) if isinstance(dynamic, dict) else {}
    dynamic_policy_rl = dynamic.get("policy_rl", {}) if isinstance(dynamic, dict) else {}
    dynamic_pulse_index = dynamic.get("pulse_index", {}) if isinstance(dynamic, dict) else {}
    dynamic_policy_rl_ope = (
        dynamic_policy_rl.get("offline_policy_evaluation", {})
        if isinstance(dynamic_policy_rl, dict)
        else {}
    )
    h_back = horizon.get("backtest", {}) if isinstance(horizon, dict) else {}
    h1 = h_back.get("h1", {}) if isinstance(h_back, dict) else {}
    h3 = h_back.get("h3", {}) if isinstance(h_back, dict) else {}
    model_metrics = pulse.get("model_metrics", {})
    if not model or model.get("status") != "ok":
        return {"status": "skipped", "reason": "trajectory_regime_model_missing_or_invalid"}

    return {
        "status": "ok",
        "n_regimes": int(model.get("n_regimes", 0)),
        "silhouette_mean": model.get("silhouette_mean"),
        "silhouette_median": model.get("silhouette_median"),
        "latest_distribution": model.get("latest_distribution", {}),
        "incremental_status": inc.get("status", "missing"),
        "best_variant": inc.get("best_variant"),
        "delta_roc_auc": inc.get("delta_roc_auc"),
        "delta_roc_auc_regime_only": inc.get("delta_roc_auc_regime_only"),
        "delta_roc_auc_transition_only": inc.get("delta_roc_auc_transition_only"),
        "delta_roc_auc_network_only": inc.get("delta_roc_auc_network_only"),
        "delta_roc_auc_full": inc.get("delta_roc_auc_full"),
        "delta_roc_auc_full_pulse_graph": inc.get("delta_roc_auc_full_pulse_graph"),
        "delta_average_precision": inc.get("delta_average_precision"),
        "delta_brier": inc.get("delta_brier"),
        "cross_continent_status": cross.get("status"),
        "cross_continent_mean_roc_auc": cross.get("mean_roc_auc"),
        "cross_continent_mean_delta_auc_vs_logit": cross.get("mean_delta_roc_auc_vs_logit"),
        "cross_continent_share_auc_gain_vs_logit": cross.get("share_continents_auc_gain_vs_logit"),
        "cross_continent_share_auc_gain_p_lt_0_10": cross.get("share_continents_auc_gain_p_lt_0_10"),
        "cross_continent_hardest": (cross.get("hardest_continent") or {}).get("continent") if isinstance(cross, dict) else None,
        "shock_response_status": shock.get("status"),
        "shock_response_n_shocks": shock.get("n_shocks"),
        "shock_response_worst_regime_t1": (shock.get("worst_regime_t1") or {}).get("regime") if isinstance(shock, dict) else None,
        "continent_calibration_status": calib.get("status"),
        "continent_calibration_alpha": calib.get("selected_alpha"),
        "continent_calibration_delta_auc": calib.get("delta_roc_auc"),
        "continent_calibration_delta_brier": calib.get("delta_brier"),
        "uncertainty_status": uncertainty.get("status"),
        "uncertainty_eval_coverage": uncertainty.get("evaluation_coverage"),
        "uncertainty_mean_interval_width": uncertainty.get("mean_interval_width_risk_score"),
        "multi_horizon_status": horizon.get("status"),
        "multi_horizon_h1_auc": h1.get("roc_auc"),
        "multi_horizon_h3_auc": h3.get("roc_auc"),
        "multi_horizon_h1_top20_stall_rate": h1.get("top20_observed_stall_rate"),
        "multi_horizon_h3_top20_stall_rate": h3.get("top20_observed_stall_rate"),
        "dynamic_hazard_status": dynamic_hazard.get("status"),
        "dynamic_hazard_auc": dynamic_hazard.get("roc_auc"),
        "dynamic_hazard_blend_delta_auc": (dynamic_hazard.get("blend_comparison") or {}).get("delta_auc_vs_base")
        if isinstance(dynamic_hazard, dict)
        else None,
        "dynamic_hazard_blend_delta_auc_p_value": (dynamic_hazard.get("blend_comparison") or {}).get("delta_auc_p_value")
        if isinstance(dynamic_hazard, dict)
        else None,
        "dynamic_graph_status": dynamic_graph.get("status"),
        "dynamic_graph_delta_auc": dynamic_graph.get("delta_auc_vs_base"),
        "dynamic_graph_delta_auc_p_value": dynamic_graph.get("delta_auc_p_value"),
        "dynamic_graph_selected_candidate": dynamic_graph.get("selected_candidate"),
        "dynamic_cycle_status": dynamic_cycle.get("status"),
        "dynamic_cycle_latest_phase": dynamic_cycle.get("latest_cycle_phase"),
        "dynamic_cycle_latest_tension": dynamic_cycle.get("latest_cycle_tension"),
        "dynamic_main_fusion_status": dynamic_fusion.get("status"),
        "dynamic_main_fusion_selected_candidate": dynamic_fusion.get("selected_candidate"),
        "dynamic_main_fusion_alpha": dynamic_fusion.get("selected_alpha_dynamic"),
        "dynamic_main_fusion_alpha_graph": dynamic_fusion.get("selected_alpha_graph"),
        "dynamic_main_fusion_delta_auc": dynamic_fusion.get("delta_auc_vs_base"),
        "dynamic_main_fusion_delta_auc_p_value": dynamic_fusion.get("delta_auc_p_value"),
        "dynamic_main_fusion_delta_brier": dynamic_fusion.get("delta_brier_vs_base"),
        "dynamic_event_status": dynamic_event.get("status"),
        "dynamic_event_n_events": dynamic_event.get("n_events"),
        "dynamic_sync_status": dynamic_sync.get("status"),
        "dynamic_sync_edges": dynamic_sync.get("n_edges"),
        "dynamic_sync_positive_edges": dynamic_sync.get("n_positive_edges"),
        "dynamic_sync_positive_edges_p_lt_0_10": dynamic_sync.get("n_positive_edges_p_lt_0_10"),
        "dynamic_sync_positive_edges_p_lt_0_05": dynamic_sync.get("n_positive_edges_p_lt_0_05"),
        "dynamic_policy_status": dynamic_policy.get("status"),
        "dynamic_policy_scenarios": dynamic_policy.get("n_scenarios"),
        "dynamic_policy_top_scenario": (dynamic_policy.get("scenarios_ranked") or [{}])[0].get("scenario")
        if isinstance(dynamic_policy, dict)
        else None,
        "dynamic_policy_top_scenario_mean_delta_h1": (dynamic_policy.get("scenarios_ranked") or [{}])[0].get("mean_delta_h1")
        if isinstance(dynamic_policy, dict)
        else None,
        "dynamic_policy_top_scenario_p_value": (dynamic_policy.get("scenarios_ranked") or [{}])[0].get("mean_delta_h1_p_value")
        if isinstance(dynamic_policy, dict)
        else None,
        "dynamic_policy_rl_status": dynamic_policy_rl.get("status"),
        "dynamic_policy_rl_entropy": dynamic_policy_rl.get("policy_entropy"),
        "dynamic_policy_rl_rule_selected_share": dynamic_policy_rl.get("rule_selected_share"),
        "dynamic_policy_rl_ope_delta_dr": (dynamic_policy_rl_ope.get("delta_vs_behavior") or {}).get("dr")
        if isinstance(dynamic_policy_rl_ope, dict)
        else None,
        "dynamic_policy_rl_ope_delta_snips": (dynamic_policy_rl_ope.get("delta_vs_behavior") or {}).get("snips")
        if isinstance(dynamic_policy_rl_ope, dict)
        else None,
        "dynamic_policy_rl_ope_ess": dynamic_policy_rl_ope.get("effective_sample_size")
        if isinstance(dynamic_policy_rl_ope, dict)
        else None,
        "dynamic_pulse_index_status": dynamic_pulse_index.get("status"),
        "dynamic_pulse_index_mean_latest": dynamic_pulse_index.get("latest_mean_index"),
        "model_selected_variant": model_metrics.get("selected_variant"),
        "model_sample_weight_mode": model_metrics.get("sample_weight_mode"),
    }


def run_experiment_enhancements(panel: pd.DataFrame, fast_mode: bool = False) -> Dict[str, Any]:
    """Run additional experiment robustness checks and uncertainty diagnostics."""
    df = _prepare_panel_for_benchmark(panel, target="composite_index")
    feature_cols = list(df.attrs.get("feature_cols", []))
    if len(df) < 200:
        out = {"status": "skipped", "reason": "too_few_rows_for_enhancement"}
        dump_json(DATA_OUTPUTS / "experiment_enhancements.json", out)
        return out

    n_boot = 150 if fast_mode else 500
    n_perm = 80 if fast_mode else 400
    run_loo_causal = not fast_mode

    temporal_df = _temporal_split_sensitivity(df, feature_cols=feature_cols)
    temporal_df.to_csv(DATA_OUTPUTS / "experiment_temporal_split_sensitivity.csv", index=False)

    spatial_df = _spatial_dispersion(df, feature_cols=feature_cols)
    spatial_df.to_csv(DATA_OUTPUTS / "experiment_spatial_ood_dispersion.csv", index=False)

    transfer = _continent_transfer_matrix(df)
    pulse = _pulse_uncertainty_and_calibration(n_boot=n_boot, random_state=42)
    pulse_fairness = _pulse_group_fairness(panel)
    pulse_decision = _pulse_decision_curve()
    did_panel, did_design = _resolve_did_design_panel(panel)
    did_treatment_year = _resolve_treatment_reference_year(did_panel, fallback=2020)
    stacked_placebo = None
    econ_path = DATA_OUTPUTS / "econometric_summary.json"
    if econ_path.exists():
        try:
            with econ_path.open("r", encoding="utf-8") as f:
                econ_payload = json.load(f)
            if isinstance(econ_payload, dict):
                stacked_placebo = econ_payload.get("did_stacked_lead_placebo")
        except Exception:  # noqa: BLE001
            stacked_placebo = None

    with _temporary_log_level("src.econometrics", logging.WARNING):
        placebo_df = _did_placebo_years(did_panel, stacked_placebo=stacked_placebo)
        placebo_df.to_csv(DATA_OUTPUTS / "experiment_did_placebo.csv", index=False)
        perm = _did_permutation_test(did_panel, n_perm=n_perm, random_state=42)
        spec_curve = _did_specification_curve(did_panel, fast_mode=fast_mode)
        neg_ctrl = _did_negative_controls(did_panel, treatment_year=int(did_treatment_year))
        loo_df = _leave_one_continent_out_stability(did_panel, include_causal=run_loo_causal)
        loo_df.to_csv(DATA_OUTPUTS / "experiment_leave_one_continent_out.csv", index=False)

    pretrend = _event_pretrend_diagnostics()
    matched_diag = _matched_did_diagnostics()
    staggered_diag = _staggered_did_diagnostics()
    nyt_diag = _not_yet_treated_did_diagnostics()
    id_score = _identification_scorecard_diagnostics()
    policy_source_diag = _policy_source_sensitivity_diagnostics()
    ai_incrementality_diag = _ai_incrementality_diagnostics()
    mechanism_diag = _mechanism_decomposition_diagnostics()
    dynamic_phase_diag = _dynamic_phase_heterogeneity_diagnostics()
    dynamic_phase_rule_diag = _dynamic_phase_rule_sensitivity_diagnostics()
    pulse_regime = _pulse_trajectory_regime_diagnostics()

    summary = {
        "n_rows": int(len(panel)),
        "n_cities": int(panel["city_id"].nunique()),
        "temporal_split_sensitivity": {
            "n_splits": int(len(temporal_df)),
            "linear_rmse_mean": float(temporal_df["linear_rmse"].mean()) if not temporal_df.empty else None,
            "linear_rmse_std": float(temporal_df["linear_rmse"].std()) if not temporal_df.empty else None,
            "rf_rmse_mean": float(temporal_df["rf_rmse"].mean()) if not temporal_df.empty else None,
            "rf_rmse_std": float(temporal_df["rf_rmse"].std()) if not temporal_df.empty else None,
        },
        "spatial_ood_dispersion": {
            "n_continents": int(spatial_df["left_out_continent"].nunique()) if not spatial_df.empty else 0,
            "linear_rmse_mean": float(spatial_df["linear_rmse"].mean()) if not spatial_df.empty else None,
            "linear_rmse_std": float(spatial_df["linear_rmse"].std()) if not spatial_df.empty else None,
            "rf_rmse_mean": float(spatial_df["rf_rmse"].mean()) if not spatial_df.empty else None,
            "rf_rmse_std": float(spatial_df["rf_rmse"].std()) if not spatial_df.empty else None,
        },
        "continent_transfer_matrix": transfer,
        "pulse_uncertainty_calibration": pulse,
        "pulse_group_fairness": pulse_fairness,
        "pulse_decision_curve": pulse_decision,
        "did_design": did_design,
        "did_design_rows": int(len(did_panel)),
        "did_treatment_reference_year": int(did_treatment_year),
        "did_placebo_summary": placebo_df.to_dict(orient="records"),
        "did_permutation": perm,
        "did_specification_curve": spec_curve,
        "did_negative_controls": neg_ctrl,
        "did_matched_diagnostics": matched_diag,
        "staggered_did_diagnostics": staggered_diag,
        "not_yet_treated_did_diagnostics": nyt_diag,
        "identification_scorecard_diagnostics": id_score,
        "policy_source_sensitivity_diagnostics": policy_source_diag,
        "ai_incrementality_diagnostics": ai_incrementality_diag,
        "mechanism_decomposition_diagnostics": mechanism_diag,
        "dynamic_phase_heterogeneity_diagnostics": dynamic_phase_diag,
        "dynamic_phase_rule_sensitivity_diagnostics": dynamic_phase_rule_diag,
        "pulse_trajectory_regime_diagnostics": pulse_regime,
        "leave_one_continent_out": loo_df.to_dict(orient="records"),
        "event_pretrend_diagnostics": pretrend,
    }
    dump_json(DATA_OUTPUTS / "experiment_enhancements.json", summary)
    LOGGER.info(
        "Experiment enhancements done: splits=%s, transfer=%s, perm=%s, specs=%s, loo=%s",
        len(temporal_df),
        transfer.get("n_pairs", 0) if isinstance(transfer, dict) else 0,
        perm.get("n_permutations_used", 0) if isinstance(perm, dict) else 0,
        spec_curve.get("n_specs", 0) if isinstance(spec_curve, dict) else 0,
        len(loo_df),
    )
    return summary
