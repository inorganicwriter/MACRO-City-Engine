from __future__ import annotations

"""AI explainability diagnostics for submission-ready reporting."""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score

from .utils import DATA_OUTPUTS, dump_json

LOGGER = logging.getLogger(__name__)


FEATURE_CANDIDATES: List[str] = [
    "acceleration_score",
    "growth_1",
    "accel_1",
    "critical_transition_score",
    "dynamic_hazard_score",
    "graph_diffusion_score",
    "dynamic_pulse_index",
    "regime_switch_rate_3y",
    "regime_forward_risk",
    "regime_transition_entropy",
    "stall_probability_base_pre_fusion",
]


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or len(y) < 3:
        return None
    sx = pd.Series(x).rank(method="average")
    sy = pd.Series(y).rank(method="average")
    corr = sx.corr(sy)
    if corr is None or pd.isna(corr):
        return None
    return float(corr)


def _counterfactual_feature_contrib(
    model: RandomForestClassifier,
    x_test: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    """Approximate SHAP-style contributions via leave-one-feature-out counterfactuals."""
    p_full = model.predict_proba(x_test)[:, 1]
    n_feat = x_test.shape[1]
    out = np.zeros(n_feat, dtype=float)
    for j in range(n_feat):
        x_cf = x_test.copy()
        x_cf[:, j] = baseline[j]
        p_cf = model.predict_proba(x_cf)[:, 1]
        out[j] = float(np.mean(np.abs(p_full - p_cf)))
    return out


def _yearly_importance_drift(df: pd.DataFrame, feature_cols: List[str], random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    years = sorted(df["year"].astype(int).unique().tolist())
    rng = np.random.default_rng(random_state)

    for year in years:
        sub = df[df["year"] == year].dropna(subset=feature_cols + ["stall_next"]).copy()
        if len(sub) < 80:
            continue
        y = sub["stall_next"].astype(int).to_numpy(dtype=int)
        if np.unique(y).size < 2:
            continue

        idx = np.arange(len(sub))
        rng.shuffle(idx)
        cut = int(round(len(sub) * 0.70))
        cut = max(40, min(cut, len(sub) - 20))
        tr_idx = idx[:cut]
        te_idx = idx[cut:]
        if len(tr_idx) < 40 or len(te_idx) < 20:
            continue

        x_all = sub[feature_cols].to_numpy(dtype=float)
        x_train = x_all[tr_idx]
        y_train = y[tr_idx]
        x_test = x_all[te_idx]
        y_test = y[te_idx]
        if np.unique(y_train).size < 2:
            continue
        if np.unique(y_test).size < 2:
            continue

        model = RandomForestClassifier(
            n_estimators=360,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=100 + year,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)

        perm = permutation_importance(
            model,
            x_test,
            y_test,
            n_repeats=10,
            random_state=year + 13,
            scoring="roc_auc",
        )
        pred = model.predict_proba(x_test)[:, 1]
        for i, feat in enumerate(feature_cols):
            rows.append(
                {
                    "year": int(year),
                    "feature": feat,
                    "importance_perm_auc": float(perm.importances_mean[i]),
                    "importance_tree_gain": float(model.feature_importances_[i]),
                    "n_train": int(len(tr_idx)),
                    "n_test": int(len(te_idx)),
                    "auc": float(roc_auc_score(y_test, pred)),
                    "brier": float(brier_score_loss(y_test, pred)),
                }
            )

    year_df = pd.DataFrame(rows)
    if year_df.empty:
        return year_df, pd.DataFrame()

    summary_rows: List[Dict[str, Any]] = []
    for feat, sub in year_df.groupby("feature", as_index=False):
        imp = pd.to_numeric(sub["importance_perm_auc"], errors="coerce")
        years_feat = pd.to_numeric(sub["year"], errors="coerce")
        slope = np.nan
        if imp.notna().sum() >= 3 and years_feat.notna().sum() >= 3:
            try:
                slope = float(np.polyfit(years_feat.to_numpy(dtype=float), imp.to_numpy(dtype=float), deg=1)[0])
            except Exception:  # noqa: BLE001
                slope = np.nan
        summary_rows.append(
            {
                "feature": str(feat),
                "year_count": int(sub["year"].nunique()),
                "mean_perm_auc_importance": float(imp.mean()),
                "std_perm_auc_importance": float(imp.std(ddof=0)),
                "min_perm_auc_importance": float(imp.min()),
                "max_perm_auc_importance": float(imp.max()),
                "slope_perm_auc_importance_per_year": slope,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_perm_auc_importance", "std_perm_auc_importance"], ascending=[False, False]
    )
    return year_df, summary_df


def run_ai_explainability_suite(panel: pd.DataFrame) -> Dict[str, Any]:
    """Build explainability consistency and feature-drift diagnostics."""
    path = DATA_OUTPUTS / "pulse_ai_scores.csv"
    if not path.exists():
        out = {"status": "skipped", "reason": "pulse_ai_scores_missing"}
        dump_json(DATA_OUTPUTS / "ai_explainability_summary.json", out)
        return out

    df = pd.read_csv(path)
    required = {"year", "stall_next", *FEATURE_CANDIDATES}
    feat = [c for c in FEATURE_CANDIDATES if c in df.columns]
    if len(feat) < 6:
        out = {"status": "skipped", "reason": "insufficient_explainability_features"}
        dump_json(DATA_OUTPUTS / "ai_explainability_summary.json", out)
        return out

    use = df[["year", "stall_next", *feat]].copy()
    use = use.dropna(subset=["stall_next"] + feat).copy()
    use["stall_next"] = pd.to_numeric(use["stall_next"], errors="coerce")
    use = use.dropna(subset=["stall_next"]).copy()
    use["stall_next"] = use["stall_next"].astype(int)
    if use.empty or use["stall_next"].nunique() < 2:
        out = {"status": "skipped", "reason": "invalid_stall_labels"}
        dump_json(DATA_OUTPUTS / "ai_explainability_summary.json", out)
        return out

    max_year = int(use["year"].max())
    eval_df = use[use["year"] < max_year].copy()
    if eval_df.empty or eval_df["stall_next"].nunique() < 2:
        eval_df = use.copy()

    years = sorted(eval_df["year"].astype(int).unique().tolist())
    if len(years) >= 4:
        split_year = int(years[-2])
    else:
        split_year = int(years[-1])
    train = eval_df[eval_df["year"] <= split_year].copy()
    test = eval_df[eval_df["year"] > split_year].copy()
    if len(test) < 80 or test["stall_next"].nunique() < 2:
        cut = int(round(len(eval_df) * 0.75))
        train = eval_df.iloc[:cut].copy()
        test = eval_df.iloc[cut:].copy()

    if len(train) < 120 or len(test) < 60:
        out = {"status": "skipped", "reason": "insufficient_rows_for_explainability"}
        dump_json(DATA_OUTPUTS / "ai_explainability_summary.json", out)
        return out

    if train["stall_next"].nunique() < 2 or test["stall_next"].nunique() < 2:
        # Fallback to a stratified random split when temporal split collapses to one class.
        try:
            from sklearn.model_selection import train_test_split

            tr, te = train_test_split(
                eval_df,
                test_size=0.25,
                random_state=42,
                stratify=eval_df["stall_next"].astype(int),
            )
            train = tr.copy()
            test = te.copy()
        except Exception:  # noqa: BLE001
            out = {"status": "skipped", "reason": "single_class_split_after_temporal_partition"}
            dump_json(DATA_OUTPUTS / "ai_explainability_summary.json", out)
            return out
        if train["stall_next"].nunique() < 2 or test["stall_next"].nunique() < 2:
            out = {"status": "skipped", "reason": "single_class_split_after_stratified_fallback"}
            dump_json(DATA_OUTPUTS / "ai_explainability_summary.json", out)
            return out

    x_train = train[feat].to_numpy(dtype=float)
    y_train = train["stall_next"].to_numpy(dtype=int)
    x_test = test[feat].to_numpy(dtype=float)
    y_test = test["stall_next"].to_numpy(dtype=int)

    model = RandomForestClassifier(
        n_estimators=520,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    p_test = model.predict_proba(x_test)[:, 1]

    perm = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=14,
        random_state=42,
        scoring="roc_auc" if np.unique(y_test).size >= 2 else "neg_brier_score",
    )
    perm_imp = perm.importances_mean.astype(float)

    shap_method = "counterfactual_leave_one_feature_out"
    shap_imp: np.ndarray | None = None
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)
        if isinstance(shap_values, list):
            vals = np.asarray(shap_values[-1], dtype=float)
        else:
            vals = np.asarray(shap_values, dtype=float)
        if vals.ndim == 3:  # e.g., multiclass
            vals = vals[:, :, -1]
        shap_imp = np.mean(np.abs(vals), axis=0)
        shap_method = "shap_tree_explainer"
    except Exception:  # noqa: BLE001
        baseline = np.nanmedian(x_train, axis=0).astype(float)
        shap_imp = _counterfactual_feature_contrib(model, x_test, baseline)

    if shap_imp is None or len(shap_imp) != len(feat):
        out = {"status": "skipped", "reason": "failed_to_build_shap_or_proxy_importance"}
        dump_json(DATA_OUTPUTS / "ai_explainability_summary.json", out)
        return out

    consistency = pd.DataFrame(
        {
            "feature": feat,
            "importance_permutation_auc": perm_imp,
            "importance_tree_gain": model.feature_importances_.astype(float),
            "importance_shap_like": shap_imp.astype(float),
        }
    )
    consistency["rank_permutation"] = consistency["importance_permutation_auc"].rank(ascending=False, method="min")
    consistency["rank_shap_like"] = consistency["importance_shap_like"].rank(ascending=False, method="min")
    consistency["rank_gap"] = (consistency["rank_permutation"] - consistency["rank_shap_like"]).abs()
    consistency = consistency.sort_values("rank_permutation")
    consistency.to_csv(DATA_OUTPUTS / "ai_explainability_consistency.csv", index=False)

    topk = min(5, len(consistency))
    top_perm = set(consistency.nsmallest(topk, "rank_permutation")["feature"].astype(str).tolist())
    top_shap = set(consistency.nsmallest(topk, "rank_shap_like")["feature"].astype(str).tolist())
    overlap = len(top_perm.intersection(top_shap))
    overlap_ratio = float(overlap / max(topk, 1))

    sp = _safe_spearman(
        consistency["importance_permutation_auc"].to_numpy(dtype=float),
        consistency["importance_shap_like"].to_numpy(dtype=float),
    )

    drift_year_df, drift_summary_df = _yearly_importance_drift(eval_df, feature_cols=feat, random_state=42)
    drift_year_df.to_csv(DATA_OUTPUTS / "ai_feature_importance_by_year.csv", index=False)
    drift_summary_df.to_csv(DATA_OUTPUTS / "ai_feature_importance_drift_summary.csv", index=False)

    # Stability across years using ranked permutation importances.
    mean_rank_corr = None
    if (not drift_year_df.empty) and drift_year_df["year"].nunique() >= 3:
        mat = (
            drift_year_df.pivot_table(
                index="feature",
                columns="year",
                values="importance_perm_auc",
                aggfunc="mean",
            )
            .fillna(0.0)
            .sort_index()
        )
        year_cols = mat.columns.tolist()
        pair_corr: List[float] = []
        for i in range(len(year_cols)):
            for j in range(i + 1, len(year_cols)):
                c = _safe_spearman(
                    mat[year_cols[i]].to_numpy(dtype=float),
                    mat[year_cols[j]].to_numpy(dtype=float),
                )
                if c is not None:
                    pair_corr.append(float(c))
        if pair_corr:
            mean_rank_corr = float(np.mean(pair_corr))

    auc = float(roc_auc_score(y_test, p_test)) if np.unique(y_test).size >= 2 else None
    brier = float(brier_score_loss(y_test, p_test))
    summary = {
        "status": "ok",
        "feature_count": int(len(feat)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "evaluation_auc": auc,
        "evaluation_brier": brier,
        "shap_method": shap_method,
        "permutation_shap_spearman": sp,
        "top5_overlap_count": int(overlap),
        "top5_overlap_ratio": overlap_ratio,
        "mean_yearly_rank_corr": mean_rank_corr,
        "consistency_file": str(DATA_OUTPUTS / "ai_explainability_consistency.csv"),
        "year_importance_file": str(DATA_OUTPUTS / "ai_feature_importance_by_year.csv"),
        "year_drift_summary_file": str(DATA_OUTPUTS / "ai_feature_importance_drift_summary.csv"),
    }
    dump_json(DATA_OUTPUTS / "ai_explainability_summary.json", summary)
    LOGGER.info(
        "AI explainability suite done: spearman=%.3f, top5 overlap=%.2f, method=%s",
        summary["permutation_shap_spearman"] if summary["permutation_shap_spearman"] is not None else float("nan"),
        summary["top5_overlap_ratio"],
        summary["shap_method"],
    )
    return summary
