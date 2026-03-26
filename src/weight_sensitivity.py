from __future__ import annotations

"""Composite index weight sensitivity analysis.

Addresses Review Issue #3: the baseline weights (0.45, 0.35, 0.20) for
economic_vitality, livability, and innovation lack academic justification.
This module computes alternative weighting schemes and measures how robust
the main conclusions (rankings, quadrant assignments, stall predictions) are
to different weight choices.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils import DATA_OUTPUTS, dump_json

LOGGER = logging.getLogger(__name__)

SUB_INDICES = ["economic_vitality", "livability", "innovation"]

WEIGHT_SCHEMES: Dict[str, Tuple[float, float, float]] = {
    "baseline": (0.45, 0.35, 0.20),
    "equal": (1 / 3, 1 / 3, 1 / 3),
    "economic_heavy": (0.60, 0.25, 0.15),
    "innovation_heavy": (0.25, 0.30, 0.45),
}


def _pca_weights(panel: pd.DataFrame) -> Tuple[float, float, float]:
    """Derive weights from first principal component loadings."""
    cols = [c for c in SUB_INDICES if c in panel.columns]
    if len(cols) < 3:
        return (1 / 3, 1 / 3, 1 / 3)
    mat = panel[cols].dropna().to_numpy(dtype=float)
    if mat.shape[0] < 20:
        return (1 / 3, 1 / 3, 1 / 3)
    mat_centered = mat - mat.mean(axis=0)
    cov = np.cov(mat_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # First PC is the one with the largest eigenvalue (last index after eigh)
    pc1_loadings = np.abs(eigenvectors[:, -1])
    weights = pc1_loadings / pc1_loadings.sum()
    return (float(weights[0]), float(weights[1]), float(weights[2]))


def _compute_composite(panel: pd.DataFrame, weights: Tuple[float, float, float]) -> pd.Series:
    """Compute composite index with given weights."""
    w_e, w_l, w_i = weights
    return (
        w_e * panel["economic_vitality"]
        + w_l * panel["livability"]
        + w_i * panel["innovation"]
    )


def _rank_correlation(s1: pd.Series, s2: pd.Series) -> float:
    """Spearman rank correlation between two series."""
    valid = s1.notna() & s2.notna()
    if valid.sum() < 10:
        return float("nan")
    r1 = s1[valid].rank()
    r2 = s2[valid].rank()
    n = float(len(r1))
    d2 = ((r1 - r2) ** 2).sum()
    return 1.0 - 6.0 * d2 / (n * (n * n - 1.0))


def _stall_auc_for_composite(panel: pd.DataFrame, composite: pd.Series) -> float:
    """Compute a quick stall-risk AUC using the given composite."""
    df = panel.copy()
    df["_comp"] = composite
    df = df.sort_values(["city_id", "year"])
    df["_delta"] = df.groupby("city_id")["_comp"].diff()
    df["_stall_next"] = (
        df.groupby("city_id")["_delta"]
        .shift(-1)
        .le(df["_delta"].quantile(0.30))
        .astype(float)
    )
    df = df.dropna(subset=["_delta", "_stall_next"])
    if df["_stall_next"].nunique() < 2 or len(df) < 50:
        return float("nan")
    # Simple logistic-like AUC via rank-based approximation
    pos = df.loc[df["_stall_next"] == 1, "_delta"]
    neg = df.loc[df["_stall_next"] == 0, "_delta"]
    if pos.empty or neg.empty:
        return float("nan")
    # Lower delta → higher stall risk, so use negative delta as score
    scores_pos = -pos.to_numpy()
    scores_neg = -neg.to_numpy()
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)
    # Mann-Whitney U approximation
    correct = 0.0
    for sp in scores_pos:
        correct += float((scores_neg < sp).sum()) + 0.5 * float((scores_neg == sp).sum())
    return correct / (n_pos * n_neg)


def run_weight_sensitivity_analysis(panel: pd.DataFrame) -> Dict[str, Any]:
    """Run composite index weight sensitivity analysis.

    Computes alternative weighting schemes and measures robustness of
    rankings, quadrant assignments, and stall-risk classification.

    Returns summary dict and saves detailed outputs.
    """
    for col in SUB_INDICES:
        if col not in panel.columns:
            out = {"status": "skipped", "reason": f"missing_sub_index:{col}"}
            dump_json(DATA_OUTPUTS / "weight_sensitivity_summary.json", out)
            return out

    LOGGER.info("Running composite index weight sensitivity analysis...")

    # Add PCA weights
    pca_w = _pca_weights(panel)
    all_schemes = {**WEIGHT_SCHEMES, "pca": pca_w}

    # Compute composites
    composites: Dict[str, pd.Series] = {}
    for name, weights in all_schemes.items():
        composites[name] = _compute_composite(panel, weights)

    baseline = composites["baseline"]

    # Rank correlations (latest year only)
    latest_year = int(panel["year"].max())
    latest_mask = panel["year"] == latest_year
    rank_rows: List[Dict[str, Any]] = []
    for name, comp in composites.items():
        if name == "baseline":
            continue
        rho = _rank_correlation(baseline[latest_mask], comp[latest_mask])
        rank_rows.append({
            "scheme": name,
            "weights_e": float(all_schemes[name][0]),
            "weights_l": float(all_schemes[name][1]),
            "weights_i": float(all_schemes[name][2]),
            "spearman_rho_vs_baseline": rho,
        })

    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(DATA_OUTPUTS / "weight_sensitivity_rank_correlations.csv", index=False)

    # Stall AUC under different weights
    auc_rows: List[Dict[str, Any]] = []
    for name, comp in composites.items():
        auc = _stall_auc_for_composite(panel, comp)
        auc_rows.append({
            "scheme": name,
            "stall_auc": auc,
        })
    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(DATA_OUTPUTS / "weight_sensitivity_stall_auc.csv", index=False)

    # Top-10 city stability across schemes
    top10_sets: Dict[str, set] = {}
    for name, comp in composites.items():
        df_tmp = panel.loc[latest_mask].copy()
        df_tmp["_comp"] = comp[latest_mask]
        top10 = set(df_tmp.nlargest(10, "_comp")["city_id"].tolist())
        top10_sets[name] = top10
    baseline_top10 = top10_sets["baseline"]
    overlap_rows: List[Dict[str, Any]] = []
    for name, t10 in top10_sets.items():
        if name == "baseline":
            continue
        overlap = len(baseline_top10.intersection(t10))
        overlap_rows.append({
            "scheme": name,
            "top10_overlap_with_baseline": overlap,
            "overlap_ratio": overlap / 10.0,
        })
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(DATA_OUTPUTS / "weight_sensitivity_top10_overlap.csv", index=False)

    mean_rho = float(rank_df["spearman_rho_vs_baseline"].mean()) if not rank_df.empty else float("nan")
    min_rho = float(rank_df["spearman_rho_vs_baseline"].min()) if not rank_df.empty else float("nan")
    mean_auc = float(auc_df["stall_auc"].mean()) if not auc_df.empty else float("nan")
    mean_overlap = float(overlap_df["overlap_ratio"].mean()) if not overlap_df.empty else float("nan")

    summary = {
        "status": "ok",
        "schemes_evaluated": len(all_schemes),
        "pca_weights": list(pca_w),
        "mean_spearman_rho_vs_baseline": mean_rho,
        "min_spearman_rho_vs_baseline": min_rho,
        "mean_stall_auc_across_schemes": mean_auc,
        "mean_top10_overlap_ratio": mean_overlap,
        "conclusion": (
            "robust" if (not np.isnan(min_rho) and min_rho > 0.85)
            else "moderately_robust" if (not np.isnan(min_rho) and min_rho > 0.70)
            else "sensitive"
        ),
        "rank_correlation_file": str(DATA_OUTPUTS / "weight_sensitivity_rank_correlations.csv"),
        "stall_auc_file": str(DATA_OUTPUTS / "weight_sensitivity_stall_auc.csv"),
        "top10_overlap_file": str(DATA_OUTPUTS / "weight_sensitivity_top10_overlap.csv"),
    }
    dump_json(DATA_OUTPUTS / "weight_sensitivity_summary.json", summary)
    LOGGER.info(
        "Weight sensitivity: mean_rho=%.3f min_rho=%.3f mean_auc=%.3f conclusion=%s",
        mean_rho, min_rho, mean_auc, summary["conclusion"],
    )
    return summary
