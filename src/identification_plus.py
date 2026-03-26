from __future__ import annotations

"""Identification-focused diagnostics for top-tier econometric presentation."""

import json
from itertools import combinations
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .utils import DATA_OUTPUTS, dump_json


def _safe_read_csv(path: str) -> pd.DataFrame:
    full = DATA_OUTPUTS / path
    if not full.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(full)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _safe_read_json(path: str) -> Dict[str, Any]:
    full = DATA_OUTPUTS / path
    if not full.exists():
        return {}
    try:
        with full.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _ols_slope_t(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    if n < 3:
        return float("nan"), float("nan")
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    xx = x - x_mean
    yy = y - y_mean
    sxx = float(np.sum(xx * xx))
    if sxx <= 0:
        return float("nan"), float("nan")
    slope = float(np.sum(xx * yy) / sxx)
    resid = y - (y_mean + slope * (x - x_mean))
    rss = float(np.sum(resid * resid))
    dof = n - 2
    if dof <= 0:
        return slope, float("nan")
    sigma2 = rss / dof
    se = float(np.sqrt(max(1e-12, sigma2 / sxx)))
    t_value = slope / se if se > 0 else float("nan")
    return slope, float(t_value)


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def _build_event_pretrend_geometry() -> pd.DataFrame:
    pts = _safe_read_csv("econometric_source_event_study_points.csv")
    econ = _safe_read_json("econometric_summary.json")

    # Include main design event-study points for direct comparability.
    main_pts = (econ.get("event_study_fe", {}) or {}).get("points", [])
    if isinstance(main_pts, list):
        rows: List[dict[str, Any]] = []
        for p in main_pts:
            if not isinstance(p, dict):
                continue
            rows.append(
                {
                    "variant": "main_design",
                    "rel_year": p.get("rel_year"),
                    "coef": p.get("coef"),
                    "stderr": p.get("stderr"),
                    "t_value": p.get("t_value"),
                    "source_channel": "main",
                    "design_variant": econ.get("main_design_variant"),
                }
            )
        if rows:
            pts = pd.concat([pts, pd.DataFrame(rows)], ignore_index=True)

    if pts.empty:
        return pd.DataFrame(
            columns=[
                "variant",
                "design_variant",
                "source_channel",
                "n_pre",
                "n_post",
                "pre_slope",
                "pre_slope_t",
                "pre_mean_abs_coef",
                "pre_max_abs_t",
                "post_mean_coef",
                "post_cum_coef",
                "post_peak_abs_coef",
                "pretrend_pass",
            ]
        )

    pts["rel_year"] = pd.to_numeric(pts.get("rel_year"), errors="coerce")
    pts["coef"] = pd.to_numeric(pts.get("coef"), errors="coerce")
    pts["t_value"] = pd.to_numeric(pts.get("t_value"), errors="coerce")
    pts = pts.dropna(subset=["rel_year", "coef"]).copy()
    if pts.empty:
        return pd.DataFrame()

    rows_out: List[dict[str, Any]] = []
    for variant, sub in pts.groupby("variant", dropna=False):
        pre = sub[sub["rel_year"] < 0].copy()
        post = sub[sub["rel_year"] >= 0].copy()
        slope, slope_t = _ols_slope_t(pre["rel_year"].to_numpy(dtype=float), pre["coef"].to_numpy(dtype=float))
        pre_max_abs_t = float(np.nanmax(np.abs(pre["t_value"].to_numpy(dtype=float)))) if (not pre.empty and pre["t_value"].notna().any()) else float("nan")
        post_cum = float(np.nansum(post["coef"].to_numpy(dtype=float))) if not post.empty else float("nan")
        row = {
            "variant": str(variant),
            "design_variant": str(sub["design_variant"].dropna().iloc[0]) if ("design_variant" in sub.columns and sub["design_variant"].notna().any()) else "",
            "source_channel": str(sub["source_channel"].dropna().iloc[0]) if ("source_channel" in sub.columns and sub["source_channel"].notna().any()) else "",
            "n_pre": int(len(pre)),
            "n_post": int(len(post)),
            "pre_slope": slope,
            "pre_slope_t": slope_t,
            "pre_mean_abs_coef": float(np.nanmean(np.abs(pre["coef"].to_numpy(dtype=float)))) if not pre.empty else float("nan"),
            "pre_max_abs_t": pre_max_abs_t,
            "post_mean_coef": float(np.nanmean(post["coef"].to_numpy(dtype=float))) if not post.empty else float("nan"),
            "post_cum_coef": post_cum,
            "post_peak_abs_coef": float(np.nanmax(np.abs(post["coef"].to_numpy(dtype=float)))) if not post.empty else float("nan"),
        }
        n_pre = int(row["n_pre"])
        if n_pre >= 4:
            pretrend_pass = bool(
                np.isfinite(row["pre_slope_t"])
                and np.isfinite(row["pre_max_abs_t"])
                and abs(float(row["pre_slope_t"])) <= 1.96
                and float(row["pre_max_abs_t"]) <= 2.0
            )
        elif n_pre >= 2:
            # With only two or three pre-period points, slope t-statistics are unstable
            # and can overstate smooth low-amplitude drift. Fall back to bounded
            # pre-period magnitude diagnostics instead.
            pretrend_pass = bool(
                np.isfinite(row["pre_mean_abs_coef"])
                and np.isfinite(row["pre_max_abs_t"])
                and float(row["pre_mean_abs_coef"]) <= 0.05
                and float(row["pre_max_abs_t"]) <= 2.0
            )
        else:
            pretrend_pass = False
        row["pretrend_pass"] = int(pretrend_pass)
        rows_out.append(row)

    out = pd.DataFrame(rows_out)
    if out.empty:
        return out
    return out.sort_values(["pretrend_pass", "pre_max_abs_t"], ascending=[False, True]).reset_index(drop=True)


def _build_design_concordance() -> tuple[pd.DataFrame, pd.DataFrame]:
    sens = _safe_read_csv("econometric_policy_source_sensitivity.csv")
    if sens.empty:
        return (
            pd.DataFrame(columns=["variant_i", "variant_j", "sign_agree", "ci_overlap", "sig_class_agree", "effect_distance_score", "concordance_score"]),
            pd.DataFrame(),
        )
    if "status" in sens.columns:
        sens = sens[sens["status"].astype(str) == "ok"].copy()
    if sens.empty:
        return pd.DataFrame(), pd.DataFrame()

    for c in ["did_coef", "did_stderr", "did_p_value"]:
        sens[c] = pd.to_numeric(sens.get(c), errors="coerce")
    sens = sens.dropna(subset=["did_coef"]).copy()
    if sens.empty:
        return pd.DataFrame(), pd.DataFrame()

    by_var = sens.set_index("variant")
    rows: List[dict[str, Any]] = []
    variants = by_var.index.tolist()
    for a, b in combinations(variants, 2):
        ra = by_var.loc[a]
        rb = by_var.loc[b]
        if isinstance(ra, pd.DataFrame):
            ra = ra.iloc[0]
        if isinstance(rb, pd.DataFrame):
            rb = rb.iloc[0]
        ca = float(ra["did_coef"])
        cb = float(rb["did_coef"])
        sa = float(ra["did_stderr"]) if np.isfinite(ra["did_stderr"]) else float("nan")
        sb = float(rb["did_stderr"]) if np.isfinite(rb["did_stderr"]) else float("nan")
        pa = float(ra["did_p_value"]) if np.isfinite(ra["did_p_value"]) else float("nan")
        pb = float(rb["did_p_value"]) if np.isfinite(rb["did_p_value"]) else float("nan")

        sign_agree = float(np.sign(ca) == np.sign(cb))
        ci_overlap = float("nan")
        if np.isfinite(sa) and np.isfinite(sb):
            lo_a, hi_a = ca - 1.96 * sa, ca + 1.96 * sa
            lo_b, hi_b = cb - 1.96 * sb, cb + 1.96 * sb
            ci_overlap = float(max(lo_a, lo_b) <= min(hi_a, hi_b))

        sig_a = pa < 0.10 if np.isfinite(pa) else np.nan
        sig_b = pb < 0.10 if np.isfinite(pb) else np.nan
        sig_class_agree = float(sig_a == sig_b) if (sig_a is not np.nan and sig_b is not np.nan) else float("nan")

        denom = abs(ca) + abs(cb) + 1e-6
        effect_distance_score = float(np.clip(1.0 - abs(ca - cb) / denom, 0.0, 1.0))

        parts = [sign_agree, ci_overlap, sig_class_agree, effect_distance_score]
        arr = np.array([p for p in parts if np.isfinite(p)], dtype=float)
        concordance = float(np.mean(arr)) if arr.size > 0 else float("nan")
        rows.append(
            {
                "variant_i": str(a),
                "variant_j": str(b),
                "sign_agree": sign_agree,
                "ci_overlap": ci_overlap,
                "sig_class_agree": sig_class_agree,
                "effect_distance_score": effect_distance_score,
                "concordance_score": concordance,
            }
        )

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        return pair_df, pd.DataFrame()
    pair_df = pair_df.sort_values("concordance_score", ascending=False).reset_index(drop=True)

    # Build symmetric matrix for heatmap.
    mat = pd.DataFrame(np.eye(len(variants), dtype=float), index=variants, columns=variants)
    for row in pair_df.itertuples(index=False):
        mat.loc[row.variant_i, row.variant_j] = float(row.concordance_score)
        mat.loc[row.variant_j, row.variant_i] = float(row.concordance_score)
    mat = mat.reset_index().rename(columns={"index": "variant"})
    return pair_df, mat


def _build_identification_stress(pretrend: pd.DataFrame) -> pd.DataFrame:
    sens = _safe_read_csv("econometric_policy_source_sensitivity.csv")
    if sens.empty:
        return pd.DataFrame(
            columns=[
                "variant",
                "design_variant",
                "did_coef",
                "did_t_value",
                "did_p_value",
                "identification_strength",
                "pretrend_pass",
                "pre_max_abs_t",
                "robustness_pass_rate",
                "signal_score",
                "credibility_score",
                "pretrend_score",
                "consistency_score",
                "resilience_score_0_100",
                "stress_risk_0_100",
                "stress_grade",
            ]
        )

    if "status" in sens.columns:
        sens = sens[sens["status"].astype(str) == "ok"].copy()
    if sens.empty:
        return pd.DataFrame()

    for c in [
        "did_coef",
        "did_t_value",
        "did_p_value",
        "bootstrap_p_value",
        "permutation_p_value",
        "wild_bootstrap_p_value",
        "lead_placebo_share_p_lt_0_10",
        "identification_strength",
        "effect_sign_consistent_with_main",
    ]:
        sens[c] = pd.to_numeric(sens.get(c), errors="coerce")

    pt = pretrend[["variant", "pretrend_pass", "pre_max_abs_t", "pre_slope_t"]].copy() if not pretrend.empty else pd.DataFrame(columns=["variant", "pretrend_pass", "pre_max_abs_t", "pre_slope_t"])
    out = sens.merge(pt, on="variant", how="left")

    checks = [
        out["did_p_value"] < 0.10,
        out["bootstrap_p_value"] < 0.10,
        out["permutation_p_value"] < 0.10,
        out["wild_bootstrap_p_value"] < 0.10,
        out["lead_placebo_share_p_lt_0_10"] <= 0.20,
    ]
    avail = np.zeros(len(out), dtype=float)
    passed = np.zeros(len(out), dtype=float)
    for cond in checks:
        mask = ~cond.isna()
        avail += mask.astype(float).to_numpy(dtype=float)
        passed += (cond.fillna(False) & mask).astype(float).to_numpy(dtype=float)
    out["robustness_pass_rate"] = np.where(avail > 0, passed / np.maximum(avail, 1.0), np.nan)

    out["signal_score"] = out["did_t_value"].abs().apply(lambda x: _clip01(float(x) / 3.0) if np.isfinite(x) else 0.0)
    out["credibility_score"] = out["identification_strength"].apply(lambda x: _clip01(float(x) / 100.0) if np.isfinite(x) else 0.0)
    out["pretrend_score"] = out["pre_max_abs_t"].apply(lambda x: _clip01(1.0 - float(x) / 3.0) if np.isfinite(x) else 0.0)
    out["consistency_score"] = out["effect_sign_consistent_with_main"].apply(lambda x: 1.0 if float(x) > 0 else 0.0 if np.isfinite(x) else 0.5)
    out["robustness_pass_rate"] = out["robustness_pass_rate"].apply(lambda x: _clip01(float(x)) if np.isfinite(x) else 0.0)
    out["pretrend_pass"] = pd.to_numeric(out.get("pretrend_pass"), errors="coerce").fillna(0.0)

    out["resilience_score_0_100"] = 100.0 * (
        0.34 * out["credibility_score"]
        + 0.26 * out["robustness_pass_rate"]
        + 0.18 * out["pretrend_score"]
        + 0.14 * out["signal_score"]
        + 0.08 * out["consistency_score"]
    )
    out["stress_risk_0_100"] = 100.0 - out["resilience_score_0_100"]

    grades: List[str] = []
    for v in out["resilience_score_0_100"].to_numpy(dtype=float):
        if v >= 75.0:
            grades.append("A")
        elif v >= 60.0:
            grades.append("B")
        elif v >= 45.0:
            grades.append("C")
        else:
            grades.append("D")
    out["stress_grade"] = grades

    keep = [
        "variant",
        "design_variant",
        "did_coef",
        "did_t_value",
        "did_p_value",
        "identification_strength",
        "pretrend_pass",
        "pre_max_abs_t",
        "robustness_pass_rate",
        "signal_score",
        "credibility_score",
        "pretrend_score",
        "consistency_score",
        "resilience_score_0_100",
        "stress_risk_0_100",
        "stress_grade",
    ]
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan
    out = out[keep].sort_values("resilience_score_0_100", ascending=False).reset_index(drop=True)
    return out


def _build_leave_continent_stability(main_did: float | None) -> pd.DataFrame:
    loo = _safe_read_csv("experiment_leave_one_continent_out.csv")
    if loo.empty:
        return pd.DataFrame(
            columns=[
                "n_splits",
                "did_mean",
                "did_std",
                "did_cv",
                "did_min",
                "did_max",
                "did_range",
                "sign_switch_count_vs_main",
                "causal_att_mean",
                "causal_att_std",
            ]
        )

    loo["did_coef"] = pd.to_numeric(loo.get("did_coef"), errors="coerce")
    loo["causal_st_att"] = pd.to_numeric(loo.get("causal_st_att"), errors="coerce")
    did = loo["did_coef"].dropna().to_numpy(dtype=float)
    att = loo["causal_st_att"].dropna().to_numpy(dtype=float)

    did_mean = float(np.mean(did)) if did.size > 0 else float("nan")
    did_std = float(np.std(did, ddof=0)) if did.size > 0 else float("nan")
    did_cv = float(did_std / (abs(did_mean) + 1e-6)) if did.size > 0 else float("nan")
    did_min = float(np.min(did)) if did.size > 0 else float("nan")
    did_max = float(np.max(did)) if did.size > 0 else float("nan")
    did_range = did_max - did_min if (np.isfinite(did_min) and np.isfinite(did_max)) else float("nan")

    if did.size > 0 and main_did is not None and np.isfinite(main_did):
        main_sign = np.sign(float(main_did))
        switch_count = int(np.sum(np.sign(did) != main_sign))
    else:
        switch_count = 0

    out = pd.DataFrame(
        [
            {
                "n_splits": int(len(loo)),
                "did_mean": did_mean,
                "did_std": did_std,
                "did_cv": did_cv,
                "did_min": did_min,
                "did_max": did_max,
                "did_range": did_range,
                "sign_switch_count_vs_main": switch_count,
                "causal_att_mean": float(np.mean(att)) if att.size > 0 else float("nan"),
                "causal_att_std": float(np.std(att, ddof=0)) if att.size > 0 else float("nan"),
            }
        ]
    )
    return out


def run_identification_plus_suite() -> Dict[str, Any]:
    """Build additional identification diagnostics used for paper reinforcement."""
    pretrend = _build_event_pretrend_geometry()
    concord_pair, concord_mat = _build_design_concordance()
    stress = _build_identification_stress(pretrend)

    main_did = None
    try:
        if not stress.empty and "main_design" in set(stress["variant"].astype(str)):
            main_did = float(stress.loc[stress["variant"].astype(str) == "main_design", "did_coef"].iloc[0])
    except Exception:  # noqa: BLE001
        main_did = None
    stability = _build_leave_continent_stability(main_did)

    pretrend_path = DATA_OUTPUTS / "idplus_event_pretrend_geometry.csv"
    pair_path = DATA_OUTPUTS / "idplus_design_concordance_pairs.csv"
    mat_path = DATA_OUTPUTS / "idplus_design_concordance_matrix.csv"
    stress_path = DATA_OUTPUTS / "idplus_identification_stress_index.csv"
    stability_path = DATA_OUTPUTS / "idplus_leave_continent_stability_summary.csv"

    pretrend.to_csv(pretrend_path, index=False)
    concord_pair.to_csv(pair_path, index=False)
    concord_mat.to_csv(mat_path, index=False)
    stress.to_csv(stress_path, index=False)
    stability.to_csv(stability_path, index=False)

    best_variant = ""
    best_score = float("nan")
    if not stress.empty:
        top = stress.iloc[0]
        best_variant = str(top.get("variant", ""))
        best_score = float(top.get("resilience_score_0_100", float("nan")))

    summary = {
        "status": "ok",
        "pretrend_rows": int(len(pretrend)),
        "concordance_pairs": int(len(concord_pair)),
        "stress_rows": int(len(stress)),
        "best_resilient_variant": best_variant,
        "best_resilience_score_0_100": best_score,
        "leave_continent_stability": stability.iloc[0].to_dict() if not stability.empty else {},
        "artifacts": {
            "pretrend_geometry_csv": str(pretrend_path),
            "concordance_pairs_csv": str(pair_path),
            "concordance_matrix_csv": str(mat_path),
            "stress_index_csv": str(stress_path),
            "leave_continent_stability_csv": str(stability_path),
        },
    }
    dump_json(DATA_OUTPUTS / "idplus_summary.json", summary)
    return summary
