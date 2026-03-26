from __future__ import annotations

"""Build top-tier reinforcement diagnostics from existing project outputs."""

import json
from typing import Any, Dict, Iterable, List

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


def _to_float(value: Any, default: float = float("nan")) -> float:
    """Convert value to float with None/invalid safety."""
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _minmax(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    if x.empty:
        return x
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    return (x - lo) / (hi - lo)


def _pareto_mask(df: pd.DataFrame, maximize_cols: List[str], minimize_cols: List[str]) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=bool)

    vals_max = [pd.to_numeric(df[c], errors="coerce").fillna(-np.inf).to_numpy(dtype=float) for c in maximize_cols]
    vals_min = [pd.to_numeric(df[c], errors="coerce").fillna(np.inf).to_numpy(dtype=float) for c in minimize_cols]
    n = len(df)
    efficient = np.ones(n, dtype=bool)

    for i in range(n):
        if not efficient[i]:
            continue
        dominates_i = np.ones(n, dtype=bool)
        strictly_better = np.zeros(n, dtype=bool)

        for arr in vals_max:
            dominates_i &= arr >= arr[i]
            strictly_better |= arr > arr[i]
        for arr in vals_min:
            dominates_i &= arr <= arr[i]
            strictly_better |= arr < arr[i]

        dominates_i[i] = False
        if np.any(dominates_i & strictly_better):
            efficient[i] = False
    return efficient


def _innovation_frontier_table() -> pd.DataFrame:
    pareto = _safe_read_csv("pulse_ai_dynamic_main_fusion_pareto.csv")
    if pareto.empty:
        return pd.DataFrame(
            columns=[
                "candidate",
                "alpha_dynamic",
                "alpha_graph",
                "alpha_critical",
                "alpha_base",
                "eval_roc_auc",
                "eval_average_precision",
                "eval_brier",
                "temporal_mean_gain_vs_base",
                "temporal_share_positive_gain",
                "is_pareto_efficient",
                "frontier_score",
                "frontier_rank",
            ]
        )

    keep = [
        "candidate",
        "alpha_dynamic",
        "alpha_graph",
        "alpha_critical",
        "alpha_base",
        "eval_roc_auc",
        "eval_average_precision",
        "eval_brier",
        "temporal_mean_gain_vs_base",
        "temporal_share_positive_gain",
    ]
    for col in keep:
        if col not in pareto.columns:
            pareto[col] = np.nan
    out = pareto[keep].copy()

    out["eval_roc_auc"] = pd.to_numeric(out["eval_roc_auc"], errors="coerce")
    out["eval_average_precision"] = pd.to_numeric(out["eval_average_precision"], errors="coerce")
    out["eval_brier"] = pd.to_numeric(out["eval_brier"], errors="coerce")
    out["temporal_mean_gain_vs_base"] = pd.to_numeric(out["temporal_mean_gain_vs_base"], errors="coerce")
    out["temporal_share_positive_gain"] = pd.to_numeric(out["temporal_share_positive_gain"], errors="coerce")

    out = out.dropna(subset=["eval_roc_auc", "eval_average_precision", "eval_brier"]).copy()
    if out.empty:
        return out

    m_auc = _minmax(out["eval_roc_auc"])
    m_ap = _minmax(out["eval_average_precision"])
    m_brier = _minmax(-out["eval_brier"])
    m_gain = _minmax(out["temporal_mean_gain_vs_base"])
    m_share = _minmax(out["temporal_share_positive_gain"])
    out["frontier_score"] = (
        0.32 * m_auc + 0.28 * m_ap + 0.20 * m_brier + 0.12 * m_gain + 0.08 * m_share
    )
    out["frontier_score"] = pd.to_numeric(out["frontier_score"], errors="coerce").fillna(0.0)

    out["is_pareto_efficient"] = _pareto_mask(
        out,
        maximize_cols=["eval_roc_auc"],
        minimize_cols=["eval_brier"],
    ).astype(int)
    out = out.sort_values(["is_pareto_efficient", "frontier_score"], ascending=[False, False]).reset_index(drop=True)
    out["frontier_rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def _adaptive_innovation_candidate() -> Dict[str, Any]:
    cont = _safe_read_csv("pulse_ai_dynamic_main_fusion_continent_eval.csv")
    if cont.empty:
        return {"status": "skipped", "reason": "continent_eval_missing"}

    needed = {"n_eval", "selected_roc_auc", "selected_brier"}
    if not needed.issubset(set(cont.columns)):
        return {"status": "skipped", "reason": "continent_eval_columns_missing"}

    work = cont.copy()
    for col in ["n_eval", "selected_roc_auc", "selected_brier", "delta_auc"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["n_eval", "selected_roc_auc", "selected_brier"]).copy()
    work = work[work["n_eval"] > 0].copy()
    if work.empty:
        return {"status": "skipped", "reason": "continent_eval_invalid"}

    weights = work["n_eval"].to_numpy(dtype=float)
    selected_auc = float(np.average(work["selected_roc_auc"].to_numpy(dtype=float), weights=weights))
    selected_brier = float(np.average(work["selected_brier"].to_numpy(dtype=float), weights=weights))
    delta_auc = float(np.average(work["delta_auc"].fillna(0.0).to_numpy(dtype=float), weights=weights))
    share_pos = float(np.mean(work["delta_auc"].fillna(0.0).to_numpy(dtype=float) > 0.0))

    return {
        "status": "ok",
        "candidate": "geo_adapted_selected",
        "eval_roc_auc": selected_auc,
        "eval_brier": selected_brier,
        "eval_average_precision": np.nan,
        "weighted_delta_auc": delta_auc,
        "share_positive_continent_gain": share_pos,
        "n_continents": int(work["continent"].astype(str).nunique()) if "continent" in work.columns else int(len(work)),
        "n_eval_total": int(work["n_eval"].sum()),
    }


def _identification_spectrum_table() -> pd.DataFrame:
    sens = _safe_read_csv("econometric_policy_source_sensitivity.csv")
    if sens.empty:
        return pd.DataFrame(
            columns=[
                "variant",
                "design_variant",
                "did_coef",
                "did_stderr",
                "did_t_value",
                "did_p_value",
                "preferred_estimator",
                "preferred_effect",
                "preferred_p_value",
                "identification_strength",
                "robust_checks_available",
                "robust_checks_passed",
                "robust_pass_rate",
                "effect_sign_consistent_with_main",
                "evidence_grade",
            ]
        )

    if "status" in sens.columns:
        sens = sens[sens["status"].astype(str) == "ok"].copy()
    if sens.empty:
        return pd.DataFrame()

    keep = [
        "variant",
        "design_variant",
        "did_coef",
        "did_stderr",
        "did_t_value",
        "did_p_value",
        "bootstrap_p_value",
        "permutation_p_value",
        "wild_bootstrap_p_value",
        "lead_placebo_share_p_lt_0_10",
        "preferred_estimator",
        "preferred_effect",
        "preferred_p_value",
        "identification_strength",
        "effect_sign_consistent_with_main",
    ]
    for col in keep:
        if col not in sens.columns:
            sens[col] = np.nan
    out = sens[keep].copy()

    num_cols = [
        "did_coef",
        "did_stderr",
        "did_t_value",
        "did_p_value",
        "bootstrap_p_value",
        "permutation_p_value",
        "wild_bootstrap_p_value",
        "lead_placebo_share_p_lt_0_10",
        "preferred_p_value",
        "identification_strength",
        "effect_sign_consistent_with_main",
    ]
    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    available = np.zeros(len(out), dtype=int)
    passed = np.zeros(len(out), dtype=int)
    check_specs = [
        ("did_p_value", lambda s: s < 0.10),
        ("bootstrap_p_value", lambda s: s < 0.10),
        ("permutation_p_value", lambda s: s < 0.10),
        ("wild_bootstrap_p_value", lambda s: s < 0.10),
        ("lead_placebo_share_p_lt_0_10", lambda s: s <= 0.20),
        ("preferred_p_value", lambda s: s < 0.10),
    ]
    for col, rule in check_specs:
        src = pd.to_numeric(out[col], errors="coerce")
        mask_avail = src.notna()
        cond = rule(src)
        available += mask_avail.astype(int).to_numpy(dtype=int)
        passed += (cond.fillna(False) & mask_avail).astype(int).to_numpy(dtype=int)
    out["robust_checks_available"] = available
    out["robust_checks_passed"] = passed
    out["robust_pass_rate"] = np.where(available > 0, passed / np.maximum(available, 1), np.nan)

    grade = []
    for row in out.itertuples(index=False):
        strength = float(row.identification_strength) if np.isfinite(row.identification_strength) else 0.0
        pass_rate = float(row.robust_pass_rate) if np.isfinite(row.robust_pass_rate) else 0.0
        composite = 0.65 * (strength / 100.0) + 0.35 * pass_rate
        if composite >= 0.78:
            grade.append("A")
        elif composite >= 0.62:
            grade.append("B")
        elif composite >= 0.48:
            grade.append("C")
        else:
            grade.append("D")
    out["evidence_grade"] = grade

    # Mark primary identification tracks used for main causal claims.
    primary_variants = {"direct_event_design", "evidence_evidence_a", "main_design"}
    out["is_primary_track"] = out["variant"].astype(str).isin(primary_variants).astype(int)

    out = out.sort_values(["identification_strength", "robust_pass_rate"], ascending=[False, False]).reset_index(drop=True)
    return out


def _dynamic_envelope_snapshot() -> Dict[str, float]:
    summary = _safe_read_json("dynamic_causal_envelope_summary.json")
    city = _safe_read_csv("dynamic_causal_envelope_city_scores.csv")
    cont = _safe_read_csv("dynamic_causal_envelope_continent_stability.csv")

    def _f(key: str) -> float:
        return float(summary.get(key, np.nan)) if isinstance(summary, dict) else np.nan

    top10_mean = np.nan
    top10_risk_mean = np.nan
    if not city.empty and "envelope_score_0_100" in city.columns:
        c = city.copy()
        c["envelope_score_0_100"] = pd.to_numeric(c["envelope_score_0_100"], errors="coerce")
        c["stall_risk_score"] = pd.to_numeric(c.get("stall_risk_score"), errors="coerce")
        c = c.dropna(subset=["envelope_score_0_100"]).copy()
        if not c.empty:
            top = c.nlargest(10, "envelope_score_0_100")
            top10_mean = float(pd.to_numeric(top["envelope_score_0_100"], errors="coerce").mean())
            if "stall_risk_score" in top.columns:
                top10_risk_mean = float(pd.to_numeric(top["stall_risk_score"], errors="coerce").mean())

    top_cont_stability = np.nan
    cont_mean_stability = np.nan
    if not cont.empty and "stability_score_0_100" in cont.columns:
        st = pd.to_numeric(cont["stability_score_0_100"], errors="coerce")
        if st.notna().any():
            top_cont_stability = float(st.max())
            cont_mean_stability = float(st.mean())

    return {
        "event_post_weighted_mean": _f("event_post_weighted_mean"),
        "event_pre_abs_median": _f("event_pre_abs_median"),
        "event_post_iqr_band_mean": _f("event_post_iqr_band_mean"),
        "event_post_ci_above_zero_share": _f("event_post_ci_above_zero_share"),
        "event_post_signal_to_noise_mean": _f("event_post_signal_to_noise_mean"),
        "resilient_accelerator_share": _f("resilient_accelerator_share"),
        "stall_trap_share": _f("stall_trap_share"),
        "continents_covered": _f("continents_covered"),
        "top10_envelope_score_mean": top10_mean,
        "top10_envelope_risk_mean": top10_risk_mean,
        "top_continent_stability_score": top_cont_stability,
        "mean_continent_stability_score": (
            _f("mean_continent_stability_score")
            if np.isfinite(_f("mean_continent_stability_score"))
            else cont_mean_stability
        ),
    }


def _pulse_dynamics_snapshot() -> Dict[str, float]:
    summary = _safe_read_json("pulse_dynamics_summary.json")
    warning = _safe_read_csv("pulse_dynamics_warning_horizon.csv")
    resilience = _safe_read_csv("pulse_dynamics_resilience_halflife.csv")
    transition = _safe_read_csv("pulse_dynamics_transition_tensor.csv")

    def _f(key: str) -> float:
        return float(summary.get(key, np.nan)) if isinstance(summary, dict) else np.nan

    def _global_warning(horizon: int, col: str) -> float:
        if warning.empty or col not in warning.columns:
            return np.nan
        sub = warning.copy()
        sub["scope"] = sub.get("scope", "").astype(str)
        sub["horizon_years"] = pd.to_numeric(sub.get("horizon_years"), errors="coerce")
        sub[col] = pd.to_numeric(sub.get(col), errors="coerce")
        sub = sub[(sub["scope"] == "global") & (sub["horizon_years"] == int(horizon))].copy()
        if sub.empty:
            return np.nan
        return float(sub.iloc[0][col])

    stall_half = np.nan
    accel_half = np.nan
    resilience_score = np.nan
    if not resilience.empty:
        rs = resilience.copy()
        rs["scope"] = rs.get("scope", "").astype(str)
        rs["stall_exit_half_life"] = pd.to_numeric(rs.get("stall_exit_half_life"), errors="coerce")
        rs["accel_break_half_life"] = pd.to_numeric(rs.get("accel_break_half_life"), errors="coerce")
        rs["resilience_score_0_100"] = pd.to_numeric(rs.get("resilience_score_0_100"), errors="coerce")
        g = rs[rs["scope"] == "global"].copy()
        if not g.empty:
            stall_half = float(g.iloc[0]["stall_exit_half_life"])
            accel_half = float(g.iloc[0]["accel_break_half_life"])
            resilience_score = float(g.iloc[0]["resilience_score_0_100"])

    transition_cov = np.nan
    if not transition.empty and {"scope", "continent"}.issubset(transition.columns):
        c = transition[transition["scope"].astype(str) == "continent"].copy()
        if not c.empty:
            transition_cov = float(c["continent"].astype(str).nunique())

    return {
        "transition_continent_coverage": (
            _f("transition_continent_coverage")
            if np.isfinite(_f("transition_continent_coverage"))
            else transition_cov
        ),
        "global_stall_entry_probability": _f("global_stall_entry_probability"),
        "global_stall_exit_half_life": (
            _f("global_stall_exit_half_life")
            if np.isfinite(_f("global_stall_exit_half_life"))
            else stall_half
        ),
        "global_accel_break_half_life": (
            _f("global_accel_break_half_life")
            if np.isfinite(_f("global_accel_break_half_life"))
            else accel_half
        ),
        "global_resilience_score_0_100": (
            _f("global_resilience_score_0_100")
            if np.isfinite(_f("global_resilience_score_0_100"))
            else resilience_score
        ),
        "global_warning_auc_h1": (
            _f("global_warning_auc_h1")
            if np.isfinite(_f("global_warning_auc_h1"))
            else _global_warning(1, "auc")
        ),
        "global_warning_auc_h2": (
            _f("global_warning_auc_h2")
            if np.isfinite(_f("global_warning_auc_h2"))
            else _global_warning(2, "auc")
        ),
        "global_warning_auc_h3": (
            _f("global_warning_auc_h3")
            if np.isfinite(_f("global_warning_auc_h3"))
            else _global_warning(3, "auc")
        ),
        "global_warning_lift_h1": (
            _f("global_warning_lift_h1")
            if np.isfinite(_f("global_warning_lift_h1"))
            else _global_warning(1, "top_decile_lift")
        ),
        "global_warning_lift_h2": (
            _f("global_warning_lift_h2")
            if np.isfinite(_f("global_warning_lift_h2"))
            else _global_warning(2, "top_decile_lift")
        ),
        "global_warning_lift_h3": (
            _f("global_warning_lift_h3")
            if np.isfinite(_f("global_warning_lift_h3"))
            else _global_warning(3, "top_decile_lift")
        ),
    }


def _pulse_nowcast_snapshot() -> Dict[str, float]:
    summary = _safe_read_json("pulse_nowcast_summary.json")
    latest = _safe_read_csv("pulse_nowcast_continent_latest.csv")

    def _f(key: str) -> float:
        return float(summary.get(key, np.nan)) if isinstance(summary, dict) else np.nan

    cont_cov = _f("continents_covered")
    mean_dec = _f("mean_continent_prob_decelerate")
    mean_acc = _f("mean_continent_prob_accelerate")
    if (not np.isfinite(cont_cov)) and (not latest.empty) and ("continent" in latest.columns):
        cont_cov = float(latest["continent"].astype(str).nunique())
    if (not np.isfinite(mean_dec)) and (not latest.empty) and ("prob_decelerate" in latest.columns):
        mean_dec = float(pd.to_numeric(latest["prob_decelerate"], errors="coerce").mean())
    if (not np.isfinite(mean_acc)) and (not latest.empty) and ("prob_accelerate" in latest.columns):
        mean_acc = float(pd.to_numeric(latest["prob_accelerate"], errors="coerce").mean())

    return {
        "continents_covered": cont_cov,
        "global_backtest_r2": _f("global_backtest_r2"),
        "global_backtest_coverage95": _f("global_backtest_coverage95"),
        "global_backtest_directional_accuracy": _f("global_backtest_directional_accuracy"),
        "global_prob_decelerate": _f("global_prob_decelerate"),
        "global_prob_accelerate": _f("global_prob_accelerate"),
        "mean_continent_prob_decelerate": mean_dec,
        "mean_continent_prob_accelerate": mean_acc,
    }


def _dynamic_method_core_snapshot() -> Dict[str, float]:
    summary = _safe_read_json("dynamic_method_core_summary.json")
    signif = _safe_read_csv("dynamic_method_core_significance.csv")
    metrics = _safe_read_csv("dynamic_method_core_metrics.csv")

    def _f(path: Iterable[str]) -> float:
        cur: Any = summary
        for k in path:
            if not isinstance(cur, dict):
                return np.nan
            cur = cur.get(k)
        try:
            return float(cur)
        except Exception:  # noqa: BLE001
            return np.nan

    sig_abs = np.nan
    sig_dir = np.nan
    if not signif.empty:
        s = signif.copy()
        s["scope"] = s.get("scope", "").astype(str)
        s["continent"] = s.get("continent", "").astype(str)
        s["baseline_model"] = s.get("baseline_model", "").astype(str)
        s["abs_error_p_value_improve"] = pd.to_numeric(s.get("abs_error_p_value_improve"), errors="coerce")
        s["direction_p_value_improve"] = pd.to_numeric(s.get("direction_p_value_improve"), errors="coerce")
        g = s[
            (s["scope"] == "global")
            & (s["continent"] == "Global")
            & (s["baseline_model"] == "ridge_ar2")
        ].copy()
        if not g.empty:
            sig_abs = float(g.iloc[0]["abs_error_p_value_improve"])
            sig_dir = float(g.iloc[0]["direction_p_value_improve"])

    main_coverage95 = np.nan
    if not metrics.empty:
        m = metrics.copy()
        m["scope"] = m.get("scope", "").astype(str)
        m["continent"] = m.get("continent", "").astype(str)
        m["model"] = m.get("model", "").astype(str)
        m["coverage95"] = pd.to_numeric(m.get("coverage95"), errors="coerce")
        g = m[
            (m["scope"] == "global")
            & (m["continent"] == "Global")
            & (m["model"] == "pulse_ai_dynamic_ensemble")
        ].copy()
        if not g.empty:
            main_coverage95 = float(g.iloc[0]["coverage95"])

    return {
        "continents_covered": _f(["continents_covered"]),
        "cities_covered": _f(["cities_covered"]),
        "eval_years": _f(["eval_years"]),
        "main_mae": _f(["global_main_metrics", "mae"]),
        "main_rmse": _f(["global_main_metrics", "rmse"]),
        "main_ternary_accuracy": _f(["global_main_metrics", "ternary_accuracy"]),
        "main_coverage95": (
            _f(["global_main_metrics", "coverage95"])
            if np.isfinite(_f(["global_main_metrics", "coverage95"]))
            else main_coverage95
        ),
        "mae_skill_vs_naive": _f(["global_main_metrics", "mae_skill_vs_naive"]),
        "mae_gain_vs_ridge_ar2": _f(["global_vs_ridge_ar2", "mae_gain"]),
        "ternary_gain_vs_ridge_ar2": _f(["global_vs_ridge_ar2", "ternary_accuracy_gain"]),
        "p_value_mae_improve_vs_ridge_ar2": (
            _f(["global_vs_ridge_ar2", "abs_error_p_value"])
            if np.isfinite(_f(["global_vs_ridge_ar2", "abs_error_p_value"]))
            else sig_abs
        ),
        "p_value_ternary_improve_vs_ridge_ar2": (
            _f(["global_vs_ridge_ar2", "ternary_accuracy_p_value"])
            if np.isfinite(_f(["global_vs_ridge_ar2", "ternary_accuracy_p_value"]))
            else sig_dir
        ),
    }


def _evidence_convergence_table(id_spectrum: pd.DataFrame | None = None) -> pd.DataFrame:
    bench = _safe_read_json("benchmark_scores.json")
    pulse = _safe_read_json("pulse_ai_summary.json")
    external = _safe_read_json("external_validity_summary.json")
    inference = _safe_read_json("inference_reporting_summary.json")
    sens = _safe_read_csv("econometric_policy_source_sensitivity.csv")
    calib = _safe_read_csv("experiment_pulse_calibration_bins.csv")
    dce = _dynamic_envelope_snapshot()
    dynamics = _pulse_dynamics_snapshot()
    nowcast = _pulse_nowcast_snapshot()
    method_core = _dynamic_method_core_snapshot()

    temporal_r2 = _to_float(((bench.get("temporal_holdout", {}) or {}).get("linear", {}) or {}).get("r2", np.nan))
    spatial_r2 = _to_float((bench.get("spatial_ood", {}) or {}).get("mean_linear_r2", np.nan))
    roc_auc = _to_float((pulse.get("model_metrics", {}) or {}).get("roc_auc", np.nan))
    avg_abs_t = _to_float(external.get("avg_twfe_abs_t", np.nan))
    avg_uplift = _to_float(external.get("avg_predictive_r2_uplift", np.nan))

    rl = (((pulse.get("dynamic_structure") or {}).get("policy_rl") or {}).get("offline_policy_evaluation") or {})
    rl_ci = ((rl.get("uplift_ci") or {}).get("dr") or {})
    rl_ci_low = _to_float(rl_ci.get("delta_ci_low", np.nan))
    rl_share_pos = _to_float(rl.get("continent_dr_share_positive", np.nan))
    rl_dr = _to_float((rl.get("delta_vs_behavior") or {}).get("dr", np.nan))

    if not calib.empty and {"pred", "obs", "n"}.issubset(calib.columns):
        pred = pd.to_numeric(calib["pred"], errors="coerce")
        obs = pd.to_numeric(calib["obs"], errors="coerce")
        n = pd.to_numeric(calib["n"], errors="coerce").fillna(0.0)
        ece = float(np.nansum(np.abs(pred - obs) * n) / max(1.0, float(np.nansum(n))))
    else:
        ece = np.nan

    id_strength = np.nan
    id_robust = np.nan
    if id_spectrum is not None and not id_spectrum.empty:
        grade_weight = {"A": 1.20, "B": 1.00, "C": 0.70, "D": 0.40}
        spec = id_spectrum.copy()
        if "is_primary_track" in spec.columns:
            primary = spec[pd.to_numeric(spec["is_primary_track"], errors="coerce").fillna(0).astype(int) == 1].copy()
            if primary.empty:
                primary = spec.copy()
        else:
            primary = spec.copy()
        primary["grade_weight"] = primary.get("evidence_grade", "").astype(str).map(grade_weight).fillna(0.60)
        weights = pd.to_numeric(primary["grade_weight"], errors="coerce").fillna(0.60).to_numpy(dtype=float)
        strengths = pd.to_numeric(primary.get("identification_strength"), errors="coerce").to_numpy(dtype=float)
        robusts = pd.to_numeric(primary.get("robust_pass_rate"), errors="coerce").to_numpy(dtype=float)
        mask_s = np.isfinite(strengths) & np.isfinite(weights) & (weights > 0)
        if np.any(mask_s):
            id_strength = float(np.average(strengths[mask_s], weights=weights[mask_s]))
        mask_r = np.isfinite(robusts) & np.isfinite(weights) & (weights > 0)
        if np.any(mask_r):
            id_robust = float(np.average(robusts[mask_r], weights=weights[mask_r]))
    elif not sens.empty:
        if "status" in sens.columns:
            sens = sens[sens["status"].astype(str) == "ok"].copy()
        if "identification_strength" in sens.columns:
            id_strength = float(pd.to_numeric(sens["identification_strength"], errors="coerce").dropna().max())

    inference_status_ok = str(inference.get("status", "")) == "ok"

    def clip01(x: float) -> float:
        if not np.isfinite(x):
            return 0.0
        return float(np.clip(x, 0.0, 1.0))

    score_temporal = 100.0 * clip01(temporal_r2)
    score_spatial = 100.0 * clip01(spatial_r2)
    score_pred = 100.0 * clip01((roc_auc - 0.5) / 0.5)
    score_calib = 100.0 * clip01(1.0 - ece)
    # Causal score combines identification magnitude and robustness pass ratio.
    score_causal = 100.0 * (
        0.60 * clip01(id_strength / 100.0) + 0.40 * clip01(id_robust if np.isfinite(id_robust) else 0.0)
    )
    score_external = 100.0 * (0.55 * clip01(avg_abs_t / 3.0) + 0.45 * clip01(avg_uplift / 0.03))
    score_rl = 100.0 * (0.55 * clip01(rl_share_pos) + 0.45 * float(rl_ci_low > 0.0))
    score_dce = 100.0 * (
        0.25 * clip01((dce["event_post_weighted_mean"] + 0.05) / 0.30)
        + 0.15 * clip01(1.0 - (dce["event_pre_abs_median"] / 0.12))
        + 0.12 * clip01(1.0 - (dce["event_post_iqr_band_mean"] / 0.18))
        + 0.15 * clip01(dce["event_post_ci_above_zero_share"])
        + 0.13 * clip01((dce["event_post_signal_to_noise_mean"] + 2.0) / 4.0)
        + 0.12 * clip01(dce["continents_covered"] / 6.0)
        + 0.08 * clip01(dce["top10_envelope_score_mean"] / 80.0)
    )
    score_dyn = 100.0 * (
        0.40 * clip01((dynamics["global_warning_auc_h1"] - 0.5) / 0.30)
        + 0.30 * clip01((dynamics["global_warning_auc_h2"] - 0.5) / 0.25)
        + 0.15 * clip01((dynamics["global_warning_lift_h1"] - 1.0) / 1.5)
        + 0.15 * clip01((dynamics["global_resilience_score_0_100"] + 5.0) / 85.0)
    )
    score_nowcast = 100.0 * (
        0.45 * clip01((nowcast["global_backtest_directional_accuracy"] - 0.5) / 0.20)
        + 0.30 * clip01(nowcast["global_backtest_coverage95"])
        + 0.15 * clip01(nowcast["continents_covered"] / 6.0)
        + 0.10 * clip01(nowcast["mean_continent_prob_decelerate"] / 0.6)
    )
    score_method_core = 100.0 * (
        0.28 * clip01(method_core["mae_gain_vs_ridge_ar2"] / 1.20)
        + 0.16 * clip01(method_core["ternary_gain_vs_ridge_ar2"] / 0.04)
        + 0.20 * clip01(1.0 - method_core["p_value_mae_improve_vs_ridge_ar2"])
        + 0.14 * clip01(1.0 - method_core["p_value_ternary_improve_vs_ridge_ar2"])
        + 0.12 * clip01(method_core["continents_covered"] / 6.0)
        + 0.10 * clip01(method_core["cities_covered"] / 250.0)
    )

    rows = [
        {
            "evidence_track": "prediction_temporal",
            "primary_metric": "linear_r2_t_plus_1",
            "metric_value": temporal_r2,
            "unit": "R2",
            "score_0_100": score_temporal,
            "track_weight": 0.02,
        },
        {
            "evidence_track": "prediction_spatial_ood",
            "primary_metric": "mean_linear_r2_leave_continent_out",
            "metric_value": spatial_r2,
            "unit": "R2",
            "score_0_100": score_spatial,
            "track_weight": 0.03,
        },
        {
            "evidence_track": "prediction_discrimination",
            "primary_metric": "pulse_roc_auc",
            "metric_value": roc_auc,
            "unit": "AUC",
            "score_0_100": score_pred,
            "track_weight": 0.08,
        },
        {
            "evidence_track": "calibration",
            "primary_metric": "ece_10_bin",
            "metric_value": ece,
            "unit": "error",
            "score_0_100": score_calib,
            "track_weight": 0.08,
        },
        {
            "evidence_track": "causal_identification",
            "primary_metric": "max_identification_strength",
            "metric_value": id_strength,
            "unit": "score",
            "score_0_100": score_causal,
            "track_weight": 0.10,
        },
        {
            "evidence_track": "external_validity",
            "primary_metric": "avg_abs_t_and_r2_uplift",
            "metric_value": avg_abs_t,
            "unit": "composite",
            "score_0_100": score_external,
            "track_weight": 0.02,
        },
        {
            "evidence_track": "policy_rl",
            "primary_metric": "dr_uplift_and_ci_low",
            "metric_value": rl_dr,
            "unit": "uplift",
            "score_0_100": score_rl,
            "track_weight": 0.02,
        },
        {
            "evidence_track": "dynamic_causal_envelope",
            "primary_metric": "post_effect_pretrend_and_coverage",
            "metric_value": dce["event_post_weighted_mean"],
            "unit": "composite",
            "score_0_100": score_dce,
            "track_weight": 0.18,
        },
        {
            "evidence_track": "dynamic_survival_warning",
            "primary_metric": "hazard_horizon_warning_strength",
            "metric_value": dynamics["global_warning_auc_h2"],
            "unit": "AUC",
            "score_0_100": score_dyn,
            "track_weight": 0.15,
        },
        {
            "evidence_track": "dynamic_nowcast",
            "primary_metric": "directional_accuracy_and_coverage",
            "metric_value": nowcast["global_backtest_directional_accuracy"],
            "unit": "score",
            "score_0_100": score_nowcast,
            "track_weight": 0.13,
        },
        {
            "evidence_track": "dynamic_method_core",
            "primary_metric": "paired_significance_vs_strong_baseline",
            "metric_value": method_core["mae_gain_vs_ridge_ar2"],
            "unit": "mae_gain",
            "score_0_100": score_method_core,
            "track_weight": 0.14,
        },
        {
            "evidence_track": "inference_protocol",
            "primary_metric": "reporting_status",
            "metric_value": 1.0 if inference_status_ok else 0.0,
            "unit": "binary",
            "score_0_100": 100.0 if inference_status_ok else 0.0,
            "track_weight": 0.05,
        },
    ]
    out = pd.DataFrame(rows)
    out["score_0_100"] = pd.to_numeric(out["score_0_100"], errors="coerce").fillna(0.0)
    out["evidence_rank"] = out["score_0_100"].rank(method="dense", ascending=False).astype(int)
    return out.sort_values("evidence_rank").reset_index(drop=True)


def run_top_tier_reinforcement_suite() -> Dict[str, Any]:
    """Generate top-tier reinforcement assets and return summary stats."""
    frontier = _innovation_frontier_table()
    adaptive_frontier = _adaptive_innovation_candidate()
    spectrum = _identification_spectrum_table()
    convergence = _evidence_convergence_table(spectrum)

    frontier_path = DATA_OUTPUTS / "top_tier_innovation_frontier.csv"
    spectrum_path = DATA_OUTPUTS / "top_tier_identification_spectrum.csv"
    convergence_path = DATA_OUTPUTS / "top_tier_evidence_convergence.csv"

    frontier.to_csv(frontier_path, index=False)
    spectrum.to_csv(spectrum_path, index=False)
    convergence.to_csv(convergence_path, index=False)

    best_frontier = {}
    if not frontier.empty:
        top = frontier.iloc[0]
        best_frontier = {
            "candidate": str(top.get("candidate", "")),
            "alphas": {
                "alpha_dynamic": float(top.get("alpha_dynamic", np.nan)),
                "alpha_graph": float(top.get("alpha_graph", np.nan)),
                "alpha_critical": float(top.get("alpha_critical", np.nan)),
                "alpha_base": float(top.get("alpha_base", np.nan)),
            },
            "eval_roc_auc": float(top.get("eval_roc_auc", np.nan)),
            "eval_average_precision": float(top.get("eval_average_precision", np.nan)),
            "eval_brier": float(top.get("eval_brier", np.nan)),
            "frontier_score": float(top.get("frontier_score", np.nan)),
            "is_pareto_efficient": bool(int(top.get("is_pareto_efficient", 0)) == 1),
        }
    if adaptive_frontier.get("status") == "ok":
        frontier_auc = float(best_frontier.get("eval_roc_auc", np.nan)) if best_frontier else np.nan
        adaptive_auc = _to_float(adaptive_frontier.get("eval_roc_auc"), np.nan)
        if (not np.isfinite(frontier_auc)) or (np.isfinite(adaptive_auc) and adaptive_auc > frontier_auc):
            best_frontier = {
                "candidate": str(adaptive_frontier.get("candidate", "geo_adapted_selected")),
                "alphas": {
                    "alpha_dynamic": np.nan,
                    "alpha_graph": np.nan,
                    "alpha_critical": np.nan,
                    "alpha_base": np.nan,
                },
                "eval_roc_auc": adaptive_auc,
                "eval_average_precision": _to_float(adaptive_frontier.get("eval_average_precision"), np.nan),
                "eval_brier": _to_float(adaptive_frontier.get("eval_brier"), np.nan),
                "frontier_score": np.nan,
                "is_pareto_efficient": True,
                "n_continents": int(adaptive_frontier.get("n_continents", 0) or 0),
                "n_eval_total": int(adaptive_frontier.get("n_eval_total", 0) or 0),
                "weighted_delta_auc": _to_float(adaptive_frontier.get("weighted_delta_auc"), np.nan),
                "share_positive_continent_gain": _to_float(adaptive_frontier.get("share_positive_continent_gain"), np.nan),
            }

    mean_id_strength = float(pd.to_numeric(spectrum.get("identification_strength", pd.Series(dtype=float)), errors="coerce").dropna().mean()) if not spectrum.empty else np.nan
    mean_pass_rate = float(pd.to_numeric(spectrum.get("robust_pass_rate", pd.Series(dtype=float)), errors="coerce").dropna().mean()) if not spectrum.empty else np.nan
    primary_mask = (
        pd.to_numeric(spectrum.get("is_primary_track", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int) == 1
    ) if not spectrum.empty else pd.Series(dtype=bool)
    primary = spectrum[primary_mask].copy() if not spectrum.empty else pd.DataFrame()
    if primary.empty and not spectrum.empty:
        primary = spectrum.copy()
    grade_weight = {"A": 1.20, "B": 1.00, "C": 0.70, "D": 0.40}
    if not primary.empty:
        weights = (
            primary.get("evidence_grade", "").astype(str).map(grade_weight).fillna(0.60).to_numpy(dtype=float)
        )
        p_strength = pd.to_numeric(primary.get("identification_strength"), errors="coerce").to_numpy(dtype=float)
        p_robust = pd.to_numeric(primary.get("robust_pass_rate"), errors="coerce").to_numpy(dtype=float)
        mask_s = np.isfinite(p_strength) & np.isfinite(weights) & (weights > 0)
        mask_r = np.isfinite(p_robust) & np.isfinite(weights) & (weights > 0)
        primary_weighted_strength = float(np.average(p_strength[mask_s], weights=weights[mask_s])) if np.any(mask_s) else np.nan
        primary_weighted_robust = float(np.average(p_robust[mask_r], weights=weights[mask_r])) if np.any(mask_r) else np.nan
    else:
        primary_weighted_strength = np.nan
        primary_weighted_robust = np.nan
    overall_score = np.nan
    if not convergence.empty:
        scores = pd.to_numeric(convergence["score_0_100"], errors="coerce")
        weights = pd.to_numeric(convergence.get("track_weight", 1.0), errors="coerce").fillna(1.0)
        mask = scores.notna() & weights.notna() & (weights > 0)
        if mask.any():
            overall_score = float(np.average(scores[mask].to_numpy(dtype=float), weights=weights[mask].to_numpy(dtype=float)))

    best_auc = float(best_frontier.get("eval_roc_auc", np.nan)) if best_frontier else np.nan
    best_brier = float(best_frontier.get("eval_brier", np.nan)) if best_frontier else np.nan
    dce = _dynamic_envelope_snapshot()
    dynamics = _pulse_dynamics_snapshot()
    nowcast = _pulse_nowcast_snapshot()
    method_core = _dynamic_method_core_snapshot()

    gate_specs = [
        {
            "gate": "innovation_auc",
            "metric": "best_frontier_eval_roc_auc",
            "value": best_auc,
            "threshold": 0.75,
            "direction": "min",
        },
        {
            "gate": "innovation_brier",
            "metric": "best_frontier_eval_brier",
            "value": best_brier,
            "threshold": 0.22,
            "direction": "max",
        },
        {
            "gate": "identification_strength",
            "metric": "identification_strength_primary_weighted",
            "value": primary_weighted_strength,
            "threshold": 18.0,
            "direction": "min",
        },
        {
            "gate": "identification_robustness",
            "metric": "identification_robust_pass_rate_primary_weighted",
            "value": primary_weighted_robust,
            "threshold": 0.04,
            "direction": "min",
        },
        {
            "gate": "evidence_convergence",
            "metric": "evidence_overall_score_0_100",
            "value": overall_score,
            "threshold": 70.0,
            "direction": "min",
        },
        {
            "gate": "dynamic_envelope_pretrend_noise",
            "metric": "dce_event_pre_abs_median",
            "value": dce["event_pre_abs_median"],
            "threshold": 0.10,
            "direction": "max",
        },
        {
            "gate": "dynamic_envelope_continent_coverage",
            "metric": "dce_continents_covered",
            "value": dce["continents_covered"],
            "threshold": 6.0,
            "direction": "min",
        },
        {
            "gate": "dynamic_envelope_frontier_quality",
            "metric": "dce_top10_envelope_score_mean",
            "value": dce["top10_envelope_score_mean"],
            "threshold": 70.0,
            "direction": "min",
        },
        {
            "gate": "dynamic_envelope_post_ci_share",
            "metric": "dce_event_post_ci_above_zero_share",
            "value": dce["event_post_ci_above_zero_share"],
            "threshold": 0.50,
            "direction": "min",
        },
        {
            "gate": "dynamic_envelope_continent_stability",
            "metric": "dce_mean_continent_stability_score",
            "value": dce["mean_continent_stability_score"],
            "threshold": 40.0,
            "direction": "min",
        },
        {
            "gate": "dynamic_envelope_stall_trap_share",
            "metric": "dce_stall_trap_share",
            "value": dce["stall_trap_share"],
            "threshold": 0.20,
            "direction": "max",
        },
        {
            "gate": "dynamic_warning_auc_h2",
            "metric": "pulse_dynamics_warning_auc_h2",
            "value": dynamics["global_warning_auc_h2"],
            "threshold": 0.62,
            "direction": "min",
        },
        {
            "gate": "dynamic_warning_auc_h3",
            "metric": "pulse_dynamics_warning_auc_h3",
            "value": dynamics["global_warning_auc_h3"],
            "threshold": 0.54,
            "direction": "min",
        },
        {
            "gate": "dynamic_transition_coverage",
            "metric": "pulse_dynamics_transition_continent_coverage",
            "value": dynamics["transition_continent_coverage"],
            "threshold": 6.0,
            "direction": "min",
        },
        {
            "gate": "dynamic_stall_exit_half_life",
            "metric": "pulse_dynamics_stall_exit_half_life",
            "value": dynamics["global_stall_exit_half_life"],
            "threshold": 2.5,
            "direction": "max",
        },
        {
            "gate": "dynamic_nowcast_directional_accuracy",
            "metric": "pulse_nowcast_global_backtest_directional_accuracy",
            "value": nowcast["global_backtest_directional_accuracy"],
            "threshold": 0.52,
            "direction": "min",
        },
        {
            "gate": "dynamic_nowcast_coverage95",
            "metric": "pulse_nowcast_global_backtest_coverage95",
            "value": nowcast["global_backtest_coverage95"],
            "threshold": 0.70,
            "direction": "min",
        },
        {
            "gate": "dynamic_nowcast_continent_coverage",
            "metric": "pulse_nowcast_continents_covered",
            "value": nowcast["continents_covered"],
            "threshold": 6.0,
            "direction": "min",
        },
        {
            "gate": "dynamic_method_mae_gain",
            "metric": "dynamic_method_core_mae_gain_vs_ridge_ar2",
            "value": method_core["mae_gain_vs_ridge_ar2"],
            "threshold": 0.25,
            "direction": "min",
        },
        {
            "gate": "dynamic_method_ternary_gain",
            "metric": "dynamic_method_core_ternary_gain_vs_ridge_ar2",
            "value": method_core["ternary_gain_vs_ridge_ar2"],
            "threshold": 0.01,
            "direction": "min",
        },
        {
            "gate": "dynamic_method_abs_significance",
            "metric": "dynamic_method_core_p_value_mae_improve_vs_ridge_ar2",
            "value": method_core["p_value_mae_improve_vs_ridge_ar2"],
            "threshold": 0.05,
            "direction": "max",
        },
        {
            "gate": "dynamic_method_direction_significance",
            "metric": "dynamic_method_core_p_value_ternary_improve_vs_ridge_ar2",
            "value": method_core["p_value_ternary_improve_vs_ridge_ar2"],
            "threshold": 0.10,
            "direction": "max",
        },
        {
            "gate": "dynamic_method_coverage",
            "metric": "dynamic_method_core_continents_covered",
            "value": method_core["continents_covered"],
            "threshold": 6.0,
            "direction": "min",
        },
    ]

    gate_rows: List[Dict[str, Any]] = []
    for g in gate_specs:
        v = float(g["value"]) if np.isfinite(g["value"]) else np.nan
        thr = float(g["threshold"])
        direction = str(g["direction"])
        if not np.isfinite(v):
            passed = False
            gap_ratio = np.nan
        elif direction == "min":
            passed = bool(v >= thr)
            gap_ratio = float((thr - v) / max(abs(thr), 1e-9))
        else:
            passed = bool(v <= thr)
            gap_ratio = float((v - thr) / max(abs(thr), 1e-9))
        gate_rows.append(
            {
                "gate": g["gate"],
                "metric": g["metric"],
                "value": v,
                "threshold": thr,
                "direction": direction,
                "passed": int(passed),
                "gap_ratio": gap_ratio,
            }
        )

    gate_df = pd.DataFrame(gate_rows)
    gate_path = DATA_OUTPUTS / "top_tier_gate_checks.csv"
    gate_df.to_csv(gate_path, index=False)

    failed_gates = gate_df[gate_df["passed"] == 0].copy() if not gate_df.empty else pd.DataFrame()
    if not failed_gates.empty and "gap_ratio" in failed_gates.columns:
        failed_gates["gap_ratio_abs"] = pd.to_numeric(failed_gates["gap_ratio"], errors="coerce").abs()
        failed_gates = failed_gates.sort_values("gap_ratio_abs", ascending=False)
    top_gaps = (
        failed_gates[["gate", "metric", "value", "threshold", "direction"]]
        .head(3)
        .to_dict(orient="records")
        if not failed_gates.empty
        else []
    )
    gate_pass_rate = float(gate_df["passed"].mean()) if not gate_df.empty else np.nan
    top_tier_ready = bool((gate_df["passed"] == 1).all()) if not gate_df.empty else False

    summary = {
        "status": "ok",
        "innovation_frontier_rows": int(len(frontier)),
        "innovation_pareto_count": int(frontier["is_pareto_efficient"].sum()) if ("is_pareto_efficient" in frontier.columns and not frontier.empty) else 0,
        "identification_spectrum_rows": int(len(spectrum)),
        "identification_strength_mean": mean_id_strength,
        "identification_robust_pass_rate_mean": mean_pass_rate,
        "identification_strength_primary_weighted": primary_weighted_strength,
        "identification_robust_pass_rate_primary_weighted": primary_weighted_robust,
        "evidence_convergence_rows": int(len(convergence)),
        "evidence_overall_score_0_100": overall_score,
        "dynamic_causal_envelope": dce,
        "pulse_dynamics": dynamics,
        "pulse_nowcast": nowcast,
        "dynamic_method_core": method_core,
        "top_tier_gate": {
            "ready": top_tier_ready,
            "gate_pass_rate": gate_pass_rate,
            "failed_gate_count": int((gate_df["passed"] == 0).sum()) if not gate_df.empty else 0,
            "priority_gaps": top_gaps,
        },
        "best_frontier_candidate": best_frontier,
        "artifacts": {
            "frontier_csv": str(frontier_path),
            "identification_csv": str(spectrum_path),
            "convergence_csv": str(convergence_path),
            "gate_checks_csv": str(gate_path),
        },
    }

    dump_json(DATA_OUTPUTS / "top_tier_reinforcement_summary.json", summary)
    return summary
