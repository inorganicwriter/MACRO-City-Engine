from __future__ import annotations

"""Prepare dashboard artifacts from processed outputs."""

import logging
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pandas.errors import EmptyDataError

from .utils import DATA_OUTPUTS, DATA_PROCESSED, WEB_DATA_DIR, dump_json

LOGGER = logging.getLogger(__name__)


def _top_cities_snapshot(panel: pd.DataFrame, year: int, metric: str, n: int = 20) -> List[Dict[str, object]]:
    subset = panel.loc[panel["year"] == year].sort_values(metric, ascending=False).head(n)
    return subset[
        ["city_id", "city_name", "country", "continent", "year", metric, "latitude", "longitude"]
    ].rename(columns={metric: "score"}).to_dict(orient="records")


def _copy_csv_if_readable(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    try:
        df = pd.read_csv(src)
    except EmptyDataError:
        LOGGER.warning("Skip empty CSV during dashboard export: %s", src)
        return
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Skip unreadable CSV during dashboard export: %s (%s)", src, exc)
        return
    df.to_csv(dst, index=False)


def export_dashboard_data(panel: pd.DataFrame) -> None:
    """Generate compact JSON payloads for the web dashboard."""
    latest_year = int(panel["year"].max())

    city_points = panel[
        [
            "city_id",
            "city_name",
            "country",
            "continent",
            "year",
            "latitude",
            "longitude",
            "economic_vitality",
            "livability",
            "innovation",
            "composite_index",
            "treated_city",
        ]
    ].copy()
    city_points.to_csv(WEB_DATA_DIR / "city_points.csv", index=False)

    top_composite = _top_cities_snapshot(panel, latest_year, "composite_index", n=20)
    top_economic = _top_cities_snapshot(panel, latest_year, "economic_vitality", n=20)

    summary = {
        "latest_year": latest_year,
        "city_count": int(panel["city_id"].nunique()),
        "records": int(len(panel)),
        "top_composite": top_composite,
        "top_economic": top_economic,
    }
    dump_json(WEB_DATA_DIR / "dashboard_summary.json", summary)

    global_yearly = (
        panel.groupby("year", as_index=False)
        .agg(
            economic_vitality=("economic_vitality", "mean"),
            livability=("livability", "mean"),
            innovation=("innovation", "mean"),
            composite_index=("composite_index", "mean"),
        )
        .sort_values("year")
    )
    global_yearly.to_csv(WEB_DATA_DIR / "global_yearly.csv", index=False)

    if (DATA_OUTPUTS / "econometric_summary.json").exists():
        with (DATA_OUTPUTS / "econometric_summary.json").open("r", encoding="utf-8") as f:
            summary_json = json.load(f)
        dump_json(WEB_DATA_DIR / "econometric_summary.json", summary_json)

    for name in [
        "causal_st_summary.json",
        "causal_st_ablation.json",
        "benchmark_scores.json",
        "experiment_enhancements.json",
        "external_validity_summary.json",
        "pulse_state_summary.json",
        "pulse_ai_summary.json",
        "idplus_summary.json",
        "dynamic_causal_envelope_summary.json",
        "pulse_dynamics_summary.json",
        "pulse_nowcast_summary.json",
        "dynamic_method_core_summary.json",
        "top_tier_reinforcement_summary.json",
        "submission_readiness.json",
        "submission_extensions_summary.json",
        "robustness_audit_summary.json",
        "reproducibility_manifest.json",
    ]:
        src = DATA_OUTPUTS / name
        if src.exists():
            with src.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            dump_json(WEB_DATA_DIR / name, payload)

    cov_path = DATA_PROCESSED / "coverage_summary.json"
    if cov_path.exists():
        with cov_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        dump_json(WEB_DATA_DIR / "coverage_summary.json", payload)

    policy_path = DATA_PROCESSED / "policy_design.json"
    if policy_path.exists():
        with policy_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        dump_json(WEB_DATA_DIR / "policy_design.json", payload)

    policy_audit_path = DATA_PROCESSED / "policy_event_registry_audit.json"
    if policy_audit_path.exists():
        with policy_audit_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        dump_json(WEB_DATA_DIR / "policy_event_registry_audit.json", payload)

    policy_quality_path = DATA_PROCESSED / "policy_event_registry_quality_report.json"
    if policy_quality_path.exists():
        with policy_quality_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        dump_json(WEB_DATA_DIR / "policy_event_registry_quality_report.json", payload)

    policy_ai_path = DATA_PROCESSED / "policy_event_ai_inference_summary.json"
    if policy_ai_path.exists():
        with policy_ai_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        dump_json(WEB_DATA_DIR / "policy_event_ai_inference_summary.json", payload)

    policy_macro_rule_path = DATA_PROCESSED / "policy_event_objective_macro_summary.json"
    if policy_macro_rule_path.exists():
        with policy_macro_rule_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        dump_json(WEB_DATA_DIR / "policy_event_objective_macro_summary.json", payload)

    policy_indicator_path = DATA_PROCESSED / "policy_event_objective_indicator_summary.json"
    if policy_indicator_path.exists():
        with policy_indicator_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        dump_json(WEB_DATA_DIR / "policy_event_objective_indicator_summary.json", payload)

    policy_evidence_path = DATA_PROCESSED / "policy_event_evidence_summary.json"
    if policy_evidence_path.exists():
        with policy_evidence_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        dump_json(WEB_DATA_DIR / "policy_event_evidence_summary.json", payload)

    pulse_probs = DATA_OUTPUTS / "pulse_state_probabilities.csv"
    if pulse_probs.exists():
        try:
            df = pd.read_csv(pulse_probs)
        except EmptyDataError:
            LOGGER.warning("Skip empty pulse_state_probabilities.csv during dashboard export.")
            df = pd.DataFrame()
        if not df.empty:
            latest = int(df["year"].max()) if not df.empty else latest_year
            latest_rows = df[df["year"] == latest].copy()
            latest_rows.to_csv(WEB_DATA_DIR / "pulse_state_latest.csv", index=False)

    pulse_ai_latest = DATA_OUTPUTS / "pulse_ai_city_latest.csv"
    if pulse_ai_latest.exists():
        _copy_csv_if_readable(pulse_ai_latest, WEB_DATA_DIR / "pulse_ai_city_latest.csv")

    for csv_name in [
        "external_validity_indicator_results.csv",
        "external_validity_rank_corr_by_year.csv",
        "pulse_ai_regime_year_share.csv",
        "pulse_ai_regime_transition_matrix.csv",
        "pulse_ai_trajectory_regimes.csv",
        "pulse_ai_regime_by_year.csv",
        "pulse_ai_cross_continent_generalization.csv",
        "pulse_ai_shock_years.csv",
        "pulse_ai_shock_irf_regime.csv",
        "pulse_ai_shock_irf_vulnerability.csv",
        "pulse_ai_horizon_forecast.csv",
        "pulse_ai_dynamic_hazard_latest.csv",
        "pulse_ai_dynamic_graph_latest.csv",
        "pulse_ai_dynamic_graph_curve.csv",
        "pulse_ai_dynamic_cycle.csv",
        "pulse_ai_dynamic_critical_latest.csv",
        "pulse_ai_dynamic_critical_decile.csv",
        "pulse_ai_dynamic_state_event_study.csv",
        "pulse_ai_dynamic_gate_weights.csv",
        "pulse_ai_dynamic_main_fusion_curve.csv",
        "pulse_ai_dynamic_main_fusion_eval.csv",
        "pulse_ai_dynamic_main_fusion_backtest.csv",
        "pulse_ai_dynamic_main_fusion_pareto.csv",
        "pulse_ai_dynamic_main_fusion_state_gate.csv",
        "pulse_ai_dynamic_main_fusion_continent_eval.csv",
        "pulse_ai_dynamic_main_fusion_continent_robust_curve.csv",
        "pulse_ai_dynamic_main_fusion_continent_adaptive_weights.csv",
        "pulse_ai_dynamic_phase_field.csv",
        "pulse_ai_dynamic_phase_latest.csv",
        "pulse_ai_dynamic_sync_network.csv",
        "pulse_ai_dynamic_policy_lab.csv",
        "pulse_ai_dynamic_policy_lab_summary.csv",
        "pulse_ai_dynamic_policy_rl_city.csv",
        "pulse_ai_dynamic_policy_rl_action_summary.csv",
        "pulse_ai_dynamic_policy_rl_state_value.csv",
        "pulse_ai_dynamic_policy_rl_ope.csv",
        "pulse_ai_dynamic_policy_rl_ablation.csv",
        "pulse_ai_dynamic_policy_rl_continent_ope.csv",
        "pulse_ai_dynamic_policy_rl_continent_action.csv",
        "pulse_ai_dynamic_index_series.csv",
        "pulse_ai_dynamic_index_latest.csv",
        "pulse_ai_dynamic_index_continent_year.csv",
        "dynamic_phase_heterogeneity_composite_index.csv",
        "dynamic_phase_rule_sensitivity_composite_index.csv",
        "econometric_policy_source_sensitivity.csv",
        "econometric_source_event_study_points.csv",
        "dose_response_bins_policy_dose_composite_index.csv",
        "dose_response_bins_policy_dose_external_direct_composite_index.csv",
        "benchmark_prospective_governance_eval.csv",
        "idplus_event_pretrend_geometry.csv",
        "idplus_design_concordance_pairs.csv",
        "idplus_design_concordance_matrix.csv",
        "idplus_identification_stress_index.csv",
        "idplus_leave_continent_stability_summary.csv",
        "dynamic_causal_envelope_event.csv",
        "dynamic_causal_envelope_event_bootstrap.csv",
        "dynamic_causal_envelope_regime.csv",
        "dynamic_causal_envelope_regime_summary.csv",
        "dynamic_causal_envelope_city_scores.csv",
        "dynamic_causal_envelope_continent_year.csv",
        "dynamic_causal_envelope_continent_stability.csv",
        "pulse_dynamics_state_panel.csv",
        "pulse_dynamics_state_thresholds.csv",
        "pulse_dynamics_transition_tensor.csv",
        "pulse_dynamics_stall_spells.csv",
        "pulse_dynamics_accel_spells.csv",
        "pulse_dynamics_spell_hazard.csv",
        "pulse_dynamics_resilience_halflife.csv",
        "pulse_dynamics_warning_horizon.csv",
        "pulse_nowcast_continent_latest.csv",
        "pulse_nowcast_continent_history.csv",
        "pulse_nowcast_backtest_metrics.csv",
        "pulse_nowcast_global.csv",
        "dynamic_method_core_predictions.csv",
        "dynamic_method_core_metrics.csv",
        "dynamic_method_core_significance.csv",
        "dynamic_method_core_ablation.csv",
        "top_tier_gate_checks.csv",
        "top_tier_innovation_frontier.csv",
        "top_tier_identification_spectrum.csv",
        "top_tier_evidence_convergence.csv",
        "robustness_gate_checks.csv",
        "robustness_diagnostic_snapshot.csv",
        "reproducibility_artifact_hashes.csv",
    ]:
        src = DATA_OUTPUTS / csv_name
        _copy_csv_if_readable(src, WEB_DATA_DIR / csv_name)

    for csv_name in [
        "coverage_by_country.csv",
        "coverage_by_continent.csv",
        "policy_event_evidence_by_continent.csv",
        "policy_event_registry_enriched.csv",
        "policy_event_source_links.csv",
    ]:
        src = DATA_PROCESSED / csv_name
        _copy_csv_if_readable(src, WEB_DATA_DIR / csv_name)

    LOGGER.info("Dashboard data exported to %s", WEB_DATA_DIR)
