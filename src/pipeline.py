from __future__ import annotations

"""End-to-end pipeline orchestration for the MACRO-City Engine project."""

import logging
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .ai_explainability import run_ai_explainability_suite
from .benchmark_eval import run_benchmark_suite
from .causal_st import run_causal_st_analysis, run_causal_st_experiment_matrix
from .config import load_config
from .dynamic_causal_envelope import run_dynamic_causal_envelope_suite
from .dynamic_method_core import run_dynamic_method_core_suite
from .econometrics import run_econometric_suite
from .external_validity import run_external_validity_suite
from .exogenous_shock_design import run_exogenous_shock_suite
from .exogenous_shock_heterogeneity import run_exogenous_shock_heterogeneity_suite
from .export_dashboard import export_dashboard_data
from .experiment_enhancements import run_experiment_enhancements
from .global_data import build_global_city_panel
from .identification_plus import run_identification_plus_suite
from .inference_reporting import run_inference_reporting_suite
from .modeling import train_all_targets
from .observed_evidence import run_observed_evidence_suite
from .provenance import audit_and_filter_objective_sources, build_global_coverage_report
from .pulse_ai import run_pulse_ai_engine
from .pulse_dynamics import run_pulse_dynamics_suite
from .pulse_nowcast import run_pulse_nowcast_suite
from .pulse_state import estimate_pulse_states
from .realtime_monitor import generate_realtime_monitor_snapshot
from .representation import build_city_embeddings
from .research_matrix import build_research_matrix_report
from .submission_extensions import run_submission_extensions
from .top_tier_reinforcement import run_top_tier_reinforcement_suite
from .utils import DATA_PROCESSED, REPORTS_DIR, dump_json, ensure_project_dirs
from .weight_sensitivity import run_weight_sensitivity_analysis

LOGGER = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure project-wide logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_pipeline(
    max_cities: int | None = None,
    strict_real_data: bool | None = None,
    require_policy_events: bool = False,
    auto_build_policy_events: bool = False,
    augment_policy_events_for_sensitivity: bool = False,
    enable_city_macro_disaggregation: bool = True,
    use_city_observed_primary_spec: bool = True,
    normalize_within_year: bool = False,
    prefer_pca_composite: bool = True,
    enforce_verified_sources: bool = True,
    min_verified_city_retention: float = 0.55,
    min_external_direct_share: float = 0.70,
    max_ai_inferred_share: float = 0.30,
    econometrics_fast: bool = False,
) -> Dict[str, Any]:
    """Run the full global urban analytics workflow."""
    ensure_project_dirs()
    config = load_config()

    city_limit = config.max_cities_default if max_cities is None else max_cities
    strict_mode = config.strict_real_data if strict_real_data is None else strict_real_data

    LOGGER.info("Step 1/7: Building global city panel...")
    panel_raw = build_global_city_panel(
        config,
        max_cities=city_limit,
        strict_real_data=strict_mode,
        require_policy_events=bool(require_policy_events),
        auto_build_policy_events=bool(auto_build_policy_events),
        augment_policy_events_for_sensitivity=bool(augment_policy_events_for_sensitivity),
        enable_city_macro_disaggregation=bool(enable_city_macro_disaggregation),
        use_city_observed_primary_spec=bool(use_city_observed_primary_spec),
        normalize_within_year=bool(normalize_within_year),
        prefer_pca_composite=bool(prefer_pca_composite),
    )

    LOGGER.info("Step 2/7: Auditing objective-source provenance...")
    panel, source_audit = audit_and_filter_objective_sources(
        panel_raw,
        strict_mode=bool(strict_mode),
        enforce_verified=bool(enforce_verified_sources),
        min_verified_city_retention_for_verified_filter=float(min_verified_city_retention),
    )
    if panel.empty:
        msg = "No rows left after objective-source filtering. Unable to continue."
        raise RuntimeError(msg)
    reliability_gate = _validate_real_source_integrity(
        source_audit=source_audit,
        strict_mode=bool(strict_mode),
        enforce_verified_sources=bool(enforce_verified_sources),
        min_verified_city_retention=float(min_verified_city_retention),
        min_external_direct_share=float(min_external_direct_share),
        max_ai_inferred_share=float(max_ai_inferred_share),
    )
    panel.to_csv(DATA_PROCESSED / "global_city_panel_strict.csv", index=False)
    coverage_report = build_global_coverage_report(panel, strict_mode=bool(strict_mode))

    LOGGER.info("Step 3/7: Building AI representations, pulse states, and pulse-risk engine...")
    embeddings = build_city_embeddings(panel)
    pulse_state = estimate_pulse_states(panel)
    pulse_ai = run_pulse_ai_engine(panel)

    LOGGER.info("Step 4/7: Training predictive models...")
    model_metrics = train_all_targets(panel)

    LOGGER.info("Step 5/7: Running econometric and causal-spatiotemporal analysis...")
    econometrics = run_econometric_suite(panel, fast_mode=bool(econometrics_fast))
    causal_st = run_causal_st_analysis(panel)
    causal_st_ablation = run_causal_st_experiment_matrix(panel)

    LOGGER.info("Step 6/7: Running benchmark protocol and robustness enhancements...")
    benchmark = run_benchmark_suite(panel)
    observed_evidence = run_observed_evidence_suite(panel)
    experiment_enhancements = run_experiment_enhancements(panel)
    ai_explainability = run_ai_explainability_suite(panel)
    inference_reporting = run_inference_reporting_suite()
    external_validity = run_external_validity_suite(panel)
    identification_plus = run_identification_plus_suite()
    dynamic_causal_envelope = run_dynamic_causal_envelope_suite()
    pulse_dynamics = run_pulse_dynamics_suite()
    pulse_nowcast = run_pulse_nowcast_suite()
    dynamic_method_core = run_dynamic_method_core_suite()
    exogenous_shock = run_exogenous_shock_suite(panel)
    exogenous_shock_heterogeneity = run_exogenous_shock_heterogeneity_suite(panel)
    top_tier_reinforcement = run_top_tier_reinforcement_suite()
    submission_extensions = run_submission_extensions()
    weight_sensitivity = run_weight_sensitivity_analysis(panel)

    LOGGER.info("Step 7/7: Exporting dashboard data and reports...")
    export_dashboard_data(panel)
    realtime_status = generate_realtime_monitor_snapshot(trigger="pipeline_step7")
    _write_summary_report(
        panel,
        model_metrics,
        econometrics,
        causal_st,
        causal_st_ablation,
        benchmark,
        experiment_enhancements,
        observed_evidence,
        ai_explainability,
        inference_reporting,
        external_validity,
        identification_plus,
        dynamic_causal_envelope,
        dynamic_method_core,
        exogenous_shock,
        exogenous_shock_heterogeneity,
        top_tier_reinforcement,
        pulse_state,
        pulse_ai,
        source_audit,
        coverage_report,
        reliability_gate,
        realtime_status,
        strict_real_data=bool(strict_mode),
    )
    build_research_matrix_report()

    summary = {
        "city_count": int(panel["city_id"].nunique()),
        "year_count": int(panel["year"].nunique()),
        "records": int(len(panel)),
        "source_audit": source_audit,
        "coverage_report": coverage_report,
        "reliability_gate": reliability_gate,
        "embedding_dim": int(len([c for c in embeddings.columns if c.startswith("emb_")])),
        "strict_real_data": bool(strict_mode),
        "enable_city_macro_disaggregation": bool(enable_city_macro_disaggregation),
        "use_city_observed_primary_spec": bool(use_city_observed_primary_spec),
        "normalize_within_year": bool(normalize_within_year),
        "prefer_pca_composite": bool(prefer_pca_composite),
        "augment_policy_events_for_sensitivity": bool(augment_policy_events_for_sensitivity),
        "enforce_verified_sources": bool(enforce_verified_sources),
        "econometrics_fast": bool(econometrics_fast),
        "outputs": {
            "panel": str(DATA_PROCESSED / "global_city_panel.csv"),
            "panel_strict": str(DATA_PROCESSED / "global_city_panel_strict.csv"),
            "model_metrics": "data/outputs/model_metrics.json",
            "model_ai_incrementality": "data/outputs/model_ai_incrementality.json",
            "model_ai_incrementality_table": "data/outputs/model_ai_incrementality.csv",
            "econometrics": "data/outputs/econometric_summary.json",
            "econometric_policy_source_sensitivity": "data/outputs/econometric_policy_source_sensitivity.csv",
            "econometric_source_event_study_points": "data/outputs/econometric_source_event_study_points.csv",
            "econometric_mechanism_decomposition": "data/outputs/mechanism_decomposition_main_did_treatment.csv",
            "econometric_dose_response_bins": "data/outputs/dose_response_bins_policy_dose_composite_index.csv",
            "econometric_dose_response_bins_external_direct": "data/outputs/dose_response_bins_policy_dose_external_direct_composite_index.csv",
            "causal_st": "data/outputs/causal_st_summary.json",
            "causal_st_ablation": "data/outputs/causal_st_ablation.json",
            "benchmark": "data/outputs/benchmark_scores.json",
            "observed_evidence": "data/outputs/observed_evidence_summary.json",
            "observed_measurement_audit": "data/outputs/observed_measurement_audit.csv",
            "observed_feature_group_ablation": "data/outputs/observed_feature_group_ablation.csv",
            "observed_feature_group_summary": "data/outputs/observed_feature_group_summary.csv",
            "observed_cross_source_city_year": "data/outputs/observed_cross_source_city_year.csv",
            "observed_cross_source_pairs": "data/outputs/observed_cross_source_pairs.csv",
            "benchmark_prospective_eval": "data/outputs/benchmark_prospective_governance_eval.csv",
            "experiment_enhancements": "data/outputs/experiment_enhancements.json",
            "ai_explainability": "data/outputs/ai_explainability_summary.json",
            "ai_explainability_consistency": "data/outputs/ai_explainability_consistency.csv",
            "ai_feature_importance_by_year": "data/outputs/ai_feature_importance_by_year.csv",
            "ai_feature_importance_drift_summary": "data/outputs/ai_feature_importance_drift_summary.csv",
            "inference_reporting": "data/outputs/inference_reporting_summary.json",
            "inference_main_results": "data/outputs/inference_main_results.csv",
            "inference_multiple_testing": "data/outputs/inference_multiple_testing.csv",
            "external_validity": "data/outputs/external_validity_summary.json",
            "identification_plus": "data/outputs/idplus_summary.json",
            "idplus_pretrend_geometry": "data/outputs/idplus_event_pretrend_geometry.csv",
            "idplus_concordance_pairs": "data/outputs/idplus_design_concordance_pairs.csv",
            "idplus_concordance_matrix": "data/outputs/idplus_design_concordance_matrix.csv",
            "idplus_stress_index": "data/outputs/idplus_identification_stress_index.csv",
            "idplus_leave_continent_stability": "data/outputs/idplus_leave_continent_stability_summary.csv",
            "dynamic_causal_envelope_summary": "data/outputs/dynamic_causal_envelope_summary.json",
            "dynamic_causal_envelope_event": "data/outputs/dynamic_causal_envelope_event.csv",
            "dynamic_causal_envelope_event_bootstrap": "data/outputs/dynamic_causal_envelope_event_bootstrap.csv",
            "dynamic_causal_envelope_regime": "data/outputs/dynamic_causal_envelope_regime.csv",
            "dynamic_causal_envelope_regime_summary": "data/outputs/dynamic_causal_envelope_regime_summary.csv",
            "dynamic_causal_envelope_city_scores": "data/outputs/dynamic_causal_envelope_city_scores.csv",
            "dynamic_causal_envelope_continent_year": "data/outputs/dynamic_causal_envelope_continent_year.csv",
            "dynamic_causal_envelope_continent_stability": "data/outputs/dynamic_causal_envelope_continent_stability.csv",
            "pulse_dynamics_summary": "data/outputs/pulse_dynamics_summary.json",
            "pulse_dynamics_state_panel": "data/outputs/pulse_dynamics_state_panel.csv",
            "pulse_dynamics_state_thresholds": "data/outputs/pulse_dynamics_state_thresholds.csv",
            "pulse_dynamics_transition_tensor": "data/outputs/pulse_dynamics_transition_tensor.csv",
            "pulse_dynamics_stall_spells": "data/outputs/pulse_dynamics_stall_spells.csv",
            "pulse_dynamics_accel_spells": "data/outputs/pulse_dynamics_accel_spells.csv",
            "pulse_dynamics_spell_hazard": "data/outputs/pulse_dynamics_spell_hazard.csv",
            "pulse_dynamics_resilience_halflife": "data/outputs/pulse_dynamics_resilience_halflife.csv",
            "pulse_dynamics_warning_horizon": "data/outputs/pulse_dynamics_warning_horizon.csv",
            "pulse_nowcast_summary": "data/outputs/pulse_nowcast_summary.json",
            "pulse_nowcast_continent_latest": "data/outputs/pulse_nowcast_continent_latest.csv",
            "pulse_nowcast_continent_history": "data/outputs/pulse_nowcast_continent_history.csv",
            "pulse_nowcast_backtest_metrics": "data/outputs/pulse_nowcast_backtest_metrics.csv",
            "pulse_nowcast_global": "data/outputs/pulse_nowcast_global.csv",
            "dynamic_method_core_summary": "data/outputs/dynamic_method_core_summary.json",
            "dynamic_method_core_predictions": "data/outputs/dynamic_method_core_predictions.csv",
            "dynamic_method_core_metrics": "data/outputs/dynamic_method_core_metrics.csv",
            "dynamic_method_core_significance": "data/outputs/dynamic_method_core_significance.csv",
            "dynamic_method_core_ablation": "data/outputs/dynamic_method_core_ablation.csv",
            "exogenous_shock": "data/outputs/exoshock_summary.json",
            "exoshock_year_index": "data/outputs/exoshock_year_index.csv",
            "exoshock_baseline_exposure": "data/outputs/exoshock_city_baseline_exposure.csv",
            "exoshock_event_response": "data/outputs/exoshock_event_response.csv",
            "exoshock_placebo_distribution": "data/outputs/exoshock_placebo_year_distribution.csv",
            "exoshock_leave_one": "data/outputs/exoshock_leave_one_shock_out.csv",
            "exoshock_heterogeneity": "data/outputs/exoshock_heterogeneity_summary.json",
            "exoshock_policy_type_year_index": "data/outputs/exoshock_policy_type_year_index.csv",
            "exoshock_policy_type_event_response": "data/outputs/exoshock_policy_type_event_response.csv",
            "exoshock_continent_event_response": "data/outputs/exoshock_continent_event_response.csv",
            "exoshock_channel_decomposition": "data/outputs/exoshock_channel_decomposition.csv",
            "exoshock_channel_summary": "data/outputs/exoshock_channel_summary.csv",
            "top_tier_reinforcement": "data/outputs/top_tier_reinforcement_summary.json",
            "top_tier_innovation_frontier": "data/outputs/top_tier_innovation_frontier.csv",
            "top_tier_identification_spectrum": "data/outputs/top_tier_identification_spectrum.csv",
            "top_tier_evidence_convergence": "data/outputs/top_tier_evidence_convergence.csv",
            "submission_extensions_summary": "data/outputs/submission_extensions_summary.json",
            "robustness_audit_summary": "data/outputs/robustness_audit_summary.json",
            "robustness_gate_checks": "data/outputs/robustness_gate_checks.csv",
            "robustness_diagnostic_snapshot": "data/outputs/robustness_diagnostic_snapshot.csv",
            "reproducibility_manifest": "data/outputs/reproducibility_manifest.json",
            "reproducibility_hashes": "data/outputs/reproducibility_artifact_hashes.csv",
            "pulse_state": "data/outputs/pulse_state_summary.json",
            "pulse_ai": "data/outputs/pulse_ai_summary.json",
            "source_audit": "data/processed/source_audit_summary.json",
            "coverage_report": "data/processed/coverage_summary.json",
            "city_macro_observed_summary": "data/processed/city_macro_observed_summary.json",
            "road_tier_year_summary": "data/processed/road_tier_year_summary.csv",
            "social_sentiment_summary": "data/processed/social_sentiment_summary.json",
            "policy_design": "data/processed/policy_design.json",
            "policy_registry_audit": "data/processed/policy_event_registry_audit.json",
            "policy_registry_enriched": "data/processed/policy_event_registry_enriched.csv",
            "policy_registry_source_links": "data/processed/policy_event_source_links.csv",
            "policy_registry_quality": "data/processed/policy_event_registry_quality_report.json",
            "policy_objective_indicator": "data/processed/policy_event_objective_indicator_summary.json",
            "policy_objective_macro": "data/processed/policy_event_objective_macro_summary.json",
            "policy_ai_inference": "data/processed/policy_event_ai_inference_summary.json",
            "policy_evidence": "data/processed/policy_event_evidence_summary.json",
            "realtime_status": "web/static/data/realtime_status.json",
            "realtime_country_monitor": "web/static/data/realtime_country_monitor.csv",
            "realtime_alerts": "web/static/data/realtime_alerts.csv",
            "realtime_sentinel": "web/static/data/realtime_sentinel.csv",
            "pulse_ai_policy_rl_city": "web/static/data/pulse_ai_dynamic_policy_rl_city.csv",
            "pulse_ai_policy_rl_action_summary": "web/static/data/pulse_ai_dynamic_policy_rl_action_summary.csv",
            "pulse_ai_policy_rl_state_value": "web/static/data/pulse_ai_dynamic_policy_rl_state_value.csv",
            "pulse_ai_policy_rl_ope": "web/static/data/pulse_ai_dynamic_policy_rl_ope.csv",
            "pulse_ai_policy_rl_ablation": "web/static/data/pulse_ai_dynamic_policy_rl_ablation.csv",
            "pulse_ai_dynamic_index_latest": "web/static/data/pulse_ai_dynamic_index_latest.csv",
            "pulse_ai_dynamic_index_continent": "web/static/data/pulse_ai_dynamic_index_continent_year.csv",
            "realtime_snapshot_history": "data/outputs/realtime_monitor_history.jsonl",
            "dashboard": "web/static/data",
        },
        "submission_extensions": submission_extensions,
        "observed_evidence": observed_evidence,
        "pulse_dynamics": pulse_dynamics,
        "pulse_nowcast": pulse_nowcast,
        "dynamic_method_core": dynamic_method_core,
        "realtime_monitor": realtime_status,
    }
    dump_json(REPORTS_DIR / "pipeline_summary.json", summary)
    return summary


def _write_summary_report(
    panel: pd.DataFrame,
    model_metrics: Dict[str, Any],
    econometrics: Dict[str, Any],
    causal_st: Dict[str, Any],
    causal_st_ablation: Dict[str, Any],
    benchmark: Dict[str, Any],
    experiment_enhancements: Dict[str, Any],
    observed_evidence: Dict[str, Any],
    ai_explainability: Dict[str, Any],
    inference_reporting: Dict[str, Any],
    external_validity: Dict[str, Any],
    identification_plus: Dict[str, Any],
    dynamic_causal_envelope: Dict[str, Any],
    dynamic_method_core: Dict[str, Any],
    exogenous_shock: Dict[str, Any],
    exogenous_shock_heterogeneity: Dict[str, Any],
    top_tier_reinforcement: Dict[str, Any],
    pulse_state: Dict[str, Any],
    pulse_ai: Dict[str, Any],
    source_audit: Dict[str, Any],
    coverage_report: Dict[str, Any],
    reliability_gate: Dict[str, Any],
    realtime_status: Dict[str, Any],
    strict_real_data: bool,
) -> None:
    """Write markdown report for challenge/project documentation."""
    latest_year = int(panel["year"].max())
    top = (
        panel.loc[panel["year"] == latest_year, ["city_name", "country", "composite_index"]]
        .sort_values("composite_index", ascending=False)
        .head(10)
    )

    lines = [
        "# MACRO-City Engine Global Report",
        "",
        "## Dataset Overview",
        f"- City count: {panel['city_id'].nunique()}",
        f"- Year range: {int(panel['year'].min())}-{int(panel['year'].max())}",
        f"- Observations: {len(panel)}",
        f"- Strict real data mode: {strict_real_data}",
        "",
        "## Top Cities (Latest Year)",
        "| Rank | City | Country | Composite |",
        "|---:|---|---|---:|",
    ]

    for idx, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(f"| {idx} | {row.city_name} | {row.country} | {row.composite_index:.2f} |")

    lines.extend(
        [
            "",
            "## Source Audit",
            "```json",
            json.dumps(source_audit, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Source Reliability Gate",
            "```json",
            json.dumps(reliability_gate, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Realtime Monitor Snapshot",
            "```json",
            json.dumps(realtime_status, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Coverage Diagnostics",
            "```json",
            json.dumps(coverage_report, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Modeling Metrics",
            "```json",
            json.dumps(model_metrics, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Econometric Findings",
            "```json",
            json.dumps(econometrics, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Causal-ST Findings",
            "```json",
            json.dumps(causal_st, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Causal-ST Ablation",
            "```json",
            json.dumps(causal_st_ablation, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Pulse State Summary",
            "```json",
            json.dumps(pulse_state, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Pulse AI Summary",
            "```json",
            json.dumps(pulse_ai, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Benchmark Protocol",
            "```json",
            json.dumps(benchmark, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Experiment Enhancements",
            "```json",
            json.dumps(experiment_enhancements, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Observed Evidence Diagnostics",
            "```json",
            json.dumps(observed_evidence, ensure_ascii=False, indent=2),
            "```",
            "",
            "## AI Explainability Diagnostics",
            "```json",
            json.dumps(ai_explainability, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Inference Reporting",
            "```json",
            json.dumps(inference_reporting, ensure_ascii=False, indent=2),
            "```",
            "",
            "## External Validity",
            "```json",
            json.dumps(external_validity, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Identification Plus Diagnostics",
            "```json",
            json.dumps(identification_plus, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Dynamic Causal Envelope",
            "```json",
            json.dumps(dynamic_causal_envelope, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Dynamic Method Core",
            "```json",
            json.dumps(dynamic_method_core, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Exogenous Shock Diagnostics",
            "```json",
            json.dumps(exogenous_shock, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Exogenous Shock Heterogeneity",
            "```json",
            json.dumps(exogenous_shock_heterogeneity, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Top-Tier Reinforcement Diagnostics",
            "```json",
            json.dumps(top_tier_reinforcement, ensure_ascii=False, indent=2),
            "```",
        ]
    )

    report = "\n".join(lines)
    (REPORTS_DIR / "final_report.md").write_text(report, encoding="utf-8")


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def _validate_real_source_integrity(
    *,
    source_audit: Dict[str, Any],
    strict_mode: bool,
    enforce_verified_sources: bool,
    min_verified_city_retention: float,
    min_external_direct_share: float,
    max_ai_inferred_share: float,
) -> Dict[str, Any]:
    gate: Dict[str, Any] = {
        "status": "ok",
        "strict_mode": bool(strict_mode),
        "enforce_verified_sources": bool(enforce_verified_sources),
        "min_verified_city_retention": float(min_verified_city_retention),
        "min_external_direct_share": float(min_external_direct_share),
        "max_ai_inferred_share": float(max_ai_inferred_share),
        "checks": [],
    }
    if not strict_mode:
        gate["status"] = "skipped"
        gate["reason"] = "strict_mode_disabled"
        return gate

    city_retention = float(source_audit.get("city_retention_ratio") or 0.0)
    verified_ratio = float(source_audit.get("verified_row_ratio") or 0.0)
    objective_city_ratio = float(source_audit.get("objective_complete_city_ratio") or 0.0)
    filter_basis = str(source_audit.get("filter_basis") or "")
    gate["verified_row_ratio_before_filter"] = verified_ratio
    gate["objective_complete_city_ratio_before_filter"] = objective_city_ratio
    gate["verified_city_retention_ratio"] = city_retention
    gate["source_filter_basis"] = filter_basis
    if enforce_verified_sources:
        metric_name = "verified_city_retention"
        metric_value = city_retention
        if "objective_city_complete_fallback" in filter_basis:
            metric_name = "objective_city_retention_fallback"
            metric_value = objective_city_ratio
        ok_retention = bool(metric_value + 1e-12 >= float(min_verified_city_retention))
        gate["checks"].append(
            {
                "name": metric_name,
                "status": "ok" if ok_retention else "failed",
                "value": metric_value,
                "threshold": float(min_verified_city_retention),
            }
        )
        if not ok_retention:
            msg = (
                "Source-retention check below threshold: "
                f"{metric_value:.3f} < {float(min_verified_city_retention):.3f}. "
                "Please improve objective-source coverage before modeling."
            )
            gate["status"] = "failed"
            gate["failure_reason"] = msg
            raise RuntimeError(msg)

    policy_audit = _read_json_if_exists(DATA_PROCESSED / "policy_event_registry_audit.json")
    if not policy_audit:
        gate["checks"].append(
            {
                "name": "policy_registry_audit_presence",
                "status": "failed",
                "value": None,
                "threshold": "required_in_strict_mode",
            }
        )
        msg = "Missing policy_event_registry_audit.json in strict mode."
        gate["status"] = "failed"
        gate["failure_reason"] = msg
        raise RuntimeError(msg)

    ext_share = policy_audit.get("iso3_share_external_direct")
    ai_share = policy_audit.get("iso3_share_ai_inferred")
    ext_share_val = float(ext_share) if ext_share is not None else None
    ai_share_val = float(ai_share) if ai_share is not None else None
    gate["policy_iso3_share_external_direct"] = ext_share_val
    gate["policy_iso3_share_ai_inferred"] = ai_share_val

    if ext_share_val is not None:
        ok_ext = bool(ext_share_val + 1e-12 >= float(min_external_direct_share))
        gate["checks"].append(
            {
                "name": "external_direct_share",
                "status": "ok" if ok_ext else "failed",
                "value": ext_share_val,
                "threshold": float(min_external_direct_share),
            }
        )
        if not ok_ext:
            msg = (
                "External-direct policy-event ISO3 share below threshold: "
                f"{ext_share_val:.3f} < {float(min_external_direct_share):.3f}."
            )
            gate["status"] = "failed"
            gate["failure_reason"] = msg
            raise RuntimeError(msg)

    if ai_share_val is not None:
        ok_ai = bool(ai_share_val <= float(max_ai_inferred_share) + 1e-12)
        gate["checks"].append(
            {
                "name": "ai_inferred_share_cap",
                "status": "ok" if ok_ai else "failed",
                "value": ai_share_val,
                "threshold": float(max_ai_inferred_share),
            }
        )
        if not ok_ai:
            msg = (
                "AI-inferred policy-event ISO3 share exceeds cap: "
                f"{ai_share_val:.3f} > {float(max_ai_inferred_share):.3f}."
            )
            gate["status"] = "failed"
            gate["failure_reason"] = msg
            raise RuntimeError(msg)

    return gate
