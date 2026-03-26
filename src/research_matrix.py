from __future__ import annotations

"""Assemble top-tier style experiment matrix report."""

import json
from pathlib import Path
from typing import Any, Dict

from .utils import DATA_OUTPUTS, DATA_PROCESSED, REPORTS_DIR


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_research_matrix_report() -> Path:
    """Build markdown report summarizing benchmark/ablation/audit results."""
    benchmark = _load_json(DATA_OUTPUTS / "benchmark_scores.json")
    cst = _load_json(DATA_OUTPUTS / "causal_st_summary.json")
    cst_ab = _load_json(DATA_OUTPUTS / "causal_st_ablation.json")
    source_audit = _load_json(DATA_PROCESSED / "source_audit_summary.json")
    coverage = _load_json(DATA_PROCESSED / "coverage_summary.json")
    policy_design = _load_json(DATA_PROCESSED / "policy_design.json")
    policy_registry = _load_json(DATA_PROCESSED / "policy_event_registry_audit.json")
    policy_macro = _load_json(DATA_PROCESSED / "policy_event_objective_macro_summary.json")
    policy_ai = _load_json(DATA_PROCESSED / "policy_event_ai_inference_summary.json")
    policy_evidence = _load_json(DATA_PROCESSED / "policy_event_evidence_summary.json")
    econ = _load_json(DATA_OUTPUTS / "econometric_summary.json")
    pulse = _load_json(DATA_OUTPUTS / "pulse_state_summary.json")
    pulse_ai = _load_json(DATA_OUTPUTS / "pulse_ai_summary.json")
    experiment = _load_json(DATA_OUTPUTS / "experiment_enhancements.json")
    ai_explain = _load_json(DATA_OUTPUTS / "ai_explainability_summary.json")
    inference = _load_json(DATA_OUTPUTS / "inference_reporting_summary.json")
    external = _load_json(DATA_OUTPUTS / "external_validity_summary.json")
    idplus = _load_json(DATA_OUTPUTS / "idplus_summary.json")
    exoshock = _load_json(DATA_OUTPUTS / "exoshock_summary.json")
    exoshock_het = _load_json(DATA_OUTPUTS / "exoshock_heterogeneity_summary.json")
    top_tier = _load_json(DATA_OUTPUTS / "top_tier_reinforcement_summary.json")

    lines = [
        "# MACRO-City Engine Experiment Matrix",
        "",
        "## Data Objectivity Audit",
        "```json",
        json.dumps(source_audit, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Coverage Diagnostics",
        "```json",
        json.dumps(coverage, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Policy Design & Registry Audit",
        "```json",
        json.dumps(
            {
                "policy_design": policy_design,
                "policy_registry_audit": policy_registry,
                "policy_objective_macro": policy_macro,
                "policy_ai_inference": policy_ai,
                "policy_evidence": policy_evidence,
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Benchmark (t->t+1 + Spatial OOD)",
        "```json",
        json.dumps(benchmark, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Causal-ST Main",
        "```json",
        json.dumps(cst, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Causal-ST Ablation",
        "```json",
        json.dumps(cst_ab, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Pulse State Summary",
        "```json",
        json.dumps(pulse, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Pulse AI Engine",
        "```json",
        json.dumps(pulse_ai, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Experiment Enhancements",
        "```json",
        json.dumps(experiment, ensure_ascii=False, indent=2),
        "```",
        "",
        "## AI Explainability Diagnostics",
        "```json",
        json.dumps(ai_explain, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Inference Reporting",
        "```json",
        json.dumps(inference, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Econometric Core",
        "```json",
        json.dumps(
            {
                "did_two_way_fe": econ.get("did_two_way_fe", {}),
                "did_twfe_cluster_bootstrap": econ.get("did_twfe_cluster_bootstrap", {}),
                "did_twfe_permutation": econ.get("did_twfe_permutation", {}),
                "did_twfe_wild_bootstrap": econ.get("did_twfe_wild_bootstrap", {}),
                "did_twfe_lead_placebo": econ.get("did_twfe_lead_placebo", {}),
                "did_matched_trend": econ.get("did_matched_trend", {}),
                "staggered_did": econ.get("staggered_did", {}),
                "not_yet_treated_did": econ.get("not_yet_treated_did", {}),
                "identification_scorecard": econ.get("identification_scorecard", {}),
                "dml_did": econ.get("dml_did", {}),
                "dr_did": econ.get("dr_did", {}),
                "event_study_fe_meta": econ.get("event_study_fe", {}).get("meta", {}),
                "matrix_completion_counterfactual": econ.get("matrix_completion_counterfactual", {}),
                "synthetic_control": econ.get("synthetic_control", {}),
                "direct_event_design": econ.get("direct_event_design", {}),
                "evidence_tier_design": econ.get("evidence_tier_design", {}),
                "evidence_tier_robustness": econ.get("evidence_tier_robustness", {}),
                "external_validity": external,
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Top-Tier Reinforcement",
        "```json",
        json.dumps(top_tier, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Identification Plus",
        "```json",
        json.dumps(idplus, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Exogenous Shock Design",
        "```json",
        json.dumps(exoshock, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Exogenous Shock Heterogeneity",
        "```json",
        json.dumps(exoshock_het, ensure_ascii=False, indent=2),
        "```",
    ]

    out = REPORTS_DIR / "research_matrix.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
