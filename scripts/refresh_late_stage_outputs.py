from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_eval import run_benchmark_suite
from src.causal_st import run_causal_st_analysis, run_causal_st_experiment_matrix
from src.export_dashboard import export_dashboard_data
from src.pipeline import _read_json_if_exists, _write_summary_report, setup_logging
from src.realtime_monitor import generate_realtime_monitor_snapshot
from src.research_matrix import build_research_matrix_report
from src.utils import DATA_OUTPUTS, DATA_PROCESSED, REPORTS_DIR, dump_json

LOGGER = logging.getLogger(__name__)


def _load_existing_summary() -> Dict[str, Any]:
    payload = _read_json_if_exists(REPORTS_DIR / "pipeline_summary.json")
    return payload if isinstance(payload, dict) else {}


def _load_inputs() -> Dict[str, Dict[str, Any]]:
    return {
        "model_metrics": _read_json_if_exists(DATA_OUTPUTS / "model_metrics.json"),
        "econometrics": _read_json_if_exists(DATA_OUTPUTS / "econometric_summary.json"),
        "experiment_enhancements": _read_json_if_exists(DATA_OUTPUTS / "experiment_enhancements.json"),
        "observed_evidence": _read_json_if_exists(DATA_OUTPUTS / "observed_evidence_summary.json"),
        "ai_explainability": _read_json_if_exists(DATA_OUTPUTS / "ai_explainability_summary.json"),
        "inference_reporting": _read_json_if_exists(DATA_OUTPUTS / "inference_reporting_summary.json"),
        "external_validity": _read_json_if_exists(DATA_OUTPUTS / "external_validity_summary.json"),
        "identification_plus": _read_json_if_exists(DATA_OUTPUTS / "idplus_summary.json"),
        "dynamic_causal_envelope": _read_json_if_exists(DATA_OUTPUTS / "dynamic_causal_envelope_summary.json"),
        "dynamic_method_core": _read_json_if_exists(DATA_OUTPUTS / "dynamic_method_core_summary.json"),
        "exogenous_shock": _read_json_if_exists(DATA_OUTPUTS / "exoshock_summary.json"),
        "exogenous_shock_heterogeneity": _read_json_if_exists(DATA_OUTPUTS / "exoshock_heterogeneity_summary.json"),
        "top_tier_reinforcement": _read_json_if_exists(DATA_OUTPUTS / "top_tier_reinforcement_summary.json"),
        "pulse_state": _read_json_if_exists(DATA_OUTPUTS / "pulse_state_summary.json"),
        "pulse_ai": _read_json_if_exists(DATA_OUTPUTS / "pulse_ai_summary.json"),
        "source_audit": _read_json_if_exists(DATA_PROCESSED / "source_audit_summary.json"),
        "coverage_report": _read_json_if_exists(DATA_PROCESSED / "coverage_summary.json"),
    }


def _build_summary(
    panel: pd.DataFrame,
    existing: Dict[str, Any],
    realtime_status: Dict[str, Any],
    benchmark: Dict[str, Any],
    causal_st: Dict[str, Any],
    causal_st_ablation: Dict[str, Any],
) -> Dict[str, Any]:
    summary = dict(existing)
    summary.update(
        {
            "city_count": int(panel["city_id"].nunique()),
            "year_count": int(panel["year"].nunique()),
            "records": int(len(panel)),
            "realtime_monitor": realtime_status,
            "refreshed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    outputs = dict(existing.get("outputs", {}))
    outputs.update(
        {
            "panel": str(DATA_PROCESSED / "global_city_panel.csv"),
            "panel_strict": str(DATA_PROCESSED / "global_city_panel_strict.csv"),
            "causal_st": "data/outputs/causal_st_summary.json",
            "causal_st_ablation": "data/outputs/causal_st_ablation.json",
            "benchmark": "data/outputs/benchmark_scores.json",
            "realtime_status": "web/static/data/realtime_status.json",
            "dashboard": "web/static/data",
        }
    )
    summary["outputs"] = outputs
    summary["late_stage_refresh"] = {
        "status": "ok",
        "causal_st_status": causal_st.get("status", "ok"),
        "causal_st_ablation_variants": len(causal_st_ablation.get("variants", [])),
        "benchmark_status": benchmark.get("status", "ok"),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh late-stage outputs from an existing strict panel.")
    parser.add_argument(
        "--panel",
        default=str(DATA_PROCESSED / "global_city_panel_strict.csv"),
        help="Path to the processed panel CSV.",
    )
    parser.add_argument(
        "--benchmark-target",
        default="auto",
        help="Target variable for benchmark evaluation.",
    )
    args = parser.parse_args()

    setup_logging(logging.INFO)
    panel_path = Path(args.panel)
    LOGGER.info("Loading strict panel from %s", panel_path)
    panel = pd.read_csv(panel_path, low_memory=False)

    LOGGER.info("Refreshing Causal-ST outputs...")
    causal_st = run_causal_st_analysis(panel)
    causal_st_ablation = run_causal_st_experiment_matrix(panel)

    benchmark_target = None if str(args.benchmark_target).strip().lower() == "auto" else args.benchmark_target
    LOGGER.info("Refreshing benchmark outputs for target=%s...", benchmark_target or "auto")
    benchmark = run_benchmark_suite(panel, target=benchmark_target)

    LOGGER.info("Refreshing dashboard snapshot and reports...")
    export_dashboard_data(panel)
    realtime_status = generate_realtime_monitor_snapshot(trigger="late_stage_refresh")
    existing = _load_existing_summary()
    inputs = _load_inputs()
    reliability_gate = existing.get("reliability_gate", {})
    strict_real_data = bool(existing.get("strict_real_data", True))
    _write_summary_report(
        panel,
        inputs["model_metrics"],
        inputs["econometrics"],
        causal_st,
        causal_st_ablation,
        benchmark,
        inputs["experiment_enhancements"],
        inputs["observed_evidence"],
        inputs["ai_explainability"],
        inputs["inference_reporting"],
        inputs["external_validity"],
        inputs["identification_plus"],
        inputs["dynamic_causal_envelope"],
        inputs["dynamic_method_core"],
        inputs["exogenous_shock"],
        inputs["exogenous_shock_heterogeneity"],
        inputs["top_tier_reinforcement"],
        inputs["pulse_state"],
        inputs["pulse_ai"],
        inputs["source_audit"],
        inputs["coverage_report"],
        reliability_gate,
        realtime_status,
        strict_real_data=strict_real_data,
    )
    build_research_matrix_report()
    summary = _build_summary(
        panel=panel,
        existing=existing,
        realtime_status=realtime_status,
        benchmark=benchmark,
        causal_st=causal_st,
        causal_st_ablation=causal_st_ablation,
    )
    dump_json(REPORTS_DIR / "pipeline_summary.json", summary)
    LOGGER.info("Late-stage refresh complete.")


if __name__ == "__main__":
    main()
