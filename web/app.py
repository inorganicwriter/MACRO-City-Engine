from __future__ import annotations

"""Flask app for global city analytics dashboard."""

import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from src.realtime_monitor import generate_realtime_monitor_snapshot

ROOT = Path(__file__).resolve().parents[1]
WEB_DATA = ROOT / "web" / "static" / "data"
DATA_OUTPUTS = ROOT / "data" / "outputs"
DATA_PROCESSED = ROOT / "data" / "processed"

app = Flask(__name__, template_folder=str(ROOT / "web" / "templates"), static_folder=str(ROOT / "web" / "static"))

_REGIME_DYNAMICS_CACHE: dict[str, object] = {"signature": None, "global": [], "continent": []}


def _truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        val = int(raw)
    except ValueError:
        return int(default)
    return int(max(10, val))


class RealtimeMonitorRuntime:
    """Background runtime that periodically refreshes realtime monitor snapshots."""

    def __init__(self, refresh_seconds: int = 900) -> None:
        self.refresh_seconds = int(max(10, refresh_seconds))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._version = 0
        self._last_status: dict[str, object] = {}
        self._last_error: str | None = None
        self._last_update_epoch = 0.0

    def _with_runtime_fields(self, payload: dict[str, object], reason: str) -> dict[str, object]:
        out = dict(payload)
        out["runtime_version"] = int(self._version)
        out["runtime_refresh_seconds"] = int(self.refresh_seconds)
        out["runtime_reason"] = str(reason)
        out["runtime_server_time_utc"] = datetime.now(timezone.utc).isoformat()
        out["runtime_last_error"] = self._last_error
        return out

    def trigger_update(self, reason: str = "manual") -> dict[str, object]:
        try:
            payload = generate_realtime_monitor_snapshot(trigger=reason)
            status_ok = True
            error_msg = None
        except Exception as exc:  # noqa: BLE001
            payload = {
                "status": "failed",
                "trigger": reason,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }
            status_ok = False
            error_msg = str(exc)

        with self._lock:
            self._version += 1
            self._last_error = error_msg if not status_ok else None
            self._last_update_epoch = time.time()
            self._last_status = self._with_runtime_fields(payload, reason)
            return dict(self._last_status)

    def get_status(self) -> dict[str, object]:
        with self._lock:
            status = dict(self._last_status)
            stale = bool((time.time() - self._last_update_epoch) > self.refresh_seconds and self._thread is None)
        if not status:
            status_path = WEB_DATA / "realtime_status.json"
            if status_path.exists():
                with status_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                with self._lock:
                    if not self._last_status:
                        self._version += 1
                        self._last_status = self._with_runtime_fields(payload, "warm_start")
                        self._last_update_epoch = time.time()
                    status = dict(self._last_status)
            else:
                status = self.trigger_update(reason="startup_on_demand")
        elif stale:
            status = self.trigger_update(reason="stale_on_demand")
        return status

    def _loop(self) -> None:
        self.trigger_update(reason="startup")
        while not self._stop.wait(self.refresh_seconds):
            self.trigger_update(reason="scheduled")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="urban-pulse-realtime-runtime", daemon=True)
        self._thread.start()


REALTIME_RUNTIME = RealtimeMonitorRuntime(refresh_seconds=_safe_int_env("URBAN_PULSE_MONITOR_REFRESH_SECONDS", 900))
if not _truthy_env(os.environ.get("URBAN_PULSE_DISABLE_BACKGROUND_MONITOR")):
    REALTIME_RUNTIME.start()


def _read_city_points() -> pd.DataFrame:
    path = WEB_DATA / "city_points.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_global_yearly() -> pd.DataFrame:
    path = WEB_DATA / "global_yearly.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_web_csv(name: str) -> pd.DataFrame:
    path = WEB_DATA / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _resolve_data_path(name: str, fallback_dirs: list[Path] | None = None) -> Path | None:
    primary = WEB_DATA / name
    if primary.exists():
        return primary
    for base in (fallback_dirs or []):
        p = base / name
        if p.exists():
            return p
    return None


def _read_csv_with_fallback(name: str, fallback_dirs: list[Path] | None = None) -> pd.DataFrame:
    path = _resolve_data_path(name, fallback_dirs=fallback_dirs)
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json_with_fallback(name: str, fallback_dirs: list[Path] | None = None) -> dict[str, object]:
    path = _resolve_data_path(name, fallback_dirs=fallback_dirs)
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _fmt_metric(value: object, digits: int = 3) -> str:
    x = _safe_float(value, default=float("nan"))
    if not np.isfinite(x):
        return "--"
    return f"{x:.{max(0, int(digits))}f}"


def _pick_nowcast_coverage95(payload: dict[str, object]) -> float:
    if not isinstance(payload, dict):
        return float("nan")
    cov95 = _safe_float(payload.get("global_backtest_coverage95"), default=float("nan"))
    if np.isfinite(cov95):
        return cov95
    return _safe_float(payload.get("global_backtest_coverage_95"), default=float("nan"))


def _parse_optional_int(value: object) -> int | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _build_top_tier_story_bundle(continent: str = "all", year: int | None = None) -> dict[str, object]:
    selected_continent = str(continent or "all").strip() or "all"
    selected_continent_key = selected_continent.lower()
    selected_year = _parse_optional_int(year)

    top = _read_json_with_fallback(
        "top_tier_reinforcement_summary.json",
        fallback_dirs=[DATA_OUTPUTS],
    )
    gate_df = _read_csv_with_fallback(
        "top_tier_gate_checks.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    evidence_df = _read_csv_with_fallback(
        "top_tier_evidence_convergence.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    spectrum_df = _read_csv_with_fallback(
        "top_tier_identification_spectrum.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    frontier_df = _read_csv_with_fallback(
        "top_tier_innovation_frontier.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    dashboard = _read_json_with_fallback(
        "dashboard_summary.json",
        fallback_dirs=[DATA_OUTPUTS],
    )
    nowcast = _read_json_with_fallback(
        "pulse_nowcast_summary.json",
        fallback_dirs=[DATA_OUTPUTS],
    )
    if isinstance(nowcast, dict):
        cov95 = _pick_nowcast_coverage95(nowcast)
        if np.isfinite(cov95):
            nowcast["global_backtest_coverage95"] = float(cov95)
    dynamics = _read_json_with_fallback(
        "pulse_dynamics_summary.json",
        fallback_dirs=[DATA_OUTPUTS],
    )
    method_core = _read_json_with_fallback(
        "dynamic_method_core_summary.json",
        fallback_dirs=[DATA_OUTPUTS],
    )
    readiness = _read_json_with_fallback(
        "submission_readiness.json",
        fallback_dirs=[DATA_OUTPUTS],
    )
    dynamic_index_df = _read_csv_with_fallback(
        "pulse_ai_dynamic_index_latest.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )

    gate_rows = gate_df.to_dict(orient="records") if not gate_df.empty else []
    evidence_rows = evidence_df.to_dict(orient="records") if not evidence_df.empty else []
    id_rows = spectrum_df.to_dict(orient="records") if not spectrum_df.empty else []
    frontier_rows = frontier_df.to_dict(orient="records") if not frontier_df.empty else []

    failed_gates: list[dict[str, object]] = []
    for row in gate_rows:
        passed = int(_safe_float(row.get("passed"), default=0.0))
        if passed != 1:
            failed_gates.append(dict(row))
    failed_gates = sorted(
        failed_gates,
        key=lambda r: abs(_safe_float(r.get("gap_ratio"), default=0.0)),
        reverse=True,
    )

    best_frontier: dict[str, object] = {}
    if frontier_rows:
        best_frontier = max(
            frontier_rows,
            key=lambda r: _safe_float(r.get("frontier_score"), default=float("-inf")),
        )

    primary_id_rows = [
        r for r in id_rows if int(_safe_float(r.get("is_primary_track"), default=0.0)) == 1
    ]
    strongest_id: dict[str, object] = {}
    if primary_id_rows:
        strongest_id = max(
            primary_id_rows,
            key=lambda r: _safe_float(r.get("identification_strength"), default=float("-inf")),
        )

    weakest_evidence: dict[str, object] = {}
    if evidence_rows:
        weakest_evidence = min(
            evidence_rows,
            key=lambda r: _safe_float(r.get("score_0_100"), default=float("inf")),
        )

    dynamic_snapshot: dict[str, object] = {
        "scope_continent": selected_continent,
        "scope_year": selected_year,
        "city_count": 0,
        "mean_stall_risk_score": float("nan"),
        "mean_acceleration_score": float("nan"),
        "fragile_boom_count": 0,
        "fragile_boom_share": float("nan"),
        "high_stall_risk_count": 0,
        "high_stall_risk_share": float("nan"),
        "high_risk_non_fragile_count": 0,
        "stable_low_risk_count": 0,
    }
    top_risk_rows: list[dict[str, object]] = []
    continent_profile_rows: list[dict[str, object]] = []
    if not dynamic_index_df.empty:
        dyn = dynamic_index_df.copy()
        dyn["year"] = pd.to_numeric(dyn.get("year"), errors="coerce")
        dyn["stall_risk_score"] = pd.to_numeric(dyn.get("stall_risk_score"), errors="coerce")
        dyn["acceleration_score"] = pd.to_numeric(dyn.get("acceleration_score"), errors="coerce")
        dyn["dynamic_pulse_index"] = pd.to_numeric(dyn.get("dynamic_pulse_index"), errors="coerce")

        if selected_continent_key != "all":
            dyn = dyn[dyn["continent"].astype(str).str.lower() == selected_continent_key]

        if selected_year is not None:
            year_subset = dyn[dyn["year"] == selected_year]
            if not year_subset.empty:
                dyn = year_subset
            else:
                selected_year = None

        if selected_year is None and not dyn.empty:
            year_vals = dyn["year"].dropna()
            if not year_vals.empty:
                selected_year = int(year_vals.max())
                dyn = dyn[dyn["year"] == selected_year]

        total_n = int(len(dyn))
        if total_n > 0:
            fragile_mask = (dyn["acceleration_score"] >= 50.0) & (dyn["stall_risk_score"] >= 50.0)
            high_risk_mask = dyn["stall_risk_score"] >= 60.0
            high_risk_non_fragile_mask = high_risk_mask & (~fragile_mask)

            fragile_n = int(fragile_mask.sum())
            high_risk_n = int(high_risk_mask.sum())
            high_risk_non_fragile_n = int(high_risk_non_fragile_mask.sum())
            stable_n = int(max(0, total_n - fragile_n - high_risk_non_fragile_n))
            dynamic_snapshot = {
                "scope_continent": selected_continent,
                "scope_year": selected_year,
                "city_count": total_n,
                "mean_stall_risk_score": float(dyn["stall_risk_score"].mean()),
                "mean_acceleration_score": float(dyn["acceleration_score"].mean()),
                "fragile_boom_count": fragile_n,
                "fragile_boom_share": float(fragile_n / max(total_n, 1)),
                "high_stall_risk_count": high_risk_n,
                "high_stall_risk_share": float(high_risk_n / max(total_n, 1)),
                "high_risk_non_fragile_count": high_risk_non_fragile_n,
                "stable_low_risk_count": stable_n,
            }
            top_risk_rows = (
                dyn.sort_values(["stall_risk_score", "acceleration_score"], ascending=[False, False])
                .head(20)[
                    [
                        "city_name",
                        "country",
                        "continent",
                        "year",
                        "stall_risk_score",
                        "acceleration_score",
                        "dynamic_pulse_index",
                        "trajectory_regime",
                    ]
                ]
                .to_dict(orient="records")
            )

        if selected_year is not None:
            profile = dynamic_index_df.copy()
            profile["year"] = pd.to_numeric(profile.get("year"), errors="coerce")
            profile = profile[profile["year"] == selected_year]
            if not profile.empty:
                profile["stall_risk_score"] = pd.to_numeric(profile.get("stall_risk_score"), errors="coerce")
                profile["acceleration_score"] = pd.to_numeric(profile.get("acceleration_score"), errors="coerce")
                profile["fragile"] = (
                    (profile["acceleration_score"] >= 50.0) & (profile["stall_risk_score"] >= 50.0)
                ).astype(float)
                continent_profile_rows = (
                    profile.groupby("continent", as_index=False)
                    .agg(
                        city_count=("city_id", "count"),
                        mean_stall_risk_score=("stall_risk_score", "mean"),
                        mean_acceleration_score=("acceleration_score", "mean"),
                        fragile_boom_share=("fragile", "mean"),
                    )
                    .sort_values("mean_stall_risk_score", ascending=False)
                    .to_dict(orient="records")
                )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_filters": {
            "continent": selected_continent,
            "year": selected_year,
        },
        "dashboard_summary": dashboard,
        "top_tier_summary": top,
        "gate_checks": gate_rows,
        "evidence_convergence": evidence_rows,
        "identification_spectrum": id_rows,
        "innovation_frontier": frontier_rows,
        "pulse_nowcast_summary": nowcast,
        "pulse_dynamics_summary": dynamics,
        "dynamic_method_core_summary": method_core,
        "submission_readiness": readiness,
        "dynamic_snapshot": dynamic_snapshot,
        "dynamic_top_risk_cities": top_risk_rows,
        "dynamic_continent_profile": continent_profile_rows,
        "highlights": {
            "failed_gates": failed_gates,
            "best_frontier_candidate": best_frontier,
            "strongest_primary_identification": strongest_id,
            "weakest_evidence_track": weakest_evidence,
        },
    }


def _compose_top_tier_story_markdown(bundle: dict[str, object]) -> str:
    selected_filters = bundle.get("selected_filters", {}) if isinstance(bundle, dict) else {}
    dashboard = bundle.get("dashboard_summary", {}) if isinstance(bundle, dict) else {}
    top = bundle.get("top_tier_summary", {}) if isinstance(bundle, dict) else {}
    nowcast = bundle.get("pulse_nowcast_summary", {}) if isinstance(bundle, dict) else {}
    dynamics = bundle.get("pulse_dynamics_summary", {}) if isinstance(bundle, dict) else {}
    method_core = bundle.get("dynamic_method_core_summary", {}) if isinstance(bundle, dict) else {}
    readiness = bundle.get("submission_readiness", {}) if isinstance(bundle, dict) else {}
    dynamic_snapshot = bundle.get("dynamic_snapshot", {}) if isinstance(bundle, dict) else {}
    top_risk_rows = bundle.get("dynamic_top_risk_cities", []) if isinstance(bundle, dict) else []
    highlights = bundle.get("highlights", {}) if isinstance(bundle, dict) else {}
    gate = top.get("top_tier_gate", {}) if isinstance(top, dict) else {}
    failed = highlights.get("failed_gates", []) if isinstance(highlights, dict) else []
    weakest_evidence = highlights.get("weakest_evidence_track", {}) if isinstance(highlights, dict) else {}
    strongest_id = highlights.get("strongest_primary_identification", {}) if isinstance(highlights, dict) else {}
    best_frontier = highlights.get("best_frontier_candidate", {}) if isinstance(highlights, dict) else {}
    method_delta = (
        method_core.get("global_vs_ridge_ar2", {})
        if isinstance(method_core, dict) and isinstance(method_core.get("global_vs_ridge_ar2", {}), dict)
        else {}
    )

    city_count = int(_safe_float(dashboard.get("city_count"), default=0.0))
    records = int(_safe_float(dashboard.get("records"), default=0.0))
    latest_year = int(_safe_float(dashboard.get("latest_year"), default=0.0))
    evidence_score = _fmt_metric(top.get("evidence_overall_score_0_100"), 3)
    gate_pass_rate = _fmt_metric(gate.get("gate_pass_rate"), 3)
    gate_ready = bool(gate.get("ready", False))
    nowcast_acc = _fmt_metric(nowcast.get("global_backtest_directional_accuracy"), 3)
    nowcast_cov = _fmt_metric(_pick_nowcast_coverage95(nowcast), 3)
    resilience = _fmt_metric(dynamics.get("global_resilience_score_0_100"), 2)
    warning_auc_h2 = _fmt_metric(dynamics.get("global_warning_auc_h2"), 3)
    method_mae_gain = _fmt_metric(method_delta.get("mae_gain"), 3)
    method_ternary_p = _fmt_metric(method_delta.get("ternary_accuracy_p_value"), 3)

    done_items = int(_safe_float(readiness.get("done_items"), default=0.0))
    total_items = int(_safe_float(readiness.get("total_items"), default=0.0))
    generated_at = str(bundle.get("generated_at_utc", "--"))
    scope_continent = str(selected_filters.get("continent", "all"))
    scope_year = selected_filters.get("year", "--")
    fragile_share = _fmt_metric(100.0 * _safe_float(dynamic_snapshot.get("fragile_boom_share"), default=float("nan")), 1)
    high_risk_share = _fmt_metric(100.0 * _safe_float(dynamic_snapshot.get("high_stall_risk_share"), default=float("nan")), 1)
    dynamic_city_count = int(_safe_float(dynamic_snapshot.get("city_count"), default=0.0))

    lines: list[str] = []
    lines.append("# MACRO-City Engine · Top-tier Review Story")
    lines.append("")
    lines.append(f"- Generated at (UTC): {generated_at}")
    lines.append(f"- Scope filter: continent={scope_continent}, year={scope_year}")
    lines.append(f"- Gate status: {'READY' if gate_ready else 'NOT READY'}")
    lines.append(f"- Gate pass rate: {gate_pass_rate}")
    lines.append(f"- Evidence score (0-100): {evidence_score}")
    lines.append("")
    lines.append("## 1) Question and Contribution")
    lines.append("- Core question: how to evaluate cities as dynamic systems (acceleration + stall risk) rather than static ranks.")
    lines.append("- Contribution: one closed chain from prediction -> identification -> policy-RL -> top-tier gate diagnostics.")
    lines.append("")
    lines.append("## 2) Data and Scope")
    lines.append(f"- Global panel: {city_count} cities, {records} city-year rows, latest year {latest_year}.")
    lines.append("- Objective-source construction and reproducible artifact export are enforced in the pipeline.")
    lines.append("")
    lines.append("## 3) Dynamic AI Evidence")
    lines.append(f"- Nowcast directional accuracy: {nowcast_acc}; 95% interval coverage: {nowcast_cov}.")
    lines.append(f"- Global resilience score: {resilience}; warning AUC at 2Y horizon: {warning_auc_h2}.")
    lines.append(f"- Dynamic-method MAE gain vs AR(2)-ridge baseline: {method_mae_gain} (ternary p-value={method_ternary_p}).")
    lines.append(f"- Fragile-boom share under scope: {fragile_share}% (high stall-risk share={high_risk_share}%, n={dynamic_city_count}).")
    lines.append("")
    lines.append("## 4) Identification and Causal Discipline")
    if isinstance(strongest_id, dict) and strongest_id:
        variant = str(strongest_id.get("design_variant", "--"))
        strength = _fmt_metric(strongest_id.get("identification_strength"), 3)
        robust_rate = _fmt_metric(100.0 * _safe_float(strongest_id.get("robust_pass_rate"), default=float("nan")), 1)
        lines.append(f"- Strongest primary identification track: `{variant}` (strength={strength}, robust pass rate={robust_rate}%).")
    else:
        lines.append("- Primary identification track summary is unavailable in current export.")
    lines.append("")
    lines.append("## 5) Innovation Frontier")
    if isinstance(best_frontier, dict) and best_frontier:
        auc = _fmt_metric(best_frontier.get("eval_roc_auc"), 3)
        brier = _fmt_metric(best_frontier.get("eval_brier"), 3)
        score = _fmt_metric(best_frontier.get("frontier_score"), 3)
        lines.append(f"- Best frontier candidate: AUC={auc}, Brier={brier}, frontier score={score}.")
    else:
        lines.append("- Innovation frontier candidate summary is unavailable in current export.")
    lines.append("")
    lines.append("## 6) Gate Gaps and Action Queue")
    if isinstance(failed, list) and failed:
        for idx, row in enumerate(failed[:8], start=1):
            gate_name = str(row.get("gate", "--"))
            metric = str(row.get("metric", "--"))
            val = _fmt_metric(row.get("value"), 4)
            thr = _fmt_metric(row.get("threshold"), 4)
            lines.append(f"{idx}. FAIL `{gate_name}` ({metric}): value={val}, threshold={thr}.")
    else:
        lines.append("- All top-tier gate checks pass in this export snapshot.")
    if isinstance(weakest_evidence, dict) and weakest_evidence:
        w_track = str(weakest_evidence.get("evidence_track", "--"))
        w_score = _fmt_metric(weakest_evidence.get("score_0_100"), 2)
        lines.append(f"- Weakest evidence track to prioritize next: `{w_track}` (score={w_score}).")
    if isinstance(top_risk_rows, list) and top_risk_rows:
        lines.append("- Top high-risk cities under current scope:")
        for row in top_risk_rows[:6]:
            city = str(row.get("city_name", "--"))
            country = str(row.get("country", "--"))
            risk = _fmt_metric(row.get("stall_risk_score"), 1)
            accel = _fmt_metric(row.get("acceleration_score"), 1)
            lines.append(f"  - {city} ({country}): risk={risk}, accel={accel}")
    lines.append("")
    lines.append("## 7) Submission Readiness")
    lines.append(f"- Checklist completion: {done_items}/{total_items}.")
    lines.append("- Recommended workflow: keep this story page as reviewer-facing summary and append full robustness tables in the manuscript appendix.")
    lines.append("")
    return "\n".join(lines)


def _regime_rank(name: str) -> int:
    mapping = {
        "structural_decline": 0,
        "stalling_plateau": 1,
        "volatile_rebound": 2,
        "late_takeoff": 3,
        "stable_mature": 4,
        "persistent_accelerator": 5,
    }
    key = str(name).strip()
    return int(mapping.get(key, 2))


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _attach_regime_rate_confidence_bands(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    n = pd.to_numeric(out["n_transitions"], errors="coerce").fillna(0.0).clip(lower=1.0)
    z = 1.96
    for metric in ["switch_rate", "upgrade_share", "downgrade_share", "self_share"]:
        p = pd.to_numeric(out[metric], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
        se = np.sqrt((p * (1.0 - p)) / n)
        out[f"{metric}_ci_low"] = np.clip(p - (z * se), 0.0, 1.0)
        out[f"{metric}_ci_high"] = np.clip(p + (z * se), 0.0, 1.0)

    p_up = pd.to_numeric(out["upgrade_share"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    p_down = pd.to_numeric(out["downgrade_share"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    diff = p_up - p_down
    var_diff = np.clip((p_up + p_down - np.square(diff)) / n, a_min=0.0, a_max=None)
    se_diff = np.sqrt(var_diff)
    out["net_flow_ci_low_pp"] = 100.0 * (diff - (z * se_diff))
    out["net_flow_ci_high_pp"] = 100.0 * (diff + (z * se_diff))
    return out


def _annotate_regime_dynamics(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = group_cols or []
    sort_cols = [*group_cols, "year"] if group_cols else ["year"]
    out = df.copy().sort_values(sort_cols)

    p_up = pd.to_numeric(out["upgrade_share"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    p_down = pd.to_numeric(out["downgrade_share"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    n = pd.to_numeric(out["n_transitions"], errors="coerce").fillna(0.0).clip(lower=1.0)
    diff = p_up - p_down
    var_diff = np.clip((p_up + p_down - np.square(diff)) / n, a_min=0.0, a_max=None)
    se_diff = np.sqrt(var_diff)
    z_stat = np.where(se_diff > 1e-12, diff / se_diff, np.nan)
    p_values = np.array(
        [2.0 * (1.0 - _normal_cdf(abs(float(v)))) if np.isfinite(v) else np.nan for v in z_stat],
        dtype=float,
    )
    out["net_flow_z_stat"] = z_stat
    out["net_flow_p_value"] = p_values
    out["net_flow_significant_5pct"] = (p_values < 0.05).astype(int)

    if group_cols:
        out["switch_rate_delta_pp"] = (
            out.groupby(group_cols)["switch_rate"].diff().astype(float) * 100.0
        )
        delta_std = out.groupby(group_cols)["switch_rate_delta_pp"].transform(
            lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0))
        )
        out["switch_rate_delta_z"] = out["switch_rate_delta_pp"] / delta_std.replace(0.0, np.nan)
    else:
        out["switch_rate_delta_pp"] = out["switch_rate"].diff().astype(float) * 100.0
        std_val = float(pd.to_numeric(out["switch_rate_delta_pp"], errors="coerce").std(ddof=0))
        out["switch_rate_delta_z"] = out["switch_rate_delta_pp"] / (std_val if std_val > 1e-12 else np.nan)

    out["switch_break_flag"] = (
        pd.to_numeric(out["switch_rate_delta_z"], errors="coerce").abs().fillna(0.0) >= 1.96
    ).astype(int)
    return out


@app.get("/")
def portal():
    return render_template("dashboard_unified.html", initial_tab="overview")


@app.get("/dashboard/full")
def index():
    return render_template("dashboard_unified.html", initial_tab="overview")


@app.get("/dashboard/dynamics")
def dashboard_dynamics():
    return render_template("dashboard_unified.html", initial_tab="dynamics")


@app.get("/dashboard/method-core")
def dashboard_method_core():
    return render_template("dashboard_unified.html", initial_tab="evidence")


@app.get("/dashboard/realtime")
def dashboard_realtime():
    return render_template("dashboard_unified.html", initial_tab="overview")


@app.get("/dashboard/identification")
def dashboard_identification():
    return render_template("dashboard_unified.html", initial_tab="evidence")


@app.get("/dashboard/external-validity")
def dashboard_external_validity():
    return render_template("dashboard_unified.html", initial_tab="evidence")


@app.get("/dashboard/policy-rl")
def dashboard_policy_rl():
    return render_template("dashboard_unified.html", initial_tab="actions")


@app.get("/dashboard/top-tier")
def dashboard_top_tier():
    return render_template("dashboard_unified.html", initial_tab="actions")


@app.get("/dashboard/top-tier-story")
def dashboard_top_tier_story():
    return render_template("dashboard_unified.html", initial_tab="actions")


@app.get("/api/frontend_bundle")
def api_frontend_bundle():
    continent = request.args.get("continent", "all")
    year = _parse_optional_int(request.args.get("year"))
    bundle = _build_top_tier_story_bundle(continent=continent, year=year)

    selected = bundle.get("selected_filters", {}) if isinstance(bundle, dict) else {}
    selected_continent = str(selected.get("continent", "all"))
    selected_year = _parse_optional_int(selected.get("year"))

    nowcast_global_df = _read_csv_with_fallback(
        "pulse_nowcast_global.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    nowcast_global_rows: list[dict[str, object]] = []
    year_candidates: set[int] = set()
    if not nowcast_global_df.empty:
        ng = nowcast_global_df.copy()
        ng["year"] = pd.to_numeric(ng.get("year"), errors="coerce")
        ng = ng.dropna(subset=["year"]).copy()
        ng["year"] = ng["year"].astype(int)
        year_candidates.update(int(v) for v in ng["year"].dropna().unique().tolist())
        if selected_year is not None:
            ng = ng[ng["year"] <= selected_year]
        nowcast_global_rows = ng.sort_values("year").to_dict(orient="records")

    dynamic_index_df = _read_csv_with_fallback(
        "pulse_ai_dynamic_index_latest.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    dynamic_index_rows: list[dict[str, object]] = []
    continent_candidates: set[str] = set()
    if not dynamic_index_df.empty:
        dyn = dynamic_index_df.copy()
        dyn["year"] = pd.to_numeric(dyn.get("year"), errors="coerce")
        dyn = dyn.dropna(subset=["year"]).copy()
        dyn["year"] = dyn["year"].astype(int)
        year_candidates.update(int(v) for v in dyn["year"].dropna().unique().tolist())
        continent_candidates.update(
            str(v) for v in dyn.get("continent", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        )

        if selected_continent.strip().lower() != "all":
            dyn = dyn[dyn["continent"].astype(str).str.lower() == selected_continent.strip().lower()]
        if selected_year is not None:
            dyn = dyn[dyn["year"] == selected_year]

        if not dyn.empty:
            dynamic_index_rows = dyn.sort_values(
                ["stall_risk_score", "acceleration_score"],
                ascending=[False, False],
            ).to_dict(orient="records")

    top = bundle.get("top_tier_summary", {}) if isinstance(bundle, dict) else {}
    gate = top.get("top_tier_gate", {}) if isinstance(top, dict) else {}
    nowcast_summary = bundle.get("pulse_nowcast_summary", {}) if isinstance(bundle, dict) else {}
    cov95 = _pick_nowcast_coverage95(nowcast_summary if isinstance(nowcast_summary, dict) else {})
    if isinstance(nowcast_summary, dict) and np.isfinite(cov95):
        nowcast_summary["global_backtest_coverage95"] = float(cov95)

    payload = {
        "status": "ok",
        "active_filters": {
            "continent": selected_continent,
            "year": selected_year,
        },
        "filter_options": {
            "continents": sorted(continent_candidates),
            "years": sorted(year_candidates),
        },
        "kpis": {
            "gate_ready": bool(gate.get("ready", False)),
            "gate_pass_rate": _safe_float(gate.get("gate_pass_rate"), default=float("nan")),
            "evidence_score_0_100": _safe_float(top.get("evidence_overall_score_0_100"), default=float("nan")),
            "nowcast_directional_accuracy": _safe_float(
                nowcast_summary.get("global_backtest_directional_accuracy"),
                default=float("nan"),
            ),
            "nowcast_coverage95": cov95,
            "fragile_boom_share": _safe_float(
                (bundle.get("dynamic_snapshot", {}) if isinstance(bundle, dict) else {}).get("fragile_boom_share"),
                default=float("nan"),
            ),
        },
        "top_tier_summary": top,
        "gate_checks": bundle.get("gate_checks", []),
        "evidence_convergence": bundle.get("evidence_convergence", []),
        "identification_spectrum": bundle.get("identification_spectrum", []),
        "nowcast_summary": nowcast_summary,
        "nowcast_global": nowcast_global_rows,
        "dynamic_snapshot": bundle.get("dynamic_snapshot", {}),
        "dynamic_index_rows": dynamic_index_rows,
        "dynamic_top_risk_cities": bundle.get("dynamic_top_risk_cities", []),
        "method_core_summary": bundle.get("dynamic_method_core_summary", {}),
        "submission_readiness": bundle.get("submission_readiness", {}),
        "highlights": bundle.get("highlights", {}),
    }
    return jsonify(payload)


@app.get("/api/cities")
def api_cities():
    points = _read_city_points()
    if points.empty:
        return jsonify([])

    metric = request.args.get("metric", "composite_index")
    year = int(request.args.get("year", int(points["year"].max())))
    continent = request.args.get("continent", "all").strip().lower()
    keyword = request.args.get("q", "").strip().lower()

    if metric not in {"composite_index", "economic_vitality", "livability", "innovation"}:
        metric = "composite_index"

    subset = points[points["year"] == year].copy()
    if continent != "all":
        subset = subset[subset["continent"].str.lower() == continent]
    if keyword:
        subset = subset[
            subset["city_name"].str.lower().str.contains(keyword, na=False)
            | subset["country"].str.lower().str.contains(keyword, na=False)
        ]
    subset = subset.sort_values(metric, ascending=False)

    records = subset[
        [
            "city_id",
            "city_name",
            "country",
            "continent",
            "latitude",
            "longitude",
            "year",
            "treated_city",
            "economic_vitality",
            "livability",
            "innovation",
            "composite_index",
        ]
    ].to_dict(orient="records")

    return jsonify(records)


@app.get("/api/city/<city_id>")
def api_city_series(city_id: str):
    points = _read_city_points()
    if points.empty:
        return jsonify([])

    subset = points[points["city_id"] == city_id].sort_values("year")
    return jsonify(
        subset[
            ["year", "economic_vitality", "livability", "innovation", "composite_index", "treated_city"]
        ].to_dict(orient="records")
    )


@app.get("/api/global")
def api_global_series():
    yearly = _read_global_yearly()
    if yearly.empty:
        return jsonify([])
    return jsonify(yearly.sort_values("year").to_dict(orient="records"))


@app.get("/api/econometrics")
def api_econometrics():
    path = WEB_DATA / "econometric_summary.json"
    if not path.exists():
        return jsonify({})

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/econometric_policy_source_sensitivity")
def api_econometric_policy_source_sensitivity():
    df = _read_csv_with_fallback(
        "econometric_policy_source_sensitivity.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/econometric_source_event_study_points")
def api_econometric_source_event_study_points():
    df = _read_csv_with_fallback(
        "econometric_source_event_study_points.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/idplus_summary")
def api_idplus_summary():
    payload = _read_json_with_fallback(
        "idplus_summary.json",
        fallback_dirs=[DATA_OUTPUTS],
    )
    return jsonify(payload)


@app.get("/api/idplus_stress_index")
def api_idplus_stress_index():
    df = _read_csv_with_fallback(
        "idplus_identification_stress_index.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/idplus_concordance_pairs")
def api_idplus_concordance_pairs():
    df = _read_csv_with_fallback(
        "idplus_design_concordance_pairs.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/idplus_pretrend_geometry")
def api_idplus_pretrend_geometry():
    df = _read_csv_with_fallback(
        "idplus_event_pretrend_geometry.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/idplus_leave_continent_stability")
def api_idplus_leave_continent_stability():
    df = _read_csv_with_fallback(
        "idplus_leave_continent_stability_summary.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/dce_summary")
def api_dce_summary():
    payload = _read_json_with_fallback(
        "dynamic_causal_envelope_summary.json",
        fallback_dirs=[DATA_OUTPUTS],
    )
    return jsonify(payload)


@app.get("/api/dce_event")
def api_dce_event():
    df = _read_csv_with_fallback(
        "dynamic_causal_envelope_event.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/dce_event_bootstrap")
def api_dce_event_bootstrap():
    df = _read_csv_with_fallback(
        "dynamic_causal_envelope_event_bootstrap.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/dce_city_scores")
def api_dce_city_scores():
    df = _read_csv_with_fallback(
        "dynamic_causal_envelope_city_scores.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/dce_continent_stability")
def api_dce_continent_stability():
    df = _read_csv_with_fallback(
        "dynamic_causal_envelope_continent_stability.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/dce_continent_year")
def api_dce_continent_year():
    df = _read_csv_with_fallback(
        "dynamic_causal_envelope_continent_year.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/dce_regime_summary")
def api_dce_regime_summary():
    df = _read_csv_with_fallback(
        "dynamic_causal_envelope_regime_summary.csv",
        fallback_dirs=[DATA_OUTPUTS],
    )
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/dashboard_summary")
def api_dashboard_summary():
    path = WEB_DATA / "dashboard_summary.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/top_tier")
def api_top_tier():
    path = WEB_DATA / "top_tier_reinforcement_summary.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/top_tier_gate_checks")
def api_top_tier_gate_checks():
    path = WEB_DATA / "top_tier_gate_checks.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/top_tier_identification_spectrum")
def api_top_tier_identification_spectrum():
    path = WEB_DATA / "top_tier_identification_spectrum.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/top_tier_evidence_convergence")
def api_top_tier_evidence_convergence():
    path = WEB_DATA / "top_tier_evidence_convergence.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/top_tier_innovation_frontier")
def api_top_tier_innovation_frontier():
    path = WEB_DATA / "top_tier_innovation_frontier.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/top_tier_story_bundle")
def api_top_tier_story_bundle():
    continent = request.args.get("continent", "all")
    year = _parse_optional_int(request.args.get("year"))
    return jsonify(_build_top_tier_story_bundle(continent=continent, year=year))


@app.get("/api/top_tier_story_markdown")
def api_top_tier_story_markdown():
    continent = request.args.get("continent", "all")
    year = _parse_optional_int(request.args.get("year"))
    bundle = _build_top_tier_story_bundle(continent=continent, year=year)
    content = _compose_top_tier_story_markdown(bundle)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    headers = {
        "Content-Disposition": f'attachment; filename="macro_city_engine_top_tier_story_{stamp}.md"',
        "Cache-Control": "no-store",
    }
    return Response(content, mimetype="text/markdown; charset=utf-8", headers=headers)


@app.get("/api/submission_readiness")
def api_submission_readiness():
    payload = _read_json_with_fallback(
        "submission_readiness.json",
        fallback_dirs=[DATA_OUTPUTS],
    )
    return jsonify(payload)


@app.get("/api/benchmark")
def api_benchmark():
    path = WEB_DATA / "benchmark_scores.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/experiment_enhancements")
def api_experiment_enhancements():
    path = WEB_DATA / "experiment_enhancements.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/robustness_audit")
def api_robustness_audit():
    path = WEB_DATA / "robustness_audit_summary.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/robustness_gate_checks")
def api_robustness_gate_checks():
    path = WEB_DATA / "robustness_gate_checks.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/reproducibility_manifest")
def api_reproducibility_manifest():
    path = WEB_DATA / "reproducibility_manifest.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/external_validity")
def api_external_validity():
    path = WEB_DATA / "external_validity_summary.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/external_validity_indicators")
def api_external_validity_indicators():
    path = WEB_DATA / "external_validity_indicator_results.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/external_validity_rank_corr")
def api_external_validity_rank_corr():
    path = WEB_DATA / "external_validity_rank_corr_by_year.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/coverage_summary")
def api_coverage_summary():
    path = WEB_DATA / "coverage_summary.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/policy_design")
def api_policy_design():
    path = WEB_DATA / "policy_design.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/policy_registry_audit")
def api_policy_registry_audit():
    path = WEB_DATA / "policy_event_registry_audit.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/policy_registry_quality")
def api_policy_registry_quality():
    path = WEB_DATA / "policy_event_registry_quality_report.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/policy_registry_events")
def api_policy_registry_events():
    path = WEB_DATA / "policy_event_registry_enriched.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    iso3 = request.args.get("iso3", "").strip().upper()
    if iso3:
        df = df[df["iso3"].astype(str).str.upper() == iso3]
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/policy_ai_inference")
def api_policy_ai_inference():
    path = WEB_DATA / "policy_event_ai_inference_summary.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/policy_objective_macro")
def api_policy_objective_macro():
    path = WEB_DATA / "policy_event_objective_macro_summary.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/policy_evidence")
def api_policy_evidence():
    path = WEB_DATA / "policy_event_evidence_summary.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/coverage_continent")
def api_coverage_continent():
    path = WEB_DATA / "coverage_by_continent.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/causal_st")
def api_causal_st():
    path = WEB_DATA / "causal_st_summary.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/causal_st_ablation")
def api_causal_st_ablation():
    path = WEB_DATA / "causal_st_ablation.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/pulse_state")
def api_pulse_state():
    path = WEB_DATA / "pulse_state_latest.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    city_id = request.args.get("city_id", "").strip()
    if city_id:
        df = df[df["city_id"] == city_id]
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai")
def api_pulse_ai():
    path = WEB_DATA / "pulse_ai_summary.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/pulse_ai_cities")
def api_pulse_ai_cities():
    path = WEB_DATA / "pulse_ai_city_latest.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    city_id = request.args.get("city_id", "").strip()
    if city_id:
        df = df[df["city_id"] == city_id]
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_regime_share")
def api_pulse_ai_regime_share():
    path = WEB_DATA / "pulse_ai_regime_year_share.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_regime_dynamics")
def api_pulse_ai_regime_dynamics():
    regime_path = WEB_DATA / "pulse_ai_regime_by_year.csv"
    points_path = WEB_DATA / "city_points.csv"
    shock_path = WEB_DATA / "pulse_ai_shock_years.csv"
    signature = (
        float(regime_path.stat().st_mtime) if regime_path.exists() else None,
        float(points_path.stat().st_mtime) if points_path.exists() else None,
        float(shock_path.stat().st_mtime) if shock_path.exists() else None,
    )

    use_cache = _REGIME_DYNAMICS_CACHE.get("signature") == signature
    if use_cache:
        global_metrics = pd.DataFrame(_REGIME_DYNAMICS_CACHE.get("global", []))
        continent_metrics = pd.DataFrame(_REGIME_DYNAMICS_CACHE.get("continent", []))
    else:
        if not regime_path.exists():
            return jsonify({"global": [], "continent": []})

        regime = pd.read_csv(regime_path)
        required_cols = {"city_id", "year", "trajectory_regime"}
        if regime.empty or not required_cols.issubset(regime.columns):
            return jsonify({"global": [], "continent": []})

        regime["city_id"] = regime["city_id"].astype(str)
        regime["trajectory_regime"] = regime["trajectory_regime"].astype(str)
        regime["year"] = pd.to_numeric(regime["year"], errors="coerce")
        regime = regime.dropna(subset=["city_id", "year", "trajectory_regime"]).copy()
        if regime.empty:
            return jsonify({"global": [], "continent": []})

        regime["year"] = regime["year"].astype(int)
        regime = regime.sort_values(["city_id", "year"]).copy()
        regime["prev_regime"] = regime.groupby("city_id")["trajectory_regime"].shift(1)

        transitions = regime.dropna(subset=["prev_regime"]).copy()
        if transitions.empty:
            return jsonify({"global": [], "continent": []})

        transitions["prev_regime"] = transitions["prev_regime"].astype(str)
        transitions["switch"] = (transitions["trajectory_regime"] != transitions["prev_regime"]).astype(float)
        transitions["rank_prev"] = transitions["prev_regime"].map(_regime_rank)
        transitions["rank_curr"] = transitions["trajectory_regime"].map(_regime_rank)
        transitions["is_upgrade"] = (transitions["rank_curr"] > transitions["rank_prev"]).astype(float)
        transitions["is_downgrade"] = (transitions["rank_curr"] < transitions["rank_prev"]).astype(float)
        transitions["is_self"] = (transitions["rank_curr"] == transitions["rank_prev"]).astype(float)

        if points_path.exists():
            points = pd.read_csv(points_path, usecols=["city_id", "continent"])
            points["city_id"] = points["city_id"].astype(str)
            points["continent"] = points["continent"].astype(str)
            city_continent = points.dropna(subset=["city_id", "continent"]).drop_duplicates(subset=["city_id"])
            transitions = transitions.merge(city_continent, on="city_id", how="left")
            regime = regime.merge(city_continent, on="city_id", how="left")
        else:
            transitions["continent"] = "Unknown"
            regime["continent"] = "Unknown"

        transitions["continent"] = transitions["continent"].fillna("Unknown").astype(str)
        regime["continent"] = regime["continent"].fillna("Unknown").astype(str)

        def _entropy(series: pd.Series) -> float:
            probs = series.value_counts(normalize=True)
            return float(-sum(float(p) * math.log(max(float(p), 1e-12)) for p in probs))

        global_metrics = (
            transitions.groupby("year", as_index=False)
            .agg(
                n_transitions=("city_id", "count"),
                switch_rate=("switch", "mean"),
                upgrade_share=("is_upgrade", "mean"),
                downgrade_share=("is_downgrade", "mean"),
                self_share=("is_self", "mean"),
            )
            .sort_values("year")
        )
        global_entropy = (
            regime.groupby("year")["trajectory_regime"]
            .apply(_entropy)
            .reset_index(name="regime_entropy")
            .sort_values("year")
        )
        global_metrics = global_metrics.merge(global_entropy, on="year", how="left")
        global_metrics["net_flow_pp"] = 100.0 * (
            global_metrics["upgrade_share"] - global_metrics["downgrade_share"]
        )

        continent_metrics = (
            transitions.groupby(["continent", "year"], as_index=False)
            .agg(
                n_transitions=("city_id", "count"),
                switch_rate=("switch", "mean"),
                upgrade_share=("is_upgrade", "mean"),
                downgrade_share=("is_downgrade", "mean"),
                self_share=("is_self", "mean"),
            )
            .sort_values(["continent", "year"])
        )
        continent_entropy = (
            regime.groupby(["continent", "year"])["trajectory_regime"]
            .apply(_entropy)
            .reset_index(name="regime_entropy")
            .sort_values(["continent", "year"])
        )
        continent_metrics = continent_metrics.merge(continent_entropy, on=["continent", "year"], how="left")
        continent_metrics["net_flow_pp"] = 100.0 * (
            continent_metrics["upgrade_share"] - continent_metrics["downgrade_share"]
        )

        shock_years: set[int] = set()
        if shock_path.exists():
            shock_df = pd.read_csv(shock_path)
            if "shock_year" in shock_df.columns:
                shock_years = {
                    int(v)
                    for v in pd.to_numeric(shock_df["shock_year"], errors="coerce").dropna().astype(int).tolist()
                }
        global_metrics["shock_anchor"] = global_metrics["year"].astype(int).isin(shock_years).astype(int)
        continent_metrics["shock_anchor"] = continent_metrics["year"].astype(int).isin(shock_years).astype(int)

        global_metrics = _attach_regime_rate_confidence_bands(global_metrics)
        continent_metrics = _attach_regime_rate_confidence_bands(continent_metrics)
        global_metrics = _annotate_regime_dynamics(global_metrics, group_cols=[])
        continent_metrics = _annotate_regime_dynamics(continent_metrics, group_cols=["continent"])

        _REGIME_DYNAMICS_CACHE["signature"] = signature
        _REGIME_DYNAMICS_CACHE["global"] = global_metrics.to_dict(orient="records")
        _REGIME_DYNAMICS_CACHE["continent"] = continent_metrics.to_dict(orient="records")

    continent_filter = request.args.get("continent", "all").strip().lower()
    if continent_filter and continent_filter != "all":
        continent_metrics = continent_metrics[continent_metrics["continent"].astype(str).str.lower() == continent_filter]

    payload = {
        "global": global_metrics.to_dict(orient="records"),
        "continent": continent_metrics.to_dict(orient="records"),
    }
    return jsonify(payload)


@app.get("/api/pulse_ai_regime_transition")
def api_pulse_ai_regime_transition():
    path = WEB_DATA / "pulse_ai_regime_transition_matrix.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_trajectory_regimes")
def api_pulse_ai_trajectory_regimes():
    path = WEB_DATA / "pulse_ai_trajectory_regimes.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_cross_continent")
def api_pulse_ai_cross_continent():
    path = WEB_DATA / "pulse_ai_cross_continent_generalization.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_shock_irf_regime")
def api_pulse_ai_shock_irf_regime():
    path = WEB_DATA / "pulse_ai_shock_irf_regime.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_shock_years")
def api_pulse_ai_shock_years():
    path = WEB_DATA / "pulse_ai_shock_years.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_horizon")
def api_pulse_ai_horizon():
    path = WEB_DATA / "pulse_ai_horizon_forecast.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_cycle")
def api_pulse_ai_dynamic_cycle():
    path = WEB_DATA / "pulse_ai_dynamic_cycle.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_phase_field")
def api_pulse_ai_dynamic_phase_field():
    path = WEB_DATA / "pulse_ai_dynamic_phase_field.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_phase_latest")
def api_pulse_ai_dynamic_phase_latest():
    path = WEB_DATA / "pulse_ai_dynamic_phase_latest.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_hazard")
def api_pulse_ai_dynamic_hazard():
    path = WEB_DATA / "pulse_ai_dynamic_hazard_latest.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_graph")
def api_pulse_ai_dynamic_graph():
    path = WEB_DATA / "pulse_ai_dynamic_graph_latest.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_graph_curve")
def api_pulse_ai_dynamic_graph_curve():
    path = WEB_DATA / "pulse_ai_dynamic_graph_curve.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_event")
def api_pulse_ai_dynamic_event():
    path = WEB_DATA / "pulse_ai_dynamic_state_event_study.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_critical")
def api_pulse_ai_dynamic_critical():
    path = WEB_DATA / "pulse_ai_dynamic_critical_decile.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_fusion")
def api_pulse_ai_dynamic_fusion():
    path = WEB_DATA / "pulse_ai_dynamic_main_fusion_curve.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_fusion_backtest")
def api_pulse_ai_dynamic_fusion_backtest():
    path = WEB_DATA / "pulse_ai_dynamic_main_fusion_backtest.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_fusion_pareto")
def api_pulse_ai_dynamic_fusion_pareto():
    path = WEB_DATA / "pulse_ai_dynamic_main_fusion_pareto.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_fusion_eval")
def api_pulse_ai_dynamic_fusion_eval():
    path = WEB_DATA / "pulse_ai_dynamic_main_fusion_eval.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_fusion_state_gate")
def api_pulse_ai_dynamic_fusion_state_gate():
    path = WEB_DATA / "pulse_ai_dynamic_main_fusion_state_gate.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_fusion_continent")
def api_pulse_ai_dynamic_fusion_continent():
    path = WEB_DATA / "pulse_ai_dynamic_main_fusion_continent_eval.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_fusion_continent_robust")
def api_pulse_ai_dynamic_fusion_continent_robust():
    path = WEB_DATA / "pulse_ai_dynamic_main_fusion_continent_robust_curve.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_fusion_continent_adaptive")
def api_pulse_ai_dynamic_fusion_continent_adaptive():
    path = WEB_DATA / "pulse_ai_dynamic_main_fusion_continent_adaptive_weights.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_sync")
def api_pulse_ai_dynamic_sync():
    path = WEB_DATA / "pulse_ai_dynamic_sync_network.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_policy")
def api_pulse_ai_dynamic_policy():
    path = WEB_DATA / "pulse_ai_dynamic_policy_lab_summary.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_policy_rl")
def api_pulse_ai_dynamic_policy_rl():
    path = WEB_DATA / "pulse_ai_dynamic_policy_rl_action_summary.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_policy_rl_city")
def api_pulse_ai_dynamic_policy_rl_city():
    path = WEB_DATA / "pulse_ai_dynamic_policy_rl_city.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    continent = request.args.get("continent", "").strip().lower()
    if continent and continent != "all":
        df = df[df["continent"].astype(str).str.lower() == continent]
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_policy_rl_state")
def api_pulse_ai_dynamic_policy_rl_state():
    path = WEB_DATA / "pulse_ai_dynamic_policy_rl_state_value.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_policy_rl_ope")
def api_pulse_ai_dynamic_policy_rl_ope():
    path = WEB_DATA / "pulse_ai_dynamic_policy_rl_ope.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_policy_rl_ablation")
def api_pulse_ai_dynamic_policy_rl_ablation():
    path = WEB_DATA / "pulse_ai_dynamic_policy_rl_ablation.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_policy_rl_continent_ope")
def api_pulse_ai_dynamic_policy_rl_continent_ope():
    path = WEB_DATA / "pulse_ai_dynamic_policy_rl_continent_ope.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    estimator = request.args.get("estimator", "").strip().lower()
    if estimator:
        df = df[df["estimator"].astype(str).str.lower() == estimator]
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_policy_rl_continent_action")
def api_pulse_ai_dynamic_policy_rl_continent_action():
    path = WEB_DATA / "pulse_ai_dynamic_policy_rl_continent_action.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    continent = request.args.get("continent", "").strip().lower()
    if continent and continent != "all":
        df = df[df["continent"].astype(str).str.lower() == continent]
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_index_latest")
def api_pulse_ai_dynamic_index_latest():
    path = WEB_DATA / "pulse_ai_dynamic_index_latest.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    continent = request.args.get("continent", "").strip().lower()
    if continent and continent != "all":
        df = df[df["continent"].astype(str).str.lower() == continent]
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_ai_dynamic_index_continent")
def api_pulse_ai_dynamic_index_continent():
    path = WEB_DATA / "pulse_ai_dynamic_index_continent_year.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_dynamics_summary")
def api_pulse_dynamics_summary():
    path = WEB_DATA / "pulse_dynamics_summary.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/pulse_dynamics_transition")
def api_pulse_dynamics_transition():
    path = WEB_DATA / "pulse_dynamics_transition_tensor.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    scope = request.args.get("scope", "").strip().lower()
    if scope in {"global", "continent"}:
        df = df[df["scope"].astype(str).str.lower() == scope]
    continent = request.args.get("continent", "").strip().lower()
    if continent and continent != "all":
        df = df[df["continent"].astype(str).str.lower() == continent]
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_dynamics_spell_hazard")
def api_pulse_dynamics_spell_hazard():
    path = WEB_DATA / "pulse_dynamics_spell_hazard.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    spell_kind = request.args.get("spell_kind", "").strip().lower()
    if spell_kind:
        df = df[df["spell_kind"].astype(str).str.lower() == spell_kind]
    scope = request.args.get("scope", "").strip().lower()
    if scope in {"global", "continent"}:
        df = df[df["scope"].astype(str).str.lower() == scope]
    continent = request.args.get("continent", "").strip().lower()
    if continent and continent != "all":
        df = df[df["continent"].astype(str).str.lower() == continent]
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_dynamics_resilience")
def api_pulse_dynamics_resilience():
    path = WEB_DATA / "pulse_dynamics_resilience_halflife.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    scope = request.args.get("scope", "").strip().lower()
    if scope in {"global", "continent"}:
        df = df[df["scope"].astype(str).str.lower() == scope]
    continent = request.args.get("continent", "").strip().lower()
    if continent and continent != "all":
        df = df[df["continent"].astype(str).str.lower() == continent]
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_dynamics_warning")
def api_pulse_dynamics_warning():
    path = WEB_DATA / "pulse_dynamics_warning_horizon.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    scope = request.args.get("scope", "").strip().lower()
    if scope in {"global", "continent"}:
        df = df[df["scope"].astype(str).str.lower() == scope]
    continent = request.args.get("continent", "").strip().lower()
    if continent and continent != "all":
        df = df[df["continent"].astype(str).str.lower() == continent]
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_nowcast_summary")
def api_pulse_nowcast_summary():
    path = WEB_DATA / "pulse_nowcast_summary.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/pulse_nowcast_latest")
def api_pulse_nowcast_latest():
    path = WEB_DATA / "pulse_nowcast_continent_latest.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    continent = request.args.get("continent", "").strip().lower()
    if continent and continent != "all":
        df = df[df["continent"].astype(str).str.lower() == continent]
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_nowcast_history")
def api_pulse_nowcast_history():
    path = WEB_DATA / "pulse_nowcast_continent_history.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    scope = request.args.get("scope", "").strip().lower()
    if scope in {"global", "continent"}:
        df = df[df["scope"].astype(str).str.lower() == scope]
    continent = request.args.get("continent", "").strip().lower()
    if continent and continent != "all":
        df = df[df["continent"].astype(str).str.lower() == continent]
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/pulse_nowcast_global")
def api_pulse_nowcast_global():
    path = WEB_DATA / "pulse_nowcast_global.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/dynamic_method_core_summary")
def api_dynamic_method_core_summary():
    path = WEB_DATA / "dynamic_method_core_summary.json"
    if not path.exists():
        return jsonify({})
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/api/dynamic_method_core_metrics")
def api_dynamic_method_core_metrics():
    path = WEB_DATA / "dynamic_method_core_metrics.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    scope = request.args.get("scope", "").strip().lower()
    if scope in {"global", "continent"}:
        df = df[df["scope"].astype(str).str.lower() == scope]
    continent = request.args.get("continent", "").strip().lower()
    if continent and continent != "all":
        df = df[df["continent"].astype(str).str.lower() == continent]
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/dynamic_method_core_significance")
def api_dynamic_method_core_significance():
    path = WEB_DATA / "dynamic_method_core_significance.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    scope = request.args.get("scope", "").strip().lower()
    if scope in {"global", "continent"}:
        df = df[df["scope"].astype(str).str.lower() == scope]
    continent = request.args.get("continent", "").strip().lower()
    if continent and continent != "all":
        df = df[df["continent"].astype(str).str.lower() == continent]
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/dynamic_method_core_ablation")
def api_dynamic_method_core_ablation():
    path = WEB_DATA / "dynamic_method_core_ablation.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/causal_st_policy")
def api_causal_st_policy():
    path = ROOT / "data" / "outputs" / "causal_st_policy_simulation.csv"
    if not path.exists():
        return jsonify([])
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/realtime/status")
def api_realtime_status():
    return jsonify(REALTIME_RUNTIME.get_status())


@app.get("/api/realtime/countries")
def api_realtime_countries():
    df = _read_web_csv("realtime_country_monitor.csv")
    if df.empty:
        return jsonify([])
    continent = request.args.get("continent", "").strip().lower()
    if continent and continent != "all":
        df = df[df["continent"].astype(str).str.lower() == continent]
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/realtime/alerts")
def api_realtime_alerts():
    df = _read_web_csv("realtime_alerts.csv")
    if df.empty:
        return jsonify([])
    level = request.args.get("level", "").strip().lower()
    if level:
        df = df[df["alert_level"].astype(str).str.lower() == level]
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/realtime/continents")
def api_realtime_continents():
    df = _read_web_csv("realtime_continent_monitor.csv")
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/realtime/sentinel")
def api_realtime_sentinel():
    df = _read_web_csv("realtime_sentinel.csv")
    if df.empty:
        return jsonify([])
    continent = request.args.get("continent", "").strip().lower()
    if continent and continent != "all":
        df = df[df["continent"].astype(str).str.lower() == continent]
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/realtime/trigger", methods=["GET", "POST"])
def api_realtime_trigger():
    reason = request.args.get("reason", "").strip() or "manual_api"
    payload = REALTIME_RUNTIME.trigger_update(reason=reason)
    return jsonify(payload)


@app.get("/api/realtime/stream")
def api_realtime_stream():
    def event_stream():
        last_version = -1
        while True:
            payload = REALTIME_RUNTIME.get_status()
            version = int(payload.get("runtime_version", 0))
            if version != last_version:
                data = json.dumps(payload, ensure_ascii=False)
                yield f"event: monitor\nid: {version}\ndata: {data}\n\n"
                last_version = version
            else:
                yield ": heartbeat\n\n"
            time.sleep(2.0)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream", headers=headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
