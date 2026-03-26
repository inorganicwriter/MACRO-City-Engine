from __future__ import annotations

"""Realtime monitoring snapshot generation for MACRO-City Engine tracking."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .utils import DATA_OUTPUTS, DATA_PROCESSED, REPORTS_DIR, WEB_DATA_DIR, dump_json

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RealtimeMonitorConfig:
    """Configuration for realtime nowcast and alert generation."""

    nowcast_fraction_year: float = 0.25
    min_history_years: int = 3
    warn_risk_threshold: float = 70.0
    critical_risk_threshold: float = 85.0
    low_accel_threshold: float = 45.0
    high_accel_threshold: float = 70.0
    top_countries: int = 60
    top_alerts: int = 120
    top_sentinel: int = 120
    shock_z_threshold: float = 2.0
    changepoint_alert_threshold: float = 0.72


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read CSV %s: %s", path, exc)
        return pd.DataFrame()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read JSON %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.fillna(default).astype(float)


def _clip(values: pd.Series | np.ndarray, lo: float = 0.0, hi: float = 100.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, lo, hi)


def _safe_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _calc_city_trend(group: pd.DataFrame, min_history_years: int) -> pd.Series:
    """Estimate per-city slope and volatility from recent history."""
    g = group.sort_values("year")
    tail = g.tail(max(2, int(min_history_years)))

    years = _to_numeric(tail["year"]).to_numpy(dtype=float)
    composite = _to_numeric(tail["composite_index"]).to_numpy(dtype=float)
    if len(years) >= 2 and np.isfinite(years).all() and np.isfinite(composite).all():
        slope = float(np.polyfit(years, composite, deg=1)[0])
    else:
        slope = 0.0

    diffs = np.diff(composite) if len(composite) >= 2 else np.array([0.0], dtype=float)
    vol = float(np.nanstd(diffs)) if diffs.size else 0.0
    prev = float(composite[-2]) if len(composite) >= 2 else float(composite[-1])
    latest = float(composite[-1]) if len(composite) >= 1 else 0.0

    comp_all = _to_numeric(g["composite_index"]).to_numpy(dtype=float)
    diffs_all = np.diff(comp_all) if len(comp_all) >= 2 else np.array([0.0], dtype=float)
    baseline = diffs_all[:-1] if len(diffs_all) >= 2 else diffs_all
    diff_mean = float(np.nanmean(baseline)) if baseline.size else 0.0
    diff_std = float(np.nanstd(baseline)) if baseline.size else 0.0
    if not np.isfinite(diff_std) or diff_std < 1e-6:
        diff_std = max(0.5, float(np.nanstd(diffs_all)) if diffs_all.size else 0.5)
    last_diff = float(diffs_all[-1]) if diffs_all.size else 0.0
    shock_z = float((last_diff - diff_mean) / max(diff_std, 1e-6))

    k = 0.2
    scale = max(diff_std, 0.6)
    cusum_pos = 0.0
    cusum_neg = 0.0
    for d in diffs_all:
        centered = (float(d) - diff_mean) / scale
        cusum_pos = max(0.0, cusum_pos + centered - k)
        cusum_neg = max(0.0, cusum_neg - centered - k)
    cp_raw = max(cusum_pos, cusum_neg)
    cp_prob = float(1.0 - np.exp(-cp_raw / 3.0))

    return pd.Series(
        {
            "composite_trend_per_year": slope,
            "composite_volatility": vol,
            "prev_composite_index": prev,
            "latest_composite_index": latest,
            "composite_diff_mean": diff_mean,
            "composite_diff_std": diff_std,
            "composite_last_diff": last_diff,
            "composite_shock_z": shock_z,
            "composite_cusum_pos": float(cusum_pos),
            "composite_cusum_neg": float(cusum_neg),
            "composite_changepoint_prob": cp_prob,
        }
    )


def _alert_level(
    nowcast_risk: float,
    nowcast_accel: float,
    source_quality: float,
    cfg: RealtimeMonitorConfig,
) -> str:
    if nowcast_risk >= cfg.critical_risk_threshold and nowcast_accel <= cfg.low_accel_threshold:
        return "critical_stall"
    if nowcast_risk >= cfg.warn_risk_threshold:
        return "stall_warning"
    if nowcast_accel >= cfg.high_accel_threshold and nowcast_risk <= 40.0:
        return "acceleration_opportunity"
    if source_quality < 75.0:
        return "data_watch"
    return "stable_watch"


def _alert_priority(level: str) -> int:
    mapping = {
        "critical_stall": 4,
        "stall_warning": 3,
        "data_watch": 2,
        "acceleration_opportunity": 1,
        "stable_watch": 0,
    }
    return mapping.get(str(level), 0)


def _sentinel_type(
    *,
    shock_z: float,
    changepoint_prob: float,
    nowcast_risk: float,
    nowcast_accel: float,
    signal_drift: float,
    cfg: RealtimeMonitorConfig,
) -> str:
    if shock_z <= -cfg.shock_z_threshold and nowcast_risk >= cfg.warn_risk_threshold:
        return "downshift_break"
    if (
        shock_z >= cfg.shock_z_threshold
        and nowcast_accel >= cfg.high_accel_threshold
        and nowcast_risk <= cfg.warn_risk_threshold
    ):
        return "upshift_break"
    if changepoint_prob >= cfg.changepoint_alert_threshold and nowcast_risk >= cfg.warn_risk_threshold:
        return "fragile_transition"
    if abs(shock_z) >= cfg.shock_z_threshold:
        return "volatility_jump"
    if signal_drift <= -2.0 and nowcast_risk >= 60.0:
        return "slowdown_watch"
    if changepoint_prob >= max(0.5, cfg.changepoint_alert_threshold - 0.15):
        return "transition_watch"
    return "steady_transition"


def _sentinel_priority(level: str) -> int:
    mapping = {
        "downshift_break": 6,
        "fragile_transition": 5,
        "volatility_jump": 4,
        "slowdown_watch": 3,
        "transition_watch": 2,
        "upshift_break": 1,
        "steady_transition": 0,
    }
    return mapping.get(str(level), 0)


def _build_city_monitor(
    city_points: pd.DataFrame,
    pulse_latest: pd.DataFrame,
    source_city: pd.DataFrame,
    cfg: RealtimeMonitorConfig,
) -> Tuple[pd.DataFrame, int]:
    if city_points.empty:
        return pd.DataFrame(), 0

    points = city_points.copy()
    points["year"] = _to_numeric(points["year"]).round().astype(int)
    points["composite_index"] = _to_numeric(points["composite_index"])
    points["economic_vitality"] = _to_numeric(points.get("economic_vitality", pd.Series(dtype=float)))
    points["livability"] = _to_numeric(points.get("livability", pd.Series(dtype=float)))
    points["innovation"] = _to_numeric(points.get("innovation", pd.Series(dtype=float)))

    latest_year = int(points["year"].max())
    latest_points = points.loc[points["year"] == latest_year].copy()
    if latest_points.empty:
        return pd.DataFrame(), latest_year

    trend = (
        points.groupby("city_id", as_index=False)
        .apply(
            lambda g: _calc_city_trend(g, min_history_years=cfg.min_history_years),
            include_groups=False,
        )
        .reset_index()
    )
    if "level_1" in trend.columns:
        trend = trend.drop(columns=["level_1"])

    out = latest_points.merge(trend, on="city_id", how="left")

    pulse_cols = [
        "city_id",
        "acceleration_score",
        "stall_risk_score",
        "stall_risk_low",
        "stall_risk_high",
        "stall_risk_interval_width",
        "accel_shift_1y",
        "risk_shift_1y",
        "forecast_risk_delta_h1",
        "phase_label",
        "trajectory_regime",
        "archetype",
        "pulse_quadrant",
    ]
    if not pulse_latest.empty:
        available = [c for c in pulse_cols if c in pulse_latest.columns]
        out = out.merge(pulse_latest[available].copy(), on="city_id", how="left")

    if not source_city.empty:
        src = source_city.copy()
        for col in ["verified_ratio", "objective_ratio", "is_verified_complete_city"]:
            if col in src.columns:
                src[col] = _to_numeric(src[col], default=np.nan)
        keep = [c for c in ["city_id", "verified_ratio", "objective_ratio", "is_verified_complete_city"] if c in src.columns]
        out = out.merge(src[keep], on="city_id", how="left")

    def _series_or_default(col: str, default: float) -> pd.Series:
        if col in out.columns:
            return out[col]
        return pd.Series(default, index=out.index, dtype=float)

    out["composite_trend_per_year"] = _to_numeric(_series_or_default("composite_trend_per_year", 0.0))
    out["composite_volatility"] = _to_numeric(_series_or_default("composite_volatility", 0.0))
    out["prev_composite_index"] = _to_numeric(_series_or_default("prev_composite_index", float("nan")))
    out["prev_composite_index"] = out["prev_composite_index"].fillna(out["composite_index"])

    out["acceleration_score"] = _to_numeric(_series_or_default("acceleration_score", 50.0), default=50.0)
    out["stall_risk_score"] = _to_numeric(_series_or_default("stall_risk_score", 50.0), default=50.0)
    out["stall_risk_interval_width"] = _to_numeric(_series_or_default("stall_risk_interval_width", 0.0), default=0.0)
    out["accel_shift_1y"] = _to_numeric(_series_or_default("accel_shift_1y", 0.0), default=0.0)
    out["risk_shift_1y"] = _to_numeric(_series_or_default("risk_shift_1y", 0.0), default=0.0)
    out["forecast_risk_delta_h1"] = _to_numeric(_series_or_default("forecast_risk_delta_h1", float("nan")), default=np.nan)
    out["forecast_risk_delta_h1"] = out["forecast_risk_delta_h1"].fillna(out["risk_shift_1y"] * 0.8)
    out["verified_ratio"] = _to_numeric(_series_or_default("verified_ratio", float("nan")), default=np.nan)

    nowcast_composite = out["composite_index"] + out["composite_trend_per_year"] * float(cfg.nowcast_fraction_year)
    nowcast_accel = out["acceleration_score"] + 0.35 * out["accel_shift_1y"]
    nowcast_risk = out["stall_risk_score"] + out["forecast_risk_delta_h1"] * float(cfg.nowcast_fraction_year)

    out["nowcast_composite_index"] = _clip(nowcast_composite)
    out["nowcast_acceleration_score"] = _clip(nowcast_accel)
    out["nowcast_stall_risk_score"] = _clip(nowcast_risk)
    out["nowcast_delta_composite"] = out["nowcast_composite_index"] - out["composite_index"]
    out["nowcast_stall_pressure"] = out["nowcast_stall_risk_score"] - out["nowcast_acceleration_score"]
    out["nowcast_pulse_balance"] = out["nowcast_acceleration_score"] - out["nowcast_stall_risk_score"]
    out["signal_drift_1y"] = out["composite_index"] - out["prev_composite_index"]

    out["composite_shock_z"] = _to_numeric(_series_or_default("composite_shock_z", 0.0), default=0.0)
    out["composite_changepoint_prob"] = _to_numeric(_series_or_default("composite_changepoint_prob", 0.0), default=0.0)
    out["composite_changepoint_prob"] = np.clip(out["composite_changepoint_prob"], 0.0, 1.0)

    uncertainty = 3.2 * out["stall_risk_interval_width"] + 1.6 * out["composite_volatility"].abs()
    out["uncertainty_score"] = _clip(uncertainty)

    src_quality = np.where(np.isfinite(out["verified_ratio"]), out["verified_ratio"] * 100.0, 65.0)
    out["source_quality_score"] = _clip(src_quality)

    out["alert_level"] = [
        _alert_level(float(r), float(a), float(s), cfg)
        for r, a, s in zip(
            out["nowcast_stall_risk_score"].tolist(),
            out["nowcast_acceleration_score"].tolist(),
            out["source_quality_score"].tolist(),
        )
    ]
    out["alert_priority"] = out["alert_level"].map(_alert_priority).astype(int)
    out["alert_score"] = (
        out["nowcast_stall_risk_score"]
        - 0.55 * out["nowcast_acceleration_score"]
        + 0.25 * out["uncertainty_score"]
        + 0.18 * (100.0 - out["source_quality_score"])
    )
    out["alert_score"] = _clip(out["alert_score"], lo=0.0, hi=200.0)

    shock_mag = _clip(np.abs(out["composite_shock_z"]) * 25.0, lo=0.0, hi=100.0)
    drift_penalty = _clip((-out["signal_drift_1y"]) * 8.0, lo=0.0, hi=100.0)
    cp_score = _clip(out["composite_changepoint_prob"] * 100.0, lo=0.0, hi=100.0)
    source_penalty = _clip(100.0 - out["source_quality_score"], lo=0.0, hi=100.0)

    out["sentinel_score"] = (
        0.36 * out["nowcast_stall_risk_score"]
        + 0.24 * drift_penalty
        + 0.18 * cp_score
        + 0.12 * shock_mag
        + 0.06 * out["uncertainty_score"]
        + 0.04 * source_penalty
    )
    out["sentinel_score"] = _clip(out["sentinel_score"], lo=0.0, hi=200.0)
    out["sentinel_type"] = [
        _sentinel_type(
            shock_z=float(z),
            changepoint_prob=float(cp),
            nowcast_risk=float(r),
            nowcast_accel=float(a),
            signal_drift=float(d),
            cfg=cfg,
        )
        for z, cp, r, a, d in zip(
            out["composite_shock_z"].tolist(),
            out["composite_changepoint_prob"].tolist(),
            out["nowcast_stall_risk_score"].tolist(),
            out["nowcast_acceleration_score"].tolist(),
            out["signal_drift_1y"].tolist(),
        )
    ]
    out["sentinel_priority"] = out["sentinel_type"].map(_sentinel_priority).astype(int)
    sentinel_rank = (
        out[["city_id", "sentinel_priority", "sentinel_score", "city_name"]]
        .sort_values(["sentinel_priority", "sentinel_score", "city_name"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    sentinel_rank["sentinel_rank"] = np.arange(1, len(sentinel_rank) + 1, dtype=int)
    out = out.merge(sentinel_rank[["city_id", "sentinel_rank"]], on="city_id", how="left")

    out = out.sort_values(["alert_priority", "alert_score", "city_name"], ascending=[False, False, True]).reset_index(drop=True)
    out["alert_rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out, latest_year


def _aggregate_country(city_monitor: pd.DataFrame, cfg: RealtimeMonitorConfig) -> pd.DataFrame:
    if city_monitor.empty:
        return pd.DataFrame()

    def _count_if(level: str) -> Any:
        return lambda x: int((x == level).sum())

    grouped = (
        city_monitor.groupby(["country", "continent"], as_index=False)
        .agg(
            city_count=("city_id", "nunique"),
            mean_composite=("nowcast_composite_index", "mean"),
            mean_accel=("nowcast_acceleration_score", "mean"),
            mean_risk=("nowcast_stall_risk_score", "mean"),
            mean_uncertainty=("uncertainty_score", "mean"),
            mean_source_quality=("source_quality_score", "mean"),
            mean_sentinel_score=("sentinel_score", "mean"),
            mean_changepoint_prob=("composite_changepoint_prob", "mean"),
            critical_city_count=("alert_level", _count_if("critical_stall")),
            warning_city_count=("alert_level", _count_if("stall_warning")),
            data_watch_city_count=("alert_level", _count_if("data_watch")),
            opportunity_city_count=("alert_level", _count_if("acceleration_opportunity")),
            sentinel_break_city_count=("sentinel_type", lambda s: int(s.isin(["downshift_break", "fragile_transition"]).sum())),
        )
        .sort_values(["mean_risk", "country"], ascending=[False, True])
    )
    grouped["critical_share"] = grouped["critical_city_count"] / grouped["city_count"].replace(0, 1)
    grouped["warning_share"] = grouped["warning_city_count"] / grouped["city_count"].replace(0, 1)
    grouped["sentinel_break_share"] = grouped["sentinel_break_city_count"] / grouped["city_count"].replace(0, 1)
    grouped["nowcast_stall_pressure"] = grouped["mean_risk"] - grouped["mean_accel"]
    grouped["nowcast_pulse_balance"] = grouped["mean_accel"] - grouped["mean_risk"]

    def _country_alert(row: pd.Series) -> str:
        if float(row["critical_share"]) >= 0.25 or float(row["mean_risk"]) >= 82.0 or float(row["sentinel_break_share"]) >= 0.35:
            return "critical"
        if (
            float(row["warning_share"]) >= 0.30
            or float(row["mean_risk"]) >= cfg.warn_risk_threshold
            or float(row["mean_changepoint_prob"]) >= cfg.changepoint_alert_threshold
        ):
            return "warning"
        if float(row["mean_accel"]) >= cfg.high_accel_threshold and float(row["mean_risk"]) <= 42.0:
            return "opportunity"
        return "watch"

    grouped["country_alert_level"] = grouped.apply(_country_alert, axis=1)

    top_city = city_monitor.sort_values(["country", "alert_priority", "alert_score"], ascending=[True, False, False]).drop_duplicates(
        subset=["country"], keep="first"
    )
    top_city = top_city[
        ["country", "city_name", "nowcast_stall_risk_score", "nowcast_acceleration_score", "alert_level"]
    ].rename(
        columns={
            "city_name": "top_alert_city",
            "nowcast_stall_risk_score": "top_city_risk",
            "nowcast_acceleration_score": "top_city_accel",
            "alert_level": "top_city_alert_level",
        }
    )
    grouped = grouped.merge(top_city, on="country", how="left")
    grouped = grouped.sort_values(
        ["country_alert_level", "nowcast_stall_pressure", "sentinel_break_share", "critical_share", "country"],
        ascending=[True, False, False, False, True],
        key=lambda s: s.map({"critical": 0, "warning": 1, "watch": 2, "opportunity": 3}) if s.name == "country_alert_level" else s,
    )
    grouped["country_rank"] = np.arange(1, len(grouped) + 1, dtype=int)
    return grouped.reset_index(drop=True)


def _aggregate_continent(country_monitor: pd.DataFrame) -> pd.DataFrame:
    if country_monitor.empty:
        return pd.DataFrame()

    grouped = (
        country_monitor.groupby("continent", as_index=False)
        .agg(
            country_count=("country", "nunique"),
            city_count=("city_count", "sum"),
            mean_composite=("mean_composite", "mean"),
            mean_accel=("mean_accel", "mean"),
            mean_risk=("mean_risk", "mean"),
            mean_source_quality=("mean_source_quality", "mean"),
            mean_sentinel_score=("mean_sentinel_score", "mean"),
            mean_changepoint_prob=("mean_changepoint_prob", "mean"),
            critical_country_share=("country_alert_level", lambda s: float((s == "critical").mean())),
            warning_country_share=("country_alert_level", lambda s: float((s == "warning").mean())),
        )
        .sort_values("mean_risk", ascending=False)
    )
    grouped["nowcast_stall_pressure"] = grouped["mean_risk"] - grouped["mean_accel"]
    grouped["nowcast_pulse_balance"] = grouped["mean_accel"] - grouped["mean_risk"]
    return grouped.reset_index(drop=True)


def _build_alert_table(city_monitor: pd.DataFrame, cfg: RealtimeMonitorConfig) -> pd.DataFrame:
    if city_monitor.empty:
        return pd.DataFrame()
    alerts = city_monitor[city_monitor["alert_level"] != "stable_watch"].copy()
    alerts = alerts.sort_values(["alert_priority", "alert_score"], ascending=[False, False]).head(int(cfg.top_alerts))
    keep_cols = [
        "alert_rank",
        "city_id",
        "city_name",
        "country",
        "continent",
        "alert_level",
        "alert_score",
        "nowcast_stall_risk_score",
        "nowcast_acceleration_score",
        "nowcast_composite_index",
        "nowcast_delta_composite",
        "uncertainty_score",
        "source_quality_score",
        "sentinel_score",
        "sentinel_type",
        "composite_shock_z",
        "composite_changepoint_prob",
        "trajectory_regime",
        "phase_label",
        "archetype",
        "pulse_quadrant",
    ]
    keep_cols = [c for c in keep_cols if c in alerts.columns]
    return alerts[keep_cols].reset_index(drop=True)


def _build_sentinel_table(city_monitor: pd.DataFrame, cfg: RealtimeMonitorConfig) -> pd.DataFrame:
    if city_monitor.empty:
        return pd.DataFrame()
    sent = city_monitor[
        (city_monitor["sentinel_type"] != "steady_transition")
        | (np.abs(city_monitor["composite_shock_z"]) >= max(1.2, cfg.shock_z_threshold * 0.6))
        | (city_monitor["composite_changepoint_prob"] >= max(0.4, cfg.changepoint_alert_threshold - 0.2))
    ].copy()
    sent = sent.sort_values(["sentinel_priority", "sentinel_score"], ascending=[False, False]).head(int(cfg.top_sentinel))
    keep_cols = [
        "sentinel_rank",
        "city_id",
        "city_name",
        "country",
        "continent",
        "sentinel_type",
        "sentinel_score",
        "composite_shock_z",
        "composite_changepoint_prob",
        "signal_drift_1y",
        "nowcast_stall_risk_score",
        "nowcast_acceleration_score",
        "nowcast_stall_pressure",
        "alert_level",
        "trajectory_regime",
        "phase_label",
    ]
    keep_cols = [c for c in keep_cols if c in sent.columns]
    return sent[keep_cols].reset_index(drop=True)


def _snapshot_signature(paths: Iterable[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        ts = _safe_mtime(p)
        out[p.name] = ts or "missing"
    return out


def _write_versioned_snapshot(
    version_tag: str,
    city_monitor: pd.DataFrame,
    country_monitor: pd.DataFrame,
    continent_monitor: pd.DataFrame,
    alerts: pd.DataFrame,
    sentinel: pd.DataFrame,
    status: Dict[str, Any],
    *,
    data_outputs_dir: Path,
) -> None:
    snap_dir = data_outputs_dir / "realtime_snapshots" / version_tag
    snap_dir.mkdir(parents=True, exist_ok=True)
    city_monitor.to_csv(snap_dir / "realtime_city_monitor.csv", index=False)
    country_monitor.to_csv(snap_dir / "realtime_country_monitor.csv", index=False)
    continent_monitor.to_csv(snap_dir / "realtime_continent_monitor.csv", index=False)
    alerts.to_csv(snap_dir / "realtime_alerts.csv", index=False)
    sentinel.to_csv(snap_dir / "realtime_sentinel.csv", index=False)
    dump_json(snap_dir / "realtime_status.json", status)


def generate_realtime_monitor_snapshot(
    *,
    config: RealtimeMonitorConfig | None = None,
    trigger: str = "manual",
    web_data_dir: Path | None = None,
    data_processed_dir: Path | None = None,
    data_outputs_dir: Path | None = None,
    reports_dir: Path | None = None,
    write_versioned_snapshot: bool = True,
) -> Dict[str, Any]:
    """Generate realtime monitoring artifacts from current dashboard outputs."""
    cfg = config or RealtimeMonitorConfig()
    web_dir = WEB_DATA_DIR if web_data_dir is None else Path(web_data_dir)
    processed_dir = DATA_PROCESSED if data_processed_dir is None else Path(data_processed_dir)
    outputs_dir = DATA_OUTPUTS if data_outputs_dir is None else Path(data_outputs_dir)
    rep_dir = REPORTS_DIR if reports_dir is None else Path(reports_dir)

    t0 = datetime.now(timezone.utc)
    city_points_path = web_dir / "city_points.csv"
    pulse_latest_path = web_dir / "pulse_ai_city_latest.csv"
    source_city_path = processed_dir / "source_audit_city.csv"
    source_summary_path = processed_dir / "source_audit_summary.json"
    pipeline_summary_path = rep_dir / "pipeline_summary.json"

    city_points = _read_csv(city_points_path)
    pulse_latest = _read_csv(pulse_latest_path)
    source_city = _read_csv(source_city_path)
    source_summary = _read_json(source_summary_path)
    pipeline_summary = _read_json(pipeline_summary_path)

    city_monitor, latest_year = _build_city_monitor(city_points, pulse_latest, source_city, cfg)
    country_monitor = _aggregate_country(city_monitor, cfg)
    continent_monitor = _aggregate_continent(country_monitor)
    alerts = _build_alert_table(city_monitor, cfg)
    sentinel = _build_sentinel_table(city_monitor, cfg)

    generated_at = datetime.now(timezone.utc)
    version_tag = generated_at.strftime("%Y%m%dT%H%M%SZ")
    sig = _snapshot_signature([city_points_path, pulse_latest_path, source_city_path, source_summary_path, pipeline_summary_path])
    signature_token = "|".join([f"{k}:{sig[k]}" for k in sorted(sig.keys())])
    signature_hash = str(abs(hash(signature_token)))

    reliability_gate = pipeline_summary.get("reliability_gate", {}) if pipeline_summary else {}
    status: Dict[str, Any] = {
        "status": "ok" if not city_monitor.empty else "empty",
        "trigger": trigger,
        "generated_at_utc": generated_at.isoformat(),
        "version_tag": version_tag,
        "latest_data_year": int(latest_year) if latest_year else None,
        "city_count": int(city_monitor["city_id"].nunique()) if not city_monitor.empty else 0,
        "country_count": int(country_monitor["country"].nunique()) if not country_monitor.empty else 0,
        "continent_count": int(continent_monitor["continent"].nunique()) if not continent_monitor.empty else 0,
        "alert_city_count": int((city_monitor["alert_level"] != "stable_watch").sum()) if not city_monitor.empty else 0,
        "critical_city_count": int((city_monitor["alert_level"] == "critical_stall").sum()) if not city_monitor.empty else 0,
        "warning_city_count": int((city_monitor["alert_level"] == "stall_warning").sum()) if not city_monitor.empty else 0,
        "opportunity_city_count": int((city_monitor["alert_level"] == "acceleration_opportunity").sum()) if not city_monitor.empty else 0,
        "sentinel_city_count": int(len(sentinel)),
        "sentinel_break_count": int((city_monitor["sentinel_type"] == "downshift_break").sum()) if not city_monitor.empty else 0,
        "sentinel_fragile_transition_count": int((city_monitor["sentinel_type"] == "fragile_transition").sum()) if not city_monitor.empty else 0,
        "source_signature": sig,
        "source_signature_hash": signature_hash,
        "source_reliability": {
            "enforce_verified": bool(source_summary.get("enforce_verified")) if source_summary else None,
            "verified_row_ratio": source_summary.get("verified_row_ratio"),
            "verified_complete_city_ratio": source_summary.get("verified_complete_city_ratio"),
            "city_retention_ratio": source_summary.get("city_retention_ratio"),
            "reliability_gate_status": reliability_gate.get("status"),
        },
        "runtime_seconds": (generated_at - t0).total_seconds(),
        "nowcast_config": {
            "fraction_year": float(cfg.nowcast_fraction_year),
            "min_history_years": int(cfg.min_history_years),
            "warn_risk_threshold": float(cfg.warn_risk_threshold),
            "critical_risk_threshold": float(cfg.critical_risk_threshold),
            "low_accel_threshold": float(cfg.low_accel_threshold),
            "high_accel_threshold": float(cfg.high_accel_threshold),
            "shock_z_threshold": float(cfg.shock_z_threshold),
            "changepoint_alert_threshold": float(cfg.changepoint_alert_threshold),
        },
    }

    web_dir.mkdir(parents=True, exist_ok=True)
    city_monitor.to_csv(web_dir / "realtime_city_monitor.csv", index=False)
    country_monitor.to_csv(web_dir / "realtime_country_monitor.csv", index=False)
    continent_monitor.to_csv(web_dir / "realtime_continent_monitor.csv", index=False)
    alerts.to_csv(web_dir / "realtime_alerts.csv", index=False)
    sentinel.to_csv(web_dir / "realtime_sentinel.csv", index=False)
    dump_json(web_dir / "realtime_status.json", status)

    if write_versioned_snapshot:
        _write_versioned_snapshot(
            version_tag,
            city_monitor,
            country_monitor,
            continent_monitor,
            alerts,
            sentinel,
            status,
            data_outputs_dir=outputs_dir,
        )

    hist_path = outputs_dir / "realtime_monitor_history.jsonl"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with hist_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(status, ensure_ascii=False) + "\n")

    return status
