from __future__ import annotations

"""Dynamic causal envelope diagnostics for city-level pulse evolution."""

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .utils import DATA_OUTPUTS, dump_json


def _safe_read_csv(name: str) -> pd.DataFrame:
    path = DATA_OUTPUTS / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _safe_read_json(name: str) -> Dict[str, Any]:
    path = DATA_OUTPUTS / name
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return default


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def _minmax_series(s: pd.Series, invert: bool = False, default: float = 0.5) -> pd.Series:
    arr = pd.to_numeric(s, errors="coerce").astype(float)
    if arr.empty:
        return pd.Series(dtype=float)
    valid = arr.dropna()
    if valid.empty:
        out = pd.Series(np.full(len(arr), default, dtype=float), index=arr.index)
        return 1.0 - out if invert else out
    lo = float(valid.min())
    hi = float(valid.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        out = pd.Series(np.full(len(arr), default, dtype=float), index=arr.index)
        return 1.0 - out if invert else out
    out = (arr - lo) / (hi - lo)
    out = out.fillna(default).clip(lower=0.0, upper=1.0)
    return 1.0 - out if invert else out


def _zscore_series(s: pd.Series) -> pd.Series:
    arr = pd.to_numeric(s, errors="coerce").astype(float)
    valid = arr.dropna()
    if valid.empty:
        return pd.Series(np.zeros(len(arr), dtype=float), index=arr.index)
    std = float(valid.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(np.zeros(len(arr), dtype=float), index=arr.index)
    mu = float(valid.mean())
    return ((arr - mu) / std).fillna(0.0)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    if weights.size != values.size:
        weights = np.ones_like(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all() or float(np.sum(w)) <= 1e-12:
        w = np.ones_like(values, dtype=float)
    return float(np.average(values, weights=w))


def _append_main_event_points(pts: pd.DataFrame) -> pd.DataFrame:
    econ = _safe_read_json("econometric_summary.json")
    main = (econ.get("event_study_fe", {}) or {}).get("points", [])
    if not isinstance(main, list) or not main:
        return pts

    rows: List[Dict[str, Any]] = []
    for p in main:
        if not isinstance(p, dict):
            continue
        rows.append(
            {
                "variant": "main_design",
                "design_variant": econ.get("main_design_variant"),
                "source_channel": "main",
                "source_sensitivity_mode": "main",
                "rel_year": p.get("rel_year"),
                "coef": p.get("coef"),
                "stderr": p.get("stderr"),
                "t_value": p.get("t_value"),
            }
        )
    if not rows:
        return pts
    extra = pd.DataFrame(rows)
    if pts.empty:
        return extra
    return pd.concat([pts, extra], ignore_index=True)


def _event_points_with_weights() -> pd.DataFrame:
    pts = _safe_read_csv("econometric_source_event_study_points.csv")
    pts = _append_main_event_points(pts)
    if pts.empty:
        return pd.DataFrame()

    pts["rel_year"] = pd.to_numeric(pts.get("rel_year"), errors="coerce")
    pts["coef"] = pd.to_numeric(pts.get("coef"), errors="coerce")
    pts["variant"] = pts.get("variant", pd.Series(["unknown"] * len(pts))).astype(str)
    pts = pts.dropna(subset=["rel_year", "coef"]).copy()
    if pts.empty:
        return pts

    stress = _safe_read_csv("idplus_identification_stress_index.csv")
    weight_map: Dict[str, float] = {}
    default_weight = 0.55
    if not stress.empty and {"variant", "resilience_score_0_100"}.issubset(stress.columns):
        tmp = stress[["variant", "resilience_score_0_100"]].copy()
        tmp["variant"] = tmp["variant"].astype(str)
        tmp["resilience_score_0_100"] = pd.to_numeric(tmp["resilience_score_0_100"], errors="coerce")
        tmp = tmp.dropna(subset=["resilience_score_0_100"])
        if not tmp.empty:
            weight_map = {
                str(r.variant): _clip01(float(r.resilience_score_0_100) / 100.0)
                for r in tmp.itertuples(index=False)
            }
            if weight_map:
                default_weight = float(np.mean(list(weight_map.values())))

    pts["variant_weight"] = pts["variant"].map(weight_map).fillna(default_weight).clip(lower=0.05, upper=1.0)
    return pts


def _build_event_envelope() -> pd.DataFrame:
    pts = _event_points_with_weights()
    if pts.empty:
        return pd.DataFrame(
            columns=[
                "rel_year",
                "n_variants",
                "coef_min",
                "coef_q25",
                "coef_median",
                "coef_q75",
                "coef_max",
                "coef_weighted_mean",
                "coef_std",
                "coef_band_iqr",
                "coef_band_full",
                "share_positive",
                "share_negative",
                "share_conflict",
                "post_period",
            ]
        )

    rows: List[Dict[str, Any]] = []
    for rel_year, grp in pts.groupby("rel_year", sort=True):
        coef = pd.to_numeric(grp["coef"], errors="coerce").dropna().to_numpy(dtype=float)
        w = pd.to_numeric(grp["variant_weight"], errors="coerce").fillna(0.55).to_numpy(dtype=float)
        if coef.size == 0:
            continue
        if len(w) != len(coef):
            w = np.ones_like(coef, dtype=float)
        if np.sum(w) <= 1e-12:
            w = np.ones_like(coef, dtype=float)

        share_pos = float(np.mean(coef > 0))
        share_neg = float(np.mean(coef < 0))
        rows.append(
            {
                "rel_year": int(rel_year),
                "n_variants": int(len(coef)),
                "coef_min": float(np.min(coef)),
                "coef_q25": float(np.quantile(coef, 0.25)),
                "coef_median": float(np.median(coef)),
                "coef_q75": float(np.quantile(coef, 0.75)),
                "coef_max": float(np.max(coef)),
                "coef_weighted_mean": _weighted_mean(coef, w),
                "coef_std": float(np.std(coef, ddof=0)),
                "share_positive": share_pos,
                "share_negative": share_neg,
                "share_conflict": float(1.0 - abs(share_pos - share_neg)),
                "post_period": int(float(rel_year) >= 0.0),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["coef_band_iqr"] = pd.to_numeric(out["coef_q75"], errors="coerce") - pd.to_numeric(out["coef_q25"], errors="coerce")
    out["coef_band_full"] = pd.to_numeric(out["coef_max"], errors="coerce") - pd.to_numeric(out["coef_min"], errors="coerce")
    return out.sort_values("rel_year").reset_index(drop=True)


def _centered_pre_abs_mean(event: pd.DataFrame, value_col: str = "coef_weighted_mean") -> float:
    if event.empty or value_col not in event.columns or "post_period" not in event.columns:
        return float("nan")
    pre = event[event["post_period"] == 0].copy()
    if pre.empty:
        return float("nan")
    vals = pd.to_numeric(pre[value_col], errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    centered = vals - float(vals.mean())
    return float(np.mean(np.abs(centered.to_numpy(dtype=float))))


def _build_event_envelope_bootstrap(draws: int = 800, random_state: int = 42) -> pd.DataFrame:
    pts = _event_points_with_weights()
    if pts.empty:
        return pd.DataFrame(
            columns=[
                "rel_year",
                "n_variants",
                "weighted_mean",
                "ci_low_95",
                "ci_high_95",
                "boot_std",
                "boot_se",
                "signal_to_noise",
                "post_period",
            ]
        )

    rng = np.random.default_rng(random_state)
    rows: List[Dict[str, Any]] = []
    for rel_year, grp in pts.groupby("rel_year", sort=True):
        coef = pd.to_numeric(grp["coef"], errors="coerce").dropna().to_numpy(dtype=float)
        w = pd.to_numeric(grp["variant_weight"], errors="coerce").fillna(0.55).to_numpy(dtype=float)
        n = int(len(coef))
        if n == 0:
            continue
        if len(w) != n:
            w = np.ones(n, dtype=float)
        point = _weighted_mean(coef, w)
        if n == 1:
            ci_low = point
            ci_high = point
            boot_std = 0.0
            boot_se = 0.0
        else:
            boot = np.zeros(draws, dtype=float)
            for b in range(draws):
                idx = rng.integers(0, n, size=n)
                boot[b] = _weighted_mean(coef[idx], w[idx])
            ci_low = float(np.quantile(boot, 0.025))
            ci_high = float(np.quantile(boot, 0.975))
            boot_std = float(np.std(boot, ddof=0))
            boot_se = float(np.std(boot, ddof=0))
        rows.append(
            {
                "rel_year": int(rel_year),
                "n_variants": n,
                "weighted_mean": point,
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
                "boot_std": boot_std,
                "boot_se": boot_se,
                # Use a floor to avoid inflated SNR when bootstrap variance is near zero.
                "signal_to_noise": float(point / max(boot_std, 0.02)),
                "post_period": int(float(rel_year) >= 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("rel_year").reset_index(drop=True)


def _build_regime_event_envelope() -> pd.DataFrame:
    st = _safe_read_csv("pulse_ai_dynamic_state_event_study.csv")
    if st.empty:
        return pd.DataFrame(
            columns=[
                "event_type",
                "event_time",
                "n_obs",
                "mean_rel_growth_adj",
                "mean_rel_risk_adj",
                "event_phase",
                "growth_z",
                "risk_z",
                "risk_forward_shift_z",
                "envelope_signal",
            ]
        )

    keep = ["event_type", "event_time", "n_obs", "mean_rel_growth_adj", "mean_rel_risk_adj"]
    for col in keep:
        if col not in st.columns:
            st[col] = np.nan
    out = st[keep].copy()
    out["event_type"] = out["event_type"].astype(str)
    out["event_time"] = pd.to_numeric(out["event_time"], errors="coerce")
    out["n_obs"] = pd.to_numeric(out["n_obs"], errors="coerce")
    out["mean_rel_growth_adj"] = pd.to_numeric(out["mean_rel_growth_adj"], errors="coerce")
    out["mean_rel_risk_adj"] = pd.to_numeric(out["mean_rel_risk_adj"], errors="coerce")
    out = out.dropna(subset=["event_type", "event_time"]).copy()
    if out.empty:
        return out

    out["event_phase"] = np.where(
        out["event_time"] < 0,
        "pre",
        np.where(out["event_time"] == 0, "impact", "post"),
    )

    rows: List[pd.DataFrame] = []
    for _, grp in out.groupby("event_type", sort=True):
        g = grp.sort_values("event_time").copy()
        g["growth_z"] = _zscore_series(g["mean_rel_growth_adj"])
        g["risk_z"] = _zscore_series(g["mean_rel_risk_adj"])
        risk_forward_shift = g["mean_rel_risk_adj"].shift(-1) - g["mean_rel_risk_adj"]
        g["risk_forward_shift_z"] = _zscore_series(risk_forward_shift).fillna(0.0)
        g["envelope_signal"] = (
            0.62 * g["growth_z"] - 0.48 * g["risk_z"] - 0.22 * g["risk_forward_shift_z"]
        )
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["event_type", "event_time"]).reset_index(drop=True)


def _build_regime_summary(regime_event: pd.DataFrame) -> pd.DataFrame:
    if regime_event.empty:
        return pd.DataFrame(
            columns=[
                "event_type",
                "n_rows",
                "n_obs_total",
                "pre_signal_mean",
                "impact_signal",
                "post_signal_mean",
                "growth_impact",
                "risk_impact",
                "post_risk_relief",
                "absorption_score_0_100",
            ]
        )

    rows: List[Dict[str, Any]] = []
    for event_type, grp in regime_event.groupby("event_type", sort=True):
        g = grp.sort_values("event_time")
        pre = g[g["event_time"] < 0]
        imp = g[g["event_time"] == 0]
        post = g[g["event_time"] > 0]
        pre_signal = float(pd.to_numeric(pre["envelope_signal"], errors="coerce").mean()) if not pre.empty else float("nan")
        impact_signal = float(pd.to_numeric(imp["envelope_signal"], errors="coerce").mean()) if not imp.empty else float("nan")
        post_signal = float(pd.to_numeric(post["envelope_signal"], errors="coerce").mean()) if not post.empty else float("nan")
        growth_impact = float(pd.to_numeric(imp["mean_rel_growth_adj"], errors="coerce").mean()) if not imp.empty else float("nan")
        risk_impact = float(pd.to_numeric(imp["mean_rel_risk_adj"], errors="coerce").mean()) if not imp.empty else float("nan")
        post_risk = float(pd.to_numeric(post["mean_rel_risk_adj"], errors="coerce").mean()) if not post.empty else float("nan")
        pre_risk = float(pd.to_numeric(pre["mean_rel_risk_adj"], errors="coerce").mean()) if not pre.empty else float("nan")
        post_relief = pre_risk - post_risk if np.isfinite(pre_risk) and np.isfinite(post_risk) else float("nan")
        rows.append(
            {
                "event_type": str(event_type),
                "n_rows": int(len(g)),
                "n_obs_total": int(pd.to_numeric(g["n_obs"], errors="coerce").fillna(0).sum()),
                "pre_signal_mean": pre_signal,
                "impact_signal": impact_signal,
                "post_signal_mean": post_signal,
                "growth_impact": growth_impact,
                "risk_impact": risk_impact,
                "post_risk_relief": post_relief,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    score_raw = (
        0.38 * pd.to_numeric(out["pre_signal_mean"], errors="coerce").fillna(0.0)
        + 0.34 * pd.to_numeric(out["post_signal_mean"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(out["post_risk_relief"], errors="coerce").fillna(0.0)
        - 0.08 * pd.to_numeric(out["risk_impact"], errors="coerce").fillna(0.0)
    )
    out["absorption_score_0_100"] = 100.0 * _minmax_series(score_raw, default=0.5)
    return out.sort_values("absorption_score_0_100", ascending=False).reset_index(drop=True)


def _build_city_envelope_scores() -> pd.DataFrame:
    series = _safe_read_csv("pulse_ai_dynamic_index_series.csv")
    if series.empty:
        return pd.DataFrame(
            columns=[
                "city_id",
                "city_name",
                "country",
                "continent",
                "year",
                "trend_3y",
                "volatility_5y",
                "drawdown_5y",
                "positive_delta_share_5y",
                "momentum_gap",
                "stall_risk_score",
                "dynamic_hazard_fused_score",
                "acceleration_score",
                "envelope_score_0_100",
                "fragility_gap",
                "envelope_class",
                "trajectory_regime",
                "kinetic_state",
            ]
        )

    keep = [
        "city_id",
        "city_name",
        "country",
        "continent",
        "year",
        "dynamic_pulse_index",
        "dynamic_pulse_delta_1y",
        "dynamic_pulse_trend_3y",
        "pulse_accel_velocity",
        "pulse_risk_velocity",
    ]
    for col in keep:
        if col not in series.columns:
            series[col] = np.nan
    s = series[keep].copy()
    s["year"] = pd.to_numeric(s["year"], errors="coerce")
    s["dynamic_pulse_index"] = pd.to_numeric(s["dynamic_pulse_index"], errors="coerce")
    s["dynamic_pulse_delta_1y"] = pd.to_numeric(s["dynamic_pulse_delta_1y"], errors="coerce")
    s["dynamic_pulse_trend_3y"] = pd.to_numeric(s["dynamic_pulse_trend_3y"], errors="coerce")
    s["pulse_accel_velocity"] = pd.to_numeric(s["pulse_accel_velocity"], errors="coerce")
    s["pulse_risk_velocity"] = pd.to_numeric(s["pulse_risk_velocity"], errors="coerce")
    s = s.dropna(subset=["city_id", "year"]).copy()
    if s.empty:
        return pd.DataFrame()

    city_rows: List[Dict[str, Any]] = []
    for city_id, grp in s.groupby("city_id", sort=False):
        g = grp.sort_values("year").copy()
        g5 = g.tail(5).copy()
        g3 = g.tail(3).copy()

        latest = g.iloc[-1]
        deltas = pd.to_numeric(g5["dynamic_pulse_delta_1y"], errors="coerce").dropna()
        index_tail = pd.to_numeric(g5["dynamic_pulse_index"], errors="coerce").dropna()
        trend3 = _safe_float(latest.get("dynamic_pulse_trend_3y"))
        if not np.isfinite(trend3) and len(g3) >= 2:
            trend3 = _safe_float(g3.iloc[-1].get("dynamic_pulse_index")) - _safe_float(g3.iloc[0].get("dynamic_pulse_index"))
        drawdown = float(index_tail.max() - index_tail.iloc[-1]) if len(index_tail) > 0 else float("nan")
        volatility = float(deltas.std(ddof=0)) if len(deltas) > 0 else float("nan")
        pos_share = float(np.mean(deltas > 0)) if len(deltas) > 0 else float("nan")
        accel_v = _safe_float(latest.get("pulse_accel_velocity"), 0.0)
        risk_v = _safe_float(latest.get("pulse_risk_velocity"), 0.0)
        city_rows.append(
            {
                "city_id": str(city_id),
                "city_name": str(latest.get("city_name", "")),
                "country": str(latest.get("country", "")),
                "continent": str(latest.get("continent", "")),
                "year": int(latest.get("year")),
                "trend_3y": trend3,
                "volatility_5y": volatility,
                "drawdown_5y": drawdown,
                "positive_delta_share_5y": pos_share,
                "momentum_gap": float(accel_v - risk_v),
            }
        )
    city = pd.DataFrame(city_rows)

    hazard = _safe_read_csv("pulse_ai_dynamic_hazard_latest.csv")
    if not hazard.empty:
        hz = hazard.copy()
        for c in [
            "stall_risk_score",
            "dynamic_hazard_fused_score",
            "turning_point_risk",
            "damping_pressure",
        ]:
            hz[c] = pd.to_numeric(hz.get(c), errors="coerce")
        cols = [
            "city_id",
            "stall_risk_score",
            "dynamic_hazard_fused_score",
            "turning_point_risk",
            "damping_pressure",
            "trajectory_regime",
            "kinetic_state",
        ]
        for col in cols:
            if col not in hz.columns:
                hz[col] = np.nan
        city = city.merge(hz[cols], on="city_id", how="left")

    latest = _safe_read_csv("pulse_ai_city_latest.csv")
    if not latest.empty:
        lt = latest.copy()
        lt["acceleration_score"] = pd.to_numeric(lt.get("acceleration_score"), errors="coerce")
        lt["stall_risk_score"] = pd.to_numeric(lt.get("stall_risk_score"), errors="coerce")
        city = city.merge(
            lt[["city_id", "acceleration_score", "stall_risk_score"]].rename(
                columns={"stall_risk_score": "stall_risk_score_latest"}
            ),
            on="city_id",
            how="left",
        )
        city["stall_risk_score"] = city["stall_risk_score"].fillna(city["stall_risk_score_latest"])
        city = city.drop(columns=["stall_risk_score_latest"])
    else:
        if "acceleration_score" not in city.columns:
            city["acceleration_score"] = np.nan

    risk_proxy = pd.to_numeric(city.get("stall_risk_score"), errors="coerce").fillna(
        pd.to_numeric(city.get("dynamic_hazard_fused_score"), errors="coerce")
    )
    growth_c = _minmax_series(city["trend_3y"], default=0.5)
    stability_c = _minmax_series(city["volatility_5y"], invert=True, default=0.5)
    drawdown_c = _minmax_series(city["drawdown_5y"], invert=True, default=0.5)
    persist_c = _minmax_series(city["positive_delta_share_5y"], default=0.5)
    momentum_c = _minmax_series(city["momentum_gap"], default=0.5)
    risk_c = _minmax_series(risk_proxy, invert=True, default=0.5)

    city["envelope_score_0_100"] = 100.0 * (
        0.26 * growth_c
        + 0.19 * stability_c
        + 0.13 * drawdown_c
        + 0.15 * persist_c
        + 0.11 * momentum_c
        + 0.16 * risk_c
    )
    city["stall_risk_score"] = risk_proxy
    city["fragility_gap"] = pd.to_numeric(city["stall_risk_score"], errors="coerce").fillna(50.0) - pd.to_numeric(
        city["envelope_score_0_100"],
        errors="coerce",
    ).fillna(50.0)

    def _classify(row: pd.Series) -> str:
        env = _safe_float(row.get("envelope_score_0_100"), 50.0)
        risk = _safe_float(row.get("stall_risk_score"), 50.0)
        momentum = _safe_float(row.get("momentum_gap"), 0.0)
        if env >= 70.0 and risk <= 45.0:
            return "resilient_accelerator"
        if env >= 58.0 and risk > 45.0:
            return "pressured_growth"
        if env < 45.0 and risk >= 60.0:
            return "stall_trap"
        if momentum >= 0.0:
            return "adaptive_transition"
        return "fragile_transition"

    city["envelope_class"] = city.apply(_classify, axis=1)
    keep_cols = [
        "city_id",
        "city_name",
        "country",
        "continent",
        "year",
        "trend_3y",
        "volatility_5y",
        "drawdown_5y",
        "positive_delta_share_5y",
        "momentum_gap",
        "stall_risk_score",
        "dynamic_hazard_fused_score",
        "acceleration_score",
        "envelope_score_0_100",
        "fragility_gap",
        "envelope_class",
        "trajectory_regime",
        "kinetic_state",
        "turning_point_risk",
        "damping_pressure",
    ]
    for c in keep_cols:
        if c not in city.columns:
            city[c] = np.nan
    return city[keep_cols].sort_values("envelope_score_0_100", ascending=False).reset_index(drop=True)


def _build_continent_year_envelope() -> pd.DataFrame:
    series = _safe_read_csv("pulse_ai_dynamic_index_series.csv")
    if series.empty:
        return pd.DataFrame(
            columns=[
                "continent",
                "year",
                "city_count",
                "envelope_score_mean",
                "envelope_score_p25",
                "envelope_score_p75",
                "envelope_score_std",
                "mean_delta_1y",
                "mean_trend_3y",
                "yoy_change",
            ]
        )

    keep = [
        "continent",
        "year",
        "dynamic_pulse_index",
        "dynamic_pulse_delta_1y",
        "dynamic_pulse_trend_3y",
        "pulse_accel_velocity",
        "pulse_risk_velocity",
    ]
    for col in keep:
        if col not in series.columns:
            series[col] = np.nan
    d = series[keep].copy()
    d["continent"] = d["continent"].astype(str)
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["dynamic_pulse_index"] = pd.to_numeric(d["dynamic_pulse_index"], errors="coerce")
    d["dynamic_pulse_delta_1y"] = pd.to_numeric(d["dynamic_pulse_delta_1y"], errors="coerce")
    d["dynamic_pulse_trend_3y"] = pd.to_numeric(d["dynamic_pulse_trend_3y"], errors="coerce")
    d["pulse_accel_velocity"] = pd.to_numeric(d["pulse_accel_velocity"], errors="coerce")
    d["pulse_risk_velocity"] = pd.to_numeric(d["pulse_risk_velocity"], errors="coerce")
    d = d.dropna(subset=["continent", "year"]).copy()
    if d.empty:
        return pd.DataFrame()

    momentum_gap = d["pulse_accel_velocity"].fillna(0.0) - d["pulse_risk_velocity"].fillna(0.0)
    row_score = 100.0 * (
        0.45 * _minmax_series(d["dynamic_pulse_index"], default=0.5)
        + 0.25 * _minmax_series(d["dynamic_pulse_trend_3y"], default=0.5)
        + 0.15 * _minmax_series(d["dynamic_pulse_delta_1y"].abs(), invert=True, default=0.5)
        + 0.15 * _minmax_series(momentum_gap, default=0.5)
    )
    d["row_envelope_score"] = row_score

    agg = (
        d.groupby(["continent", "year"], as_index=False)
        .agg(
            city_count=("row_envelope_score", "size"),
            envelope_score_mean=("row_envelope_score", "mean"),
            envelope_score_p25=("row_envelope_score", lambda x: float(np.quantile(np.asarray(x, dtype=float), 0.25))),
            envelope_score_p75=("row_envelope_score", lambda x: float(np.quantile(np.asarray(x, dtype=float), 0.75))),
            envelope_score_std=("row_envelope_score", "std"),
            mean_delta_1y=("dynamic_pulse_delta_1y", "mean"),
            mean_trend_3y=("dynamic_pulse_trend_3y", "mean"),
        )
        .sort_values(["continent", "year"])
    )
    agg["yoy_change"] = agg.groupby("continent")["envelope_score_mean"].diff().fillna(0.0)
    return agg.reset_index(drop=True)


def _safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    xv = np.asarray(x, dtype=float)[mask]
    yv = np.asarray(y, dtype=float)[mask]
    if len(xv) < 2:
        return float("nan")
    if float(np.nanstd(xv, ddof=0)) <= 1e-12:
        return float("nan")
    try:
        return float(np.polyfit(xv, yv, deg=1)[0])
    except Exception:  # noqa: BLE001
        return float("nan")


def _build_continent_stability(continent_year: pd.DataFrame, draws: int = 400, random_state: int = 42) -> pd.DataFrame:
    if continent_year.empty:
        return pd.DataFrame(
            columns=[
                "continent",
                "year_count",
                "score_last",
                "score_mean",
                "trend_slope",
                "trend_ci_low_95",
                "trend_ci_high_95",
                "yoy_volatility",
                "yoy_positive_share",
                "stability_score_0_100",
                "stability_rank",
            ]
        )

    rng = np.random.default_rng(random_state)
    rows: List[Dict[str, Any]] = []
    for continent, grp in continent_year.groupby("continent", sort=True):
        g = grp.sort_values("year").copy()
        x = pd.to_numeric(g["year"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g["envelope_score_mean"], errors="coerce").to_numpy(dtype=float)
        x0 = x - np.nanmin(x) if len(x) else x
        slope = _safe_slope(x0, y)
        yoy = pd.to_numeric(g["yoy_change"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        boot = np.zeros(draws, dtype=float)
        if len(g) >= 3:
            for b in range(draws):
                idx = rng.integers(0, len(g), size=len(g))
                xb = x0[idx]
                yb = y[idx]
                boot[b] = _safe_slope(xb, yb)
            boot_valid = boot[np.isfinite(boot)]
            if boot_valid.size == 0:
                ci_low = slope
                ci_high = slope
            else:
                ci_low = float(np.nanquantile(boot_valid, 0.025))
                ci_high = float(np.nanquantile(boot_valid, 0.975))
        else:
            ci_low = slope
            ci_high = slope

        rows.append(
            {
                "continent": str(continent),
                "year_count": int(len(g)),
                "score_last": float(y[-1]) if len(y) else np.nan,
                "score_mean": float(np.nanmean(y)) if len(y) else np.nan,
                "trend_slope": slope,
                "trend_ci_low_95": ci_low,
                "trend_ci_high_95": ci_high,
                "yoy_volatility": float(np.nanstd(yoy, ddof=0)) if len(yoy) else np.nan,
                "yoy_positive_share": float(np.nanmean(yoy > 0)) if len(yoy) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    score_c = _minmax_series(out["score_mean"], default=0.5)
    slope_c = _minmax_series(out["trend_slope"], default=0.5)
    vol_c = _minmax_series(out["yoy_volatility"], invert=True, default=0.5)
    pos_c = _minmax_series(out["yoy_positive_share"], default=0.5)
    out["stability_score_0_100"] = 100.0 * (0.36 * score_c + 0.28 * slope_c + 0.22 * vol_c + 0.14 * pos_c)
    out = out.sort_values("stability_score_0_100", ascending=False).reset_index(drop=True)
    out["stability_rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def run_dynamic_causal_envelope_suite() -> Dict[str, Any]:
    """Generate dynamic-causal-envelope artifacts for dashboard and paper diagnostics."""
    event = _build_event_envelope()
    event_boot = _build_event_envelope_bootstrap()
    regime_event = _build_regime_event_envelope()
    regime_summary = _build_regime_summary(regime_event)
    city = _build_city_envelope_scores()
    continent_year = _build_continent_year_envelope()
    continent_stability = _build_continent_stability(continent_year)

    event_path = DATA_OUTPUTS / "dynamic_causal_envelope_event.csv"
    event_boot_path = DATA_OUTPUTS / "dynamic_causal_envelope_event_bootstrap.csv"
    regime_path = DATA_OUTPUTS / "dynamic_causal_envelope_regime.csv"
    regime_summary_path = DATA_OUTPUTS / "dynamic_causal_envelope_regime_summary.csv"
    city_path = DATA_OUTPUTS / "dynamic_causal_envelope_city_scores.csv"
    continent_path = DATA_OUTPUTS / "dynamic_causal_envelope_continent_year.csv"
    continent_stability_path = DATA_OUTPUTS / "dynamic_causal_envelope_continent_stability.csv"

    event.to_csv(event_path, index=False)
    event_boot.to_csv(event_boot_path, index=False)
    regime_event.to_csv(regime_path, index=False)
    regime_summary.to_csv(regime_summary_path, index=False)
    city.to_csv(city_path, index=False)
    continent_year.to_csv(continent_path, index=False)
    continent_stability.to_csv(continent_stability_path, index=False)

    post = event[event["post_period"] == 1].copy() if (not event.empty and "post_period" in event.columns) else pd.DataFrame()
    pre = event[event["post_period"] == 0].copy() if (not event.empty and "post_period" in event.columns) else pd.DataFrame()

    top_resilient: Dict[str, Any] = {}
    top_fragile: Dict[str, Any] = {}
    if not city.empty:
        top = city.iloc[0]
        top_resilient = {
            "city_id": str(top.get("city_id", "")),
            "city_name": str(top.get("city_name", "")),
            "country": str(top.get("country", "")),
            "continent": str(top.get("continent", "")),
            "envelope_score_0_100": _safe_float(top.get("envelope_score_0_100")),
            "stall_risk_score": _safe_float(top.get("stall_risk_score")),
            "class": str(top.get("envelope_class", "")),
        }
        frag = city.sort_values("fragility_gap", ascending=False).iloc[0]
        top_fragile = {
            "city_id": str(frag.get("city_id", "")),
            "city_name": str(frag.get("city_name", "")),
            "country": str(frag.get("country", "")),
            "continent": str(frag.get("continent", "")),
            "fragility_gap": _safe_float(frag.get("fragility_gap")),
            "envelope_score_0_100": _safe_float(frag.get("envelope_score_0_100")),
            "stall_risk_score": _safe_float(frag.get("stall_risk_score")),
            "class": str(frag.get("envelope_class", "")),
        }

    resilient_share = (
        float(np.mean(city["envelope_class"].astype(str) == "resilient_accelerator")) if not city.empty else float("nan")
    )
    stall_trap_share = float(np.mean(city["envelope_class"].astype(str) == "stall_trap")) if not city.empty else float("nan")

    best_regime = regime_summary.iloc[0].to_dict() if not regime_summary.empty else {}
    post_boot = event_boot[event_boot["post_period"] == 1].copy() if (not event_boot.empty and "post_period" in event_boot.columns) else pd.DataFrame()
    post_ci_above_zero_share = (
        float(np.mean(pd.to_numeric(post_boot.get("ci_low_95"), errors="coerce").fillna(-np.inf) > 0.0))
        if not post_boot.empty
        else float("nan")
    )
    post_snr_mean = (
        float(pd.to_numeric(post_boot.get("signal_to_noise"), errors="coerce").replace([np.inf, -np.inf], np.nan).mean())
        if not post_boot.empty
        else float("nan")
    )
    top_stability = continent_stability.iloc[0].to_dict() if not continent_stability.empty else {}
    bottom_stability = continent_stability.iloc[-1].to_dict() if not continent_stability.empty else {}

    summary = {
        "status": "ok",
        "event_rows": int(len(event)),
        "event_bootstrap_rows": int(len(event_boot)),
        "regime_event_rows": int(len(regime_event)),
        "regime_summary_rows": int(len(regime_summary)),
        "city_rows": int(len(city)),
        "continent_year_rows": int(len(continent_year)),
        "continent_stability_rows": int(len(continent_stability)),
        # Pre-period noise should measure drift around the local pre-period baseline,
        # not penalize a common level offset across designs.
        "event_pre_abs_median": _centered_pre_abs_mean(event, value_col="coef_weighted_mean"),
        "event_post_weighted_mean": float(pd.to_numeric(post.get("coef_weighted_mean"), errors="coerce").mean()) if not post.empty else float("nan"),
        "event_post_iqr_band_mean": float(pd.to_numeric(post.get("coef_band_iqr"), errors="coerce").mean()) if not post.empty else float("nan"),
        "event_post_ci_above_zero_share": post_ci_above_zero_share,
        "event_post_signal_to_noise_mean": post_snr_mean,
        "resilient_accelerator_share": resilient_share,
        "stall_trap_share": stall_trap_share,
        "top_resilient_city": top_resilient,
        "top_fragility_city": top_fragile,
        "best_regime_absorption": best_regime,
        "top_continent_stability": top_stability,
        "bottom_continent_stability": bottom_stability,
        "mean_continent_stability_score": float(pd.to_numeric(continent_stability.get("stability_score_0_100"), errors="coerce").mean()) if not continent_stability.empty else float("nan"),
        "continents_covered": int(continent_year["continent"].nunique()) if not continent_year.empty else 0,
        "year_min": int(continent_year["year"].min()) if not continent_year.empty else None,
        "year_max": int(continent_year["year"].max()) if not continent_year.empty else None,
        "artifacts": {
            "event_csv": str(event_path),
            "event_bootstrap_csv": str(event_boot_path),
            "regime_event_csv": str(regime_path),
            "regime_summary_csv": str(regime_summary_path),
            "city_scores_csv": str(city_path),
            "continent_year_csv": str(continent_path),
            "continent_stability_csv": str(continent_stability_path),
        },
    }
    dump_json(DATA_OUTPUTS / "dynamic_causal_envelope_summary.json", summary)
    return summary
