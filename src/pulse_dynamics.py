from __future__ import annotations

"""Dynamic transition-survival diagnostics for MACRO-City Engine."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .utils import DATA_OUTPUTS, dump_json


STATE_ORDER: List[str] = [
    "resilient_accelerator",
    "fragile_boom",
    "transitional",
    "recovery_window",
    "stall_trap",
]

STATE_RANK: Dict[str, int] = {
    "stall_trap": 0,
    "recovery_window": 1,
    "transitional": 2,
    "fragile_boom": 3,
    "resilient_accelerator": 4,
}


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = float(successes) / float(total)
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z / denom) * np.sqrt((p * (1.0 - p) / total) + (z * z) / (4.0 * total * total))
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _resolve_prediction_col(df: pd.DataFrame) -> str | None:
    for candidate in ["dynamic_hazard_fused_probability", "dynamic_hazard_probability", "stall_probability"]:
        if candidate in df.columns:
            return candidate
    return None


def _assign_dynamic_state(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["stall_risk_score"] = _safe_float(out["stall_risk_score"])
    out["acceleration_score"] = _safe_float(out["acceleration_score"])

    yearly = (
        out.groupby("year", as_index=False)
        .agg(
            risk_q35=("stall_risk_score", lambda s: float(s.quantile(0.35))),
            risk_q65=("stall_risk_score", lambda s: float(s.quantile(0.65))),
            accel_q35=("acceleration_score", lambda s: float(s.quantile(0.35))),
            accel_q65=("acceleration_score", lambda s: float(s.quantile(0.65))),
        )
        .sort_values("year")
    )
    out = out.merge(yearly, on="year", how="left")

    cond = [
        (out["stall_risk_score"] >= out["risk_q65"]) & (out["acceleration_score"] <= out["accel_q35"]),
        (out["stall_risk_score"] <= out["risk_q35"]) & (out["acceleration_score"] >= out["accel_q65"]),
        (out["stall_risk_score"] >= out["risk_q65"]) & (out["acceleration_score"] >= out["accel_q65"]),
        (out["stall_risk_score"] <= out["risk_q35"]) & (out["acceleration_score"] <= out["accel_q35"]),
    ]
    out["pulse_dynamic_state"] = np.select(
        cond,
        ["stall_trap", "resilient_accelerator", "fragile_boom", "recovery_window"],
        default="transitional",
    )
    return out, yearly


def _build_transition_tensor(panel: pd.DataFrame) -> pd.DataFrame:
    cols = ["city_id", "continent", "year", "pulse_dynamic_state"]
    work = panel[cols].dropna(subset=["city_id", "year", "pulse_dynamic_state"]).copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "scope",
                "continent",
                "from_state",
                "to_state",
                "n_transitions",
                "from_total",
                "probability",
                "prob_ci_low_95",
                "prob_ci_high_95",
                "upgrade_step",
            ]
        )

    work = work.sort_values(["city_id", "year"]).copy()
    work["to_state"] = work.groupby("city_id")["pulse_dynamic_state"].shift(-1)
    work["next_year"] = work.groupby("city_id")["year"].shift(-1)
    trans = work[(work["next_year"] == (work["year"] + 1)) & work["to_state"].notna()].copy()
    if trans.empty:
        return pd.DataFrame(
            columns=[
                "scope",
                "continent",
                "from_state",
                "to_state",
                "n_transitions",
                "from_total",
                "probability",
                "prob_ci_low_95",
                "prob_ci_high_95",
                "upgrade_step",
            ]
        )

    records: List[Dict[str, Any]] = []
    continents = sorted(trans["continent"].dropna().astype(str).unique().tolist())
    scopes: List[tuple[str, str | None, pd.DataFrame]] = [("global", None, trans)]
    scopes.extend([("continent", cont, trans[trans["continent"].astype(str) == cont].copy()) for cont in continents])

    for scope_name, scope_continent, sub in scopes:
        if sub.empty:
            continue
        counts = (
            sub.groupby(["pulse_dynamic_state", "to_state"], as_index=False)
            .size()
            .rename(columns={"pulse_dynamic_state": "from_state", "size": "n_transitions"})
        )
        from_tot = counts.groupby("from_state", as_index=False)["n_transitions"].sum().rename(columns={"n_transitions": "from_total"})
        counts = counts.merge(from_tot, on="from_state", how="left")
        counts["probability"] = counts["n_transitions"] / counts["from_total"].replace(0, np.nan)

        for row in counts.itertuples(index=False):
            n = int(row.n_transitions)
            total = int(row.from_total)
            lo, hi = _wilson_interval(n, total)
            from_state = str(row.from_state)
            to_state = str(row.to_state)
            records.append(
                {
                    "scope": scope_name,
                    "continent": "Global" if scope_continent is None else str(scope_continent),
                    "from_state": from_state,
                    "to_state": to_state,
                    "n_transitions": n,
                    "from_total": total,
                    "probability": float(row.probability) if np.isfinite(row.probability) else np.nan,
                    "prob_ci_low_95": lo,
                    "prob_ci_high_95": hi,
                    "upgrade_step": int(STATE_RANK.get(to_state, 2) - STATE_RANK.get(from_state, 2)),
                }
            )

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["state_order"] = out["from_state"].astype(str).map({s: i for i, s in enumerate(STATE_ORDER)}).fillna(999)
    out = out.sort_values(["scope", "continent", "state_order", "to_state"]).drop(columns=["state_order"]).reset_index(drop=True)
    return out


def _extract_spells(panel: pd.DataFrame, target_state: str) -> pd.DataFrame:
    cols = ["city_id", "continent", "year", "pulse_dynamic_state"]
    work = panel[cols].dropna(subset=["city_id", "year", "pulse_dynamic_state"]).copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "city_id",
                "continent",
                "state",
                "start_year",
                "end_year",
                "duration",
                "exit_observed",
            ]
        )

    rows: List[Dict[str, Any]] = []
    for city_id, grp in work.sort_values(["city_id", "year"]).groupby("city_id"):
        g = grp.sort_values("year")
        years = g["year"].astype(int).to_list()
        states = g["pulse_dynamic_state"].astype(str).to_list()
        cont = str(g["continent"].iloc[-1]) if "continent" in g.columns else "Unknown"

        i = 0
        n = len(g)
        while i < n:
            if states[i] != target_state:
                i += 1
                continue
            j = i
            while (j + 1) < n and years[j + 1] == (years[j] + 1) and states[j + 1] == target_state:
                j += 1

            has_follow = (j + 1) < n and years[j + 1] == (years[j] + 1)
            rows.append(
                {
                    "city_id": str(city_id),
                    "continent": cont,
                    "state": str(target_state),
                    "start_year": int(years[i]),
                    "end_year": int(years[j]),
                    "duration": int(j - i + 1),
                    "exit_observed": int(bool(has_follow and states[j + 1] != target_state)),
                }
            )
            i = j + 1

    return pd.DataFrame(rows)


def _build_spell_hazard(spells: pd.DataFrame, spell_kind: str) -> pd.DataFrame:
    cols = [
        "scope",
        "continent",
        "spell_kind",
        "duration",
        "at_risk",
        "exits",
        "hazard",
        "hazard_ci_low_95",
        "hazard_ci_high_95",
        "survival_after_t",
    ]
    if spells.empty:
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, Any]] = []
    continents = sorted(spells["continent"].dropna().astype(str).unique().tolist())
    scopes: List[tuple[str, str | None, pd.DataFrame]] = [("global", None, spells)]
    scopes.extend([("continent", c, spells[spells["continent"].astype(str) == c].copy()) for c in continents])

    for scope_name, scope_continent, sub in scopes:
        if sub.empty:
            continue
        max_d = int(pd.to_numeric(sub["duration"], errors="coerce").max())
        if max_d <= 0:
            continue
        survival = 1.0
        for t in range(1, max_d + 1):
            at_risk = int((pd.to_numeric(sub["duration"], errors="coerce") >= t).sum())
            if at_risk <= 0:
                break
            exits = int(
                (
                    (pd.to_numeric(sub["duration"], errors="coerce") == t)
                    & (pd.to_numeric(sub["exit_observed"], errors="coerce").fillna(0).astype(int) == 1)
                ).sum()
            )
            hz = float(exits) / float(at_risk)
            lo, hi = _wilson_interval(exits, at_risk)
            survival = survival * max(0.0, 1.0 - hz)
            rows.append(
                {
                    "scope": scope_name,
                    "continent": "Global" if scope_continent is None else str(scope_continent),
                    "spell_kind": spell_kind,
                    "duration": int(t),
                    "at_risk": at_risk,
                    "exits": exits,
                    "hazard": hz,
                    "hazard_ci_low_95": lo,
                    "hazard_ci_high_95": hi,
                    "survival_after_t": float(survival),
                }
            )

    return pd.DataFrame(rows, columns=cols)


def _half_life_for_group(hazard: pd.DataFrame) -> tuple[float, float]:
    if hazard.empty:
        return np.nan, np.nan
    h = hazard.sort_values("duration")
    half_life = np.nan
    survival_end = np.nan
    for row in h.itertuples(index=False):
        survival_end = float(row.survival_after_t)
        if (not np.isfinite(half_life)) and np.isfinite(survival_end) and survival_end <= 0.5:
            half_life = float(row.duration)
    return half_life, survival_end


def _build_resilience_halflife(
    stall_spells: pd.DataFrame,
    accel_spells: pd.DataFrame,
    hazard: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "scope",
        "continent",
        "stall_spell_count",
        "stall_mean_duration",
        "stall_exit_half_life",
        "stall_survival_end",
        "accel_spell_count",
        "accel_mean_duration",
        "accel_break_half_life",
        "accel_survival_end",
        "resilience_gap_t",
        "resilience_score_0_100",
    ]
    all_continents = sorted(
        set(stall_spells.get("continent", pd.Series(dtype=str)).dropna().astype(str).tolist())
        | set(accel_spells.get("continent", pd.Series(dtype=str)).dropna().astype(str).tolist())
    )
    groups: List[tuple[str, str | None]] = [("global", None)]
    groups.extend([("continent", c) for c in all_continents])

    rows: List[Dict[str, Any]] = []
    for scope_name, continent in groups:
        if scope_name == "global":
            stall_sub = stall_spells.copy()
            accel_sub = accel_spells.copy()
            hz_sub = hazard.copy()
            scope_label = "Global"
        else:
            stall_sub = stall_spells[stall_spells["continent"].astype(str) == str(continent)].copy()
            accel_sub = accel_spells[accel_spells["continent"].astype(str) == str(continent)].copy()
            hz_sub = hazard[hazard["continent"].astype(str) == str(continent)].copy()
            scope_label = str(continent)

        stall_hz = hz_sub[hz_sub["spell_kind"].astype(str) == "stall_trap"].copy()
        accel_hz = hz_sub[hz_sub["spell_kind"].astype(str) == "resilient_accelerator"].copy()
        stall_half, stall_surv = _half_life_for_group(stall_hz)
        accel_half, accel_surv = _half_life_for_group(accel_hz)
        stall_mean_d = float(pd.to_numeric(stall_sub.get("duration"), errors="coerce").mean()) if not stall_sub.empty else np.nan
        accel_mean_d = float(pd.to_numeric(accel_sub.get("duration"), errors="coerce").mean()) if not accel_sub.empty else np.nan

        part_exit = 0.5 * _clip01((4.0 - stall_half) / 4.0) if np.isfinite(stall_half) else 0.0
        part_hold = 0.5 * _clip01(accel_half / 4.0) if np.isfinite(accel_half) else 0.0
        score = 100.0 * (part_exit + part_hold)
        rows.append(
            {
                "scope": scope_name,
                "continent": scope_label,
                "stall_spell_count": int(len(stall_sub)),
                "stall_mean_duration": stall_mean_d,
                "stall_exit_half_life": stall_half,
                "stall_survival_end": stall_surv,
                "accel_spell_count": int(len(accel_sub)),
                "accel_mean_duration": accel_mean_d,
                "accel_break_half_life": accel_half,
                "accel_survival_end": accel_surv,
                "resilience_gap_t": (accel_half - stall_half) if (np.isfinite(accel_half) and np.isfinite(stall_half)) else np.nan,
                "resilience_score_0_100": score,
            }
        )

    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out["resilience_rank"] = out["resilience_score_0_100"].rank(method="dense", ascending=False).astype(int)
    return out


def _build_warning_horizon(panel: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    cols = [
        "scope",
        "continent",
        "horizon_years",
        "n_obs",
        "positive_rate",
        "auc",
        "average_precision",
        "brier",
        "top_decile_precision",
        "top_decile_lift",
        "warning_score_0_100",
        "prediction_col",
    ]
    work = panel[["city_id", "continent", "year", "stall_next", pred_col]].copy()
    work["stall_next"] = _safe_float(work["stall_next"])
    work[pred_col] = _safe_float(work[pred_col]).clip(lower=0.0, upper=1.0)
    work = work.sort_values(["city_id", "year"]).copy()

    if work.empty:
        return pd.DataFrame(columns=cols)

    for h in [1, 2, 3]:
        event = np.zeros(len(work), dtype=float)
        valid_any = np.zeros(len(work), dtype=bool)
        for lag in range(h):
            shifted = work.groupby("city_id")["stall_next"].shift(-lag)
            arr = pd.to_numeric(shifted, errors="coerce")
            valid = arr.notna().to_numpy()
            valid_any |= valid
            event = np.maximum(event, arr.fillna(0.0).to_numpy(dtype=float))
        work[f"stall_within_{h}y"] = np.where(valid_any, event, np.nan)

    rows: List[Dict[str, Any]] = []
    continents = sorted(work["continent"].dropna().astype(str).unique().tolist())
    scopes: List[tuple[str, str | None, pd.DataFrame]] = [("global", None, work)]
    scopes.extend([("continent", c, work[work["continent"].astype(str) == c].copy()) for c in continents])

    for scope_name, continent, sub in scopes:
        if sub.empty:
            continue
        for h in [1, 2, 3]:
            y = pd.to_numeric(sub[f"stall_within_{h}y"], errors="coerce")
            p = pd.to_numeric(sub[pred_col], errors="coerce")
            mask = y.notna() & p.notna()
            yv = y[mask].astype(int).to_numpy(dtype=int)
            pv = p[mask].to_numpy(dtype=float)
            if len(yv) < 30:
                continue
            base_rate = float(np.mean(yv))

            auc = np.nan
            ap = np.nan
            brier = np.nan
            if np.unique(yv).size >= 2:
                auc = float(roc_auc_score(yv, pv))
                ap = float(average_precision_score(yv, pv))
                brier = float(brier_score_loss(yv.astype(float), pv))
            else:
                ap = float(base_rate)
                brier = float(brier_score_loss(yv.astype(float), pv))

            q = float(np.quantile(pv, 0.90))
            top_mask = pv >= q
            top_precision = float(np.mean(yv[top_mask])) if np.any(top_mask) else np.nan
            top_lift = (top_precision / max(base_rate, 1e-6)) if np.isfinite(top_precision) else np.nan

            ap_norm = np.nan
            if np.isfinite(ap):
                denom = max(1e-6, 1.0 - base_rate)
                ap_norm = (ap - base_rate) / denom

            score = 100.0 * (
                0.50 * _clip01((auc - 0.5) / 0.5)
                + 0.30 * _clip01(ap_norm)
                + 0.20 * _clip01((top_lift - 1.0) / 2.0)
            )
            rows.append(
                {
                    "scope": scope_name,
                    "continent": "Global" if continent is None else str(continent),
                    "horizon_years": int(h),
                    "n_obs": int(len(yv)),
                    "positive_rate": base_rate,
                    "auc": auc,
                    "average_precision": ap,
                    "brier": brier,
                    "top_decile_precision": top_precision,
                    "top_decile_lift": top_lift,
                    "warning_score_0_100": score,
                    "prediction_col": pred_col,
                }
            )

    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out = out.sort_values(["scope", "continent", "horizon_years"]).reset_index(drop=True)
    return out


def run_pulse_dynamics_suite() -> Dict[str, Any]:
    """Generate transition, spell-hazard, and horizon-warning diagnostics for pulse dynamics."""
    src = DATA_OUTPUTS / "pulse_ai_scores.csv"
    if not src.exists():
        out = {"status": "skipped", "reason": "pulse_ai_scores_missing"}
        dump_json(DATA_OUTPUTS / "pulse_dynamics_summary.json", out)
        return out

    raw = pd.read_csv(src)
    if raw.empty:
        out = {"status": "skipped", "reason": "pulse_ai_scores_empty"}
        dump_json(DATA_OUTPUTS / "pulse_dynamics_summary.json", out)
        return out

    required = {"city_id", "continent", "year", "stall_risk_score", "acceleration_score", "stall_next"}
    if not required.issubset(set(raw.columns)):
        out = {"status": "skipped", "reason": "pulse_ai_scores_missing_required_columns"}
        dump_json(DATA_OUTPUTS / "pulse_dynamics_summary.json", out)
        return out

    panel, thresholds = _assign_dynamic_state(raw)
    pred_col = _resolve_prediction_col(panel)
    if pred_col is None:
        out = {"status": "skipped", "reason": "prediction_column_missing"}
        dump_json(DATA_OUTPUTS / "pulse_dynamics_summary.json", out)
        return out

    for opt_col in ["city_name", "country"]:
        if opt_col not in panel.columns:
            panel[opt_col] = ""

    state_cols = [
        "city_id",
        "city_name",
        "country",
        "continent",
        "year",
        "stall_risk_score",
        "acceleration_score",
        "stall_next",
        pred_col,
        "pulse_dynamic_state",
        "risk_q35",
        "risk_q65",
        "accel_q35",
        "accel_q65",
    ]
    panel[state_cols].to_csv(DATA_OUTPUTS / "pulse_dynamics_state_panel.csv", index=False)
    thresholds.to_csv(DATA_OUTPUTS / "pulse_dynamics_state_thresholds.csv", index=False)

    transitions = _build_transition_tensor(panel)
    transitions.to_csv(DATA_OUTPUTS / "pulse_dynamics_transition_tensor.csv", index=False)

    stall_spells = _extract_spells(panel, target_state="stall_trap")
    accel_spells = _extract_spells(panel, target_state="resilient_accelerator")
    stall_spells.to_csv(DATA_OUTPUTS / "pulse_dynamics_stall_spells.csv", index=False)
    accel_spells.to_csv(DATA_OUTPUTS / "pulse_dynamics_accel_spells.csv", index=False)

    hazard_stall = _build_spell_hazard(stall_spells, spell_kind="stall_trap")
    hazard_accel = _build_spell_hazard(accel_spells, spell_kind="resilient_accelerator")
    hazard = pd.concat([hazard_stall, hazard_accel], ignore_index=True)
    hazard.to_csv(DATA_OUTPUTS / "pulse_dynamics_spell_hazard.csv", index=False)

    resilience = _build_resilience_halflife(stall_spells, accel_spells, hazard)
    resilience.to_csv(DATA_OUTPUTS / "pulse_dynamics_resilience_halflife.csv", index=False)

    warning = _build_warning_horizon(panel, pred_col=pred_col)
    warning.to_csv(DATA_OUTPUTS / "pulse_dynamics_warning_horizon.csv", index=False)

    transition_cov = (
        int(
            transitions.loc[transitions["scope"].astype(str) == "continent", "continent"]
            .astype(str)
            .nunique()
        )
        if not transitions.empty
        else 0
    )
    global_transition = transitions[transitions["scope"].astype(str) == "global"].copy()
    total_trans = float(pd.to_numeric(global_transition.get("n_transitions"), errors="coerce").sum()) if not global_transition.empty else 0.0
    stall_entry = (
        float(
            pd.to_numeric(
                global_transition.loc[global_transition["to_state"].astype(str) == "stall_trap", "n_transitions"],
                errors="coerce",
            ).sum()
        )
        / total_trans
        if total_trans > 0.0
        else np.nan
    )

    global_warning = warning[warning["scope"].astype(str) == "global"].copy()
    global_res = resilience[resilience["scope"].astype(str) == "global"].copy()
    res_row = global_res.iloc[0] if not global_res.empty else pd.Series(dtype=object)

    def _global_warning_metric(h: int, col: str) -> float:
        if global_warning.empty:
            return np.nan
        sub = global_warning[global_warning["horizon_years"] == int(h)]
        if sub.empty:
            return np.nan
        return float(pd.to_numeric(sub.iloc[0].get(col), errors="coerce"))

    summary = {
        "status": "ok",
        "n_rows": int(len(panel)),
        "n_cities": int(panel["city_id"].nunique()),
        "year_count": int(panel["year"].nunique()),
        "continents_covered": int(panel["continent"].dropna().nunique()),
        "prediction_col": pred_col,
        "state_threshold_rows": int(len(thresholds)),
        "transition_rows": int(len(transitions)),
        "stall_spell_count": int(len(stall_spells)),
        "accel_spell_count": int(len(accel_spells)),
        "hazard_rows": int(len(hazard)),
        "warning_rows": int(len(warning)),
        "resilience_rows": int(len(resilience)),
        "transition_continent_coverage": transition_cov,
        "global_stall_entry_probability": stall_entry,
        "global_stall_exit_half_life": float(pd.to_numeric(res_row.get("stall_exit_half_life"), errors="coerce"))
        if not global_res.empty
        else np.nan,
        "global_accel_break_half_life": float(pd.to_numeric(res_row.get("accel_break_half_life"), errors="coerce"))
        if not global_res.empty
        else np.nan,
        "global_resilience_score_0_100": float(pd.to_numeric(res_row.get("resilience_score_0_100"), errors="coerce"))
        if not global_res.empty
        else np.nan,
        "global_warning_auc_h1": _global_warning_metric(1, "auc"),
        "global_warning_auc_h2": _global_warning_metric(2, "auc"),
        "global_warning_auc_h3": _global_warning_metric(3, "auc"),
        "global_warning_lift_h1": _global_warning_metric(1, "top_decile_lift"),
        "global_warning_lift_h2": _global_warning_metric(2, "top_decile_lift"),
        "global_warning_lift_h3": _global_warning_metric(3, "top_decile_lift"),
    }

    dump_json(DATA_OUTPUTS / "pulse_dynamics_summary.json", summary)
    return summary
