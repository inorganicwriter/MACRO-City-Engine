from __future__ import annotations

"""Event-level exogenous shock diagnostics based on objective policy events."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .utils import DATA_OUTPUTS, DATA_RAW, dump_json


@dataclass(frozen=True)
class OLSResult:
    coef: float
    stderr: float
    t_value: float
    p_value: float
    n_obs: int


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _ols_hc1(y: np.ndarray, x: np.ndarray) -> OLSResult:
    n, k = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    meat = np.zeros((k, k), dtype=float)
    for i in range(n):
        xi = x[i : i + 1, :]
        ui = float(resid[i])
        meat += (ui * ui) * (xi.T @ xi)
    hc1 = (n / max(1, n - k)) * (xtx_inv @ meat @ xtx_inv)
    se = float(np.sqrt(max(1e-12, hc1[1, 1]))) if k > 1 else float(np.sqrt(max(1e-12, hc1[0, 0])))
    coef = float(beta[1]) if k > 1 else float(beta[0])
    t_value = coef / se if se > 0 else float("nan")
    p_value = 2.0 * (1.0 - _normal_cdf(abs(t_value))) if np.isfinite(t_value) else float("nan")
    return OLSResult(coef=coef, stderr=se, t_value=t_value, p_value=p_value, n_obs=n)


def _two_way_demean(df: pd.DataFrame, col: str, city_col: str = "city_id", year_col: str = "year") -> pd.Series:
    x = pd.to_numeric(df[col], errors="coerce")
    city_mean = x.groupby(df[city_col]).transform("mean")
    year_mean = x.groupby(df[year_col]).transform("mean")
    grand_mean = float(np.nanmean(x.to_numpy(dtype=float)))
    return x - city_mean - year_mean + grand_mean


def _zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = float(np.nanmean(x.to_numpy(dtype=float)))
    sd = float(np.nanstd(x.to_numpy(dtype=float), ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    return (x - mu) / sd


def _minmax(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo = float(np.nanmin(x.to_numpy(dtype=float)))
    hi = float(np.nanmax(x.to_numpy(dtype=float)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    return (x - lo) / (hi - lo)


def _build_shock_year_index(registry: pd.DataFrame, panel_years: np.ndarray) -> pd.DataFrame:
    years = pd.DataFrame({"year": np.arange(int(np.min(panel_years)), int(np.max(panel_years)) + 1, dtype=int)})
    reg = registry.copy()
    reg["start_year"] = pd.to_numeric(reg["start_year"], errors="coerce")
    reg["policy_intensity"] = pd.to_numeric(reg["policy_intensity"], errors="coerce")
    reg = reg.dropna(subset=["start_year"]).copy()
    reg["start_year"] = reg["start_year"].astype(int)

    yearly = (
        reg.groupby("start_year", as_index=False)
        .agg(
            event_count=("iso3", "size"),
            country_coverage=("iso3", "nunique"),
            mean_intensity=("policy_intensity", "mean"),
            median_intensity=("policy_intensity", "median"),
        )
        .rename(columns={"start_year": "year"})
    )
    out = years.merge(yearly, on="year", how="left")
    for c in ["event_count", "country_coverage", "mean_intensity", "median_intensity"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out["z_event_count"] = _zscore(out["event_count"])
    out["z_country_coverage"] = _zscore(out["country_coverage"])
    out["z_mean_intensity"] = _zscore(out["mean_intensity"])
    out["shock_index_raw"] = 0.50 * out["z_event_count"] + 0.35 * out["z_country_coverage"] + 0.15 * out["z_mean_intensity"]
    out["shock_index"] = _minmax(out["shock_index_raw"])

    q75 = float(out["shock_index"].quantile(0.75))
    out["is_shock_year"] = (out["shock_index"] >= q75).astype(int)
    return out.sort_values("year").reset_index(drop=True)


def _build_city_baseline_exposure(panel: pd.DataFrame, pre_end_year: int) -> pd.DataFrame:
    df = panel.copy()
    for c in ["policy_dose_external_direct", "policy_intensity_external_direct"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")
    pre = df[df["year"] <= int(pre_end_year)].copy()
    if pre.empty:
        pre = df.copy()

    expo = (
        pre.groupby("city_id", as_index=False)
        .agg(
            baseline_dose=("policy_dose_external_direct", "mean"),
            baseline_intensity=("policy_intensity_external_direct", "mean"),
            city_name=("city_name", "first"),
            country=("country", "first"),
            iso3=("iso3", "first"),
            continent=("continent", "first"),
        )
    )
    expo["baseline_exposure"] = 0.6 * expo["baseline_dose"].fillna(0.0) + 0.4 * expo["baseline_intensity"].fillna(0.0)
    q30 = float(expo["baseline_exposure"].quantile(0.30))
    q70 = float(expo["baseline_exposure"].quantile(0.70))
    expo["exposure_group"] = np.where(
        expo["baseline_exposure"] >= q70,
        "high",
        np.where(expo["baseline_exposure"] <= q30, "low", "mid"),
    )
    return expo


def _event_group_response(
    panel: pd.DataFrame,
    exposure: pd.DataFrame,
    shock_years: List[int],
    event_window: tuple[int, int] = (-2, 2),
) -> pd.DataFrame:
    df = panel[["city_id", "year", "composite_index"]].copy()
    df["composite_index"] = pd.to_numeric(df["composite_index"], errors="coerce")
    exp = exposure[["city_id", "exposure_group"]].copy()
    merged = df.merge(exp, on="city_id", how="inner")

    lo, hi = int(event_window[0]), int(event_window[1])
    rows: List[dict[str, Any]] = []
    for shock_year in shock_years:
        sub = merged[(merged["year"] >= shock_year + lo) & (merged["year"] <= shock_year + hi)].copy()
        if sub.empty:
            continue

        base = merged[merged["year"] == shock_year - 1][["city_id", "composite_index"]].rename(columns={"composite_index": "base_index"})
        sub = sub.merge(base, on="city_id", how="left")
        sub["rel_index"] = sub["composite_index"] - sub["base_index"]
        sub["event_time"] = sub["year"] - shock_year

        grp = (
            sub[sub["exposure_group"].isin(["high", "low"])]
            .groupby(["event_time", "exposure_group"], as_index=False)
            .agg(mean_rel_index=("rel_index", "mean"), n_city=("city_id", "nunique"))
        )
        if grp.empty:
            continue
        piv = grp.pivot(index="event_time", columns="exposure_group", values="mean_rel_index").reset_index()
        n_piv = grp.pivot(index="event_time", columns="exposure_group", values="n_city").reset_index()
        piv = piv.merge(n_piv, on="event_time", how="left", suffixes=("", "_n"))
        for row in piv.itertuples(index=False):
            high = float(getattr(row, "high", np.nan))
            low = float(getattr(row, "low", np.nan))
            n_high = int(getattr(row, "high_n", 0)) if np.isfinite(getattr(row, "high_n", np.nan)) else 0
            n_low = int(getattr(row, "low_n", 0)) if np.isfinite(getattr(row, "low_n", np.nan)) else 0
            rows.append(
                {
                    "shock_year": int(shock_year),
                    "event_time": int(row.event_time),
                    "high_mean_rel_index": high,
                    "low_mean_rel_index": low,
                    "effect_high_minus_low": high - low if (np.isfinite(high) and np.isfinite(low)) else np.nan,
                    "n_high": n_high,
                    "n_low": n_low,
                }
            )
    return pd.DataFrame(rows)


def _first_stage_and_reduced_form(panel: pd.DataFrame, shock_index: pd.DataFrame, exposure: pd.DataFrame) -> Dict[str, Any]:
    df = panel.copy()
    df = df.merge(shock_index[["year", "shock_index", "is_shock_year"]], on="year", how="left")
    df = df.merge(exposure[["city_id", "baseline_exposure"]], on="city_id", how="left")

    for c in ["shock_index", "baseline_exposure", "did_treatment_external_direct", "composite_index"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["instrument_shiftshare"] = df["shock_index"].fillna(0.0) * df["baseline_exposure"].fillna(0.0)

    work = df.dropna(subset=["instrument_shiftshare", "did_treatment_external_direct", "composite_index"]).copy()
    if work.empty:
        return {"status": "failed", "reason": "empty_working_panel"}

    y1 = _two_way_demean(work, "did_treatment_external_direct").to_numpy(dtype=float)
    y2 = _two_way_demean(work, "composite_index").to_numpy(dtype=float)
    z = _two_way_demean(work, "instrument_shiftshare").to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(work), dtype=float), z])

    fs = _ols_hc1(y1, x)
    rf = _ols_hc1(y2, x)
    f_stat = float(fs.t_value * fs.t_value) if np.isfinite(fs.t_value) else float("nan")
    iv = float(rf.coef / fs.coef) if (np.isfinite(fs.coef) and abs(fs.coef) > 1e-10) else float("nan")

    return {
        "status": "ok",
        "n_obs": int(len(work)),
        "first_stage": {
            "coef": fs.coef,
            "stderr": fs.stderr,
            "t_value": fs.t_value,
            "p_value": fs.p_value,
            "f_stat_approx": f_stat,
        },
        "reduced_form": {
            "coef": rf.coef,
            "stderr": rf.stderr,
            "t_value": rf.t_value,
            "p_value": rf.p_value,
        },
        "iv_wald_ratio": iv,
    }


def _build_placebo_distribution(event_resp: pd.DataFrame, shock_years: List[int], panel_years: np.ndarray) -> pd.DataFrame:
    if event_resp.empty:
        return pd.DataFrame(columns=["placebo_year", "effect_t0_high_minus_low", "is_real_shock_year"])
    years = sorted(set(int(y) for y in panel_years))
    valid_years = [y for y in years if (y - 2 in years) and (y + 2 in years)]
    t0 = event_resp[event_resp["event_time"] == 0][["shock_year", "effect_high_minus_low"]].copy()

    # Existing shock-year effects.
    real_map = {int(r.shock_year): float(r.effect_high_minus_low) for r in t0.itertuples(index=False)}

    rows: List[dict[str, Any]] = []
    for y in valid_years:
        rows.append(
            {
                "placebo_year": int(y),
                "effect_t0_high_minus_low": real_map.get(int(y), np.nan),
                "is_real_shock_year": int(y in set(shock_years)),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["is_real_shock_year", "placebo_year"], ascending=[False, True]).reset_index(drop=True)


def _build_leave_one_shock_out(event_resp: pd.DataFrame, shock_years: List[int]) -> pd.DataFrame:
    if event_resp.empty:
        return pd.DataFrame(columns=["excluded_shock_year", "post_mean_effect", "post_event_rows"])
    out_rows: List[dict[str, Any]] = []
    for sy in shock_years:
        sub = event_resp[event_resp["shock_year"] != sy].copy()
        post = sub[sub["event_time"] >= 0].copy()
        out_rows.append(
            {
                "excluded_shock_year": int(sy),
                "post_mean_effect": float(np.nanmean(post["effect_high_minus_low"].to_numpy(dtype=float))) if not post.empty else np.nan,
                "post_event_rows": int(len(post)),
            }
        )
    return pd.DataFrame(out_rows).sort_values("excluded_shock_year").reset_index(drop=True)


def run_exogenous_shock_suite(panel: pd.DataFrame, registry_path: Path | None = None) -> Dict[str, Any]:
    """Run event-level exogenous-shock diagnostics and persist artifacts."""
    registry_path = (DATA_RAW / "policy_events_registry.csv") if registry_path is None else Path(registry_path)
    if not registry_path.exists():
        summary = {"status": "failed", "reason": "policy_events_registry_missing"}
        dump_json(DATA_OUTPUTS / "exoshock_summary.json", summary)
        return summary

    registry = pd.read_csv(registry_path)
    shock_index = _build_shock_year_index(registry, panel["year"].to_numpy(dtype=int))
    shock_years = shock_index.loc[shock_index["is_shock_year"] == 1, "year"].astype(int).tolist()
    if not shock_years:
        shock_years = shock_index.sort_values("shock_index", ascending=False).head(3)["year"].astype(int).tolist()

    pre_end = min(shock_years) - 1 if shock_years else int(panel["year"].min()) + 3
    exposure = _build_city_baseline_exposure(panel, pre_end_year=pre_end)
    event_resp = _event_group_response(panel, exposure, shock_years, event_window=(-2, 2))
    fs_rf = _first_stage_and_reduced_form(panel, shock_index, exposure)
    placebo = _build_placebo_distribution(event_resp, shock_years, panel["year"].to_numpy(dtype=int))
    leave_one = _build_leave_one_shock_out(event_resp, shock_years)

    # Persist artifacts
    shock_index_path = DATA_OUTPUTS / "exoshock_year_index.csv"
    exposure_path = DATA_OUTPUTS / "exoshock_city_baseline_exposure.csv"
    event_resp_path = DATA_OUTPUTS / "exoshock_event_response.csv"
    placebo_path = DATA_OUTPUTS / "exoshock_placebo_year_distribution.csv"
    leave_one_path = DATA_OUTPUTS / "exoshock_leave_one_shock_out.csv"

    shock_index.to_csv(shock_index_path, index=False)
    exposure.to_csv(exposure_path, index=False)
    event_resp.to_csv(event_resp_path, index=False)
    placebo.to_csv(placebo_path, index=False)
    leave_one.to_csv(leave_one_path, index=False)

    summary = {
        "status": "ok",
        "registry_rows": int(len(registry)),
        "shock_years": [int(y) for y in shock_years],
        "shock_year_count": int(len(shock_years)),
        "pre_period_end_year": int(pre_end),
        "first_stage_reduced_form": fs_rf,
        "event_response_rows": int(len(event_resp)),
        "placebo_rows": int(len(placebo)),
        "leave_one_rows": int(len(leave_one)),
        "artifacts": {
            "shock_index_csv": str(shock_index_path),
            "baseline_exposure_csv": str(exposure_path),
            "event_response_csv": str(event_resp_path),
            "placebo_distribution_csv": str(placebo_path),
            "leave_one_shock_csv": str(leave_one_path),
        },
    }
    dump_json(DATA_OUTPUTS / "exoshock_summary.json", summary)
    return summary
