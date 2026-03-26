from __future__ import annotations

"""Heterogeneity and mechanism diagnostics for exogenous-shock design."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .utils import DATA_OUTPUTS, DATA_RAW, dump_json


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


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


def _build_baseline_exposure(panel: pd.DataFrame, pre_end_year: int) -> pd.DataFrame:
    df = panel.copy()
    for c in ["policy_dose_external_direct", "policy_intensity_external_direct"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")
    pre = df[df["year"] <= int(pre_end_year)].copy()
    if pre.empty:
        pre = df.copy()

    out = (
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
    out["baseline_exposure"] = 0.6 * out["baseline_dose"].fillna(0.0) + 0.4 * out["baseline_intensity"].fillna(0.0)
    q30 = float(out["baseline_exposure"].quantile(0.30))
    q70 = float(out["baseline_exposure"].quantile(0.70))
    out["exposure_group"] = np.where(
        out["baseline_exposure"] >= q70,
        "high",
        np.where(out["baseline_exposure"] <= q30, "low", "mid"),
    )
    return out


def _policy_type_year_index(registry: pd.DataFrame, panel_years: np.ndarray) -> pd.DataFrame:
    reg = registry.copy()
    reg["start_year"] = pd.to_numeric(reg["start_year"], errors="coerce")
    reg["policy_intensity"] = pd.to_numeric(reg["policy_intensity"], errors="coerce")
    reg = reg.dropna(subset=["start_year"]).copy()
    reg["start_year"] = reg["start_year"].astype(int)
    reg["policy_type"] = reg["policy_name"].astype(str).str.split("_").str[0].str.lower()
    reg = reg[reg["policy_type"].str.len() > 0].copy()

    years = np.arange(int(np.min(panel_years)), int(np.max(panel_years)) + 1, dtype=int)
    types = sorted(reg["policy_type"].dropna().astype(str).unique().tolist())
    grid = pd.MultiIndex.from_product([types, years], names=["policy_type", "year"]).to_frame(index=False)

    agg = (
        reg.groupby(["policy_type", "start_year"], as_index=False)
        .agg(
            event_count=("iso3", "size"),
            country_coverage=("iso3", "nunique"),
            mean_intensity=("policy_intensity", "mean"),
        )
        .rename(columns={"start_year": "year"})
    )
    out = grid.merge(agg, on=["policy_type", "year"], how="left")
    for c in ["event_count", "country_coverage", "mean_intensity"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    rows: List[pd.DataFrame] = []
    for ptype, sub in out.groupby("policy_type", as_index=False):
        s = sub.copy()
        s["z_event_count"] = _zscore(s["event_count"])
        s["z_country_coverage"] = _zscore(s["country_coverage"])
        s["z_mean_intensity"] = _zscore(s["mean_intensity"])
        s["shock_index_raw"] = 0.50 * s["z_event_count"] + 0.35 * s["z_country_coverage"] + 0.15 * s["z_mean_intensity"]
        s["shock_index_type"] = _minmax(s["shock_index_raw"])
        q75 = float(s["shock_index_type"].quantile(0.75))
        s["is_shock_year_type"] = (s["shock_index_type"] >= q75).astype(int)
        rows.append(s)
    out2 = pd.concat(rows, ignore_index=True) if rows else out
    return out2.sort_values(["policy_type", "year"]).reset_index(drop=True)


def _event_effect(
    panel: pd.DataFrame,
    exposure: pd.DataFrame,
    outcome_col: str,
    shock_years: List[int],
    event_window: tuple[int, int] = (-2, 2),
    group_col: str | None = None,
) -> pd.DataFrame:
    df_cols = ["city_id", "year", outcome_col] + ([group_col] if group_col else [])
    df = panel[df_cols].copy()
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")
    df = df.dropna(subset=[outcome_col])
    exp_cols = ["city_id", "exposure_group"]
    if group_col and (group_col not in df.columns) and (group_col in exposure.columns):
        exp_cols.append(group_col)
    exp = exposure[exp_cols].drop_duplicates(subset=["city_id"]).copy()
    merged = df.merge(exp, on="city_id", how="inner")
    if group_col and group_col not in merged.columns:
        raise RuntimeError(f"group_col not found in panel: {group_col}")

    lo, hi = int(event_window[0]), int(event_window[1])
    rows: List[dict[str, Any]] = []
    for shock_year in shock_years:
        sub = merged[(merged["year"] >= shock_year + lo) & (merged["year"] <= shock_year + hi)].copy()
        if sub.empty:
            continue
        base = merged[merged["year"] == shock_year - 1][["city_id", outcome_col]].rename(columns={outcome_col: "base_outcome"})
        sub = sub.merge(base, on="city_id", how="left")
        sub["rel_outcome"] = sub[outcome_col] - sub["base_outcome"]
        sub["event_time"] = sub["year"] - shock_year
        sub = sub[sub["exposure_group"].isin(["high", "low"])].copy()
        if sub.empty:
            continue

        group_vals = [("__all__", sub)] if group_col is None else [(str(g), gdf) for g, gdf in sub.groupby(group_col, dropna=False)]
        for gname, gdf in group_vals:
            agg = (
                gdf.groupby(["event_time", "exposure_group"], as_index=False)
                .agg(mean_rel_outcome=("rel_outcome", "mean"), n_city=("city_id", "nunique"))
            )
            if agg.empty:
                continue
            piv = agg.pivot(index="event_time", columns="exposure_group", values="mean_rel_outcome").reset_index()
            n_piv = agg.pivot(index="event_time", columns="exposure_group", values="n_city").reset_index()
            piv = piv.merge(n_piv, on="event_time", how="left", suffixes=("", "_n"))
            for row in piv.itertuples(index=False):
                high = float(getattr(row, "high", np.nan))
                low = float(getattr(row, "low", np.nan))
                n_high = int(getattr(row, "high_n", 0)) if np.isfinite(getattr(row, "high_n", np.nan)) else 0
                n_low = int(getattr(row, "low_n", 0)) if np.isfinite(getattr(row, "low_n", np.nan)) else 0
                rows.append(
                    {
                        "group": gname,
                        "shock_year": int(shock_year),
                        "event_time": int(row.event_time),
                        "high_mean_rel_outcome": high,
                        "low_mean_rel_outcome": low,
                        "effect_high_minus_low": high - low if (np.isfinite(high) and np.isfinite(low)) else np.nan,
                        "n_high": n_high,
                        "n_low": n_low,
                    }
                )
    return pd.DataFrame(rows)


def run_exogenous_shock_heterogeneity_suite(
    panel: pd.DataFrame,
    registry_path: Path | None = None,
) -> Dict[str, Any]:
    """Build exogenous-shock heterogeneity and mechanism artifacts."""
    registry_path = (DATA_RAW / "policy_events_registry.csv") if registry_path is None else Path(registry_path)
    registry = _safe_read_csv(registry_path)
    if registry.empty:
        summary = {"status": "failed", "reason": "policy_registry_missing_or_empty"}
        dump_json(DATA_OUTPUTS / "exoshock_heterogeneity_summary.json", summary)
        return summary

    type_year = _policy_type_year_index(registry, panel["year"].to_numpy(dtype=int))
    # Global shock years from existing exoshock index if available, fallback to top years by total shock index.
    global_index = _safe_read_csv(DATA_OUTPUTS / "exoshock_year_index.csv")
    if not global_index.empty and {"year", "is_shock_year", "shock_index"}.issubset(global_index.columns):
        global_shock_years = (
            global_index.loc[pd.to_numeric(global_index["is_shock_year"], errors="coerce").fillna(0.0) > 0.5, "year"]
            .dropna()
            .astype(int)
            .tolist()
        )
        if not global_shock_years:
            global_shock_years = (
                global_index.assign(shock_index=pd.to_numeric(global_index["shock_index"], errors="coerce"))
                .sort_values("shock_index", ascending=False)["year"]
                .dropna()
                .astype(int)
                .head(3)
                .tolist()
            )
    else:
        tmp = (
            type_year.groupby("year", as_index=False)["shock_index_type"]
            .mean()
            .sort_values("shock_index_type", ascending=False)
        )
        global_shock_years = tmp["year"].astype(int).head(3).tolist()

    pre_end = min(global_shock_years) - 1 if global_shock_years else int(panel["year"].min()) + 3
    exposure = _build_baseline_exposure(panel, pre_end_year=pre_end)

    # 1) Policy-type-specific responses.
    rows_policy: List[pd.DataFrame] = []
    for ptype, sub in type_year.groupby("policy_type", as_index=False):
        sy = sub.loc[sub["is_shock_year_type"] > 0, "year"].astype(int).tolist()
        if not sy:
            sy = sub.sort_values("shock_index_type", ascending=False)["year"].astype(int).head(2).tolist()
        resp = _event_effect(panel, exposure, outcome_col="composite_index", shock_years=sy, group_col=None)
        if resp.empty:
            continue
        resp["policy_type"] = str(ptype)
        rows_policy.append(resp)
    policy_resp = pd.concat(rows_policy, ignore_index=True) if rows_policy else pd.DataFrame(
        columns=[
            "policy_type",
            "group",
            "shock_year",
            "event_time",
            "high_mean_rel_outcome",
            "low_mean_rel_outcome",
            "effect_high_minus_low",
            "n_high",
            "n_low",
        ]
    )

    # 2) Continent heterogeneity on global shock years.
    cont_resp = _event_effect(
        panel,
        exposure,
        outcome_col="composite_index",
        shock_years=global_shock_years,
        group_col="continent",
    )
    if not cont_resp.empty:
        cont_resp = cont_resp.rename(columns={"group": "continent"})

    # 3) Mechanism channel decomposition.
    channels = ["composite_index", "economic_vitality", "livability", "innovation"]
    channel_rows: List[pd.DataFrame] = []
    channel_summary_rows: List[dict[str, Any]] = []
    for ch in channels:
        if ch not in panel.columns:
            continue
        resp = _event_effect(panel, exposure, outcome_col=ch, shock_years=global_shock_years, group_col=None)
        if resp.empty:
            continue
        resp["channel"] = ch
        channel_rows.append(resp)
        agg = resp.groupby("event_time", as_index=False).agg(effect_mean=("effect_high_minus_low", "mean"), effect_std=("effect_high_minus_low", "std"), n=("effect_high_minus_low", "size"))
        pre_mean = float(np.nanmean(agg.loc[agg["event_time"] < 0, "effect_mean"].to_numpy(dtype=float))) if (agg["event_time"] < 0).any() else np.nan
        post_mean = float(np.nanmean(agg.loc[agg["event_time"] >= 0, "effect_mean"].to_numpy(dtype=float))) if (agg["event_time"] >= 0).any() else np.nan
        t0 = float(agg.loc[agg["event_time"] == 0, "effect_mean"].iloc[0]) if (agg["event_time"] == 0).any() else np.nan
        t1 = float(agg.loc[agg["event_time"] == 1, "effect_mean"].iloc[0]) if (agg["event_time"] == 1).any() else np.nan
        channel_summary_rows.append(
            {
                "channel": ch,
                "pre_mean_effect": pre_mean,
                "post_mean_effect": post_mean,
                "delta_post_minus_pre": post_mean - pre_mean if (np.isfinite(post_mean) and np.isfinite(pre_mean)) else np.nan,
                "t0_effect": t0,
                "t1_effect": t1,
            }
        )
    channel_resp = pd.concat(channel_rows, ignore_index=True) if channel_rows else pd.DataFrame()
    channel_summary = pd.DataFrame(channel_summary_rows)

    # Persist artifacts.
    type_year_path = DATA_OUTPUTS / "exoshock_policy_type_year_index.csv"
    policy_resp_path = DATA_OUTPUTS / "exoshock_policy_type_event_response.csv"
    cont_resp_path = DATA_OUTPUTS / "exoshock_continent_event_response.csv"
    channel_resp_path = DATA_OUTPUTS / "exoshock_channel_decomposition.csv"
    channel_sum_path = DATA_OUTPUTS / "exoshock_channel_summary.csv"

    type_year.to_csv(type_year_path, index=False)
    policy_resp.to_csv(policy_resp_path, index=False)
    cont_resp.to_csv(cont_resp_path, index=False)
    channel_resp.to_csv(channel_resp_path, index=False)
    channel_summary.to_csv(channel_sum_path, index=False)

    top_policy_type = ""
    if not policy_resp.empty:
        tmp = (
            policy_resp[policy_resp["event_time"] >= 0]
            .groupby("policy_type", as_index=False)["effect_high_minus_low"]
            .mean()
            .sort_values("effect_high_minus_low", ascending=False)
        )
        if not tmp.empty:
            top_policy_type = str(tmp.iloc[0]["policy_type"])

    summary = {
        "status": "ok",
        "global_shock_years": [int(y) for y in global_shock_years],
        "policy_type_count": int(type_year["policy_type"].nunique()) if not type_year.empty else 0,
        "policy_type_event_rows": int(len(policy_resp)),
        "continent_event_rows": int(len(cont_resp)),
        "channel_rows": int(len(channel_resp)),
        "top_policy_type_post_effect": top_policy_type,
        "artifacts": {
            "policy_type_year_index_csv": str(type_year_path),
            "policy_type_event_response_csv": str(policy_resp_path),
            "continent_event_response_csv": str(cont_resp_path),
            "channel_decomposition_csv": str(channel_resp_path),
            "channel_summary_csv": str(channel_sum_path),
        },
    }
    dump_json(DATA_OUTPUTS / "exoshock_heterogeneity_summary.json", summary)
    return summary
