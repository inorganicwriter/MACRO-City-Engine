from __future__ import annotations

"""External validity checks with independent World Bank outcomes."""

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
from requests import RequestException, Response
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .utils import DATA_OUTPUTS, DATA_RAW, dump_json

LOGGER = logging.getLogger(__name__)

EXTERNAL_INDICATORS: Dict[str, str] = {
    "patent_residents": "IP.PAT.RESD",
    "researchers_per_million": "SP.POP.SCIE.RD.P6",
    "high_tech_exports_share": "TX.VAL.TECH.MF.ZS",
    "employment_rate": "SL.EMP.TOTL.SP.ZS",
    "urban_population_share": "SP.URB.TOTL.IN.ZS",
    "electricity_access": "EG.ELC.ACCS.ZS",
    "fixed_broadband_subscriptions": "IT.NET.BBND.P2",
    "pm25_exposure": "EN.ATM.PM25.MC.M3",
}

DERIVED_INDICATORS: List[str] = ["electricity_broadband_proxy"]


@dataclass(frozen=True)
class OLSResult:
    coef: np.ndarray
    stderr: np.ndarray
    t_value: np.ndarray
    variable_names: List[str]
    r2: float
    n_obs: int


def _request_json(
    url: str,
    *,
    timeout: int = 45,
    retries: int = 3,
    backoff_seconds: float = 1.0,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp: Response = requests.get(url, timeout=timeout)
            if resp.status_code in {429, 500, 502, 503, 504}:
                resp.raise_for_status()
            resp.raise_for_status()
            return resp.json()
        except (RequestException, ValueError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    msg = f"World Bank request failed: {url} ({last_error})"
    raise RuntimeError(msg)


def _world_bank_url(indicator_code: str, start_year: int, end_year: int) -> str:
    return (
        "https://api.worldbank.org/v2/country/all/indicator/"
        f"{indicator_code}?format=json&date={start_year}:{end_year}&per_page=20000"
    )


def _fetch_indicator(indicator_code: str, start_year: int, end_year: int) -> pd.DataFrame:
    payload = _request_json(_world_bank_url(indicator_code, start_year, end_year), timeout=60, retries=4)
    if not isinstance(payload, list) or len(payload) < 2:
        msg = f"Invalid World Bank payload for indicator {indicator_code}"
        raise RuntimeError(msg)

    rows: List[dict] = []
    for item in payload[1]:
        iso3 = item.get("countryiso3code")
        year_raw = item.get("date")
        value = item.get("value")
        if not iso3 or value is None:
            continue
        try:
            rows.append({"iso3": str(iso3).upper(), "year": int(year_raw), "value": float(value)})
        except (TypeError, ValueError):
            continue
    return pd.DataFrame(rows)


def _is_external_cache_valid(df: pd.DataFrame, iso3_set: set[str], years: set[int]) -> bool:
    needed = {"iso3", "year", *EXTERNAL_INDICATORS.keys(), "external_source"}
    if not needed.issubset(df.columns):
        return False
    return iso3_set.issubset(set(df["iso3"].astype(str))) and years.issubset(set(df["year"].astype(int)))


def _add_derived_external_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"electricity_access", "fixed_broadband_subscriptions"}.issubset(out.columns):
        ele = pd.to_numeric(out["electricity_access"], errors="coerce")
        bbd = pd.to_numeric(out["fixed_broadband_subscriptions"], errors="coerce")
        ele_min, ele_max = float(ele.min()), float(ele.max())
        bbd_min, bbd_max = float(bbd.min()), float(bbd.max())
        ele_n = (ele - ele_min) / max(ele_max - ele_min, 1e-8)
        bbd_n = (bbd - bbd_min) / max(bbd_max - bbd_min, 1e-8)
        out["electricity_broadband_proxy"] = 100.0 * (0.70 * ele_n.fillna(0.0) + 0.30 * bbd_n.fillna(0.0))
    else:
        out["electricity_broadband_proxy"] = np.nan
    return out


def _load_from_wb_extra_panel(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    extra_path = DATA_RAW / "wb_extra_panel.csv"
    if not extra_path.exists():
        return pd.DataFrame()

    try:
        extra = pd.read_csv(extra_path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    needed = {"iso3", "year", *EXTERNAL_INDICATORS.keys()}
    if not needed.issubset(set(extra.columns)):
        return pd.DataFrame()

    start_year = int(panel["year"].min())
    end_year = int(panel["year"].max())
    iso3_values = sorted(panel["iso3"].astype(str).str.upper().unique().tolist())
    years = list(range(start_year, end_year + 1))
    full = pd.MultiIndex.from_product([iso3_values, years], names=["iso3", "year"]).to_frame(index=False)

    for col in ["iso3", "external_source"]:
        if col in extra.columns:
            extra[col] = extra[col].astype(str)
    if "external_source" not in extra.columns:
        if "extra_wb_source" in extra.columns:
            extra["external_source"] = extra["extra_wb_source"].astype(str)
        else:
            extra["external_source"] = "world_bank"

    keep_cols = ["iso3", "year", "external_source", *EXTERNAL_INDICATORS.keys()]
    merged = full.merge(extra[keep_cols], on=["iso3", "year"], how="left")
    merged["external_source"] = merged["external_source"].fillna("world_bank")

    for col in EXTERNAL_INDICATORS.keys():
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged[col] = merged.groupby("iso3")[col].transform(lambda x: x.interpolate(limit_direction="both"))
        med = float(merged[col].median()) if not pd.isna(merged[col].median()) else 0.0
        merged[col] = merged[col].fillna(med)

    merged = _add_derived_external_indicators(merged)
    merged["external_panel_origin"] = "wb_extra_panel_cache"
    return merged


def _load_external_panel(
    panel: pd.DataFrame,
    use_cache: bool = True,
) -> pd.DataFrame:
    wb_extra = _load_from_wb_extra_panel(panel)
    if not wb_extra.empty:
        return wb_extra

    cache_path = DATA_RAW / "wb_external_validity_panel.csv"
    start_year = int(panel["year"].min())
    end_year = int(panel["year"].max())
    iso3_values = sorted(panel["iso3"].astype(str).str.upper().unique().tolist())
    iso3_set = set(iso3_values)
    years = set(range(start_year, end_year + 1))

    if use_cache and cache_path.exists():
        cached = pd.read_csv(cache_path)
        if _is_external_cache_valid(cached, iso3_set, years):
            return cached[(cached["iso3"].isin(iso3_values)) & (cached["year"].between(start_year, end_year))].copy()

    frames: List[pd.DataFrame] = []
    for name, code in EXTERNAL_INDICATORS.items():
        try:
            ind = _fetch_indicator(code, start_year=start_year, end_year=end_year)
            ind = ind.rename(columns={"value": name})
            frames.append(ind)
            ind.to_csv(DATA_RAW / f"wb_external_{name}.csv", index=False)
            LOGGER.info("External indicator loaded: %s (%s rows)", name, len(ind))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("External indicator fetch skipped for %s: %s", name, exc)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["iso3", "year"], how="outer")

    full = pd.MultiIndex.from_product([iso3_values, list(range(start_year, end_year + 1))], names=["iso3", "year"]).to_frame(
        index=False
    )
    merged = full.merge(merged, on=["iso3", "year"], how="left")
    merged["external_source"] = "world_bank"

    for col in EXTERNAL_INDICATORS.keys():
        if col not in merged.columns:
            merged[col] = np.nan
        merged[col] = merged.groupby("iso3")[col].transform(lambda x: x.interpolate(limit_direction="both"))
        med = float(merged[col].median()) if not pd.isna(merged[col].median()) else 0.0
        merged[col] = merged[col].fillna(med)

    merged = _add_derived_external_indicators(merged)
    merged["external_panel_origin"] = "world_bank_api_fetch"
    merged.to_csv(cache_path, index=False)
    return merged


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _two_way_within(df: pd.DataFrame, cols: List[str], unit_col: str, time_col: str) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        unit_mean = out.groupby(unit_col)[col].transform("mean")
        time_mean = out.groupby(time_col)[col].transform("mean")
        grand_mean = float(out[col].mean())
        out[f"{col}_tw"] = out[col] - unit_mean - time_mean + grand_mean
    return out


def _ols_hc1(y: np.ndarray, x: np.ndarray, names: List[str]) -> OLSResult:
    xtx = x.T @ x
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    n, k = x.shape
    meat = np.zeros((k, k))
    for i in range(n):
        xi = x[i : i + 1, :]
        ui2 = float(resid[i] ** 2)
        meat += ui2 * (xi.T @ xi)
    scale = n / max(1, n - k)
    cov = scale * (xtx_inv @ meat @ xtx_inv)
    stderr = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    tval = beta / stderr
    ssr = float((resid**2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - (ssr / sst if sst > 0 else 0.0)
    return OLSResult(coef=beta, stderr=stderr, t_value=tval, variable_names=names, r2=r2, n_obs=n)


def _predictive_uplift(
    df: pd.DataFrame,
    outcome_col: str,
    controls: List[str],
) -> Dict[str, float | int]:
    years = sorted(df["year"].astype(int).unique().tolist())
    if len(years) < 4:
        return {"status": "skipped", "reason": "insufficient_years"}  # type: ignore[return-value]

    split_year = years[-3] if len(years) >= 5 else years[-2]
    train = df[df["year"] <= split_year].copy()
    test = df[df["year"] > split_year].copy()
    if len(train) < 80 or len(test) < 40:
        return {"status": "skipped", "reason": "insufficient_rows"}  # type: ignore[return-value]

    x_base_train = train[controls].to_numpy(dtype=float)
    x_base_test = test[controls].to_numpy(dtype=float)
    x_plus_train = train[controls + ["composite_index"]].to_numpy(dtype=float)
    x_plus_test = test[controls + ["composite_index"]].to_numpy(dtype=float)
    y_train = train[outcome_col].to_numpy(dtype=float)
    y_test = test[outcome_col].to_numpy(dtype=float)

    m_base = LinearRegression()
    m_plus = LinearRegression()
    m_base.fit(x_base_train, y_train)
    m_plus.fit(x_plus_train, y_train)

    p_base = m_base.predict(x_base_test)
    p_plus = m_plus.predict(x_plus_test)
    r2_base = float(r2_score(y_test, p_base))
    r2_plus = float(r2_score(y_test, p_plus))
    return {
        "status": "ok",
        "split_year": int(split_year),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "r2_base": r2_base,
        "r2_plus_composite": r2_plus,
        "r2_uplift": float(r2_plus - r2_base),
    }


def run_external_validity_suite(panel: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate whether MACRO-City Engine signals generalize to independent external outcomes."""
    required = {"iso3", "year", "composite_index", "log_gdp_pc", "internet_users", "unemployment"}
    if not required.issubset(panel.columns):
        miss = sorted(required.difference(set(panel.columns)))
        out = {"status": "skipped", "reason": f"missing_columns_{','.join(miss)}"}
        dump_json(DATA_OUTPUTS / "external_validity_summary.json", out)
        return out

    external = _load_external_panel(panel)
    if external.empty:
        out = {"status": "skipped", "reason": "external_indicators_unavailable"}
        dump_json(DATA_OUTPUTS / "external_validity_summary.json", out)
        return out

    country_panel = (
        panel.groupby(["iso3", "year"], as_index=False)
        .agg(
            composite_index=("composite_index", "mean"),
            log_gdp_pc=("log_gdp_pc", "mean"),
            internet_users=("internet_users", "mean"),
            unemployment=("unemployment", "mean"),
            city_count=("city_id", "nunique"),
        )
        .copy()
    )

    merged = country_panel.merge(external, on=["iso3", "year"], how="inner")
    indicator_results: List[Dict[str, Any]] = []
    rank_rows: List[Dict[str, Any]] = []
    indicator_list = list(EXTERNAL_INDICATORS.keys()) + list(DERIVED_INDICATORS)

    for indicator in indicator_list:
        if indicator not in merged.columns:
            continue

        tmp = merged[["iso3", "year", "composite_index", "log_gdp_pc", "internet_users", "unemployment", indicator]].copy()
        tmp = tmp.sort_values(["iso3", "year"])
        tmp["outcome_t1"] = tmp.groupby("iso3")[indicator].shift(-1)
        tmp = tmp.dropna(subset=["outcome_t1"]).copy()
        if len(tmp) < 120:
            continue

        cols = ["outcome_t1", "composite_index", "log_gdp_pc", "internet_users", "unemployment"]
        tw = _two_way_within(tmp, cols, unit_col="iso3", time_col="year")
        y = tw["outcome_t1_tw"].to_numpy(dtype=float)
        x_cols = ["composite_index_tw", "log_gdp_pc_tw", "internet_users_tw", "unemployment_tw"]
        x = tw[x_cols].to_numpy(dtype=float)
        names = ["composite_index", "log_gdp_pc", "internet_users", "unemployment"]
        ols = _ols_hc1(y, x, names)
        idx = names.index("composite_index")
        coef = float(ols.coef[idx])
        se = float(ols.stderr[idx])
        t_val = float(ols.t_value[idx])
        p_val = float(2.0 * (1.0 - _normal_cdf(abs(t_val))))

        uplift = _predictive_uplift(tmp, outcome_col="outcome_t1", controls=["log_gdp_pc", "internet_users", "unemployment"])
        rho_years: List[float] = []
        for year, sub in tmp.groupby("year"):
            if len(sub) < 12:
                continue
            rho = sub["composite_index"].rank(method="average").corr(sub[indicator].rank(method="average"))
            if pd.isna(rho):
                continue
            rho_years.append(float(rho))
            rank_rows.append({"indicator": indicator, "year": int(year), "spearman_rank_corr": float(rho), "n_obs": int(len(sub))})

        indicator_results.append(
            {
                "indicator": indicator,
                "n_obs": int(ols.n_obs),
                "twfe_coef_composite": coef,
                "twfe_stderr_composite": se,
                "twfe_t_value_composite": t_val,
                "twfe_p_value_composite": p_val,
                "twfe_r2_within": float(ols.r2),
                "mean_spearman_rank_corr": float(np.mean(rho_years)) if rho_years else None,
                "predictive_uplift": uplift,
            }
        )

    reg_df = pd.DataFrame(indicator_results)
    rank_df = pd.DataFrame(rank_rows)
    reg_df.to_csv(DATA_OUTPUTS / "external_validity_indicator_results.csv", index=False)
    rank_df.to_csv(DATA_OUTPUTS / "external_validity_rank_corr_by_year.csv", index=False)

    if reg_df.empty:
        out = {"status": "skipped", "reason": "no_valid_external_indicator_estimates"}
        dump_json(DATA_OUTPUTS / "external_validity_summary.json", out)
        return out

    uplift_series = reg_df["predictive_uplift"].apply(
        lambda x: float(x.get("r2_uplift")) if isinstance(x, dict) and x.get("status") == "ok" else np.nan
    )
    out = {
        "status": "ok",
        "indicators_evaluated": int(reg_df["indicator"].nunique()),
        "external_panel_origin": str(merged.get("external_panel_origin", pd.Series(["unknown"])).astype(str).iloc[0]),
        "avg_twfe_abs_t": float(np.mean(np.abs(reg_df["twfe_t_value_composite"].to_numpy(dtype=float)))),
        "avg_twfe_p_value": float(np.mean(reg_df["twfe_p_value_composite"].to_numpy(dtype=float))),
        "avg_predictive_r2_uplift": float(np.nanmean(uplift_series.to_numpy(dtype=float)))
        if uplift_series.notna().any()
        else None,
        "indicator_results": indicator_results,
    }
    dump_json(DATA_OUTPUTS / "external_validity_summary.json", out)
    LOGGER.info(
        "External validity suite done: indicators=%s, avg|t|=%.3f",
        out["indicators_evaluated"],
        out["avg_twfe_abs_t"],
    )
    return out
