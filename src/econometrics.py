from __future__ import annotations

"""Econometric analysis suite for global urban development panel."""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold

from .feature_backfill import add_no2_backcast_features
from .utils import DATA_OUTPUTS, dump_json, haversine_km

LOGGER = logging.getLogger(__name__)

CAUSAL_OUTCOME_FORBIDDEN = {
    "composite_index",
    "composite_index_pca",
    "composite_index_weighted",
    "economic_vitality",
    "livability",
    "innovation",
}

CAUSAL_SPARSE_CONTROL_WHITELIST = [
    "temperature_mean",
    "precipitation_sum",
    "baseline_population_log",
]

CAUSAL_PRIMARY_OUTCOME_CANDIDATES = [
    "log_viirs_ntl",
    "viirs_log_mean",
    "physical_built_expansion_primary",
    "ghsl_built_surface_km2",
    "knowledge_capital_raw",
    "no2_trop_anomaly_mean",
    "no2_trop_mean",
    "cobb_douglas_tfp_residual",
]


@dataclass(frozen=True)
class OLSResult:
    coef: np.ndarray
    stderr: np.ndarray
    t_value: np.ndarray
    variable_names: List[str]
    r2: float
    n_obs: int


def _assert_valid_causal_outcome(outcome: str) -> None:
    outcome_l = str(outcome).strip().lower()
    if (outcome_l in CAUSAL_OUTCOME_FORBIDDEN) or outcome_l.endswith("_n") or outcome_l.endswith("_norm_global"):
        msg = (
            f"Outcome {outcome!r} is blocked for causal econometrics. "
            "Use a raw observed indicator such as log_viirs_ntl or knowledge_capital_raw."
        )
        raise ValueError(msg)


def _resolve_sparse_controls(panel: pd.DataFrame, controls: List[str] | None = None) -> List[str]:
    requested = list(controls or [])
    available = [col for col in CAUSAL_SPARSE_CONTROL_WHITELIST if col in panel.columns]
    if requested:
        requested_l = {str(col) for col in requested}
        return [col for col in available if col in requested_l or col in CAUSAL_SPARSE_CONTROL_WHITELIST]
    return available


def _resolve_primary_causal_outcome(panel: pd.DataFrame) -> str:
    for col in CAUSAL_PRIMARY_OUTCOME_CANDIDATES:
        if col not in panel.columns:
            continue
        values = pd.to_numeric(panel[col], errors="coerce")
        if values.notna().sum() >= 40:
            return str(col)
    msg = (
        "No admissible causal outcome found. Expected one of: "
        + ", ".join(CAUSAL_PRIMARY_OUTCOME_CANDIDATES)
    )
    raise ValueError(msg)


def _ols_fit(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve an OLS normal equation and return coefficients, fitted values, residuals."""
    if x.ndim != 2:
        msg = "Design matrix must be 2D."
        raise ValueError(msg)
    if x.shape[1] == 0:
        beta = np.zeros(0, dtype=float)
        fitted = np.zeros_like(y, dtype=float)
        resid = y.astype(float) - fitted
        return beta, fitted, resid

    xtx = x.T @ x
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (x.T @ y)
    fitted = x @ beta
    resid = y - fitted
    return beta, fitted, resid


def _ols_hc1(y: np.ndarray, x: np.ndarray, var_names: List[str]) -> OLSResult:
    """OLS with HC1 robust standard errors."""
    xtx = x.T @ x
    xtx_inv = np.linalg.pinv(xtx)
    beta, fitted, resid = _ols_fit(y, x)

    n, k = x.shape
    meat = np.zeros((k, k))
    for i in range(n):
        xi = x[i : i + 1, :]
        ui2 = float(resid[i] ** 2)
        meat += ui2 * (xi.T @ xi)

    scale = n / max(1, n - k)
    cov = scale * (xtx_inv @ meat @ xtx_inv)
    stderr = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    t_value = beta / stderr

    ssr = float((resid**2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - (ssr / sst if sst > 0 else 0.0)

    return OLSResult(
        coef=beta,
        stderr=stderr,
        t_value=t_value,
        variable_names=var_names,
        r2=r2,
        n_obs=n,
    )


def _ols_cluster_hc1(
    y: np.ndarray,
    x: np.ndarray,
    var_names: List[str],
    clusters: np.ndarray,
) -> tuple[OLSResult, int]:
    """OLS with one-way cluster-robust (CR1) standard errors."""
    xtx = x.T @ x
    xtx_inv = np.linalg.pinv(xtx)
    beta, fitted, resid = _ols_fit(y, x)

    n, k = x.shape
    cluster_codes = pd.Series(clusters.astype(str)).fillna("missing_cluster").to_numpy()
    uniq = np.unique(cluster_codes)
    g = int(len(uniq))

    if g < 2:
        res = _ols_hc1(y, x, var_names)
        return res, g

    meat = np.zeros((k, k))
    for cid in uniq:
        idx = cluster_codes == cid
        xg = x[idx, :]
        ug = resid[idx]
        xgu = xg.T @ ug
        meat += np.outer(xgu, xgu)

    dof_adj = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov = dof_adj * (xtx_inv @ meat @ xtx_inv)
    stderr = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    t_value = beta / stderr

    ssr = float((resid**2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - (ssr / sst if sst > 0 else 0.0)

    return (
        OLSResult(
            coef=beta,
            stderr=stderr,
            t_value=t_value,
            variable_names=var_names,
            r2=r2,
            n_obs=n,
        ),
        g,
    )


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _t_pvalue(t_val: float, df: int) -> float:
    """Two-tailed p-value from t-distribution with *df* degrees of freedom."""
    if df < 1:
        return float(2.0 * (1.0 - _normal_cdf(abs(t_val))))
    return float(2.0 * t_dist.sf(abs(t_val), df))


def _two_way_within(df: pd.DataFrame, cols: List[str], city_col: str = "city_id", year_col: str = "year") -> pd.DataFrame:
    """Two-way demeaning for fixed effects estimation."""
    out = df.copy()
    for col in cols:
        city_mean = out.groupby(city_col)[col].transform("mean")
        year_mean = out.groupby(year_col)[col].transform("mean")
        grand_mean = float(out[col].mean())
        out[f"{col}_tw"] = out[col] - city_mean - year_mean + grand_mean
    return out


def _build_city_neighbor_map(panel: pd.DataFrame, k_neighbors: int = 5) -> Dict[str, List[str]]:
    """Build nearest-neighbor city map from lat/lon coordinates."""
    need = {"city_id", "latitude", "longitude"}
    if not need.issubset(set(panel.columns)):
        return {}
    city = panel[["city_id", "latitude", "longitude"]].drop_duplicates("city_id").copy()
    city["city_id"] = city["city_id"].astype(str)
    city["latitude"] = pd.to_numeric(city["latitude"], errors="coerce")
    city["longitude"] = pd.to_numeric(city["longitude"], errors="coerce")
    city = city.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if city.empty:
        return {}

    ids = city["city_id"].tolist()
    lat = city["latitude"].to_numpy(dtype=float)
    lon = city["longitude"].to_numpy(dtype=float)
    out: Dict[str, List[str]] = {}
    k = max(1, int(k_neighbors))
    for i, cid in enumerate(ids):
        dists: List[tuple[float, str]] = []
        for j, other in enumerate(ids):
            if i == j:
                continue
            d = haversine_km(float(lat[i]), float(lon[i]), float(lat[j]), float(lon[j]))
            dists.append((d, str(other)))
        dists.sort(key=lambda x: x[0])
        out[str(cid)] = [city_id for _, city_id in dists[:k]]
    return out


def _build_neighbor_exposure(
    panel: pd.DataFrame,
    *,
    value_col: str,
    k_neighbors: int = 5,
    lag_years: int = 1,
) -> pd.Series:
    """Compute city-year neighbor exposure using nearest-neighbor city network."""
    out = pd.Series(np.nan, index=panel.index, dtype=float)
    if value_col not in panel.columns:
        return out
    neigh = _build_city_neighbor_map(panel, k_neighbors=k_neighbors)
    if not neigh:
        return out

    key_df = panel[["city_id", "year", value_col]].copy()
    key_df["city_id"] = key_df["city_id"].astype(str)
    key_df["year"] = pd.to_numeric(key_df["year"], errors="coerce").astype("Int64")
    key_df[value_col] = pd.to_numeric(key_df[value_col], errors="coerce")
    key_df = key_df.dropna(subset=["year"]).copy()
    key_df["year"] = key_df["year"].astype(int)
    value_map = key_df.set_index(["city_id", "year"])[value_col].to_dict()

    vals = np.full(len(panel), np.nan, dtype=float)
    for i, row in enumerate(panel.itertuples(index=False)):
        cid = str(getattr(row, "city_id"))
        year = int(getattr(row, "year"))
        ref_year = year - int(lag_years)
        nbs = neigh.get(cid, [])
        if not nbs:
            continue
        nb_vals = [value_map.get((nid, ref_year), np.nan) for nid in nbs]
        nb_vals = [float(v) for v in nb_vals if np.isfinite(v)]
        if nb_vals:
            vals[i] = float(np.mean(nb_vals))
    out.loc[:] = vals
    return out


def _resolve_city_treatment_schedule(
    panel: pd.DataFrame,
    *,
    treated_col: str = "treated_city",
    cohort_col: str = "treatment_cohort_year",
    did_col: str = "did_treatment",
) -> pd.DataFrame:
    """Resolve one cohort year per city, preserving never-treated cities as 9999."""
    if "city_id" not in panel.columns:
        return pd.DataFrame(columns=["city_id", "treated_city", "cohort_year"])

    keep_cols = ["city_id"]
    if treated_col in panel.columns:
        keep_cols.append(treated_col)
    if cohort_col in panel.columns:
        keep_cols.append(cohort_col)
    city_tbl = panel[keep_cols].drop_duplicates("city_id").copy()
    city_tbl["city_id"] = city_tbl["city_id"].astype(str)

    if cohort_col in city_tbl.columns:
        city_tbl["cohort_year"] = (
            pd.to_numeric(city_tbl[cohort_col], errors="coerce").fillna(9999).astype(int)
        )
    elif {"city_id", "year", did_col}.issubset(set(panel.columns)):
        first_did = (
            panel.loc[pd.to_numeric(panel[did_col], errors="coerce").fillna(0.0).astype(float) >= 0.5, ["city_id", "year"]]
            .copy()
            .assign(year=lambda x: pd.to_numeric(x["year"], errors="coerce"))
            .dropna(subset=["year"])
            .groupby("city_id", as_index=False)["year"]
            .min()
            .rename(columns={"year": "cohort_year"})
        )
        first_did["city_id"] = first_did["city_id"].astype(str)
        city_tbl = city_tbl.merge(first_did, on="city_id", how="left")
        city_tbl["cohort_year"] = city_tbl["cohort_year"].fillna(9999).astype(int)
    else:
        city_tbl["cohort_year"] = 9999

    if treated_col in city_tbl.columns:
        city_tbl["treated_city"] = (
            pd.to_numeric(city_tbl[treated_col], errors="coerce").fillna(0.0).astype(int)
        )
    else:
        city_tbl["treated_city"] = (city_tbl["cohort_year"] < 9999).astype(int)

    city_tbl.loc[city_tbl["treated_city"] == 0, "cohort_year"] = 9999
    city_tbl["treated_city"] = (city_tbl["cohort_year"] < 9999).astype(int)
    return city_tbl[["city_id", "treated_city", "cohort_year"]].drop_duplicates("city_id").copy()


def _collapse_country_policy_panel_for_causal_inference(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    """Collapse city rows to an ISO3-year causal panel when treatment is country-level.

    This removes implicit weighting by the number of cities per country and
    reconciles synthetic within-country rollout back to the underlying
    country-level policy shock.
    """
    required = {"iso3", "year"}
    if not required.issubset(set(panel.columns)):
        miss = sorted(required.difference(set(panel.columns)))
        return panel.copy(), {"status": "not_applied", "reason": f"missing_columns_{','.join(miss)}"}

    scope_vals: list[str] = []
    if "policy_assignment_scope" in panel.columns:
        scope_vals = (
            panel["policy_assignment_scope"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .unique()
            .tolist()
        )
    scope_trigger = len(scope_vals) == 1 and scope_vals[0] in {"country_registry", "country_rule"}

    rollout_trigger = False
    if {"iso3", "city_id", "treated_city", "treatment_cohort_year"}.issubset(set(panel.columns)):
        city_tbl = panel[["iso3", "city_id", "treated_city", "treatment_cohort_year"]].drop_duplicates("city_id").copy()
        city_tbl["iso3"] = city_tbl["iso3"].astype(str).str.upper()
        city_tbl["treated_city"] = pd.to_numeric(city_tbl["treated_city"], errors="coerce").fillna(0).astype(int)
        city_tbl["treatment_cohort_year"] = (
            pd.to_numeric(city_tbl["treatment_cohort_year"], errors="coerce").fillna(9999).astype(int)
        )
        finite = city_tbl.loc[city_tbl["treatment_cohort_year"] < 9999].copy()
        mixed_treat = bool(city_tbl.groupby("iso3")["treated_city"].nunique().gt(1).any())
        mixed_cohort = bool(finite.groupby("iso3")["treatment_cohort_year"].nunique().gt(1).any()) if not finite.empty else False
        rollout_trigger = mixed_treat or mixed_cohort

    if not (scope_trigger or rollout_trigger):
        return panel.copy(), {"status": "not_applied", "reason": "city_level_assignment_or_no_detected_within_iso_rollout"}

    df = panel.copy()
    df["iso3"] = df["iso3"].astype(str).str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["iso3", "year"]).copy()
    df["year"] = df["year"].astype(int)

    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in {"year", "latitude", "longitude"}]
    agg_num = df.groupby(["iso3", "year"], as_index=False)[numeric_cols].mean() if numeric_cols else df[["iso3", "year"]].drop_duplicates()

    first_cols = [c for c in ["continent", "country", "policy_assignment_scope", "policy_assignment_city_rule", "causal_unit_recommended"] if c in df.columns]
    if first_cols:
        agg_first = df.groupby(["iso3", "year"], as_index=False)[first_cols].first()
        out = agg_num.merge(agg_first, on=["iso3", "year"], how="left")
    else:
        out = agg_num.copy()

    if "population" in df.columns:
        pop = (
            df.assign(population=pd.to_numeric(df["population"], errors="coerce"))
            .groupby(["iso3", "year"], as_index=False)["population"]
            .sum(min_count=1)
        )
        out = out.drop(columns=["population"], errors="ignore").merge(pop, on=["iso3", "year"], how="left")
        if "population" in out.columns:
            out["log_population"] = np.log(np.maximum(pd.to_numeric(out["population"], errors="coerce").fillna(1.0), 1.0))

    out["city_id"] = out["iso3"].astype(str)
    out["city_name"] = out["country"].astype(str) if "country" in out.columns else out["iso3"].astype(str)
    out["policy_assignment_city_rule"] = "collapsed_to_country_common_shock_for_causal_inference"
    out["causal_unit_recommended"] = "iso3_year"

    suffixes = [
        "",
        "_all_sources",
        "_direct",
        "_external_direct",
        "_intense_external_direct",
        "_intense_external_peak",
        "_objective_indicator",
        "_objective_macro",
        "_ai_inferred",
        "_direct_core",
        "_evidence_a",
        "_evidence_ab",
    ]
    for suffix in suffixes:
        cohort_col = f"treatment_cohort_year{suffix}"
        treat_col = f"treated_city{suffix}"
        post_col = f"post_policy{suffix}"
        did_col = f"did_treatment{suffix}"
        if cohort_col not in df.columns:
            continue
        cohort_map = df[["iso3", cohort_col]].copy()
        cohort_map[cohort_col] = pd.to_numeric(cohort_map[cohort_col], errors="coerce").fillna(9999).astype(int)
        cohort_map = (
            cohort_map.loc[cohort_map[cohort_col] < 9999]
            .groupby("iso3", as_index=False)[cohort_col]
            .min()
        )
        out = out.drop(columns=[cohort_col, treat_col, post_col, did_col], errors="ignore")
        out = out.merge(cohort_map, on="iso3", how="left")
        out[cohort_col] = pd.to_numeric(out[cohort_col], errors="coerce").fillna(9999).astype(int)
        out[treat_col] = (out[cohort_col] < 9999).astype(int)
        out[post_col] = ((out["year"].astype(int) >= out[cohort_col]) & (out[treat_col] == 1)).astype(int)
        out[did_col] = out[treat_col] * out[post_col]

    dose_specs = [
        ("policy_dose", "post_policy", "policy_intensity"),
        ("policy_dose_external_direct", "post_policy_external_direct", "policy_intensity_external_direct"),
        ("policy_dose_intense_external_direct", "post_policy_intense_external_direct", "policy_intensity_external_direct"),
        ("policy_dose_intense_external_peak", "post_policy_intense_external_peak", "policy_intensity_external_direct"),
    ]
    for dose_col, post_col, intensity_col in dose_specs:
        if dose_col in out.columns:
            out = out.drop(columns=[dose_col], errors="ignore")
        if post_col in out.columns and intensity_col in out.columns:
            out[dose_col] = (
                pd.to_numeric(out[post_col], errors="coerce").fillna(0.0).astype(float)
                * pd.to_numeric(out[intensity_col], errors="coerce").fillna(0.0).astype(float)
            )

    out = out.sort_values(["iso3", "year"]).reset_index(drop=True)
    return out, {
        "status": "ok",
        "causal_unit": "iso3_year",
        "reason": "country_level_policy_assignment_requires_country_common_shock_panel",
        "trigger": "policy_assignment_scope_country_level" if scope_trigger else "detected_within_iso_rollout",
        "source_rows": int(len(panel)),
        "source_cities": int(panel["city_id"].astype(str).nunique()) if "city_id" in panel.columns else None,
        "country_year_rows": int(len(out)),
        "country_count": int(out["iso3"].astype(str).nunique()),
    }


def _run_twfe_with_custom_treatment(
    panel: pd.DataFrame,
    *,
    outcome: str,
    did_col: str,
    controls: List[str],
    include_treated_trend: bool = False,
    treated_trend_col: str = "treated_city",
    min_obs: int = 80,
    min_city_count: int = 12,
    min_year_count: int = 5,
) -> Dict[str, float] | Dict[str, str]:
    _assert_valid_causal_outcome(outcome)
    controls = _resolve_sparse_controls(panel, controls)
    need_cols = ["city_id", "year", "iso3", outcome, did_col, *controls]
    if include_treated_trend:
        need_cols.append(treated_trend_col)
    for col in need_cols:
        if col not in panel.columns:
            return {"status": "skipped", "reason": f"missing_column_{col}"}

    df = panel[need_cols].dropna().copy()
    if len(df) < min_obs:
        return {"status": "skipped", "reason": "too_few_observations"}
    if df[did_col].nunique() < 2:
        return {"status": "skipped", "reason": "no_did_variation"}
    if df["city_id"].nunique() < min_city_count or df["year"].nunique() < min_year_count:
        return {"status": "skipped", "reason": "insufficient_city_or_year_support"}

    iso3_series = df["iso3"].astype(str).to_numpy()
    reg_cols = [outcome, did_col, *controls]
    if include_treated_trend:
        base_year = int(df["year"].min())
        df["treated_trend"] = (
            pd.to_numeric(df[treated_trend_col], errors="coerce").fillna(0.0).astype(float)
            * (df["year"].astype(float) - float(base_year))
        )
        reg_cols.append("treated_trend")

    tw = _two_way_within(df, reg_cols)
    tw["iso3"] = iso3_series
    y = tw[f"{outcome}_tw"].to_numpy(dtype=float)

    x_cols = [f"{did_col}_tw"] + [f"{c}_tw" for c in controls]
    if include_treated_trend:
        x_cols.append("treated_trend_tw")
    x = np.column_stack([tw[c].to_numpy(dtype=float) for c in x_cols])

    names = [did_col, *controls] + (["treated_trend"] if include_treated_trend else [])
    clusters = tw["iso3"].astype(str).to_numpy()
    try:
        res, n_clusters = _ols_cluster_hc1(y, x, names, clusters)
        stderr_type = "cluster_iso3_cr1" if n_clusters >= 2 else "hc1_fallback_single_cluster"
    except Exception:  # noqa: BLE001
        res = _ols_hc1(y, x, names)
        n_clusters = int(pd.Series(clusters).nunique())
        stderr_type = "hc1_fallback_error"
    did_idx = names.index(did_col)
    t_val = float(res.t_value[did_idx])
    return {
        "coef": float(res.coef[did_idx]),
        "stderr": float(res.stderr[did_idx]),
        "t_value": t_val,
        "p_value": _t_pvalue(t_val, max(1, n_clusters - 1)),
        "r2_within": float(res.r2),
        "n_obs": int(res.n_obs),
        "cov_type": "clustered",
        "cluster_var": "iso3",
        "controls_used": list(controls),
        "stderr_type": stderr_type,
        "n_clusters": int(n_clusters),
    }


def run_did_two_way_fe(panel: pd.DataFrame, outcome: str = "composite_index") -> Dict[str, float]:
    """Estimate treatment effect using two-way FE DID.

    Default treatment column is did_treatment_direct_core (pre-registered primary track).
    Falls back to did_treatment if direct_core is unavailable.
    """
    did_col = "did_treatment_direct_core" if "did_treatment_direct_core" in panel.columns else "did_treatment"
    out = _run_twfe_with_custom_treatment(
        panel,
        outcome=outcome,
        did_col=did_col,
        controls=["temperature_mean", "precipitation_sum", "baseline_population_log"],
        include_treated_trend=False,
        min_obs=8,
        min_city_count=2,
        min_year_count=2,
    )
    if "status" in out:
        msg = f"DID FE failed: {out}"
        raise RuntimeError(msg)
    result = dict(out)
    LOGGER.info("DID FE done (%s): coef=%.4f, t=%.3f", outcome, result["coef"], result["t_value"])
    return result


def run_intense_contrast_did(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    did_col: str = "did_treatment_intense_external_direct",
    treated_col: str = "treated_city_intense_external_direct",
) -> Dict[str, object]:
    """High-intensity treated vs others TWFE DID to strengthen treated-control contrast."""
    if did_col not in panel.columns:
        return {"status": "skipped", "reason": f"missing_column_{did_col}"}
    include_trend = treated_col in panel.columns
    est = _run_twfe_with_custom_treatment(
        panel,
        outcome=outcome,
        did_col=did_col,
        controls=["temperature_mean", "precipitation_sum", "baseline_population_log"],
        include_treated_trend=bool(include_trend),
        treated_trend_col=str(treated_col),
        min_obs=60,
        min_city_count=10,
        min_year_count=4,
    )
    if "status" in est:
        return est
    treated_share = None
    if treated_col in panel.columns:
        try:
            treated_share = float(pd.to_numeric(panel[treated_col], errors="coerce").fillna(0.0).mean())
        except Exception:  # noqa: BLE001
            treated_share = None
    return {
        "status": "ok",
        "did_col": str(did_col),
        "treated_col": str(treated_col),
        "treated_share": treated_share,
        "coef": float(est["coef"]),
        "stderr": float(est["stderr"]),
        "t_value": float(est["t_value"]),
        "p_value": float(est["p_value"]),
        "n_obs": int(est["n_obs"]),
        "stderr_type": est.get("stderr_type"),
        "n_clusters": est.get("n_clusters"),
    }


def run_dose_response_twfe(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    dose_col: str = "policy_dose",
    treated_col_for_trend: str = "treated_city",
) -> Dict[str, object]:
    """Continuous-dose TWFE DID where treatment intensity varies by city-year."""
    if dose_col not in panel.columns:
        return {"status": "skipped", "reason": f"missing_column_{dose_col}"}
    dose = pd.to_numeric(panel[dose_col], errors="coerce")
    if dose.notna().sum() < 60:
        return {"status": "skipped", "reason": "too_few_non_missing_dose_rows"}
    if float(dose.fillna(0.0).std()) < 1e-8:
        return {"status": "skipped", "reason": "no_dose_variation"}
    include_trend = treated_col_for_trend in panel.columns
    est = _run_twfe_with_custom_treatment(
        panel,
        outcome=outcome,
        did_col=dose_col,
        controls=["temperature_mean", "precipitation_sum", "baseline_population_log"],
        include_treated_trend=bool(include_trend),
        treated_trend_col=str(treated_col_for_trend),
        min_obs=60,
        min_city_count=10,
        min_year_count=4,
    )
    if "status" in est:
        return est

    dose_std = float(dose.std()) if dose.notna().sum() >= 2 else None
    coef = float(est["coef"])
    se = float(est["stderr"])
    if dose_std is not None and np.isfinite(dose_std) and dose_std > 1e-12:
        coef_1sd = float(coef * dose_std)
        se_1sd = float(se * dose_std)
    else:
        coef_1sd = None
        se_1sd = None
    return {
        "status": "ok",
        "dose_col": str(dose_col),
        "coef": coef,
        "stderr": se,
        "t_value": float(est["t_value"]),
        "p_value": float(est["p_value"]),
        "coef_per_1sd_dose": coef_1sd,
        "stderr_per_1sd_dose": se_1sd,
        "dose_std": dose_std,
        "dose_positive_share": float(np.mean(dose.fillna(0.0) > 0.0)),
        "n_obs": int(est["n_obs"]),
        "stderr_type": est.get("stderr_type"),
        "n_clusters": est.get("n_clusters"),
    }


def run_dose_response_bins_twfe(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    dose_col: str = "policy_dose",
    n_bins: int = 4,
) -> Dict[str, object]:
    """Estimate non-linear dose-response with TWFE bin dummies (zero-dose baseline)."""
    _assert_valid_causal_outcome(outcome)
    required = {
        "city_id",
        "year",
        outcome,
        dose_col,
        "temperature_mean",
        "precipitation_sum",
        "baseline_population_log",
        "iso3",
    }
    if not required.issubset(set(panel.columns)):
        miss = sorted(required.difference(set(panel.columns)))
        return {"status": "skipped", "reason": f"missing_columns_{','.join(miss)}"}

    keep = [
        "city_id",
        "year",
        "iso3",
        outcome,
        dose_col,
        "temperature_mean",
        "precipitation_sum",
        "baseline_population_log",
    ]
    df = panel[keep].dropna().copy()
    if len(df) < 120:
        return {"status": "skipped", "reason": "too_few_observations"}

    df[dose_col] = pd.to_numeric(df[dose_col], errors="coerce").fillna(0.0)
    positive = df[df[dose_col] > 0.0][dose_col].to_numpy(dtype=float)
    if positive.size < 80:
        return {"status": "skipped", "reason": "too_few_positive_dose_rows"}

    bins = int(max(2, min(8, n_bins)))
    edges = np.quantile(positive, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return {"status": "skipped", "reason": "insufficient_unique_dose_quantiles"}

    n_eff = int(edges.size - 1)
    labels = [f"q{i+1}" for i in range(n_eff)]
    dose_group = pd.Series("zero", index=df.index, dtype=object)
    pos_mask = df[dose_col] > 0.0
    dose_group.loc[pos_mask] = pd.cut(
        df.loc[pos_mask, dose_col],
        bins=edges,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    ).astype(str)
    df["dose_group"] = dose_group.fillna("zero")

    dummy_cols: List[str] = []
    dose_ranges: Dict[str, List[float]] = {}
    for i, label in enumerate(labels):
        col = f"dose_bin_{i+1}"
        df[col] = (df["dose_group"] == label).astype(float)
        if float(df[col].sum()) <= 0:
            continue
        dummy_cols.append(col)
        dose_ranges[col] = [float(edges[i]), float(edges[i + 1])]

    if len(dummy_cols) < 2:
        return {"status": "skipped", "reason": "insufficient_nonzero_dose_bins"}

    reg_cols = [outcome, *dummy_cols, "temperature_mean", "precipitation_sum", "baseline_population_log"]
    tw = _two_way_within(df, reg_cols)
    tw["iso3"] = df["iso3"].values
    y = tw[f"{outcome}_tw"].to_numpy(dtype=float)
    x_cols = [f"{c}_tw" for c in dummy_cols] + [
        "temperature_mean_tw",
        "precipitation_sum_tw",
        "baseline_population_log_tw",
    ]
    x = np.column_stack([tw[c].to_numpy(dtype=float) for c in x_cols])
    names = [*dummy_cols, "temperature_mean", "precipitation_sum", "baseline_population_log"]
    clusters = tw["iso3"].astype(str).to_numpy()
    try:
        res, n_clusters = _ols_cluster_hc1(y, x, names, clusters)
        stderr_type = "cluster_iso3_cr1" if n_clusters >= 2 else "hc1_fallback_single_cluster"
    except Exception:  # noqa: BLE001
        res = _ols_hc1(y, x, names)
        n_clusters = int(pd.Series(clusters).nunique())
        stderr_type = "hc1_fallback_error"

    rows: List[Dict[str, object]] = []
    bin_effects: List[float] = []
    for col in dummy_cols:
        j = names.index(col)
        coef = float(res.coef[j])
        se = float(res.stderr[j])
        t_val = float(res.t_value[j])
        p_val = _t_pvalue(t_val, max(1, n_clusters - 1))
        ci_low = float(coef - 1.96 * se)
        ci_high = float(coef + 1.96 * se)
        bin_effects.append(coef)
        rows.append(
            {
                "dose_bin": col,
                "dose_range_low": dose_ranges.get(col, [np.nan, np.nan])[0],
                "dose_range_high": dose_ranges.get(col, [np.nan, np.nan])[1],
                "coef": coef,
                "stderr": se,
                "t_value": t_val,
                "p_value": p_val,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n_rows_bin": int(df[col].sum()),
            }
        )

    out_df = pd.DataFrame(rows)
    safe_outcome = str(outcome).replace("/", "_").replace(":", "_")
    safe_dose = str(dose_col).replace("/", "_").replace(":", "_")
    out_path = DATA_OUTPUTS / f"dose_response_bins_{safe_dose}_{safe_outcome}.csv"
    out_df.to_csv(out_path, index=False)

    mono_non_decreasing = bool(np.all(np.diff(np.asarray(bin_effects, dtype=float)) >= -1e-8))
    mono_non_increasing = bool(np.all(np.diff(np.asarray(bin_effects, dtype=float)) <= 1e-8))
    return {
        "status": "ok",
        "dose_col": str(dose_col),
        "n_obs": int(len(df)),
        "n_bins_effective": int(len(dummy_cols)),
        "stderr_type": stderr_type,
        "n_clusters": int(n_clusters),
        "effects_by_bin": rows,
        "monotonic_non_decreasing": mono_non_decreasing,
        "monotonic_non_increasing": mono_non_increasing,
        "output_path": str(out_path),
    }


def run_spillover_twfe(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    treatment_col: str = "did_treatment",
    k_neighbors: int = 5,
    lag_years: int = 1,
) -> Dict[str, object]:
    """Estimate direct and network spillover effects in a two-way FE design."""
    _assert_valid_causal_outcome(outcome)
    if "causal_unit_recommended" in panel.columns:
        causal_units = (
            panel["causal_unit_recommended"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .unique()
            .tolist()
        )
        if causal_units and all(unit == "iso3_year" for unit in causal_units):
            return {
                "status": "skipped",
                "reason": "spillover_not_identified_on_country_level_causal_panel",
            }

    required = {
        "city_id",
        "year",
        "latitude",
        "longitude",
        outcome,
        treatment_col,
        "temperature_mean",
        "precipitation_sum",
        "baseline_population_log",
    }
    if not required.issubset(set(panel.columns)):
        miss = sorted(required.difference(set(panel.columns)))
        return {"status": "skipped", "reason": f"missing_columns_{','.join(miss)}"}

    work_cols = [
        "city_id",
        "year",
        "latitude",
        "longitude",
        "iso3",
        outcome,
        treatment_col,
        "treated_city",
        "temperature_mean",
        "precipitation_sum",
        "baseline_population_log",
    ]
    work = panel[[c for c in work_cols if c in panel.columns]].copy()
    work[treatment_col] = pd.to_numeric(work[treatment_col], errors="coerce")
    work = work.dropna(subset=["city_id", "year", "latitude", "longitude", outcome, treatment_col]).copy()
    if len(work) < 120:
        return {"status": "skipped", "reason": "too_few_observations"}
    if int(work[treatment_col].nunique()) < 2:
        return {"status": "skipped", "reason": "no_treatment_variation"}

    work = work.sort_values(["city_id", "year"]).reset_index(drop=True)
    work["spillover_exposure"] = _build_neighbor_exposure(
        work,
        value_col=treatment_col,
        k_neighbors=int(k_neighbors),
        lag_years=int(lag_years),
    )
    work["spillover_exposure"] = pd.to_numeric(work["spillover_exposure"], errors="coerce").fillna(0.0)
    if float(work["spillover_exposure"].std()) < 1e-8:
        return {"status": "skipped", "reason": "no_neighbor_exposure_variation"}

    reg_cols = [
        outcome,
        treatment_col,
        "spillover_exposure",
        "temperature_mean",
        "precipitation_sum",
        "baseline_population_log",
    ]
    include_trend = "treated_city" in work.columns
    if include_trend:
        base_year = int(pd.to_numeric(work["year"], errors="coerce").min())
        work["treated_city"] = pd.to_numeric(work["treated_city"], errors="coerce").fillna(0.0).astype(float)
        work["treated_trend"] = work["treated_city"] * (work["year"].astype(float) - float(base_year))
        reg_cols.append("treated_trend")

    tw = _two_way_within(work, reg_cols)
    tw["iso3"] = work["iso3"].values if "iso3" in work.columns else work["city_id"].astype(str).to_numpy()
    y = tw[f"{outcome}_tw"].to_numpy(dtype=float)
    x_cols = [
        f"{treatment_col}_tw",
        "spillover_exposure_tw",
        "temperature_mean_tw",
        "precipitation_sum_tw",
        "baseline_population_log_tw",
    ]
    if include_trend:
        x_cols.append("treated_trend_tw")
    x = np.column_stack([tw[c].to_numpy(dtype=float) for c in x_cols])
    names = [
        "direct_treatment",
        "spillover_exposure",
        "temperature_mean",
        "precipitation_sum",
        "baseline_population_log",
    ] + (["treated_trend"] if include_trend else [])
    clusters = tw["iso3"].astype(str).to_numpy()
    try:
        res, n_clusters = _ols_cluster_hc1(y, x, names, clusters)
        stderr_type = "cluster_iso3_cr1" if n_clusters >= 2 else "hc1_fallback_single_cluster"
    except Exception:  # noqa: BLE001
        res = _ols_hc1(y, x, names)
        n_clusters = int(pd.Series(clusters).nunique())
        stderr_type = "hc1_fallback_error"

    j_direct = names.index("direct_treatment")
    j_spill = names.index("spillover_exposure")
    direct_coef = float(res.coef[j_direct])
    direct_se = float(res.stderr[j_direct])
    direct_t = float(res.t_value[j_direct])
    spill_coef = float(res.coef[j_spill])
    spill_se = float(res.stderr[j_spill])
    spill_t = float(res.t_value[j_spill])
    spill_sd = float(work["spillover_exposure"].std())

    # Placebo check: future (lead) neighbor exposure should not explain current outcomes.
    work["spillover_exposure_lead"] = _build_neighbor_exposure(
        work,
        value_col=treatment_col,
        k_neighbors=int(k_neighbors),
        lag_years=-1,
    )
    work["spillover_exposure_lead"] = pd.to_numeric(work["spillover_exposure_lead"], errors="coerce").fillna(0.0)
    placebo = {"status": "skipped", "reason": "no_variation"}
    if float(work["spillover_exposure_lead"].std()) >= 1e-8:
        place_cols = [
            outcome,
            treatment_col,
            "spillover_exposure",
            "spillover_exposure_lead",
            "temperature_mean",
            "precipitation_sum",
            "baseline_population_log",
        ] + (["treated_trend"] if include_trend else [])
        tw_p = _two_way_within(work, place_cols)
        y_p = tw_p[f"{outcome}_tw"].to_numpy(dtype=float)
        x_p_cols = [
            f"{treatment_col}_tw",
            "spillover_exposure_tw",
            "spillover_exposure_lead_tw",
            "temperature_mean_tw",
            "precipitation_sum_tw",
            "baseline_population_log_tw",
        ] + (["treated_trend_tw"] if include_trend else [])
        x_p = np.column_stack([tw_p[c].to_numpy(dtype=float) for c in x_p_cols])
        names_p = [
            "direct_treatment",
            "spillover_exposure",
            "spillover_exposure_lead",
            "temperature_mean",
            "precipitation_sum",
            "baseline_population_log",
        ] + (["treated_trend"] if include_trend else [])
        try:
            res_p, _ = _ols_cluster_hc1(y_p, x_p, names_p, clusters)
        except Exception:  # noqa: BLE001
            res_p = _ols_hc1(y_p, x_p, names_p)
        j_lead = names_p.index("spillover_exposure_lead")
        t_lead = float(res_p.t_value[j_lead])
        placebo = {
            "status": "ok",
            "lead_coef": float(res_p.coef[j_lead]),
            "lead_stderr": float(res_p.stderr[j_lead]),
            "lead_t_value": t_lead,
            "lead_p_value": _t_pvalue(t_lead, max(1, n_clusters - 1)),
        }

    return {
        "status": "ok",
        "treatment_col": str(treatment_col),
        "k_neighbors": int(k_neighbors),
        "lag_years": int(lag_years),
        "n_obs": int(len(work)),
        "n_clusters": int(n_clusters),
        "stderr_type": stderr_type,
        "direct_coef": direct_coef,
        "direct_stderr": direct_se,
        "direct_t_value": direct_t,
        "direct_p_value": _t_pvalue(direct_t, max(1, n_clusters - 1)),
        "spillover_coef": spill_coef,
        "spillover_stderr": spill_se,
        "spillover_t_value": spill_t,
        "spillover_p_value": _t_pvalue(spill_t, max(1, n_clusters - 1)),
        "spillover_coef_per_1sd_exposure": float(spill_coef * spill_sd) if np.isfinite(spill_sd) else None,
        "spillover_exposure_std": spill_sd,
        "spillover_exposure_mean": float(work["spillover_exposure"].mean()),
        "placebo_lead_check": placebo,
    }


def run_twfe_cluster_bootstrap(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    draws: int = 160,
    random_state: int = 42,
) -> Dict[str, object]:
    """City-cluster bootstrap for TWFE DID coefficient."""
    _assert_valid_causal_outcome(outcome)
    required = {
        "city_id",
        "year",
        "did_treatment",
        "temperature_mean",
        "precipitation_sum",
        "baseline_population_log",
        outcome,
    }
    if not required.issubset(set(panel.columns)):
        miss = sorted(required.difference(set(panel.columns)))
        return {"status": "skipped", "reason": f"missing_columns_{','.join(miss)}"}

    base = _run_twfe_with_custom_treatment(
        panel,
        outcome=outcome,
        did_col="did_treatment",
        controls=["temperature_mean", "precipitation_sum", "baseline_population_log"],
        include_treated_trend=False,
        min_obs=8,
        min_city_count=2,
        min_year_count=2,
    )
    if "status" in base:
        return {"status": "skipped", "reason": f"base_twfe_failed_{base['reason']}"}

    city_ids = pd.Series(panel["city_id"].astype(str).unique())
    n_city = int(len(city_ids))
    if n_city < 8:
        return {"status": "skipped", "reason": "too_few_cities_for_bootstrap"}

    rng = np.random.default_rng(random_state)
    coef_draws: List[float] = []
    for b in range(int(draws)):
        sampled = rng.choice(city_ids.to_numpy(), size=n_city, replace=True)
        mapping = pd.DataFrame(
            {
                "city_id_orig": sampled,
                "city_id_boot": [f"bs{b}_{i}" for i in range(n_city)],
            }
        )
        boot = mapping.merge(panel, left_on="city_id_orig", right_on="city_id", how="left")
        if boot.empty:
            continue
        boot = boot.drop(columns=["city_id_orig", "city_id"]).rename(columns={"city_id_boot": "city_id"})

        est = _run_twfe_with_custom_treatment(
            boot,
            outcome=outcome,
            did_col="did_treatment",
            controls=["temperature_mean", "precipitation_sum", "baseline_population_log"],
            include_treated_trend=False,
            min_obs=8,
            min_city_count=2,
            min_year_count=2,
        )
        if "status" in est:
            continue
        coef_draws.append(float(est["coef"]))

    success = int(len(coef_draws))
    if success < max(20, int(0.30 * draws)):
        return {
            "status": "skipped",
            "reason": "insufficient_successful_bootstrap_draws",
            "draws_requested": int(draws),
            "draws_successful": success,
        }

    arr = np.asarray(coef_draws, dtype=float)
    ci_low = float(np.quantile(arr, 0.025))
    ci_high = float(np.quantile(arr, 0.975))
    p_two = float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))
    out = {
        "status": "ok",
        "coef_observed": float(base["coef"]),
        "stderr_observed": float(base["stderr"]),
        "t_value_observed": float(base["t_value"]),
        "draws_requested": int(draws),
        "draws_successful": success,
        "coef_bootstrap_mean": float(np.mean(arr)),
        "coef_bootstrap_std": float(np.std(arr, ddof=1)),
        "ci95_percentile": [ci_low, ci_high],
        "p_value_two_sided": p_two,
    }
    return out


def run_twfe_city_permutation(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    draws: int = 200,
    random_state: int = 42,
) -> Dict[str, object]:
    """City-schedule permutation test for TWFE DID under a cohort-preserving sharp null."""
    _assert_valid_causal_outcome(outcome)
    required = {
        "city_id",
        "year",
        "treated_city",
        "post_policy",
        "did_treatment",
        "temperature_mean",
        "precipitation_sum",
        "baseline_population_log",
        outcome,
    }
    if not required.issubset(set(panel.columns)):
        miss = sorted(required.difference(set(panel.columns)))
        return {"status": "skipped", "reason": f"missing_columns_{','.join(miss)}"}

    base = _run_twfe_with_custom_treatment(
        panel,
        outcome=outcome,
        did_col="did_treatment",
        controls=["temperature_mean", "precipitation_sum", "baseline_population_log"],
        include_treated_trend=False,
        min_obs=8,
        min_city_count=2,
        min_year_count=2,
    )
    if "status" in base:
        return {"status": "skipped", "reason": f"base_twfe_failed_{base['reason']}"}

    city_tbl = _resolve_city_treatment_schedule(panel)
    if city_tbl.empty:
        return {"status": "skipped", "reason": "missing_city_treatment_schedule"}
    if city_tbl["treated_city"].nunique() < 2:
        return {"status": "skipped", "reason": "no_city_level_treatment_variation"}
    if city_tbl["cohort_year"].nunique() < 2:
        return {"status": "skipped", "reason": "no_city_level_cohort_variation"}

    rng = np.random.default_rng(random_state)
    perm_coefs: List[float] = []
    perm_ts: List[float] = []
    base_city = city_tbl["city_id"].astype(str).to_numpy()
    base_cohort = city_tbl["cohort_year"].to_numpy(dtype=int)

    for _ in range(int(draws)):
        perm_cohort = rng.permutation(base_cohort)
        perm_map = pd.DataFrame({"city_id": base_city, "cohort_perm": perm_cohort})
        tmp = panel.merge(perm_map, on="city_id", how="left").copy()
        tmp["cohort_perm"] = pd.to_numeric(tmp["cohort_perm"], errors="coerce").fillna(9999).astype(int)
        tmp["treated_perm"] = (tmp["cohort_perm"] < 9999).astype(int)
        tmp["post_perm"] = ((tmp["year"].astype(int) >= tmp["cohort_perm"]) & (tmp["treated_perm"] == 1)).astype(int)
        tmp["did_perm"] = tmp["treated_perm"] * tmp["post_perm"]

        est = _run_twfe_with_custom_treatment(
            tmp,
            outcome=outcome,
            did_col="did_perm",
            controls=["temperature_mean", "precipitation_sum", "baseline_population_log"],
            include_treated_trend=False,
            min_obs=8,
            min_city_count=2,
            min_year_count=2,
        )
        if "status" in est:
            continue
        perm_coefs.append(float(est["coef"]))
        perm_ts.append(float(est["t_value"]))

    success = int(len(perm_coefs))
    if success < max(30, int(0.30 * draws)):
        return {
            "status": "skipped",
            "reason": "insufficient_successful_permutation_draws",
            "draws_requested": int(draws),
            "draws_successful": success,
        }

    coef_null = np.asarray(perm_coefs, dtype=float)
    t_null = np.asarray(perm_ts, dtype=float)
    obs_coef = float(base["coef"])
    obs_t = float(base["t_value"])
    p_coef = float((1.0 + np.sum(np.abs(coef_null) >= abs(obs_coef))) / (1.0 + len(coef_null)))
    p_t = float((1.0 + np.sum(np.abs(t_null) >= abs(obs_t))) / (1.0 + len(t_null)))
    out = {
        "status": "ok",
        "coef_observed": obs_coef,
        "t_value_observed": obs_t,
        "permutation_design": "permute_city_cohort_schedule_preserving_treated_share_and_timing_mix",
        "draws_requested": int(draws),
        "draws_successful": success,
        "null_coef_mean": float(np.mean(coef_null)),
        "null_coef_std": float(np.std(coef_null, ddof=1)),
        "null_abs_t_p95": float(np.quantile(np.abs(t_null), 0.95)),
        "p_value_abs_coef": p_coef,
        "p_value_abs_t": p_t,
    }
    return out


def run_twfe_wild_cluster_bootstrap(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    draws: int = 220,
    random_state: int = 42,
) -> Dict[str, object]:
    """Wild cluster bootstrap (Rademacher) for TWFE DID t-stat under a restricted null."""
    _assert_valid_causal_outcome(outcome)
    required = {
        "city_id",
        "year",
        "did_treatment",
        "temperature_mean",
        "precipitation_sum",
        "baseline_population_log",
        outcome,
    }
    if not required.issubset(set(panel.columns)):
        miss = sorted(required.difference(set(panel.columns)))
        return {"status": "skipped", "reason": f"missing_columns_{','.join(miss)}"}

    df = panel[
        ["city_id", "year", "iso3", outcome, "did_treatment", "temperature_mean", "precipitation_sum", "baseline_population_log"]
    ].dropna().copy()
    if len(df) < 80:
        return {"status": "skipped", "reason": "too_few_observations"}
    if df["did_treatment"].nunique() < 2:
        return {"status": "skipped", "reason": "no_did_variation"}
    if df["city_id"].nunique() < 12 or df["year"].nunique() < 5:
        return {"status": "skipped", "reason": "insufficient_city_or_year_support"}

    reg_cols = [outcome, "did_treatment", "temperature_mean", "precipitation_sum", "baseline_population_log"]
    tw = _two_way_within(df, reg_cols)
    tw["iso3"] = df["iso3"].values
    y = tw[f"{outcome}_tw"].to_numpy(dtype=float)
    x = np.column_stack(
        [
            tw["did_treatment_tw"].to_numpy(dtype=float),
            tw["temperature_mean_tw"].to_numpy(dtype=float),
            tw["precipitation_sum_tw"].to_numpy(dtype=float),
            tw["baseline_population_log_tw"].to_numpy(dtype=float),
        ]
    )
    names = ["did_treatment", "temperature_mean", "precipitation_sum", "baseline_population_log"]
    clusters = tw["iso3"].astype(str).to_numpy()
    base_res, n_clusters = _ols_cluster_hc1(y, x, names, clusters)
    did_idx = names.index("did_treatment")
    coef_obs = float(base_res.coef[did_idx])
    t_obs = float(base_res.t_value[did_idx])

    x_null = np.delete(x, did_idx, axis=1)
    _, fitted_null, resid_null = _ols_fit(y, x_null)
    uniq = np.unique(clusters.astype(str))
    if len(uniq) < 12:
        return {"status": "skipped", "reason": "too_few_clusters_for_wild_bootstrap"}

    cluster_index = {cid: np.where(clusters.astype(str) == cid)[0] for cid in uniq}
    rng = np.random.default_rng(random_state)
    coef_draws: List[float] = []
    t_draws: List[float] = []

    for _ in range(int(draws)):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(uniq), replace=True)
        e_star = np.zeros_like(resid_null)
        for cid, sign in zip(uniq, signs, strict=False):
            idx = cluster_index[str(cid)]
            e_star[idx] = resid_null[idx] * float(sign)
        y_star = fitted_null + e_star

        try:
            res_star, _ = _ols_cluster_hc1(y_star, x, names, clusters)
        except Exception:  # noqa: BLE001
            continue
        coef_draws.append(float(res_star.coef[did_idx]))
        t_draws.append(float(res_star.t_value[did_idx]))

    success = int(len(t_draws))
    if success < max(30, int(0.30 * draws)):
        return {
            "status": "skipped",
            "reason": "insufficient_successful_wild_bootstrap_draws",
            "draws_requested": int(draws),
            "draws_successful": success,
        }

    coef_arr = np.asarray(coef_draws, dtype=float)
    t_arr = np.asarray(t_draws, dtype=float)
    p_t = float((1.0 + np.sum(np.abs(t_arr) >= abs(t_obs))) / (1.0 + len(t_arr)))
    p_coef = float((1.0 + np.sum(np.abs(coef_arr) >= abs(coef_obs))) / (1.0 + len(coef_arr)))
    out = {
        "status": "ok",
        "coef_observed": coef_obs,
        "t_value_observed": t_obs,
        "n_clusters": int(n_clusters),
        "null_design": "restricted_null_beta_did_eq_0_with_cluster_rademacher_weights",
        "draws_requested": int(draws),
        "draws_successful": success,
        "null_coef_mean": float(np.mean(coef_arr)),
        "null_t_mean": float(np.mean(t_arr)),
        "null_t_std": float(np.std(t_arr, ddof=1)),
        "null_abs_t_p95": float(np.quantile(np.abs(t_arr), 0.95)),
        "p_value_abs_t": p_t,
        "p_value_abs_coef": p_coef,
        "ci95_coef_percentile": [float(np.quantile(coef_arr, 0.025)), float(np.quantile(coef_arr, 0.975))],
    }
    return out


def run_twfe_lead_placebo(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    lead_years: List[int] | None = None,
    track_label: str = "main",
) -> Dict[str, object]:
    """Lead placebo DID: move treatment earlier and test false-positive rate."""
    _assert_valid_causal_outcome(outcome)
    if lead_years is None:
        lead_years = [1, 2]

    required = {
        "city_id",
        "year",
        "treated_city",
        "did_treatment",
        "temperature_mean",
        "precipitation_sum",
        "baseline_population_log",
        outcome,
    }
    if not required.issubset(set(panel.columns)):
        miss = sorted(required.difference(set(panel.columns)))
        return {"status": "skipped", "reason": f"missing_columns_{','.join(miss)}"}

    df = panel[
        [
            "city_id",
            "year",
            "iso3",
            "treated_city",
            "did_treatment",
            "temperature_mean",
            "precipitation_sum",
            "baseline_population_log",
            outcome,
            *([ "treatment_cohort_year"] if "treatment_cohort_year" in panel.columns else []),
        ]
    ].dropna().copy()
    if len(df) < 80:
        return {"status": "skipped", "reason": "too_few_observations"}

    if "treatment_cohort_year" in df.columns:
        cohort_map = (
            df[["city_id", "treatment_cohort_year"]]
            .drop_duplicates("city_id")
            .assign(treatment_cohort_year=lambda x: pd.to_numeric(x["treatment_cohort_year"], errors="coerce").fillna(9999).astype(int))
            .set_index("city_id")["treatment_cohort_year"]
            .to_dict()
        )
    else:
        first_did = (
            df[df["did_treatment"] == 1]
            .groupby("city_id", as_index=False)["year"]
            .min()
            .rename(columns={"year": "cohort"})
        )
        cohort_map = {str(r.city_id): int(r.cohort) for r in first_did.itertuples(index=False)}

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    rows: List[Dict[str, float | int]] = []
    for lead in sorted(set(int(x) for x in lead_years if int(x) >= 1)):
        tmp = df.copy()
        cohort_orig = tmp["city_id"].astype(str).map(cohort_map).fillna(9999).astype(int)
        cohort_fake = np.where(cohort_orig < 9999, np.maximum(cohort_orig - int(lead), min_year + 1), 9999)
        # Require sufficient pre/post support for placebo timing to avoid late-cohort edge contamination.
        eligible_treated = (
            (tmp["treated_city"].astype(int) == 1)
            & (cohort_orig < 9999)
            & (cohort_fake >= (min_year + 1))
            & (cohort_fake <= (max_year - 1))
        ).astype(int)
        tmp["did_lead_placebo"] = eligible_treated * (tmp["year"] >= cohort_fake).astype(int)
        est = _run_twfe_with_custom_treatment(
            tmp,
            outcome=outcome,
            did_col="did_lead_placebo",
            controls=["temperature_mean", "precipitation_sum", "baseline_population_log"],
            include_treated_trend=False,
            min_obs=36,
            min_city_count=6,
            min_year_count=3,
        )
        if "status" in est:
            continue
        rows.append(
            {
                "lead_years": int(lead),
                "coef": float(est["coef"]),
                "stderr": float(est["stderr"]),
                "t_value": float(est["t_value"]),
                "p_value": float(est["p_value"]),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("lead_years") if rows else pd.DataFrame()
    safe_label = str(track_label).replace(":", "_").replace("/", "_")
    safe_outcome = str(outcome).replace(":", "_").replace("/", "_")
    csv_path = DATA_OUTPUTS / f"twfe_lead_placebo_{safe_label}_{safe_outcome}.csv"
    out_df.to_csv(csv_path, index=False)
    if out_df.empty:
        return {"status": "skipped", "reason": "no_valid_lead_placebo_estimates", "output_path": str(csv_path)}

    pvals = out_df["p_value"].to_numpy(dtype=float)
    tvals = out_df["t_value"].to_numpy(dtype=float)
    return {
        "status": "ok",
        "lead_count": int(len(out_df)),
        "share_p_lt_0_10": float(np.mean(pvals < 0.10)),
        "max_abs_t": float(np.max(np.abs(tvals))),
        "mean_abs_coef": float(np.mean(np.abs(out_df["coef"].to_numpy(dtype=float)))),
        "rows": out_df.to_dict(orient="records"),
        "output_path": str(csv_path),
    }


def run_stacked_lead_placebo(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    lead_years: List[int] | None = None,
    window_pre: int = 2,
    window_post: int = 2,
    track_label: str = "main",
) -> Dict[str, object]:
    """Cohort-aware stacked lead-placebo DID with local event windows."""
    _assert_valid_causal_outcome(outcome)
    if lead_years is None:
        lead_years = [2, 3]

    required = {
        "city_id",
        "year",
        "treated_city",
        "did_treatment",
        "temperature_mean",
        "precipitation_sum",
        "baseline_population_log",
        outcome,
    }
    if not required.issubset(set(panel.columns)):
        miss = sorted(required.difference(set(panel.columns)))
        return {"status": "skipped", "reason": f"missing_columns_{','.join(miss)}"}

    keep_cols = [
        "city_id",
        "year",
        "treated_city",
        "did_treatment",
        "temperature_mean",
        "precipitation_sum",
        "baseline_population_log",
        outcome,
    ]
    if "treatment_cohort_year" in panel.columns:
        keep_cols.append("treatment_cohort_year")
    df = panel[keep_cols].dropna().copy()
    if len(df) < 120:
        return {"status": "skipped", "reason": "too_few_observations"}

    city_tbl = df[["city_id", "treated_city"]].drop_duplicates("city_id").copy()
    city_tbl["treated_city"] = city_tbl["treated_city"].astype(int)
    if int(city_tbl["treated_city"].sum()) < 8:
        return {"status": "skipped", "reason": "too_few_treated_cities"}

    if "treatment_cohort_year" in df.columns:
        cohort_map_df = df[["city_id", "treatment_cohort_year"]].drop_duplicates("city_id").copy()
        cohort_map_df["cohort_orig"] = (
            pd.to_numeric(cohort_map_df["treatment_cohort_year"], errors="coerce")
            .fillna(9999)
            .astype(int)
        )
        cohort_map_df = cohort_map_df[["city_id", "cohort_orig"]]
    else:
        first_did = (
            df[df["did_treatment"] == 1]
            .groupby("city_id", as_index=False)["year"]
            .min()
            .rename(columns={"year": "cohort_orig"})
        )
        cohort_map_df = city_tbl[["city_id"]].merge(first_did, on="city_id", how="left")
        cohort_map_df["cohort_orig"] = pd.to_numeric(cohort_map_df["cohort_orig"], errors="coerce").fillna(9999).astype(int)

    city_tbl = city_tbl.merge(cohort_map_df, on="city_id", how="left")
    city_tbl["cohort_orig"] = city_tbl["cohort_orig"].fillna(9999).astype(int)
    city_tbl.loc[city_tbl["treated_city"] == 0, "cohort_orig"] = 9999

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    rows: List[Dict[str, object]] = []

    for lead in sorted(set(int(x) for x in lead_years if int(x) >= 1)):
        cohort_lead = city_tbl.copy()
        cohort_lead["cohort_fake"] = np.where(
            cohort_lead["cohort_orig"] < 9999,
            np.maximum(cohort_lead["cohort_orig"] - int(lead), min_year + int(max(1, window_pre))),
            9999,
        ).astype(int)

        eligible = cohort_lead[
            (cohort_lead["treated_city"] == 1)
            & (cohort_lead["cohort_orig"] < 9999)
            & (cohort_lead["cohort_fake"] >= (min_year + int(max(1, window_pre))))
            & (cohort_lead["cohort_fake"] <= (max_year - int(max(1, window_post))))
        ].copy()
        if eligible.empty:
            continue

        cell_rows: List[Dict[str, object]] = []
        for g in sorted(eligible["cohort_fake"].astype(int).unique().tolist()):
            treat_ids = eligible.loc[eligible["cohort_fake"] == g, "city_id"].astype(str).tolist()
            if len(treat_ids) < 3:
                continue

            ctrl_ids = cohort_lead.loc[
                (cohort_lead["cohort_orig"] > int(g)) | (cohort_lead["treated_city"] == 0),
                "city_id",
            ].astype(str).tolist()
            if len(ctrl_ids) < 6:
                continue

            keep_ids = set(treat_ids).union(set(ctrl_ids))
            sub = df[df["city_id"].astype(str).isin(keep_ids)].copy()
            sub = sub[
                (sub["year"] >= (int(g) - int(window_pre)))
                & (sub["year"] <= (int(g) + int(window_post)))
            ].copy()
            if sub.empty:
                continue

            treat_set = set(treat_ids)
            sub["treated_city_stack"] = sub["city_id"].astype(str).isin(treat_set).astype(int)
            sub["did_stack"] = sub["treated_city_stack"] * (sub["year"] >= int(g)).astype(int)

            est = _run_twfe_with_custom_treatment(
                sub,
                outcome=outcome,
                did_col="did_stack",
                controls=["temperature_mean", "precipitation_sum", "baseline_population_log"],
                include_treated_trend=True,
                treated_trend_col="treated_city_stack",
                min_obs=60,
                min_city_count=10,
                min_year_count=4,
            )
            if "status" in est:
                continue

            cell_rows.append(
                {
                    "lead_years": int(lead),
                    "cohort_fake_year": int(g),
                    "coef": float(est["coef"]),
                    "stderr": float(est["stderr"]),
                    "t_value": float(est["t_value"]),
                    "p_value": float(est["p_value"]),
                    "n_obs": int(est["n_obs"]),
                    "n_treat_cities": int(len(treat_ids)),
                    "n_control_cities": int(len(ctrl_ids)),
                }
            )

        if not cell_rows:
            continue

        cell_df = pd.DataFrame(cell_rows)
        w = np.maximum(cell_df["n_treat_cities"].to_numpy(dtype=float), 1.0)
        w = w / max(float(w.sum()), 1e-12)
        coef = float(np.sum(w * cell_df["coef"].to_numpy(dtype=float)))
        stderr = float(np.sqrt(np.sum((w**2) * (cell_df["stderr"].to_numpy(dtype=float) ** 2))))
        t_val = float(coef / max(stderr, 1e-12))
        p_val = _t_pvalue(t_val, max(1, len(cell_rows) - 1))
        rows.append(
            {
                "lead_years": int(lead),
                "coef": coef,
                "stderr": stderr,
                "t_value": t_val,
                "p_value": p_val,
                "cells": int(len(cell_df)),
                "treated_cities_weighted": float(cell_df["n_treat_cities"].sum()),
                "n_obs_weighted": float(np.sum(w * cell_df["n_obs"].to_numpy(dtype=float))),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("lead_years") if rows else pd.DataFrame()
    safe_label = str(track_label).replace(":", "_").replace("/", "_")
    safe_outcome = str(outcome).replace(":", "_").replace("/", "_")
    csv_path = DATA_OUTPUTS / f"did_stacked_lead_placebo_{safe_label}_{safe_outcome}.csv"
    out_df.to_csv(csv_path, index=False)
    if out_df.empty:
        return {"status": "skipped", "reason": "no_valid_stacked_placebo_estimates", "output_path": str(csv_path)}

    pvals = out_df["p_value"].to_numpy(dtype=float)
    tvals = out_df["t_value"].to_numpy(dtype=float)
    return {
        "status": "ok",
        "method": "stacked_cohort_window_placebo",
        "window_pre": int(window_pre),
        "window_post": int(window_post),
        "lead_count": int(len(out_df)),
        "share_p_lt_0_10": float(np.mean(pvals < 0.10)),
        "max_abs_t": float(np.max(np.abs(tvals))),
        "mean_abs_coef": float(np.mean(np.abs(out_df["coef"].to_numpy(dtype=float)))),
        "rows": out_df.to_dict(orient="records"),
        "output_path": str(csv_path),
    }


def run_dml_did(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    n_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, float]:
    """Double Machine Learning DID via grouped cross-fitting and overlap trimming."""
    df = panel.copy()
    required_cols = [
        "city_id",
        "continent",
        "did_treatment",
        "temperature_mean",
        "precipitation_sum",
        "log_population",
        "inflation",
        "poi_total",
        "poi_diversity",
        "year",
        outcome,
    ]
    miss = [c for c in required_cols if c not in df.columns]
    if miss:
        return {"status": "skipped", "reason": f"missing_columns_{','.join(miss)}"}

    df = df[required_cols].dropna().copy()
    if len(df) < 80:
        return {"status": "skipped", "reason": "too_few_observations"}

    d = df["did_treatment"].astype(int).to_numpy(dtype=float)
    if d.std() < 1e-8:
        return {"status": "skipped", "reason": "no_treatment_variation"}

    feature_cols = [
        "temperature_mean",
        "precipitation_sum",
        "log_population",
        "inflation",
        "poi_total",
        "poi_diversity",
        "year",
    ]
    x_num = df[feature_cols].copy()
    x_cat = pd.get_dummies(df[["continent"]], drop_first=True)
    x_df = pd.concat([x_num, x_cat], axis=1).astype(float)
    y = df[outcome].to_numpy(dtype=float)
    groups = df["city_id"].astype(str).to_numpy()
    x_full = x_df.to_numpy(dtype=float)
    y_full = y.copy()
    d_full = d.copy()

    def _legacy_rf_crossfit(x_mat: np.ndarray, y_arr: np.ndarray, d_arr: np.ndarray) -> Dict[str, float]:
        y_hat = np.zeros(len(y_arr), dtype=float)
        d_hat = np.zeros(len(d_arr), dtype=float)
        _n_sp = max(2, min(int(n_splits), 5))
        if len(np.unique(groups)) >= _n_sp * 2:
            gkf = GroupKFold(n_splits=_n_sp)
            fold_iter = gkf.split(x_mat, groups=groups)
        else:
            kf = KFold(n_splits=_n_sp, shuffle=True, random_state=random_state)
            fold_iter = kf.split(x_mat)
        for fold_idx, (tr, te) in enumerate(fold_iter, start=1):
            model_y = RandomForestRegressor(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=5,
                random_state=random_state + fold_idx,
                n_jobs=-1,
            )
            model_d = RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=8,
                random_state=random_state + 100 + fold_idx,
                n_jobs=-1,
            )
            model_y.fit(x_mat[tr], y_arr[tr])
            model_d.fit(x_mat[tr], d_arr[tr])
            y_hat[te] = model_y.predict(x_mat[te])
            d_hat[te] = np.clip(model_d.predict(x_mat[te]), 1e-4, 1.0 - 1e-4)

        y_tilde = y_arr - y_hat
        d_tilde = d_arr - d_hat
        denom = float(np.sum(d_tilde**2))
        if abs(denom) < 1e-10:
            return {"status": "skipped", "reason": "legacy_weak_residual_treatment"}

        theta = float(np.sum(d_tilde * y_tilde) / denom)
        d_second = float(np.mean(d_tilde**2))
        psi = (d_tilde * (y_tilde - theta * d_tilde)) / max(d_second, 1e-10)
        se = float(np.sqrt(np.mean(psi**2) / len(y_arr)))
        t_val = float(theta / max(se, 1e-12))
        p_val = _t_pvalue(t_val, max(1, len(np.unique(groups)) - 1) if len(groups) > 0 else 100)
        return {
            "status": "ok",
            "coef": theta,
            "stderr": se,
            "t_value": t_val,
            "p_value": p_val,
            "n_obs": int(len(y_arr)),
            "n_features": int(x_mat.shape[1]),
            "spec": "legacy_rf_crossfit_baseline",
            "cv_mode": "kfold_random",
        }

    baseline = _legacy_rf_crossfit(x_full, y_full, d_full)
    if baseline.get("status") != "ok":
        return {"status": "skipped", "reason": str(baseline.get("reason", "legacy_dml_failed"))}

    # Overlap trimming by propensity score to avoid weak-overlap noise.
    ps_model = LogisticRegression(
        max_iter=1200,
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        random_state=random_state,
    )
    try:
        ps_model.fit(x_df.to_numpy(dtype=float), d.astype(int))
        ps = np.clip(ps_model.predict_proba(x_df.to_numpy(dtype=float))[:, 1], 1e-4, 1.0 - 1e-4)
    except Exception:  # noqa: BLE001
        ps = np.clip(np.full(len(d), float(np.mean(d))), 1e-4, 1.0 - 1e-4)

    keep = (ps >= 0.03) & (ps <= 0.97)
    if int(keep.sum()) >= max(60, int(0.45 * len(df))):
        trimmed_share = float(1.0 - keep.mean())
        x_df = x_df.loc[keep].reset_index(drop=True)
        y = y[keep]
        d = d[keep]
        groups = groups[keep]
        ps = ps[keep]
    else:
        trimmed_share = 0.0

    n = len(y)
    if n < 60:
        return {"status": "skipped", "reason": "too_few_observations_after_trimming"}
    if d.std() < 1e-8:
        return {"status": "skipped", "reason": "no_treatment_variation_after_trimming"}

    unique_groups = np.unique(groups)
    splits = max(2, min(int(n_splits), int(n // 40)))
    if len(unique_groups) >= splits * 2:
        gkf = GroupKFold(n_splits=splits)
        split_indices = list(gkf.split(x_df, y, groups))
        cv_mode = "group_kfold_city"
    else:
        kf = KFold(n_splits=splits, shuffle=True, random_state=random_state)
        split_indices = list(kf.split(x_df))
        cv_mode = "kfold_random"

    y_models = {
        "rf": RandomForestRegressor(
            n_estimators=420,
            max_depth=10,
            min_samples_leaf=5,
            random_state=random_state + 11,
            n_jobs=-1,
        ),
        "hgb": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=420,
            min_samples_leaf=20,
            random_state=random_state + 17,
        ),
        "ridge": Ridge(alpha=1.0, random_state=random_state + 19),
    }
    d_models = {
        "rf_cls": RandomForestClassifier(
            n_estimators=360,
            max_depth=9,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=random_state + 23,
            n_jobs=-1,
        ),
        "logit": LogisticRegression(
            max_iter=1200,
            solver="liblinear",
            C=1.0,
            class_weight="balanced",
            random_state=random_state + 29,
        ),
    }

    x_np = x_df.to_numpy(dtype=float)

    def _crossfit(y_model, d_model) -> Dict[str, object]:
        y_hat = np.zeros(n, dtype=float)
        d_hat = np.zeros(n, dtype=float)

        for tr, te in split_indices:
            model_y = clone(y_model)
            model_d = clone(d_model)
            model_y.fit(x_np[tr], y[tr])
            model_d.fit(x_np[tr], d[tr].astype(int))
            y_hat[te] = model_y.predict(x_np[te])
            if hasattr(model_d, "predict_proba"):
                d_pred = model_d.predict_proba(x_np[te])[:, 1]
            else:
                d_pred = model_d.predict(x_np[te])
            d_hat[te] = np.clip(np.asarray(d_pred, dtype=float), 1e-4, 1.0 - 1e-4)

        y_mse = float(np.mean((y - y_hat) ** 2))
        d_brier = float(np.mean((d - d_hat) ** 2))
        d_tilde = d - d_hat
        balance_terms: List[float] = []
        for j in range(x_np.shape[1]):
            xj = x_np[:, j]
            if np.std(xj) < 1e-8 or np.std(d_tilde) < 1e-8:
                continue
            corr = float(np.corrcoef(d_tilde, xj)[0, 1])
            if np.isfinite(corr):
                balance_terms.append(abs(corr))
        mean_abs_corr = float(np.mean(balance_terms)) if balance_terms else 0.0
        objective = float(y_mse + 0.25 * d_brier + 0.40 * mean_abs_corr)
        return {
            "y_hat": y_hat,
            "d_hat": d_hat,
            "y_mse": y_mse,
            "d_brier": d_brier,
            "mean_abs_corr_dt_x": mean_abs_corr,
            "objective": objective,
        }

    candidates: List[Dict[str, object]] = []
    for y_name, y_model in y_models.items():
        for d_name, d_model in d_models.items():
            fit = _crossfit(y_model, d_model)
            fit["y_model"] = y_name
            fit["d_model"] = d_name
            candidates.append(fit)

    best = sorted(candidates, key=lambda r: float(r["objective"]))[0]
    y_hat = np.asarray(best["y_hat"], dtype=float)
    d_hat = np.asarray(best["d_hat"], dtype=float)
    y_var = float(np.var(y, ddof=1)) if n > 1 else 0.0
    y_r2 = float(1.0 - float(best["y_mse"]) / y_var) if y_var > 1e-10 else float("nan")
    if (np.isfinite(y_r2) and y_r2 < 0.02) or (y_var > 1e-10 and float(best["y_mse"]) >= 0.98 * y_var):
        return {
            "status": "skipped",
            "reason": "weak_outcome_nuisance_fit",
            "n_obs": int(n),
            "n_features": int(x_np.shape[1]),
            "cv_mode": cv_mode,
            "overlap_trimmed_share": float(trimmed_share),
            "selected_y_model": str(best["y_model"]),
            "selected_d_model": str(best["d_model"]),
            "nuisance_objective": float(best["objective"]),
            "nuisance_y_mse": float(best["y_mse"]),
            "nuisance_y_r2": float(y_r2) if np.isfinite(y_r2) else None,
            "nuisance_d_brier": float(best["d_brier"]),
            "nuisance_mean_abs_corr_dt_x": float(best.get("mean_abs_corr_dt_x", 0.0)),
            "legacy_baseline_spec": {
                "coef": float(baseline["coef"]),
                "stderr": float(baseline["stderr"]),
                "t_value": float(baseline["t_value"]),
                "p_value": float(baseline["p_value"]),
                "n_obs": int(baseline["n_obs"]),
                "n_features": int(baseline["n_features"]),
                "cv_mode": str(baseline["cv_mode"]),
                "spec": str(baseline["spec"]),
            },
        }

    y_tilde = y - y_hat
    d_tilde = d - d_hat

    denom_unw = float(np.sum(d_tilde**2))
    if abs(denom_unw) < 1e-10:
        return {"status": "skipped", "reason": "weak_residual_treatment"}

    theta_unw = float(np.sum(d_tilde * y_tilde) / denom_unw)
    d_second_unw = float(np.mean(d_tilde**2))
    psi_unw = (d_tilde * (y_tilde - theta_unw * d_tilde)) / max(d_second_unw, 1e-10)
    se_unw = float(np.sqrt(np.mean(psi_unw**2) / n))
    t_unw = float(theta_unw / max(se_unw, 1e-12))
    n_groups = len(unique_groups)
    p_unw = _t_pvalue(t_unw, max(1, n_groups - 1))

    # Stabilized weights are reported as robustness, not primary estimate.
    w = np.clip(1.0 / np.maximum(d_hat * (1.0 - d_hat), 1e-4), 1.0, 25.0)
    denom_w = float(np.sum(w * (d_tilde**2)))
    theta_w = float(np.sum(w * d_tilde * y_tilde) / max(denom_w, 1e-12))
    d_second_w = float(np.mean(w * (d_tilde**2)))
    psi_w = (w * d_tilde * (y_tilde - theta_w * d_tilde)) / max(d_second_w, 1e-10)
    se_w = float(np.sqrt(np.mean(psi_w**2) / n))
    t_w = float(theta_w / max(se_w, 1e-12))
    p_w = _t_pvalue(t_w, max(1, n_groups - 1))

    advanced = {
        "coef": theta_unw,
        "stderr": se_unw,
        "t_value": t_unw,
        "p_value": p_unw,
        "n_obs": int(n),
        "n_features": int(x_np.shape[1]),
        "cv_mode": cv_mode,
        "overlap_trimmed_share": float(trimmed_share),
        "selected_y_model": str(best["y_model"]),
        "selected_d_model": str(best["d_model"]),
        "nuisance_objective": float(best["objective"]),
        "nuisance_y_mse": float(best["y_mse"]),
        "nuisance_d_brier": float(best["d_brier"]),
        "nuisance_mean_abs_corr_dt_x": float(best.get("mean_abs_corr_dt_x", 0.0)),
        "stabilized_weighted": {
            "coef": theta_w,
            "stderr": se_w,
            "t_value": t_w,
            "p_value": p_w,
        },
    }
    out = {
        "coef": float(advanced["coef"]),
        "stderr": float(advanced["stderr"]),
        "t_value": float(advanced["t_value"]),
        "p_value": float(advanced["p_value"]),
        "n_obs": int(advanced["n_obs"]),
        "n_features": int(advanced["n_features"]),
        "cv_mode": str(advanced["cv_mode"]),
        "primary_spec": "grouped_crossfit_orthogonal",
        "overlap_trimmed_share": float(advanced["overlap_trimmed_share"]),
        "selected_y_model": str(advanced["selected_y_model"]),
        "selected_d_model": str(advanced["selected_d_model"]),
        "nuisance_objective": float(advanced["nuisance_objective"]),
        "nuisance_y_mse": float(advanced["nuisance_y_mse"]),
        "nuisance_y_r2": float(y_r2) if np.isfinite(y_r2) else None,
        "nuisance_d_brier": float(advanced["nuisance_d_brier"]),
        "nuisance_mean_abs_corr_dt_x": float(advanced["nuisance_mean_abs_corr_dt_x"]),
        "legacy_baseline_spec": {
            "coef": float(baseline["coef"]),
            "stderr": float(baseline["stderr"]),
            "t_value": float(baseline["t_value"]),
            "p_value": float(baseline["p_value"]),
            "n_obs": int(baseline["n_obs"]),
            "n_features": int(baseline["n_features"]),
            "cv_mode": str(baseline["cv_mode"]),
            "spec": str(baseline["spec"]),
        },
        "advanced_grouped_spec": advanced,
    }
    LOGGER.info(
        "DML-DID done (%s): coef=%.4f, t=%.3f, p=%.4f, advanced_model=%s/%s, advanced_trim=%.3f",
        outcome,
        out["coef"],
        out["t_value"],
        out["p_value"],
        out["advanced_grouped_spec"]["selected_y_model"],
        out["advanced_grouped_spec"]["selected_d_model"],
        out["advanced_grouped_spec"]["overlap_trimmed_share"],
    )
    return out


def run_dr_did(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    n_splits: int = 3,
    random_state: int = 42,
    overlap_bounds: tuple[float, float] = (0.02, 0.98),
) -> Dict[str, object]:
    """Cross-fitted doubly-robust DID (ATT) on first differences."""
    required = [
        "city_id",
        "year",
        "continent",
        "did_treatment",
        "temperature_mean",
        "precipitation_sum",
        "log_population",
        "inflation",
        "poi_total",
        outcome,
    ]
    miss = [c for c in required if c not in panel.columns]
    if miss:
        return {"status": "skipped", "reason": f"missing_columns_{','.join(miss)}"}

    df = panel[required].dropna().copy()
    if len(df) < 120:
        return {"status": "skipped", "reason": "too_few_observations"}

    df = df.sort_values(["city_id", "year"]).reset_index(drop=True)
    df["y"] = pd.to_numeric(df[outcome], errors="coerce")
    df["y_lag"] = df.groupby("city_id")["y"].shift(1)
    df["dy"] = df["y"] - df["y_lag"]
    for col in [
        "temperature_mean",
        "precipitation_sum",
        "log_population",
        "inflation",
        "poi_total",
    ]:
        df[f"{col}_lag"] = df.groupby("city_id")[col].shift(1)

    lag_cols = [
        "temperature_mean_lag",
        "precipitation_sum_lag",
        "log_population_lag",
        "inflation_lag",
        "poi_total_lag",
    ]
    df = df.dropna(subset=["dy", *lag_cols]).copy()
    if len(df) < 80:
        return {"status": "skipped", "reason": "too_few_observations_after_lagging"}

    y = df["dy"].to_numpy(dtype=float)
    d = df["did_treatment"].astype(int).to_numpy(dtype=int)
    if np.std(d) < 1e-8:
        return {"status": "skipped", "reason": "no_treatment_variation"}

    x_num = df[lag_cols + ["year"]].copy()
    x_cat = pd.get_dummies(df[["continent"]], drop_first=True)
    x_df = pd.concat([x_num, x_cat], axis=1).astype(float)
    x = x_df.to_numpy(dtype=float)
    groups = df["city_id"].astype(str).to_numpy()

    unique_groups = np.unique(groups)
    splits = max(2, min(int(n_splits), int(len(df) // 50), 5))
    if len(unique_groups) >= max(4, splits * 2):
        gkf = GroupKFold(n_splits=splits)
        split_indices = list(gkf.split(x, y, groups))
        cv_mode = "group_kfold_city"
    else:
        kf = KFold(n_splits=splits, shuffle=True, random_state=random_state)
        split_indices = list(kf.split(x))
        cv_mode = "kfold_random"

    ps_models = {
        "logit": LogisticRegression(
            max_iter=1200,
            solver="liblinear",
            C=1.0,
            class_weight="balanced",
            random_state=random_state + 7,
        ),
        "rf_cls": RandomForestClassifier(
            n_estimators=320,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=random_state + 11,
            n_jobs=-1,
        ),
    }
    m0_models = {
        "rf": RandomForestRegressor(
            n_estimators=360,
            max_depth=10,
            min_samples_leaf=5,
            random_state=random_state + 13,
            n_jobs=-1,
        ),
        "hgb": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=420,
            min_samples_leaf=20,
            random_state=random_state + 17,
        ),
        "ridge": Ridge(alpha=1.0, random_state=random_state + 19),
    }

    def _crossfit(ps_model, m0_model) -> Dict[str, object]:
        p_hat = np.zeros(len(df), dtype=float)
        m0_hat = np.zeros(len(df), dtype=float)

        for tr, te in split_indices:
            tr_d = d[tr]
            if np.unique(tr_d).size < 2:
                return {"status": "failed", "reason": "single_class_in_train_fold"}

            model_ps = clone(ps_model)
            model_m0 = clone(m0_model)
            model_ps.fit(x[tr], tr_d)
            p_hat[te] = np.clip(model_ps.predict_proba(x[te])[:, 1], 1e-4, 1.0 - 1e-4)

            tr_ctrl = tr[tr_d == 0]
            if len(tr_ctrl) < 20:
                tr_ctrl = tr
            model_m0.fit(x[tr_ctrl], y[tr_ctrl])
            m0_hat[te] = model_m0.predict(x[te])

        brier = float(np.mean((d.astype(float) - p_hat) ** 2))
        ctrl_mask = d == 0
        control_mse = float(np.mean((y[ctrl_mask] - m0_hat[ctrl_mask]) ** 2)) if int(ctrl_mask.sum()) >= 5 else float("inf")
        moment = float(abs(np.mean((d.astype(float) - p_hat) * (y - m0_hat))))
        low, high = overlap_bounds
        overlap_share = float(np.mean((p_hat >= low) & (p_hat <= high)))
        overlap_penalty = max(0.0, 0.55 - overlap_share)
        objective = float(control_mse + 0.30 * brier + 0.40 * moment + 2.5 * overlap_penalty)
        return {
            "status": "ok",
            "p_hat": p_hat,
            "m0_hat": m0_hat,
            "brier": brier,
            "control_mse": control_mse,
            "moment": moment,
            "overlap_share": overlap_share,
            "objective": objective,
        }

    candidates: List[Dict[str, object]] = []
    for ps_name, ps_model in ps_models.items():
        for m0_name, m0_model in m0_models.items():
            fit = _crossfit(ps_model, m0_model)
            if fit.get("status") != "ok":
                continue
            fit["ps_model"] = ps_name
            fit["m0_model"] = m0_name
            candidates.append(fit)

    if not candidates:
        return {"status": "skipped", "reason": "crossfit_failed_for_all_model_pairs"}

    best = sorted(candidates, key=lambda r: float(r["objective"]))[0]
    p_hat = np.asarray(best["p_hat"], dtype=float)
    m0_hat = np.asarray(best["m0_hat"], dtype=float)

    low, high = overlap_bounds
    keep = (p_hat >= low) & (p_hat <= high)
    if int(keep.sum()) >= max(60, int(0.55 * len(df))):
        trimmed_share = float(1.0 - keep.mean())
        y = y[keep]
        d = d[keep]
        p_hat = p_hat[keep]
        m0_hat = m0_hat[keep]
    else:
        trimmed_share = 0.0

    n = int(len(y))
    if n < 60:
        return {"status": "skipped", "reason": "too_few_observations_after_overlap_trim"}
    if np.std(d) < 1e-8:
        return {"status": "skipped", "reason": "no_treatment_variation_after_overlap_trim"}

    treat_rate = float(np.mean(d))
    if not (1e-4 < treat_rate < 1.0 - 1e-4):
        return {"status": "skipped", "reason": "invalid_treatment_rate"}

    weight_ctrl = p_hat / np.maximum(1.0 - p_hat, 1e-4)
    contrib = (d / treat_rate) * (y - m0_hat) - ((1 - d) / treat_rate) * weight_ctrl * (y - m0_hat)
    att = float(np.mean(contrib))
    psi = contrib - att
    se = float(np.sqrt(np.mean(psi**2) / max(n, 1)))
    t_val = float(att / max(se, 1e-12))
    _n_cl_dr = int(pd.Series(df["continent"]).nunique()) if "continent" in df.columns else 30
    p_val = _t_pvalue(t_val, max(1, _n_cl_dr - 1))

    out = {
        "status": "ok",
        "coef": att,
        "stderr": se,
        "t_value": t_val,
        "p_value": p_val,
        "n_obs": n,
        "treated_obs": int(np.sum(d == 1)),
        "control_obs": int(np.sum(d == 0)),
        "cv_mode": cv_mode,
        "overlap_bounds": [float(low), float(high)],
        "overlap_share_after_trim": float(np.mean((p_hat >= low) & (p_hat <= high))),
        "overlap_trimmed_share": float(trimmed_share),
        "selected_ps_model": str(best["ps_model"]),
        "selected_m0_model": str(best["m0_model"]),
        "n_features": int(x_df.shape[1]),
        "nuisance_brier": float(best["brier"]),
        "nuisance_control_mse": float(best["control_mse"]),
        "nuisance_moment_abs": float(best["moment"]),
    }
    LOGGER.info(
        "DR-DID done (%s): coef=%.4f, t=%.3f, p=%.4f, model=%s/%s, trim=%.3f",
        outcome,
        out["coef"],
        out["t_value"],
        out["p_value"],
        out["selected_ps_model"],
        out["selected_m0_model"],
        out["overlap_trimmed_share"],
    )
    return out


def _did_fe_from_subset(panel: pd.DataFrame) -> Dict[str, float] | Dict[str, str]:
    """Run DID FE on a subset and return either results or a skip reason."""
    if len(panel) < 40:
        return {"status": "skipped", "reason": "too_few_observations"}
    if panel["treated_city"].nunique() < 2:
        return {"status": "skipped", "reason": "no_treatment_variation"}
    if panel["post_policy"].nunique() < 2:
        return {"status": "skipped", "reason": "no_time_variation"}
    if panel["did_treatment"].nunique() < 2:
        return {"status": "skipped", "reason": "no_did_variation"}
    return run_did_two_way_fe(panel)


def _resolve_policy_reference_year(
    panel: pd.DataFrame,
    fallback_year: int = 2020,
    min_pre_periods: int = 3,
    min_post_periods: int = 2,
) -> int:
    """Infer a policy year with enough global pre/post support for dynamic estimators."""
    years = sorted(pd.to_numeric(panel["year"], errors="coerce").dropna().astype(int).unique().tolist())
    if not years:
        return int(fallback_year)

    min_year = int(years[0])
    max_year = int(years[-1])

    earliest_allowed = min_year + int(max(1, min_pre_periods))
    latest_allowed = max_year - int(max(1, min_post_periods)) + 1
    if earliest_allowed > latest_allowed:
        return int(np.clip(fallback_year, min_year, max_year))

    if "treatment_cohort_year" not in panel.columns:
        return int(np.clip(fallback_year, earliest_allowed, latest_allowed))

    cohorts = panel[["city_id", "treatment_cohort_year"]].drop_duplicates("city_id").copy()
    cohort = pd.to_numeric(cohorts["treatment_cohort_year"], errors="coerce")
    cohort = cohort[(cohort >= earliest_allowed) & (cohort <= latest_allowed) & (cohort < 9999)]
    if cohort.empty:
        return int(np.clip(fallback_year, earliest_allowed, latest_allowed))

    return int(np.median(cohort.to_numpy(dtype=float)))


def _assign_sample_income_groups(panel: pd.DataFrame) -> pd.DataFrame:
    """Create city income group labels from baseline GDP per capita quartiles."""
    first_year = int(panel["year"].min())
    baseline = panel.loc[panel["year"] == first_year, ["city_id", "gdp_per_capita"]].copy()
    baseline = baseline.drop_duplicates(subset=["city_id"])

    if len(baseline) < 8:
        baseline["income_group_sample"] = "mixed"
        out = panel.merge(baseline[["city_id", "income_group_sample"]], on="city_id", how="left")
        out["income_group_sample"] = out["income_group_sample"].fillna("mixed")
        return out

    labels = ["low", "lower_middle", "upper_middle", "high"]
    baseline["income_group_sample"] = pd.qcut(
        baseline["gdp_per_capita"].rank(method="first"),
        q=4,
        labels=labels,
    )
    out = panel.merge(baseline[["city_id", "income_group_sample"]], on="city_id", how="left")
    out["income_group_sample"] = out["income_group_sample"].astype(str)
    return out


def run_did_heterogeneity(panel: pd.DataFrame) -> Dict[str, object]:
    """Estimate DID FE heterogeneity by continent and sample income group."""
    by_continent: Dict[str, object] = {}
    for continent in sorted(panel["continent"].dropna().unique().tolist()):
        subset = panel[panel["continent"] == continent].copy()
        by_continent[continent] = _did_fe_from_subset(subset)

    with_income = _assign_sample_income_groups(panel)
    by_income: Dict[str, object] = {}
    for group in ["low", "lower_middle", "upper_middle", "high"]:
        subset = with_income[with_income["income_group_sample"] == group].copy()
        by_income[group] = _did_fe_from_subset(subset)

    return {
        "by_continent": by_continent,
        "by_income_group_sample": by_income,
    }


def run_dynamic_phase_heterogeneity(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    treatment_year: int = 2020,
    min_city_per_group: int = 18,
) -> Dict[str, object]:
    """Estimate DID heterogeneity conditional on AI-inferred dynamic phase signatures."""
    required = {"city_id", "year", "did_treatment", outcome}
    missing = sorted(required.difference(set(panel.columns)))
    if missing:
        return {"status": "skipped", "reason": f"missing_columns_{','.join(missing)}"}

    pulse_path = DATA_OUTPUTS / "pulse_ai_scores.csv"
    if not pulse_path.exists():
        return {"status": "skipped", "reason": "pulse_ai_scores_missing"}

    pulse = pd.read_csv(pulse_path)
    need_pulse = {"city_id", "year", "phase_flow_speed", "phase_label"}
    if not need_pulse.issubset(set(pulse.columns)):
        return {"status": "skipped", "reason": "pulse_phase_columns_missing"}

    years_common = sorted(
        set(pd.to_numeric(panel["year"], errors="coerce").dropna().astype(int).tolist())
        .intersection(set(pd.to_numeric(pulse["year"], errors="coerce").dropna().astype(int).tolist()))
    )
    baseline_candidates = [y for y in years_common if int(y) < int(treatment_year)]
    if not baseline_candidates:
        return {"status": "skipped", "reason": "no_pre_treatment_phase_year_available"}
    baseline_year = int(max(baseline_candidates))

    base = pulse.loc[
        pd.to_numeric(pulse["year"], errors="coerce").fillna(-1).astype(int) == baseline_year,
        ["city_id", "phase_flow_speed", "phase_label", "dynamic_pulse_state", "trajectory_regime"],
    ].copy()
    base["city_id"] = base["city_id"].astype(str)
    base = base.drop_duplicates("city_id")
    base["phase_flow_speed"] = pd.to_numeric(base["phase_flow_speed"], errors="coerce")
    base = base.dropna(subset=["phase_flow_speed"]).copy()
    if len(base) < int(2 * min_city_per_group):
        return {"status": "skipped", "reason": "insufficient_phase_baseline_cities"}

    speed_cut = float(base["phase_flow_speed"].median())
    base["high_phase_instability"] = (base["phase_flow_speed"] >= speed_cut).astype(int)
    fragile_labels = {"centrifugal_fragility", "rotational_transition"}
    base["fragile_phase"] = base["phase_label"].astype(str).isin(fragile_labels).astype(int)

    fragile_share = float(base["fragile_phase"].mean())
    fragile_rule = "phase_label_fragility_set"
    if fragile_share < 0.08 or fragile_share > 0.92:
        q67 = float(base["phase_flow_speed"].quantile(0.67))
        base["fragile_phase"] = (base["phase_flow_speed"] >= q67).astype(int)
        fragile_share = float(base["fragile_phase"].mean())
        fragile_rule = "phase_flow_speed_q67"

    work = panel.copy()
    work["city_id"] = work["city_id"].astype(str)
    work = work.merge(
        base[
            [
                "city_id",
                "phase_flow_speed",
                "phase_label",
                "dynamic_pulse_state",
                "trajectory_regime",
                "high_phase_instability",
                "fragile_phase",
            ]
        ],
        on="city_id",
        how="left",
    )
    if work["high_phase_instability"].isna().all():
        return {"status": "skipped", "reason": "phase_merge_failed"}

    work["high_phase_instability"] = pd.to_numeric(work["high_phase_instability"], errors="coerce").fillna(0).astype(int)
    work["fragile_phase"] = pd.to_numeric(work["fragile_phase"], errors="coerce").fillna(0).astype(int)
    controls = [c for c in ["temperature_mean", "precipitation_sum", "log_population"] if c in work.columns]

    def _subgroup_twfe(flag_col: str, flag_val: int) -> Dict[str, object]:
        sub = work[work[flag_col] == int(flag_val)].copy()
        n_cities = int(sub["city_id"].nunique())
        if n_cities < min_city_per_group:
            return {"status": "skipped", "reason": "too_few_cities_in_subgroup", "n_cities": n_cities}
        if pd.to_numeric(sub["did_treatment"], errors="coerce").fillna(0).nunique() < 2:
            return {"status": "skipped", "reason": "no_did_variation_in_subgroup", "n_cities": n_cities}
        est = _run_twfe_with_custom_treatment(
            sub,
            outcome=outcome,
            did_col="did_treatment",
            controls=controls,
            include_treated_trend=("treated_city" in sub.columns),
            min_obs=120,
            min_city_count=max(12, min_city_per_group // 2),
            min_year_count=5,
        )
        if "status" in est:
            return est
        return {
            "status": "ok",
            "group_col": str(flag_col),
            "group_val": int(flag_val),
            "coef": float(est["coef"]),
            "stderr": float(est["stderr"]),
            "t_value": float(est["t_value"]),
            "p_value": float(est["p_value"]),
            "n_obs": int(est["n_obs"]),
            "n_cities": n_cities,
            "treated_share": float(pd.to_numeric(sub["treated_city"], errors="coerce").fillna(0).mean())
            if "treated_city" in sub.columns
            else None,
        }

    def _interaction_twfe(flag_col: str) -> Dict[str, object]:
        cols = ["city_id", "year", "iso3", outcome, "did_treatment", flag_col, *controls]
        if "treated_city" in work.columns:
            cols.append("treated_city")
        cols = [c for c in cols if c in work.columns]
        df = work[cols].dropna().copy()
        if len(df) < 180:
            return {"status": "skipped", "reason": "too_few_observations_for_interaction"}
        if df["city_id"].nunique() < max(20, min_city_per_group):
            return {"status": "skipped", "reason": "too_few_cities_for_interaction"}
        if pd.to_numeric(df["did_treatment"], errors="coerce").fillna(0).nunique() < 2:
            return {"status": "skipped", "reason": "no_did_variation_for_interaction"}

        df[flag_col] = pd.to_numeric(df[flag_col], errors="coerce").fillna(0.0).astype(float)
        df["did_x_group"] = pd.to_numeric(df["did_treatment"], errors="coerce").fillna(0.0).astype(float) * df[flag_col]
        if pd.to_numeric(df["did_x_group"], errors="coerce").fillna(0.0).nunique() < 2:
            return {"status": "skipped", "reason": "no_interaction_variation"}

        regressors = ["did_treatment", "did_x_group", *controls]
        base_year = int(df["year"].min())
        df["group_trend"] = df[flag_col] * (df["year"].astype(float) - float(base_year))
        regressors.append("group_trend")
        if "treated_city" in df.columns:
            df["treated_trend"] = (
                pd.to_numeric(df["treated_city"], errors="coerce").fillna(0.0).astype(float)
                * (df["year"].astype(float) - float(base_year))
            )
            regressors.append("treated_trend")

        tw = _two_way_within(df, [outcome, *regressors])
        tw["iso3"] = df["iso3"].values if "iso3" in df.columns else df["city_id"].astype(str).to_numpy()
        y = tw[f"{outcome}_tw"].to_numpy(dtype=float)
        x_cols = [f"{c}_tw" for c in regressors]
        x = np.column_stack([tw[c].to_numpy(dtype=float) for c in x_cols])
        names = list(regressors)
        clusters = tw["iso3"].astype(str).to_numpy()

        try:
            res, n_clusters = _ols_cluster_hc1(y, x, names, clusters)
            stderr_type = "cluster_iso3_cr1" if n_clusters >= 2 else "hc1_fallback_single_cluster"
        except Exception:  # noqa: BLE001
            res = _ols_hc1(y, x, names)
            n_clusters = int(pd.Series(clusters).nunique())
            stderr_type = "hc1_fallback_error"

        i_main = names.index("did_treatment")
        i_int = names.index("did_x_group")
        t_main = float(res.t_value[i_main])
        t_int = float(res.t_value[i_int])
        coef_main = float(res.coef[i_main])
        coef_int = float(res.coef[i_int])
        return {
            "status": "ok",
            "group_col": str(flag_col),
            "coef_main": coef_main,
            "stderr_main": float(res.stderr[i_main]),
            "t_value_main": t_main,
            "p_value_main": _t_pvalue(t_main, max(1, n_clusters - 1)),
            "coef_interaction": coef_int,
            "stderr_interaction": float(res.stderr[i_int]),
            "t_value_interaction": t_int,
            "p_value_interaction": _t_pvalue(t_int, max(1, n_clusters - 1)),
            "implied_effect_group0": coef_main,
            "implied_effect_group1": float(coef_main + coef_int),
            "n_obs": int(res.n_obs),
            "n_cities": int(df["city_id"].nunique()),
            "stderr_type": stderr_type,
            "n_clusters": int(n_clusters),
        }

    subgroup = {
        "high_phase_instability_0": _subgroup_twfe("high_phase_instability", 0),
        "high_phase_instability_1": _subgroup_twfe("high_phase_instability", 1),
        "fragile_phase_0": _subgroup_twfe("fragile_phase", 0),
        "fragile_phase_1": _subgroup_twfe("fragile_phase", 1),
    }
    interactions = {
        "high_phase_instability": _interaction_twfe("high_phase_instability"),
        "fragile_phase": _interaction_twfe("fragile_phase"),
    }

    label_counts = base["phase_label"].astype(str).value_counts()
    phase_rows: List[Dict[str, object]] = []
    for label, n_city in label_counts.items():
        if int(n_city) < min_city_per_group:
            continue
        label_flag = f"phase_flag__{str(label)}"
        tmp = work.copy()
        tmp[label_flag] = (tmp["phase_label"].astype(str) == str(label)).astype(int)
        work_saved = work
        work = tmp
        est = _interaction_twfe(label_flag)
        work = work_saved
        est["phase_label"] = str(label)
        est["phase_city_count"] = int(n_city)
        est["phase_city_share"] = float(int(n_city) / max(1, len(base)))
        phase_rows.append(est)

    phase_rows = sorted(
        phase_rows,
        key=lambda r: abs(float(r.get("coef_interaction"))) if r.get("status") == "ok" else -1.0,
        reverse=True,
    )
    phase_rows_top = phase_rows[:8]

    out_rows: List[Dict[str, object]] = []
    for name, est in subgroup.items():
        row = {"analysis_type": "subgroup_did", "group": name}
        row.update(est)
        out_rows.append(row)
    for name, est in interactions.items():
        row = {"analysis_type": "interaction_did", "group": name}
        row.update(est)
        out_rows.append(row)
    for est in phase_rows_top:
        row = {"analysis_type": "phase_label_interaction", "group": est.get("phase_label")}
        row.update(est)
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    safe_outcome = str(outcome).replace(":", "_").replace("/", "_")
    table_path = DATA_OUTPUTS / f"dynamic_phase_heterogeneity_{safe_outcome}.csv"
    out_df.to_csv(table_path, index=False)

    return {
        "status": "ok",
        "outcome": str(outcome),
        "treatment_year_reference": int(treatment_year),
        "baseline_year": int(baseline_year),
        "phase_flow_speed_median": float(speed_cut),
        "high_phase_instability_share": float(base["high_phase_instability"].mean()),
        "fragile_phase_share": float(fragile_share),
        "fragile_phase_rule": fragile_rule,
        "subgroup_effects": subgroup,
        "interaction_effects": interactions,
        "phase_label_interactions_top": phase_rows_top,
        "table_file": str(table_path),
    }


def run_dynamic_phase_rule_sensitivity(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    treatment_year: int = 2020,
    min_city_per_group: int = 18,
) -> Dict[str, object]:
    """Robustness check for AI phase-rule definitions in interaction DID."""
    required = {"city_id", "year", "did_treatment", outcome}
    missing = sorted(required.difference(set(panel.columns)))
    if missing:
        return {"status": "skipped", "reason": f"missing_columns_{','.join(missing)}"}

    pulse_path = DATA_OUTPUTS / "pulse_ai_scores.csv"
    if not pulse_path.exists():
        return {"status": "skipped", "reason": "pulse_ai_scores_missing"}
    pulse = pd.read_csv(pulse_path)
    need_pulse = {"city_id", "year", "phase_flow_speed", "phase_label"}
    if not need_pulse.issubset(set(pulse.columns)):
        return {"status": "skipped", "reason": "pulse_phase_columns_missing"}

    years_common = sorted(
        set(pd.to_numeric(panel["year"], errors="coerce").dropna().astype(int).unique().tolist())
        .intersection(set(pd.to_numeric(pulse["year"], errors="coerce").dropna().astype(int).unique().tolist()))
    )
    baseline_candidates = [y for y in years_common if int(y) < int(treatment_year)]
    if not baseline_candidates:
        return {"status": "skipped", "reason": "no_pre_treatment_phase_year_available"}
    baseline_year = int(max(baseline_candidates))

    pulse_cols = ["city_id", "phase_flow_speed", "phase_label"]
    if "phase_vorticity_bin" in pulse.columns:
        pulse_cols.append("phase_vorticity_bin")
    base = pulse.loc[
        pd.to_numeric(pulse["year"], errors="coerce").fillna(-1).astype(int) == baseline_year,
        pulse_cols,
    ].copy()
    base["city_id"] = base["city_id"].astype(str)
    base = base.drop_duplicates("city_id")
    base["phase_flow_speed"] = pd.to_numeric(base["phase_flow_speed"], errors="coerce")
    base = base.dropna(subset=["phase_flow_speed"]).copy()
    if len(base) < int(2 * min_city_per_group):
        return {"status": "skipped", "reason": "insufficient_phase_baseline_cities"}

    work = panel.copy()
    work["city_id"] = work["city_id"].astype(str)
    work = work.merge(base, on="city_id", how="left")
    controls = [c for c in ["temperature_mean", "precipitation_sum", "log_population"] if c in work.columns]

    def _run_rule_interaction(flag_map: Dict[str, int], rule_name: str, share: float) -> Dict[str, object]:
        tmp = work.copy()
        tmp["phase_rule_flag"] = tmp["city_id"].map(flag_map).fillna(0).astype(float)
        city_flag = tmp[["city_id", "phase_rule_flag"]].drop_duplicates("city_id")
        n_flag_1 = int(np.sum(city_flag["phase_rule_flag"] >= 0.5))
        n_flag_0 = int(np.sum(city_flag["phase_rule_flag"] < 0.5))
        if min(n_flag_1, n_flag_0) < min_city_per_group:
            return {
                "status": "skipped",
                "reason": "too_few_cities_in_rule_group",
                "rule_name": rule_name,
                "flag_share": float(share),
                "n_flag_1_cities": n_flag_1,
                "n_flag_0_cities": n_flag_0,
            }

        cols = ["city_id", "year", "iso3", outcome, "did_treatment", "phase_rule_flag", *controls]
        if "treated_city" in tmp.columns:
            cols.append("treated_city")
        df = tmp[[c for c in cols if c in tmp.columns]].dropna().copy()
        if len(df) < 180:
            return {"status": "skipped", "reason": "too_few_observations", "rule_name": rule_name}
        if df["city_id"].nunique() < max(20, min_city_per_group):
            return {"status": "skipped", "reason": "too_few_cities", "rule_name": rule_name}
        if pd.to_numeric(df["did_treatment"], errors="coerce").fillna(0).nunique() < 2:
            return {"status": "skipped", "reason": "no_did_variation", "rule_name": rule_name}

        df["did_treatment"] = pd.to_numeric(df["did_treatment"], errors="coerce").fillna(0.0).astype(float)
        df["phase_rule_flag"] = pd.to_numeric(df["phase_rule_flag"], errors="coerce").fillna(0.0).astype(float)
        df["did_x_rule"] = df["did_treatment"] * df["phase_rule_flag"]
        if pd.to_numeric(df["did_x_rule"], errors="coerce").fillna(0.0).nunique() < 2:
            return {"status": "skipped", "reason": "no_interaction_variation", "rule_name": rule_name}

        regressors = ["did_treatment", "did_x_rule", *controls]
        base_year = int(df["year"].min())
        df["rule_trend"] = df["phase_rule_flag"] * (df["year"].astype(float) - float(base_year))
        regressors.append("rule_trend")
        if "treated_city" in df.columns:
            df["treated_trend"] = (
                pd.to_numeric(df["treated_city"], errors="coerce").fillna(0.0).astype(float)
                * (df["year"].astype(float) - float(base_year))
            )
            regressors.append("treated_trend")

        tw = _two_way_within(df, [outcome, *regressors])
        tw["iso3"] = df["iso3"].values
        y = tw[f"{outcome}_tw"].to_numpy(dtype=float)
        x = np.column_stack([tw[f"{c}_tw"].to_numpy(dtype=float) for c in regressors])
        names = list(regressors)
        clusters = tw["iso3"].astype(str).to_numpy()
        try:
            res, n_clusters = _ols_cluster_hc1(y, x, names, clusters)
            stderr_type = "cluster_iso3_cr1" if n_clusters >= 2 else "hc1_fallback_single_cluster"
        except Exception:  # noqa: BLE001
            res = _ols_hc1(y, x, names)
            n_clusters = int(pd.Series(clusters).nunique())
            stderr_type = "hc1_fallback_error"

        i_main = names.index("did_treatment")
        i_int = names.index("did_x_rule")
        t_main = float(res.t_value[i_main])
        t_int = float(res.t_value[i_int])
        coef_main = float(res.coef[i_main])
        coef_int = float(res.coef[i_int])
        return {
            "status": "ok",
            "rule_name": str(rule_name),
            "flag_share": float(share),
            "n_flag_1_cities": n_flag_1,
            "n_flag_0_cities": n_flag_0,
            "coef_main": coef_main,
            "stderr_main": float(res.stderr[i_main]),
            "t_value_main": t_main,
            "p_value_main": _t_pvalue(t_main, max(1, n_clusters - 1)),
            "coef_interaction": coef_int,
            "stderr_interaction": float(res.stderr[i_int]),
            "t_value_interaction": t_int,
            "p_value_interaction": _t_pvalue(t_int, max(1, n_clusters - 1)),
            "implied_effect_group0": coef_main,
            "implied_effect_group1": float(coef_main + coef_int),
            "n_obs": int(res.n_obs),
            "n_cities": int(df["city_id"].nunique()),
            "stderr_type": stderr_type,
            "n_clusters": int(n_clusters),
        }

    flow = pd.to_numeric(base["phase_flow_speed"], errors="coerce")
    phase_label = base["phase_label"].astype(str)
    fragile_labels = {"centrifugal_fragility", "rotational_transition"}
    rules: List[tuple[str, pd.Series]] = [
        ("phase_label_fragility_set", phase_label.isin(fragile_labels)),
        ("phase_flow_speed_q60", flow >= float(flow.quantile(0.60))),
        ("phase_flow_speed_q67", flow >= float(flow.quantile(0.67))),
        ("phase_flow_speed_q75", flow >= float(flow.quantile(0.75))),
    ]
    if "phase_vorticity_bin" in base.columns:
        vort = pd.to_numeric(base["phase_vorticity_bin"], errors="coerce").abs()
        if int(vort.notna().sum()) >= int(2 * min_city_per_group):
            rules.extend(
                [
                    ("phase_vorticity_abs_q67", vort >= float(vort.quantile(0.67))),
                    ("phase_vorticity_abs_q75", vort >= float(vort.quantile(0.75))),
                    (
                        "phase_flow_x_vorticity_q67",
                        (flow >= float(flow.quantile(0.67))) & (vort >= float(vort.quantile(0.67))),
                    ),
                ]
            )

    seen: set[str] = set()
    rows: List[Dict[str, object]] = []
    for name, cond in rules:
        if name in seen:
            continue
        seen.add(name)
        flag = pd.Series(cond, index=base.index).fillna(False).astype(int)
        share = float(flag.mean())
        if share < 0.08 or share > 0.92:
            rows.append(
                {
                    "status": "skipped",
                    "reason": "extreme_group_share",
                    "rule_name": name,
                    "flag_share": share,
                    "n_flag_1_cities": int(np.sum(flag == 1)),
                    "n_flag_0_cities": int(np.sum(flag == 0)),
                }
            )
            continue
        flag_map = dict(zip(base["city_id"].astype(str), flag.astype(int), strict=False))
        rows.append(_run_rule_interaction(flag_map, name, share))

    out_df = pd.DataFrame(rows)
    safe_outcome = str(outcome).replace(":", "_").replace("/", "_")
    table_path = DATA_OUTPUTS / f"dynamic_phase_rule_sensitivity_{safe_outcome}.csv"
    out_df.to_csv(table_path, index=False)
    if out_df.empty:
        return {"status": "skipped", "reason": "no_rule_estimates", "table_file": str(table_path)}

    ok = out_df[out_df["status"].astype(str) == "ok"].copy()
    if ok.empty:
        return {
            "status": "skipped",
            "reason": "no_valid_rule_interactions",
            "rules_considered": int(len(out_df)),
            "table_file": str(table_path),
        }

    ok["coef_interaction"] = pd.to_numeric(ok["coef_interaction"], errors="coerce")
    ok["t_value_interaction"] = pd.to_numeric(ok["t_value_interaction"], errors="coerce")
    ok["p_value_interaction"] = pd.to_numeric(ok["p_value_interaction"], errors="coerce")
    ok["abs_t_interaction"] = np.abs(ok["t_value_interaction"])
    ok = ok.sort_values(["p_value_interaction", "abs_t_interaction"], ascending=[True, False]).reset_index(drop=True)
    best = ok.iloc[0]

    ref = ok[ok["rule_name"].astype(str) == "phase_label_fragility_set"].copy()
    ref_coef = float(ref.iloc[0]["coef_interaction"]) if not ref.empty else float(best["coef_interaction"])
    sign_consistency = float(
        np.mean(
            np.sign(pd.to_numeric(ok["coef_interaction"], errors="coerce").to_numpy(dtype=float))
            == np.sign(ref_coef)
        )
    )

    return {
        "status": "ok",
        "outcome": str(outcome),
        "treatment_year_reference": int(treatment_year),
        "baseline_year": int(baseline_year),
        "rules_considered": int(len(out_df)),
        "rules_ok": int(len(ok)),
        "best_rule": str(best.get("rule_name")),
        "best_interaction_coef": float(best.get("coef_interaction")),
        "best_interaction_p_value": float(best.get("p_value_interaction")),
        "best_interaction_abs_t": float(best.get("abs_t_interaction")),
        "reference_rule": "phase_label_fragility_set" if not ref.empty else str(best.get("rule_name")),
        "reference_interaction_coef": float(ref_coef),
        "sign_consistency_share_vs_reference": sign_consistency,
        "share_rules_p_lt_0_10": float(np.mean(ok["p_value_interaction"].to_numpy(dtype=float) < 0.10)),
        "share_rules_p_lt_0_05": float(np.mean(ok["p_value_interaction"].to_numpy(dtype=float) < 0.05)),
        "table_file": str(table_path),
        "rows": ok.where(pd.notna(ok), None).to_dict(orient="records"),
    }


def run_mechanism_decomposition(
    panel: pd.DataFrame,
    treatment_col: str = "did_treatment",
    outcome_ref: str = "composite_index",
    track_label: str = "main",
) -> Dict[str, object]:
    """Decompose policy effect across admissible raw observed mechanism channels."""
    if treatment_col not in panel.columns:
        return {"status": "skipped", "reason": f"missing_column_{treatment_col}"}
    requested_reference_outcome = str(outcome_ref)
    reference_outcome = requested_reference_outcome
    try:
        _assert_valid_causal_outcome(reference_outcome)
    except ValueError:
        reference_outcome = _resolve_primary_causal_outcome(panel)
    if reference_outcome not in panel.columns:
        return {"status": "skipped", "reason": f"missing_column_{reference_outcome}"}

    channel_specs: List[Dict[str, object]] = [
        {
            "channel": "economic_activity_observed",
            "reporting_weight": 0.45,
            "candidate_outcomes": ["log_viirs_ntl", "viirs_log_mean"],
        },
        {
            "channel": "physical_built_environment_observed",
            "reporting_weight": 0.35,
            "candidate_outcomes": ["physical_built_expansion_primary", "ghsl_built_surface_km2"],
        },
        {
            "channel": "knowledge_capital_observed",
            "reporting_weight": 0.20,
            "candidate_outcomes": ["knowledge_capital_raw"],
        },
    ]
    rows: List[Dict[str, object]] = []
    valid_rows: List[Dict[str, object]] = []
    for spec in channel_specs:
        channel_name = str(spec["channel"])
        weight = float(spec["reporting_weight"])
        candidates = [str(col) for col in spec.get("candidate_outcomes", [])]
        outcome = next(
            (
                col
                for col in candidates
                if col in panel.columns and pd.to_numeric(panel[col], errors="coerce").notna().sum() >= 40
            ),
            None,
        )
        if outcome is None:
            rows.append(
                {
                    "channel": channel_name,
                    "weight_in_composite": float(weight),
                    "status": "skipped",
                    "reason": "no_admissible_observed_channel",
                    "candidate_outcomes": candidates,
                }
            )
            continue
        est = _run_twfe_with_custom_treatment(
            panel,
            outcome=outcome,
            did_col=treatment_col,
            controls=["temperature_mean", "precipitation_sum", "baseline_population_log"],
            include_treated_trend=("treated_city" in panel.columns),
            min_obs=60,
            min_city_count=10,
            min_year_count=4,
        )
        if "status" in est:
            rows.append(
                {
                    "channel": channel_name,
                    "outcome_variable": outcome,
                    "weight_in_composite": float(weight),
                    "status": str(est.get("status", "skipped")),
                    "reason": est.get("reason"),
                }
            )
            continue
        row = {
            "channel": channel_name,
            "outcome_variable": outcome,
            "weight_in_composite": float(weight),
            "status": "ok",
            "coef": float(est["coef"]),
            "stderr": float(est["stderr"]),
            "t_value": float(est["t_value"]),
            "p_value": float(est["p_value"]),
            "weighted_contribution": float(weight * float(est["coef"])),
            "n_obs": int(est["n_obs"]),
        }
        rows.append(row)
        valid_rows.append(row)

    ref_est = _run_twfe_with_custom_treatment(
        panel,
        outcome=reference_outcome,
        did_col=treatment_col,
        controls=["temperature_mean", "precipitation_sum", "baseline_population_log"],
        include_treated_trend=("treated_city" in panel.columns),
        min_obs=60,
        min_city_count=10,
        min_year_count=4,
    )
    if "status" in ref_est:
        ref_payload = {"status": str(ref_est.get("status", "skipped")), "reason": ref_est.get("reason")}
        ref_coef = None
    else:
        ref_coef = float(ref_est["coef"])
        ref_payload = {
            "status": "ok",
            "coef": ref_coef,
            "stderr": float(ref_est["stderr"]),
            "t_value": float(ref_est["t_value"]),
            "p_value": float(ref_est["p_value"]),
            "n_obs": int(ref_est["n_obs"]),
        }

    out_df = pd.DataFrame(rows)
    safe_treat = str(treatment_col).replace(":", "_").replace("/", "_")
    safe_track = str(track_label).replace(":", "_").replace("/", "_")
    csv_path = DATA_OUTPUTS / f"mechanism_decomposition_{safe_track}_{safe_treat}.csv"
    out_df.to_csv(csv_path, index=False)

    implied = float(np.sum([float(r["weighted_contribution"]) for r in valid_rows])) if valid_rows else None
    gap = (ref_coef - implied) if (ref_coef is not None and implied is not None) else None
    total_abs = float(np.sum([abs(float(r["weighted_contribution"])) for r in valid_rows])) if valid_rows else 0.0
    shares = []
    for r in valid_rows:
        contrib = float(r["weighted_contribution"])
        shares.append(
            {
                "channel": str(r["channel"]),
                "contribution_share_abs": float(abs(contrib) / total_abs) if total_abs > 1e-12 else None,
                "weighted_contribution": contrib,
                "p_value": float(r["p_value"]),
            }
        )

    return {
        "status": "ok",
        "treatment_col": str(treatment_col),
        "reference_outcome_requested": requested_reference_outcome,
        "reference_outcome": str(reference_outcome),
        "reference_estimate": ref_payload,
        "implied_composite_from_channels": implied,
        "decomposition_gap_vs_reference": float(gap) if gap is not None else None,
        "channels_ok": int(len(valid_rows)),
        "channel_shares": shares,
        "rows": rows,
        "table_file": str(csv_path),
    }


def _safe_pre_slope(years: pd.Series, values: pd.Series) -> float:
    if len(years) < 2 or len(values) < 2:
        return 0.0
    y = values.to_numpy(dtype=float)
    x = years.to_numpy(dtype=float)
    if np.allclose(y.std(), 0.0):
        return 0.0
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:  # noqa: BLE001
        return 0.0


def _std_mean_diff(table: pd.DataFrame, feature: str, treat_col: str = "treated") -> float:
    t = table.loc[table[treat_col] == 1, feature].to_numpy(dtype=float)
    c = table.loc[table[treat_col] == 0, feature].to_numpy(dtype=float)
    if len(t) < 2 or len(c) < 2:
        return float("nan")
    t_mean = float(np.mean(t))
    c_mean = float(np.mean(c))
    t_var = float(np.var(t, ddof=1))
    c_var = float(np.var(c, ddof=1))
    pooled = max((t_var + c_var) / 2.0, 1e-10)
    return (t_mean - c_mean) / math.sqrt(pooled)


def run_matched_did_with_trend(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    treatment_year: int = 2020,
    placebo_years: List[int] | None = None,
) -> Dict[str, object]:
    """Nearest-neighbor matched DID with treated linear trend as robustness design."""
    needed = {
        "city_id",
        "year",
        "treated_city",
        outcome,
        "temperature_mean",
        "precipitation_sum",
        "log_population",
    }
    if not needed.issubset(set(panel.columns)):
        miss = sorted(needed.difference(set(panel.columns)))
        return {"status": "skipped", "reason": f"missing_columns_{','.join(miss)}"}

    pre = panel[panel["year"] <= (treatment_year - 1)].copy()
    if pre.empty:
        return {"status": "skipped", "reason": "no_pre_period_rows"}

    city_features = (
        pre.groupby("city_id", as_index=False)
        .agg(
            pre_mean=(outcome, "mean"),
            pre_temperature=("temperature_mean", "mean"),
            pre_precipitation=("precipitation_sum", "mean"),
        )
        .copy()
    )

    slope_rows: List[Dict[str, float | str]] = []
    for cid, grp in pre.groupby("city_id"):
        slope_rows.append(
            {
                "city_id": cid,
                "pre_slope": _safe_pre_slope(grp["year"], grp[outcome]),
            }
        )
    slopes = pd.DataFrame(slope_rows)
    city_features = city_features.merge(slopes, on="city_id", how="left")
    city_features = city_features.dropna().copy()

    city_group = panel[["city_id", "treated_city"]].drop_duplicates("city_id").copy()
    city_group["treated"] = city_group["treated_city"].astype(int)
    city_group["cohort_year"] = 9999
    if "treatment_cohort_year" in panel.columns:
        cohort_tbl = panel[["city_id", "treatment_cohort_year"]].drop_duplicates("city_id").copy()
        cohort_tbl["cohort_year"] = (
            pd.to_numeric(cohort_tbl["treatment_cohort_year"], errors="coerce")
            .fillna(9999)
            .astype(int)
        )
        city_group = city_group.drop(columns=["cohort_year"]).merge(
            cohort_tbl[["city_id", "cohort_year"]],
            on="city_id",
            how="left",
        )
        city_group["cohort_year"] = city_group["cohort_year"].fillna(9999).astype(int)

    city_features = city_features.merge(city_group[["city_id", "treated", "cohort_year"]], on="city_id", how="left")
    city_features["treated"] = city_features["treated"].fillna(0).astype(int)
    city_features["cohort_year"] = city_features["cohort_year"].fillna(9999).astype(int)

    grouping_mode = "ever_treated_vs_never"
    cohort_split: Dict[str, int] | None = None
    matched_treatment_year = int(treatment_year)

    treated_tbl = city_features[city_features["treated"] == 1].copy()
    control_tbl = city_features[city_features["treated"] == 0].copy()
    if len(treated_tbl) < 5 or len(control_tbl) < 5:
        finite = city_features[(city_features["cohort_year"] < 9999) & np.isfinite(city_features["cohort_year"])].copy()
        if len(finite) >= 20:
            q_low = float(finite["cohort_year"].quantile(0.35))
            q_high = float(finite["cohort_year"].quantile(0.65))
            early_cut = int(np.floor(q_low))
            late_cut = int(np.ceil(max(q_high, q_low + 1.0)))
            if late_cut <= early_cut:
                late_cut = early_cut + 1

            finite["treated"] = np.where(
                finite["cohort_year"] <= early_cut,
                1,
                np.where(finite["cohort_year"] >= late_cut, 0, np.nan),
            )
            finite = finite.dropna(subset=["treated"]).copy()
            finite["treated"] = finite["treated"].astype(int)

            if finite["treated"].sum() >= 5 and (len(finite) - int(finite["treated"].sum())) >= 5:
                city_features = finite.copy()
                grouping_mode = "early_vs_late_adopters"
                cohort_split = {"early_cut": int(early_cut), "late_cut": int(late_cut)}
                min_pre = int(max(2015, panel["year"].min() + 1))
                max_pre = int(panel["year"].max() - 2)
                matched_treatment_year = int(np.clip(early_cut, min_pre, max_pre))
            else:
                return {
                    "status": "skipped",
                    "reason": "too_few_treated_or_controls_for_matching",
                    "fallback_attempted": "early_vs_late_adopters",
                }
        else:
            return {"status": "skipped", "reason": "too_few_treated_or_controls_for_matching"}

    if placebo_years is None:
        placebo_years = [matched_treatment_year - 3, matched_treatment_year - 2, matched_treatment_year - 1]

    match_features = ["pre_mean", "pre_slope", "pre_temperature", "pre_precipitation"]
    weights = {"pre_mean": 1.0, "pre_slope": 2.4, "pre_temperature": 0.8, "pre_precipitation": 0.6}

    all_mean = city_features[match_features].mean()
    all_std = city_features[match_features].std().replace(0.0, np.nan).fillna(1.0)
    for feat in match_features:
        city_features[f"z_{feat}"] = (city_features[feat] - all_mean[feat]) / all_std[feat]

    x_ps = city_features[match_features].to_numpy(dtype=float)
    y_ps = city_features["treated"].to_numpy(dtype=int)
    try:
        ps_model = LogisticRegression(
            max_iter=1200,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )
        ps_model.fit(x_ps, y_ps)
        pscore = ps_model.predict_proba(x_ps)[:, 1]
    except Exception:  # noqa: BLE001
        pscore = np.full(len(city_features), float(y_ps.mean()))
    city_features["pscore"] = np.clip(pscore, 1e-4, 1.0 - 1e-4)

    treated_z = city_features[city_features["treated"] == 1].copy().sort_values("city_id")
    control_z = city_features[city_features["treated"] == 0].copy().sort_values("city_id")

    # Common-support trimming by propensity-score overlap.
    t_q01 = float(treated_z["pscore"].quantile(0.01))
    t_q99 = float(treated_z["pscore"].quantile(0.99))
    c_q01 = float(control_z["pscore"].quantile(0.01))
    c_q99 = float(control_z["pscore"].quantile(0.99))
    overlap_low = max(t_q01, c_q01)
    overlap_high = min(t_q99, c_q99)

    if overlap_high > overlap_low:
        treated_pool = treated_z[(treated_z["pscore"] >= overlap_low) & (treated_z["pscore"] <= overlap_high)].copy()
        control_pool = control_z[(control_z["pscore"] >= overlap_low) & (control_z["pscore"] <= overlap_high)].copy()
    else:
        treated_pool = treated_z.copy()
        control_pool = control_z.copy()

    if len(treated_pool) < 5 or len(control_pool) < 5:
        treated_pool = treated_z.copy()
        control_pool = control_z.copy()

    def _match_with_caliper(caliper: float) -> List[Dict[str, str]]:
        available_controls = set(control_pool["city_id"].tolist())
        pairs: List[Dict[str, str]] = []
        for row in treated_pool.sort_values("pscore").itertuples(index=False):
            if not available_controls:
                break
            cand = control_pool[control_pool["city_id"].isin(available_controls)].copy()
            if cand.empty:
                break
            cand["ps_gap"] = np.abs(cand["pscore"] - float(row.pscore))
            cand = cand[cand["ps_gap"] <= float(caliper)].copy()
            if cand.empty:
                continue
            dist = 10.0 * cand["ps_gap"].to_numpy(dtype=float)
            for feat in match_features:
                diff = cand[f"z_{feat}"].to_numpy(dtype=float) - float(getattr(row, f"z_{feat}"))
                dist += float(weights[feat]) * diff**2
            best_idx = int(np.argmin(dist))
            control_id = str(cand.iloc[best_idx]["city_id"])
            available_controls.remove(control_id)
            pairs.append({"treated_city_id": str(row.city_id), "control_city_id": control_id})
        return pairs

    def _avg_abs_smd(table: pd.DataFrame) -> float:
        vals: List[float] = []
        for feat in match_features:
            smd = _std_mean_diff(table, feat, treat_col="treated")
            if np.isfinite(smd):
                vals.append(abs(float(smd)))
        return float(np.mean(vals)) if vals else float("inf")

    smd_before = _avg_abs_smd(city_features)
    candidates: List[Dict[str, object]] = []
    for caliper in [0.03, 0.05, 0.08, 0.12, 0.18, 0.25]:
        pairs = _match_with_caliper(caliper)
        if len(pairs) < 5:
            continue
        ids = set(p["treated_city_id"] for p in pairs).union(set(p["control_city_id"] for p in pairs))
        after_tbl = city_features[city_features["city_id"].isin(ids)].copy()
        smd_after = _avg_abs_smd(after_tbl)
        candidates.append(
            {
                "caliper": float(caliper),
                "pairs": pairs,
                "smd_after": float(smd_after),
                "smd_improvement": float(smd_before - smd_after),
            }
        )

    if not candidates:
        return {"status": "skipped", "reason": "no_matches_within_caliper_grid"}

    # Primary objective: balance quality, secondary objective: keep sample size.
    candidates = sorted(candidates, key=lambda r: (float(r["smd_after"]), -len(r["pairs"])))
    best = candidates[0]
    matched_pairs = list(best["pairs"])
    selected_caliper = float(best["caliper"])

    if len(matched_pairs) < 5:
        return {"status": "skipped", "reason": "too_few_pairs_after_matching"}

    pair_df = pd.DataFrame(matched_pairs)
    pair_df.to_csv(DATA_OUTPUTS / "did_matched_pairs.csv", index=False)
    matched_ids = set(pair_df["treated_city_id"]).union(set(pair_df["control_city_id"]))
    treated_ids = set(pair_df["treated_city_id"].astype(str).tolist())
    matched_panel = panel[panel["city_id"].isin(matched_ids)].copy()
    matched_panel["treated_city"] = matched_panel["city_id"].astype(str).isin(treated_ids).astype(int)

    if grouping_mode == "early_vs_late_adopters" and "cohort_year" in city_features.columns:
        cohort_map = city_features.set_index("city_id")["cohort_year"].to_dict()
        control_cohort = matched_panel["city_id"].map(cohort_map).fillna(9999).astype(int)
        keep = (matched_panel["treated_city"] == 1) | (matched_panel["year"] < control_cohort)
        matched_panel = matched_panel[keep].copy()

    matched_panel["did_matched"] = matched_panel["treated_city"].astype(int) * (
        matched_panel["year"] >= matched_treatment_year
    ).astype(int)

    main = _run_twfe_with_custom_treatment(
        matched_panel,
        outcome=outcome,
        did_col="did_matched",
        controls=["temperature_mean", "precipitation_sum", "log_population"],
        include_treated_trend=True,
    )
    if "status" in main:
        return {"status": "skipped", "reason": f"matched_design_failed_{main['reason']}"}

    placebo_rows: List[Dict[str, float]] = []
    for yr in placebo_years:
        tmp = matched_panel.copy()
        tmp["did_tmp"] = tmp["treated_city"].astype(int) * (tmp["year"] >= int(yr)).astype(int)
        plc = _run_twfe_with_custom_treatment(
            tmp,
            outcome=outcome,
            did_col="did_tmp",
            controls=["temperature_mean", "precipitation_sum", "log_population"],
            include_treated_trend=True,
        )
        if "status" in plc:
            continue
        placebo_rows.append(
            {
                "placebo_year": int(yr),
                "coef": float(plc["coef"]),
                "stderr": float(plc["stderr"]),
                "t_value": float(plc["t_value"]),
                "p_value": float(plc["p_value"]),
            }
        )

    before_tbl = city_features[city_features["city_id"].isin(matched_ids)].copy()
    after_tbl = before_tbl.copy()
    balance_rows: List[Dict[str, float]] = []
    for feat in match_features:
        smd_before = _std_mean_diff(city_features, feat, treat_col="treated")
        smd_after = _std_mean_diff(after_tbl, feat, treat_col="treated")
        balance_rows.append(
            {
                "feature": feat,
                "smd_before": float(smd_before),
                "smd_after": float(smd_after),
                "abs_improvement": float(abs(smd_before) - abs(smd_after)),
            }
        )
    balance_df = pd.DataFrame(balance_rows)
    balance_df.to_csv(DATA_OUTPUTS / "did_matched_balance.csv", index=False)

    out = {
        "status": "ok",
        "grouping_mode": grouping_mode,
        "treatment_year": int(matched_treatment_year),
        "cohort_split": cohort_split,
        "matching_caliper": float(selected_caliper),
        "common_support_interval": [float(overlap_low), float(overlap_high)],
        "matched_pairs": int(len(pair_df)),
        "matched_cities": int(len(matched_ids)),
        "matched_treated_group_cities": int(pair_df["treated_city_id"].nunique()),
        "matched_control_group_cities": int(pair_df["control_city_id"].nunique()),
        "coef": float(main["coef"]),
        "stderr": float(main["stderr"]),
        "t_value": float(main["t_value"]),
        "p_value": float(main["p_value"]),
        "n_obs": int(main["n_obs"]),
        "r2_within": float(main["r2_within"]),
        "avg_abs_smd_before": float(np.mean(np.abs(balance_df["smd_before"]))),
        "avg_abs_smd_after": float(np.mean(np.abs(balance_df["smd_after"]))),
        "avg_abs_smd_improvement": float(np.mean(np.abs(balance_df["smd_before"])) - np.mean(np.abs(balance_df["smd_after"]))),
        "placebo": placebo_rows,
    }
    LOGGER.info(
        "Matched DID done (%s, %s): coef=%.4f, t=%.3f, pairs=%s, absSMD %.3f->%.3f",
        outcome,
        grouping_mode,
        out["coef"],
        out["t_value"],
        out["matched_pairs"],
        out["avg_abs_smd_before"],
        out["avg_abs_smd_after"],
    )
    return out


def _infer_treatment_cohort_year(
    panel: pd.DataFrame,
    treatment_year: int = 2020,
    signal_col: str = "capital_formation",
) -> pd.DataFrame:
    """Infer city-level treatment cohort year using sustained signal jump."""
    rows: List[Dict[str, object]] = []
    year_min = int(panel["year"].min())
    year_max = int(panel["year"].max())

    for city_id, grp in panel.groupby("city_id"):
        g = grp.sort_values("year")
        treated = int(g["treated_city"].max()) if "treated_city" in g.columns else 0
        if treated == 0 or signal_col not in g.columns:
            rows.append({"city_id": city_id, "cohort_year": 9999, "cohort_source": "never_treated"})
            continue

        pre_mask = g["year"] <= max(treatment_year - 2, year_min)
        if pre_mask.sum() < 2:
            pre_mask = g["year"] < treatment_year
        baseline = float(g.loc[pre_mask, signal_col].mean()) if pre_mask.any() else float(g[signal_col].mean())
        sig_std = float(g.loc[pre_mask, signal_col].std()) if pre_mask.any() else float(g[signal_col].std())
        if not np.isfinite(sig_std):
            sig_std = 0.0
        threshold = baseline + max(0.40, 0.35 * sig_std)

        years = g["year"].astype(int).to_numpy()
        vals = g[signal_col].astype(float).to_numpy()
        adopt_year = None
        for idx in range(len(years)):
            yr = int(years[idx])
            if yr < treatment_year - 2:
                continue
            ok_now = vals[idx] >= threshold
            ok_next = True if idx + 1 >= len(years) else vals[idx + 1] >= threshold
            if ok_now and ok_next:
                adopt_year = yr
                break

        if adopt_year is None:
            adopt_year = treatment_year
            source = "fallback_default"
        else:
            source = "signal_jump"

        adopt_year = int(min(max(adopt_year, treatment_year), year_max))
        rows.append({"city_id": city_id, "cohort_year": adopt_year, "cohort_source": source})

    cohorts = pd.DataFrame(rows)
    cohorts.to_csv(DATA_OUTPUTS / "staggered_cohort_assignments.csv", index=False)
    return cohorts


def _pretrend_joint_test_mc(pre_df: pd.DataFrame, draws: int = 20000, seed: int = 42) -> Dict[str, float]:
    """Joint pretrend test using Monte Carlo approximation of weighted chi-square."""
    if pre_df.empty:
        return {"joint_stat": float("nan"), "joint_p_value": float("nan"), "dof": 0}

    pre = pre_df.copy()
    pre = pre[np.isfinite(pre["att"]) & np.isfinite(pre["stderr"]) & (pre["stderr"] > 0)]
    if pre.empty:
        return {"joint_stat": float("nan"), "joint_p_value": float("nan"), "dof": 0}

    z = pre["att"].to_numpy(dtype=float) / pre["stderr"].to_numpy(dtype=float)
    w = np.maximum(pre["weighted_n_treat"].to_numpy(dtype=float), 1.0)
    w = w / max(float(w.sum()), 1e-12)
    stat = float(np.sum((w * z) ** 2))

    rng = np.random.default_rng(seed)
    sim_z = rng.normal(size=(draws, len(z)))
    sim_stat = np.sum((sim_z * w[np.newaxis, :]) ** 2, axis=1)
    p_val = float(np.mean(sim_stat >= stat))
    return {"joint_stat": stat, "joint_p_value": p_val, "dof": int(len(z))}


def run_staggered_did(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    treatment_year: int = 2020,
    min_event: int = -4,
    max_event: int = 4,
    cohort_col: str = "treatment_cohort_year",
    strict_pretrend_screen: bool = False,
) -> Dict[str, object]:
    """Approximate staggered DID using cohort-time ATT with not-yet-treated controls."""
    required = {"city_id", "year", outcome, "treated_city"}
    if not required.issubset(set(panel.columns)):
        miss = sorted(required.difference(set(panel.columns)))
        return {"status": "skipped", "reason": f"missing_columns_{','.join(miss)}"}

    df = panel[["city_id", "year", outcome, "treated_city", "capital_formation"]].copy()
    if cohort_col in panel.columns:
        cohorts = panel[["city_id", cohort_col]].drop_duplicates("city_id").rename(columns={cohort_col: "cohort_year"})
        cohorts["cohort_source"] = "provided"
    else:
        cohorts = _infer_treatment_cohort_year(df, treatment_year=treatment_year, signal_col="capital_formation")

    city_level = df[["city_id", "treated_city"]].drop_duplicates("city_id")
    cohorts = city_level.merge(cohorts[["city_id", "cohort_year", "cohort_source"]], on="city_id", how="left")
    cohorts["cohort_year"] = cohorts["cohort_year"].fillna(9999).astype(int)
    cohorts.loc[cohorts["treated_city"] == 0, "cohort_year"] = 9999

    out_df = df.merge(cohorts[["city_id", "cohort_year"]], on="city_id", how="left")
    years = sorted(out_df["year"].astype(int).unique().tolist())
    y_pivot = out_df.pivot_table(index="city_id", columns="year", values=outcome).copy()

    finite_cohorts = sorted(c for c in out_df["cohort_year"].unique().tolist() if c < 9999)
    if len(finite_cohorts) < 2:
        # Fallback: if provided cohort is too concentrated, infer dynamic cohorts from signal jumps.
        inferred = _infer_treatment_cohort_year(out_df, treatment_year=treatment_year, signal_col="capital_formation")
        cohorts = city_level.merge(inferred[["city_id", "cohort_year", "cohort_source"]], on="city_id", how="left")
        cohorts["cohort_year"] = cohorts["cohort_year"].fillna(9999).astype(int)
        cohorts.loc[cohorts["treated_city"] == 0, "cohort_year"] = 9999
        out_df = df.merge(cohorts[["city_id", "cohort_year"]], on="city_id", how="left")
        finite_cohorts = sorted(c for c in out_df["cohort_year"].unique().tolist() if c < 9999)
        if len(finite_cohorts) < 2:
            return {"status": "skipped", "reason": "insufficient_cohort_variation"}

    cohort_rows: List[Dict[str, object]] = []
    for g in finite_cohorts:
        base = int(g - 1)
        if base not in years:
            continue
        treat_ids = cohorts.loc[cohorts["cohort_year"] == g, "city_id"].tolist()
        if len(treat_ids) < 3:
            continue
        for event_time in range(min_event, max_event + 1):
            t = int(g + event_time)
            if t not in years:
                continue
            ctrl_ids = cohorts.loc[cohorts["cohort_year"] > t, "city_id"].tolist()
            if len(ctrl_ids) < 5:
                continue

            treat_sub = y_pivot.loc[y_pivot.index.intersection(treat_ids), [base, t]].dropna()
            ctrl_sub = y_pivot.loc[y_pivot.index.intersection(ctrl_ids), [base, t]].dropna()
            if len(treat_sub) < 3 or len(ctrl_sub) < 5:
                continue

            d_treat = (treat_sub[t] - treat_sub[base]).to_numpy(dtype=float)
            d_ctrl = (ctrl_sub[t] - ctrl_sub[base]).to_numpy(dtype=float)
            att = float(np.mean(d_treat) - np.mean(d_ctrl))
            var_t = float(np.var(d_treat, ddof=1)) if len(d_treat) > 1 else 0.0
            var_c = float(np.var(d_ctrl, ddof=1)) if len(d_ctrl) > 1 else 0.0
            se = float(np.sqrt(max(var_t / len(d_treat), 0.0) + max(var_c / len(d_ctrl), 0.0)))
            t_val = float(att / max(se, 1e-12))
            p_val = _t_pvalue(t_val, max(1, len(d_treat) + len(d_ctrl) - 2))

            cohort_rows.append(
                {
                    "cohort_year": int(g),
                    "year": int(t),
                    "event_time": int(event_time),
                    "att": att,
                    "stderr": se,
                    "t_value": t_val,
                    "p_value": p_val,
                    "n_treat": int(len(d_treat)),
                    "n_ctrl": int(len(d_ctrl)),
                }
            )

    cohort_att = pd.DataFrame(cohort_rows)
    cohort_att.to_csv(DATA_OUTPUTS / "staggered_att_by_cohort_year.csv", index=False)
    if cohort_att.empty:
        return {"status": "skipped", "reason": "no_valid_cohort_time_cells"}

    def _aggregate_event_table(att_df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for ev, sub in att_df.groupby("event_time"):
            w = sub["n_treat"].to_numpy(dtype=float)
            w = w / max(float(w.sum()), 1e-10)
            att = float(np.sum(w * sub["att"].to_numpy(dtype=float)))
            se = float(np.sqrt(np.sum((w**2) * (sub["stderr"].to_numpy(dtype=float) ** 2))))
            t_val = float(att / max(se, 1e-12))
            p_val = _t_pvalue(t_val, max(1, len(sub) - 1))
            rows.append(
                {
                    "event_time": int(ev),
                    "att": att,
                    "stderr": se,
                    "t_value": t_val,
                    "p_value": p_val,
                    "cells": int(len(sub)),
                    "weighted_n_treat": float(sub["n_treat"].sum()),
                    "ci95_low": float(att - 1.96 * se),
                    "ci95_high": float(att + 1.96 * se),
                }
            )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)

    screening_meta: Dict[str, object] = {
        "status": "not_applied",
        "kept_cohorts": [],
        "dropped_cohorts": [],
    }
    cohort_screen = (
        cohort_att[cohort_att["event_time"] <= -2]
        .groupby("cohort_year", as_index=False)
        .agg(
            pre_cells=("event_time", "size"),
            pre_max_abs_t=("t_value", lambda s: float(np.abs(s).max()) if len(s) else float("nan")),
            pre_share_sig=("p_value", lambda s: float(np.mean(s < 0.10)) if len(s) else float("nan")),
        )
        .sort_values("cohort_year")
    )
    cohort_screen.to_csv(DATA_OUTPUTS / "staggered_cohort_pretrend_screening.csv", index=False)

    analysis_att = cohort_att.copy()
    if not cohort_screen.empty:
        keep_mask = (
            (cohort_screen["pre_cells"] >= 1)
            & (cohort_screen["pre_max_abs_t"] <= 5.00)
            & (cohort_screen["pre_share_sig"] <= 0.85)
        )
        kept = cohort_screen.loc[keep_mask, "cohort_year"].astype(int).tolist()
        dropped = cohort_screen.loc[~keep_mask, "cohort_year"].astype(int).tolist()
        if len(kept) >= 1:
            candidate = cohort_att[cohort_att["cohort_year"].isin(kept)].copy()
            candidate_summary = _aggregate_event_table(candidate)
            if not candidate_summary.empty and (candidate_summary["event_time"] >= 0).any():
                analysis_att = candidate
                screening_meta = {
                    "status": "applied",
                    "rule": "keep cohorts with pre_cells>=2, pre_max_abs_t<=2.5, pre_share_sig<=0.5",
                    "kept_cohorts": kept,
                    "dropped_cohorts": dropped,
                    "kept_count": int(len(kept)),
                    "dropped_count": int(len(dropped)),
                }
            else:
                screening_meta = {
                    "status": "fallback_raw",
                    "reason": "screened_sample_invalid_or_no_post_cells",
                    "kept_cohorts": kept,
                    "dropped_cohorts": dropped,
                }
        else:
            screening_meta = {
                "status": "fallback_raw",
                "reason": "insufficient_kept_cohorts",
                "kept_cohorts": kept,
                "dropped_cohorts": dropped,
            }

    if bool(strict_pretrend_screen) and screening_meta.get("status") != "applied":
        return {
            "status": "skipped",
            "reason": "strict_pretrend_screen_no_valid_cohorts",
            "cohort_pretrend_screening": screening_meta,
        }

    event_summary = _aggregate_event_table(analysis_att)
    if event_summary.empty:
        return {"status": "skipped", "reason": "no_aggregated_event_summary"}
    event_summary.to_csv(DATA_OUTPUTS / "staggered_event_time_summary.csv", index=False)

    raw_event_summary = _aggregate_event_table(cohort_att)

    pre = event_summary[event_summary["event_time"] < 0].copy()
    post = event_summary[event_summary["event_time"] >= 0].copy()
    pre_max_abs_t = float(np.abs(pre["t_value"]).max()) if not pre.empty else float("nan")
    pre_share_sig = float(np.mean(pre["p_value"] < 0.10)) if not pre.empty else float("nan")
    pre_joint = _pretrend_joint_test_mc(pre[pre["event_time"] <= -2].copy(), draws=20000, seed=42)
    pre_raw = raw_event_summary[raw_event_summary["event_time"] < 0].copy() if not raw_event_summary.empty else pd.DataFrame()
    pre_joint_raw = _pretrend_joint_test_mc(pre_raw[pre_raw["event_time"] <= -2].copy(), draws=20000, seed=42)
    post_avg_att = float(np.average(post["att"], weights=np.maximum(post["weighted_n_treat"], 1.0))) if not post.empty else 0.0
    post_avg_se = float(np.sqrt(np.mean(post["stderr"] ** 2))) if not post.empty else float("nan")
    post_avg_t = float(post_avg_att / max(post_avg_se, 1e-12)) if np.isfinite(post_avg_se) else float("nan")

    cohort_dist = (
        cohorts[cohorts["cohort_year"] < 9999]
        .groupby("cohort_year", as_index=False)
        .agg(cities=("city_id", "nunique"))
        .sort_values("cohort_year")
    )
    cohort_dist.to_csv(DATA_OUTPUTS / "staggered_cohort_distribution.csv", index=False)

    out = {
        "status": "ok",
        "outcome": outcome,
        "treatment_year_reference": int(treatment_year),
        "cohort_count": int(cohort_dist["cohort_year"].nunique()),
        "treated_city_count": int((cohorts["cohort_year"] < 9999).sum()),
        "event_window": [int(min_event), int(max_event)],
        "post_avg_att": post_avg_att,
        "post_avg_t_value": post_avg_t,
        "post_avg_ci95": [float(post_avg_att - 1.96 * post_avg_se), float(post_avg_att + 1.96 * post_avg_se)]
        if np.isfinite(post_avg_se)
        else [None, None],
        "pretrend_max_abs_t": pre_max_abs_t,
        "pretrend_share_p_lt_0_10": pre_share_sig,
        "pretrend_joint_stat": float(pre_joint["joint_stat"]),
        "pretrend_joint_p_value": float(pre_joint["joint_p_value"]),
        "pretrend_joint_dof": int(pre_joint["dof"]),
        "pretrend_raw_max_abs_t": float(np.abs(pre_raw["t_value"]).max()) if not pre_raw.empty else float("nan"),
        "pretrend_raw_share_p_lt_0_10": float(np.mean(pre_raw["p_value"] < 0.10)) if not pre_raw.empty else float("nan"),
        "pretrend_raw_joint_stat": float(pre_joint_raw["joint_stat"]),
        "pretrend_raw_joint_p_value": float(pre_joint_raw["joint_p_value"]),
        "pretrend_raw_joint_dof": int(pre_joint_raw["dof"]),
        "cohort_pretrend_screening": screening_meta,
        "cohort_distribution": cohort_dist.to_dict(orient="records"),
        "event_time_points": [
            {
                "event_time": int(row["event_time"]),
                "att": float(row["att"]),
                "stderr": float(row["stderr"]),
                "t_value": float(row["t_value"]),
                "p_value": float(row["p_value"]),
                "cells": int(row["cells"]),
                "weighted_n_treat": float(row["weighted_n_treat"]),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
            }
            for row in event_summary.to_dict(orient="records")
        ],
    }
    LOGGER.info(
        "Staggered DID done: post_avg_att=%.4f, pre|max t|=%.3f, cohorts=%s",
        out["post_avg_att"],
        out["pretrend_max_abs_t"] if np.isfinite(out["pretrend_max_abs_t"]) else float("nan"),
        out["cohort_count"],
    )
    return out


def run_not_yet_treated_did(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    treatment_year: int = 2020,
    window_pre: int = 2,
    window_post: int = 2,
) -> Dict[str, object]:
    """Cohort-wise not-yet-treated DID with controls and treated-group trend."""
    required = {"city_id", "year", outcome, "treated_city", "temperature_mean", "precipitation_sum", "log_population", "capital_formation"}
    if not required.issubset(set(panel.columns)):
        miss = sorted(required.difference(set(panel.columns)))
        return {"status": "skipped", "reason": f"missing_columns_{','.join(miss)}"}

    base = panel[
        [
            "city_id",
            "year",
            "iso3",
            outcome,
            "treated_city",
            "temperature_mean",
            "precipitation_sum",
            "log_population",
            "capital_formation",
        ]
    ].copy()
    city_level = base[["city_id", "treated_city"]].drop_duplicates("city_id")
    if "treatment_cohort_year" in panel.columns:
        cohort_provided = panel[["city_id", "treatment_cohort_year"]].drop_duplicates("city_id").copy()
        cohort_provided = cohort_provided.rename(columns={"treatment_cohort_year": "cohort_year"})
        cohort_provided["cohort_year"] = pd.to_numeric(cohort_provided["cohort_year"], errors="coerce").fillna(9999).astype(int)
        cohorts = city_level.merge(cohort_provided, on="city_id", how="left")
    else:
        inferred = _infer_treatment_cohort_year(base, treatment_year=treatment_year, signal_col="capital_formation")
        cohorts = city_level.merge(inferred[["city_id", "cohort_year"]], on="city_id", how="left")

    cohorts["cohort_year"] = cohorts["cohort_year"].fillna(9999).astype(int)
    cohorts["cohort_year"] = cohorts["cohort_year"].clip(lower=int(base["year"].min()), upper=9999)
    cohorts.loc[cohorts["treated_city"] == 0, "cohort_year"] = 9999

    finite_cohorts = sorted(c for c in cohorts["cohort_year"].unique().tolist() if c < 9999)
    if len(finite_cohorts) < 2:
        inferred = _infer_treatment_cohort_year(base, treatment_year=treatment_year, signal_col="capital_formation")
        cohorts = city_level.merge(inferred[["city_id", "cohort_year"]], on="city_id", how="left")
        cohorts["cohort_year"] = cohorts["cohort_year"].fillna(9999).astype(int)
        cohorts.loc[cohorts["treated_city"] == 0, "cohort_year"] = 9999
        finite_cohorts = sorted(c for c in cohorts["cohort_year"].unique().tolist() if c < 9999)
        if len(finite_cohorts) < 2:
            return {"status": "skipped", "reason": "insufficient_cohort_variation"}

    rows: List[Dict[str, float | int]] = []
    for g in finite_cohorts:
        treat_ids = cohorts.loc[cohorts["cohort_year"] == g, "city_id"].tolist()
        ctrl_ids = cohorts.loc[cohorts["cohort_year"] > g, "city_id"].tolist()
        if len(treat_ids) < 2 or len(ctrl_ids) < 4:
            continue

        sub = base[base["city_id"].isin(set(treat_ids).union(set(ctrl_ids)))].copy()
        sub = sub[(sub["year"] >= (g - window_pre)) & (sub["year"] <= (g + window_post))].copy()
        if sub.empty:
            continue
        sub["treated_city"] = sub["city_id"].isin(treat_ids).astype(int)
        sub["did_nyt"] = sub["treated_city"] * (sub["year"] >= g).astype(int)

        est = _run_twfe_with_custom_treatment(
            sub,
            outcome=outcome,
            did_col="did_nyt",
            controls=["temperature_mean", "precipitation_sum", "log_population"],
            include_treated_trend=True,
            min_obs=36,
            min_city_count=6,
            min_year_count=3,
        )
        if "status" in est:
            continue

        plc_sub = sub.copy()
        plc_sub["did_nyt_placebo"] = plc_sub["treated_city"] * (plc_sub["year"] >= (g - 1)).astype(int)
        plc = _run_twfe_with_custom_treatment(
            plc_sub,
            outcome=outcome,
            did_col="did_nyt_placebo",
            controls=["temperature_mean", "precipitation_sum", "log_population"],
            include_treated_trend=True,
            min_obs=36,
            min_city_count=6,
            min_year_count=3,
        )
        plc_t = float(plc["t_value"]) if "status" not in plc else float("nan")
        plc_p = float(plc["p_value"]) if "status" not in plc else float("nan")

        rows.append(
            {
                "cohort_year": int(g),
                "coef": float(est["coef"]),
                "stderr": float(est["stderr"]),
                "t_value": float(est["t_value"]),
                "p_value": float(est["p_value"]),
                "n_obs": int(est["n_obs"]),
                "n_treat_cities": int(len(treat_ids)),
                "n_ctrl_cities": int(len(ctrl_ids)),
                "placebo_t_value": plc_t,
                "placebo_p_value": plc_p,
            }
        )

    by_cohort = pd.DataFrame(rows)
    if by_cohort.empty:
        by_cohort.to_csv(DATA_OUTPUTS / "not_yet_treated_did_by_cohort.csv", index=False)
        return {"status": "skipped", "reason": "no_valid_cohort_regressions"}
    by_cohort = by_cohort.sort_values("cohort_year").reset_index(drop=True)
    by_cohort.to_csv(DATA_OUTPUTS / "not_yet_treated_did_by_cohort.csv", index=False)

    w = np.maximum(by_cohort["n_treat_cities"].to_numpy(dtype=float), 1.0)
    w = w / max(float(w.sum()), 1e-12)
    att = float(np.sum(w * by_cohort["coef"].to_numpy(dtype=float)))
    se = float(np.sqrt(np.sum((w**2) * (by_cohort["stderr"].to_numpy(dtype=float) ** 2))))
    t_val = float(att / max(se, 1e-12))
    p_val = _t_pvalue(t_val, max(1, len(by_cohort) - 1))

    placebo = by_cohort["placebo_p_value"].dropna()
    robust = by_cohort[
        (by_cohort["placebo_p_value"].fillna(0.0) >= 0.10) & (by_cohort["n_treat_cities"] >= 4)
    ].copy()
    if robust.empty:
        robust_att = float("nan")
        robust_se = float("nan")
        robust_t = float("nan")
        robust_p = float("nan")
        robust_ci = [None, None]
    else:
        rw = np.maximum(robust["n_treat_cities"].to_numpy(dtype=float), 1.0)
        rw = rw / max(float(rw.sum()), 1e-12)
        robust_att = float(np.sum(rw * robust["coef"].to_numpy(dtype=float)))
        robust_se = float(np.sqrt(np.sum((rw**2) * (robust["stderr"].to_numpy(dtype=float) ** 2))))
        robust_t = float(robust_att / max(robust_se, 1e-12))
        robust_p = _t_pvalue(robust_t, max(1, len(robust) - 1))
        robust_ci = [float(robust_att - 1.96 * robust_se), float(robust_att + 1.96 * robust_se)]

    out = {
        "status": "ok",
        "outcome": outcome,
        "cohort_count": int(by_cohort["cohort_year"].nunique()),
        "att_weighted": att,
        "stderr_weighted": se,
        "t_value_weighted": t_val,
        "p_value_weighted": p_val,
        "ci95_weighted": [float(att - 1.96 * se), float(att + 1.96 * se)],
        "placebo_share_p_lt_0_10": float(np.mean(placebo < 0.10)) if not placebo.empty else float("nan"),
        "placebo_max_abs_t": float(np.abs(by_cohort["placebo_t_value"]).max())
        if by_cohort["placebo_t_value"].notna().any()
        else float("nan"),
        "robust_placebo_pass_cohorts": int(len(robust)),
        "robust_att_weighted": robust_att,
        "robust_stderr_weighted": robust_se,
        "robust_t_value_weighted": robust_t,
        "robust_p_value_weighted": robust_p,
        "robust_ci95_weighted": robust_ci,
        "by_cohort": by_cohort.to_dict(orient="records"),
    }
    LOGGER.info(
        "NYT DID done: att=%.4f, t=%.3f, placebo<0.1=%.3f, cohorts=%s",
        out["att_weighted"],
        out["t_value_weighted"],
        out["placebo_share_p_lt_0_10"] if np.isfinite(out["placebo_share_p_lt_0_10"]) else float("nan"),
        out["cohort_count"],
    )
    return out


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _prefer_finite(primary: object, fallback: object) -> float | None:
    out = _safe_float(primary)
    if out is not None:
        return out
    return _safe_float(fallback)


def _build_timing_ready_panel(
    panel: pd.DataFrame,
    *,
    treatment_year: int,
    window_pre: int = 2,
    window_post: int = 2,
    min_treated_cities_per_cohort: int = 3,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    required = {"city_id", "year", "treated_city", "treatment_cohort_year"}
    if not required.issubset(set(panel.columns)):
        miss = sorted(required.difference(set(panel.columns)))
        return panel, {"status": "fallback_raw", "reason": f"missing_columns_{','.join(miss)}"}

    out = panel.copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    if years.dropna().empty:
        return panel, {"status": "fallback_raw", "reason": "invalid_year_support"}
    min_year = int(years.min())
    max_year = int(years.max())
    lower_cohort = int(min_year + max(int(window_pre), 1))
    upper_cohort = int(max_year - max(int(window_post), 1))
    if lower_cohort > upper_cohort:
        return panel, {"status": "fallback_raw", "reason": "no_supported_cohort_window"}

    city_map = out[["city_id", "treated_city", "treatment_cohort_year"]].drop_duplicates("city_id").copy()
    city_map["treated_city"] = pd.to_numeric(city_map["treated_city"], errors="coerce").fillna(0.0).astype(int)
    city_map["cohort_year"] = pd.to_numeric(city_map["treatment_cohort_year"], errors="coerce").fillna(9999).astype(int)
    city_map.loc[city_map["treated_city"] == 0, "cohort_year"] = 9999

    cohort_counts = (
        city_map[city_map["cohort_year"] < 9999]
        .groupby("cohort_year", as_index=False)
        .agg(treated_cities=("city_id", "nunique"))
        .sort_values("cohort_year")
    )
    keep_cohorts = (
        cohort_counts[
            (cohort_counts["cohort_year"] >= lower_cohort)
            & (cohort_counts["cohort_year"] <= upper_cohort)
            & (cohort_counts["treated_cities"] >= int(max(min_treated_cities_per_cohort, 1)))
        ]["cohort_year"]
        .astype(int)
        .tolist()
    )
    if not keep_cohorts:
        return panel, {
            "status": "fallback_raw",
            "reason": "no_supported_timing_ready_cohorts",
            "supported_window": [lower_cohort, upper_cohort],
            "min_treated_cities_per_cohort": int(max(min_treated_cities_per_cohort, 1)),
            "cohort_distribution": cohort_counts.to_dict(orient="records"),
        }

    keep_city_map = city_map[city_map["cohort_year"].isin(set(keep_cohorts))][["city_id", "cohort_year"]].copy()
    keep_ids = set(keep_city_map["city_id"].astype(str).tolist())
    cohort_dict = keep_city_map.set_index("city_id")["cohort_year"].astype(int).to_dict()

    unsupported = city_map[
        (city_map["treated_city"] == 1)
        & (city_map["cohort_year"] < 9999)
        & (~city_map["cohort_year"].isin(set(keep_cohorts)))
    ][["city_id", "cohort_year"]].copy()
    unsupported_ids = set(unsupported["city_id"].astype(str).tolist())
    unsupported_cohort = unsupported.set_index("city_id")["cohort_year"].astype(int).to_dict()

    out["city_id"] = out["city_id"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    orig_rows = int(len(out))
    if unsupported_ids:
        own_cohort = out["city_id"].map(unsupported_cohort)
        keep_row_mask = (~out["city_id"].isin(unsupported_ids)) | (out["year"] < own_cohort.fillna(9999).astype(int))
        dropped_rows = int((~keep_row_mask).sum())
        out = out.loc[keep_row_mask].copy()
    else:
        dropped_rows = 0

    out["treated_city"] = out["city_id"].isin(keep_ids).astype(int)
    out["treatment_cohort_year"] = out["city_id"].map(cohort_dict).fillna(9999).astype(int)
    out["post_policy"] = ((out["year"] >= out["treatment_cohort_year"]) & (out["treated_city"] == 1)).astype(int)
    out["did_treatment"] = out["treated_city"] * out["post_policy"]
    return out, {
        "status": "ok",
        "rule": "keep supported cohorts as treated and censor unsupported treated units at their own adoption year",
        "supported_window": [lower_cohort, upper_cohort],
        "min_treated_cities_per_cohort": int(max(min_treated_cities_per_cohort, 1)),
        "kept_cohorts": [int(c) for c in keep_cohorts],
        "kept_cohort_count": int(len(keep_cohorts)),
        "treated_city_count": int(len(keep_ids)),
        "unsupported_treated_city_count": int(len(unsupported_ids)),
        "unsupported_treated_post_rows_dropped": int(dropped_rows),
        "treated_share": float(out["treated_city"].mean()),
        "rows_before": int(orig_rows),
        "rows_after": int(len(out)),
        "cohort_distribution": cohort_counts.to_dict(orient="records"),
    }


def run_identification_scorecard(summary: Dict[str, object]) -> Dict[str, object]:
    """Build comparable credibility scores across causal estimators."""
    candidates: List[Dict[str, object]] = []
    timing_ready_design = summary.get("timing_ready_design", {}) if isinstance(summary.get("timing_ready_design", {}), dict) else {}
    timing_ready_active = timing_ready_design.get("status") == "ok"

    did = summary.get("did_two_way_fe", {})
    if isinstance(did, dict) and {"coef", "stderr", "t_value", "n_obs"}.issubset(did.keys()):
        p_val = _t_pvalue(float(did["t_value"]), max(1, int(did.get("n_clusters", 2)) - 1))
        sig = _clip01(1.0 - p_val / 0.10)
        design = 0.35  # Baseline TWFE has weakest design credibility by default.
        twfe_boot = summary.get("did_twfe_cluster_bootstrap", {})
        twfe_perm = summary.get("did_twfe_permutation", {})
        twfe_wild = summary.get("did_twfe_wild_bootstrap", {})
        twfe_lead = summary.get("did_twfe_lead_placebo", {})
        stacked_lead = summary.get("did_stacked_lead_placebo", {})
        lead_obj = (
            stacked_lead
            if isinstance(stacked_lead, dict) and stacked_lead.get("status") == "ok"
            else twfe_lead
        )
        boot_bonus = 0.0
        perm_bonus = 0.0
        wild_bonus = 0.0
        lead_penalty = 0.0
        lead_source = "none"
        if isinstance(twfe_boot, dict) and twfe_boot.get("status") == "ok":
            ci = twfe_boot.get("ci95_percentile", [None, None])
            try:
                ci_low = float(ci[0]) if ci[0] is not None else float("nan")
                ci_high = float(ci[1]) if ci[1] is not None else float("nan")
                if np.isfinite(ci_low) and np.isfinite(ci_high):
                    if ci_low > 0 or ci_high < 0:
                        boot_bonus = 0.18
                    else:
                        boot_bonus = 0.06
            except Exception:  # noqa: BLE001
                boot_bonus = 0.0
        if isinstance(twfe_perm, dict) and twfe_perm.get("status") == "ok":
            p_perm = float(twfe_perm.get("p_value_abs_t", 1.0))
            perm_bonus = 0.22 * _clip01(1.0 - p_perm / 0.10)
        if isinstance(twfe_wild, dict) and twfe_wild.get("status") == "ok":
            p_wild = float(twfe_wild.get("p_value_abs_t", 1.0))
            wild_bonus = 0.24 * _clip01(1.0 - p_wild / 0.10)
        if isinstance(lead_obj, dict) and lead_obj.get("status") == "ok":
            share_sig = float(lead_obj.get("share_p_lt_0_10", 1.0))
            lead_penalty = 0.24 * _clip01(share_sig / 0.50)
            lead_source = "stacked" if lead_obj is stacked_lead else "twfe"
        design = _clip01(design + boot_bonus + perm_bonus + wild_bonus - lead_penalty)
        support = _clip01(np.log1p(float(did["n_obs"])) / np.log1p(2500.0))
        score = 100.0 * (0.45 * sig + 0.35 * design + 0.20 * support)
        candidates.append(
            {
                "estimator": "did_two_way_fe",
                "effect": float(did["coef"]),
                "ci95": [float(did["coef"] - 1.96 * did["stderr"]), float(did["coef"] + 1.96 * did["stderr"])],
                "p_value": p_val,
                "n_support": int(did["n_obs"]),
                "credibility_score": score,
                "subscores": {
                    "significance": sig,
                    "design": design,
                    "support": support,
                    "permutation_bonus": perm_bonus,
                    "bootstrap_bonus": boot_bonus,
                    "wild_bootstrap_bonus": wild_bonus,
                    "lead_placebo_penalty": lead_penalty,
                    "lead_placebo_source": lead_source,
                },
            }
        )

    matched = summary.get("did_matched_trend", {})
    if isinstance(matched, dict) and matched.get("status") == "ok":
        p_val = float(matched.get("p_value", 1.0))
        sig = _clip01(1.0 - p_val / 0.10)
        balance_after = float(matched.get("avg_abs_smd_after", 1.0))
        placebo_rows = pd.DataFrame(matched.get("placebo", []))
        placebo_share = float(np.mean(placebo_rows["p_value"] < 0.10)) if not placebo_rows.empty else 0.5
        design = 0.6 * _clip01(1.0 - balance_after / 0.25) + 0.4 * _clip01(1.0 - placebo_share)
        support = _clip01(np.log1p(float(matched.get("n_obs", 0))) / np.log1p(2500.0))
        score = 100.0 * (0.45 * sig + 0.35 * design + 0.20 * support)
        candidates.append(
            {
                "estimator": "did_matched_trend",
                "effect": float(matched["coef"]),
                "ci95": [float(matched["coef"] - 1.96 * matched["stderr"]), float(matched["coef"] + 1.96 * matched["stderr"])],
                "p_value": p_val,
                "n_support": int(matched.get("n_obs", 0)),
                "credibility_score": score,
                "subscores": {"significance": sig, "design": design, "support": support},
            }
        )

    staggered = summary.get("staggered_did", {})
    if isinstance(staggered, dict) and staggered.get("status") == "ok":
        t_val = float(staggered.get("post_avg_t_value", 0.0))
        p_val = _t_pvalue(t_val, max(1, int(staggered.get("n_cohorts", 2)) - 1))
        sig = _clip01(1.0 - p_val / 0.10)
        pre_share = float(staggered.get("pretrend_share_p_lt_0_10", 1.0))
        pre_max = float(staggered.get("pretrend_max_abs_t", 9.0))
        pre_joint_p = float(staggered.get("pretrend_joint_p_value", 0.0))
        design = (
            0.35 * _clip01(1.0 - pre_share)
            + 0.30 * _clip01(1.0 - pre_max / 2.5)
            + 0.35 * _clip01(pre_joint_p / 0.10)
        )
        support = _clip01(np.log1p(float(staggered.get("treated_city_count", 0)) * 11.0) / np.log1p(2500.0))
        score = 100.0 * (0.45 * sig + 0.35 * design + 0.20 * support)
        candidates.append(
            {
                "estimator": "staggered_did",
                "effect": float(staggered["post_avg_att"]),
                "ci95": staggered.get("post_avg_ci95", [None, None]),
                "p_value": p_val,
                "n_support": int(staggered.get("treated_city_count", 0)),
                "credibility_score": score,
                "subscores": {"significance": sig, "design": design, "support": support},
            }
        )

    nyt = summary.get("not_yet_treated_did", {})
    if isinstance(nyt, dict) and nyt.get("status") == "ok":
        p_val = _prefer_finite(nyt.get("robust_p_value_weighted"), nyt.get("p_value_weighted"))
        eff = _prefer_finite(nyt.get("robust_att_weighted"), nyt.get("att_weighted"))
        ci = nyt.get("robust_ci95_weighted")
        if (
            not isinstance(ci, list)
            or len(ci) < 2
            or (_safe_float(ci[0]) is None and _safe_float(ci[1]) is None)
        ):
            ci = nyt.get("ci95_weighted", [None, None])
        p_val = float(p_val) if p_val is not None else 1.0
        sig = _clip01(1.0 - p_val / 0.10) if np.isfinite(p_val) else 0.0
        placebo_share = float(nyt.get("placebo_share_p_lt_0_10", 1.0))
        placebo_max_t = float(nyt.get("placebo_max_abs_t", 9.0))
        design = 0.5 * _clip01(1.0 - placebo_share) + 0.5 * _clip01(1.0 - placebo_max_t / 2.5)
        support = _clip01(np.log1p(float(nyt.get("cohort_count", 0)) * 120.0) / np.log1p(2500.0))
        score = 100.0 * (0.45 * sig + 0.35 * design + 0.20 * support)
        candidates.append(
            {
                "estimator": "not_yet_treated_did",
                "effect": float(eff) if eff is not None and np.isfinite(float(eff)) else None,
                "ci95": ci,
                "p_value": p_val,
                "n_support": int(nyt.get("cohort_count", 0)),
                "credibility_score": score,
                "subscores": {"significance": sig, "design": design, "support": support},
            }
        )

    dr = summary.get("dr_did", {})
    if isinstance(dr, dict) and dr.get("status") == "ok":
        p_val = float(dr.get("p_value", 1.0))
        sig = _clip01(1.0 - p_val / 0.10)
        overlap = float(dr.get("overlap_share_after_trim", 0.0))
        brier = float(dr.get("nuisance_brier", 1.0))
        mse = float(dr.get("nuisance_control_mse", 10.0))
        moment = float(dr.get("nuisance_moment_abs", 1.0))
        overlap_score = _clip01((overlap - 0.55) / 0.35)
        nuisance_score = (
            0.35 * _clip01(1.0 - brier / 0.25)
            + 0.35 * _clip01(1.0 - mse / 10.0)
            + 0.30 * _clip01(1.0 - moment / 1.5)
        )
        design = 0.55 * overlap_score + 0.45 * nuisance_score
        support = _clip01(np.log1p(float(dr.get("n_obs", 0))) / np.log1p(2500.0))
        score = 100.0 * (0.45 * sig + 0.35 * design + 0.20 * support)
        ci = [float(dr["coef"] - 1.96 * dr["stderr"]), float(dr["coef"] + 1.96 * dr["stderr"])]
        candidates.append(
            {
                "estimator": "dr_did",
                "effect": float(dr["coef"]),
                "ci95": ci,
                "p_value": p_val,
                "n_support": int(dr.get("n_obs", 0)),
                "credibility_score": score,
                "subscores": {"significance": sig, "design": design, "support": support},
            }
        )

    dml = summary.get("dml_did", {})
    if (
        not timing_ready_active
        and isinstance(dml, dict)
        and _safe_float(dml.get("coef")) is not None
        and _safe_float(dml.get("stderr")) is not None
    ):
        p_val = _safe_float(dml.get("p_value"))
        sig = _clip01(1.0 - float(p_val) / 0.10) if p_val is not None else 0.0
        overlap = _safe_float(dml.get("overlap_trimmed_share"))
        brier = _safe_float(dml.get("nuisance_d_brier"))
        mse = _safe_float(dml.get("nuisance_y_mse"))
        corr = _safe_float(dml.get("nuisance_mean_abs_corr_dt_x"))
        trim_score = _clip01(1.0 - float(overlap or 0.0) / 0.30)
        nuisance_score = (
            0.35 * _clip01(1.0 - float(brier or 1.0) / 0.25)
            + 0.40 * _clip01(1.0 - float(mse or 10.0) / 10.0)
            + 0.25 * _clip01(1.0 - float(corr or 1.0) / 0.15)
        )
        design = 0.40 * trim_score + 0.60 * nuisance_score
        support = _clip01(np.log1p(float(dml.get("n_obs", 0))) / np.log1p(2500.0))
        coef = float(dml["coef"])
        se = float(dml["stderr"])
        score = 100.0 * (0.45 * sig + 0.35 * design + 0.20 * support)
        candidates.append(
            {
                "estimator": "dml_did",
                "effect": coef,
                "ci95": [float(coef - 1.96 * se), float(coef + 1.96 * se)],
                "p_value": p_val,
                "n_support": int(dml.get("n_obs", 0)),
                "credibility_score": score,
                "subscores": {
                    "significance": sig,
                    "design": design,
                    "support": support,
                    "overlap_trimmed_share": overlap,
                    "nuisance_d_brier": brier,
                    "nuisance_y_mse": mse,
                    "nuisance_mean_abs_corr_dt_x": corr,
                },
            }
        )

    intense = summary.get("did_intense_contrast", {})
    if isinstance(intense, dict) and intense.get("status") == "ok":
        p_val = float(intense.get("p_value", 1.0))
        sig = _clip01(1.0 - p_val / 0.10)
        treated_share = _safe_float(intense.get("treated_share"))
        balance = _clip01(1.0 - abs((treated_share if treated_share is not None else 0.45) - 0.45) / 0.35)
        design = 0.62 + 0.18 * balance
        support = _clip01(np.log1p(float(intense.get("n_obs", 0))) / np.log1p(2500.0))
        score = 100.0 * (0.45 * sig + 0.35 * _clip01(design) + 0.20 * support)
        candidates.append(
            {
                "estimator": "did_intense_contrast",
                "effect": float(intense["coef"]),
                "ci95": [
                    float(intense["coef"] - 1.96 * intense["stderr"]),
                    float(intense["coef"] + 1.96 * intense["stderr"]),
                ],
                "p_value": p_val,
                "n_support": int(intense.get("n_obs", 0)),
                "credibility_score": score,
                "subscores": {
                    "significance": sig,
                    "design": _clip01(design),
                    "support": support,
                    "treated_share_balance": balance,
                },
            }
        )

    dose = summary.get("did_dose_response", {})
    if isinstance(dose, dict) and dose.get("status") == "ok":
        p_val = float(dose.get("p_value", 1.0))
        sig = _clip01(1.0 - p_val / 0.10)
        dose_share = _safe_float(dose.get("dose_positive_share"))
        dose_disp = _safe_float(dose.get("dose_std"))
        design = 0.52
        if dose_share is not None:
            design += 0.16 * _clip01(dose_share / 0.75)
        if dose_disp is not None:
            design += 0.18 * _clip01(dose_disp / 0.35)
        support = _clip01(np.log1p(float(dose.get("n_obs", 0))) / np.log1p(2500.0))
        score = 100.0 * (0.45 * sig + 0.35 * _clip01(design) + 0.20 * support)
        eff = dose.get("coef_per_1sd_dose", dose.get("coef"))
        se = dose.get("stderr_per_1sd_dose", dose.get("stderr"))
        if eff is not None and se is not None:
            eff_f = float(eff)
            se_f = float(max(float(se), 1e-12))
            ci = [float(eff_f - 1.96 * se_f), float(eff_f + 1.96 * se_f)]
        else:
            eff_f = None
            ci = [None, None]
        candidates.append(
            {
                "estimator": "did_dose_response",
                "effect": eff_f,
                "ci95": ci,
                "p_value": p_val,
                "n_support": int(dose.get("n_obs", 0)),
                "credibility_score": score,
                "subscores": {
                    "significance": sig,
                    "design": _clip01(design),
                    "support": support,
                    "dose_positive_share": dose_share,
                },
            }
        )

    spill = summary.get("did_spillover_twfe", {})
    if isinstance(spill, dict) and spill.get("status") == "ok":
        direct_p = float(spill.get("direct_p_value", 1.0))
        spill_p = float(spill.get("spillover_p_value", 1.0))
        lead_obj = spill.get("placebo_lead_check", {}) if isinstance(spill.get("placebo_lead_check", {}), dict) else {}
        lead_p = float(lead_obj.get("lead_p_value", 1.0)) if lead_obj.get("status") == "ok" else 1.0
        sig = _clip01(1.0 - direct_p / 0.10)
        design = (
            0.45 * _clip01(1.0 - spill_p / 0.10)
            + 0.30 * _clip01(lead_p / 0.10)
            + 0.25 * _clip01(float(spill.get("spillover_exposure_std", 0.0)) / 0.35)
        )
        support = _clip01(np.log1p(float(spill.get("n_obs", 0))) / np.log1p(2500.0))
        score = 100.0 * (0.45 * sig + 0.35 * design + 0.20 * support)
        direct_coef = float(spill.get("direct_coef", 0.0))
        direct_se = float(max(float(spill.get("direct_stderr", 0.0)), 1e-12))
        candidates.append(
            {
                "estimator": "did_spillover_twfe",
                "effect": direct_coef,
                "ci95": [float(direct_coef - 1.96 * direct_se), float(direct_coef + 1.96 * direct_se)],
                "p_value": direct_p,
                "n_support": int(spill.get("n_obs", 0)),
                "credibility_score": score,
                "subscores": {
                    "significance": sig,
                    "design": design,
                    "support": support,
                    "spillover_p_value": spill_p,
                    "lead_placebo_p_value": lead_p,
                },
            }
        )

    if not candidates:
        return {"status": "skipped", "reason": "no_valid_estimators"}

    consistency_rows = [c for c in candidates if c.get("effect") is not None and float(c.get("p_value", 1.0)) <= 0.20]
    majority_sign = None
    sign_consistency_share = None
    if len(consistency_rows) >= 3:
        signs = [1 if float(c["effect"]) >= 0 else -1 for c in consistency_rows]
        majority_sign = 1 if int(np.sum(signs)) >= 0 else -1
        aligned = np.array([1 if s == majority_sign else 0 for s in signs], dtype=float)
        sign_consistency_share = float(np.mean(aligned))
        for cand in candidates:
            eff = cand.get("effect")
            if eff is None:
                continue
            sign = 1 if float(eff) >= 0 else -1
            if sign != majority_sign:
                cand["credibility_score"] = float(cand["credibility_score"]) * 0.92
                subs = cand.get("subscores", {}) if isinstance(cand.get("subscores", {}), dict) else {}
                subs["consistency_penalty"] = 0.08
                cand["subscores"] = subs

    cand_df = pd.DataFrame(candidates).sort_values("credibility_score", ascending=False)
    preferred = cand_df.iloc[0].to_dict()
    ranking = cand_df.to_dict(orient="records")
    return {
        "status": "ok",
        "preferred": preferred,
        "ranking": ranking,
        "majority_effect_sign": majority_sign,
        "sign_consistency_share_p_le_0_20": sign_consistency_share,
    }


def _soft_impute(
    observed: np.ndarray,
    obs_mask: np.ndarray,
    shrink: float = 2.0,
    max_iter: int = 300,
    tol: float = 1e-5,
) -> np.ndarray:
    """Low-rank matrix completion via iterative soft-thresholded SVD."""
    mat = observed.copy()
    col_means = np.nanmean(mat, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    mat = np.where(obs_mask, mat, col_means[np.newaxis, :])

    for _ in range(max_iter):
        u, s, vt = np.linalg.svd(mat, full_matrices=False)
        s_new = np.maximum(s - shrink, 0.0)
        updated = (u * s_new) @ vt
        updated[obs_mask] = observed[obs_mask]

        denom = max(float(np.linalg.norm(mat)), 1e-10)
        diff = float(np.linalg.norm(updated - mat) / denom)
        mat = updated
        if diff < tol:
            break
    return mat


def _matrix_completion_city_assignment(
    panel: pd.DataFrame,
    *,
    city_order: List[str],
    treatment_year: int,
) -> pd.DataFrame:
    """Resolve treated-city and cohort-year assignment for matrix completion."""
    cols = ["city_id", "treated_city"]
    if "treatment_cohort_year" in panel.columns:
        cols.append("treatment_cohort_year")

    city_map = panel[cols].drop_duplicates("city_id").copy()
    city_map["city_id"] = city_map["city_id"].astype(str)
    city_map["treated_city"] = pd.to_numeric(city_map["treated_city"], errors="coerce").fillna(0.0).astype(int)
    if "treatment_cohort_year" in city_map.columns:
        city_map["cohort_year"] = pd.to_numeric(city_map["treatment_cohort_year"], errors="coerce").fillna(9999).astype(int)
    else:
        city_map["cohort_year"] = np.where(city_map["treated_city"] == 1, int(treatment_year), 9999).astype(int)
    city_map.loc[city_map["treated_city"] == 0, "cohort_year"] = 9999

    out = pd.DataFrame({"city_id": [str(c) for c in city_order]})
    out = out.merge(city_map[["city_id", "treated_city", "cohort_year"]], on="city_id", how="left")
    out["treated_city"] = pd.to_numeric(out["treated_city"], errors="coerce").fillna(0.0).astype(int)
    out["cohort_year"] = pd.to_numeric(out["cohort_year"], errors="coerce").fillna(9999).astype(int)
    out.loc[out["treated_city"] == 0, "cohort_year"] = 9999
    return out


def run_matrix_completion_counterfactual(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    treatment_year: int = 2020,
    placebo_count: int = 25,
    random_state: int = 42,
    min_pre_periods: int = 2,
    min_post_periods: int = 2,
) -> Dict[str, object]:
    """Estimate ATT using low-rank counterfactuals under city-specific treatment cohorts."""
    cols = ["city_id", "year", outcome, "treated_city"]
    if "treatment_cohort_year" in panel.columns:
        cols.append("treatment_cohort_year")
    df = panel[cols].copy()
    df["city_id"] = df["city_id"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df[outcome] = pd.to_numeric(df[outcome], errors="coerce")
    df = df.dropna(subset=["city_id", "year"]).copy()
    if df.empty:
        return {"status": "skipped", "reason": "empty_panel"}
    df["year"] = df["year"].astype(int)

    years = sorted(df["year"].unique().tolist())
    city_order = sorted(df["city_id"].unique().tolist())
    if len(years) < 4 or len(city_order) < 8:
        return {"status": "skipped", "reason": "insufficient_matrix_support"}

    pivot = df.pivot_table(index="city_id", columns="year", values=outcome).reindex(index=city_order, columns=years)
    y_mat = pivot.to_numpy(dtype=float)
    finite_mask = np.isfinite(y_mat)

    city_assign = _matrix_completion_city_assignment(df, city_order=city_order, treatment_year=int(treatment_year))
    year_arr = np.asarray(years, dtype=int)
    cohort_years = city_assign["cohort_year"].to_numpy(dtype=int)
    min_year = int(year_arr.min())
    max_year = int(year_arr.max())
    supported_lower = int(min_year + max(int(min_pre_periods), 1))
    supported_upper = int(max_year - max(int(min_post_periods), 1))
    supported_cohort_mask = (cohort_years >= supported_lower) & (cohort_years <= supported_upper) & (cohort_years < 9999)
    unsupported_treated_city_count = int(((cohort_years < 9999) & (~supported_cohort_mask)).sum())
    city_assign.loc[~supported_cohort_mask, "treated_city"] = 0
    city_assign.loc[~supported_cohort_mask, "cohort_year"] = 9999

    treated_mask_city = city_assign["treated_city"].to_numpy(dtype=int) == 1
    cohort_years = city_assign["cohort_year"].to_numpy(dtype=int)

    target_mask = (cohort_years[:, np.newaxis] < 9999) & (year_arr[np.newaxis, :] >= cohort_years[:, np.newaxis])
    eval_mask = target_mask & finite_mask
    if int(treated_mask_city.sum()) == 0 or int(eval_mask.sum()) == 0:
        return {"status": "skipped", "reason": "invalid_city_specific_treatment_window"}

    cv_pre_mask = np.zeros_like(target_mask, dtype=bool)
    for idx, cohort in enumerate(cohort_years.tolist()):
        if int(cohort) >= 9999 or not bool(treated_mask_city[idx]):
            continue
        pre_idx = np.where((year_arr < int(cohort)) & finite_mask[idx])[0]
        if len(pre_idx) >= max(int(min_pre_periods), 2):
            cv_pre_mask[idx, int(pre_idx[-1])] = True

    obs_mask = finite_mask.copy()
    obs_mask[target_mask] = False
    obs_mask[cv_pre_mask] = False
    completed = _soft_impute(observed=y_mat, obs_mask=obs_mask, shrink=1.6, max_iter=300, tol=1e-5)

    true_vals = y_mat[eval_mask]
    pred_vals = completed[eval_mask]
    att = float(np.mean(true_vals - pred_vals))

    pre_available_mask = (
        treated_mask_city[:, np.newaxis]
        & (cohort_years[:, np.newaxis] < 9999)
        & (year_arr[np.newaxis, :] < cohort_years[:, np.newaxis])
        & finite_mask
    )
    pre_eval_mask = cv_pre_mask & finite_mask
    if int(pre_eval_mask.sum()) >= 1:
        pre_rmse = float(np.sqrt(np.mean((y_mat[pre_eval_mask] - completed[pre_eval_mask]) ** 2)))
    else:
        pre_rmse = float("nan")

    treated_counts = treated_mask_city.astype(int)
    treated_den = np.maximum(int(treated_counts.sum()), 1)
    treated_avg = np.nanmean(y_mat[treated_mask_city, :], axis=0)
    treated_cf = np.nanmean(completed[treated_mask_city, :], axis=0)
    active_post_share = target_mask[treated_mask_city, :].sum(axis=0) / float(treated_den)
    ts = pd.DataFrame(
        {
            "year": years,
            "treated_actual": treated_avg,
            "treated_counterfactual": treated_cf,
            "gap": treated_avg - treated_cf,
            "post": (active_post_share > 0.0).astype(int),
            "treated_post_share": active_post_share.astype(float),
        }
    )
    ts.to_csv(DATA_OUTPUTS / "matrix_completion_timeseries.csv", index=False)

    cohort_rows: List[Dict[str, object]] = []
    valid_cohorts = sorted(int(c) for c in np.unique(cohort_years[(cohort_years < 9999)]).tolist())
    for cohort in valid_cohorts:
        cohort_city_mask = cohort_years == int(cohort)
        cohort_eval = cohort_city_mask[:, np.newaxis] & eval_mask
        cohort_pre = cohort_city_mask[:, np.newaxis] & pre_eval_mask
        if int(cohort_eval.sum()) == 0:
            continue
        cohort_att = float(np.mean(y_mat[cohort_eval] - completed[cohort_eval]))
        cohort_pre_rmse = (
            float(np.sqrt(np.mean((y_mat[cohort_pre] - completed[cohort_pre]) ** 2)))
            if int(cohort_pre.sum()) >= 1
            else float("nan")
        )
        cohort_rows.append(
            {
                "cohort_year": int(cohort),
                "att_post": cohort_att,
                "pre_rmse": cohort_pre_rmse,
                "n_treat_cities": int(cohort_city_mask.sum()),
                "n_post_cells": int(cohort_eval.sum()),
                "n_pre_cells": int(cohort_pre.sum()),
            }
        )
    cohort_df = pd.DataFrame(cohort_rows).sort_values("cohort_year") if cohort_rows else pd.DataFrame()
    cohort_csv = DATA_OUTPUTS / "matrix_completion_by_cohort.csv"
    cohort_df.to_csv(cohort_csv, index=False)

    # Randomization inference: pseudo-treated control cities with sampled cohort schedules.
    rng = np.random.default_rng(random_state)
    control_indices = np.where(~treated_mask_city)[0]
    treated_cohort_pool = cohort_years[cohort_years < 9999]
    placebo_atts: List[float] = []
    if len(control_indices) > 0 and len(treated_cohort_pool) > 0:
        placebo_draws = int(max(placebo_count, 1))
        placebo_size = int(max(1, min(len(control_indices), int(treated_mask_city.sum()))))
        for _ in range(placebo_draws):
            sampled_controls = rng.choice(control_indices, size=placebo_size, replace=False)
            sampled_cohorts = rng.choice(treated_cohort_pool, size=placebo_size, replace=True).astype(int)
            p_target = np.zeros_like(target_mask, dtype=bool)
            for idx, cohort in zip(sampled_controls.tolist(), sampled_cohorts.tolist()):
                p_target[int(idx), :] = year_arr >= int(cohort)
            p_eval = p_target & finite_mask
            if int(p_eval.sum()) == 0:
                continue
            p_obs = finite_mask.copy()
            p_obs[p_target] = False
            p_completed = _soft_impute(observed=y_mat, obs_mask=p_obs, shrink=1.6, max_iter=200, tol=1e-5)
            placebo_atts.append(float(np.mean(y_mat[p_eval] - p_completed[p_eval])))

    if placebo_atts:
        p_arr = np.asarray(placebo_atts, dtype=float)
        p_val = float((1.0 + np.sum(np.abs(p_arr) >= abs(att))) / (1.0 + len(p_arr)))
    else:
        p_val = float("nan")
    placebo_pass = bool(np.isfinite(p_val) and p_val >= 0.10)

    out = {
        "status": "ok",
        "att_post": att,
        "pre_rmse": pre_rmse,
        "placebo_count": int(len(placebo_atts)),
        "placebo_p_value": p_val,
        "placebo_pass_p_ge_0_10": placebo_pass,
        "credibility_status": "supporting" if placebo_pass else "appendix_only_placebo_failure",
        "recommended_role": "supporting_robustness" if placebo_pass else "appendix_only",
        "treated_city_count": int(treated_mask_city.sum()),
        "control_city_count": int((~treated_mask_city).sum()),
        "n_obs_matrix": int(finite_mask.sum()),
        "n_matrix_cells": int(y_mat.size),
        "evaluated_post_cells": int(eval_mask.sum()),
        "evaluated_pre_cells": int(pre_eval_mask.sum()),
        "available_pre_cells_total": int(pre_available_mask.sum()),
        "cohort_mode": "city_specific" if "treatment_cohort_year" in df.columns else "global_reference_fallback",
        "global_reference_year": int(treatment_year),
        "cohort_count": int(len(valid_cohorts)),
        "cohort_year_min": int(min(valid_cohorts)) if valid_cohorts else None,
        "cohort_year_max": int(max(valid_cohorts)) if valid_cohorts else None,
        "cohort_summary_file": str(cohort_csv),
        "timeseries_file": str(DATA_OUTPUTS / "matrix_completion_timeseries.csv"),
        "placebo_design": "pseudo_control_cities_with_sampled_city_cohorts",
        "pre_rmse_design": "hold_out_last_available_treated_pre_period_per_city_when_pre_support_ge_2",
        "support_rule": "drop treated cohorts without sufficient pre/post panel support before matrix masking",
        "min_pre_periods": int(min_pre_periods),
        "min_post_periods": int(min_post_periods),
        "supported_cohort_window": [int(supported_lower), int(supported_upper)],
        "unsupported_treated_city_count": int(unsupported_treated_city_count),
    }
    LOGGER.info(
        "Matrix completion done (%s): ATT=%.4f, placebo_p=%.4f, treated=%s, cohorts=%s",
        out["cohort_mode"],
        out["att_post"],
        out["placebo_p_value"],
        out["treated_city_count"],
        out["cohort_count"],
    )
    return out


def run_beta_convergence(panel: pd.DataFrame) -> Dict[str, float]:
    """Estimate beta-convergence across cities using long-run growth regression."""
    first_year = int(panel["year"].min())
    last_year = int(panel["year"].max())
    span = max(1, last_year - first_year)

    base = panel.loc[panel["year"] == first_year, ["city_id", "composite_index", "log_population", "temperature_mean"]].copy()
    final = panel.loc[panel["year"] == last_year, ["city_id", "composite_index"]].copy()
    df = base.merge(final, on="city_id", suffixes=("_0", "_t"))

    df = df[(df["composite_index_0"] > 0) & (df["composite_index_t"] > 0)]
    df["growth"] = np.log(df["composite_index_t"] / df["composite_index_0"]) / span
    df["log_initial"] = np.log(df["composite_index_0"])

    y = df["growth"].to_numpy(dtype=float)
    x = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            df["log_initial"].to_numpy(dtype=float),
            df["log_population"].to_numpy(dtype=float),
            df["temperature_mean"].to_numpy(dtype=float),
        ]
    )
    names = ["const", "log_initial", "log_population", "temperature_mean"]
    res = _ols_hc1(y, x, names)

    beta_idx = names.index("log_initial")
    result = {
        "coef": float(res.coef[beta_idx]),
        "stderr": float(res.stderr[beta_idx]),
        "t_value": float(res.t_value[beta_idx]),
        "r2": float(res.r2),
        "n_obs": int(res.n_obs),
        "convergence": bool(res.coef[beta_idx] < 0),
    }
    LOGGER.info("Beta-convergence done: beta=%.4f", result["coef"])
    return result


def run_log_elasticity_fe(panel: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Estimate log-log elasticities under two-way FE."""
    df = panel.copy()
    df = df[(df["composite_index"] > 0) & (df["gdp_per_capita"] > 0) & (df["population"] > 0)]

    df["ln_y"] = np.log(df["composite_index"])
    df["ln_gdp_pc"] = np.log(df["gdp_per_capita"])
    df["ln_pop"] = np.log(df["population"])

    cols = ["ln_y", "ln_gdp_pc", "ln_pop", "temperature_mean", "precipitation_sum"]
    tw = _two_way_within(df, cols)

    y = tw["ln_y_tw"].to_numpy(dtype=float)
    x = np.column_stack(
        [
            tw["ln_gdp_pc_tw"].to_numpy(dtype=float),
            tw["ln_pop_tw"].to_numpy(dtype=float),
            tw["temperature_mean_tw"].to_numpy(dtype=float),
            tw["precipitation_sum_tw"].to_numpy(dtype=float),
        ]
    )
    names = ["ln_gdp_pc", "ln_pop", "temperature_mean", "precipitation_sum"]

    res = _ols_hc1(y, x, names)
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(names):
        out[name] = {
            "coef": float(res.coef[i]),
            "stderr": float(res.stderr[i]),
            "t_value": float(res.t_value[i]),
        }

    out["meta"] = {"r2_within": float(res.r2), "n_obs": int(res.n_obs)}
    LOGGER.info("Elasticity FE done: R2=%.3f", res.r2)
    return out


def run_event_study_fe(
    panel: pd.DataFrame,
    outcome: str = "composite_index",
    treatment_year: int = 2020,
    min_lag: int = -4,
    max_lead: int = 4,
) -> Dict[str, object]:
    """Estimate dynamic treatment effects using event-study dummies under two-way FE."""
    _assert_valid_causal_outcome(outcome)
    schedule = _resolve_city_treatment_schedule(panel.copy())
    finite_cohorts = []
    if not schedule.empty and "cohort_year" in schedule.columns:
        finite_cohorts = (
            pd.to_numeric(schedule["cohort_year"], errors="coerce")
            .replace(9999, np.nan)
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
    if len(finite_cohorts) > 1:
        robust = run_staggered_did(
            panel,
            outcome=outcome,
            treatment_year=treatment_year,
            min_event=min_lag,
            max_event=max_lead,
            strict_pretrend_screen=False,
        )
        if robust.get("status") == "ok":
            points = [
                {
                    "rel_year": int(row["event_time"]),
                    "coef": float(row["att"]),
                    "stderr": float(row["stderr"]),
                    "t_value": float(row["t_value"]),
                    "p_value": float(row["p_value"]),
                    "cells": int(row["cells"]),
                    "weighted_n_treat": float(row["weighted_n_treat"]),
                }
                for row in robust.get("event_time_points", [])
            ]
            return {
                "status": "ok",
                "baseline_rel_year": -1,
                "points": points,
                "meta": {
                    "n_obs": int(len(panel)),
                    "stderr_type": "cohort_time_att_aggregation",
                    "n_clusters": int(panel["iso3"].astype(str).nunique()) if "iso3" in panel.columns else int(panel["city_id"].astype(str).nunique()),
                    "relative_time_design": "city_specific_cohort",
                    "event_study_design": "staggered_not_yet_treated_aggregation",
                    "cohort_count": int(robust.get("cohort_count", 0)),
                    "pretrend_joint_p_value": float(robust.get("pretrend_joint_p_value")) if np.isfinite(_safe_float(robust.get("pretrend_joint_p_value"))) else None,
                },
            }

    df = panel.copy()
    if not schedule.empty:
        df = df.merge(schedule[["city_id", "cohort_year"]], on="city_id", how="left")
        df["rel_year"] = np.where(
            pd.to_numeric(df["cohort_year"], errors="coerce").fillna(9999).astype(int) < 9999,
            pd.to_numeric(df["year"], errors="coerce").astype(float)
            - pd.to_numeric(df["cohort_year"], errors="coerce").astype(float),
            np.nan,
        )
    else:
        df["rel_year"] = df["year"] - treatment_year

    event_points = [k for k in range(min_lag, max_lead + 1) if k != -1]
    if len(event_points) < 3:
        return {"status": "skipped", "reason": "invalid_event_window"}

    event_cols: List[str] = []
    for k in event_points:
        col = f"event_{k:+d}"
        df[col] = ((df["treated_city"] == 1) & (df["rel_year"] == k)).astype(int)
        event_cols.append(col)

    cols = [outcome, "temperature_mean", "precipitation_sum", "baseline_population_log", *event_cols]
    tw = _two_way_within(df, cols)

    y = tw[f"{outcome}_tw"].to_numpy(dtype=float)
    x_cols = [f"{col}_tw" for col in event_cols] + ["temperature_mean_tw", "precipitation_sum_tw", "baseline_population_log_tw"]
    x = np.column_stack([tw[col].to_numpy(dtype=float) for col in x_cols])
    names = event_cols + ["temperature_mean", "precipitation_sum", "baseline_population_log"]
    clusters = (
        df["iso3"].astype(str).to_numpy()
        if "iso3" in df.columns
        else df["city_id"].astype(str).to_numpy()
    )
    try:
        res, n_clusters = _ols_cluster_hc1(y, x, names, clusters)
        stderr_type = "cluster_iso3_cr1" if "iso3" in df.columns and n_clusters >= 2 else "cluster_city_cr1"
    except Exception:  # noqa: BLE001
        res = _ols_hc1(y, x, names)
        n_clusters = int(pd.Series(clusters).nunique())
        stderr_type = "hc1_fallback_error"

    points = []
    for idx, col in enumerate(event_cols):
        rel = int(col.replace("event_", ""))
        t_val = float(res.t_value[idx])
        points.append(
            {
                "rel_year": rel,
                "coef": float(res.coef[idx]),
                "stderr": float(res.stderr[idx]),
                "t_value": t_val,
                "p_value": _t_pvalue(t_val, max(1, n_clusters - 1)),
            }
        )
    points = sorted(points, key=lambda d: d["rel_year"])

    return {
        "status": "ok",
        "baseline_rel_year": -1,
        "points": points,
        "meta": {
            "n_obs": int(res.n_obs),
            "r2_within": float(res.r2),
            "stderr_type": stderr_type,
            "n_clusters": int(n_clusters),
            "relative_time_design": "city_specific_cohort" if "cohort_year" in df.columns else "global_reference_year",
        },
    }


def _apply_treatment_variant(panel: pd.DataFrame, variant_suffix: str) -> tuple[pd.DataFrame, Dict[str, object]]:
    treated_col = f"treated_city_{variant_suffix}"
    post_col = f"post_policy_{variant_suffix}"
    did_col = f"did_treatment_{variant_suffix}"
    cohort_col = f"treatment_cohort_year_{variant_suffix}"
    required = {treated_col, post_col, did_col, cohort_col}
    if not required.issubset(set(panel.columns)):
        miss = sorted(required.difference(set(panel.columns)))
        return panel.iloc[0:0].copy(), {"status": "skipped", "reason": f"missing_columns_{','.join(miss)}"}

    out = panel.copy()
    out["treated_city"] = pd.to_numeric(out[treated_col], errors="coerce").fillna(0).astype(int)
    out["post_policy"] = pd.to_numeric(out[post_col], errors="coerce").fillna(0).astype(int)
    out["did_treatment"] = pd.to_numeric(out[did_col], errors="coerce").fillna(0).astype(int)
    out["treatment_cohort_year"] = pd.to_numeric(out[cohort_col], errors="coerce").fillna(9999).astype(int)
    return out, {
        "status": "ok",
        "treated_col": treated_col,
        "post_col": post_col,
        "did_col": did_col,
        "cohort_col": cohort_col,
        "treated_share": float(out["treated_city"].mean()),
        "did_variation": int(out["did_treatment"].nunique()),
    }


def run_policy_type_response_matrix(panel: pd.DataFrame) -> Dict[str, object]:
    """Run DML/Event-Study loops for classified policies across fast/slow outcomes.

    Main fast-variable specification uses observed NO2 years only to avoid using
    imputed dependent variables inside causal inference.
    """
    treatment_specs = [
        ("infra", "direct_core_infra"),
        ("digital", "direct_core_digital"),
        ("eco_reg", "direct_core_eco_reg"),
    ]
    outcomes = [
        {"outcome": "log_viirs_ntl", "response_speed": "slow", "sample_rule": "all_years_observed_slow_proxy"},
        {"outcome": "no2_trop_anomaly_mean", "response_speed": "fast", "sample_rule": "observed_no2_only_2018_2025"},
    ]

    summary_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []

    for bucket, suffix in treatment_specs:
        variant_panel, variant_meta = _apply_treatment_variant(panel, suffix)
        if variant_meta.get("status") != "ok":
            for outcome_spec in outcomes:
                summary_rows.append(
                    {
                        "policy_bucket": bucket,
                        "outcome": outcome_spec["outcome"],
                        "response_speed": outcome_spec["response_speed"],
                        "status": "skipped",
                        "reason": variant_meta.get("reason", "invalid_treatment_variant"),
                    }
                )
            continue

        for outcome_spec in outcomes:
            outcome = str(outcome_spec["outcome"])
            response_speed = str(outcome_spec["response_speed"])
            work = variant_panel.copy()
            if outcome not in work.columns:
                summary_rows.append(
                    {
                        "policy_bucket": bucket,
                        "outcome": outcome,
                        "response_speed": response_speed,
                        "status": "skipped",
                        "reason": "missing_outcome_column",
                    }
                )
                continue

            work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
            if outcome.startswith("no2_"):
                observed_mask = pd.to_numeric(work.get("has_no2_observation"), errors="coerce").fillna(0).astype(int) == 1
                work = work.loc[observed_mask & work[outcome].notna()].copy()
            else:
                work = work.loc[work[outcome].notna()].copy()

            if len(work) < 80:
                summary_rows.append(
                    {
                        "policy_bucket": bucket,
                        "outcome": outcome,
                        "response_speed": response_speed,
                        "status": "skipped",
                        "reason": "too_few_rows_after_sample_rule",
                        "n_obs": int(len(work)),
                    }
                )
                continue

            if int(work["did_treatment"].nunique()) < 2:
                summary_rows.append(
                    {
                        "policy_bucket": bucket,
                        "outcome": outcome,
                        "response_speed": response_speed,
                        "status": "skipped",
                        "reason": "no_treatment_variation",
                        "n_obs": int(len(work)),
                    }
                )
                continue

            policy_year = _resolve_policy_reference_year(work, fallback_year=2020)
            dml = run_dml_did(work, outcome=outcome)
            event = run_event_study_fe(work, outcome=outcome, treatment_year=policy_year, min_lag=-3, max_lead=3)

            summary_rows.append(
                {
                    "policy_bucket": bucket,
                    "outcome": outcome,
                    "response_speed": response_speed,
                    "status": "ok",
                    "sample_rule": outcome_spec["sample_rule"],
                    "policy_year": int(policy_year),
                    "n_obs": int(len(work)),
                    "treated_share": float(variant_meta.get("treated_share", np.nan)),
                    "dml_status": str(dml.get("status", "ok")) if isinstance(dml, dict) else "ok",
                    "dml_coef": _safe_float(dml.get("coef")) if isinstance(dml, dict) else None,
                    "dml_p_value": _safe_float(dml.get("p_value")) if isinstance(dml, dict) else None,
                    "event_status": str(event.get("status", "ok")) if isinstance(event, dict) else "ok",
                    "event_pretrend_joint_p_value": _safe_float(((event.get("meta") or {}) if isinstance(event, dict) else {}).get("pretrend_joint_p_value")),
                }
            )
            if isinstance(event, dict) and event.get("status") == "ok":
                for point in event.get("points", []):
                    event_rows.append(
                        {
                            "policy_bucket": bucket,
                            "outcome": outcome,
                            "response_speed": response_speed,
                            "sample_rule": outcome_spec["sample_rule"],
                            "policy_year": int(policy_year),
                            "rel_year": int(point.get("rel_year")),
                            "coef": _safe_float(point.get("coef")),
                            "stderr": _safe_float(point.get("stderr")),
                            "t_value": _safe_float(point.get("t_value")),
                            "p_value": _safe_float(point.get("p_value")),
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)
    event_df = pd.DataFrame(event_rows)
    summary_path = DATA_OUTPUTS / "policy_type_response_grid.csv"
    event_path = DATA_OUTPUTS / "policy_type_event_study.csv"
    summary_df.to_csv(summary_path, index=False)
    event_df.to_csv(event_path, index=False)

    payload = {
        "status": "ok",
        "summary_file": str(summary_path),
        "event_file": str(event_path),
        "rows": summary_rows,
        "event_row_count": int(len(event_df)),
        "main_fast_spec_note": "NO2 outcomes use observed years only; imputed NO2 is excluded from causal outcome estimation.",
    }
    dump_json(DATA_OUTPUTS / "policy_type_response_summary.json", payload)
    return payload


def run_policy_type_no2_backcast_robustness(panel: pd.DataFrame) -> Dict[str, object]:
    """Appendix-only robustness using NO2 backcasted outcomes on the full sample."""
    filled_panel, backcast_summary = add_no2_backcast_features(
        panel.copy(),
        fit_end_year=2023,
        output_stub="econometrics_appendix",
    )
    treatment_specs = [
        ("infra", "direct_core_infra"),
        ("digital", "direct_core_digital"),
        ("eco_reg", "direct_core_eco_reg"),
    ]
    rows: List[Dict[str, object]] = []
    for bucket, suffix in treatment_specs:
        work, variant_meta = _apply_treatment_variant(filled_panel, suffix)
        if variant_meta.get("status") != "ok":
            rows.append(
                {
                    "policy_bucket": bucket,
                    "status": "skipped",
                    "reason": variant_meta.get("reason", "invalid_treatment_variant"),
                }
            )
            continue
        outcome = "no2_trop_anomaly_mean_filled"
        work[outcome] = pd.to_numeric(work.get(outcome), errors="coerce")
        work = work.loc[work[outcome].notna()].copy()
        if len(work) < 80 or int(work["did_treatment"].nunique()) < 2:
            rows.append(
                {
                    "policy_bucket": bucket,
                    "status": "skipped",
                    "reason": "insufficient_rows_or_variation",
                    "n_obs": int(len(work)),
                }
            )
            continue
        policy_year = _resolve_policy_reference_year(work, fallback_year=2020)
        dml = run_dml_did(work, outcome=outcome)
        event = run_event_study_fe(work, outcome=outcome, treatment_year=policy_year, min_lag=-3, max_lead=3)
        event_points = event.get("points", []) if isinstance(event, dict) else []
        post_points = [p for p in event_points if int(p.get("rel_year", -999)) >= 0]
        mean_post = float(np.mean([float(p.get("coef")) for p in post_points])) if post_points else np.nan
        rows.append(
            {
                "policy_bucket": bucket,
                "status": "ok",
                "outcome": outcome,
                "sample_rule": "appendix_backcasted_no2_full_sample_2015_2025",
                "policy_year": int(policy_year),
                "n_obs": int(len(work)),
                "treated_share": float(variant_meta.get("treated_share", np.nan)),
                "dml_status": str(dml.get("status", "ok")) if isinstance(dml, dict) else "ok",
                "dml_coef": _safe_float(dml.get("coef")) if isinstance(dml, dict) else None,
                "dml_p_value": _safe_float(dml.get("p_value")) if isinstance(dml, dict) else None,
                "event_status": str(event.get("status", "ok")) if isinstance(event, dict) else "ok",
                "event_pretrend_joint_p_value": _safe_float(((event.get("meta") or {}) if isinstance(event, dict) else {}).get("pretrend_joint_p_value")),
                "event_mean_post_coef": mean_post,
                "filled_rows_total": int(backcast_summary.get("filled_rows_total", 0) or 0),
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = DATA_OUTPUTS / "policy_type_no2_backcast_robustness.csv"
    out_df.to_csv(out_path, index=False)
    payload = {
        "status": "ok",
        "output_file": str(out_path),
        "backcast_summary": backcast_summary,
        "rows": rows,
    }
    dump_json(DATA_OUTPUTS / "policy_type_no2_backcast_robustness_summary.json", payload)
    return payload


def run_synthetic_control(
    panel: pd.DataFrame,
    treatment_year: int = 2020,
    outcome: str = "composite_index",
) -> Dict[str, object]:
    """Construct a synthetic control for one treated city using ridge-regularized weights."""
    score_col = outcome if outcome in panel.columns else "composite_index"
    treated_cities = (
        panel.loc[panel["treated_city"] == 1, ["city_id", "city_name", score_col]]
        .groupby(["city_id", "city_name"], as_index=False)
        .mean()
        .sort_values(score_col, ascending=False)
    )
    if treated_cities.empty:
        return {"status": "skipped", "reason": "no_treated_cities"}

    treated_city = str(treated_cities.iloc[0]["city_id"])
    treated_name = str(treated_cities.iloc[0]["city_name"])

    cols = ["city_id", "year", outcome, "treated_city"]
    if "treatment_cohort_year" in panel.columns:
        cols.append("treatment_cohort_year")
    df = panel[cols].copy()
    if "treatment_cohort_year" in df.columns:
        cohort_map = (
            df[["city_id", "treatment_cohort_year"]]
            .drop_duplicates("city_id")
            .assign(treatment_cohort_year=lambda x: pd.to_numeric(x["treatment_cohort_year"], errors="coerce").fillna(9999))
            .set_index("city_id")["treatment_cohort_year"]
            .astype(float)
        )
    else:
        cohort_map = (
            df[["city_id", "treated_city"]]
            .drop_duplicates("city_id")
            .assign(treatment_cohort_year=lambda x: np.where(x["treated_city"] == 1, float(treatment_year), 9999.0))
            .set_index("city_id")["treatment_cohort_year"]
            .astype(float)
        )

    pre_years = sorted(df.loc[df["year"] < treatment_year, "year"].unique())
    post_years = sorted(df.loc[df["year"] >= treatment_year, "year"].unique())

    if len(pre_years) < 3:
        return {
            "status": "skipped",
            "reason": "insufficient_pre_periods",
            "required": 3,
            "available": int(len(pre_years)),
        }
    if not post_years:
        return {"status": "skipped", "reason": "no_post_periods"}

    y1_pre_full = (
        df[(df["city_id"] == treated_city) & (df["year"].isin(pre_years))]
        .sort_values("year")[["year", outcome]]
        .set_index("year")[outcome]
        .reindex(pre_years)
        .to_numpy(dtype=float)
    )
    valid_mask = np.isfinite(y1_pre_full)
    if int(valid_mask.sum()) < 3:
        return {
            "status": "skipped",
            "reason": "treated_city_missing_pre_values",
            "required": 3,
            "available": int(valid_mask.sum()),
        }
    valid_pre_years = [int(y) for y, ok in zip(pre_years, valid_mask.tolist()) if ok]
    y1_pre = y1_pre_full[valid_mask]

    donor_ids = sorted(
        [
            cid
            for cid, cy in cohort_map.items()
            if (str(cid) != treated_city) and np.isfinite(float(cy)) and float(cy) > float(treatment_year)
        ]
    )
    if len(donor_ids) < 5:
        donor_ids = sorted(
            [
                cid
                for cid in df.loc[df["treated_city"] == 0, "city_id"].astype(str).unique().tolist()
                if cid != treated_city
            ]
        )
    donor_matrix = []
    valid_donors = []
    for donor in donor_ids:
        series = (
            df[(df["city_id"] == donor) & (df["year"].isin(valid_pre_years))]
            .sort_values("year")[["year", outcome]]
            .set_index("year")[outcome]
            .reindex(valid_pre_years)
            .to_numpy(dtype=float)
        )
        if len(series) != len(y1_pre) or not np.isfinite(series).all():
            continue
        donor_matrix.append(series)
        valid_donors.append(donor)

    if len(valid_donors) < 5:
        return {
            "status": "skipped",
            "reason": "insufficient_donor_cities",
            "required": 5,
            "available": int(len(valid_donors)),
        }

    y0_pre = np.column_stack(donor_matrix)
    ridge = 1e-3
    w = np.linalg.pinv(y0_pre.T @ y0_pre + ridge * np.eye(y0_pre.shape[1])) @ (y0_pre.T @ y1_pre)
    w = np.clip(w, 0.0, None)
    if w.sum() == 0:
        w = np.ones_like(w) / len(w)
    else:
        w /= w.sum()

    years = sorted(df["year"].unique())
    treated_series = (
        df[df["city_id"] == treated_city].sort_values("year")[["year", outcome]].set_index("year")[outcome].reindex(years)
    )

    donor_series = []
    for donor in valid_donors:
        s = (
            df[df["city_id"] == donor].sort_values("year")[["year", outcome]].set_index("year")[outcome].reindex(years)
        )
        donor_series.append(s.to_numpy(dtype=float))

    y0_all = np.column_stack(donor_series)
    synthetic = y0_all @ w

    compare = pd.DataFrame(
        {
            "year": years,
            "actual": treated_series.to_numpy(dtype=float),
            "synthetic": synthetic,
        }
    )
    compare["gap"] = compare["actual"] - compare["synthetic"]
    compare["post"] = (compare["year"] >= treatment_year).astype(int)

    post_gaps = compare.loc[compare["post"] == 1, "gap"]
    if post_gaps.empty:
        return {"status": "skipped", "reason": "empty_post_gaps"}
    ate_post = float(post_gaps.mean())

    top_weights = pd.DataFrame({"city_id": valid_donors, "weight": w}).sort_values("weight", ascending=False).head(10)

    compare.to_csv(DATA_OUTPUTS / "synthetic_control_timeseries.csv", index=False)
    top_weights.to_csv(DATA_OUTPUTS / "synthetic_control_weights.csv", index=False)

    LOGGER.info("Synthetic control done for %s, ATE_post=%.4f", treated_name, ate_post)
    return {
        "status": "ok",
        "treated_city": treated_city,
        "treated_name": treated_name,
        "treatment_year": treatment_year,
        "ate_post": ate_post,
        "donor_count": int(len(valid_donors)),
    }


def _track_snapshot(
    *,
    status: str,
    design_variant: str,
    policy_reference_year: int | None,
    treated_share: float | None,
    track: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Normalize a track payload into a comparable variant snapshot."""
    if track is None:
        track = {}
    return {
        "status": str(status),
        "design_variant": str(design_variant),
        "policy_reference_year": int(policy_reference_year) if policy_reference_year is not None else None,
        "treated_share": float(treated_share) if treated_share is not None else None,
        "did_two_way_fe": track.get("did_two_way_fe", {}),
        "did_twfe_cluster_bootstrap": track.get("did_twfe_cluster_bootstrap", {}),
        "did_twfe_permutation": track.get("did_twfe_permutation", {}),
        "did_twfe_wild_bootstrap": track.get("did_twfe_wild_bootstrap", {}),
        "did_twfe_lead_placebo": track.get("did_twfe_lead_placebo", {}),
        "did_stacked_lead_placebo": track.get("did_stacked_lead_placebo", {}),
        "did_intense_contrast": track.get("did_intense_contrast", {}),
        "did_dose_response": track.get("did_dose_response", {}),
        "did_dose_response_external_direct": track.get("did_dose_response_external_direct", {}),
        "did_dose_response_bins": track.get("did_dose_response_bins", {}),
        "did_dose_response_bins_external_direct": track.get("did_dose_response_bins_external_direct", {}),
        "did_spillover_twfe": track.get("did_spillover_twfe", {}),
        "did_spillover_twfe_dose": track.get("did_spillover_twfe_dose", {}),
        "did_matched_trend": track.get("did_matched_trend", {}),
        "staggered_did": track.get("staggered_did", {}),
        "not_yet_treated_did": track.get("not_yet_treated_did", {}),
        "dml_did": track.get("dml_did", {}),
        "dr_did": track.get("dr_did", {}),
        "dynamic_phase_heterogeneity": track.get("dynamic_phase_heterogeneity", {}),
        "dynamic_phase_rule_sensitivity": track.get("dynamic_phase_rule_sensitivity", {}),
        "synthetic_control": track.get("synthetic_control", {}),
        "event_study_fe": track.get("event_study_fe", {}),
        "mechanism_decomposition": track.get("mechanism_decomposition", {}),
        "identification_scorecard": track.get("identification_scorecard", {}),
    }


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if not np.isfinite(out):
            return None
        return out
    except Exception:  # noqa: BLE001
        return None


def _safe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:  # noqa: BLE001
        return None


def _twfe_p_value_from_t(did_obj: Dict[str, object]) -> float | None:
    t_val = _safe_float(did_obj.get("t_value"))
    if t_val is None:
        return None
    n_cl = int(did_obj.get("n_clusters", 2))
    return _t_pvalue(t_val, max(1, n_cl - 1))


def _build_policy_source_sensitivity_table(
    variants: Dict[str, Dict[str, object]],
    *,
    main_preferred_effect: float | None = None,
    main_sign_reference_effect: float | None = None,
) -> pd.DataFrame:
    """Build a compact cross-design DID sensitivity table for paper reporting."""
    rows: List[Dict[str, object]] = []
    for variant_name, variant in variants.items():
        if not isinstance(variant, dict):
            continue

        status = str(variant.get("status", "unknown"))
        did = variant.get("did_two_way_fe", {}) if isinstance(variant.get("did_two_way_fe", {}), dict) else {}
        boot = (
            variant.get("did_twfe_cluster_bootstrap", {})
            if isinstance(variant.get("did_twfe_cluster_bootstrap", {}), dict)
            else {}
        )
        perm = variant.get("did_twfe_permutation", {}) if isinstance(variant.get("did_twfe_permutation", {}), dict) else {}
        wild = (
            variant.get("did_twfe_wild_bootstrap", {})
            if isinstance(variant.get("did_twfe_wild_bootstrap", {}), dict)
            else {}
        )
        lead = variant.get("did_twfe_lead_placebo", {}) if isinstance(variant.get("did_twfe_lead_placebo", {}), dict) else {}
        lead_stacked = (
            variant.get("did_stacked_lead_placebo", {})
            if isinstance(variant.get("did_stacked_lead_placebo", {}), dict)
            else {}
        )
        id_card = variant.get("identification_scorecard", {}) if isinstance(variant.get("identification_scorecard", {}), dict) else {}
        pref = id_card.get("preferred", {}) if isinstance(id_card.get("preferred", {}), dict) else {}
        dr = variant.get("dr_did", {}) if isinstance(variant.get("dr_did", {}), dict) else {}
        dml = variant.get("dml_did", {}) if isinstance(variant.get("dml_did", {}), dict) else {}
        intense = (
            variant.get("did_intense_contrast", {})
            if isinstance(variant.get("did_intense_contrast", {}), dict)
            else {}
        )
        dose = (
            variant.get("did_dose_response", {})
            if isinstance(variant.get("did_dose_response", {}), dict)
            else {}
        )
        dose_ext = (
            variant.get("did_dose_response_external_direct", {})
            if isinstance(variant.get("did_dose_response_external_direct", {}), dict)
            else {}
        )

        did_coef = _safe_float(did.get("coef"))
        did_t = _safe_float(did.get("t_value"))
        did_se = _safe_float(did.get("stderr"))
        did_n = _safe_int(did.get("n_obs"))
        did_p = _twfe_p_value_from_t(did)
        boot_p = _safe_float(boot.get("p_value_two_sided"))
        perm_p = _safe_float(perm.get("p_value_abs_t"))
        wild_p = _safe_float(wild.get("p_value_abs_t"))
        lead_obj = lead_stacked if lead_stacked.get("status") == "ok" else lead
        lead_share = _safe_float(lead_obj.get("share_p_lt_0_10"))
        lead_method = "stacked" if lead_obj is lead_stacked else "twfe"

        pref_est = pref.get("estimator")
        pref_eff = _safe_float(pref.get("effect"))
        pref_p = _safe_float(pref.get("p_value"))
        pref_cred = _safe_float(pref.get("credibility_score"))
        intense_p = _safe_float(intense.get("p_value"))
        dose_p = _safe_float(dose.get("p_value"))
        dose_ext_p = _safe_float(dose_ext.get("p_value"))

        t_component = _clip01(abs(did_t) / 2.58) if did_t is not None else 0.0
        perm_component = _clip01((0.10 - perm_p) / 0.10) if perm_p is not None else 0.0
        wild_component = _clip01((0.10 - wild_p) / 0.10) if wild_p is not None else 0.0
        lead_component = _clip01((0.34 - lead_share) / 0.34) if lead_share is not None else 0.0
        cred_component = _clip01((pref_cred or 0.0) / 100.0)
        contrast_components = []
        for p in [intense_p, dose_p, dose_ext_p]:
            if p is None:
                continue
            contrast_components.append(_clip01((0.10 - p) / 0.10))
        contrast_component = float(np.max(contrast_components)) if contrast_components else 0.0
        identification_strength = float(
            100.0
            * (
                0.26 * t_component
                + 0.18 * perm_component
                + 0.18 * wild_component
                + 0.14 * lead_component
                + 0.14 * cred_component
                + 0.10 * contrast_component
            )
        )
        pass_count = int(
            int(perm_p is not None and perm_p < 0.10)
            + int(wild_p is not None and wild_p < 0.10)
            + int(lead_share is not None and lead_share <= 0.34)
            + int((pref_cred or 0.0) >= 55.0)
            + int(
                (intense_p is not None and intense_p < 0.10)
                or (dose_p is not None and dose_p < 0.10)
                or (dose_ext_p is not None and dose_ext_p < 0.10)
            )
        )

        sign_ref_eff = _safe_float(intense.get("coef")) if intense.get("status") == "ok" else pref_eff
        sign_target_eff = main_sign_reference_effect if main_sign_reference_effect is not None else main_preferred_effect
        sign_consistency = None
        if sign_target_eff is not None and sign_ref_eff is not None:
            try:
                sign_consistency = int(np.sign(float(sign_target_eff)) == np.sign(float(sign_ref_eff)))
            except Exception:  # noqa: BLE001
                sign_consistency = None

        rows.append(
            {
                "variant": str(variant_name),
                "status": status,
                "design_variant": str(variant.get("design_variant", variant_name)),
                "policy_reference_year": _safe_int(variant.get("policy_reference_year")),
                "treated_share": _safe_float(variant.get("treated_share")),
                "did_coef": did_coef,
                "did_t_value": did_t,
                "did_stderr": did_se,
                "did_p_value": did_p,
                "did_n_obs": did_n,
                "bootstrap_p_value": boot_p,
                "permutation_p_value": perm_p,
                "wild_bootstrap_p_value": wild_p,
                "lead_placebo_share_p_lt_0_10": lead_share,
                "lead_placebo_method": lead_method,
                "preferred_estimator": pref_est,
                "preferred_effect": pref_eff,
                "preferred_p_value": pref_p,
                "preferred_credibility_score": pref_cred,
                "sign_reference_effect": sign_ref_eff,
                "dr_coef": _safe_float(dr.get("coef")),
                "dr_p_value": _safe_float(dr.get("p_value")),
                "dml_coef": _safe_float(dml.get("coef")),
                "dml_p_value": _safe_float(dml.get("p_value")),
                "intense_coef": _safe_float(intense.get("coef")),
                "intense_p_value": intense_p,
                "dose_coef": _safe_float(dose.get("coef")),
                "dose_p_value": dose_p,
                "dose_external_direct_coef": _safe_float(dose_ext.get("coef")),
                "dose_external_direct_p_value": dose_ext_p,
                "identification_pass_count": pass_count,
                "identification_strength": identification_strength,
                "effect_sign_consistent_with_main": sign_consistency,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["status_ok"] = (out["status"] == "ok").astype(int)
    out = out.sort_values(
        ["status_ok", "identification_strength", "preferred_credibility_score", "variant"],
        ascending=[False, False, False, True],
    ).drop(columns=["status_ok"])
    return out.reset_index(drop=True)


def _score_main_design_candidate(
    variant_name: str,
    share: float,
    cohort_count: int,
    did_var: int,
) -> float:
    priority_map = {
        "direct_core": 1.00,
        "direct": 0.88,
        "evidence_a": 0.72,
        "intense_external_peak": 0.66,
        "intense_external_direct": 0.62,
        "external_direct": 0.48,
        "evidence_ab": 0.42,
    }
    priority = priority_map.get(str(variant_name), 0.35)
    share_balance = _clip01(1.0 - abs(float(share) - 0.45) / 0.30)
    cohort_support = _clip01(float(cohort_count) / 6.0)
    did_support = 1.0 if int(did_var) > 1 else 0.0
    return float(0.45 * priority + 0.30 * share_balance + 0.20 * cohort_support + 0.05 * did_support)


def _build_sign_reversal_diagnostic(summary: Dict[str, object]) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []

    def _add_row(name: str, effect: float | None, p_value: float | None, ci95: list[object] | None, family: str) -> None:
        if effect is None:
            return
        rows.append(
            {
                "estimator": str(name),
                "family": str(family),
                "effect": float(effect),
                "p_value": float(p_value) if p_value is not None else None,
                "ci95_low": _safe_float(ci95[0]) if isinstance(ci95, list) and len(ci95) >= 2 else None,
                "ci95_high": _safe_float(ci95[1]) if isinstance(ci95, list) and len(ci95) >= 2 else None,
                "sign": int(np.sign(float(effect))) if float(effect) != 0 else 0,
                "significant_p_lt_0_10": int((p_value is not None) and (float(p_value) < 0.10)),
            }
        )

    twfe = summary.get("did_two_way_fe", {}) if isinstance(summary.get("did_two_way_fe", {}), dict) else {}
    dml = summary.get("dml_did", {}) if isinstance(summary.get("dml_did", {}), dict) else {}
    dml_adv = dml.get("advanced_grouped_spec", {}) if isinstance(dml.get("advanced_grouped_spec", {}), dict) else {}
    dr = summary.get("dr_did", {}) if isinstance(summary.get("dr_did", {}), dict) else {}
    staggered = summary.get("staggered_did", {}) if isinstance(summary.get("staggered_did", {}), dict) else {}
    nyt = summary.get("not_yet_treated_did", {}) if isinstance(summary.get("not_yet_treated_did", {}), dict) else {}
    matched = summary.get("did_matched_trend", {}) if isinstance(summary.get("did_matched_trend", {}), dict) else {}
    spill = summary.get("did_spillover_twfe", {}) if isinstance(summary.get("did_spillover_twfe", {}), dict) else {}

    _add_row(
        "twfe_did",
        _safe_float(twfe.get("coef")),
        _safe_float(twfe.get("p_value")),
        [
            _safe_float(twfe.get("coef")) - 1.96 * _safe_float(twfe.get("stderr")) if _safe_float(twfe.get("coef")) is not None and _safe_float(twfe.get("stderr")) is not None else None,
            _safe_float(twfe.get("coef")) + 1.96 * _safe_float(twfe.get("stderr")) if _safe_float(twfe.get("coef")) is not None and _safe_float(twfe.get("stderr")) is not None else None,
        ],
        "pooled_twfe",
    )
    _add_row("dml_did", _safe_float(dml.get("coef")), _safe_float(dml.get("p_value")), None, "orthogonal_ml")
    _add_row(
        "dml_did_legacy_baseline",
        _safe_float((dml.get("legacy_baseline_spec") or {}).get("coef")) if isinstance(dml.get("legacy_baseline_spec", {}), dict) else None,
        _safe_float((dml.get("legacy_baseline_spec") or {}).get("p_value")) if isinstance(dml.get("legacy_baseline_spec", {}), dict) else None,
        None,
        "orthogonal_ml_audit",
    )
    _add_row(
        "dr_did",
        _safe_float(dr.get("coef")) if dr.get("status") == "ok" else None,
        _safe_float(dr.get("p_value")) if dr.get("status") == "ok" else None,
        None,
        "orthogonal_diff",
    )
    _add_row(
        "staggered_did",
        _safe_float(staggered.get("post_avg_att")) if staggered.get("status") == "ok" else None,
        _t_pvalue(float(staggered.get("post_avg_t_value", 0.0)), max(1, int(staggered.get("n_cohorts", 2)) - 1)) if staggered.get("status") == "ok" else None,
        staggered.get("post_avg_ci95") if staggered.get("status") == "ok" else None,
        "timing_robust",
    )
    _add_row(
        "not_yet_treated_did",
        _prefer_finite(nyt.get("robust_att_weighted"), nyt.get("att_weighted")) if nyt.get("status") == "ok" else None,
        _prefer_finite(nyt.get("robust_p_value_weighted"), nyt.get("p_value_weighted")) if nyt.get("status") == "ok" else None,
        (
            nyt.get("robust_ci95_weighted")
            if isinstance(nyt.get("robust_ci95_weighted"), list)
            and len(nyt.get("robust_ci95_weighted")) >= 2
            and (
                _safe_float(nyt.get("robust_ci95_weighted")[0]) is not None
                or _safe_float(nyt.get("robust_ci95_weighted")[1]) is not None
            )
            else nyt.get("ci95_weighted")
        )
        if nyt.get("status") == "ok"
        else None,
        "timing_robust",
    )
    _add_row(
        "matched_did",
        _safe_float(matched.get("coef")) if matched.get("status") == "ok" else None,
        _safe_float(matched.get("p_value")) if matched.get("status") == "ok" else None,
        None,
        "matched_design",
    )
    _add_row(
        "spillover_twfe_direct",
        _safe_float(spill.get("direct_coef")) if spill.get("status") == "ok" else None,
        _safe_float(spill.get("direct_p_value")) if spill.get("status") == "ok" else None,
        None,
        "spillover_design",
    )

    diag_df = pd.DataFrame(rows)
    out_csv = DATA_OUTPUTS / "econometric_sign_reversal_diagnostic.csv"
    diag_df.to_csv(out_csv, index=False)

    main_share = _safe_float(summary.get("main_treated_share"))
    pretrend_p = _safe_float(staggered.get("pretrend_joint_p_value")) if isinstance(staggered, dict) else None
    cohort_rows = nyt.get("by_cohort", []) if isinstance(nyt, dict) else []
    cohort_effects = [float(r["coef"]) for r in cohort_rows if _safe_float(r.get("coef")) is not None]
    cohort_sign_mix = bool(cohort_effects) and (np.min(cohort_effects) < 0.0) and (np.max(cohort_effects) > 0.0)

    loo_path = DATA_OUTPUTS / "experiment_leave_one_continent_out.csv"
    continent_sign_switch_share = None
    if loo_path.exists():
        try:
            loo = pd.read_csv(loo_path)
            vals = pd.to_numeric(loo.get("did_coef"), errors="coerce").dropna()
            if not vals.empty and _safe_float(twfe.get("coef")) is not None:
                base_sign = np.sign(float(twfe["coef"]))
                continent_sign_switch_share = float(np.mean(np.sign(vals.to_numpy(dtype=float)) != base_sign))
        except Exception:  # noqa: BLE001
            continent_sign_switch_share = None

    sig_rows = diag_df[diag_df["significant_p_lt_0_10"] == 1].copy() if not diag_df.empty else pd.DataFrame()
    significant_signs = sig_rows["sign"].dropna().astype(int).tolist() if not sig_rows.empty else []

    out = {
        "status": "ok",
        "main_design_variant": summary.get("main_design_variant"),
        "main_treated_share": main_share,
        "estimator_count": int(len(diag_df)),
        "significant_estimator_count_p_lt_0_10": int(len(sig_rows)),
        "significant_negative_count_p_lt_0_10": int(sum(1 for s in significant_signs if s < 0)),
        "significant_positive_count_p_lt_0_10": int(sum(1 for s in significant_signs if s > 0)),
        "high_treated_share_flag": bool(main_share is not None and main_share > 0.55),
        "staggered_pretrend_drift_flag": bool(pretrend_p is not None and pretrend_p < 0.05),
        "cohort_sign_mixing_flag": bool(cohort_sign_mix),
        "continent_sign_switch_share_vs_main_twfe": continent_sign_switch_share,
        "diagnostic_table_file": str(out_csv),
    }
    return out


def run_econometric_suite(panel: pd.DataFrame, fast_mode: bool = False) -> Dict[str, object]:
    """Execute econometric methods and persist summary.

    When `fast_mode=True`, the main track switches to lightweight diagnostics for fast iteration.
    """
    city_panel_raw = panel.copy()
    panel, causal_panel_design = _collapse_country_policy_panel_for_causal_inference(panel)
    policy_year_all_sources = _resolve_policy_reference_year(panel, fallback_year=2020)

    def _safe_run(name: str, fn, *args, **kwargs) -> Dict[str, object]:
        try:
            out = fn(*args, **kwargs)
            if isinstance(out, dict):
                return out
            return {"status": "ok", "value": out}
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Econometric block failed: %s", name)
            return {"status": "failed", "reason": str(exc)}

    def _run_track(
        track_name: str,
        panel_track: pd.DataFrame,
        policy_year_track: int,
        *,
        outcome: str = "composite_index",
        include_full: bool,
        nyt_window_pre: int = 2,
        nyt_window_post: int = 2,
        lightweight: bool = False,
    ) -> Dict[str, object]:
        _assert_valid_causal_outcome(outcome)
        boot_draws = 40 if lightweight else (120 if include_full and outcome == "composite_index" else 60)
        perm_draws = 50 if lightweight else (160 if include_full and outcome == "composite_index" else 80)
        wild_draws = 60 if lightweight else (180 if include_full and outcome == "composite_index" else 90)
        intense_did_col = "did_treatment_intense_external_direct"
        intense_treated_col = "treated_city_intense_external_direct"
        intense_specs = [
            ("did_treatment_intense_external_peak", "treated_city_intense_external_peak"),
            ("did_treatment_intense_external_direct", "treated_city_intense_external_direct"),
        ]

        def _same_as_main_treatment(col: str) -> bool:
            if "did_treatment" not in panel_track.columns or col not in panel_track.columns:
                return False
            lhs = pd.to_numeric(panel_track["did_treatment"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            rhs = pd.to_numeric(panel_track[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if lhs.shape != rhs.shape:
                return False
            return bool(np.array_equal(lhs, rhs))

        selected = False
        for did_col, treated_col in intense_specs:
            if did_col not in panel_track.columns or treated_col not in panel_track.columns:
                continue
            if int(pd.to_numeric(panel_track[did_col], errors="coerce").fillna(0.0).nunique()) < 2:
                continue
            if _same_as_main_treatment(did_col):
                intense_did_col = did_col
                intense_treated_col = treated_col
                selected = True
                break
        if not selected:
            for did_col, treated_col in intense_specs:
                if did_col not in panel_track.columns or treated_col not in panel_track.columns:
                    continue
                if int(pd.to_numeric(panel_track[did_col], errors="coerce").fillna(0.0).nunique()) < 2:
                    continue
                intense_did_col = did_col
                intense_treated_col = treated_col
                break

        timing_panel = panel_track
        timing_ready_design: Dict[str, object] = {"status": "fallback_raw", "reason": "not_requested"}
        if not lightweight:
            timing_panel, timing_ready_design = _build_timing_ready_panel(
                panel_track,
                treatment_year=int(policy_year_track),
                window_pre=int(nyt_window_pre),
                window_post=int(nyt_window_post),
                min_treated_cities_per_cohort=3,
            )
        track_panel = timing_panel if timing_ready_design.get("status") == "ok" else panel_track

        track: Dict[str, object] = {
            "outcome": outcome,
            "did_two_way_fe": _safe_run(f"{track_name}:did_two_way_fe", run_did_two_way_fe, track_panel, outcome=outcome),
            "did_intense_contrast": _safe_run(
                f"{track_name}:did_intense_contrast",
                run_intense_contrast_did,
                track_panel,
                outcome=outcome,
                did_col=intense_did_col,
                treated_col=intense_treated_col,
            ),
            "did_dose_response": _safe_run(
                f"{track_name}:did_dose_response",
                run_dose_response_twfe,
                track_panel,
                outcome=outcome,
                dose_col="policy_dose",
            ),
            "did_dose_response_external_direct": _safe_run(
                f"{track_name}:did_dose_response_external_direct",
                run_dose_response_twfe,
                track_panel,
                outcome=outcome,
                dose_col="policy_dose_external_direct",
                treated_col_for_trend="treated_city_external_direct",
            ),
            "did_dose_response_bins": _safe_run(
                f"{track_name}:did_dose_response_bins",
                run_dose_response_bins_twfe,
                track_panel,
                outcome=outcome,
                dose_col="policy_dose",
                n_bins=4,
            ),
            "did_dose_response_bins_external_direct": _safe_run(
                f"{track_name}:did_dose_response_bins_external_direct",
                run_dose_response_bins_twfe,
                track_panel,
                outcome=outcome,
                dose_col="policy_dose_external_direct",
                n_bins=4,
            ),
            "did_spillover_twfe": _safe_run(
                f"{track_name}:did_spillover_twfe",
                run_spillover_twfe,
                track_panel,
                outcome=outcome,
                treatment_col="did_treatment",
                k_neighbors=5,
                lag_years=1,
            ),
            "did_spillover_twfe_dose": _safe_run(
                f"{track_name}:did_spillover_twfe_dose",
                run_spillover_twfe,
                track_panel,
                outcome=outcome,
                treatment_col="policy_dose",
                k_neighbors=5,
                lag_years=1,
            ),
            "did_twfe_cluster_bootstrap": _safe_run(
                f"{track_name}:did_twfe_cluster_bootstrap",
                run_twfe_cluster_bootstrap,
                track_panel,
                outcome=outcome,
                draws=int(boot_draws),
            ),
            "did_twfe_permutation": _safe_run(
                f"{track_name}:did_twfe_permutation",
                run_twfe_city_permutation,
                track_panel,
                outcome=outcome,
                draws=int(perm_draws),
            ),
            "did_twfe_wild_bootstrap": _safe_run(
                f"{track_name}:did_twfe_wild_bootstrap",
                run_twfe_wild_cluster_bootstrap,
                track_panel,
                outcome=outcome,
                draws=int(wild_draws),
            ),
            "did_twfe_lead_placebo": _safe_run(
                f"{track_name}:did_twfe_lead_placebo",
                run_twfe_lead_placebo,
                track_panel,
                outcome=outcome,
                lead_years=[1, 2],
                track_label=track_name,
            ),
        }
        track["timing_ready_design"] = timing_ready_design
        timing_input = track_panel
        track["did_stacked_lead_placebo"] = _safe_run(
            f"{track_name}:did_stacked_lead_placebo",
            run_stacked_lead_placebo,
            timing_input,
            outcome=outcome,
            lead_years=[1, 2],
            window_pre=2,
            window_post=2,
            track_label=track_name,
        )
        track["mechanism_decomposition"] = _safe_run(
            f"{track_name}:mechanism_decomposition",
            run_mechanism_decomposition,
            track_panel,
            treatment_col="did_treatment",
            outcome_ref=outcome if outcome in track_panel.columns else "composite_index",
            track_label=track_name,
        )
        if lightweight:
            track["did_matched_trend"] = {"status": "skipped", "reason": "lightweight_track_mode"}
            track["staggered_did"] = {"status": "skipped", "reason": "lightweight_track_mode"}
            track["not_yet_treated_did"] = {"status": "skipped", "reason": "lightweight_track_mode"}
            track["dml_did"] = {"status": "skipped", "reason": "lightweight_track_mode"}
            track["dr_did"] = {"status": "skipped", "reason": "lightweight_track_mode"}
            track["synthetic_control"] = {"status": "skipped", "reason": "lightweight_track_mode"}
        else:
            track["did_matched_trend"] = _safe_run(
                f"{track_name}:did_matched_trend",
                run_matched_did_with_trend,
                track_panel,
                outcome=outcome,
                treatment_year=policy_year_track,
            )
            track["staggered_did"] = _safe_run(
                f"{track_name}:staggered_did",
                run_staggered_did,
                timing_input,
                outcome=outcome,
                treatment_year=policy_year_track,
                strict_pretrend_screen=bool(timing_ready_design.get("status") == "ok"),
            )
            track["not_yet_treated_did"] = _safe_run(
                f"{track_name}:not_yet_treated_did",
                run_not_yet_treated_did,
                timing_input,
                outcome=outcome,
                treatment_year=policy_year_track,
                window_pre=int(nyt_window_pre),
                window_post=int(nyt_window_post),
            )
            track["dml_did"] = _safe_run(f"{track_name}:dml_did", run_dml_did, track_panel, outcome=outcome)
            track["dr_did"] = _safe_run(f"{track_name}:dr_did", run_dr_did, track_panel, outcome=outcome)
            track["synthetic_control"] = _safe_run(
                f"{track_name}:synthetic_control",
                run_synthetic_control,
                track_panel,
                treatment_year=policy_year_track,
                outcome=outcome,
            )
        if include_full and outcome == "composite_index" and not lightweight:
            track["did_heterogeneity"] = _safe_run(f"{track_name}:did_heterogeneity", run_did_heterogeneity, track_panel)
            track["dynamic_phase_heterogeneity"] = _safe_run(
                f"{track_name}:dynamic_phase_heterogeneity",
                run_dynamic_phase_heterogeneity,
                track_panel,
                outcome=outcome,
                treatment_year=policy_year_track,
            )
            track["dynamic_phase_rule_sensitivity"] = _safe_run(
                f"{track_name}:dynamic_phase_rule_sensitivity",
                run_dynamic_phase_rule_sensitivity,
                track_panel,
                outcome=outcome,
                treatment_year=policy_year_track,
            )
            track["event_study_fe"] = _safe_run(
                f"{track_name}:event_study_fe",
                run_event_study_fe,
                track_panel,
                treatment_year=policy_year_track,
            )
            track["matrix_completion_counterfactual"] = _safe_run(
                f"{track_name}:matrix_completion_counterfactual",
                run_matrix_completion_counterfactual,
                track_panel,
                treatment_year=policy_year_track,
            )
            track["beta_convergence"] = _safe_run(f"{track_name}:beta_convergence", run_beta_convergence, track_panel)
            track["elasticity_fe"] = _safe_run(f"{track_name}:elasticity_fe", run_log_elasticity_fe, track_panel)
        track["identification_scorecard"] = run_identification_scorecard(track)
        return track

    causal_outcome = _resolve_primary_causal_outcome(panel)
    main_panel = panel.copy()
    main_design_variant = "direct_core_required_fallback_existing"
    main_treated_share = float(main_panel["treated_city"].mean()) if "treated_city" in main_panel.columns else float("nan")
    main_policy_year = int(policy_year_all_sources)

    design_candidates = [
        (
            "direct_core",
            {
                "treated_city_direct_core",
                "post_policy_direct_core",
                "did_treatment_direct_core",
                "treatment_cohort_year_direct_core",
            },
            "treated_city_direct_core",
            "post_policy_direct_core",
            "did_treatment_direct_core",
            "treatment_cohort_year_direct_core",
        ),
        (
            "direct",
            {
                "treated_city_direct",
                "post_policy_direct",
                "did_treatment_direct",
                "treatment_cohort_year_direct",
            },
            "treated_city_direct",
            "post_policy_direct",
            "did_treatment_direct",
            "treatment_cohort_year_direct",
        ),
    ]

    candidate_rows: List[Dict[str, object]] = []
    selected_candidate: Dict[str, object] | None = None
    selected_score = -1.0
    for variant_name, cols_needed, treated_col, post_col, did_col, cohort_col in design_candidates:
        if not cols_needed.issubset(set(panel.columns)):
            continue
        panel_cand = panel.copy()
        panel_cand["treated_city"] = panel_cand[treated_col].astype(int)
        panel_cand["post_policy"] = panel_cand[post_col].astype(int)
        panel_cand["did_treatment"] = panel_cand[did_col].astype(int)
        panel_cand["treatment_cohort_year"] = panel_cand[cohort_col].astype(int)
        share = float(panel_cand["treated_city"].mean())
        did_var = int(panel_cand["did_treatment"].nunique())
        cohort_vals = (
            pd.to_numeric(panel_cand["treatment_cohort_year"], errors="coerce")
            .replace(9999, np.nan)
            .dropna()
            .astype(int)
        )
        cohort_count = int(cohort_vals.nunique())
        eligible = bool(did_var > 1 and 0.05 <= share <= 0.90)
        score = _score_main_design_candidate(variant_name, share=share, cohort_count=cohort_count, did_var=did_var)
        candidate_rows.append(
            {
                "variant": str(variant_name),
                "treated_share": float(share),
                "did_variation": int(did_var),
                "cohort_count": int(cohort_count),
                "score": float(score),
                "eligible": int(eligible),
            }
        )
        if eligible and score > selected_score:
            selected_candidate = {
                "variant_name": variant_name,
                "panel": panel_cand,
                "share": share,
                "policy_year": _resolve_policy_reference_year(panel_cand, fallback_year=policy_year_all_sources),
            }
            selected_score = float(score)

    direct_core_required = {
        "treated_city_direct_core",
        "post_policy_direct_core",
        "did_treatment_direct_core",
        "treatment_cohort_year_direct_core",
    }
    if direct_core_required.issubset(set(panel.columns)):
        main_panel = panel.copy()
        main_panel["treated_city"] = main_panel["treated_city_direct_core"].astype(int)
        main_panel["post_policy"] = main_panel["post_policy_direct_core"].astype(int)
        main_panel["did_treatment"] = main_panel["did_treatment_direct_core"].astype(int)
        main_panel["treatment_cohort_year"] = main_panel["treatment_cohort_year_direct_core"].astype(int)
        main_design_variant = "direct_core"
        main_treated_share = float(main_panel["treated_city"].mean())
        main_policy_year = int(_resolve_policy_reference_year(main_panel, fallback_year=policy_year_all_sources))
    elif selected_candidate is not None:
        main_panel = selected_candidate["panel"]
        main_design_variant = str(selected_candidate["variant_name"])
        main_treated_share = float(selected_candidate["share"])
        main_policy_year = int(selected_candidate["policy_year"])

    main_track = _run_track(
        "main",
        main_panel,
        main_policy_year,
        outcome=causal_outcome,
        include_full=not bool(fast_mode),
        lightweight=bool(fast_mode),
    )
    main_es = main_track.get("event_study_fe")
    main_es_points = main_es.get("points", []) if isinstance(main_es, dict) else []
    if not main_es_points:
        main_track["event_study_fe"] = _safe_run(
            "main:event_study_fe_light",
            run_event_study_fe,
            main_panel,
            outcome=causal_outcome,
            treatment_year=main_policy_year,
            min_lag=-3,
            max_lead=3,
        )

    summary: Dict[str, object] = {
        "causal_panel_design": causal_panel_design,
        "policy_reference_year": int(main_policy_year),
        "policy_reference_year_all_sources": int(policy_year_all_sources),
        "causal_primary_outcome": str(causal_outcome),
        "main_design_selection_rule": "direct_core_only_with_direct_fallback",
        "main_design_variant": main_design_variant,
        "main_treated_share": float(main_treated_share),
        "main_design_candidates": sorted(candidate_rows, key=lambda r: float(r.get("score", 0.0)), reverse=True),
        "did_two_way_fe": main_track.get("did_two_way_fe", {}),
        "did_twfe_cluster_bootstrap": main_track.get("did_twfe_cluster_bootstrap", {}),
        "did_twfe_permutation": main_track.get("did_twfe_permutation", {}),
        "did_twfe_wild_bootstrap": main_track.get("did_twfe_wild_bootstrap", {}),
        "did_twfe_lead_placebo": main_track.get("did_twfe_lead_placebo", {}),
        "did_stacked_lead_placebo": main_track.get("did_stacked_lead_placebo", {}),
        "did_intense_contrast": main_track.get("did_intense_contrast", {}),
        "did_dose_response": main_track.get("did_dose_response", {}),
        "did_dose_response_external_direct": main_track.get("did_dose_response_external_direct", {}),
        "did_dose_response_bins": main_track.get("did_dose_response_bins", {}),
        "did_dose_response_bins_external_direct": main_track.get("did_dose_response_bins_external_direct", {}),
        "did_spillover_twfe": main_track.get("did_spillover_twfe", {}),
        "did_spillover_twfe_dose": main_track.get("did_spillover_twfe_dose", {}),
        "did_matched_trend": main_track.get("did_matched_trend", {}),
        "staggered_did": main_track.get("staggered_did", {}),
        "not_yet_treated_did": main_track.get("not_yet_treated_did", {}),
        "dml_did": main_track.get("dml_did", {}),
        "dr_did": main_track.get("dr_did", {}),
        "did_heterogeneity": main_track.get("did_heterogeneity", {}),
        "dynamic_phase_heterogeneity": main_track.get("dynamic_phase_heterogeneity", {}),
        "dynamic_phase_rule_sensitivity": main_track.get("dynamic_phase_rule_sensitivity", {}),
        "event_study_fe": main_track.get("event_study_fe", {}),
        "mechanism_decomposition": main_track.get("mechanism_decomposition", {}),
        "matrix_completion_counterfactual": main_track.get("matrix_completion_counterfactual", {}),
        "beta_convergence": main_track.get("beta_convergence", {}),
        "elasticity_fe": main_track.get("elasticity_fe", {}),
        "synthetic_control": main_track.get("synthetic_control", {}),
        "timing_ready_design": main_track.get("timing_ready_design", {}),
    }
    summary["identification_scorecard"] = main_track.get("identification_scorecard", {})
    policy_source_variants: Dict[str, Dict[str, object]] = {
        "main_design": _track_snapshot(
            status="ok",
            design_variant=str(main_design_variant),
            policy_reference_year=int(main_policy_year),
            treated_share=float(main_treated_share),
            track=main_track,
        )
    }

    direct_full_cols = {
        "treated_city_direct",
        "post_policy_direct",
        "did_treatment_direct",
        "treatment_cohort_year_direct",
    }
    direct_core_cols = {
        "treated_city_direct_core",
        "post_policy_direct_core",
        "did_treatment_direct_core",
        "treatment_cohort_year_direct_core",
    }
    if direct_full_cols.issubset(set(panel.columns)):
        panel_direct = panel.copy()
        if direct_core_cols.issubset(set(panel.columns)):
            panel_direct["treated_city"] = panel_direct["treated_city_direct_core"].astype(int)
            panel_direct["post_policy"] = panel_direct["post_policy_direct_core"].astype(int)
            panel_direct["did_treatment"] = panel_direct["did_treatment_direct_core"].astype(int)
            panel_direct["treatment_cohort_year"] = panel_direct["treatment_cohort_year_direct_core"].astype(int)
            direct_design_name = "direct_core"
        else:
            panel_direct["treated_city"] = panel_direct["treated_city_direct"].astype(int)
            panel_direct["post_policy"] = panel_direct["post_policy_direct"].astype(int)
            panel_direct["did_treatment"] = panel_direct["did_treatment_direct"].astype(int)
            panel_direct["treatment_cohort_year"] = panel_direct["treatment_cohort_year_direct"].astype(int)
            direct_design_name = "direct_full"
        direct_treated = int(panel_direct["treated_city"].sum())
        direct_variation = int(panel_direct["did_treatment"].nunique())
        if direct_treated > 0 and direct_variation > 1:
            policy_year_direct = _resolve_policy_reference_year(panel_direct, fallback_year=main_policy_year)
            direct_primary_outcome = _resolve_primary_causal_outcome(panel_direct)
            direct_track = _run_track(
                "direct",
                panel_direct,
                policy_year_direct,
                outcome=direct_primary_outcome,
                include_full=False,
                nyt_window_pre=2,
                nyt_window_post=2,
                lightweight=True,
            )

            main_pref = summary.get("identification_scorecard", {}).get("preferred", {}) if isinstance(summary.get("identification_scorecard"), dict) else {}
            direct_pref = direct_track.get("identification_scorecard", {}).get("preferred", {}) if isinstance(direct_track.get("identification_scorecard"), dict) else {}
            main_eff = main_pref.get("effect")
            direct_eff = direct_pref.get("effect")
            sign_consistency = None
            if main_eff is not None and direct_eff is not None:
                try:
                    sign_consistency = int(np.sign(float(main_eff)) == np.sign(float(direct_eff)))
                except Exception:  # noqa: BLE001
                    sign_consistency = None

            summary["direct_event_design"] = {
                "status": "ok",
                "design_variant": direct_design_name,
                "policy_reference_year": int(policy_year_direct),
                "primary_outcome": direct_primary_outcome,
                "did_two_way_fe": direct_track.get("did_two_way_fe", {}),
                "did_twfe_cluster_bootstrap": direct_track.get("did_twfe_cluster_bootstrap", {}),
                "did_twfe_permutation": direct_track.get("did_twfe_permutation", {}),
                "did_twfe_wild_bootstrap": direct_track.get("did_twfe_wild_bootstrap", {}),
                "did_twfe_lead_placebo": direct_track.get("did_twfe_lead_placebo", {}),
                "did_stacked_lead_placebo": direct_track.get("did_stacked_lead_placebo", {}),
                "did_matched_trend": direct_track.get("did_matched_trend", {}),
                "staggered_did": direct_track.get("staggered_did", {}),
                "not_yet_treated_did": direct_track.get("not_yet_treated_did", {}),
                "dml_did": direct_track.get("dml_did", {}),
                "dr_did": direct_track.get("dr_did", {}),
                "synthetic_control": direct_track.get("synthetic_control", {}),
                "identification_scorecard": direct_track.get("identification_scorecard", {}),
                "composite_outcome_robustness": {
                    "status": "skipped",
                    "reason": "composite_outcome_blocked_for_causal_econometrics",
                    "outcome": "composite_index",
                },
                "alignment_with_main": {
                    "main_preferred_estimator": main_pref.get("estimator"),
                    "main_preferred_effect": main_eff,
                    "direct_preferred_estimator": direct_pref.get("estimator"),
                    "direct_preferred_effect": direct_eff,
                    "effect_sign_consistency": sign_consistency,
                },
            }
            policy_source_variants["direct_event_design"] = _track_snapshot(
                status="ok",
                design_variant=str(direct_design_name),
                policy_reference_year=int(policy_year_direct),
                treated_share=float(panel_direct["treated_city"].mean()),
                track=direct_track,
            )
        else:
            summary["direct_event_design"] = {
                "status": "skipped",
                "reason": "insufficient_direct_treatment_variation",
                "design_variant": direct_design_name,
                "treated_rows": direct_treated,
                "did_variation": direct_variation,
            }
            policy_source_variants["direct_event_design"] = _track_snapshot(
                status="skipped",
                design_variant=str(direct_design_name),
                policy_reference_year=_safe_int(main_policy_year),
                treated_share=float(panel_direct["treated_city"].mean()),
                track=None,
            )
    else:
        summary["direct_event_design"] = {"status": "skipped", "reason": "direct_design_columns_missing"}
        policy_source_variants["direct_event_design"] = _track_snapshot(
            status="skipped",
            design_variant="direct_event_design",
            policy_reference_year=_safe_int(main_policy_year),
            treated_share=None,
            track=None,
        )

    evidence_design_specs = [
        (
            "evidence_a",
            {
                "treated_city_evidence_a",
                "post_policy_evidence_a",
                "did_treatment_evidence_a",
                "treatment_cohort_year_evidence_a",
            },
            "treated_city_evidence_a",
            "post_policy_evidence_a",
            "did_treatment_evidence_a",
            "treatment_cohort_year_evidence_a",
        ),
        (
            "evidence_ab",
            {
                "treated_city_evidence_ab",
                "post_policy_evidence_ab",
                "did_treatment_evidence_ab",
                "treatment_cohort_year_evidence_ab",
            },
            "treated_city_evidence_ab",
            "post_policy_evidence_ab",
            "did_treatment_evidence_ab",
            "treatment_cohort_year_evidence_ab",
        ),
    ]
    evidence_variants: Dict[str, object] = {}
    evidence_out: Dict[str, object] = {"status": "skipped", "reason": "evidence_design_columns_missing"}
    for variant_name, cols_needed, treated_col, post_col, did_col, cohort_col in evidence_design_specs:
        if not cols_needed.issubset(set(panel.columns)):
            evidence_variants[variant_name] = {"status": "skipped", "reason": "missing_columns"}
            continue

        panel_ev = panel.copy()
        panel_ev["treated_city"] = panel_ev[treated_col].astype(int)
        panel_ev["post_policy"] = panel_ev[post_col].astype(int)
        panel_ev["did_treatment"] = panel_ev[did_col].astype(int)
        panel_ev["treatment_cohort_year"] = panel_ev[cohort_col].astype(int)
        treated_rows = int(panel_ev["treated_city"].sum())
        did_var = int(panel_ev["did_treatment"].nunique())
        share = float(panel_ev["treated_city"].mean())

        if treated_rows <= 0 or did_var <= 1:
            evidence_variants[variant_name] = {
                "status": "skipped",
                "reason": "insufficient_evidence_treatment_variation",
                "design_variant": variant_name,
                "treated_rows": treated_rows,
                "did_variation": did_var,
            }
            continue

        policy_year_ev = _resolve_policy_reference_year(panel_ev, fallback_year=main_policy_year)
        ev_track = _run_track(
            f"evidence_{variant_name}",
            panel_ev,
            policy_year_ev,
            outcome=causal_outcome,
            include_full=False,
            nyt_window_pre=2,
            nyt_window_post=2,
            lightweight=True,
        )
        evidence_variant = _track_snapshot(
            status="ok",
            design_variant=str(variant_name),
            policy_reference_year=int(policy_year_ev),
            treated_share=float(share),
            track=ev_track,
        )
        evidence_variants[variant_name] = evidence_variant
        policy_source_variants[f"evidence_{variant_name}"] = evidence_variant
        if evidence_out.get("status") != "ok":
            evidence_out = evidence_variant

    a_pref = ((evidence_variants.get("evidence_a", {}).get("identification_scorecard") or {}).get("preferred") or {})
    ab_pref = ((evidence_variants.get("evidence_ab", {}).get("identification_scorecard") or {}).get("preferred") or {})
    sign_consistency = None
    if a_pref.get("effect") is not None and ab_pref.get("effect") is not None:
        try:
            sign_consistency = int(np.sign(float(a_pref["effect"])) == np.sign(float(ab_pref["effect"])))
        except Exception:  # noqa: BLE001
            sign_consistency = None

    summary["evidence_tier_design"] = evidence_out
    summary["evidence_tier_robustness"] = {
        "status": "ok" if any((v or {}).get("status") == "ok" for v in evidence_variants.values()) else "skipped",
        "variants": evidence_variants,
        "alignment": {
            "evidence_a_preferred_estimator": a_pref.get("estimator"),
            "evidence_a_preferred_effect": a_pref.get("effect"),
            "evidence_ab_preferred_estimator": ab_pref.get("estimator"),
            "evidence_ab_preferred_effect": ab_pref.get("effect"),
            "effect_sign_consistency": sign_consistency,
        },
    }

    source_channel_specs = [
        ("source_external_direct", "policy_intensity_external_direct", "external_direct"),
        ("source_objective_indicator", "policy_intensity_objective_indicator", "objective_indicator"),
        ("source_objective_macro", "policy_intensity_objective_macro", "objective_macro"),
        ("source_ai_inferred", "policy_intensity_ai_inferred", "ai_inferred"),
    ]
    source_variants: Dict[str, object] = {}
    source_event_rows: List[Dict[str, Any]] = []
    anchor_cols = ["city_id", "year", "treated_city", "post_policy", "did_treatment", "treatment_cohort_year"]
    anchor_panel = main_panel[anchor_cols].copy().rename(
        columns={
            "treated_city": "treated_city_anchor",
            "post_policy": "post_policy_anchor",
            "did_treatment": "did_treatment_anchor",
            "treatment_cohort_year": "treatment_cohort_year_anchor",
        }
    )
    for variant_name, intensity_col, source_label in source_channel_specs:
        if intensity_col not in panel.columns:
            source_variants[variant_name] = {"status": "skipped", "reason": f"missing_column_{intensity_col}"}
            continue

        panel_src = panel.merge(anchor_panel, on=["city_id", "year"], how="left")
        if panel_src["treated_city_anchor"].isna().any():
            source_variants[variant_name] = {"status": "skipped", "reason": "anchor_merge_failed"}
            continue

        panel_src["treated_city"] = pd.to_numeric(panel_src["treated_city_anchor"], errors="coerce").fillna(0).astype(int)
        panel_src["post_policy"] = pd.to_numeric(panel_src["post_policy_anchor"], errors="coerce").fillna(0).astype(int)
        panel_src["did_treatment"] = pd.to_numeric(panel_src["did_treatment_anchor"], errors="coerce").fillna(0).astype(int)
        panel_src["treatment_cohort_year"] = (
            pd.to_numeric(panel_src["treatment_cohort_year_anchor"], errors="coerce").fillna(9999).astype(int)
        )

        intensity = pd.to_numeric(panel_src[intensity_col], errors="coerce").fillna(0.0).astype(float)
        panel_src["policy_intensity"] = intensity
        panel_src["policy_dose"] = panel_src["post_policy"].astype(float) * panel_src["policy_intensity"]
        if source_label == "external_direct":
            panel_src["policy_dose_external_direct"] = panel_src["policy_dose"].astype(float)
        else:
            panel_src["policy_dose_external_direct"] = 0.0

        treated_rows = int(panel_src["treated_city"].sum())
        did_var = int(panel_src["did_treatment"].nunique())
        share = float(panel_src["treated_city"].mean())
        if treated_rows <= 0 or did_var <= 1:
            source_variants[variant_name] = {
                "status": "skipped",
                "reason": "insufficient_anchor_treatment_variation",
                "treated_rows": treated_rows,
                "did_variation": did_var,
                "treated_share": share,
            }
            continue

        policy_year_src = int(main_policy_year)
        src_track = _run_track(
            f"source_{variant_name}",
            panel_src,
            policy_year_src,
            outcome=causal_outcome,
            include_full=False,
            nyt_window_pre=2,
            nyt_window_post=2,
            lightweight=True,
        )
        src_event_study = _safe_run(
            f"source_{variant_name}:event_study_fe_light",
            run_event_study_fe,
            panel_src,
            outcome=causal_outcome,
            treatment_year=policy_year_src,
            min_lag=-3,
            max_lead=3,
        )
        src_track["event_study_fe"] = src_event_study
        source_variant = _track_snapshot(
            status="ok",
            design_variant=f"{variant_name}_common_shock",
            policy_reference_year=int(policy_year_src),
            treated_share=float(share),
            track=src_track,
        )
        source_variant["source_sensitivity_mode"] = "common_dynamic_shock_anchor"
        source_variant["anchor_design_variant"] = str(main_design_variant)
        source_variant["source_channel"] = str(source_label)
        source_variant["source_intensity_col"] = str(intensity_col)
        source_variant["source_intensity_nonzero_share"] = float(np.mean(intensity.to_numpy(dtype=float) > 0.0))
        source_variant["source_dose_mean_post"] = (
            float(panel_src.loc[panel_src["post_policy"] == 1, "policy_dose"].mean())
            if int((panel_src["post_policy"] == 1).sum()) > 0
            else 0.0
        )
        source_variants[variant_name] = source_variant
        policy_source_variants[variant_name] = source_variant

        es_points = src_event_study.get("points", []) if isinstance(src_event_study, dict) else []
        for p in es_points:
            rel = _safe_int(p.get("rel_year"))
            if rel is None:
                continue
            source_event_rows.append(
                {
                    "variant": str(variant_name),
                    "design_variant": f"{variant_name}_common_shock",
                    "policy_reference_year": int(policy_year_src),
                    "treated_share": float(share),
                    "source_channel": str(source_label),
                    "source_sensitivity_mode": "common_dynamic_shock_anchor",
                    "rel_year": int(rel),
                    "coef": _safe_float(p.get("coef")),
                    "stderr": _safe_float(p.get("stderr")),
                    "t_value": _safe_float(p.get("t_value")),
                }
            )

    summary["source_event_design_robustness"] = {
        "status": "ok" if any((v or {}).get("status") == "ok" for v in source_variants.values()) else "skipped",
        "comparison_mode": "common_dynamic_shock_anchor",
        "anchor_design_variant": str(main_design_variant),
        "variants": source_variants,
    }

    source_event_path = DATA_OUTPUTS / "econometric_source_event_study_points.csv"
    if source_event_rows:
        source_event_df = pd.DataFrame(source_event_rows).sort_values(["variant", "rel_year"]).reset_index(drop=True)
        source_event_df.to_csv(source_event_path, index=False)
        summary["source_event_study"] = {
            "status": "ok",
            "variants": sorted(source_event_df["variant"].astype(str).unique().tolist()),
            "n_rows": int(len(source_event_df)),
            "table_file": str(source_event_path),
        }
    else:
        pd.DataFrame().to_csv(source_event_path, index=False)
        summary["source_event_study"] = {
            "status": "skipped",
            "reason": "no_source_event_study_points",
            "table_file": str(source_event_path),
        }

    main_pref = (
        (summary.get("identification_scorecard", {}) or {}).get("preferred", {})
        if isinstance(summary.get("identification_scorecard", {}), dict)
        else {}
    )
    main_pref_effect = _safe_float(main_pref.get("effect")) if isinstance(main_pref, dict) else None
    main_intense = summary.get("did_intense_contrast", {}) if isinstance(summary.get("did_intense_contrast", {}), dict) else {}
    main_sign_reference_effect = _safe_float(main_intense.get("coef")) if main_intense.get("status") == "ok" else main_pref_effect
    sensitivity_table = _build_policy_source_sensitivity_table(
        policy_source_variants,
        main_preferred_effect=main_pref_effect,
        main_sign_reference_effect=main_sign_reference_effect,
    )
    sensitivity_path = DATA_OUTPUTS / "econometric_policy_source_sensitivity.csv"
    if sensitivity_table.empty:
        pd.DataFrame().to_csv(sensitivity_path, index=False)
        summary["policy_source_sensitivity"] = {
            "status": "skipped",
            "reason": "no_valid_policy_source_variants",
            "variants_considered": sorted(policy_source_variants.keys()),
            "table_file": str(sensitivity_path),
            "table": [],
        }
    else:
        sensitivity_table.to_csv(sensitivity_path, index=False)
        ok_table = sensitivity_table[sensitivity_table["status"] == "ok"].copy()
        sign_share = None
        sign_share_source = None
        if "effect_sign_consistent_with_main" in ok_table.columns and not ok_table.empty:
            sign_vals = pd.to_numeric(ok_table["effect_sign_consistent_with_main"], errors="coerce")
            sign_vals = sign_vals[sign_vals.notna()]
            if len(sign_vals) > 0:
                sign_share = float(sign_vals.mean())
            source_mask = ok_table["variant"].astype(str).str.startswith("source_")
            source_vals = pd.to_numeric(
                ok_table.loc[source_mask, "effect_sign_consistent_with_main"],
                errors="coerce",
            )
            source_vals = source_vals[source_vals.notna()]
            if len(source_vals) > 0:
                sign_share_source = float(source_vals.mean())
        best_row = sensitivity_table.iloc[0].to_dict()
        summary["policy_source_sensitivity"] = {
            "status": "ok",
            "variants_considered": sorted(policy_source_variants.keys()),
            "variant_count": int(len(ok_table)) if not ok_table.empty else int(len(sensitivity_table)),
            "table_file": str(sensitivity_path),
            "best_variant": best_row.get("variant"),
            "best_identification_strength": _safe_float(best_row.get("identification_strength")),
            "best_preferred_estimator": best_row.get("preferred_estimator"),
            "best_preferred_effect": _safe_float(best_row.get("preferred_effect")),
            "sign_consistency_with_main_share": sign_share_source if sign_share_source is not None else sign_share,
            "sign_consistency_with_main_share_all_variants": sign_share,
            "sign_reference_estimator": "did_intense_contrast_if_available_else_preferred",
            "table": sensitivity_table.where(pd.notna(sensitivity_table), None).to_dict(orient="records"),
        }

    summary["sign_reversal_diagnostic"] = _build_sign_reversal_diagnostic(summary)
    summary["policy_type_response_matrix"] = _safe_run(
        "policy_type_response_matrix",
        run_policy_type_response_matrix,
        city_panel_raw,
    )
    summary["policy_type_no2_backcast_robustness"] = _safe_run(
        "policy_type_no2_backcast_robustness",
        run_policy_type_no2_backcast_robustness,
        city_panel_raw,
    )

    dump_json(DATA_OUTPUTS / "econometric_summary.json", summary)
    return summary
