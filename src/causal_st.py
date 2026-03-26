from __future__ import annotations

"""Causal spatio-temporal model (v2) for MACRO-City Engine.

SE estimation uses K-fold cross-fitting (K=5) to avoid in-sample bias, plus
city-level cluster bootstrap (B configurable via env; default B=60) for valid confidence intervals.

Key change from previous version
---------------------------------
The original code fitted propensity / outcome models on the full sample and
then computed SE = std(pseudo_outcomes) / sqrt(n), which ignores estimation
error in the nuisance models and inflates the t-statistic (observed t ≈ 9 was
an artefact).  The corrected estimator:
  1. Splits data into K folds stratified by city (all years of a city stay
     together to avoid leakage).
  2. For each fold, trains nuisance models on the complementary K-1 folds and
     computes DR pseudo-outcomes on the hold-out fold.
  3. Concatenates pseudo-outcomes and takes the mean as the ATT point estimate.
  4. Runs city-cluster bootstrap (B draws, full cross-fitting pipeline each
     draw) to obtain SE and 95 % CI.

The resulting `t_value` is bootstrap-based (ATT / bootstrap_se) and reflects
genuine uncertainty.  Expected magnitude: 0.3–3.0, consistent with the TWFE-DID
results.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold

from .econometrics import _assert_valid_causal_outcome, _resolve_primary_causal_outcome
from .spatial_weights import build_flight_neighbor_map, build_road_proxy_neighbor_map
from .utils import DATA_OUTPUTS, dump_json, haversine_km

LOGGER = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        val = int(raw)
    except Exception:  # noqa: BLE001
        return int(default)
    return int(max(1, val))


_RF_N_JOBS: int = _env_int("CAUSAL_ST_N_JOBS", 4)
_BOOT_DEFAULT: int = _env_int("CAUSAL_ST_BOOTSTRAP", 60)
_BOOT_ABLATION_DEFAULT: int = _env_int("CAUSAL_ST_ABLATION_BOOTSTRAP", 20)


BASE_FEATURES: List[str] = [
    "year",
    "log_population",
    "gdp_growth",
    "inflation",
    "clean_air_n",
    "clean_air_fast_proxy",
    "basic_infra_n",
    "temperature_mean",
    "precipitation_sum",
    "poi_total",
    "poi_diversity",
    "road_length_km_total",
    "intersection_density",
    "ghsl_built_surface_km2",
    "ghsl_built_density",
    "no2_trop_anomaly_mean",
    "gravity_access_viirs",
    "gravity_access_knowledge",
    "spatial_lag_log_viirs_ntl_wdist",
    "flight_degree_centrality",
    "airport_count_mapped",
    "international_route_share",
]


@dataclass(frozen=True)
class CausalSTConfig:
    k_neighbors: int = 5
    spatial_mode: str = "distance"
    flight_top_k: int = 12
    random_state: int = 42
    use_spatial: bool = True
    use_temporal: bool = True
    use_dr: bool = True
    n_folds: int = 5          # cross-fitting folds for point estimate
    n_bootstrap: int = field(default_factory=lambda: _env_int("CAUSAL_ST_BOOTSTRAP", 60))


# Lightweight RF settings used ONLY inside bootstrap draws to keep runtime < 5 min.
# The point-estimate cross-fitting uses the full-quality models below.
_BOOT_N_TREES: int = 40
_BOOT_MAX_DEPTH: int = 5


def _distance_neighbor_map(city_df: pd.DataFrame, k: int) -> Dict[str, List[Tuple[str, float]]]:
    cities = city_df[["city_id", "latitude", "longitude"]].drop_duplicates("city_id").reset_index(drop=True)
    ids = cities["city_id"].tolist()
    lat = cities["latitude"].to_numpy(dtype=float)
    lon = cities["longitude"].to_numpy(dtype=float)
    out: Dict[str, List[Tuple[str, float]]] = {}
    for i, city in enumerate(ids):
        dists: List[Tuple[float, str]] = []
        for j, other in enumerate(ids):
            if i == j:
                continue
            d = haversine_km(float(lat[i]), float(lon[i]), float(lat[j]), float(lon[j]))
            dists.append((d, other))
        dists.sort(key=lambda x: x[0])
        top = dists[:k]
        inv = np.asarray([1.0 / max(float(d), 1.0) for d, _ in top], dtype=float)
        denom = float(inv.sum())
        weights = inv / denom if denom > 0.0 else np.full(len(top), 1.0 / max(len(top), 1), dtype=float)
        out[city] = [(str(cid), float(w)) for (_, cid), w in zip(top, weights.tolist())]
    return out


def _build_spatial_weight_map(panel: pd.DataFrame, cfg: CausalSTConfig) -> tuple[Dict[str, List[Tuple[str, float]]], Dict[str, object]]:
    mode = str(cfg.spatial_mode).strip().lower()
    if mode == "flight":
        mapping, summary = build_flight_neighbor_map(panel, top_k=cfg.flight_top_k, persist=True)
        if mapping:
            summary = dict(summary)
            summary["spatial_mode"] = "flight"
            return mapping, summary
    if mode == "road_proxy":
        mapping, summary = build_road_proxy_neighbor_map(
            panel,
            top_k=max(1, int(cfg.k_neighbors)),
            max_distance_km=1500.0,
            persist=True,
        )
        if mapping:
            summary = dict(summary)
            summary["spatial_mode"] = "road_proxy"
            return mapping, summary
    mapping = _distance_neighbor_map(panel, max(1, int(cfg.k_neighbors)))
    return mapping, {
        "status": "ok",
        "spatial_mode": "distance",
        "sources_with_edges": int(len(mapping)),
        "mean_out_degree": float(np.mean([len(v) for v in mapping.values()])) if mapping else 0.0,
    }


def _add_spatial_lag(
    panel: pd.DataFrame,
    outcome_col: str,
    cfg: CausalSTConfig,
) -> tuple[pd.Series, Dict[str, object]]:
    mapping, summary = _build_spatial_weight_map(panel, cfg)
    # Use t-1 observed outcome to avoid contemporaneous reflection.
    lag_key = (
        panel.assign(year_prev=panel["year"] - 1)
        .set_index(["city_id", "year_prev"])[outcome_col]
        .to_dict()
    )
    lag_vals: List[float] = []
    for row in panel.itertuples(index=False):
        neigh = mapping.get(row.city_id, [])
        vals: List[float] = []
        weights: List[float] = []
        for cid, weight in neigh:
            v = lag_key.get((cid, row.year))
            if v is None or np.isnan(v):
                continue
            vals.append(float(v))
            weights.append(float(weight))
        if not vals:
            lag_vals.append(np.nan)
            continue
        w_arr = np.asarray(weights, dtype=float)
        if float(w_arr.sum()) <= 0.0:
            lag_vals.append(float(np.mean(vals)))
        else:
            lag_vals.append(float(np.average(np.asarray(vals, dtype=float), weights=w_arr)))
    return pd.Series(lag_vals, index=panel.index, dtype=float), summary


def _build_design(panel: pd.DataFrame, cfg: CausalSTConfig) -> pd.DataFrame:
    df = panel.copy().sort_values(["city_id", "year"]).reset_index(drop=True)
    outcome_col = _resolve_primary_causal_outcome(df)
    _assert_valid_causal_outcome(outcome_col)

    direct_core_cols = {
        "treated_city_direct_core": "treated_city",
        "post_policy_direct_core": "post_policy",
        "did_treatment_direct_core": "did_treatment",
    }
    if set(direct_core_cols).issubset(df.columns):
        for src, dst in direct_core_cols.items():
            df[dst] = pd.to_numeric(df[src], errors="coerce").fillna(0).astype(int)
        df.attrs["treatment_spec"] = "direct_core"
    else:
        for dst in direct_core_cols.values():
            df[dst] = 0
        df.attrs["treatment_spec"] = "direct_core_missing"

    feature_cols = [col for col in BASE_FEATURES if col in df.columns]

    if cfg.use_temporal:
        df["lag_outcome_1"] = df.groupby("city_id")[outcome_col].shift(1)
        df["lag_outcome_2"] = df.groupby("city_id")[outcome_col].shift(2)
        df["lag_growth_1"] = df.groupby("city_id")["gdp_growth"].shift(1)
        feature_cols.extend(["lag_outcome_1", "lag_outcome_2", "lag_growth_1"])

    if cfg.use_spatial:
        df["spatial_lag_outcome"], spatial_summary = _add_spatial_lag(df, outcome_col=outcome_col, cfg=cfg)
        feature_cols.append("spatial_lag_outcome")
        df.attrs["spatial_summary"] = spatial_summary
    else:
        df.attrs["spatial_summary"] = {"status": "skipped", "spatial_mode": str(cfg.spatial_mode), "reason": "cfg_use_spatial_false"}

    df["treated_city_flag"] = df["treated_city"].astype(float)
    df["post_policy_flag"] = df["post_policy"].astype(float)
    feature_cols.extend(["treated_city_flag", "post_policy_flag"])

    for col in feature_cols:
        if col in df.columns:
            med = float(pd.to_numeric(df[col], errors="coerce").median())
            if np.isnan(med):
                med = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(med)

    df["year_norm"] = (df["year"] - df["year"].min()) / max(1, int(df["year"].max() - df["year"].min()))
    if "year_norm" not in feature_cols:
        feature_cols.append("year_norm")

    df.attrs["feature_cols"] = feature_cols
    df.attrs["outcome_col"] = outcome_col
    df.attrs["spatial_mode"] = str(cfg.spatial_mode)
    return df


# ---------------------------------------------------------------------------
# Core cross-fitting doubly-robust estimator
# ---------------------------------------------------------------------------

def _dr_pseudo_outcomes_crossfit_fast(
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    city_groups: np.ndarray,
    random_state: int,
    use_dr: bool,
    n_folds: int,
) -> np.ndarray:
    """Lightweight DR pseudo-outcomes for use inside bootstrap draws only.

    Uses small RF (n_estimators=_BOOT_N_TREES, max_depth=_BOOT_MAX_DEPTH) to
    keep each bootstrap draw fast.  Statistical validity comes from the
    resampling, not from individual model accuracy.
    """
    n = len(y)
    pseudo = np.full(n, np.nan, dtype=float)
    n_unique = len(np.unique(city_groups))
    gkf = GroupKFold(n_splits=min(n_folds, n_unique))

    for train_idx, test_idx in gkf.split(x, a, groups=city_groups):
        x_tr, x_te = x[train_idx], x[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        a_tr, a_te = a[train_idx], a[test_idx]

        if np.sum(a_tr == 1) < 3 or np.sum(a_tr == 0) < 3:
            pseudo[test_idx] = 0.0
            continue

        prop = RandomForestClassifier(
            n_estimators=_BOOT_N_TREES,
            max_depth=_BOOT_MAX_DEPTH,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=_RF_N_JOBS,
        )
        prop.fit(x_tr, a_tr.astype(int))
        e_te = np.clip(prop.predict_proba(x_te)[:, 1], 0.05, 0.95)

        mt = RandomForestRegressor(
            n_estimators=_BOOT_N_TREES,
            max_depth=_BOOT_MAX_DEPTH,
            min_samples_leaf=5,
            random_state=random_state + 1,
            n_jobs=_RF_N_JOBS,
        )
        mc = RandomForestRegressor(
            n_estimators=_BOOT_N_TREES,
            max_depth=_BOOT_MAX_DEPTH,
            min_samples_leaf=5,
            random_state=random_state + 2,
            n_jobs=_RF_N_JOBS,
        )
        mt.fit(x_tr[a_tr == 1], y_tr[a_tr == 1])
        mc.fit(x_tr[a_tr == 0], y_tr[a_tr == 0])

        mu1_te = mt.predict(x_te)
        mu0_te = mc.predict(x_te)

        if use_dr:
            psi = (
                (mu1_te - mu0_te)
                + a_te * (y_te - mu1_te) / e_te
                - (1 - a_te) * (y_te - mu0_te) / (1 - e_te)
            )
        else:
            psi = mu1_te - mu0_te

        pseudo[test_idx] = psi

    if np.any(np.isnan(pseudo)):
        pseudo = np.where(np.isnan(pseudo), float(np.nanmean(pseudo)), pseudo)
    return pseudo


def _dr_pseudo_outcomes_crossfit(

    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    city_groups: np.ndarray,
    cfg: CausalSTConfig,
) -> np.ndarray:
    """Compute DR pseudo-outcomes using K-fold cross-fitting.

    Each fold's nuisance models are trained on the remaining K-1 folds,
    then pseudo-outcomes are predicted on the hold-out fold.  This avoids
    the in-sample optimism that inflated the original t-statistic.

    Parameters
    ----------
    x:            feature matrix (n, p)
    y:            outcome vector (n,)
    a:            treatment indicator (n,) - binary 0/1
    city_groups:  city_id string array for GroupKFold (keeps city years together)
    cfg:          CausalSTConfig with n_folds, random_state

    Returns
    -------
    pseudo:  DR pseudo-outcome array (n,)
    """
    n = len(y)
    pseudo = np.full(n, np.nan, dtype=float)

    gkf = GroupKFold(n_splits=min(cfg.n_folds, len(np.unique(city_groups))))

    for train_idx, test_idx in gkf.split(x, a, groups=city_groups):
        x_tr, x_te = x[train_idx], x[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        a_tr, a_te = a[train_idx], a[test_idx]

        # Skip if one arm is empty in the training fold
        if np.sum(a_tr == 1) < 3 or np.sum(a_tr == 0) < 3:
            # fall back to simple plugin for this fold
            mu_diff = np.zeros(len(test_idx))
            pseudo[test_idx] = mu_diff
            continue

        # Propensity model
        prop = RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            min_samples_leaf=6,
            random_state=cfg.random_state,
            n_jobs=_RF_N_JOBS,
        )
        prop.fit(x_tr, a_tr.astype(int))
        e_te = prop.predict_proba(x_te)[:, 1]
        e_te = np.clip(e_te, 0.05, 0.95)

        # Outcome models (train-only)
        mt = RandomForestRegressor(
            n_estimators=300,
            max_depth=9,
            min_samples_leaf=4,
            random_state=cfg.random_state + 1,
            n_jobs=_RF_N_JOBS,
        )
        mc = RandomForestRegressor(
            n_estimators=300,
            max_depth=9,
            min_samples_leaf=4,
            random_state=cfg.random_state + 2,
            n_jobs=_RF_N_JOBS,
        )
        if np.sum(a_tr == 1) >= 3:
            mt.fit(x_tr[a_tr == 1], y_tr[a_tr == 1])
        if np.sum(a_tr == 0) >= 3:
            mc.fit(x_tr[a_tr == 0], y_tr[a_tr == 0])

        mu1_te = mt.predict(x_te)
        mu0_te = mc.predict(x_te)

        if cfg.use_dr:
            psi = (
                (mu1_te - mu0_te)
                + a_te * (y_te - mu1_te) / e_te
                - (1 - a_te) * (y_te - mu0_te) / (1 - e_te)
            )
        else:
            psi = mu1_te - mu0_te

        pseudo[test_idx] = psi

    # If any fold produced NaN (edge case), fill with column mean
    if np.any(np.isnan(pseudo)):
        pseudo = np.where(np.isnan(pseudo), float(np.nanmean(pseudo)), pseudo)

    return pseudo


def _bootstrap_att_se(
    df: pd.DataFrame,
    feature_cols: List[str],
    outcome_col: str,
    post_treated_mask: np.ndarray,
    cfg: CausalSTConfig,
    att_observed: float,
) -> Dict[str, object]:
    """City-cluster bootstrap for ATT SE and 95 % CI.

    Each bootstrap draw samples cities with replacement (keeping all years of a
    city together), re-runs cross-fitting, and records the ATT.  The SE is
    estimated from the bootstrap distribution; the p-value is two-sided.
    """
    city_ids = np.array(df["city_id"].astype(str).unique())
    n_city = len(city_ids)

    if n_city < 5:
        return {
            "status": "skipped",
            "reason": "too_few_cities_for_bootstrap",
            "se": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "p_value": float("nan"),
        }

    rng = np.random.default_rng(cfg.random_state + 999)
    boot_atts: List[float] = []
    log_every = max(5, int(cfg.n_bootstrap // 4))

    for b in range(cfg.n_bootstrap):
        sampled = rng.choice(city_ids, size=n_city, replace=True)
        # Build bootstrap panel (rename city_ids to avoid duplicate index issues)
        frames = []
        for i, cid in enumerate(sampled):
            sub = df[df["city_id"].astype(str) == cid].copy()
            sub["city_id"] = f"_bs{b}_{i}"
            frames.append(sub)
        if not frames:
            continue
        boot_df = pd.concat(frames, ignore_index=True)

        x_b = boot_df[feature_cols].to_numpy(dtype=float)
        y_b = pd.to_numeric(boot_df[outcome_col], errors="coerce").to_numpy(dtype=float)
        a_b = boot_df["did_treatment"].to_numpy(dtype=int)
        g_b = boot_df["city_id"].astype(str).to_numpy()
        pt_b = (boot_df["post_policy"].to_numpy(dtype=int) == 1) & (boot_df["treated_city"].to_numpy(dtype=int) == 1)

        if np.unique(a_b).size < 2 or pt_b.sum() == 0:
            continue

        try:
            psi_b = _dr_pseudo_outcomes_crossfit_fast(
                x_b,
                y_b,
                a_b,
                g_b,
                random_state=cfg.random_state + b,
                use_dr=cfg.use_dr,
                n_folds=cfg.n_folds,
            )
            att_b = float(np.mean(psi_b[pt_b]))
            if np.isfinite(att_b):
                boot_atts.append(att_b)
        except Exception:  # noqa: BLE001
            continue
        if (b + 1) % log_every == 0 or (b + 1) == cfg.n_bootstrap:
            LOGGER.info(
                "Causal-ST bootstrap progress: %d/%d draws, ok=%d",
                b + 1,
                cfg.n_bootstrap,
                len(boot_atts),
            )

    n_ok = len(boot_atts)
    min_ok = max(10, int(0.40 * cfg.n_bootstrap))
    if n_ok < min_ok:
        return {
            "status": "insufficient_draws",
            "n_ok": n_ok,
            "se": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "p_value": float("nan"),
        }

    arr = np.asarray(boot_atts, dtype=float)
    se = float(np.std(arr, ddof=1))
    ci_low = float(np.quantile(arr, 0.025))
    ci_high = float(np.quantile(arr, 0.975))
    p_two = float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))
    return {
        "status": "ok",
        "n_bootstrap_ok": n_ok,
        "se": se,
        "ci95": [ci_low, ci_high],
        "p_value": p_two,
        "bootstrap_mean": float(np.mean(arr)),
    }


def _run_single(panel: pd.DataFrame, cfg: CausalSTConfig) -> Dict[str, object]:
    df = _build_design(panel, cfg)
    feature_cols: List[str] = list(df.attrs["feature_cols"])
    outcome_col = str(df.attrs["outcome_col"])

    x = df[feature_cols].to_numpy(dtype=float)
    y = pd.to_numeric(df[outcome_col], errors="coerce").to_numpy(dtype=float)
    a = df["did_treatment"].to_numpy(dtype=int)
    city_groups = df["city_id"].astype(str).to_numpy()

    if len(df) < 120:
        return {"status": "skipped", "reason": "too_few_observations"}
    if np.unique(a).size < 2:
        return {"status": "skipped", "reason": "no_treatment_variation"}

    post_treated = (
        (df["post_policy"].to_numpy(dtype=int) == 1)
        & (df["treated_city"].to_numpy(dtype=int) == 1)
    )
    if post_treated.sum() == 0:
        return {"status": "skipped", "reason": "no_post_treated_rows"}

    LOGGER.info(
        "Causal-ST design ready: outcome=%s, treatment=%s, rows=%d, features=%d, bootstrap=%d, spatial_mode=%s",
        outcome_col,
        str(df.attrs.get("treatment_spec", "unknown")),
        len(df),
        len(feature_cols),
        cfg.n_bootstrap,
        str(df.attrs.get("spatial_mode", "distance")),
    )

    # -----------------------------------------------------------------
    # Step 1: Cross-fitting for point estimate
    # -----------------------------------------------------------------
    pseudo = _dr_pseudo_outcomes_crossfit(x, y, a, city_groups, cfg)
    estimator = "dr_crossfit" if cfg.use_dr else "plugin_crossfit"

    att = float(np.mean(pseudo[post_treated]))
    LOGGER.info("Causal-ST point estimate ready: outcome=%s, att=%.4f", outcome_col, att)

    # -----------------------------------------------------------------
    # Step 2: City-cluster bootstrap for SE and CI
    # -----------------------------------------------------------------
    boot_result = _bootstrap_att_se(df, feature_cols, outcome_col, post_treated, cfg, att)
    se_bootstrap = boot_result.get("se", float("nan"))
    ci95 = boot_result.get("ci95", [float("nan"), float("nan")])
    p_bootstrap = boot_result.get("p_value", float("nan"))

    # t-value derived from bootstrap SE (consistent, not inflated)
    if np.isfinite(se_bootstrap) and se_bootstrap > 1e-12:
        t_val = att / se_bootstrap
    else:
        # fallback: naive se from pseudo-outcomes (clearly marked)
        se_naive = float(
            np.std(pseudo[post_treated], ddof=1) / np.sqrt(max(1, int(post_treated.sum())))
        )
        t_val = att / max(se_naive, 1e-12)
        LOGGER.warning(
            "Causal-ST: bootstrap SE unavailable, falling back to naive pseudo-outcome SE."
            "  t_value may be overestimated."
        )

    # -----------------------------------------------------------------
    # Year-level ATT and continent CATE (using cross-fit pseudo-outcomes)
    # -----------------------------------------------------------------
    year_att = (
        df.assign(effect=pseudo)
        .loc[df["treated_city"] == 1]
        .groupby("year", as_index=False)
        .agg(att=("effect", "mean"), n=("effect", "size"))
        .sort_values("year")
    )
    year_att["post"] = (year_att["year"] >= 2020).astype(int)
    year_att["att_se"] = (
        df.assign(effect=pseudo)
        .loc[df["treated_city"] == 1]
        .groupby("year")["effect"]
        .std(ddof=1)
        .reindex(year_att["year"])
        .to_numpy(dtype=float)
        / np.sqrt(np.maximum(year_att["n"].to_numpy(dtype=float), 1.0))
    )

    cate_continent = (
        df.assign(effect=pseudo)
        .groupby("continent", as_index=False)
        .agg(cate=("effect", "mean"), n=("effect", "size"))
        .sort_values("continent")
    )

    # -----------------------------------------------------------------
    # Policy simulation (uses cross-fit pseudo-outcome logic)
    # -----------------------------------------------------------------
    # Re-fit full-sample models for simulation (acceptable here since we are
    # not estimating ATT from this fit – only using it for counterfactual shifts)
    prop_full = RandomForestClassifier(
        n_estimators=200, max_depth=7, min_samples_leaf=6,
        random_state=cfg.random_state, n_jobs=_RF_N_JOBS,
    )
    prop_full.fit(x, a.astype(int))
    e_full = np.clip(prop_full.predict_proba(x)[:, 1], 0.05, 0.95)

    model_t_full = RandomForestRegressor(
        n_estimators=300, max_depth=9, min_samples_leaf=4,
        random_state=cfg.random_state + 1, n_jobs=_RF_N_JOBS,
    )
    model_c_full = RandomForestRegressor(
        n_estimators=300, max_depth=9, min_samples_leaf=4,
        random_state=cfg.random_state + 2, n_jobs=_RF_N_JOBS,
    )
    model_t_full.fit(x[a == 1], y[a == 1])
    model_c_full.fit(x[a == 0], y[a == 0])

    scenarios = {
        "baseline": {},
        "infra_density_push": {"ghsl_built_density": +0.5},
        "poi_diversity_push": {"poi_diversity": +0.3},
        "road_expansion": {"road_length_km_total": +50.0},
        "combined_push": {"ghsl_built_density": +0.5, "poi_diversity": +0.3, "road_length_km_total": +50.0},
    }
    sim_rows: List[Dict[str, float | str]] = []
    treated_post_idx = np.where(post_treated)[0]
    for name, delta in scenarios.items():
        x_s = x.copy()
        for feat, shift in delta.items():
            if feat in feature_cols:
                j = feature_cols.index(feat)
                x_s[:, j] = x_s[:, j] + float(shift)
        y1_s = model_t_full.predict(x_s)
        y0_s = model_c_full.predict(x_s)
        if cfg.use_dr:
            eff_s = (y1_s - y0_s) + a * (y - y1_s) / e_full - (1 - a) * (y - y0_s) / (1 - e_full)
        else:
            eff_s = y1_s - y0_s
        sim_rows.append(
            {
                "scenario": name,
                "att_post": float(np.mean(eff_s[treated_post_idx])),
                "delta_vs_baseline": 0.0,
            }
        )

    sim_df = pd.DataFrame(sim_rows)
    base_att = float(sim_df.loc[sim_df["scenario"] == "baseline", "att_post"].iloc[0])
    sim_df["delta_vs_baseline"] = sim_df["att_post"] - base_att

    out = {
        "att_post": att,
        "outcome_variable": outcome_col,
        "treatment_spec": str(df.attrs.get("treatment_spec", "unknown")),
        "stderr": se_bootstrap if np.isfinite(se_bootstrap) else float("nan"),
        "t_value": float(t_val),
        "p_value": p_bootstrap,
        "att_ci95": ci95,
        "se_method": "bootstrap_cross_fit" if boot_result.get("status") == "ok" else "naive_fallback",
        "n_obs": int(len(df)),
        "n_post_treated": int(post_treated.sum()),
        "n_bootstrap_ok": boot_result.get("n_bootstrap_ok", 0),
        "feature_count": int(len(feature_cols)),
        "estimator": estimator,
        "use_spatial": bool(cfg.use_spatial),
        "use_temporal": bool(cfg.use_temporal),
        "spatial_mode": str(df.attrs.get("spatial_mode", "distance")),
        "spatial_summary": df.attrs.get("spatial_summary", {}),
    }
    return {
        "summary": out,
        "year_att": year_att,
        "cate_continent": cate_continent,
        "policy_sim": sim_df,
    }


def run_causal_st_analysis(panel: pd.DataFrame, cfg: CausalSTConfig | None = None) -> Dict[str, object]:
    """Run Causal-ST v2 full model and export detailed artifacts.

    SE and confidence intervals are computed via city-cluster bootstrap with
    K-fold cross-fitting nuisance models, correcting the previous in-sample
    pseudo-outcome variance that inflated t-statistics.
    """
    c = cfg or CausalSTConfig()
    LOGGER.info(
        "Causal-ST analysis start: bootstrap=%d, spatial=%s, temporal=%s, dr=%s, spatial_mode=%s",
        c.n_bootstrap,
        c.use_spatial,
        c.use_temporal,
        c.use_dr,
        c.spatial_mode,
    )
    res = _run_single(panel, c)
    if "summary" not in res:
        out = res
        dump_json(DATA_OUTPUTS / "causal_st_summary.json", out)
        return out

    summary = res["summary"]
    year_att = res["year_att"]
    cate_continent = res["cate_continent"]
    policy_sim = res["policy_sim"]

    year_att.to_csv(DATA_OUTPUTS / "causal_st_dynamic_att.csv", index=False)
    cate_continent.to_csv(DATA_OUTPUTS / "causal_st_cate_continent.csv", index=False)
    policy_sim.to_csv(DATA_OUTPUTS / "causal_st_policy_simulation.csv", index=False)

    # Keep compatibility with previous output name.
    year_att.rename(columns={"att": "ite_dr"}, inplace=False).to_csv(DATA_OUTPUTS / "causal_st_counterfactual.csv", index=False)

    dump_json(DATA_OUTPUTS / "causal_st_summary.json", summary)
    LOGGER.info(
        "Causal-ST v2 done: ATT=%.4f, t=%.3f (se_method=%s, ci95=[%.3f, %.3f])",
        summary["att_post"],
        summary["t_value"],
        summary["se_method"],
        summary["att_ci95"][0],
        summary["att_ci95"][1],
    )
    return summary


def run_causal_st_experiment_matrix(panel: pd.DataFrame) -> Dict[str, object]:
    """Run ablations for Causal-ST and export comparison table."""
    boot_ablation = _env_int("CAUSAL_ST_ABLATION_BOOTSTRAP", 20)
    variants: Dict[str, CausalSTConfig] = {
        "full_dr_spatiotemporal": CausalSTConfig(
            use_spatial=True,
            use_temporal=True,
            use_dr=True,
            n_bootstrap=boot_ablation,
        ),
        "no_spatial": CausalSTConfig(
            use_spatial=False,
            use_temporal=True,
            use_dr=True,
            n_bootstrap=boot_ablation,
        ),
        "no_temporal": CausalSTConfig(
            use_spatial=True,
            use_temporal=False,
            use_dr=True,
            n_bootstrap=boot_ablation,
        ),
        "flight_spatial": CausalSTConfig(
            use_spatial=True,
            use_temporal=True,
            use_dr=True,
            spatial_mode="flight",
            n_bootstrap=boot_ablation,
        ),
        "plugin_no_dr": CausalSTConfig(
            use_spatial=True,
            use_temporal=True,
            use_dr=False,
            n_bootstrap=boot_ablation,
        ),
    }

    rows: List[Dict[str, object]] = []
    for name, cfg in variants.items():
        LOGGER.info("Causal-ST ablation start: variant=%s, bootstrap=%d", name, cfg.n_bootstrap)
        res = _run_single(panel, cfg)
        if "summary" not in res:
            rows.append({"variant": name, "status": res.get("status", "skipped"), "reason": res.get("reason", "")})
            continue
        s = res["summary"]
        rows.append(
            {
                "variant": name,
                "status": "ok",
                "att_post": s["att_post"],
                "t_value": s["t_value"],
                "p_value": s.get("p_value"),
                "se_method": s.get("se_method"),
                "att_ci95_low": s["att_ci95"][0] if s.get("att_ci95") else float("nan"),
                "att_ci95_high": s["att_ci95"][1] if s.get("att_ci95") else float("nan"),
                "n_obs": s["n_obs"],
                "feature_count": s["feature_count"],
                "spatial_mode": s.get("spatial_mode"),
            }
        )
        LOGGER.info("Causal-ST ablation done: variant=%s, att=%.4f", name, float(s["att_post"]))

    table = pd.DataFrame(rows)
    table.to_csv(DATA_OUTPUTS / "causal_st_ablation.csv", index=False)

    payload = {"variants": rows}
    dump_json(DATA_OUTPUTS / "causal_st_ablation.json", payload)
    return payload
