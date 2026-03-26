from __future__ import annotations

"""Theory-anchored spatial and urban feature builders."""

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .utils import DATA_PROCESSED, dump_json, haversine_km


def _safe_series(values: pd.Series | Iterable[float], index: pd.Index | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.Series(values, index=index, dtype=float)


def _global_minmax(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    mask = vals.notna()
    out = pd.Series(np.nan, index=vals.index, dtype=float)
    if not mask.any():
        return out
    lo = float(vals[mask].min())
    hi = float(vals[mask].max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return out
    if np.isclose(lo, hi):
        out.loc[mask] = 0.5
        return out
    out.loc[mask] = (vals.loc[mask] - lo) / (hi - lo)
    return out.astype(float)


def entropy_weighted_score(
    df: pd.DataFrame,
    columns: List[str],
    *,
    prefix: str,
) -> tuple[pd.Series, Dict[str, float]]:
    """Entropy Weight Method over already normalized columns."""
    valid_cols = [col for col in columns if col in df.columns]
    if not valid_cols:
        return pd.Series(np.nan, index=df.index, dtype=float), {}

    block = df[valid_cols].apply(pd.to_numeric, errors="coerce")
    if block.notna().sum().sum() == 0:
        return pd.Series(np.nan, index=df.index, dtype=float), {col: 0.0 for col in valid_cols}

    work = block.copy()
    for col in valid_cols:
        med = float(pd.to_numeric(work[col], errors="coerce").median())
        if not np.isfinite(med):
            med = 0.5
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(med).clip(0.0, 1.0)

    n = max(int(len(work)), 1)
    eps = 1e-12
    col_sums = work.sum(axis=0).replace(0.0, np.nan)
    prob = work.div(col_sums, axis=1).fillna(1.0 / float(n))
    entropy = -(prob * np.log(prob.clip(lower=eps))).sum(axis=0) / np.log(float(max(n, 2)))
    divergence = (1.0 - entropy).clip(lower=0.0)
    if float(divergence.sum()) <= eps:
        weights = pd.Series(
            np.full(len(valid_cols), 1.0 / max(len(valid_cols), 1), dtype=float),
            index=valid_cols,
            dtype=float,
        )
    else:
        weights = (divergence / divergence.sum()).astype(float)

    score = work.to_numpy(dtype=float) @ weights.to_numpy(dtype=float)
    score_s = pd.Series(score, index=df.index, dtype=float)
    weights_dict = {f"{prefix}:{col}": float(weights[col]) for col in valid_cols}
    return score_s, weights_dict


def fit_cobb_douglas_vitality(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, float | int | str]]:
    """Fit a Cobb-Douglas style vitality equation and return fitted/residual series."""
    out = df.copy()
    out["cobb_douglas_log_A_proxy"] = np.log1p(pd.to_numeric(out.get("knowledge_capital_raw"), errors="coerce").clip(lower=0.0))
    out["cobb_douglas_log_K"] = np.log1p(pd.to_numeric(out.get("ghsl_built_volume_m3"), errors="coerce").clip(lower=0.0))
    if "baseline_population_log" in out.columns:
        out["cobb_douglas_log_L"] = pd.to_numeric(out.get("baseline_population_log"), errors="coerce")
    else:
        out["cobb_douglas_log_L"] = np.log1p(pd.to_numeric(out.get("population"), errors="coerce").clip(lower=0.0))
    y = pd.to_numeric(out.get("log_viirs_ntl"), errors="coerce")
    x = pd.DataFrame(
        {
            "A": out["cobb_douglas_log_A_proxy"],
            "K": out["cobb_douglas_log_K"],
            "L": out["cobb_douglas_log_L"],
        }
    )
    valid = y.notna() & x.notna().all(axis=1)
    if int(valid.sum()) < 50:
        out["cobb_douglas_log_vitality_fit"] = np.nan
        out["cobb_douglas_tfp_residual"] = np.nan
        out["economic_vitality_cobb_douglas"] = np.nan
        return out, {
            "status": "insufficient_rows",
            "valid_rows": int(valid.sum()),
        }

    x_fit = np.column_stack(
        [
            np.ones(int(valid.sum()), dtype=float),
            x.loc[valid, "A"].to_numpy(dtype=float),
            x.loc[valid, "K"].to_numpy(dtype=float),
            x.loc[valid, "L"].to_numpy(dtype=float),
        ]
    )
    y_fit = y.loc[valid].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x_fit, y_fit, rcond=None)
    fitted = np.full(len(out), np.nan, dtype=float)
    fitted[valid.to_numpy(dtype=bool)] = x_fit @ beta
    resid = y.to_numpy(dtype=float) - fitted
    out["cobb_douglas_log_vitality_fit"] = pd.Series(fitted, index=out.index, dtype=float)
    out["cobb_douglas_tfp_residual"] = pd.Series(resid, index=out.index, dtype=float)
    out["economic_vitality_cobb_douglas"] = 100.0 * _global_minmax(out["cobb_douglas_log_vitality_fit"])

    y_hat = out.loc[valid, "cobb_douglas_log_vitality_fit"].to_numpy(dtype=float)
    ssr = float(np.nansum((y_fit - y_hat) ** 2))
    sst = float(np.nansum((y_fit - np.nanmean(y_fit)) ** 2))
    r2 = 1.0 - (ssr / sst if sst > 0 else 0.0)
    return out, {
        "status": "ok",
        "valid_rows": int(valid.sum()),
        "intercept": float(beta[0]),
        "elasticity_A": float(beta[1]),
        "elasticity_K": float(beta[2]),
        "elasticity_L": float(beta[3]),
        "r2": float(r2),
    }


def build_static_weight_matrices(
    city_frame: pd.DataFrame,
    *,
    baseline_signal: pd.Series | None = None,
    economic_gap_floor: float = 0.25,
) -> tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Build row-standardized distance and economic-similarity weight matrices."""
    city = city_frame[["city_id", "latitude", "longitude"]].drop_duplicates("city_id").copy()
    city["city_id"] = city["city_id"].astype(str)
    city["latitude"] = pd.to_numeric(city["latitude"], errors="coerce")
    city["longitude"] = pd.to_numeric(city["longitude"], errors="coerce")
    city = city.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    ids = city["city_id"].tolist()
    n = len(ids)
    if n == 0:
        return [], np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))

    dist_km = np.zeros((n, n), dtype=float)
    inv_dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = haversine_km(
                float(city.loc[i, "latitude"]),
                float(city.loc[i, "longitude"]),
                float(city.loc[j, "latitude"]),
                float(city.loc[j, "longitude"]),
            )
            dist_km[i, j] = d
            inv_dist[i, j] = 1.0 / max(d, 1.0)

    if baseline_signal is None:
        signal = np.arange(n, dtype=float)
    else:
        sig = _safe_series(baseline_signal)
        sig_map = {str(k): float(v) for k, v in sig.dropna().items()}
        signal = np.asarray([sig_map.get(cid, np.nan) for cid in ids], dtype=float)
        finite = np.isfinite(signal)
        fill = float(np.nanmedian(signal[finite])) if finite.any() else 0.0
        signal = np.where(np.isfinite(signal), signal, fill)

    econ_raw = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            gap = abs(float(signal[i]) - float(signal[j]))
            econ_raw[i, j] = 1.0 / max(gap, float(economic_gap_floor))

    def _row_standardize(mat: np.ndarray) -> np.ndarray:
        row_sum = mat.sum(axis=1, keepdims=True)
        out = np.zeros_like(mat, dtype=float)
        valid = row_sum.squeeze() > 0
        out[valid] = mat[valid] / row_sum[valid]
        return out

    return ids, _row_standardize(inv_dist), _row_standardize(econ_raw), dist_km


def export_spatial_weight_artifacts(
    panel: pd.DataFrame,
    *,
    baseline_year: int = 2015,
) -> Dict[str, object]:
    """Export static distance/economic weight matrices and edge summaries."""
    city = panel[["city_id", "latitude", "longitude"]].drop_duplicates("city_id").copy()
    baseline = (
        panel.loc[pd.to_numeric(panel.get("year"), errors="coerce") == int(baseline_year), ["city_id", "log_viirs_ntl"]]
        .drop_duplicates("city_id")
        .set_index("city_id")["log_viirs_ntl"]
    )
    ids, w_dist, w_econ, dist_km = build_static_weight_matrices(city, baseline_signal=baseline)
    if not ids:
        summary = {"status": "missing_city_geometry_or_baseline_signal", "city_count": 0}
        dump_json(DATA_PROCESSED / "spatial_weight_summary.json", summary)
        return summary

    idx = pd.Index(ids, name="city_id")
    pd.DataFrame(w_dist, index=idx, columns=ids).to_csv(DATA_PROCESSED / "spatial_weight_matrix_distance.csv")
    pd.DataFrame(w_econ, index=idx, columns=ids).to_csv(DATA_PROCESSED / "spatial_weight_matrix_economic.csv")
    pd.DataFrame(dist_km, index=idx, columns=ids).to_csv(DATA_PROCESSED / "city_pairwise_distance_km.csv")

    edges: List[dict[str, object]] = []
    for i, src in enumerate(ids):
        if w_dist.shape[1] == 0:
            continue
        top_idx = np.argsort(w_dist[i])[::-1][:8]
        for j in top_idx:
            weight = float(w_dist[i, j])
            if i == j or weight <= 0.0:
                continue
            edges.append(
                {
                    "city_id": src,
                    "neighbor_city_id": ids[j],
                    "distance_km": float(dist_km[i, j]),
                    "w_dist": weight,
                    "w_econ": float(w_econ[i, j]),
                }
            )
    pd.DataFrame(edges).to_csv(DATA_PROCESSED / "spatial_weight_top_edges.csv", index=False)
    summary = {
        "status": "ok",
        "city_count": int(len(ids)),
        "baseline_year": int(baseline_year),
        "files": {
            "w_dist": str(DATA_PROCESSED / "spatial_weight_matrix_distance.csv"),
            "w_econ": str(DATA_PROCESSED / "spatial_weight_matrix_economic.csv"),
            "distance_km": str(DATA_PROCESSED / "city_pairwise_distance_km.csv"),
            "top_edges": str(DATA_PROCESSED / "spatial_weight_top_edges.csv"),
        },
    }
    dump_json(DATA_PROCESSED / "spatial_weight_summary.json", summary)
    return summary


def add_spatial_structure_features(panel: pd.DataFrame, *, baseline_year: int = 2015) -> tuple[pd.DataFrame, Dict[str, object]]:
    """Add gravity accessibility and spatial-lag features using static city matrices."""
    out = panel.copy().sort_values(["city_id", "year"]).reset_index(drop=True)
    city = out[["city_id", "latitude", "longitude"]].drop_duplicates("city_id").copy()
    baseline = (
        out.loc[pd.to_numeric(out.get("year"), errors="coerce") == int(baseline_year), ["city_id", "log_viirs_ntl"]]
        .drop_duplicates("city_id")
        .set_index("city_id")["log_viirs_ntl"]
    )
    ids, w_dist, w_econ, dist_km = build_static_weight_matrices(city, baseline_signal=baseline)
    if not ids:
        return out, {"status": "missing_city_coordinates", "city_count": 0}

    raw_inv_dist = np.zeros_like(dist_km, dtype=float)
    mask = dist_km > 0.0
    raw_inv_dist[mask] = 1.0 / np.maximum(dist_km[mask], 1.0)
    city_order = {cid: idx for idx, cid in enumerate(ids)}

    out["gravity_access_viirs"] = np.nan
    out["gravity_access_knowledge"] = np.nan
    out["gravity_access_population"] = np.nan
    out["spatial_lag_log_viirs_ntl_wdist"] = np.nan
    out["spatial_lag_log_viirs_ntl_wecon"] = np.nan
    out["spatial_lag_knowledge_wdist"] = np.nan
    out["spatial_degree_nonzero"] = float(max(len(ids) - 1, 0))

    for year, sub in out.groupby("year", sort=True):
        idx_rows = sub.index.to_numpy(dtype=int)
        pos = np.asarray([city_order.get(str(cid), -1) for cid in sub["city_id"].astype(str)], dtype=int)
        valid_rows = pos >= 0
        if not valid_rows.any():
            continue

        viirs_vec = np.full(len(ids), np.nan, dtype=float)
        know_vec = np.full(len(ids), np.nan, dtype=float)
        pop_vec = np.full(len(ids), np.nan, dtype=float)

        year_city = sub.copy()
        year_city["city_id"] = year_city["city_id"].astype(str)
        viirs_map = pd.to_numeric(year_city.get("log_viirs_ntl"), errors="coerce")
        know_map = np.log1p(pd.to_numeric(year_city.get("knowledge_capital_raw"), errors="coerce").clip(lower=0.0))
        pop_map = pd.to_numeric(year_city.get("baseline_population_log"), errors="coerce")
        city_to_viirs = dict(zip(year_city["city_id"], viirs_map))
        city_to_know = dict(zip(year_city["city_id"], know_map))
        city_to_pop = dict(zip(year_city["city_id"], pop_map))

        for cid, pos_idx in city_order.items():
            viirs_vec[pos_idx] = city_to_viirs.get(cid, np.nan)
            know_vec[pos_idx] = city_to_know.get(cid, np.nan)
            pop_vec[pos_idx] = city_to_pop.get(cid, np.nan)

        def _fill(vec: np.ndarray) -> np.ndarray:
            finite = np.isfinite(vec)
            fill = float(np.nanmedian(vec[finite])) if finite.any() else 0.0
            return np.where(np.isfinite(vec), vec, fill)

        viirs_f = _fill(viirs_vec)
        know_f = _fill(know_vec)
        pop_f = _fill(pop_vec)

        gravity_viirs = raw_inv_dist @ viirs_f
        gravity_know = raw_inv_dist @ know_f
        gravity_pop = raw_inv_dist @ pop_f
        lag_viirs_dist = w_dist @ viirs_f
        lag_viirs_econ = w_econ @ viirs_f
        lag_know_dist = w_dist @ know_f

        out.loc[idx_rows[valid_rows], "gravity_access_viirs"] = gravity_viirs[pos[valid_rows]]
        out.loc[idx_rows[valid_rows], "gravity_access_knowledge"] = gravity_know[pos[valid_rows]]
        out.loc[idx_rows[valid_rows], "gravity_access_population"] = gravity_pop[pos[valid_rows]]
        out.loc[idx_rows[valid_rows], "spatial_lag_log_viirs_ntl_wdist"] = lag_viirs_dist[pos[valid_rows]]
        out.loc[idx_rows[valid_rows], "spatial_lag_log_viirs_ntl_wecon"] = lag_viirs_econ[pos[valid_rows]]
        out.loc[idx_rows[valid_rows], "spatial_lag_knowledge_wdist"] = lag_know_dist[pos[valid_rows]]

    summary = export_spatial_weight_artifacts(out, baseline_year=int(baseline_year))
    summary["status"] = "ok"
    summary["feature_columns"] = [
        "gravity_access_viirs",
        "gravity_access_knowledge",
        "gravity_access_population",
        "spatial_lag_log_viirs_ntl_wdist",
        "spatial_lag_log_viirs_ntl_wecon",
        "spatial_lag_knowledge_wdist",
    ]
    return out, summary

