from __future__ import annotations

"""Dynamic pulse index nowcasting with uncertainty and directional risk diagnostics."""

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .utils import DATA_OUTPUTS, dump_json


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def _harmonic_design(t: np.ndarray, period: float) -> np.ndarray:
    x = np.asarray(t, dtype=float)
    w = (2.0 * np.pi) / max(float(period), 1e-6)
    return np.column_stack(
        [
            np.ones_like(x, dtype=float),
            x,
            np.sin(w * x),
            np.cos(w * x),
            np.sin(2.0 * w * x),
            np.cos(2.0 * w * x),
        ]
    )


def _ridge_predict(train_t: np.ndarray, train_y: np.ndarray, pred_t: float, period: float, alpha: float = 1.2) -> float:
    X = _harmonic_design(train_t, period=period)
    y = np.asarray(train_y, dtype=float)
    if X.shape[0] < X.shape[1]:
        # Fallback to least squares when sample is small.
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    else:
        xtx = X.T @ X
        reg = np.eye(X.shape[1], dtype=float)
        reg[0, 0] = 0.0
        xtx = xtx + float(alpha) * reg
        xty = X.T @ y
        try:
            beta = np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    x_new = _harmonic_design(np.asarray([pred_t], dtype=float), period=period)
    return float((x_new @ beta)[0])


def _rolling_harmonic_blend(
    t: np.ndarray,
    y: np.ndarray,
    period: float,
    min_train: int,
) -> Dict[str, Any]:
    n = len(y)
    preds = np.full(n, np.nan, dtype=float)
    trend_arr = np.full(n, np.nan, dtype=float)

    diffs = np.diff(y)
    trend = float(np.median(diffs)) if diffs.size > 0 else 0.0
    if not np.isfinite(trend):
        trend = 0.0
    trend_arr[0] = trend

    t_rel = np.asarray(t - t[0], dtype=float)
    for i in range(1, n):
        naive = float(y[i - 1] + trend)
        if i >= min_train:
            harm = _ridge_predict(
                train_t=t_rel[:i],
                train_y=y[:i],
                pred_t=float(t_rel[i]),
                period=period,
                alpha=1.2,
            )
            pred = 0.68 * harm + 0.32 * naive
        else:
            pred = naive
        preds[i] = pred

        delta = float(y[i] - y[i - 1])
        trend = 0.76 * trend + 0.24 * delta
        if not np.isfinite(trend):
            trend = delta if np.isfinite(delta) else 0.0
        trend_arr[i] = trend

    eval_idx = np.array([i for i in range(min_train, n) if np.isfinite(preds[i])], dtype=int)
    if eval_idx.size == 0:
        return {
            "preds": preds,
            "trend_arr": trend_arr,
            "trend_last": float(trend_arr[-1]) if np.isfinite(trend_arr[-1]) else 0.0,
            "directional_accuracy": np.nan,
            "rmse": np.nan,
            "eval_idx": eval_idx,
        }

    true_delta = y[eval_idx] - y[eval_idx - 1]
    pred_delta = preds[eval_idx] - y[eval_idx - 1]
    directional = float(np.mean(np.sign(true_delta) == np.sign(pred_delta)))
    rmse = float(np.sqrt(np.mean(np.square(y[eval_idx] - preds[eval_idx]))))
    return {
        "preds": preds,
        "trend_arr": trend_arr,
        "trend_last": float(trend_arr[-1]) if np.isfinite(trend_arr[-1]) else 0.0,
        "directional_accuracy": directional,
        "rmse": rmse,
        "eval_idx": eval_idx,
    }


def _series_nowcast(
    years: np.ndarray,
    values: np.ndarray,
    accel_band: float = 2.0,
) -> Dict[str, Any]:
    y = np.asarray(values, dtype=float)
    t = np.asarray(years, dtype=int)
    mask = np.isfinite(y)
    y = y[mask]
    t = t[mask]
    if y.size < 4:
        return {
            "status": "insufficient_history",
            "history": pd.DataFrame(
                columns=[
                    "year",
                    "index_mean",
                    "forecast_mean",
                    "forecast_ci_low_95",
                    "forecast_ci_high_95",
                    "trend_estimate",
                    "forecast_error",
                    "is_forecast",
                ]
            ),
            "latest": {},
            "metrics": {},
        }

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    diffs = np.diff(y)
    min_train = max(5, int(len(y) // 2))
    periods = np.arange(2.0, 6.76, 0.25)
    y_scale = float(np.std(y, ddof=0))
    if (not np.isfinite(y_scale)) or y_scale <= 1e-6:
        y_scale = 1.0

    best: Dict[str, Any] | None = None
    for p in periods:
        cur = _rolling_harmonic_blend(t=t, y=y, period=float(p), min_train=min_train)
        dir_acc = float(cur.get("directional_accuracy", np.nan))
        rmse = float(cur.get("rmse", np.nan))
        rmse_skill = _clip01(1.0 - (rmse / max(y_scale, 1e-6))) if np.isfinite(rmse) else 0.0
        score = 0.78 * _clip01(dir_acc) + 0.22 * rmse_skill
        cur["period"] = float(p)
        cur["selection_score"] = float(score)
        if (best is None) or (float(cur["selection_score"]) > float(best["selection_score"])):
            best = cur

    if best is None:
        best = _rolling_harmonic_blend(t=t, y=y, period=3.0, min_train=min_train)
        best["period"] = 3.0
        best["selection_score"] = 0.0

    best_p = float(best.get("period", 3.0))
    preds = np.asarray(best.get("preds"), dtype=float)
    trend_arr = np.asarray(best.get("trend_arr"), dtype=float)
    trend_last = float(best.get("trend_last", 0.0))
    eval_idx = np.asarray(best.get("eval_idx"), dtype=int)

    rows: List[Dict[str, Any]] = []
    residuals: List[float] = []
    pred_vals: List[float] = []
    true_vals: List[float] = []
    pred_deltas: List[float] = []
    true_deltas: List[float] = []
    for i in range(len(y)):
        pred = float(preds[i]) if np.isfinite(preds[i]) else np.nan
        err = float(y[i] - pred) if np.isfinite(pred) else np.nan
        if np.isfinite(err):
            residuals.append(err)
            pred_vals.append(pred)
            true_vals.append(float(y[i]))
            if i >= 1:
                pred_deltas.append(float(pred - y[i - 1]))
                true_deltas.append(float(y[i] - y[i - 1]))
        rows.append(
            {
                "year": int(t[i]),
                "index_mean": float(y[i]),
                "forecast_mean": pred,
                "forecast_ci_low_95": np.nan,
                "forecast_ci_high_95": np.nan,
                "trend_estimate": float(trend_arr[i]) if np.isfinite(trend_arr[i]) else np.nan,
                "forecast_error": err,
                "is_forecast": 0,
            }
        )

    sigma_resid = float(np.std(np.asarray(residuals, dtype=float), ddof=0)) if residuals else float("nan")
    if (not np.isfinite(sigma_resid)) or sigma_resid <= 1e-6:
        sigma_resid = float(np.std(diffs, ddof=0)) if diffs.size > 1 else 1.0
    sigma_resid = float(max(0.8, sigma_resid))

    for r in rows:
        if np.isfinite(r["forecast_mean"]):
            r["forecast_ci_low_95"] = float(r["forecast_mean"] - 1.96 * sigma_resid)
            r["forecast_ci_high_95"] = float(r["forecast_mean"] + 1.96 * sigma_resid)

    last_year = int(t[-1])
    last_val = float(y[-1])
    t_rel = t - t[0]
    next_rel = float(t_rel[-1] + 1.0)
    harm_next = _ridge_predict(
        train_t=t_rel.astype(float),
        train_y=y.astype(float),
        pred_t=next_rel,
        period=best_p,
        alpha=1.2,
    )
    naive_next = float(last_val + trend_last)
    nowcast = float(0.68 * harm_next + 0.32 * naive_next)
    ci80 = (float(nowcast - 1.28155 * sigma_resid), float(nowcast + 1.28155 * sigma_resid))
    ci95 = (float(nowcast - 1.96 * sigma_resid), float(nowcast + 1.96 * sigma_resid))

    sigma_delta = float(np.std(np.asarray(true_deltas, dtype=float), ddof=0)) if len(true_deltas) > 1 else sigma_resid
    sigma_delta = float(max(0.5, sigma_delta))
    z_dec = float((-accel_band - trend_last) / sigma_delta)
    z_acc = float((accel_band - trend_last) / sigma_delta)
    p_dec = _clip01(_normal_cdf(z_dec))
    p_acc = _clip01(1.0 - _normal_cdf(z_acc))
    p_stable = _clip01(1.0 - p_dec - p_acc)

    rows.append(
        {
            "year": int(last_year + 1),
            "index_mean": np.nan,
            "forecast_mean": nowcast,
            "forecast_ci_low_95": ci95[0],
            "forecast_ci_high_95": ci95[1],
            "trend_estimate": trend_last,
            "forecast_error": np.nan,
            "is_forecast": 1,
        }
    )

    yt = np.asarray(true_vals, dtype=float)
    yp = np.asarray(pred_vals, dtype=float)
    mae = float(np.mean(np.abs(yt - yp))) if yt.size > 0 else np.nan
    rmse = float(np.sqrt(np.mean(np.square(yt - yp)))) if yt.size > 0 else np.nan
    if yt.size > 0 and np.std(yt) > 1e-9:
        r2 = float(1.0 - np.mean(np.square(yt - yp)) / np.var(yt))
    else:
        r2 = np.nan
    coverage95 = (
        float(
            np.mean(
                (yt >= (yp - 1.96 * sigma_resid))
                & (yt <= (yp + 1.96 * sigma_resid))
            )
        )
        if yt.size > 0
        else np.nan
    )
    sign_acc = (
        float(
            np.mean(
                np.sign(np.asarray(true_deltas, dtype=float))
                == np.sign(np.asarray(pred_deltas, dtype=float))
            )
        )
        if pred_deltas
        else np.nan
    )

    return {
        "status": "ok",
        "history": pd.DataFrame(rows),
        "latest": {
            "last_year": int(last_year),
            "last_index": last_val,
            "nowcast_year": int(last_year + 1),
            "nowcast_index_mean": nowcast,
            "nowcast_ci_low_80": ci80[0],
            "nowcast_ci_high_80": ci80[1],
            "nowcast_ci_low_95": ci95[0],
            "nowcast_ci_high_95": ci95[1],
            "trend_estimate": float(trend_last),
            "selected_period": best_p,
            "selection_score": float(best.get("selection_score", np.nan)),
            "sigma_residual": sigma_resid,
            "sigma_delta": sigma_delta,
            "prob_decelerate": p_dec,
            "prob_accelerate": p_acc,
            "prob_stable": p_stable,
            "signal": (
                "decelerate_risk"
                if p_dec >= max(0.45, p_acc + 0.10)
                else "accelerate_window"
                if p_acc >= max(0.45, p_dec + 0.10)
                else "mixed"
            ),
        },
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "coverage95": coverage95,
            "directional_accuracy": sign_acc,
            "n_backtest": int(yt.size),
            "selected_period": best_p,
        },
    }


def run_pulse_nowcast_suite() -> Dict[str, Any]:
    """Build continent/global pulse nowcast tables and summary."""
    src = DATA_OUTPUTS / "pulse_ai_dynamic_index_continent_year.csv"
    if not src.exists():
        out = {"status": "skipped", "reason": "pulse_ai_dynamic_index_continent_year_missing"}
        dump_json(DATA_OUTPUTS / "pulse_nowcast_summary.json", out)
        return out

    df = pd.read_csv(src)
    if df.empty:
        out = {"status": "skipped", "reason": "pulse_ai_dynamic_index_continent_year_empty"}
        dump_json(DATA_OUTPUTS / "pulse_nowcast_summary.json", out)
        return out

    need = {"continent", "year", "dynamic_pulse_index_mean", "city_count"}
    if not need.issubset(set(df.columns)):
        out = {"status": "skipped", "reason": "pulse_ai_dynamic_index_continent_year_missing_columns"}
        dump_json(DATA_OUTPUTS / "pulse_nowcast_summary.json", out)
        return out

    work = df.copy()
    work["continent"] = work["continent"].astype(str)
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["dynamic_pulse_index_mean"] = pd.to_numeric(work["dynamic_pulse_index_mean"], errors="coerce")
    work["city_count"] = pd.to_numeric(work["city_count"], errors="coerce")
    work = work.dropna(subset=["continent", "year", "dynamic_pulse_index_mean"]).copy()
    if work.empty:
        out = {"status": "skipped", "reason": "no_valid_rows_after_coerce"}
        dump_json(DATA_OUTPUTS / "pulse_nowcast_summary.json", out)
        return out

    history_rows: List[pd.DataFrame] = []
    latest_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    valid_continents = 0

    for cont, grp in work.groupby("continent", as_index=False):
        sub = grp.sort_values("year").copy()
        res = _series_nowcast(
            years=sub["year"].to_numpy(dtype=int),
            values=sub["dynamic_pulse_index_mean"].to_numpy(dtype=float),
            accel_band=2.0,
        )
        if res.get("status") != "ok":
            continue
        valid_continents += 1

        hist = res["history"].copy()
        hist["continent"] = str(cont)
        history_rows.append(hist)

        latest = dict(res["latest"])
        latest["continent"] = str(cont)
        latest["city_count_last"] = int(pd.to_numeric(sub["city_count"], errors="coerce").fillna(0).iloc[-1])
        latest_rows.append(latest)

        metrics = dict(res["metrics"])
        metrics["continent"] = str(cont)
        metric_rows.append(metrics)

    history_df = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    latest_df = pd.DataFrame(latest_rows)
    metrics_df = pd.DataFrame(metric_rows)

    years = sorted(work["year"].dropna().astype(int).unique().tolist())
    global_rows: List[Dict[str, Any]] = []
    for year in years:
        sub = work[work["year"].astype(int) == int(year)].copy()
        weights = pd.to_numeric(sub["city_count"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        vals = pd.to_numeric(sub["dynamic_pulse_index_mean"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(weights) & np.isfinite(vals) & (weights > 0)
        if not np.any(mask):
            continue
        global_rows.append(
            {
                "year": int(year),
                "dynamic_pulse_index_mean": float(np.average(vals[mask], weights=weights[mask])),
                "city_count": float(np.sum(weights[mask])),
            }
        )
    global_df = pd.DataFrame(global_rows).sort_values("year")
    global_res = _series_nowcast(
        years=global_df["year"].to_numpy(dtype=int),
        values=global_df["dynamic_pulse_index_mean"].to_numpy(dtype=float),
        accel_band=1.5,
    ) if not global_df.empty else {"status": "insufficient_history", "history": pd.DataFrame(), "latest": {}, "metrics": {}}

    global_hist = global_res["history"].copy() if isinstance(global_res.get("history"), pd.DataFrame) else pd.DataFrame()
    if not global_hist.empty:
        global_hist["scope"] = "global"
        global_hist = global_hist.rename(columns={"index_mean": "dynamic_pulse_index_mean"})
        global_hist["city_count"] = np.nan
        global_hist = global_hist[
            [
                "scope",
                "year",
                "dynamic_pulse_index_mean",
                "forecast_mean",
                "forecast_ci_low_95",
                "forecast_ci_high_95",
                "trend_estimate",
                "forecast_error",
                "is_forecast",
                "city_count",
            ]
        ]
    else:
        global_hist = pd.DataFrame(
            columns=[
                "scope",
                "year",
                "dynamic_pulse_index_mean",
                "forecast_mean",
                "forecast_ci_low_95",
                "forecast_ci_high_95",
                "trend_estimate",
                "forecast_error",
                "is_forecast",
                "city_count",
            ]
        )

    if not history_df.empty:
        history_df = history_df.rename(columns={"index_mean": "dynamic_pulse_index_mean"})
        history_df["scope"] = "continent"
        history_df["city_count"] = np.nan
        history_df = history_df[
            [
                "scope",
                "continent",
                "year",
                "dynamic_pulse_index_mean",
                "forecast_mean",
                "forecast_ci_low_95",
                "forecast_ci_high_95",
                "trend_estimate",
                "forecast_error",
                "is_forecast",
                "city_count",
            ]
        ]
    else:
        history_df = pd.DataFrame(
            columns=[
                "scope",
                "continent",
                "year",
                "dynamic_pulse_index_mean",
                "forecast_mean",
                "forecast_ci_low_95",
                "forecast_ci_high_95",
                "trend_estimate",
                "forecast_error",
                "is_forecast",
                "city_count",
            ]
        )

    global_hist["continent"] = "Global"
    all_hist = pd.concat([history_df, global_hist], ignore_index=True)
    all_hist = all_hist.sort_values(["scope", "continent", "year"]).reset_index(drop=True)

    latest_path = DATA_OUTPUTS / "pulse_nowcast_continent_latest.csv"
    history_path = DATA_OUTPUTS / "pulse_nowcast_continent_history.csv"
    metric_path = DATA_OUTPUTS / "pulse_nowcast_backtest_metrics.csv"
    global_path = DATA_OUTPUTS / "pulse_nowcast_global.csv"

    latest_df.to_csv(latest_path, index=False)
    all_hist.to_csv(history_path, index=False)
    metrics_df.to_csv(metric_path, index=False)
    global_hist.to_csv(global_path, index=False)

    global_metrics = global_res.get("metrics", {}) if isinstance(global_res, dict) else {}
    global_latest = global_res.get("latest", {}) if isinstance(global_res, dict) else {}

    mean_prob_dec = float(pd.to_numeric(latest_df.get("prob_decelerate"), errors="coerce").mean()) if not latest_df.empty else np.nan
    mean_prob_acc = float(pd.to_numeric(latest_df.get("prob_accelerate"), errors="coerce").mean()) if not latest_df.empty else np.nan
    max_dec_row = latest_df.sort_values("prob_decelerate", ascending=False).head(1) if not latest_df.empty else pd.DataFrame()
    max_acc_row = latest_df.sort_values("prob_accelerate", ascending=False).head(1) if not latest_df.empty else pd.DataFrame()

    summary = {
        "status": "ok",
        "continent_rows": int(len(latest_df)),
        "history_rows": int(len(all_hist)),
        "metrics_rows": int(len(metrics_df)),
        "continents_covered": int(valid_continents),
        "global_nowcast_year": int(global_latest.get("nowcast_year")) if global_latest else None,
        "global_nowcast_index_mean": float(global_latest.get("nowcast_index_mean", np.nan)) if global_latest else np.nan,
        "global_nowcast_ci_low_95": float(global_latest.get("nowcast_ci_low_95", np.nan)) if global_latest else np.nan,
        "global_nowcast_ci_high_95": float(global_latest.get("nowcast_ci_high_95", np.nan)) if global_latest else np.nan,
        "global_prob_decelerate": float(global_latest.get("prob_decelerate", np.nan)) if global_latest else np.nan,
        "global_prob_accelerate": float(global_latest.get("prob_accelerate", np.nan)) if global_latest else np.nan,
        "global_signal": str(global_latest.get("signal", "")) if global_latest else "",
        "global_backtest_mae": float(global_metrics.get("mae", np.nan)) if global_metrics else np.nan,
        "global_backtest_rmse": float(global_metrics.get("rmse", np.nan)) if global_metrics else np.nan,
        "global_backtest_r2": float(global_metrics.get("r2", np.nan)) if global_metrics else np.nan,
        "global_backtest_coverage95": float(global_metrics.get("coverage95", np.nan)) if global_metrics else np.nan,
        "global_backtest_directional_accuracy": float(global_metrics.get("directional_accuracy", np.nan))
        if global_metrics
        else np.nan,
        "mean_continent_prob_decelerate": mean_prob_dec,
        "mean_continent_prob_accelerate": mean_prob_acc,
        "top_decelerate_continent": (
            str(max_dec_row.iloc[0]["continent"]) if not max_dec_row.empty else None
        ),
        "top_accelerate_continent": (
            str(max_acc_row.iloc[0]["continent"]) if not max_acc_row.empty else None
        ),
        "artifacts": {
            "latest_csv": str(latest_path),
            "history_csv": str(history_path),
            "metrics_csv": str(metric_path),
            "global_csv": str(global_path),
        },
    }

    dump_json(DATA_OUTPUTS / "pulse_nowcast_summary.json", summary)
    return summary
