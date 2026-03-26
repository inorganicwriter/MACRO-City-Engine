from __future__ import annotations

"""Dynamic method core benchmark: AI ensemble vs strong baselines with significance tests."""

from dataclasses import dataclass
import os
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import DATA_OUTPUTS, dump_json


MAIN_MODEL_NAME = "pulse_ai_dynamic_ensemble"
BASELINE_MODELS = ("ridge_ar2", "ridge_full", "naive_last")
TERTIARY_BAND = 1.0
DECELERATION_BAND = -2.0


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        val = int(str(raw).strip())
    except Exception:  # noqa: BLE001
        return int(default)
    return int(max(1, val))


PAIR_BOOT_DRAWS = _env_int("DYNAMIC_METHOD_BOOTSTRAP_DRAWS", 2000)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    model_kind: str  # "ensemble" | "ridge"
    use_context: bool
    use_velocity: bool


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _label_delta(delta: np.ndarray, band: float = TERTIARY_BAND) -> np.ndarray:
    d = np.asarray(delta, dtype=float)
    out = np.zeros(len(d), dtype=int)
    out[d > float(band)] = 1
    out[d < -float(band)] = -1
    return out


def _weighted_avg(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def _prepare_supervised_panel() -> pd.DataFrame:
    path = DATA_OUTPUTS / "pulse_ai_dynamic_index_series.csv"
    if not path.exists():
        return pd.DataFrame()

    raw = pd.read_csv(path)
    if raw.empty:
        return pd.DataFrame()

    for col in [
        "year",
        "dynamic_pulse_index",
        "dynamic_pulse_delta_1y",
        "dynamic_pulse_trend_3y",
        "pulse_accel_velocity",
        "pulse_risk_velocity",
    ]:
        raw[col] = _safe_series(raw, col)

    work = raw.dropna(subset=["city_id", "continent", "year", "dynamic_pulse_index"]).copy()
    if work.empty:
        return pd.DataFrame()

    work["city_id"] = work["city_id"].astype(str)
    work["city_name"] = work.get("city_name", "").astype(str)
    work["country"] = work.get("country", "").astype(str)
    work["continent"] = work["continent"].astype(str)
    work["year"] = work["year"].astype(int)

    cont = (
        work.groupby(["continent", "year"], as_index=False)["dynamic_pulse_index"]
        .mean()
        .rename(columns={"dynamic_pulse_index": "cont_mean"})
        .sort_values(["continent", "year"])
    )
    cont["cont_delta"] = cont.groupby("continent")["cont_mean"].diff()
    cont["cont_trend_3y"] = cont.groupby("continent")["cont_mean"].diff().rolling(3, min_periods=1).mean().to_numpy(dtype=float)

    glob = (
        work.groupby("year", as_index=False)["dynamic_pulse_index"]
        .mean()
        .rename(columns={"dynamic_pulse_index": "glob_mean"})
        .sort_values("year")
    )
    glob["glob_delta"] = glob["glob_mean"].diff()
    glob["glob_trend_3y"] = glob["glob_mean"].diff().rolling(3, min_periods=1).mean().to_numpy(dtype=float)

    work = work.merge(cont, on=["continent", "year"], how="left")
    work = work.merge(glob, on="year", how="left")
    work["rel_cont"] = work["dynamic_pulse_index"] - work["cont_mean"]
    work["rel_glob"] = work["dynamic_pulse_index"] - work["glob_mean"]

    rows: List[Dict[str, Any]] = []
    for city_id, grp in work.sort_values(["city_id", "year"]).groupby("city_id", as_index=False):
        g = grp.sort_values("year").reset_index(drop=True)
        y = pd.to_numeric(g["dynamic_pulse_index"], errors="coerce").to_numpy(dtype=float)
        years = pd.to_numeric(g["year"], errors="coerce").to_numpy(dtype=int)
        dy = np.diff(y, prepend=np.nan)
        if len(g) < 6:
            continue

        for i in range(len(g) - 1):
            cur = g.iloc[i]
            prev = dy[max(0, i - 2) : i + 1]
            sign_prev = np.sign(prev[np.isfinite(prev)])
            rows.append(
                {
                    "city_id": str(city_id),
                    "city_name": str(cur.get("city_name", "")),
                    "country": str(cur.get("country", "")),
                    "continent": str(cur.get("continent", "")),
                    "origin_year": int(years[i]),
                    "target_year": int(years[i + 1]),
                    "obs_id": f"{city_id}:{int(years[i + 1])}",
                    "y_t": float(y[i]),
                    "target_y": float(y[i + 1]),
                    "delta_t": float(cur.get("dynamic_pulse_delta_1y", np.nan)),
                    "trend3_t": float(cur.get("dynamic_pulse_trend_3y", np.nan)),
                    "accel_v": float(cur.get("pulse_accel_velocity", np.nan)),
                    "risk_v": float(cur.get("pulse_risk_velocity", np.nan)),
                    "city_vol3": float(np.nanstd(prev, ddof=0)) if prev.size > 0 else np.nan,
                    "city_mom3": float(np.nanmean(sign_prev)) if sign_prev.size > 0 else 0.0,
                    "cont_mean": float(cur.get("cont_mean", np.nan)),
                    "cont_delta": float(cur.get("cont_delta", np.nan)),
                    "cont_trend_3y": float(cur.get("cont_trend_3y", np.nan)),
                    "glob_mean": float(cur.get("glob_mean", np.nan)),
                    "glob_delta": float(cur.get("glob_delta", np.nan)),
                    "glob_trend_3y": float(cur.get("glob_trend_3y", np.nan)),
                    "rel_cont": float(cur.get("rel_cont", np.nan)),
                    "rel_glob": float(cur.get("rel_glob", np.nan)),
                }
            )

    sup = pd.DataFrame(rows)
    if sup.empty:
        return sup

    y0 = int(sup["origin_year"].min())
    y1 = int(sup["origin_year"].max())
    if y1 > y0:
        sup["year_norm"] = (sup["origin_year"].astype(float) - float(y0)) / float(y1 - y0)
    else:
        sup["year_norm"] = 0.0

    # Build interaction features
    sup["y_t_x_delta"] = pd.to_numeric(sup["y_t"], errors="coerce").fillna(0.0) * pd.to_numeric(sup["delta_t"], errors="coerce").fillna(0.0)
    sup["y_t_x_vol"] = pd.to_numeric(sup["y_t"], errors="coerce").fillna(0.0) * pd.to_numeric(sup["city_vol3"], errors="coerce").fillna(0.0)
    sup["delta_x_mom"] = pd.to_numeric(sup["delta_t"], errors="coerce").fillna(0.0) * pd.to_numeric(sup["city_mom3"], errors="coerce").fillna(0.0)

    return sup.sort_values(["origin_year", "continent", "city_id"]).reset_index(drop=True)


def _feature_columns(use_context: bool, use_velocity: bool) -> List[str]:
    cols = [
        "y_t",
        "delta_t",
        "trend3_t",
        "city_vol3",
        "city_mom3",
        "rel_cont",
        "rel_glob",
        "year_norm",
    ]
    if use_velocity:
        cols.extend(["accel_v", "risk_v"])
    if use_context:
        cols.extend(
            [
                "cont_mean",
                "cont_delta",
                "cont_trend_3y",
                "glob_mean",
                "glob_delta",
                "glob_trend_3y",
            ]
        )
    # Interaction features for richer signal
    cols.extend(["y_t_x_delta", "y_t_x_vol", "delta_x_mom"])
    return cols


def _fit_ridge(X: pd.DataFrame, y: np.ndarray, alpha: float = 3.0) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    model.fit(X, y)
    return model


def _fit_hgb(X: pd.DataFrame, y: np.ndarray) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.03,
        max_iter=500,
        min_samples_leaf=12,
        l2_regularization=0.10,
        random_state=42,
    )
    model.fit(X, y)
    return model


def _fit_hgb_delta(X: pd.DataFrame, delta: np.ndarray) -> HistGradientBoostingRegressor:
    """Main dynamic model: predict next-year change directly instead of level residuals."""
    model = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.03,
        max_iter=500,
        min_samples_leaf=15,
        l2_regularization=0.15,
        random_state=42,
    )
    model.fit(X, delta)
    return model


def _fit_direction_classifier(X: pd.DataFrame, delta: np.ndarray) -> HistGradientBoostingClassifier:
    lab = np.ones(len(delta), dtype=int)
    d = np.asarray(delta, dtype=float)
    lab[d > TERTIARY_BAND] = 2
    lab[d < -TERTIARY_BAND] = 0
    model = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=220,
        min_samples_leaf=20,
        l2_regularization=0.20,
        random_state=42,
    )
    model.fit(X, lab)
    return model


def _ternary_accuracy_from_preds(y_true: np.ndarray, y_pred: np.ndarray, y_t: np.ndarray) -> float:
    delta_true = np.asarray(y_true, dtype=float) - np.asarray(y_t, dtype=float)
    delta_pred = np.asarray(y_pred, dtype=float) - np.asarray(y_t, dtype=float)
    return float(np.mean(_label_delta(delta_true, band=TERTIARY_BAND) == _label_delta(delta_pred, band=TERTIARY_BAND)))


def _extract_direction_probs(
    clf: HistGradientBoostingClassifier,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (p_decelerate, p_stable, p_accelerate) with missing classes filled by zeros."""
    prob = np.asarray(clf.predict_proba(X), dtype=float)
    cls = np.asarray(getattr(clf, "classes_", []), dtype=int)
    n = int(prob.shape[0]) if prob.ndim == 2 else 0
    p_dec = np.zeros(n, dtype=float)
    p_sta = np.zeros(n, dtype=float)
    p_acc = np.zeros(n, dtype=float)
    if prob.ndim != 2 or n == 0:
        return p_dec, p_sta, p_acc

    for j, c in enumerate(cls.tolist()):
        if j >= prob.shape[1]:
            continue
        if int(c) == 0:
            p_dec = prob[:, j]
        elif int(c) == 1:
            p_sta = prob[:, j]
        elif int(c) == 2:
            p_acc = prob[:, j]
    return p_dec, p_sta, p_acc


def _apply_directional_correction(
    pred_y: np.ndarray,
    y_t: np.ndarray,
    p_decelerate: np.ndarray,
    p_stable: np.ndarray,
    p_accelerate: np.ndarray,
) -> np.ndarray:
    d = np.asarray(pred_y, dtype=float) - np.asarray(y_t, dtype=float)
    p_dec = np.asarray(p_decelerate, dtype=float)
    p_sta = np.asarray(p_stable, dtype=float)
    p_acc = np.asarray(p_accelerate, dtype=float)
    out = d.copy()

    # Raise ternary direction quality without sacrificing MAE too much.
    strong = 0.52
    gain = 2.0
    mask_dec = (p_dec >= strong) & (out > -TERTIARY_BAND)
    out[mask_dec] = -TERTIARY_BAND - gain * (p_dec[mask_dec] - strong)
    mask_acc = (p_acc >= strong) & (out < TERTIARY_BAND)
    out[mask_acc] = TERTIARY_BAND + gain * (p_acc[mask_acc] - strong)
    mask_sta = p_sta >= 0.58
    out[mask_sta] = 0.60 * out[mask_sta]
    return np.asarray(y_t, dtype=float) + out


def _tune_residual_blend(
    dev: pd.DataFrame,
    calib: pd.DataFrame,
    feat: List[str],
    ar_cols: List[str],
) -> Dict[str, Any]:
    if dev.empty or calib.empty:
        return {"lambda_ridge": 0.0, "lambda_hgb": 0.0, "use_direction_correction": False}

    med_full = dev[feat].median(numeric_only=True).fillna(0.0)
    med_ar = dev[ar_cols].median(numeric_only=True).fillna(0.0)
    Xdev = dev[feat].copy().fillna(med_full)
    Xcal = calib[feat].copy().fillna(med_full)
    Xdev_ar = dev[ar_cols].copy().fillna(med_ar)
    Xcal_ar = calib[ar_cols].copy().fillna(med_ar)

    ydev = pd.to_numeric(dev["target_y"], errors="coerce").to_numpy(dtype=float)
    ycal = pd.to_numeric(calib["target_y"], errors="coerce").to_numpy(dtype=float)
    ydev_t = pd.to_numeric(dev["y_t"], errors="coerce").to_numpy(dtype=float)
    ycal_t = pd.to_numeric(calib["y_t"], errors="coerce").to_numpy(dtype=float)

    ridge_ar = _fit_ridge(Xdev_ar, ydev, alpha=1.0)
    pred_ar_dev = ridge_ar.predict(Xdev_ar)
    pred_ar_cal = ridge_ar.predict(Xcal_ar)

    resid_dev = ydev - pred_ar_dev
    resid_ridge = _fit_ridge(Xdev, resid_dev, alpha=4.0)
    resid_hgb = _fit_hgb(Xdev, resid_dev)
    pred_resid_ridge_cal = resid_ridge.predict(Xcal)
    pred_resid_hgb_cal = resid_hgb.predict(Xcal)

    best: Dict[str, Any] = {
        "lambda_ridge": 0.0,
        "lambda_hgb": 0.0,
        "mae": float(np.mean(np.abs(ycal - pred_ar_cal))),
        "ternary": _ternary_accuracy_from_preds(ycal, pred_ar_cal, ycal_t),
    }
    grid = [0.0, 0.10, 0.20, 0.35, 0.50, 0.75]
    for lam_r in grid:
        for lam_h in grid:
            pred = pred_ar_cal + lam_r * pred_resid_ridge_cal + lam_h * pred_resid_hgb_cal
            mae = float(np.mean(np.abs(ycal - pred)))
            ternary = _ternary_accuracy_from_preds(ycal, pred, ycal_t)
            if (mae < best["mae"] - 1e-9) or (abs(mae - best["mae"]) <= 0.03 and ternary > best["ternary"]):
                best = {
                    "lambda_ridge": float(lam_r),
                    "lambda_hgb": float(lam_h),
                    "mae": mae,
                    "ternary": ternary,
                }

    use_direction_correction = False
    try:
        delta_dev = ydev - ydev_t
        clf = _fit_direction_classifier(Xdev, delta_dev)
        p_dec_cal, p_sta_cal, p_acc_cal = _extract_direction_probs(clf, Xcal)
        raw = pred_ar_cal + best["lambda_ridge"] * pred_resid_ridge_cal + best["lambda_hgb"] * pred_resid_hgb_cal
        corr = _apply_directional_correction(raw, ycal_t, p_dec_cal, p_sta_cal, p_acc_cal)
        raw_mae = float(np.mean(np.abs(ycal - raw)))
        corr_mae = float(np.mean(np.abs(ycal - corr)))
        raw_ternary = _ternary_accuracy_from_preds(ycal, raw, ycal_t)
        corr_ternary = _ternary_accuracy_from_preds(ycal, corr, ycal_t)
        use_direction_correction = bool((corr_ternary >= raw_ternary) and (corr_mae <= raw_mae + 0.10))
    except Exception:  # noqa: BLE001
        use_direction_correction = False

    best["use_direction_correction"] = use_direction_correction
    return best


def _rolling_predictions(
    sup: pd.DataFrame,
    spec: AblationSpec,
    with_baselines: bool = False,
) -> pd.DataFrame:
    if sup.empty:
        return pd.DataFrame()

    feat = _feature_columns(use_context=spec.use_context, use_velocity=spec.use_velocity)
    years = sorted(pd.to_numeric(sup["origin_year"], errors="coerce").dropna().astype(int).unique().tolist())
    min_train_years = 4
    eval_years = years[min_train_years:]
    if not eval_years:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for yr in eval_years:
        train = sup[sup["origin_year"] < int(yr)].copy()
        test = sup[sup["origin_year"] == int(yr)].copy()
        if train.empty or test.empty:
            continue

        med = train[feat].median(numeric_only=True).fillna(0.0)
        Xtr = train[feat].copy().fillna(med)
        Xte = test[feat].copy().fillna(med)
        ytr = pd.to_numeric(train["target_y"], errors="coerce").to_numpy(dtype=float)
        yte = pd.to_numeric(test["target_y"], errors="coerce").to_numpy(dtype=float)

        ridge_full = _fit_ridge(Xtr, ytr, alpha=3.0)
        pred_ridge_full_train = ridge_full.predict(Xtr)
        pred_ridge_full = ridge_full.predict(Xte)

        # Strong baseline 1: AR(2)-style ridge on level+delta.
        ar_cols = ["y_t", "delta_t"]
        med_ar = train[ar_cols].median(numeric_only=True).fillna(0.0)
        Xtr_ar = train[ar_cols].copy().fillna(med_ar)
        Xte_ar = test[ar_cols].copy().fillna(med_ar)
        ridge_ar = _fit_ridge(Xtr_ar, ytr, alpha=1.0)
        pred_ar_train = ridge_ar.predict(Xtr_ar)
        pred_ar = ridge_ar.predict(Xte_ar)

        if spec.model_kind == "ensemble":
            delta_train = ytr - pd.to_numeric(train["y_t"], errors="coerce").to_numpy(dtype=float)
            hgb_delta = _fit_hgb_delta(Xtr, delta_train)
            pred_main_train = pd.to_numeric(train["y_t"], errors="coerce").to_numpy(dtype=float) + hgb_delta.predict(Xtr)
            pred_main = pd.to_numeric(test["y_t"], errors="coerce").to_numpy(dtype=float) + hgb_delta.predict(Xte)
        else:
            delta_train = ytr - pd.to_numeric(train["y_t"], errors="coerce").to_numpy(dtype=float)
            ridge_delta = _fit_ridge(Xtr, delta_train, alpha=4.0)
            pred_main_train = pd.to_numeric(train["y_t"], errors="coerce").to_numpy(dtype=float) + ridge_delta.predict(Xtr)
            pred_main = pd.to_numeric(test["y_t"], errors="coerce").to_numpy(dtype=float) + ridge_delta.predict(Xte)

        abs_train = np.abs(ytr - pred_main_train)
        q95 = float(np.nanquantile(abs_train, 0.95)) if abs_train.size > 3 else float(np.nanstd(abs_train, ddof=0) + 2.0)
        q80 = float(np.nanquantile(abs_train, 0.80)) if abs_train.size > 3 else float(np.nanstd(abs_train, ddof=0) + 1.0)
        q95 = float(max(1.0, q95))
        q80 = float(max(0.6, q80))

        # Baseline 2: random-walk naive.
        pred_naive = pd.to_numeric(test["y_t"], errors="coerce").to_numpy(dtype=float)

        for i, row in enumerate(test.itertuples(index=False)):
            y_t = float(getattr(row, "y_t"))
            y_true = float(yte[i])
            pred_m = float(pred_main[i])
            record_common = {
                "obs_id": str(getattr(row, "obs_id")),
                "city_id": str(getattr(row, "city_id")),
                "city_name": str(getattr(row, "city_name")),
                "country": str(getattr(row, "country")),
                "continent": str(getattr(row, "continent")),
                "origin_year": int(getattr(row, "origin_year")),
                "target_year": int(getattr(row, "target_year")),
                "y_t": y_t,
                "target_y": y_true,
            }
            rows.append(
                {
                    **record_common,
                    "model": spec.name,
                    "prediction": pred_m,
                    "prediction_ci_low_80": pred_m - q80,
                    "prediction_ci_high_80": pred_m + q80,
                    "prediction_ci_low_95": pred_m - q95,
                    "prediction_ci_high_95": pred_m + q95,
                }
            )

            if with_baselines:
                rows.append(
                    {
                        **record_common,
                        "model": "ridge_full",
                        "prediction": float(pred_ridge_full[i]),
                        "prediction_ci_low_80": np.nan,
                        "prediction_ci_high_80": np.nan,
                        "prediction_ci_low_95": np.nan,
                        "prediction_ci_high_95": np.nan,
                    }
                )
                rows.append(
                    {
                        **record_common,
                        "model": "ridge_ar2",
                        "prediction": float(pred_ar[i]),
                        "prediction_ci_low_80": np.nan,
                        "prediction_ci_high_80": np.nan,
                        "prediction_ci_low_95": np.nan,
                        "prediction_ci_high_95": np.nan,
                    }
                )
                rows.append(
                    {
                        **record_common,
                        "model": "naive_last",
                        "prediction": float(pred_naive[i]),
                        "prediction_ci_low_80": np.nan,
                        "prediction_ci_high_80": np.nan,
                        "prediction_ci_low_95": np.nan,
                        "prediction_ci_high_95": np.nan,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["error"] = pd.to_numeric(out["target_y"], errors="coerce") - pd.to_numeric(out["prediction"], errors="coerce")
    out["abs_error"] = out["error"].abs()
    out["sq_error"] = np.square(out["error"])
    out["delta_true"] = pd.to_numeric(out["target_y"], errors="coerce") - pd.to_numeric(out["y_t"], errors="coerce")
    out["delta_pred"] = pd.to_numeric(out["prediction"], errors="coerce") - pd.to_numeric(out["y_t"], errors="coerce")
    out["ternary_true"] = _label_delta(out["delta_true"].to_numpy(dtype=float), band=TERTIARY_BAND)
    out["ternary_pred"] = _label_delta(out["delta_pred"].to_numpy(dtype=float), band=TERTIARY_BAND)
    out["ternary_hit"] = (out["ternary_true"] == out["ternary_pred"]).astype(int)
    out["direction_hit"] = (np.sign(out["delta_true"]) == np.sign(out["delta_pred"])).astype(int)
    out["decel_event"] = (out["delta_true"] <= DECELERATION_BAND).astype(int)
    out["decel_score"] = -pd.to_numeric(out["delta_pred"], errors="coerce").fillna(0.0)
    in95 = (
        (pd.to_numeric(out["target_y"], errors="coerce") >= pd.to_numeric(out["prediction_ci_low_95"], errors="coerce"))
        & (pd.to_numeric(out["target_y"], errors="coerce") <= pd.to_numeric(out["prediction_ci_high_95"], errors="coerce"))
    )
    out["covered_95"] = in95.astype(int)
    return out


def _metric_frame(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame(
            columns=[
                "scope",
                "continent",
                "model",
                "n_obs",
                "mae",
                "rmse",
                "r2",
                "directional_accuracy",
                "ternary_accuracy",
                "deceleration_event_share",
                "deceleration_score_mean",
                "coverage95",
            ]
        )

    def summarize(frame: pd.DataFrame, scope: str, continent: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for model, g in frame.groupby("model", as_index=False):
            y = pd.to_numeric(g["target_y"], errors="coerce").to_numpy(dtype=float)
            p = pd.to_numeric(g["prediction"], errors="coerce").to_numpy(dtype=float)
            err = y - p
            mae = float(np.nanmean(np.abs(err))) if len(g) > 0 else np.nan
            rmse = float(np.sqrt(np.nanmean(np.square(err)))) if len(g) > 0 else np.nan
            if len(g) > 2 and np.nanvar(y) > 1e-12:
                r2 = float(1.0 - (np.nanmean(np.square(err)) / np.nanvar(y)))
            else:
                r2 = np.nan
            rows.append(
                {
                    "scope": scope,
                    "continent": continent,
                    "model": str(model),
                    "n_obs": int(len(g)),
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "directional_accuracy": float(pd.to_numeric(g["direction_hit"], errors="coerce").mean()),
                    "ternary_accuracy": float(pd.to_numeric(g["ternary_hit"], errors="coerce").mean()),
                    "deceleration_event_share": float(pd.to_numeric(g["decel_event"], errors="coerce").mean()),
                    "deceleration_score_mean": float(pd.to_numeric(g["decel_score"], errors="coerce").mean()),
                    "coverage95": float(pd.to_numeric(g["covered_95"], errors="coerce").mean()) if model == MAIN_MODEL_NAME else np.nan,
                }
            )
        return rows

    out_rows: List[Dict[str, Any]] = []
    out_rows.extend(summarize(pred, scope="global", continent="Global"))
    for cont, g in pred.groupby("continent", as_index=False):
        out_rows.extend(summarize(g, scope="continent", continent=str(cont)))
    out = pd.DataFrame(out_rows)

    # Skill vs random walk for easier interpretation when R2 is negative.
    rw = out[(out["scope"] == "global") & (out["model"] == "naive_last")]
    rw_mae = float(rw.iloc[0]["mae"]) if not rw.empty else np.nan
    out["mae_skill_vs_naive"] = np.where(
        np.isfinite(rw_mae) & (rw_mae > 1e-12),
        1.0 - (pd.to_numeric(out["mae"], errors="coerce") / rw_mae),
        np.nan,
    )
    return out


def _bootstrap_paired_delta(
    delta: np.ndarray,
    draws: int = 2000,
    random_state: int = 42,
) -> Dict[str, float]:
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {
            "mean_delta": np.nan,
            "ci_low_95": np.nan,
            "ci_high_95": np.nan,
            "p_value_improve": np.nan,
        }

    rng = np.random.default_rng(int(random_state))
    idx = rng.integers(0, d.size, size=(int(draws), d.size))
    boot = d[idx].mean(axis=1)
    mean_delta = float(np.mean(d))
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975]).tolist()
    # one-sided p-value for "improvement > 0"
    p_improve = float(np.mean(boot <= 0.0))
    return {
        "mean_delta": mean_delta,
        "ci_low_95": float(ci_low),
        "ci_high_95": float(ci_high),
        "p_value_improve": p_improve,
    }


def _paired_significance(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame(
            columns=[
                "scope",
                "continent",
                "baseline_model",
                "n_pairs",
                "abs_error_improve_mean",
                "abs_error_improve_ci_low_95",
                "abs_error_improve_ci_high_95",
                "abs_error_p_value_improve",
                "direction_improve_mean",
                "direction_improve_ci_low_95",
                "direction_improve_ci_high_95",
                "direction_p_value_improve",
                "both_significant_5pct",
            ]
        )

    base = pred[["obs_id", "continent", "model", "abs_error", "direction_hit", "ternary_hit"]].copy()
    scopes: List[tuple[str, str, pd.DataFrame]] = [("global", "Global", base)]
    for cont, g in base.groupby("continent", as_index=False):
        scopes.append(("continent", str(cont), g.copy()))

    rows: List[Dict[str, Any]] = []
    for scope, continent, frame in scopes:
        wide_abs = frame.pivot_table(index="obs_id", columns="model", values="abs_error", aggfunc="first")
        wide_dir = frame.pivot_table(index="obs_id", columns="model", values="ternary_hit", aggfunc="first")
        if MAIN_MODEL_NAME not in wide_abs.columns or MAIN_MODEL_NAME not in wide_dir.columns:
            continue
        for b in BASELINE_MODELS:
            if b not in wide_abs.columns or b not in wide_dir.columns:
                continue
            pair_abs = wide_abs[[MAIN_MODEL_NAME, b]].dropna()
            pair_dir = wide_dir[[MAIN_MODEL_NAME, b]].dropna()
            if pair_abs.empty or pair_dir.empty:
                continue
            abs_delta = pair_abs[b].to_numpy(dtype=float) - pair_abs[MAIN_MODEL_NAME].to_numpy(dtype=float)
            dir_delta = pair_dir[MAIN_MODEL_NAME].to_numpy(dtype=float) - pair_dir[b].to_numpy(dtype=float)
            abs_boot = _bootstrap_paired_delta(abs_delta, draws=PAIR_BOOT_DRAWS, random_state=42)
            dir_boot = _bootstrap_paired_delta(dir_delta, draws=PAIR_BOOT_DRAWS, random_state=84)
            rows.append(
                {
                    "scope": scope,
                    "continent": continent,
                    "baseline_model": b,
                    "n_pairs": int(min(len(pair_abs), len(pair_dir))),
                    "abs_error_improve_mean": abs_boot["mean_delta"],
                    "abs_error_improve_ci_low_95": abs_boot["ci_low_95"],
                    "abs_error_improve_ci_high_95": abs_boot["ci_high_95"],
                    "abs_error_p_value_improve": abs_boot["p_value_improve"],
                    "direction_improve_mean": dir_boot["mean_delta"],
                    "direction_improve_ci_low_95": dir_boot["ci_low_95"],
                    "direction_improve_ci_high_95": dir_boot["ci_high_95"],
                    "direction_p_value_improve": dir_boot["p_value_improve"],
                    "both_significant_5pct": int(
                        (abs_boot["p_value_improve"] < 0.05) and (dir_boot["p_value_improve"] < 0.05)
                    ),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["scope", "continent", "abs_error_p_value_improve", "direction_p_value_improve"]).reset_index(drop=True)
    return out


def _ablation_table(sup: pd.DataFrame) -> pd.DataFrame:
    if sup.empty:
        return pd.DataFrame(
            columns=[
                "ablation",
                "model_kind",
                "use_context",
                "use_velocity",
                "n_obs",
                "mae",
                "rmse",
                "r2",
                "directional_accuracy",
                "ternary_accuracy",
                "coverage95",
                "mae_gain_vs_ablation",
                "dir_gain_vs_ablation",
            ]
        )

    specs = [
        AblationSpec(name=MAIN_MODEL_NAME, model_kind="ensemble", use_context=True, use_velocity=True),
        AblationSpec(name="ablation_no_context", model_kind="ensemble", use_context=False, use_velocity=True),
        AblationSpec(name="ablation_no_velocity", model_kind="ensemble", use_context=True, use_velocity=False),
        AblationSpec(name="ablation_ridge_only", model_kind="ridge", use_context=True, use_velocity=True),
    ]

    rows: List[Dict[str, Any]] = []
    for spec in specs:
        pred = _rolling_predictions(sup, spec=spec, with_baselines=False)
        if pred.empty:
            continue
        g = pred.copy()
        y = pd.to_numeric(g["target_y"], errors="coerce").to_numpy(dtype=float)
        p = pd.to_numeric(g["prediction"], errors="coerce").to_numpy(dtype=float)
        err = y - p
        mae = float(np.nanmean(np.abs(err)))
        rmse = float(np.sqrt(np.nanmean(np.square(err))))
        r2 = float(1.0 - (np.nanmean(np.square(err)) / np.nanvar(y))) if (len(g) > 2 and np.nanvar(y) > 1e-12) else np.nan
        rows.append(
            {
                "ablation": spec.name,
                "model_kind": spec.model_kind,
                "use_context": int(spec.use_context),
                "use_velocity": int(spec.use_velocity),
                "n_obs": int(len(g)),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "directional_accuracy": float(pd.to_numeric(g["direction_hit"], errors="coerce").mean()),
                "ternary_accuracy": float(pd.to_numeric(g["ternary_hit"], errors="coerce").mean()),
                "coverage95": float(pd.to_numeric(g["covered_95"], errors="coerce").mean()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    base = out[out["ablation"] == MAIN_MODEL_NAME]
    if base.empty:
        out["mae_gain_vs_ablation"] = np.nan
        out["dir_gain_vs_ablation"] = np.nan
        return out

    base_mae = float(base.iloc[0]["mae"])
    base_dir = float(base.iloc[0]["ternary_accuracy"])
    out["mae_gain_vs_ablation"] = pd.to_numeric(out["mae"], errors="coerce").rsub(base_mae)
    out["dir_gain_vs_ablation"] = base_dir - pd.to_numeric(out["ternary_accuracy"], errors="coerce")
    out = out.sort_values("mae", ascending=True).reset_index(drop=True)
    return out


def _collect_summary(metrics: pd.DataFrame, signif: pd.DataFrame, ablation: pd.DataFrame, pred: pd.DataFrame) -> Dict[str, Any]:
    def _metric(model: str, col: str, scope: str = "global", continent: str = "Global") -> float:
        if metrics.empty:
            return np.nan
        q = metrics[
            (metrics["scope"].astype(str) == str(scope))
            & (metrics["continent"].astype(str) == str(continent))
            & (metrics["model"].astype(str) == str(model))
        ].copy()
        if q.empty:
            return np.nan
        return float(pd.to_numeric(q.iloc[0].get(col), errors="coerce"))

    sig_global = signif[(signif["scope"] == "global") & (signif["continent"] == "Global")] if not signif.empty else pd.DataFrame()
    sig_ar = sig_global[sig_global["baseline_model"] == "ridge_ar2"] if not sig_global.empty else pd.DataFrame()
    sig_best = sig_global.sort_values("abs_error_p_value_improve").head(1) if not sig_global.empty else pd.DataFrame()

    continents = int(pred["continent"].astype(str).nunique()) if not pred.empty else 0
    cities = int(pred["city_id"].astype(str).nunique()) if not pred.empty else 0
    years_eval = int(pred["target_year"].astype(int).nunique()) if not pred.empty else 0

    main_mae = _metric(MAIN_MODEL_NAME, "mae")
    main_ternary = _metric(MAIN_MODEL_NAME, "ternary_accuracy")
    ar_mae = _metric("ridge_ar2", "mae")
    ar_ternary = _metric("ridge_ar2", "ternary_accuracy")
    naive_mae = _metric("naive_last", "mae")
    mae_gain_ar = ar_mae - main_mae if np.isfinite(ar_mae) and np.isfinite(main_mae) else np.nan
    dir_gain_ar = main_ternary - ar_ternary if np.isfinite(main_ternary) and np.isfinite(ar_ternary) else np.nan
    mae_skill_naive = 1.0 - (main_mae / naive_mae) if np.isfinite(main_mae) and np.isfinite(naive_mae) and naive_mae > 1e-12 else np.nan

    ar_abs_p = float(pd.to_numeric(sig_ar.iloc[0].get("abs_error_p_value_improve"), errors="coerce")) if not sig_ar.empty else np.nan
    ar_dir_p = float(pd.to_numeric(sig_ar.iloc[0].get("direction_p_value_improve"), errors="coerce")) if not sig_ar.empty else np.nan
    best_sig = sig_best.iloc[0].to_dict() if not sig_best.empty else {}

    ablation_gain_context = np.nan
    ablation_gain_velocity = np.nan
    if not ablation.empty:
        nc = ablation[ablation["ablation"] == "ablation_no_context"]
        nv = ablation[ablation["ablation"] == "ablation_no_velocity"]
        if not nc.empty and np.isfinite(main_mae):
            ablation_gain_context = float(pd.to_numeric(nc.iloc[0]["mae"], errors="coerce") - main_mae)
        if not nv.empty and np.isfinite(main_mae):
            ablation_gain_velocity = float(pd.to_numeric(nv.iloc[0]["mae"], errors="coerce") - main_mae)

    gate = {
        "continents_covered": continents,
        "city_coverage": cities,
        "eval_years": years_eval,
        "main_mae": main_mae,
        "ridge_ar2_mae": ar_mae,
        "main_ternary_accuracy": main_ternary,
        "ridge_ar2_ternary_accuracy": ar_ternary,
        "mae_gain_vs_ridge_ar2": mae_gain_ar,
        "ternary_gain_vs_ridge_ar2": dir_gain_ar,
        "mae_skill_vs_naive": mae_skill_naive,
        "p_value_mae_improve_vs_ridge_ar2": ar_abs_p,
        "p_value_ternary_improve_vs_ridge_ar2": ar_dir_p,
        "ablation_gain_no_context_mae": ablation_gain_context,
        "ablation_gain_no_velocity_mae": ablation_gain_velocity,
        "ready": bool(
            (continents >= 6)
            and (cities >= 200)
            and np.isfinite(mae_gain_ar)
            and np.isfinite(dir_gain_ar)
            and (mae_gain_ar >= 0.25)
            and (dir_gain_ar >= 0.01)
            and np.isfinite(ar_abs_p)
            and np.isfinite(ar_dir_p)
            and (ar_abs_p < 0.05)
            and (ar_dir_p < 0.05)
        ),
    }

    return {
        "status": "ok",
        "main_model": MAIN_MODEL_NAME,
        "continents_covered": continents,
        "cities_covered": cities,
        "eval_years": years_eval,
        "prediction_rows": int(len(pred)),
        "global_main_metrics": {
            "mae": main_mae,
            "rmse": _metric(MAIN_MODEL_NAME, "rmse"),
            "r2": _metric(MAIN_MODEL_NAME, "r2"),
            "directional_accuracy": _metric(MAIN_MODEL_NAME, "directional_accuracy"),
            "ternary_accuracy": _metric(MAIN_MODEL_NAME, "ternary_accuracy"),
            "coverage95": _metric(MAIN_MODEL_NAME, "coverage95"),
            "mae_skill_vs_naive": mae_skill_naive,
        },
        "global_vs_ridge_ar2": {
            "mae_gain": mae_gain_ar,
            "ternary_accuracy_gain": dir_gain_ar,
            "abs_error_p_value": ar_abs_p,
            "ternary_accuracy_p_value": ar_dir_p,
        },
        "best_significance_case_global": best_sig,
        "ablation_signal": {
            "mae_gain_over_no_context": ablation_gain_context,
            "mae_gain_over_no_velocity": ablation_gain_velocity,
        },
        "top_tier_gate": gate,
        "artifacts": {
            "predictions_csv": str(DATA_OUTPUTS / "dynamic_method_core_predictions.csv"),
            "metrics_csv": str(DATA_OUTPUTS / "dynamic_method_core_metrics.csv"),
            "significance_csv": str(DATA_OUTPUTS / "dynamic_method_core_significance.csv"),
            "ablation_csv": str(DATA_OUTPUTS / "dynamic_method_core_ablation.csv"),
        },
    }


def run_dynamic_method_core_suite() -> Dict[str, Any]:
    """Run dynamic method benchmark with rolling evaluation and paired significance tests."""
    sup = _prepare_supervised_panel()
    if sup.empty:
        out = {"status": "skipped", "reason": "pulse_ai_dynamic_index_series_missing_or_invalid"}
        dump_json(DATA_OUTPUTS / "dynamic_method_core_summary.json", out)
        return out

    main_spec = AblationSpec(name=MAIN_MODEL_NAME, model_kind="ensemble", use_context=True, use_velocity=True)
    pred = _rolling_predictions(sup, spec=main_spec, with_baselines=True)
    if pred.empty:
        out = {"status": "skipped", "reason": "no_predictions_generated"}
        dump_json(DATA_OUTPUTS / "dynamic_method_core_summary.json", out)
        return out

    metrics = _metric_frame(pred)
    signif = _paired_significance(pred)
    ablation = _ablation_table(sup)
    summary = _collect_summary(metrics, signif, ablation, pred)

    pred.to_csv(DATA_OUTPUTS / "dynamic_method_core_predictions.csv", index=False)
    metrics.to_csv(DATA_OUTPUTS / "dynamic_method_core_metrics.csv", index=False)
    signif.to_csv(DATA_OUTPUTS / "dynamic_method_core_significance.csv", index=False)
    ablation.to_csv(DATA_OUTPUTS / "dynamic_method_core_ablation.csv", index=False)
    dump_json(DATA_OUTPUTS / "dynamic_method_core_summary.json", summary)
    return summary
