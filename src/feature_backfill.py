from __future__ import annotations

"""Feature backfilling helpers used by predictive and benchmark modules."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .utils import DATA_OUTPUTS, dump_json

LOGGER = logging.getLogger(__name__)

NO2_DIRECT_TARGETS: List[str] = [
    "no2_trop_mean",
    "no2_trop_p90",
    "no2_trop_anomaly_mean",
    "no2_trop_anomaly_abs_mean",
    "no2_recent_spike",
]

NO2_FILLED_MAP: Dict[str, str] = {
    "no2_trop_mean": "no2_trop_mean_filled",
    "no2_trop_p90": "no2_trop_p90_filled",
    "no2_trop_yoy_mean": "no2_trop_yoy_mean_filled",
    "no2_trop_anomaly_mean": "no2_trop_anomaly_mean_filled",
    "no2_trop_anomaly_abs_mean": "no2_trop_anomaly_abs_mean_filled",
    "no2_recent_spike": "no2_recent_spike_filled",
}

NO2_PREDICTOR_CANDIDATES: List[str] = [
    "_year_index",
    "latitude",
    "longitude",
    "temperature_mean",
    "precipitation_sum",
    "climate_comfort",
    "baseline_population_log",
    "_viirs_log1p",
    "viirs_ntl_yoy",
    "viirs_recent_drop",
    "viirs_lit_area_km2",
    "ghsl_built_surface_km2",
    "ghsl_built_surface_yoy",
    "ghsl_built_density",
    "road_length_km_total",
    "road_growth_intensity",
    "arterial_share",
    "intersection_density",
    "transport_intensity",
    "commerce_ratio",
    "flight_connectivity_total",
    "international_route_share",
    "gravity_access_viirs",
    "spatial_lag_log_viirs_ntl_wdist",
    "policy_intensity_sum_iso_year",
    "policy_event_count_iso_year",
]


def _numeric_series(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(np.full(len(frame), default, dtype=float), index=frame.index)


def _build_no2_predictor_frame(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    if "year" in panel.columns:
        year_vals = pd.to_numeric(panel["year"], errors="coerce")
        year_min = float(year_vals.min()) if year_vals.notna().any() else 0.0
        out["_year_index"] = year_vals - year_min
    else:
        out["_year_index"] = 0.0
    viirs_mean = np.clip(_numeric_series(panel, "viirs_ntl_mean", default=0.0), a_min=0.0, a_max=None)
    out["_viirs_log1p"] = np.log1p(viirs_mean)
    for col in NO2_PREDICTOR_CANDIDATES:
        if col in out.columns:
            continue
        out[col] = _numeric_series(panel, col)
    return out


def add_no2_backcast_features(
    panel: pd.DataFrame,
    fit_end_year: int | None = None,
    output_stub: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Add NO2-filled features for ML modules without overwriting raw observed columns.

    The helper fits on observed NO2 rows only and backcasts missing early years using
    contemporaneous non-NO2 proxies. When ``fit_end_year`` is provided, fitting is
    restricted to rows with ``year <= fit_end_year`` to avoid look-ahead leakage in
    out-of-time validation.
    """

    out = panel.copy()
    for raw_col, filled_col in NO2_FILLED_MAP.items():
        base = _numeric_series(out, raw_col)
        out[raw_col] = base
        out[filled_col] = base
    out["no2_backcast_flag"] = 0
    out["has_no2_observation_or_backcast"] = pd.to_numeric(
        out.get("has_no2_observation"), errors="coerce"
    ).fillna(0).astype(int)
    out["no2_backcast_source"] = np.where(out["no2_trop_mean"].notna(), "observed", "missing")

    year_vals = pd.to_numeric(out.get("year"), errors="coerce")
    observed_mask = out["no2_trop_mean"].notna()
    fit_mask = observed_mask.copy()
    if fit_end_year is not None and year_vals.notna().any():
        fit_mask = fit_mask & (year_vals <= int(fit_end_year))

    predictor_df = _build_no2_predictor_frame(out)
    fit_predictors = predictor_df.loc[fit_mask].copy()

    usable_predictors: List[str] = []
    fill_values: Dict[str, float] = {}
    for col in predictor_df.columns:
        s = pd.to_numeric(fit_predictors[col], errors="coerce")
        if float(s.notna().mean()) < 0.55:
            continue
        fill_values[col] = float(s.median()) if s.notna().any() else 0.0
        usable_predictors.append(col)

    summary: Dict[str, Any] = {
        "status": "skipped",
        "fit_end_year": int(fit_end_year) if fit_end_year is not None else None,
        "observed_rows_total": int(observed_mask.sum()),
        "fit_rows_total": int(fit_mask.sum()),
        "predictor_count": int(len(usable_predictors)),
        "targets": {},
    }
    if fit_mask.sum() < 200 or len(usable_predictors) < 6:
        if output_stub:
            dump_json(DATA_OUTPUTS / f"{output_stub}_no2_backcast_summary.json", summary)
        return out, summary

    x_all = np.column_stack(
        [
            pd.to_numeric(predictor_df[col], errors="coerce").fillna(fill_values[col]).to_numpy(dtype=float)
            for col in usable_predictors
        ]
    )
    missing_mask_global = out["no2_trop_mean"].isna()

    for raw_target in NO2_DIRECT_TARGETS:
        filled_target = NO2_FILLED_MAP[raw_target]
        y_fit = pd.to_numeric(out.loc[fit_mask, raw_target], errors="coerce")
        valid_fit_mask = y_fit.notna().to_numpy(dtype=bool)
        train_rows = int(valid_fit_mask.sum())
        if train_rows < 160:
            summary["targets"][raw_target] = {
                "status": "skipped",
                "reason": "too_few_training_rows",
                "train_rows": train_rows,
            }
            continue
        x_fit = x_all[np.asarray(fit_mask), :][valid_fit_mask]
        y_fit_arr = y_fit.to_numpy(dtype=float)[valid_fit_mask]
        model = HistGradientBoostingRegressor(
            random_state=42,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=20,
            l2_regularization=0.02,
        )
        try:
            model.fit(x_fit, y_fit_arr)
            pred_all = model.predict(x_all)
        except Exception as exc:  # noqa: BLE001
            summary["targets"][raw_target] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}",
                "train_rows": train_rows,
            }
            continue

        out.loc[missing_mask_global, filled_target] = pred_all[missing_mask_global.to_numpy(dtype=bool)]
        if raw_target == "no2_trop_anomaly_abs_mean":
            out[filled_target] = np.clip(pd.to_numeric(out[filled_target], errors="coerce"), a_min=0.0, a_max=None)
        summary["targets"][raw_target] = {
            "status": "ok",
            "train_rows": train_rows,
            "filled_rows": int(missing_mask_global.sum()),
        }

    mean_filled = pd.to_numeric(out[NO2_FILLED_MAP["no2_trop_mean"]], errors="coerce")
    yoy_filled = mean_filled.groupby(out["city_id"]).diff()
    raw_yoy = pd.to_numeric(out["no2_trop_yoy_mean"], errors="coerce")
    out[NO2_FILLED_MAP["no2_trop_yoy_mean"]] = raw_yoy.where(raw_yoy.notna(), yoy_filled)
    out["no2_backcast_flag"] = (mean_filled.notna() & out["no2_trop_mean"].isna()).astype(int)
    out["has_no2_observation_or_backcast"] = mean_filled.notna().astype(int)
    out["no2_backcast_source"] = np.where(
        out["no2_trop_mean"].notna(),
        "observed",
        np.where(out["no2_backcast_flag"].eq(1), "model_backcast", "missing"),
    )

    coverage = (
        out.groupby("year")[
            ["no2_trop_mean", NO2_FILLED_MAP["no2_trop_mean"], "no2_backcast_flag"]
        ]
        .agg(
            observed_share=("no2_trop_mean", lambda s: float(pd.to_numeric(s, errors="coerce").notna().mean())),
            filled_share=(NO2_FILLED_MAP["no2_trop_mean"], lambda s: float(pd.to_numeric(s, errors="coerce").notna().mean())),
            backcast_share=("no2_backcast_flag", "mean"),
        )
        .reset_index()
    )
    if output_stub:
        coverage.to_csv(DATA_OUTPUTS / f"{output_stub}_no2_backcast_coverage.csv", index=False)

    summary.update(
        {
            "status": "ok",
            "filled_rows_total": int(out["no2_backcast_flag"].sum()),
            "filled_feature_non_null_ratio": float(mean_filled.notna().mean()) if len(mean_filled) else 0.0,
            "coverage_file": str(DATA_OUTPUTS / f"{output_stub}_no2_backcast_coverage.csv") if output_stub else None,
            "used_predictors": usable_predictors,
        }
    )
    if output_stub:
        dump_json(DATA_OUTPUTS / f"{output_stub}_no2_backcast_summary.json", summary)
    LOGGER.info(
        "NO2 backcast ready: stub=%s, observed_rows=%s, filled_rows=%s, predictors=%s",
        output_stub or "adhoc",
        int(observed_mask.sum()),
        int(out["no2_backcast_flag"].sum()),
        len(usable_predictors),
    )
    return out, summary
