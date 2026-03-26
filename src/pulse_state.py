from __future__ import annotations

"""Dynamic pulse-state estimation for cities."""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from .utils import DATA_OUTPUTS, dump_json

LOGGER = logging.getLogger(__name__)


STATE_ORDER = [
    "accelerating_expansion",
    "decelerating_expansion",
    "deepening_contraction",
    "bottoming_recovery",
]


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    expv = np.exp(shifted)
    return expv / np.maximum(expv.sum(axis=1, keepdims=True), 1e-12)


def estimate_pulse_states(panel: pd.DataFrame) -> Dict[str, object]:
    """Estimate city-year pulse state probabilities from dynamic features."""
    df = panel[["city_id", "city_name", "year", "composite_index"]].copy()
    df = df.sort_values(["city_id", "year"]).reset_index(drop=True)

    df["composite_smooth"] = (
        df.groupby("city_id")["composite_index"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        .astype(float)
    )
    df["log_comp"] = np.log1p(np.maximum(df["composite_smooth"], 0.0))
    df["velocity"] = df.groupby("city_id")["log_comp"].diff().fillna(0.0)
    df["acceleration"] = df.groupby("city_id")["velocity"].diff().fillna(0.0)
    df["volatility"] = (
        df.groupby("city_id")["velocity"]
        .transform(lambda x: x.rolling(window=4, min_periods=2).std())
        .fillna(0.0)
    )

    for col in ["velocity", "acceleration", "volatility"]:
        mean = df.groupby("city_id")[col].transform("mean")
        std = df.groupby("city_id")[col].transform("std").replace(0.0, 1.0).fillna(1.0)
        df[f"{col}_z"] = ((df[col] - mean) / std).fillna(0.0)

    v = df["velocity_z"].to_numpy(dtype=float)
    a = df["acceleration_z"].to_numpy(dtype=float)
    vol = df["volatility_z"].to_numpy(dtype=float)

    scores = np.column_stack(
        [
            1.2 * v + 1.0 * a - 0.35 * vol,
            0.8 * v - 1.0 * a - 0.20 * vol,
            -1.3 * v - 0.6 * a + 0.50 * vol,
            -0.4 * v + 1.1 * a - 0.20 * vol,
        ]
    )
    probs = _softmax(scores)

    for idx, state in enumerate(STATE_ORDER):
        df[f"prob_{state}"] = probs[:, idx]

    state_idx = probs.argmax(axis=1)
    df["pulse_state"] = [STATE_ORDER[i] for i in state_idx]

    out_cols = [
        "city_id",
        "city_name",
        "year",
        "composite_index",
        "composite_smooth",
        "velocity",
        "acceleration",
        "volatility",
        "pulse_state",
    ] + [f"prob_{s}" for s in STATE_ORDER]
    states_out = df[out_cols].copy()
    states_out.to_csv(DATA_OUTPUTS / "pulse_state_probabilities.csv", index=False)

    trans = []
    for city_id, g in states_out.sort_values(["city_id", "year"]).groupby("city_id"):
        vals = g["pulse_state"].tolist()
        for i in range(len(vals) - 1):
            trans.append({"city_id": city_id, "from_state": vals[i], "to_state": vals[i + 1]})

    if trans:
        trans_df = pd.DataFrame(trans)
        mat = (
            trans_df.groupby(["from_state", "to_state"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values(["from_state", "to_state"])
        )
        mat["probability"] = mat["count"] / mat.groupby("from_state")["count"].transform("sum")
    else:
        mat = pd.DataFrame(columns=["from_state", "to_state", "count", "probability"])
    mat.to_csv(DATA_OUTPUTS / "pulse_transition_matrix.csv", index=False)

    latest = states_out.sort_values("year").groupby("city_id", as_index=False).tail(1)
    latest_dist = latest["pulse_state"].value_counts(normalize=True).to_dict()

    summary = {
        "n_rows": int(len(states_out)),
        "n_cities": int(states_out["city_id"].nunique()),
        "latest_state_distribution": {k: float(v) for k, v in latest_dist.items()},
        "states": STATE_ORDER,
    }
    dump_json(DATA_OUTPUTS / "pulse_state_summary.json", summary)
    LOGGER.info("Pulse states estimated: %s rows, %s cities", summary["n_rows"], summary["n_cities"])
    return summary

