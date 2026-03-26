from __future__ import annotations

"""Shared utilities for paths, geometry, and lightweight transforms."""

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("URBAN_PULSE_OUTPUT_ROOT", str(ROOT))).resolve()

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = OUTPUT_ROOT / "data" / "processed"
DATA_OUTPUTS = OUTPUT_ROOT / "data" / "outputs"
MODELS_DIR = OUTPUT_ROOT / "models"
REPORTS_DIR = OUTPUT_ROOT / "reports"
WEB_DATA_DIR = OUTPUT_ROOT / "web" / "static" / "data"


def ensure_project_dirs() -> None:
    """Create standard output directories if missing."""
    for p in [DATA_RAW, DATA_PROCESSED, DATA_OUTPUTS, MODELS_DIR, REPORTS_DIR, WEB_DATA_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def minmax_scale(series: np.ndarray) -> np.ndarray:
    """Scale a numeric array to `[0, 1]` while handling degenerate ranges."""
    if series.size == 0:
        return series
    vmin = np.nanmin(series)
    vmax = np.nanmax(series)
    if math.isclose(vmax, vmin):
        return np.zeros_like(series)
    return (series - vmin) / (vmax - vmin)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def point_line_distance_km(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """Distance from a point to a line segment on a small-area planar approximation."""
    # For small urban extents, an equirectangular projection gives adequate accuracy.
    lat0 = math.radians((a[0] + b[0] + point[0]) / 3.0)

    def to_xy(lat: float, lon: float) -> tuple[float, float]:
        x = math.radians(lon) * math.cos(lat0) * 6371.0
        y = math.radians(lat) * 6371.0
        return x, y

    px, py = to_xy(point[0], point[1])
    ax, ay = to_xy(a[0], a[1])
    bx, by = to_xy(b[0], b[1])

    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    denom = abx * abx + aby * aby
    if denom == 0:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)

    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    """Write UTF-8 JSON with stable pretty-printing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def quantile_bins(values: Iterable[float], q: int = 5) -> np.ndarray:
    """Bucket values into quantile bins indexed from `0` to `q-1`."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    quantiles = np.quantile(arr, np.linspace(0, 1, q + 1))
    return np.digitize(arr, quantiles[1:-1], right=True)


# Ensure writable artifact directories exist for both pipeline and unit-test entrypoints.
ensure_project_dirs()
