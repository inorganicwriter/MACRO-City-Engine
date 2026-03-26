from __future__ import annotations

"""Project configuration loading and validation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class BBox:
    south: float
    west: float
    north: float
    east: float


@dataclass(frozen=True)
class TimeRange:
    start_year: int
    end_year: int


@dataclass(frozen=True)
class InterventionConfig:
    name: str
    start_year: int
    line_start: tuple[float, float]
    line_end: tuple[float, float]
    treatment_radius_km: float


@dataclass(frozen=True)
class APIConfig:
    overpass_url: str
    open_meteo_url: str


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    city: str
    scope: str
    max_cities_default: int
    strict_real_data: bool
    strict_no_synthetic: bool
    bbox: BBox
    cell_size_km: float
    time_range: TimeRange
    intervention: InterventionConfig
    api: APIConfig
    random_seed: int


def _resolve(path: str | Path | None) -> Path:
    """Resolve config path from explicit value or default project root."""
    if path is None:
        return Path(__file__).resolve().parents[1] / "config.json"
    return Path(path).resolve()


def _to_latlon_pair(raw: Any) -> tuple[float, float]:
    """Convert list-like input to a `(lat, lon)` tuple."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        msg = "line_start/line_end must be a 2-item list: [lat, lon]."
        raise ValueError(msg)
    return float(raw[0]), float(raw[1])


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load config JSON and return a strongly-typed dataclass tree."""
    config_path = _resolve(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)

    if raw["time_range"]["start_year"] > raw["time_range"]["end_year"]:
        msg = "time_range.start_year must be <= time_range.end_year."
        raise ValueError(msg)
    if float(raw["cell_size_km"]) <= 0:
        msg = "cell_size_km must be positive."
        raise ValueError(msg)

    return ProjectConfig(
        project_name=raw["project_name"],
        city=raw["city"],
        scope=raw.get("scope", "global_cities"),
        max_cities_default=int(raw.get("max_cities_default", 50)),
        strict_real_data=bool(raw.get("strict_real_data", False)),
        strict_no_synthetic=bool(raw.get("strict_no_synthetic", False)),
        bbox=BBox(**raw["bbox"]),
        cell_size_km=float(raw["cell_size_km"]),
        time_range=TimeRange(**raw["time_range"]),
        intervention=InterventionConfig(
            name=raw["intervention"]["name"],
            start_year=int(raw["intervention"]["start_year"]),
            line_start=_to_latlon_pair(raw["intervention"]["line_start"]),
            line_end=_to_latlon_pair(raw["intervention"]["line_end"]),
            treatment_radius_km=float(raw["intervention"]["treatment_radius_km"]),
        ),
        api=APIConfig(**raw["api"]),
        random_seed=int(raw["random_seed"]),
    )
