from __future__ import annotations

"""Structured spatial-weight helpers for distance and network spillovers."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils import DATA_PROCESSED, dump_json


def build_flight_weight_matrix(
    panel: pd.DataFrame,
    *,
    top_k: int | None = 12,
    persist: bool = True,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    """Build a row-standardized flight-network weight matrix from route edges."""
    edge_path = DATA_PROCESSED / "openflights_city_route_edges.csv"
    city_ids = sorted(panel["city_id"].astype(str).dropna().unique().tolist()) if "city_id" in panel.columns else []
    empty = pd.DataFrame(columns=["source_city_id", "target_city_id", "route_rows", "weight"])
    summary: Dict[str, object] = {
        "status": "skipped",
        "edge_file": str(edge_path),
        "cities": int(len(city_ids)),
        "rows": 0,
    }
    if not city_ids or not edge_path.exists():
        if persist:
            empty.to_csv(DATA_PROCESSED / "spatial_weight_matrix_flight.csv", index=False)
            dump_json(DATA_PROCESSED / "spatial_weight_matrix_flight_summary.json", summary)
        return empty, summary

    edges = pd.read_csv(edge_path)
    required = {"source_city_id", "target_city_id", "route_rows"}
    if not required.issubset(set(edges.columns)):
        summary["reason"] = "missing_edge_columns"
        if persist:
            empty.to_csv(DATA_PROCESSED / "spatial_weight_matrix_flight.csv", index=False)
            dump_json(DATA_PROCESSED / "spatial_weight_matrix_flight_summary.json", summary)
        return empty, summary

    work = edges.copy()
    work["source_city_id"] = work["source_city_id"].astype(str)
    work["target_city_id"] = work["target_city_id"].astype(str)
    work["route_rows"] = pd.to_numeric(work["route_rows"], errors="coerce")
    work = work[
        work["source_city_id"].isin(city_ids)
        & work["target_city_id"].isin(city_ids)
        & (work["source_city_id"] != work["target_city_id"])
        & work["route_rows"].notna()
        & (work["route_rows"] > 0.0)
    ].copy()
    if work.empty:
        summary["reason"] = "no_eligible_edges"
        if persist:
            empty.to_csv(DATA_PROCESSED / "spatial_weight_matrix_flight.csv", index=False)
            dump_json(DATA_PROCESSED / "spatial_weight_matrix_flight_summary.json", summary)
        return empty, summary

    work = (
        work.groupby(["source_city_id", "target_city_id"], as_index=False)["route_rows"]
        .sum()
        .sort_values(["source_city_id", "route_rows"], ascending=[True, False])
    )
    if top_k is not None and int(top_k) >= 1:
        work = work.groupby("source_city_id", group_keys=False).head(int(top_k)).copy()
    denom = work.groupby("source_city_id")["route_rows"].transform("sum")
    work["weight"] = np.where(denom > 0.0, work["route_rows"] / denom, 0.0)
    work = work.sort_values(["source_city_id", "weight", "target_city_id"], ascending=[True, False, True]).reset_index(drop=True)

    source_degree = work.groupby("source_city_id")["target_city_id"].nunique()
    summary = {
        "status": "ok",
        "edge_file": str(edge_path),
        "cities": int(len(city_ids)),
        "rows": int(len(work)),
        "sources_with_edges": int(source_degree.index.nunique()),
        "mean_out_degree": float(source_degree.mean()) if not source_degree.empty else 0.0,
        "median_out_degree": float(source_degree.median()) if not source_degree.empty else 0.0,
        "top_k": int(top_k) if top_k is not None else None,
    }
    if persist:
        work.to_csv(DATA_PROCESSED / "spatial_weight_matrix_flight.csv", index=False)
        dump_json(DATA_PROCESSED / "spatial_weight_matrix_flight_summary.json", summary)
    return work, summary


def build_flight_neighbor_map(
    panel: pd.DataFrame,
    *,
    top_k: int | None = 12,
    persist: bool = True,
) -> tuple[Dict[str, List[Tuple[str, float]]], Dict[str, object]]:
    """Return a sparse neighbor map keyed by source city with row-standardized flight weights."""
    weights, summary = build_flight_weight_matrix(panel, top_k=top_k, persist=persist)
    neighbor_map: Dict[str, List[Tuple[str, float]]] = {}
    for row in weights.itertuples(index=False):
        neighbor_map.setdefault(str(row.source_city_id), []).append((str(row.target_city_id), float(row.weight)))
    return neighbor_map, summary


def build_road_proxy_weight_matrix(
    panel: pd.DataFrame,
    *,
    top_k: int | None = 8,
    max_distance_km: float = 1500.0,
    persist: bool = True,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    """Build a sparse road-structure-informed accessibility matrix.

    This is not a shortest-path routing matrix. It uses city-level OSM road
    structure signals together with inter-city geographic distance to produce a
    conservative accessibility proxy for regional robustness checks.
    """
    required = {
        "city_id",
        "continent",
        "latitude",
        "longitude",
        "road_length_km_total",
        "arterial_share",
        "intersection_density",
    }
    empty = pd.DataFrame(columns=["source_city_id", "target_city_id", "distance_km", "raw_score", "weight"])
    city_ids = sorted(panel["city_id"].astype(str).dropna().unique().tolist()) if "city_id" in panel.columns else []
    summary: Dict[str, object] = {
        "status": "skipped",
        "cities": int(len(city_ids)),
        "rows": 0,
        "top_k": int(top_k) if top_k is not None else None,
        "max_distance_km": float(max_distance_km),
        "matrix_type": "road_structure_accessibility_proxy",
    }
    if not required.issubset(panel.columns):
        summary["reason"] = "missing_required_columns"
        if persist:
            empty.to_csv(DATA_PROCESSED / "spatial_weight_matrix_road_proxy.csv", index=False)
            dump_json(DATA_PROCESSED / "spatial_weight_matrix_road_proxy_summary.json", summary)
        return empty, summary

    latest_year = int(pd.to_numeric(panel["year"], errors="coerce").max()) if "year" in panel.columns else None
    city_df = panel.copy()
    if latest_year is not None:
        city_df = city_df[pd.to_numeric(city_df["year"], errors="coerce") == latest_year].copy()
    city_df = (
        city_df[
            [
                "city_id",
                "continent",
                "latitude",
                "longitude",
                "road_length_km_total",
                "arterial_share",
                "intersection_density",
            ]
        ]
        .drop_duplicates("city_id")
        .copy()
    )
    if city_df.empty:
        summary["reason"] = "no_city_rows"
        if persist:
            empty.to_csv(DATA_PROCESSED / "spatial_weight_matrix_road_proxy.csv", index=False)
            dump_json(DATA_PROCESSED / "spatial_weight_matrix_road_proxy_summary.json", summary)
        return empty, summary

    work = city_df.copy()
    work["road_length_km_total"] = pd.to_numeric(work["road_length_km_total"], errors="coerce").clip(lower=0.0).fillna(0.0)
    work["arterial_share"] = pd.to_numeric(work["arterial_share"], errors="coerce").clip(lower=0.0, upper=1.0).fillna(0.0)
    work["intersection_density"] = pd.to_numeric(work["intersection_density"], errors="coerce").clip(lower=0.0).fillna(0.0)
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work = work.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if work.empty:
        summary["reason"] = "no_valid_coordinates"
        if persist:
            empty.to_csv(DATA_PROCESSED / "spatial_weight_matrix_road_proxy.csv", index=False)
            dump_json(DATA_PROCESSED / "spatial_weight_matrix_road_proxy_summary.json", summary)
        return empty, summary

    road_score = (
        np.log1p(work["road_length_km_total"].to_numpy(dtype=float))
        * (0.5 + work["arterial_share"].to_numpy(dtype=float))
        * np.sqrt(1.0 + work["intersection_density"].to_numpy(dtype=float))
    )
    work["road_access_score"] = np.where(np.isfinite(road_score), road_score, 0.0)

    rows: List[Dict[str, object]] = []
    ids = work["city_id"].astype(str).tolist()
    cont = work["continent"].astype(str).tolist()
    lat = work["latitude"].to_numpy(dtype=float)
    lon = work["longitude"].to_numpy(dtype=float)
    score = work["road_access_score"].to_numpy(dtype=float)

    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.sqrt(a))
        return float(r * c)

    for i, src in enumerate(ids):
        cand: List[Dict[str, object]] = []
        for j, tgt in enumerate(ids):
            if i == j or cont[i] != cont[j]:
                continue
            d_km = _haversine(float(lat[i]), float(lon[i]), float(lat[j]), float(lon[j]))
            if not np.isfinite(d_km) or d_km <= 0.0 or d_km > float(max_distance_km):
                continue
            raw = float(np.sqrt(max(score[i], 0.0) * max(score[j], 0.0)) / max(d_km, 25.0))
            if raw <= 0.0 or not np.isfinite(raw):
                continue
            cand.append(
                {
                    "source_city_id": src,
                    "target_city_id": tgt,
                    "distance_km": float(d_km),
                    "raw_score": raw,
                }
            )
        if not cand:
            continue
        cand_df = pd.DataFrame(cand).sort_values(["raw_score", "target_city_id"], ascending=[False, True])
        if top_k is not None and int(top_k) >= 1:
            cand_df = cand_df.head(int(top_k)).copy()
        denom = float(cand_df["raw_score"].sum())
        cand_df["weight"] = cand_df["raw_score"] / denom if denom > 0.0 else 1.0 / max(len(cand_df), 1)
        rows.extend(cand_df.to_dict(orient="records"))

    if not rows:
        summary["reason"] = "no_eligible_edges"
        if persist:
            empty.to_csv(DATA_PROCESSED / "spatial_weight_matrix_road_proxy.csv", index=False)
            dump_json(DATA_PROCESSED / "spatial_weight_matrix_road_proxy_summary.json", summary)
        return empty, summary

    out = pd.DataFrame(rows).sort_values(["source_city_id", "weight", "target_city_id"], ascending=[True, False, True]).reset_index(drop=True)
    source_degree = out.groupby("source_city_id")["target_city_id"].nunique()
    summary = {
        "status": "ok",
        "cities": int(len(ids)),
        "rows": int(len(out)),
        "sources_with_edges": int(source_degree.index.nunique()),
        "mean_out_degree": float(source_degree.mean()) if not source_degree.empty else 0.0,
        "median_out_degree": float(source_degree.median()) if not source_degree.empty else 0.0,
        "top_k": int(top_k) if top_k is not None else None,
        "max_distance_km": float(max_distance_km),
        "latest_year": latest_year,
        "matrix_type": "road_structure_accessibility_proxy",
    }
    if persist:
        out.to_csv(DATA_PROCESSED / "spatial_weight_matrix_road_proxy.csv", index=False)
        dump_json(DATA_PROCESSED / "spatial_weight_matrix_road_proxy_summary.json", summary)
    return out, summary


def build_road_proxy_neighbor_map(
    panel: pd.DataFrame,
    *,
    top_k: int | None = 8,
    max_distance_km: float = 1500.0,
    persist: bool = True,
) -> tuple[Dict[str, List[Tuple[str, float]]], Dict[str, object]]:
    """Return sparse road-structure-informed neighbor map."""
    weights, summary = build_road_proxy_weight_matrix(
        panel,
        top_k=top_k,
        max_distance_km=max_distance_km,
        persist=persist,
    )
    neighbor_map: Dict[str, List[Tuple[str, float]]] = {}
    for row in weights.itertuples(index=False):
        neighbor_map.setdefault(str(row.source_city_id), []).append((str(row.target_city_id), float(row.weight)))
    return neighbor_map, summary
