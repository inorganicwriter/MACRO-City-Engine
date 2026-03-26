from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import math

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.causal_st import CausalSTConfig, _run_single
from src.spatial_weights import build_road_proxy_weight_matrix
from src.utils import DATA_OUTPUTS, DATA_PROCESSED, dump_json


LOGGER = logging.getLogger("road_proxy_robustness")


def _slug(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("&", "and")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _bootstrap() -> int:
    raw = os.environ.get("ROAD_PROXY_BOOTSTRAP", "16")
    try:
        return max(8, int(raw))
    except Exception:  # noqa: BLE001
        return 16


def _load_panel() -> pd.DataFrame:
    panel_path = Path("data/processed/global_city_panel.csv")
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)
    return pd.read_csv(panel_path)


def _road_relevant_core(panel: pd.DataFrame, *, top_k: int, max_distance_km: float, min_neighbors: int) -> pd.DataFrame:
    base = panel[panel["continent"].astype(str).isin(["Europe", "North America"])].copy()
    if base.empty:
        return base
    weights, _summary = build_road_proxy_weight_matrix(
        base,
        top_k=top_k,
        max_distance_km=max_distance_km,
        persist=False,
    )
    if weights.empty:
        return base.iloc[0:0].copy()
    degree = weights.groupby("source_city_id")["target_city_id"].nunique()
    keep_city_ids = set(degree[degree >= int(min_neighbors)].index.astype(str).tolist())
    return base[base["city_id"].astype(str).isin(keep_city_ids)].copy()


def _run_variant(name: str, panel: pd.DataFrame, cfg: CausalSTConfig) -> Dict[str, object]:
    res = _run_single(panel, cfg)
    if "summary" not in res:
        payload = {"variant": name, "status": res.get("status", "skipped"), "reason": res.get("reason", "")}
    else:
        summary = dict(res["summary"])
        summary["variant"] = name
        summary["cities"] = int(panel["city_id"].astype(str).nunique())
        summary["years"] = sorted(pd.to_numeric(panel["year"], errors="coerce").dropna().astype(int).unique().tolist())
        ci = summary.get("att_ci95")
        att = float(summary.get("att_post", float("nan")))
        n_ok = int(summary.get("n_bootstrap_ok", 0) or 0)
        cities = int(summary.get("cities", 0) or 0)
        ci_is_valid = (
            isinstance(ci, (list, tuple))
            and len(ci) == 2
            and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in ci)
        )
        point_outside_ci = bool(ci_is_valid and not (float(ci[0]) <= att <= float(ci[1])))
        small_sample_flag = bool(cities < 20)
        incomplete_bootstrap_flag = bool(n_ok < int(cfg.n_bootstrap))
        if point_outside_ci and (small_sample_flag or incomplete_bootstrap_flag):
            summary["variant_status"] = "unstable_small_sample_bootstrap_geometry"
            summary["stability_note"] = (
                "Point estimate falls outside percentile CI in a small-sample road-topology run; "
                "t-value and p-value are suppressed and the row should not be interpreted substantively."
            )
            summary["t_value"] = float("nan")
            summary["p_value"] = float("nan")
            summary["att_ci95"] = [float("nan"), float("nan")]
        else:
            summary["variant_status"] = "ok"
            summary["stability_note"] = ""
        payload = summary
    dump_json(DATA_OUTPUTS / f"causal_st_{name}_summary.json", payload)
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    panel = _load_panel()
    boot = _bootstrap()
    cfg = CausalSTConfig(
        use_spatial=True,
        use_temporal=True,
        use_dr=True,
        spatial_mode="road_proxy",
        n_bootstrap=boot,
        k_neighbors=8,
    )
    min_neighbors = max(1, int(os.environ.get("ROAD_PROXY_MIN_NEIGHBORS", "5")))
    variants: List[tuple[str, pd.DataFrame]] = [("road_proxy_global", panel.copy())]
    dense_core = _road_relevant_core(
        panel,
        top_k=cfg.k_neighbors,
        max_distance_km=1500.0,
        min_neighbors=min_neighbors,
    )
    if not dense_core.empty:
        variants.append(("road_proxy_dense_core", dense_core))
    continents = sorted(str(v) for v in panel["continent"].dropna().astype(str).unique().tolist())
    for continent in continents:
        sub = panel[panel["continent"].astype(str) == continent].copy()
        if sub.empty:
            continue
        variants.append((f"road_proxy_{_slug(continent)}", sub))
    rows: List[Dict[str, object]] = []
    for name, sub in variants:
        LOGGER.info("running %s rows=%d cities=%d", name, len(sub), sub["city_id"].astype(str).nunique())
        rows.append(_run_variant(name, sub, cfg))
    pd.DataFrame(rows).to_csv(DATA_OUTPUTS / "causal_st_road_proxy_robustness.csv", index=False)
    dump_json(DATA_OUTPUTS / "causal_st_road_proxy_robustness.json", {"variants": rows})
    global_row = next((row for row in rows if row.get("variant") == "road_proxy_global"), None)
    if isinstance(global_row, dict) and isinstance(global_row.get("spatial_summary"), dict):
        dump_json(DATA_PROCESSED / "spatial_weight_matrix_road_proxy_summary.json", global_row["spatial_summary"])


if __name__ == "__main__":
    main()
