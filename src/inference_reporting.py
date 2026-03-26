from __future__ import annotations

"""Standardized uncertainty and multiple-testing reporting utilities."""

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .utils import DATA_OUTPUTS, dump_json


def _as_float(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:  # noqa: BLE001
        return None
    if not np.isfinite(x):
        return None
    return x


def _ci_from_se(coef: float | None, se: float | None) -> tuple[float | None, float | None]:
    if coef is None or se is None:
        return None, None
    return float(coef - 1.96 * se), float(coef + 1.96 * se)


def _bh_adjust(pvals: List[float]) -> List[float]:
    if not pvals:
        return []
    arr = np.asarray(pvals, dtype=float)
    n = len(arr)
    order = np.argsort(arr)
    ranked = arr[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = float(min(prev, ranked[i] * n / max(rank, 1)))
        q[i] = val
        prev = val
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return [float(x) for x in out.tolist()]


def _load_json(path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def run_inference_reporting_suite() -> Dict[str, Any]:
    """Export standardized CI/bootstrap/multiple-testing artifacts."""
    econ = _load_json(DATA_OUTPUTS / "econometric_summary.json")
    if not econ:
        out = {"status": "skipped", "reason": "econometric_summary_missing"}
        dump_json(DATA_OUTPUTS / "inference_reporting_summary.json", out)
        return out

    rows_main: List[Dict[str, Any]] = []

    def _add_estimator(name: str, payload: Dict[str, Any]) -> None:
        coef = _as_float(payload.get("coef"))
        se = _as_float(payload.get("stderr"))
        p = _as_float(payload.get("p_value"))
        ci_low, ci_high = _ci_from_se(coef, se)
        rows_main.append(
            {
                "estimator": name,
                "coef": coef,
                "stderr": se,
                "t_value": _as_float(payload.get("t_value")),
                "p_value": p,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n_obs": _as_float(payload.get("n_obs")),
            }
        )

    _add_estimator("twfe_did", econ.get("did_two_way_fe", {}) if isinstance(econ.get("did_two_way_fe"), dict) else {})
    _add_estimator("dml_did", econ.get("dml_did", {}) if isinstance(econ.get("dml_did"), dict) else {})
    _add_estimator("dr_did", econ.get("dr_did", {}) if isinstance(econ.get("dr_did"), dict) else {})
    _add_estimator(
        "staggered_post_avg",
        {
            "coef": (econ.get("staggered_did", {}) or {}).get("post_avg_att"),
            "stderr": np.nan,
            "t_value": (econ.get("staggered_did", {}) or {}).get("post_avg_t_value"),
            "p_value": np.nan,
            "n_obs": (econ.get("staggered_did", {}) or {}).get("treated_city_count"),
        },
    )
    _add_estimator(
        "not_yet_treated_weighted",
        {
            "coef": (econ.get("not_yet_treated_did", {}) or {}).get("att_weighted"),
            "stderr": (econ.get("not_yet_treated_did", {}) or {}).get("stderr_weighted"),
            "t_value": (econ.get("not_yet_treated_did", {}) or {}).get("t_value_weighted"),
            "p_value": (econ.get("not_yet_treated_did", {}) or {}).get("p_value_weighted"),
            "n_obs": (econ.get("not_yet_treated_did", {}) or {}).get("n_treated_cohorts"),
        },
    )

    boot = econ.get("did_twfe_cluster_bootstrap", {})
    if isinstance(boot, dict):
        rows_main.append(
            {
                "estimator": "twfe_cluster_bootstrap",
                "coef": _as_float(boot.get("coef_observed")),
                "stderr": _as_float(boot.get("coef_bootstrap_std")),
                "t_value": _as_float(boot.get("t_value_observed")),
                "p_value": _as_float(boot.get("p_value_two_sided")),
                "ci95_low": _as_float((boot.get("ci95_percentile") or [None, None])[0]),
                "ci95_high": _as_float((boot.get("ci95_percentile") or [None, None])[1]),
                "n_obs": np.nan,
            }
        )

    main_df = pd.DataFrame(rows_main)
    main_df.to_csv(DATA_OUTPUTS / "inference_main_results.csv", index=False)

    p_rows: List[Dict[str, Any]] = []
    policy_path = DATA_OUTPUTS / "econometric_policy_source_sensitivity.csv"
    if policy_path.exists():
        policy = pd.read_csv(policy_path)
        if "did_p_value" in policy.columns:
            for row in policy.itertuples(index=False):
                p = _as_float(getattr(row, "did_p_value", np.nan))
                if p is None:
                    continue
                p_rows.append(
                    {
                        "family": "policy_source_sensitivity",
                        "test_id": str(getattr(row, "variant", "unknown")),
                        "raw_p_value": p,
                    }
                )

    ext_path = DATA_OUTPUTS / "external_validity_indicator_results.csv"
    if ext_path.exists():
        ext = pd.read_csv(ext_path)
        if "twfe_p_value_composite" in ext.columns:
            for row in ext.itertuples(index=False):
                p = _as_float(getattr(row, "twfe_p_value_composite", np.nan))
                if p is None:
                    continue
                p_rows.append(
                    {
                        "family": "external_validity_indicators",
                        "test_id": str(getattr(row, "indicator", "unknown")),
                        "raw_p_value": p,
                    }
                )

    mt_df = pd.DataFrame(p_rows)
    family_summary: Dict[str, Dict[str, Any]] = {}
    if not mt_df.empty:
        mt_df["raw_p_value"] = pd.to_numeric(mt_df["raw_p_value"], errors="coerce")
        mt_df = mt_df.dropna(subset=["raw_p_value"]).copy()
        mt_df = mt_df.sort_values(["family", "raw_p_value"]).reset_index(drop=True)

        out_parts: List[pd.DataFrame] = []
        for family, sub in mt_df.groupby("family", dropna=False, sort=False):
            fam = sub.copy().reset_index(drop=True)
            pvals = fam["raw_p_value"].to_numpy(dtype=float).tolist()
            fam["bh_q_value"] = _bh_adjust(pvals)
            fam_n = float(len(fam))
            fam["bonferroni_p"] = np.minimum(1.0, fam["raw_p_value"] * fam_n)
            out_parts.append(fam)

            q = pd.to_numeric(fam["bh_q_value"], errors="coerce")
            family_summary[str(family)] = {
                "n_tests": int(len(fam)),
                "bh_q_lt_0_10_count": int((q < 0.10).sum()) if q.notna().any() else 0,
                "bh_q_lt_0_10_share": float((q < 0.10).mean()) if q.notna().any() else 0.0,
            }

        mt_df = pd.concat(out_parts, ignore_index=True)
        mt_df.to_csv(DATA_OUTPUTS / "inference_multiple_testing.csv", index=False)
    else:
        pd.DataFrame(columns=["family", "test_id", "raw_p_value", "bh_q_value", "bonferroni_p"]).to_csv(
            DATA_OUTPUTS / "inference_multiple_testing.csv",
            index=False,
        )

    out = {
        "status": "ok",
        "main_result_rows": int(len(main_df)),
        "multiple_testing_rows": int(len(mt_df)),
        "main_results_file": str(DATA_OUTPUTS / "inference_main_results.csv"),
        "multiple_testing_file": str(DATA_OUTPUTS / "inference_multiple_testing.csv"),
        "share_bh_q_lt_0_10": float(np.mean(mt_df["bh_q_value"] < 0.10)) if (not mt_df.empty) else None,
        "share_bh_q_lt_0_05": float(np.mean(mt_df["bh_q_value"] < 0.05)) if (not mt_df.empty) else None,
        "family_summary": family_summary,
    }
    dump_json(DATA_OUTPUTS / "inference_reporting_summary.json", out)
    return out
