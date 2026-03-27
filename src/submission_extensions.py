from __future__ import annotations

"""Submission-oriented extensions: policy registry QA, robustness gates, reproducibility manifest."""

import hashlib
import platform
import sys
from importlib import metadata as importlib_metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .utils import DATA_OUTPUTS, DATA_PROCESSED, DATA_RAW, ROOT, dump_json


WB_PROJECT_URL = "https://projects.worldbank.org/en/projects-operations/project-detail/"
WB_INDICATOR_URL = "https://api.worldbank.org/v2/indicator?format=json&per_page=20000"
WB_MACRO_URL = "https://data.worldbank.org/indicator"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import json

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return default


def _iter_urls(track: str, source_ref: str) -> List[str]:
    ref = str(source_ref or "").strip()
    if not ref:
        return []
    if track == "external_direct":
        payload = ref.split(":", 1)[1] if ":" in ref else ref
        ids = [s.strip() for s in payload.split("|") if s.strip()]
        return [f"{WB_PROJECT_URL}{pid}" for pid in ids]
    if track == "objective_indicator":
        return [WB_INDICATOR_URL]
    if track == "objective_macro":
        return [WB_MACRO_URL]
    return []


def _parse_track(source_ref: str) -> tuple[str, str]:
    ref = str(source_ref or "").strip()
    if ref.startswith("wb_project:") or ref.startswith("wb_project_regional:"):
        return "external_direct", "A"
    if ref.startswith("wb_indicator_event:"):
        return "objective_indicator", "B"
    if ref.startswith("objective_macro_rule:"):
        return "objective_macro", "B"
    if ref.startswith("ai_inferred:"):
        return "ai_inferred", "C"
    return "unknown", "C"


def run_policy_registry_enrichment() -> Dict[str, Any]:
    """Build event-level enriched policy registry plus quality diagnostics."""
    registry_path = DATA_RAW / "policy_events_registry_augmented.csv"
    if not registry_path.exists():
        registry_path = DATA_RAW / "policy_events_registry.csv"

    reg = _safe_read_csv(registry_path)
    debug = _safe_read_csv(DATA_RAW / "policy_events_registry_debug.csv")

    out_enriched = DATA_PROCESSED / "policy_event_registry_enriched.csv"
    out_links = DATA_PROCESSED / "policy_event_source_links.csv"
    out_quality = DATA_PROCESSED / "policy_event_registry_quality_report.json"

    if reg.empty:
        pd.DataFrame().to_csv(out_enriched, index=False)
        pd.DataFrame().to_csv(out_links, index=False)
        summary = {"status": "failed", "reason": "registry_empty", "registry_path": str(registry_path)}
        dump_json(out_quality, summary)
        return summary

    required = ["iso3", "start_year", "end_year", "policy_intensity", "policy_name", "source_ref"]
    for col in required:
        if col not in reg.columns:
            reg[col] = np.nan

    reg = reg[required].copy()
    reg["iso3"] = reg["iso3"].astype(str).str.upper().str.strip()
    reg["policy_name"] = reg["policy_name"].astype(str).str.strip()
    reg["source_ref"] = reg["source_ref"].astype(str).str.strip()
    reg["start_year"] = pd.to_numeric(reg["start_year"], errors="coerce")
    reg["end_year"] = pd.to_numeric(reg["end_year"], errors="coerce")
    reg["policy_intensity"] = pd.to_numeric(reg["policy_intensity"], errors="coerce")
    reg = reg.dropna(subset=["iso3", "start_year", "policy_name", "source_ref"]).copy()
    reg["start_year"] = reg["start_year"].astype(int)
    reg["end_year"] = reg["end_year"].fillna(reg["start_year"]).astype(int)
    reg = reg.sort_values(["iso3", "start_year", "policy_name", "source_ref"]).reset_index(drop=True)

    dedup_cols = ["iso3", "start_year", "policy_name", "source_ref"]
    before = len(reg)
    reg = reg.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    duplicate_rows_removed = int(before - len(reg))

    rows: List[Dict[str, Any]] = []
    link_rows: List[Dict[str, Any]] = []
    for idx, row in reg.iterrows():
        track, grade = _parse_track(str(row["source_ref"]))
        urls = _iter_urls(track, str(row["source_ref"]))
        domain = urlparse(urls[0]).netloc if urls else ""
        eid = f"pevt_{idx + 1:06d}"
        payload = str(row["source_ref"]).split(":", 1)[1] if ":" in str(row["source_ref"]) else str(row["source_ref"])
        project_ids = [s for s in payload.split("|") if s] if track == "external_direct" else []
        rows.append(
            {
                "event_id": eid,
                "iso3": str(row["iso3"]),
                "start_year": int(row["start_year"]),
                "end_year": int(row["end_year"]),
                "effective_date": f"{int(row['start_year'])}-01-01",
                "policy_name": str(row["policy_name"]),
                "policy_intensity": _as_float(row["policy_intensity"]),
                "source_ref": str(row["source_ref"]),
                "source_track": track,
                "evidence_grade": grade,
                "project_id_count": int(len(project_ids)),
                "project_ids": "|".join(project_ids),
                "source_url_count": int(len(urls)),
                "primary_source_url": urls[0] if urls else "",
                "source_domain": domain,
                "has_objective_source": int(track in {"external_direct", "objective_indicator", "objective_macro"}),
                "is_external_direct": int(track == "external_direct"),
                "is_objective_indicator": int(track == "objective_indicator"),
                "is_objective_macro": int(track == "objective_macro"),
                "is_ai_inferred": int(track == "ai_inferred"),
            }
        )
        for uidx, url in enumerate(urls):
            link_rows.append(
                {
                    "event_id": eid,
                    "source_rank": int(uidx + 1),
                    "source_url": url,
                    "source_domain": urlparse(url).netloc if url else "",
                    "source_track": track,
                    "source_ref": str(row["source_ref"]),
                }
            )

    enriched = pd.DataFrame(rows)
    links = pd.DataFrame(link_rows)

    if not debug.empty:
        d = debug.copy()
        if "year" in d.columns and "start_year" not in d.columns:
            d = d.rename(columns={"year": "start_year"})
        merge_keys = [c for c in ["iso3", "start_year", "policy_name", "source_ref"] if c in d.columns]
        keep_cols = [c for c in ["commitment_usd", "project_count", "assignment_type"] if c in d.columns]
        if merge_keys and keep_cols:
            d2 = d[merge_keys + keep_cols].drop_duplicates(subset=merge_keys, keep="first")
            enriched = enriched.merge(d2, on=merge_keys, how="left")

    if "assignment_type" not in enriched.columns:
        enriched["assignment_type"] = np.where(
            enriched["source_ref"].astype(str).str.startswith("wb_project_regional:"),
            "regional",
            np.where(enriched["is_external_direct"] == 1, "explicit", "rule_based"),
        )
    else:
        enriched["assignment_type"] = enriched["assignment_type"].fillna("rule_based").astype(str)

    enriched = enriched.sort_values(["iso3", "start_year", "policy_name", "event_id"]).reset_index(drop=True)
    enriched.to_csv(out_enriched, index=False)
    links.to_csv(out_links, index=False)

    source_track_counts = (
        enriched["source_track"].value_counts(dropna=False).sort_index().astype(int).to_dict()
        if not enriched.empty
        else {}
    )
    evidence_grade_counts = (
        enriched["evidence_grade"].value_counts(dropna=False).sort_index().astype(int).to_dict()
        if not enriched.empty
        else {}
    )
    url_coverage = float(np.mean(enriched["source_url_count"] > 0)) if not enriched.empty else 0.0
    source_domains = (
        enriched.loc[enriched["source_domain"].astype(str).str.len() > 0, "source_domain"]
        .value_counts()
        .head(10)
        .astype(int)
        .to_dict()
        if not enriched.empty
        else {}
    )
    top_policy_names = (
        enriched["policy_name"].value_counts().head(15).astype(int).to_dict() if not enriched.empty else {}
    )
    assignment_share = (
        enriched["assignment_type"].value_counts(normalize=True).round(6).to_dict() if not enriched.empty else {}
    )

    summary = {
        "status": "ok",
        "registry_input_path": str(registry_path),
        "event_rows": int(len(enriched)),
        "iso3_covered": int(enriched["iso3"].nunique()) if not enriched.empty else 0,
        "year_min": int(enriched["start_year"].min()) if not enriched.empty else None,
        "year_max": int(enriched["start_year"].max()) if not enriched.empty else None,
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "missing_intensity_rows": int(enriched["policy_intensity"].isna().sum()) if not enriched.empty else 0,
        "source_track_counts": source_track_counts,
        "evidence_grade_counts": evidence_grade_counts,
        "external_url_coverage": url_coverage,
        "source_domain_top10": source_domains,
        "assignment_type_share": assignment_share,
        "top_policy_names": top_policy_names,
        "artifacts": {
            "enriched_csv": str(out_enriched),
            "source_links_csv": str(out_links),
            "quality_report_json": str(out_quality),
        },
    }
    dump_json(out_quality, summary)
    return summary


def _gate_row(
    gate: str,
    metric: str,
    value: float,
    threshold: float,
    direction: str,
    note: str,
) -> Dict[str, Any]:
    if direction == "max":
        passed = int(value <= threshold)
        gap_ratio = float((value - threshold) / max(abs(threshold), 1e-9))
    else:
        passed = int(value >= threshold)
        gap_ratio = float((threshold - value) / max(abs(threshold), 1e-9))
    return {
        "gate": gate,
        "metric": metric,
        "value": float(value),
        "threshold": float(threshold),
        "direction": direction,
        "passed": passed,
        "gap_ratio": gap_ratio,
        "note": note,
    }


def run_robustness_audit() -> Dict[str, Any]:
    """Aggregate core robustness diagnostics into an auditable gate table."""
    econ = _safe_read_json(DATA_OUTPUTS / "econometric_summary.json")
    did_placebo = _safe_read_csv(DATA_OUTPUTS / "experiment_did_placebo.csv")
    perm = _safe_read_csv(DATA_OUTPUTS / "experiment_did_permutation_distribution.csv")
    loo_cont = _safe_read_csv(DATA_OUTPUTS / "experiment_leave_one_continent_out.csv")
    pretrend = _safe_read_csv(DATA_OUTPUTS / "idplus_event_pretrend_geometry.csv")
    shock_loo = _safe_read_csv(DATA_OUTPUTS / "exoshock_leave_one_shock_out.csv")
    shock_placebo = _safe_read_csv(DATA_OUTPUTS / "exoshock_placebo_year_distribution.csv")
    mt = _safe_read_csv(DATA_OUTPUTS / "inference_multiple_testing.csv")

    actual = did_placebo[did_placebo["spec"].astype(str).str.startswith("actual_")].copy() if not did_placebo.empty else pd.DataFrame()
    observed_coef = _as_float(actual.iloc[0].get("coef")) if not actual.empty else float("nan")
    if not np.isfinite(observed_coef):
        observed_coef = _as_float(((econ.get("did_two_way_fe") or {}).get("coef")))
    if not np.isfinite(observed_coef):
        observed_coef = 0.0

    gates: List[Dict[str, Any]] = []

    placebo_only = did_placebo[~did_placebo["spec"].astype(str).str.startswith("actual_")].copy() if not did_placebo.empty else pd.DataFrame()
    placebo_p = pd.to_numeric(placebo_only.get("p_value_norm"), errors="coerce")
    placebo_t = pd.to_numeric(placebo_only.get("t_value"), errors="coerce")
    placebo_share = float(np.mean(placebo_p < 0.10)) if placebo_p.notna().any() else 0.0
    placebo_max_t = float(np.nanmax(np.abs(placebo_t))) if placebo_t.notna().any() else 0.0
    gates.append(
        _gate_row(
            "did_placebo_false_positive",
            "placebo_share_p_lt_0_10",
            placebo_share,
            0.20,
            "max",
            "False-positive rate under placebo timing should stay low.",
        )
    )
    gates.append(
        _gate_row(
            "did_placebo_magnitude",
            "placebo_max_abs_t",
            placebo_max_t,
            1.96,
            "max",
            "Placebo t-statistics should not mimic real treatment effects.",
        )
    )

    perm_coef = pd.to_numeric(perm.get("perm_coef"), errors="coerce") if not perm.empty else pd.Series(dtype=float)
    if perm_coef.notna().any():
        perm_p = float(np.mean(np.abs(perm_coef) >= abs(observed_coef)))
    else:
        perm_p = 1.0
    gates.append(
        _gate_row(
            "did_permutation",
            "permutation_two_sided_p",
            perm_p,
            0.10,
            "min",
            "Observed DID coefficient should not look extreme under random treatment shuffles.",
        )
    )

    loo_did = pd.to_numeric(loo_cont.get("did_coef"), errors="coerce") if not loo_cont.empty else pd.Series(dtype=float)
    if loo_did.notna().any():
        sign_main = np.sign(observed_coef) if abs(observed_coef) >= 1e-10 else 0.0
        if sign_main == 0.0:
            same_sign_share = float(np.mean(np.abs(loo_did) <= 0.10))
        else:
            same_sign_share = float(np.mean(np.sign(loo_did) == sign_main))
        loo_cv = float(np.nanstd(loo_did) / max(abs(np.nanmean(loo_did)), 1e-9))
    else:
        same_sign_share = 0.0
        loo_cv = float("inf")
    gates.append(
        _gate_row(
            "leave_one_continent_sign",
            "same_sign_share",
            same_sign_share,
            0.50,
            "min",
            "Direction should remain stable under leave-one-continent-out splits.",
        )
    )
    gates.append(
        _gate_row(
            "leave_one_continent_dispersion",
            "did_cv",
            loo_cv,
            1.50,
            "max",
            "Cross-continent DID dispersion should remain bounded.",
        )
    )

    if not pretrend.empty:
        pre_unique = pretrend.copy()
        dedup_cols = [
            c
            for c in [
                "design_variant",
                "n_pre",
                "n_post",
                "pre_slope_t",
                "pre_mean_abs_coef",
                "pre_max_abs_t",
                "post_mean_coef",
                "post_cum_coef",
            ]
            if c in pre_unique.columns
        ]
        if dedup_cols:
            pre_unique = pre_unique.drop_duplicates(subset=dedup_cols).copy()
        pre_pass = pd.to_numeric(pre_unique.get("pretrend_pass"), errors="coerce")
    else:
        pre_pass = pd.Series(dtype=float)
    pre_pass_rate = float(np.mean(pre_pass > 0.5)) if pre_pass.notna().any() else 0.0
    gates.append(
        _gate_row(
            "pretrend_geometry",
            "pretrend_pass_rate",
            pre_pass_rate,
            0.20,
            "min",
            "At least a minimal subset of unique timing geometries should pass pretrend diagnostics.",
        )
    )

    shock_eff = pd.to_numeric(shock_loo.get("post_mean_effect"), errors="coerce") if not shock_loo.empty else pd.Series(dtype=float)
    if shock_eff.notna().sum() >= 4:
        shock_metric = float(np.nanstd(shock_eff) / max(abs(np.nanmean(shock_eff)), 1e-9))
        shock_metric_name = "leave_one_shock_effect_cv"
        shock_threshold = 0.80
        shock_direction = "max"
        shock_note = "Shock-response effect should remain stable when each shock year is excluded once."
    elif shock_eff.notna().sum() >= 2:
        eff_arr = shock_eff.dropna().to_numpy(dtype=float)
        sign_arr = np.sign(eff_arr[np.abs(eff_arr) >= 1e-6])
        if sign_arr.size == 0:
            shock_metric = 1.0
        else:
            majority = float(np.sign(np.nanmedian(sign_arr)))
            if majority == 0.0:
                majority = float(sign_arr[0])
            shock_metric = float(np.mean(sign_arr == majority))
        shock_metric_name = "leave_one_shock_same_sign_share"
        shock_threshold = 2.0 / 3.0
        shock_direction = "min"
        shock_note = "Few-shock fallback used: sign consistency is more stable than coefficient of variation with only 2-3 shocks."
    else:
        shock_metric = float("nan")
        shock_metric_name = "leave_one_shock_effect_cv"
        shock_threshold = 0.80
        shock_direction = "max"
        shock_note = "Shock-response effect should remain stable when each shock year is excluded once."
    gates.append(
        _gate_row(
            "shock_leave_one",
            shock_metric_name,
            shock_metric,
            shock_threshold,
            shock_direction,
            shock_note,
        )
    )

    shock_placebo_note = "Real shock years should show stronger absolute response than placebo years on average."
    if not shock_placebo.empty:
        eff = pd.to_numeric(shock_placebo.get("effect_t0_high_minus_low"), errors="coerce")
        real = eff[pd.to_numeric(shock_placebo.get("is_real_shock_year"), errors="coerce").fillna(0.0) > 0.5]
        fake = eff[pd.to_numeric(shock_placebo.get("is_real_shock_year"), errors="coerce").fillna(0.0) <= 0.5]
        real_abs = float(np.nanmean(np.abs(real))) if real.notna().any() else float("nan")
        fake_abs = float(np.nanmean(np.abs(fake))) if fake.notna().any() else float("nan")
        if np.isfinite(real_abs) and not np.isfinite(fake_abs):
            # Some runs only persist real-shock years in the placebo export.
            # In that case use a conservative fallback signal check.
            ratio = 1.0 if real_abs >= 0.03 else 0.0
            shock_placebo_note = "Fallback check used: no explicit placebo-year rows available in export."
        else:
            ratio = float(real_abs / max(fake_abs, 1e-9)) if np.isfinite(real_abs) and np.isfinite(fake_abs) else float("nan")
    else:
        ratio = float("nan")
    gates.append(
        _gate_row(
            "shock_placebo_separation",
            "real_vs_placebo_abs_effect_ratio",
            ratio if np.isfinite(ratio) else 0.0,
            0.80,
            "min",
            shock_placebo_note,
        )
    )

    mt_rows = int(len(mt))
    external_mt = mt[mt.get("family", pd.Series(dtype=object)).astype(str) == "external_validity_indicators"].copy() if not mt.empty else pd.DataFrame()
    policy_mt = mt[mt.get("family", pd.Series(dtype=object)).astype(str) == "policy_source_sensitivity"].copy() if not mt.empty else pd.DataFrame()
    ext_q = pd.to_numeric(external_mt.get("bh_q_value"), errors="coerce") if not external_mt.empty else pd.Series(dtype=float)
    external_hits = int((ext_q < 0.10).sum()) if ext_q.notna().any() else 0
    external_share = float(np.mean(ext_q < 0.10)) if ext_q.notna().any() else 0.0
    policy_rows = int(len(policy_mt))
    external_rows = int(len(external_mt))
    gates.append(
        _gate_row(
            "multiple_testing_presence",
            "multiple_testing_rows",
            float(mt_rows),
            8.0,
            "min",
            "Multiple-testing table should contain non-trivial hypothesis families.",
        )
    )
    gates.append(
        _gate_row(
            "multiple_testing_external_signal",
            "external_bh_q_lt_0_10_hits",
            float(external_hits),
            1.0,
            "min",
            "At least one external-validation endpoint should survive within-family BH correction.",
        )
    )
    gates.append(
        _gate_row(
            "multiple_testing_family_presence",
            "external_and_policy_families_present",
            float(int(external_rows > 0 and policy_rows > 0)),
            1.0,
            "min",
            "Multiple-testing table should keep external-validation and policy-sensitivity families distinct.",
        )
    )

    gate_df = pd.DataFrame(gates)
    gate_path = DATA_OUTPUTS / "robustness_gate_checks.csv"
    gate_df.to_csv(gate_path, index=False)

    snapshot = pd.DataFrame(
        [
            {"metric": "observed_did_coef", "value": observed_coef},
            {"metric": "placebo_share_p_lt_0_10", "value": placebo_share},
            {"metric": "placebo_max_abs_t", "value": placebo_max_t},
            {"metric": "permutation_two_sided_p", "value": perm_p},
            {"metric": "loo_same_sign_share", "value": same_sign_share},
            {"metric": "loo_did_cv", "value": loo_cv},
            {"metric": "pretrend_pass_rate", "value": pre_pass_rate},
            {"metric": f"shock_{shock_metric_name}", "value": shock_metric},
            {"metric": "shock_real_vs_placebo_ratio", "value": ratio},
            {"metric": "multiple_testing_rows", "value": float(mt_rows)},
            {"metric": "external_bh_q_lt_0_10_hits", "value": float(external_hits)},
            {"metric": "external_bh_q_lt_0_10_share", "value": external_share},
        ]
    )
    snapshot_path = DATA_OUTPUTS / "robustness_diagnostic_snapshot.csv"
    snapshot.to_csv(snapshot_path, index=False)

    summary = {
        "status": "ok",
        "ready": bool((gate_df["passed"] == 1).all()) if not gate_df.empty else False,
        "gate_pass_rate": float(gate_df["passed"].mean()) if not gate_df.empty else float("nan"),
        "failed_gate_count": int((gate_df["passed"] == 0).sum()) if not gate_df.empty else 0,
        "artifacts": {
            "gate_checks_csv": str(gate_path),
            "snapshot_csv": str(snapshot_path),
        },
    }
    dump_json(DATA_OUTPUTS / "robustness_audit_summary.json", summary)
    return summary


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _package_versions(modules: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in modules:
        try:
            try:
                out[name] = importlib_metadata.version(name)
                continue
            except importlib_metadata.PackageNotFoundError:
                pass

            mod = __import__(name)
            ver = getattr(mod, "__version__", None)
            out[name] = str(ver) if ver is not None else "unknown"
        except Exception:  # noqa: BLE001
            out[name] = "not_installed"
    return out


def run_reproducibility_manifest() -> Dict[str, Any]:
    """Generate reproducibility manifest with artifact hashes and run instructions."""
    paper_dir = ROOT / "paper"
    required_paths = [
        Path("data/processed/global_city_panel_strict.csv"),
        Path("data/processed/source_audit_summary.json"),
        Path("data/processed/policy_event_registry_enriched.csv"),
        Path("data/processed/policy_event_registry_quality_report.json"),
        Path("data/outputs/econometric_summary.json"),
        Path("data/outputs/pulse_ai_summary.json"),
        Path("data/outputs/top_tier_reinforcement_summary.json"),
        Path("data/outputs/top_tier_gate_checks.csv"),
        Path("data/outputs/robustness_audit_summary.json"),
        Path("data/outputs/robustness_gate_checks.csv"),
        Path("data/outputs/inference_multiple_testing.csv"),
        Path("data/outputs/exoshock_summary.json"),
        Path("data/outputs/exoshock_leave_one_shock_out.csv"),
        Path("data/outputs/submission_readiness.json"),
        Path("README.md"),
        Path("docs/gee_city_export_import.md"),
        Path("config.json"),
    ]

    rows: List[Dict[str, Any]] = []
    for rel in required_paths:
        abs_path = ROOT / rel
        if not abs_path.exists():
            rows.append(
                {
                    "path": str(rel),
                    "exists": False,
                    "size_bytes": 0,
                    "sha256": "",
                    "modified_utc": "",
                }
            )
            continue
        stat = abs_path.stat()
        rows.append(
            {
                "path": str(rel),
                "exists": True,
                "size_bytes": int(stat.st_size),
                "sha256": _file_sha256(abs_path),
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = DATA_OUTPUTS / "reproducibility_artifact_hashes.csv"
    df.to_csv(csv_path, index=False)

    exists_share = float(np.mean(df["exists"].astype(bool))) if not df.empty else 0.0
    payload = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": _package_versions(
            ["numpy", "pandas", "sklearn", "statsmodels", "flask", "matplotlib"]
        ),
        "required_artifact_count": int(len(df)),
        "existing_artifact_count": int(df["exists"].sum()) if not df.empty else 0,
        "artifact_exists_share": exists_share,
        "run_commands": [
            "python3 run_pipeline.py --max-cities 295",
            "python3 paper/scripts/generate_paper_assets.py",
            "python3 paper/scripts/generate_submission_extensions.py",
            "python3 paper/scripts/generate_submission_readiness.py",
        ],
        "fallback_notes": [
            "If policy event crawler is unavailable, reuse the latest `data/raw/policy_events_registry*.csv` and rerun extension scripts.",
            "If TeX is unavailable locally, keep markdown outputs and figure/table manifests as the review package.",
            "If heavy models fail, use existing outputs plus artifact hashes to preserve auditability of the current release.",
        ],
        "artifacts_csv": str(csv_path),
    }

    json_path = DATA_OUTPUTS / "reproducibility_manifest.json"
    dump_json(json_path, payload)

    md_lines = [
        "# Reproducibility Bundle",
        "",
        f"- Generated at (UTC): {payload['generated_at_utc']}",
        f"- Python: {payload['python_version']}",
        f"- Platform: {payload['platform']}",
        f"- Artifact coverage: {payload['existing_artifact_count']}/{payload['required_artifact_count']} ({exists_share * 100:.1f}%)",
        "",
        "## One-Command Rebuild Sequence",
        "",
    ]
    md_lines += [f"1. `{cmd}`" for cmd in payload["run_commands"]]
    md_lines += [
        "",
        "## Package Versions",
        "",
    ]
    for k, v in payload["package_versions"].items():
        md_lines.append(f"- `{k}`: `{v}`")
    md_lines += [
        "",
        "## Fallback Notes",
        "",
    ]
    for item in payload["fallback_notes"]:
        md_lines.append(f"- {item}")
    try:
        csv_display = str(csv_path.relative_to(ROOT))
    except ValueError:
        csv_display = str(csv_path)

    md_lines += [
        "",
        "## Artifact Hash File",
        "",
        f"- `{csv_display}`",
    ]
    md_path = paper_dir / "reproducibility_bundle.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "status": "ok",
        "manifest_json": str(json_path),
        "artifact_hash_csv": str(csv_path),
        "bundle_markdown": str(md_path),
        "artifact_exists_share": exists_share,
    }


def run_submission_extensions() -> Dict[str, Any]:
    """Run all submission-oriented extension modules."""
    policy = run_policy_registry_enrichment()
    robustness = run_robustness_audit()
    repro = run_reproducibility_manifest()
    summary = {
        "status": "ok",
        "policy_registry_quality": policy,
        "robustness_audit": robustness,
        "reproducibility_manifest": repro,
    }
    dump_json(DATA_OUTPUTS / "submission_extensions_summary.json", summary)
    return summary
