from __future__ import annotations

"""Objective policy-event registry builder from World Bank public APIs."""

import logging
import math
import re
import time
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import requests
from requests import RequestException

from .policy_taxonomy import classify_policy_record, policy_type_coarse as shared_policy_type_coarse
from .utils import DATA_RAW, dump_json

LOGGER = logging.getLogger(__name__)

PROJECTS_API_URL = "https://search.worldbank.org/api/v2/projects"
COUNTRY_API_URL = "https://api.worldbank.org/v2/country"

POLICY_KEYWORDS: Dict[str, List[str]] = {
    "digital_connectivity": [
        "digital",
        "broadband",
        "internet",
        "fiber",
        "ict",
        "telecom",
        "e-government",
        "data center",
        "5g",
        "mobile network",
    ],
    "urban_mobility": [
        "transport",
        "metro",
        "rail",
        "bus",
        "mobility",
        "logistics",
        "port",
        "airport",
        "road",
        "transit",
    ],
    "urban_services": [
        "urban",
        "city",
        "housing",
        "water",
        "sanitation",
        "waste",
        "sewerage",
        "district",
        "municipal",
        "resilience",
    ],
    "innovation_industry": [
        "innovation",
        "research",
        "technology",
        "industrial",
        "manufacturing",
        "special economic zone",
        "enterprise",
        "startup",
        "skills",
    ],
    "energy_transition": [
        "energy",
        "electricity",
        "power",
        "grid",
        "renewable",
        "solar",
        "wind",
        "efficiency",
        "electrification",
        "transmission",
        "distribution",
    ],
    "institutional_reform": [
        "governance",
        "institutional",
        "public administration",
        "public sector",
        "fiscal",
        "regulatory",
        "capacity building",
        "service delivery",
        "decentralization",
        "municipal finance",
    ],
    "climate_resilience": [
        "climate",
        "adaptation",
        "resilience",
        "flood",
        "drought",
        "disaster risk",
        "coastal",
        "stormwater",
        "nature-based",
    ],
}


def _request_json(url: str, params: dict[str, Any], timeout: int = 60, retries: int = 4) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code in {429, 500, 502, 503, 504}:
                resp.raise_for_status()
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                msg = "unexpected_payload_type"
                raise RuntimeError(msg)
            return payload
        except (RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(0.8 * (2 ** (attempt - 1)))
    msg = f"request_failed_after_retries: {last_error}"
    raise RuntimeError(msg)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_as_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(_as_text(v) for v in value.values())
    return str(value)


def _extract_year(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    m = re.search(r"(19|20)\d{2}", text)
    if not m:
        return None
    year = int(m.group(0))
    if year < 1980 or year > 2100:
        return None
    return year


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:  # noqa: BLE001
        return 0.0


def _extract_projects(payload: dict[str, Any]) -> List[dict[str, Any]]:
    rows = payload.get("projects", {})
    if isinstance(rows, dict):
        return [v for v in rows.values() if isinstance(v, dict)]
    if isinstance(rows, list):
        return [v for v in rows if isinstance(v, dict)]
    return []


def _extract_total(payload: dict[str, Any]) -> int | None:
    raw = payload.get("total")
    if raw is None:
        return None
    try:
        val = int(str(raw).strip())
    except Exception:  # noqa: BLE001
        return None
    if val <= 0:
        return None
    return val


def _load_country_metadata() -> pd.DataFrame:
    cache_path = DATA_RAW / "wb_country_code_map.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        required = {"iso2", "iso3", "country_name", "region_name"}
        if required.issubset(cached.columns):
            return cached.copy()

    payload = requests.get(COUNTRY_API_URL, params={"format": "json", "per_page": 400}, timeout=60).json()
    if not isinstance(payload, list) or len(payload) < 2:
        msg = "failed_to_load_country_code_map"
        raise RuntimeError(msg)
    rows: List[dict[str, str]] = []
    for item in payload[1]:
        iso3 = str(item.get("id") or "").upper().strip()
        iso2 = str(item.get("iso2Code") or "").upper().strip()
        if len(iso3) != 3 or len(iso2) != 2 or iso2 == "NA":
            continue
        rows.append(
            {
                "iso2": iso2,
                "iso3": iso3,
                "country_name": str(item.get("name") or "").strip(),
                "region_name": str(item.get("region", {}).get("value") or "").strip(),
            }
        )
    out = pd.DataFrame(rows).drop_duplicates(subset=["iso2"], keep="first")
    out.to_csv(cache_path, index=False)
    return out


def _load_iso2_to_iso3_map() -> dict[str, str]:
    meta = _load_country_metadata()
    return {
        str(r["iso2"]).upper(): str(r["iso3"]).upper()
        for _, r in meta.iterrows()
        if str(r["iso2"]).strip() and str(r["iso3"]).strip()
    }


def _fetch_projects_global(rows_per_page: int = 500, max_pages: int = 80) -> List[dict[str, Any]]:
    out: List[dict[str, Any]] = []
    seen_ids: set[str] = set()
    total_expected: int | None = None
    for page in range(1, max_pages + 1):
        offset = (page - 1) * rows_per_page
        params = {
            "format": "json",
            "rows": rows_per_page,
            # search API relies on offset pagination (`os`) instead of page.
            "os": offset,
            "status_exact": "Active^Closed",
            "fl": "id,project_name,project_abstract,boardapprovaldate,closingdate,totalcommamt,approvalfy,countrycode,countryname,countryshortname,sector1,theme1,mjtheme_namecode",
        }
        payload = _request_json(PROJECTS_API_URL, params=params, timeout=60, retries=4)
        if total_expected is None:
            total_expected = _extract_total(payload)
        projects = _extract_projects(payload)
        if not projects:
            break

        new_cnt = 0
        for p in projects:
            pid = str(p.get("id") or "").strip()
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            out.append(p)
            new_cnt += 1

        LOGGER.info(
            "Policy crawler offset page %s (os=%s): fetched=%s new=%s unique=%s total=%s",
            page,
            offset,
            len(projects),
            new_cnt,
            len(out),
            total_expected,
        )
        if new_cnt == 0:
            break
        if total_expected is not None and len(out) >= total_expected:
            break
        if len(projects) < rows_per_page:
            break
    return out


def _project_text(row: dict[str, Any]) -> str:
    parts = [
        _as_text(row.get("project_name")),
        _as_text(row.get("project_abstract")),
        _as_text(row.get("sector1")),
        _as_text(row.get("theme1")),
        _as_text(row.get("mjtheme_namecode")),
    ]
    return " ".join(p for p in parts if p).lower()


def _infer_policy_type(text: str) -> str | None:
    hits: List[tuple[str, int]] = []
    for policy_type, kws in POLICY_KEYWORDS.items():
        score = 0
        for kw in kws:
            if kw in text:
                score += 1
        if score > 0:
            hits.append((policy_type, score))
    if not hits:
        return None
    hits.sort(key=lambda x: x[1], reverse=True)
    return hits[0][0]


def _policy_type_coarse(policy_type: str) -> str:
    return shared_policy_type_coarse(policy_type)


def _extract_iso2_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        txt = raw.strip().upper()
        if not txt:
            return []
        if len(txt) == 2 and txt.isalpha():
            return [txt]
        # WB fields may contain multi-country values like "BD;IN;NP" or comma-separated names.
        tokens = re.findall(r"[A-Z]{2}", txt)
        out = [tok for tok in tokens if tok.isalpha()]
        return sorted(set(out))
    if isinstance(raw, list):
        out: List[str] = []
        for x in raw:
            out.extend(_extract_iso2_list(x))
        return sorted(set(out))
    if isinstance(raw, dict):
        out: List[str] = []
        for v in raw.values():
            out.extend(_extract_iso2_list(v))
        return sorted(set(out))
    return []


def _normalize_text_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _build_country_name_map(meta: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in meta.itertuples(index=False):
        iso3 = str(row.iso3).upper().strip()
        name = _normalize_text_label(str(row.country_name))
        if name and len(iso3) == 3:
            out[name] = iso3
    return out


def _infer_iso3_from_project_fields(
    project: dict[str, Any],
    *,
    iso3_set: set[str],
    iso2_to_iso3: dict[str, str],
    country_name_to_iso3: dict[str, str],
    iso3_to_region: dict[str, str],
) -> tuple[List[str], str]:
    """Infer ISO3 list when WB project countrycode is missing/ambiguous."""
    iso2_candidates: List[str] = []
    for field in ["countryname", "countryshortname", "borrower"]:
        iso2_candidates.extend(_extract_iso2_list(project.get(field)))
    mapped_iso3 = sorted({iso2_to_iso3.get(code, "") for code in iso2_candidates if iso2_to_iso3.get(code, "") in iso3_set})
    if mapped_iso3:
        return mapped_iso3, "explicit_iso2_from_text"

    text_parts = [
        _as_text(project.get("countryname")),
        _as_text(project.get("countryshortname")),
        _as_text(project.get("borrower")),
    ]
    text_norm = _normalize_text_label(" ".join(text_parts))
    tokens = [t for t in re.split(r"[;,/|]", text_norm) if t.strip()]
    exact_hits = sorted({country_name_to_iso3[t.strip()] for t in tokens if t.strip() in country_name_to_iso3 and country_name_to_iso3[t.strip()] in iso3_set})
    if exact_hits:
        return exact_hits, "explicit_country_name"

    region_rules = [
        ("western and central africa", lambda r: "africa" in r),
        ("eastern and southern africa", lambda r: "africa" in r),
        ("central africa", lambda r: "africa" in r),
        ("middle east and north africa", lambda r: "middle east" in r or "north africa" in r),
        ("europe and central asia", lambda r: "europe" in r or "central asia" in r),
        ("western balkans", lambda r: "europe" in r),
        ("central asia", lambda r: "central asia" in r),
        ("caribbean", lambda r: "latin america" in r or "caribbean" in r),
        ("organization of eastern caribbean states", lambda r: "latin america" in r or "caribbean" in r),
        ("pacific islands", lambda r: "east asia" in r or "pacific" in r),
        ("south east asia", lambda r: "east asia" in r or "south asia" in r),
    ]
    for phrase, matcher in region_rules:
        if phrase in text_norm:
            region_iso = sorted(
                [
                    iso3
                    for iso3 in iso3_set
                    if matcher(_normalize_text_label(iso3_to_region.get(iso3, "")))
                ]
            )
            if region_iso:
                return region_iso, f"regional_{phrase.replace(' ', '_')}"
    return [], "unresolved"


def _scale_group(values: pd.Series) -> pd.Series:
    arr = values.to_numpy(dtype=float)
    if len(arr) == 0:
        return pd.Series([], dtype=float, index=values.index)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if math.isclose(vmin, vmax):
        return pd.Series(np.ones_like(arr), index=values.index)
    return pd.Series((arr - vmin) / (vmax - vmin), index=values.index)


def build_policy_event_registry(
    iso3_codes: Iterable[str],
    *,
    start_year: int,
    end_year: int,
    min_commitment_usd: float = 1_000_000.0,
) -> dict[str, Any]:
    """Build `policy_events_registry.csv` from World Bank project metadata."""
    iso3_set = {str(c).upper().strip() for c in iso3_codes if str(c).strip()}
    if not iso3_set:
        return {"status": "skipped", "reason": "empty_iso3_set"}

    errors: List[dict[str, str]] = []
    try:
        meta = _load_country_metadata()
        iso2_to_iso3 = _load_iso2_to_iso3_map()
        country_name_to_iso3 = _build_country_name_map(meta)
        iso3_to_region = {
            str(r["iso3"]).upper(): str(r["region_name"])
            for _, r in meta.iterrows()
            if str(r["iso3"]).strip()
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "reason": f"country_code_map_error: {exc}"}

    try:
        projects = _fetch_projects_global(rows_per_page=500, max_pages=80)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "reason": f"project_api_error: {exc}"}

    rows: List[dict[str, Any]] = []
    for p in projects:
        text = _project_text(p)
        policy_type = _infer_policy_type(text)
        if policy_type is None:
            continue

        year = _extract_year(p.get("boardapprovaldate")) or _extract_year(p.get("approvalfy")) or _extract_year(
            p.get("closingdate")
        )
        if year is None or year < start_year or year > end_year:
            continue

        amount = _to_float(p.get("totalcommamt"))
        if amount < min_commitment_usd:
            continue

        pid = str(p.get("id") or "").strip()
        if not pid:
            continue

        iso2_list = _extract_iso2_list(p.get("countrycode"))
        assign_mode = "explicit_countrycode"
        iso3_list: List[str] = []
        if iso2_list:
            iso3_list = sorted(
                {
                    iso2_to_iso3.get(iso2, "")
                    for iso2 in iso2_list
                    if iso2_to_iso3.get(iso2, "") in iso3_set
                }
            )
        else:
            iso3_list, assign_mode = _infer_iso3_from_project_fields(
                p,
                iso3_set=iso3_set,
                iso2_to_iso3=iso2_to_iso3,
                country_name_to_iso3=country_name_to_iso3,
                iso3_to_region=iso3_to_region,
            )
            if not iso3_list:
                cn = str(p.get("countryname") or p.get("countryshortname") or "").strip()
                errors.append({"project_id": pid, "error": f"missing_countrycode:{cn}"})
                continue

        for iso3 in iso3_list:
            cls = classify_policy_record(policy_type, "wb_project")
            rows.append(
                {
                    "iso3": iso3,
                    "year": int(year),
                    "policy_name": policy_type,
                    "policy_type_coarse": cls["policy_type_coarse"],
                    "policy_subtype": cls["policy_subtype"],
                    "policy_treatment_bucket": cls["policy_treatment_bucket"],
                    "policy_evidence_track": cls["policy_evidence_track"],
                    "policy_direct_core_evidence_eligible": cls["policy_direct_core_evidence_eligible"],
                    "project_id": pid,
                    "commitment_usd": float(amount),
                    "assignment_type": assign_mode,
                }
            )

    cand = pd.DataFrame(rows)
    cand.to_csv(DATA_RAW / "policy_events_registry_candidates.csv", index=False)
    if cand.empty:
        error_projects = len({str(e.get("project_id")) for e in errors if str(e.get("project_id", "")).strip()})
        summary = {
            "status": "failed",
            "reason": "no_policy_events_detected",
            "countries_requested": len(iso3_set),
            "projects_scanned": int(len(projects)),
            "projects_with_country_parse_errors": int(error_projects),
            "errors": errors[:60],
        }
        dump_json(DATA_RAW / "policy_events_fetch_log.json", summary)
        return summary

    agg = (
        cand.groupby(["iso3", "year", "policy_name"], as_index=False)
        .agg(
            policy_type_coarse=("policy_type_coarse", "first"),
            policy_subtype=("policy_subtype", "first"),
            policy_treatment_bucket=("policy_treatment_bucket", "first"),
            policy_evidence_track=("policy_evidence_track", "first"),
            policy_direct_core_evidence_eligible=("policy_direct_core_evidence_eligible", "max"),
            commitment_usd=("commitment_usd", "sum"),
            project_count=("project_id", "nunique"),
            source_ref=("project_id", lambda s: "|".join(sorted(s.astype(str).unique().tolist())[:6])),
            assignment_type=(
                "assignment_type",
                lambda s: "regional" if np.any(pd.Series(s).astype(str).str.startswith("regional_")) else "explicit",
            ),
        )
        .copy()
    )
    agg["log_commitment"] = np.log1p(agg["commitment_usd"])
    agg["country_scaled_commitment"] = agg.groupby("iso3")["log_commitment"].transform(_scale_group)
    agg["country_scaled_projects"] = agg.groupby("iso3")["project_count"].transform(_scale_group)
    agg["policy_intensity"] = np.clip(
        0.75 * agg["country_scaled_commitment"] + 0.25 * agg["country_scaled_projects"], 0.05, 1.0
    )

    country_q = agg.groupby("iso3")["commitment_usd"].transform(lambda s: float(s.quantile(0.55)))
    filt = agg[(agg["commitment_usd"] >= country_q) | (agg["project_count"] >= 2)].copy()
    if filt.empty:
        filt = agg.copy()

    registry = filt[
        [
            "iso3",
            "year",
            "policy_intensity",
            "policy_name",
            "policy_type_coarse",
            "policy_subtype",
            "policy_treatment_bucket",
            "policy_evidence_track",
            "policy_direct_core_evidence_eligible",
            "source_ref",
            "commitment_usd",
            "project_count",
            "assignment_type",
        ]
    ].copy()
    registry["source_ref"] = np.where(
        registry["assignment_type"] == "regional",
        "wb_project_regional:" + registry["source_ref"].astype(str),
        "wb_project:" + registry["source_ref"].astype(str),
    )
    registry = registry.sort_values(["iso3", "year", "policy_name"]).reset_index(drop=True)
    registry = registry.rename(columns={"year": "start_year"})
    registry["end_year"] = registry["start_year"].astype(int)
    registry_out = registry[["iso3", "start_year", "end_year", "policy_intensity", "policy_name", "source_ref"]].copy()
    registry_classified = registry[
        [
            "iso3",
            "start_year",
            "end_year",
            "policy_intensity",
            "policy_name",
            "policy_type_coarse",
            "policy_subtype",
            "policy_treatment_bucket",
            "policy_evidence_track",
            "policy_direct_core_evidence_eligible",
            "source_ref",
            "commitment_usd",
            "project_count",
            "assignment_type",
        ]
    ].copy()

    registry_out.to_csv(DATA_RAW / "policy_events_registry.csv", index=False)
    registry_classified.to_csv(DATA_RAW / "policy_events_registry_classified.csv", index=False)
    registry.to_csv(DATA_RAW / "policy_events_registry_debug.csv", index=False)

    error_projects = len({str(e.get("project_id")) for e in errors if str(e.get("project_id", "")).strip()})
    summary = {
        "status": "ok",
        "countries_requested": int(len(iso3_set)),
        "countries_with_events": int(registry_out["iso3"].nunique()),
        "event_rows": int(len(registry_out)),
        "policy_types": int(registry_out["policy_name"].nunique()),
        "projects_scanned": int(len(projects)),
        "year_range": [int(registry_out["start_year"].min()), int(registry_out["start_year"].max())],
        "projects_with_country_parse_errors": int(error_projects),
        "errors": errors[:60],
    }
    dump_json(DATA_RAW / "policy_events_fetch_log.json", summary)
    return summary
