from __future__ import annotations

"""Shared policy taxonomy helpers used across registry builders and panel logic."""

import re
from typing import Any


_DIGITAL_KEYWORDS = (
    "digital",
    "broadband",
    "internet",
    "fiber",
    "ict",
    "telecom",
    "5g",
    "data center",
    "e government",
    "innovation",
    "industry",
    "industrial",
    "manufacturing",
    "startup",
    "research",
    "technology",
)

_INFRA_MOBILITY_KEYWORDS = (
    "mobility",
    "transport",
    "metro",
    "rail",
    "road",
    "bus",
    "transit",
    "airport",
    "port",
    "logistics",
)

_INFRA_SERVICES_KEYWORDS = (
    "urban",
    "city",
    "housing",
    "water",
    "sanitation",
    "waste",
    "sewerage",
    "municipal",
    "service delivery",
)

_ECO_CLIMATE_KEYWORDS = (
    "climate",
    "adaptation",
    "resilience",
    "flood",
    "drought",
    "disaster",
    "stormwater",
    "coastal",
    "nature based",
)

_ECO_ENERGY_KEYWORDS = (
    "energy",
    "electricity",
    "power",
    "grid",
    "renewable",
    "solar",
    "wind",
    "electrification",
    "transmission",
    "distribution",
    "efficiency",
)

_ECO_GOVERNANCE_KEYWORDS = (
    "institution",
    "governance",
    "regulation",
    "regulatory",
    "fiscal",
    "public administration",
    "public sector",
    "capacity building",
    "decentralization",
    "municipal finance",
)

_EXACT_SUBTYPE_MAP = {
    "digital connectivity": "digital_connectivity",
    "digital connectivity ai inferred": "digital_connectivity",
    "digital connectivity objective indicator event": "digital_connectivity",
    "digital connectivity objective macro proxy": "digital_connectivity",
    "innovation industry": "digital_innovation_industry",
    "urban mobility": "infra_mobility_transport",
    "urban services": "infra_urban_services",
    "energy transition": "eco_reg_energy_transition",
    "institutional reform": "eco_reg_institutional_reform",
    "climate resilience": "eco_reg_climate_resilience",
}


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def policy_evidence_track(source_ref: Any) -> str:
    src = _normalize_text(source_ref)
    if "objective ai inferred" in src:
        return "ai_inferred"
    if "wb indicator event" in src:
        return "objective_indicator"
    if "objective macro rule" in src:
        return "objective_macro"
    return "external_direct"


def policy_subtype(policy_name: Any) -> str:
    name = _normalize_text(policy_name)
    if not name:
        return "other"
    exact = _EXACT_SUBTYPE_MAP.get(name)
    if exact is not None:
        return exact
    if any(tok in name for tok in _DIGITAL_KEYWORDS):
        if any(tok in name for tok in ("innovation", "industry", "industrial", "manufacturing", "startup", "research")):
            return "digital_innovation_industry"
        return "digital_connectivity"
    if any(tok in name for tok in _INFRA_MOBILITY_KEYWORDS):
        return "infra_mobility_transport"
    if any(tok in name for tok in _INFRA_SERVICES_KEYWORDS):
        return "infra_urban_services"
    if any(tok in name for tok in _ECO_CLIMATE_KEYWORDS):
        return "eco_reg_climate_resilience"
    if any(tok in name for tok in _ECO_ENERGY_KEYWORDS):
        return "eco_reg_energy_transition"
    if any(tok in name for tok in _ECO_GOVERNANCE_KEYWORDS):
        return "eco_reg_institutional_reform"
    return "other"


def policy_type_coarse(policy_name: Any) -> str:
    subtype = policy_subtype(policy_name)
    if subtype.startswith("digital_"):
        return "digital"
    if subtype.startswith("infra_"):
        return "infra"
    if subtype.startswith("eco_reg_"):
        return "eco_reg"
    return "other"


def policy_treatment_bucket(policy_name: Any) -> str:
    coarse = policy_type_coarse(policy_name)
    if coarse in {"infra", "digital", "eco_reg"}:
        return coarse
    return "other"


def policy_direct_core_evidence_eligible(source_ref: Any) -> int:
    return int(policy_evidence_track(source_ref) == "external_direct")


def classify_policy_record(policy_name: Any, source_ref: Any) -> dict[str, Any]:
    return {
        "policy_type_coarse": policy_type_coarse(policy_name),
        "policy_subtype": policy_subtype(policy_name),
        "policy_treatment_bucket": policy_treatment_bucket(policy_name),
        "policy_evidence_track": policy_evidence_track(source_ref),
        "policy_direct_core_evidence_eligible": policy_direct_core_evidence_eligible(source_ref),
    }
