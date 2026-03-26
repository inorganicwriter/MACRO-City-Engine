from __future__ import annotations

"""Materialize a classified policy registry for downstream causal analysis."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.policy_taxonomy import classify_policy_record
from src.utils import DATA_PROCESSED, DATA_RAW, dump_json


def build_policy_event_taxonomy(
    src_path: Path | None = None,
    out_path: Path | None = None,
) -> dict[str, object]:
    src_path = src_path or (DATA_RAW / "policy_events_registry.csv")
    out_path = out_path or (DATA_RAW / "policy_events_registry_classified.csv")
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    raw = pd.read_csv(src_path)
    if raw.empty:
        raw.to_csv(out_path, index=False)
        summary = {"status": "empty", "source_path": str(src_path), "output_path": str(out_path), "event_rows": 0}
        dump_json(DATA_PROCESSED / "policy_event_taxonomy_summary.json", summary)
        return summary

    classified = raw.copy()
    meta = classified.apply(
        lambda row: classify_policy_record(row.get("policy_name"), row.get("source_ref")),
        axis=1,
        result_type="expand",
    )
    for col in meta.columns:
        classified[col] = meta[col]

    classified.to_csv(out_path, index=False)

    subtype_counts = (
        classified.groupby(["policy_type_coarse", "policy_subtype"], as_index=False)
        .size()
        .rename(columns={"size": "event_rows"})
        .sort_values(["event_rows", "policy_type_coarse", "policy_subtype"], ascending=[False, True, True])
    )
    subtype_counts.to_csv(DATA_PROCESSED / "policy_event_taxonomy_counts.csv", index=False)

    summary = {
        "status": "ok",
        "source_path": str(src_path),
        "output_path": str(out_path),
        "event_rows": int(len(classified)),
        "policy_types": int(classified["policy_name"].astype(str).nunique()),
        "coarse_types": classified["policy_type_coarse"].astype(str).value_counts().to_dict(),
        "subtypes": subtype_counts.head(20).to_dict(orient="records"),
        "evidence_tracks": classified["policy_evidence_track"].astype(str).value_counts().to_dict(),
        "direct_core_evidence_eligible_rows": int(
            pd.to_numeric(classified["policy_direct_core_evidence_eligible"], errors="coerce").fillna(0).astype(int).sum()
        ),
    }
    dump_json(DATA_PROCESSED / "policy_event_taxonomy_summary.json", summary)
    return summary


if __name__ == "__main__":
    result = build_policy_event_taxonomy()
    print(result)
