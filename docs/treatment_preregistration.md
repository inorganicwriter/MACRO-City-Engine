# Treatment Pre-Registration: Primary Track Selection

## Primary Treatment Track

**`direct_core`** is designated as the pre-registered primary treatment track
for all DID analyses in this study.

## Selection Rationale

1. **External validity**: `direct_core` relies exclusively on externally
   verifiable policy events (trade agreements, infrastructure mandates,
   regulatory changes) sourced from official government publications and
   international organization databases.

2. **Temporal precision**: Treatment timing is pinned to the policy
   effective date rather than inferred from endogenous data patterns,
   avoiding reverse-causality concerns.

3. **Coverage-confidence trade-off**: `direct_core` covers fewer countries
   than the `all_sources_fallback` track but achieves higher confidence in
   treatment assignment, reducing measurement error in the treatment
   indicator.

4. **Pre-commitment**: This track was selected before examining DID
   results to avoid specification searching across the multiple treatment
   definitions constructed during data assembly.

## Robustness Tracks

The following tracks are reported as robustness checks only:

| Track | Description | Role |
|-------|-------------|------|
| `direct` | Broader direct policy events | Sensitivity to inclusion criteria |
| `evidence_a` | Grade-A evidence only | Stricter external validity |
| `intense_external_direct` | High-intensity subset | Dose-response check |
| `intense_external_peak` | Peak-intensity subset | Extreme-contrast check |
| `all_sources_fallback` | Union of all sources | Maximum coverage bound |

## Implementation

- `run_did_two_way_fe()` defaults to `did_treatment_direct_core`
- `PRIMARY_TREATMENT_TRACK = "direct_core"` constant in `global_data.py`
- All other tracks are estimated but clearly labeled as robustness
