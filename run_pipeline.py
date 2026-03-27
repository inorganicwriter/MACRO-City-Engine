from __future__ import annotations

"""CLI entry for running full pipeline."""

import argparse
import logging
import os
import warnings

from src.pipeline import run_pipeline, setup_logging


def _configure_runtime() -> None:
    """Set stable runtime defaults for long full-pipeline runs."""
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    warnings.filterwarnings(
        "ignore",
        message=(
            r".*`?sklearn\.utils\.parallel\.delayed`?\s+should be used with\s+"
            r"`?sklearn\.utils\.parallel\.Parallel`?.*"
        ),
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*X has feature names, but .* was fitted without feature names.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*X does not have valid feature names, but .* was fitted with feature names.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*KMeans is known to have a memory leak on Windows with MKL.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Degrees of freedom <= 0 for slice.*",
        category=RuntimeWarning,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MACRO-City Engine global pipeline.")
    parser.add_argument("--max-cities", type=int, default=None, help="Maximum number of cities to include.")
    parser.add_argument(
        "--require-policy-events",
        action="store_true",
        help="Require auditable policy event registry (data/raw/policy_events_registry.csv).",
    )
    parser.add_argument(
        "--auto-build-policy-events",
        action="store_true",
        help="Auto-crawl policy events from objective APIs when registry is missing.",
    )
    parser.add_argument(
        "--enable-augmented-policy-events",
        action="store_true",
        help=(
            "Enable objective-indicator/macro/AI event augmentation. "
            "Keep disabled for main causal estimates; use only for sensitivity analysis."
        ),
    )
    parser.add_argument(
        "--enable-city-macro-disaggregation",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Country-to-city macro disaggregation is now enabled by default."
        ),
    )
    parser.add_argument(
        "--disable-city-macro-disaggregation",
        action="store_true",
        help="Disable physical-share macro downscaling and keep country-year macro controls only.",
    )
    parser.add_argument(
        "--use-legacy-macro-mixed-spec",
        action="store_true",
        help=(
            "Use the legacy macro-mixed composite construction. "
            "Default uses the city-observed primary composite specification."
        ),
    )
    parser.add_argument(
        "--normalize-within-year",
        action="store_true",
        help="Use within-year MinMax normalization instead of the default global-panel normalization.",
    )
    parser.add_argument(
        "--use-manual-composite",
        action="store_true",
        help="Use the manual weighted composite as the main target instead of PCA-first composite.",
    )
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Allow synthetic fallback when external data sources fail.",
    )
    parser.add_argument(
        "--allow-non-verified-sources",
        action="store_true",
        help="Relax strict filtering to allow objectively imputed POI rows (not recommended for final research runs).",
    )
    parser.add_argument(
        "--min-verified-city-retention",
        type=float,
        default=0.55,
        help="Minimum retained-city ratio under verified-source filtering in strict mode.",
    )
    parser.add_argument(
        "--min-external-direct-share",
        type=float,
        default=0.70,
        help="Minimum ISO3 share for external-direct policy events in strict mode.",
    )
    parser.add_argument(
        "--max-ai-inferred-share",
        type=float,
        default=0.30,
        help="Maximum ISO3 share for AI-inferred policy events in strict mode.",
    )
    parser.add_argument(
        "--econometrics-fast",
        action="store_true",
        help="Use lightweight econometric mode to avoid the long Step 5 full sweep.",
    )
    return parser.parse_args()


def main() -> None:
    _configure_runtime()
    args = parse_args()
    setup_logging(logging.INFO)
    summary = run_pipeline(
        max_cities=args.max_cities,
        strict_real_data=not args.allow_synthetic_fallback,
        require_policy_events=bool(args.require_policy_events),
        auto_build_policy_events=bool(args.auto_build_policy_events or args.require_policy_events),
        augment_policy_events_for_sensitivity=bool(args.enable_augmented_policy_events),
        enable_city_macro_disaggregation=bool(args.enable_city_macro_disaggregation or (not args.disable_city_macro_disaggregation)),
        use_city_observed_primary_spec=not bool(args.use_legacy_macro_mixed_spec),
        normalize_within_year=bool(args.normalize_within_year),
        prefer_pca_composite=not bool(args.use_manual_composite),
        enforce_verified_sources=not bool(args.allow_non_verified_sources),
        min_verified_city_retention=float(args.min_verified_city_retention),
        min_external_direct_share=float(args.min_external_direct_share),
        max_ai_inferred_share=float(args.max_ai_inferred_share),
        econometrics_fast=bool(args.econometrics_fast),
    )
    print("Pipeline finished.")
    print(summary)


if __name__ == "__main__":
    main()
