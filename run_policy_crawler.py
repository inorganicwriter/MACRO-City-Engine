from __future__ import annotations

"""CLI entry for auto-building auditable policy event registry."""

import argparse
import logging

from src.city_catalog import load_city_catalog
from src.policy_event_crawler import build_policy_event_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build policy_events_registry.csv from objective APIs.")
    parser.add_argument("--max-cities", type=int, default=295, help="Use sampled city catalog to determine ISO3 scope.")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year for project events.")
    parser.add_argument("--end-year", type=int, default=2025, help="End year for project events.")
    parser.add_argument(
        "--min-commitment-usd",
        type=float,
        default=500000.0,
        help="Minimum project commitment to include (USD).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cities = load_city_catalog(max_cities=args.max_cities)
    summary = build_policy_event_registry(
        cities["iso3"].astype(str).str.upper().unique().tolist(),
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        min_commitment_usd=float(args.min_commitment_usd),
    )
    print(summary)


if __name__ == "__main__":
    main()
