from __future__ import annotations

import argparse
import json
import logging

from src.historical_viirs import import_historical_viirs_to_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import historical EOG VIIRS GeoTIFF files into viirs_city_monthly.csv.")
    parser.add_argument("--source-root", type=str, required=True, help="Directory containing EOG monthly/annual VIIRS GeoTIFF files.")
    parser.add_argument("--max-cities", type=int, default=295)
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--no-merge-existing", action="store_true", help="Overwrite target monthly panel instead of merging with existing rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    summary = import_historical_viirs_to_raw(
        source_root=args.source_root,
        max_cities=int(args.max_cities),
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        merge_existing=not bool(args.no_merge_existing),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
