from __future__ import annotations

import argparse
import json
import logging

from src.gee_city_observed import (
    import_gee_ghsl_city_yearly,
    import_gee_no2_city_monthly,
    import_gee_viirs_city_monthly,
    prepare_gee_city_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and import Google Earth Engine city-observed data bundles.")
    parser.add_argument("--prepare-bundle", action="store_true", help="Write city points CSV and Earth Engine JS export scripts.")
    parser.add_argument("--import-viirs", type=str, default=None, help="Path to exported GEE VIIRS monthly CSV.")
    parser.add_argument("--import-ghsl", type=str, default=None, help="Path to exported GEE GHSL yearly CSV.")
    parser.add_argument("--import-no2", type=str, default=None, help="Path to exported GEE NO2 monthly CSV.")
    parser.add_argument("--max-cities", type=int, default=295)
    parser.add_argument("--buffer-m", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for generated bundle files.")
    parser.add_argument("--asset-id", type=str, default="users/your_username/gee_city_points")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--no-merge-existing", action="store_true", help="Overwrite existing raw CSVs instead of merging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    summary: dict[str, object] = {"status": "ok"}
    if bool(args.prepare_bundle):
        summary["prepare_bundle"] = prepare_gee_city_bundle(
            max_cities=int(args.max_cities),
            buffer_m=int(args.buffer_m),
            output_dir=args.output_dir,
            asset_id=str(args.asset_id),
            start_year=int(args.start_year),
            end_year=int(args.end_year),
        )

    if args.import_viirs:
        summary["import_viirs"] = import_gee_viirs_city_monthly(
            source_path=args.import_viirs,
            merge_existing=not bool(args.no_merge_existing),
            max_cities=int(args.max_cities),
        )

    if args.import_ghsl:
        summary["import_ghsl"] = import_gee_ghsl_city_yearly(
            source_path=args.import_ghsl,
            merge_existing=not bool(args.no_merge_existing),
            max_cities=int(args.max_cities),
        )

    if args.import_no2:
        summary["import_no2"] = import_gee_no2_city_monthly(
            source_path=args.import_no2,
            merge_existing=not bool(args.no_merge_existing),
            max_cities=int(args.max_cities),
        )

    if not bool(args.prepare_bundle) and not args.import_viirs and not args.import_ghsl and not args.import_no2:
        summary = {"status": "skipped", "reason": "no_action_requested"}

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
