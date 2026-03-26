from __future__ import annotations

"""CLI entry for unified real-data crawler."""

import argparse
import json
import logging
import os

from src.data_crawler import crawl_global_real_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl real global data sources for Urban Pulse.")
    parser.add_argument("--max-cities", type=int, default=295, help="Maximum cities sampled from city catalog.")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year for crawling.")
    parser.add_argument("--end-year", type=int, default=2025, help="End year for crawling.")
    parser.add_argument("--min-commitment-usd", type=float, default=0.0, help="Minimum WB project commitment for policy events.")
    parser.add_argument("--non-strict", action="store_true", help="Allow non-strict crawling mode.")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache usage.")
    parser.add_argument("--skip-macro", action="store_true", help="Skip World Bank macro panel crawling.")
    parser.add_argument("--skip-extra-wb", action="store_true", help="Skip extra World Bank indicators crawling.")
    parser.add_argument(
        "--extra-wb-cache-only",
        action="store_true",
        help="Only read cached extra WB indicator files and do not request API.",
    )
    parser.add_argument("--skip-policy", action="store_true", help="Skip policy-event crawling.")
    parser.add_argument("--skip-weather", action="store_true", help="Skip weather crawling.")
    parser.add_argument("--skip-poi", action="store_true", help="Skip POI crawling.")
    parser.add_argument("--crawl-road", action="store_true", help="Crawl city road snapshot and build yearly road panel.")
    parser.add_argument("--crawl-viirs", action="store_true", help="Crawl NOAA VIIRS nightly samples and build city monthly panel.")
    parser.add_argument(
        "--historical-viirs-root",
        type=str,
        default=None,
        help="Local directory of EOG historical VIIRS GeoTIFF files to import and merge into viirs_city_monthly.csv.",
    )
    parser.add_argument("--gee-prepare-bundle", action="store_true", help="Prepare GEE city-points CSV and JS export scripts.")
    parser.add_argument("--gee-output-dir", type=str, default=None, help="Output directory for generated GEE bundle files.")
    parser.add_argument("--gee-asset-id", type=str, default="users/your_username/gee_city_points", help="Template Earth Engine asset ID used in generated JS scripts.")
    parser.add_argument("--gee-buffer-m", type=int, default=5000, help="City buffer radius for GEE export templates.")
    parser.add_argument("--gee-viirs-csv", type=str, default=None, help="Path to exported GEE VIIRS monthly CSV.")
    parser.add_argument("--gee-ghsl-csv", type=str, default=None, help="Path to exported GEE GHSL yearly CSV.")
    parser.add_argument("--crawl-osm-history", action="store_true", help="Crawl city-year OSM history signals from ohsome API.")
    parser.add_argument("--crawl-poi-yearly", action="store_true", help="Crawl city-year POI category counts from ohsome API.")
    parser.add_argument(
        "--crawl-social-sentiment",
        action="store_true",
        help="Crawl social-media discourse proxy and build city-year sentiment panel.",
    )
    parser.add_argument(
        "--social-max-records",
        type=int,
        default=80,
        help="Max GDELT records fetched per city-year query when crawling social sentiment.",
    )
    parser.add_argument("--road-radius-m", type=int, default=2000, help="Road query radius in meters for Overpass snapshot.")
    parser.add_argument(
        "--strict-weather-circuit",
        type=int,
        default=None,
        help="Strict weather circuit-breaker threshold for consecutive failures.",
    )
    parser.add_argument(
        "--strict-skip-live-poi",
        action="store_true",
        help="Skip live OSM fetch in strict mode and use objective POI pool imputation path.",
    )
    parser.add_argument("--strict-poi-timeout", type=int, default=None, help="Strict POI request timeout (seconds).")
    parser.add_argument("--strict-poi-retries", type=int, default=None, help="Strict POI request retries per endpoint.")
    parser.add_argument("--strict-poi-backoff", type=float, default=None, help="Strict POI request retry backoff factor.")
    parser.add_argument("--strict-poi-sleep", type=float, default=None, help="Strict POI sleep between city requests (seconds).")
    parser.add_argument(
        "--strict-poi-max-consec-fail",
        type=int,
        default=None,
        help="Strict POI circuit-breaker threshold for consecutive failed cities.",
    )
    parser.add_argument(
        "--strict-poi-max-total-fail",
        type=int,
        default=None,
        help="Strict POI circuit-breaker threshold for total failed cities in one run.",
    )
    parser.add_argument(
        "--strict-poi-primary-two-endpoints",
        action="store_true",
        help="Use only first two Overpass endpoints in strict mode (default uses all mirrors).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.strict_weather_circuit is not None:
        os.environ["URBAN_PULSE_STRICT_WEATHER_CIRCUIT"] = str(int(args.strict_weather_circuit))
    if bool(args.strict_skip_live_poi):
        os.environ["URBAN_PULSE_STRICT_SKIP_LIVE_POI"] = "1"
    if args.strict_poi_timeout is not None:
        os.environ["URBAN_PULSE_STRICT_POI_TIMEOUT"] = str(int(args.strict_poi_timeout))
    if args.strict_poi_retries is not None:
        os.environ["URBAN_PULSE_STRICT_POI_RETRIES"] = str(int(args.strict_poi_retries))
    if args.strict_poi_backoff is not None:
        os.environ["URBAN_PULSE_STRICT_POI_BACKOFF"] = str(float(args.strict_poi_backoff))
    if args.strict_poi_sleep is not None:
        os.environ["URBAN_PULSE_STRICT_POI_SLEEP"] = str(float(args.strict_poi_sleep))
    if args.strict_poi_max_consec_fail is not None:
        os.environ["URBAN_PULSE_STRICT_POI_MAX_CONSEC_FAIL"] = str(int(args.strict_poi_max_consec_fail))
    if args.strict_poi_max_total_fail is not None:
        os.environ["URBAN_PULSE_STRICT_POI_MAX_TOTAL_FAIL"] = str(int(args.strict_poi_max_total_fail))
    if bool(args.strict_poi_primary_two_endpoints):
        os.environ["URBAN_PULSE_STRICT_POI_ALL_ENDPOINTS"] = "0"

    summary = crawl_global_real_sources(
        max_cities=int(args.max_cities),
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        strict_real_data=not bool(args.non_strict),
        use_cache=not bool(args.no_cache),
        policy_min_commitment_usd=float(args.min_commitment_usd),
        crawl_macro=not bool(args.skip_macro),
        crawl_extra_world_bank=not bool(args.skip_extra_wb),
        extra_world_bank_cache_only=bool(args.extra_wb_cache_only),
        crawl_policy_events=not bool(args.skip_policy),
        crawl_weather=not bool(args.skip_weather),
        crawl_poi=not bool(args.skip_poi),
        crawl_road=bool(args.crawl_road),
        crawl_viirs=bool(args.crawl_viirs),
        crawl_osm_history=bool(args.crawl_osm_history),
        crawl_poi_yearly=bool(args.crawl_poi_yearly),
        crawl_social_sentiment=bool(args.crawl_social_sentiment),
        social_max_records=int(args.social_max_records),
        road_radius_m=int(args.road_radius_m),
        historical_viirs_root=args.historical_viirs_root,
        gee_prepare_bundle=bool(args.gee_prepare_bundle),
        gee_output_dir=args.gee_output_dir,
        gee_asset_id=str(args.gee_asset_id),
        gee_buffer_m=int(args.gee_buffer_m),
        gee_viirs_csv=args.gee_viirs_csv,
        gee_ghsl_csv=args.gee_ghsl_csv,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
