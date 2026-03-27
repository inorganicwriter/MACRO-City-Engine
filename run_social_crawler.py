from __future__ import annotations

"""CLI entry for crawling city-level social sentiment signals."""

import argparse
import json
import logging

from src.social_sentiment import build_city_social_sentiment_yearly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl social sentiment proxies for MACRO-City Engine.")
    parser.add_argument("--max-cities", type=int, default=295, help="Maximum cities sampled from city catalog.")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year for crawling.")
    parser.add_argument("--end-year", type=int, default=2025, help="End year for crawling.")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache usage.")
    parser.add_argument("--max-records", type=int, default=80, help="Max records per city-year query.")
    parser.add_argument("--timeout", type=int, default=25, help="HTTP timeout (seconds) per query.")
    parser.add_argument("--sleep", type=float, default=0.15, help="Sleep interval between queries (seconds).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    summary = build_city_social_sentiment_yearly(
        max_cities=int(args.max_cities),
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        use_cache=not bool(args.no_cache),
        max_records=int(args.max_records),
        request_timeout=int(args.timeout),
        sleep_seconds=float(args.sleep),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
