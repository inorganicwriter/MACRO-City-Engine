from __future__ import annotations

"""CLI for generating realtime monitor snapshots without full pipeline rerun."""

import argparse
import logging
import time

from src.pipeline import setup_logging
from src.realtime_monitor import generate_realtime_monitor_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate realtime MACRO-City Engine monitor snapshots.")
    parser.add_argument(
        "--loop-seconds",
        type=int,
        default=0,
        help="If >0, run in loop with the given interval (seconds).",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=0,
        help="Optional loop cap when --loop-seconds > 0; 0 means infinite.",
    )
    parser.add_argument(
        "--trigger",
        type=str,
        default="cli_manual",
        help="Trigger label written into realtime_status.json.",
    )
    return parser.parse_args()


def _run_once(trigger: str) -> dict[str, object]:
    status = generate_realtime_monitor_snapshot(trigger=trigger)
    print(
        {
            "status": status.get("status"),
            "generated_at_utc": status.get("generated_at_utc"),
            "latest_data_year": status.get("latest_data_year"),
            "city_count": status.get("city_count"),
            "country_count": status.get("country_count"),
            "alert_city_count": status.get("alert_city_count"),
            "warning_city_count": status.get("warning_city_count"),
            "sentinel_city_count": status.get("sentinel_city_count"),
            "source_signature_hash": status.get("source_signature_hash"),
        }
    )
    return status


def main() -> None:
    args = parse_args()
    setup_logging(logging.INFO)

    loop_seconds = int(max(0, args.loop_seconds))
    max_loops = int(max(0, args.max_loops))
    trigger = str(args.trigger).strip() or "cli_manual"

    if loop_seconds <= 0:
        _run_once(trigger)
        return

    n = 0
    while True:
        _run_once(f"{trigger}_loop")
        n += 1
        if max_loops > 0 and n >= max_loops:
            break
        time.sleep(loop_seconds)


if __name__ == "__main__":
    main()
