from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="https://dbr.abs.gov.au/compare.html?lyr=gccsa&rgn0=1GSYD",
    )
    parser.add_argument("--wait-seconds", type=float, default=8.0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="abs-probe-") as tmpdir:
        options = Options()
        options.use_chromium = True
        options.add_argument(f"--user-data-dir={tmpdir}")
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-sync")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1400,1000")

        driver = webdriver.Edge(options=options)
        try:
            driver.get(args.url)
            time.sleep(args.wait_seconds)
            anchors = driver.find_elements(By.TAG_NAME, "a")
            payload = []
            for a in anchors:
                anchor_id = a.get_attribute("id") or ""
                href = a.get_attribute("href") or ""
                text = (a.text or "").strip()
                title = a.get_attribute("title") or ""
                if anchor_id.startswith("dload") or "Download" in text or "download" in href.lower():
                    payload.append(
                        {
                            "id": anchor_id,
                            "text": text,
                            "title": title,
                            "href": href,
                            "onclick": a.get_attribute("onclick") or "",
                            "outer_html": a.get_attribute("outerHTML") or "",
                        }
                    )
            scripts = [
                script.get_attribute("src") or ""
                for script in driver.find_elements(By.TAG_NAME, "script")
            ]
            print(
                json.dumps(
                    {"url": driver.current_url, "downloads": payload, "scripts": scripts},
                    ensure_ascii=True,
                    indent=2,
                )
            )
        finally:
            driver.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
