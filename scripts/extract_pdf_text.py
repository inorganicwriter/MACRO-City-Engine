from __future__ import annotations

import argparse
from pathlib import Path
import unicodedata

from pypdf import PdfReader


def _safe_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--max-pages", type=int, default=0)
    args = parser.parse_args()

    reader = PdfReader(str(args.pdf_path))
    total_pages = len(reader.pages)
    page_limit = args.max_pages if args.max_pages and args.max_pages > 0 else total_pages
    patterns = [p.lower() for p in args.pattern]

    print(f"pages={total_pages}")
    for page_no in range(min(total_pages, page_limit)):
        text = reader.pages[page_no].extract_text() or ""
        if not patterns:
            print(f"\n--- page {page_no + 1} ---\n{_safe_text(text[:4000])}")
            continue
        text_lower = text.lower()
        if any(pattern in text_lower for pattern in patterns):
            print(f"\n--- page {page_no + 1} ---")
            print(_safe_text(text[:8000]))


if __name__ == "__main__":
    main()
