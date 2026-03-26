from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _col_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch) - 64)
    return value


def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for si in root.findall("a:si", NS):
        parts = [node.text or "" for node in si.findall(".//a:t", NS)]
        values.append("".join(parts))
    return values


def _sheet_map(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels
    }
    out: list[tuple[str, str]] = []
    for sheet in wb.findall("a:sheets/a:sheet", NS):
        rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = rel_map[rid]
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        out.append((sheet.attrib["name"], target))
    return out


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    value_node = cell.find("a:v", NS)
    if value_node is None:
        texts = [node.text or "" for node in cell.findall(".//a:t", NS)]
        return "".join(texts).strip()
    raw = value_node.text or ""
    if cell.attrib.get("t") == "s":
        return shared[int(raw)] if raw.isdigit() and int(raw) < len(shared) else raw
    return raw.strip()


def _rows(zf: zipfile.ZipFile, sheet_path: str, shared: list[str]) -> list[list[str]]:
    root = ET.fromstring(zf.read(sheet_path))
    out: list[list[str]] = []
    for row in root.findall(".//a:sheetData/a:row", NS):
        values: dict[int, str] = {}
        for cell in row.findall("a:c", NS):
            ref = cell.attrib.get("r", "")
            if not ref:
                continue
            values[_col_index(ref)] = _cell_value(cell, shared)
        if not values:
            continue
        width = max(values)
        out.append([values.get(i, "") for i in range(1, width + 1)])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xlsx_path", type=Path)
    parser.add_argument("--sheet", default="")
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--max-rows", type=int, default=200)
    args = parser.parse_args()

    with zipfile.ZipFile(args.xlsx_path) as zf:
        shared = _shared_strings(zf)
        sheets = _sheet_map(zf)
        print("sheets=", [name for name, _ in sheets])
        for sheet_name, sheet_path in sheets:
            if args.sheet and args.sheet != sheet_name:
                continue
            print(f"\n--- sheet {sheet_name} ---")
            shown = 0
            patterns = [p.lower() for p in args.pattern]
            for row in _rows(zf, sheet_path, shared):
                line = " | ".join(value.strip() for value in row if value.strip())
                if not line:
                    continue
                if patterns and not any(pattern in line.lower() for pattern in patterns):
                    continue
                print(line)
                shown += 1
                if shown >= args.max_rows:
                    break


if __name__ == "__main__":
    main()
