from __future__ import annotations

"""Local importer for historical VIIRS night-light GeoTIFF products."""

import gzip
import logging
import math
import os
import re
import struct
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .city_catalog import load_city_catalog
from .utils import DATA_RAW, dump_json

LOGGER = logging.getLogger(__name__)

# EOG VIIRS rasters are very large BigTIFFs; PIL's decompression-bomb guard
# blocks legitimate reads unless we disable the pixel-count cap.
Image.MAX_IMAGE_PIXELS = None

_TIF_RE = re.compile(r"\.(tif|tiff)(\.gz)?$", re.IGNORECASE)
_DATE_RANGE_RE = re.compile(r"(?P<start>\d{8})-(?P<end>\d{8})")
_YEAR_MONTH_RE = re.compile(r"(?P<year>20\d{2})[-_](?P<month>\d{2})")
_YEAR_RE = re.compile(r"(?<!\d)(?P<year>20\d{2})(?!\d)")

_TAG_MODEL_PIXEL_SCALE = 33550
_TAG_MODEL_TIEPOINT = 33922
_TAG_IMAGE_WIDTH = 256
_TAG_IMAGE_LENGTH = 257
_TAG_BITS_PER_SAMPLE = 258
_TAG_COMPRESSION = 259
_TAG_STRIP_OFFSETS = 273
_TAG_SAMPLES_PER_PIXEL = 277
_TAG_ROWS_PER_STRIP = 278
_TAG_STRIP_BYTE_COUNTS = 279
_TAG_SAMPLE_FORMAT = 339

_RADIANCE_KEYS = ("avg_rade9", "avg_rade9h", "average")
_COVERAGE_KEYS = ("cf_cvg",)

_TIFF_TYPE_SIZES = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 1,
    7: 1,
    8: 2,
    9: 4,
    10: 8,
    11: 4,
    12: 8,
    16: 8,
    17: 8,
    18: 8,
}


def _sample_points(cities: pd.DataFrame, step_deg: float = 0.01) -> List[dict]:
    offsets = [(0.0, 0.0), (step_deg, 0.0), (-step_deg, 0.0), (0.0, step_deg), (0.0, -step_deg)]
    points: List[dict] = []
    for row in cities.to_dict(orient="records"):
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        for sample_id, (dlat, dlon) in enumerate(offsets):
            points.append(
                {
                    "city_id": str(row["city_id"]),
                    "city_name": str(row["city_name"]),
                    "sample_id": int(sample_id),
                    "lat": lat + dlat,
                    "lon": lon + dlon,
                }
            )
    return points


def _discover_viirs_files(root: Path) -> List[Path]:
    paths: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and _TIF_RE.search(path.name):
            paths.append(path)
    return sorted(paths)


def _band_key_from_name(name: str) -> str | None:
    low = name.lower()
    for key in _COVERAGE_KEYS:
        if key in low:
            return key
    for key in _RADIANCE_KEYS:
        if key in low:
            return key
    return None


def _parse_temporal_meta(path: Path) -> Dict[str, Any] | None:
    name = path.name
    low = name.lower()
    band_key = _band_key_from_name(name)
    if band_key is None:
        return None

    start_year: int | None = None
    month: int | None = None
    temporal = "unknown"

    m = _DATE_RANGE_RE.search(name)
    if m:
        start = m.group("start")
        end = m.group("end")
        start_year = int(start[:4])
        start_month = int(start[4:6])
        start_day = int(start[6:8])
        end_year = int(end[:4])
        end_month = int(end[4:6])
        end_day = int(end[6:8])
        start_ord = start_year * 372 + start_month * 31 + start_day
        end_ord = end_year * 372 + end_month * 31 + end_day
        span = end_ord - start_ord
        if span >= 300:
            temporal = "annual"
            month = None
        else:
            temporal = "monthly"
            month = start_month
    else:
        ym = _YEAR_MONTH_RE.search(name)
        if ym:
            start_year = int(ym.group("year"))
            month = int(ym.group("month"))
            temporal = "monthly"
        else:
            y = _YEAR_RE.search(name)
            if y:
                start_year = int(y.group("year"))
                temporal = "annual"

    if start_year is None:
        return None

    version = "unknown"
    if "v22" in low:
        version = "v22"
    elif "v21" in low:
        version = "v21"
    elif "v20" in low or "v2_" in low or "v2." in low:
        version = "v20"
    elif "v10" in low:
        version = "v10"

    config = "unknown"
    for key in ["vcmslcfg", "vcmcfg", "vcmsl", "vcm"]:
        if key in low:
            config = key
            break

    return {
        "year": int(start_year),
        "month": int(month) if month is not None else None,
        "temporal": temporal,
        "band_key": band_key,
        "version": version,
        "config": config,
        "filename": name,
        "path": str(path),
    }


@contextmanager
def _open_tiff_path(path: Path) -> Iterator[Path]:
    if path.suffix.lower() != ".gz":
        yield path
        return

    tmp_dir = DATA_RAW / "_tmp_viirs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False, dir=tmp_dir)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        with gzip.open(path, "rb") as src, tmp_path.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        yield tmp_path
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


def _geo_transform_from_image(img: Image.Image) -> Tuple[float, float, float, float]:
    tags = getattr(img, "tag_v2", None)
    if tags is None:
        msg = "GeoTIFF tags are unavailable."
        raise RuntimeError(msg)
    scale = tags.get(_TAG_MODEL_PIXEL_SCALE)
    tie = tags.get(_TAG_MODEL_TIEPOINT)
    if not scale or not tie or len(scale) < 2 or len(tie) < 6:
        msg = "GeoTIFF is missing ModelPixelScale or ModelTiepoint tags."
        raise RuntimeError(msg)

    sx = float(scale[0])
    sy = float(scale[1])
    i0 = float(tie[0])
    j0 = float(tie[1])
    x0 = float(tie[3])
    y0 = float(tie[4])
    origin_x = x0 - i0 * sx
    origin_y = y0 + j0 * sy
    return origin_x, origin_y, sx, sy


def _read_struct(fp: Any, fmt: str) -> Tuple[Any, ...]:
    size = struct.calcsize(fmt)
    raw = fp.read(size)
    if len(raw) != size:
        msg = f"Unable to read {size} bytes for format {fmt!r}."
        raise RuntimeError(msg)
    return struct.unpack(fmt, raw)


def _decode_tiff_values(raw: bytes, field_type: int, count: int, endian: str) -> Tuple[Any, ...]:
    if field_type == 1:
        return tuple(raw[i] for i in range(count))
    if field_type == 2:
        return tuple(raw.rstrip(b"\x00").decode("ascii", errors="ignore"))
    if field_type == 3:
        return struct.unpack(endian + ("H" * count), raw)
    if field_type == 4:
        return struct.unpack(endian + ("I" * count), raw)
    if field_type == 5:
        vals = struct.unpack(endian + ("II" * count), raw)
        return tuple(vals[i] / vals[i + 1] if vals[i + 1] else float("nan") for i in range(0, len(vals), 2))
    if field_type == 11:
        return struct.unpack(endian + ("f" * count), raw)
    if field_type == 12:
        return struct.unpack(endian + ("d" * count), raw)
    if field_type == 16:
        return struct.unpack(endian + ("Q" * count), raw)
    if field_type == 17:
        return struct.unpack(endian + ("q" * count), raw)
    if field_type == 18:
        return struct.unpack(endian + ("Q" * count), raw)
    return tuple(raw)


def _load_tiff_array_values(
    fp: Any,
    *,
    value_offset: int,
    field_type: int,
    count: int,
    endian: str,
    offset_size: int,
) -> Tuple[Any, ...]:
    elem_size = _TIFF_TYPE_SIZES.get(field_type)
    if elem_size is None:
        msg = f"Unsupported TIFF field type: {field_type}"
        raise RuntimeError(msg)

    total_size = elem_size * int(count)
    if total_size <= int(offset_size):
        raw = int(value_offset).to_bytes(int(offset_size), byteorder="little" if endian == "<" else "big", signed=False)[
            :total_size
        ]
        return _decode_tiff_values(raw, field_type, int(count), endian)

    pos = fp.tell()
    fp.seek(int(value_offset))
    raw = fp.read(total_size)
    fp.seek(pos)
    if len(raw) != total_size:
        msg = f"Unable to read TIFF payload at offset {value_offset}."
        raise RuntimeError(msg)
    return _decode_tiff_values(raw, field_type, int(count), endian)


def _read_tiff_binary_meta(path: Path) -> Dict[str, Any]:
    with path.open("rb") as fp:
        byte_order = fp.read(2)
        if byte_order == b"II":
            endian = "<"
        elif byte_order == b"MM":
            endian = ">"
        else:
            msg = f"Unsupported TIFF byte order for {path}"
            raise RuntimeError(msg)

        version = _read_struct(fp, endian + "H")[0]
        if int(version) == 43:
            offset_size = _read_struct(fp, endian + "H")[0]
            _ = _read_struct(fp, endian + "H")[0]
            if int(offset_size) != 8:
                msg = f"Unsupported BigTIFF offset size {offset_size} for {path}"
                raise RuntimeError(msg)
            first_ifd = _read_struct(fp, endian + "Q")[0]
            fp.seek(int(first_ifd))
            entry_count = _read_struct(fp, endian + "Q")[0]
            entry_fmt = endian + "HHQQ"
        elif int(version) == 42:
            offset_size = 4
            first_ifd = _read_struct(fp, endian + "I")[0]
            fp.seek(int(first_ifd))
            entry_count = _read_struct(fp, endian + "H")[0]
            entry_fmt = endian + "HHII"
        else:
            msg = f"Unsupported TIFF version {version} for {path}"
            raise RuntimeError(msg)

        tags: Dict[int, Tuple[int, int, int]] = {}
        for _ in range(int(entry_count)):
            tag, field_type, count, value_offset = _read_struct(fp, entry_fmt)
            tags[int(tag)] = (int(field_type), int(count), int(value_offset))

        def _tag_values(tag_id: int, default: Tuple[Any, ...] | None = None) -> Tuple[Any, ...] | None:
            spec = tags.get(int(tag_id))
            if spec is None:
                return default
            field_type, count, value_offset = spec
            return _load_tiff_array_values(
                fp,
                value_offset=value_offset,
                field_type=field_type,
                count=count,
                endian=endian,
                offset_size=offset_size,
            )

        width_vals = _tag_values(_TAG_IMAGE_WIDTH)
        height_vals = _tag_values(_TAG_IMAGE_LENGTH)
        bits_vals = _tag_values(_TAG_BITS_PER_SAMPLE)
        compression_vals = _tag_values(_TAG_COMPRESSION)
        strip_offsets = _tag_values(_TAG_STRIP_OFFSETS)
        rows_per_strip_vals = _tag_values(_TAG_ROWS_PER_STRIP)
        strip_byte_counts = _tag_values(_TAG_STRIP_BYTE_COUNTS)
        sample_format_vals = _tag_values(_TAG_SAMPLE_FORMAT, default=(1,))
        samples_per_pixel_vals = _tag_values(_TAG_SAMPLES_PER_PIXEL, default=(1,))
        scale_vals = _tag_values(_TAG_MODEL_PIXEL_SCALE)
        tie_vals = _tag_values(_TAG_MODEL_TIEPOINT)

    required = {
        "width": width_vals,
        "height": height_vals,
        "bits": bits_vals,
        "compression": compression_vals,
        "strip_offsets": strip_offsets,
        "rows_per_strip": rows_per_strip_vals,
        "strip_byte_counts": strip_byte_counts,
        "scale": scale_vals,
        "tie": tie_vals,
    }
    missing = [name for name, vals in required.items() if not vals]
    if missing:
        msg = f"TIFF metadata missing required tags for {path}: {', '.join(missing)}"
        raise RuntimeError(msg)

    scale = tuple(float(v) for v in scale_vals[:2])
    tie = tuple(float(v) for v in tie_vals[:6])
    sx = float(scale[0])
    sy = float(scale[1])
    i0 = float(tie[0])
    j0 = float(tie[1])
    x0 = float(tie[3])
    y0 = float(tie[4])
    origin_x = x0 - i0 * sx
    origin_y = y0 + j0 * sy

    return {
        "width": int(width_vals[0]),
        "height": int(height_vals[0]),
        "bits_per_sample": int(bits_vals[0]),
        "compression": int(compression_vals[0]),
        "rows_per_strip": int(rows_per_strip_vals[0]),
        "strip_offsets": [int(v) for v in strip_offsets],
        "strip_byte_counts": [int(v) for v in strip_byte_counts],
        "sample_format": int(sample_format_vals[0]) if sample_format_vals else 1,
        "samples_per_pixel": int(samples_per_pixel_vals[0]) if samples_per_pixel_vals else 1,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "scale_x": sx,
        "scale_y": sy,
    }


def _dtype_from_tiff_meta(meta: Dict[str, Any]) -> np.dtype:
    bits = int(meta["bits_per_sample"])
    sample_format = int(meta.get("sample_format", 1))
    if bits == 32 and sample_format == 3:
        return np.dtype("<f4")
    if bits == 32 and sample_format == 1:
        return np.dtype("<u4")
    if bits == 32 and sample_format == 2:
        return np.dtype("<i4")
    if bits == 16 and sample_format == 1:
        return np.dtype("<u2")
    if bits == 16 and sample_format == 2:
        return np.dtype("<i2")
    msg = f"Unsupported TIFF sample format={sample_format}, bits={bits}"
    raise RuntimeError(msg)


def _sample_uncompressed_strip_tiff(path: Path, point_records: List[dict]) -> Dict[str, float]:
    meta = _read_tiff_binary_meta(path)
    if int(meta["compression"]) != 1:
        msg = f"Only uncompressed TIFF strips are supported, got compression={meta['compression']}"
        raise RuntimeError(msg)

    width = int(meta["width"])
    height = int(meta["height"])
    rows_per_strip = int(meta["rows_per_strip"])
    dtype = _dtype_from_tiff_meta(meta)
    bytes_per_sample = dtype.itemsize
    samples_per_pixel = int(meta.get("samples_per_pixel", 1))
    row_specs: Dict[int, List[Tuple[str, int]]] = {}

    for rec in point_records:
        col = int(round((float(rec["lon"]) - float(meta["origin_x"])) / float(meta["scale_x"])))
        row = int(round((float(meta["origin_y"]) - float(rec["lat"])) / float(meta["scale_y"])))
        if col < 0 or row < 0 or col >= width or row >= height:
            continue
        row_specs.setdefault(int(row), []).append((str(rec["city_id"]), int(col)))

    if not row_specs:
        return {}

    strip_offsets = [int(v) for v in meta["strip_offsets"]]
    strip_byte_counts = [int(v) for v in meta["strip_byte_counts"]]
    city_vals: Dict[str, List[float]] = {}

    with path.open("rb") as fp:
        for row, specs in row_specs.items():
            strip_index = int(row // rows_per_strip)
            within_strip_row = int(row % rows_per_strip)
            strip_offset = int(strip_offsets[strip_index])
            strip_size = int(strip_byte_counts[strip_index])
            fp.seek(strip_offset)
            strip = fp.read(strip_size)
            if len(strip) != strip_size:
                continue

            row_stride = width * samples_per_pixel * bytes_per_sample
            row_start = within_strip_row * row_stride
            for city_id, col in specs:
                byte_start = row_start + (int(col) * samples_per_pixel * bytes_per_sample)
                byte_end = byte_start + bytes_per_sample
                if byte_end > len(strip):
                    continue
                val = np.frombuffer(strip[byte_start:byte_end], dtype=dtype, count=1)
                if val.size == 0:
                    continue
                out = float(val[0])
                if not np.isfinite(out) or out <= -900.0:
                    continue
                city_vals.setdefault(str(city_id), []).append(out)

    out: Dict[str, float] = {}
    for city_id, vals in city_vals.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        out[city_id] = float(np.median(arr))
    return out


def _get_pixel_value(img: Image.Image, col: int, row: int) -> float:
    value = img.getpixel((int(col), int(row)))
    if isinstance(value, tuple):
        if not value:
            return float("nan")
        value = value[0]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(out):
        return float("nan")
    return out


def _sample_geotiff_city_values(path: Path, point_records: List[dict]) -> Dict[str, float]:
    with _open_tiff_path(path) as tif_path:
        return _sample_uncompressed_strip_tiff(tif_path, point_records)


def _priority_for_source(source: str) -> int:
    s = str(source).lower()
    if "eog_monthly" in s:
        return 3
    if "noaa_viirs_nightly_radiance" in s:
        return 2
    if "eog_annual" in s:
        return 1
    return 0


def _collapse_viirs_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keys = ["city_id", "year", "month"]
    work = df.copy()
    if "viirs_source" not in work.columns:
        work["viirs_source"] = "unknown"
    work["_priority"] = work["viirs_source"].map(_priority_for_source).fillna(0).astype(int)
    all_cols = [c for c in work.columns if c not in {"_priority"}]

    rows: List[dict] = []
    for _, grp in work.groupby(keys, sort=False):
        grp = grp.sort_values("_priority")
        top = grp.iloc[-1]
        row: Dict[str, Any] = {k: top[k] for k in keys}
        for col in all_cols:
            if col in keys:
                continue
            series = grp[col]
            if pd.api.types.is_numeric_dtype(series):
                valid = pd.to_numeric(series, errors="coerce").dropna()
                row[col] = float(valid.iloc[-1]) if not valid.empty else np.nan
            else:
                valid = series.dropna().astype(str)
                row[col] = valid.iloc[-1] if not valid.empty else ""
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.sort_values(keys).reset_index(drop=True)
    return out


def build_historical_viirs_city_monthly(
    *,
    source_root: str | os.PathLike[str],
    max_cities: int = 295,
    start_year: int = 2014,
    end_year: int = 2025,
    step_deg: float = 0.01,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    root = Path(source_root).expanduser().resolve()
    if not root.exists():
        msg = f"Historical VIIRS root does not exist: {root}"
        raise FileNotFoundError(msg)

    cities = load_city_catalog(max_cities=int(max_cities)).copy()
    points = _sample_points(cities, step_deg=float(step_deg))
    files = _discover_viirs_files(root)

    rows: List[dict] = []
    skipped: List[dict] = []
    processed = 0

    for path in files:
        meta = _parse_temporal_meta(path)
        if meta is None:
            skipped.append({"path": str(path), "reason": "unrecognized_filename_pattern"})
            continue
        year = int(meta["year"])
        month = meta["month"]
        if year < int(start_year) or year > int(end_year):
            continue
        if meta["temporal"] == "monthly" and (month is None or month < 1 or month > 12):
            skipped.append({"path": str(path), "reason": "invalid_month"})
            continue

        band_key = str(meta["band_key"])
        if band_key not in _RADIANCE_KEYS and band_key not in _COVERAGE_KEYS:
            continue

        try:
            sampled = _sample_geotiff_city_values(path, points)
            processed += 1
        except Exception as exc:  # noqa: BLE001
            skipped.append({"path": str(path), "reason": f"sample_failed:{type(exc).__name__}"})
            LOGGER.warning("Historical VIIRS sample failed for %s: %s", path, exc)
            continue

        temporal = str(meta["temporal"])
        source = f"eog_{temporal}_{meta['version']}_{meta['config']}"
        out_month = int(month) if temporal == "monthly" and month is not None else 7
        for city_id, value in sampled.items():
            row = {
                "city_id": str(city_id),
                "year": int(year),
                "month": int(out_month),
                "viirs_source": source,
                "viirs_product_temporal": temporal,
                "viirs_product_version": str(meta["version"]),
                "viirs_product_config": str(meta["config"]),
                "viirs_filename": str(path.name),
                "radiance": np.nan,
                "cf_cvg": np.nan,
                "lit_area_km2": np.nan,
            }
            if band_key in _COVERAGE_KEYS:
                row["cf_cvg"] = float(value)
            else:
                row["radiance"] = float(value)
            rows.append(row)

    imported = _collapse_viirs_rows(pd.DataFrame(rows))
    summary = {
        "status": "ok",
        "source_root": str(root),
        "files_discovered": int(len(files)),
        "files_processed": int(processed),
        "rows": int(len(imported)),
        "cities": int(imported["city_id"].nunique()) if not imported.empty else 0,
        "year_min": int(imported["year"].min()) if not imported.empty else None,
        "year_max": int(imported["year"].max()) if not imported.empty else None,
        "source_counts": imported.get("viirs_source", pd.Series(dtype=str)).value_counts(dropna=False).to_dict()
        if not imported.empty
        else {},
        "skipped_files_preview": skipped[:20],
    }
    return imported, summary


def merge_viirs_monthly_panels(existing: pd.DataFrame, imported: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return _collapse_viirs_rows(imported)
    if imported.empty:
        return _collapse_viirs_rows(existing)
    return _collapse_viirs_rows(pd.concat([existing, imported], ignore_index=True, sort=False))


def import_historical_viirs_to_raw(
    *,
    source_root: str | os.PathLike[str],
    max_cities: int = 295,
    start_year: int = 2014,
    end_year: int = 2025,
    merge_existing: bool = True,
    output_path: Path | None = None,
) -> Dict[str, Any]:
    imported, summary = build_historical_viirs_city_monthly(
        source_root=source_root,
        max_cities=int(max_cities),
        start_year=int(start_year),
        end_year=int(end_year),
    )

    dest = output_path or (DATA_RAW / "viirs_city_monthly.csv")
    existing = pd.DataFrame()
    if merge_existing and dest.exists():
        try:
            existing = pd.read_csv(dest)
        except Exception:  # noqa: BLE001
            existing = pd.DataFrame()

    merged = merge_viirs_monthly_panels(existing, imported)
    merged.to_csv(dest, index=False)

    summary["output_path"] = str(dest)
    summary["rows_after_merge"] = int(len(merged))
    summary["cities_after_merge"] = int(merged["city_id"].nunique()) if "city_id" in merged.columns else 0
    summary["merge_existing"] = bool(merge_existing)
    dump_json(DATA_RAW / "viirs_historical_import_summary.json", summary)
    return summary
