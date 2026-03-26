# Historical VIIRS Import

This project can now ingest local EOG VIIRS GeoTIFF products and merge them into `data/raw/viirs_city_monthly.csv`.

## Why this exists

The NOAA nightly ArcGIS service currently exposed in the project only covers a short 2025 window.
The broader historical VIIRS monthly and annual products are hosted by the Earth Observation Group (EOG)
and now require user login.

## Supported local sources

- Monthly non-tiled VIIRS composites:
  - `https://eogdata.mines.edu/nighttime_light/monthly_notile/`
- Annual VIIRS composites:
  - `https://eogdata.mines.edu/nighttime_light/annual/v20/`
  - `https://eogdata.mines.edu/nighttime_light/annual/v21/`
  - `https://eogdata.mines.edu/nighttime_light/annual/v22/`

## What to download

Recommended:

- Monthly radiance files:
  - `avg_rade9`, `avg_rade9h`, or `average`
- Monthly cloud-free coverage files:
  - `cf_cvg`
- Annual radiance files:
  - `average` or `avg_rade9`

The importer uses radiance as the main signal and keeps `cf_cvg` when available.

## Suggested local layout

Place downloaded files anywhere under one root, for example:

```text
data/raw/eog_viirs/
  monthly/
  annual/
```

The importer scans recursively, so nested year folders are fine.

## Import command

```bash
python3 run_historical_viirs_import.py \
  --source-root data/raw/eog_viirs \
  --start-year 2014 \
  --end-year 2025 \
  --max-cities 295
```

This writes:

- `data/raw/viirs_city_monthly.csv`
- `data/raw/viirs_historical_import_summary.json`

## Unified crawler path

You can also merge historical VIIRS during the normal crawler run:

```bash
python3 run_data_crawler.py \
  --crawl-viirs \
  --historical-viirs-root data/raw/eog_viirs \
  --max-cities 295 \
  --start-year 2015 \
  --end-year 2025
```

## Current limitation

The project cannot fetch EOG historical files anonymously. Direct file requests are redirected to the EOG login system.
You need either:

- downloaded GeoTIFF files, or
- an authenticated download workflow outside the current project.
