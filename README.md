# MACRO-City Engine

MACRO-City Engine is a city-observation and analysis repository for measuring temporal mismatches between fast urban activity and slower structural adjustment.

## What this repository does

- collects and harmonizes multi-source city observations
- builds city-year analytical panels
- measures dynamic urban risk states, including fragile boom
- produces validation, policy-timing, and related analysis outputs
- stores the current public paper package

## Main components

- `src/`: core analysis code
- `run_pipeline.py`: end-to-end analysis pipeline
- `run_data_crawler.py`: data collection entry point
- `run_social_crawler.py`: social-signal collection entry point
- `run_realtime_monitor.py`: monitoring output generation
- `paper/nature_cities_submission/`: manuscript, supplementary information, and figure assets

## How to use it

Run the full pipeline:

```bash
python3 run_pipeline.py
```

Run realtime monitoring only:

```bash
python3 run_realtime_monitor.py
```

Main outputs are written under:

- `data/processed/`
- `data/outputs/`
