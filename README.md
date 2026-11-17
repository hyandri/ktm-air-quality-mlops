# Hawa (AQI-MLOPS)

Hourly PM2.5 forecasting for Kathmandu (24 h ahead) with an MLOps pipeline.
Current stage: **Stage 1 – baselines scored on validation; weather and ML models next.**

## Layout

```
AQI-MLOPS/
├── data/
│   ├── raw/               untouched downloads (openaq/, openmeteo/, open_data_nepal/)
│   ├── interim/           half-processed data
│   └── processed/         clean files ready for modelling
├── src/hawa/
│   ├── config.py          paths, coordinates, cleaning constants
│   ├── ingestion/         RAW pulls from APIs only (no cleaning)
│   │   ├── http.py          shared GET with retries
│   │   ├── openaq.py        locations + sensor history
│   │   └── openmeteo.py     weather
│   ├── processing/
│   │   └── clean.py       cleaning rules -> data/processed
│   ├── models/            baselines, LightGBM
│   ├── evaluation/        splits, metrics, scoring
│   └── analysis/          one-off checks that guide decisions
│       ├── sensor_scorecard.py
│       └── openmeteo_probe.py
├── reports/               result tables (baseline scores, ...)
├── tests/
├── notebooks/
└── docs/
```

Rule of thumb: `ingestion` fetches, `processing` changes data, `analysis` only reports.

## Documentation

- [docs/decisions.md](docs/decisions.md): what we tried, the numbers, what we chose
- [data/README.md](data/README.md): every data file, its source, licence, cleaning and known problems
- [src/README.md](src/README.md): code overview, conventions, how to add modules
- per-folder docs: [ingestion](src/hawa/ingestion/README.md), [processing](src/hawa/processing/README.md), [models](src/hawa/models/README.md), [evaluation](src/hawa/evaluation/README.md), [analysis](src/hawa/analysis/README.md)

## Setup (once)

```
cd ~/projects/AQI-MLOPS
pip install -r requirements.txt     # same as: pip install -e ".[dev]"
cp .env.example .env        # then put your OpenAQ key in .env
pytest                      # should pass
```

## Commands (run from the project root)

```
python -m hawa.ingestion.openaq locations
python -m hawa.ingestion.openaq download --location-id 3459 --start 2021-03-01 --end 2026-11-16
python -m hawa.ingestion.openmeteo --source archive --start 2017-03-01 --end 2026-11-05
python -m hawa.processing.clean
python -m hawa.processing.build_dataset
python -m hawa.processing.features
python -m hawa.evaluation.evaluate_baselines
python -m hawa.evaluation.evaluate_models
python -m hawa.evaluation.walk_forward
python -m hawa.evaluation.walk_forward
python -m hawa.analysis.sensor_scorecard --start 2025-09-01
python -m hawa.analysis.openmeteo_probe
```

## Data sources and licences

- AirNow-derived US Embassy data via Open Data Nepal (CC BY-SA) and OpenAQ.
- AirGradient sensors via OpenAQ (CC BY 4.0 reported; confirm per sensor).
- Open-Meteo weather (CC BY 4.0, non-commercial free tier). Keep the attribution.
