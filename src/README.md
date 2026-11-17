# `src/` : source code

All project code is one Python package called `hawa`, inside `src/hawa/`.
Install it once (editable) and run everything with `python -m`:

```
pip install -r requirements.txt
python -m hawa.<folder>.<file> [options]        # always from the project root
```

## Module map

```
src/hawa/
├── config.py       paths, coordinates, weather variable list, cleaning constants
├── ingestion/      fetch RAW data from APIs. Does not change values.
├── processing/     change data: cleaning, joining, features
├── models/         forecasting models (baselines now; Ridge, LightGBM later)
├── evaluation/     time-based splits, metrics, scoring
└── analysis/       one-off checks that report numbers, to guide decisions
```

**Rule of thumb: `ingestion` fetches, `processing` changes, `analysis` only reports.**
Imports go one way: `analysis` may use `processing` and `ingestion`; `processing` may use `config`;
`ingestion` never imports from `processing` or `analysis`.

Docs per folder (add a new README when you add a new folder):

| Folder | Doc |
|---|---|
| `ingestion/` | [ingestion/README.md](hawa/ingestion/README.md) |
| `processing/` | [processing/README.md](hawa/processing/README.md) |
| `models/` | [models/README.md](hawa/models/README.md) |
| `evaluation/` | [evaluation/README.md](hawa/evaluation/README.md) |
| `analysis/` | [analysis/README.md](hawa/analysis/README.md) |

Data files and their meaning: [`data/README.md`](../data/README.md).

## `config.py`

One place for things several modules need. Change them here, not inside modules.

| Name | Meaning |
|---|---|
| `PROJECT_ROOT`, `DATA_DIR`, `RAW_DIR`, `INTERIM_DIR`, `PROCESSED_DIR` | project paths (found from the file location, so they work from any folder) |
| `OPENAQ_RAW`, `OPENMETEO_RAW`, `OPEN_DATA_NEPAL_RAW` | raw data sub-folders |
| `KATHMANDU_BBOX` | `min_lon,min_lat,max_lon,max_lat` = `85.15,27.55,85.60,27.85` |
| `EMBASSY_LAT`, `EMBASSY_LON` | US Embassy point, also used for weather |
| `OPENAQ_PM25_PARAMETER_ID` | `2` (PM2.5 in OpenAQ) |
| `WEATHER_VARIABLES` | the 11 candidate weather variables |
| `ERROR_VALUES`, `FLAT_RUN_HOURS` | cleaning constants (`985`, `12`) |
| `MODEL_WEATHER_VARIABLES`, `FORECAST_WEATHER_START` | weather columns used in models; date from which archived forecasts are used |
| `HORIZONS`, `LOCAL_UTC_OFFSET_H` | 24 forecast hours; Kathmandu offset 5.75 h |
| `WINTER_MONTHS`, `TRAIN_WINTERS`, `VALIDATION_WINTERS`, `TEST_WINTERS`, `TRAIN_ALL_END`, `VALIDATION_ALL`, `TEST_ALL_START` | split definitions (see `evaluation/README.md`) |
| `REPORTS_DIR` | `reports/`: result tables from evaluation scripts |

`config.py` also reads a `.env` file in the project root (lines like `OPENAQ_API_KEY=...`).
Real environment variables win over `.env`. Never commit `.env` or paste the key in chat.

## Conventions

- **Time:** always UTC. Hourly timestamp = start of the hour.
- **Commands:** every module with a command-line has `main(argv=None)`, so it can be called from Python or tests too.
- **Safe to stop and re-run:** downloads are cached in `_cache/` folders and resume.
- **Rate limits:** the code sleeps between calls. Do not remove the sleeps.
- **Each file starts with a docstring** explaining what it does and how to run it. `python -m hawa.<...> --help` prints it.
- **Tests:** `pytest` from the project root. Add a test with every cleaning rule or feature.

## Adding something new

1. Pick the folder by the rule of thumb above.
2. Put paths and constants in `config.py`, not in the new file.
3. Give the file a docstring with the run command.
4. Add a short entry in that folder's README (or create a README for a new folder and link it in the table above).
5. Add a test in `tests/`.
