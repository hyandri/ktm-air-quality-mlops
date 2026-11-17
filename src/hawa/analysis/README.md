# `analysis/` : checks that only report

One-off investigations that guide decisions. They download or read data and print or save numbers.
They do **not** change files in `data/processed/`.

| File | Question it answers |
|---|---|
| `sensor_scorecard.py` | which live OpenAQ sensors have complete PM2.5 data? |
| `sensor_transfer.py` | does the Embassy-trained model still work on live (AirGradient) sensors? |
| `openmeteo_probe.py` | which weather variables does each Open-Meteo API really return, and since when? |

## `sensor_scorecard.py`

```
python -m hawa.analysis.sensor_scorecard --start 2025-09-01
python -m hawa.analysis.sensor_scorecard --max 3                  # quick trial
python -m hawa.analysis.sensor_scorecard --ids 6093549,5676345    # chosen sensors
```

Needs `data/raw/openaq/kathmandu_locations.csv` (from `python -m hawa.ingestion.openaq locations`).
For each selected location it downloads raw PM2.5 (via `hawa.ingestion.openaq`), cleans it with the project rules (via `hawa.processing.clean`) and counts **valid hours only**. Safe to stop and re-run.

Default selection: last data within 7 days and at least 300 days of history. `--ids` overrides this.
Other options: `--end`, `--window-days`, `--endpoint`, `--timeout`, `--out`, `--out-dir`.

Output: `data/raw/openaq/scorecard.csv` (by default), sorted by `missing_pct`.
**Careful:** the default name overwrites the file of the same name. The corrected results we use now are in `scorecard_v2.csv`; pass `--out data/raw/openaq/scorecard_v2.csv` to refresh it.

| Column | Meaning |
|---|---|
| `valid_hours` | hours with a valid value |
| `missing_pct` | share of hours between the first and last valid hour with no valid value |
| `winter_missing_pct` | same, for Dec–Feb only. Winter is the most important season |
| `longest_gap_h` | longest run of missing hours |
| `gaps_over_24h` | how many gaps are longer than 24 h |
| `median_value` | median PM2.5, as a sanity check |
| `n_raw_null` | raw rows that had no value |
| `pct_flagged` | share of rows OpenAQ flagged (kept, see `processing/README.md`) |

Low `missing_pct` means complete data, not accurate data. Calibration against a reference station is a separate check.

## `openmeteo_probe.py`

```
python -m hawa.analysis.openmeteo_probe
```

Sends a few small requests and prints, for each API and period, how many of the 11 candidate variables return values and which are empty. It also tests how far back the Historical Forecast API goes.

Result on 16 Nov 2026: ERA5 returns all 11 variables; the Historical Forecast API works from 2021-01-01 but has no `boundary_layer_height`; the live forecast returns all 11; Previous Runs works from Feb 2024.
Re-run it if Open-Meteo changes (new variables, new start dates).

## `sensor_transfer.py`

```
python -m hawa.models.train --train-until 2024-12-01 --no-track   # once: the frozen model
python -m hawa.analysis.sensor_transfer
python -m hawa.analysis.sensor_transfer --track                   # also an MLflow run (hawa-sensor-transfer)
```

For every downloaded `loc<ID>_sensor<SID>_pm25_hours.csv` and one winter (default 2025/26):

| Part | What | Columns |
|---|---|---|
| A. Transfer score | the frozen model (trained before 2024-12-01) predicts from the sensor's own PM2.5 history; scored against the two baselines on the sensor's own values | `n_origins`, `mae_model`, `mae_bar`, `gain_pct` |
| B. Agreement | sensor vs Embassy at the same hours | `overlap_h`, `corr`, `bias` (sensor minus Embassy), `ratio` (mean sensor / mean Embassy) |

Also `dist_km` (straight line to the Embassy) and, once, the same model scored on the Embassy for that winter (the reference). The sensor's table is built with the same functions as in training. A sensor needs at least 500 scored hours (`--min-origins`) and 200 overlap hours to get a number. Writes `reports/sensor_transfer.csv`.

**Limits:** about one winter per sensor; sensors are at different places, so agreement does not prove the instruments are interchangeable; winter 2025/26 is also a locked test winter for the Embassy model, so a later calibration based on these numbers must be treated as a new experiment.
