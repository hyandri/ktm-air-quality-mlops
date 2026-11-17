# `analysis/` : checks that only report

One-off investigations that guide decisions. They download or read data and print or save numbers.
They do **not** change files in `data/processed/`.

| File | Question it answers |
|---|---|
| `sensor_scorecard.py` | which live OpenAQ sensors have complete PM2.5 data? |
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
