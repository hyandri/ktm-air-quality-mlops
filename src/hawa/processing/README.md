# `processing/` : change data

Turns raw files into clean, modelling-ready files. Reads from `data/raw/`, writes to `data/processed/`.

| File | Job |
|---|---|
| `clean.py` | clean PM2.5 and build one hourly file per station |
| `build_dataset.py` | join clean PM2.5 and weather into one modelling table with 24 targets |
| `features.py` | add PM2.5 history features (lags, rolling means, trend) |

## `clean.py`

```
python -m hawa.processing.clean
```

Defaults match the folder layout, so no arguments are needed. Options: `--embassy-odn`, `--embassy-oaq`, `--phora-odn`, `--phora-oaq`, `--out-dir`.

**Output** (`data/processed/`): `embassy_pm25_hourly_clean.csv` and `phora_pm25_hourly_clean.csv`
with columns `timestamp_utc, station, pm25, source`. The command also prints valid-hour counts, stuck hours removed, how much the two sources overlap, and winter coverage.

### The cleaning rules

Applied in this order, to every source:

1. **Keep valid numbers only.** Drop empty values, `-999`, any negative value, and the error code `985`.
2. **One value per hour.** If an hour appears twice, keep the smaller value. Timestamp = hour start, UTC.
3. **Drop flat runs.** 12 or more consecutive hours with exactly the same value = stuck sensor. (A run only counts across consecutive hours, so a gap breaks it.)

Not dropped on purpose: **flagged rows.** We compared flagged values with neighbouring hours and they look normal.

The constants are in `config.py` (`ERROR_VALUES`, `FLAT_RUN_HOURS`).

### Merging the two sources

- Old Open Data Nepal timestamps mark the END of the hour (at minute `:15`). The code floors to the hour and shifts back 1 hour.
- OpenAQ timestamps already mark the hour START.
- After the shift, both sources are identical on the overlap (250 hours for Embassy, 256 for Phora), so the files can be stitched.
- On overlapping hours the Open Data Nepal value is kept. `source` says where each row came from.

### Functions

| Function | Does |
|---|---|
| `read_odn(path)` | Open Data Nepal CSV → `hour_start`, `v` (PM2.5 only, shifted) |
| `openaq_to_frame(raw)`, `read_openaq(path)` | OpenAQ raw table or CSV → `hour_start`, `v` |
| `valid_series(d)` | rules 1 and 2 → Series indexed by `hour_start` |
| `drop_flat_runs(s)` | rule 3 → `(series, hours_removed)` |
| `clean_frame(d)` | all three rules → `(series, hours_removed)` |
| `build_station(name, odn_path, openaq_path)` | cleaned, merged DataFrame for one station |
| `winter_coverage(df)` | text: share of Dec–Feb hours that have data, per winter |

### Adding a station

Add its raw files, call `build_station(...)` in `main()`, and write the file. Add a test in `tests/test_clean.py` if you add a rule.

### Tests

`tests/test_clean.py` checks: bad values are dropped, one row per hour, a flat run is removed only at 12+ hours, a run needs consecutive hours, and Open Data Nepal timestamps shift back 1 hour.

## `build_dataset.py`

```
python -m hawa.processing.build_dataset                       # Embassy, weather = auto
python -m hawa.processing.build_dataset --station phora --weather era5
```

Joins clean PM2.5 and weather into one table, **one row per hour** on a full hourly grid.
Row `t` is a forecast origin: "it is hour `t`; what will PM2.5 be 1 to 24 hours later?"

| Column group | Columns | Notes |
|---|---|---|
| time | `timestamp_utc` | hour start, UTC |
| current PM2.5 | `pm25` | empty if the hour is missing |
| weather | the 10 `MODEL_WEATHER_VARIABLES` in `config.py` | boundary layer height is left out |
| weather label | `weather_source` | `historical_forecast` or `era5` |
| time features | `hour_local`, `month`, `dayofweek`, `dayofyear`, `hour_sin/cos`, `doy_sin/cos` | Kathmandu local time (UTC+05:45); `hour_local` is a float, e.g. 5.75 |
| targets | `y_h01` ... `y_h24` | PM2.5 at `t+1h` ... `t+24h`. **Labels only, never features** |

`--weather` options:

| Value | Meaning |
|---|---|
| `auto` (default) | archived forecasts from 2021-01-01, ERA5 before |
| `forecast` | archived forecasts only (rows before 2021 have no weather) |
| `era5` | ERA5 for every year. Use it to measure the train/live mismatch |

Output: `data/processed/<station>_dataset_<weather>.csv`. The command prints row counts, missing shares, how many target values exist and the number of usable rows per winter.

Rules this file follows:
- **Nothing is filled or interpolated.** A missing hour stays empty; models and splits decide what to do with it.
- **No future information in features.** Only targets look forward. `tests/test_build_dataset.py` checks that changing a future PM2.5 value does not change an earlier row's features.
- **Weather is the value for hour `t` itself.** Future weather (the forecast for `t+1 ... t+24`) is not in the table yet. Whether to add it is a modelling decision for Stage 1.
- **Alignment note:** PM2.5 is the average over the hour starting at `t`; Open-Meteo instant values are at time `t`, and `precipitation` is the sum of the hour before `t`. We do not shift them. This small offset is accepted for now.

Functions: `read_pm25`, `read_weather`, `choose_weather`, `add_time_features`, `add_targets`, `build`, `report`.

## `features.py`

```
python -m hawa.processing.features        # reads <station>_dataset_auto.csv, writes <station>_features_auto.csv
```

Adds what the model learns from: the recent history of PM2.5. Every feature at row `t` uses only hours `t, t-1, t-2, ...`; a test changes the future and checks nothing earlier moves.

| Feature | Meaning |
|---|---|
| `pm25_filled` | current PM2.5; if the last 1–3 hours are missing, the last real value is carried forward |
| `pm25_age_h` | hours since the last real measurement (0 = measured now) |
| `pm25_lag001 ... lag168` | PM2.5 1, 2, 3, 6, 12, 24, 48, 168 hours earlier (168 h = one week) |
| `pm25_mean006 ... mean168` | average of the last 6, 24, 72, 168 hours |
| `pm25_std024` | spread of the last 24 hours (is the air changing fast?) |
| `pm25_change03` | now minus 3 hours ago (rising or falling) |

**Missing-data rule (decided):**
- A gap is bridged for at most `FILL_LIMIT_H = 3` hours, by carrying the last real value **forward**.
- We do not interpolate, because interpolation needs the next value, which is the future.
- Longer gaps stay empty. `pm25_age_h` tells the model how old the last real value is.
- `pm25` (raw) and the targets `y_h01 ... y_h24` are **never** filled.
- On the Embassy this rescues about 4,200 hours (PM2.5 available 71.3% → 76.4%).

`feature_columns(df)` lists the columns a model may use as inputs (34 now: 10 weather, 8 time, 16 PM2.5 history). It excludes raw `pm25`, `weather_source` and all targets.

Constants are in `config.py`: `FILL_LIMIT_H`, `LAG_HOURS`, `ROLL_WINDOWS`.

Not included yet: future weather (the forecast for `t+1 ... t+24`) and weather lags. We add them as experiments, so we can measure whether they help.

### Next in this folder (planned)

Weather experiments (future forecast values as features).
