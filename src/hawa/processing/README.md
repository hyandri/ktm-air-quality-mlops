# `processing/` : change data

Turns raw files into clean, modelling-ready files. Reads from `data/raw/`, writes to `data/processed/`.

| File | Job |
|---|---|
| `clean.py` | clean PM2.5 and build one hourly file per station |

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

### Next in this folder (planned)

`build_dataset.py`: join clean PM2.5 and weather on a full hourly grid, add the 24 h target and time features.
