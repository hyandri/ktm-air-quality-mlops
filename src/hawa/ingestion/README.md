# `ingestion/` : pull RAW data

Fetches data from APIs and saves it. **No cleaning, no judging.** Values stay as the API gave them
(JSON is only flattened into a table). Output goes to `data/raw/`.

| File | Job |
|---|---|
| `http.py` | one shared GET helper with retries |
| `openaq.py` | OpenAQ v3: list locations, download sensor history |
| `openmeteo.py` | Open-Meteo: download hourly weather |

## `http.py`

`get_json(url, *, params, headers, timeout=90, retries=5, pause_s=1.0, rate_limit_wait_s=60)` returns parsed JSON.

| Situation | What happens |
|---|---|
| timeout or connection error | wait `min(10*attempt, 60)` s, retry |
| HTTP 429 (rate limit) | wait 60 s, retry |
| HTTP 5xx | wait `5*attempt` s, retry |
| other error (bad key, bad request) | raise `ApiError` with the reason the API gave |
| success | sleep `pause_s`, return the JSON |

After `retries` failed attempts it raises `ApiError("gave up ...")`.

## `openaq.py`

API: `https://api.openaq.org/v3`. Key in `.env` as `OPENAQ_API_KEY`. Free limits: 60 requests/minute, 2,000/hour (the code waits 1.1 s after each call).

### Commands

```
python -m hawa.ingestion.openaq locations
python -m hawa.ingestion.openaq download --location-id 3459 --start 2021-03-01 --end 2026-11-16 --params pm25
```

`locations` options: `--bbox` (default Kathmandu Valley), `--out`, `--show`, `--timeout`.
It writes `data/raw/openaq/kathmandu_locations.csv` with `location_id, name, provider, owner, is_monitor, lat, lon, first_utc, last_utc, history_days, days_since_last`. The first/last dates are for the whole location, not for one sensor.

`download` options:

| Option | Default | Use |
|---|---|---|
| `--location-id` | required | OpenAQ location |
| `--start`, `--end` | required | `YYYY-MM-DD` |
| `--params` | `pm25` | comma list, for example `pm25,temperature` |
| `--window-days` | 30 | request size. Use `7` if you get timeouts |
| `--endpoint` | `hours` | `hours` = hourly averages, `measurements` = original values |
| `--timeout` | 120 | seconds per request. Use `180` on a slow connection |
| `--out-dir` | `data/raw/openaq` | |

Output: `loc<ID>_sensor<SID>_<parameter>_<endpoint>.csv`. `period.datetimeFrom.utc` is the hour **start**.

### Functions (for use from other code)

| Function | Returns |
|---|---|
| `list_locations(bbox, timeout)` | DataFrame of PM2.5 locations |
| `get_sensors(location_id, timeout)` | list of sensors (one per parameter) at a location |
| `download_location(location_id, start, end, params, out_dir, window_days, endpoint, timeout, quiet)` | list of CSV paths written |

### Good to know

- Bounding box order is `min_lon, min_lat, max_lon, max_lat`. PM2.5 is `parameters_id=2`.
- A location can have several sensors. We pick the sensor by parameter name.
- Finished 30-day windows are cached as JSON in `<out-dir>/_cache/sensor<SID>_<endpoint>/`. Windows ending in the last 3 days are not cached, so they refresh.
- The AirNow-derived locations (Embassy 3459, Phora 3460) return many rows with an empty `value`. That is normal; the cleaning step handles it.
- The OpenAQ S3 archive (`openaq-data-archive`) is an alternative for bulk history. We have not used it.

## `openmeteo.py`

No key. Free for non-commercial use (about 10,000 calls/day). Licence CC BY 4.0, keep the attribution.

### Command

```
python -m hawa.ingestion.openmeteo --source archive --start 2017-03-01 --end 2026-11-05
python -m hawa.ingestion.openmeteo --source historical_forecast --start 2021-01-01 --end 2026-11-15
```

| `--source` | What | Range we verified |
|---|---|---|
| `archive` | ERA5 reanalysis | from 1940 (we use 2017+) |
| `historical_forecast` | archived real forecasts | works from 2021-01-01 |
| `forecast` | live forecast, up to 16 days ahead (for daily inference) | live |

Other options: `--lat`, `--lon` (default: US Embassy), `--variables` (comma list; default is `config.WEATHER_VARIABLES`), `--out-dir`.

Output: `openmeteo_<source>_<lat>_<lon>.csv` with `time` (UTC), the variables, `source`, `fetched_at_utc`.

### Good to know

- Requests go in 1-year chunks. Finished chunks are cached in `_cache/<source>/<lat>_<lon>_<start>_<end>.csv`. Chunks ending in the last 7 days are not cached.
- `precipitation` is the sum of the preceding hour; the rest are instant values.
- `boundary_layer_height` is incomplete (see `data/README.md`) and is not used in the first models.
- A `previous_runs` URL exists in the code (forecasts as seen 1–7 days earlier, from about Jan 2024). It is used only by `analysis/openmeteo_probe.py`.
