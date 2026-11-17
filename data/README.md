# Data folder

What is in `data/`, where it came from, and how to rebuild it.
Data files are **not in git** (see `.gitignore`); this README is. Later we track data with DVC.

## Folder map

```
data/
├── raw/          untouched downloads. Never edit by hand.
│   ├── open_data_nepal/   old Embassy + Phora files (2017 → Mar 2021)
│   ├── openaq/            OpenAQ API downloads (Embassy, Phora, AirGradient sensors)
│   └── openmeteo/         weather downloads
├── interim/      half-processed data (empty for now)
└── processed/    clean files ready for modelling
```

Rule: `raw` is only written by `hawa.ingestion`, `processed` only by `hawa.processing`.
If a file in `processed` looks wrong, fix the code and rebuild it. Do not edit the file.

## Conventions (all files)

- **Time is UTC.** Kathmandu local time is UTC+05:45.
- **Hourly timestamps mean the START of the hour** in `processed/` and in OpenAQ files.
  Old Open Data Nepal files use the END of the hour; `hawa.processing.clean` shifts them back 1 hour.
- **PM2.5 unit:** µg/m³.
- **Missing data:** `processed/` PM2.5 files contain valid hours only (missing hours are absent rows).
  Weather files have all hours; a missing value is empty (NaN).

---

## `raw/open_data_nepal/`

| File | What |
|---|---|
| `us_embassy.csv` | US Embassy Kathmandu, AirNow data copied to Open Data Nepal (CKAN) |
| `phora_durbar.csv` | US Phora Durbar monitor, same source |

- **Source:** https://opendatanepal.com/dataset/air-quality-data-in-kathmandu
- **Licence:** CC BY-SA (as listed on the portal; keep attribution).
- **Format:** long format. Columns: `locationId, location, city, country, utc, local, parameter, value, unit, latitude, longitude`. One row per parameter per hour. We use `parameter == "pm25"`.
- **Period:** 2017-03-02 → 2021-03-12 (about 31.4k PM2.5 rows for Embassy, 31.8k for Phora).
- **Missing marker:** `-999`. Also seen: error value `985`, and some negative values. All are dropped.
- **Timestamps:** `utc` ends at minute `:15` and marks the END of the hour.
- AirNow stopped publishing the Embassy feed on 4 Mar 2025, so this source is history only.

## `raw/openaq/`

| File | What |
|---|---|
| `kathmandu_locations.csv` | every PM2.5 location in the Kathmandu Valley box (62 when we checked) |
| `loc3459_sensor7710_pm25_measurements.csv` | Embassy, 2021-02 → 2026-06 |
| `loc3460_sensor7711_pm25_measurements.csv` | Phora Durbar, 2021-02 → 2025-03 |
| `loc<ID>_sensor<SID>_pm25_hours.csv` | one AirGradient (or other) sensor, hourly averages |
| `scorecard_v2.csv` | sensor ratings with corrected counting (use this one) |
| `scorecard.csv` | **old, wrong**: counted rows without values as data. Do not use. |
| `_cache/` | raw API answers (JSON), so downloads can resume |

- **Source:** OpenAQ API v3. Embassy/Phora here are the AirNow data re-published by OpenAQ.
  Shifted by +1 hour, OpenAQ values equal the Open Data Nepal values (100% identical on the overlap).
- **Columns** (flattened JSON): `value`, `period.datetimeFrom.utc` (hour START), `period.datetimeTo.utc`, `flagInfo.hasFlags`, `coverage.*`, plus `location_id, sensor_id, parameter_name, fetched_at_utc`.
- **Licence:** to confirm per provider. AirGradient network reported as CC BY 4.0; the AirNow-derived data needs a check before we publish anything.
- **Important:** many rows have an empty `value`. Count valid hours, not rows.

### Live AirGradient sensors: shortlist (snapshot 16 Nov 2026, from `scorecard_v2.csv`)

Ranked by missing hours, with winter (Dec–Feb) gaps in mind.

| Sensor (location id) | Missing | Missing in winter |
|---|---|---|
| Golfutar SC-17 (6093549) | 8.4% | 0.5% |
| Chovar SC-07 (5676345) | 12.6% | 1.1% |
| Lagankhel SC-05 (5509788) | 14.7% | 0.6% |
| Sundarighat SC-23 (6094668) | 16.4% | 3.1% |
| Kritipur Ward 3 SC-25 (6094706) | 15.2% | 10.1% |
| Shantichowk SC-11 (6070692) | 17.3% | 9.0% |
| Tankeshwor SC-18 (6093550) | 19.9% | 11.9% |
| Kadhaghari SC-42 (6128821) | 19.5% | 10.2% |

Avoid (half or more of the hours missing): Nakhipot, Purano naikap, Balkumari, Haugal Ganesh Temple, Ramkot, Tripureshowr.
These sensors started in late 2025, so each has about one winter of history. Low missing data does not prove accurate readings; they still need a calibration check against a reference station.

## `raw/openmeteo/`

| File | What | Coverage |
|---|---|---|
| `openmeteo_archive_27.738703_85.336205.csv` | ERA5 reanalysis ("what the weather really was") | 2017-03-01 → 2026-11-05, 84,888 rows |
| `openmeteo_historical_forecast_27.738703_85.336205.csv` | archived real forecasts (what a live model would have seen) | 2021-01-01 → 2026-11-15, 51,480 rows (no gaps) |

- **Source:** Open-Meteo. **Licence:** CC BY 4.0, keep the attribution. Free for non-commercial use (about 10,000 calls/day).
- **Location:** one point, the US Embassy (27.738703, 85.336205). We assume Phora (about 3 km away) shares the same weather grid cell; not yet verified.
- **Columns:** `time` (UTC) and 11 variables: `temperature_2m, relative_humidity_2m, dew_point_2m, surface_pressure, precipitation, cloud_cover, wind_speed_10m, wind_direction_10m, wind_gusts_10m, shortwave_radiation, boundary_layer_height`, then `source`, `fetched_at_utc`.
- **Units** (Open-Meteo defaults): °C, %, hPa, mm, km/h, degrees, W/m², m.
- **`precipitation` is the sum of the preceding hour.** The other variables are instant values.
- **Boundary layer height (BLH) is unreliable:** ERA5 is missing one 182-day block in 2024 (4,368 hours); the archived forecasts miss 62.4% of hours (empty in the early years). Decision: BLH is **not used** in the first models.
- Train/serve mismatch: models are trained on ERA5 or archived forecasts, but predict from live forecasts. Archived forecasts (2021+) match live use better; ERA5 is the fallback for 2017–2020.

## `processed/`

| File | Rows | Range (UTC) |
|---|---|---|
| `embassy_pm25_hourly_clean.csv` | 58,106 | 2017-03-02 22:00 → 2026-06-17 22:00 |
| `phora_pm25_hourly_clean.csv` | 46,775 | 2017-03-02 22:00 → 2025-03-24 13:00 |

- **Columns:** `timestamp_utc` (hour start), `station`, `pm25`, `source` (`open_data_nepal` or `openaq`).
- **Valid hours only.** Build a full hourly grid before making features.
- **Rebuild:** `python -m hawa.processing.clean` (see `src/hawa/processing/README.md` for the rules).

### Winter coverage (Dec–Feb, share of hours with a valid value)

| Winter | Embassy | Phora |
|---|---|---|
| 2017/18 | 56% | 52% |
| 2018/19 | 89% | 92% |
| 2019/20 | 80% | 87% |
| 2020/21 | 78% | 95% |
| 2021/22 | 95% | 68% |
| 2022/23 | **2%** | 38% |
| 2023/24 | 94% | 35% |
| 2024/25 | 88% | 70% |
| 2025/26 | 82% | 0% (feed ended) |

### Known problems

- **Embassy:** winter 2022/23 is almost empty. A 156-day gap runs from Mar to Aug 2025.
- **Phora:** a stuck sensor shows exactly `92.0` for 448 hours (6–26 Jan 2024); it is removed. Readings also drift in 2024–25 (correlation with Embassy 0.59 in 2025, against 0.75–0.91 earlier). Use Phora mainly for the early years.
- **Phora/Embassy in 2017:** correlation only 0.55, so treat the first year with care.
- **Flagged rows:** 26.8% of Embassy rows are flagged. We checked 709 flagged values against their neighbours (median difference −0.5 µg/m³) and they look normal, so they are kept.

## Planned split (chronological, to confirm)

- **Train:** 2017 → 2022, winters 2017/18 to 2021/22.
- **Skip:** winter 2022/23 (almost empty for Embassy).
- **Validation:** winter 2023/24.
- **Test:** winters 2024/25 and 2025/26. Look at these only at the end.
- **Live test:** winter 2026/27 with the AirGradient shortlist.
