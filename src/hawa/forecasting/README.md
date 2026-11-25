# forecasting/

Live batch job: forecast the next 24 hours for every sensor in `config.LIVE_SENSORS`.

```bash
python -m hawa.forecasting.batch                      # needs OPENAQ_API_KEY in .env
python -m hawa.forecasting.batch --model models/embassy_to_2026-06-17
python -m hawa.forecasting.batch --now 2026-11-16T10:20:00Z --out /tmp/test.csv   # pretend time
```

## What one run does

1. Origin = the last COMPLETE hour (now floored to the hour, minus 1 h). The running hour is never used.
2. For each sensor: fetch the last 240 h from OpenAQ, clean it with the same code as training, build the 24 features
   with the same `add_features` as training (no separate live code, so no training/serving skew).
3. Skip a sensor if its newest real value is older than 6 h (`MAX_DATA_AGE_H`). The reason is printed. One bad sensor never stops the others.
4. Predict h1..h24 with the model bundle. Horizon h is the value for the hour starting at origin + h.
5. Append to `data/forecasts/forecasts.csv`. The key is (sensor_id, model_version, origin_utc, horizon),
   so running twice in the same hour does not duplicate rows.

The `model_version` label is `<station>_to_<last training origin date>`, read from the bundle's metadata,
so the DVC bundle `models/embassy_latest` still gets a label that says which data it was trained on.

## Output columns

`location_id, sensor_id, sensor_name, model_version, origin_utc, target_hour_utc, horizon,
forecast_pm25_sensor_scale, data_age_h, created_utc`

## `loop.py`: the hourly schedule

```bash
python -m hawa.forecasting.loop --model models/embassy_latest
```

Runs `batch --to-db` and then `actuals --to-db` once at start and then at minute 10 of every hour, forever. One failed run is printed and
the loop tries again next hour. This is what the `forecaster` container runs (`docs/docker.md`).
`batch` also warns when the installed LightGBM / scikit-learn / pandas differ from the ones that trained the model.

## `actuals.py`: forecast vs real

```bash
python -m hawa.forecasting.actuals            # CSV in, reports/live_eval.csv out, prints MAE per horizon group
python -m hawa.forecasting.actuals --to-db    # last 7 days from the database, saved into the live_eval table
```

The hourly loop runs the `--to-db` form after every forecast, so `GET /metrics` is always fresh.

Fetches the real values again and compares, on the same rows: the model, persistence (value at the origin hour),
and same hour yesterday. Forecast hours with no real value yet are left out. After one day only the first
horizons have real values; after a few days all 24 do. Few rows means a smoke test, not a result.

## Important limits

- Values are on each sensor's own scale. AirGradient sensors read about 1.56x the Embassy, and the model learned the Embassy.
  They are NOT calibrated yet. Do not read them as health warnings.
- The model is the all-data bundle (not the frozen one). Winter 2025/26 was a locked Embassy test winter, so any scale
  correction against a reference is a new experiment.
- Forecasts are kept so that, later, we can compare them with what really happened (monitoring).
