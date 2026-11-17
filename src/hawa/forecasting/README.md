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

## Output columns

`location_id, sensor_id, sensor_name, model_version, origin_utc, target_hour_utc, horizon,
forecast_pm25_sensor_scale, data_age_h, created_utc`

## Important limits

- Values are on each sensor's own scale. AirGradient sensors read about 1.56x the Embassy, and the model learned the Embassy.
  They are NOT calibrated yet. Do not read them as health warnings.
- The model is the all-data bundle (not the frozen one). Winter 2025/26 was a locked Embassy test winter, so any scale
  correction against a reference is a new experiment.
- Forecasts are kept so that, later, we can compare them with what really happened (monitoring).
