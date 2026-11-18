# api/

A small read-only web service (FastAPI) on top of the database. The batch job writes; the API only reads.

```bash
uvicorn hawa.api.app:app --reload       # then open http://127.0.0.1:8000/docs  (interactive page, try the calls there)
```

| Endpoint | Returns |
|---|---|
| `GET /health` | `status`: `ok`, `stale` (newest forecast older than 3 h), `no_data`, or `db_down` (HTTP 503); newest forecast age; the last batch run (sensors ok / skipped and why) |
| `GET /sensors` | sensors that have forecasts and their newest forecast time |
| `GET /sensors/{sensor_id}/forecast/latest` | the next 24 hours for one sensor, with `age_h` and a `stale` flag. 404 if the sensor has none |

Every forecast answer carries `unit` and `note`: values are on the sensor's own scale, not calibrated, not a health warning.

Try it:

```bash
curl -s localhost:8000/health
curl -s localhost:8000/sensors
curl -s localhost:8000/sensors/6093549/forecast/latest      # use a sensor_id from /sensors
```

`create_app(engine, clock)` is a factory, so tests give it a temporary database and a fixed clock.
The stale limit is `config.API_STALE_AFTER_H`.
