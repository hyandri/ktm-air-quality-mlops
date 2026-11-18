# db/

PostgreSQL storage for forecasts. Plain SQLAlchemy Core, no ORM, no migration tool (see `docs/decisions.md` O13).

| File | What it does |
|---|---|
| `schema.py` | the two tables: `forecasts` and `batch_runs` |
| `store.py` | `make_engine`, `init_db`, `save_forecasts` (upsert), `save_run`, `load_csv`, and the read queries |
| `init.py` | command line: create the tables, optionally copy an old `forecasts.csv` into them |

## Tables

**`forecasts`** one row per (sensor, model version, origin, horizon). That tuple is the primary key, so running the
job twice for the same hour replaces rows instead of adding them. Columns: `sensor_id, model_version, origin_utc,
horizon` (key), `location_id, sensor_name, target_hour_utc, forecast_pm25_sensor_scale, data_age_h, created_utc`.

**`batch_runs`** one row per job run: `created_utc, origin_utc, model_version, n_ok, n_skipped, skipped` (JSON text
with the sensors that were skipped and why). `GET /health` shows the last one.

All times are UTC. Values are on the sensor's own scale (not calibrated).

## Commands

```bash
docker compose up -d db                                    # start PostgreSQL (data survives restarts)
python -m hawa.db.init                                     # create the tables (safe to repeat)
python -m hawa.db.init --from-csv data/forecasts/forecasts.csv   # one-time: copy the old CSV in
python -m hawa.forecasting.batch --to-db                   # forecast and save to CSV and database
docker compose exec db psql -U hawa -d hawa -c "select sensor_name, count(*) from forecasts group by 1"
```

The connection comes from `DATABASE_URL` (environment or `.env`). Default: the docker-compose database on port 5433.

## Tests

By default the tests use a temporary SQLite file. To run them on real PostgreSQL, point them at an EMPTY test
database (its tables are dropped and recreated before every test):

```bash
docker compose exec db createdb -U hawa hawa_test
HAWA_TEST_DATABASE_URL=postgresql+psycopg://hawa:hawa@localhost:5433/hawa_test python -m pytest -q tests/test_db_api.py
```
