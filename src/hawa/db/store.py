"""Read and write the database. Small functions, no ORM.

`save_forecasts` is an upsert on the table key: running the job twice for the same hour never duplicates rows,
and the newer values replace the older ones (same behaviour as the CSV).
"""
import json

import pandas as pd
import sqlalchemy as sa

from hawa import config
from hawa.db import schema

KEY_COLUMNS = ["sensor_id", "model_version", "origin_utc", "horizon"]
FORECAST_COLUMNS = ["location_id", "sensor_name", "target_hour_utc", "forecast_pm25_sensor_scale", "data_age_h",
                    "created_utc"]


def make_engine(url=None):
    return sa.create_engine(url or config.DATABASE_URL, pool_pre_ping=True)


def init_db(engine):
    """Create the tables if they do not exist. Safe to run many times."""
    schema.metadata.create_all(engine)


def utc(value):
    """A datetime that is always timezone-aware UTC (SQLite returns naive values)."""
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _upsert(engine, table, rows, key_columns):
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif engine.dialect.name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:
        raise NotImplementedError(f"upsert not written for {engine.dialect.name}")
    stmt = insert(table)
    update = {c.name: stmt.excluded[c.name] for c in table.columns if c.name not in key_columns}
    stmt = stmt.on_conflict_do_update(index_elements=key_columns, set_=update)
    with engine.begin() as conn:
        conn.execute(stmt, rows)


def save_forecasts(engine, df):
    """Write forecast rows (the same columns as forecasts.csv). Returns the number of rows."""
    if df is None or len(df) == 0:
        return 0
    d = df.copy()
    for col in ("origin_utc", "target_hour_utc", "created_utc"):
        d[col] = [utc(v).to_pydatetime() for v in d[col]]
    d = d[KEY_COLUMNS + FORECAST_COLUMNS]
    d = d.astype(object).where(d.notna(), None)    # NaN -> NULL
    rows = d.to_dict("records")
    for r in rows:
        r["sensor_id"], r["location_id"], r["horizon"] = int(r["sensor_id"]), int(r["location_id"]), int(r["horizon"])
    _upsert(engine, schema.forecasts, rows, KEY_COLUMNS)
    return len(rows)


def save_run(engine, created_utc, origin_utc, model_version, n_ok, skipped):
    with engine.begin() as conn:
        conn.execute(sa.insert(schema.batch_runs).values(
            created_utc=utc(created_utc).to_pydatetime(), origin_utc=utc(origin_utc).to_pydatetime(),
            model_version=model_version, n_ok=int(n_ok), n_skipped=len(skipped),
            skipped=json.dumps([list(s) for s in skipped])))


def load_csv(engine, path):
    """Copy an existing forecasts.csv into the database (one-time backfill). Returns the number of rows."""
    return save_forecasts(engine, pd.read_csv(path))


# ------------------------------------------------------------------------------ queries
def list_sensors(engine):
    """One row per sensor with its newest forecast origin."""
    f = schema.forecasts
    q = (sa.select(f.c.sensor_id, sa.func.max(f.c.sensor_name).label("sensor_name"),
                   sa.func.max(f.c.location_id).label("location_id"),
                   sa.func.max(f.c.origin_utc).label("newest_origin_utc"))
         .group_by(f.c.sensor_id).order_by(f.c.sensor_id))
    with engine.connect() as conn:
        return [dict(r._mapping, newest_origin_utc=utc(r.newest_origin_utc)) for r in conn.execute(q)]


def latest_forecast(engine, sensor_id):
    """The newest forecast of one sensor: rows sorted by horizon, or None if the sensor has none.

    "Newest" = the latest origin; if two model versions share it, the one written last.
    """
    f = schema.forecasts
    with engine.connect() as conn:
        top = conn.execute(sa.select(f.c.model_version, f.c.origin_utc).where(f.c.sensor_id == sensor_id)
                           .order_by(f.c.origin_utc.desc(), f.c.created_utc.desc()).limit(1)).first()
        if top is None:
            return None
        rows = conn.execute(sa.select(f).where(f.c.sensor_id == sensor_id, f.c.model_version == top.model_version,
                                              f.c.origin_utc == top.origin_utc).order_by(f.c.horizon)).all()
    out = []
    for r in rows:
        d = dict(r._mapping)
        for k in ("origin_utc", "target_hour_utc", "created_utc"):
            d[k] = utc(d[k])
        out.append(d)
    return out


def newest_origin(engine):
    with engine.connect() as conn:
        value = conn.execute(sa.select(sa.func.max(schema.forecasts.c.origin_utc))).scalar()
    return None if value is None else utc(value)


def last_run(engine):
    b = schema.batch_runs
    with engine.connect() as conn:
        r = conn.execute(sa.select(b).order_by(b.c.created_utc.desc(), b.c.id.desc()).limit(1)).first()
    if r is None:
        return None
    d = dict(r._mapping)
    d["skipped"] = json.loads(d["skipped"])
    for k in ("created_utc", "origin_utc"):
        d[k] = utc(d[k])
    return d
