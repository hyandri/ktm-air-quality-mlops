"""Database tables (SQLAlchemy Core). Works on PostgreSQL (real use) and SQLite (tests).

forecasts   one row per (sensor, model version, origin, horizon). The key makes a re-run idempotent.
batch_runs  one row per batch job run: when, which origin and model, how many sensors ok / skipped and why.
            The API health check reads it.

All times are UTC. Values are PM2.5 on the SENSOR's own scale (not calibrated, see docs/decisions.md O8, O10).
"""
import sqlalchemy as sa

metadata = sa.MetaData()

forecasts = sa.Table(
    "forecasts", metadata,
    sa.Column("sensor_id", sa.BigInteger, primary_key=True),
    sa.Column("model_version", sa.String(80), primary_key=True),
    sa.Column("origin_utc", sa.DateTime(timezone=True), primary_key=True),
    sa.Column("horizon", sa.SmallInteger, primary_key=True),
    sa.Column("location_id", sa.BigInteger, nullable=False),
    sa.Column("sensor_name", sa.String(120), nullable=False),
    sa.Column("target_hour_utc", sa.DateTime(timezone=True), nullable=False),
    sa.Column("forecast_pm25_sensor_scale", sa.Float, nullable=False),
    sa.Column("data_age_h", sa.Float),
    sa.Column("created_utc", sa.DateTime(timezone=True), nullable=False),
    sa.Index("ix_forecasts_sensor_origin", "sensor_id", "origin_utc"),
)

batch_runs = sa.Table(
    "batch_runs", metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("created_utc", sa.DateTime(timezone=True), nullable=False),
    sa.Column("origin_utc", sa.DateTime(timezone=True), nullable=False),
    sa.Column("model_version", sa.String(80), nullable=False),
    sa.Column("n_ok", sa.Integer, nullable=False),
    sa.Column("n_skipped", sa.Integer, nullable=False),
    sa.Column("skipped", sa.Text, nullable=False),   # JSON text: [[sensor name, reason], ...]
)
