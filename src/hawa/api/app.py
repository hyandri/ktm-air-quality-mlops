"""FastAPI service: serves the newest saved forecasts from the database.

Run from the project root (database running, tables created):
    uvicorn hawa.api.app:app --reload          then open http://127.0.0.1:8000/docs

Endpoints
    GET /health                              service alive? database reachable? how old is the newest forecast?
    GET /sensors                             sensors that have forecasts, with their newest forecast time
    GET /sensors/{sensor_id}/forecast/latest the next 24 hours for one sensor

The API only READS. The batch job (hawa.forecasting.batch --to-db) writes.
Values are PM2.5 in ug/m3 on each sensor's OWN scale: not calibrated, not a health warning.
"""
from datetime import datetime, timezone

import sqlalchemy as sa
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from hawa import config
from hawa.db import store

UNIT = "ug/m3, on the sensor's own scale"
NOTE = "Not calibrated: AirGradient sensors read about 1.56x higher than the US Embassy. Not a health warning."


class ForecastPoint(BaseModel):
    horizon: int
    target_hour_utc: datetime
    pm25: float


class ForecastResponse(BaseModel):
    sensor_id: int
    sensor_name: str
    model_version: str
    origin_utc: datetime
    created_utc: datetime
    data_age_h: float | None
    age_h: float
    stale: bool
    unit: str = UNIT
    note: str = NOTE
    forecast: list[ForecastPoint]


class SensorOut(BaseModel):
    sensor_id: int
    sensor_name: str
    location_id: int
    newest_origin_utc: datetime


def _hours(delta):
    return round(delta.total_seconds() / 3600, 2)


def create_app(engine=None, clock=None):
    """`engine` and `clock` can be replaced in tests."""
    engine = engine or store.make_engine()
    clock = clock or (lambda: datetime.now(timezone.utc))
    app = FastAPI(title="Hawa: Kathmandu PM2.5 forecast", version="0.1")

    @app.get("/health")
    def health():
        try:
            newest = store.newest_origin(engine)
            run = store.last_run(engine)
        except sa.exc.SQLAlchemyError:
            return JSONResponse(status_code=503, content={"status": "db_down"})
        if newest is None:
            return {"status": "no_data", "newest_origin_utc": None, "newest_origin_age_h": None, "last_run": None}
        age = _hours(store.utc(clock()) - newest)
        last = None if run is None else {
            "created_utc": run["created_utc"].isoformat(), "model_version": run["model_version"],
            "n_ok": run["n_ok"], "n_skipped": run["n_skipped"], "skipped": run["skipped"]}
        return {"status": "ok" if age <= config.API_STALE_AFTER_H else "stale",
                "newest_origin_utc": newest.isoformat(), "newest_origin_age_h": age, "last_run": last}

    @app.get("/sensors", response_model=list[SensorOut])
    def sensors():
        return store.list_sensors(engine)

    @app.get("/sensors/{sensor_id}/forecast/latest", response_model=ForecastResponse)
    def latest(sensor_id: int):
        rows = store.latest_forecast(engine, sensor_id)
        if not rows:
            raise HTTPException(status_code=404, detail=f"no forecast for sensor {sensor_id}")
        first = rows[0]
        age = _hours(store.utc(clock()) - first["origin_utc"])
        return ForecastResponse(
            sensor_id=first["sensor_id"], sensor_name=first["sensor_name"], model_version=first["model_version"],
            origin_utc=first["origin_utc"], created_utc=first["created_utc"], data_age_h=first["data_age_h"],
            age_h=age, stale=age > config.API_STALE_AFTER_H,
            forecast=[ForecastPoint(horizon=r["horizon"], target_hour_utc=r["target_hour_utc"],
                                    pm25=r["forecast_pm25_sensor_scale"]) for r in rows])

    return app


app = create_app()
