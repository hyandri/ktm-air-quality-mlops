import json

import pandas as pd
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from hawa import config
from hawa.api import app as api
from hawa.db import schema, store

ORIGIN = pd.Timestamp("2026-11-18T03:00:00Z")
NOW = pd.Timestamp("2026-11-18T04:20:00Z")           # 1.33 h after the origin: fresh


def _forecast_rows(sensor_id=11, name="Chovar", origin=ORIGIN, version="m1", base=50.0, created="2026-11-18T04:10:00Z"):
    h = list(range(1, 25))
    return pd.DataFrame({
        "location_id": 5, "sensor_id": sensor_id, "sensor_name": name, "model_version": version,
        "origin_utc": origin.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_hour_utc": [(origin + pd.Timedelta(hours=x)).strftime("%Y-%m-%dT%H:%M:%SZ") for x in h],
        "horizon": h, "forecast_pm25_sensor_scale": [base + x for x in h], "data_age_h": 0.0, "created_utc": created})


@pytest.fixture
def engine(db_engine):
    return db_engine


def _client(engine, now=NOW):
    return TestClient(api.create_app(engine, clock=lambda: now.to_pydatetime()))


# ----------------------------------------------------------------------------- store
def test_save_twice_does_not_duplicate_and_newer_values_win(engine):
    assert store.save_forecasts(engine, _forecast_rows(base=50.0)) == 24
    store.save_forecasts(engine, _forecast_rows(base=80.0))
    with engine.connect() as c:
        assert c.execute(sa.select(sa.func.count()).select_from(schema.forecasts)).scalar() == 24
    rows = store.latest_forecast(engine, 11)
    assert rows[0]["forecast_pm25_sensor_scale"] == 81.0


def test_latest_forecast_takes_the_newest_origin_in_order_of_horizon(engine):
    store.save_forecasts(engine, _forecast_rows(origin=ORIGIN - pd.Timedelta(hours=1), base=10.0))
    store.save_forecasts(engine, _forecast_rows(origin=ORIGIN, base=50.0))
    rows = store.latest_forecast(engine, 11)
    assert [r["horizon"] for r in rows] == list(range(1, 25))
    assert rows[0]["origin_utc"] == ORIGIN and rows[0]["forecast_pm25_sensor_scale"] == 51.0
    assert store.latest_forecast(engine, 999) is None


def test_same_origin_two_model_versions_gives_the_one_written_last(engine):
    store.save_forecasts(engine, _forecast_rows(version="old", base=10.0, created="2026-11-18T04:00:00Z"))
    store.save_forecasts(engine, _forecast_rows(version="new", base=90.0, created="2026-11-18T04:10:00Z"))
    assert store.latest_forecast(engine, 11)[0]["model_version"] == "new"


def test_list_sensors_and_empty_database(engine):
    assert store.list_sensors(engine) == [] and store.newest_origin(engine) is None
    store.save_forecasts(engine, _forecast_rows(11, "Chovar"))
    store.save_forecasts(engine, _forecast_rows(22, "Lagankhel", origin=ORIGIN - pd.Timedelta(hours=2)))
    out = store.list_sensors(engine)
    assert [s["sensor_name"] for s in out] == ["Chovar", "Lagankhel"]
    assert out[1]["newest_origin_utc"] == ORIGIN - pd.Timedelta(hours=2)


def test_save_run_roundtrip(engine):
    assert store.last_run(engine) is None
    store.save_run(engine, NOW, ORIGIN, "m1", 7, [("stale one", "newest real value is 9 h old")])
    r = store.last_run(engine)
    assert r["n_ok"] == 7 and r["n_skipped"] == 1 and r["skipped"] == [["stale one", "newest real value is 9 h old"]]
    assert r["origin_utc"] == ORIGIN


def test_load_csv_copies_old_forecasts(engine, tmp_path):
    path = tmp_path / "f.csv"
    pd.concat([_forecast_rows(11), _forecast_rows(22, "L")]).to_csv(path, index=False)
    assert store.load_csv(engine, path) == 48


# ------------------------------------------------------------------------------- API
def test_health_no_data_ok_stale_and_db_down(engine, tmp_path):
    assert _client(engine).get("/health").json()["status"] == "no_data"
    store.save_forecasts(engine, _forecast_rows())
    store.save_run(engine, NOW, ORIGIN, "m1", 8, [])
    body = _client(engine).get("/health").json()
    assert body["status"] == "ok" and body["newest_origin_age_h"] == 1.33
    assert body["last_run"]["n_ok"] == 8 and body["last_run"]["skipped"] == []
    late = _client(engine, NOW + pd.Timedelta(hours=config.API_STALE_AFTER_H)).get("/health").json()
    assert late["status"] == "stale"
    broken = sa.create_engine(f"sqlite:///{tmp_path / 'missing_dir' / 'x.db'}")
    r = _client(broken).get("/health")
    assert r.status_code == 503 and r.json() == {"status": "db_down"}


def test_forecast_endpoint_shape_and_honesty_fields(engine):
    store.save_forecasts(engine, _forecast_rows())
    r = _client(engine).get("/sensors/11/forecast/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["sensor_name"] == "Chovar" and body["model_version"] == "m1" and body["stale"] is False
    assert len(body["forecast"]) == 24 and body["forecast"][0]["horizon"] == 1
    assert body["forecast"][0]["target_hour_utc"].startswith("2026-11-18T04:00:00")
    assert "sensor's own scale" in body["unit"] and "Not calibrated" in body["note"]


def test_forecast_endpoint_marks_old_forecasts_stale_and_unknown_sensor_404(engine):
    store.save_forecasts(engine, _forecast_rows())
    old = _client(engine, NOW + pd.Timedelta(hours=10)).get("/sensors/11/forecast/latest").json()
    assert old["stale"] is True and old["age_h"] > 10
    assert _client(engine).get("/sensors/999/forecast/latest").status_code == 404
    assert _client(engine).get("/sensors/abc/forecast/latest").status_code == 422


def test_sensors_endpoint(engine):
    store.save_forecasts(engine, _forecast_rows(11, "Chovar"))
    body = _client(engine).get("/sensors").json()
    assert body == [{"sensor_id": 11, "sensor_name": "Chovar", "location_id": 5,
                     "newest_origin_utc": "2026-11-18T03:00:00Z"}]


def test_skipped_json_is_plain_text(engine):
    store.save_run(engine, NOW, ORIGIN, "m1", 0, [("a", "x")])
    with engine.connect() as c:
        assert json.loads(c.execute(sa.select(schema.batch_runs.c.skipped)).scalar()) == [["a", "x"]]


def test_api_creates_the_tables_at_start_so_an_empty_database_is_no_data_not_db_down(tmp_path):
    fresh = sa.create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")        # no tables yet
    with TestClient(api.create_app(fresh, clock=lambda: NOW.to_pydatetime())) as client:   # `with` runs the start-up
        assert client.get("/health").json()["status"] == "no_data"
