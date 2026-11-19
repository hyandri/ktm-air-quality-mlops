import json

import numpy as np
import pandas as pd
import pytest

from hawa import config
from hawa.forecasting import batch
from hawa.ingestion import openaq
from hawa.models import bundle, train
from hawa.processing import build_dataset as b
from hawa.processing import features

NOW = pd.Timestamp("2026-11-16T10:20:00Z")


def _wave(idx, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(60 + 30 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + rng.normal(0, 4, len(idx)), index=idx)


@pytest.fixture(scope="module")
def model_bundle():
    old = dict(config.LIGHTGBM_PARAMS)
    config.LIGHTGBM_PARAMS.update(n_estimators=40, learning_rate=0.3)
    idx = pd.date_range("2023-01-01", "2024-06-01", freq="h", tz="UTC")
    df = features.add_features(b.add_targets(b.add_time_features(pd.DataFrame({"pm25": _wave(idx)}))))
    models, meta = train.train(df)
    config.LIGHTGBM_PARAMS.clear()
    config.LIGHTGBM_PARAMS.update(old)
    return bundle.Bundle(models=models, metadata=meta)


def _raw(last_hour_start, hours=240, seed=1):
    idx = pd.date_range(end=last_hour_start, periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({"period.datetimeFrom.utc": idx.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "value": _wave(idx, seed).to_numpy()})


def test_last_complete_hour():
    assert batch.last_complete_hour(NOW) == pd.Timestamp("2026-11-16T09:00:00Z")
    assert batch.last_complete_hour(pd.Timestamp("2026-11-16T10:00:00Z")) == pd.Timestamp("2026-11-16T09:00:00Z")


def test_forecast_has_24_rows_with_the_right_target_hours(model_bundle):
    origin = batch.last_complete_hour(NOW)
    series = _raw(origin).set_index(pd.to_datetime(_raw(origin)["period.datetimeFrom.utc"], utc=True))["value"]
    fc, reason = batch.forecast_sensor(model_bundle, series, origin)
    assert reason == "ok" and len(fc) == 24
    assert fc["target_hour_utc"].iloc[0] == origin + pd.Timedelta(hours=1)
    assert fc["target_hour_utc"].iloc[-1] == origin + pd.Timedelta(hours=24)
    assert (fc["forecast_pm25_sensor_scale"] >= 0).all() and fc["data_age_h"].iloc[0] == 0


def test_live_features_equal_the_training_features(model_bundle):
    """The batch job builds its inputs with the same code as training: a long history gives the same row."""
    idx = pd.date_range("2026-10-01", "2026-11-16T09:00:00", freq="h", tz="UTC")
    series = _wave(idx, 5)
    origin = idx[-1]
    live = batch.live_features(series, origin)
    full = features.add_features(b.add_time_features(pd.DataFrame({"pm25": series})))
    cols = model_bundle.metadata["feature_columns"]
    pd.testing.assert_series_equal(live.iloc[0][cols], full.iloc[-1][cols], check_names=False)


def test_stale_sensor_is_skipped_and_others_still_run(model_bundle, tmp_path):
    sensors = [{"name": "fresh", "location_id": 1, "sensor_id": 11},
               {"name": "stale", "location_id": 2, "sensor_id": 22},
               {"name": "broken", "location_id": 3, "sensor_id": 33},
               {"name": "empty", "location_id": 4, "sensor_id": 44}]
    origin = batch.last_complete_hour(NOW)

    def fetch(s, now):
        if s["name"] == "fresh":
            return _raw(origin)
        if s["name"] == "stale":
            return _raw(origin - pd.Timedelta(hours=20))
        if s["name"] == "empty":
            return pd.DataFrame()
        raise RuntimeError("network down")

    out = tmp_path / "f.csv"
    n_ok, skipped, _ = batch.run(sensors, model_bundle, "m1", NOW, out, fetch=fetch)
    assert n_ok == 1 and {n for n, _ in skipped} == {"stale", "broken", "empty"}
    written = pd.read_csv(out)
    assert set(written["sensor_name"]) == {"fresh"} and len(written) == 24
    assert dict(skipped)["stale"].startswith("newest real value is")


def test_rerun_same_hour_makes_no_duplicates_but_new_hour_appends(model_bundle, tmp_path):
    sensors = [{"name": "a", "location_id": 1, "sensor_id": 11}]
    out = tmp_path / "f.csv"
    fetch = lambda s, now: _raw(batch.last_complete_hour(now))                      # noqa: E731
    batch.run(sensors, model_bundle, "m1", NOW, out, fetch=fetch)
    batch.run(sensors, model_bundle, "m1", NOW + pd.Timedelta(minutes=5), out, fetch=fetch)
    assert len(pd.read_csv(out)) == 24
    batch.run(sensors, model_bundle, "m1", NOW + pd.Timedelta(hours=1), out, fetch=fetch)
    assert len(pd.read_csv(out)) == 48


def test_fetch_recent_hours_builds_the_request_and_flattens(monkeypatch):
    seen = {}

    def fake_pages(path, params, timeout=120):
        seen.update(path=path, **params)
        return [{"value": 5.0, "period": {"datetimeFrom": {"utc": "2026-11-16T08:00:00Z"}}}]

    monkeypatch.setattr(openaq, "_get_all_pages", fake_pages)
    df = openaq.fetch_recent_hours(7, 99, hours_back=48, now=NOW.to_pydatetime())
    assert seen["path"] == "/sensors/99/hours" and seen["datetime_to"] == "2026-11-16T10:20:00Z"
    assert seen["datetime_from"] == "2026-11-14T10:20:00Z"
    assert list(df["location_id"]) == [7] and "period.datetimeFrom.utc" in df.columns


def test_find_latest_bundle_ignores_frozen_ones(tmp_path):
    cases = (("a", None, "2026-01-01"), ("b", None, "2026-06-01"), ("c", "2024-12-01", "2026-09-01"))
    for name, until, created in cases:
        d = tmp_path / name
        d.mkdir()
        meta = {"train_until_local_date": until, "created_utc": created}
        (d / "metadata.json").write_text(json.dumps(meta))
    assert batch.find_latest_bundle(tmp_path).name == "b"


def test_version_name_comes_from_the_data_not_the_folder():
    meta = {"station": "embassy", "train_last_origin_utc": "2026-06-17T20:00:00+00:00"}
    assert batch.version_name(meta, "embassy_latest") == "embassy_to_2026-06-17"
    assert batch.version_name({}, "embassy_latest") == "embassy_latest"


def test_run_with_engine_writes_the_database_too(model_bundle, tmp_path, db_engine):
    from hawa.db import store

    engine = db_engine
    sensors = [{"name": "a", "location_id": 1, "sensor_id": 11}, {"name": "gone", "location_id": 2, "sensor_id": 22}]

    def fetch(s, now):
        return _raw(batch.last_complete_hour(now)) if s["name"] == "a" else None

    batch.run(sensors, model_bundle, "m1", NOW, tmp_path / "f.csv", fetch=fetch, engine=engine)
    later = NOW + pd.Timedelta(minutes=5)
    batch.run(sensors, model_bundle, "m1", later, tmp_path / "f.csv", fetch=fetch, engine=engine)
    rows = store.latest_forecast(engine, 11)
    assert len(rows) == 24 and rows[0]["model_version"] == "m1"
    assert store.latest_forecast(engine, 22) is None
    assert store.last_run(engine)["n_ok"] == 1 and store.last_run(engine)["skipped"][0][0] == "gone"


def test_version_warnings_name_the_library_that_differs():
    import lightgbm

    same = {"versions": {"lightgbm": lightgbm.__version__}}
    assert batch.version_warnings(same) == []
    other = {"versions": {"lightgbm": "0.0.1", "pandas": pd.__version__}}
    out = batch.version_warnings(other)
    assert len(out) == 1 and out[0].startswith("lightgbm: model trained with 0.0.1")
    assert batch.version_warnings({}) == []
