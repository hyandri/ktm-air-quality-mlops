import numpy as np
import pandas as pd
import pytest

from hawa import config
from hawa.analysis import sensor_transfer as st
from hawa.models import bundle, train
from hawa.processing import build_dataset as b
from hawa.processing import features


def _series(start, end, seed=0, offset=0.0, scale=1.0):
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    base = 60 + 30 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + rng.normal(0, 4, len(idx))
    return pd.Series(np.clip(base * scale + offset, 1, None), index=idx)


def test_haversine_embassy_to_phora_is_about_3_5_km():
    d = st.haversine_km(27.738703, 85.336206, 27.712464, 85.315703)
    assert 3.2 < d < 3.8


def test_sensor_files_finds_only_hour_files(tmp_path):
    for name in ("loc123_sensor45_pm25_hours.csv", "loc9_sensor8_pm25_measurements.csv", "notes.txt"):
        (tmp_path / name).write_text("x")
    assert st.sensor_files(tmp_path) == [(123, tmp_path / "loc123_sensor45_pm25_hours.csv")]


def test_agreement_for_a_scaled_copy():
    emb = _series("2025-11-01", "2026-03-01")
    sensor = 2.0 * emb + 5.0
    a = st.agreement(sensor, emb, 2025)
    assert a["corr"] == pytest.approx(1.0) and a["ratio"] > 1.9
    assert a["bias"] > 0 and a["overlap_h"] > 2000


def test_agreement_with_too_little_overlap_is_nan():
    emb = _series("2025-12-01", "2025-12-03")
    a = st.agreement(emb, emb, 2025, min_hours=200)
    assert a["overlap_h"] > 0 and np.isnan(a["corr"])


def test_transfer_score_on_a_predictable_sensor(tmp_path, monkeypatch):
    monkeypatch.setitem(config.LIGHTGBM_PARAMS, "n_estimators", 60)
    monkeypatch.setitem(config.LIGHTGBM_PARAMS, "learning_rate", 0.3)     # few trees, so learn fast
    train_series = _series("2023-01-01", "2024-06-01", seed=1)
    df = features.add_features(b.add_targets(b.add_time_features(pd.DataFrame({"pm25": train_series}))))
    models, meta = train.train(df)
    mb = bundle.Bundle(models=models, metadata=meta)

    sensor = _series("2025-10-01", "2026-03-01", seed=2)
    out = st.transfer_score(mb, st.feature_table(sensor), 2025, min_origins=500)
    assert out["n_origins"] > 1500 and out["gain_pct"] > 0       # a model that learned the pattern should beat the bar
    few = st.transfer_score(mb, st.feature_table(sensor.iloc[:200]), 2025, min_origins=500)
    assert np.isnan(few["gain_pct"])                              # no winter origins in 200 hours of October


def test_find_frozen_bundle(tmp_path):
    for name, until in (("a", None), ("b", st.FROZEN_UNTIL)):
        d = tmp_path / name
        d.mkdir()
        (d / "metadata.json").write_text('{"train_until_local_date": %s}' % ("null" if until is None else f'"{until}"'))
    assert st.find_frozen_bundle(tmp_path).name == "b"
    assert st.find_frozen_bundle(tmp_path / "nothing") is None


def test_command_end_to_end_on_fake_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(config.LIGHTGBM_PARAMS, "n_estimators", 40)
    monkeypatch.setitem(config.LIGHTGBM_PARAMS, "learning_rate", 0.3)
    raw, proc, models, reports = (tmp_path / d for d in ("raw", "proc", "models", "reports"))
    for d in (raw, proc, models, reports):
        d.mkdir()

    emb = _series("2023-01-01", "2026-03-01", seed=1)
    emb_table = features.add_features(b.add_targets(b.add_time_features(pd.DataFrame({"pm25": emb}))))
    emb_table.reset_index(names="timestamp_utc").to_csv(proc / "embassy_features_auto.csv", index=False)
    pd.DataFrame({"timestamp_utc": emb.index, "pm25": emb.values}).to_csv(
        proc / "embassy_pm25_hourly_clean.csv", index=False)

    frozen, meta = train.train(emb_table, train_until=st.FROZEN_UNTIL)
    bundle.save_bundle(models / "embassy_frozen", frozen, meta)

    sensor = _series("2025-10-01", "2026-03-01", seed=2, offset=2.0)
    pd.DataFrame({"period.datetimeFrom.utc": sensor.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "value": sensor.values}).to_csv(raw / "loc111_sensor222_pm25_hours.csv", index=False)
    pd.DataFrame({"location_id": [111], "name": ["Test sensor"], "provider": ["AirGradient"],
                  "lat": [27.74], "lon": [85.34]}).to_csv(raw / "kathmandu_locations.csv", index=False)

    st.main(["--raw-dir", str(raw), "--locations", str(raw / "kathmandu_locations.csv"),
             "--embassy-clean", str(proc / "embassy_pm25_hourly_clean.csv"),
             "--embassy-features", str(proc / "embassy_features_auto.csv"),
             "--model", str(models / "embassy_frozen"), "--out", str(reports / "t.csv")])
    out = pd.read_csv(reports / "t.csv")
    assert list(out["location_id"]) == [111] and out.loc[0, "n_origins"] > 1500
    assert out.loc[0, "dist_km"] < 1 and out.loc[0, "corr"] > 0.5
    assert "Sensors scored: 1 of 1" in capsys.readouterr().out
