import numpy as np
import pandas as pd
import pytest

from hawa import config
from hawa.analysis import calibration as cal
from hawa.analysis import sensor_transfer as st
from hawa.models import bundle, train
from hawa.processing import build_dataset as b
from hawa.processing import features


def _wave(start, end, seed=0, scale=1.0):
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    base = 60 + 30 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + rng.normal(0, 4, len(idx))
    return pd.Series(np.clip(base * scale, 1, None), index=idx)


def test_monthly_table_finds_the_ratio_and_drops_thin_months():
    emb = _wave("2025-09-01", "2025-11-10")
    sensor = 1.5 * emb
    t = cal.monthly_table(cal.pair(sensor, emb), min_hours=100)
    assert set(t["month"]) == {9, 10, 11}
    assert t["ratio"].round(3).eq(1.5).all() and (t["corr"] > 0.999).all()
    assert cal.monthly_table(cal.pair(sensor, emb), min_hours=100000).empty


def test_the_factor_is_fitted_outside_the_locked_test_winters():
    """The sensor reads 1.5x everywhere, but 3x inside the locked winter 2025/26. The fit must not see the 3x."""
    emb = _wave("2025-09-01", "2026-06-01")
    sensor = 1.5 * emb
    in_test = pd.Series(np.asarray(cal.splits.winter_start_year(sensor.index)) == 2025, index=sensor.index)
    sensor[in_test] = 3.0 * emb[in_test]
    pairs = cal.pair(sensor, emb)
    assert cal.fit_ratio(pairs) == pytest.approx(1.5, rel=1e-6)
    assert len(cal.fit_pairs(pairs)) < len(pairs)
    all_hours_ratio = pairs["s"].mean() / pairs["e"].mean()
    assert all_hours_ratio > 1.7                                            # using every hour WOULD have leaked the 3x


def test_fit_ratio_needs_enough_hours():
    emb = _wave("2025-09-01", "2025-09-10")
    assert np.isnan(cal.fit_ratio(cal.pair(1.5 * emb, emb), min_hours=500))


def test_pooled_factor_is_the_median_of_the_others():
    own = {1: 1.4, 2: 1.6, 3: 3.0, 4: np.nan}
    assert cal.pooled_factor(own, exclude=3) == pytest.approx(1.5)         # median of 1.4 and 1.6
    assert cal.pooled_factor(own, exclude=1) == pytest.approx(2.3)         # median of 1.6 and 3.0
    assert np.isnan(cal.pooled_factor({1: 1.4}, exclude=1))


def _table(raw, pooled, raw_mae=10.0, pooled_mae=6.0):
    return pd.DataFrame({"raw_gain": raw, "pooled_gain": pooled, "raw_match_mae": raw_mae,
                         "pooled_match_mae": pooled_mae})


def _monthly(ratios):
    return pd.DataFrame({"location_id": 1, "name": "s", "month": range(1, len(ratios) + 1), "n": 500, "ratio": ratios,
                         "corr": 0.9})


def test_verdict_adopts_when_all_rules_hold_and_rejects_otherwise():
    t = _table([5, 6, 4, 5, 6], [10, 11, 9, 10, 11])
    adopt, facts, text = cal.verdict(t, _monthly([1.5, 1.55, 1.6]))
    assert adopt and text == "ADOPT the pooled factor." and facts["median_improvement"] == 5.0

    small = _table([5, 6, 4, 5, 6], [6, 7, 5, 6, 7])                        # +1 point only
    adopt, _, text = cal.verdict(small, _monthly([1.5, 1.5]))
    assert not adopt and "improves by 1.0 points" in text

    mixed = _table([5, 6, 4, 5, 6], [10, 11, 1, 1, 1])                      # better on only 40%
    adopt, _, text = cal.verdict(mixed, _monthly([1.5, 1.5]))
    assert not adopt and "40% of sensors" in text

    worse_match = _table([5, 6, 4], [10, 11, 9], raw_mae=6.0, pooled_mae=8.0)
    adopt, _, text = cal.verdict(worse_match, _monthly([1.5, 1.5]))
    assert not adopt and "scale match" in text


def test_a_changing_seasonal_ratio_blocks_adoption_even_if_the_gain_is_large():
    t = _table([5, 6, 4, 5, 6], [10, 11, 9, 10, 11])
    adopt, facts, text = cal.verdict(t, _monthly([1.2, 1.9]))               # 1.58x between months
    assert not adopt and facts["season_spread"] > cal.SEASON_SPREAD and "does not fit every season" in text


def test_verdict_with_nothing_scored():
    t = _table([np.nan], [np.nan])
    adopt, _, text = cal.verdict(t, _monthly([1.5]))
    assert not adopt and "no verdict" in text.lower()


def test_command_end_to_end_on_fake_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(config.LIGHTGBM_PARAMS, "n_estimators", 60)
    monkeypatch.setitem(config.LIGHTGBM_PARAMS, "learning_rate", 0.3)
    raw, proc, models, reports = (tmp_path / d for d in ("raw", "proc", "models", "reports"))
    for d in (raw, proc, models, reports):
        d.mkdir()

    emb = _wave("2023-01-01", "2026-06-01", seed=1)
    emb_table = features.add_features(b.add_targets(b.add_time_features(pd.DataFrame({"pm25": emb}))))
    pd.DataFrame({"timestamp_utc": emb.index, "pm25": emb.values}).to_csv(
        proc / "embassy_pm25_hourly_clean.csv", index=False)
    frozen, meta = train.train(emb_table, train_until=st.FROZEN_UNTIL)
    bundle.save_bundle(models / "embassy_frozen", frozen, meta)

    locations = []
    for i, seed in enumerate((2, 3, 4)):
        sensor = _wave("2025-09-15", "2026-06-01", seed=seed, scale=1.5)       # reads 1.5x, same daily pattern
        pd.DataFrame({"period.datetimeFrom.utc": sensor.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "value": sensor.values}).to_csv(raw / f"loc{100 + i}_sensor{200 + i}_pm25_hours.csv", index=False)
        locations.append({"location_id": 100 + i, "name": f"S{i}", "provider": "AirGradient",
                          "lat": 27.74 + 0.01 * i, "lon": 85.34})
    far = _wave("2025-09-15", "2026-06-01", seed=9, scale=1.5)
    pd.DataFrame({"period.datetimeFrom.utc": far.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "value": far.values}).to_csv(raw / "loc999_sensor999_pm25_hours.csv", index=False)
    locations.append({"location_id": 999, "name": "far", "provider": "x", "lat": 28.5, "lon": 85.3})
    pd.DataFrame(locations).to_csv(raw / "kathmandu_locations.csv", index=False)

    cal.main(["--raw-dir", str(raw), "--locations", str(raw / "kathmandu_locations.csv"),
              "--embassy-clean", str(proc / "embassy_pm25_hourly_clean.csv"),
              "--model", str(models / "embassy_frozen"), "--out-dir", str(reports)])
    test = pd.read_csv(reports / "calibration_test.csv")
    monthly = pd.read_csv(reports / "calibration_monthly.csv")
    assert len(test) == 3 and set(test["location_id"]) == {100, 101, 102}      # the far sensor was left out
    assert test["own_ratio"].round(1).eq(1.5).all() and test["pooled_factor"].round(1).eq(1.5).all()
    assert (test["pooled_match_mae"] < test["raw_match_mae"]).all()             # scale now agrees with the Embassy
    assert test["pooled_bias"].abs().max() < 3 and test["raw_bias"].min() > 20
    assert (test["pooled_gain"] > test["raw_gain"]).all()                       # the model likes the corrected input
    assert monthly["ratio"].round(1).eq(1.5).all()
    out = capsys.readouterr().out
    assert "left out: 1" in out and "ADOPT the pooled factor." in out
