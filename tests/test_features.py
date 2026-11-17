import numpy as np
import pandas as pd
import pytest

from hawa.processing import features


def _df(values, start="2024-01-01 00:00"):
    idx = pd.date_range(start, periods=len(values), freq="h", tz="UTC", name="timestamp_utc")
    return pd.DataFrame({"pm25": values}, index=idx)


def test_short_gap_is_carried_forward_long_gap_is_not():
    v = [10.0, np.nan, np.nan, np.nan, np.nan, 50.0]        # 4 missing hours
    f = features.add_features(_df(v))
    assert list(f["pm25_filled"].iloc[:4]) == [10.0, 10.0, 10.0, 10.0]   # 3 carried
    assert np.isnan(f["pm25_filled"].iloc[4])                              # 4th stays empty
    assert f["pm25_filled"].iloc[5] == 50.0


def test_raw_pm25_is_never_changed():
    v = [10.0, np.nan, 12.0]
    f = features.add_features(_df(v))
    assert np.isnan(f["pm25"].iloc[1])


def test_age_counts_hours_since_real_value():
    f = features.add_features(_df([5.0, np.nan, np.nan, 7.0]))
    assert list(f["pm25_age_h"]) == [0, 1, 2, 0]


def test_lag_and_rolling_values():
    f = features.add_features(_df(list(range(1, 31))))     # 1..30
    assert f["pm25_lag024"].iloc[25] == 2.0                # value 24 hours before row 25 (value 26)
    assert f["pm25_mean006"].iloc[10] == np.mean([6, 7, 8, 9, 10, 11])
    assert f["pm25_change03"].iloc[10] == 3.0


def test_features_do_not_use_the_future():
    v = list(np.random.default_rng(0).uniform(10, 100, 400))
    a = features.add_features(_df(v))
    v2 = v.copy()
    v2[300:] = [999.0] * 100                                # change the future only
    b = features.add_features(_df(v2))
    cols = features.feature_columns(a)
    pd.testing.assert_frame_equal(a.iloc[:300][cols], b.iloc[:300][cols])


def test_irregular_grid_rejected():
    d = _df([1.0, 2.0, 3.0]).drop(_df([1.0, 2.0, 3.0]).index[1])
    with pytest.raises(ValueError):
        features.add_features(d)


def test_feature_columns_exclude_weather_pm25_and_targets():
    from hawa import config
    df = _df(list(range(30)))
    df[config.MODEL_WEATHER_VARIABLES[0]] = 1.0
    df["y_h01"] = 1.0
    cols = features.feature_columns(features.add_features(df))
    assert config.MODEL_WEATHER_VARIABLES[0] not in cols
    assert "pm25" not in cols and "y_h01" not in cols and "pm25_filled" in cols


def test_live_features_equal_training_features():
    """Training/serving skew check: features computed from the history up to hour t only (what a live
    forecast has) must equal the features of the same row computed on the full table (training)."""
    import numpy as np

    from hawa.processing import build_dataset as b
    rng = np.random.default_rng(3)
    idx = pd.date_range("2022-01-01", periods=600, freq="h", tz="UTC", name="timestamp_utc")
    pm = pd.Series(rng.uniform(10, 150, len(idx)), index=idx)
    pm.iloc[[50, 51, 52, 53, 54, 200, 201, 400]] = np.nan          # gaps, including one longer than the 3 h fill
    full = features.add_features(b.add_time_features(pd.DataFrame({"pm25": pm})))
    cols = features.feature_columns(full)
    for t in (60, 203, 300, 599):
        live = features.add_features(b.add_time_features(pd.DataFrame({"pm25": pm.iloc[: t + 1]})))
        pd.testing.assert_series_equal(live.iloc[-1][cols], full.iloc[t][cols], check_names=False)
