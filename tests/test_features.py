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
