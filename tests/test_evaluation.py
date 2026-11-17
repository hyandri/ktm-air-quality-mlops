import numpy as np
import pandas as pd
import pytest

from hawa import config
from hawa.evaluation import metrics, splits
from hawa.models import baselines


def _df(n=120, start="2024-01-01 00:00"):
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC", name="timestamp_utc")
    df = pd.DataFrame({"pm25": np.arange(n, dtype=float)}, index=idx)
    for h in range(1, 25):
        df[f"y_h{h:02d}"] = df["pm25"].shift(-h)
    return df


# ------------------------------------------------------------------- splits
def test_winter_year_naming():
    idx = pd.DatetimeIndex(["2024-01-15 00:00", "2023-12-15 00:00", "2023-07-01 00:00"], tz="UTC")
    assert list(splits.winter_start_year(idx)) == [2023, 2023, -1]


def test_splits_do_not_overlap():
    idx = pd.date_range("2017-03-01", "2026-06-01", freq="h", tz="UTC")
    for scope in splits.SCOPES:
        m = {s: splits.split_mask(idx, s, scope) for s in splits.SPLITS}
        assert not (m["train"] & m["validation"]).any()
        assert not (m["train"] & m["test"]).any()
        assert not (m["validation"] & m["test"]).any()


def test_winter_scope_uses_configured_winters_and_skips_2022():
    idx = pd.date_range("2022-12-05", periods=24, freq="h", tz="UTC")  # winter 2022/23
    for s in splits.SPLITS:
        assert not splits.split_mask(idx, s, "winter").any()
    val = pd.date_range("2024-01-10", periods=24, freq="h", tz="UTC")
    assert splits.split_mask(val, "validation", "winter").all()


def test_train_targets_do_not_reach_into_validation():
    # last training origin + 24 h must stay before the first validation origin (both scopes)
    idx = pd.date_range("2017-03-01", "2026-06-01", freq="h", tz="UTC")
    for scope in splits.SCOPES:
        tr = idx[splits.split_mask(idx, "train", scope)].max()
        va = idx[splits.split_mask(idx, "validation", scope)].min()
        assert tr + pd.Timedelta(hours=24) < va


# ---------------------------------------------------------------- baselines
def test_persistence_repeats_current_value():
    df = _df()
    p = baselines.persistence(df)
    assert (p["y_h05"] == df["pm25"]).all()


def test_same_hour_yesterday_uses_value_24h_before_target():
    df = _df()
    p = baselines.same_hour_yesterday(df)
    t = df.index[50]
    assert p.loc[t, "y_h06"] == df.loc[t + pd.Timedelta(hours=6 - 24), "pm25"]
    assert p.loc[t, "y_h24"] == df.loc[t, "pm25"]


def test_baselines_use_no_future_values():
    df = _df()
    df2 = df.copy()
    df2.loc[df2.index[80:], "pm25"] = 999.0                    # change the future
    for fn in baselines.BASELINES.values():
        a, b = fn(df), fn(df2)
        pd.testing.assert_frame_equal(a.iloc[:80], b.iloc[:80])


def test_baseline_rejects_irregular_grid():
    df = _df().drop(_df().index[10])
    with pytest.raises(ValueError):
        baselines.persistence(df)


# ------------------------------------------------------------------ metrics
def test_score_values_and_common_rows():
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    cols = metrics.target_columns(1)
    y = pd.DataFrame({"y_h01": [10.0, 20.0, 30.0, np.nan]}, index=idx)
    a = pd.DataFrame({"y_h01": [12.0, 18.0, np.nan, 5.0]}, index=idx)   # missing at row 3
    b = pd.DataFrame({"y_h01": [10.0, 20.0, 33.0, 5.0]}, index=idx)
    s = metrics.score(y, {"a": a, "b": b}, horizons=1).set_index("model")
    assert s.loc["a", "n"] == s.loc["b", "n"] == 2          # same rows for both models
    assert s.loc["a", "mae"] == 2.0 and s.loc["b", "mae"] == 0.0
    assert s.loc["a", "rmse"] == 2.0
