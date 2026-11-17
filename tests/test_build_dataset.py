import numpy as np
import pandas as pd

from hawa.processing import build_dataset as bd


def _pm25(n=60, start="2023-01-01 00:00"):
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC", name="timestamp_utc")
    return pd.Series(np.arange(n, dtype=float), index=idx)


def _weather(index, value):
    return pd.DataFrame({"temperature_2m": value}, index=index)


def test_targets_are_future_values():
    s = _pm25()
    df = bd.build(s, _weather(s.index, 1.0), pd.Series("era5", index=s.index))
    t = s.index[10]
    assert df.loc[t, "y_h01"] == s.iloc[11]
    assert df.loc[t, "y_h24"] == s.iloc[34]


def test_targets_empty_at_the_end():
    s = _pm25()
    df = bd.build(s, _weather(s.index, 1.0), pd.Series("era5", index=s.index))
    assert df["y_h24"].iloc[-24:].isna().all()
    assert df["y_h24"].iloc[:-24].notna().all()


def test_missing_pm25_stays_missing_and_shows_in_targets():
    s = _pm25()
    s.iloc[20] = np.nan
    df = bd.build(s, _weather(s.index, 1.0), pd.Series("era5", index=s.index))
    assert np.isnan(df["pm25"].iloc[20])
    assert np.isnan(df["y_h05"].iloc[15])          # row 15 + 5 hours = missing hour 20
    assert not np.isnan(df["y_h04"].iloc[15])


def test_features_do_not_use_the_future():
    # changing a FUTURE pm25 value must not change features of an earlier row
    s = _pm25()
    a = bd.build(s, _weather(s.index, 1.0), pd.Series("era5", index=s.index))
    s2 = s.copy()
    s2.iloc[40] = 999.0
    b = bd.build(s2, _weather(s2.index, 1.0), pd.Series("era5", index=s.index))
    feature_cols = [c for c in a.columns if not c.startswith("y_h")]
    row = 30
    pd.testing.assert_series_equal(a.iloc[row][feature_cols], b.iloc[row][feature_cols])


def test_weather_auto_switches_source_at_start_date():
    idx = pd.date_range("2020-12-30", periods=96, freq="h", tz="UTC")
    era5 = _weather(idx, 1.0)
    fc = _weather(idx, 2.0)
    w, src = bd.choose_weather(era5, fc, "auto", forecast_start="2021-01-01")
    before, after = idx < pd.Timestamp("2021-01-01", tz="UTC"), idx >= pd.Timestamp("2021-01-01", tz="UTC")
    assert (w.loc[before, "temperature_2m"] == 1.0).all() and (src[before] == "era5").all()
    assert (w.loc[after, "temperature_2m"] == 2.0).all() and (src[after] == "historical_forecast").all()


def test_time_features_use_kathmandu_local_time():
    s = _pm25(2, "2024-01-01 00:00")                  # 00:00 UTC = 05:45 local
    df = bd.build(s, _weather(s.index, 1.0), pd.Series("era5", index=s.index))
    assert df["hour_local"].iloc[0] == 5.75
    assert df["month"].iloc[0] == 1
    assert df["hour_local"].iloc[1] == 6.75


def test_read_pm25_makes_regular_grid(tmp_path):
    f = tmp_path / "p.csv"
    f.write_text("timestamp_utc,station,pm25,source\n"
                 "2024-01-01 00:00:00+00:00,x,10,a\n"
                 "2024-01-01 03:00:00+00:00,x,13,a\n")
    s = bd.read_pm25(f)
    assert len(s) == 4 and s.isna().sum() == 2
