import numpy as np
import pandas as pd

from hawa.processing import clean


def _frame(values, start="2024-01-01 00:00", freq="h"):
    idx = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    return pd.DataFrame({"hour_start": idx, "v": values})


def test_valid_series_drops_bad_values():
    d = _frame([10, np.nan, -999, -1, 985, 20])
    s = clean.valid_series(d)
    assert list(s.values) == [10, 20]


def test_valid_series_keeps_one_row_per_hour():
    d = pd.concat([_frame([10, 20]), _frame([11, 21])])
    s = clean.valid_series(d)
    assert len(s) == 2 and s.index.is_unique


def test_flat_run_removed_only_when_long_enough():
    stuck = [92.0] * 12                       # exactly the limit -> removed
    ok = [92.0] * 11                          # one short -> kept
    s_stuck, n = clean.drop_flat_runs(clean.valid_series(_frame([1, 2] + stuck + [3, 4])))
    s_ok, m = clean.drop_flat_runs(clean.valid_series(_frame([1, 2] + ok + [3, 4])))
    assert n == 12 and len(s_stuck) == 4
    assert m == 0 and len(s_ok) == 15


def test_flat_run_needs_consecutive_hours():
    # same value 12 times but with a gap in the middle is NOT a stuck run
    d = pd.concat([_frame([50.0] * 6, "2024-01-01 00:00"), _frame([50.0] * 6, "2024-01-01 08:00")])
    s, n = clean.drop_flat_runs(clean.valid_series(d))
    assert n == 0 and len(s) == 12


def test_odn_timestamps_shift_back_one_hour(tmp_path):
    f = tmp_path / "odn.csv"
    f.write_text(
        "locationId,location,city,country,utc,local,parameter,value,unit,latitude,longitude\n"
        "1,x,x,NP,2021-03-12T18:15:00+00:00,x,pm25,50,u,0,0\n"
        "1,x,x,NP,2021-03-12T18:15:00+00:00,x,o3,0.05,ppm,0,0\n"
    )
    d = clean.read_odn(f)
    assert len(d) == 1                                   # only pm25
    assert d["hour_start"].iloc[0] == pd.Timestamp("2021-03-12 17:00", tz="UTC")
