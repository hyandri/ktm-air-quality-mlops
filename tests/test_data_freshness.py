import numpy as np
import pandas as pd

from hawa.analysis import data_freshness as df_


def _raw(start, hours, offset=0.0):
    idx = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    vals = 40 + 10 * np.sin(np.arange(hours) * 2 * np.pi / 24) + np.arange(hours) * 0.01 + offset
    return pd.DataFrame({"period.datetimeFrom.utc": idx.strftime("%Y-%m-%dT%H:%M:%SZ"), "value": vals})


def test_summarise_counts_hours_after_old_end_and_overlap():
    full = df_.valid_hours(_raw("2026-06-01", 24 * 14))                    # 14 days of one continuous record
    old = full[full.index < "2026-06-11"]                                 # our raw file: first 10 days
    new = full[full.index >= "2026-06-05"]                                # fresh download: from day 5 on
    r = df_.summarise(new, old, "2026-06-05")
    assert r["valid_hours_after_old_end"] == 24 * 4 and r["overlap_hours"] == 24 * 6
    assert r["overlap_identical_share"] == 1.0 and r["monthly"].sum() == len(new)


def test_a_changed_value_on_the_overlap_lowers_the_identical_share():
    full = df_.valid_hours(_raw("2026-06-01", 24 * 4))
    changed = full.copy()
    changed.iloc[:24] += 5.0                                              # first day differs
    r = df_.summarise(changed, full, "2026-06-01")
    assert r["overlap_identical_share"] == 0.75


def test_identical_data_gives_100_percent_overlap():
    raw = _raw("2026-06-01", 24 * 5)
    s = df_.valid_hours(raw)
    r = df_.summarise(s, s, "2026-06-01")
    assert r["overlap_identical_share"] == 1.0 and r["valid_hours_after_old_end"] == 0


def test_check_station_reports_empty_download_and_never_writes_to_raw(tmp_path, monkeypatch):
    out = df_.check_station("embassy", "2026-06-01", "2026-06-10", fetch=lambda *a, **k: [], out_dir=tmp_path)
    assert out == {"station": "embassy", "error": "OpenAQ returned no rows for this period"}
