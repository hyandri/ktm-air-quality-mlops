import numpy as np
import pandas as pd

from hawa.forecasting import actuals

NOW = pd.Timestamp("2026-11-18T04:17:00Z")
ORIGIN = pd.Timestamp("2026-11-17T09:00:00Z")


def _series(hours=300):
    idx = pd.date_range(end="2026-11-18T03:00:00Z", periods=hours, freq="h", tz="UTC")
    return pd.Series(np.arange(hours, dtype=float), index=idx)        # value = position, easy to check by hand


def _forecasts(value=100.0):
    h = np.arange(1, 25)
    return pd.DataFrame({
        "location_id": 1, "sensor_id": 11, "sensor_name": "a", "model_version": "m", "horizon": h,
        "origin_utc": ORIGIN.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_hour_utc": (ORIGIN + pd.to_timedelta(h, unit="h")).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_pm25_sensor_scale": value})


def test_history_hours_covers_yesterday_before_the_oldest_origin():
    assert actuals.history_hours(ORIGIN, NOW) == 19 + 24 + 2


def test_only_hours_with_a_real_value_are_compared():
    rows = actuals.compare(_forecasts(), _series())
    # the series ends at 03:00 on 18 Nov; origin 09:00 on 17 Nov + 18 h = 03:00 -> horizons 1..18
    assert list(rows["horizon"]) == list(range(1, 19))


def test_errors_are_computed_against_the_right_values():
    s = _series()
    rows = actuals.compare(_forecasts(100.0), s)
    r = rows[rows["horizon"] == 1].iloc[0]
    t = ORIGIN + pd.Timedelta(hours=1)
    assert r["actual"] == s[t]
    assert r["persistence"] == s[ORIGIN]
    assert r["same_hour_yesterday"] == s[t - pd.Timedelta(hours=24)]
    assert r["err_model"] == abs(100.0 - s[t]) and r["err_persistence"] == 1.0 and r["err_yesterday"] == 24.0


def test_summary_gain_is_positive_for_a_perfect_model():
    s = _series()
    f = _forecasts()
    f["forecast_pm25_sensor_scale"] = s.reindex(pd.to_datetime(f["target_hour_utc"], utc=True)).fillna(0).to_numpy()
    rows = actuals.compare(f, s)
    out = actuals.summary(rows)
    assert (out["mae_model"] == 0).all() and (out["gain_vs_best_baseline_pct"] == 100).all()
    assert out["horizons"].iloc[-1] == "h1-h24" and out["n"].iloc[-1] == 18


def test_evaluate_reports_broken_sensor_and_keeps_going():
    raw = pd.DataFrame({"period.datetimeFrom.utc": _series().index.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "value": np.linspace(10, 60, 300)})
    f2 = _forecasts()
    f2["sensor_id"] = 22
    forecasts = pd.concat([_forecasts(), f2], ignore_index=True)
    sensors = [{"name": "a", "location_id": 1, "sensor_id": 11}, {"name": "b", "location_id": 2, "sensor_id": 22}]

    def fetch(s, now, hours_back):
        if s["name"] == "b":
            raise RuntimeError("network down")
        return raw

    rows, problems = actuals.evaluate(forecasts, fetch, NOW, sensors)
    assert set(rows["sensor_id"]) == {11} and problems[0][0] == "b"
