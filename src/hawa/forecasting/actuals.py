"""Live check: compare saved forecasts with what really happened.

For every forecast row in data/forecasts/forecasts.csv whose target hour is now in the past, we fetch the real
hourly values again and compare:
  - the model forecast,
  - persistence (the value at the origin hour, repeated),
  - same hour yesterday (the value 24 h before the target hour).
All three are scored on the SAME rows, so the comparison is fair. Rows where the real value is still missing
are left out (the hour has not happened yet, or the sensor had no valid value).

Run from the project root (needs OPENAQ_API_KEY in .env):
    python -m hawa.forecasting.actuals

Writes: reports/live_eval.csv (one row per compared forecast hour). Prints MAE per group of horizons.
Few days of data = few rows: read the numbers as a smoke test, not as a result.
"""
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from hawa import config
from hawa.forecasting.batch import openaq_to_series
from hawa.ingestion import openaq

GROUPS = [(1, 6), (7, 12), (13, 18), (19, 24)]


def history_hours(origin_min, now):
    """Hours of history needed: from 24 h before the oldest origin until now (plus a small margin)."""
    return int((now - origin_min) / pd.Timedelta(hours=1)) + 24 + 2


def openaq_fetch_since(sensor, now, hours_back):
    return openaq.fetch_recent_hours(sensor["location_id"], sensor["sensor_id"],
                                     hours_back=hours_back, now=now.to_pydatetime())


def compare(forecasts, series):
    """Join forecasts of ONE sensor with its real series. Returns one row per forecast hour that has all values."""
    f = forecasts.copy()
    target = pd.to_datetime(f["target_hour_utc"], utc=True)
    origin = pd.to_datetime(f["origin_utc"], utc=True)
    f["actual"] = series.reindex(target).to_numpy()
    f["persistence"] = series.reindex(origin).to_numpy()
    f["same_hour_yesterday"] = series.reindex(target - pd.Timedelta(hours=24)).to_numpy()
    f = f.dropna(subset=["actual", "persistence", "same_hour_yesterday"])
    f["err_model"] = (f["forecast_pm25_sensor_scale"] - f["actual"]).abs()
    f["err_persistence"] = (f["persistence"] - f["actual"]).abs()
    f["err_yesterday"] = (f["same_hour_yesterday"] - f["actual"]).abs()
    return f


def evaluate(forecasts, fetch, now, sensors):
    """Compare all forecasts. `fetch(sensor, now, hours_back)` returns the raw OpenAQ table."""
    now = pd.Timestamp(now).tz_convert("UTC")
    parts, problems = [], []
    for s in sensors:
        f = forecasts[forecasts["sensor_id"] == s["sensor_id"]]
        if f.empty:
            continue
        origin_min = pd.to_datetime(f["origin_utc"], utc=True).min()
        try:
            raw = fetch(s, now, history_hours(origin_min, now))
        except Exception as err:
            problems.append((s["name"], f"fetch failed: {err}"))
            continue
        if raw is None or raw.empty:
            problems.append((s["name"], "API returned no rows"))
            continue
        parts.append(compare(f, openaq_to_series(raw)))
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return rows, problems


def summary(rows):
    """MAE of model / persistence / yesterday per horizon group, and the gain over the best baseline.

    The best baseline is chosen for EACH horizon (persistence is better early, yesterday later) and then averaged
    over the rows of the group. A single baseline for all 24 hours would be a weaker bar and flatter the model.
    """
    by_h = rows.groupby("horizon")[["err_persistence", "err_yesterday"]].transform("mean")
    rows = rows.assign(err_best_baseline=by_h.min(axis=1))
    out = []
    for lo, hi in GROUPS + [(1, 24)]:
        r = rows[rows["horizon"].between(lo, hi)]
        if r.empty:
            continue
        m, best = r["err_model"].mean(), r["err_best_baseline"].mean()
        gain = 100 * (best - m) / best if best else np.nan
        out.append({"horizons": f"h{lo}-h{hi}", "n": len(r), "mae_model": m,
                    "mae_persistence": r["err_persistence"].mean(), "mae_yesterday": r["err_yesterday"].mean(),
                    "mae_best_baseline": best, "gain_vs_best_baseline_pct": gain})
    return pd.DataFrame(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forecasts", default=str(config.FORECASTS_DIR / "forecasts.csv"))
    p.add_argument("--out", default=str(config.REPORTS_DIR / "live_eval.csv"))
    p.add_argument("--now", help="pretend it is this UTC time (default: now)")
    a = p.parse_args(argv)
    now = pd.Timestamp(a.now) if a.now else pd.Timestamp(datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    forecasts = pd.read_csv(a.forecasts)
    rows, problems = evaluate(forecasts, openaq_fetch_since, now, config.LIVE_SENSORS)
    for name, why in problems:
        print(f"  skipped {name}: {why}")
    if rows.empty:
        print("no forecast hour has a real value yet")
        return
    rows.to_csv(a.out, index=False)
    n_sensors = rows["sensor_id"].nunique()
    print(f"compared {len(rows)} forecast hours from {n_sensors} sensors, {rows['origin_utc'].nunique()} origins")
    print(summary(rows).round(2).to_string(index=False))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
