"""Simple forecasting baselines. They learn nothing; every later model must beat them.

Each baseline takes the dataset (one row per hour, regular hourly index, column `pm25`)
and returns a table with columns y_h01 ... y_h24: the prediction made at hour t for t+h.
Only values at or before hour t are used (no future information).

  persistence          PM2.5 at t+h will equal PM2.5 now.
  same_hour_yesterday  PM2.5 at t+h will equal PM2.5 at the same clock hour on the previous day,
                       i.e. the value at t+h-24 (needs h <= 24; for h=24 it equals persistence).
"""
import pandas as pd

from hawa import config


def _check(df):
    idx = df.index
    if len(idx) > 1 and not (idx[1:] - idx[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("dataset must be on a regular hourly grid")


def persistence(df, horizons=config.HORIZONS):
    _check(df)
    return pd.DataFrame({f"y_h{h:02d}": df["pm25"] for h in range(1, horizons + 1)}, index=df.index)


def same_hour_yesterday(df, horizons=config.HORIZONS):
    _check(df)
    return pd.DataFrame({f"y_h{h:02d}": df["pm25"].shift(24 - h) for h in range(1, horizons + 1)},
                        index=df.index)


BASELINES = {
    "persistence": persistence,
    "same_hour_yesterday": same_hour_yesterday,
}
