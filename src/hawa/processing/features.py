"""Add PM2.5 history features to the modelling table.

Every feature for row `t` uses only PM2.5 at hours t, t-1, t-2 ... (never the future).

Features
  pm25_filled     current PM2.5; if the last 1-3 hours are missing, the last real value is carried forward
  pm25_age_h      hours since the last REAL measurement (0 = measured now)
  pm25_lagNNN     PM2.5 NNN hours earlier (1, 2, 3, 6, 12, 24, 48, 168 h = 1 week)
  pm25_meanNNN    average of the last NNN hours (6, 24, 72, 168)
  pm25_std024     spread (standard deviation) of the last 24 hours
  pm25_change03   PM2.5 now minus 3 hours ago (is it rising or falling?)

Missing-data rule: a gap is bridged only for FILL_LIMIT_H (3) hours, and only by carrying the
last real value FORWARD. Interpolation is not used because it needs the next value (the future).
Longer gaps stay empty. The targets (y_h01 ... y_h24) and the raw `pm25` are never filled.

Run from the project root:
    python -m hawa.processing.features                 # Embassy, weather = auto
Needs:  data/processed/<station>_dataset_<weather>.csv   (hawa.processing.build_dataset)
Writes: data/processed/<station>_features_<weather>.csv
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from hawa import config


def add_features(df):
    """Return `df` plus the PM2.5 history features. `df` needs a regular hourly index and `pm25`."""
    idx = df.index
    if len(idx) > 1 and not (idx[1:] - idx[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("dataset must be on a regular hourly grid")

    raw = df["pm25"]
    filled = raw.ffill(limit=config.FILL_LIMIT_H)

    pos = pd.Series(np.arange(len(df), dtype=float), index=idx)
    last_real = pos.where(raw.notna()).ffill()
    new = {"pm25_filled": filled, "pm25_age_h": pos - last_real}

    for k in config.LAG_HOURS:
        new[f"pm25_lag{k:03d}"] = filled.shift(k)
    for w in config.ROLL_WINDOWS:
        new[f"pm25_mean{w:03d}"] = filled.rolling(w, min_periods=math.ceil(w / 2)).mean()
    new["pm25_std024"] = filled.rolling(24, min_periods=12).std()
    new["pm25_change03"] = filled - filled.shift(3)

    return pd.concat([df, pd.DataFrame(new, index=idx)], axis=1)


def feature_columns(df, include_weather=True):
    """Columns a model may use as inputs: everything except raw pm25, weather_source and targets.

    include_weather=False also drops the weather columns (for the "does weather help?" test).
    """
    skip = {"pm25", "weather_source"}
    if not include_weather:
        skip |= set(config.MODEL_WEATHER_VARIABLES)
    return [c for c in df.columns if c not in skip and not c.startswith("y_h")]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--station", default="embassy")
    p.add_argument("--weather", default="auto")
    p.add_argument("--dataset", help="explicit input path")
    p.add_argument("--out", help="explicit output path")
    a = p.parse_args(argv)

    src = Path(a.dataset or config.PROCESSED_DIR / f"{a.station}_dataset_{a.weather}.csv")
    df = pd.read_csv(src, parse_dates=["timestamp_utc"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc").sort_index()

    out_df = add_features(df)
    cols = feature_columns(out_df)
    print(f"rows: {len(out_df)} | input columns for models: {len(cols)}")
    print(f"pm25 real: {out_df['pm25'].notna().mean():.1%} | "
          f"pm25_filled available: {out_df['pm25_filled'].notna().mean():.1%}")
    print(f"hours rescued by the 3 h carry-forward: "
          f"{int(out_df['pm25_filled'].notna().sum() - out_df['pm25'].notna().sum())}")

    out = Path(a.out or config.PROCESSED_DIR / f"{a.station}_features_{a.weather}.csv")
    out_df.reset_index().to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
