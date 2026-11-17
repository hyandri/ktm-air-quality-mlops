"""Join clean PM2.5 and weather into one modelling table (one row per hour).

Row `t` = a forecast origin: "it is hour t, what will PM2.5 be at t+1 ... t+24?"
  * features  = only information known at hour t (current PM2.5, weather at t, time of day/year)
  * targets   = `y_h01` ... `y_h24` = PM2.5 at t+1 ... t+24 (labels only; never use as features)
Missing hours stay empty (NaN). Nothing is filled or interpolated here.

Weather source (--weather):
  auto      archived forecasts from 2021-01-01, ERA5 before that   (default)
  forecast  archived forecasts only (rows before 2021 have no weather)
  era5      ERA5 for all years (use it to measure the train/live mismatch)

Run from the project root:
    python -m hawa.processing.build_dataset
    python -m hawa.processing.build_dataset --station phora --weather era5

Needs: data/processed/<station>_pm25_hourly_clean.csv  (hawa.processing.clean)
       data/raw/openmeteo/openmeteo_archive_*.csv and openmeteo_historical_forecast_*.csv
Writes: data/processed/<station>_dataset_<weather>.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hawa import config

STATIONS = {"embassy": "us_embassy", "phora": "phora_durbar"}


# ------------------------------------------------------------------- readers
def read_pm25(path):
    """Clean PM2.5 file -> Series on a regular hourly UTC grid (NaN where missing)."""
    d = pd.read_csv(path, parse_dates=["timestamp_utc"])
    d["timestamp_utc"] = pd.to_datetime(d["timestamp_utc"], utc=True)
    s = d.set_index("timestamp_utc")["pm25"].sort_index()
    s = s[~s.index.duplicated()]
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h", name="timestamp_utc"))


def read_weather(path, variables):
    """Open-Meteo raw CSV -> DataFrame indexed by UTC hour with the wanted variables."""
    d = pd.read_csv(path)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d.drop_duplicates("time").set_index("time").sort_index()
    return d[list(variables)]


def choose_weather(era5, forecast, mode, forecast_start=config.FORECAST_WEATHER_START):
    """Pick the weather per hour. Returns (weather DataFrame, source label Series)."""
    start = pd.Timestamp(forecast_start, tz="UTC")
    if mode == "era5":
        return era5, pd.Series("era5", index=era5.index)
    if mode == "forecast":
        return forecast, pd.Series("historical_forecast", index=forecast.index)
    if mode != "auto":
        raise ValueError(f"unknown weather mode: {mode}")
    idx = era5.index.union(forecast.index)
    use_fc = idx >= start
    w = era5.reindex(idx)
    w.loc[use_fc] = forecast.reindex(idx[use_fc])
    src = pd.Series(np.where(use_fc, "historical_forecast", "era5"), index=idx)
    return w, src


# ------------------------------------------------------------------ features
def add_time_features(df):
    """Time of day / year in Kathmandu local time (UTC+05:45), plus sin/cos for linear models."""
    local = df.index + pd.Timedelta(hours=config.LOCAL_UTC_OFFSET_H)
    df["hour_local"] = (df.index.hour + config.LOCAL_UTC_OFFSET_H) % 24  # float, e.g. 5.75
    df["month"] = local.month
    df["dayofweek"] = local.day_of_week
    df["dayofyear"] = local.day_of_year
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_local"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_local"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    return df


def add_targets(df, horizons=config.HORIZONS):
    """y_hNN = PM2.5 NN hours after the row's hour. Needs a regular hourly index."""
    targets = {f"y_h{h:02d}": df["pm25"].shift(-h) for h in range(1, horizons + 1)}
    return pd.concat([df, pd.DataFrame(targets, index=df.index)], axis=1)


# --------------------------------------------------------------------- build
def build(pm25, weather, weather_source):
    df = pd.DataFrame({"pm25": pm25})
    df = df.join(weather, how="left")
    df["weather_source"] = weather_source.reindex(df.index)
    df = add_time_features(df)
    df = add_targets(df)
    df.index.name = "timestamp_utc"
    return df


def report(df, weather_cols):
    n = len(df)
    print(f"\nrows: {n}  | {df.index.min()} -> {df.index.max()}")
    print(f"pm25 valid: {df['pm25'].notna().mean():.1%}")
    print("weather_source:", df["weather_source"].value_counts(dropna=False).to_dict())
    miss = df[weather_cols].isna().mean().mul(100).round(1)
    print("weather missing % (worst 3):", miss.sort_values(ascending=False).head(3).to_dict())
    for h in (1, 6, 12, 24):
        print(f"  target y_h{h:02d} available: {df[f'y_h{h:02d}'].notna().mean():.1%}")
    usable = df["pm25"].notna() & df["y_h24"].notna() & df[weather_cols].notna().all(axis=1)
    print(f"rows with pm25, weather and y_h24 all present: {usable.sum()} ({usable.mean():.1%})")
    local = df.index + pd.Timedelta(hours=config.LOCAL_UTC_OFFSET_H)
    winter = pd.Series(local.month.isin([12, 1, 2]), index=df.index)
    wy = np.where(local.month == 12, local.year, local.year - 1)
    by = usable[winter].groupby(wy[winter.values]).sum()
    print("usable rows per winter (Dec-Feb, by start year):", by.to_dict())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--station", choices=list(STATIONS), default="embassy")
    p.add_argument("--weather", choices=["auto", "forecast", "era5"], default="auto")
    p.add_argument("--pm25-file", help="default: data/processed/<station>_pm25_hourly_clean.csv")
    p.add_argument("--out", help="default: data/processed/<station>_dataset_<weather>.csv")
    a = p.parse_args(argv)

    pm25_path = Path(a.pm25_file or config.PROCESSED_DIR / f"{a.station}_pm25_hourly_clean.csv")
    tag = f"{config.EMBASSY_LAT}_{config.EMBASSY_LON}"
    variables = config.MODEL_WEATHER_VARIABLES
    era5 = read_weather(config.OPENMETEO_RAW / f"openmeteo_archive_{tag}.csv", variables)
    fc = read_weather(config.OPENMETEO_RAW / f"openmeteo_historical_forecast_{tag}.csv", variables)

    weather, src = choose_weather(era5, fc, a.weather)
    df = build(read_pm25(pm25_path), weather, src)
    report(df, variables)

    out = Path(a.out or config.PROCESSED_DIR / f"{a.station}_dataset_{a.weather}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()