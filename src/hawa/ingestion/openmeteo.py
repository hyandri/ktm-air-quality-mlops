"""Pull RAW hourly weather from Open-Meteo. Nothing is cleaned here.

No API key needed. Free for non-commercial use, data licence CC BY 4.0
(keep the attribution). Free fair-use limit is about 10,000 calls per day.

Sources (--source)
  archive              ERA5 reanalysis, 1940 -> ~5 days ago
  historical_forecast  archived real forecasts, from 2021/2022
  forecast             live forecast, today -> +16 days (daily inference)

Run from the project root:
    python -m hawa.ingestion.openmeteo --source archive --start 2017-03-01 --end 2026-11-05
    python -m hawa.ingestion.openmeteo --source historical_forecast --start 2022-01-01 --end 2026-11-15

Requests go in 1-year chunks; finished chunks are cached in <out-dir>/_cache.
Times are UTC. Note: `precipitation` is the sum of the PRECEDING hour,
the other variables are instant values.
"""
import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from hawa import config
from hawa.ingestion.http import get_json

URLS = {
    "archive": "https://archive-api.open-meteo.com/v1/archive",
    "historical_forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "forecast": "https://api.open-meteo.com/v1/forecast",
    "previous_runs": "https://previous-runs-api.open-meteo.com/v1/forecast",
}


def fetch_hourly(source, lat, lon, start, end, variables):
    """One request -> DataFrame with a UTC `time` column plus the variables."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": "UTC",
    }
    data = get_json(URLS[source], params=params, timeout=90, pause_s=1.0)
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _year_chunks(start, end):
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    while s <= e:
        c_end = min(date(s.year, 12, 31), e)
        yield s.isoformat(), c_end.isoformat()
        s = c_end + timedelta(days=1)


def download(source, start, end, lat=config.EMBASSY_LAT, lon=config.EMBASSY_LON,
             variables=None, out_dir=config.OPENMETEO_RAW):
    """Download weather in year chunks and write one CSV. Returns its path."""
    variables = variables or config.WEATHER_VARIABLES
    out_dir = Path(out_dir)
    cache = out_dir / "_cache" / source
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for s, e in _year_chunks(start, end):
        f = cache / f"{lat}_{lon}_{s}_{e}.csv"
        if f.exists():
            print(f"{s}..{e}: cached")
            frames.append(pd.read_csv(f, parse_dates=["time"]))
            continue
        print(f"{s}..{e}: downloading")
        df = fetch_hourly(source, lat, lon, s, e, variables)
        if date.fromisoformat(e) < date.today() - timedelta(days=7):  # recent data can still change
            df.to_csv(f, index=False)
        frames.append(df)
    df = pd.concat(frames).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["source"] = source
    df["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
    out = out_dir / f"openmeteo_{source}_{lat}_{lon}.csv"
    df.to_csv(out, index=False)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["archive", "historical_forecast", "forecast"], required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--lat", type=float, default=config.EMBASSY_LAT)
    p.add_argument("--lon", type=float, default=config.EMBASSY_LON)
    p.add_argument("--variables", help="comma list; default = all candidates in config.py")
    p.add_argument("--out-dir", default=str(config.OPENMETEO_RAW))
    a = p.parse_args(argv)

    variables = a.variables.split(",") if a.variables else None
    out = download(a.source, a.start, a.end, a.lat, a.lon, variables, a.out_dir)
    df = pd.read_csv(out)
    cols = [v for v in (variables or config.WEATHER_VARIABLES) if v in df]
    print(f"\nSaved {len(df)} rows -> {out}")
    print(f"First {df['time'].min()} | Last {df['time'].max()}")
    print("Missing % per variable:\n" + df[cols].isna().mean().mul(100).round(1).to_string())


if __name__ == "__main__":
    main()
