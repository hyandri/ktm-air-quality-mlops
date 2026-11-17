"""Check what each Open-Meteo API really offers, before we build on it.

Sends a few small requests and prints, for each API and period, how many of our
candidate weather variables come back with real values, and which are empty.
Also shows how far back the Historical Forecast API goes.

Run from the project root:
    python -m hawa.analysis.openmeteo_probe
"""
import argparse
from datetime import date, timedelta

from hawa import config
from hawa.ingestion.http import ApiError
from hawa.ingestion.openmeteo import fetch_hourly


def probe_one(label, source, lat, lon, start, end, variables=None):
    variables = variables or config.WEATHER_VARIABLES
    try:
        df = fetch_hourly(source, lat, lon, start, end, variables)
    except ApiError as e:
        print(f"  [{label}] {source} {start}..{end}: FAILED -> {e}")
        return
    ok = [v for v in variables if v in df and df[v].notna().any()]
    empty = [v for v in variables if v not in ok]
    print(f"  [{label}] {source} {start}..{end}: {len(df)} rows | "
          f"data for {len(ok)}/{len(variables)} variables")
    if empty:
        print(f"      no data for: {', '.join(empty)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lat", type=float, default=config.EMBASSY_LAT)
    p.add_argument("--lon", type=float, default=config.EMBASSY_LON)
    a = p.parse_args(argv)
    lat, lon = a.lat, a.lon
    print(f"Location: {lat}, {lon}\n")

    print("1) ERA5 archive: start and end of our data range")
    probe_one("2017", "archive", lat, lon, "2017-03-01", "2017-03-03")
    probe_one("2025", "archive", lat, lon, "2025-12-01", "2025-12-03")
    recent = (date.today() - timedelta(days=10)).isoformat()
    probe_one("recent", "archive", lat, lon, recent, recent)

    print("\n2) Historical Forecast API: how far back does it really go?")
    for d in ["2021-01-01", "2021-06-01", "2022-01-01", "2022-06-01", "2023-01-01", "2024-01-01"]:
        end = (date.fromisoformat(d) + timedelta(days=2)).isoformat()
        probe_one(d, "historical_forecast", lat, lon, d, end)

    print("\n3) Live forecast (used for daily inference)")
    today = date.today()
    probe_one("next 3 days", "forecast", lat, lon, today.isoformat(), (today + timedelta(days=2)).isoformat())

    print("\n4) Previous Runs API: values as forecast 1 day earlier")
    names = [f"{v}_previous_day1" for v in ("temperature_2m", "wind_speed_10m", "precipitation")]
    probe_one("2024-02", "previous_runs", lat, lon, "2024-02-01", "2024-02-03", names)

    print("\n'no data for' lines = variables we cannot use from that source/period.")


if __name__ == "__main__":
    main()
