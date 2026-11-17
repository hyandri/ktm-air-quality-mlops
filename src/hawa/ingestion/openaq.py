"""Pull RAW data from the OpenAQ v3 API. Nothing is cleaned or judged here.

This module only answers two questions:
  * which locations exist?                       -> `locations`
  * give me the history of one location          -> `download`
Raw API answers are cached as JSON and also written as flat CSV files under
data/raw/openaq/. Quality checks and cleaning live in `hawa.processing`.

Setup (once): put your key in a `.env` file at the project root:
    OPENAQ_API_KEY=your-key-here

Run from the project root:
    python -m hawa.ingestion.openaq locations
    python -m hawa.ingestion.openaq download --location-id 3459 \\
        --start 2021-03-01 --end 2026-11-16 --params pm25

Options for `download`:
    --window-days 7          smaller requests (use this if you get timeouts)
    --endpoint measurements  original values instead of hourly averages
    --timeout 180            seconds to wait for each request

Finished windows are cached in <out-dir>/_cache, so an interrupted run
continues where it stopped. Windows that end in the last 3 days are not
cached, so they refresh next time.
Free limits: 60 requests/minute, 2,000/hour (the code waits between calls).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from hawa import config
from hawa.ingestion.http import get_json

BASE_URL = "https://api.openaq.org/v3"
PAGE_LIMIT = 1000


# ------------------------------------------------------------------ API calls
def _get(path, params=None, timeout=120):
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        sys.exit("OPENAQ_API_KEY is not set. Add it to the .env file in the project root.")
    return get_json(f"{BASE_URL}{path}", params=params, headers={"X-API-Key": key},
                    timeout=timeout, pause_s=1.1)  # stay under 60 requests/minute


def _get_all_pages(path, params, timeout=120):
    results, page = [], 1
    while page <= 200:  # safety stop
        data = _get(path, {**params, "limit": PAGE_LIMIT, "page": page}, timeout)
        batch = data.get("results", [])
        results.extend(batch)
        if len(batch) < PAGE_LIMIT:
            break
        page += 1
    return results


def _utc(obj):
    """Pull the UTC string out of {'utc': ..., 'local': ...} (or None)."""
    return obj.get("utc") if isinstance(obj, dict) else None


def _to_iso(d):
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_day(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


# ------------------------------------------------------------------ locations
def list_locations(bbox=config.KATHMANDU_BBOX, timeout=120):
    """Every location in the box that measures PM2.5, as a DataFrame."""
    found = _get_all_pages(
        "/locations", {"bbox": bbox, "parameters_id": config.OPENAQ_PM25_PARAMETER_ID}, timeout
    )
    rows = []
    for x in found:
        c = x.get("coordinates") or {}
        rows.append({
            "location_id": x.get("id"),
            "name": x.get("name"),
            "provider": (x.get("provider") or {}).get("name"),
            "owner": (x.get("owner") or {}).get("name"),
            "is_monitor": x.get("isMonitor"),
            "lat": c.get("latitude"),
            "lon": c.get("longitude"),
            "first_utc": _utc(x.get("datetimeFirst")),
            "last_utc": _utc(x.get("datetimeLast")),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["first_utc"] = pd.to_datetime(df["first_utc"], utc=True, errors="coerce")
    df["last_utc"] = pd.to_datetime(df["last_utc"], utc=True, errors="coerce")
    df["history_days"] = (df["last_utc"] - df["first_utc"]).dt.days
    df["days_since_last"] = (pd.Timestamp.now(tz="UTC") - df["last_utc"]).dt.days
    return df.sort_values("first_utc").reset_index(drop=True)


def get_sensors(location_id, timeout=120):
    """Sensors (one per measured parameter) at a location."""
    return _get(f"/locations/{location_id}/sensors", timeout=timeout).get("results", [])


# ------------------------------------------------------------------- download
def _fetch_rows(sensor_id, endpoint, start, end, step, cache_dir, timeout, label, quiet):
    """Download one sensor window by window. Finished windows are cached as JSON."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fresh_after = datetime.now(timezone.utc) - timedelta(days=3)
    n_windows = -(-(end - start) // step)
    rows, cur, i = [], start, 0
    while cur < end:
        i += 1
        nxt = min(cur + step, end)
        f = cache_dir / f"{cur:%Y%m%d}_{nxt:%Y%m%d}.json"
        if f.exists():
            batch = json.loads(f.read_text())
        else:
            batch = _get_all_pages(
                f"/sensors/{sensor_id}/{endpoint}",
                {"datetime_from": _to_iso(cur), "datetime_to": _to_iso(nxt)},
                timeout,
            )
            if nxt < fresh_after:  # do not cache windows that are still filling
                f.write_text(json.dumps(batch))
        rows += batch
        if not quiet:
            print(f"  {label} window {i}/{n_windows} {cur:%Y-%m-%d}: {len(batch)} rows")
        cur = nxt
    return rows


def _rows_to_frame(rows, location_id, sensor_id, parameter):
    """Flatten the API's nested JSON into a table. No values are changed."""
    df = pd.json_normalize(rows)
    df.insert(0, "location_id", location_id)
    df.insert(1, "sensor_id", sensor_id)
    df.insert(2, "parameter_name", parameter)
    df["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
    time_cols = [c for c in df.columns if c.lower().endswith("datetimefrom.utc")]
    if time_cols:
        df = df.drop_duplicates(subset=time_cols[0]).sort_values(time_cols[0]).reset_index(drop=True)
    return df


def fetch_recent_hours(location_id, sensor_id, hours_back=240, now=None, timeout=60):
    """Raw hourly PM2.5 of one sensor for the last `hours_back` hours (no cache: live data changes).

    Returns the same flat table as `download_location` writes, or an empty table if the API has no rows.
    """
    now = now or datetime.now(timezone.utc)
    rows = _get_all_pages(f"/sensors/{sensor_id}/hours",
                          {"datetime_from": _to_iso(now - timedelta(hours=hours_back)),
                           "datetime_to": _to_iso(now)}, timeout)
    if not rows:
        return pd.DataFrame()
    return _rows_to_frame(rows, location_id, sensor_id, "pm25")


def download_location(location_id, start, end, params=("pm25",), out_dir=config.OPENAQ_RAW,
                      window_days=30, endpoint="hours", timeout=120, quiet=False):
    """Download raw history of the wanted parameters at one location.

    Returns the list of CSV files written:
    <out_dir>/loc<ID>_sensor<SID>_<parameter>_<endpoint>.csv
    """
    wanted = set(params)
    sensors = get_sensors(location_id, timeout)
    chosen = [s for s in sensors if (s.get("parameter") or {}).get("name") in wanted]
    if not quiet:
        print("Sensors at location:", [(s.get("id"), (s.get("parameter") or {}).get("name"))
                                       for s in sensors])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_d, end_d = _parse_day(start), _parse_day(end)
    written = []
    for s in chosen:
        sid, name = s["id"], s["parameter"]["name"]
        cache = out_dir / "_cache" / f"sensor{sid}_{endpoint}"
        rows = _fetch_rows(sid, endpoint, start_d, end_d, timedelta(days=window_days),
                           cache, timeout, name, quiet)
        if not rows:
            if not quiet:
                print(f"sensor {sid} ({name}): no rows")
            continue
        df = _rows_to_frame(rows, location_id, sid, name)
        path = out_dir / f"loc{location_id}_sensor{sid}_{name}_{endpoint}.csv"
        df.to_csv(path, index=False)
        written.append(path)
    return written


# ------------------------------------------------------------------------ CLI
def _cmd_locations(a):
    df = list_locations(a.bbox, a.timeout)
    if df.empty:
        sys.exit("No locations returned. Check the bbox and your key.")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df)} PM2.5 locations in the box. Saved to {out}\n")
    print("By provider:\n", df["provider"].value_counts(dropna=False).to_string(), "\n")
    cols = ["location_id", "name", "provider", "first_utc", "history_days", "days_since_last"]
    print(df[cols].head(a.show).to_string(index=False))
    live = df[df["days_since_last"] <= 7]
    print(f"\nLive (data in the last 7 days): {len(live)} of {len(df)}")
    print("Note: first/last dates are for the whole location, not for one sensor.")


def _cmd_download(a):
    params = [p.strip() for p in a.params.split(",")]
    paths = download_location(a.location_id, a.start, a.end, params, a.out_dir,
                              a.window_days, a.endpoint, a.timeout)
    if not paths:
        sys.exit(f"Nothing downloaded for {params} at location {a.location_id}.")
    for p in paths:
        df = pd.read_csv(p)
        tcol = next(c for c in df.columns if c.lower().endswith("datetimefrom.utc"))
        print(f"\n{p}\n  rows={len(df)}  first={df[tcol].min()}  last={df[tcol].max()}")
    print("\nThis is RAW data. For quality numbers run the cleaning step (hawa.processing.clean).")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    loc = sub.add_parser("locations", help="list PM2.5 locations in a box")
    loc.add_argument("--bbox", default=config.KATHMANDU_BBOX)
    loc.add_argument("--out", default=str(config.OPENAQ_RAW / "kathmandu_locations.csv"))
    loc.add_argument("--show", type=int, default=80, help="rows to print")
    loc.add_argument("--timeout", type=int, default=120)
    loc.set_defaults(func=_cmd_locations)

    dl = sub.add_parser("download", help="download raw history for one location")
    dl.add_argument("--location-id", type=int, required=True)
    dl.add_argument("--start", required=True, help="YYYY-MM-DD")
    dl.add_argument("--end", required=True, help="YYYY-MM-DD")
    dl.add_argument("--params", default="pm25")
    dl.add_argument("--out-dir", default=str(config.OPENAQ_RAW))
    dl.add_argument("--window-days", type=int, default=30)
    dl.add_argument("--endpoint", choices=["hours", "measurements"], default="hours")
    dl.add_argument("--timeout", type=int, default=120)
    dl.set_defaults(func=_cmd_download)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
