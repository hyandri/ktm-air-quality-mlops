"""Does OpenAQ have newer PM2.5 data for the training stations than our raw files? Report only.

Our Embassy training data ends where the last download ended (2026-06-17). That may be only the download date.
This check downloads the period since a start date into a SCRATCH folder (data/interim/openaq_fresh/, not data/raw),
cleans it with the project rules, and reports:
  - the newest valid hour per station and the number of valid hours since our raw data ended,
  - valid hours per month (so gaps are visible),
  - on the days that both our raw file and the new download cover: how many hours agree (same data source?).

Run from the project root (needs OPENAQ_API_KEY in .env):
    python -m hawa.analysis.data_freshness                       # from 2026-05-15 to today
    python -m hawa.analysis.data_freshness --start 2026-06-01 --station embassy

Nothing in data/raw or data/processed is changed. Writes reports/data_freshness.csv (monthly valid hours).
Adding the new data to the pipeline is a separate, deliberate step (it needs `dvc add data/raw` afterwards).
"""
import argparse
from datetime import date

import pandas as pd

from hawa import config
from hawa.ingestion import openaq
from hawa.processing import clean

STATIONS = {  # name: (location id, sensor id, raw file already in data/raw/openaq)
    "embassy": (3459, 7710, "loc3459_sensor7710_pm25_measurements.csv"),
    "phora": (3460, 7711, "loc3460_sensor7711_pm25_measurements.csv"),
}


def valid_hours(raw):
    """Raw OpenAQ table -> clean hourly Series (project cleaning rules)."""
    series, _ = clean.clean_frame(clean.openaq_to_frame(raw))
    return series


def summarise(new, old, since):
    """Compare a fresh Series with the Series already in our raw file.

    Returns a dict: newest valid hour, valid hours after our data ended, monthly valid hours (as a Series),
    and the overlap check (hours in both, share that are identical).
    """
    old_last = old.index.max() if len(old) else None
    after = new[new.index > old_last] if old_last is not None else new
    both = new.index.intersection(old.index)
    same = float((new.loc[both].round(3) == old.loc[both].round(3)).mean()) if len(both) else None
    monthly = new.groupby(new.index.strftime("%Y-%m")).size()
    return {"old_last_utc": old_last, "new_first_utc": new.index.min() if len(new) else None,
            "new_last_utc": new.index.max() if len(new) else None, "valid_hours_after_old_end": int(len(after)),
            "overlap_hours": int(len(both)), "overlap_identical_share": same, "monthly": monthly,
            "since": since}


def check_station(name, start, end, fetch=openaq.download_location, out_dir=config.INTERIM_DIR / "openaq_fresh"):
    loc, sensor, raw_file = STATIONS[name]
    paths = fetch(loc, start, end, ("pm25",), out_dir, endpoint="hours", quiet=True)
    if not paths:
        return {"station": name, "error": "OpenAQ returned no rows for this period"}
    new = valid_hours(pd.read_csv(paths[0]))
    old = valid_hours(pd.read_csv(config.OPENAQ_RAW / raw_file))
    return {"station": name, **summarise(new, old, start)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2026-05-15", help="YYYY-MM-DD (use a date a bit before our raw data ends)")
    p.add_argument("--end", default=date.today().isoformat(), help="YYYY-MM-DD (default: today)")
    p.add_argument("--station", choices=list(STATIONS), help="default: all")
    p.add_argument("--out", default=str(config.REPORTS_DIR / "data_freshness.csv"))
    a = p.parse_args(argv)

    rows = []
    for name in ([a.station] if a.station else list(STATIONS)):
        r = check_station(name, a.start, a.end)
        if "error" in r:
            print(f"{name}: {r['error']}")
            continue
        print(f"\n{name}")
        print(f"  our raw data ends:        {r['old_last_utc']}")
        print(f"  new download covers:      {r['new_first_utc']}  ->  {r['new_last_utc']}")
        print(f"  valid hours AFTER our end: {r['valid_hours_after_old_end']}")
        share = r["overlap_identical_share"]
        print(f"  overlap: {r['overlap_hours']} hours in both, "
              f"{'n/a' if share is None else f'{100 * share:.1f}% identical'}")
        print("  valid hours per month:", r["monthly"].to_dict())
        rows += [{"station": name, "month": m, "valid_hours": int(n)} for m, n in r["monthly"].items()]
    if rows:
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
