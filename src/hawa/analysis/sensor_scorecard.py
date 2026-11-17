"""Rate live OpenAQ sensors by how complete their PM2.5 data is.

Step 1 of the data research: which sensors are good enough for live forecasts?
For each selected location this downloads raw PM2.5 (via hawa.ingestion.openaq),
cleans it with the project rules (hawa.processing.clean) and counts VALID hours only.

Needs data/raw/openaq/kathmandu_locations.csv (create it with
`python -m hawa.ingestion.openaq locations`).

Run from the project root:
    python -m hawa.analysis.sensor_scorecard --start 2025-09-01
    python -m hawa.analysis.sensor_scorecard --max 3        # quick trial
    python -m hawa.analysis.sensor_scorecard --ids 6093549,5676345

Safe to stop and re-run: downloads are cached.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hawa import config
from hawa.ingestion import openaq
from hawa.processing import clean


def sensor_stats(raw):
    """Coverage numbers for one sensor from its raw table (valid hours only)."""
    s, stuck = clean.clean_frame(clean.openaq_to_frame(raw))
    out = {"rows": len(raw), "valid_hours": len(s), "stuck_hours": stuck}
    v = pd.to_numeric(raw.get("value"), errors="coerce")
    out["n_raw_null"] = int(v.isna().sum())
    if "flagInfo.hasFlags" in raw.columns:
        out["pct_flagged"] = round(100 * raw["flagInfo.hasFlags"].astype(bool).mean(), 1)
    if s.empty:
        return {**out, "missing_pct": 100.0}
    t = s.index
    full = pd.date_range(t.min(), t.max(), freq="h")
    present = full.isin(t)
    gaps = t.to_series().diff().dropna()
    winter = full.month.isin([12, 1, 2])
    return {
        **out,
        "first_utc": t.min(),
        "last_utc": t.max(),
        "expected_hours": len(full),
        "missing_pct": round(100 * (1 - present.mean()), 1),
        "winter_missing_pct": round(100 * (1 - present[winter].mean()), 1) if winter.any() else None,
        "longest_gap_h": int(gaps.max() / pd.Timedelta(hours=1)) - 1 if len(gaps) else 0,
        "gaps_over_24h": int((gaps > pd.Timedelta(hours=24)).sum()),
        "median_value": round(float(s.median()), 1),
    }


def select_locations(loc, ids, max_days_since_last, min_history_days):
    loc = loc.copy()
    loc["first_utc"] = pd.to_datetime(loc["first_utc"], utc=True)
    loc["last_utc"] = pd.to_datetime(loc["last_utc"], utc=True)
    loc["days_since_last"] = (pd.Timestamp.now(tz="UTC") - loc["last_utc"]).dt.days
    if ids:
        sel = loc[loc["location_id"].isin({int(x) for x in ids.split(",")})]
    else:
        sel = loc[(loc["days_since_last"] <= max_days_since_last)
                  & (loc["history_days"] >= min_history_days)]
    return sel.sort_values("first_utc")


def main(argv=None):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--locations", default=str(config.OPENAQ_RAW / "kathmandu_locations.csv"))
    p.add_argument("--start", default="2025-09-01", help="YYYY-MM-DD")
    p.add_argument("--end", default=today)
    p.add_argument("--min-history-days", type=int, default=300)
    p.add_argument("--max-days-since-last", type=int, default=7)
    p.add_argument("--ids", help="comma-separated location IDs (overrides the filters)")
    p.add_argument("--max", type=int, help="only the first N locations (trial run)")
    p.add_argument("--out", default=str(config.OPENAQ_RAW / "scorecard.csv"))
    p.add_argument("--out-dir", default=str(config.OPENAQ_RAW))
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--endpoint", choices=["hours", "measurements"], default="hours")
    p.add_argument("--timeout", type=int, default=120)
    a = p.parse_args(argv)

    sel = select_locations(pd.read_csv(a.locations), a.ids, a.max_days_since_last, a.min_history_days)
    if a.max:
        sel = sel.head(a.max)
    print(f"{len(sel)} locations selected. Safe to stop and re-run.\n")

    results = []
    for n, (_, row) in enumerate(sel.iterrows(), 1):
        lid = int(row["location_id"])
        print(f"[{n}/{len(sel)}] {lid} {row['name']}")
        paths = openaq.download_location(lid, a.start, a.end, ("pm25",), a.out_dir,
                                         a.window_days, a.endpoint, a.timeout, quiet=True)
        if not paths:
            print("  no pm25 data, skipped")
            continue
        raw = pd.read_csv(paths[0])
        q = sensor_stats(raw)
        q.update({"location_id": lid, "name": row["name"], "provider": row.get("provider"),
                  "sensor_id": int(raw["sensor_id"].iloc[0])})
        results.append(q)
        print(f"  valid_hours={q['valid_hours']} missing={q['missing_pct']}% "
              f"longest_gap_h={q.get('longest_gap_h')}")
        pd.DataFrame(results).to_csv(a.out, index=False)  # save as we go

    if not results:
        sys.exit("No results.")
    sc = pd.DataFrame(results)
    cols = ["location_id", "name", "provider", "first_utc", "last_utc", "valid_hours",
            "missing_pct", "winter_missing_pct", "longest_gap_h", "gaps_over_24h",
            "median_value", "n_raw_null", "pct_flagged"]
    sc = sc.sort_values("missing_pct")[[c for c in cols if c in sc.columns]]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sc.to_csv(a.out, index=False)
    print(f"\nScorecard saved to {a.out}\n")
    print(sc.to_string(index=False))


if __name__ == "__main__":
    main()
