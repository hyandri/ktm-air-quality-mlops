"""Turn raw PM2.5 downloads into clean hourly files (one per station).

Inputs (raw files):
  - Open Data Nepal CSVs (2017 -> Mar 2021)      data/raw/open_data_nepal/
  - OpenAQ CSVs from hawa.ingestion.openaq        data/raw/openaq/

Cleaning rules (same for every file):
  1. keep only valid numbers: drop empty, -999, any negative value, error code 985
  2. one value per hour; timestamp = START of the hour, UTC
  3. drop flat runs: FLAT_RUN_HOURS or more hours in a row with the same value
Old Open Data Nepal stamps mean "end of the hour", so they are shifted back 1 hour.
If both sources have the same hour, the Open Data Nepal value is kept
(checked: they are identical once the shift is applied).

Run from the project root (defaults match the folder layout):
    python -m hawa.processing.clean
"""
import argparse
from pathlib import Path

import pandas as pd

from hawa import config


# ------------------------------------------------------------------- loaders
def read_odn(path):
    """Open Data Nepal long-format CSV -> frame with `hour_start` (UTC) and `v`."""
    d = pd.read_csv(path)
    d = d[d["parameter"] == "pm25"].copy()
    t = pd.to_datetime(d["utc"], utc=True)
    d["hour_start"] = t.dt.floor("h") - pd.Timedelta(hours=1)  # end-of-hour -> start
    d["v"] = pd.to_numeric(d["value"], errors="coerce")
    return d[["hour_start", "v"]]


def openaq_to_frame(raw):
    """OpenAQ raw table (from hawa.ingestion.openaq) -> frame with `hour_start`, `v`."""
    tcol = next(c for c in raw.columns if c.lower().endswith("datetimefrom.utc"))
    d = pd.DataFrame({
        "hour_start": pd.to_datetime(raw[tcol], utc=True).dt.floor("h"),
        "v": pd.to_numeric(raw["value"], errors="coerce"),
    })
    return d


def read_openaq(path):
    return openaq_to_frame(pd.read_csv(path))


# --------------------------------------------------------------- the rules
def valid_series(d):
    """Rules 1 and 2: valid numbers only, one per hour. Returns Series indexed by hour_start."""
    d = d[d["v"].notna() & (d["v"] >= 0) & ~d["v"].isin(config.ERROR_VALUES)]
    return d.sort_values(["hour_start", "v"]).drop_duplicates("hour_start").set_index("hour_start")["v"]


def drop_flat_runs(s):
    """Rule 3: remove runs of >= FLAT_RUN_HOURS identical values in consecutive hours.

    Returns (clean_series, number_of_hours_removed).
    """
    if s.empty:
        return s, 0
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
    run = (full != full.shift()).cumsum()
    size = full.groupby(run).transform("size")
    stuck = (size >= config.FLAT_RUN_HOURS) & full.notna()
    return full[~stuck].dropna(), int(stuck.sum())


def clean_frame(d):
    """Apply all three rules to a `hour_start`/`v` frame. Returns (series, stuck_hours)."""
    return drop_flat_runs(valid_series(d))


# ------------------------------------------------------------ build a station
def build_station(station, odn_path, openaq_path):
    a, stuck_a = clean_frame(read_odn(odn_path))
    b, stuck_b = clean_frame(read_openaq(openaq_path))
    both = pd.concat({"odn": a, "oaq": b}, axis=1)
    overlap = both.dropna()
    same = float((overlap["odn"] == overlap["oaq"]).mean()) if len(overlap) else float("nan")
    s = both["odn"].combine_first(both["oaq"])
    source = pd.Series("openaq", index=s.index)
    source[both["odn"].notna()] = "open_data_nepal"
    out = pd.DataFrame({"timestamp_utc": s.index, "station": station,
                        "pm25": s.values, "source": source.values})
    print(f"{station}: {len(out)} valid hours | {out.timestamp_utc.min()} -> {out.timestamp_utc.max()}")
    print(f"  stuck hours dropped: odn={stuck_a} openaq={stuck_b} | "
          f"overlap hours={len(overlap)}, identical={same:.1%}")
    return out


def winter_coverage(df, first_year=2017, last_year=2025):
    """Share of Dec-Feb hours that have a valid value, per winter (text, for printing)."""
    t = pd.DatetimeIndex(df["timestamp_utc"])
    parts = []
    for y in range(first_year, last_year + 1):
        lo, hi = pd.Timestamp(f"{y}-12-01", tz="UTC"), pd.Timestamp(f"{y + 1}-03-01", tz="UTC")
        n = ((t >= lo) & (t < hi)).sum()
        parts.append(f"{y}/{str(y + 1)[2:]}: {100 * n / ((hi - lo).days * 24):.0f}%")
    return " | ".join(parts)


def main(argv=None):
    odn, oaq, out = config.OPEN_DATA_NEPAL_RAW, config.OPENAQ_RAW, config.PROCESSED_DIR
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embassy-odn", default=str(odn / "us_embassy.csv"))
    p.add_argument("--embassy-oaq", default=str(oaq / "loc3459_sensor7710_pm25_measurements.csv"))
    p.add_argument("--phora-odn", default=str(odn / "phora_durbar.csv"))
    p.add_argument("--phora-oaq", default=str(oaq / "loc3460_sensor7711_pm25_measurements.csv"))
    p.add_argument("--out-dir", default=str(out))
    a = p.parse_args(argv)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stations = {
        "embassy": build_station("us_embassy", a.embassy_odn, a.embassy_oaq),
        "phora": build_station("phora_durbar", a.phora_odn, a.phora_oaq),
    }
    for name, df in stations.items():
        df.to_csv(out_dir / f"{name}_pm25_hourly_clean.csv", index=False)
        print(f"{name} winter coverage -> {winter_coverage(df)}")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
