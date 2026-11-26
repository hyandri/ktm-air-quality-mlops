"""Calibration experiment: can one simple number fix the scale difference between the sensors and the Embassy?

The model learned the Embassy. Live sensors (AirGradient) read higher (about 1.56x in winter 2025/26). Questions:
  1. How big is the ratio sensor / Embassy, and is it the same in every month?   (diagnosis, all paired hours)
  2. Does dividing the sensor by a fixed factor give the model better input?     (test, locked winter only)

How the test is kept honest
  * The factor is FITTED only on paired hours OUTSIDE the locked test winters (config.TEST_WINTERS), i.e. never on
    Dec 2024 - Feb 2026. It is SCORED on winter 2025/26 (--winter 2025), which the factor never saw.
  * Two factors are compared with "no correction" (raw):
      pooled  median of the OTHER sensors' own ratios (leave one out). This is what a new sensor would get.
      own     the sensor's own ratio. Needs Embassy overlap, so it is only a ceiling.
  * Ratio = mean(sensor) / mean(Embassy) over the paired hours. Correction = sensor / factor.
  * Two scores per variant, on the winter's hours:
      match  corrected sensor vs Embassy at the same hours: MAE and bias (does the scale now agree?)
      gain   frozen model (trained before 2024-12-01) on the corrected sensor series: % lower MAE than the best
             baseline (same score as hawa.analysis.sensor_transfer; it is on corrected units for both).

Decision rule (written BEFORE looking at results). Adopt the pooled factor only if ALL hold:
  - median gain improves by at least ADOPT_MIN_GAIN_POINTS over raw,
  - it improves on at least ADOPT_MIN_SHARE of the sensors,
  - the median match MAE (vs Embassy) is lower than raw.
Seasonality warning: if the monthly median ratios differ by more than SEASON_SPREAD (max / min), a single factor is
not enough and the verdict says so, whatever the numbers above say.

Limits
  * A sensor sits at another place than the Embassy: the ratio mixes instrument bias and real local differences.
    Only sensors within --max-km of the Embassy are used (default 10).
  * The winter 2025/26 transfer numbers were also used to choose the live sensor shortlist (O9): optimistic.
  * Winter 2025/26 is the ONLY winter with many sensors. One winter, so this is evidence, not proof.

Run from the project root:
    python -m hawa.models.train --train-until 2024-12-01 --no-track     # once: the frozen model (if not yet made)
    python -m hawa.analysis.calibration

Needs:  data/raw/openaq/loc<ID>_sensor<SID>_pm25_hours.csv, data/raw/openaq/kathmandu_locations.csv,
        data/processed/embassy_pm25_hourly_clean.csv, and the frozen model bundle in models/
Writes: reports/calibration_monthly.csv (ratio per sensor and month), reports/calibration_test.csv (per sensor)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hawa import config
from hawa.analysis import sensor_transfer as st
from hawa.evaluation import splits
from hawa.models import bundle
from hawa.processing import build_dataset

FIT_MIN_H = 500            # paired hours outside the test winters needed to trust a sensor's own ratio
MONTH_MIN_H = 100          # paired hours needed to report a month
ADOPT_MIN_GAIN_POINTS = 2.0
ADOPT_MIN_SHARE = 0.6
SEASON_SPREAD = 1.25
VARIANTS = ("raw", "pooled", "own")


# -------------------------------------------------------------------------------------- diagnosis
def local_month(index):
    return (index + pd.Timedelta(hours=config.LOCAL_UTC_OFFSET_H)).month


def pair(sensor, embassy):
    """Hours where both series are valid. Columns: s (sensor), e (Embassy)."""
    return pd.concat({"s": sensor, "e": embassy}, axis=1, sort=True).dropna()


def monthly_table(pairs, min_hours=MONTH_MIN_H):
    """Per calendar month (Kathmandu time): paired hours, ratio of means, correlation."""
    rows = []
    for month, d in pairs.groupby(local_month(pairs.index)):
        if len(d) >= min_hours:
            rows.append({"month": int(month), "n": len(d), "ratio": float(d["s"].mean() / d["e"].mean()),
                         "corr": float(d["s"].corr(d["e"]))})
    return pd.DataFrame(rows, columns=["month", "n", "ratio", "corr"])


# -------------------------------------------------------------------------------------- fitting
def fit_pairs(pairs):
    """Paired hours that may be used to FIT a factor: everything outside the locked test winters."""
    in_test = pd.Series(splits.winter_start_year(pairs.index)).isin(config.TEST_WINTERS).to_numpy()
    return pairs[~in_test]


def fit_ratio(pairs, min_hours=FIT_MIN_H):
    fp = fit_pairs(pairs)
    if len(fp) < min_hours:
        return np.nan
    return float(fp["s"].mean() / fp["e"].mean())


def pooled_factor(own_ratios, exclude):
    """Median of the other sensors' own ratios (leave one out). NaN if no other sensor has one."""
    others = [v for k, v in own_ratios.items() if k != exclude and np.isfinite(v)]
    return float(np.median(others)) if others else np.nan


# -------------------------------------------------------------------------------------- scoring
def winter_match(corrected, embassy, winter_year, min_hours=200):
    """Corrected sensor vs Embassy at the winter's paired hours: MAE and bias (sensor minus Embassy)."""
    both = pair(corrected, embassy)
    both = both[splits.winter_start_year(both.index) == winter_year]
    if len(both) < min_hours:
        return {"mae": np.nan, "bias": np.nan}
    diff = both["s"] - both["e"]
    return {"mae": float(diff.abs().mean()), "bias": float(diff.mean())}


def evaluate_sensor(series, embassy, factors, model_bundle, winter_year, min_origins):
    """Score the variants (raw / pooled / own) of one sensor. `factors`: {"raw": 1.0, "pooled": x, "own": y}."""
    out = {}
    for name in VARIANTS:
        f = factors.get(name, np.nan)
        if not np.isfinite(f) or f <= 0:
            out.update({f"{name}_gain": np.nan, f"{name}_match_mae": np.nan, f"{name}_bias": np.nan})
            continue
        corrected = series / f
        match = winter_match(corrected, embassy, winter_year)
        out[f"{name}_gain"] = st.transfer_score(model_bundle, st.feature_table(corrected), winter_year,
                                                min_origins)["gain_pct"]
        out[f"{name}_match_mae"], out[f"{name}_bias"] = match["mae"], match["bias"]
    return out


# -------------------------------------------------------------------------------------- verdict
def season_spread(monthly):
    """(max / min) of the per-calendar-month MEDIAN ratio across sensors. NaN if fewer than 2 months."""
    if monthly.empty:
        return np.nan
    med = monthly.groupby("month")["ratio"].median()
    return float(med.max() / med.min()) if len(med) >= 2 else np.nan


def verdict(test, monthly):
    """Apply the decision rule. Returns (adopt: bool, facts: dict, text: str)."""
    d = test.dropna(subset=["raw_gain", "pooled_gain"])
    facts = {"n_sensors": int(len(d)), "season_spread": season_spread(monthly)}
    if d.empty:
        return False, facts, "No sensor could be scored with both raw and pooled input: no verdict."
    improvement = d["pooled_gain"] - d["raw_gain"]
    facts.update(median_gain_raw=float(d["raw_gain"].median()), median_gain_pooled=float(d["pooled_gain"].median()),
                 median_improvement=float(improvement.median()), share_improved=float((improvement > 0).mean()),
                 median_match_raw=float(d["raw_match_mae"].median()),
                 median_match_pooled=float(d["pooled_match_mae"].median()))
    checks = [facts["median_improvement"] >= ADOPT_MIN_GAIN_POINTS,
              facts["share_improved"] >= ADOPT_MIN_SHARE,
              facts["median_match_pooled"] < facts["median_match_raw"]]
    seasonal = np.isfinite(facts["season_spread"]) and facts["season_spread"] > SEASON_SPREAD
    adopt = all(checks) and not seasonal
    reasons = []
    if not checks[0]:
        reasons.append(f"median gain improves by {facts['median_improvement']:.1f} points "
                       f"(need {ADOPT_MIN_GAIN_POINTS})")
    if not checks[1]:
        reasons.append(f"better on {100 * facts['share_improved']:.0f}% of sensors (need {100 * ADOPT_MIN_SHARE:.0f}%)")
    if not checks[2]:
        reasons.append("the scale match with the Embassy did not improve")
    if seasonal:
        reasons.append(f"the monthly ratio changes by {facts['season_spread']:.2f}x (limit {SEASON_SPREAD}): "
                       "one factor does not fit every season")
    text = ("ADOPT the pooled factor." if adopt else "DO NOT adopt a single factor: " + "; ".join(reasons) + ".")
    return adopt, facts, text


# -------------------------------------------------------------------------------------- run
def run(raw_dir, locations_path, model_dir, embassy_clean, winter_year, min_origins, max_km):
    model_bundle = bundle.load_bundle(model_dir)
    loc = pd.read_csv(locations_path).set_index("location_id")
    embassy = build_dataset.read_pm25(embassy_clean).dropna()

    sensors, skipped = {}, 0
    for loc_id, path in st.sensor_files(raw_dir):
        if loc_id not in loc.index:
            skipped += 1
            continue
        info = loc.loc[loc_id]
        dist = st.haversine_km(config.EMBASSY_LAT, config.EMBASSY_LON, info["lat"], info["lon"])
        if dist > max_km:
            skipped += 1
            continue
        try:
            series = st.load_sensor_series(path)
        except (KeyError, StopIteration, ValueError):
            skipped += 1
            continue
        if not series.empty:
            sensors[loc_id] = {"series": series, "name": info["name"], "dist_km": dist}
    print(f"sensors within {max_km} km of the Embassy: {len(sensors)} (left out: {skipped})")

    monthly_rows, own = [], {}
    for loc_id, s in sensors.items():
        pairs = pair(s["series"], embassy)
        own[loc_id] = fit_ratio(pairs)
        for r in monthly_table(pairs).to_dict("records"):
            monthly_rows.append({"location_id": loc_id, "name": s["name"], **r})
    monthly = pd.DataFrame(monthly_rows, columns=["location_id", "name", "month", "n", "ratio", "corr"])

    rows = []
    for loc_id, s in sensors.items():
        factors = {"raw": 1.0, "pooled": pooled_factor(own, loc_id), "own": own[loc_id]}
        row = {"location_id": loc_id, "name": s["name"], "dist_km": s["dist_km"],
               "own_ratio": own[loc_id], "pooled_factor": factors["pooled"]}
        row.update(evaluate_sensor(s["series"], embassy, factors, model_bundle, winter_year, min_origins))
        rows.append(row)
    test = pd.DataFrame(rows).sort_values("dist_km").reset_index(drop=True)
    return monthly, test


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="frozen bundle folder (default: the one trained with "
                                   f"--train-until {st.FROZEN_UNTIL})")
    p.add_argument("--raw-dir", default=str(config.OPENAQ_RAW))
    p.add_argument("--locations", default=str(config.OPENAQ_RAW / "kathmandu_locations.csv"))
    p.add_argument("--embassy-clean", default=str(config.PROCESSED_DIR / "embassy_pm25_hourly_clean.csv"))
    p.add_argument("--winter", type=int, default=2025, help="scored winter start year (2025 = Dec 2025 - Feb 2026)")
    p.add_argument("--max-km", type=float, default=10.0, help="only sensors this close to the Embassy")
    p.add_argument("--min-origins", type=int, default=500, help="fewer scored hours than this: no gain score")
    p.add_argument("--out-dir", default=str(config.REPORTS_DIR))
    a = p.parse_args(argv)

    model_dir = Path(a.model) if a.model else st.find_frozen_bundle()
    if model_dir is None:
        raise SystemExit(f"No frozen model found. Run first:\n  python -m hawa.models.train "
                         f"--train-until {st.FROZEN_UNTIL} --no-track")
    print(f"frozen model: {model_dir.name}")
    monthly, test = run(a.raw_dir, a.locations, model_dir, a.embassy_clean, a.winter, a.min_origins, a.max_km)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(out / "calibration_monthly.csv", index=False)
    test.to_csv(out / "calibration_test.csv", index=False)

    print("\n1. Ratio sensor / Embassy by calendar month (across sensors; ratio = mean sensor / mean Embassy)")
    if monthly.empty:
        print("   no month has enough paired hours")
    else:
        g = monthly.groupby("month")["ratio"].agg(sensors="count", median="median", low="min", high="max")
        print(g.round(2).to_string())
    spread = season_spread(monthly)
    print(f"   spread of the monthly medians (max/min): {spread:.2f}  (a single factor needs <= {SEASON_SPREAD})")

    show = test.copy()
    show["name"] = show["name"].astype(str).str.slice(0, 22)
    cols = ["name", "dist_km", "own_ratio", "pooled_factor", "raw_gain", "pooled_gain", "own_gain",
            "raw_match_mae", "pooled_match_mae", "raw_bias", "pooled_bias"]
    print(f"\n2. Winter {a.winter}/{str(a.winter + 1)[2:]}: factors were fitted OUTSIDE the locked test winters")
    print("   gain = % lower MAE than the best baseline; match_mae = corrected sensor vs Embassy (ug/m3); "
          "bias = sensor - Embassy\n")
    print(show[cols].round(2).to_string(index=False))

    adopt, facts, text = verdict(test, monthly)
    print("\n3. Decision rule (fixed before the run)")
    for k in ("n_sensors", "median_gain_raw", "median_gain_pooled", "median_improvement", "share_improved",
              "median_match_raw", "median_match_pooled"):
        if k in facts:
            print(f"   {k}: {facts[k]:.2f}" if isinstance(facts[k], float) else f"   {k}: {facts[k]}")
    print(f"   -> {text}")
    print(f"\nSaved -> {out / 'calibration_monthly.csv'}, {out / 'calibration_test.csv'}")


if __name__ == "__main__":
    main()
