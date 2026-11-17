"""Sensor transfer test: does the Embassy-trained model still work on live (AirGradient) sensors?

The model was trained on Embassy PM2.5. The live product will read other sensors. For every downloaded
OpenAQ sensor file this script does two checks, for one winter (default 2025/26, Dec-Feb):

  A. Transfer score  the FROZEN model (trained only on data before 2024-12-01) gets the sensor's own PM2.5
                     history as input and is scored against the same two baselines on the sensor's own
                     future values. Gain > 0 means the model still beats the bar on that sensor.
                     The same model is also scored on the Embassy for the same winter (reference).
  B. Agreement       the sensor against the Embassy at the same hours: correlation, bias, ratio of means.

Limits (read before quoting any number):
  * Each sensor has about one winter of history, so a few thousand hours at best.
  * Sensors sit at different places. Agreement with the Embassy shows the readings look alike, not that the
    instruments are interchangeable.
  * The winter 2025/26 is one of our locked test winters for the Embassy model. Nothing about the model is
    changed here; if a later step uses these numbers (for example a calibration), treat it as a new
    experiment with its own check.

Run from the project root:
    python -m hawa.models.train --train-until 2024-12-01 --no-track     # once: makes the frozen model
    python -m hawa.analysis.sensor_transfer
    python -m hawa.analysis.sensor_transfer --track                     # also write an MLflow run

Needs:  data/raw/openaq/loc<ID>_sensor<SID>_pm25_hours.csv, data/raw/openaq/kathmandu_locations.csv,
        data/processed/embassy_pm25_hourly_clean.csv, data/processed/embassy_features_auto.csv,
        and the frozen model bundle in models/
Writes: reports/sensor_transfer.csv
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from hawa import config
from hawa.evaluation import metrics, splits
from hawa.evaluation.evaluate_baselines import load_dataset
from hawa.models import baselines, bundle, learners
from hawa.processing import build_dataset, clean, features

FILE_RE = re.compile(r"loc(\d+)_sensor(\d+)_pm25_hours\.csv$")
FROZEN_UNTIL = "2024-12-01"


def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance between two points on the Earth, in km."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def sensor_files(raw_dir):
    """[(location_id, path)] for every downloaded hourly sensor file."""
    found = []
    for p in sorted(Path(raw_dir).glob("loc*_sensor*_pm25_hours.csv")):
        m = FILE_RE.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return found


def load_sensor_series(path):
    """Raw OpenAQ hours file -> clean valid-hours Series (index = hour start, UTC). Project cleaning rules."""
    s, _ = clean.clean_frame(clean.openaq_to_frame(pd.read_csv(path)))
    return s


def feature_table(series):
    """Valid-hours Series -> modelling table (regular hourly grid, time features, targets, PM2.5 features).

    Uses the SAME functions as training, so the inputs are built the same way.
    """
    grid = pd.date_range(series.index.min(), series.index.max(), freq="h", name="timestamp_utc")
    df = pd.DataFrame({"pm25": series.reindex(grid)})
    df = build_dataset.add_targets(build_dataset.add_time_features(df))
    return features.add_features(df)


def find_frozen_bundle(models_dir=config.MODELS_DIR):
    """The bundle trained with --train-until 2024-12-01 (newest first), or None."""
    for meta_file in sorted(Path(models_dir).glob("*/metadata.json"), reverse=True):
        if json.loads(meta_file.read_text()).get("train_until_local_date") == FROZEN_UNTIL:
            return meta_file.parent
    return None


def transfer_score(model_bundle, table, winter_year, min_origins=500):
    """Part A on one table: mean MAE of the model and of the best baseline over the winter's origins."""
    idx = table.index[splits.winter_start_year(table.index) == winter_year]
    out = {"n_origins": 0, "mae_model": np.nan, "mae_bar": np.nan, "gain_pct": np.nan}
    if len(idx) == 0:
        return out
    cols = model_bundle.metadata["feature_columns"]
    preds = {name: fn(table).loc[idx] for name, fn in baselines.BASELINES.items()}
    preds["model"] = learners.predict_models(model_bundle.models, table.loc[idx], cols)
    sc = metrics.score(table.loc[idx, metrics.target_columns()], preds)
    out["n_origins"] = int(sc[sc["horizon"] == 1]["n"].iloc[0])
    if out["n_origins"] < min_origins:
        return out
    mae = sc.groupby("model")["mae"].mean()
    out["mae_model"] = float(mae["model"])
    out["mae_bar"] = float(mae[list(baselines.BASELINES)].min())
    out["gain_pct"] = 100 * (1 - out["mae_model"] / out["mae_bar"])
    return out


def agreement(sensor, embassy, winter_year, min_hours=200):
    """Part B: sensor vs Embassy at the same winter hours (both are valid-hour Series, UTC hour start)."""
    both = pd.concat({"s": sensor, "e": embassy}, axis=1, sort=True).dropna()
    both = both[splits.winter_start_year(both.index) == winter_year]
    out = {"overlap_h": len(both), "corr": np.nan, "bias": np.nan, "ratio": np.nan}
    if len(both) < min_hours:
        return out
    out["corr"] = float(both["s"].corr(both["e"]))
    out["bias"] = float((both["s"] - both["e"]).mean())
    out["ratio"] = float(both["s"].mean() / both["e"].mean())
    return out


def run(raw_dir, locations_path, model_dir, embassy_features, embassy_clean, winter_year, min_origins):
    model_bundle = bundle.load_bundle(model_dir)
    loc = pd.read_csv(locations_path).set_index("location_id")
    embassy_series = build_dataset.read_pm25(embassy_clean).dropna()

    emb_table = load_dataset(embassy_features)
    ref = transfer_score(model_bundle, emb_table, winter_year, min_origins)
    print(f"Reference: frozen model on the Embassy, winter {winter_year}/{str(winter_year + 1)[2:]}: "
          f"MAE {ref['mae_model']:.2f} vs bar {ref['mae_bar']:.2f} -> gain {ref['gain_pct']:.1f}% "
          f"({ref['n_origins']} origins)\n")

    rows = []
    for loc_id, path in sensor_files(raw_dir):
        try:
            series = load_sensor_series(path)
        except (KeyError, StopIteration, ValueError) as err:     # a file with no usable table
            print(f"  skip {path.name}: {err}")
            continue
        if series.empty:
            continue
        info = loc.loc[loc_id] if loc_id in loc.index else None
        row = {"location_id": loc_id,
               "name": info["name"] if info is not None else "?",
               "provider": info["provider"] if info is not None else "?",
               "dist_km": (haversine_km(config.EMBASSY_LAT, config.EMBASSY_LON, info["lat"], info["lon"])
                           if info is not None else np.nan)}
        row.update(agreement(series, embassy_series, winter_year))
        row.update(transfer_score(model_bundle, feature_table(series), winter_year, min_origins))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dist_km").reset_index(drop=True), ref


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help=f"bundle folder (default: the one trained with --train-until {FROZEN_UNTIL})")
    p.add_argument("--raw-dir", default=str(config.OPENAQ_RAW))
    p.add_argument("--locations", default=str(config.OPENAQ_RAW / "kathmandu_locations.csv"))
    p.add_argument("--embassy-clean", default=str(config.PROCESSED_DIR / "embassy_pm25_hourly_clean.csv"))
    p.add_argument("--embassy-features", default=str(config.PROCESSED_DIR / "embassy_features_auto.csv"))
    p.add_argument("--winter", type=int, default=2025, help="winter start year (2025 = Dec 2025 - Feb 2026)")
    p.add_argument("--min-origins", type=int, default=500, help="fewer scored hours than this: no score")
    p.add_argument("--out", default=str(config.REPORTS_DIR / "sensor_transfer.csv"))
    p.add_argument("--track", action="store_true", help="also log an MLflow run")
    a = p.parse_args(argv)

    model_dir = Path(a.model) if a.model else find_frozen_bundle()
    if model_dir is None:
        raise SystemExit(f"No frozen model found. Run first:\n  python -m hawa.models.train "
                         f"--train-until {FROZEN_UNTIL} --no-track")
    print(f"frozen model: {model_dir.name}")
    table, ref = run(a.raw_dir, a.locations, model_dir, a.embassy_features, a.embassy_clean, a.winter,
                     a.min_origins)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    show = table.copy()
    show["name"] = show["name"].astype(str).str.slice(0, 24)
    cols = ["location_id", "name", "dist_km", "overlap_h", "corr", "bias", "ratio", "n_origins",
            "mae_model", "mae_bar", "gain_pct"]
    print("Sorted by distance from the Embassy.")
    print("bias = sensor minus Embassy (ug/m3); ratio = mean sensor / mean Embassy; gain_pct = % below the bar\n")
    print(show[cols].round(2).to_string(index=False))
    scored = table.dropna(subset=["gain_pct"])
    beat = int((scored["gain_pct"] > 0).sum())
    print(f"\nSensors scored: {len(scored)} of {len(table)} | model beats the bar on {beat} | "
          f"median gain {scored['gain_pct'].median():.1f}% (Embassy reference {ref['gain_pct']:.1f}%)")
    print(f"Saved -> {out}")

    if a.track:
        import mlflow

        from hawa import tracking
        from hawa.versioning import git_commit
        with tracking.run("hawa-sensor-transfer", f"winter-{a.winter}",
                          {"git_commit": git_commit(), "model": model_dir.name}) as r:
            mlflow.log_params({"winter": a.winter, "min_origins": a.min_origins})
            mlflow.log_metrics({"sensors_scored": len(scored), "sensors_beating_bar": beat,
                                "median_gain_pct": float(scored["gain_pct"].median()) if len(scored) else float("nan"),
                                "embassy_reference_gain_pct": ref["gain_pct"]})
            mlflow.log_artifact(str(out))
            print(f"MLflow run: {r.info.run_id}")


if __name__ == "__main__":
    main()
