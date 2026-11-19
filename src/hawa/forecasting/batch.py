"""Batch forecast: for each live sensor, forecast PM2.5 for the next 24 hours and save it.

One run = one forecast "origin": the last COMPLETE hour (hour start = floor(now) - 1 h). For each sensor:
  1. fetch the last 240 h of hourly PM2.5 from OpenAQ,
  2. clean it with the same rules as in training (hawa.processing.clean),
  3. build the model inputs with the same functions as in training (so the features match),
  4. run the 24 models and write 24 rows: the forecast for origin+1 h ... origin+24 h.

Safety rule: if the newest real measurement is older than MAX_DATA_AGE_H hours (sensor offline, or stuck and
removed by the cleaning rules), the sensor is skipped and the reason is printed. No forecast is better than
a forecast from old data.

Values are on the SENSOR's own scale (AirGradient reads about 55% higher than the Embassy in winter 2025/26,
see docs/decisions.md O8). They are not calibrated and not reference-grade.

Run from the project root (needs OPENAQ_API_KEY in .env, and a model from `python -m hawa.models.train`):
    python -m hawa.forecasting.batch
    python -m hawa.forecasting.batch --now 2026-11-16T10:20:00Z      # as if it were that time
    python -m hawa.forecasting.batch --model models/embassy_latest
    python -m hawa.forecasting.batch --to-db                         # also save to PostgreSQL (docker compose up -d db)

Writes: data/forecasts/forecasts.csv (appended; re-running the same hour does not create duplicates)
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hawa import config
from hawa.db import store
from hawa.ingestion import openaq
from hawa.models import bundle, learners
from hawa.processing import build_dataset, clean, features

KEY = ["location_id", "sensor_id", "model_version", "origin_utc", "horizon"]
COLUMNS = ["location_id", "sensor_id", "sensor_name", "model_version", "origin_utc", "target_hour_utc",
           "horizon", "forecast_pm25_sensor_scale", "data_age_h", "created_utc"]


def last_complete_hour(now):
    """Hour start of the last hour that is fully over (UTC)."""
    return pd.Timestamp(now).tz_convert("UTC").floor("h") - pd.Timedelta(hours=1)


def live_features(series, origin):
    """Model inputs for ONE row: the origin hour, built from the valid-hours Series (hour start, UTC).

    The hourly grid runs up to the origin, so hours without data after the last measurement stay empty and
    `pm25_age_h` shows how old the newest real value is. Same functions as in training.
    """
    series = series[series.index <= origin]
    if series.empty:
        return None
    grid = pd.date_range(series.index.min(), origin, freq="h", name="timestamp_utc")
    df = pd.DataFrame({"pm25": series.reindex(grid)})
    df = features.add_features(build_dataset.add_time_features(df))
    return df.iloc[[-1]]


def forecast_sensor(model_bundle, series, origin, max_age_h=config.MAX_DATA_AGE_H):
    """24 forecasts for one sensor. Returns (DataFrame or None, reason). `reason` is "ok" or why it was skipped."""
    row = live_features(series, origin)
    if row is None:
        return None, "no valid data"
    age = row["pm25_age_h"].iloc[0]
    if pd.isna(age) or age > max_age_h:
        old = "unknown" if pd.isna(age) else f"{age:.0f}"
        return None, f"newest real value is {old} h old (limit {max_age_h} h)"
    pred = learners.predict_models(model_bundle.models, row, model_bundle.metadata["feature_columns"])
    values = pred.iloc[0].to_numpy()
    out = pd.DataFrame({"horizon": range(1, len(values) + 1), "forecast_pm25_sensor_scale": values})
    out["origin_utc"] = origin
    out["target_hour_utc"] = origin + pd.to_timedelta(out["horizon"], unit="h")
    out["data_age_h"] = float(age)
    return out, "ok"


def openaq_to_series(raw):
    """Raw OpenAQ table -> clean hourly Series (hour start, UTC), same rules as training."""
    series, _ = clean.clean_frame(clean.openaq_to_frame(raw))
    return series


def openaq_fetch(sensor, now):
    return openaq.fetch_recent_hours(sensor["location_id"], sensor["sensor_id"],
                                     hours_back=config.LIVE_HISTORY_HOURS, now=now.to_pydatetime())


def run(sensors, model_bundle, model_version, now, out_path, fetch=openaq_fetch, max_age_h=config.MAX_DATA_AGE_H,
        engine=None):
    """Forecast every sensor, append to `out_path`. `fetch(sensor, now)` returns the raw OpenAQ table.

    With an `engine` (SQLAlchemy), the same rows are also saved to the database, and the run is logged there.
    """
    now = pd.Timestamp(now).tz_convert("UTC")
    origin = last_complete_hour(now)
    parts, skipped = [], []
    for s in sensors:
        try:
            raw = fetch(s, now)
        except Exception as err:               # one failing sensor must not stop the others
            skipped.append((s["name"], f"fetch failed: {err}"))
            continue
        if raw is None or raw.empty:
            skipped.append((s["name"], "API returned no rows"))
            continue
        series = openaq_to_series(raw)
        fc, reason = forecast_sensor(model_bundle, series, origin, max_age_h)
        if fc is None:
            skipped.append((s["name"], reason))
            continue
        fc.insert(0, "location_id", s["location_id"])
        fc.insert(1, "sensor_id", s["sensor_id"])
        fc.insert(2, "sensor_name", s["name"])
        fc.insert(3, "model_version", model_version)
        parts.append(fc)
    new = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
    new["created_utc"] = now.isoformat()
    new = new[COLUMNS]
    for col in ("origin_utc", "target_hour_utc"):      # one text format, so old and new rows compare equal
        new[col] = pd.to_datetime(new[col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    fresh = new

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and len(new):
        old = pd.read_csv(out_path)
        for col in ("origin_utc", "target_hour_utc"):
            old[col] = pd.to_datetime(old[col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        new = pd.concat([old, new], ignore_index=True)
        new = new.drop_duplicates(KEY, keep="last")
    if len(new) or not out_path.exists():
        new.to_csv(out_path, index=False)
    if engine is not None:
        store.save_forecasts(engine, fresh)
        store.save_run(engine, now, origin, model_version, len(parts), skipped)
    return len(parts), skipped, origin


def version_name(meta, fallback):
    """Model version label: <station>_to_<last training origin date>. The same data gives the same label,
    whatever the folder is called (the DVC pipeline always writes `embassy_latest`)."""
    try:
        return f"{meta['station']}_to_{pd.Timestamp(meta['train_last_origin_utc']):%Y-%m-%d}"
    except (KeyError, ValueError):
        return fallback


def version_warnings(meta):
    """Differences between the library versions that trained the model and the ones installed now."""
    import lightgbm
    import sklearn
    installed = {"lightgbm": lightgbm.__version__, "scikit_learn": sklearn.__version__, "pandas": pd.__version__}
    trained = meta.get("versions", {})
    return [f"{k}: model trained with {trained[k]}, installed {installed[k]}"
            for k in installed if k in trained and trained[k] != installed[k]]


def find_latest_bundle(models_dir=config.MODELS_DIR):
    """Newest bundle trained on all data (not a frozen one), or None."""
    best = None
    for meta_file in Path(models_dir).glob("*/metadata.json"):
        meta = json.loads(meta_file.read_text())
        if meta.get("train_until_local_date") is None:
            if best is None or meta["created_utc"] > best[0]:
                best = (meta["created_utc"], meta_file.parent)
    return best[1] if best else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="bundle folder (default: newest bundle trained on all data)")
    p.add_argument("--now", help="pretend it is this UTC time, e.g. 2026-11-16T10:20:00Z (default: now)")
    p.add_argument("--out", default=str(config.FORECASTS_DIR / "forecasts.csv"))
    p.add_argument("--to-db", action="store_true",
                   help="also save to the database (DATABASE_URL); the CSV is always written")
    a = p.parse_args(argv)

    model_dir = Path(a.model) if a.model else find_latest_bundle()
    if model_dir is None:
        raise SystemExit("No model found. Run first:  python -m hawa.models.train")
    now = pd.Timestamp(a.now) if a.now else pd.Timestamp(datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    model_bundle = bundle.load_bundle(model_dir)
    for warning in version_warnings(model_bundle.metadata):
        print(f"WARNING library version differs ({warning}). Predictions may differ or loading may fail.")

    version = version_name(model_bundle.metadata, model_dir.name)
    engine = None
    if a.to_db:
        engine = store.make_engine()
        store.init_db(engine)
    n_ok, skipped, origin = run(config.LIVE_SENSORS, model_bundle, version, now, a.out, engine=engine)
    print(f"model: {version} | origin (last complete hour): {origin:%Y-%m-%d %H:%M} UTC")
    print(f"forecasts written for {n_ok} of {len(config.LIVE_SENSORS)} sensors -> {a.out}"
          + (" and the database" if engine is not None else ""))
    for name, reason in skipped:
        print(f"  skipped {name}: {reason}")


if __name__ == "__main__":
    main()
