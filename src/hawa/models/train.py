"""Train the 24 LightGBM models and save them as a bundle (models + metadata).

Run from the project root:
    python -m hawa.models.train                              # train on ALL available origins
    python -m hawa.models.train --train-until 2024-12-01     # only origins before that date (minus 24 h)
    python -m hawa.models.train --no-track                   # do not write an MLflow run
    python -m hawa.models.train --name embassy_latest        # fixed folder name (used by the DVC pipeline)

`--train-until` makes a "frozen" model, like the one in the final test (before 2024-12-01), which is
what the sensor transfer test needs (the model must not have seen the period it is tested on).

Needs:  data/processed/<station>_features_<weather>.csv   (hawa.processing.features)
Writes: models/<station>_to_<last training origin>/{models.joblib, metadata.json}
        and an MLflow run in experiment "hawa-train" (see hawa.tracking)
"""
import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn

from hawa import config
from hawa.evaluation import splits
from hawa.evaluation.evaluate_baselines import load_dataset
from hawa.models import bundle, learners
from hawa.processing.features import feature_columns
from hawa.versioning import git_commit, sha256_file


def train(df, train_until=None, horizons=config.HORIZONS):
    """Train on `df`. Returns (models, metadata). `train_until`: 'YYYY-MM-DD' local date or None (all)."""
    cols = feature_columns(df)
    if train_until:
        mask = splits.origins_before(df.index, train_until)
    else:
        mask = np.ones(len(df), dtype=bool)
    # origins that can be used: at least the first target has a real value
    usable = mask & df["y_h01"].notna().to_numpy()
    if not usable.any():
        raise ValueError("no training rows")
    models = learners.fit_models(df, cols, mask, horizons)
    used = df.index[usable]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "horizons": horizons,
        "feature_columns": cols,
        "lightgbm_params": config.LIGHTGBM_PARAMS,
        "random_seed": config.RANDOM_SEED,
        "train_until_local_date": train_until,
        "train_gap_hours": config.TRAIN_GAP_H,
        "n_train_origins": int(usable.sum()),
        "train_first_origin_utc": used.min().isoformat(),
        "train_last_origin_utc": used.max().isoformat(),
        "versions": {"python": platform.python_version(), "lightgbm": lightgbm.__version__,
                     "scikit_learn": sklearn.__version__, "pandas": pd.__version__},
    }
    return models, metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--station", default="embassy")
    p.add_argument("--weather", default="auto")
    p.add_argument("--dataset", help="explicit features file")
    p.add_argument("--train-until", help="local date YYYY-MM-DD: train only on origins before it (minus 24 h)")
    p.add_argument("--out-dir", default=str(config.MODELS_DIR))
    p.add_argument("--name", help="bundle folder name (default: <station>_to_<last training origin date>). "
                                  "DVC uses a fixed name so the output path never changes.")
    p.add_argument("--no-track", action="store_true", help="do not log an MLflow run")
    a = p.parse_args(argv)

    path = Path(a.dataset or config.PROCESSED_DIR / f"{a.station}_features_{a.weather}.csv")
    df = load_dataset(path)
    print(f"{a.station} | training 24 models | train-until: {a.train_until or 'all data'}")
    models, meta = train(df, a.train_until)
    meta.update({"station": a.station, "dataset_file": path.name, "dataset_sha256": sha256_file(path)})

    last = pd.Timestamp(meta["train_last_origin_utc"]).strftime("%Y-%m-%d")
    out = Path(a.out_dir) / (a.name or f"{a.station}_to_{last}")
    bundle.save_bundle(out, models, meta)
    print(f"train origins: {meta['n_train_origins']} ({meta['train_first_origin_utc'][:10]} .. {last})")
    print(f"features: {len(meta['feature_columns'])} | git commit: {meta['git_commit'][:8]}")
    print(f"Saved -> {out}")

    if not a.no_track:
        log_to_mlflow(out, meta)


def log_to_mlflow(bundle_path, meta):
    """Log one run: settings as params, data and code identity as tags, the bundle folder as files."""
    import mlflow

    from hawa import tracking  # imported here so `--no-track` and the tests do not need MLflow

    tags = {"station": meta["station"], "git_commit": meta["git_commit"],
            "dataset_sha256": meta["dataset_sha256"], "dataset_file": meta["dataset_file"]}
    with tracking.run("hawa-train", Path(bundle_path).name, tags) as r:
        mlflow.log_params({**meta["lightgbm_params"], "random_seed": meta["random_seed"],
                           "train_until_local_date": meta["train_until_local_date"] or "all",
                           "n_features": len(meta["feature_columns"]), "horizons": meta["horizons"]})
        mlflow.log_metrics({"n_train_origins": meta["n_train_origins"]})
        mlflow.log_artifacts(str(bundle_path), artifact_path="bundle")
        print(f"MLflow run: {r.info.run_id}")


if __name__ == "__main__":
    main()
