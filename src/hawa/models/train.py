"""Train the 24 LightGBM models and save them as a bundle (models + metadata).

Run from the project root:
    python -m hawa.models.train                              # train on ALL available origins
    python -m hawa.models.train --train-until 2024-12-01     # only origins before that date (minus 24 h)

`--train-until` makes a "frozen" model, like the one in the final test (before 2024-12-01), which is
what the sensor transfer test needs (the model must not have seen the period it is tested on).

Needs:  data/processed/<station>_features_<weather>.csv   (hawa.processing.features)
Writes: models/<station>_to_<last training origin>/{models.joblib, metadata.json}
"""
import argparse
import hashlib
import platform
import subprocess
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


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                             cwd=config.PROJECT_ROOT)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
        "git_commit": _git_commit(),
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
    a = p.parse_args(argv)

    path = Path(a.dataset or config.PROCESSED_DIR / f"{a.station}_features_{a.weather}.csv")
    df = load_dataset(path)
    print(f"{a.station} | training 24 models | train-until: {a.train_until or 'all data'}")
    models, meta = train(df, a.train_until)
    meta.update({"station": a.station, "dataset_file": path.name, "dataset_sha256": _sha256(path)})

    last = pd.Timestamp(meta["train_last_origin_utc"]).strftime("%Y-%m-%d")
    out = Path(a.out_dir) / f"{a.station}_to_{last}"
    bundle.save_bundle(out, models, meta)
    print(f"train origins: {meta['n_train_origins']} ({meta['train_first_origin_utc'][:10]} .. {last})")
    print(f"features: {len(meta['feature_columns'])} | git commit: {meta['git_commit'][:8]}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
