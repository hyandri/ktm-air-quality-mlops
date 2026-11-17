"""Small LightGBM tuning test, scored with walk-forward (the same 4 winters as walk_forward.py).

Run from the project root:
    python -m hawa.evaluation.tune                 # all settings in config.TUNING_GRID (about 12 min)
    python -m hawa.evaluation.tune --only base,small_trees

RULES, written BEFORE seeing results (so we do not pick the lucky one):
  1. A setting is only a candidate if it wins in all 4 winters.
  2. It must beat "base" by at least 0.3 points of mean gain. Smaller differences are noise
     (we saw differences of that size between identical-quality variants before).
  3. If several pass, take the simplest (fewer / smaller trees).
  4. If none pass, keep the base settings. That is a fine result.
The TEST winters are never used here.

Needs:  data/processed/<station>_features_<weather>.csv
Writes: reports/tuning_<station>.csv  (same format as walkforward_<station>.csv)
"""
import argparse

import pandas as pd

from hawa import config
from hawa.evaluation.evaluate_baselines import load_dataset
from hawa.evaluation.walk_forward import print_summary, run_fold
from hawa.processing.features import feature_columns


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--station", default="embassy")
    p.add_argument("--weather", default="auto")
    p.add_argument("--dataset", help="explicit features file")
    p.add_argument("--only", help="comma-separated names from config.TUNING_GRID")
    a = p.parse_args(argv)

    names = a.only.split(",") if a.only else list(config.TUNING_GRID)
    models = {n: config.TUNING_GRID[n] for n in names}
    df = load_dataset(a.dataset or config.PROCESSED_DIR / f"{a.station}_features_{a.weather}.csv")
    cols = feature_columns(df)
    print(f"{a.station} | tuning over winters {config.WALK_FORWARD_WINTERS} | settings: {names}")
    all_scores = pd.concat([run_fold(df, y, cols, models) for y in config.WALK_FORWARD_WINTERS],
                           ignore_index=True)

    out = config.REPORTS_DIR / f"tuning_{a.station}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    all_scores.to_csv(out, index=False)
    print_summary(all_scores, names)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
