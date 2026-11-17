"""Score the baselines on a split and save the per-horizon table.

Run from the project root:
    python -m hawa.evaluation.evaluate_baselines                       # validation, winter
    python -m hawa.evaluation.evaluate_baselines --scope all           # validation, whole year
    python -m hawa.evaluation.evaluate_baselines --split test --final  # test: final results only

The TEST split is protected: it is scored only with --final, so it is not used by accident
while we are still choosing models.

Needs: data/processed/<station>_dataset_<weather>.csv   (hawa.processing.build_dataset)
Writes: reports/baselines_<station>_<split>_<scope>.csv  (long table: model, horizon, n, mae, rmse)
"""
import argparse
import sys

import pandas as pd

from hawa import config
from hawa.evaluation import metrics, splits
from hawa.models import baselines


def load_dataset(path):
    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.set_index("timestamp_utc").sort_index()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--station", default="embassy")
    p.add_argument("--weather", default="auto", help="which dataset file (baselines do not use weather)")
    p.add_argument("--dataset", help="explicit dataset path")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--scope", choices=splits.SCOPES, default="winter")
    p.add_argument("--final", action="store_true", help="required to score the test split")
    a = p.parse_args(argv)

    if a.split == "test" and not a.final:
        sys.exit("The test split is for FINAL results only. Add --final if you really mean it.")

    path = a.dataset or config.PROCESSED_DIR / f"{a.station}_dataset_{a.weather}.csv"
    df = load_dataset(path)
    part = df[splits.split_mask(df.index, a.split, a.scope)]
    print(f"{a.station} | split={a.split} scope={a.scope} | origins in split: {len(part)}")

    # baselines are computed on the FULL grid first (they look back in time), then cut to the split
    preds = {name: fn(df).loc[part.index] for name, fn in baselines.BASELINES.items()}
    scores = metrics.score(part[metrics.target_columns()], preds)

    out = config.REPORTS_DIR / f"baselines_{a.station}_{a.split}_{a.scope}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out, index=False)

    print("\nMAE by horizon (ug/m3), same rows for both models:\n")
    print(metrics.summarize(scores).to_string())
    print(f"\nSaved full per-horizon table -> {out}")


if __name__ == "__main__":
    main()
