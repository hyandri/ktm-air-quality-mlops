"""FINAL test: score the chosen model once on the locked test winters 2024/25 and 2025/26.

Run from the project root (needs --final, on purpose):
    python -m hawa.evaluation.final_test --final

RULES
  * Run it ONCE, after all choices are made (features, loss, settings are fixed in config.py).
  * Do NOT change the model after seeing these numbers. Write them into docs/decisions.md as they are,
    good or bad, and commit them before any new experiment.

What is scored (same baselines as before, same 24 horizons, same rows for every model):
    frozen      trained on everything before winter 2024 (minus 24 h), scored on 2024/25 AND 2025/26.
                Like a model deployed in Dec 2024 and never retrained.
    retrained   trained on everything before winter 2025 (minus 24 h), scored on 2025/26 only.
                Like the same system retrained once a year.
    "frozen vs retrained" on 2025/26 shows what retraining is worth.

Needs:  data/processed/<station>_features_<weather>.csv
Writes: reports/final_test_<station>.csv  (fold, model, horizon, n, mae, rmse)
"""
import argparse
import sys

import pandas as pd

from hawa import config
from hawa.evaluation import metrics, splits
from hawa.evaluation.evaluate_baselines import load_dataset
from hawa.evaluation.walk_forward import print_summary
from hawa.models import baselines, learners
from hawa.processing.features import feature_columns

# (model name, scored test winter, training ends before winter)
RUNS = (
    ("frozen", 2024, 2024),
    ("frozen", 2025, 2024),
    ("retrained", 2025, 2025),
)


def label(winter):
    return f"{winter}/{str(winter + 1)[2:]}"


def score_run(df, cols, name, eval_winter, train_until):
    train_mask, eval_mask = splits.test_masks(df.index, eval_winter, train_until)
    eval_index = df.index[eval_mask]
    print(f"  {name} on {label(eval_winter)}: train origins {int(train_mask.sum())}, "
          f"scored origins {len(eval_index)}")
    return learners.fit_predict(df, cols, train_mask, eval_index), eval_index


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--station", default="embassy")
    p.add_argument("--weather", default="auto")
    p.add_argument("--dataset", help="explicit features file")
    p.add_argument("--final", action="store_true", help="required: confirms this is the one final run")
    a = p.parse_args(argv)
    if not a.final:
        sys.exit("This scores the locked TEST winters. Run it once, when all choices are final. "
                 "Add --final if you mean it.")

    df = load_dataset(a.dataset or config.PROCESSED_DIR / f"{a.station}_features_{a.weather}.csv")
    cols = feature_columns(df)
    print(f"{a.station} | FINAL test on winters {config.TEST_WINTERS} | {len(cols)} input columns")

    parts = []
    for winter in config.TEST_WINTERS:
        runs = [r for r in RUNS if r[1] == winter]
        preds, eval_index = {}, None
        for name, ew, tu in runs:
            preds[name], eval_index = score_run(df, cols, name, ew, tu)
        for name, fn in baselines.BASELINES.items():
            preds[name] = fn(df).loc[eval_index]
        sc = metrics.score(df.loc[eval_index, metrics.target_columns()], preds)
        sc.insert(0, "fold", label(winter))
        parts.append(sc)
    all_scores = pd.concat(parts, ignore_index=True)

    out = config.REPORTS_DIR / f"final_test_{a.station}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    all_scores.to_csv(out, index=False)

    print_summary(all_scores, ["frozen", "retrained"])
    base = list(baselines.BASELINES)
    for fold, g in all_scores.groupby("fold"):
        print(f"\n% lower MAE than the best baseline at chosen horizons, winter {fold}:\n")
        print(metrics.improvement_vs_best_baseline(g, base, horizons=(1, 3, 6, 12, 18, 24)).to_string())
    print(f"\nSaved -> {out}")
    print("Now write these numbers into docs/decisions.md (E5) and commit BEFORE any new experiment.")


if __name__ == "__main__":
    main()
