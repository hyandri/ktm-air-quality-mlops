"""Train Ridge and LightGBM, score them against the baselines on the VALIDATION winter.

Run from the project root:
    python -m hawa.evaluation.evaluate_models --train-scope winter   # train on winters only
    python -m hawa.evaluation.evaluate_models --train-scope all      # train on all seasons

Training data (always before the validation winter, see evaluation/README.md):
    winter  origins in the training winters 2017/18 ... 2021/22
    all     every origin before 2022-12-01 (all seasons)
Scoring is always on validation winter 2023/24, on the same rows for every model.
The TEST winters are not used here. Settings are not tuned yet.

Needs:  data/processed/<station>_features_<weather>.csv  (hawa.processing.features)
Writes: reports/models_<station>_validation_train-<scope>.csv
"""
import argparse

from hawa import config
from hawa.evaluation import metrics, splits
from hawa.evaluation.evaluate_baselines import load_dataset
from hawa.models import baselines, learners
from hawa.processing.features import feature_columns


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--station", default="embassy")
    p.add_argument("--weather", default="auto")
    p.add_argument("--dataset", help="explicit features file")
    p.add_argument("--train-scope", choices=splits.SCOPES, default="winter")
    p.add_argument("--models", default=",".join(learners.MODEL_NAMES))
    a = p.parse_args(argv)

    df = load_dataset(a.dataset or config.PROCESSED_DIR / f"{a.station}_features_{a.weather}.csv")
    cols = feature_columns(df)
    train_mask = splits.split_mask(df.index, "train", a.train_scope)
    val_index = df.index[splits.split_mask(df.index, "validation", "winter")]
    print(f"{a.station} | train scope={a.train_scope}: {int(train_mask.sum())} origins | "
          f"validation winter: {len(val_index)} origins | {len(cols)} input columns")

    preds = {name: fn(df).loc[val_index] for name, fn in baselines.BASELINES.items()}
    for name in a.models.split(","):
        print(f"training {name} (24 horizons) ...")
        preds[f"{name}_train-{a.train_scope}"] = learners.fit_predict(name, df, cols, train_mask, val_index)

    scores = metrics.score(df.loc[val_index, metrics.target_columns()], preds)
    out = config.REPORTS_DIR / f"models_{a.station}_validation_train-{a.train_scope}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out, index=False)

    print("\nMAE (ug/m3), same rows for every model:\n")
    print(metrics.summarize(scores).to_string())
    print("\n% lower MAE than the best baseline at that horizon (positive = better):\n")
    print(metrics.improvement_vs_best_baseline(scores, list(baselines.BASELINES)).to_string())
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
