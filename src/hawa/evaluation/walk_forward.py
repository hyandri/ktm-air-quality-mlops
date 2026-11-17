"""Walk-forward comparison: score several winters in turn, not just one.

One validation winter is a fragile judge (winters differ a lot). Here every fold scores one winter,
training on all earlier data (expanding window), exactly like a live system would:
    fold 2019 -> train before Dec 2019, score winter 2019/20
    fold 2020 -> train before Dec 2020, score winter 2020/21
    fold 2021 -> train before Dec 2021, score winter 2021/22
    fold 2023 -> train before Dec 2023, score winter 2023/24
The TEST winters (2024/25, 2025/26) are never used. Folds are listed in config.WALK_FORWARD_WINTERS.

Run from the project root:
    python -m hawa.evaluation.walk_forward
    python -m hawa.evaluation.walk_forward --track    # also write an MLflow run (experiment "hawa-walk-forward")

Needs:  data/processed/<station>_features_<weather>.csv
Writes: reports/walkforward_<station>.csv  (fold, model, horizon, n, mae, rmse)
"""
import argparse

import pandas as pd

from hawa import config
from hawa.evaluation import metrics, splits
from hawa.evaluation.evaluate_baselines import load_dataset
from hawa.models import baselines, learners
from hawa.processing.features import feature_columns
from hawa.versioning import git_commit


def run_fold(df, winter_year, cols):
    """Score baselines and LightGBM on one winter. Returns the long score table."""
    train_mask, eval_mask = splits.walk_forward_masks(df.index, winter_year)
    eval_index = df.index[eval_mask]
    preds = {name: fn(df).loc[eval_index] for name, fn in baselines.BASELINES.items()}
    preds["lightgbm"] = learners.fit_predict(df, cols, train_mask, eval_index)
    sc = metrics.score(df.loc[eval_index, metrics.target_columns()], preds)
    sc.insert(0, "fold", f"{winter_year}/{str(winter_year + 1)[2:]}")
    print(f"  fold {winter_year}/{str(winter_year + 1)[2:]}: train origins {int(train_mask.sum())}, "
          f"scored origins {len(eval_index)}")
    return sc


def fold_table(scores, baseline_names):
    """Rows = folds, columns = models: mean MAE over 24 horizons. Plus the best-baseline column."""
    t = scores.groupby(["fold", "model"])["mae"].mean().unstack("model")
    t["best_baseline"] = t[list(baseline_names)].min(axis=1)
    return t


def print_summary(all_scores, model_names):
    """Print mean MAE per winter and the % gain over the best baseline for the given models."""
    t = fold_table(all_scores, list(baselines.BASELINES))
    print("\nMean MAE over 24 horizons, per scored winter (ug/m3):\n")
    print(t.round(2).to_string())
    gain = (100 * (1 - t[model_names].div(t["best_baseline"], axis=0))).round(1)
    gain.loc["mean of folds"] = gain.mean().round(1)
    gain.loc["folds won"] = (gain.iloc[:-1] > 0).sum().astype(int)
    print("\n% lower MAE than the best baseline (positive = better):\n")
    print(gain.to_string())


def summary_metrics(all_scores, model="lightgbm"):
    """Numbers worth tracking: per winter the model MAE, the bar and the % gain; plus mean gain and folds won."""
    t = fold_table(all_scores, list(baselines.BASELINES))
    gain = 100 * (1 - t[model] / t["best_baseline"])
    out = {}
    for fold in t.index:
        key = str(fold).replace("/", "_")
        out[f"mae_{key}"] = float(t.loc[fold, model])
        out[f"bar_{key}"] = float(t.loc[fold, "best_baseline"])
        out[f"gain_pct_{key}"] = float(gain[fold])
    out["mean_gain_pct"] = float(gain.mean())
    out["folds_won"] = int((gain > 0).sum())
    return out


def log_to_mlflow(all_scores, csv_path, station):
    """One MLflow run: model settings, winters, the metrics above and the per-horizon table."""
    import mlflow

    from hawa import tracking
    with tracking.run("hawa-walk-forward", f"{station}-walk-forward", {"station": station,
                                                                         "git_commit": git_commit()}) as r:
        mlflow.log_params({**config.LIGHTGBM_PARAMS, "winters": str(config.WALK_FORWARD_WINTERS),
                           "random_seed": config.RANDOM_SEED})
        mlflow.log_metrics(summary_metrics(all_scores))
        mlflow.log_artifact(str(csv_path))
        print(f"MLflow run: {r.info.run_id}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--station", default="embassy")
    p.add_argument("--weather", default="auto")
    p.add_argument("--dataset", help="explicit features file")
    p.add_argument("--track", action="store_true", help="also log an MLflow run")
    a = p.parse_args(argv)

    df = load_dataset(a.dataset or config.PROCESSED_DIR / f"{a.station}_features_{a.weather}.csv")
    cols = feature_columns(df)
    print(f"{a.station} | walk-forward over winters {config.WALK_FORWARD_WINTERS}")
    all_scores = pd.concat([run_fold(df, y, cols) for y in config.WALK_FORWARD_WINTERS],
                           ignore_index=True)

    out = config.REPORTS_DIR / f"walkforward_{a.station}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    all_scores.to_csv(out, index=False)

    print_summary(all_scores, ["lightgbm"])
    print(f"\nSaved -> {out}")
    if a.track:
        log_to_mlflow(all_scores, out, a.station)


if __name__ == "__main__":
    main()
