# `evaluation/` : splits, metrics, scoring

How we decide whether a forecast is good. Rules here protect us from fooling ourselves.

| File | Job |
|---|---|
| `splits.py` | time-based train / validation / test masks |
| `metrics.py` | per-horizon MAE and RMSE, fair comparison between models |
| `evaluate_baselines.py` | score the baselines on a split, save a report |
| `evaluate_models.py` | train LightGBM, score them with the baselines on validation |
| `walk_forward.py` | score several winters in turn (expanding training window) |

## Splits (`splits.py`)

Splits are **chronological, never random**. A row is a forecast origin (hour `t`); its targets look up to 24 h ahead, so we cut by origin time. Dates are Kathmandu local time. A winter is named by its start year: `2023` = Dec 2023 – Feb 2024.

Settings live in `config.py`:

| Split | Scope `winter` (current focus) | Scope `all` (planned, later) |
|---|---|---|
| train | winters 2017, 2018, 2019, 2020, 2021 | all seasons before 2022-12-01 |
| validation | winter 2023 (2023/24) | 2023-12-01 → 2024-11-30 |
| test | winters 2024 and 2025 | from 2024-12-01 |

- **Winter 2022/23 is skipped:** the Embassy has almost no data (2% coverage).
- Validation is used to choose models and settings. **Test is used once, at the end.**
- `evaluate_baselines.py` refuses to score the test split unless you pass `--final`.
- Tests check: splits never overlap, the skipped winter is in no split, and the last training target stays before the first validation origin.

`split_mask(index, split, scope)` returns a boolean array for a UTC hourly index.

## Metrics (`metrics.py`)

- **MAE** (mean absolute error) and **RMSE**, in µg/m³, **for each horizon** 1 … 24 h.
- A row is skipped if the target, or any model's prediction, is missing. **All models are scored on the same rows** for a given horizon, so the comparison is fair. `n` says how many rows were used.
- `score(targets, predictions)` returns a long table (`model, horizon, n, mae, rmse`); `summarize(scores)` makes a short table.
- We never fill missing targets (see `data/README.md`).

## Scoring baselines (`evaluate_baselines.py`)

```
python -m hawa.evaluation.evaluate_baselines                        # validation, winter
python -m hawa.evaluation.evaluate_baselines --scope all            # validation, whole year
python -m hawa.evaluation.evaluate_baselines --split test --final   # final results only
```

Reads `data/processed/<station>_dataset_<weather>.csv`. Writes `reports/baselines_<station>_<split>_<scope>.csv`.
Baselines are computed on the full hourly grid (they look backward), then cut to the split.
Options: `--station`, `--weather`, `--dataset`.

## Scoring models (`evaluate_models.py`)

```
python -m hawa.evaluation.evaluate_models
```

- **Training data:** every origin before 2022-12-01 (all seasons). Winter-only training was removed because it was worse (`docs/decisions.md`, M1).
- **Scoring:** validation winter 2023/24, same rows for every model (baselines included).
- **The test winters are never used here.** Only validation, until we decide the final model.
- Prints the MAE table and the "% lower MAE than the best baseline at that horizon" table (`metrics.improvement_vs_best_baseline`). Positive = better than the bar.
- Writes `reports/models_<station>_validation.csv`.
- Needs `data/processed/<station>_features_<weather>.csv` (from `hawa.processing.features`).

## Walk-forward (`walk_forward.py`)

One validation winter is a fragile judge: winters differ a lot (mean PM2.5 from 58 to 87, and the "copy yesterday" error from 13.5 to 26). So we score several winters, each time training only on earlier data, like a live system:

| Fold (scored winter) | Trained on |
|---|---|
| 2019/20 | everything before Dec 2019 (minus 24 h) |
| 2020/21 | everything before Dec 2020 |
| 2021/22 | everything before Dec 2021 |
| 2023/24 | everything before Dec 2023 |

```
python -m hawa.evaluation.walk_forward
```

- Folds are in `config.WALK_FORWARD_WINTERS`. **2017 and 2018 are skipped** (too little earlier data), 2022 has no data, and **the test winters are refused** by `splits.walk_forward_masks`.
- Prints mean MAE per winter, the % gain over the best baseline, and "folds won" (how many winters the model beats the bar in).
- A model should win in most folds, not just on average. Writes `reports/walkforward_<station>.csv`.
- Takes about 2 minutes.
- `--track` also writes an MLflow run (experiment `hawa-walk-forward`): model settings, per winter the MAE, the bar and the % gain, the mean gain, folds won, and the per-horizon CSV.

## Final test (`final_test.py`)

```
python -m hawa.evaluation.final_test --final
```

Scores the locked test winters 2024/25 and 2025/26. It refuses to run without `--final`. **Run it once**, after every choice is made, and do not change the model afterwards.

| Model | Trained on | Scored on |
|---|---|---|
| `frozen` | everything before winter 2024 (minus 24 h) | 2024/25 and 2025/26 (a model deployed in Dec 2024 and never retrained) |
| `retrained` | everything before winter 2025 (minus 24 h) | 2025/26 (the same system retrained once a year) |

Same baselines, same rows for every model. Prints mean MAE per winter, the % gain over the best baseline, and the gain at h1, 3, 6, 12, 18, 24. Writes `reports/final_test_<station>.csv`. `splits.test_masks` allows only test winters and checks that training never reaches into the scored winter.
