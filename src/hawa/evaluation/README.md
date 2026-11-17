# `evaluation/` : splits, metrics, scoring

How we decide whether a forecast is good. Rules here protect us from fooling ourselves.

| File | Job |
|---|---|
| `splits.py` | time-based train / validation / test masks |
| `metrics.py` | per-horizon MAE and RMSE, fair comparison between models |
| `evaluate_baselines.py` | score the baselines on a split, save a report |

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

## Adding a model to the comparison

Make a function that returns `y_h01 … y_h24` predictions on the same index, add it to the `predictions` dict, and call `metrics.score`. Use the same split and scope so numbers are comparable.
