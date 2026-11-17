# `models/` : forecasting models

| File | Job |
|---|---|
| `baselines.py` | simple baselines every real model must beat |
| `learners.py` | LightGBM, one model per forecast horizon: `fit_models`, `predict_models`, `fit_predict` |
| `train.py` | train the 24 models on the full data and save them as a bundle |
| `bundle.py` | save and load a bundle (models + metadata) |

## `baselines.py`

Baselines learn nothing. Each takes the dataset (regular hourly index, column `pm25`) and returns a table `y_h01 … y_h24`: the prediction made at hour `t` for `t+h`. They use only values at or before hour `t`; a test changes the future and checks the predictions do not move.

| Baseline | Rule |
|---|---|
| `persistence` | PM2.5 at `t+h` = PM2.5 now |
| `same_hour_yesterday` | PM2.5 at `t+h` = the value one day before the target, i.e. at `t+h-24`. For `h=24` it equals persistence |

Registry: `BASELINES = {name: function}`. The dataset must be on a regular hourly grid (the code raises an error otherwise).

### How to read the results

- Persistence is very good at short horizons and gets worse as `h` grows.
- Same-hour-yesterday has about the same error at every horizon, because its error is always "today versus yesterday at that hour".
- A real model should beat the **better baseline at each horizon**, not only on average.

## `learners.py`

**Direct multi-horizon forecasting:** for every `h = 1 ... 24` we train a separate model that predicts PM2.5 at `t+h` from the features known at hour `t`. That is 24 models, all with the same inputs.

Model: LightGBM, gradient-boosted trees (settings in `config.LIGHTGBM_PARAMS`). It handles missing inputs by itself. Loss is absolute error (`objective="l1"`), which matches the MAE score. Ridge and the squared-error loss were removed after they lost (see `docs/decisions.md`, M2, M4, M6).

- Training rows must have a real target for that horizon. Inputs may have gaps.
- Predictions are clipped at 0 (PM2.5 cannot be negative).
- Settings were checked in one small tuning test (6 settings, walk-forward; `docs/decisions.md`, M9). The random seed is `config.RANDOM_SEED`.
- Functions: `make_model()`, `fit_models(df, feature_cols, train_mask, horizons)` -> `{h: model}`, `predict_models(models, X, feature_cols)`, `fit_predict(...)` (both in one go, used by evaluation).
- Tests (`tests/test_learners.py`): the models learn a simple pattern, training uses only the masked rows (rows outside the mask never influence it), predictions are never negative.

### Planned here

Experiments, one at a time, each measured against the baselines: tuning, model saving (MLflow).

## `train.py` and `bundle.py`: saving a model

```
python -m hawa.models.train                            # train on all available origins
python -m hawa.models.train --train-until 2024-12-01   # a "frozen" model: only origins before that date (minus 24 h)
python -m hawa.models.train --no-track                 # do not write an MLflow run
```

Each training also writes an **MLflow run** (experiment `hawa-train`): settings as params, git commit and data hash as tags, and the bundle files. See `src/hawa/tracking.py`.

Writes a **bundle**, a folder `models/<station>_to_<last training origin>/`:

| File | Content |
|---|---|
| `models.joblib` | the 24 fitted LightGBM models, `{horizon: model}` |
| `metadata.json` | feature columns **in order**, LightGBM settings, seed, training data range and size, the data file name and its SHA-256 hash, git commit, library versions |

Why the metadata matters: a forecast job must build exactly the features listed there, in that order. A test (`test_live_features_equal_training_features`) checks that features built from the history up to hour `t` (what a live job has) equal the training features of the same row. `models/` is not in git (large, versioned with MLflow/DVC later).

Tests: `tests/test_train.py` (metadata, training gap, save/load gives identical predictions).
