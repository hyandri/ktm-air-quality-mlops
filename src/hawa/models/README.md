# `models/` : forecasting models

| File | Job |
|---|---|
| `baselines.py` | simple baselines every real model must beat |
| `learners.py` | LightGBM, one model per forecast horizon |

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
- **Nothing is tuned yet.** Settings are reasonable defaults, fixed before looking at results. The random seed is `config.RANDOM_SEED`.
- Functions: `make_model()`, `fit_predict(df, feature_cols, train_mask, predict_index, horizons)`.
- Tests (`tests/test_learners.py`): the models learn a simple pattern, training uses only the masked rows (rows outside the mask never influence it), predictions are never negative.

### Planned here

Experiments, one at a time, each measured against the baselines: tuning, model saving (MLflow).
