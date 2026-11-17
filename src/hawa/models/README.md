# `models/` : forecasting models

| File | Job |
|---|---|
| `baselines.py` | simple baselines every real model must beat |

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

### Planned here

Ridge regression, then LightGBM (Stage 1), using lag/rolling features. Baseline numbers are written to `reports/`.
