# Retraining policy

Status: written down, **manual**. No automatic retraining is built, on purpose (reasons below).

## Today

| | |
|---|---|
| Champion (the model in use) | `models/embassy_latest`, trained on Embassy data up to 2026-06-17 |
| Training data source | Embassy PM2.5. The OpenAQ feed stopped on 2026-06-17 (decision O16) |
| Live labels | only the live sensors, which read about 1.3 to 1.7 times the Embassy depending on the season (O17) |
| Watching it | `GET /metrics` (live error vs two baselines, and uptime), `GET /health` |

## Why we do not retrain automatically now

1. There is no new data of the kind that trained the model. Retraining on the same data gives the same model.
2. Sensor data covers about one winter. A model trained on it could not be tested on a different winter,
   so we could not show that it is better.
3. The scale differs by season (O17), so mixing sensor data into training needs a seasonal correction first.

## When we would consider a new model (any one of these)

- **The Embassy feed returns.** Run `python -m hawa.analysis.data_freshness`. At least 30 days of new valid hours: plan a retrain.
- **A second sensor winter exists** (Dec 2026 to Feb 2027 is complete). We can then test a sensor-aware model on a winter it did not see.
- **The monitoring turns bad.** In `/metrics?days=14`, the model is worse than the best baseline overall, with uptime of at least
  70% and at least 150 compared hours in every horizon group. Then first look for a bug (inputs, sensor failures), not for a new model.

## How a challenger must be judged

- Same rows, same horizons, same baselines as the champion. Never on data it trained on.
- It must beat the **best baseline** AND the **champion**. Promote only if its error is at least 3% lower than the champion's
  on the same data. Otherwise keep the champion. A tie is a "keep".
- **Winter 2026/27 is the next locked test winter.** Nobody tunes, selects or calibrates on it before the final test.
  Decided now, before any of its data exists.
- The current locked winters (2024/25 and 2025/26) stay locked for the Embassy model.

## How to promote (manual)

```bash
dvc repro                                # new models/embassy_latest (the old bundle stays in git history of dvc.lock)
python docker/make_constraints.py        # library versions may have changed
docker compose build && docker compose up -d --wait
```

The new bundle gets a new `model_version` label (`<station>_to_<last training date>`), so old and new forecasts stay
separate in the database and `/metrics` can show each one.

**Rollback:** `git checkout <old commit> -- dvc.lock && dvc checkout`, then rebuild and restart as above.

## Not covered yet

- Monitoring of the INPUTS (drift in the PM2.5 distribution), not only of the error.
- Alerts: someone has to look at `/metrics`.
- A seasonal sensor correction (a month-by-month factor), once there are two sensor winters.
