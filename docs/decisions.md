# Decisions log

What we decided, why, and the evidence. Keep entries short. When an experiment switch or script is
removed from the code, its result stays here, so no knowledge is lost.
Numbers are mean MAE over 24 horizons (µg/m³), Embassy, validation winter 2023/24, unless noted.

## Data

| # | Decision | Why |
|---|---|---|
| D1 | Embassy is the main station; Phora only for early years | Embassy winter coverage 78–95% (except 2022/23: 2%); Phora drifts after 2022 and had a 448 h stuck block |
| D2 | Train on history, use AirGradient sensors for live forecasts later | AirGradient sensors only started in late 2025 |
| D3 | Cleaning rules: drop empty, -999, negative, 985, flat runs ≥ 12 h; one value per hour | see `src/hawa/processing/README.md` |
| D4 | Flagged rows are kept | 709 flagged values compared with neighbours: median difference −0.5 µg/m³ |
| D5 | Never fill the target. Features may carry the last real value forward for ≤ 3 h; no interpolation | interpolation uses the future; 4,161 hours rescued |
| D6 | Weather: archived forecasts from 2021, ERA5 before. Boundary layer height not used | BLH missing in 62% of archived forecasts and one 182-day block of ERA5 |

## Evaluation

| # | Decision | Why |
|---|---|---|
| E1 | Chronological splits by winter. Test winters 2024/25 and 2025/26 are locked until the final model | avoids leakage and peeking |
| E2 | Winter 2022/23 is skipped | Embassy has 1 usable row there |
| E3 | The bar to beat is the better of persistence and same-hour-yesterday, per horizon | `evaluation/README.md` |
| E4 | Compare models with walk-forward over 4 winters, not one | winters differ a lot (copy-yesterday error 13.5 – 26) |

## Modelling experiments (validation winter 2023/24)

| # | Question | Result | Decision |
|---|---|---|---|
| M1 | Train on winters only, or all seasons? | LightGBM 18.4 vs 17.6; Ridge 18.7 vs 17.4 (all seasons better) | train on all seasons |
| M2 | Does the loss matter? | LightGBM squared error 17.6 → absolute error (L1) 15.9 | use L1 (it matches the MAE score) |
| M3 | Does weather help? | L1 LightGBM with weather 15.94, without 15.94 (same mean) | open: re-test in walk-forward |
| M4 | Ridge vs LightGBM | Ridge 17.4 (squared error), LightGBM L1 15.9 | LightGBM L1 leads; Ridge is a reference |
| M5 | Best model vs the bar (16.2) | LightGBM L1 +0.5% on this winter | one winter is too fragile: use walk-forward (E4) |

## Open questions

- Does weather help in walk-forward? (M3) Does the ERA5 / forecast mix in training hurt it?
- Future weather (forecast for the next 24 h) as features: not tried yet.
- Tuning of LightGBM settings: not done.

## Cleanup gate (before each new stage)

Remove code that lost its experiment, and record it above. Candidates now:
- `--train-scope winter` results and the `winter` option in `evaluate_models.py` if M1 stays as is
- the weather switch, if walk-forward shows no gain (M3)
- `reports/*.csv` from superseded runs; old `data/raw/openaq/scorecard.csv`
- the unused `previous_runs` URL in `ingestion/openmeteo.py`
