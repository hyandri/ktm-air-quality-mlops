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
| E4 | Compare models with walk-forward over 4 winters (2019/20, 2020/21, 2021/22, 2023/24), not one | winters differ a lot (copy-yesterday error 13.5 – 26) |

## Modelling experiments

Single-winter runs (M1–M5, validation 2023/24) first, then walk-forward (M6, M7).

| # | Question | Result | Decision |
|---|---|---|---|
| M1 | Train on winters only, or all seasons? | LightGBM 18.4 vs 17.6; Ridge 18.7 vs 17.4 (all seasons better) | **train on all seasons.** Winter-only option removed |
| M2 | Does the loss matter? | LightGBM squared error 17.6 → absolute error (L1) 15.9 | **L1 is the loss** (it matches the MAE score). `objective="l1"` is now fixed in `config.LIGHTGBM_PARAMS` |
| M3 | Does weather (value at the origin hour) help? | Single winter: 15.94 with and without. Walk-forward: mean MAE difference per winter +0.21, −0.08, −0.27, −0.02 (weather better in 10, 13, 23, 14 of 24 horizons) | **no measurable gain.** The switch stays as an option because future-weather features (M8) reuse it; remove if M8 also fails |
| M4 | Ridge vs LightGBM | Single winter: Ridge 17.4, LightGBM L1 15.9. Walk-forward: Ridge beat the bar in 1 of 4 winters (mean −1.8%) | **Ridge removed** (code, `RIDGE_ALPHA`, `LINEAR_EXCLUDE`, its tests) |
| M5 | Best model vs the bar (16.2) on 2023/24 | LightGBM L1 +0.5% | one winter is too fragile: use walk-forward (E4) |
| M6 | LightGBM L1 vs the bar, walk-forward | Beats the bar in 4 of 4 winters. Mean gain +12.5% (no weather), +12.7% (weather). All 24 horizons won in 2019/20, 2020/21, 2021/22; only 15 (no weather) / 13 (weather) of 24 in 2023/24 | **LightGBM L1 is the current model** |
| M7 | Where is it strong or weak? | Strong at h1–6 in every winter (2021/22: 13.4 vs 18.2). In 2023/24 it loses to the bar at h=7 and h=14–21. The baseline was also much better that winter (h1–6: 15.3 vs 18–25 in the other winters) | long horizons are the weak spot. Earlier idea "calm winters are easier" was **not supported**: 2021/22 is equally calm and gained ~19% |

## Experiment in progress

**M8: future weather.** For horizon h the model also gets the weather at hour t+h (`fut_*` columns,
`learners.future_weather`). Variant `lightgbm_futureweather` (no current-hour weather) is compared with
`lightgbm_noweather` in walk-forward.
- This is an **optimistic upper bound**: the archived weather is close to what really happened, but a real
  forecast made at hour t has errors. Hourly "Historical Forecast" values are not true 1–24 h-ahead forecasts.
- If there is **no gain even here**, drop weather as a direction (and remove the switches at the next gate).
- If there is a gain, the next step is to test with true lead-time forecasts (Open-Meteo Previous Runs API,
  only from Feb 2024), which is a fair but short test.
- Result: _to be filled after the run._

## Open questions

- M8: future weather as features: built, **result pending** (see below).
- Why is 2023/24 harder? Is it the latest winter, so closer to the locked test winters?
- Does the ERA5 / forecast-weather mix in training hurt? (Weather is archived forecasts from 2021, ERA5 before.)
- Tuning of LightGBM settings: not done.
- Sin/cos calendar features were made for Ridge. Trees do not need them: check if they can go.

## Cleanup gate (before each new stage)

Remove code that lost its experiment, and record it above.

**Done (stage 1 → 2):**
- Ridge learner, its settings and tests (M4)
- `--train-scope winter` in `evaluate_models.py` (M1)
- `--objective` switch and the squared-error LightGBM default (M2)
- `--models` option (only one model left)
- walk-forward variants renamed: `lightgbm_noweather`, `lightgbm_weather`
- old result files in `reports/` (they carry removed model names) and `data/raw/openaq/scorecard.csv` (old, wrong)

**Kept on purpose:**
- the weather switch (needed for M8)
- the `previous_runs` URL in `ingestion/openmeteo.py`: it is used by `analysis/openmeteo_probe.py`

**Next gate candidates:** weather switch if M8 fails; sin/cos calendar features if trees do not need them.
