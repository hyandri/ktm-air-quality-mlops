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
| E5 | Final test protocol: run `final_test.py --final` once, with settings frozen in `config.py`. Two models: `frozen` (trained before winter 2024, scored on 2024/25 and 2025/26) and `retrained` (trained before winter 2025, scored on 2025/26). No model changes after seeing the numbers; record them here first | an honest, never-seen score. Frozen vs retrained also shows what yearly retraining is worth |

## Modelling experiments

Single-winter runs (M1–M5, validation 2023/24) first, then walk-forward (M6–M9).

| # | Question | Result | Decision |
|---|---|---|---|
| M1 | Train on winters only, or all seasons? | LightGBM 18.4 vs 17.6; Ridge 18.7 vs 17.4 (all seasons better) | **train on all seasons.** Winter-only option removed |
| M2 | Does the loss matter? | LightGBM squared error 17.6 → absolute error (L1) 15.9 | **L1 is the loss** (it matches the MAE score). `objective="l1"` is now fixed in `config.LIGHTGBM_PARAMS` |
| M3 | Does weather (value at the origin hour) help? | Single winter: 15.94 with and without. Walk-forward: mean MAE difference per winter +0.21, −0.08, −0.27, −0.02 (weather better in 10, 13, 23, 14 of 24 horizons) | **no measurable gain.** Weather removed from model inputs (see M8) |
| M4 | Ridge vs LightGBM | Single winter: Ridge 17.4, LightGBM L1 15.9. Walk-forward: Ridge beat the bar in 1 of 4 winters (mean −1.8%) | **Ridge removed** (code, `RIDGE_ALPHA`, `LINEAR_EXCLUDE`, its tests) |
| M5 | Best model vs the bar (16.2) on 2023/24 | LightGBM L1 +0.5% | one winter is too fragile: use walk-forward (E4) |
| M6 | LightGBM L1 vs the bar, walk-forward | Beats the bar in 4 of 4 winters. Mean gain +12.5% (no weather), +12.7% (weather). All 24 horizons won in 2019/20, 2020/21, 2021/22; only 15 (no weather) / 13 (weather) of 24 in 2023/24 | **LightGBM L1 is the current model** |
| M7 | Where is it strong or weak? | Strong at h1–6 in every winter (2021/22: 13.4 vs 18.2). In 2023/24 it loses to the bar at h=7 and h=14–21. The baseline was also much better that winter (h1–6: 15.3 vs 18–25 in the other winters) | long horizons are the weak spot. Earlier idea "calm winters are easier" was **not supported**: 2021/22 is equally calm and gained ~19% |
| M8 | Does future weather (weather at t+h, `fut_*`) help? Walk-forward, optimistic test | Mean gain over the bar: 13.5% vs 12.5% (no weather) and 12.7% (weather at t). Gain in MAE per winter: 0.37, 0.35, 0.18, 0.04 µg/m³ (about 1%). Better in all 4 winters, but small. In 2023/24 it does not fix the weak horizons (h14–16 worse than no weather) | **small gain, upper bound only.** A real forecast has errors, so the true gain is likely smaller. **Weather removed from the models** (decision: not worth the complexity) |
| M9 | Do other LightGBM settings help? 6 settings, walk-forward (rules fixed before the run: win 4/4, beat base by ≥ 0.3 points, simplest wins) | Mean gain over the bar: base 12.5, small_trees 12.6, small_trees_strict 12.7, big_trees 12.3, fewer_columns 12.6, **slow_more_trees 12.9** (600 trees, learning rate 0.025). All won 4/4. Only slow_more_trees beat base by ≥ 0.3 (about 0.08 µg/m³ MAE, better in all 4 winters). 2023/24 stays weak (+3.3%) | **adopt slow_more_trees** (only one passed the rule). The gain is tiny: settings barely matter, the model has reached a plateau. `tune.py` and the grid removed |

## MLOps decisions

| # | Decision | Why |
|---|---|---|
| O1 | Basic CI from the start: GitHub Actions runs `ruff check`, `pytest` and `--help` of the main commands on Python 3.10 and 3.12 | cheap, and every later step is checked automatically |
| O2 | A trained model is a **bundle**: `models.joblib` (24 models) + `metadata.json` (feature list in order, settings, data range, data hash, git commit, versions) | a live forecast must build exactly the training features; the metadata makes a model traceable |
| O3 | A test checks that features built from history up to `t` equal the training features of that row | catches training/serving skew early (a risk both Claude and ChatGPT named) |
| O4 | `train --train-until DATE` makes a frozen model | the sensor transfer test needs a model that has not seen the tested period |
| O5 | Do not automate retraining yet. First log forecast, actual, error, horizon, model version (monitoring), then decide | a retrained model can be worse; evidence first |
| O6 | MLflow runs in a local SQLite file (`mlflow.db`) with files in `mlartifacts/`; no server. Two experiments: `hawa-train`, `hawa-walk-forward`. Both are git-ignored | simplest setup that gives a real run history and UI; a server can come with Docker later |
| O7 | The saved model is logged to MLflow as the whole **bundle folder**, not 24 separate MLflow models. A registry / pyfunc wrapper comes when we build serving | 24 models belong together as one version; wrapping them as one served model is a serving decision |
| O8 | Sensor transfer test: the frozen model (trained before 2024-12-01) is scored on each AirGradient sensor's own history for winter 2025/26, against the same two baselines, plus agreement with the Embassy (`analysis/sensor_transfer.py`). **Result (winter 2025/26, 38 sensor files, 34 scored):** the model beats the bar on 29 of 34 sensors, median gain 13.1% (Embassy reference 15.2%, sensor range −13.8% to +26.5%). Sensors move with the Embassy (correlation 0.44–0.91; closest sensors 0.78–0.91) but read **much higher**: ratio of means 1.19–2.10 (median about 1.55), bias +11 to +57 µg/m³. 4 sensors had too little winter data. No clear trend with distance; the unstable sensors are mostly those with fewer than 1,000 overlap hours. Meaning: live forecasting is credible in relative terms (beats the baselines), but absolute levels differ, so a scale/calibration decision is needed before showing PM2.5 values or health categories. Use a shortlist of complete sensors. | the model was trained on the Embassy but the product reads other sensors; this decides if live forecasting is credible |

## Final test result (E5)

Model: LightGBM, absolute-error loss, 600 trees, learning rate 0.025, 24 PM2.5-history and time inputs, no weather.
Embassy station, winters only, trained on all seasons. Test winters were never used for any choice. No changes were made after this run.

| Winter | Model | Mean MAE | Best baseline | Gain over the bar |
|---|---|---|---|---|
| 2024/25 | frozen (trained before winter 2024) | 15.42 | 18.87 | **+18.3%** |
| 2025/26 | frozen | 11.44 | 13.50 | **+15.2%** |
| 2025/26 | retrained (trained before winter 2025) | 11.29 | 13.50 | **+16.3%** |

Gain over the bar by horizon (mean of the per-horizon percentages in brackets):

| Winter / model | h1 | h3 | h6 | h12 | h18 | h24 |
|---|---|---|---|---|---|---|
| 2024/25 frozen (17.5) | 27.8 | 27.8 | 20.0 | 15.9 | 14.3 | 13.1 |
| 2025/26 frozen (14.6) | 26.4 | 21.7 | 14.8 | 13.2 | 11.5 | 13.7 |
| 2025/26 retrained (15.6) | 27.2 | 22.9 | 16.5 | 14.3 | 12.2 | 14.3 |

Scored origins: 2,160 per winter. Training origins: 67,893 (frozen), 76,653 (retrained).

What it says:
- The model beats the best baseline in both unseen winters (+18.3%, +15.2%), in line with the walk-forward winters (+11.5 to +18.7), so the weak 2023/24 (+3.3%) was the odd winter.
- The gain is largest at short horizons (about +27% at h1) and shrinks to about +13% at h24, but stays positive at every horizon shown.
- Retraining once a year helped a little (+1.1 points, about 0.15 µg/m³ MAE; better at every horizon shown). A model one year old lost little. Only one winter shows this, so treat it as a hint.
- Limits: one station (Embassy), winters only, two test winters. Whole-year performance is not tested. Not a health-warning claim.

### O9. Live sensor shortlist
The batch job forecasts 8 sensors in `config.LIVE_SENSORS` (AirGradient sensors near Kathmandu plus KIAS).
Rule used: at least 1,700 scored hours and at least 10% gain over the best baseline in the sensor transfer test.
Caveat: the shortlist was chosen on the same winter it was scored on, so the gain is optimistic.

### O10. Batch forecast design
- Origin = last complete hour; horizon h = hour start at origin + h. The running hour is never used.
- Features are built by the training code (`add_features`); a test checks live and training rows are equal.
- A sensor with its newest real value older than 6 h is skipped, with a printed reason.
- Output is an idempotent CSV (dedupe on sensor, model, origin, horizon). It is the base for forecast-vs-actual monitoring.
- Values are on the sensor scale and not calibrated. Calibration is a later experiment (see open questions).

### O11. Live check of forecasts (`forecasting/actuals.py`)
Each saved forecast hour is compared with the real value once it exists, against persistence and same-hour-yesterday
(same rows for all three). This is the first monitoring step. It is on the sensor scale, so the model and the baselines
are compared fairly even without calibration. Early numbers (a few days) are a smoke test only.

### O12. DVC pipeline (`dvc.yaml`)
- Stages: clean → dataset → features → train, and walk_forward. `dvc repro` runs only the stages whose inputs changed.
  Inputs are listed by hand (code + data files), so a stage reruns when its code, `config.py` or its data changes.
- Raw data is tracked with `dvc add data/raw` (the files stay out of git; git keeps only `data/raw.dvc`).
  `dvc.lock` records the hash of every input and output of the last run.
- Left out on purpose: downloads (API keys, already cached), the frozen model and the locked final test (must stay a
  deliberate manual step), the sensor transfer test, and the live batch forecast (it runs on a schedule, not on data change).
- The train stage writes `models/embassy_latest` (fixed path, DVC needs one). Forecast rows still get the label
  `embassy_to_<date>` from the bundle metadata.
- Limit: `config.py` is one dep for every stage, so changing a setting reruns all stages. Fine for now.

## Open questions

- Why is 2023/24 harder? Is it the latest winter, so closer to the locked test winters?
- Weather was tested only on winters (Dec–Feb, dry). Does it matter in other seasons (pre-monsoon smoke, monsoon rain)? Test when we add whole-year tables.
- Scale: how to calibrate sensor-scale forecasts (AirGradient is about 1.56x the Embassy) without touching the locked test winters?
- Sin/cos calendar features were made for Ridge. Trees do not need them: check if they can go.

## Cleanup gate (before each new stage)

Remove code that lost its experiment, and record it above.

**Done (stage 1 → 2):**
- `evaluation/tune.py`, `config.TUNING_GRID` and the `params` override (M9 finished; the code is in git history)
- weather as model input: `--feature-set` switch, `include_weather`, `fut_*` future-weather columns, weather walk-forward variants (M3, M8). Kept: the weather download and the weather columns in the dataset table, so weather can be added back without re-downloading. Results: M3, M8 above
- Ridge learner, its settings and tests (M4)
- `--train-scope winter` in `evaluate_models.py` (M1)
- `--objective` switch and the squared-error LightGBM default (M2)
- `--models` option (only one model left)
- walk-forward variants renamed: `lightgbm_noweather`, `lightgbm_weather`
- old result files in `reports/` (they carry removed model names) and `data/raw/openaq/scorecard.csv` (old, wrong)

**Kept on purpose:**
- the `previous_runs` URL in `ingestion/openmeteo.py`: it is used by `analysis/openmeteo_probe.py`

**Next gate candidates:** the weather download and dataset join (nothing uses them now); sin/cos calendar features if trees do not need them.
