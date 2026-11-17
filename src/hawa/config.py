"""Project-wide settings: paths, locations and cleaning constants.

Everything that more than one module needs lives here, so it is changed in one place.
"""
import os
from pathlib import Path

# src/hawa/config.py -> project root is two folders up from "hawa"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path):
    """Tiny .env reader (KEY=VALUE per line). Real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")

# ----------------------------------------------------------------- data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"   # saved model bundles (not in git)
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"              # MLflow run history (SQLite file, not in git)
MLFLOW_ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"  # files MLflow stores per run (not in git)
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB}"
PROCESSED_DIR = DATA_DIR / "processed"

OPENAQ_RAW = RAW_DIR / "openaq"
OPENMETEO_RAW = RAW_DIR / "openmeteo"
OPEN_DATA_NEPAL_RAW = RAW_DIR / "open_data_nepal"

# ------------------------------------------------------------------ locations
# min_lon, min_lat, max_lon, max_lat (rough Kathmandu Valley box)
KATHMANDU_BBOX = "85.15,27.55,85.60,27.85"
EMBASSY_LAT, EMBASSY_LON = 27.738703, 85.336205  # also used as the weather point
OPENAQ_PM25_PARAMETER_ID = 2

# --------------------------------------------------------------------- weather
WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "boundary_layer_height",
]

# Variables used in the first models. Boundary layer height is left out because the
# archived forecasts miss most of it (see data/README.md).
MODEL_WEATHER_VARIABLES = [v for v in WEATHER_VARIABLES if v != "boundary_layer_height"]
# Archived forecasts (what a live model would have seen) exist from this date;
# before it only ERA5 is available.
FORECAST_WEATHER_START = "2021-01-01"

# ------------------------------------------------------------------- dataset
HORIZONS = 24             # forecast 1..24 hours ahead
LOCAL_UTC_OFFSET_H = 5.75  # Kathmandu = UTC+05:45

# -------------------------------------------------------------- cleaning rules
ERROR_VALUES = {985.0}  # seen in the AirNow-derived data; treated as an error code
FLAT_RUN_HOURS = 12     # this many identical values in a row = stuck sensor

# ----------------------------------------------------------- evaluation splits
# A winter is named by the year it STARTS: 2023 = Dec 2023 - Feb 2024 (Kathmandu local time).
# Winter 2022/23 is skipped: the Embassy has almost no data then.
WINTER_MONTHS = (12, 1, 2)
TRAIN_WINTERS = (2017, 2018, 2019, 2020, 2021)
VALIDATION_WINTERS = (2023,)
TEST_WINTERS = (2024, 2025)
# Whole-year scope (used later): train = all seasons before 2022-12-01,
# validation = 2023-12-01 .. 2024-11-30, test = from 2024-12-01 (all local time).
TRAIN_ALL_END = "2022-12-01"
VALIDATION_ALL = ("2023-12-01", "2024-12-01")
TEST_ALL_START = "2024-12-01"

# ------------------------------------------------------------------- features
FILL_LIMIT_H = 3                      # carry the last real PM2.5 forward for at most this many hours (features only)
LAG_HOURS = (1, 2, 3, 6, 12, 24, 48, 168)       # PM2.5 this many hours before the forecast origin
ROLL_WINDOWS = (6, 24, 72, 168)       # rolling mean windows (hours, ending at the origin hour)

# --------------------------------------------------------------------- models
RANDOM_SEED = 42
LIGHTGBM_PARAMS = dict(objective="l1", n_estimators=600, learning_rate=0.025, num_leaves=31, min_child_samples=50,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

# ----------------------------------------------------------------- walk-forward
# Each winter below is scored in turn, training on ALL data before it (expanding window).
# 2017 and 2018 are skipped (too little earlier data), 2022 has no data, and the TEST winters are never used.
WALK_FORWARD_WINTERS = (2019, 2020, 2021, 2023)
TRAIN_GAP_H = 24   # training origins must end this many hours before the scored winter starts
