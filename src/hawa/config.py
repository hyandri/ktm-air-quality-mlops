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

# -------------------------------------------------------------- cleaning rules
ERROR_VALUES = {985.0}  # seen in the AirNow-derived data; treated as an error code
FLAT_RUN_HOURS = 12     # this many identical values in a row = stuck sensor
