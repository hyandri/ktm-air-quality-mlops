"""Time-based train / validation / test splits. Never random.

A row is a forecast origin (hour t). Its targets look up to 24 h ahead, so splits are
cut by the origin time. All dates are Kathmandu local time (UTC+05:45).

Two scopes:
  winter  only Dec-Feb origins of the chosen winters   (current focus)
  all     every origin in a date range, all seasons    (planned, later)

Winters are named by their start year: 2023 = Dec 2023 - Feb 2024.
The settings live in config.py.
"""
import pandas as pd

from hawa import config

SPLITS = ("train", "validation", "test")
SCOPES = ("winter", "all")


def _local(index):
    return index + pd.Timedelta(hours=config.LOCAL_UTC_OFFSET_H)


def winter_start_year(index):
    """Start year of the winter an hour belongs to (NaN-like -1 outside Dec-Feb)."""
    local = _local(index)
    in_winter = local.month.isin(config.WINTER_MONTHS)
    year = pd.Series(local.year, index=index)
    year[local.month.isin([1, 2])] -= 1
    return year.where(in_winter, -1).to_numpy()


def split_mask(index, split, scope="winter"):
    """Boolean numpy array: which rows of `index` (UTC DatetimeIndex) belong to `split`."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    if scope == "winter":
        years = {"train": config.TRAIN_WINTERS, "validation": config.VALIDATION_WINTERS,
                 "test": config.TEST_WINTERS}[split]
        return pd.Series(winter_start_year(index)).isin(years).to_numpy()
    local = _local(index).tz_localize(None)
    if split == "train":
        return local < pd.Timestamp(config.TRAIN_ALL_END)
    if split == "validation":
        lo, hi = config.VALIDATION_ALL
        return (local >= pd.Timestamp(lo)) & (local < pd.Timestamp(hi))
    return local >= pd.Timestamp(config.TEST_ALL_START)
