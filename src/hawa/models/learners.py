"""Learning model: LightGBM (gradient-boosted trees), one model per forecast horizon.

"Direct" forecasting: for each h = 1 ... 24 we train a separate model that predicts PM2.5 at t+h
from the features known at hour t. The 24 models share the same inputs.

Loss is absolute error (L1), because the score is MAE (see docs/decisions.md, M2).
Predictions are clipped at 0 (PM2.5 cannot be negative).
Settings (not tuned yet) are in config.py.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from hawa import config


def make_model(params=None):
    """LightGBM with the settings in config.LIGHTGBM_PARAMS. `params` overrides single keys."""
    return LGBMRegressor(random_state=config.RANDOM_SEED, **{**config.LIGHTGBM_PARAMS, **(params or {})})


def fit_predict(df, feature_cols, train_mask, predict_index, horizons=config.HORIZONS, params=None):
    """Train one model per horizon on `df[train_mask]`, predict for `predict_index`.

    Training rows need a real target for that horizon. `params` overrides LightGBM settings.
    Returns a DataFrame y_h01 ... y_hNN.
    """
    cols = list(feature_cols)
    out = {}
    for h in range(1, horizons + 1):
        target = f"y_h{h:02d}"
        keep = train_mask & df[target].notna().to_numpy()
        model = make_model(params).fit(df.loc[keep, cols], df.loc[keep, target])
        out[target] = np.clip(model.predict(df.loc[predict_index, cols]), 0, None)
    return pd.DataFrame(out, index=predict_index)
