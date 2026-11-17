"""Learning models: Ridge (linear) and LightGBM (trees). One model per forecast horizon.

"Direct" forecasting: for each h = 1 ... 24 we train a separate model that predicts PM2.5 at t+h
from the features known at hour t. The 24 models share the same inputs.

  ridge     linear model with ridge penalty. Missing inputs are filled with the training median and
            inputs are standardised (both learned from training rows only).
  lightgbm  gradient-boosted trees. Handles missing inputs by itself.

Predictions are clipped at 0 (PM2.5 cannot be negative).
Settings (not tuned yet) are in config.py.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hawa import config

MODEL_NAMES = ("ridge", "lightgbm")


def make_model(name):
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             Ridge(alpha=config.RIDGE_ALPHA))
    if name == "lightgbm":
        return LGBMRegressor(random_state=config.RANDOM_SEED, **config.LIGHTGBM_PARAMS)
    raise ValueError(f"unknown model: {name}")


def columns_for(name, feature_cols):
    if name == "ridge":
        return [c for c in feature_cols if c not in config.LINEAR_EXCLUDE]
    return list(feature_cols)


def fit_predict(name, df, feature_cols, train_mask, predict_index, horizons=config.HORIZONS):
    """Train one model per horizon on `df[train_mask]`, predict for `predict_index`.

    Training rows need a real target for that horizon. Returns a DataFrame y_h01 ... y_hNN.
    """
    cols = columns_for(name, feature_cols)
    X_pred = df.loc[predict_index, cols]
    out = {}
    for h in range(1, horizons + 1):
        target = f"y_h{h:02d}"
        rows = df[train_mask & df[target].notna().to_numpy()]
        model = make_model(name).fit(rows[cols], rows[target])
        out[target] = np.clip(model.predict(X_pred), 0, None)
    return pd.DataFrame(out, index=predict_index)
