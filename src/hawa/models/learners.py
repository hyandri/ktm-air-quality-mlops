"""Learning model: LightGBM (gradient-boosted trees), one model per forecast horizon.

"Direct" forecasting: for each h = 1 ... 24 we train a separate model that predicts PM2.5 at t+h
from the features known at hour t. The 24 models share the same inputs.

Optional "future weather" (experiment M8): the model for horizon h also gets the weather at hour
t+h (columns `fut_<variable>`), as if a weather forecast for the target hour were available.
CAUTION: the archived weather is close to what actually happened, so this is an optimistic test.
A real forecast issued at hour t has errors, which these columns do not have.

Loss is absolute error (L1), because the score is MAE (see docs/decisions.md, M2).
Predictions are clipped at 0 (PM2.5 cannot be negative).
Settings (not tuned yet) are in config.py.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from hawa import config


def make_model():
    return LGBMRegressor(random_state=config.RANDOM_SEED, **config.LIGHTGBM_PARAMS)


def future_weather(df, h):
    """Weather at hour t+h, placed on the row of origin t. Needs a regular hourly index."""
    cols = [c for c in config.MODEL_WEATHER_VARIABLES if c in df.columns]
    return df[cols].shift(-h).add_prefix("fut_")


def fit_predict(df, feature_cols, train_mask, predict_index, horizons=config.HORIZONS,
                use_future_weather=False):
    """Train one model per horizon on `df[train_mask]`, predict for `predict_index`.

    Training rows need a real target for that horizon.
    use_future_weather=True adds the `fut_*` columns (weather at t+h) for horizon h.
    Returns a DataFrame y_h01 ... y_hNN.
    """
    cols = list(feature_cols)
    out = {}
    for h in range(1, horizons + 1):
        target = f"y_h{h:02d}"
        X = df[cols]
        if use_future_weather:
            X = pd.concat([X, future_weather(df, h)], axis=1)
        keep = train_mask & df[target].notna().to_numpy()
        model = make_model().fit(X[keep], df.loc[keep, target])
        out[target] = np.clip(model.predict(X.loc[predict_index]), 0, None)
    return pd.DataFrame(out, index=predict_index)
