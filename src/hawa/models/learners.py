"""Learning model: LightGBM (gradient-boosted trees), one model per forecast horizon.

"Direct" forecasting: for each h = 1 ... 24 we train a separate model that predicts PM2.5 at t+h
from the features known at hour t. The 24 models share the same inputs.

Loss is absolute error (L1), because the score is MAE (see docs/decisions.md, M2).
Predictions are clipped at 0 (PM2.5 cannot be negative).
Settings are in config.py (small tuning test done: docs/decisions.md, M9).

Three functions:
  fit_models     train the 24 models            -> {h: fitted model}
  predict_models use them on a feature table    -> DataFrame y_h01 ... y_hNN
  fit_predict    both in one go (used by the evaluation scripts)
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from hawa import config


def make_model():
    return LGBMRegressor(random_state=config.RANDOM_SEED, **config.LIGHTGBM_PARAMS)


def fit_models(df, feature_cols, train_mask, horizons=config.HORIZONS):
    """Train one model per horizon on `df[train_mask]`. Returns {h: fitted model}.

    Training rows need a real target for that horizon.
    """
    cols = list(feature_cols)
    models = {}
    for h in range(1, horizons + 1):
        target = f"y_h{h:02d}"
        keep = train_mask & df[target].notna().to_numpy()
        models[h] = make_model().fit(df.loc[keep, cols], df.loc[keep, target])
    return models


def predict_models(models, X, feature_cols):
    """Predict with {h: model}. `X` must contain `feature_cols`. Returns y_h01 ... y_hNN, clipped at 0."""
    X = X[list(feature_cols)]
    out = {f"y_h{h:02d}": np.clip(m.predict(X), 0, None) for h, m in models.items()}
    return pd.DataFrame(out, index=X.index)


def fit_predict(df, feature_cols, train_mask, predict_index, horizons=config.HORIZONS):
    """Train on `df[train_mask]`, predict for `predict_index`. Returns a DataFrame y_h01 ... y_hNN."""
    models = fit_models(df, feature_cols, train_mask, horizons)
    return predict_models(models, df.loc[predict_index], feature_cols)
