"""Forecast error per horizon.

Predictions and targets are tables with columns y_h01 ... y_h24 (same shape as the
targets in the dataset). Rows where a prediction or the target is missing are skipped,
and every model is scored on the SAME rows for a given horizon, so the comparison is fair.
"""
import numpy as np
import pandas as pd

from hawa import config


def target_columns(horizons=config.HORIZONS):
    return [f"y_h{h:02d}" for h in range(1, horizons + 1)]


def score(targets, predictions, horizons=config.HORIZONS):
    """Per-horizon MAE and RMSE.

    targets      DataFrame with y_hNN columns
    predictions  dict: model name -> DataFrame with y_hNN columns (same index as targets)
    Returns a long DataFrame: model, horizon, n, mae, rmse.
    """
    rows = []
    for h in range(1, horizons + 1):
        col = f"y_h{h:02d}"
        y = targets[col]
        ok = y.notna()
        for p in predictions.values():
            ok &= p[col].notna()
        for name, p in predictions.items():
            err = (p.loc[ok, col] - y[ok]).to_numpy(dtype=float)
            rows.append({
                "model": name, "horizon": h, "n": int(ok.sum()),
                "mae": float(np.abs(err).mean()) if err.size else np.nan,
                "rmse": float(np.sqrt((err ** 2).mean())) if err.size else np.nan,
            })
    return pd.DataFrame(rows)


def summarize(scores, horizons=(1, 3, 6, 12, 18, 24)):
    """Wide table: MAE for chosen horizons plus the mean over all horizons, per model."""
    wide = scores.pivot(index="model", columns="horizon", values="mae")
    out = wide[[h for h in horizons if h in wide.columns]].copy()
    out.columns = [f"MAE h{h:02d}" for h in out.columns]
    out["MAE mean(1-24)"] = wide.mean(axis=1)
    out["RMSE mean(1-24)"] = scores.groupby("model")["rmse"].mean()
    out["n (h=1)"] = scores[scores["horizon"] == 1].set_index("model")["n"]
    return out.round(2)
