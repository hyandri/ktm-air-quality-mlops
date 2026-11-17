import numpy as np
import pandas as pd

from hawa.evaluation import metrics
from hawa.models import learners


def _df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC", name="timestamp_utc")
    x = rng.uniform(10, 100, n)
    df = pd.DataFrame({"f1": x}, index=idx)
    for h in (1, 2):
        df[f"y_h{h:02d}"] = x * 0.5 + h
    return df


def test_fit_predict_shape_and_learning():
    df = _df()
    train = np.arange(len(df)) < 300
    p = learners.fit_predict(df, ["f1"], train, df.index[300:], horizons=2)
    assert list(p.columns) == ["y_h01", "y_h02"] and len(p) == 100
    mae = (p["y_h01"] - df.loc[p.index, "y_h01"]).abs().mean()
    assert mae < 8                                        # it learned the 0.5*x + 1 pattern


def test_training_uses_only_masked_rows():
    df = _df()
    df.loc[df.index[300:], "f1"] = 10_000.0                # "future" rows with absurd values
    df.loc[df.index[300:], "y_h01"] = 10_000.0
    train = np.arange(len(df)) < 300
    p = learners.fit_predict(df, ["f1"], train, df.index[:50], horizons=1)
    assert p["y_h01"].max() < 200                          # model never saw the 10,000 rows


def test_predictions_never_negative():
    df = _df()
    df["y_h01"] = -5.0 + df["f1"] * 0.0
    train = np.arange(len(df)) < 300
    p = learners.fit_predict(df, ["f1"], train, df.index[300:], horizons=1)
    assert (p["y_h01"] >= 0).all()


def test_loss_is_absolute_error():
    assert learners.make_model().get_params()["objective"] == "l1"


def test_improvement_vs_best_baseline():
    sc = pd.DataFrame({"model": ["b1", "b2", "m"], "horizon": [1, 1, 1],
                       "n": [5, 5, 5], "mae": [10.0, 20.0, 8.0], "rmse": [1, 1, 1]})
    r = metrics.improvement_vs_best_baseline(sc, ["b1", "b2"], horizons=(1,))
    assert r.loc["m", "h01"] == 20.0                        # (10-8)/10



def test_params_override_only_the_given_keys():
    m = learners.make_model({"num_leaves": 7})
    assert m.get_params()["num_leaves"] == 7
    assert m.get_params()["objective"] == "l1"             # others stay at the config values


def test_tuning_grid_only_uses_known_lightgbm_keys():
    from hawa import config
    for name, overrides in config.TUNING_GRID.items():
        assert set(overrides) <= set(config.LIGHTGBM_PARAMS), name
        learners.make_model(overrides)                      # builds without error
