import numpy as np
import pandas as pd
import pytest

from hawa import config
from hawa.models import bundle, learners, train
from hawa.processing import build_dataset as b
from hawa.processing import features


def _table(n=1500, horizons=2, seed=0):
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC", name="timestamp_utc")
    rng = np.random.default_rng(seed)
    pm = pd.Series(50 + 20 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 5, n), index=idx)
    df = b.add_targets(b.add_time_features(pd.DataFrame({"pm25": pm})), horizons=horizons)
    return features.add_features(df)


def test_train_returns_models_and_metadata():
    df = _table()
    models, meta = train.train(df, horizons=2)
    assert sorted(models) == [1, 2]
    assert meta["feature_columns"] == features.feature_columns(df)
    assert meta["n_train_origins"] > 0 and meta["horizons"] == 2
    assert meta["lightgbm_params"]["objective"] == "l1"


def test_train_until_keeps_a_gap_before_the_date():
    df = _table()
    _, meta = train.train(df, train_until="2022-02-15", horizons=2)
    last = pd.Timestamp(meta["train_last_origin_utc"]) + pd.Timedelta(hours=config.LOCAL_UTC_OFFSET_H)
    assert last.tz_localize(None) + pd.Timedelta(hours=config.TRAIN_GAP_H) <= pd.Timestamp("2022-02-15")


def test_bundle_roundtrip_gives_identical_predictions(tmp_path):
    df = _table()
    models, meta = train.train(df, horizons=2)
    bundle.save_bundle(tmp_path / "m", models, meta)
    loaded = bundle.load_bundle(tmp_path / "m")
    cols = loaded.metadata["feature_columns"]
    X = df.iloc[-100:]
    a = learners.predict_models(models, X, cols)
    c = learners.predict_models(loaded.models, X, cols)
    pd.testing.assert_frame_equal(a, c)
    assert loaded.metadata["feature_columns"] == meta["feature_columns"]


def test_fit_predict_still_matches_fit_then_predict():
    df = _table()
    cols = features.feature_columns(df)
    train_mask = np.arange(len(df)) < 1000
    idx = df.index[1000:1100]
    a = learners.fit_predict(df, cols, train_mask, idx, horizons=2)
    b2 = learners.predict_models(learners.fit_models(df, cols, train_mask, 2), df.loc[idx], cols)
    pd.testing.assert_frame_equal(a, b2)


def test_no_training_rows_is_an_error():
    df = _table()
    with pytest.raises(ValueError):
        train.train(df, train_until="2000-01-01", horizons=2)
