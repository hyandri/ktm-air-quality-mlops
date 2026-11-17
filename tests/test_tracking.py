import mlflow
import numpy as np
import pandas as pd
import pytest

from hawa import config
from hawa.evaluation import walk_forward
from hawa.models import train
from hawa.processing import build_dataset as b
from hawa.processing import features


def _use_temp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setitem(config.LIGHTGBM_PARAMS, "n_estimators", 5)        # keep the test fast


def _features_file(tmp_path, n=700):
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC", name="timestamp_utc")
    rng = np.random.default_rng(0)
    pm = pd.Series(50 + 20 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 5, n), index=idx)
    df = features.add_features(b.add_targets(b.add_time_features(pd.DataFrame({"pm25": pm}))))
    path = tmp_path / "demo_features_auto.csv"
    df.reset_index().to_csv(path, index=False)
    return path


def test_train_logs_an_mlflow_run(tmp_path, monkeypatch):
    _use_temp_store(monkeypatch, tmp_path)
    path = _features_file(tmp_path)
    train.main(["--dataset", str(path), "--station", "demo", "--out-dir", str(tmp_path / "models")])
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=["hawa-train"])
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["params.objective"] == "l1" and row["params.n_estimators"] == "5"
    assert row["tags.station"] == "demo" and len(row["tags.dataset_sha256"]) == 64
    files = [f.path for f in mlflow.MlflowClient().list_artifacts(row["run_id"], "bundle")]
    assert "bundle/models.joblib" in files and "bundle/metadata.json" in files


def test_no_track_leaves_no_mlflow_store(tmp_path, monkeypatch):
    _use_temp_store(monkeypatch, tmp_path)
    path = _features_file(tmp_path)
    train.main(["--dataset", str(path), "--station", "demo", "--out-dir", str(tmp_path / "models"), "--no-track"])
    assert not (tmp_path / "mlflow.db").exists()
    assert any((tmp_path / "models").iterdir())                      # the bundle is still saved


def _fake_scores():
    rows = []
    for fold, (mae, bar) in {"2019/20": (8.0, 10.0), "2020/21": (9.0, 9.0)}.items():
        for model, v in (("lightgbm", mae), ("persistence", bar + 5), ("same_hour_yesterday", bar)):
            for h in (1, 2):
                rows.append({"fold": fold, "model": model, "horizon": h, "n": 10, "mae": v, "rmse": v})
    return pd.DataFrame(rows)


def test_summary_metrics():
    m = walk_forward.summary_metrics(_fake_scores())
    assert m["gain_pct_2019_20"] == pytest.approx(20.0) and m["gain_pct_2020_21"] == pytest.approx(0.0)
    assert m["mean_gain_pct"] == pytest.approx(10.0) and m["folds_won"] == 1
    assert m["bar_2019_20"] == 10.0


def test_walk_forward_logs_to_mlflow(tmp_path, monkeypatch):
    _use_temp_store(monkeypatch, tmp_path)
    csv = tmp_path / "wf.csv"
    scores = _fake_scores()
    scores.to_csv(csv, index=False)
    walk_forward.log_to_mlflow(scores, csv, "demo")
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=["hawa-walk-forward"])
    assert len(runs) == 1 and runs.iloc[0]["metrics.mean_gain_pct"] == pytest.approx(10.0)
