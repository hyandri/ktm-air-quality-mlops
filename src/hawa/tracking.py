"""MLflow tracking: the one place that knows where runs are stored.

Runs go to a local SQLite file (config.MLFLOW_DB), files to config.MLFLOW_ARTIFACT_DIR. No server needed.
To look at them:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
then open http://127.0.0.1:5000

Experiments:
    hawa-train          one run per trained model bundle (settings, data hash, git commit, the bundle files)
    hawa-walk-forward   one run per walk-forward evaluation (gain over the baseline per winter)
"""
import contextlib

import mlflow

from hawa import config


def setup(experiment):
    """Point MLflow at our local store and select (or create) the experiment."""
    config.MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=config.MLFLOW_ARTIFACT_DIR.as_uri())
    mlflow.set_experiment(experiment)


@contextlib.contextmanager
def run(experiment, run_name, tags=None):
    """Open an MLflow run: `with tracking.run("hawa-train", "name", {"git_commit": ...}): ...`"""
    setup(experiment)
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as active:
        yield active
