"""Save and load a trained model bundle: the 24 models plus a `metadata.json` that says how they were made.

A bundle is a folder:
    models.joblib    {horizon: fitted LightGBM}
    metadata.json    feature columns, settings, training data range, data file hash, versions, git commit

The feature list in the metadata is what a forecast job must build, in this order.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import joblib

MODELS_FILE = "models.joblib"
METADATA_FILE = "metadata.json"


@dataclass
class Bundle:
    models: dict
    metadata: dict


def save_bundle(path, models, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, path / MODELS_FILE)
    (path / METADATA_FILE).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return path


def load_bundle(path):
    path = Path(path)
    models = joblib.load(path / MODELS_FILE)
    metadata = json.loads((path / METADATA_FILE).read_text())
    return Bundle(models=models, metadata=metadata)
