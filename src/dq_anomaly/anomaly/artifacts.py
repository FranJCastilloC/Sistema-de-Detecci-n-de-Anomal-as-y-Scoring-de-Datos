"""Loading the trained artifacts produced by the training script."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dq_anomaly.anomaly.autoencoder import AutoencoderDetector
from dq_anomaly.anomaly.isolation_forest import IsolationForestDetector
from dq_anomaly.anomaly.preprocess import CreditCardPreprocessor
from dq_anomaly.config import PATHS


@dataclass
class ModelBundle:
    """Everything needed to score a new batch the way the models were trained."""

    preprocessor: CreditCardPreprocessor
    autoencoder: AutoencoderDetector
    isolation_forest: IsolationForestDetector | None
    calibration_scores: dict[str, np.ndarray]

    @property
    def feature_names(self) -> list[str]:
        return self.preprocessor.feature_names_


def bundle_available(model_dir: Path | None = None) -> bool:
    model_dir = Path(model_dir or PATHS.models)
    return (model_dir / "autoencoder.pt").exists() and (
        model_dir / "preprocessor.joblib"
    ).exists()


def load_bundle(model_dir: Path | None = None) -> ModelBundle:
    """Load the trained bundle, raising a clear error when it is absent."""
    model_dir = Path(model_dir or PATHS.models)
    if not bundle_available(model_dir):
        raise FileNotFoundError(
            f"No trained model found in {model_dir}. "
            "Run `python scripts/03_train_models.py` first."
        )
    preprocessor = CreditCardPreprocessor.load(model_dir / "preprocessor.joblib")
    autoencoder = AutoencoderDetector.load(model_dir / "autoencoder.pt")

    forest_path = model_dir / "isolation_forest.joblib"
    forest = IsolationForestDetector.load(forest_path) if forest_path.exists() else None

    calibration: dict[str, np.ndarray] = {}
    calibration_path = model_dir / "calibration_scores.npz"
    if calibration_path.exists():
        with np.load(calibration_path) as archive:
            calibration = {key: archive[key] for key in archive.files}

    return ModelBundle(preprocessor, autoencoder, forest, calibration)
