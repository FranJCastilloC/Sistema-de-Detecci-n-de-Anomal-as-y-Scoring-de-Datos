"""Classical baseline: Isolation Forest over the same feature matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class IFConfig:
    n_estimators: int = 200
    max_samples: int = 8192
    max_features: float = 1.0
    contamination: str | float = "auto"
    random_state: int = 42
    n_jobs: int = -1


class IsolationForestDetector:
    """Wraps scikit-learn so the score direction matches the rest of the project."""

    name = "isolation_forest"

    def __init__(self, config: IFConfig | None = None) -> None:
        self.config = config or IFConfig()
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            max_features=self.config.max_features,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            bootstrap=False,
        )

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        # A subsample larger than the data is a no-op that scikit-learn warns
        # about on every small batch; clamp it instead.
        self.model.set_params(max_samples=min(self.config.max_samples, len(X)))
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher means more anomalous, inverting scikit-learn's convention."""
        return -self.model.score_samples(X).astype("float64")

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> "IsolationForestDetector":
        return joblib.load(Path(path))
