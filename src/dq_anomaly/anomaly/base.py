"""Shared contract for anomaly scorers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AnomalyScorer(Protocol):
    """Any detector in this project scores higher for more anomalous rows.

    Fixing the direction here matters: scikit-learn's ``score_samples`` returns
    the opposite convention, and a sign error would be invisible in the code but
    catastrophic in the metrics.
    """

    name: str

    def fit(self, X: np.ndarray) -> "AnomalyScorer": ...

    def score(self, X: np.ndarray) -> np.ndarray: ...

    def save(self, path: Path) -> Path: ...
