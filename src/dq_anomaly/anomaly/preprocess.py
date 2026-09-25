"""Feature construction for the credit card transactions.

V1-V28 are already PCA components, so the work here is confined to the two raw
columns and to scaling.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

V_COLUMNS = [f"V{i}" for i in range(1, 29)]


class CreditCardPreprocessor:
    """Builds the model matrix and holds the scaler fitted on training data only.

    ``Time`` is deliberately not used as a feature. It is a monotonically
    increasing offset from the first transaction in the file, so it carries row
    position rather than anything a future batch could reproduce. What is
    genuinely reusable is the time of day, encoded on a circle so that 23:59 and
    00:01 sit next to each other.

    ``Amount`` is log-transformed before scaling: its raw range spans four
    orders of magnitude, and without the transform the autoencoder spends its
    capacity reconstructing a handful of very large transactions.
    """

    def __init__(self, use_hour: bool = True) -> None:
        self.use_hour = use_hour
        self.scaler = StandardScaler()
        self.feature_names_: list[str] = []
        self._fitted = False

    def _raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[V_COLUMNS].astype("float64").copy()
        features["log_amount"] = np.log1p(df["Amount"].astype("float64").clip(lower=0))
        if self.use_hour:
            hour = (df["Time"].astype("float64") / 3600.0) % 24.0
            features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
            features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        return features

    def fit(self, df: pd.DataFrame) -> "CreditCardPreprocessor":
        features = self._raw_features(df)
        self.feature_names_ = list(features.columns)
        self.scaler.fit(features.to_numpy())
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("preprocessor must be fitted before transform")
        features = self._raw_features(df)[self.feature_names_]
        return self.scaler.transform(features.to_numpy()).astype("float32")

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> "CreditCardPreprocessor":
        return joblib.load(Path(path))


def temporal_split(
    df: pd.DataFrame, train_frac: float = 0.6, calib_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically into train / calibration / test windows.

    The data covers two consecutive days, so a time-ordered split is both the
    honest evaluation and the realistic story: fit on history, score the batch
    that arrives next. ``mergesort`` is required because ``Time`` has many
    duplicate values and the default quicksort is unstable, which would silently
    reshuffle tied rows between runs and change split membership.
    """
    ordered = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    calib_end = int(n * (train_frac + calib_frac))
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:calib_end].copy(),
        ordered.iloc[calib_end:].copy(),
    )
