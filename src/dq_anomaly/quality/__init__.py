"""Rule-based data quality scoring.

This subpackage is deliberately independent of :mod:`dq_anomaly.anomaly`: a
quality score must be computable for any batch, with no trained model present.
A test enforces that the import graph stays one-directional.
"""

from dq_anomaly.quality.profile import (
    ColumnExpectation,
    CrossFieldRule,
    TableExpectation,
    detect_profile,
    infer_profile,
    load_profile,
)
from dq_anomaly.quality.scorer import (
    DIMENSIONS,
    DimensionScore,
    QualityReport,
    score_dataframe,
)

__all__ = [
    "ColumnExpectation", "CrossFieldRule", "TableExpectation", "detect_profile",
    "infer_profile", "load_profile", "DIMENSIONS", "DimensionScore",
    "QualityReport", "score_dataframe",
]
