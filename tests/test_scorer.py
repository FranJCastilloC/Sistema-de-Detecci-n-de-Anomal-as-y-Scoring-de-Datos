"""Aggregation behaviour, including the two properties the design depends on."""

from __future__ import annotations

import pandas as pd
import pytest

from dq_anomaly.quality.profile import ColumnExpectation, TableExpectation
from dq_anomaly.quality.scorer import (
    aggregate_arithmetic,
    aggregate_geometric,
    score_dataframe,
    to_grade,
)

WEIGHTS = {"completeness": 0.25, "validity": 0.25, "consistency": 0.20,
           "uniqueness": 0.15, "plausibility": 0.15}


def test_geometric_mean_punishes_a_dead_dimension_far_harder_than_arithmetic():
    scores = {"completeness": 1.0, "validity": 1.0, "consistency": 1.0,
              "uniqueness": 0.0, "plausibility": 1.0}
    geometric = aggregate_geometric(scores, WEIGHTS)
    arithmetic = aggregate_arithmetic(scores, WEIGHTS)
    assert arithmetic == pytest.approx(85.0)
    # The whole reason for choosing it: one unusable dimension must not be
    # averaged away by four healthy ones.
    assert geometric < 55.0


def test_perfect_batch_scores_one_hundred(tiny_clean_df, tiny_profile):
    report = score_dataframe(tiny_clean_df, tiny_profile)
    assert report.global_score == pytest.approx(100.0, abs=1e-6)
    assert report.grade == "A"


def test_grade_boundaries_are_inclusive():
    healthy = {"completeness": 1.0}
    assert to_grade(90.0, healthy) == "A"
    assert to_grade(89.99, healthy) == "B"
    assert to_grade(80.0, healthy) == "B"
    assert to_grade(70.0, healthy) == "C"
    assert to_grade(60.0, healthy) == "D"
    assert to_grade(59.99, healthy) == "F"


def test_a_collapsed_dimension_caps_the_grade_at_d():
    assert to_grade(95.0, {"completeness": 1.0, "uniqueness": 0.4}) == "D"


def test_broken_primary_key_forces_grade_f(tiny_clean_df, tiny_profile):
    frame = pd.concat([tiny_clean_df, tiny_clean_df.iloc[[0]]], ignore_index=True)
    report = score_dataframe(frame, tiny_profile)
    assert report.grade == "F"
    # A broken key is not a gradeable condition, even at a near-perfect score.
    assert report.global_score > 90


def test_dimension_with_nothing_to_check_is_excluded_not_scored_as_perfect():
    frame = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
    profile = TableExpectation(name="t", primary_key=[], columns=[
        ColumnExpectation(name="v", dtype="float")])
    report = score_dataframe(frame, profile)
    assert report.dimension_scores["consistency"].weight == 0.0


def test_quality_package_does_not_depend_on_the_anomaly_package():
    """The architectural contract: quality scoring works with no model present."""
    import importlib
    import sys

    for module in [name for name in sys.modules if name.startswith("dq_anomaly.anomaly")]:
        del sys.modules[module]
    importlib.import_module("dq_anomaly.quality")
    assert not any(name.startswith("dq_anomaly.anomaly") for name in sys.modules)
