"""Behavioural tests for the detectors, the preprocessing and the thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dq_anomaly.anomaly.autoencoder import AEConfig, AutoencoderDetector
from dq_anomaly.anomaly.evaluate import evaluate_scores
from dq_anomaly.anomaly.isolation_forest import IsolationForestDetector
from dq_anomaly.anomaly.preprocess import CreditCardPreprocessor, temporal_split
from dq_anomaly.anomaly.thresholds import (
    threshold_best_f1,
    threshold_from_review_budget,
)


def _synthetic_transactions(n: int = 2_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)}
    data["Time"] = np.sort(rng.uniform(0, 172_800, n))
    data["Amount"] = rng.gamma(2, 30, n)
    data["Class"] = 0
    return pd.DataFrame(data)


def test_preprocessor_builds_the_expected_feature_set():
    frame = _synthetic_transactions()
    preprocessor = CreditCardPreprocessor()
    matrix = preprocessor.fit_transform(frame)
    assert matrix.shape == (len(frame), 31)
    assert preprocessor.feature_names_[-3:] == ["log_amount", "hour_sin", "hour_cos"]
    assert "Time" not in preprocessor.feature_names_
    assert not np.isnan(matrix).any()


def test_hour_encoding_is_continuous_across_midnight():
    before = pd.DataFrame({**{f"V{i}": [0.0] for i in range(1, 29)},
                           "Time": [86_399.0], "Amount": [10.0]})
    after = pd.DataFrame({**{f"V{i}": [0.0] for i in range(1, 29)},
                          "Time": [86_401.0], "Amount": [10.0]})
    preprocessor = CreditCardPreprocessor()
    preprocessor.fit(_synthetic_transactions())
    gap = np.abs(preprocessor.transform(before) - preprocessor.transform(after))
    assert gap.max() < 0.1


def test_scaler_is_fitted_on_training_rows_only():
    frame = _synthetic_transactions(3_000)
    train, _, test = temporal_split(frame)
    preprocessor = CreditCardPreprocessor().fit(train)
    # Leakage would centre the test window exactly on zero.
    assert abs(preprocessor.transform(test).mean()) > 1e-6
    assert abs(preprocessor.transform(train).mean()) < 1e-6


def test_temporal_split_is_ordered_and_stable_with_tied_timestamps():
    frame = _synthetic_transactions(1_000)
    frame.loc[:, "Time"] = np.repeat(np.arange(100, dtype=float), 10)
    first = temporal_split(frame)
    second = temporal_split(frame)
    for left, right in zip(first, second):
        pd.testing.assert_frame_equal(left, right)
    assert first[0]["Time"].max() <= first[1]["Time"].min()


def test_preprocessor_survives_a_joblib_round_trip(tmp_path):
    frame = _synthetic_transactions()
    preprocessor = CreditCardPreprocessor().fit(frame)
    path = preprocessor.save(tmp_path / "pre.joblib")
    restored = CreditCardPreprocessor.load(path)
    np.testing.assert_allclose(preprocessor.transform(frame), restored.transform(frame))


def test_autoencoder_scores_outliers_far_above_normals():
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, size=(3_000, 8)).astype("float32")
    detector = AutoencoderDetector(AEConfig(input_dim=8, max_epochs=40)).fit(train)
    normal = detector.score(rng.normal(0, 1, size=(400, 8)).astype("float32"))
    shifted = detector.score(rng.normal(8, 1, size=(40, 8)).astype("float32"))
    assert np.median(shifted) > 10 * np.median(normal)


def test_autoencoder_training_is_reproducible():
    rng = np.random.default_rng(1)
    train = rng.normal(0, 1, size=(1_500, 6)).astype("float32")
    probe = rng.normal(0, 1, size=(50, 6)).astype("float32")
    config = AEConfig(input_dim=6, max_epochs=25, seed=123)
    first = AutoencoderDetector(config).fit(train).score(probe)
    second = AutoencoderDetector(config).fit(train).score(probe)
    np.testing.assert_allclose(first, second)


def test_per_feature_error_points_at_the_corrupted_field():
    rng = np.random.default_rng(2)
    train = rng.normal(0, 1, size=(3_000, 6)).astype("float32")
    detector = AutoencoderDetector(AEConfig(input_dim=6, max_epochs=60)).fit(train)
    probe = rng.normal(0, 1, size=(1, 6)).astype("float32")
    probe[0, 3] = 12.0
    assert int(np.argmax(detector.per_feature_errors(probe)[0])) == 3


def test_autoencoder_round_trips_through_disk(tmp_path):
    rng = np.random.default_rng(3)
    train = rng.normal(0, 1, size=(1_200, 5)).astype("float32")
    detector = AutoencoderDetector(AEConfig(input_dim=5, max_epochs=20)).fit(train)
    path = detector.save(tmp_path / "ae.pt")
    restored = AutoencoderDetector.load(path)
    np.testing.assert_allclose(detector.score(train), restored.score(train), rtol=1e-6)


def test_isolation_forest_scores_higher_for_anomalies():
    rng = np.random.default_rng(4)
    train = rng.normal(0, 1, size=(2_000, 5))
    detector = IsolationForestDetector().fit(train)
    assert detector.score(rng.normal(9, 1, size=(30, 5))).mean() > \
        detector.score(rng.normal(0, 1, size=(200, 5))).mean()


def test_review_budget_flags_the_requested_share():
    scores = np.linspace(0, 1, 10_000)
    threshold = threshold_from_review_budget(scores, 0.005)
    assert (scores >= threshold.value).sum() == pytest.approx(50, abs=2)
    assert threshold.is_oracle is False


def test_best_f1_threshold_stays_marked_as_an_oracle():
    """Guards against the oracle quietly becoming the headline number."""
    rng = np.random.default_rng(5)
    scores = rng.random(1_000)
    labels = (scores > 0.9).astype(int)
    assert threshold_best_f1(scores, labels).is_oracle is True


def test_random_scores_yield_pr_auc_near_the_prevalence():
    rng = np.random.default_rng(6)
    labels = (rng.random(20_000) < 0.01).astype(int)
    scores = rng.random(20_000)
    metrics = evaluate_scores(labels, scores, threshold_from_review_budget(scores, 0.005))
    assert metrics["pr_auc"] == pytest.approx(labels.mean(), abs=0.006)


def test_reviewer_workload_arithmetic_is_consistent():
    rng = np.random.default_rng(7)
    labels = (rng.random(10_000) < 0.02).astype(int)
    scores = rng.random(10_000) + labels * 0.5
    metrics = evaluate_scores(labels, scores, threshold_from_review_budget(scores, 0.01))
    assert metrics["review_hours"] == pytest.approx(metrics["alerts"] * 90 / 3600)
    assert metrics["true_positives"] + metrics["false_positives"] == metrics["alerts"]
