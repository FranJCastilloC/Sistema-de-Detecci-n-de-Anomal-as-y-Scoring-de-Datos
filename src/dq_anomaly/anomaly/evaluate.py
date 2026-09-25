"""Metrics for a heavily imbalanced detection problem."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from dq_anomaly.anomaly.thresholds import Threshold

#: Assumed manual effort to clear one alert, used to express false positives as
#: staffing rather than as an abstract rate.
SECONDS_PER_REVIEW = 90


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Threshold,
    k_values: tuple[int, ...] = (50, 100, 285, 500, 1000),
) -> dict:
    """Ranking quality, the chosen operating point, and reviewer workload."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    prevalence = float(y_true.mean())

    average_precision = float(average_precision_score(y_true, scores))
    metrics = {
        "n_records": int(len(y_true)),
        "n_positives": int(y_true.sum()),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        # Average precision, not the trapezoid under the PR curve: trapezoidal
        # integration of precision-recall is optimistically biased.
        "pr_auc": average_precision,
        "pr_auc_lift": float(average_precision / prevalence) if prevalence else float("nan"),
        "threshold_name": threshold.name,
        "threshold_value": float(threshold.value),
        "threshold_rule": threshold.rule,
        "threshold_is_oracle": bool(threshold.is_oracle),
    }

    predicted = (scores >= threshold.value).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    alerts = int(predicted.sum())

    metrics.update(
        {
            "alerts": alerts,
            "flag_rate": float(alerts / len(y_true)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        }
    )

    order = np.argsort(-scores, kind="mergesort")
    ranked = y_true[order]
    for k in k_values:
        k = min(k, len(ranked))
        hits = int(ranked[:k].sum())
        metrics[f"precision_at_{k}"] = float(hits / k)
        metrics[f"recall_at_{k}"] = float(hits / max(1, y_true.sum()))

    review_hours = alerts * SECONDS_PER_REVIEW / 3600.0
    metrics.update(
        {
            "review_hours": float(review_hours),
            "false_positives_per_true_positive": float(fp / tp) if tp else float("inf"),
            "review_hours_per_fraud_found": float(review_hours / tp) if tp else float("inf"),
            "alerts_per_1000_records": float(1000 * alerts / len(y_true)),
        }
    )
    return metrics


def bootstrap_interval(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold_value: float,
    n_resamples: int = 200,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap intervals for PR-AUC and recall at the operating point.

    Cheap because the scores are already computed, and it answers the question a
    single-number comparison cannot: whether the gap between two models is
    larger than the noise in a test window holding fewer than a hundred frauds.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    pr_aucs, recalls = [], []
    for _ in range(n_resamples):
        index = rng.integers(0, n, size=n)
        sample_y = y_true[index]
        if sample_y.sum() == 0:
            continue
        sample_scores = scores[index]
        pr_aucs.append(average_precision_score(sample_y, sample_scores))
        predicted = sample_scores >= threshold_value
        recalls.append(predicted[sample_y == 1].mean())
    return {
        "pr_auc_ci95": [float(np.percentile(pr_aucs, 2.5)), float(np.percentile(pr_aucs, 97.5))],
        "recall_ci95": [float(np.percentile(recalls, 2.5)), float(np.percentile(recalls, 97.5))],
        "n_resamples": len(pr_aucs),
    }


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Assemble per-model metric dictionaries into the headline table."""
    columns = [
        "model", "pr_auc", "pr_auc_lift", "roc_auc", "precision", "recall", "f1",
        "alerts", "true_positives", "false_positives", "false_negatives",
        "precision_at_100", "review_hours", "review_hours_per_fraud_found",
        "threshold_name", "threshold_is_oracle",
    ]
    rows = []
    for name, metrics in results.items():
        row = {"model": name}
        row.update({key: metrics.get(key) for key in columns if key != "model"})
        rows.append(row)
    table = pd.DataFrame(rows)[columns]
    return table.sort_values("pr_auc", ascending=False).reset_index(drop=True)
