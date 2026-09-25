"""Turning a continuous anomaly score into a flag.

Threshold choice is where most unsupervised fraud write-ups quietly cheat: they
pick the cut-off that maximises F1 on the labelled test set, a number no
deployment could ever have known in advance. Each strategy here records whether
it needed labels, and the oracle one is flagged as such everywhere it appears.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_recall_curve


@dataclass(frozen=True)
class Threshold:
    """A decision cut-off plus how it was obtained."""

    name: str
    value: float
    rule: str
    expected_flag_rate: float
    is_oracle: bool = False


def threshold_from_review_budget(
    calibration_scores: np.ndarray, budget_frac: float = 0.005
) -> Threshold:
    """Flag the top ``budget_frac`` of an unlabelled calibration window.

    This is the operating point a real team actually sets: how many records can
    a reviewer clear in a day. It uses no labels, and because every model is
    held to the same alert volume, their precision and recall are directly
    comparable instead of being confounded by different flag rates.
    """
    quantile = float(np.quantile(calibration_scores, 1.0 - budget_frac))
    return Threshold(
        name=f"budget@{budget_frac:.2%}",
        value=quantile,
        rule=f"{(1 - budget_frac) * 100:.2f}th percentile of unlabelled calibration scores",
        expected_flag_rate=budget_frac,
    )


def threshold_from_train_percentile(
    train_scores: np.ndarray, percentile: float = 99.0
) -> Threshold:
    """Percentile of the training scores.

    Reported mainly to show the failure mode: the model has already minimised
    error on these rows, so the resulting cut-off is optimistically low and
    over-flags on data it has not seen.
    """
    value = float(np.percentile(train_scores, percentile))
    return Threshold(
        name=f"train_p{percentile:g}",
        value=value,
        rule=f"{percentile:g}th percentile of training reconstruction error",
        expected_flag_rate=1.0 - percentile / 100.0,
    )


def threshold_best_f1(scores: np.ndarray, y_true: np.ndarray) -> Threshold:
    """The cut-off maximising F1 on labelled data.

    Not achievable in production: it reads the answers. Kept as an upper bound
    so the honest operating point can be compared against the ceiling.
    """
    precision, recall, cuts = precision_recall_curve(y_true, scores)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    best = int(np.argmax(f1[:-1])) if len(cuts) else 0
    value = float(cuts[best]) if len(cuts) else float(scores.max())
    return Threshold(
        name="best_f1_oracle",
        value=value,
        rule="argmax F1 over labelled test scores (uses labels; not deployable)",
        expected_flag_rate=float((scores >= value).mean()),
        is_oracle=True,
    )
