"""Aggregation of rule results into a single batch quality score."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from dq_anomaly.quality.profile import TableExpectation
from dq_anomaly.quality.rules import RuleResult, run_all_rules

DIMENSIONS = ("completeness", "validity", "consistency", "uniqueness", "plausibility")

#: Floor applied before taking logs, so a fully failed dimension does not send
#: the geometric mean to zero and erase the information in the other four.
SCORE_FLOOR = 0.01

GRADE_BANDS = ((90.0, "A"), (80.0, "B"), (70.0, "C"), (60.0, "D"))


@dataclass
class DimensionScore:
    dimension: str
    score: float
    weight: float
    rules: list[RuleResult] = field(default_factory=list)

    @property
    def n_failed_rules(self) -> int:
        return sum(1 for rule in self.rules if not rule.passed)


@dataclass
class QualityReport:
    """The quality verdict for one batch."""

    table: str
    n_rows: int
    n_columns: int
    global_score: float
    global_score_arithmetic: float
    grade: str
    dimension_scores: dict[str, DimensionScore]
    rules: list[RuleResult]
    profile_inferred: bool
    generated_at: str

    @property
    def failed_rules(self) -> list[RuleResult]:
        return [rule for rule in self.rules if not rule.passed]

    def dimension_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "dimension": name,
                    "score": round(score.score * 100, 2),
                    "weight": score.weight,
                    "rules": len(score.rules),
                    "failed_rules": score.n_failed_rules,
                }
                for name, score in self.dimension_scores.items()
            ]
        )

    def to_dict(self) -> dict:
        return {
            "table": self.table,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "global_score": round(self.global_score, 2),
            "global_score_arithmetic": round(self.global_score_arithmetic, 2),
            "grade": self.grade,
            "profile_inferred": self.profile_inferred,
            "generated_at": self.generated_at,
            "dimensions": {
                name: {
                    "score": round(score.score * 100, 2),
                    "weight": score.weight,
                    "rules_evaluated": len(score.rules),
                    "rules_failed": score.n_failed_rules,
                }
                for name, score in self.dimension_scores.items()
            },
        }


def aggregate_geometric(
    dimension_scores: dict[str, float], weights: dict[str, float], floor: float = SCORE_FLOOR
) -> float:
    """Weighted geometric mean of dimension scores, expressed on a 0-100 scale.

    Quality dimensions are not substitutable: a batch whose primary keys are 40%
    duplicated is unusable no matter how complete it is. An arithmetic mean lets
    four healthy dimensions hide one dead one; the geometric mean is dominated
    by the weakest dimension, which is the behaviour a quality gate needs.
    """
    usable = {name: score for name, score in dimension_scores.items() if name in weights}
    total_weight = sum(weights[name] for name in usable)
    if total_weight <= 0:
        return 0.0
    log_sum = sum(
        weights[name] * np.log(max(score, floor)) for name, score in usable.items()
    )
    return float(100.0 * np.exp(log_sum / total_weight))


def aggregate_arithmetic(
    dimension_scores: dict[str, float], weights: dict[str, float]
) -> float:
    """Weighted arithmetic mean, reported alongside for contrast."""
    usable = {name: score for name, score in dimension_scores.items() if name in weights}
    total_weight = sum(weights[name] for name in usable)
    if total_weight <= 0:
        return 0.0
    return float(
        100.0 * sum(weights[name] * score for name, score in usable.items()) / total_weight
    )


def to_grade(
    score: float, dimension_scores: dict[str, float], primary_key_ok: bool = True
) -> str:
    """Letter grade, with two hard overrides.

    A broken primary key is not a gradeable condition, and a single collapsed
    dimension caps the grade no matter how high the weighted score lands.
    """
    if not primary_key_ok:
        return "F"
    grade = "F"
    for threshold, letter in GRADE_BANDS:
        if score >= threshold:
            grade = letter
            break
    if dimension_scores and min(dimension_scores.values()) < 0.50:
        order = ["A", "B", "C", "D", "F"]
        grade = order[max(order.index(grade), order.index("D"))]
    return grade


def score_dataframe(
    df: pd.DataFrame,
    profile: TableExpectation,
    reference_tables: dict[str, pd.DataFrame] | None = None,
) -> QualityReport:
    """Run every rule and aggregate the outcome into a batch score."""
    rules = run_all_rules(df, profile, reference_tables)
    weights = {name: profile.dimension_weights.get(name, 0.0) for name in DIMENSIONS}

    dimension_scores: dict[str, DimensionScore] = {}
    raw_scores: dict[str, float] = {}
    for dimension in DIMENSIONS:
        members = [rule for rule in rules if rule.dimension == dimension]
        if not members:
            # A dimension with nothing to check must not be scored as perfect;
            # it is excluded from the aggregate entirely.
            dimension_scores[dimension] = DimensionScore(dimension, 1.0, 0.0, [])
            continue
        total_weight = sum(rule.weight for rule in members)
        score = sum(rule.score * rule.weight for rule in members) / total_weight
        dimension_scores[dimension] = DimensionScore(
            dimension, float(score), weights.get(dimension, 0.0), members
        )
        raw_scores[dimension] = float(score)

    active_weights = {name: weights[name] for name in raw_scores if weights.get(name, 0) > 0}
    global_score = aggregate_geometric(raw_scores, active_weights)
    arithmetic = aggregate_arithmetic(raw_scores, active_weights)

    primary_key_ok = all(
        rule.passed for rule in rules if rule.rule_id == "expect_primary_key_to_be_unique"
    )
    grade = to_grade(global_score, raw_scores, primary_key_ok)

    return QualityReport(
        table=profile.name,
        n_rows=int(len(df)),
        n_columns=int(df.shape[1]),
        global_score=global_score,
        global_score_arithmetic=arithmetic,
        grade=grade,
        dimension_scores=dimension_scores,
        rules=rules,
        profile_inferred=profile.inferred,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
