"""The issue model shared by the rule engine and the anomaly model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from dq_anomaly.config import load_yaml


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        return {"critical": 0, "high": 1, "medium": 2, "low": 3}[self.value]


class IssueSource(str, Enum):
    QUALITY_RULE = "quality_rule"
    ANOMALY_MODEL = "anomaly_model"


@dataclass
class Issue:
    """One finding: what is wrong, how bad it is, and what to do about it."""

    issue_id: str
    source: IssueSource
    dimension: str
    rule_id: str
    scope: str
    table: str
    column: str | None
    record_keys: list[str]
    n_records: int
    pct_affected: float
    reason: str
    severity: Severity
    severity_score: float
    recommended_action: str
    evidence: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    detected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict:
        return {
            "issue_id": self.issue_id,
            "priority": self.priority,
            "severity": self.severity.value,
            "severity_score": round(self.severity_score, 2),
            "source": self.source.value,
            "dimension": self.dimension,
            "rule_id": self.rule_id,
            "scope": self.scope,
            "table": self.table,
            "column": self.column,
            "n_records": self.n_records,
            "pct_affected": round(self.pct_affected, 6),
            "reason": self.reason,
            "recommended_action": self.recommended_action,
            "record_keys": self.record_keys,
            "evidence": self.evidence,
            "detected_at": self.detected_at,
        }


class SeverityModel:
    """Computes a 0-100 severity from rule impact, extent and confidence."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_yaml("severity.yaml")
        self.bands = self.config["bands"]
        self.weights = self.config["weights"]
        self.always_critical = set(self.config.get("always_critical", []))
        self.actions = self.config.get("actions", {})
        self.rule_actions = self.config.get("rule_actions", {})

    def score(
        self,
        impact: float,
        n_affected: int,
        n_rows: int,
        dimension_weight: float,
        max_dimension_weight: float,
        confidence: float = 1.0,
    ) -> float:
        # Extent is square-rooted on purpose: with a linear term every realistic
        # defect rate (a fraction of a percent) would round to "low", whereas a
        # 1% breach of a critical field plainly is not low.
        extent = math.sqrt(n_affected / n_rows) if n_rows else 0.0
        normalised_weight = (
            dimension_weight / max_dimension_weight if max_dimension_weight else 1.0
        )
        blended = (
            self.weights["impact"] * impact
            + self.weights["extent"] * extent
            + self.weights["confidence"] * confidence
        )
        return float(100.0 * max(0.2, normalised_weight) * blended)

    def band(self, score: float, rule_id: str = "") -> Severity:
        if rule_id in self.always_critical or rule_id.split(".")[0] in self.always_critical:
            return Severity.CRITICAL
        if score >= self.bands["critical"]:
            return Severity.CRITICAL
        if score >= self.bands["high"]:
            return Severity.HIGH
        if score >= self.bands["medium"]:
            return Severity.MEDIUM
        return Severity.LOW

    def action_for(self, rule_id: str) -> str:
        family, _, remainder = rule_id.partition(".")
        if family == "consistency" and remainder in self.rule_actions:
            return " ".join(self.rule_actions[remainder].split())
        if family in self.actions:
            return " ".join(self.actions[family].split())
        return " ".join(self.actions.get("anomaly", "Review manually.").split())


def prioritize(issues: list[Issue]) -> list[Issue]:
    """Rank issues and assign a dense 1..N priority.

    The sort key is fully deterministic, including the tie-breakers, so the same
    batch always produces the same register.
    """
    ordered = sorted(
        issues,
        key=lambda issue: (
            -issue.severity_score,
            -issue.n_records,
            issue.rule_id,
            issue.record_keys[0] if issue.record_keys else "",
        ),
    )
    for position, issue in enumerate(ordered, start=1):
        issue.priority = position
    return ordered
