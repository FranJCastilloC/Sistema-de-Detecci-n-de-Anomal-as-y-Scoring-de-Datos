"""Assemble a single batch report from quality rules and model flags."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dq_anomaly.quality.profile import TableExpectation
from dq_anomaly.quality.rules import RuleResult
from dq_anomaly.quality.scorer import QualityReport
from dq_anomaly.report.issues import Issue, IssueSource, Severity, SeverityModel, prioritize

#: Rules scoring at or above this are treated as satisfied; a completeness rule
#: with an explicit null tolerance can fail rows while still being within spec.
PASSING_SCORE = 0.999
MAX_SAMPLE_KEYS = 10
DEFAULT_MAX_RECORD_ISSUES = 200


@dataclass
class BatchReport:
    """Everything a reviewer needs for one batch of data."""

    batch_id: str
    table: str
    n_rows: int
    quality: QualityReport
    issues: list[Issue]
    anomaly_summary: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def severity_counts(self) -> dict[str, int]:
        counts = {severity.value: 0 for severity in Severity}
        for issue in self.issues:
            counts[issue.severity.value] += 1
        return counts

    def issues_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(
                columns=["priority", "severity", "severity_score", "dimension",
                         "rule_id", "column", "n_records", "reason", "recommended_action"]
            )
        return pd.DataFrame([issue.to_dict() for issue in self.issues])

    def to_dict(self) -> dict:
        return {
            "schema_version": "1.0",
            "batch_id": self.batch_id,
            "table": self.table,
            "n_rows": self.n_rows,
            "generated_at": self.generated_at,
            "quality": self.quality.to_dict(),
            "anomaly": self.anomaly_summary,
            "severity_counts": self.severity_counts,
            "n_issues": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self, top_n: int = 25) -> str:
        quality = self.quality
        lines = [
            f"# Data quality report - {self.table}",
            "",
            f"- **Batch**: `{self.batch_id}`",
            f"- **Rows**: {self.n_rows:,}  |  **Columns**: {quality.n_columns}",
            f"- **Generated**: {self.generated_at}",
            f"- **Profile**: {'inferred from the batch' if quality.profile_inferred else 'curated expectation suite'}",
            "",
            "## Quality score",
            "",
            f"**{quality.global_score:.1f} / 100 (grade {quality.grade})**",
            "",
            f"Weighted geometric mean across dimensions. The arithmetic mean would "
            f"read {quality.global_score_arithmetic:.1f}; the gap is the penalty for "
            f"uneven quality across dimensions.",
            "",
            "| Dimension | Score | Weight | Rules | Failed |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, dimension in quality.dimension_scores.items():
            lines.append(
                f"| {name} | {dimension.score * 100:.1f} | {dimension.weight:.2f} | "
                f"{len(dimension.rules)} | {dimension.n_failed_rules} |"
            )

        if self.anomaly_summary:
            summary = self.anomaly_summary
            lines += [
                "",
                "## Anomaly detection",
                "",
                f"- Model: **{summary.get('model', 'n/a')}**",
                f"- Records flagged: **{summary.get('n_flagged', 0):,}** "
                f"({summary.get('flag_rate', 0):.2%} of the batch)",
                f"- Threshold: {summary.get('threshold_value', float('nan')):.6g} "
                f"({summary.get('threshold_rule', 'n/a')})",
            ]
            if summary.get("n_labelled_positives") is not None:
                lines.append(
                    f"- Known positives in batch: {summary['n_labelled_positives']}, "
                    f"of which {summary.get('n_caught', 0)} were flagged"
                )

        counts = self.severity_counts
        lines += [
            "",
            "## Issues",
            "",
            f"{len(self.issues)} open issues: "
            f"{counts['critical']} critical, {counts['high']} high, "
            f"{counts['medium']} medium, {counts['low']} low.",
            "",
            f"### Top {min(top_n, len(self.issues))} by priority",
            "",
            "| # | Severity | Dimension | Rule | Records | Reason | Recommended action |",
            "| ---: | --- | --- | --- | ---: | --- | --- |",
        ]
        for issue in self.issues[:top_n]:
            reason = issue.reason.replace("|", "/")
            action = issue.recommended_action.replace("|", "/")
            lines.append(
                f"| {issue.priority} | {issue.severity.value} | {issue.dimension} | "
                f"`{issue.rule_id}` | {issue.n_records:,} | {reason} | {action} |"
            )

        lines += ["", "## All rules evaluated", "",
                  "| Rule | Dimension | Checked | Failed | Score |",
                  "| --- | --- | ---: | ---: | ---: |"]
        for rule in sorted(quality.rules, key=lambda r: (r.dimension, r.rule_id)):
            lines.append(
                f"| `{rule.rule_id}` | {rule.dimension} | {rule.n_checked:,} | "
                f"{rule.n_failed:,} | {rule.score:.4f} |"
            )
        return "\n".join(lines) + "\n"

    def write(self, out_dir: Path, top_n: int = 25) -> dict[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "json": out_dir / "report.json",
            "markdown": out_dir / "report.md",
            "issues": out_dir / "issues.csv",
        }
        paths["json"].write_text(self.to_json(), encoding="utf-8")
        paths["markdown"].write_text(self.to_markdown(top_n=top_n), encoding="utf-8")
        self.issues_frame().to_csv(paths["issues"], index=False)
        return paths


def _sample_keys(df: pd.DataFrame, index: pd.Index, key_columns: list[str]) -> list[str]:
    sample = index[:MAX_SAMPLE_KEYS]
    if key_columns and all(column in df.columns for column in key_columns):
        return [
            "|".join(str(value) for value in row)
            for row in df.loc[sample, key_columns].to_numpy()
        ]
    return [str(value) for value in sample]


def issues_from_quality(
    df: pd.DataFrame,
    quality: QualityReport,
    profile: TableExpectation,
    severity_model: SeverityModel,
    table: str,
) -> list[Issue]:
    """One issue per failing rule, describing the breach and the fix."""
    weights = profile.dimension_weights
    max_weight = max(weights.values()) if weights else 1.0
    issues: list[Issue] = []

    for counter, rule in enumerate(
        sorted(quality.rules, key=lambda r: r.rule_id), start=1
    ):
        if rule.n_failed == 0 or rule.score >= PASSING_SCORE:
            continue
        pct = rule.n_failed / max(1, rule.n_checked)
        score = severity_model.score(
            impact=rule.impact,
            n_affected=rule.n_failed,
            n_rows=max(1, rule.n_checked),
            dimension_weight=weights.get(rule.dimension, 0.1),
            max_dimension_weight=max_weight,
            confidence=0.6 if rule.dimension == "plausibility" else 1.0,
        )
        issues.append(Issue(
            issue_id=f"ISS-Q{counter:04d}",
            source=IssueSource.QUALITY_RULE,
            dimension=rule.dimension,
            rule_id=rule.rule_id,
            scope="column" if rule.column else "dataset",
            table=table,
            column=rule.column,
            record_keys=_sample_keys(df, rule.failing_index, profile.primary_key),
            n_records=rule.n_failed,
            pct_affected=pct,
            reason=f"{rule.description}. {rule.n_failed:,} of {rule.n_checked:,} "
                   f"rows checked failed ({pct:.2%}).",
            severity=severity_model.band(score, rule.rule_id),
            severity_score=score,
            recommended_action=severity_model.action_for(rule.rule_id),
            evidence={"rule_score": round(rule.score, 6), **{
                key: value for key, value in rule.detail.items()
                if isinstance(value, (int, float, str, list, bool, type(None)))
            }},
        ))
    return issues


def issues_from_anomalies(
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold_value: float,
    threshold_rule: str,
    profile: TableExpectation,
    severity_model: SeverityModel,
    table: str,
    model_name: str,
    per_feature_errors: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    feature_values: np.ndarray | None = None,
    reconstructions: np.ndarray | None = None,
    max_issues: int = DEFAULT_MAX_RECORD_ISSUES,
) -> tuple[list[Issue], dict[str, Any]]:
    """One issue per flagged record, explained by its worst-reconstructed fields."""
    scores = np.asarray(scores, dtype=float)
    flagged = np.flatnonzero(scores >= threshold_value)
    percentiles = pd.Series(scores).rank(pct=True).to_numpy()
    threshold_pct = float((scores < threshold_value).mean())
    median_score = float(np.median(scores)) or 1e-12

    ordered = flagged[np.argsort(-scores[flagged], kind="mergesort")]
    issues: list[Issue] = []
    for counter, position in enumerate(ordered[:max_issues], start=1):
        confidence = float(np.clip(
            (percentiles[position] - threshold_pct) / max(1e-9, 1 - threshold_pct), 0.0, 1.0
        ))
        severity_score = severity_model.score(
            impact=0.60, n_affected=1, n_rows=max(1, len(scores)),
            dimension_weight=1.0, max_dimension_weight=1.0, confidence=confidence,
        )
        # A single record is a tiny share of the batch, so the extent term stays
        # near zero; severity is driven by how far past the threshold it sits.
        severity_score = max(severity_score, 100 * (0.45 * 0.60 + 0.25 * confidence))

        reason = (
            f"Anomaly score {scores[position]:.5g} "
            f"({percentiles[position]:.4%} percentile, "
            f"{scores[position] / median_score:.1f}x the batch median)."
        )
        evidence: dict[str, Any] = {
            "score": float(scores[position]),
            "percentile": float(percentiles[position]),
            "threshold": float(threshold_value),
        }
        if per_feature_errors is not None and feature_names:
            errors = per_feature_errors[position]
            top = np.argsort(-errors)[:3]
            fragments = []
            top_features = []
            for feature_index in top:
                name = feature_names[feature_index]
                entry: dict[str, Any] = {
                    "feature": name,
                    "squared_error": float(errors[feature_index]),
                    "share_of_error": float(errors[feature_index] / max(errors.sum(), 1e-12)),
                }
                if feature_values is not None and reconstructions is not None:
                    observed = float(feature_values[position, feature_index])
                    expected = float(reconstructions[position, feature_index])
                    entry.update({"observed": observed, "expected": expected})
                    fragments.append(f"{name} (observed {observed:.2f} vs expected {expected:.2f})")
                else:
                    fragments.append(name)
                top_features.append(entry)
            evidence["top_features"] = top_features
            reason += " Largest deviations: " + ", ".join(fragments) + "."

        issues.append(Issue(
            issue_id=f"ISS-A{counter:04d}",
            source=IssueSource.ANOMALY_MODEL,
            dimension="anomaly",
            rule_id=f"anomaly.{model_name}",
            scope="record",
            table=table,
            column=None,
            record_keys=_sample_keys(df, df.index[[position]], profile.primary_key),
            n_records=1,
            pct_affected=1 / max(1, len(scores)),
            reason=reason,
            severity=severity_model.band(severity_score),
            severity_score=severity_score,
            recommended_action=severity_model.action_for("anomaly"),
            evidence=evidence,
        ))

    if len(ordered) > max_issues:
        issues.append(Issue(
            issue_id="ISS-A0000",
            source=IssueSource.ANOMALY_MODEL,
            dimension="anomaly",
            rule_id=f"anomaly.{model_name}.truncated",
            scope="dataset",
            table=table,
            column=None,
            record_keys=[],
            n_records=int(len(ordered) - max_issues),
            pct_affected=float((len(ordered) - max_issues) / max(1, len(scores))),
            reason=f"{len(ordered) - max_issues:,} further records exceeded the anomaly "
                   f"threshold but were not listed individually (register capped at "
                   f"{max_issues}).",
            severity=Severity.MEDIUM,
            severity_score=40.0,
            recommended_action="Increase the register cap or tighten the review budget "
                               "to bring the alert volume within review capacity.",
            evidence={"n_truncated": int(len(ordered) - max_issues)},
        ))

    summary = {
        "model": model_name,
        "n_flagged": int(len(flagged)),
        "flag_rate": float(len(flagged) / max(1, len(scores))),
        "threshold_value": float(threshold_value),
        "threshold_rule": threshold_rule,
        "score_median": median_score,
        "score_max": float(scores.max()) if len(scores) else 0.0,
    }
    return issues, summary


def build_batch_report(
    df: pd.DataFrame,
    quality: QualityReport,
    profile: TableExpectation,
    table: str,
    batch_id: str | None = None,
    anomaly_issues: list[Issue] | None = None,
    anomaly_summary: dict[str, Any] | None = None,
    severity_model: SeverityModel | None = None,
) -> BatchReport:
    """Combine quality findings and model flags into one prioritized register."""
    severity_model = severity_model or SeverityModel()
    issues = issues_from_quality(df, quality, profile, severity_model, table)
    if anomaly_issues:
        issues.extend(anomaly_issues)
    return BatchReport(
        batch_id=batch_id or datetime.now(timezone.utc).strftime("batch-%Y%m%dT%H%M%S"),
        table=table,
        n_rows=int(len(df)),
        quality=quality,
        issues=prioritize(issues),
        anomaly_summary=anomaly_summary or {},
    )
