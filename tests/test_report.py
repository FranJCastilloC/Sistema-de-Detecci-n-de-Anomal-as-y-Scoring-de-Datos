"""Issue severity, ordering and report serialisation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dq_anomaly.pipeline import AuditConfig, audit_batch
from dq_anomaly.quality.scorer import score_dataframe
from dq_anomaly.report.builder import build_batch_report, issues_from_anomalies
from dq_anomaly.report.issues import Issue, IssueSource, Severity, SeverityModel, prioritize


def _issue(issue_id: str, score: float, records: int, rule: str = "r") -> Issue:
    return Issue(
        issue_id=issue_id, source=IssueSource.QUALITY_RULE, dimension="validity",
        rule_id=rule, scope="column", table="t", column="c", record_keys=["k"],
        n_records=records, pct_affected=0.1, reason="", severity=Severity.LOW,
        severity_score=score, recommended_action="",
    )


def test_severity_bands_follow_the_configured_cut_offs():
    model = SeverityModel()
    assert model.band(80.0) is Severity.CRITICAL
    assert model.band(60.0) is Severity.HIGH
    assert model.band(30.0) is Severity.MEDIUM
    assert model.band(10.0) is Severity.LOW


def test_primary_key_violations_are_always_critical():
    model = SeverityModel()
    assert model.band(1.0, "expect_primary_key_to_be_unique") is Severity.CRITICAL


def test_extent_is_square_rooted_so_small_breaches_are_not_rounded_away():
    model = SeverityModel()
    small = model.score(impact=0.9, n_affected=100, n_rows=10_000,
                        dimension_weight=1.0, max_dimension_weight=1.0)
    linear_equivalent = 100 * (0.45 * 0.9 + 0.30 * 0.01 + 0.25 * 1.0)
    assert small > linear_equivalent


def test_priority_is_a_dense_sequence_ordered_by_severity():
    issues = prioritize([_issue("a", 10, 1), _issue("b", 90, 5), _issue("c", 50, 2)])
    assert [issue.priority for issue in issues] == [1, 2, 3]
    assert [issue.issue_id for issue in issues] == ["b", "c", "a"]


def test_priority_ordering_is_stable_across_runs():
    build = lambda: [_issue("a", 50, 3, "r2"), _issue("b", 50, 3, "r1"),
                     _issue("c", 50, 3, "r3")]
    first = [issue.issue_id for issue in prioritize(build())]
    second = [issue.issue_id for issue in prioritize(build())]
    assert first == second == ["b", "a", "c"]


def test_record_issues_are_capped_and_the_remainder_is_reported(tiny_clean_df, tiny_profile):
    scores = np.linspace(0, 1, len(tiny_clean_df))
    issues, summary = issues_from_anomalies(
        df=tiny_clean_df, scores=scores, threshold_value=0.0,
        threshold_rule="all", profile=tiny_profile, severity_model=SeverityModel(),
        table="t", model_name="test", max_issues=5,
    )
    truncation = [issue for issue in issues if issue.rule_id.endswith("truncated")]
    assert len(issues) == 6
    assert len(truncation) == 1
    # The count must never be silently lost.
    assert truncation[0].n_records == len(tiny_clean_df) - 5
    assert summary["n_flagged"] == len(tiny_clean_df)


def test_report_serialises_to_valid_json(tiny_clean_df, tiny_profile):
    quality = score_dataframe(tiny_clean_df, tiny_profile)
    report = build_batch_report(tiny_clean_df, quality, tiny_profile, "tiny", batch_id="b1")
    payload = json.loads(report.to_json())
    assert payload["schema_version"] == "1.0"
    assert payload["batch_id"] == "b1"
    assert payload["quality"]["grade"] == "A"


def test_markdown_report_contains_the_expected_sections(tiny_clean_df, tiny_profile):
    quality = score_dataframe(tiny_clean_df, tiny_profile)
    report = build_batch_report(tiny_clean_df, quality, tiny_profile, "tiny")
    markdown = report.to_markdown()
    for heading in ("# Data quality report", "## Quality score", "## Issues",
                    "## All rules evaluated"):
        assert heading in markdown


def test_report_writes_all_three_artifacts(tmp_path, tiny_clean_df, tiny_profile):
    quality = score_dataframe(tiny_clean_df, tiny_profile)
    report = build_batch_report(tiny_clean_df, quality, tiny_profile, "tiny")
    paths = report.write(tmp_path / "out")
    assert all(path.exists() for path in paths.values())


def test_audit_infers_a_profile_for_an_unrecognised_schema():
    """The quality-only path the app falls back to for an unknown schema."""
    frame = pd.DataFrame({
        "ticket": [f"T-{i}" for i in range(200)],
        "amount_local": np.linspace(1, 500, 200),
        "channel": ["web", "branch"] * 100,
    })
    result = audit_batch(frame, config=AuditConfig(use_model=False),
                         table_name="unknown_batch")
    assert result.scores is None
    assert 0 <= result.quality.global_score <= 100
    assert result.profile.inferred is True
    assert any("inferred" in note for note in result.notes)


def test_audit_auto_detects_a_bundled_profile_from_the_column_signature(erp_tables_small):
    result = audit_batch(erp_tables_small["po_lines"], config=AuditConfig(use_model=False),
                         table_name="po_lines_sample")
    assert result.profile.inferred is False
    assert result.profile.name == "erp_po_lines"


def test_audit_flags_injected_defects_in_the_erp_batch(erp_tables_small):
    from dq_anomaly.data.defect_injector import inject_defects
    from dq_anomaly.quality.profile import load_profile

    dirty, ledger = inject_defects(erp_tables_small, seed=99)
    result = audit_batch(
        dirty["po_lines"], profile=load_profile("profile_erp_po_lines.yaml"),
        reference_tables=dirty, table_name="erp_po_lines",
        config=AuditConfig(use_model=False),
    )
    flagged_rules = {rule.rule_id for rule in result.quality.rules if rule.n_failed}
    assert "expect_column_values_to_not_be_null.unit_price" in flagged_rules
    assert "consistency.material_id_exists_in_material_master" in flagged_rules
    assert len(result.report.issues) > 0
    assert all(issue.recommended_action for issue in result.report.issues)
