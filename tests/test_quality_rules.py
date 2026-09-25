"""Each rule is asserted against a fixture with a known number of violations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dq_anomaly.quality.profile import ColumnExpectation, TableExpectation, check_expression
from dq_anomaly.quality.rules import (
    check_completeness,
    check_consistency,
    check_plausibility,
    check_uniqueness,
    check_validity,
)


def test_completeness_is_perfect_on_a_clean_batch(tiny_clean_df, tiny_profile):
    results = check_completeness(tiny_clean_df, tiny_profile)
    assert all(result.score == pytest.approx(1.0) for result in results)


def test_completeness_respects_declared_null_tolerance(tiny_clean_df):
    frame = tiny_clean_df.copy()
    frame.loc[0:3, "quantity"] = np.nan  # 4 of 40 rows = 10%
    tolerant = TableExpectation(name="t", columns=[
        ColumnExpectation(name="quantity", dtype="float", max_null_rate=0.10)])
    strict = TableExpectation(name="t", columns=[
        ColumnExpectation(name="quantity", dtype="float", max_null_rate=0.0)])

    assert check_completeness(frame, tolerant)[0].score == pytest.approx(1.0)
    assert check_completeness(frame, strict)[0].score == pytest.approx(0.9)


def test_missing_column_scores_zero_rather_than_being_skipped(tiny_clean_df, tiny_profile):
    frame = tiny_clean_df.drop(columns=["unit_price"])
    results = {result.rule_id: result for result in check_completeness(frame, tiny_profile)}
    assert results["expect_column_to_exist.unit_price"].score == 0.0


def test_range_rule_reports_the_exact_failing_rows(tiny_clean_df, tiny_profile):
    frame = tiny_clean_df.copy()
    frame.loc[[5, 11], "quantity"] = -1.0
    results = {r.rule_id: r for r in check_validity(frame, tiny_profile)}
    failing = results["expect_column_values_to_be_between.quantity"]
    assert failing.n_failed == 2
    assert set(failing.failing_index) == {5, 11}


def test_type_rule_catches_locale_formatted_numbers(tiny_clean_df, tiny_profile):
    frame = tiny_clean_df.copy()
    frame["quantity"] = frame["quantity"].astype(object)
    frame.loc[3, "quantity"] = "1.234,00"
    results = {r.rule_id: r for r in check_validity(frame, tiny_profile)}
    assert results["expect_column_values_to_be_of_type.quantity"].n_failed == 1


def test_allowed_values_flags_casing_drift(tiny_clean_df, tiny_profile):
    frame = tiny_clean_df.copy()
    frame.loc[0, "category"] = "a"
    results = {r.rule_id: r for r in check_validity(frame, tiny_profile)}
    assert results["expect_column_values_to_be_in_set.category"].n_failed == 1


def test_row_duplicates_and_key_duplicates_are_reported_separately(tiny_clean_df, tiny_profile):
    frame = pd.concat([tiny_clean_df, tiny_clean_df.iloc[[0]]], ignore_index=True)
    results = {r.rule_id: r for r in check_uniqueness(frame, tiny_profile)}
    assert results["expect_table_rows_to_be_unique"].n_failed == 1
    # Both copies of the colliding key are reported, not just the second.
    assert results["expect_primary_key_to_be_unique"].n_failed == 2


def test_consistency_ignores_rows_whose_operands_are_missing(tiny_clean_df, tiny_profile):
    frame = tiny_clean_df.copy()
    frame.loc[0, "unit_price"] = np.nan   # not a consistency failure
    frame.loc[1, "total"] = 999.0         # a real consistency failure
    results = check_consistency(frame, tiny_profile)
    assert results[0].n_failed == 1
    assert set(results[0].failing_index) == {1}


def test_robust_outlier_test_survives_heavy_contamination():
    clean = np.linspace(8.0, 12.0, 70)
    values = np.concatenate([clean, np.full(30, 1000.0)])
    frame = pd.DataFrame({"v": values})
    profile = TableExpectation(name="t", columns=[
        ColumnExpectation(name="v", dtype="float")], outlier_columns=["v"])
    result = check_plausibility(frame, profile, cut=4.0)[0]
    assert result.n_failed == 30

    # The same data under a mean/standard-deviation rule: the contamination
    # inflates the standard deviation enough to hide itself, which is precisely
    # why the rule is built on the median and MAD instead.
    z_naive = np.abs(values - values.mean()) / values.std(ddof=0)
    assert (z_naive > 4.0).sum() == 0


def test_outlier_rule_skips_a_constant_column_without_dividing_by_zero():
    frame = pd.DataFrame({"v": np.full(100, 5.0)})
    profile = TableExpectation(name="t", columns=[
        ColumnExpectation(name="v", dtype="float")], outlier_columns=["v"])
    assert check_plausibility(frame, profile) == []


def test_expression_allowlist_rejects_foreign_names_and_calls():
    with pytest.raises(ValueError):
        check_expression("a + evil", {"a"})
    with pytest.raises(ValueError):
        check_expression("__import__('os').system('ls')", {"a"})
    check_expression("abs(a - b) <= 0.1", {"a", "b", "abs"})
