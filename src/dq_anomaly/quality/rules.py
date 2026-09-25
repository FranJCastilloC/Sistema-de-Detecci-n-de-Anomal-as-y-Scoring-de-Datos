"""The individual data quality checks.

Every check returns a :class:`RuleResult` carrying both a 0-1 score and the
exact index of the rows that failed. Keeping the failing index (not just a
count) is what lets the report quote real examples and lets the defect-recall
evaluation match findings against the injected ground truth cell by cell.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dq_anomaly.quality.profile import (
    ColumnExpectation,
    CrossFieldRule,
    TableExpectation,
    check_expression,
)

#: Minimum number of observations before a robust outlier test is meaningful.
MIN_OUTLIER_SAMPLE = 30


@dataclass
class RuleResult:
    """Outcome of a single check."""

    rule_id: str
    dimension: str
    column: str | None
    description: str
    n_checked: int
    n_failed: int
    score: float
    failing_index: pd.Index
    impact: float = 0.5
    weight: float = 1.0
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.n_failed == 0

    @property
    def fail_rate(self) -> float:
        return 0.0 if self.n_checked == 0 else self.n_failed / self.n_checked


def _empty_index() -> pd.Index:
    return pd.Index([], dtype="int64")


def _result(
    rule_id: str, dimension: str, column: str | None, description: str,
    n_checked: int, failing: pd.Index, impact: float, weight: float = 1.0,
    score: float | None = None, **detail: Any,
) -> RuleResult:
    n_failed = int(len(failing))
    if score is None:
        score = 1.0 if n_checked == 0 else 1.0 - n_failed / n_checked
    return RuleResult(
        rule_id=rule_id, dimension=dimension, column=column, description=description,
        n_checked=int(n_checked), n_failed=n_failed, score=float(np.clip(score, 0.0, 1.0)),
        failing_index=failing, impact=impact, weight=weight, detail=detail,
    )


def _coerce(series: pd.Series, dtype: str) -> pd.Series:
    """Coerce to the expected dtype, turning unparseable values into NaT/NaN."""
    if dtype in {"int", "float"}:
        return pd.to_numeric(series, errors="coerce")
    if dtype == "datetime":
        return pd.to_datetime(series, errors="coerce")
    if dtype == "bool":
        mapping = {"true": True, "false": False, "1": True, "0": False,
                   "yes": True, "no": False}
        if pd.api.types.is_bool_dtype(series):
            return series
        lowered = series.astype("string").str.strip().str.lower()
        return lowered.map(mapping).astype("object")
    return series


# --- completeness ---------------------------------------------------------------

def check_completeness(df: pd.DataFrame, profile: TableExpectation) -> list[RuleResult]:
    """Penalise nulls beyond each column's declared tolerance."""
    results = []
    for expectation in profile.columns:
        if expectation.name not in df.columns:
            results.append(_result(
                f"expect_column_to_exist.{expectation.name}", "completeness",
                expectation.name, f"Column '{expectation.name}' is missing from the batch",
                len(df), df.index, impact=0.9, weight=expectation.weight, score=0.0,
            ))
            continue
        series = df[expectation.name]
        failing = df.index[series.isna()]
        null_rate = float(len(failing) / len(df)) if len(df) else 0.0
        tolerance = expectation.max_null_rate
        # A column declared 5%-nullable scores a clean 1.0 at 5% nulls.
        excess = max(0.0, null_rate - tolerance)
        score = 1.0 - excess / max(1e-9, 1.0 - tolerance)
        results.append(_result(
            f"expect_column_values_to_not_be_null.{expectation.name}", "completeness",
            expectation.name,
            f"'{expectation.name}' is {null_rate:.2%} null (tolerance {tolerance:.2%})",
            len(df), failing, impact=0.9 if expectation.required else 0.4,
            weight=expectation.weight, score=score,
            null_rate=null_rate, tolerance=tolerance,
        ))
    return results


# --- validity -------------------------------------------------------------------

def check_validity(df: pd.DataFrame, profile: TableExpectation) -> list[RuleResult]:
    """Type parseability, numeric ranges, controlled vocabularies and patterns."""
    results: list[RuleResult] = []
    for expectation in profile.columns:
        if expectation.name not in df.columns:
            continue
        series = df[expectation.name]
        observed = series.notna()
        n_observed = int(observed.sum())
        if n_observed == 0:
            continue
        results.extend(_type_rule(df, series, observed, n_observed, expectation))
        results.extend(_range_rules(df, series, expectation))
        results.extend(_date_range_rule(df, series, expectation))
        results.extend(_vocabulary_rule(df, series, observed, n_observed, expectation))
        results.extend(_regex_rule(df, series, observed, n_observed, expectation))
    return results


def _type_rule(df, series, observed, n_observed, expectation: ColumnExpectation):
    if expectation.dtype == "string":
        return []
    coerced = _coerce(series, expectation.dtype)
    unparseable = observed & coerced.isna()
    return [_result(
        f"expect_column_values_to_be_of_type.{expectation.name}", "validity",
        expectation.name,
        f"'{expectation.name}' must parse as {expectation.dtype}",
        n_observed, df.index[unparseable], impact=0.7, weight=expectation.weight,
        expected_dtype=expectation.dtype,
    )]


def _range_rules(df, series, expectation: ColumnExpectation):
    if expectation.min is None and expectation.max is None:
        return []
    numeric = _coerce(series, "float") if expectation.dtype != "datetime" else None
    if numeric is None:
        return []
    valid = numeric.notna()
    n_valid = int(valid.sum())
    if n_valid == 0:
        return []
    below = valid & (numeric < expectation.min) if expectation.min is not None else pd.Series(False, index=series.index)
    above = valid & (numeric > expectation.max) if expectation.max is not None else pd.Series(False, index=series.index)
    failing = df.index[below | above]
    bounds = f"[{expectation.min}, {expectation.max}]"
    return [_result(
        f"expect_column_values_to_be_between.{expectation.name}", "validity",
        expectation.name, f"'{expectation.name}' must fall within {bounds}",
        n_valid, failing, impact=0.75, weight=expectation.weight,
        min=expectation.min, max=expectation.max,
    )]


def _resolve_date(token: str) -> pd.Timestamp:
    """Resolve a profile date bound, including the relative token ``today``."""
    if token == "today":
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(token)


def _date_range_rule(df, series, expectation: ColumnExpectation):
    if expectation.min_date is None and expectation.max_date is None:
        return []
    parsed = pd.to_datetime(series, errors="coerce")
    valid = parsed.notna()
    n_valid = int(valid.sum())
    if n_valid == 0:
        return []
    failing_mask = pd.Series(False, index=series.index)
    if expectation.min_date is not None:
        failing_mask |= valid & (parsed < _resolve_date(expectation.min_date))
    if expectation.max_date is not None:
        failing_mask |= valid & (parsed > _resolve_date(expectation.max_date))
    bounds = f"[{expectation.min_date}, {expectation.max_date}]"
    return [_result(
        f"expect_column_dates_to_be_between.{expectation.name}", "validity",
        expectation.name, f"'{expectation.name}' must fall within {bounds}",
        n_valid, df.index[failing_mask], impact=0.7, weight=expectation.weight,
        min_date=expectation.min_date, max_date=expectation.max_date,
    )]


def _vocabulary_rule(df, series, observed, n_observed, expectation: ColumnExpectation):
    if not expectation.allowed_values:
        return []
    allowed = set(expectation.allowed_values)
    as_text = series.astype("string")
    failing = df.index[observed & ~as_text.isin(allowed)]
    return [_result(
        f"expect_column_values_to_be_in_set.{expectation.name}", "validity",
        expectation.name,
        f"'{expectation.name}' must be one of {sorted(allowed)}",
        n_observed, failing, impact=0.55, weight=expectation.weight,
        allowed_values=sorted(allowed),
    )]


def _regex_rule(df, series, observed, n_observed, expectation: ColumnExpectation):
    if not expectation.regex:
        return []
    pattern = re.compile(expectation.regex)
    as_text = series.astype("string")
    matches = as_text.str.fullmatch(pattern).fillna(False).astype(bool)
    failing = df.index[observed & ~matches]
    return [_result(
        f"expect_column_values_to_match_regex.{expectation.name}", "validity",
        expectation.name, f"'{expectation.name}' must match /{expectation.regex}/",
        n_observed, failing, impact=0.6, weight=expectation.weight,
        regex=expectation.regex,
    )]


# --- uniqueness -----------------------------------------------------------------

def check_uniqueness(df: pd.DataFrame, profile: TableExpectation) -> list[RuleResult]:
    """Exact row duplicates and primary-key collisions, reported separately."""
    results = []
    duplicated_rows = df.duplicated(keep="first")
    results.append(_result(
        "expect_table_rows_to_be_unique", "uniqueness", None,
        "Rows must not be exact duplicates of one another",
        len(df), df.index[duplicated_rows], impact=0.7,
    ))

    key_columns = [column for column in profile.primary_key if column in df.columns]
    if key_columns:
        duplicated_keys = df.duplicated(subset=key_columns, keep=False)
        results.append(_result(
            "expect_primary_key_to_be_unique", "uniqueness", ",".join(key_columns),
            f"Primary key {key_columns} must identify at most one row",
            len(df), df.index[duplicated_keys], impact=1.0, weight=2.0,
            primary_key=key_columns,
        ))

    for expectation in profile.columns:
        if not expectation.unique or expectation.name not in df.columns:
            continue
        series = df[expectation.name]
        duplicated = series.duplicated(keep=False) & series.notna()
        results.append(_result(
            f"expect_column_values_to_be_unique.{expectation.name}", "uniqueness",
            expectation.name, f"'{expectation.name}' must be unique",
            len(df), df.index[duplicated], impact=0.8, weight=expectation.weight,
        ))
    return results


# --- consistency ----------------------------------------------------------------

def check_consistency(
    df: pd.DataFrame,
    profile: TableExpectation,
    reference_tables: dict[str, pd.DataFrame] | None = None,
) -> list[RuleResult]:
    """Cross-column arithmetic, date ordering, and cross-table referential rules."""
    reference_tables = reference_tables or {}
    results: list[RuleResult] = []
    known_names = set(profile.column_map) | {"abs", "round"}

    for rule in profile.cross_field_rules:
        if rule.kind == "expression":
            results.extend(_expression_rule(df, rule, known_names))
        elif rule.kind == "foreign_key":
            results.extend(_foreign_key_rule(df, rule, reference_tables))
        elif rule.kind == "lookup_match":
            results.extend(_lookup_match_rule(df, rule, reference_tables))
    return results


def _expression_rule(df: pd.DataFrame, rule: CrossFieldRule, known_names: set[str]):
    tree = check_expression(rule.expression or "", known_names, rule.id)
    referenced = {node.id for node in ast.walk(tree)
                  if isinstance(node, ast.Name)} - {"abs", "round"}
    if not referenced.issubset(set(df.columns)):
        return []

    namespace = {column: df[column] for column in referenced}
    # Numeric comparisons need real numbers: a column holding "1.234,00" text
    # would otherwise raise instead of being reported as a consistency failure.
    for name, series in list(namespace.items()):
        if not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series):
            coerced = pd.to_numeric(series, errors="coerce")
            if coerced.notna().any():
                namespace[name] = coerced

    # A row missing one of the operands is not a consistency failure: the
    # completeness and type rules already report it. Evaluating it anyway would
    # double-count the same defect, because NaN comparisons return False rather
    # than propagating as missing.
    evaluable = pd.Series(True, index=df.index)
    for series in namespace.values():
        evaluable &= series.notna()
    namespace.update({"abs": abs, "round": round})

    outcome = eval(compile(tree, f"<rule:{rule.id}>", "eval"), {"__builtins__": {}}, namespace)
    outcome = pd.Series(outcome, index=df.index).astype("boolean")
    failing = df.index[evaluable & ~outcome.fillna(True).astype(bool)]
    return [_result(
        f"consistency.{rule.id}", "consistency", None, rule.description,
        int(evaluable.sum()), failing, impact=rule.impact,
        expression=rule.expression,
    )]


def _foreign_key_rule(df, rule: CrossFieldRule, reference_tables):
    reference = reference_tables.get(rule.ref_table or "")
    if reference is None or rule.column not in df.columns or rule.ref_column not in reference.columns:
        return []
    series = df[rule.column]
    observed = series.notna()
    known = set(reference[rule.ref_column].dropna().astype(str))
    failing = df.index[observed & ~series.astype(str).isin(known)]
    return [_result(
        f"consistency.{rule.id}", "consistency", rule.column, rule.description,
        int(observed.sum()), failing, impact=rule.impact,
        ref_table=rule.ref_table, ref_column=rule.ref_column,
    )]


def _lookup_match_rule(df, rule: CrossFieldRule, reference_tables):
    reference = reference_tables.get(rule.ref_table or "")
    if reference is None:
        return []
    needed = {rule.column, rule.local_key}
    if not needed.issubset(set(df.columns)):
        return []
    if not {rule.ref_key, rule.ref_column}.issubset(set(reference.columns)):
        return []
    lookup = (
        reference[[rule.ref_key, rule.ref_column]]
        .dropna(subset=[rule.ref_key])
        .drop_duplicates(subset=[rule.ref_key])
        .set_index(rule.ref_key)[rule.ref_column]
    )
    expected = df[rule.local_key].astype(str).map(lookup.rename(index=str))
    observed = df[rule.column].notna() & expected.notna()
    mismatched = observed & (df[rule.column].astype(str) != expected.astype(str))
    return [_result(
        f"consistency.{rule.id}", "consistency", rule.column, rule.description,
        int(observed.sum()), df.index[mismatched], impact=rule.impact,
        ref_table=rule.ref_table, ref_column=rule.ref_column,
    )]


# --- plausibility ---------------------------------------------------------------

def check_plausibility(
    df: pd.DataFrame, profile: TableExpectation, cut: float | None = None
) -> list[RuleResult]:
    """Flag statistically implausible values with a robust z-score.

    Median and MAD are used rather than mean and standard deviation because the
    latter are themselves inflated by the very outliers being hunted; MAD
    tolerates up to 50% contamination before it breaks down.
    """
    cut = cut if cut is not None else profile.outlier_cut
    results = []
    for column in profile.outlier_columns:
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        valid = numeric.notna()
        n_valid = int(valid.sum())
        if n_valid < MIN_OUTLIER_SAMPLE:
            continue
        values = numeric[valid]
        # Money and quantity columns are typically log-normal. A robust z-score
        # on the raw scale would flag the entire right tail of such a column as
        # implausible, which is a property of the distribution rather than a
        # data defect, so a strongly right-skewed non-negative column is tested
        # on the log scale instead.
        log_scaled = False
        if (values >= 0).all() and len(values) > 3 and float(values.skew()) > 2.0:
            numeric = np.log1p(numeric)
            values = numeric[valid]
            log_scaled = True
        median = float(values.median())
        mad = float((values - median).abs().median())
        scale = 1.4826 * mad
        if scale <= 0:
            # A constant-ish column: fall back to the IQR so a degenerate MAD
            # does not silently divide by zero.
            q1, q3 = values.quantile([0.25, 0.75])
            scale = float(q3 - q1) / 1.349
        if scale <= 0:
            continue
        robust_z = (numeric - median).abs() / scale
        failing = df.index[valid & (robust_z > cut)]
        results.append(_result(
            f"expect_column_values_to_be_plausible.{column}", "plausibility", column,
            f"'{column}' values beyond {cut} robust standard deviations of the median",
            n_valid, failing, impact=0.35,
            median=median, robust_scale=scale, cut=cut, log_scaled=log_scaled,
        ))
    return results


def run_all_rules(
    df: pd.DataFrame,
    profile: TableExpectation,
    reference_tables: dict[str, pd.DataFrame] | None = None,
) -> list[RuleResult]:
    """Run every dimension's checks against a batch."""
    return [
        *check_completeness(df, profile),
        *check_validity(df, profile),
        *check_uniqueness(df, profile),
        *check_consistency(df, profile, reference_tables),
        *check_plausibility(df, profile),
    ]
