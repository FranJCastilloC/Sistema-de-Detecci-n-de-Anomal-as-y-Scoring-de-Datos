"""Measure the quality engine itself against the injected defect ledger.

Most data quality tooling can only describe a batch. Because the synthetic ERP
extract ships with a ground-truth record of every corruption, the rules can be
scored the way a detector should be: recall per defect family, and the share of
flagged cells that were never actually corrupted.
"""

from __future__ import annotations

import pandas as pd

from dq_anomaly.data.defect_injector import PRIMARY_KEYS
from dq_anomaly.quality.profile import TableExpectation
from dq_anomaly.quality.rules import RuleResult

#: Outlier rules are statistical suspicions, not violations: a value flagged as
#: implausible was not necessarily corrupted. They are reported separately so
#: their false positives do not contaminate the deterministic rules' precision.
STATISTICAL_DIMENSIONS = {"plausibility"}


def flagged_cells(
    df: pd.DataFrame, rules: list[RuleResult], table: str
) -> pd.DataFrame:
    """Expand rule results into one row per (row_key, column) cell flagged."""
    key_column = PRIMARY_KEYS.get(table)
    records = []
    for rule in rules:
        if rule.n_failed == 0:
            continue
        index = rule.failing_index
        keys = (
            df.loc[index, key_column].astype(str).tolist()
            if key_column in df.columns
            else [str(value) for value in index]
        )
        for key in keys:
            records.append(
                {
                    "table": table,
                    "row_key": key,
                    "column": rule.column,
                    "rule_id": rule.rule_id,
                    "dimension": rule.dimension,
                }
            )
    if not records:
        return pd.DataFrame(columns=["table", "row_key", "column", "rule_id", "dimension"])
    return pd.DataFrame(records)


def evaluate_defect_recall(
    ledger: pd.DataFrame, findings: pd.DataFrame
) -> pd.DataFrame:
    """Per defect family: how many injected defects the rules recovered."""
    if ledger.empty:
        return pd.DataFrame()

    # A defect is recovered when some rule flagged that row, and either the rule
    # names the same column or it is a row-level rule (duplicates, cross-field).
    flagged_pairs = set(
        zip(findings["table"].astype(str), findings["row_key"].astype(str),
            findings["column"].astype("string").fillna("*"))
    )
    flagged_rows = set(zip(findings["table"].astype(str), findings["row_key"].astype(str)))

    rows = []
    for (table, code), group in ledger.groupby(["table", "defect_code"], sort=True):
        detected = 0
        for _, defect in group.iterrows():
            key = (str(defect["table"]), str(defect["row_key"]))
            column = defect["column"]
            if column is None or (isinstance(column, float) and pd.isna(column)):
                hit = key in flagged_rows
            else:
                hit = (key + (str(column),)) in flagged_pairs or (
                    key + ("*",)
                ) in flagged_pairs
            detected += int(hit)
        rows.append(
            {
                "table": table,
                "defect_code": code,
                "injected": len(group),
                "detected": detected,
                "recall": round(detected / len(group), 4),
            }
        )
    return pd.DataFrame(rows).sort_values(["table", "defect_code"]).reset_index(drop=True)


def evaluate_rule_precision(
    ledger: pd.DataFrame, findings: pd.DataFrame
) -> pd.DataFrame:
    """Per rule: how many flagged cells correspond to a known injected defect."""
    if findings.empty:
        return pd.DataFrame()
    corrupted = set(
        zip(ledger["table"].astype(str), ledger["row_key"].astype(str))
    )
    findings = findings.copy()
    findings["is_known_defect"] = [
        (str(table), str(key)) in corrupted
        for table, key in zip(findings["table"], findings["row_key"])
    ]
    summary = (
        findings.groupby(["table", "rule_id", "dimension"], sort=True)
        .agg(flagged=("is_known_defect", "size"), true_positives=("is_known_defect", "sum"))
        .reset_index()
    )
    summary["precision"] = (summary["true_positives"] / summary["flagged"]).round(4)
    summary["is_statistical"] = summary["dimension"].isin(STATISTICAL_DIMENSIONS)
    return summary.sort_values(["is_statistical", "precision"]).reset_index(drop=True)
