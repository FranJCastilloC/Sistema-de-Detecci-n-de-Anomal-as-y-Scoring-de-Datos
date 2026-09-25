"""Shared fixtures. Nothing here touches the network or the large dataset."""

from __future__ import annotations

import sys
from pathlib import Path

# See scripts/_bootstrap.py: a macOS-hidden .pth silently disables the editable
# install, so the suite puts src/ on the path itself rather than relying on it.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import pytest

from dq_anomaly.data.erp_generator import ERPVolumes, generate_erp_dataset
from dq_anomaly.quality.profile import ColumnExpectation, CrossFieldRule, TableExpectation


@pytest.fixture
def tiny_clean_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"R-{i:03d}" for i in range(1, 41)],
            "category": ["A", "B", "C", "D"] * 10,
            "quantity": np.arange(1, 41, dtype="float64"),
            "unit_price": np.full(40, 2.0),
            "total": np.arange(1, 41, dtype="float64") * 2.0,
        }
    )


@pytest.fixture
def tiny_profile() -> TableExpectation:
    return TableExpectation(
        name="tiny",
        primary_key=["row_id"],
        columns=[
            ColumnExpectation(name="row_id", dtype="string", required=True, unique=True),
            ColumnExpectation(name="category", dtype="string", required=True,
                              allowed_values=["A", "B", "C", "D"]),
            ColumnExpectation(name="quantity", dtype="float", required=True, min=0.5),
            ColumnExpectation(name="unit_price", dtype="float", required=True, min=0.01),
            ColumnExpectation(name="total", dtype="float", required=True, min=0.0),
        ],
        cross_field_rules=[
            CrossFieldRule(
                id="total_matches",
                kind="expression",
                expression="abs(total - quantity * unit_price) <= 0.01",
                description="Total must equal quantity times unit price",
            )
        ],
        outlier_columns=["quantity"],
    )


@pytest.fixture(scope="session")
def erp_tables_small() -> dict[str, pd.DataFrame]:
    return generate_erp_dataset(
        seed=7,
        volumes=ERPVolumes(vendors=20, materials=120, bom_headers=15, purchase_orders=200),
    )
