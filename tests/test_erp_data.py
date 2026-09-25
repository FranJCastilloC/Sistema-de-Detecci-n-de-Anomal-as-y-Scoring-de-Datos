"""The synthetic extract must be deterministic and relationally sound."""

from __future__ import annotations

import pandas as pd

from dq_anomaly.data.defect_injector import DEFAULT_DEFECT_SPECS, inject_defects
from dq_anomaly.data.erp_generator import ERPVolumes, generate_erp_dataset

SMALL = ERPVolumes(vendors=20, materials=120, bom_headers=15, purchase_orders=200)


def test_generation_is_deterministic_for_a_given_seed():
    first = generate_erp_dataset(seed=7, volumes=SMALL)
    second = generate_erp_dataset(seed=7, volumes=SMALL)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_different_seeds_produce_different_data():
    first = generate_erp_dataset(seed=7, volumes=SMALL)
    second = generate_erp_dataset(seed=8, volumes=SMALL)
    assert not first["po_lines"].equals(second["po_lines"])


def test_clean_extract_has_no_dangling_foreign_keys(erp_tables_small):
    tables = erp_tables_small
    assert tables["po_lines"]["material_id"].isin(tables["materials"]["material_id"]).all()
    assert tables["po_lines"]["po_id"].isin(tables["purchase_orders"]["po_id"]).all()
    assert tables["purchase_orders"]["vendor_id"].isin(tables["vendors"]["vendor_id"]).all()
    assert tables["bom_lines"]["bom_id"].isin(tables["bom_headers"]["bom_id"]).all()


def test_clean_order_headers_agree_with_their_line_items(erp_tables_small):
    tables = erp_tables_small
    totals = tables["po_lines"].groupby("po_id")["net_value"].sum().round(2)
    headers = tables["purchase_orders"].set_index("po_id")["total_net_value"].round(2)
    pd.testing.assert_series_equal(headers, totals, check_names=False)


def test_injection_does_not_mutate_the_clean_tables(erp_tables_small):
    before = {name: frame.copy(deep=True) for name, frame in erp_tables_small.items()}
    inject_defects(erp_tables_small, seed=99)
    for name, frame in before.items():
        pd.testing.assert_frame_equal(erp_tables_small[name], frame)


def test_injection_is_deterministic_for_a_given_seed(erp_tables_small):
    _, first = inject_defects(erp_tables_small, seed=99)
    _, second = inject_defects(erp_tables_small, seed=99)
    pd.testing.assert_frame_equal(first, second)


def test_ledger_accounts_for_every_actual_mutation(erp_tables_small):
    """Guards the classic failure mode: a ledger that drifts from the data.

    Every cell that differs between the clean and corrupted tables must appear
    in the ledger, and vice versa.
    """
    dirty, ledger = inject_defects(erp_tables_small, seed=99)
    key_columns = {"vendors": "vendor_id", "materials": "material_id",
                   "purchase_orders": "po_id", "po_lines": "po_line_id"}

    for table, key_column in key_columns.items():
        clean_frame = erp_tables_small[table]
        dirty_frame = dirty[table].iloc[:len(clean_frame)]
        recorded = set(
            zip(ledger.loc[ledger["table"] == table, "row_key"],
                ledger.loc[ledger["table"] == table, "column"].astype(object))
        )
        for column in clean_frame.columns:
            left = clean_frame[column].astype(object).where(clean_frame[column].notna(), None)
            right = dirty_frame[column].astype(object).where(dirty_frame[column].notna(), None)
            changed = left.ne(right)
            for key in clean_frame.loc[changed.to_numpy(), key_column]:
                assert (str(key), column) in recorded, (
                    f"{table}.{column} row {key} was mutated but is not in the ledger"
                )


def test_every_declared_defect_family_is_actually_injected(erp_tables_small):
    _, ledger = inject_defects(erp_tables_small, seed=99)
    assert set(ledger["defect_code"]) == {spec.code for spec in DEFAULT_DEFECT_SPECS}
