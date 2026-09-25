"""Controlled corruption of the synthetic ERP extract.

Every mutation is recorded in a ground-truth ledger. That ledger is what makes
the quality engine measurable: instead of only describing a batch, we can ask
what fraction of known defects each rule actually recovered, and how many cells
it flagged that were never corrupted.

All mutations go through ``df.loc[rows, col] = value``. Under pandas 3's
copy-on-write semantics chained assignment silently does nothing, which would
produce a ledger that quietly disagrees with the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

PRIMARY_KEYS = {
    "vendors": "vendor_id",
    "materials": "material_id",
    "bom_headers": "bom_id",
    "bom_lines": "bom_line_id",
    "purchase_orders": "po_id",
    "po_lines": "po_line_id",
}

_MOJIBAKE = {"é": "Ã©", "á": "Ã¡", "ó": "Ã³", "ñ": "Ã±", "ü": "Ã¼"}
_TERM_VARIANTS = ["NET 30 days", "net30", "N30", "30 DAYS", "Net-30"]


@dataclass(frozen=True)
class DefectSpec:
    """One family of injected defects."""

    code: str
    table: str
    column: str | None
    rate: float
    dimension: str
    description: str


DEFAULT_DEFECT_SPECS: list[DefectSpec] = [
    DefectSpec("MISSING_CRITICAL", "po_lines", "unit_price", 0.012, "completeness",
               "Unit price blanked out on purchase order lines"),
    DefectSpec("MISSING_OPTIONAL", "vendors", "tax_id", 0.030, "completeness",
               "Vendor tax id missing"),
    DefectSpec("TYPE_MISMATCH", "po_lines", "quantity", 0.005, "validity",
               "Quantity stored as European-formatted text"),
    DefectSpec("NEGATIVE_QUANTITY", "po_lines", "quantity", 0.004, "validity",
               "Negative ordered quantity"),
    DefectSpec("ZERO_PRICE", "po_lines", "unit_price", 0.006, "validity",
               "Unit price of zero"),
    DefectSpec("OUT_OF_RANGE_PRICE", "materials", "standard_price", 0.003, "plausibility",
               "Fat-fingered standard price, orders of magnitude too high"),
    DefectSpec("INVALID_CATEGORY", "vendors", "payment_terms", 0.020, "validity",
               "Payment terms outside the controlled vocabulary"),
    DefectSpec("INVALID_UOM", "po_lines", "uom", 0.008, "consistency",
               "Line unit of measure disagrees with the material master"),
    DefectSpec("DUPLICATE_ROW", "purchase_orders", None, 0.005, "uniqueness",
               "Purchase order header duplicated with the same key"),
    DefectSpec("BROKEN_FK", "po_lines", "material_id", 0.004, "consistency",
               "Line points at a material that does not exist"),
    DefectSpec("ARITHMETIC_INCONSISTENCY", "po_lines", "net_value", 0.009, "consistency",
               "Net value does not equal quantity times unit price"),
    DefectSpec("DATE_ORDER_VIOLATION", "purchase_orders", "delivery_date", 0.007, "consistency",
               "Delivery date precedes the order date"),
    DefectSpec("FUTURE_DATE", "purchase_orders", "order_date", 0.002, "validity",
               "Order dated years into the future"),
    DefectSpec("ENCODING_NOISE", "materials", "description", 0.010, "validity",
               "Mis-decoded characters and trailing whitespace in free text"),
]

LEDGER_COLUMNS = [
    "defect_id", "table", "row_key", "column", "defect_code",
    "dimension", "original_value", "corrupted_value",
]


def _pick_rows(frame: pd.DataFrame, rate: float, rng: np.random.Generator) -> np.ndarray:
    n = max(1, int(round(len(frame) * rate)))
    n = min(n, len(frame))
    return rng.choice(frame.index.to_numpy(), size=n, replace=False)


def _record(
    records: list[dict], table: str, keys, column: str | None, spec: DefectSpec,
    originals, corrupted,
) -> None:
    for key, original, new in zip(keys, originals, corrupted):
        records.append(
            {
                "table": table,
                "row_key": str(key),
                "column": column,
                "defect_code": spec.code,
                "dimension": spec.dimension,
                "original_value": "" if original is None else str(original),
                "corrupted_value": "" if new is None else str(new),
            }
        )


def inject_defects(
    tables: dict[str, pd.DataFrame],
    specs: list[DefectSpec] | None = None,
    seed: int = 1337,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return corrupted copies of ``tables`` plus the ground-truth defect ledger.

    The input tables are never mutated.
    """
    specs = specs or DEFAULT_DEFECT_SPECS
    rng = np.random.default_rng(seed)
    dirty = {name: frame.copy(deep=True) for name, frame in tables.items()}
    records: list[dict] = []

    handlers: dict[str, Callable] = {
        "MISSING_CRITICAL": _inject_missing,
        "MISSING_OPTIONAL": _inject_missing,
        "TYPE_MISMATCH": _inject_type_mismatch,
        "NEGATIVE_QUANTITY": _inject_negative,
        "ZERO_PRICE": _inject_zero,
        "OUT_OF_RANGE_PRICE": _inject_price_outlier,
        "INVALID_CATEGORY": _inject_invalid_category,
        "INVALID_UOM": _inject_invalid_uom,
        "DUPLICATE_ROW": _inject_duplicate_rows,
        "BROKEN_FK": _inject_broken_fk,
        "ARITHMETIC_INCONSISTENCY": _inject_arithmetic,
        "DATE_ORDER_VIOLATION": _inject_date_order,
        "FUTURE_DATE": _inject_future_date,
        "ENCODING_NOISE": _inject_encoding_noise,
    }

    for spec in specs:
        handler = handlers[spec.code]
        handler(dirty, spec, rng, records)

    ledger = pd.DataFrame(records)
    if ledger.empty:
        ledger = pd.DataFrame(columns=LEDGER_COLUMNS)
    else:
        ledger.insert(0, "defect_id", [f"DEF-{i:06d}" for i in range(1, len(ledger) + 1)])
        ledger = ledger[LEDGER_COLUMNS]
    return dirty, ledger


# --- individual defect handlers -------------------------------------------------

def _keys(frame: pd.DataFrame, table: str, rows: np.ndarray):
    return frame.loc[rows, PRIMARY_KEYS[table]].to_numpy()


def _inject_missing(dirty, spec, rng, records):
    frame = dirty[spec.table]
    rows = _pick_rows(frame, spec.rate, rng)
    originals = frame.loc[rows, spec.column].tolist()
    frame.loc[rows, spec.column] = np.nan
    _record(records, spec.table, _keys(frame, spec.table, rows), spec.column, spec,
            originals, [None] * len(rows))


def _inject_type_mismatch(dirty, spec, rng, records):
    frame = dirty[spec.table]
    rows = _pick_rows(frame, spec.rate, rng)
    originals = frame.loc[rows, spec.column].tolist()
    # A float column cannot hold text under pandas 3, so widen it first. This
    # mirrors what a CSV export from a mis-configured locale actually produces.
    frame[spec.column] = frame[spec.column].astype(object)
    corrupted = [f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                 for value in originals]
    frame.loc[rows, spec.column] = corrupted
    _record(records, spec.table, _keys(frame, spec.table, rows), spec.column, spec,
            originals, corrupted)


def _inject_negative(dirty, spec, rng, records):
    frame = dirty[spec.table]
    numeric = pd.to_numeric(frame[spec.column], errors="coerce")
    eligible = frame.loc[numeric.notna()]
    rows = _pick_rows(eligible, spec.rate * len(frame) / max(len(eligible), 1), rng)
    originals = numeric.loc[rows].tolist()
    corrupted = [-abs(value) for value in originals]
    frame.loc[rows, spec.column] = corrupted
    _record(records, spec.table, _keys(frame, spec.table, rows), spec.column, spec,
            originals, corrupted)


def _inject_zero(dirty, spec, rng, records):
    frame = dirty[spec.table]
    eligible = frame.loc[frame[spec.column].notna()]
    rows = _pick_rows(eligible, spec.rate * len(frame) / max(len(eligible), 1), rng)
    originals = frame.loc[rows, spec.column].tolist()
    frame.loc[rows, spec.column] = 0.0
    _record(records, spec.table, _keys(frame, spec.table, rows), spec.column, spec,
            originals, [0.0] * len(rows))


def _inject_price_outlier(dirty, spec, rng, records):
    frame = dirty[spec.table]
    rows = _pick_rows(frame, spec.rate, rng)
    originals = frame.loc[rows, spec.column].tolist()
    factors = rng.uniform(50, 500, size=len(rows))
    corrupted = np.round(np.asarray(originals, dtype=float) * factors, 2)
    frame.loc[rows, spec.column] = corrupted
    _record(records, spec.table, _keys(frame, spec.table, rows), spec.column, spec,
            originals, corrupted.tolist())


def _inject_invalid_category(dirty, spec, rng, records):
    frame = dirty[spec.table]
    rows = _pick_rows(frame, spec.rate, rng)
    originals = frame.loc[rows, spec.column].tolist()
    corrupted = rng.choice(_TERM_VARIANTS, size=len(rows)).tolist()
    frame[spec.column] = frame[spec.column].astype(object)
    frame.loc[rows, spec.column] = corrupted
    _record(records, spec.table, _keys(frame, spec.table, rows), spec.column, spec,
            originals, corrupted)


def _inject_invalid_uom(dirty, spec, rng, records):
    frame = dirty[spec.table]
    rows = _pick_rows(frame, spec.rate, rng)
    originals = frame.loc[rows, spec.column].tolist()
    pool = ["EA", "KG", "L", "M", "BOX"]
    corrupted = [
        rng.choice([u for u in pool if u != original]) for original in originals
    ]
    frame[spec.column] = frame[spec.column].astype(object)
    frame.loc[rows, spec.column] = corrupted
    _record(records, spec.table, _keys(frame, spec.table, rows), spec.column, spec,
            originals, corrupted)


def _inject_duplicate_rows(dirty, spec, rng, records):
    frame = dirty[spec.table]
    rows = _pick_rows(frame, spec.rate, rng)
    clones = frame.loc[rows].copy()
    keys = clones[PRIMARY_KEYS[spec.table]].to_numpy()
    dirty[spec.table] = pd.concat([frame, clones], ignore_index=True)
    _record(records, spec.table, keys, None, spec, keys.tolist(), keys.tolist())


def _inject_broken_fk(dirty, spec, rng, records):
    frame = dirty[spec.table]
    rows = _pick_rows(frame, spec.rate, rng)
    originals = frame.loc[rows, spec.column].tolist()
    corrupted = [f"MAT-9{rng.integers(10000, 99999)}" for _ in rows]
    frame[spec.column] = frame[spec.column].astype(object)
    frame.loc[rows, spec.column] = corrupted
    _record(records, spec.table, _keys(frame, spec.table, rows), spec.column, spec,
            originals, corrupted)


def _inject_arithmetic(dirty, spec, rng, records):
    frame = dirty[spec.table]
    rows = _pick_rows(frame, spec.rate, rng)
    originals = frame.loc[rows, spec.column].tolist()
    factors = rng.uniform(1.05, 1.40, size=len(rows))
    corrupted = np.round(np.asarray(originals, dtype=float) * factors, 2)
    frame.loc[rows, spec.column] = corrupted
    _record(records, spec.table, _keys(frame, spec.table, rows), spec.column, spec,
            originals, corrupted.tolist())


def _inject_date_order(dirty, spec, rng, records):
    frame = dirty[spec.table]
    rows = _pick_rows(frame, spec.rate, rng)
    originals = frame.loc[rows, spec.column].tolist()
    shift = pd.to_timedelta(rng.integers(5, 120, size=len(rows)), unit="D")
    corrupted = frame.loc[rows, "order_date"] - shift
    frame.loc[rows, spec.column] = corrupted.to_numpy()
    _record(records, spec.table, _keys(frame, spec.table, rows), spec.column, spec,
            originals, corrupted.tolist())


def _inject_future_date(dirty, spec, rng, records):
    frame = dirty[spec.table]
    rows = _pick_rows(frame, spec.rate, rng)
    originals = frame.loc[rows, spec.column].tolist()
    shift = pd.to_timedelta(rng.integers(365, 1825, size=len(rows)), unit="D")
    corrupted = pd.Timestamp.today().normalize() + shift
    frame.loc[rows, spec.column] = corrupted.to_numpy()
    _record(records, spec.table, _keys(frame, spec.table, rows), spec.column, spec,
            originals, corrupted.tolist())


def _inject_encoding_noise(dirty, spec, rng, records):
    frame = dirty[spec.table]
    rows = _pick_rows(frame, spec.rate, rng)
    originals = frame.loc[rows, spec.column].tolist()
    corrupted = []
    for value in originals:
        text = str(value)
        for good, bad in _MOJIBAKE.items():
            text = text.replace(good, bad)
        corrupted.append(text.replace("E", "Ã©", 1) + "   ")
    frame[spec.column] = frame[spec.column].astype(object)
    frame.loc[rows, spec.column] = corrupted
    _record(records, spec.table, _keys(frame, spec.table, rows), spec.column, spec,
            originals, corrupted)
