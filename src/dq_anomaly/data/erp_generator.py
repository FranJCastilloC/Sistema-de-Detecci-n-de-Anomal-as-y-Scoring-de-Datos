"""Synthetic ERP dataset generator.

Produces a small but relationally consistent SAP-style extract: vendors,
materials, bills of materials and purchase orders with their line items.
Every value is drawn from a seeded generator, so the same seed always yields
byte-identical tables.

Faker is used only for the handful of genuinely "textual" fields (company
names, tax ids, user names). Those are drawn once into small pools and then
indexed with numpy; calling Faker once per row would dominate the runtime at
90k line items.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

PAYMENT_TERMS = ["NET30", "NET45", "NET60", "IMMEDIATE"]
CURRENCIES = ["EUR", "USD", "GBP", "MXN"]
PLANTS = ["1000", "1100", "2000", "3000"]
UOMS = ["EA", "KG", "L", "M", "BOX"]
PO_STATUSES = ["OPEN", "PARTIAL", "CLOSED", "CANCELLED"]
BOM_STATUSES = ["ACTIVE", "INACTIVE", "DRAFT"]
MATERIAL_GROUPS = [
    "FASTENERS", "RAW-METAL", "RAW-PLASTIC", "ELECTRONICS", "PACKAGING",
    "CHEMICALS", "TOOLING", "CONSUMABLES", "SUBASSEMBLY", "FINISHED",
    "SPARES", "LABELS",
]

_DESC_HEAD = [
    "HEX BOLT", "SOCKET SCREW", "FLAT WASHER", "COMPRESSION SPRING", "BALL BEARING",
    "ALUMINIUM SHEET", "STEEL ROD", "COPPER WIRE", "PVC TUBE", "RUBBER GASKET",
    "CERAMIC CAPACITOR", "POWER RESISTOR", "LINEAR REGULATOR", "RIBBON CABLE",
    "CARTON BOX", "STRETCH FILM", "ADHESIVE LABEL", "EPOXY RESIN", "CUTTING FLUID",
    "DRILL BIT", "END MILL", "SAFETY GLOVE", "FILTER CARTRIDGE", "DRIVE SHAFT",
]
_DESC_SPEC = ["M4X12", "M6X25", "M8X40", "M10X60", "3MM", "5MM", "10MM", "22UF",
              "100R", "3V3", "12V", "A2", "A4", "GRADE-8", "TYPE-B", "XL"]
_DESC_FINISH = ["ZINC", "BLACK OX", "STAINLESS", "ANODISED", "GALVANISED", "PLAIN",
                "NICKEL", "POWDER CT"]


@dataclass(frozen=True)
class ERPVolumes:
    """Row counts for each generated table."""

    vendors: int = 300
    materials: int = 5_000
    bom_headers: int = 1_200
    purchase_orders: int = 20_000


def _make_vendors(n: int, rng: np.random.Generator, fake: Faker) -> pd.DataFrame:
    names = [fake.company() for _ in range(n)]
    created = _random_dates(rng, n, "2015-01-01", "2023-06-30")
    return pd.DataFrame(
        {
            "vendor_id": [f"VEN-{i:05d}" for i in range(1, n + 1)],
            "vendor_name": names,
            "country_code": rng.choice(
                ["DE", "US", "MX", "CN", "ES", "IT", "PL", "GB", "DO", "BR"], size=n
            ),
            "tax_id": [fake.bothify("??#########").upper() for _ in range(n)],
            "payment_terms": rng.choice(PAYMENT_TERMS, size=n, p=[0.5, 0.25, 0.15, 0.10]),
            "currency": rng.choice(CURRENCIES, size=n, p=[0.45, 0.35, 0.12, 0.08]),
            "is_blocked": rng.random(n) < 0.04,
            "created_at": created,
        }
    )


def _make_materials(n: int, rng: np.random.Generator) -> pd.DataFrame:
    head = rng.choice(_DESC_HEAD, size=n)
    spec = rng.choice(_DESC_SPEC, size=n)
    finish = rng.choice(_DESC_FINISH, size=n)
    descriptions = np.char.add(
        np.char.add(np.char.add(head.astype(str), " "), spec.astype(str)),
        np.char.add(" ", finish.astype(str)),
    )
    # Log-normal prices: most parts are cheap, a few are expensive. This is the
    # real shape of an ERP price column and it matters for the outlier rules.
    prices = np.round(np.exp(rng.normal(2.2, 1.35, size=n)), 2).clip(0.5, 5_000.0)
    return pd.DataFrame(
        {
            "material_id": [f"MAT-{i:06d}" for i in range(1, n + 1)],
            "description": descriptions,
            "material_group": rng.choice(MATERIAL_GROUPS, size=n),
            "base_uom": rng.choice(UOMS, size=n, p=[0.55, 0.15, 0.08, 0.12, 0.10]),
            "standard_price": prices,
            "currency": rng.choice(CURRENCIES, size=n, p=[0.5, 0.35, 0.1, 0.05]),
            "plant": rng.choice(PLANTS, size=n),
            "safety_stock": rng.integers(0, 500, size=n),
            "is_active": rng.random(n) < 0.93,
            "created_at": _random_dates(rng, n, "2014-01-01", "2023-12-31"),
        }
    )


def _make_boms(
    n_headers: int, materials: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    finished = materials.loc[
        materials["material_group"].isin(["FINISHED", "SUBASSEMBLY"]), "material_id"
    ].to_numpy()
    if len(finished) < n_headers:
        finished = materials["material_id"].to_numpy()
    parents = rng.choice(finished, size=n_headers, replace=True)
    valid_from = _random_dates(rng, n_headers, "2018-01-01", "2023-01-01")
    headers = pd.DataFrame(
        {
            "bom_id": [f"BOM-{i:06d}" for i in range(1, n_headers + 1)],
            "parent_material_id": parents,
            "plant": rng.choice(PLANTS, size=n_headers),
            "base_quantity": rng.choice([1, 1, 1, 10, 100], size=n_headers),
            "valid_from": valid_from,
            "valid_to": valid_from + pd.to_timedelta(rng.integers(365, 3650, n_headers), unit="D"),
            "status": rng.choice(BOM_STATUSES, size=n_headers, p=[0.8, 0.12, 0.08]),
        }
    )

    n_lines_per = rng.integers(3, 13, size=n_headers)
    bom_ids = np.repeat(headers["bom_id"].to_numpy(), n_lines_per)
    total = int(n_lines_per.sum())
    positions = np.concatenate([np.arange(1, k + 1) * 10 for k in n_lines_per])
    components = rng.choice(materials["material_id"].to_numpy(), size=total, replace=True)
    lines = pd.DataFrame(
        {
            "bom_line_id": [f"BML-{i:07d}" for i in range(1, total + 1)],
            "bom_id": bom_ids,
            "position": positions,
            "component_material_id": components,
            "quantity": np.round(rng.gamma(2.0, 1.5, size=total) + 0.1, 3),
            "uom": rng.choice(UOMS, size=total, p=[0.6, 0.15, 0.07, 0.10, 0.08]),
            "scrap_pct": np.round(rng.beta(1.5, 12.0, size=total) * 15.0, 2),
        }
    )
    return headers, lines


def _make_purchase_orders(
    n_orders: int, vendors: pd.DataFrame, materials: pd.DataFrame,
    rng: np.random.Generator, fake: Faker,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = [fake.user_name() for _ in range(40)]
    vendor_idx = rng.integers(0, len(vendors), size=n_orders)
    order_date = _random_dates(rng, n_orders, "2023-01-01", "2024-12-31")
    lead_days = rng.integers(3, 91, size=n_orders)

    orders = pd.DataFrame(
        {
            "po_id": [f"PO-{4500000000 + i}" for i in range(1, n_orders + 1)],
            "vendor_id": vendors["vendor_id"].to_numpy()[vendor_idx],
            "order_date": order_date,
            "delivery_date": order_date + pd.to_timedelta(lead_days, unit="D"),
            "currency": vendors["currency"].to_numpy()[vendor_idx],
            "plant": rng.choice(PLANTS, size=n_orders),
            "created_by": rng.choice(users, size=n_orders),
            "status": rng.choice(PO_STATUSES, size=n_orders, p=[0.25, 0.15, 0.55, 0.05]),
        }
    )

    n_lines_per = rng.integers(1, 9, size=n_orders)
    total = int(n_lines_per.sum())
    po_ids = np.repeat(orders["po_id"].to_numpy(), n_lines_per)
    positions = np.concatenate([np.arange(1, k + 1) * 10 for k in n_lines_per])
    mat_idx = rng.integers(0, len(materials), size=total)

    quantity = np.round(rng.gamma(2.2, 18.0, size=total) + 1.0, 2)
    # Purchase price drifts around the material's standard price.
    unit_price = np.round(
        materials["standard_price"].to_numpy()[mat_idx] * rng.normal(1.0, 0.12, size=total).clip(0.6, 1.6),
        2,
    ).clip(0.01, None)
    net_value = np.round(quantity * unit_price, 2)

    status_per_line = np.repeat(orders["status"].to_numpy(), n_lines_per)
    receipt_ratio = np.select(
        [status_per_line == "CLOSED", status_per_line == "PARTIAL", status_per_line == "OPEN"],
        [np.ones(total), rng.uniform(0.1, 0.9, size=total), np.zeros(total)],
        default=np.zeros(total),
    )
    goods_receipt_qty = np.round(quantity * receipt_ratio, 2)
    invoice_qty = np.round(goods_receipt_qty * rng.uniform(0.85, 1.0, size=total), 2)

    line_delivery = np.repeat(orders["delivery_date"].to_numpy(), n_lines_per)
    lines = pd.DataFrame(
        {
            "po_line_id": [f"POL-{i:07d}" for i in range(1, total + 1)],
            "po_id": po_ids,
            "position": positions,
            "material_id": materials["material_id"].to_numpy()[mat_idx],
            "quantity": quantity,
            "uom": materials["base_uom"].to_numpy()[mat_idx],
            "unit_price": unit_price,
            "net_value": net_value,
            "currency": np.repeat(orders["currency"].to_numpy(), n_lines_per),
            "delivery_date": line_delivery + pd.to_timedelta(
                rng.integers(0, 8, size=total), unit="D"
            ),
            "goods_receipt_qty": goods_receipt_qty,
            "invoice_qty": invoice_qty,
        }
    )

    totals = lines.groupby("po_id", sort=False)["net_value"].sum().round(2)
    orders["total_net_value"] = orders["po_id"].map(totals).astype("float64")
    return orders, lines


def _random_dates(
    rng: np.random.Generator, n: int, start: str, end: str
) -> pd.Series:
    start_ts = pd.Timestamp(start).value // 10**9
    end_ts = pd.Timestamp(end).value // 10**9
    seconds = rng.integers(start_ts, end_ts, size=n)
    return pd.to_datetime(seconds, unit="s").floor("D")


def generate_erp_dataset(
    seed: int = 42, volumes: ERPVolumes | None = None
) -> dict[str, pd.DataFrame]:
    """Generate a clean, relationally consistent ERP extract."""
    volumes = volumes or ERPVolumes()
    rng = np.random.default_rng(seed)
    fake = Faker("en_US")
    Faker.seed(seed)

    vendors = _make_vendors(volumes.vendors, rng, fake)
    materials = _make_materials(volumes.materials, rng)
    bom_headers, bom_lines = _make_boms(volumes.bom_headers, materials, rng)
    orders, lines = _make_purchase_orders(
        volumes.purchase_orders, vendors, materials, rng, fake
    )
    return {
        "vendors": vendors,
        "materials": materials,
        "bom_headers": bom_headers,
        "bom_lines": bom_lines,
        "purchase_orders": orders,
        "po_lines": lines,
    }


def write_erp_dataset(
    tables: dict[str, pd.DataFrame], out_dir: Path, suffix: str = ""
) -> dict[str, Path]:
    """Write each table to CSV, returning the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, frame in tables.items():
        path = out_dir / f"{name}{suffix}.csv"
        frame.to_csv(path, index=False)
        written[name] = path
    return written
