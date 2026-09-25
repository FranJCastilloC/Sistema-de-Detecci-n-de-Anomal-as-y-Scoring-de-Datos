"""Generate the example batch reports shipped in results/examples/."""

from __future__ import annotations

import pandas as pd

from _bootstrap import ensure_src_on_path  # noqa: E402  (must precede dq_anomaly)

ensure_src_on_path()

from dq_anomaly.anomaly.artifacts import bundle_available, load_bundle
from dq_anomaly.config import PATHS, set_global_seed
from dq_anomaly.data.loaders import load_creditcard
from dq_anomaly.pipeline import AuditConfig, audit_batch
from dq_anomaly.quality.profile import load_profile

ERP_REFERENCE_TABLES = ["materials", "purchase_orders", "vendors"]


def build_erp_report() -> None:
    """Audit the corrupted ERP purchase order lines."""
    frame = pd.read_csv(PATHS.data_synthetic / "po_lines.csv", parse_dates=["delivery_date"])
    references = {
        name: pd.read_csv(PATHS.data_synthetic / f"{name}.csv")
        for name in ERP_REFERENCE_TABLES
    }
    result = audit_batch(
        frame,
        profile=load_profile("profile_erp_po_lines.yaml"),
        reference_tables=references,
        table_name="erp_po_lines",
        batch_id="erp-po-lines-2024Q4",
        config=AuditConfig(use_model=False),
    )
    paths = result.report.write(PATHS.examples / "erp_po_lines")
    print(f"ERP batch  : score {result.quality.global_score:.2f} "
          f"(grade {result.quality.grade}), {len(result.report.issues)} issues "
          f"-> {paths['markdown'].parent}")


def build_creditcard_report() -> None:
    """Audit one day of transactions with the trained autoencoder."""
    if not bundle_available():
        print("No trained model found; skipping the credit card example report.")
        return
    frame = load_creditcard()
    # Score the most recent 20,000 transactions as if they were today's batch.
    batch = frame.sort_values("Time", kind="mergesort").tail(20_000).reset_index(drop=True)
    result = audit_batch(
        batch,
        profile=load_profile("profile_creditcard.yaml"),
        bundle=load_bundle(),
        table_name="creditcard_transactions",
        batch_id="creditcard-latest-20k",
        label_column="Class",
        config=AuditConfig(use_model=True, max_record_issues=100),
    )
    paths = result.report.write(PATHS.examples / "creditcard_batch")
    summary = result.report.anomaly_summary
    print(f"Card batch : score {result.quality.global_score:.2f} "
          f"(grade {result.quality.grade}), {summary.get('n_flagged', 0)} records flagged, "
          f"{summary.get('n_caught', 0)}/{summary.get('n_labelled_positives', 0)} known frauds "
          f"caught -> {paths['markdown'].parent}")


def main() -> None:
    PATHS.ensure()
    set_global_seed(42)
    build_erp_report()
    build_creditcard_report()


if __name__ == "__main__":
    main()
