"""Generate the synthetic ERP extract, both clean and deliberately corrupted."""

from __future__ import annotations

from _bootstrap import ensure_src_on_path  # noqa: E402  (must precede dq_anomaly)

ensure_src_on_path()

from dq_anomaly.config import PATHS, set_global_seed
from dq_anomaly.data.defect_injector import inject_defects
from dq_anomaly.data.erp_generator import ERPVolumes, generate_erp_dataset, write_erp_dataset

SEED = 42
DEFECT_SEED = 1337


def main() -> None:
    PATHS.ensure()
    set_global_seed(SEED)

    clean = generate_erp_dataset(seed=SEED, volumes=ERPVolumes())
    dirty, ledger = inject_defects(clean, seed=DEFECT_SEED)

    write_erp_dataset(clean, PATHS.data_synthetic / "clean")
    write_erp_dataset(dirty, PATHS.data_synthetic)
    ledger_path = PATHS.data_synthetic / "defect_ledger.csv"
    ledger.to_csv(ledger_path, index=False)

    for name, frame in dirty.items():
        print(f"{name:18s} {len(frame):>7,} rows")
    print(f"\nInjected {len(ledger):,} defects across "
          f"{ledger['defect_code'].nunique()} families -> {ledger_path}")


if __name__ == "__main__":
    main()
