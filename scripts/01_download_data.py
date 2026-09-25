"""Download the public credit card fraud dataset into data/raw/."""

from __future__ import annotations

from _bootstrap import ensure_src_on_path  # noqa: E402  (must precede dq_anomaly)

ensure_src_on_path()

from dq_anomaly.config import PATHS
from dq_anomaly.data.loaders import download_creditcard, load_creditcard


def main() -> None:
    PATHS.ensure()
    path = download_creditcard()
    frame = load_creditcard(path)
    fraud = int(frame["Class"].sum())
    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"Rows: {len(frame):,} | Columns: {frame.shape[1]} | "
          f"Fraud: {fraud} ({fraud / len(frame):.4%})")


if __name__ == "__main__":
    main()
