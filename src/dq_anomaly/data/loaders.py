"""Acquisition of the public credit card fraud dataset.

The dataset is the ULB / Worldline "Credit Card Fraud Detection" set, the same
data distributed on Kaggle. It is pulled from OpenML instead, where it is
mirrored under a public licence and needs no credentials, so anyone who clones
this repository can reproduce the results with a single command.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import pandas as pd

from dq_anomaly.config import PATHS

OPENML_DATA_ID = 1597
OPENML_PARQUET_URL = "https://data.openml.org/datasets/0000/1597/dataset_1597.pq"
OPENML_ARFF_URL = "https://openml.org/data/v1/download/1673544/creditcard.arff"

EXPECTED_ROWS = 284_807
EXPECTED_COLUMNS = 31
EXPECTED_FRAUD = 492

RAW_FILENAME = "dataset_1597.pq"
PROCESSED_FILENAME = "creditcard.parquet"

#: A small, committed slice of the most recent transactions. The full 73 MB
#: dataset is too large to track, so this is what the hosted demo scores.
SAMPLE_FILENAME = "creditcard_sample.parquet"
SAMPLE_ROWS = 20_000


def download_creditcard(dest: Path | None = None, force: bool = False) -> Path:
    """Download the raw dataset, returning the local path.

    Tries the OpenML parquet mirror first (73 MB); falls back to scikit-learn's
    OpenML client, which fetches and parses the 150 MB ARFF.
    """
    dest = Path(dest or PATHS.data_raw / RAW_FILENAME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest

    partial = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(OPENML_PARQUET_URL, timeout=600) as response, \
                open(partial, "wb") as handle:
            shutil.copyfileobj(response, handle)
        partial.replace(dest)
        return dest
    except Exception:  # noqa: BLE001 - any transport failure falls back
        partial.unlink(missing_ok=True)

    from sklearn.datasets import fetch_openml

    frame = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True).frame
    frame.to_parquet(dest, index=False)
    return dest


def load_creditcard(path: Path | None = None) -> pd.DataFrame:
    """Load the dataset with the target cast to a usable integer dtype.

    OpenML types ``Class`` as a nominal ``{'0','1'}``, so it arrives as a
    category of strings; summing it without a cast silently misbehaves.
    """
    path = Path(path or PATHS.data_raw / RAW_FILENAME)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/01_download_data.py` first."
        )
    frame = pd.read_parquet(path)
    frame["Class"] = frame["Class"].astype("int64").astype("int8")

    if frame.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(f"unexpected shape {frame.shape}, expected "
                         f"({EXPECTED_ROWS}, {EXPECTED_COLUMNS})")
    fraud = int(frame["Class"].sum())
    if fraud != EXPECTED_FRAUD:
        raise ValueError(f"expected {EXPECTED_FRAUD} fraud rows, found {fraud}")
    return frame


def load_table(source, kind: str | None = None) -> pd.DataFrame:
    """Read a CSV or parquet batch from a path or an uploaded file object."""
    if kind is None:
        name = str(getattr(source, "name", source)).lower()
        kind = "parquet" if name.endswith((".parquet", ".pq")) else "csv"
    if kind == "parquet":
        return pd.read_parquet(source)
    return pd.read_csv(source)


def write_creditcard_sample(n_rows: int = SAMPLE_ROWS) -> Path:
    """Carve the most recent transactions out into a committable sample."""
    frame = load_creditcard()
    sample = (
        frame.sort_values("Time", kind="mergesort").tail(n_rows).reset_index(drop=True)
    )
    PATHS.data_samples.mkdir(parents=True, exist_ok=True)
    path = PATHS.data_samples / SAMPLE_FILENAME
    sample.to_parquet(path, index=False, compression="zstd")
    return path


def load_creditcard_sample() -> pd.DataFrame:
    """Load the committed demo slice, falling back to the full dataset.

    This is what makes the hosted demo work: the deployment only ever has what
    is in the repository, and the full dataset is not.
    """
    path = PATHS.data_samples / SAMPLE_FILENAME
    if path.exists():
        frame = pd.read_parquet(path)
        frame["Class"] = frame["Class"].astype("int8")
        return frame
    frame = load_creditcard()
    return (
        frame.sort_values("Time", kind="mergesort")
        .tail(SAMPLE_ROWS)
        .reset_index(drop=True)
    )
