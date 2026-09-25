"""Command line entry point: `python -m dq_anomaly.cli --help`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from dq_anomaly.config import PATHS, set_global_seed
from dq_anomaly.pipeline import AuditConfig, audit_batch
from dq_anomaly.quality.profile import load_profile

app = typer.Typer(
    add_completion=False,
    help="Audit a batch of tabular data for quality issues and anomalies.",
)


@app.command()
def download() -> None:
    """Fetch the public credit card fraud dataset into data/raw/."""
    from dq_anomaly.data.loaders import download_creditcard, load_creditcard

    PATHS.ensure()
    path = download_creditcard()
    frame = load_creditcard(path)
    typer.echo(f"{path} | {len(frame):,} rows | {int(frame['Class'].sum())} fraud")


@app.command("generate-erp")
def generate_erp() -> None:
    """Generate the synthetic ERP extract and its defect ledger."""
    from dq_anomaly.data.defect_injector import inject_defects
    from dq_anomaly.data.erp_generator import generate_erp_dataset, write_erp_dataset

    PATHS.ensure()
    set_global_seed()
    clean = generate_erp_dataset(seed=42)
    dirty, ledger = inject_defects(clean, seed=1337)
    write_erp_dataset(clean, PATHS.data_synthetic / "clean")
    write_erp_dataset(dirty, PATHS.data_synthetic)
    ledger.to_csv(PATHS.data_synthetic / "defect_ledger.csv", index=False)
    typer.echo(f"{len(ledger):,} defects injected across "
               f"{ledger['defect_code'].nunique()} families")


@app.command("audit")
def audit(
    input_path: Path = typer.Option(..., "--input", "-i", exists=True,
                                    help="CSV or parquet batch to audit"),
    profile_name: str | None = typer.Option(
        None, "--profile", "-p",
        help="Profile file in config/ (auto-detected or inferred when omitted)"),
    output_dir: Path | None = typer.Option(
        None, "--out", "-o", help="Where to write the report (default results/examples/<name>)"),
    reference_dir: Path | None = typer.Option(
        None, "--references", "-r",
        help="Directory of CSVs used to resolve foreign-key rules"),
    review_budget: float = typer.Option(
        0.005, "--budget", "-b", min=0.0001, max=0.5,
        help="Share of the batch the review team can clear"),
    no_model: bool = typer.Option(False, "--no-model", help="Skip anomaly detection"),
    label_column: str | None = typer.Option(
        None, "--label-column", help="Known-positive column, when the batch is labelled"),
) -> None:
    """Score a batch and write a prioritized issue report."""
    set_global_seed()
    PATHS.ensure()

    frame = (
        pd.read_parquet(input_path)
        if input_path.suffix in {".parquet", ".pq"}
        else pd.read_csv(input_path)
    )
    profile = load_profile(profile_name) if profile_name else None

    references = None
    if reference_dir:
        references = {
            path.stem: pd.read_csv(path) for path in sorted(Path(reference_dir).glob("*.csv"))
        }

    result = audit_batch(
        frame,
        profile=profile,
        reference_tables=references,
        table_name=input_path.stem,
        batch_id=input_path.stem,
        label_column=label_column,
        config=AuditConfig(review_budget=review_budget, use_model=not no_model),
    )

    destination = Path(output_dir or PATHS.examples / input_path.stem)
    paths = result.report.write(destination)

    quality = result.quality
    typer.echo(f"Quality score : {quality.global_score:.2f}/100 (grade {quality.grade})")
    typer.echo(f"Rows          : {result.report.n_rows:,}")
    if result.report.anomaly_summary:
        summary = result.report.anomaly_summary
        typer.echo(f"Anomalies     : {summary['n_flagged']:,} flagged "
                   f"({summary['flag_rate']:.2%}) by {summary['model']}")
    counts = result.report.severity_counts
    typer.echo(f"Issues        : {len(result.report.issues)} "
               f"({counts['critical']} critical, {counts['high']} high, "
               f"{counts['medium']} medium, {counts['low']} low)")
    for note in result.notes:
        typer.echo(f"Note          : {note}")
    typer.echo(f"Report        : {paths['markdown']}")


@app.command("top-issues")
def top_issues(
    report_path: Path = typer.Option(..., "--report", exists=True,
                                     help="issues.csv produced by `audit`"),
    limit: int = typer.Option(10, "--limit", "-n"),
) -> None:
    """Print the highest-priority issues from a generated report."""
    frame = pd.read_csv(report_path)
    columns = [c for c in ["priority", "severity", "dimension", "rule_id", "n_records", "reason"]
               if c in frame.columns]
    typer.echo(frame.head(limit)[columns].to_string(index=False))


if __name__ == "__main__":
    app()
