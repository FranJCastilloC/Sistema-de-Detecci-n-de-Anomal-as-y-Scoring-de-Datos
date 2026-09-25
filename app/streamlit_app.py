"""Interactive demo: drop in a CSV, get a quality score, anomalies and issues.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dq_anomaly.anomaly.artifacts import bundle_available, load_bundle  # noqa: E402
from dq_anomaly.data.defect_injector import inject_defects  # noqa: E402
from dq_anomaly.data.erp_generator import generate_erp_dataset  # noqa: E402
from dq_anomaly.data.loaders import load_creditcard_sample  # noqa: E402
from dq_anomaly.pipeline import AuditConfig, audit_batch  # noqa: E402
from dq_anomaly.quality.profile import load_profile  # noqa: E402

SEVERITY_COLOURS = {
    "critical": "#b3261e",
    "high": "#e8710a",
    "medium": "#f2b705",
    "low": "#6b7f8c",
}
GRADE_COLOURS = {"A": "#2e7d32", "B": "#68a03a", "C": "#f2b705", "D": "#e8710a", "F": "#b3261e"}

st.set_page_config(page_title="Data Quality & Anomaly Audit", page_icon="•", layout="wide")


@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_bundle() if bundle_available() else None


@st.cache_data(show_spinner="Generating the synthetic ERP extract...")
def load_erp_tables(corrupted: bool) -> dict[str, pd.DataFrame]:
    """Build the ERP extract in memory rather than reading it from disk.

    Generation takes well under a second, so the hosted demo does not need the
    CSVs to be committed - and the seeds guarantee it is the same extract the
    metrics in the README were measured on.
    """
    tables = generate_erp_dataset(seed=42)
    if corrupted:
        tables, _ = inject_defects(tables, seed=1337)
    return tables


@st.cache_data(show_spinner="Loading transactions...")
def load_creditcard_batch() -> pd.DataFrame:
    return load_creditcard_sample()


@st.cache_data(show_spinner="Reading uploaded file...")
def read_upload(payload: bytes, filename: str) -> pd.DataFrame:
    import io

    if filename.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(io.BytesIO(payload))
    return pd.read_csv(io.BytesIO(payload))


def score_badge(score: float, grade: str) -> str:
    colour = GRADE_COLOURS.get(grade, "#6b7f8c")
    return (
        f"<div style='border-left:6px solid {colour};padding:0.4rem 0 0.4rem 0.9rem'>"
        f"<div style='font-size:2.6rem;font-weight:700;line-height:1'>{score:.1f}"
        f"<span style='font-size:1.1rem;font-weight:400;color:#6b7f8c'> / 100</span></div>"
        f"<div style='color:{colour};font-weight:600;margin-top:0.2rem'>Grade {grade}</div>"
        "</div>"
    )


# --- sidebar -------------------------------------------------------------------

st.sidebar.title("Batch to audit")
source = st.sidebar.radio(
    "Source",
    ["ERP purchase order lines (corrupted)", "ERP purchase order lines (clean)",
     "Credit card transactions", "Upload a file"],
    help="The bundled batches are the ones the project was built and measured on.",
)

uploaded = None
if source == "Upload a file":
    uploaded = st.sidebar.file_uploader("CSV or parquet", type=["csv", "parquet", "pq"])

review_budget = st.sidebar.slider(
    "Review budget (% of batch)", 0.1, 5.0, 0.5, 0.1,
    help="How much of the batch a reviewer can realistically clear. "
         "This sets the anomaly threshold.",
) / 100.0

use_model = st.sidebar.checkbox("Run anomaly detection", value=True)
max_issues = st.sidebar.number_input("Max record-level issues listed", 20, 1000, 200, 20)

bundle = get_bundle()
if bundle is None:
    st.sidebar.warning(
        "No trained model in `models/`. Quality scoring still works; run "
        "`python scripts/03_train_models.py` to enable the autoencoder."
    )

# --- load the batch --------------------------------------------------------------

profile = None
references = None
table_name = "uploaded_batch"
label_column = None

if source.startswith("ERP"):
    tables = load_erp_tables(corrupted="corrupted" in source)
    frame = tables["po_lines"]
    references = {name: tables[name] for name in ("materials", "purchase_orders", "vendors")}
    profile = load_profile("profile_erp_po_lines.yaml")
    table_name = "erp_po_lines"
elif source == "Credit card transactions":
    try:
        frame = load_creditcard_batch()
    except FileNotFoundError:
        st.error("No transaction data available. Run "
                 "`python scripts/01_download_data.py` to fetch it.")
        st.stop()
    profile = load_profile("profile_creditcard.yaml")
    table_name = "creditcard_transactions"
    label_column = "Class"
else:
    if uploaded is None:
        st.title("Data Quality & Anomaly Audit")
        st.info("Upload a CSV or parquet file in the sidebar, or pick one of the "
                "bundled batches, to run an audit.")
        st.stop()
    frame = read_upload(uploaded.getvalue(), uploaded.name)
    table_name = Path(uploaded.name).stem

if len(frame) > 200_000:
    st.warning(f"Batch has {len(frame):,} rows; sampling 200,000 for the demo.")
    frame = frame.sample(200_000, random_state=42).reset_index(drop=True)


@st.cache_data(show_spinner="Auditing batch...")
def run_audit(
    frame: pd.DataFrame, table_name: str, profile_name: str | None, budget: float,
    use_model: bool, max_issues: int, label_column: str | None,
    reference_tables: dict[str, pd.DataFrame] | None,
):
    return audit_batch(
        frame,
        profile=load_profile(profile_name) if profile_name else None,
        reference_tables=reference_tables,
        bundle=get_bundle(),
        table_name=table_name,
        label_column=label_column,
        config=AuditConfig(review_budget=budget, use_model=use_model,
                           max_record_issues=int(max_issues)),
    )


profile_file = {
    "erp_po_lines": "profile_erp_po_lines.yaml",
    "creditcard_transactions": "profile_creditcard.yaml",
}.get(table_name)

result = run_audit(frame, table_name, profile_file, review_budget, use_model,
                   max_issues, label_column, references)
report = result.report
quality = result.quality

# --- header ---------------------------------------------------------------------

st.title("Data Quality & Anomaly Audit")
st.caption(f"Batch `{report.batch_id}` — {report.n_rows:,} rows, {quality.n_columns} columns")

for note in result.notes:
    st.info(note, icon=None)

overview, quality_tab, anomaly_tab, issues_tab = st.tabs(
    ["Overview", "Quality detail", "Anomalies", "Issues & report"]
)

with overview:
    left, right = st.columns([1, 2])
    with left:
        st.markdown(score_badge(quality.global_score, quality.grade), unsafe_allow_html=True)
        st.caption(
            f"Weighted geometric mean. An arithmetic mean would read "
            f"{quality.global_score_arithmetic:.1f}."
        )
    with right:
        counts = report.severity_counts
        columns = st.columns(4)
        for column, (severity, count) in zip(columns, counts.items()):
            column.metric(severity.capitalize(), count)

    dimensions = quality.dimension_table()
    if not dimensions.empty:
        figure = px.bar(
            dimensions, x="dimension", y="score", text="score",
            color="score", color_continuous_scale=["#b3261e", "#f2b705", "#2e7d32"],
            range_color=(60, 100),
        )
        figure.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        figure.update_layout(
            yaxis_range=[min(60, dimensions["score"].min() - 5), 104],
            yaxis_title="Score", xaxis_title=None, coloraxis_showscale=False,
            height=330, margin=dict(t=30, b=10),
        )
        st.plotly_chart(figure, use_container_width=True)

    failed = len(quality.failed_rules)
    st.markdown(
        f"**Verdict.** {failed} of {len(quality.rules)} rules found at least one "
        f"violation, producing {len(report.issues)} prioritized issues. "
        + ("A duplicated primary key caps the grade at F regardless of the score."
           if quality.grade == "F" and quality.global_score >= 60 else "")
    )

with quality_tab:
    st.subheader("Rules by dimension")
    for name, dimension in quality.dimension_scores.items():
        if not dimension.rules:
            continue
        label = (f"{name} — {dimension.score * 100:.1f}/100 "
                 f"({dimension.n_failed_rules} of {len(dimension.rules)} rules failing)")
        with st.expander(label, expanded=dimension.n_failed_rules > 0):
            frame_rules = pd.DataFrame([
                {
                    "rule": rule.rule_id,
                    "column": rule.column or "-",
                    "checked": rule.n_checked,
                    "failed": rule.n_failed,
                    "score": round(rule.score, 4),
                    "description": rule.description,
                }
                for rule in sorted(dimension.rules, key=lambda r: r.score)
            ])
            st.dataframe(frame_rules, use_container_width=True, hide_index=True)

with anomaly_tab:
    if result.scores is None:
        st.info("Anomaly detection did not run for this batch. Enable it in the "
                "sidebar, or train a model with `python scripts/03_train_models.py`.")
    else:
        scores = result.scores
        threshold = result.threshold
        flagged = int(result.flags.sum())

        columns = st.columns(4)
        columns[0].metric("Records flagged", f"{flagged:,}")
        columns[1].metric("Flag rate", f"{flagged / len(scores):.2%}")
        columns[2].metric("Review hours", f"{flagged * 90 / 3600:.1f}")
        summary = report.anomaly_summary
        if summary.get("n_labelled_positives") is not None:
            columns[3].metric(
                "Known positives caught",
                f"{summary.get('n_caught', 0)}/{summary['n_labelled_positives']}",
            )
        else:
            columns[3].metric("Model", result.anomaly_model or "n/a")

        st.caption(f"Threshold {threshold.value:.6g} — {threshold.rule}")

        positive = scores[scores > 0]
        figure = go.Figure()
        figure.add_trace(go.Histogram(
            x=np.log10(positive) if len(positive) else scores,
            nbinsx=70, marker_color="#8c8c8c", name="all records",
        ))
        if threshold.value > 0:
            figure.add_vline(
                x=float(np.log10(threshold.value)), line_color="#1f4e79", line_width=2,
                annotation_text="threshold", annotation_position="top",
            )
        figure.update_layout(
            xaxis_title="log10(anomaly score)", yaxis_title="Records", yaxis_type="log",
            height=360, margin=dict(t=30, b=10), showlegend=False,
        )
        st.plotly_chart(figure, use_container_width=True)

        st.subheader("Highest-scoring records")
        top = np.argsort(-scores)[:50]
        preview = frame.iloc[top].copy()
        preview.insert(0, "anomaly_score", scores[top])
        st.dataframe(preview, use_container_width=True, height=340)

with issues_tab:
    frame_issues = report.issues_frame()
    if frame_issues.empty:
        st.success("No issues found in this batch.")
    else:
        chosen = st.multiselect(
            "Severity", ["critical", "high", "medium", "low"],
            default=["critical", "high", "medium"],
        )
        visible = frame_issues[frame_issues["severity"].isin(chosen)] if chosen else frame_issues
        st.caption(f"{len(visible):,} of {len(frame_issues):,} issues shown, "
                   "ordered by priority.")
        st.dataframe(
            visible[["priority", "severity", "severity_score", "dimension", "rule_id",
                     "column", "n_records", "reason", "recommended_action"]],
            use_container_width=True, height=430, hide_index=True,
            column_config={
                "severity_score": st.column_config.ProgressColumn(
                    "severity score", min_value=0, max_value=100, format="%.0f"),
                "reason": st.column_config.TextColumn("reason", width="large"),
                "recommended_action": st.column_config.TextColumn("recommended action",
                                                                  width="large"),
            },
        )

        left, right = st.columns(2)
        left.download_button("Download report (Markdown)", report.to_markdown(),
                             file_name=f"{report.batch_id}_report.md", use_container_width=True)
        right.download_button("Download report (JSON)", report.to_json(),
                              file_name=f"{report.batch_id}_report.json",
                              use_container_width=True)
