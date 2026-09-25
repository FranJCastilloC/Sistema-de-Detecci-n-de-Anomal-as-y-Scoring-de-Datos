"""Figures and secondary tables for the model comparison and the quality engine."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from _bootstrap import ensure_src_on_path  # noqa: E402  (must precede dq_anomaly)

ensure_src_on_path()

from dq_anomaly.config import PATHS, load_yaml, set_global_seed
from dq_anomaly.data.defect_injector import inject_defects
from dq_anomaly.data.erp_generator import generate_erp_dataset
from dq_anomaly.quality.defect_eval import (
    evaluate_defect_recall,
    evaluate_rule_precision,
    flagged_cells,
)
from dq_anomaly.quality.profile import load_profile
from dq_anomaly.quality.scorer import score_dataframe

PRIMARY = "#1f4e79"
SECONDARY = "#c1512d"
MUTED = "#8c8c8c"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

AE_KEY = "Autoencoder_(trained_on_normals)"
IF_KEY = "Isolation_Forest_(trained_on_normals)"


def _load_scores() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(PATHS.metrics / "test_scores.npz") as archive:
        y_test = archive["y_test"]
        scores = {key: archive[key] for key in archive.files if key != "y_test"}
    return y_test, scores


def plot_curves(y_test: np.ndarray, scores: dict[str, np.ndarray]) -> None:
    prevalence = float(y_test.mean())
    figure, (pr_axis, roc_axis) = plt.subplots(1, 2, figsize=(11, 4.2))

    for key, colour, style in (
        (AE_KEY, PRIMARY, "-"), (IF_KEY, SECONDARY, "--"),
        ("Logistic_Regression_(supervised_ceiling)", MUTED, ":"),
    ):
        if key not in scores:
            continue
        label = key.replace("_", " ")
        precision, recall, _ = precision_recall_curve(y_test, scores[key])
        pr_axis.plot(recall, precision, style, color=colour, label=label, linewidth=1.6)
        fpr, tpr, _ = roc_curve(y_test, scores[key])
        roc_axis.plot(fpr, tpr, style, color=colour, label=label, linewidth=1.6)

    pr_axis.axhline(prevalence, color="black", linestyle=(0, (1, 3)), linewidth=1,
                    label=f"random baseline ({prevalence:.4f})")
    pr_axis.set(xlabel="Recall", ylabel="Precision",
                title="Precision-Recall (the metric that matters here)")
    pr_axis.legend(fontsize=7.5, loc="upper right")

    roc_axis.plot([0, 1], [0, 1], color="black", linestyle=(0, (1, 3)), linewidth=1)
    roc_axis.set(xlabel="False positive rate", ylabel="True positive rate",
                 title="ROC (flattering at 0.17% prevalence)")
    roc_axis.legend(fontsize=7.5, loc="lower right")

    figure.suptitle("Fraud detection on the held-out test window", fontsize=11)
    figure.tight_layout()
    figure.savefig(PATHS.figures / "pr_roc_curves.png", bbox_inches="tight")
    plt.close(figure)


def plot_score_distribution(y_test: np.ndarray, scores: dict[str, np.ndarray],
                            threshold: float) -> None:
    values = scores[AE_KEY]
    figure, axis = plt.subplots(figsize=(8, 4.2))
    bins = np.logspace(np.log10(max(values.min(), 1e-6)), np.log10(values.max()), 70)
    axis.hist(values[y_test == 0], bins=bins, color=MUTED, alpha=0.75,
              label=f"legitimate (n={int((y_test == 0).sum()):,})")
    axis.hist(values[y_test == 1], bins=bins, color=SECONDARY, alpha=0.85,
              label=f"fraud (n={int(y_test.sum())})")
    axis.axvline(threshold, color=PRIMARY, linewidth=1.8,
                 label="review budget threshold (0.5%)")
    axis.set(xscale="log", yscale="log", xlabel="Reconstruction error (anomaly score)",
             ylabel="Records", title="Autoencoder score distribution on the test window")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(PATHS.figures / "score_distribution.png", bbox_inches="tight")
    plt.close(figure)


def plot_training_history(payload: dict) -> None:
    history = payload.get("autoencoder_history", {})
    if not history.get("train_loss"):
        return
    figure, axis = plt.subplots(figsize=(7, 3.8))
    axis.plot(history["train_loss"], color=PRIMARY, label="train", linewidth=1.4)
    axis.plot(history["val_loss"], color=SECONDARY, label="validation", linewidth=1.4)
    axis.set(xlabel="Epoch", ylabel="MSE", yscale="log",
             title="Autoencoder training (normal transactions only)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(PATHS.figures / "training_history.png", bbox_inches="tight")
    plt.close(figure)


def plot_reviewer_workload(payload: dict) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    for name, colour, marker in (
        ("Autoencoder (trained on normals)", PRIMARY, "o"),
        ("Isolation Forest (trained on normals)", SECONDARY, "s"),
    ):
        sweep = payload["oracle_and_sweeps"].get(name, {}).get("budget_sweep", [])
        if not sweep:
            continue
        hours = [entry["review_hours"] for entry in sweep]
        recall = [entry["recall"] for entry in sweep]
        axis.plot(hours, recall, marker=marker, color=colour, label=name, linewidth=1.6)
        for entry, x, y in zip(sweep, hours, recall):
            axis.annotate(f"{entry['flag_rate']:.1%}", (x, y), fontsize=7,
                          xytext=(3, -9), textcoords="offset points", color=colour)
    axis.set(xlabel="Manual review hours for the batch (90 s per alert)",
             ylabel="Share of fraud caught",
             title="What each review budget buys")
    axis.legend(fontsize=8, loc="lower right")
    axis.grid(alpha=0.25, linewidth=0.5)
    figure.tight_layout()
    figure.savefig(PATHS.figures / "reviewer_workload.png", bbox_inches="tight")
    plt.close(figure)


def plot_quality_dimensions(scores_by_table: dict[str, dict]) -> None:
    tables = list(scores_by_table)
    dimensions = ["completeness", "validity", "consistency", "uniqueness", "plausibility"]
    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    width = 0.15
    positions = np.arange(len(dimensions))
    palette = [PRIMARY, SECONDARY, "#4c8c5a", MUTED]
    for index, table in enumerate(tables):
        values = [scores_by_table[table]["dimensions"][d] * 100 for d in dimensions]
        axis.bar(positions + index * width, values, width, label=table,
                 color=palette[index % len(palette)])
    axis.set(xticks=positions + width * 1.5, ylim=(90, 100.5), ylabel="Dimension score",
             title="Data quality by dimension on the corrupted ERP extract")
    axis.set_xticklabels(dimensions)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(PATHS.figures / "quality_dimensions.png", bbox_inches="tight")
    plt.close(figure)


def evaluate_quality_engine() -> dict[str, dict]:
    """Score the quality rules against the injected ground truth."""
    clean = generate_erp_dataset(seed=42)
    dirty, ledger = inject_defects(clean, seed=1337)
    tables = ["po_lines", "purchase_orders", "vendors", "materials"]

    findings, summary = [], {}
    for table in tables:
        profile = load_profile(f"profile_erp_{table}.yaml")
        report = score_dataframe(dirty[table], profile, reference_tables=dirty)
        findings.append(flagged_cells(dirty[table], report.rules, table))
        summary[table] = {
            "global_score": round(report.global_score, 2),
            "global_score_arithmetic": round(report.global_score_arithmetic, 2),
            "grade": report.grade,
            "dimensions": {
                name: round(score.score, 6)
                for name, score in report.dimension_scores.items()
            },
        }

    all_findings = pd.concat(findings, ignore_index=True)
    recall = evaluate_defect_recall(ledger, all_findings)
    precision = evaluate_rule_precision(ledger, all_findings)
    recall.to_csv(PATHS.metrics / "defect_recall.csv", index=False)
    precision.to_csv(PATHS.metrics / "rule_precision.csv", index=False)

    deterministic = precision.loc[~precision["is_statistical"]]
    (PATHS.metrics / "quality_engine_summary.json").write_text(json.dumps({
        "tables": summary,
        "defects_injected": int(recall["injected"].sum()),
        "defects_detected": int(recall["detected"].sum()),
        "overall_recall": round(float(recall["detected"].sum() / recall["injected"].sum()), 4),
        "deterministic_rule_precision": round(
            float(deterministic["true_positives"].sum() / deterministic["flagged"].sum()), 4
        ),
        "families_at_full_recall": int((recall["recall"] == 1.0).sum()),
        "n_families": int(len(recall)),
    }, indent=2), encoding="utf-8")

    print(recall.to_string(index=False))
    print(f"\nOverall defect recall: "
          f"{recall['detected'].sum() / recall['injected'].sum():.2%}")
    return summary


def main() -> None:
    PATHS.ensure()
    set_global_seed(int(load_yaml("model.yaml")["seed"]))

    payload = json.loads((PATHS.metrics / "model_comparison.json").read_text(encoding="utf-8"))
    y_test, scores = _load_scores()
    threshold = payload["results"]["Autoencoder (trained on normals)"]["threshold_value"]

    plot_curves(y_test, scores)
    plot_score_distribution(y_test, scores, threshold)
    plot_training_history(payload)
    plot_reviewer_workload(payload)

    summary = evaluate_quality_engine()
    plot_quality_dimensions(summary)

    table = pd.read_csv(PATHS.metrics / "model_comparison.csv")
    markdown = table.drop(columns=["threshold_name", "threshold_is_oracle"]).round(4)
    (PATHS.metrics / "model_comparison.md").write_text(
        "# Baseline vs autoencoder\n\n"
        "Held-out temporal test window, identical features, identical 0.5% alert budget.\n\n"
        + markdown.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nFigures written to {PATHS.figures}")


if __name__ == "__main__":
    main()
