"""The single audit entry point shared by the CLI, the notebooks and the app."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dq_anomaly.anomaly.artifacts import ModelBundle, bundle_available, load_bundle
from dq_anomaly.anomaly.isolation_forest import IFConfig, IsolationForestDetector
from dq_anomaly.anomaly.thresholds import Threshold, threshold_from_review_budget
from dq_anomaly.quality.profile import (
    TableExpectation,
    detect_profile,
    infer_profile,
    load_bundled_profiles,
)
from dq_anomaly.quality.scorer import QualityReport, score_dataframe
from dq_anomaly.report.builder import BatchReport, build_batch_report, issues_from_anomalies
from dq_anomaly.report.issues import Issue, SeverityModel


@dataclass
class AuditConfig:
    review_budget: float = 0.005
    max_record_issues: int = 200
    use_model: bool = True
    fallback_isolation_forest: bool = True
    seed: int = 42


@dataclass
class AuditResult:
    """The outcome of auditing one batch."""

    report: BatchReport
    quality: QualityReport
    profile: TableExpectation
    scores: np.ndarray | None = None
    flags: np.ndarray | None = None
    threshold: Threshold | None = None
    anomaly_model: str | None = None
    notes: list[str] = field(default_factory=list)


def resolve_profile(
    df: pd.DataFrame, profile: TableExpectation | None = None, name: str = "uploaded_batch"
) -> TableExpectation:
    """Use the given profile, else a matching bundled one, else infer."""
    if profile is not None:
        return profile
    try:
        detected = detect_profile(df, load_bundled_profiles())
    except Exception:  # noqa: BLE001 - a broken profile must not block an audit
        detected = None
    return detected or infer_profile(df, name)


def _score_with_bundle(
    df: pd.DataFrame, bundle: ModelBundle, budget: float
) -> tuple[np.ndarray, Threshold, np.ndarray, np.ndarray, np.ndarray, str]:
    X = bundle.preprocessor.transform(df)
    scores = bundle.autoencoder.score(X)
    calibration = bundle.calibration_scores.get("autoencoder")
    reference = calibration if calibration is not None and len(calibration) else scores
    threshold = threshold_from_review_budget(reference, budget)
    per_feature = bundle.autoencoder.per_feature_errors(X)
    reconstructions = bundle.autoencoder.reconstruct(X)
    return scores, threshold, per_feature, X, reconstructions, "autoencoder"


def _score_with_fallback(
    df: pd.DataFrame, budget: float, seed: int
) -> tuple[np.ndarray, Threshold, None, None, None, str] | None:
    """Fit an Isolation Forest on this batch when no trained model fits its schema.

    Clearly a different thing from a trained model: it can only find records that
    are unusual relative to the batch in front of it.
    """
    numeric = df.select_dtypes(include=["number"]).apply(
        lambda column: pd.to_numeric(column, errors="coerce")
    )
    numeric = numeric.loc[:, numeric.notna().mean() > 0.5]
    if numeric.shape[1] < 2 or len(numeric) < 50:
        return None
    filled = numeric.fillna(numeric.median(numeric_only=True)).to_numpy(dtype="float64")
    detector = IsolationForestDetector(IFConfig(
        max_samples=min(8192, len(filled)), random_state=seed
    )).fit(filled)
    scores = detector.score(filled)
    threshold = threshold_from_review_budget(scores, budget)
    return scores, threshold, None, None, None, "isolation_forest_fitted_on_batch"


def audit_batch(
    df: pd.DataFrame,
    profile: TableExpectation | None = None,
    reference_tables: dict[str, pd.DataFrame] | None = None,
    bundle: ModelBundle | None = None,
    config: AuditConfig | None = None,
    batch_id: str | None = None,
    table_name: str | None = None,
    label_column: str | None = None,
) -> AuditResult:
    """Score a batch for quality, detect anomalies, and build the issue register."""
    config = config or AuditConfig()
    df = df.reset_index(drop=True)
    profile = resolve_profile(df, profile, table_name or "uploaded_batch")
    table = table_name or profile.name
    notes: list[str] = []
    if profile.inferred:
        notes.append(
            "No curated profile matched this batch, so expectations were inferred "
            "from the data itself. That detects internal inconsistency, not "
            "incorrectness."
        )

    quality = score_dataframe(df, profile, reference_tables)

    severity_model = SeverityModel()
    anomaly_issues: list[Issue] = []
    anomaly_summary: dict[str, Any] = {}
    scores = flags = None
    threshold = None
    model_name = None

    if config.use_model:
        scored = None
        if bundle is None and bundle_available():
            try:
                bundle = load_bundle()
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Trained model could not be loaded: {exc}")
        if bundle is not None:
            required = {"Time", "Amount", *[f"V{i}" for i in range(1, 29)]}
            if required.issubset(set(df.columns)):
                scored = _score_with_bundle(df, bundle, config.review_budget)
            else:
                notes.append(
                    "The trained autoencoder expects the credit card transaction "
                    "schema; this batch does not match it."
                )
        if scored is None and config.fallback_isolation_forest:
            scored = _score_with_fallback(df, config.review_budget, config.seed)
            if scored is not None:
                notes.append(
                    "Anomaly scores come from an Isolation Forest fitted on this "
                    "batch, not from a trained model."
                )
        if scored is not None:
            scores, threshold, per_feature, matrix, reconstructions, model_name = scored
            flags = scores >= threshold.value
            anomaly_issues, anomaly_summary = issues_from_anomalies(
                df=df, scores=scores, threshold_value=threshold.value,
                threshold_rule=threshold.rule, profile=profile,
                severity_model=severity_model, table=table, model_name=model_name,
                per_feature_errors=per_feature,
                feature_names=bundle.feature_names if (bundle and per_feature is not None) else None,
                feature_values=matrix, reconstructions=reconstructions,
                max_issues=config.max_record_issues,
            )
            if label_column and label_column in df.columns:
                labels = pd.to_numeric(df[label_column], errors="coerce").fillna(0).to_numpy()
                anomaly_summary["n_labelled_positives"] = int(labels.sum())
                anomaly_summary["n_caught"] = int(((labels == 1) & flags).sum())

    report = build_batch_report(
        df=df, quality=quality, profile=profile, table=table, batch_id=batch_id,
        anomaly_issues=anomaly_issues, anomaly_summary=anomaly_summary,
        severity_model=severity_model,
    )
    return AuditResult(
        report=report, quality=quality, profile=profile, scores=scores,
        flags=flags, threshold=threshold, anomaly_model=model_name, notes=notes,
    )
