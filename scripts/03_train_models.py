"""Train and evaluate every detector, then write the comparison table.

Four unsupervised configurations are compared on identical features and an
identical alert budget, plus a supervised ceiling and a random floor so the
unsupervised numbers can be read against something.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from _bootstrap import ensure_src_on_path  # noqa: E402  (must precede dq_anomaly)

ensure_src_on_path()

from dq_anomaly.anomaly.autoencoder import AEConfig, AutoencoderDetector
from dq_anomaly.anomaly.evaluate import bootstrap_interval, comparison_table, evaluate_scores
from dq_anomaly.anomaly.isolation_forest import IFConfig, IsolationForestDetector
from dq_anomaly.anomaly.preprocess import CreditCardPreprocessor, temporal_split
from dq_anomaly.anomaly.thresholds import (
    threshold_best_f1,
    threshold_from_review_budget,
    threshold_from_train_percentile,
)
from dq_anomaly.config import PATHS, load_yaml, set_global_seed
from dq_anomaly.data.loaders import load_creditcard


def main() -> None:
    PATHS.ensure()
    config = load_yaml("model.yaml")
    seed = int(config["seed"])
    set_global_seed(seed)

    frame = load_creditcard()
    train, calibration, test = temporal_split(
        frame,
        train_frac=config["split"]["train_frac"],
        calib_frac=config["split"]["calibration_frac"],
    )
    train_normal = train.loc[train["Class"] == 0]
    print(f"train {len(train):,} ({int(train['Class'].sum())} fraud) | "
          f"calibration {len(calibration):,} ({int(calibration['Class'].sum())} fraud) | "
          f"test {len(test):,} ({int(test['Class'].sum())} fraud)")

    # The scaler sees only the rows the model is allowed to learn from. Fitting
    # it on the full dataset is the leak that inflates most published results.
    preprocessor = CreditCardPreprocessor().fit(train_normal)
    X_train_normal = preprocessor.transform(train_normal)
    X_train_all = preprocessor.transform(train)
    X_calibration = preprocessor.transform(calibration)
    X_test = preprocessor.transform(test)
    y_test = test["Class"].to_numpy()
    print(f"features: {len(preprocessor.feature_names_)} -> {preprocessor.feature_names_}")

    ae_config = config["autoencoder"]
    detectors: dict[str, object] = {}

    started = time.perf_counter()
    ae_clean = AutoencoderDetector(AEConfig(
        input_dim=X_train_normal.shape[1], hidden=tuple(ae_config["hidden"]),
        latent=ae_config["latent"], dropout=ae_config["dropout"], lr=ae_config["lr"],
        weight_decay=ae_config["weight_decay"], batch_size=ae_config["batch_size"],
        max_epochs=ae_config["max_epochs"], patience=ae_config["patience"],
        min_delta=ae_config["min_delta"], val_fraction=ae_config["val_fraction"],
        seed=seed, num_threads=ae_config["num_threads"],
    )).fit(X_train_normal)
    print(f"autoencoder (normals only): best epoch {ae_clean.best_epoch_}, "
          f"{time.perf_counter() - started:.1f}s")
    detectors["Autoencoder (trained on normals)"] = ae_clean

    started = time.perf_counter()
    ae_contaminated = AutoencoderDetector(AEConfig(
        input_dim=X_train_all.shape[1], hidden=tuple(ae_config["hidden"]),
        latent=ae_config["latent"], lr=ae_config["lr"],
        batch_size=ae_config["batch_size"], max_epochs=ae_config["max_epochs"],
        patience=ae_config["patience"], seed=seed, num_threads=ae_config["num_threads"],
    )).fit(X_train_all)
    print(f"autoencoder (contaminated): best epoch {ae_contaminated.best_epoch_}, "
          f"{time.perf_counter() - started:.1f}s")
    detectors["Autoencoder (fully unsupervised)"] = ae_contaminated

    if_config = IFConfig(
        n_estimators=config["isolation_forest"]["n_estimators"],
        max_samples=config["isolation_forest"]["max_samples"],
        max_features=config["isolation_forest"]["max_features"],
        contamination=config["isolation_forest"]["contamination"],
        random_state=seed,
    )
    started = time.perf_counter()
    if_clean = IsolationForestDetector(if_config).fit(X_train_normal)
    if_all = IsolationForestDetector(if_config).fit(X_train_all)
    print(f"isolation forests: {time.perf_counter() - started:.1f}s")
    detectors["Isolation Forest (trained on normals)"] = if_clean
    detectors["Isolation Forest (fully unsupervised)"] = if_all

    budget = float(config["thresholds"]["review_budget"])
    results: dict[str, dict] = {}
    scores_test: dict[str, np.ndarray] = {}
    extras: dict[str, dict] = {}

    for name, detector in detectors.items():
        calibration_scores = detector.score(X_calibration)
        test_scores = detector.score(X_test)
        scores_test[name] = test_scores
        threshold = threshold_from_review_budget(calibration_scores, budget)
        metrics = evaluate_scores(y_test, test_scores, threshold)
        metrics.update(bootstrap_interval(
            y_test, test_scores, threshold.value,
            n_resamples=config["evaluation"]["bootstrap_resamples"], seed=seed,
        ))
        results[name] = metrics

        oracle = threshold_best_f1(test_scores, y_test)
        extras[name] = {
            "oracle_best_f1": evaluate_scores(y_test, test_scores, oracle),
            "budget_sweep": [
                evaluate_scores(
                    y_test, test_scores,
                    threshold_from_review_budget(calibration_scores, sweep_budget),
                )
                for sweep_budget in config["thresholds"]["budget_sweep"]
            ],
        }

    # Supervised ceiling: what the same features yield when labels do exist.
    supervised = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    supervised.fit(X_train_all, train["Class"].to_numpy())
    supervised_calibration = supervised.predict_proba(X_calibration)[:, 1]
    supervised_test = supervised.predict_proba(X_test)[:, 1]
    scores_test["Logistic Regression (supervised ceiling)"] = supervised_test
    results["Logistic Regression (supervised ceiling)"] = evaluate_scores(
        y_test, supervised_test, threshold_from_review_budget(supervised_calibration, budget)
    )

    rng = np.random.default_rng(seed)
    random_scores = rng.random(len(y_test))
    scores_test["Random (floor)"] = random_scores
    results["Random (floor)"] = evaluate_scores(
        y_test, random_scores, threshold_from_review_budget(rng.random(len(X_calibration)), budget)
    )

    # Demonstrate why a train-derived threshold is the wrong choice.
    train_scores = ae_clean.score(X_train_normal)
    train_threshold = threshold_from_train_percentile(train_scores, 99.0)
    train_threshold_metrics = evaluate_scores(y_test, scores_test[
        "Autoencoder (trained on normals)"], train_threshold)

    table = comparison_table(results)
    table.to_csv(PATHS.metrics / "model_comparison.csv", index=False)
    payload = {
        "dataset": "ULB credit card fraud (OpenML data_id=1597)",
        "split": "temporal 60/20/20 (train / calibration / test)",
        "review_budget": budget,
        "headline_threshold": "budget@0.50% set on the unlabelled calibration window",
        "results": results,
        "oracle_and_sweeps": extras,
        "train_percentile_threshold_demo": {
            "threshold": train_threshold.__dict__,
            "metrics": train_threshold_metrics,
        },
        "autoencoder_history": ae_clean.history_,
        "feature_names": preprocessor.feature_names_,
    }
    (PATHS.metrics / "model_comparison.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )

    preprocessor.save(PATHS.models / "preprocessor.joblib")
    ae_clean.save(PATHS.models / "autoencoder.pt")
    if_clean.save(PATHS.models / "isolation_forest.joblib")
    np.savez_compressed(
        PATHS.models / "calibration_scores.npz",
        autoencoder=ae_clean.score(X_calibration),
        isolation_forest=if_clean.score(X_calibration),
    )
    (PATHS.models / "model_card.json").write_text(json.dumps({
        "trained_on": "ULB credit card fraud, temporal train window, normal rows only",
        "n_train_rows": int(len(train_normal)),
        "features": preprocessor.feature_names_,
        "autoencoder": {"best_epoch": ae_clean.best_epoch_, "config": ae_clean.config.__dict__},
        "review_budget": budget,
        "headline_metrics": {
            key: results["Autoencoder (trained on normals)"][key]
            for key in ("pr_auc", "roc_auc", "precision", "recall", "f1", "alerts")
        },
        "seed": seed,
        "determinism_note": (
            "Reproducible for a fixed torch thread count (4); float reduction "
            "order changes with thread count."
        ),
    }, indent=2, default=float), encoding="utf-8")

    np.savez_compressed(
        PATHS.metrics / "test_scores.npz",
        y_test=y_test,
        **{name.replace(" ", "_"): values for name, values in scores_test.items()},
    )

    pd.set_option("display.width", 200)
    print()
    print(table.drop(columns=["threshold_name", "threshold_is_oracle"]).to_string(index=False))


if __name__ == "__main__":
    main()
