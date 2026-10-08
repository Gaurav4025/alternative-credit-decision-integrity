"""Conventional baseline risk models for synthetic financial stress prediction.

This module intentionally implements only the baseline benchmark:

alternative financial features -> risk probability -> automated decision.

It does not implement calibration correction, uncertainty estimation,
out-of-distribution detection, evidence-quality scoring, or a safety gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.validation import assert_no_forbidden_credit_variables
from src.evaluation.metrics import (
    calibration_diagnostics,
    evaluate_probability_model,
    threshold_sensitivity,
)
from src.features.alternative_features import FEATURE_COLUMNS
from src.utils.reproducibility import DEFAULT_RANDOM_SEED

TARGET_COLUMN = "financial_stress_event"
PROBABILITY_COLUMN = "financial_stress_probability"
BASELINE_FEATURES = tuple(FEATURE_COLUMNS)
SELECTED_THRESHOLD = 0.5
SENSITIVITY_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def load_model_dataset(path: str | Path) -> pd.DataFrame:
    """Load the processed model dataset and validate baseline columns."""

    dataset = pd.read_csv(path)
    validate_baseline_dataset(dataset)
    return dataset


def validate_baseline_dataset(dataset: pd.DataFrame) -> None:
    """Validate that the baseline dataset has no obvious leakage columns."""

    required_columns = {"applicant_id", *BASELINE_FEATURES, TARGET_COLUMN}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"Missing baseline dataset columns: {sorted(missing_columns)}")

    assert_no_forbidden_credit_variables(dataset)

    forbidden_inputs = {
        "applicant_id",
        "population_group",
        PROBABILITY_COLUMN,
        TARGET_COLUMN,
    }
    leaked_features = forbidden_inputs.intersection(BASELINE_FEATURES)
    if leaked_features:
        raise ValueError(f"Leakage columns present in baseline features: {leaked_features}")


def make_train_validation_test_split(
    dataset: pd.DataFrame,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Create a deterministic 70/15/15 stratified split.

    The split first separates 70% training rows from a 30% holdout, stratified on
    ``financial_stress_event``. The holdout is then split evenly into validation
    and test sets, again stratified on the target. The test set is not used for
    threshold selection.
    """

    validate_baseline_dataset(dataset)

    train, holdout = train_test_split(
        dataset,
        test_size=0.30,
        random_state=seed,
        stratify=dataset[TARGET_COLUMN],
    )
    validation, test = train_test_split(
        holdout,
        test_size=0.50,
        random_state=seed,
        stratify=holdout[TARGET_COLUMN],
    )

    return {
        "train": train.sort_values("applicant_id").reset_index(drop=True),
        "validation": validation.sort_values("applicant_id").reset_index(drop=True),
        "test": test.sort_values("applicant_id").reset_index(drop=True),
    }


def train_baseline_models(train: pd.DataFrame, seed: int = DEFAULT_RANDOM_SEED) -> dict[str, Any]:
    """Train logistic regression and HistGradientBoosting baselines."""

    x_train = train[list(BASELINE_FEATURES)]
    y_train = train[TARGET_COLUMN]

    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1_000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    tree = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=31,
        l2_regularization=0.0,
        random_state=seed,
    )

    return {
        "logistic_regression": logistic.fit(x_train, y_train),
        "hist_gradient_boosting": tree.fit(x_train, y_train),
    }


def predict_probabilities(model: Any, dataset: pd.DataFrame) -> pd.DataFrame:
    """Generate P(financial_stress_event = 1) for a dataset split."""

    probability = model.predict_proba(dataset[list(BASELINE_FEATURES)])[:, 1]
    return pd.DataFrame(
        {
            "applicant_id": dataset["applicant_id"].to_numpy(),
            "financial_stress_event": dataset[TARGET_COLUMN].to_numpy(),
            "predicted_probability": probability,
        }
    )


def run_baseline_experiment(
    dataset_path: str | Path = "data/processed/synthetic/model_dataset.csv",
    output_dir: str | Path = "experiments/baseline",
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Train, evaluate, and save baseline model artifacts."""

    dataset = load_model_dataset(dataset_path)
    splits = make_train_validation_test_split(dataset, seed=seed)
    models = train_baseline_models(splits["train"], seed=seed)

    output_path = Path(output_dir)
    model_path = output_path / "models"
    output_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {
        "seed": seed,
        "split_strategy": "70/15/15 stratified on financial_stress_event",
        "selected_threshold": SELECTED_THRESHOLD,
        "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
        "features": list(BASELINE_FEATURES),
        "split_sizes": {name: len(split) for name, split in splits.items()},
        "models": {},
    }
    comparison_rows: list[dict[str, Any]] = []

    for model_name, model in models.items():
        validation_predictions = predict_probabilities(model, splits["validation"])
        test_predictions = predict_probabilities(model, splits["test"])

        validation_sensitivity = threshold_sensitivity(
            validation_predictions[TARGET_COLUMN],
            validation_predictions["predicted_probability"],
            thresholds=SENSITIVITY_THRESHOLDS,
        )
        test_metrics = evaluate_probability_model(
            test_predictions[TARGET_COLUMN],
            test_predictions["predicted_probability"],
            threshold=SELECTED_THRESHOLD,
        )
        calibration = calibration_diagnostics(
            test_predictions[TARGET_COLUMN],
            test_predictions["predicted_probability"],
        )

        metrics["models"][model_name] = {
            "test_metrics": test_metrics,
            "validation_threshold_sensitivity": validation_sensitivity,
            "calibration_diagnostics": calibration,
            "configuration": _model_configuration(model_name),
        }

        comparison_rows.append(
            {
                "model": model_name,
                "roc_auc": test_metrics["roc_auc"],
                "pr_auc": test_metrics["pr_auc"],
                "brier_score": test_metrics["brier_score"],
                "log_loss": test_metrics["log_loss"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
            }
        )

        output_file = (
            output_path / "predictions_logistic.csv"
            if model_name == "logistic_regression"
            else output_path / "predictions_tree.csv"
        )
        test_predictions.to_csv(output_file, index=False)
        joblib.dump(model, model_path / f"{model_name}.joblib")

    pd.DataFrame(comparison_rows).to_csv(output_path / "model_comparison.csv", index=False)
    with (output_path / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
        file.write("\n")

    return metrics


def _model_configuration(model_name: str) -> dict[str, Any]:
    """Return the documented baseline model configuration."""

    if model_name == "logistic_regression":
        return {
            "estimator": "Pipeline(StandardScaler, LogisticRegression)",
            "solver": "lbfgs",
            "max_iter": 1000,
        }
    if model_name == "hist_gradient_boosting":
        return {
            "estimator": "HistGradientBoostingClassifier",
            "learning_rate": 0.05,
            "max_iter": 150,
            "max_leaf_nodes": 31,
        }
    raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    run_baseline_experiment()
