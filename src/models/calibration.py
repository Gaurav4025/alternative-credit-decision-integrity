"""Probability calibration experiments for baseline risk models.

This module evaluates calibration only. It does not implement confidence
estimation, OOD detection, evidence-quality scoring, or a safety gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.evaluation.metrics import (
    calibration_diagnostics,
    evaluate_probability_model,
    expected_calibration_error,
)
from src.models.baseline import (
    BASELINE_FEATURES,
    SELECTED_THRESHOLD,
    TARGET_COLUMN,
    load_model_dataset,
    make_train_validation_test_split,
    predict_probabilities,
    train_baseline_models,
)
from src.utils.reproducibility import DEFAULT_RANDOM_SEED

CALIBRATION_METHODS = ("raw", "sigmoid", "isotonic")
ECE_BINS = 10


def fit_calibrated_models(
    models: dict[str, Any],
    validation: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Fit Platt and isotonic calibration on validation data only."""

    x_validation = validation[list(BASELINE_FEATURES)]
    y_validation = validation[TARGET_COLUMN]
    calibrated: dict[str, dict[str, Any]] = {}

    for model_name, model in models.items():
        calibrated[model_name] = {"raw": model}
        for method in ("sigmoid", "isotonic"):
            calibrator = CalibratedClassifierCV(
                estimator=FrozenEstimator(model),
                method=method,
            )
            calibrator.fit(x_validation, y_validation)
            calibrated[model_name][method] = calibrator

    return calibrated


def evaluate_calibration_variants(
    calibrated_models: dict[str, dict[str, Any]],
    test: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate raw and calibrated probabilities on the held-out test set."""

    metrics: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name, variants in calibrated_models.items():
        metrics[model_name] = {}
        for calibration_method, model in variants.items():
            predictions = predict_probabilities(model, test)
            model_metrics = evaluate_probability_model(
                predictions[TARGET_COLUMN],
                predictions["predicted_probability"],
                threshold=SELECTED_THRESHOLD,
            )
            ece = expected_calibration_error(
                predictions[TARGET_COLUMN],
                predictions["predicted_probability"],
                n_bins=ECE_BINS,
            )
            diagnostics = calibration_diagnostics(
                predictions[TARGET_COLUMN],
                predictions["predicted_probability"],
                n_bins=ECE_BINS,
            )
            model_metrics["ece"] = ece

            metrics[model_name][calibration_method] = {
                "test_metrics": model_metrics,
                "calibration_diagnostics": diagnostics,
            }
            comparison_rows.append(
                {
                    "model": model_name,
                    "calibration_method": calibration_method,
                    "roc_auc": model_metrics["roc_auc"],
                    "pr_auc": model_metrics["pr_auc"],
                    "brier_score": model_metrics["brier_score"],
                    "log_loss": model_metrics["log_loss"],
                    "ece": model_metrics["ece"],
                    "positive_prediction_rate": model_metrics[
                        "positive_prediction_rate"
                    ],
                    "actual_event_rate": model_metrics["actual_event_rate"],
                }
            )
            prediction_frames.append(
                predictions.assign(
                    model=model_name,
                    calibration_method=calibration_method,
                )
            )

    return (
        metrics,
        pd.DataFrame(comparison_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def run_calibration_experiment(
    dataset_path: str | Path = "data/processed/synthetic/model_dataset.csv",
    output_dir: str | Path = "experiments/calibration",
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Train baseline models, fit calibrators on validation, evaluate on test."""

    dataset = load_model_dataset(dataset_path)
    splits = make_train_validation_test_split(dataset, seed=seed)
    models = train_baseline_models(splits["train"], seed=seed)
    calibrated_models = fit_calibrated_models(models, splits["validation"])
    model_metrics, comparison, predictions = evaluate_calibration_variants(
        calibrated_models,
        splits["test"],
    )

    output_path = Path(output_dir)
    model_path = output_path / "models"
    output_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {
        "seed": seed,
        "split_strategy": "70/15/15 stratified on financial_stress_event",
        "calibration_protocol": (
            "baseline models fit on train; sigmoid and isotonic calibrators fit "
            "on validation only; test used only for final reporting"
        ),
        "ece_bins": ECE_BINS,
        "ece_binning_strategy": "10 fixed-width bins over [0, 1]; empty bins ignored",
        "features": list(BASELINE_FEATURES),
        "split_sizes": {name: len(split) for name, split in splits.items()},
        "models": model_metrics,
    }

    comparison.to_csv(output_path / "calibration_comparison.csv", index=False)
    predictions.to_csv(output_path / "calibration_predictions.csv", index=False)
    with (output_path / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
        file.write("\n")

    for model_name, variants in calibrated_models.items():
        for method, model in variants.items():
            joblib.dump(model, model_path / f"{model_name}_{method}.joblib")

    return metrics


if __name__ == "__main__":
    run_calibration_experiment()
