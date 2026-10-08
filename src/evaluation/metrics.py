"""Evaluation metrics for baseline probability models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_probability_model(
    y_true: pd.Series | np.ndarray,
    y_probability: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Evaluate binary event probabilities at a documented threshold."""

    y_true_array = np.asarray(y_true)
    y_probability_array = np.asarray(y_probability)
    y_pred = (y_probability_array >= threshold).astype(int)
    matrix = confusion_matrix(y_true_array, y_pred, labels=[0, 1])

    return {
        "roc_auc": float(roc_auc_score(y_true_array, y_probability_array)),
        "pr_auc": float(average_precision_score(y_true_array, y_probability_array)),
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "precision": float(
            precision_score(y_true_array, y_pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "brier_score": float(brier_score_loss(y_true_array, y_probability_array)),
        "log_loss": float(log_loss(y_true_array, y_probability_array, labels=[0, 1])),
        "positive_prediction_rate": float(y_pred.mean()),
        "actual_event_rate": float(y_true_array.mean()),
        "threshold": float(threshold),
    }


def threshold_sensitivity(
    y_true: pd.Series | np.ndarray,
    y_probability: pd.Series | np.ndarray,
    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
) -> list[dict[str, float]]:
    """Evaluate classification tradeoffs across candidate thresholds."""

    rows: list[dict[str, float]] = []
    y_true_array = np.asarray(y_true)
    y_probability_array = np.asarray(y_probability)
    for threshold in thresholds:
        y_pred = (y_probability_array >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true_array, y_pred)),
                "precision": float(
                    precision_score(y_true_array, y_pred, zero_division=0)
                ),
                "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
                "positive_prediction_rate": float(y_pred.mean()),
            }
        )
    return rows


def calibration_diagnostics(
    y_true: pd.Series | np.ndarray,
    y_probability: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Return reliability-diagram bins without applying calibration correction."""

    observed_rate, predicted_probability = calibration_curve(
        y_true,
        y_probability,
        n_bins=n_bins,
        strategy="uniform",
    )
    return [
        {
            "mean_predicted_probability": float(predicted),
            "observed_event_rate": float(observed),
        }
        for predicted, observed in zip(predicted_probability, observed_rate)
    ]
