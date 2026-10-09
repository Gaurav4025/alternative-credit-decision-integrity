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


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    y_probability: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error with fixed-width probability bins.

    Bins are equally spaced over ``[0, 1]``. Each non-empty bin contributes:

    ``bin_size / n_samples * abs(mean_predicted_probability - observed_event_rate)``

    Empty bins contribute zero because they contain no predictions.
    """

    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    y_true_array = np.asarray(y_true)
    y_probability_array = np.asarray(y_probability)
    if len(y_true_array) != len(y_probability_array):
        raise ValueError("y_true and y_probability must have the same length.")
    if len(y_true_array) == 0:
        raise ValueError("ECE requires at least one prediction.")
    if ((y_probability_array < 0) | (y_probability_array > 1)).any():
        raise ValueError("Probabilities must be in [0, 1].")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        lower = bin_edges[index]
        upper = bin_edges[index + 1]
        if index == n_bins - 1:
            in_bin = (y_probability_array >= lower) & (y_probability_array <= upper)
        else:
            in_bin = (y_probability_array >= lower) & (y_probability_array < upper)

        if not in_bin.any():
            continue

        bin_probability = y_probability_array[in_bin].mean()
        bin_observed_rate = y_true_array[in_bin].mean()
        bin_weight = in_bin.mean()
        ece += bin_weight * abs(bin_probability - bin_observed_rate)

    return float(ece)
