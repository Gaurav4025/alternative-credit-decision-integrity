"""Evaluation utilities for bootstrap prediction-uncertainty experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


CLASSIFICATION_THRESHOLD = 0.5
STABILITY_RANK_CORRELATION_MINIMUM = 0.90
STABILITY_MEDIAN_RELATIVE_DIFFERENCE_MAXIMUM = 0.35
SELECTIVE_COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Return Bernoulli entropy of mean probabilities with safe numerical clipping."""

    values = np.asarray(probabilities, dtype=float)
    clipped = np.clip(values, 1e-12, 1 - 1e-12)
    return -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))


def summarize_member_predictions(member_probabilities: np.ndarray) -> dict[str, np.ndarray]:
    """Summarize bootstrap member probabilities for each applicant."""

    values = np.asarray(member_probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Member probabilities must be a two-dimensional array with at least two members.")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Member probabilities must be finite values in [0, 1].")
    mean_probability = values.mean(axis=1)
    variance = values.var(axis=1, ddof=1)
    return {
        "mean_probability": mean_probability,
        "predictive_variance": variance,
        "predictive_std": np.sqrt(variance),
        "predictive_entropy": predictive_entropy(mean_probability),
        "interval_lower_05": np.quantile(values, 0.05, axis=1),
        "interval_upper_95": np.quantile(values, 0.95, axis=1),
    }


def ensemble_stability(first_member_probabilities: np.ndarray, all_member_probabilities: np.ndarray) -> dict[str, float]:
    """Compare first-half and full-ensemble variances on clean validation data."""

    first = summarize_member_predictions(first_member_probabilities)["predictive_variance"]
    full = summarize_member_predictions(all_member_probabilities)["predictive_variance"]
    rank_correlation = float(pd.Series(first).rank().corr(pd.Series(full).rank()))
    absolute_difference = np.abs(first - full)
    meaningful = full > 1e-12
    relative_difference = absolute_difference[meaningful] / full[meaningful]
    median_relative = float(np.median(relative_difference)) if len(relative_difference) else 0.0
    return {
        "rank_correlation": rank_correlation,
        "mean_absolute_difference": float(absolute_difference.mean()),
        "median_absolute_difference": float(np.median(absolute_difference)),
        "median_relative_difference": median_relative,
        "stable": bool(
            rank_correlation >= STABILITY_RANK_CORRELATION_MINIMUM
            and median_relative <= STABILITY_MEDIAN_RELATIVE_DIFFERENCE_MAXIMUM
        ),
    }


def select_uncertainty_threshold(validation_variance: np.ndarray, percentile: float = 0.95) -> float:
    """Select a model-family uncertainty threshold from validation variance only."""

    values = np.asarray(validation_variance, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Validation variances must be a non-empty finite non-negative array.")
    return float(np.quantile(values, percentile))


def evaluate_uncertainty(
    y_true: np.ndarray,
    mean_probability: np.ndarray,
    variance: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate error detection and fixed-threshold selective prediction."""

    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(mean_probability, dtype=float)
    uncertainty = np.asarray(variance, dtype=float)
    if not (len(labels) == len(probabilities) == len(uncertainty)):
        raise ValueError("Labels, probabilities, and uncertainty values must have equal lengths.")
    predictions = probabilities >= CLASSIFICATION_THRESHOLD
    errors = predictions != labels
    retained = uncertainty < threshold
    if not retained.any():
        raise ValueError("Uncertainty threshold retained no applicants.")

    metrics = {
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "uncertainty_flag_rate": float((uncertainty >= threshold).mean()),
        "retained_coverage": float(retained.mean()),
        "retained_brier_score": float(brier_score_loss(labels[retained], probabilities[retained])),
        "retained_log_loss": float(log_loss(labels[retained], probabilities[retained], labels=[0, 1])),
        "retained_classification_error_rate": float(errors[retained].mean()),
        "error_detection_auroc": float(roc_auc_score(errors, uncertainty)),
        "error_detection_auprc": float(average_precision_score(errors, uncertainty)),
        "error_rate": float(errors.mean()),
    }
    curve = selective_prediction_curve(labels, probabilities, uncertainty)
    metrics["area_under_risk_coverage_curve"] = area_under_risk_coverage_curve(labels, probabilities, uncertainty)
    return metrics, curve


def selective_prediction_curve(
    y_true: np.ndarray,
    mean_probability: np.ndarray,
    variance: np.ndarray,
    coverages: tuple[float, ...] = SELECTIVE_COVERAGES,
) -> pd.DataFrame:
    """Report prediction risk after retaining the least-uncertain applicants."""

    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(mean_probability, dtype=float)
    uncertainty = np.asarray(variance, dtype=float)
    order = np.argsort(uncertainty, kind="stable")
    rows = []
    for requested_coverage in coverages:
        retained_count = max(1, int(np.ceil(requested_coverage * len(labels))))
        selected = order[:retained_count]
        errors = (probabilities[selected] >= CLASSIFICATION_THRESHOLD) != labels[selected]
        rows.append({
            "requested_coverage": requested_coverage,
            "coverage": float(retained_count / len(labels)),
            "risk": float(errors.mean()),
            "brier_score": float(brier_score_loss(labels[selected], probabilities[selected])),
            "log_loss": float(log_loss(labels[selected], probabilities[selected], labels=[0, 1])),
        })
    return pd.DataFrame(rows)


def area_under_risk_coverage_curve(
    y_true: np.ndarray,
    mean_probability: np.ndarray,
    variance: np.ndarray,
) -> float:
    """Integrate classification risk as retained coverage grows from 1/n to 1."""

    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(mean_probability, dtype=float)
    order = np.argsort(variance, kind="stable")
    errors = ((probabilities[order] >= CLASSIFICATION_THRESHOLD) != labels[order]).astype(float)
    risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    coverage = np.arange(1, len(errors) + 1) / len(errors)
    return float(np.trapezoid(risk, coverage))
