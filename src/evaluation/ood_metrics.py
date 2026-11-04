"""Metrics for controlled OOD detection experiments, separate from risk prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def select_ood_threshold(validation_scores: pd.Series | np.ndarray, percentile: float = 0.95) -> float:
    """Select an OOD threshold from clean validation scores only."""

    values = np.asarray(validation_scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Validation scores must be a non-empty finite one-dimensional array.")
    return float(np.quantile(values, percentile))


def evaluate_ood_detection(
    clean_scores: pd.Series | np.ndarray,
    ood_scores: pd.Series | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Evaluate clean versus controlled-OOD scores at a fixed threshold."""

    clean = np.asarray(clean_scores, dtype=float)
    ood = np.asarray(ood_scores, dtype=float)
    if len(clean) == 0 or len(ood) == 0:
        raise ValueError("Both clean and controlled-OOD score arrays must be non-empty.")
    labels = np.concatenate([np.zeros(len(clean), dtype=int), np.ones(len(ood), dtype=int)])
    scores = np.concatenate([clean, ood])
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "clean_false_positive_rate": float((clean >= threshold).mean()),
        "ood_detection_rate": float((ood >= threshold).mean()),
        "artificial_ood_prevalence": float(len(ood) / len(labels)),
        "clean_count": int(len(clean)),
        "ood_count": int(len(ood)),
    }
