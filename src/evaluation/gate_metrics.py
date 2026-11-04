"""Auditable safety-gate evaluation metrics, comparison tables, and risk curves.

This module evaluates gate behavior (coverage, review, and block rates), safety
violations (unsafe automated decisions and missed financial stress events), and
unnecessary escalations (false escalations on solvent applicants).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.safety_gate import (
    BLOCKED,
    OOD_THRESHOLD,
    REVIEW_REQUIRED,
    SAFE_TO_AUTOMATE,
    UNCERTAINTY_THRESHOLDS,
)


def evaluate_gate(
    panel: pd.DataFrame,
    model_family: str,
    experiment_name: str,
) -> dict[str, Any]:
    """Compute comprehensive gate metrics for a given model family and experiment.

    Parameters
    ----------
    panel : pd.DataFrame
        Gated panel containing 'gate_outcome', 'financial_stress_event', 'model_family',
        and prediction / probability columns.
    model_family : str
        Target model family to filter and evaluate.
    experiment_name : str
        Audit label for this experiment configuration.

    Returns
    -------
    dict[str, Any]
        Dictionary of scalar metrics, distributions, and scenario breakdowns.
    """
    sub = panel[panel["model_family"] == model_family].copy()
    if sub.empty:
        raise ValueError(f"No records found for model_family='{model_family}' in panel.")

    total_cases = len(sub)
    safe_mask = sub["gate_outcome"] == SAFE_TO_AUTOMATE
    review_mask = sub["gate_outcome"] == REVIEW_REQUIRED
    block_mask = sub["gate_outcome"] == BLOCKED

    automated_count = int(safe_mask.sum())
    review_count = int(review_mask.sum())
    blocked_count = int(block_mask.sum())

    automation_coverage = automated_count / total_cases
    review_rate = review_count / total_cases
    block_rate = blocked_count / total_cases

    # Resolve predicted label
    if "candidate_label" in sub.columns:
        pred = sub["candidate_label"].astype(int)
    elif "predicted_label" in sub.columns:
        pred = sub["predicted_label"].astype(int)
    elif "mean_probability" in sub.columns:
        pred = (sub["mean_probability"] >= 0.5).astype(int)
    else:
        raise ValueError("panel must provide candidate_label, predicted_label, or mean_probability.")

    actual = sub["financial_stress_event"].astype(int)

    # Unsafe automated classifications
    # 1. Any wrong label passed through safely
    unsafe_automated_mask = safe_mask & (pred != actual)
    unsafe_automated_count = int(unsafe_automated_mask.sum())
    unsafe_automated_rate = unsafe_automated_count / total_cases

    # 2. Missed financial stress cases passed through safely (actual=1, pred=0)
    unsafe_missed_mask = safe_mask & (actual == 1) & (pred == 0)
    unsafe_automated_missed_stress_count = int(unsafe_missed_mask.sum())
    unsafe_automated_missed_stress_rate = unsafe_automated_missed_stress_count / total_cases

    # Accuracy on automated slice
    if automated_count > 0:
        automated_correct = int((safe_mask & (pred == actual)).sum())
        accuracy_on_automated = automated_correct / automated_count
        unsafe_automated_coverage_rate = unsafe_automated_count / automated_count
    else:
        accuracy_on_automated = 0.0
        unsafe_automated_coverage_rate = 0.0

    # False escalation count: escalated (REVIEW or BLOCKED) where applicant is solvent (actual=0)
    # and model prediction would have been correct (pred=0).
    false_escalation_mask = (review_mask | block_mask) & (actual == 0) & (pred == 0)
    false_escalation_count = int(false_escalation_mask.sum())
    false_escalation_rate = false_escalation_count / total_cases

    # Reason code distribution
    reason_dist: dict[str, int] = {}
    if "reason_codes" in sub.columns:
        for codes_val in sub["reason_codes"].dropna():
            if isinstance(codes_val, str) and codes_val:
                for code in codes_val.split("|"):
                    c = code.strip()
                    if c:
                        reason_dist[c] = reason_dist.get(c, 0) + 1
            elif isinstance(codes_val, (list, tuple)):
                for code in codes_val:
                    c = str(code).strip()
                    if c:
                        reason_dist[c] = reason_dist.get(c, 0) + 1
    sorted_reasons = dict(sorted(reason_dist.items(), key=lambda item: item[1], reverse=True))

    # Scenario breakdown
    scenario_breakdown: dict[str, dict[str, Any]] = {}
    if "scenario_label" in sub.columns:
        for sc_name, sc_sub in sub.groupby("scenario_label"):
            sc_tot = len(sc_sub)
            sc_safe = int((sc_sub["gate_outcome"] == SAFE_TO_AUTOMATE).sum())
            sc_rev = int((sc_sub["gate_outcome"] == REVIEW_REQUIRED).sum())
            sc_blk = int((sc_sub["gate_outcome"] == BLOCKED).sum())
            sc_pred = pred.loc[sc_sub.index]
            sc_act = actual.loc[sc_sub.index]
            sc_unsafe = int(((sc_sub["gate_outcome"] == SAFE_TO_AUTOMATE) & (sc_pred != sc_act)).sum())
            scenario_breakdown[str(sc_name)] = {
                "total_cases": sc_tot,
                "automated_count": sc_safe,
                "review_count": sc_rev,
                "blocked_count": sc_blk,
                "automation_coverage": sc_safe / sc_tot,
                "review_rate": sc_rev / sc_tot,
                "block_rate": sc_blk / sc_tot,
                "unsafe_automated_count": sc_unsafe,
            }

    # Severity breakdown
    severity_breakdown: dict[str, dict[str, Any]] = {}
    if "severity" in sub.columns:
        for sev_name, sev_sub in sub.groupby("severity"):
            sev_tot = len(sev_sub)
            sev_safe = int((sev_sub["gate_outcome"] == SAFE_TO_AUTOMATE).sum())
            sev_rev = int((sev_sub["gate_outcome"] == REVIEW_REQUIRED).sum())
            sev_blk = int((sev_sub["gate_outcome"] == BLOCKED).sum())
            sev_pred = pred.loc[sev_sub.index]
            sev_act = actual.loc[sev_sub.index]
            sev_unsafe = int(((sev_sub["gate_outcome"] == SAFE_TO_AUTOMATE) & (sev_pred != sev_act)).sum())
            severity_breakdown[str(sev_name)] = {
                "total_cases": sev_tot,
                "automated_count": sev_safe,
                "review_count": sev_rev,
                "blocked_count": sev_blk,
                "automation_coverage": sev_safe / sev_tot,
                "review_rate": sev_rev / sev_tot,
                "block_rate": sev_blk / sev_tot,
                "unsafe_automated_count": sev_unsafe,
            }

    return {
        "experiment_name": experiment_name,
        "model_family": model_family,
        "total_cases": total_cases,
        "automated_count": automated_count,
        "review_count": review_count,
        "blocked_count": blocked_count,
        "automation_coverage": float(automation_coverage),
        "review_rate": float(review_rate),
        "block_rate": float(block_rate),
        "unsafe_automated_count": unsafe_automated_count,
        "unsafe_automated_rate": float(unsafe_automated_rate),
        "unsafe_automated_coverage_rate": float(unsafe_automated_coverage_rate),
        "unsafe_automated_missed_stress_count": unsafe_automated_missed_stress_count,
        "unsafe_automated_missed_stress_rate": float(unsafe_automated_missed_stress_rate),
        "false_escalation_count": false_escalation_count,
        "false_escalation_rate": float(false_escalation_rate),
        "accuracy_on_automated": float(accuracy_on_automated),
        "reason_code_distribution": sorted_reasons,
        "scenario_breakdown": scenario_breakdown,
        "severity_breakdown": severity_breakdown,
    }


def compare_experiments(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a standardized tabular comparison across experiment metrics.

    Parameters
    ----------
    results : list[dict[str, Any]]
        List of result dictionaries returned by evaluate_gate().

    Returns
    -------
    pd.DataFrame
        Formatted comparison table.
    """
    rows = []
    for r in results:
        rows.append(
            {
                "experiment_name": r["experiment_name"],
                "model_family": r["model_family"],
                "total_cases": r["total_cases"],
                "automation_coverage": r["automation_coverage"],
                "review_rate": r["review_rate"],
                "block_rate": r["block_rate"],
                "unsafe_automated_count": r["unsafe_automated_count"],
                "unsafe_automated_rate": r["unsafe_automated_rate"],
                "unsafe_automated_missed_stress_count": r["unsafe_automated_missed_stress_count"],
                "false_escalation_count": r["false_escalation_count"],
                "false_escalation_rate": r["false_escalation_rate"],
                "accuracy_on_automated": r["accuracy_on_automated"],
            }
        )
    return pd.DataFrame(rows)


def compute_risk_coverage_curve(
    panel: pd.DataFrame,
    model_family: str,
    thresholds: np.ndarray | list[float] | None = None,
) -> pd.DataFrame:
    """Compute automation coverage vs unsafe automated rate at various OOD thresholds.

    Parameters
    ----------
    panel : pd.DataFrame
        Evaluation panel with evidence, OOD scores, uncertainty, and actual labels.
    model_family : str
        Model family to evaluate.
    thresholds : array-like, optional
        Custom grid of OOD thresholds. If omitted, a 30-point grid from 1.0 to 12.0 is used.

    Returns
    -------
    pd.DataFrame
        Risk-coverage curve data points.
    """
    sub = panel[panel["model_family"] == model_family].copy()
    if sub.empty:
        raise ValueError(f"No records found for model_family='{model_family}' in panel.")

    if thresholds is None:
        thresholds = np.linspace(1.0, 12.0, 30)

    total_cases = len(sub)
    actual = sub["financial_stress_event"].astype(int).to_numpy()

    if "candidate_label" in sub.columns:
        pred = sub["candidate_label"].astype(int).to_numpy()
    elif "mean_probability" in sub.columns:
        pred = (sub["mean_probability"].to_numpy() >= 0.5).astype(int)
    else:
        pred = (sub["predicted_label"].to_numpy() >= 0.5).astype(int)

    # Extract base signals
    consistency_violations = sub["consistency_violation_count"].to_numpy()
    observed_months = sub["observed_months"].to_numpy()
    unverified = sub["unverified_record_count"].to_numpy()
    missing_fields = sub["missing_required_field_count"].to_numpy()

    missing_months = sub["missing_month_count"].to_numpy()
    age = sub["latest_evidence_age_months"].to_numpy()

    ood_scores = sub["ood_score"].to_numpy()
    unc_var = sub["predictive_variance"].to_numpy()

    default_unc_thresh = UNCERTAINTY_THRESHOLDS[model_family]
    unc_thresh = sub.get("uncertainty_threshold", default_unc_thresh)
    if isinstance(unc_thresh, pd.Series):
        unc_thresh = unc_thresh.to_numpy()
    else:
        unc_thresh = np.full(total_cases, unc_thresh)

    # Mandatory evidence defect check -> BLOCKED
    mandatory_mask = (
        (consistency_violations > 0)
        | (observed_months == 0)
        | ((observed_months > 0) & (unverified == observed_months))
        | ((observed_months > 0) & (missing_fields >= observed_months * 12))
    )

    # Soft evidence defect check
    soft_ev_mask = (
        (missing_months > 0)
        | (missing_fields > 0)
        | (age > 0)
        | ((unverified > 0) & (unverified < observed_months))
    )

    # High uncertainty check
    high_unc_mask = unc_var >= unc_thresh

    # Non-OOD review condition
    base_review_mask = soft_ev_mask | high_unc_mask

    curve_rows = []
    for thresh in thresholds:
        ood_flag = ood_scores >= thresh

        is_blocked = mandatory_mask
        is_review = (~is_blocked) & (base_review_mask | ood_flag)
        is_safe = (~is_blocked) & (~is_review)

        safe_cnt = int(is_safe.sum())
        rev_cnt = int(is_review.sum())
        blk_cnt = int(is_blocked.sum())

        coverage = safe_cnt / total_cases
        review_rate = rev_cnt / total_cases
        block_rate = blk_cnt / total_cases

        unsafe_automated_cnt = int((is_safe & (pred != actual)).sum())
        unsafe_automated_rate = unsafe_automated_cnt / total_cases
        unsafe_cov_rate = unsafe_automated_cnt / safe_cnt if safe_cnt > 0 else 0.0

        curve_rows.append(
            {
                "model_family": model_family,
                "ood_threshold": float(thresh),
                "automated_count": safe_cnt,
                "review_count": rev_cnt,
                "blocked_count": blk_cnt,
                "automation_coverage": float(coverage),
                "review_rate": float(review_rate),
                "block_rate": float(block_rate),
                "unsafe_automated_count": unsafe_automated_cnt,
                "unsafe_automated_rate": float(unsafe_automated_rate),
                "unsafe_automated_coverage_rate": float(unsafe_cov_rate),
            }
        )

    return pd.DataFrame(curve_rows)
