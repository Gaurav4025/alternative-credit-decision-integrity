"""Matched evaluation panels and gate execution across Phase 8 experiments.

This module constructs test panels combining evidence quality, shrinkage
Mahalanobis OOD scores, and model bootstrap predictive variance for the fixed
Phase 6 test split. Labels (financial_stress_event) are retained strictly as
evaluation targets and are never passed to the safety gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.models.safety_gate import (
    BLOCKED,
    GATE_VERSION,
    OOD_THRESHOLD,
    REQUIRED_FIELD_COUNT,
    REVIEW_REQUIRED,
    SAFE_TO_AUTOMATE,
    UNCERTAINTY_THRESHOLDS,
    CandidatePrediction,
    EvidenceQualityInput,
    GateDecision,
    OODInput,
    UncertaintyInput,
    decide,
)

PROHIBITED_GATE_INPUTS = {
    "financial_stress_probability",
    "population_group",
}

EVIDENCE_SCENARIOS = (
    "missingness",
    "stale_records",
    "inconsistent_records",
    "unverified_provenance",
)
OOD_SCENARIOS = ("monetary_scale", "temporal_volatility", "transaction_pattern")
SEVERITIES = ("low", "medium", "high")
MODEL_FAMILIES = ("logistic_regression", "hist_gradient_boosting")

EVIDENCE_FIELD_COLUMNS = [
    "total_expected_months",
    "observed_months",
    "missing_month_count",
    "missing_required_field_count",
    "latest_evidence_age_months",
    "consistency_violation_count",
    "unverified_record_count",
]


def build_gate_panel(
    model_dataset_path: str | Path = "data/processed/synthetic/model_dataset.csv",
    split_ids_path: str | Path = "experiments/ood/split_applicant_ids.csv",
    evidence_quality_path: str | Path = "experiments/evidence_quality/applicant_evidence_quality.csv",
    ood_scores_path: str | Path = "experiments/ood/ood_scores.csv",
    uncertainty_scores_path: str | Path = "experiments/uncertainty/uncertainty_scores.csv",
    panel_type: Literal["all", "clean", "evidence", "ood", "combined"] = "all",
) -> pd.DataFrame:
    """Construct matched test panels across all gate conditions.

    Parameters
    ----------
    model_dataset_path : str or Path
        Source for the held-out reference evaluation label (financial_stress_event).
    split_ids_path : str or Path
        Persisted split assignments containing the 1,500 test applicants.
    evidence_quality_path : str or Path
        Phase 5 evidence quality measurements across 12 scenario/severity pairs.
    ood_scores_path : str or Path
        Phase 6 detector scores for clean reference and controlled OOD cohorts.
    uncertainty_scores_path : str or Path
        Phase 7 bootstrap uncertainty scores per applicant and model family.
    panel_type : str, default 'all'
        Subset of panels to return ('all', 'clean', 'evidence', 'ood', 'combined').

    Returns
    -------
    pd.DataFrame
        Matched panel with one row per (applicant_id, model_family, scenario_label).
    """
    model_dataset_path = Path(model_dataset_path)
    split_ids_path = Path(split_ids_path)
    evidence_quality_path = Path(evidence_quality_path)
    ood_scores_path = Path(ood_scores_path)
    uncertainty_scores_path = Path(uncertainty_scores_path)

    # 1. Load test split applicant IDs
    splits_df = pd.read_csv(split_ids_path)
    test_ids = set(splits_df.loc[splits_df["split"] == "test", "applicant_id"].astype(int))

    # 2. Load evaluation labels (financial_stress_event only)
    dataset_df = pd.read_csv(model_dataset_path)
    test_labels = dataset_df.loc[
        dataset_df["applicant_id"].isin(test_ids),
        ["applicant_id", "financial_stress_event"],
    ].copy()
    test_labels["applicant_id"] = test_labels["applicant_id"].astype(int)
    test_labels["financial_stress_event"] = test_labels["financial_stress_event"].astype(int)

    # 3. Load test uncertainty scores
    unc_df = pd.read_csv(uncertainty_scores_path)
    unc_test = unc_df.loc[
        (unc_df["evaluation_split"] == "test") & (unc_df["applicant_id"].isin(test_ids)),
        [
            "model_family",
            "applicant_id",
            "mean_probability",
            "predictive_variance",
            "uncertainty_threshold",
        ],
    ].copy()
    unc_test["applicant_id"] = unc_test["applicant_id"].astype(int)

    # 4. Load OOD Shrinkage Mahalanobis test scores
    ood_df = pd.read_csv(ood_scores_path)
    ood_test_sm = ood_df.loc[
        (ood_df["evaluation_split"] == "test")
        & (ood_df["detector"] == "shrinkage_mahalanobis")
        & (ood_df["source_applicant_id"].isin(test_ids)),
        [
            "scenario",
            "severity",
            "source_applicant_id",
            "ood_score",
            "threshold",
            "is_controlled_ood",
        ],
    ].copy()
    ood_test_sm["source_applicant_id"] = ood_test_sm["source_applicant_id"].astype(int)

    # Clean OOD reference lookup
    ood_clean = ood_test_sm.loc[
        ood_test_sm["scenario"] == "clean_reference",
        ["source_applicant_id", "ood_score", "threshold"],
    ].rename(columns={"source_applicant_id": "applicant_id", "threshold": "ood_threshold"})

    # 5. Load Evidence Quality measurements
    ev_df = pd.read_csv(evidence_quality_path)
    ev_test = ev_df.loc[
        ev_df["applicant_id"].isin(test_ids),
        [
            "scenario",
            "severity",
            "applicant_id",
            "simulated_applicant_defect_introduced",
            *EVIDENCE_FIELD_COLUMNS,
        ],
    ].copy()
    ev_test["applicant_id"] = ev_test["applicant_id"].astype(int)

    panels: list[pd.DataFrame] = []

    # --- Clean panel ---
    if panel_type in ("all", "clean", "evidence", "ood"):
        clean_base = test_labels.merge(unc_test, on="applicant_id", how="inner")
        clean_base = clean_base.merge(ood_clean, on="applicant_id", how="inner")
        clean_base["total_expected_months"] = 12
        clean_base["observed_months"] = 12
        clean_base["missing_month_count"] = 0
        clean_base["missing_required_field_count"] = 0
        clean_base["latest_evidence_age_months"] = 0
        clean_base["consistency_violation_count"] = 0
        clean_base["unverified_record_count"] = 0
        clean_base["scenario_label"] = "clean"
        clean_base["experiment_group"] = "clean"
        clean_base["severity"] = "none"
        panels.append(clean_base)

    # --- Evidence-degraded panel ---
    if panel_type in ("all", "evidence"):
        ev_degraded = ev_test.loc[ev_test["simulated_applicant_defect_introduced"]].copy()
        for (ev_scenario, ev_severity), group in ev_degraded.groupby(["scenario", "severity"]):
            deg_merged = test_labels.merge(group, on="applicant_id", how="inner")
            deg_merged = deg_merged.merge(unc_test, on="applicant_id", how="inner")
            deg_merged = deg_merged.merge(ood_clean, on="applicant_id", how="inner")
            deg_merged["scenario_label"] = f"evidence_{ev_scenario}_{ev_severity}"
            deg_merged["experiment_group"] = "evidence_degraded"
            deg_merged["severity"] = ev_severity
            deg_merged = deg_merged.drop(
                columns=["scenario", "simulated_applicant_defect_introduced"]
            )
            panels.append(deg_merged)

    # --- OOD-shifted panel ---
    if panel_type in ("all", "ood"):
        ood_shifted = ood_test_sm.loc[ood_test_sm["is_controlled_ood"]].copy()
        for (ood_sc, ood_sev), group in ood_shifted.groupby(["scenario", "severity"]):
            group_renamed = group.rename(
                columns={
                    "source_applicant_id": "applicant_id",
                    "threshold": "ood_threshold",
                }
            )
            ood_merged = test_labels.merge(group_renamed, on="applicant_id", how="inner")
            ood_merged = ood_merged.merge(unc_test, on="applicant_id", how="inner")
            ood_merged["total_expected_months"] = 12
            ood_merged["observed_months"] = 12
            ood_merged["missing_month_count"] = 0
            ood_merged["missing_required_field_count"] = 0
            ood_merged["latest_evidence_age_months"] = 0
            ood_merged["consistency_violation_count"] = 0
            ood_merged["unverified_record_count"] = 0
            ood_merged["scenario_label"] = f"ood_{ood_sc}_{ood_sev}"
            ood_merged["experiment_group"] = "ood_shifted"
            ood_merged["severity"] = ood_sev
            ood_merged = ood_merged.drop(columns=["scenario", "is_controlled_ood"])
            panels.append(ood_merged)

    # --- Combined panel ---
    if panel_type in ("all", "combined"):
        ev_degraded = ev_test.loc[ev_test["simulated_applicant_defect_introduced"]].copy()
        ood_shifted = ood_test_sm.loc[ood_test_sm["is_controlled_ood"]].copy()
        for sev in SEVERITIES:
            for ev_sc in EVIDENCE_SCENARIOS:
                ev_sub = ev_degraded.loc[
                    (ev_degraded["scenario"] == ev_sc) & (ev_degraded["severity"] == sev)
                ]
                for ood_sc in OOD_SCENARIOS:
                    ood_sub = ood_shifted.loc[
                        (ood_shifted["scenario"] == ood_sc)
                        & (ood_shifted["severity"] == sev)
                    ].rename(
                        columns={
                            "source_applicant_id": "applicant_id",
                            "threshold": "ood_threshold",
                        }
                    )
                    combo = ev_sub.merge(
                        ood_sub[["applicant_id", "ood_score", "ood_threshold"]],
                        on="applicant_id",
                        how="inner",
                    )
                    combo = combo.merge(test_labels, on="applicant_id", how="inner")
                    combo = combo.merge(unc_test, on="applicant_id", how="inner")
                    combo["scenario_label"] = f"combined_{ev_sc}_{ood_sc}_{sev}"
                    combo["experiment_group"] = "combined"
                    combo["severity"] = sev
                    combo = combo.drop(
                        columns=["scenario", "simulated_applicant_defect_introduced"]
                    )
                    panels.append(combo)

    result = pd.concat(panels, ignore_index=True)

    # Check forbidden leakage columns
    for prohibited in PROHIBITED_GATE_INPUTS:
        if prohibited in result.columns:
            result = result.drop(columns=[prohibited])

    # Reorder columns logically
    canonical_columns = [
        "applicant_id",
        "model_family",
        "scenario_label",
        "experiment_group",
        "severity",
        "financial_stress_event",
        "mean_probability",
        "predictive_variance",
        "uncertainty_threshold",
        "ood_score",
        "ood_threshold",
        *EVIDENCE_FIELD_COLUMNS,
    ]
    other_cols = [c for c in result.columns if c not in canonical_columns]
    result = result[canonical_columns + other_cols]

    return result


def apply_safety_gate(
    panel: pd.DataFrame,
    experiment_id: str = "phase8_gate",
    gate_mode: Literal["full", "evidence_only", "ood_only", "uncertainty_only", "ungated"] = "full",
) -> pd.DataFrame:
    """Execute safety gate decisions across panel rows.

    Parameters
    ----------
    panel : pd.DataFrame
        Evaluation panel containing evidence, OOD, uncertainty, and prediction columns.
    experiment_id : str, default 'phase8_gate'
        Audit identifier associated with this experiment run.
    gate_mode : str, default 'full'
        Gate operational mode:
        - 'full': All signals (evidence, OOD, uncertainty) evaluated as provided.
        - 'evidence_only': Neutral OOD and uncertainty; evidence quality only.
        - 'ood_only': Clean evidence and neutral uncertainty; OOD shift only.
        - 'uncertainty_only': Clean evidence and neutral OOD; predictive variance only.
        - 'ungated': Bypass gate, setting all decisions to SAFE_TO_AUTOMATE.

    Returns
    -------
    pd.DataFrame
        Input panel augmented with gate decisions, outcomes, and reason codes.
    """
    records: list[dict[str, object]] = []

    if gate_mode == "ungated":
        out = panel.copy()
        out["gate_outcome"] = SAFE_TO_AUTOMATE
        out["reason_codes"] = ""
        out["gate_version"] = GATE_VERSION
        out["experiment_id"] = experiment_id
        out["candidate_label"] = (out["mean_probability"] >= 0.5).astype(int)
        out["candidate_probability"] = out["mean_probability"].astype(float)
        out["candidate_model_family"] = out["model_family"]
        out["candidate_classification_threshold"] = 0.5
        return out

    for row in panel.to_dict(orient="records"):
        applicant_id = int(row["applicant_id"])
        scenario_id = str(row["scenario_label"])
        model_family = str(row["model_family"])
        prob = float(row["mean_probability"])
        label = int(prob >= 0.5)

        candidate = CandidatePrediction(
            model_family=model_family,  # type: ignore[arg-type]
            probability=prob,
            label=label,
            classification_threshold=0.5,
        )

        # 1. Evidence Quality input
        if gate_mode in ("ood_only", "uncertainty_only"):
            evidence = EvidenceQualityInput(
                total_expected_months=12,
                observed_months=12,
                missing_month_count=0,
                missing_required_field_count=0,
                latest_evidence_age_months=0,
                consistency_violation_count=0,
                unverified_record_count=0,
            )
        else:
            evidence = EvidenceQualityInput(
                total_expected_months=int(row["total_expected_months"]),
                observed_months=int(row["observed_months"]),
                missing_month_count=int(row["missing_month_count"]),
                missing_required_field_count=int(row["missing_required_field_count"]),
                latest_evidence_age_months=int(row["latest_evidence_age_months"]),
                consistency_violation_count=int(row["consistency_violation_count"]),
                unverified_record_count=int(row["unverified_record_count"]),
            )

        # 2. OOD input
        ood_thresh = float(row.get("ood_threshold", OOD_THRESHOLD))
        if gate_mode in ("evidence_only", "uncertainty_only"):
            ood = OODInput(score=0.0, threshold=ood_thresh)
        else:
            ood = OODInput(score=float(row["ood_score"]), threshold=ood_thresh)

        # 3. Uncertainty input
        default_unc_thresh = UNCERTAINTY_THRESHOLDS[model_family]
        unc_thresh = float(row.get("uncertainty_threshold", default_unc_thresh))
        if gate_mode in ("evidence_only", "ood_only"):
            uncertainty = UncertaintyInput(predictive_variance=0.0, threshold=unc_thresh)
        else:
            uncertainty = UncertaintyInput(
                predictive_variance=float(row["predictive_variance"]),
                threshold=unc_thresh,
            )

        decision = decide(
            applicant_id=applicant_id,
            scenario_id=scenario_id,
            experiment_id=experiment_id,
            evidence=evidence,
            ood=ood,
            uncertainty=uncertainty,
            candidate=candidate,
        )
        dec_dict = decision.as_dict()

        row_copy = dict(row)
        row_copy["gate_outcome"] = dec_dict["gate_outcome"]
        row_copy["reason_codes"] = dec_dict["reason_codes"]
        row_copy["gate_version"] = dec_dict["gate_version"]
        row_copy["experiment_id"] = dec_dict["experiment_id"]
        row_copy["candidate_label"] = dec_dict["candidate_label"]
        row_copy["candidate_probability"] = dec_dict["candidate_probability"]
        row_copy["candidate_model_family"] = dec_dict["candidate_model_family"]
        row_copy["candidate_classification_threshold"] = dec_dict["candidate_classification_threshold"]
        records.append(row_copy)

    return pd.DataFrame(records)
