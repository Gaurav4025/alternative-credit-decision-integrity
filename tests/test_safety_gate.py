"""Comprehensive test suite for Phase 8 safety gating, scenarios, and metrics.

Covers:
1. Pure unit tests for safety_gate.py logic, validation checks, and reason codes.
2. Integration tests for gate_scenarios.py panel construction and leakage prevention.
3. Metric verification tests for gate_metrics.py rates, distributions, and risk curves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.gate_scenarios import apply_safety_gate, build_gate_panel
from src.evaluation.gate_metrics import (
    compare_experiments,
    compute_risk_coverage_curve,
    evaluate_gate,
)
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
    _validate_candidate,
    _validate_evidence,
    _validate_identity,
    _validate_ood,
    _validate_uncertainty,
    decide,
)


# ==============================================================================
# Helper fixtures for pure unit testing
# ==============================================================================


def _clean_evidence() -> EvidenceQualityInput:
    return EvidenceQualityInput(
        total_expected_months=12,
        observed_months=12,
        missing_month_count=0,
        missing_required_field_count=0,
        latest_evidence_age_months=0,
        consistency_violation_count=0,
        unverified_record_count=0,
    )


def _clean_ood() -> OODInput:
    return OODInput(score=3.0, threshold=OOD_THRESHOLD)


def _clean_uncertainty(model_family: str = "logistic_regression") -> UncertaintyInput:
    thresh = UNCERTAINTY_THRESHOLDS[model_family]
    return UncertaintyInput(predictive_variance=thresh / 2.0, threshold=thresh)


def _clean_candidate(
    model_family: str = "logistic_regression",
    prob: float = 0.30,
) -> CandidatePrediction:
    return CandidatePrediction(
        model_family=model_family,  # type: ignore[arg-type]
        probability=prob,
        label=int(prob >= 0.5),
        classification_threshold=0.5,
    )


# ==============================================================================
# 1. Gate Logic Tests (Pure Unit)
# ==============================================================================


def test_decide_returns_safe_to_automate_when_all_signals_clean() -> None:
    for mf in ("logistic_regression", "hist_gradient_boosting"):
        decision = decide(
            applicant_id=101,
            scenario_id="clean_test",
            experiment_id="exp_clean",
            evidence=_clean_evidence(),
            ood=_clean_ood(),
            uncertainty=_clean_uncertainty(mf),
            candidate=_clean_candidate(mf),
        )
        assert decision.outcome == SAFE_TO_AUTOMATE
        assert decision.reason_codes == ()
        assert decision.applicant_id == 101
        assert decision.scenario_id == "clean_test"
        assert decision.experiment_id == "exp_clean"
        assert decision.gate_version == GATE_VERSION


@pytest.mark.parametrize(
    ("evidence_override", "expected_reason"),
    [
        (
            {"consistency_violation_count": 2},
            "EVIDENCE_CONSISTENCY_CONTRADICTION",
        ),
        (
            {"observed_months": 0, "missing_month_count": 12},
            "EVIDENCE_NO_OBSERVATIONS",
        ),
        (
            {"unverified_record_count": 12},
            "EVIDENCE_NO_VERIFIED_RECORDS",
        ),
        (
            {"missing_required_field_count": 12 * REQUIRED_FIELD_COUNT},
            "EVIDENCE_NO_USABLE_REQUIRED_FIELDS",
        ),
    ],
)
def test_decide_returns_blocked_for_each_mandatory_condition_independently(
    evidence_override: dict[str, int],
    expected_reason: str,
) -> None:
    base = {
        "total_expected_months": 12,
        "observed_months": 12,
        "missing_month_count": 0,
        "missing_required_field_count": 0,
        "latest_evidence_age_months": 0,
        "consistency_violation_count": 0,
        "unverified_record_count": 0,
    }
    base.update(evidence_override)
    ev = EvidenceQualityInput(**base)

    decision = decide(
        applicant_id=1,
        scenario_id="mandatory_test",
        experiment_id="exp_mandatory",
        evidence=ev,
        ood=_clean_ood(),
        uncertainty=_clean_uncertainty(),
        candidate=_clean_candidate(),
    )
    assert decision.outcome == BLOCKED
    assert expected_reason in decision.reason_codes


@pytest.mark.parametrize(
    ("evidence_override", "ood_override", "unc_override", "expected_reason"),
    [
        (
            {"observed_months": 11, "missing_month_count": 1},
            None,
            None,
            "EVIDENCE_MISSING_MONTH",
        ),
        (
            {"missing_required_field_count": 3},
            None,
            None,
            "EVIDENCE_MISSING_REQUIRED_FIELD",
        ),
        (
            {"latest_evidence_age_months": 2},
            None,
            None,
            "EVIDENCE_STALE_OBSERVATION",
        ),
        (
            {"unverified_record_count": 3},
            None,
            None,
            "EVIDENCE_PARTIALLY_UNVERIFIED",
        ),
        (
            None,
            {"score": OOD_THRESHOLD + 1.0},
            None,
            "OOD_SHRINKAGE_MAHALANOBIS_HIGH",
        ),
        (
            None,
            None,
            {"predictive_variance": UNCERTAINTY_THRESHOLDS["logistic_regression"] + 0.001},
            "UNCERTAINTY_LOGISTIC_VARIANCE_HIGH",
        ),
    ],
)
def test_decide_returns_review_required_for_each_soft_condition_independently(
    evidence_override: dict[str, int] | None,
    ood_override: dict[str, float] | None,
    unc_override: dict[str, float] | None,
    expected_reason: str,
) -> None:
    if evidence_override is not None:
        base_ev = {
            "total_expected_months": 12,
            "observed_months": 12,
            "missing_month_count": 0,
            "missing_required_field_count": 0,
            "latest_evidence_age_months": 0,
            "consistency_violation_count": 0,
            "unverified_record_count": 0,
        }
        base_ev.update(evidence_override)
        ev = EvidenceQualityInput(**base_ev)
    else:
        ev = _clean_evidence()

    if ood_override is not None:
        ood = OODInput(**ood_override)
    else:
        ood = _clean_ood()

    if unc_override is not None:
        thresh = UNCERTAINTY_THRESHOLDS["logistic_regression"]
        unc = UncertaintyInput(
            predictive_variance=unc_override["predictive_variance"],
            threshold=thresh,
        )
    else:
        unc = _clean_uncertainty("logistic_regression")

    decision = decide(
        applicant_id=1,
        scenario_id="soft_test",
        experiment_id="exp_soft",
        evidence=ev,
        ood=ood,
        uncertainty=unc,
        candidate=_clean_candidate("logistic_regression"),
    )
    assert decision.outcome == REVIEW_REQUIRED
    assert expected_reason in decision.reason_codes


def test_ood_alone_never_causes_blocked() -> None:
    extreme_ood = OODInput(score=9999.0, threshold=OOD_THRESHOLD)
    decision = decide(
        applicant_id=1,
        scenario_id="extreme_ood",
        experiment_id="exp_ood_alone",
        evidence=_clean_evidence(),
        ood=extreme_ood,
        uncertainty=_clean_uncertainty(),
        candidate=_clean_candidate(),
    )
    assert decision.outcome == REVIEW_REQUIRED
    assert decision.outcome != BLOCKED
    assert "OOD_SHRINKAGE_MAHALANOBIS_HIGH" in decision.reason_codes


def test_uncertainty_alone_never_causes_blocked() -> None:
    for mf in ("logistic_regression", "hist_gradient_boosting"):
        thresh = UNCERTAINTY_THRESHOLDS[mf]
        extreme_unc = UncertaintyInput(predictive_variance=9999.0, threshold=thresh)
        decision = decide(
            applicant_id=1,
            scenario_id="extreme_unc",
            experiment_id="exp_unc_alone",
            evidence=_clean_evidence(),
            ood=_clean_ood(),
            uncertainty=extreme_unc,
            candidate=_clean_candidate(mf),
        )
        assert decision.outcome == REVIEW_REQUIRED
        assert decision.outcome != BLOCKED
        expected_code = (
            "UNCERTAINTY_LOGISTIC_VARIANCE_HIGH"
            if mf == "logistic_regression"
            else "UNCERTAINTY_HGB_VARIANCE_HIGH"
        )
        assert expected_code in decision.reason_codes


def test_reason_codes_ordering_consistency() -> None:
    # A case that triggers mandatory evidence + soft evidence + OOD + uncertainty
    ev = EvidenceQualityInput(
        total_expected_months=12,
        observed_months=10,
        missing_month_count=2,
        missing_required_field_count=5,
        latest_evidence_age_months=1,
        consistency_violation_count=1,  # mandatory
        unverified_record_count=2,       # soft partial
    )
    ood = OODInput(score=10.0, threshold=5.0)
    unc = UncertaintyInput(predictive_variance=0.05, threshold=0.01)
    cand = _clean_candidate("logistic_regression")

    decision = decide(
        applicant_id=1,
        scenario_id="multi_defect",
        experiment_id="exp_multi",
        evidence=ev,
        ood=ood,
        uncertainty=unc,
        candidate=cand,
    )
    assert decision.outcome == BLOCKED
    codes = decision.reason_codes
    # Mandatory must appear first
    assert codes[0] == "EVIDENCE_CONSISTENCY_CONTRADICTION"
    # Soft evidence follows
    assert "EVIDENCE_MISSING_MONTH" in codes
    assert "EVIDENCE_MISSING_REQUIRED_FIELD" in codes
    assert "EVIDENCE_STALE_OBSERVATION" in codes
    assert "EVIDENCE_PARTIALLY_UNVERIFIED" in codes
    # OOD follows
    assert "OOD_SHRINKAGE_MAHALANOBIS_HIGH" in codes
    # Uncertainty is last
    assert codes[-1] == "UNCERTAINTY_LOGISTIC_VARIANCE_HIGH"


def test_as_dict_contains_no_leakage_columns() -> None:
    decision = decide(
        applicant_id=42,
        scenario_id="dict_test",
        experiment_id="exp_dict",
        evidence=_clean_evidence(),
        ood=_clean_ood(),
        uncertainty=_clean_uncertainty(),
        candidate=_clean_candidate(),
    )
    dec_dict = decision.as_dict()
    assert "financial_stress_event" not in dec_dict
    assert "financial_stress_probability" not in dec_dict
    assert "population_group" not in dec_dict
    assert dec_dict["applicant_id"] == 42
    assert dec_dict["gate_outcome"] == SAFE_TO_AUTOMATE
    assert dec_dict["gate_version"] == GATE_VERSION


def test_validation_functions_raise_on_bad_inputs() -> None:
    # Identity
    with pytest.raises(ValueError, match="applicant_id must be a positive integer"):
        _validate_identity(0, "sc", "exp")
    with pytest.raises(ValueError, match="scenario_id and experiment_id must be non-empty"):
        _validate_identity(1, "", "exp")

    # Evidence
    with pytest.raises(ValueError, match="non-negative"):
        _validate_evidence(EvidenceQualityInput(12, -1, 0, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="total_expected_months must be positive"):
        _validate_evidence(EvidenceQualityInput(0, 0, 0, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="cannot exceed total_expected_months"):
        _validate_evidence(EvidenceQualityInput(10, 12, 0, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="missing_month_count must equal"):
        _validate_evidence(EvidenceQualityInput(12, 10, 1, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="unverified_record_count cannot exceed"):
        _validate_evidence(EvidenceQualityInput(12, 10, 2, 0, 0, 0, 11))
    with pytest.raises(ValueError, match="missing_required_field_count exceeds"):
        _validate_evidence(EvidenceQualityInput(12, 10, 2, 10 * 12 + 1, 0, 0, 0))

    # OOD
    with pytest.raises(ValueError, match="finite non-negative"):
        _validate_ood(OODInput(score=-1.0))
    with pytest.raises(ValueError, match="finite non-negative"):
        _validate_ood(OODInput(score=float("nan")))

    # Uncertainty
    with pytest.raises(ValueError, match="Unsupported candidate model family"):
        _validate_uncertainty(UncertaintyInput(0.01, 0.05), "random_forest")
    with pytest.raises(ValueError, match="finite"):
        _validate_uncertainty(
            UncertaintyInput(float("inf"), 0.05), "logistic_regression"
        )
    with pytest.raises(ValueError, match="non-negative"):
        _validate_uncertainty(
            UncertaintyInput(-0.01, 0.05), "logistic_regression"
        )

    # Candidate
    with pytest.raises(ValueError, match="[0, 1]"):
        _validate_candidate(
            CandidatePrediction("logistic_regression", 1.5, 1, 0.5)
        )
    with pytest.raises(ValueError, match="Candidate label must match"):
        _validate_candidate(
            CandidatePrediction("logistic_regression", 0.6, 0, 0.5)
        )
    with pytest.raises(ValueError, match="classification threshold must remain 0.50"):
        _validate_candidate(
            CandidatePrediction("logistic_regression", 0.6, 1, 0.4)
        )


# ==============================================================================
# 2. Gate Scenario Tests (Integration with Real Artifacts)
# ==============================================================================


def test_build_gate_panel_returns_non_empty_with_required_columns() -> None:
    panel = build_gate_panel(panel_type="clean")
    assert not panel.empty
    assert len(panel) == 3000  # 1500 test applicants * 2 model families

    required_columns = {
        "applicant_id",
        "financial_stress_event",
        "mean_probability",
        "predictive_variance",
        "ood_score",
        "total_expected_months",
        "observed_months",
        "missing_month_count",
        "missing_required_field_count",
        "latest_evidence_age_months",
        "consistency_violation_count",
        "unverified_record_count",
        "scenario_label",
        "model_family",
    }
    assert required_columns.issubset(panel.columns)


def test_panel_has_no_forbidden_leakage_columns() -> None:
    panel = build_gate_panel(panel_type="all")
    assert "financial_stress_probability" not in panel.columns
    assert "population_group" not in panel.columns
    assert "latent_trait" not in panel.columns
    assert "financial_stress_event" in panel.columns  # Allowed strictly as evaluation target


def test_clean_scenario_rows_have_no_mandatory_failures() -> None:
    panel = build_gate_panel(panel_type="clean")
    assert (panel["consistency_violation_count"] == 0).all()
    assert (panel["observed_months"] == 12).all()
    assert (panel["missing_month_count"] == 0).all()
    assert (panel["unverified_record_count"] == 0).all()
    assert (panel["missing_required_field_count"] == 0).all()


def test_evidence_degraded_rows_have_higher_review_or_block_rates_than_clean() -> None:
    clean_panel = build_gate_panel(panel_type="clean")
    ev_panel = build_gate_panel(panel_type="evidence")

    gated_clean = apply_safety_gate(clean_panel, experiment_id="clean_test")
    gated_ev = apply_safety_gate(
        ev_panel[ev_panel["experiment_group"] == "evidence_degraded"],
        experiment_id="ev_test",
    )

    clean_escalation_rate = (
        (gated_clean["gate_outcome"] != SAFE_TO_AUTOMATE).sum() / len(gated_clean)
    )
    ev_escalation_rate = (
        (gated_ev["gate_outcome"] != SAFE_TO_AUTOMATE).sum() / len(gated_ev)
    )

    # Defective evidence must produce substantially more reviews and blocks than clean reference
    assert ev_escalation_rate > clean_escalation_rate


# ==============================================================================
# 3. Gate Metrics Tests
# ==============================================================================


def test_evaluate_gate_returns_all_required_keys_and_valid_ranges() -> None:
    clean_panel = build_gate_panel(panel_type="clean")
    gated_clean = apply_safety_gate(clean_panel, experiment_id="metrics_test")

    required_keys = {
        "experiment_name",
        "model_family",
        "total_cases",
        "automated_count",
        "review_count",
        "blocked_count",
        "automation_coverage",
        "review_rate",
        "block_rate",
        "unsafe_automated_count",
        "unsafe_automated_rate",
        "unsafe_automated_coverage_rate",
        "unsafe_automated_missed_stress_count",
        "unsafe_automated_missed_stress_rate",
        "false_escalation_count",
        "false_escalation_rate",
        "accuracy_on_automated",
        "reason_code_distribution",
        "scenario_breakdown",
        "severity_breakdown",
    }

    for mf in ("logistic_regression", "hist_gradient_boosting"):
        metrics = evaluate_gate(gated_clean, model_family=mf, experiment_name=f"clean_{mf}")
        assert required_keys.issubset(metrics.keys())

        # Rates must sum to 1.0
        rate_sum = (
            metrics["automation_coverage"]
            + metrics["review_rate"]
            + metrics["block_rate"]
        )
        assert np.isclose(rate_sum, 1.0, atol=1e-6)

        # Range checks
        assert 0.0 <= metrics["automation_coverage"] <= 1.0
        assert 0.0 <= metrics["review_rate"] <= 1.0
        assert 0.0 <= metrics["block_rate"] <= 1.0
        assert 0.0 <= metrics["unsafe_automated_rate"] <= 1.0
        assert 0.0 <= metrics["false_escalation_rate"] <= 1.0
        assert metrics["total_cases"] == (
            metrics["automated_count"]
            + metrics["review_count"]
            + metrics["blocked_count"]
        )


def test_compare_experiments_formats_table_correctly() -> None:
    clean_panel = build_gate_panel(panel_type="clean")
    gated_clean = apply_safety_gate(clean_panel, experiment_id="comp_test")
    m1 = evaluate_gate(gated_clean, model_family="logistic_regression", experiment_name="exp_lr")
    m2 = evaluate_gate(gated_clean, model_family="hist_gradient_boosting", experiment_name="exp_hgb")

    comp_df = compare_experiments([m1, m2])
    assert len(comp_df) == 2
    assert "experiment_name" in comp_df.columns
    assert "automation_coverage" in comp_df.columns
    assert "unsafe_automated_count" in comp_df.columns


def test_compute_risk_coverage_curve_monotonic_coverage() -> None:
    clean_panel = build_gate_panel(panel_type="clean")
    thresholds = [2.0, 4.0, 6.0, 8.0, 10.0]
    curve = compute_risk_coverage_curve(
        clean_panel,
        model_family="logistic_regression",
        thresholds=thresholds,
    )
    assert len(curve) == len(thresholds)
    # As OOD threshold increases (gate becomes more permissive), automation coverage should be non-decreasing
    coverages = curve["automation_coverage"].to_numpy()
    assert (np.diff(coverages) >= -1e-6).all()
