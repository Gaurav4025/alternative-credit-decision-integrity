"""Deterministic, interpretable safety rules for synthetic research decisions.

The gate deliberately keeps evidence integrity, covariate shift, and bootstrap
prediction uncertainty as separate signals.  It does not alter a candidate
model probability or make a financial-risk prediction of its own.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Literal


SAFE_TO_AUTOMATE = "SAFE_TO_AUTOMATE"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
BLOCKED = "BLOCKED"
GATE_VERSION = "phase8_hierarchical_v1"
REQUIRED_FIELD_COUNT = 12
OOD_THRESHOLD = 5.459076558162995
UNCERTAINTY_THRESHOLDS = {
    "logistic_regression": 0.000956288247270191,
    "hist_gradient_boosting": 0.01628725348136607,
}


@dataclass(frozen=True)
class EvidenceQualityInput:
    """Raw applicant-level evidence measurements produced by Phase 5."""

    total_expected_months: int
    observed_months: int
    missing_month_count: int
    missing_required_field_count: int
    latest_evidence_age_months: int
    consistency_violation_count: int
    unverified_record_count: int


@dataclass(frozen=True)
class OODInput:
    """The selected Phase 6 Shrinkage Mahalanobis score and fixed threshold."""

    score: float
    threshold: float = OOD_THRESHOLD


@dataclass(frozen=True)
class UncertaintyInput:
    """Bootstrap predictive variance for the same candidate model family."""

    predictive_variance: float
    threshold: float


@dataclass(frozen=True)
class CandidatePrediction:
    """An unchanged raw probability and 0.50-threshold candidate label."""

    model_family: Literal["logistic_regression", "hist_gradient_boosting"]
    probability: float
    label: int
    classification_threshold: float = 0.5


@dataclass(frozen=True)
class GateDecision:
    """Auditable gate result with every signal retained as supplied."""

    applicant_id: int
    scenario_id: str
    experiment_id: str
    gate_version: str
    outcome: str
    reason_codes: tuple[str, ...]
    candidate: CandidatePrediction
    evidence: EvidenceQualityInput
    ood: OODInput
    uncertainty: UncertaintyInput

    def as_dict(self) -> dict[str, object]:
        """Return a flat, serializable decision record without evaluation labels."""

        result = {
            "applicant_id": self.applicant_id,
            "scenario_id": self.scenario_id,
            "experiment_id": self.experiment_id,
            "gate_version": self.gate_version,
            "gate_outcome": self.outcome,
            "reason_codes": "|".join(self.reason_codes),
        }
        result.update({f"candidate_{key}": value for key, value in asdict(self.candidate).items()})
        result.update({f"evidence_{key}": value for key, value in asdict(self.evidence).items()})
        result.update({"ood_score": self.ood.score, "ood_threshold": self.ood.threshold})
        result.update({
            "uncertainty_predictive_variance": self.uncertainty.predictive_variance,
            "uncertainty_threshold": self.uncertainty.threshold,
        })
        return result


def decide(
    applicant_id: int,
    scenario_id: str,
    experiment_id: str,
    evidence: EvidenceQualityInput,
    ood: OODInput,
    uncertainty: UncertaintyInput,
    candidate: CandidatePrediction,
) -> GateDecision:
    """Apply the fixed hierarchy and return a deterministic, fully explained result."""

    _validate_identity(applicant_id, scenario_id, experiment_id)
    _validate_evidence(evidence)
    _validate_ood(ood)
    _validate_uncertainty(uncertainty, candidate.model_family)
    _validate_candidate(candidate)

    reasons: list[str] = []
    if evidence.consistency_violation_count > 0:
        reasons.append("EVIDENCE_CONSISTENCY_CONTRADICTION")
    if evidence.observed_months == 0:
        reasons.append("EVIDENCE_NO_OBSERVATIONS")
    if evidence.observed_months > 0 and evidence.unverified_record_count == evidence.observed_months:
        reasons.append("EVIDENCE_NO_VERIFIED_RECORDS")
    if evidence.observed_months > 0 and evidence.missing_required_field_count >= evidence.observed_months * REQUIRED_FIELD_COUNT:
        reasons.append("EVIDENCE_NO_USABLE_REQUIRED_FIELDS")
    mandatory_reasons = tuple(reasons)

    if evidence.missing_month_count > 0:
        reasons.append("EVIDENCE_MISSING_MONTH")
    if evidence.missing_required_field_count > 0:
        reasons.append("EVIDENCE_MISSING_REQUIRED_FIELD")
    if evidence.latest_evidence_age_months > 0:
        reasons.append("EVIDENCE_STALE_OBSERVATION")
    if 0 < evidence.unverified_record_count < evidence.observed_months:
        reasons.append("EVIDENCE_PARTIALLY_UNVERIFIED")
    if ood.score >= ood.threshold:
        reasons.append("OOD_SHRINKAGE_MAHALANOBIS_HIGH")
    if uncertainty.predictive_variance >= uncertainty.threshold:
        reasons.append(
            "UNCERTAINTY_LOGISTIC_VARIANCE_HIGH"
            if candidate.model_family == "logistic_regression"
            else "UNCERTAINTY_HGB_VARIANCE_HIGH"
        )

    outcome = BLOCKED if mandatory_reasons else REVIEW_REQUIRED if reasons else SAFE_TO_AUTOMATE
    return GateDecision(
        applicant_id=applicant_id,
        scenario_id=scenario_id,
        experiment_id=experiment_id,
        gate_version=GATE_VERSION,
        outcome=outcome,
        reason_codes=tuple(reasons),
        candidate=candidate,
        evidence=evidence,
        ood=ood,
        uncertainty=uncertainty,
    )


def _validate_identity(applicant_id: int, scenario_id: str, experiment_id: str) -> None:
    if not isinstance(applicant_id, int) or applicant_id <= 0:
        raise ValueError("applicant_id must be a positive integer.")
    if not scenario_id or not experiment_id:
        raise ValueError("scenario_id and experiment_id must be non-empty strings.")


def _validate_evidence(evidence: EvidenceQualityInput) -> None:
    values = asdict(evidence)
    if any(not isinstance(value, int) or value < 0 for value in values.values()):
        raise ValueError("Evidence counts and ages must be non-negative integers.")
    if evidence.total_expected_months == 0:
        raise ValueError("total_expected_months must be positive.")
    if evidence.observed_months > evidence.total_expected_months:
        raise ValueError("observed_months cannot exceed total_expected_months.")
    if evidence.missing_month_count != evidence.total_expected_months - evidence.observed_months:
        raise ValueError("missing_month_count must equal expected minus observed months.")
    if evidence.unverified_record_count > evidence.observed_months:
        raise ValueError("unverified_record_count cannot exceed observed_months.")
    if evidence.missing_required_field_count > evidence.observed_months * REQUIRED_FIELD_COUNT:
        raise ValueError("missing_required_field_count exceeds available field slots.")


def _validate_ood(ood: OODInput) -> None:
    if not isfinite(ood.score) or not isfinite(ood.threshold) or ood.score < 0 or ood.threshold < 0:
        raise ValueError("OOD score and threshold must be finite non-negative values.")


def _validate_uncertainty(uncertainty: UncertaintyInput, model_family: str) -> None:
    if model_family not in UNCERTAINTY_THRESHOLDS:
        raise ValueError("Unsupported candidate model family.")
    if not isfinite(uncertainty.predictive_variance) or not isfinite(uncertainty.threshold):
        raise ValueError("Uncertainty values must be finite.")
    if uncertainty.predictive_variance < 0 or uncertainty.threshold < 0:
        raise ValueError("Uncertainty values must be non-negative.")


def _validate_candidate(candidate: CandidatePrediction) -> None:
    if candidate.model_family not in UNCERTAINTY_THRESHOLDS:
        raise ValueError("Unsupported candidate model family.")
    if not isfinite(candidate.probability) or not 0 <= candidate.probability <= 1:
        raise ValueError("Candidate probability must be finite and in [0, 1].")
    if candidate.label not in (0, 1):
        raise ValueError("Candidate label must be 0 or 1.")
    if candidate.classification_threshold != 0.5:
        raise ValueError("The Phase 8 candidate classification threshold must remain 0.50.")
    if candidate.label != int(candidate.probability >= candidate.classification_threshold):
        raise ValueError("Candidate label must match the unchanged candidate probability threshold.")
