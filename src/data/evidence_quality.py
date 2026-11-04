"""Controlled evidence-quality experiments for synthetic financial histories.

This module is deliberately independent from the risk-model feature contract.
It models observable evidence defects and writes separate experimental artifacts;
it never changes clean data, target labels, model splits, or baseline artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.validation import assert_no_forbidden_credit_variables
from src.utils.reproducibility import DEFAULT_RANDOM_SEED

REQUIRED_EVIDENCE_FIELDS = (
    "monthly_income",
    "monthly_expenses",
    "savings_balance",
    "utility_bill_amount",
    "utility_payment_status",
    "rent_amount",
    "rent_payment_status",
    "transaction_count",
    "incoming_transaction_amount",
    "outgoing_transaction_amount",
    "end_of_month_balance",
    "net_cashflow",
)

SCENARIOS = (
    "missingness",
    "stale_records",
    "inconsistent_records",
    "unverified_provenance",
)
SEVERITIES = ("low", "medium", "high")

SEVERITY_SETTINGS = {
    "low": {"affected_rate": 0.15, "months": 1, "field_rate": 0.08, "verification_rate": 0.20},
    "medium": {"affected_rate": 0.30, "months": 3, "field_rate": 0.16, "verification_rate": 0.45},
    "high": {"affected_rate": 0.50, "months": 5, "field_rate": 0.28, "verification_rate": 0.70},
}


@dataclass(frozen=True)
class EvidenceQualityConfig:
    """Reproducible settings for the evidence-quality experiment."""

    seed: int = DEFAULT_RANDOM_SEED
    assessment_month: int = 12
    scenarios: tuple[str, ...] = SCENARIOS
    severities: tuple[str, ...] = SEVERITIES
    sample_applicants_per_variant: int = 5
    quality_dimensions: tuple[str, ...] = field(
        default_factory=lambda: (
            "completeness_score",
            "consistency_score",
            "freshness_score",
            "provenance_score",
        )
    )


def run_evidence_quality_experiment(
    raw_history_path: str | Path = "data/raw/synthetic/financial_history.csv",
    model_dataset_path: str | Path = "data/processed/synthetic/model_dataset.csv",
    output_dir: str | Path = "experiments/evidence_quality",
    config: EvidenceQualityConfig | None = None,
) -> dict[str, object]:
    """Run all configured controlled degradation variants and save artifacts.

    Applicant selection depends solely on stable applicant IDs and a fixed random
    seed. It never reads targets, stress probabilities, populations, or latents.
    The model dataset is loaded only to audit that it remains unmodified.
    """

    config = config or EvidenceQualityConfig()
    _validate_config(config)
    clean_history = pd.read_csv(raw_history_path)
    clean_model_dataset = pd.read_csv(model_dataset_path)
    _validate_clean_inputs(clean_history, clean_model_dataset, config)
    model_dataset_bytes = Path(model_dataset_path).read_bytes()

    all_metadata: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    all_samples: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    variant_seed_sequence = np.random.SeedSequence(config.seed).spawn(
        len(config.scenarios) * len(config.severities)
    )

    for (scenario, severity), child_seed in zip(
        ((scenario, severity) for scenario in config.scenarios for severity in config.severities),
        variant_seed_sequence,
    ):
        rng = np.random.default_rng(child_seed)
        degraded, record_metadata, events = create_degraded_evidence(
            clean_history=clean_history,
            scenario=scenario,
            severity=severity,
            assessment_month=config.assessment_month,
            rng=rng,
        )
        applicant_metadata = measure_evidence_quality(
            evidence_history=degraded,
            clean_history=clean_history,
            record_metadata=record_metadata,
            scenario=scenario,
            severity=severity,
            assessment_month=config.assessment_month,
        )
        events.insert(0, "severity", severity)
        events.insert(0, "scenario", scenario)
        all_metadata.append(applicant_metadata)
        all_events.append(events)
        all_samples.append(
            _sample_degraded_records(
                degraded,
                applicant_metadata,
                scenario,
                severity,
                config.sample_applicants_per_variant,
            )
        )
        summaries.append(_summarize_variant(applicant_metadata))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata = pd.concat(all_metadata, ignore_index=True)
    events = pd.concat(all_events, ignore_index=True)
    samples = pd.concat(all_samples, ignore_index=True)
    summary = pd.DataFrame(summaries)

    metadata.to_csv(output_path / "applicant_evidence_quality.csv", index=False)
    events.to_csv(output_path / "degradation_events.csv", index=False)
    samples.to_csv(output_path / "degraded_evidence_samples.csv", index=False)
    summary.to_csv(output_path / "scenario_summary.csv", index=False)

    experiment_metadata: dict[str, object] = {
        "config": asdict(config),
        "selection_basis": "fixed-seed random sampling of applicant_id only",
        "quality_dimensions": list(config.quality_dimensions),
        "quality_score_definition": (
            "provisional unweighted arithmetic mean of the four separate "
            "quality dimensions; not empirically validated and not a probability"
        ),
        "degraded_evidence_representation": (
            "degradation_events.csv is a reproducible overlay on the clean "
            "financial history; degraded_evidence_samples.csv contains materialized "
            "records for inspection"
        ),
        "derived_model_features": "not produced in this phase",
        "clean_model_dataset_unchanged": Path(model_dataset_path).read_bytes()
        == model_dataset_bytes,
        "targets_regenerated": False,
    }
    with (output_path / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment_metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return experiment_metadata


def create_degraded_evidence(
    clean_history: pd.DataFrame,
    scenario: str,
    severity: str,
    assessment_month: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create one degradation variant without using labels or population labels."""

    if scenario not in SCENARIOS or severity not in SEVERITIES:
        raise ValueError("Scenario and severity must be supported values.")
    applicant_ids = np.sort(clean_history["applicant_id"].unique())
    settings = SEVERITY_SETTINGS[severity]
    affected_count = int(round(len(applicant_ids) * settings["affected_rate"]))
    affected_ids = np.sort(rng.choice(applicant_ids, size=affected_count, replace=False))
    affected_set = set(affected_ids.tolist())

    evidence = clean_history.drop(columns=["population_group"], errors="ignore").copy(
        deep=True
    )
    evidence["simulated_verification_status"] = "verified"
    evidence["simulated_defect_introduced"] = False
    events: list[dict[str, object]] = []

    if scenario == "missingness":
        evidence, events = _apply_missingness(evidence, affected_set, severity, rng)
    elif scenario == "stale_records":
        evidence, events = _apply_staleness(
            evidence, affected_set, severity, assessment_month
        )
    elif scenario == "inconsistent_records":
        evidence, events = _apply_inconsistency(evidence, affected_set, severity, rng)
    elif scenario == "unverified_provenance":
        evidence, events = _apply_unverified_provenance(
            evidence, affected_set, severity, rng
        )

    affected_mask = evidence["applicant_id"].isin(affected_set)
    evidence.loc[affected_mask, "simulated_defect_introduced"] = True
    record_metadata = pd.DataFrame(
        {
            "applicant_id": applicant_ids,
            "simulated_defect_introduced": [
                applicant_id in affected_set for applicant_id in applicant_ids
            ],
            "simulated_defect_type": [
                scenario if applicant_id in affected_set else "none"
                for applicant_id in applicant_ids
            ],
        }
    )
    event_frame = pd.DataFrame(events, columns=_event_columns())
    return evidence, record_metadata, event_frame


def measure_evidence_quality(
    evidence_history: pd.DataFrame,
    clean_history: pd.DataFrame,
    record_metadata: pd.DataFrame,
    scenario: str,
    severity: str,
    assessment_month: int,
) -> pd.DataFrame:
    """Measure four observable evidence-quality dimensions per applicant."""

    clean_counts = clean_history.groupby("applicant_id")["month"].nunique()
    observed_counts = evidence_history.groupby("applicant_id")["month"].nunique()
    latest_month = evidence_history.groupby("applicant_id")["month"].max()
    missing_field_counts = evidence_history.groupby("applicant_id")[
        list(REQUIRED_EVIDENCE_FIELDS)
    ].apply(lambda frame: int(frame.isna().sum().sum()))
    violations = _count_consistency_violations(evidence_history)
    unverified_counts = (
        evidence_history["simulated_verification_status"].eq("unverified")
        .groupby(evidence_history["applicant_id"])
        .sum()
    )

    quality = record_metadata.copy()
    quality["total_expected_months"] = quality["applicant_id"].map(clean_counts)
    quality["observed_months"] = quality["applicant_id"].map(observed_counts).fillna(0)
    quality["missing_month_count"] = (
        quality["total_expected_months"] - quality["observed_months"]
    ).astype(int)
    quality["missing_required_field_count"] = quality["applicant_id"].map(
        missing_field_counts
    ).fillna(0).astype(int)
    quality["latest_observed_month"] = quality["applicant_id"].map(latest_month).fillna(0)
    quality["observation_age_months"] = (
        assessment_month - quality["latest_observed_month"]
    ).astype(int)
    quality["consistency_violation_count"] = quality["applicant_id"].map(
        violations
    ).fillna(0).astype(int)
    quality["unverified_record_count"] = quality["applicant_id"].map(
        unverified_counts
    ).fillna(0).astype(int)

    total_field_slots = quality["total_expected_months"] * len(REQUIRED_EVIDENCE_FIELDS)
    quality["completeness_score"] = (
        1
        - quality["missing_month_count"] / quality["total_expected_months"]
        - quality["missing_required_field_count"] / total_field_slots
    ).clip(0, 1)
    quality["consistency_score"] = (
        1 - quality["consistency_violation_count"] / quality["total_expected_months"]
    ).clip(0, 1)
    quality["freshness_score"] = (
        1 - quality["observation_age_months"] / (assessment_month - 1)
    ).clip(0, 1)
    quality["provenance_score"] = (
        1 - quality["unverified_record_count"] / quality["observed_months"].clip(lower=1)
    ).clip(0, 1)
    quality["evidence_quality_score"] = quality[
        [
            "completeness_score",
            "consistency_score",
            "freshness_score",
            "provenance_score",
        ]
    ].mean(axis=1)
    quality.insert(0, "severity", severity)
    quality.insert(0, "scenario", scenario)
    return quality


def _apply_missingness(
    evidence: pd.DataFrame,
    affected_ids: set[int],
    severity: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Remove monthly records and required field values for selected applicants."""

    settings = SEVERITY_SETTINGS[severity]
    drop_indices: list[int] = []
    events: list[dict[str, object]] = []
    for applicant_id in affected_ids:
        applicant_rows = evidence[evidence["applicant_id"] == applicant_id]
        months = applicant_rows["month"].to_numpy()
        dropped_months = rng.choice(
            months, size=min(settings["months"], len(months) - 1), replace=False
        )
        for month in dropped_months:
            index = applicant_rows.loc[applicant_rows["month"] == month].index[0]
            drop_indices.append(index)
            events.append(_event(applicant_id, int(month), "missing_month", None, "removed"))

        remaining = applicant_rows[~applicant_rows["month"].isin(dropped_months)]
        field_mask = rng.random((len(remaining), len(REQUIRED_EVIDENCE_FIELDS))) < settings["field_rate"]
        for row_position, row_index in enumerate(remaining.index):
            for field_position, field_name in enumerate(REQUIRED_EVIDENCE_FIELDS):
                if field_mask[row_position, field_position]:
                    evidence.loc[row_index, field_name] = pd.NA
                    events.append(
                        _event(
                            applicant_id,
                            int(evidence.loc[row_index, "month"]),
                            "missing_field",
                            field_name,
                            "missing",
                        )
                    )
    return evidence.drop(index=drop_indices).reset_index(drop=True), events


def _apply_staleness(
    evidence: pd.DataFrame,
    affected_ids: set[int],
    severity: str,
    assessment_month: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Remove most recent observations to create explicitly stale evidence."""

    stale_months = SEVERITY_SETTINGS[severity]["months"]
    threshold = assessment_month - stale_months + 1
    mask = evidence["applicant_id"].isin(affected_ids) & (evidence["month"] >= threshold)
    removed = evidence.loc[mask, ["applicant_id", "month"]]
    events = [
        _event(int(row.applicant_id), int(row.month), "stale_record", None, "removed")
        for row in removed.itertuples(index=False)
    ]
    return evidence.loc[~mask].reset_index(drop=True), events


def _apply_inconsistency(
    evidence: pd.DataFrame,
    affected_ids: set[int],
    severity: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Create auditable cashflow contradictions while retaining valid numbers."""

    events: list[dict[str, object]] = []
    months_to_change = SEVERITY_SETTINGS[severity]["months"]
    for applicant_id in affected_ids:
        applicant_rows = evidence[evidence["applicant_id"] == applicant_id]
        selected = rng.choice(
            applicant_rows.index,
            size=min(months_to_change, len(applicant_rows)),
            replace=False,
        )
        for row_index in selected:
            income = float(evidence.loc[row_index, "monthly_income"])
            evidence.loc[row_index, "net_cashflow"] = round(
                float(evidence.loc[row_index, "net_cashflow"])
                + income * rng.uniform(0.15, 0.35),
                2,
            )
            events.append(
                _event(
                    applicant_id,
                    int(evidence.loc[row_index, "month"]),
                    "cashflow_contradiction",
                    "net_cashflow",
                    "modified",
                )
            )
    return evidence, events


def _apply_unverified_provenance(
    evidence: pd.DataFrame,
    affected_ids: set[int],
    severity: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Mark selected records unverified without changing financial values."""

    events: list[dict[str, object]] = []
    rate = SEVERITY_SETTINGS[severity]["verification_rate"]
    for applicant_id in affected_ids:
        indexes = evidence.index[evidence["applicant_id"] == applicant_id]
        selected = rng.choice(indexes, size=int(round(len(indexes) * rate)), replace=False)
        evidence.loc[selected, "simulated_verification_status"] = "unverified"
        for row_index in selected:
            events.append(
                _event(
                    applicant_id,
                    int(evidence.loc[row_index, "month"]),
                    "unverified_source",
                    "simulated_verification_status",
                    "unverified",
                )
            )
    return evidence, events


def _count_consistency_violations(evidence: pd.DataFrame) -> pd.Series:
    """Count observable contradictory or implausible records per applicant."""

    numeric_fields = [
        "monthly_income",
        "monthly_expenses",
        "savings_balance",
        "utility_bill_amount",
        "rent_amount",
        "transaction_count",
        "incoming_transaction_amount",
        "outgoing_transaction_amount",
        "end_of_month_balance",
    ]
    invalid_numeric = evidence[numeric_fields].lt(0).any(axis=1)
    cashflow_expected = (
        evidence["monthly_income"]
        - evidence["monthly_expenses"]
        - evidence["rent_amount"]
        - evidence["utility_bill_amount"]
    )
    cashflow_mismatch = (evidence["net_cashflow"] - cashflow_expected).abs() > 0.02
    implausible_rent = evidence["rent_amount"] > 0.65 * evidence["monthly_income"]
    implausible_expenses = evidence["monthly_expenses"] > 1.45 * evidence["monthly_income"]
    violation = (invalid_numeric | cashflow_mismatch | implausible_rent | implausible_expenses).fillna(False)
    return violation.groupby(evidence["applicant_id"]).sum()


def _sample_degraded_records(
    evidence: pd.DataFrame,
    applicant_metadata: pd.DataFrame,
    scenario: str,
    severity: str,
    count: int,
) -> pd.DataFrame:
    """Materialize a small inspection sample; the event overlay is authoritative."""

    affected = applicant_metadata.loc[
        applicant_metadata["simulated_defect_introduced"], "applicant_id"
    ].head(count)
    return evidence[evidence["applicant_id"].isin(affected)].assign(
        scenario=scenario, severity=severity
    )


def _summarize_variant(metadata: pd.DataFrame) -> dict[str, object]:
    """Summarize quality dimensions for one scenario and severity."""

    return {
        "scenario": metadata["scenario"].iloc[0],
        "severity": metadata["severity"].iloc[0],
        "applicant_count": int(len(metadata)),
        "affected_applicant_count": int(metadata["simulated_defect_introduced"].sum()),
        "mean_completeness_score": float(metadata["completeness_score"].mean()),
        "mean_consistency_score": float(metadata["consistency_score"].mean()),
        "mean_freshness_score": float(metadata["freshness_score"].mean()),
        "mean_provenance_score": float(metadata["provenance_score"].mean()),
        "mean_evidence_quality_score": float(metadata["evidence_quality_score"].mean()),
    }


def _event(
    applicant_id: int,
    month: int,
    defect_type: str,
    field_name: str | None,
    action: str,
) -> dict[str, object]:
    """Create one audit event that can be applied to clean history."""

    return {
        "applicant_id": applicant_id,
        "month": month,
        "simulated_defect_type": defect_type,
        "field_name": field_name,
        "action": action,
    }


def _event_columns() -> list[str]:
    return ["applicant_id", "month", "simulated_defect_type", "field_name", "action"]


def _validate_config(config: EvidenceQualityConfig) -> None:
    """Validate fixed experiment choices before modifying any copy of evidence."""

    if config.assessment_month <= 1:
        raise ValueError("assessment_month must be greater than 1.")
    if config.sample_applicants_per_variant <= 0:
        raise ValueError("sample_applicants_per_variant must be positive.")
    if not set(config.scenarios).issubset(SCENARIOS):
        raise ValueError("Unsupported scenario in configuration.")
    if not set(config.severities).issubset(SEVERITIES):
        raise ValueError("Unsupported severity in configuration.")


def _validate_clean_inputs(
    clean_history: pd.DataFrame,
    clean_model_dataset: pd.DataFrame,
    config: EvidenceQualityConfig,
) -> None:
    """Verify clean input schema without reading targets for corruption logic."""

    assert_no_forbidden_credit_variables(clean_history)
    assert_no_forbidden_credit_variables(clean_model_dataset)
    required_history = {"applicant_id", "month", *REQUIRED_EVIDENCE_FIELDS}
    if missing := required_history - set(clean_history.columns):
        raise ValueError(f"Missing clean financial history columns: {sorted(missing)}")
    if clean_history["month"].max() < config.assessment_month:
        raise ValueError("Assessment month is beyond available evidence history.")


if __name__ == "__main__":
    run_evidence_quality_experiment()
