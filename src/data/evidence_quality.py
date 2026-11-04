"""Controlled evidence-quality experiments for synthetic financial histories.

This module creates evidence-only artifacts for future research. It never
changes clean datasets, target labels, model splits, model features, or Phase
1-4 artifacts.
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
    "monthly_income", "monthly_expenses", "savings_balance", "utility_bill_amount",
    "utility_payment_status", "rent_amount", "rent_payment_status", "transaction_count",
    "incoming_transaction_amount", "outgoing_transaction_amount", "end_of_month_balance",
    "net_cashflow",
)
SCENARIOS = ("missingness", "stale_records", "inconsistent_records", "unverified_provenance")
SEVERITIES = ("low", "medium", "high")
SEVERITY_SETTINGS = {
    "low": {"affected_rate": 0.15, "months": 1, "field_rate": 0.08, "verification_rate": 0.20},
    "medium": {"affected_rate": 0.30, "months": 3, "field_rate": 0.16, "verification_rate": 0.45},
    "high": {"affected_rate": 0.50, "months": 5, "field_rate": 0.28, "verification_rate": 0.70},
}


@dataclass(frozen=True)
class EvidenceQualityConfig:
    """Configuration for deterministic, label-blind evidence degradation."""

    seed: int = DEFAULT_RANDOM_SEED
    assessment_month: int = 12
    scenarios: tuple[str, ...] = SCENARIOS
    severities: tuple[str, ...] = SEVERITIES
    sample_applicants_per_variant: int = 5
    quality_dimensions: tuple[str, ...] = field(
        default_factory=lambda: (
            "completeness_score", "consistency_score", "freshness_score", "provenance_score"
        )
    )


def run_evidence_quality_experiment(
    raw_history_path: str | Path = "data/raw/synthetic/financial_history.csv",
    model_dataset_path: str | Path = "data/processed/synthetic/model_dataset.csv",
    output_dir: str | Path = "experiments/evidence_quality",
    config: EvidenceQualityConfig | None = None,
) -> dict[str, object]:
    """Write reproducible evidence-quality artifacts without changing inputs."""

    config = config or EvidenceQualityConfig()
    _validate_config(config)
    history_path, dataset_path = Path(raw_history_path), Path(model_dataset_path)
    clean_history = pd.read_csv(history_path)
    clean_model_dataset = pd.read_csv(dataset_path)
    _validate_clean_inputs(clean_history, clean_model_dataset, config)
    history_bytes, model_bytes = history_path.read_bytes(), dataset_path.read_bytes()

    metadata_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    sample_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    seeds = np.random.SeedSequence(config.seed).spawn(len(config.scenarios) * len(config.severities))
    variants = ((scenario, severity) for scenario in config.scenarios for severity in config.severities)

    for (scenario, severity), child_seed in zip(variants, seeds):
        degraded, applicant_metadata, events = create_degraded_evidence(
            clean_history, scenario, severity, config.assessment_month, np.random.default_rng(child_seed)
        )
        quality = measure_evidence_quality(
            degraded, clean_history, applicant_metadata, scenario, severity, config.assessment_month
        )
        metadata_frames.append(quality)
        event_frames.append(events.assign(scenario=scenario, severity=severity))
        sample_frames.append(_sample_degraded_records(
            degraded, quality, scenario, severity, config.sample_applicants_per_variant
        ))
        summaries.append(_summarize_variant(quality))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pd.concat(metadata_frames, ignore_index=True).to_csv(output_path / "applicant_evidence_quality.csv", index=False)
    pd.concat(event_frames, ignore_index=True).to_csv(output_path / "degradation_events.csv", index=False)
    pd.concat(sample_frames, ignore_index=True).to_csv(output_path / "degraded_evidence_samples.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_path / "scenario_summary.csv", index=False)

    result = {
        "config": asdict(config),
        "selection_basis": "fixed-seed random sampling of applicant_id only",
        "staleness_assumption": (
            "month remains the underlying financial observation period; stale records remain present "
            "and receive synthetic reporting-lag metadata rather than fabricated real timestamps"
        ),
        "quality_dimensions": list(config.quality_dimensions),
        "quality_score_definition": (
            "provisional unweighted arithmetic mean of separate quality dimensions; "
            "not empirically validated and not a probability"
        ),
        "clean_history_unchanged": history_path.read_bytes() == history_bytes,
        "clean_model_dataset_unchanged": dataset_path.read_bytes() == model_bytes,
        "targets_regenerated": False,
        "derived_model_features": "not produced in this phase",
    }
    with (output_path / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def create_degraded_evidence(
    clean_history: pd.DataFrame,
    scenario: str,
    severity: str,
    assessment_month: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create one variant with distinct applicant-level and record-level metadata."""

    if scenario not in SCENARIOS or severity not in SEVERITIES:
        raise ValueError("Scenario and severity must be supported values.")
    applicant_ids = np.sort(clean_history["applicant_id"].unique())
    affected_count = int(round(len(applicant_ids) * SEVERITY_SETTINGS[severity]["affected_rate"]))
    affected_ids = np.sort(rng.choice(applicant_ids, affected_count, replace=False))
    affected_set = set(affected_ids.tolist())

    evidence = clean_history.drop(columns=["population_group"], errors="ignore").copy()
    evidence["simulated_verification_status"] = "verified"
    evidence["synthetic_reporting_lag_months"] = 0
    evidence["synthetic_evidence_age_months"] = assessment_month - evidence["month"]
    evidence["simulated_record_defect_introduced"] = False
    evidence["simulated_record_defect_type"] = "none"

    if scenario == "missingness":
        evidence, events = _apply_missingness(evidence, affected_set, severity, rng)
    elif scenario == "stale_records":
        evidence, events = _apply_staleness(evidence, affected_set, severity)
    elif scenario == "inconsistent_records":
        evidence, events = _apply_inconsistency(evidence, affected_set, severity, rng)
    else:
        evidence, events = _apply_unverified_provenance(evidence, affected_set, severity, rng)

    applicant_metadata = pd.DataFrame({
        "applicant_id": applicant_ids,
        "simulated_applicant_defect_introduced": [applicant_id in affected_set for applicant_id in applicant_ids],
        "simulated_applicant_defect_type": [scenario if applicant_id in affected_set else "none" for applicant_id in applicant_ids],
    })
    return evidence, applicant_metadata, pd.DataFrame(events, columns=_event_columns())


def measure_evidence_quality(
    evidence_history: pd.DataFrame,
    clean_history: pd.DataFrame,
    applicant_metadata: pd.DataFrame,
    scenario: str,
    severity: str,
    assessment_month: int,
) -> pd.DataFrame:
    """Measure separate completeness, consistency, freshness, and provenance."""

    expected_months = clean_history.groupby("applicant_id")["month"].nunique()
    observed_months = evidence_history.groupby("applicant_id")["month"].nunique()
    missing_fields = evidence_history.groupby("applicant_id")[list(REQUIRED_EVIDENCE_FIELDS)].apply(
        lambda frame: int(frame.isna().sum().sum())
    )
    violations = _count_consistency_violations(evidence_history)
    latest_age = evidence_history.groupby("applicant_id")["synthetic_evidence_age_months"].min()
    unverified = evidence_history["simulated_verification_status"].eq("unverified").groupby(evidence_history["applicant_id"]).sum()
    defective_records = evidence_history["simulated_record_defect_introduced"].groupby(evidence_history["applicant_id"]).sum()

    quality = applicant_metadata.copy()
    quality["total_expected_months"] = quality["applicant_id"].map(expected_months)
    quality["observed_months"] = quality["applicant_id"].map(observed_months).fillna(0).astype(int)
    quality["missing_month_count"] = (quality["total_expected_months"] - quality["observed_months"]).astype(int)
    quality["missing_required_field_count"] = quality["applicant_id"].map(missing_fields).fillna(0).astype(int)
    quality["latest_evidence_age_months"] = quality["applicant_id"].map(latest_age).fillna(assessment_month).astype(int)
    quality["consistency_violation_count"] = quality["applicant_id"].map(violations).fillna(0).astype(int)
    quality["unverified_record_count"] = quality["applicant_id"].map(unverified).fillna(0).astype(int)
    quality["defective_record_count"] = quality["applicant_id"].map(defective_records).fillna(0).astype(int)

    field_slots = quality["total_expected_months"] * len(REQUIRED_EVIDENCE_FIELDS)
    quality["completeness_score"] = (1 - quality["missing_month_count"] / quality["total_expected_months"] - quality["missing_required_field_count"] / field_slots).clip(0, 1)
    quality["consistency_score"] = (1 - quality["consistency_violation_count"] / quality["total_expected_months"]).clip(0, 1)
    quality["freshness_score"] = (1 - quality["latest_evidence_age_months"] / (assessment_month - 1)).clip(0, 1)
    quality["provenance_score"] = (1 - quality["unverified_record_count"] / quality["observed_months"].clip(lower=1)).clip(0, 1)
    quality["evidence_quality_score"] = quality[["completeness_score", "consistency_score", "freshness_score", "provenance_score"]].mean(axis=1)
    quality.insert(0, "severity", severity)
    quality.insert(0, "scenario", scenario)
    return quality


def _apply_missingness(evidence: pd.DataFrame, affected_ids: set[int], severity: str, rng: np.random.Generator) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Remove non-latest months and blank fields on retained records."""

    setting = SEVERITY_SETTINGS[severity]
    drop_indices: list[int] = []
    events: list[dict[str, object]] = []
    for applicant_id in affected_ids:
        rows = evidence[evidence["applicant_id"] == applicant_id]
        candidates = rows.loc[rows["month"] < rows["month"].max(), "month"].to_numpy()
        removed_months = rng.choice(candidates, min(setting["months"], len(candidates)), replace=False)
        for month in removed_months:
            row_index = rows.loc[rows["month"] == month].index[0]
            drop_indices.append(row_index)
            events.append(_event(applicant_id, int(month), "missing_month", None, "removed", False))
        retained = rows[~rows["month"].isin(removed_months)]
        mask = rng.random((len(retained), len(REQUIRED_EVIDENCE_FIELDS))) < setting["field_rate"]
        for row_position, row_index in enumerate(retained.index):
            for field_position, field_name in enumerate(REQUIRED_EVIDENCE_FIELDS):
                if mask[row_position, field_position]:
                    evidence.loc[row_index, field_name] = pd.NA
                    _mark_record(evidence, row_index, "missing_field")
                    events.append(_event(applicant_id, int(evidence.loc[row_index, "month"]), "missing_field", field_name, "missing", True))
    return evidence.drop(index=drop_indices).reset_index(drop=True), events


def _apply_staleness(evidence: pd.DataFrame, affected_ids: set[int], severity: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Preserve records and simulate delayed reporting metadata only."""

    lag = SEVERITY_SETTINGS[severity]["months"]
    mask = evidence["applicant_id"].isin(affected_ids)
    evidence.loc[mask, "synthetic_reporting_lag_months"] = lag
    evidence.loc[mask, "synthetic_evidence_age_months"] += lag
    evidence.loc[mask, "simulated_record_defect_introduced"] = True
    evidence.loc[mask, "simulated_record_defect_type"] = "stale_reporting_lag"
    events = [
        _event(int(row.applicant_id), int(row.month), "stale_reporting_lag", "synthetic_reporting_lag_months", "modified", True)
        for row in evidence.loc[mask, ["applicant_id", "month"]].itertuples(index=False)
    ]
    return evidence, events


def _apply_inconsistency(evidence: pd.DataFrame, affected_ids: set[int], severity: str, rng: np.random.Generator) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Create cashflow contradictions while retaining valid non-missing numbers."""

    events: list[dict[str, object]] = []
    for applicant_id in affected_ids:
        rows = evidence[evidence["applicant_id"] == applicant_id]
        selected = rng.choice(rows.index, min(SEVERITY_SETTINGS[severity]["months"], len(rows)), replace=False)
        for row_index in selected:
            evidence.loc[row_index, "net_cashflow"] = round(float(evidence.loc[row_index, "net_cashflow"]) + float(evidence.loc[row_index, "monthly_income"]) * rng.uniform(0.15, 0.35), 2)
            _mark_record(evidence, row_index, "cashflow_contradiction")
            events.append(_event(applicant_id, int(evidence.loc[row_index, "month"]), "cashflow_contradiction", "net_cashflow", "modified", True))
    return evidence, events


def _apply_unverified_provenance(evidence: pd.DataFrame, affected_ids: set[int], severity: str, rng: np.random.Generator) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Change verification metadata without changing financial evidence values."""

    events: list[dict[str, object]] = []
    for applicant_id in affected_ids:
        indexes = evidence.index[evidence["applicant_id"] == applicant_id]
        count = int(round(len(indexes) * SEVERITY_SETTINGS[severity]["verification_rate"]))
        for row_index in rng.choice(indexes, count, replace=False):
            evidence.loc[row_index, "simulated_verification_status"] = "unverified"
            _mark_record(evidence, row_index, "unverified_source")
            events.append(_event(applicant_id, int(evidence.loc[row_index, "month"]), "unverified_source", "simulated_verification_status", "modified", True))
    return evidence, events


def _mark_record(evidence: pd.DataFrame, row_index: int, defect_type: str) -> None:
    evidence.loc[row_index, "simulated_record_defect_introduced"] = True
    evidence.loc[row_index, "simulated_record_defect_type"] = defect_type


def _count_consistency_violations(evidence: pd.DataFrame) -> pd.Series:
    numeric_fields = ["monthly_income", "monthly_expenses", "savings_balance", "utility_bill_amount", "rent_amount", "transaction_count", "incoming_transaction_amount", "outgoing_transaction_amount", "end_of_month_balance"]
    invalid_numeric = evidence[numeric_fields].lt(0).any(axis=1)
    expected_cashflow = evidence["monthly_income"] - evidence["monthly_expenses"] - evidence["rent_amount"] - evidence["utility_bill_amount"]
    mismatch = (evidence["net_cashflow"] - expected_cashflow).abs() > 0.02
    rent_too_high = evidence["rent_amount"] > 0.65 * evidence["monthly_income"]
    expenses_too_high = evidence["monthly_expenses"] > 1.45 * evidence["monthly_income"]
    violation = (invalid_numeric | mismatch | rent_too_high | expenses_too_high).fillna(False)
    return violation.groupby(evidence["applicant_id"]).sum()


def _sample_degraded_records(evidence: pd.DataFrame, quality: pd.DataFrame, scenario: str, severity: str, count: int) -> pd.DataFrame:
    selected = quality.loc[quality["simulated_applicant_defect_introduced"], "applicant_id"].head(count)
    return evidence[evidence["applicant_id"].isin(selected)].assign(scenario=scenario, severity=severity)


def _summarize_variant(quality: pd.DataFrame) -> dict[str, object]:
    return {
        "scenario": quality["scenario"].iloc[0], "severity": quality["severity"].iloc[0],
        "applicant_count": int(len(quality)),
        "affected_applicant_count": int(quality["simulated_applicant_defect_introduced"].sum()),
        "mean_completeness_score": float(quality["completeness_score"].mean()),
        "mean_consistency_score": float(quality["consistency_score"].mean()),
        "mean_freshness_score": float(quality["freshness_score"].mean()),
        "mean_provenance_score": float(quality["provenance_score"].mean()),
        "mean_evidence_quality_score": float(quality["evidence_quality_score"].mean()),
    }


def _event(applicant_id: int, month: int, defect_type: str, field_name: str | None, action: str, record_present_after_degradation: bool) -> dict[str, object]:
    return {"applicant_id": applicant_id, "month": month, "simulated_defect_type": defect_type, "field_name": field_name, "action": action, "record_present_after_degradation": record_present_after_degradation}


def _event_columns() -> list[str]:
    return ["applicant_id", "month", "simulated_defect_type", "field_name", "action", "record_present_after_degradation"]


def _validate_config(config: EvidenceQualityConfig) -> None:
    if config.assessment_month <= 1:
        raise ValueError("assessment_month must be greater than 1.")
    if config.sample_applicants_per_variant <= 0:
        raise ValueError("sample_applicants_per_variant must be positive.")
    if not set(config.scenarios).issubset(SCENARIOS) or not set(config.severities).issubset(SEVERITIES):
        raise ValueError("Configuration contains unsupported scenarios or severities.")


def _validate_clean_inputs(clean_history: pd.DataFrame, model_dataset: pd.DataFrame, config: EvidenceQualityConfig) -> None:
    """Validate required schema, IDs, coverage, and assessment-month consistency."""

    assert_no_forbidden_credit_variables(clean_history)
    assert_no_forbidden_credit_variables(model_dataset)
    required = {"applicant_id", "month", *REQUIRED_EVIDENCE_FIELDS}
    if missing := required - set(clean_history.columns):
        raise ValueError(f"Missing clean financial history columns: {sorted(missing)}")
    if "applicant_id" not in model_dataset.columns:
        raise ValueError("Model dataset must include applicant_id.")
    if clean_history["applicant_id"].isna().any() or model_dataset["applicant_id"].isna().any():
        raise ValueError("Applicant IDs must be non-null.")
    if model_dataset["applicant_id"].duplicated().any():
        raise ValueError("Model dataset applicant IDs must be unique.")
    if clean_history.duplicated(["applicant_id", "month"]).any():
        raise ValueError("Financial history applicant-month combinations must be unique.")
    months = pd.to_numeric(clean_history["month"], errors="coerce")
    if months.isna().any() or not np.equal(months, months.astype(int)).all():
        raise ValueError("Financial history month values must be non-null integers.")
    if months.min() != 1 or months.max() != config.assessment_month or not months.between(1, config.assessment_month).all():
        raise ValueError("Assessment month must equal the final available month within a 1..assessment window.")
    if not clean_history.groupby("applicant_id")["month"].max().eq(config.assessment_month).all():
        raise ValueError("Every applicant must have evidence through the assessment month.")
    if set(clean_history["applicant_id"]) != set(model_dataset["applicant_id"]):
        raise ValueError("Financial history and model dataset applicant IDs must match exactly.")


if __name__ == "__main__":
    run_evidence_quality_experiment()
