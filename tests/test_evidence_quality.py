from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.evidence_quality import (
    EvidenceQualityConfig,
    REQUIRED_EVIDENCE_FIELDS,
    _validate_clean_inputs,
    create_degraded_evidence,
    measure_evidence_quality,
    run_evidence_quality_experiment,
)
from src.features.alternative_features import FEATURE_COLUMNS
from src.models.baseline import BASELINE_FEATURES, TARGET_COLUMN


def _history_and_model(n_applicants: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = pd.read_csv("data/processed/synthetic/model_dataset.csv").head(n_applicants)
    history = pd.read_csv("data/raw/synthetic/financial_history.csv")
    return history[history["applicant_id"].isin(model["applicant_id"])].copy(), model.copy()


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    history, model = _history_and_model()
    history_path, model_path = tmp_path / "history.csv", tmp_path / "model.csv"
    history.to_csv(history_path, index=False)
    model.to_csv(model_path, index=False)
    return history_path, model_path


def test_fixed_seed_reproduces_artifacts_and_preserves_clean_files(tmp_path: Path) -> None:
    history_path, model_path = _write_inputs(tmp_path)
    history_bytes, model_bytes = history_path.read_bytes(), model_path.read_bytes()
    config = EvidenceQualityConfig(seed=123, sample_applicants_per_variant=2)

    run_evidence_quality_experiment(history_path, model_path, tmp_path / "first", config)
    run_evidence_quality_experiment(history_path, model_path, tmp_path / "second", config)

    for filename in [
        "applicant_evidence_quality.csv", "degradation_events.csv",
        "degraded_evidence_samples.csv", "scenario_summary.csv", "metadata.json",
    ]:
        assert (tmp_path / "first" / filename).read_bytes() == (tmp_path / "second" / filename).read_bytes()
    assert history_path.read_bytes() == history_bytes
    assert model_path.read_bytes() == model_bytes


def test_record_flags_match_actual_modifications_and_applicant_metadata_is_separate() -> None:
    history, _ = _history_and_model()
    degraded, applicants, events = create_degraded_evidence(
        history, "inconsistent_records", "medium", 12, np.random.default_rng(42)
    )
    flagged = degraded[degraded["simulated_record_defect_introduced"]]
    event_records = events[events["record_present_after_degradation"]]

    assert len(flagged) == len(event_records)
    assert set(flagged["simulated_record_defect_type"]) == {"cashflow_contradiction"}
    assert applicants["simulated_applicant_defect_introduced"].sum() < len(flagged)
    assert set(applicants.columns) == {
        "applicant_id", "simulated_applicant_defect_introduced", "simulated_applicant_defect_type"
    }
    unflagged = degraded[~degraded["simulated_record_defect_introduced"]]
    expected = unflagged["monthly_income"] - unflagged["monthly_expenses"] - unflagged["rent_amount"] - unflagged["utility_bill_amount"]
    assert (unflagged["net_cashflow"] - expected).abs().max() <= 0.02


def test_missingness_primarily_affects_completeness_without_artificial_staleness() -> None:
    history, _ = _history_and_model()
    degraded, applicants, _ = create_degraded_evidence(
        history, "missingness", "high", 12, np.random.default_rng(42)
    )
    quality = measure_evidence_quality(degraded, history, applicants, "missingness", "high", 12)
    affected = quality[quality["simulated_applicant_defect_introduced"]]

    assert (affected["missing_month_count"] > 0).all()
    assert (affected["completeness_score"] < 1).all()
    assert (affected["latest_evidence_age_months"] == 0).all()
    assert (affected["freshness_score"] == 1).all()


def test_staleness_preserves_observations_and_affects_freshness_only() -> None:
    history, _ = _history_and_model()
    degraded, applicants, _ = create_degraded_evidence(
        history, "stale_records", "medium", 12, np.random.default_rng(42)
    )
    quality = measure_evidence_quality(degraded, history, applicants, "stale_records", "medium", 12)
    affected_ids = applicants.loc[applicants["simulated_applicant_defect_introduced"], "applicant_id"]
    affected_records = degraded[degraded["applicant_id"].isin(affected_ids)]
    affected_quality = quality[quality["applicant_id"].isin(affected_ids)]

    assert len(degraded) == len(history)
    assert (affected_quality["observed_months"] == affected_quality["total_expected_months"]).all()
    assert (affected_quality["completeness_score"] == 1).all()
    assert (affected_quality["freshness_score"] < 1).all()
    assert (affected_records["synthetic_reporting_lag_months"] == 3).all()
    assert affected_records["simulated_record_defect_introduced"].all()


def test_provenance_changes_no_financial_values_and_inconsistency_is_detectable() -> None:
    history, _ = _history_and_model()
    provenance, provenance_applicants, _ = create_degraded_evidence(
        history, "unverified_provenance", "medium", 12, np.random.default_rng(42)
    )
    clean = history.drop(columns=["population_group"])
    financial_columns = ["applicant_id", "month", *REQUIRED_EVIDENCE_FIELDS]
    pd.testing.assert_frame_equal(
        provenance[financial_columns].sort_values(["applicant_id", "month"]).reset_index(drop=True),
        clean[financial_columns].sort_values(["applicant_id", "month"]).reset_index(drop=True),
    )
    assert provenance["simulated_record_defect_introduced"].sum() > 0
    assert provenance_applicants["simulated_applicant_defect_introduced"].sum() > 0

    inconsistent, inconsistent_applicants, _ = create_degraded_evidence(
        history, "inconsistent_records", "medium", 12, np.random.default_rng(42)
    )
    quality = measure_evidence_quality(
        inconsistent, history, inconsistent_applicants, "inconsistent_records", "medium", 12
    )
    affected = quality[quality["simulated_applicant_defect_introduced"]]
    assert (affected["consistency_violation_count"] > 0).all()
    assert inconsistent[list(REQUIRED_EVIDENCE_FIELDS)].select_dtypes("number").isna().sum().sum() == 0


def test_model_contract_and_phase_five_outputs_do_not_leak_targets(tmp_path: Path) -> None:
    history_path, model_path = _write_inputs(tmp_path)
    run_evidence_quality_experiment(
        history_path, model_path, tmp_path / "outputs", EvidenceQualityConfig(seed=42, sample_applicants_per_variant=2)
    )
    assert tuple(BASELINE_FEATURES) == tuple(FEATURE_COLUMNS)
    for filename in ["applicant_evidence_quality.csv", "degradation_events.csv", "degraded_evidence_samples.csv"]:
        columns = pd.read_csv(tmp_path / "outputs" / filename, nrows=1).columns
        assert TARGET_COLUMN not in columns
        assert "financial_stress_probability" not in columns
        assert "population_group" not in columns
        assert not set(FEATURE_COLUMNS).intersection(columns)


def test_input_validation_rejects_duplicate_months_bad_coverage_and_bad_assessment_month() -> None:
    history, model = _history_and_model()
    duplicate = pd.concat([history, history.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        _validate_clean_inputs(duplicate, model, EvidenceQualityConfig())

    with pytest.raises(ValueError, match="match exactly"):
        _validate_clean_inputs(history, model.iloc[:-1], EvidenceQualityConfig())

    with pytest.raises(ValueError, match="Assessment month"):
        _validate_clean_inputs(history, model, EvidenceQualityConfig(assessment_month=11))


def test_input_validation_rejects_missing_fields_null_ids_and_incomplete_assessment_coverage() -> None:
    history, model = _history_and_model()

    with pytest.raises(ValueError, match="Missing clean financial history columns"):
        _validate_clean_inputs(history.drop(columns=["net_cashflow"]), model, EvidenceQualityConfig())

    null_id_history = history.copy()
    null_id_history.loc[null_id_history.index[0], "applicant_id"] = pd.NA
    with pytest.raises(ValueError, match="non-null"):
        _validate_clean_inputs(null_id_history, model, EvidenceQualityConfig())

    incomplete_history = history[
        ~((history["applicant_id"] == history["applicant_id"].iloc[0]) & (history["month"] == 12))
    ]
    with pytest.raises(ValueError, match="Every applicant"):
        _validate_clean_inputs(incomplete_history, model, EvidenceQualityConfig())
