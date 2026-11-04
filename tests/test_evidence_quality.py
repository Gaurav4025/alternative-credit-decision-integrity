from pathlib import Path

import numpy as np
import pandas as pd

from src.data.evidence_quality import (
    EvidenceQualityConfig,
    REQUIRED_EVIDENCE_FIELDS,
    SCENARIOS,
    SEVERITIES,
    create_degraded_evidence,
    measure_evidence_quality,
    run_evidence_quality_experiment,
)
from src.features.alternative_features import FEATURE_COLUMNS
from src.models.baseline import TARGET_COLUMN


def _write_small_clean_inputs(tmp_path: Path) -> tuple[Path, Path]:
    model_dataset = pd.read_csv("data/processed/synthetic/model_dataset.csv").head(80)
    applicant_ids = set(model_dataset["applicant_id"])
    history = pd.read_csv("data/raw/synthetic/financial_history.csv")
    history = history[history["applicant_id"].isin(applicant_ids)]
    history_path = tmp_path / "financial_history.csv"
    model_path = tmp_path / "model_dataset.csv"
    history.to_csv(history_path, index=False)
    model_dataset.to_csv(model_path, index=False)
    return history_path, model_path


def test_evidence_quality_is_reproducible_with_fixed_seed(tmp_path: Path) -> None:
    history_path, model_path = _write_small_clean_inputs(tmp_path)
    config = EvidenceQualityConfig(seed=123, sample_applicants_per_variant=2)

    run_evidence_quality_experiment(history_path, model_path, tmp_path / "first", config)
    run_evidence_quality_experiment(history_path, model_path, tmp_path / "second", config)

    for filename in [
        "applicant_evidence_quality.csv",
        "degradation_events.csv",
        "degraded_evidence_samples.csv",
        "scenario_summary.csv",
        "metadata.json",
    ]:
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()


def test_severity_reduces_the_relevant_quality_dimension(tmp_path: Path) -> None:
    history_path, model_path = _write_small_clean_inputs(tmp_path)
    run_evidence_quality_experiment(
        history_path,
        model_path,
        tmp_path / "outputs",
        EvidenceQualityConfig(seed=42, sample_applicants_per_variant=2),
    )
    summary = pd.read_csv(tmp_path / "outputs" / "scenario_summary.csv")

    dimensions = {
        "missingness": "mean_completeness_score",
        "stale_records": "mean_freshness_score",
        "inconsistent_records": "mean_consistency_score",
        "unverified_provenance": "mean_provenance_score",
    }
    for scenario, dimension in dimensions.items():
        values = summary[summary["scenario"] == scenario].set_index("severity")[dimension]
        assert values["low"] > values["medium"] > values["high"]


def test_clean_inputs_and_outcome_labels_are_preserved(tmp_path: Path) -> None:
    history_path, model_path = _write_small_clean_inputs(tmp_path)
    history_before = history_path.read_bytes()
    model_before = model_path.read_bytes()
    labels_before = pd.read_csv(model_path)[["applicant_id", TARGET_COLUMN]]

    run_evidence_quality_experiment(
        history_path,
        model_path,
        tmp_path / "outputs",
        EvidenceQualityConfig(seed=42, sample_applicants_per_variant=2),
    )

    assert history_path.read_bytes() == history_before
    assert model_path.read_bytes() == model_before
    pd.testing.assert_frame_equal(
        pd.read_csv(model_path)[["applicant_id", TARGET_COLUMN]], labels_before
    )


def test_metadata_and_artifacts_have_expected_non_leaking_schema(tmp_path: Path) -> None:
    history_path, model_path = _write_small_clean_inputs(tmp_path)
    run_evidence_quality_experiment(
        history_path,
        model_path,
        tmp_path / "outputs",
        EvidenceQualityConfig(seed=42, sample_applicants_per_variant=2),
    )
    quality = pd.read_csv(tmp_path / "outputs" / "applicant_evidence_quality.csv")
    events = pd.read_csv(tmp_path / "outputs" / "degradation_events.csv")
    samples = pd.read_csv(tmp_path / "outputs" / "degraded_evidence_samples.csv")

    assert set(quality["scenario"]) == set(SCENARIOS)
    assert set(quality["severity"]) == set(SEVERITIES)
    assert events.groupby(["scenario", "severity"]).size().gt(0).all()
    assert set(REQUIRED_EVIDENCE_FIELDS).issubset(samples.columns)
    for score in [
        "completeness_score",
        "consistency_score",
        "freshness_score",
        "provenance_score",
        "evidence_quality_score",
    ]:
        assert quality[score].between(0, 1).all()
    for frame in [quality, events, samples]:
        assert TARGET_COLUMN not in frame.columns
        assert "financial_stress_probability" not in frame.columns
        assert "population_group" not in frame.columns
        assert not set(FEATURE_COLUMNS).intersection(frame.columns)


def test_inconsistency_is_detectable_and_does_not_create_invalid_numbers() -> None:
    history = pd.read_csv("data/raw/synthetic/financial_history.csv")
    history = history[history["applicant_id"] <= 30]
    degraded, record_metadata, _ = create_degraded_evidence(
        history,
        scenario="inconsistent_records",
        severity="medium",
        assessment_month=12,
        rng=np.random.default_rng(42),
    )
    quality = measure_evidence_quality(
        degraded,
        history,
        record_metadata,
        scenario="inconsistent_records",
        severity="medium",
        assessment_month=12,
    )

    affected = quality[quality["simulated_defect_introduced"]]
    assert (affected["consistency_violation_count"] > 0).all()
    numeric = degraded[list(REQUIRED_EVIDENCE_FIELDS)].select_dtypes("number")
    assert numeric.isna().sum().sum() == 0
    nonnegative = numeric.drop(columns=["net_cashflow"])
    assert (nonnegative >= 0).all().all()
