from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.ood_scenarios import (
    OOD_SCENARIOS,
    OOD_SEVERITIES,
    generate_ood_feature_cohort,
    generate_ood_history,
)
from src.data.validation import validate_financial_history
from src.evaluation.ood_metrics import evaluate_ood_detection, select_ood_threshold
from src.features.alternative_features import FEATURE_COLUMNS
from src.models.baseline import make_train_validation_test_split
from src.models.ood import (
    OOD_FEATURES,
    OODExperimentConfig,
    fit_ood_detectors,
    run_ood_experiment,
    validate_ood_features,
)


def _dataset_and_history(n_applicants: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv("data/processed/synthetic/model_dataset.csv").head(n_applicants).copy()
    history = pd.read_csv("data/raw/synthetic/financial_history.csv")
    return dataset, history[history["applicant_id"].isin(dataset["applicant_id"])].copy()


def test_ood_feature_contract_rejects_targets_and_unexpected_columns() -> None:
    dataset, _ = _dataset_and_history()
    features = dataset.loc[:, FEATURE_COLUMNS]
    assert tuple(OOD_FEATURES) == tuple(FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(validate_ood_features(features), features.astype(float))

    with pytest.raises(ValueError, match="Prohibited"):
        validate_ood_features(features.assign(financial_stress_event=0))
    with pytest.raises(ValueError, match="exactly the permitted"):
        validate_ood_features(features.drop(columns=[FEATURE_COLUMNS[0]]))


def test_ood_detectors_fit_only_training_data_and_score_far_points_higher() -> None:
    dataset, _ = _dataset_and_history()
    splits = make_train_validation_test_split(dataset, seed=42)
    train = splits["train"].loc[:, FEATURE_COLUMNS]
    detectors = fit_ood_detectors(train, seed=42)
    far = train.iloc[:20].copy()
    far += train.std() * 10

    for detector in detectors.values():
        assert detector.n_reference_samples == len(train)
        assert detector.score_samples(far).mean() > detector.score_samples(train.iloc[:20]).mean()


def test_ood_scenarios_are_reproducible_complete_and_clean_verified() -> None:
    dataset, history = _dataset_and_history()
    applicant_ids = dataset["applicant_id"].head(20)

    for scenario in OOD_SCENARIOS:
        for severity in OOD_SEVERITIES:
            first = generate_ood_history(history, applicant_ids, scenario, severity, seed=42)
            second = generate_ood_history(history, applicant_ids, scenario, severity, seed=42)
            pd.testing.assert_frame_equal(first, second)
            validate_financial_history(first)
            assert first.groupby("applicant_id")["month"].nunique().eq(12).all()
            assert first["evidence_quality_status"].eq("clean_verified").all()

            cohort = generate_ood_feature_cohort(history, applicant_ids, scenario, severity, seed=42)
            assert set(cohort["source_applicant_id"]) == set(applicant_ids)
            assert set(FEATURE_COLUMNS).issubset(cohort.columns)
            assert "financial_stress_event" not in cohort.columns
            assert cohort["evidence_quality_status"].eq("clean_verified").all()


def test_ood_metric_calculation_and_validation_only_threshold() -> None:
    validation_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    threshold = select_ood_threshold(validation_scores, percentile=0.80)
    assert threshold == pytest.approx(0.42)

    metrics = evaluate_ood_detection(np.array([0.1, 0.2]), np.array([0.7, 0.8]), threshold)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["clean_false_positive_rate"] == 0.0
    assert metrics["ood_detection_rate"] == 1.0
    assert metrics["artificial_ood_prevalence"] == 0.5


def test_ood_experiment_persists_disjoint_reproducible_splits_and_test_evaluation(tmp_path: Path) -> None:
    dataset, history = _dataset_and_history()
    dataset_path, history_path = tmp_path / "dataset.csv", tmp_path / "history.csv"
    dataset.to_csv(dataset_path, index=False)
    history.to_csv(history_path, index=False)
    dataset_bytes, history_bytes = dataset_path.read_bytes(), history_path.read_bytes()
    config = OODExperimentConfig(seed=42)

    first = run_ood_experiment(dataset_path, history_path, tmp_path / "first", config)
    second = run_ood_experiment(dataset_path, history_path, tmp_path / "second", config)

    assert first == second
    assert dataset_path.read_bytes() == dataset_bytes
    assert history_path.read_bytes() == history_bytes
    for filename in ["split_applicant_ids.csv", "ood_scores.csv", "scenario_summary.csv"]:
        assert (tmp_path / "first" / filename).read_bytes() == (tmp_path / "second" / filename).read_bytes()

    split_ids = pd.read_csv(tmp_path / "first" / "split_applicant_ids.csv")
    assert split_ids["applicant_id"].is_unique
    assert split_ids["split"].value_counts().to_dict() == {"train": 210, "validation": 45, "test": 45}

    scores = pd.read_csv(tmp_path / "first" / "ood_scores.csv")
    summary = pd.read_csv(tmp_path / "first" / "scenario_summary.csv")
    assert not {"financial_stress_event", "financial_stress_probability", "predicted_probability"}.intersection(scores.columns)
    assert set(summary["evaluation_split"]) == {"validation", "test"}
    for detector, threshold in first["thresholds"].items():
        validation_clean = scores[
            (scores["evaluation_split"] == "validation")
            & (scores["detector"] == detector)
            & (scores["scenario"] == "clean_reference")
        ]["ood_score"]
        assert threshold == pytest.approx(select_ood_threshold(validation_clean))
