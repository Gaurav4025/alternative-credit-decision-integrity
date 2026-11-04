import numpy as np
import pandas as pd
import pytest

from src.evaluation.uncertainty_metrics import (
    ensemble_stability,
    evaluate_uncertainty,
    predictive_entropy,
    select_uncertainty_threshold,
    summarize_member_predictions,
)
from src.features.alternative_features import FEATURE_COLUMNS
from src.models.baseline import TARGET_COLUMN, make_train_validation_test_split
from src.models.uncertainty import (
    MODEL_FAMILIES,
    UNCERTAINTY_FEATURES,
    fit_bootstrap_ensemble,
    load_persisted_splits,
    predict_bootstrap_ensemble,
    stratified_bootstrap_sample,
    validate_uncertainty_features,
)


def _small_splits() -> dict[str, pd.DataFrame]:
    dataset = pd.read_csv("data/processed/synthetic/model_dataset.csv").head(300).copy()
    return make_train_validation_test_split(dataset, seed=42)


def test_uncertainty_feature_contract_rejects_prohibited_inputs() -> None:
    features = _small_splits()["train"].loc[:, FEATURE_COLUMNS]
    assert tuple(UNCERTAINTY_FEATURES) == tuple(FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(validate_uncertainty_features(features), features.astype(float))
    with pytest.raises(ValueError, match="Prohibited"):
        validate_uncertainty_features(features.assign(ood_score=0.2))
    with pytest.raises(ValueError, match="exactly the permitted"):
        validate_uncertainty_features(features.drop(columns=[FEATURE_COLUMNS[0]]))


def test_persisted_phase_six_splits_are_unique_disjoint_and_reused() -> None:
    dataset = pd.read_csv("data/processed/synthetic/model_dataset.csv")
    splits = load_persisted_splits(dataset, "experiments/ood/split_applicant_ids.csv")
    ids = {name: set(frame["applicant_id"]) for name, frame in splits.items()}
    assert {name: len(frame) for name, frame in splits.items()} == {
        "train": 7000, "validation": 1500, "test": 1500
    }
    assert not ids["train"] & ids["validation"]
    assert not ids["train"] & ids["test"]
    assert not ids["validation"] & ids["test"]
    assert set.union(*ids.values()) == set(dataset["applicant_id"])


def test_stratified_bootstrap_sampling_is_deterministic_and_preserves_class_counts() -> None:
    train = _small_splits()["train"]
    first, counts = stratified_bootstrap_sample(train, seed=99)
    second, _ = stratified_bootstrap_sample(train, seed=99)
    pd.testing.assert_frame_equal(first, second)
    assert first[TARGET_COLUMN].value_counts().to_dict() == train[TARGET_COLUMN].value_counts().to_dict()
    assert counts == {int(label): int(count) for label, count in train[TARGET_COLUMN].value_counts().items()}


def test_bootstrap_members_are_training_only_deterministic_and_fit_logistic_scalers() -> None:
    splits = _small_splits()
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    seeds = [101, 102, 103]
    first = fit_bootstrap_ensemble(train, "logistic_regression", seeds)
    second = fit_bootstrap_ensemble(train, "logistic_regression", seeds)
    first_predictions = predict_bootstrap_ensemble(first, validation.loc[:, FEATURE_COLUMNS])
    second_predictions = predict_bootstrap_ensemble(second, validation.loc[:, FEATURE_COLUMNS])
    np.testing.assert_allclose(first_predictions, second_predictions)
    assert set(first.training_applicant_ids) == set(train["applicant_id"])
    assert not first.training_applicant_ids.intersection(test["applicant_id"])
    assert all(list(member.named_steps) == ["scaler", "model"] for member in first.members)

    validation_variance = summarize_member_predictions(first_predictions)["predictive_variance"]
    threshold_before_test_label_change = select_uncertainty_threshold(validation_variance)
    changed_test = test.copy()
    changed_test[TARGET_COLUMN] = 1 - changed_test[TARGET_COLUMN]
    np.testing.assert_allclose(
        predict_bootstrap_ensemble(first, test.loc[:, FEATURE_COLUMNS]),
        predict_bootstrap_ensemble(first, changed_test.loc[:, FEATURE_COLUMNS]),
    )
    assert select_uncertainty_threshold(validation_variance) == threshold_before_test_label_change


def test_bootstrap_statistics_thresholds_stability_and_metrics_are_valid() -> None:
    splits = _small_splits()
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    for family in MODEL_FAMILIES:
        ensemble = fit_bootstrap_ensemble(train, family, [201, 202, 203, 204, 205, 206])
        validation_members = predict_bootstrap_ensemble(ensemble, validation.loc[:, FEATURE_COLUMNS])
        test_members = predict_bootstrap_ensemble(ensemble, test.loc[:, FEATURE_COLUMNS])
        summary = summarize_member_predictions(validation_members)
        assert np.isfinite(summary["mean_probability"]).all()
        assert ((summary["mean_probability"] >= 0) & (summary["mean_probability"] <= 1)).all()
        assert (summary["predictive_variance"] >= 0).all()
        assert ((summary["predictive_entropy"] >= 0) & (summary["predictive_entropy"] <= np.log(2))).all()
        threshold = select_uncertainty_threshold(summary["predictive_variance"])
        assert threshold == pytest.approx(np.quantile(summary["predictive_variance"], 0.95))
        stability = ensemble_stability(validation_members[:, :3], validation_members)
        assert set(stability) == {
            "rank_correlation", "mean_absolute_difference", "median_absolute_difference",
            "median_relative_difference", "stable",
        }

        test_summary = summarize_member_predictions(test_members)
        metrics, curve = evaluate_uncertainty(
            test[TARGET_COLUMN].to_numpy(),
            test_summary["mean_probability"],
            test_summary["predictive_variance"],
            threshold,
        )
        assert 0 <= metrics["error_detection_auroc"] <= 1
        assert 0 <= metrics["error_detection_auprc"] <= 1
        assert curve["coverage"].is_monotonic_decreasing


def test_entropy_and_selective_prediction_metrics_have_known_values() -> None:
    entropy = predictive_entropy(np.array([0.0, 0.5, 1.0]))
    assert entropy[1] == pytest.approx(np.log(2))
    assert entropy[0] >= 0
    metrics, curve = evaluate_uncertainty(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.9, 0.8, 0.2]),
        np.array([0.1, 0.9, 0.2, 0.8]),
        threshold=0.85,
    )
    assert metrics["uncertainty_flag_rate"] == 0.25
    assert metrics["retained_classification_error_rate"] == pytest.approx(1 / 3)
    assert curve.iloc[0]["coverage"] == 1.0
