from pathlib import Path

import pandas as pd
import pytest

from src.data.validation import FORBIDDEN_CONVENTIONAL_CREDIT_TERMS
from src.evaluation.metrics import expected_calibration_error
from src.models.baseline import (
    BASELINE_FEATURES,
    TARGET_COLUMN,
    make_train_validation_test_split,
    predict_probabilities,
    train_baseline_models,
)
from src.models.calibration import (
    CALIBRATION_METHODS,
    evaluate_calibration_variants,
    fit_calibrated_models,
    run_calibration_experiment,
)


def _small_dataset() -> pd.DataFrame:
    dataset = pd.read_csv("data/processed/synthetic/model_dataset.csv")
    return dataset.head(600).copy()


def test_expected_calibration_error_is_valid() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_probability = pd.Series([0.1, 0.2, 0.8, 0.9])

    ece = expected_calibration_error(y_true, y_probability, n_bins=2)

    assert ece == pytest.approx(0.15)
    assert expected_calibration_error(y_true, y_true, n_bins=2) == 0.0


def test_calibration_probabilities_remain_in_range_and_have_expected_rows() -> None:
    dataset = _small_dataset()
    splits = make_train_validation_test_split(dataset, seed=42)
    models = train_baseline_models(splits["train"], seed=42)
    calibrated_models = fit_calibrated_models(models, splits["validation"])

    metrics, comparison, predictions = evaluate_calibration_variants(
        calibrated_models,
        splits["test"],
    )

    expected_rows = len(splits["test"]) * len(models) * len(CALIBRATION_METHODS)
    assert len(predictions) == expected_rows
    assert predictions["predicted_probability"].between(0, 1).all()
    assert set(comparison["calibration_method"]) == set(CALIBRATION_METHODS)
    assert set(metrics) == {"logistic_regression", "hist_gradient_boosting"}


def test_calibration_fitting_does_not_use_test_labels() -> None:
    dataset = _small_dataset()
    splits = make_train_validation_test_split(dataset, seed=42)
    models = train_baseline_models(splits["train"], seed=42)
    calibrated_models = fit_calibrated_models(models, splits["validation"])

    original_test = splits["test"].copy()
    changed_test = original_test.copy()
    changed_test[TARGET_COLUMN] = 1 - changed_test[TARGET_COLUMN]

    for variants in calibrated_models.values():
        for model in variants.values():
            original_probabilities = predict_probabilities(model, original_test)[
                "predicted_probability"
            ]
            changed_probabilities = predict_probabilities(model, changed_test)[
                "predicted_probability"
            ]
            pd.testing.assert_series_equal(
                original_probabilities,
                changed_probabilities,
                check_names=False,
            )


def test_calibration_experiment_is_reproducible_and_preserves_model_dataset(
    tmp_path: Path,
) -> None:
    dataset = _small_dataset()
    dataset_path = tmp_path / "model_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    before_contents = dataset_path.read_bytes()

    first = run_calibration_experiment(
        dataset_path=dataset_path,
        output_dir=tmp_path / "first",
        seed=42,
    )
    second = run_calibration_experiment(
        dataset_path=dataset_path,
        output_dir=tmp_path / "second",
        seed=42,
    )

    assert first == second
    assert dataset_path.read_bytes() == before_contents


def test_calibration_uses_no_forbidden_features() -> None:
    for feature in BASELINE_FEATURES:
        assert not any(
            forbidden_term in feature.lower()
            for forbidden_term in FORBIDDEN_CONVENTIONAL_CREDIT_TERMS
        )
    assert "financial_stress_probability" not in BASELINE_FEATURES
    assert "applicant_id" not in BASELINE_FEATURES
