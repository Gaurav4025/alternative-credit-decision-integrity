from pathlib import Path

import pandas as pd

from src.models.baseline import (
    BASELINE_FEATURES,
    PROBABILITY_COLUMN,
    TARGET_COLUMN,
    make_train_validation_test_split,
    predict_probabilities,
    run_baseline_experiment,
    train_baseline_models,
)


def _small_dataset() -> pd.DataFrame:
    dataset = pd.read_csv("data/processed/synthetic/model_dataset.csv")
    return dataset.head(600).copy()


def test_baseline_feature_selection_excludes_leakage_columns() -> None:
    forbidden_features = {
        "applicant_id",
        "population_group",
        PROBABILITY_COLUMN,
        TARGET_COLUMN,
    }

    assert len(BASELINE_FEATURES) == 13
    assert not forbidden_features.intersection(BASELINE_FEATURES)


def test_train_validation_test_split_is_reproducible_and_stratified() -> None:
    dataset = _small_dataset()

    first = make_train_validation_test_split(dataset, seed=42)
    second = make_train_validation_test_split(dataset, seed=42)

    for split_name in ["train", "validation", "test"]:
        pd.testing.assert_frame_equal(first[split_name], second[split_name])

    assert len(first["train"]) == 420
    assert len(first["validation"]) == 90
    assert len(first["test"]) == 90

    full_rate = dataset[TARGET_COLUMN].mean()
    for split in first.values():
        assert abs(split[TARGET_COLUMN].mean() - full_rate) < 0.05


def test_model_training_and_probability_outputs_succeed() -> None:
    dataset = _small_dataset()
    splits = make_train_validation_test_split(dataset, seed=42)
    models = train_baseline_models(splits["train"], seed=42)

    assert set(models) == {"logistic_regression", "hist_gradient_boosting"}

    for model in models.values():
        predictions = predict_probabilities(model, splits["test"])
        assert len(predictions) == len(splits["test"])
        assert predictions["predicted_probability"].between(0, 1).all()


def test_baseline_experiment_generates_metrics_and_predictions(tmp_path: Path) -> None:
    dataset_path = tmp_path / "model_dataset.csv"
    _small_dataset().to_csv(dataset_path, index=False)

    metrics = run_baseline_experiment(
        dataset_path=dataset_path,
        output_dir=tmp_path / "baseline",
        seed=42,
    )

    assert metrics["split_sizes"] == {"train": 420, "validation": 90, "test": 90}
    assert set(metrics["models"]) == {"logistic_regression", "hist_gradient_boosting"}

    for model_metrics in metrics["models"].values():
        test_metrics = model_metrics["test_metrics"]
        for key in [
            "roc_auc",
            "pr_auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "brier_score",
            "log_loss",
            "positive_prediction_rate",
            "actual_event_rate",
        ]:
            assert key in test_metrics
        assert len(model_metrics["validation_threshold_sensitivity"]) == 5

    assert (tmp_path / "baseline" / "metrics.json").exists()
    assert (tmp_path / "baseline" / "model_comparison.csv").exists()
    assert (tmp_path / "baseline" / "predictions_logistic.csv").exists()
    assert (tmp_path / "baseline" / "predictions_tree.csv").exists()
