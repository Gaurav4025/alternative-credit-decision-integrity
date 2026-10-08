import pandas as pd

from src.data.synthetic_generator import generate_dataset
from src.data.validation import FORBIDDEN_CONVENTIONAL_CREDIT_TERMS
from src.features.alternative_features import FEATURE_COLUMNS
from src.features.financial_stress_outcome import (
    build_financial_stress_outcome,
    validate_model_dataset,
)


def test_financial_stress_outcome_is_reproducible() -> None:
    _, _, features, _ = generate_dataset(n_applicants=100, months=12, seed=11)

    first = build_financial_stress_outcome(features, seed=123)
    second = build_financial_stress_outcome(features, seed=123)

    pd.testing.assert_frame_equal(first, second)


def test_model_dataset_has_valid_schema_and_ranges() -> None:
    _, _, features, _ = generate_dataset(n_applicants=100, months=12, seed=12)
    model_dataset = build_financial_stress_outcome(features, seed=42)

    validate_model_dataset(model_dataset)

    assert list(model_dataset.columns) == [
        "applicant_id",
        *FEATURE_COLUMNS,
        "financial_stress_probability",
        "financial_stress_event",
    ]
    assert model_dataset["financial_stress_probability"].between(0, 1).all()
    assert set(model_dataset["financial_stress_event"].unique()).issubset({0, 1})


def test_model_dataset_contains_no_forbidden_conventional_credit_features() -> None:
    _, _, features, _ = generate_dataset(n_applicants=50, months=12, seed=13)
    model_dataset = build_financial_stress_outcome(features, seed=42)

    for column in model_dataset.columns:
        assert not any(
            forbidden_term in column.lower()
            for forbidden_term in FORBIDDEN_CONVENTIONAL_CREDIT_TERMS
        )


def test_target_is_not_identical_to_population_label() -> None:
    profiles, _, features, _ = generate_dataset(n_applicants=400, months=12, seed=14)
    model_dataset = build_financial_stress_outcome(features, seed=42)
    merged = profiles[["applicant_id", "population_group"]].merge(
        model_dataset[["applicant_id", "financial_stress_event"]],
        on="applicant_id",
        how="inner",
    )

    event_rates = merged.groupby("population_group")["financial_stress_event"].mean()

    assert len(event_rates) == 3
    assert (event_rates > 0).all()
    assert (event_rates < 1).all()
    assert event_rates.nunique() > 1


def test_outcome_is_not_perfectly_determined_by_one_feature() -> None:
    _, _, features, _ = generate_dataset(n_applicants=600, months=12, seed=15)
    model_dataset = build_financial_stress_outcome(features, seed=42)

    correlations = (
        model_dataset[FEATURE_COLUMNS]
        .corrwith(model_dataset["financial_stress_event"])
        .abs()
        .dropna()
    )

    assert correlations.max() < 0.8


def test_expected_approximate_event_rate_range() -> None:
    _, _, features, _ = generate_dataset(n_applicants=1_000, months=12, seed=16)
    model_dataset = build_financial_stress_outcome(features, seed=42)

    event_rate = model_dataset["financial_stress_event"].mean()

    assert 0.15 <= event_rate <= 0.45
