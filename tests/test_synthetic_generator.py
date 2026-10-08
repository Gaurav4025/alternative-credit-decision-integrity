import pandas as pd

from src.data.synthetic_generator import generate_dataset
from src.data.validation import (
    FORBIDDEN_CONVENTIONAL_CREDIT_TERMS,
    validate_alternative_features,
    validate_financial_history,
)
from src.features.alternative_features import FEATURE_COLUMNS


def test_generate_dataset_is_reproducible_with_same_seed() -> None:
    first_profiles, first_history, first_features, first_metadata = generate_dataset(
        n_applicants=25,
        months=12,
        seed=123,
    )
    second_profiles, second_history, second_features, second_metadata = generate_dataset(
        n_applicants=25,
        months=12,
        seed=123,
    )

    pd.testing.assert_frame_equal(first_profiles, second_profiles)
    pd.testing.assert_frame_equal(first_history, second_history)
    pd.testing.assert_frame_equal(first_features, second_features)
    assert first_metadata == second_metadata


def test_generate_dataset_has_expected_rows_and_columns() -> None:
    profiles, history, features, metadata = generate_dataset(
        n_applicants=30,
        months=12,
        seed=42,
    )

    assert len(profiles) == 30
    assert len(history) == 30 * 12
    assert len(features) == 30
    assert metadata["n_applicants"] == 30
    assert metadata["months"] == 12

    assert set(profiles.columns) == {
        "applicant_id",
        "population_group",
        "synthetic_profile_version",
    }
    assert {
        "applicant_id",
        "month",
        "population_group",
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
    }.issubset(history.columns)
    assert ["applicant_id", *FEATURE_COLUMNS] == list(features.columns)


def test_generated_values_satisfy_financial_constraints() -> None:
    _, history, features, _ = generate_dataset(
        n_applicants=40,
        months=12,
        seed=7,
    )

    validate_financial_history(history)
    validate_alternative_features(features)

    assert (history["monthly_income"] >= 0).all()
    assert (history["monthly_expenses"] >= 0).all()
    assert (history["savings_balance"] >= 0).all()
    assert (history["rent_amount"] >= 0).all()
    assert (history["utility_bill_amount"] >= 0).all()
    assert (history["transaction_count"] >= 0).all()
    assert (history["rent_amount"] / history["monthly_income"]).max() <= 0.65


def test_no_forbidden_conventional_credit_variables_are_generated() -> None:
    profiles, history, features, metadata = generate_dataset(
        n_applicants=20,
        months=12,
        seed=99,
    )

    all_columns = (
        list(profiles.columns)
        + list(history.columns)
        + list(features.columns)
        + list(metadata.keys())
    )
    for column in all_columns:
        assert not any(
            forbidden_term in column.lower()
            for forbidden_term in FORBIDDEN_CONVENTIONAL_CREDIT_TERMS
        )


def test_population_groups_overlap_on_income_features() -> None:
    profiles, _, features, _ = generate_dataset(
        n_applicants=250,
        months=12,
        seed=314,
    )
    features_with_group = profiles[["applicant_id", "population_group"]].merge(
        features,
        on="applicant_id",
        how="inner",
    )

    group_ranges = features_with_group.groupby("population_group")[
        "monthly_income_mean"
    ].agg(["min", "max"])

    assert group_ranges.loc["volatile", "max"] > group_ranges.loc["stable", "min"]
    assert group_ranges.loc["stable", "max"] > group_ranges.loc["borderline", "min"]
