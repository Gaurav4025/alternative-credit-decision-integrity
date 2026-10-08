"""Validation checks for synthetic alternative-financial data.

These checks validate the controlled synthetic environment. They are not the
final evidence validation system for production decisioning.
"""

from __future__ import annotations

import pandas as pd


FORBIDDEN_CONVENTIONAL_CREDIT_TERMS = (
    "credit_score",
    "cibil",
    "previous_loan",
    "previous_emi",
    "credit_card_repayment",
    "number_of_previous_loans",
    "bureau",
    "delinquency",
)


def assert_no_forbidden_credit_variables(dataframe: pd.DataFrame) -> None:
    """Raise if a dataframe contains conventional credit-history variables."""

    forbidden_columns = [
        column
        for column in dataframe.columns
        if any(term in column.lower() for term in FORBIDDEN_CONVENTIONAL_CREDIT_TERMS)
    ]
    if forbidden_columns:
        raise ValueError(
            "Forbidden conventional-credit variables present: "
            f"{sorted(forbidden_columns)}"
        )


def validate_financial_history(financial_history: pd.DataFrame) -> None:
    """Validate core realism constraints for generated monthly histories."""

    required_columns = {
        "applicant_id",
        "month",
        "monthly_income",
        "monthly_expenses",
        "savings_balance",
        "utility_bill_amount",
        "rent_amount",
        "transaction_count",
        "incoming_transaction_amount",
        "outgoing_transaction_amount",
        "end_of_month_balance",
        "net_cashflow",
    }
    missing_columns = required_columns - set(financial_history.columns)
    if missing_columns:
        raise ValueError(f"Missing financial history columns: {sorted(missing_columns)}")

    assert_no_forbidden_credit_variables(financial_history)

    non_negative_columns = [
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
    negative_columns = [
        column for column in non_negative_columns if (financial_history[column] < 0).any()
    ]
    if negative_columns:
        raise ValueError(f"Negative values found in: {negative_columns}")

    rent_ratio = financial_history["rent_amount"] / financial_history[
        "monthly_income"
    ].replace(0, pd.NA)
    if (rent_ratio.dropna() > 0.65).any():
        raise ValueError("Rent exceeds 65% of monthly income for at least one record.")

    expense_ratio = financial_history["monthly_expenses"] / financial_history[
        "monthly_income"
    ].replace(0, pd.NA)
    if (expense_ratio.dropna() > 1.45).any():
        raise ValueError("Monthly expenses exceed the allowed income relationship.")

    expected_cashflow = (
        financial_history["monthly_income"]
        - financial_history["monthly_expenses"]
        - financial_history["rent_amount"]
        - financial_history["utility_bill_amount"]
    )
    max_error = (financial_history["net_cashflow"] - expected_cashflow).abs().max()
    if max_error > 0.02:
        raise ValueError("Net cashflow is inconsistent with income and outflows.")


def validate_alternative_features(features: pd.DataFrame) -> None:
    """Validate mathematical ranges for derived applicant-level features."""

    assert_no_forbidden_credit_variables(features)

    ratio_columns = [
        "savings_consistency",
        "utility_payment_regularity",
        "rent_payment_regularity",
        "positive_cashflow_ratio",
    ]
    out_of_range = [
        column
        for column in ratio_columns
        if ((features[column] < 0) | (features[column] > 1)).any()
    ]
    if out_of_range:
        raise ValueError(f"Feature ratios outside [0, 1]: {out_of_range}")

    non_negative_columns = [
        "monthly_income_mean",
        "income_variability",
        "expense_to_income_ratio",
        "cashflow_volatility",
        "transaction_frequency",
        "average_monthly_balance",
        "minimum_monthly_balance",
        "financial_buffer_ratio",
    ]
    negative_columns = [
        column for column in non_negative_columns if (features[column] < 0).any()
    ]
    if negative_columns:
        raise ValueError(f"Negative feature values found in: {negative_columns}")
