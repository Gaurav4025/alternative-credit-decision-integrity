"""Alternative-financial feature engineering from monthly histories."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "monthly_income_mean",
    "income_variability",
    "expense_to_income_ratio",
    "savings_consistency",
    "utility_payment_regularity",
    "rent_payment_regularity",
    "cashflow_volatility",
    "transaction_frequency",
    "average_monthly_balance",
    "minimum_monthly_balance",
    "positive_cashflow_ratio",
    "income_growth",
    "financial_buffer_ratio",
]


def build_alternative_features(financial_history: pd.DataFrame) -> pd.DataFrame:
    """Build applicant-level features from monthly alternative-financial history."""

    required_columns = {
        "applicant_id",
        "month",
        "monthly_income",
        "monthly_expenses",
        "savings_balance",
        "utility_payment_status",
        "rent_payment_status",
        "transaction_count",
        "end_of_month_balance",
        "net_cashflow",
    }
    missing_columns = required_columns - set(financial_history.columns)
    if missing_columns:
        raise ValueError(f"Missing financial history columns: {sorted(missing_columns)}")

    ordered_history = financial_history.sort_values(["applicant_id", "month"]).copy()
    ordered_history["utility_paid_on_time"] = (
        ordered_history["utility_payment_status"] == "on_time"
    ).astype(float)
    ordered_history["rent_paid_on_time"] = (
        ordered_history["rent_payment_status"] == "on_time"
    ).astype(float)
    ordered_history["positive_cashflow"] = (ordered_history["net_cashflow"] > 0).astype(
        float
    )

    grouped = ordered_history.groupby("applicant_id", sort=True)
    features = grouped.agg(
        monthly_income_mean=("monthly_income", "mean"),
        monthly_income_std=("monthly_income", "std"),
        monthly_expenses_mean=("monthly_expenses", "mean"),
        savings_consistency=("savings_balance", _non_decreasing_share),
        utility_payment_regularity=("utility_paid_on_time", "mean"),
        rent_payment_regularity=("rent_paid_on_time", "mean"),
        cashflow_volatility=("net_cashflow", "std"),
        transaction_frequency=("transaction_count", "mean"),
        average_monthly_balance=("end_of_month_balance", "mean"),
        minimum_monthly_balance=("end_of_month_balance", "min"),
        positive_cashflow_ratio=("positive_cashflow", "mean"),
        first_income=("monthly_income", "first"),
        last_income=("monthly_income", "last"),
    )

    features["income_variability"] = (
        features["monthly_income_std"] / features["monthly_income_mean"]
    )
    features["expense_to_income_ratio"] = (
        features["monthly_expenses_mean"] / features["monthly_income_mean"]
    )
    features["income_growth"] = (
        features["last_income"] - features["first_income"]
    ) / features["first_income"]
    features["financial_buffer_ratio"] = (
        features["average_monthly_balance"] / features["monthly_income_mean"]
    )

    features = features.replace([np.inf, -np.inf], np.nan)
    if features[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature calculation produced missing or infinite values.")

    return features[FEATURE_COLUMNS].reset_index()


def _non_decreasing_share(values: pd.Series) -> float:
    """Return share of month-to-month changes that are non-negative."""

    deltas = values.diff().dropna()
    if deltas.empty:
        return 1.0
    return float((deltas >= 0).mean())
