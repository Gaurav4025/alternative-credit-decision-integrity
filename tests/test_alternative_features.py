import pandas as pd
import pytest

from src.features.alternative_features import build_alternative_features


def test_build_alternative_features_calculates_expected_values() -> None:
    history = pd.DataFrame(
        {
            "applicant_id": [1, 1, 1],
            "month": [1, 2, 3],
            "monthly_income": [100.0, 110.0, 120.0],
            "monthly_expenses": [50.0, 55.0, 60.0],
            "savings_balance": [10.0, 20.0, 15.0],
            "utility_bill_amount": [5.0, 5.0, 5.0],
            "utility_payment_status": ["on_time", "late", "on_time"],
            "rent_amount": [20.0, 20.0, 20.0],
            "rent_payment_status": ["on_time", "on_time", "missed"],
            "transaction_count": [10, 12, 14],
            "incoming_transaction_amount": [100.0, 110.0, 120.0],
            "outgoing_transaction_amount": [75.0, 80.0, 85.0],
            "end_of_month_balance": [25.0, 30.0, 35.0],
            "net_cashflow": [25.0, 30.0, 35.0],
        }
    )

    features = build_alternative_features(history)
    row = features.iloc[0]

    assert row["monthly_income_mean"] == 110.0
    assert row["expense_to_income_ratio"] == 0.5
    assert row["savings_consistency"] == 0.5
    assert row["utility_payment_regularity"] == pytest.approx(2 / 3)
    assert row["rent_payment_regularity"] == pytest.approx(2 / 3)
    assert row["transaction_frequency"] == 12.0
    assert row["average_monthly_balance"] == 30.0
    assert row["minimum_monthly_balance"] == 25.0
    assert row["positive_cashflow_ratio"] == 1.0
    assert row["income_growth"] == 0.2
