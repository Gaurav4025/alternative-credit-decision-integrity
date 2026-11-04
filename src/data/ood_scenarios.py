"""Controlled, label-blind distribution shifts for OOD experiments.

The scenarios transform complete synthetic monthly histories and recompute the
existing feature contract. They are not evidence-quality defects and never use
outcomes, stress probabilities, population labels, latent traits, or model
predictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.data.validation import validate_alternative_features, validate_financial_history
from src.features.alternative_features import FEATURE_COLUMNS, build_alternative_features
from src.utils.reproducibility import DEFAULT_RANDOM_SEED

OOD_SCENARIOS = ("monetary_scale", "temporal_volatility", "transaction_pattern")
OOD_SEVERITIES = ("low", "medium", "high")
SCENARIO_SETTINGS = {
    "monetary_scale": {"low": 1.8, "medium": 2.8, "high": 4.0},
    "temporal_volatility": {"low": 0.20, "medium": 0.45, "high": 0.70},
    "transaction_pattern": {"low": 2.0, "medium": 4.0, "high": 7.0},
}
PROHIBITED_OOD_COLUMNS = {
    "financial_stress_event",
    "financial_stress_probability",
    "population_group",
}


@dataclass(frozen=True)
class OODScenarioConfig:
    """Fixed settings for controlled OOD cohort generation."""

    seed: int = DEFAULT_RANDOM_SEED
    assessment_month: int = 12

    def as_dict(self) -> dict[str, object]:
        """Return serializable scenario settings for experiment metadata."""

        return {"config": asdict(self), "settings": SCENARIO_SETTINGS}


def generate_ood_history(
    clean_history: pd.DataFrame,
    applicant_ids: pd.Series | np.ndarray | list[int],
    scenario: str,
    severity: str,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Create a complete, valid transformed monthly history for selected IDs."""

    _validate_scenario(scenario, severity)
    _validate_clean_history(clean_history)
    selected_ids = np.sort(np.asarray(applicant_ids, dtype=int))
    source = clean_history[clean_history["applicant_id"].isin(selected_ids)].copy()
    if set(source["applicant_id"]) != set(selected_ids):
        raise ValueError("Every selected applicant must exist in the clean financial history.")

    rng = np.random.default_rng(seed)
    transformed: list[pd.DataFrame] = []
    setting = SCENARIO_SETTINGS[scenario][severity]
    for applicant_id, history in source.groupby("applicant_id", sort=True):
        ordered = history.sort_values("month").reset_index(drop=True)
        income_multipliers, transaction_multipliers = _scenario_multipliers(
            ordered["month"].to_numpy(), scenario, float(setting), rng
        )
        transformed.append(
            _rebuild_history(ordered, income_multipliers, transaction_multipliers)
        )

    result = pd.concat(transformed, ignore_index=True)
    validate_financial_history(result)
    _validate_complete_months(result)
    result["ood_scenario"] = scenario
    result["ood_severity"] = severity
    result["evidence_quality_status"] = "clean_verified"
    return result


def generate_ood_feature_cohort(
    clean_history: pd.DataFrame,
    applicant_ids: pd.Series | np.ndarray | list[int],
    scenario: str,
    severity: str,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Recompute the permitted feature contract for one controlled OOD cohort."""

    history = generate_ood_history(clean_history, applicant_ids, scenario, severity, seed)
    features = build_alternative_features(history)
    validate_alternative_features(features)
    result = features[["applicant_id", *FEATURE_COLUMNS]].copy()
    result = result.rename(columns={"applicant_id": "source_applicant_id"})
    result["ood_scenario"] = scenario
    result["ood_severity"] = severity
    result["evidence_quality_status"] = "clean_verified"
    return result


def _scenario_multipliers(
    months: np.ndarray,
    scenario: str,
    setting: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if scenario == "monetary_scale":
        return np.full(len(months), setting), np.ones(len(months))
    if scenario == "temporal_volatility":
        phase = int(rng.integers(0, 12))
        multipliers = 1 + setting * np.sin(2 * np.pi * (months + phase) / 12)
        return multipliers, np.ones(len(months))
    return np.ones(len(months)), np.full(len(months), setting)


def _rebuild_history(
    history: pd.DataFrame,
    income_multipliers: np.ndarray,
    transaction_multipliers: np.ndarray,
) -> pd.DataFrame:
    """Rebuild values so transformed evidence remains arithmetically valid."""

    rows: list[dict[str, object]] = []
    opening_savings = max(
        0.0,
        float(history.loc[0, "savings_balance"] - max(0.0, history.loc[0, "net_cashflow"])),
    )
    running_savings = opening_savings * float(income_multipliers[0])

    for row, income_multiplier, transaction_multiplier in zip(
        history.to_dict(orient="records"), income_multipliers, transaction_multipliers
    ):
        income = _round_money(float(row["monthly_income"]) * income_multiplier)
        expense_ratio = float(row["monthly_expenses"]) / float(row["monthly_income"])
        rent_ratio = float(row["rent_amount"]) / float(row["monthly_income"])
        utility_ratio = float(row["utility_bill_amount"]) / float(row["monthly_income"])
        expenses = _round_money(income * expense_ratio)
        rent = _round_money(income * rent_ratio)
        utility = _round_money(income * utility_ratio)
        cashflow = round(income - expenses - rent - utility, 2)
        running_savings = max(0.0, running_savings + cashflow)
        balance = _round_money(running_savings + max(0.0, 0.25 * cashflow))
        total_outflow = expenses + rent + utility

        rows.append(
            {
                "applicant_id": int(row["applicant_id"]),
                "month": int(row["month"]),
                "monthly_income": income,
                "monthly_expenses": expenses,
                "savings_balance": _round_money(running_savings),
                "utility_bill_amount": utility,
                "utility_payment_status": row["utility_payment_status"],
                "rent_amount": rent,
                "rent_payment_status": row["rent_payment_status"],
                "transaction_count": int(round(float(row["transaction_count"]) * transaction_multiplier)),
                "incoming_transaction_amount": _round_money(
                    float(row["incoming_transaction_amount"]) * income_multiplier * transaction_multiplier
                ),
                "outgoing_transaction_amount": _round_money(
                    float(row["outgoing_transaction_amount"]) * income_multiplier * transaction_multiplier
                ),
                "end_of_month_balance": balance,
                "net_cashflow": cashflow,
            }
        )
    return pd.DataFrame(rows)


def _round_money(value: float) -> float:
    return round(max(0.0, value), 2)


def _validate_scenario(scenario: str, severity: str) -> None:
    if scenario not in OOD_SCENARIOS or severity not in OOD_SEVERITIES:
        raise ValueError("Scenario and severity must be supported OOD values.")


def _validate_clean_history(history: pd.DataFrame) -> None:
    forbidden = PROHIBITED_OOD_COLUMNS.intersection(history.columns)
    if forbidden - {"population_group"}:
        raise ValueError(f"Prohibited OOD columns in financial history: {sorted(forbidden)}")
    required = {
        "applicant_id", "month", "monthly_income", "monthly_expenses", "savings_balance",
        "utility_bill_amount", "utility_payment_status", "rent_amount", "rent_payment_status",
        "transaction_count", "incoming_transaction_amount", "outgoing_transaction_amount",
        "end_of_month_balance", "net_cashflow",
    }
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Missing clean financial history columns: {sorted(missing)}")


def _validate_complete_months(history: pd.DataFrame) -> None:
    expected = list(range(1, 13))
    observed = history.groupby("applicant_id")["month"].agg(lambda values: sorted(values.tolist()))
    if not observed.map(lambda values: values == expected).all():
        raise ValueError("OOD scenarios must preserve a complete 12-month history for every applicant.")
