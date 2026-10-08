"""Synthetic alternative-financial behavior generator.

The generator creates a controlled experimental environment for thin-file or
credit-invisible applicants. It avoids conventional credit-history variables and
derives model-ready features from monthly alternative-financial behavior.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.distributions import (
    LATENT_COLUMNS,
    normalize_population_proportions,
    sample_latent_traits,
    sample_population_groups,
)
from src.data.validation import (
    validate_alternative_features,
    validate_financial_history,
)
from src.features.alternative_features import build_alternative_features
from src.utils.reproducibility import DEFAULT_RANDOM_SEED

GENERATOR_VERSION = "0.1.0"


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Configuration for synthetic alternative-financial data generation."""

    n_applicants: int = 10_000
    months: int = 12
    seed: int = DEFAULT_RANDOM_SEED
    population_proportions: dict[str, float] = field(
        default_factory=lambda: {"stable": 0.4, "volatile": 0.3, "borderline": 0.3}
    )


def generate_dataset(
    n_applicants: int = 10_000,
    months: int = 12,
    seed: int = DEFAULT_RANDOM_SEED,
    population_proportions: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Generate applicant profiles, monthly history, features, and metadata."""

    if n_applicants <= 0:
        raise ValueError("n_applicants must be positive.")
    if months <= 1:
        raise ValueError("months must be greater than 1.")

    proportions = normalize_population_proportions(population_proportions)
    config = SyntheticDataConfig(
        n_applicants=n_applicants,
        months=months,
        seed=seed,
        population_proportions=proportions,
    )
    rng = np.random.default_rng(seed)

    groups = sample_population_groups(n_applicants, rng, proportions)
    latent_traits = sample_latent_traits(groups, rng)
    applicant_ids = np.arange(1, n_applicants + 1)

    applicant_profiles = pd.DataFrame(
        {
            "applicant_id": applicant_ids,
            "population_group": groups,
            "synthetic_profile_version": GENERATOR_VERSION,
        }
    )

    financial_history = _build_monthly_history(
        applicant_ids=applicant_ids,
        groups=groups,
        latent_traits=latent_traits,
        months=months,
        rng=rng,
    )
    alternative_features = build_alternative_features(financial_history)

    validate_financial_history(financial_history)
    validate_alternative_features(alternative_features)

    metadata: dict[str, object] = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "n_applicants": n_applicants,
        "months": months,
        "population_proportions": proportions,
        "latent_variables": LATENT_COLUMNS,
        "latent_variables_exposed_as_features": False,
        "forbidden_conventional_credit_variables_excluded": True,
    }

    return applicant_profiles, financial_history, alternative_features, metadata


def save_synthetic_dataset(
    output_root: str | Path = ".",
    config: SyntheticDataConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Generate and save the synthetic dataset under ``data/``."""

    config = config or SyntheticDataConfig()
    root = Path(output_root)
    raw_dir = root / "data" / "raw" / "synthetic"
    processed_dir = root / "data" / "processed" / "synthetic"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    applicant_profiles, financial_history, alternative_features, metadata = (
        generate_dataset(**asdict(config))
    )

    applicant_profiles.to_csv(raw_dir / "applicant_profiles.csv", index=False)
    financial_history.to_csv(raw_dir / "financial_history.csv", index=False)
    alternative_features.to_csv(
        processed_dir / "alternative_features.csv", index=False
    )

    with (processed_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")

    return applicant_profiles, financial_history, alternative_features, metadata


def _build_monthly_history(
    applicant_ids: np.ndarray,
    groups: np.ndarray,
    latent_traits: pd.DataFrame,
    months: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate observable monthly financial behavior from latent traits."""

    rows: list[dict[str, object]] = []

    for applicant_id, group, traits in zip(
        applicant_ids,
        groups,
        latent_traits.to_dict(orient="records"),
    ):
        base_income = _sample_base_income(str(group), rng)
        base_rent_ratio = rng.normal(0.28 - 0.05 * traits["financial_liquidity"], 0.04)
        rent_amount = _bounded_value(
            base_income * base_rent_ratio,
            lower=0.12 * base_income,
            upper=0.48 * base_income,
        )
        savings_balance = max(
            0.0,
            rng.normal(
                base_income * (0.3 + 2.2 * traits["financial_liquidity"]),
                base_income * 0.25,
            ),
        )

        income_trend = rng.normal(0.002, 0.006)
        transaction_baseline = 10 + 55 * traits["transaction_activity"]

        for month in range(1, months + 1):
            monthly_income = _generate_monthly_income(
                base_income=base_income,
                month=month,
                income_stability=traits["income_stability"],
                income_trend=income_trend,
                minimum_income=rent_amount / 0.62,
                rng=rng,
            )
            utility_bill_amount = _bounded_value(
                rng.normal(
                    monthly_income * 0.045,
                    monthly_income * (0.012 + 0.018 * (1 - traits["spending_discipline"])),
                ),
                lower=0.015 * monthly_income,
                upper=0.12 * monthly_income,
            )
            monthly_expenses = _generate_monthly_expenses(
                monthly_income=monthly_income,
                spending_discipline=traits["spending_discipline"],
                rng=rng,
            )
            rounded_income = round(monthly_income, 2)
            rounded_expenses = round(monthly_expenses, 2)
            rounded_rent = round(rent_amount, 2)
            rounded_utility = round(utility_bill_amount, 2)
            net_cashflow = (
                rounded_income - rounded_expenses - rounded_rent - rounded_utility
            )

            payment_stress = max(0.0, -net_cashflow / max(monthly_income, 1.0))
            utility_payment_status = _sample_payment_status(
                payment_discipline=traits["payment_discipline"],
                liquidity=traits["financial_liquidity"],
                stress=payment_stress,
                rng=rng,
            )
            rent_payment_status = _sample_payment_status(
                payment_discipline=traits["payment_discipline"],
                liquidity=traits["financial_liquidity"],
                stress=payment_stress + 0.04,
                rng=rng,
            )

            savings_balance = max(0.0, savings_balance + net_cashflow)
            end_of_month_balance = savings_balance + max(0.0, 0.25 * net_cashflow)
            transaction_count = max(
                0,
                int(
                    rng.normal(
                        transaction_baseline,
                        5 + 18 * (1 - traits["income_stability"]),
                    )
                ),
            )

            incoming_transaction_amount = monthly_income * rng.normal(1.0, 0.04)
            outgoing_transaction_amount = (
                monthly_expenses + rent_amount + utility_bill_amount
            ) * rng.normal(1.0, 0.04)

            row = {
                "applicant_id": int(applicant_id),
                "month": month,
                "population_group": str(group),
                "monthly_income": rounded_income,
                "monthly_expenses": rounded_expenses,
                "savings_balance": round(savings_balance, 2),
                "utility_bill_amount": rounded_utility,
                "utility_payment_status": utility_payment_status,
                "rent_amount": rounded_rent,
                "rent_payment_status": rent_payment_status,
                "transaction_count": transaction_count,
                "incoming_transaction_amount": round(
                    max(0.0, incoming_transaction_amount), 2
                ),
                "outgoing_transaction_amount": round(
                    max(0.0, outgoing_transaction_amount), 2
                ),
                "end_of_month_balance": round(end_of_month_balance, 2),
                "net_cashflow": round(net_cashflow, 2),
            }
            _validate_monthly_row(row)
            rows.append(row)

    return pd.DataFrame(rows)


def _sample_base_income(group: str, rng: np.random.Generator) -> float:
    """Sample a monthly income level with overlapping group distributions."""

    group_log_means = {
        "stable": np.log(38_000),
        "volatile": np.log(32_000),
        "borderline": np.log(35_000),
    }
    return float(rng.lognormal(mean=group_log_means[group], sigma=0.34))


def _generate_monthly_income(
    base_income: float,
    month: int,
    income_stability: float,
    income_trend: float,
    minimum_income: float,
    rng: np.random.Generator,
) -> float:
    """Generate one month of income with applicant-specific stability."""

    noise_scale = 0.04 + 0.28 * (1 - income_stability)
    seasonal_factor = 1 + 0.025 * np.sin(2 * np.pi * month / 12)
    trend_factor = 1 + income_trend * (month - 1)

    for _ in range(20):
        income = base_income * seasonal_factor * trend_factor * rng.normal(1, noise_scale)
        if income >= max(1_000, minimum_income):
            return float(income)

    raise ValueError("Unable to generate a valid monthly income.")


def _generate_monthly_expenses(
    monthly_income: float,
    spending_discipline: float,
    rng: np.random.Generator,
) -> float:
    """Generate expenses related to income and spending discipline."""

    expected_ratio = 0.28 + 0.35 * (1 - spending_discipline)
    ratio_sd = 0.04 + 0.11 * (1 - spending_discipline)

    for _ in range(20):
        ratio = rng.normal(expected_ratio, ratio_sd)
        if 0.14 <= ratio <= 0.95:
            return float(monthly_income * ratio)

    raise ValueError("Unable to generate valid monthly expenses.")


def _sample_payment_status(
    payment_discipline: float,
    liquidity: float,
    stress: float,
    rng: np.random.Generator,
) -> str:
    """Sample payment status from discipline, liquidity, and cashflow stress."""

    on_time_probability = 0.45 + 0.38 * payment_discipline + 0.15 * liquidity - 0.35 * stress
    on_time_probability = _bounded_value(on_time_probability, lower=0.08, upper=0.98)
    late_probability = _bounded_value(0.72 * (1 - on_time_probability), 0.015, 0.55)
    missed_probability = max(0.0, 1 - on_time_probability - late_probability)

    return str(
        rng.choice(
            ["on_time", "late", "missed"],
            p=[on_time_probability, late_probability, missed_probability],
        )
    )


def _bounded_value(value: float, lower: float, upper: float) -> float:
    """Bound a generated scalar where economic constraints require it."""

    return float(min(max(value, lower), upper))


def _validate_monthly_row(row: dict[str, object]) -> None:
    """Fail fast if a monthly row violates core generation constraints."""

    non_negative_fields = [
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
    for field_name in non_negative_fields:
        if float(row[field_name]) < 0:
            raise ValueError(f"{field_name} must be non-negative.")

    if float(row["rent_amount"]) > 0.65 * float(row["monthly_income"]):
        raise ValueError("Rent is unrealistically high relative to income.")

    expected_cashflow = (
        float(row["monthly_income"])
        - float(row["monthly_expenses"])
        - float(row["rent_amount"])
        - float(row["utility_bill_amount"])
    )
    if abs(float(row["net_cashflow"]) - round(expected_cashflow, 2)) > 0.02:
        raise ValueError("Net cashflow is inconsistent with monthly behavior.")


if __name__ == "__main__":
    save_synthetic_dataset()
