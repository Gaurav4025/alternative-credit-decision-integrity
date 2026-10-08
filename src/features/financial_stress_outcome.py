"""Synthetic financial stress outcome generation.

The outcome is a stochastic future financial stress event for controlled
experiments. It is not a loan default label and must not be interpreted as an
observed real-world credit outcome.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.validation import assert_no_forbidden_credit_variables
from src.features.alternative_features import FEATURE_COLUMNS
from src.utils.reproducibility import DEFAULT_RANDOM_SEED

OUTCOME_COLUMNS = ["financial_stress_probability", "financial_stress_event"]


def build_financial_stress_outcome(
    alternative_features: pd.DataFrame,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Add stochastic synthetic financial stress outcomes to feature rows.

    The formulation is:

    ``stress_score = weighted_risk_components + idiosyncratic_noise``

    ``P(stress) = sigmoid(stress_score)``

    ``financial_stress_event ~ Bernoulli(P(stress))``

    The risk components use allowed alternative-financial features only. No
    population labels, latent variables, or conventional credit-history fields
    are used.
    """

    _validate_input_features(alternative_features)
    rng = np.random.default_rng(seed)

    features = alternative_features.copy()
    components = _build_risk_components(features)

    deterministic_score = (
        -1.85
        + 1.10 * components["expense_pressure"]
        + 0.90 * components["income_instability"]
        + 0.85 * components["savings_weakness"]
        + 1.00 * components["payment_irregularity"]
        + 0.80 * components["cashflow_volatility_pressure"]
        + 1.10 * components["buffer_weakness"]
        + 0.80 * components["cashflow_weakness"]
    )
    idiosyncratic_noise = rng.normal(loc=0.0, scale=0.65, size=len(features))
    stress_score = deterministic_score + idiosyncratic_noise
    stress_probability = _sigmoid(stress_score)
    stress_event = rng.binomial(n=1, p=stress_probability)

    model_dataset = features[["applicant_id", *FEATURE_COLUMNS]].copy()
    model_dataset["financial_stress_probability"] = stress_probability
    model_dataset["financial_stress_event"] = stress_event.astype(int)

    validate_model_dataset(model_dataset)
    return model_dataset


def save_model_dataset(
    output_root: str | Path = ".",
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Read alternative features and save the synthetic model dataset."""

    root = Path(output_root)
    feature_path = root / "data" / "processed" / "synthetic" / "alternative_features.csv"
    output_path = root / "data" / "processed" / "synthetic" / "model_dataset.csv"

    alternative_features = pd.read_csv(feature_path)
    model_dataset = build_financial_stress_outcome(alternative_features, seed=seed)
    model_dataset.to_csv(output_path, index=False)
    return model_dataset


def validate_model_dataset(model_dataset: pd.DataFrame) -> None:
    """Validate ranges and schema for the synthetic outcome dataset."""

    expected_columns = {
        "applicant_id",
        *FEATURE_COLUMNS,
        "financial_stress_probability",
        "financial_stress_event",
    }
    missing_columns = expected_columns - set(model_dataset.columns)
    if missing_columns:
        raise ValueError(f"Missing model dataset columns: {sorted(missing_columns)}")

    assert_no_forbidden_credit_variables(model_dataset)

    probabilities = model_dataset["financial_stress_probability"]
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Financial stress probabilities must be in [0, 1].")

    allowed_events = {0, 1}
    observed_events = set(model_dataset["financial_stress_event"].unique())
    if not observed_events.issubset(allowed_events):
        raise ValueError("Financial stress events must be binary 0/1 values.")


def _validate_input_features(alternative_features: pd.DataFrame) -> None:
    """Validate that all required allowed feature columns are present."""

    required_columns = {"applicant_id", *FEATURE_COLUMNS}
    missing_columns = required_columns - set(alternative_features.columns)
    if missing_columns:
        raise ValueError(f"Missing alternative feature columns: {sorted(missing_columns)}")
    assert_no_forbidden_credit_variables(alternative_features)


def _build_risk_components(features: pd.DataFrame) -> pd.DataFrame:
    """Transform financial features into bounded stress-risk components."""

    components = pd.DataFrame(index=features.index)
    components["expense_pressure"] = _bounded_ratio(
        features["expense_to_income_ratio"] - 0.35,
        denominator=0.55,
    )
    components["income_instability"] = _bounded_ratio(
        features["income_variability"],
        denominator=0.45,
    )
    components["savings_weakness"] = 1 - features["savings_consistency"].clip(0, 1)
    components["payment_irregularity"] = 1 - (
        (
            features["utility_payment_regularity"].clip(0, 1)
            + features["rent_payment_regularity"].clip(0, 1)
        )
        / 2
    )
    components["cashflow_volatility_pressure"] = _bounded_ratio(
        features["cashflow_volatility"] / features["monthly_income_mean"],
        denominator=0.55,
    )
    components["buffer_weakness"] = 1 - (
        features["financial_buffer_ratio"] / 3.0
    ).clip(0, 1)
    components["cashflow_weakness"] = 1 - features["positive_cashflow_ratio"].clip(0, 1)
    return components


def _bounded_ratio(values: pd.Series, denominator: float) -> pd.Series:
    """Scale a component to [0, 1] without creating hard outcome thresholds."""

    return (values / denominator).clip(lower=0, upper=1)


def _sigmoid(values: pd.Series | np.ndarray) -> np.ndarray:
    """Compute logistic probabilities from stress scores."""

    return 1 / (1 + np.exp(-values))


if __name__ == "__main__":
    save_model_dataset()
