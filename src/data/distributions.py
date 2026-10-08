"""Distribution helpers for synthetic alternative-financial behavior.

The functions in this module generate hidden applicant traits used by the
synthetic data generator. These traits are not intended to be model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


LATENT_COLUMNS = [
    "income_stability",
    "spending_discipline",
    "savings_behavior",
    "payment_discipline",
    "financial_liquidity",
    "transaction_activity",
]

POPULATION_GROUPS = ["stable", "volatile", "borderline"]


@dataclass(frozen=True)
class PopulationSpec:
    """Latent distribution settings for one synthetic population group."""

    mean: tuple[float, ...]
    sd: float


POPULATION_SPECS = {
    "stable": PopulationSpec(mean=(0.72, 0.68, 0.64, 0.72, 0.66, 0.58), sd=0.16),
    "volatile": PopulationSpec(mean=(0.36, 0.43, 0.38, 0.48, 0.40, 0.62), sd=0.18),
    "borderline": PopulationSpec(mean=(0.54, 0.53, 0.50, 0.56, 0.52, 0.56), sd=0.19),
}


def normalize_population_proportions(
    proportions: dict[str, float] | None = None,
) -> dict[str, float]:
    """Validate and normalize population proportions."""

    if proportions is None:
        proportions = {"stable": 0.4, "volatile": 0.3, "borderline": 0.3}

    unknown_groups = set(proportions) - set(POPULATION_GROUPS)
    if unknown_groups:
        raise ValueError(f"Unknown population groups: {sorted(unknown_groups)}")

    if any(value < 0 for value in proportions.values()):
        raise ValueError("Population proportions must be non-negative.")

    total = sum(proportions.values())
    if total <= 0:
        raise ValueError("Population proportions must sum to a positive value.")

    return {group: proportions.get(group, 0.0) / total for group in POPULATION_GROUPS}


def sample_population_groups(
    n_applicants: int,
    rng: np.random.Generator,
    proportions: dict[str, float] | None = None,
) -> np.ndarray:
    """Sample applicant population groups according to normalized proportions."""

    normalized = normalize_population_proportions(proportions)
    return rng.choice(
        POPULATION_GROUPS,
        size=n_applicants,
        p=[normalized[group] for group in POPULATION_GROUPS],
    )


def sample_latent_traits(
    groups: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample correlated latent traits for each applicant.

    Traits are bounded to ``[0, 1]`` after drawing from overlapping multivariate
    normal distributions. The overlap is intentional: group labels should not be
    trivially recoverable from the eventual feature table.
    """

    correlation = np.array(
        [
            [1.00, 0.35, 0.35, 0.28, 0.35, -0.05],
            [0.35, 1.00, 0.45, 0.30, 0.38, -0.12],
            [0.35, 0.45, 1.00, 0.32, 0.55, 0.00],
            [0.28, 0.30, 0.32, 1.00, 0.34, 0.04],
            [0.35, 0.38, 0.55, 0.34, 1.00, 0.02],
            [-0.05, -0.12, 0.00, 0.04, 0.02, 1.00],
        ]
    )

    rows: list[np.ndarray] = []
    for group in groups:
        spec = POPULATION_SPECS[str(group)]
        covariance = correlation * (spec.sd**2)
        sample = rng.multivariate_normal(mean=np.array(spec.mean), cov=covariance)
        rows.append(np.clip(sample, 0.02, 0.98))

    return pd.DataFrame(rows, columns=LATENT_COLUMNS)
