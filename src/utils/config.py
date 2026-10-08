"""Configuration primitives for reproducible experiments."""

from dataclasses import dataclass

from src.utils.reproducibility import DEFAULT_RANDOM_SEED


@dataclass(frozen=True)
class ResearchConfig:
    """Minimal immutable configuration shared by early experiments."""

    random_seed: int = DEFAULT_RANDOM_SEED
    test_size: float = 0.2
