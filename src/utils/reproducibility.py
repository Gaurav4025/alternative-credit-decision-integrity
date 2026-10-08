"""Utilities for reproducible research experiments."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

DEFAULT_RANDOM_SEED = 42


def set_global_seed(seed: int = DEFAULT_RANDOM_SEED) -> int:
    """Seed Python and NumPy random number generators.

    Returns the seed so callers can record it in experiment metadata.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def deterministic_split_kwargs(
    seed: int = DEFAULT_RANDOM_SEED,
    test_size: float = 0.2,
    shuffle: bool = True,
    stratify: Any | None = None,
) -> dict[str, Any]:
    """Return standard kwargs for deterministic scikit-learn train/test splits."""

    return {
        "test_size": test_size,
        "shuffle": shuffle,
        "random_state": seed,
        "stratify": stratify,
    }
