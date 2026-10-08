import random

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.reproducibility import (
    DEFAULT_RANDOM_SEED,
    deterministic_split_kwargs,
    set_global_seed,
)


def test_set_global_seed_repeats_random_sequences() -> None:
    set_global_seed(DEFAULT_RANDOM_SEED)
    first_python_value = random.random()
    first_numpy_values = np.random.random(3)

    set_global_seed(DEFAULT_RANDOM_SEED)
    second_python_value = random.random()
    second_numpy_values = np.random.random(3)

    assert first_python_value == second_python_value
    np.testing.assert_array_equal(first_numpy_values, second_numpy_values)


def test_deterministic_split_kwargs_repeats_train_test_split() -> None:
    features = np.arange(20).reshape(10, 2)
    labels = np.array([0, 1] * 5)
    kwargs = deterministic_split_kwargs(seed=DEFAULT_RANDOM_SEED, test_size=0.3)

    first_split = train_test_split(features, labels, **kwargs)
    second_split = train_test_split(features, labels, **kwargs)

    for first, second in zip(first_split, second_split):
        np.testing.assert_array_equal(first, second)
