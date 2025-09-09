from typing import Callable, Dict, Any

import numpy as np

from .cost_functions import (
    calculate_hungarian_distance,
    modified_hungarian_distance,
    scaled_modified_hungarian_distance,
)


def _hungarian_wrapper(strategy: str, reduction: str):
    def fn(peaks1: np.ndarray, peaks2: np.ndarray, **_: Any) -> float:
        return calculate_hungarian_distance(peaks1, peaks2, strategy=strategy, reduction=reduction)

    return fn


def _modified_wrapper(matching: str):
    def fn(peaks1: np.ndarray, peaks2: np.ndarray, **kwargs: Any):
        return modified_hungarian_distance(peaks1, peaks2, matching=matching, **kwargs)

    return fn


def _modified_default(peaks1: np.ndarray, peaks2: np.ndarray, **kwargs: Any):
    return modified_hungarian_distance(peaks1, peaks2, **kwargs)


# Public mapping used throughout the codebase
DISTANCE_FUNCTION_MAP: Dict[str, Callable[..., float]] = {
    # Standard Hungarian with different unmatched strategies and reductions
    'hungarian_nn_sum': _hungarian_wrapper('nn', 'sum'),
    'hungarian_nn_mean': _hungarian_wrapper('nn', 'mean'),
    'hungarian_trunc_sum': _hungarian_wrapper('trunc', 'sum'),
    'hungarian_trunc_mean': _hungarian_wrapper('trunc', 'mean'),
    'hungarian_zero_sum': _hungarian_wrapper('zero', 'sum'),
    'hungarian_zero_mean': _hungarian_wrapper('zero', 'mean'),
    # Modified Hungarian (uncertainty-aware)
    'modified_hungarian': _modified_default,
    'modified_hungarian_zero': _modified_wrapper('zero'),
    'modified_hungarian_nn': _modified_wrapper('nn'),
    'modified_hungarian_trunc': _modified_wrapper('trunc'),
    # Legacy aliases
    'Hungarian_Distance': _hungarian_wrapper('nn', 'sum'),
    'hung_norm': _hungarian_wrapper('nn', 'mean'),
    'hung_sum': _hungarian_wrapper('nn', 'sum'),
    'hung_modified': _modified_default,
    'hung_modified_2': _modified_default,
    'modified_hungarian_distance': _modified_default,
}


# Reasonable defaults for distance function parameters
DISTANCE_FUNCTION_PARAMS: Dict[str, Dict[str, Any]] = {
    # Standard Hungarian do not require parameters
    'hungarian_nn_sum': {},
    'hungarian_nn_mean': {},
    'hungarian_trunc_sum': {},
    'hungarian_trunc_mean': {},
    'hungarian_zero_sum': {},
    'hungarian_zero_mean': {},
    # Modified Hungarian defaults tuned for HSQC
    'modified_hungarian': {'sigma_H': 0.01, 'sigma_C': 0.2, 'func_H': 0.5, 'func_C': 2.5, 'penalty_factor': 1.0},
    'modified_hungarian_zero': {'sigma_H': 0.01, 'sigma_C': 0.2, 'func_H': 0.5, 'func_C': 2.5, 'penalty_factor': 1.0},
    'modified_hungarian_nn': {'sigma_H': 0.01, 'sigma_C': 0.2, 'func_H': 0.5, 'func_C': 2.5, 'penalty_factor': 1.0},
    'modified_hungarian_trunc': {'sigma_H': 0.01, 'sigma_C': 0.2, 'func_H': 0.5, 'func_C': 2.5, 'penalty_factor': 1.0},
    # Legacy aliases
    'Hungarian_Distance': {},
    'hung_norm': {},
    'hung_sum': {},
    'hung_modified': {'sigma_H': 0.01, 'sigma_C': 0.2, 'func_H': 0.5, 'func_C': 2.5, 'penalty_factor': 1.0},
    'hung_modified_2': {'sigma_H': 0.01, 'sigma_C': 0.2, 'func_H': 0.5, 'func_C': 2.5, 'penalty_factor': 1.0},
    'modified_hungarian_distance': {'sigma_H': 0.01, 'sigma_C': 0.2, 'func_H': 0.5, 'func_C': 2.5, 'penalty_factor': 1.0},
}


__all__ = [
    'DISTANCE_FUNCTION_MAP',
    'DISTANCE_FUNCTION_PARAMS',
    'calculate_hungarian_distance',
    'modified_hungarian_distance',
    'scaled_modified_hungarian_distance',
]


