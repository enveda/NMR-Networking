from typing import Callable, Dict, Any, Optional

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
    # Zone-1 weighted: downfield (high 1H & 13C) peaks dominate the score
    'modified_hungarian_zone1': _modified_default,
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
    'modified_hungarian_zone1': {'sigma_H': 0.01, 'sigma_C': 0.2, 'func_H': 0.5, 'func_C': 2.5, 'penalty_factor': 1.0,
                                 'zone_floor': 0.03, 'zone_gamma': 2.0, 'zone_combine': 'avg',
                                 # zone 1 = downfield of 3 ppm (1H) and 50 ppm (13C);
                                 # the weight ramp is anchored to these thresholds.
                                 'H_range': (3.0, 10.0), 'C_range': (50.0, 200.0)},
    # Legacy aliases
    'Hungarian_Distance': {},
    'hung_norm': {},
    'hung_sum': {},
    'hung_modified': {'sigma_H': 0.01, 'sigma_C': 0.2, 'func_H': 0.5, 'func_C': 2.5, 'penalty_factor': 1.0},
    'hung_modified_2': {'sigma_H': 0.01, 'sigma_C': 0.2, 'func_H': 0.5, 'func_C': 2.5, 'penalty_factor': 1.0},
    'modified_hungarian_distance': {'sigma_H': 0.01, 'sigma_C': 0.2, 'func_H': 0.5, 'func_C': 2.5, 'penalty_factor': 1.0},
}


def resolve_distance_params(distance_function: str,
                            zone_floor: Optional[float] = None,
                            zone_gamma: Optional[float] = None,
                            zone_combine: Optional[str] = None) -> Dict[str, Any]:
    """Return the registry defaults for ``distance_function``, overriding any
    zone-weighting params that were explicitly supplied (non-``None``).

    Lets a CLI expose ``--zone-floor`` / ``--zone-gamma`` / ``--zone-combine``
    without hard-coding them: unset flags fall back to the registered defaults.
    """
    params = dict(DISTANCE_FUNCTION_PARAMS.get(distance_function, {}))
    if zone_floor is not None:
        params['zone_floor'] = zone_floor
    if zone_gamma is not None:
        params['zone_gamma'] = zone_gamma
    if zone_combine is not None:
        params['zone_combine'] = zone_combine
    return params


__all__ = [
    'DISTANCE_FUNCTION_MAP',
    'DISTANCE_FUNCTION_PARAMS',
    'resolve_distance_params',
    'calculate_hungarian_distance',
    'modified_hungarian_distance',
    'scaled_modified_hungarian_distance',
]


