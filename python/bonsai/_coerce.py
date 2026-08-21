"""Input coercion shared by the estimator layer: contiguous float32
views for the native module."""

from __future__ import annotations

import numpy as np


def _as_f32(a, ndim: int, name: str) -> np.ndarray:
    """Contiguous float32 view of the given rank; the native boundary's shape.

    Parameters
    ----------
    a
        Anything numpy can make an array of.
    ndim
        The rank the boundary requires: 2 for a feature matrix, 1 for a target
        or weight vector.
    name
        The argument's name, for the error message.

    Returns
    -------
    numpy.ndarray
        C-contiguous float32, rank ``ndim``.

    Raises
    ------
    ValueError
        When the array's rank is not ``ndim``.
    """
    arr = np.ascontiguousarray(a, dtype=np.float32)
    if arr.ndim != ndim:
        raise ValueError(
            f"{name} must be {ndim}-dimensional, got shape {arr.shape}")
    return arr

