"""Input coercion shared by the estimator layer: contiguous float32
views for the native module, and config values rendered as the strings the
dotted-key config system expects."""

from __future__ import annotations

import collections.abc

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


def _to_config_str(v) -> object:
    """Render a config value the way the dotted-key parser reads it back
    (bool before the generic branches: bools are ints to isinstance).

    A Mapping is handed on untouched, because the native layer renders it:
    ``tree.monotone_constraints`` may be keyed by feature name, and the names
    belong to the training data, which this layer has not seen.
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, collections.abc.Mapping):
        return v
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)
