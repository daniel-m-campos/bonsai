"""Input coercion shared by the estimator layer: contiguous float32
views for the native module."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


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
    arr = np.ascontiguousarray(_as_array(a, np.float32))
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {arr.shape}")
    return arr


def _as_targets(a, name: str, dtype: npt.DTypeLike = np.float32) -> np.ndarray:
    """Contiguous targets: a vector, or one target per column.

    A single-column 2-D array is read as a vector of one target, the way
    sklearn's regressors read it.

    Parameters
    ----------
    a
        Anything numpy can make an array of.
    name
        The argument's name, for the error message.
    dtype
        The element type: float32 for the native boundary, float64 where
        the caller's targets are scored rather than trained on.

    Returns
    -------
    numpy.ndarray
        C-contiguous ``dtype`` of rank 1, or rank 2 with at least two columns.

    Raises
    ------
    ValueError
        When the array's rank is neither 1 nor 2.
    """
    arr = np.ascontiguousarray(_as_array(a, dtype))
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = np.ascontiguousarray(arr[:, 0])
    if arr.ndim not in (1, 2):
        raise ValueError(f"{name} must be 1- or 2-dimensional, got shape {arr.shape}")
    return arr


def _as_array(a, dtype: npt.DTypeLike) -> np.ndarray:
    """Floating array of ``a``, with pandas' missing cells read as NaN.

    A pandas frame mixing nullable columns (``Int64``, ``Float64``) with
    others converts through an object array, where ``pd.NA`` cannot become a
    float; pandas' own conversion takes the missing value explicitly.

    Parameters
    ----------
    a
        Anything numpy can make an array of, or a pandas object.
    dtype
        The floating dtype of the result.

    Returns
    -------
    numpy.ndarray
        ``a`` as ``dtype``, NaN where pandas held ``pd.NA``.
    """
    if type(a).__module__.startswith("pandas"):
        return a.to_numpy(dtype=dtype, na_value=np.nan)
    return np.asarray(a, dtype=dtype)
