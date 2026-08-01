"""Input coercion shared by the estimator layer: contiguous float32
views for the native module, and config values rendered as the strings the
dotted-key config system expects."""

from __future__ import annotations

import numpy as np


def _as_2d_f32(X) -> np.ndarray:
    """Contiguous float32 2-D view; the native boundary's input shape."""
    a = np.ascontiguousarray(X, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {a.shape}")
    return a


def _as_1d_f32(y) -> np.ndarray:
    """Contiguous float32 1-D view; the native boundary's target shape."""
    a = np.ascontiguousarray(y, dtype=np.float32)
    if a.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got shape {a.shape}")
    return a


def _to_config_str(v) -> str:
    """Render a config value the way the dotted-key parser reads it back
    (bool before the generic branches: bools are ints to isinstance)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)
