"""Tests for bonsai.bench.metrics (against sklearn)."""

from __future__ import annotations

import sys

import numpy as np
from bonsai.bench import metrics


def test_metrics_against_sklearn():
    rng = np.random.default_rng(0)
    y = rng.random(500)
    pred = y + rng.normal(0, 0.1, 500)
    yb = (rng.random(500) > 0.5).astype(float)
    scores = yb * 0.6 + rng.random(500) * 0.4
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
    assert abs(metrics.r2(y, pred) - r2_score(y, pred)) < 1e-12
    assert abs(metrics.rmse(y, pred) - mean_squared_error(y, pred) ** 0.5) < 1e-12
    assert abs(metrics.mae(y, pred) - mean_absolute_error(y, pred)) < 1e-12
    assert abs(metrics.auc(yb, scores) - roc_auc_score(yb, scores)) < 1e-12
    # the numpy fallback must agree with sklearn, including under ties
    tied = np.round(scores, 1)
    import unittest.mock as mock
    with mock.patch.dict(sys.modules, {"sklearn.metrics": None, "sklearn": None}):
        fallback = metrics.auc(yb, tied)
    assert abs(fallback - roc_auc_score(yb, tied)) < 1e-12


def test_additivity_is_the_worst_row_relative_residual():
    phi = np.array([[1.0, 2.0, 0.5], [0.1, 0.1, 0.1]])  # rows sum to 3.5, 0.3
    assert metrics.additivity(phi, np.array([3.5, 0.3])) < 1e-9
    # row 1's margin moves to 1.0: |0.3 - 1.0| / max(1, |1.0|) = 0.7
    assert abs(metrics.additivity(phi, np.array([3.5, 1.0])) - 0.7) < 1e-9
    # the >= 1 floor in the denominator: a tiny margin does not blow up a
    # tiny absolute residual into a huge relative one
    assert abs(metrics.additivity(np.array([[1e-9]]), np.array([0.0])) - 1e-9) < 1e-15
