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
