"""Sample weights reach every statistic a fit computes from its rows.

A weight scales a row's gradient and hessian, and it also weighs the row in
the base score and, for the objectives that renew their leaves (``mae``,
``huber``, ``quantile``), in each leaf's value. The tests below hold the two
ends of that rule: a row whose weight is zero changes nothing whatever its
label says, and a fit whose weights are all zero is refused.

Run with  PYTHONPATH=build/python .venv/bin/python -m pytest python/tests -q
"""

from __future__ import annotations

import bonsai
import numpy as np
import pytest

N_ROWS = 400
REGRESSION_OBJECTIVES = ("mse", "mae", "huber", "quantile", "poisson")
GROWERS = ("depthwise", "leafwise", "levelwise")


def _features(seed=0, p=4):
    return np.random.default_rng(seed).standard_normal((N_ROWS, p)).astype(np.float32)


def _half_zero_weights(seed=1):
    """Weights of one and zero, drawn so both halves reach every part of the features."""
    return np.random.default_rng(seed).permutation(np.arange(N_ROWS) % 2).astype(np.float32)


def _relabelled(y, weights, value):
    """``y`` with every zero-weight row's label replaced by ``value``."""
    out = y.copy()
    out[weights == 0] = value
    return out


# INVARIANT: estimators-ignore-zero-weight-rows
# The label of a row whose sample_weight is zero never reaches the fit: whatever it says,
# every estimator predicts the same values, for every objective and for a multi-target y.
@pytest.mark.parametrize("objective", REGRESSION_OBJECTIVES)
@pytest.mark.parametrize("grower", GROWERS)
def test_regressor_ignores_the_label_of_a_zero_weight_row(objective, grower):
    X = _features()
    weights = _half_zero_weights()
    y = np.abs(X[:, 0] * 2 + X[:, 1]).astype(np.float32)

    def predictions(zero_weight_label):
        model = bonsai.BonsaiRegressor(
            n_iters=10,
            learning_rate=0.3,
            max_depth=3,
            objective=objective,
            quantile_alpha=0.8,
            grower=grower,
            n_threads=2,
        )
        return model.fit(
            X, _relabelled(y, weights, zero_weight_label), sample_weight=weights
        ).predict(X)

    np.testing.assert_array_equal(predictions(100.0), predictions(0.0))


def test_regressor_ignores_the_labels_of_a_zero_weight_row_for_every_target():
    X = _features()
    weights = _half_zero_weights()
    Y = np.stack([X[:, 0] * 2 + X[:, 1], np.sin(X[:, 1]) - X[:, 2]], axis=1).astype(np.float32)

    def predictions(zero_weight_label):
        model = bonsai.BonsaiRegressor(n_iters=10, learning_rate=0.3, max_depth=3, n_threads=2)
        return model.fit(
            X, _relabelled(Y, weights, zero_weight_label), sample_weight=weights
        ).predict(X)

    np.testing.assert_array_equal(predictions(100.0), predictions(-100.0))


@pytest.mark.parametrize("n_classes", [2, 3])
def test_classifier_ignores_the_label_of_a_zero_weight_row(n_classes):
    X = _features()
    weights = _half_zero_weights()
    y = np.digitize(X[:, 0], np.linspace(-0.5, 0.5, n_classes - 1)).astype(np.int64)

    def probabilities(zero_weight_label):
        model = bonsai.BonsaiClassifier(n_iters=10, learning_rate=0.3, max_depth=3, n_threads=2)
        return model.fit(
            X, _relabelled(y, weights, zero_weight_label), sample_weight=weights
        ).predict_proba(X)

    np.testing.assert_array_equal(probabilities(0), probabilities(n_classes - 1))


@pytest.mark.parametrize("objective", ["mae", "quantile"])
@pytest.mark.parametrize("grower", GROWERS)
def test_train_with_unit_and_zero_weights_matches_the_fit_on_the_unit_rows(objective, grower):
    """One binary feature and one split, so a prediction is the base score plus a renewed leaf."""
    rng = np.random.default_rng(3)
    X = rng.integers(0, 2, size=(N_ROWS, 1)).astype(np.float32)
    y = (X[:, 0] * 3 + rng.standard_t(2, size=N_ROWS)).astype(np.float32)
    weights = _half_zero_weights()
    kept = weights > 0
    params = {
        "parallel.n_threads": 2,
        "booster.n_iters": 1,
        "booster.learning_rate": 1.0,
        "tree.max_depth": 1,
        "tree.lambda_l2": 0.0,
        "tree.min_child_hess": 0.0,
        "dispatch.objective_name": objective,
        "objective.quantile_alpha": 0.8,
        "dispatch.grower_name": grower,
    }
    grid = np.array([[0.0], [1.0]], dtype=np.float32)

    weighted = bonsai.train(params, X, y, sample_weight=weights).predict(grid)
    kept_rows_only = bonsai.train(params, X[kept], y[kept]).predict(grid)

    np.testing.assert_array_equal(weighted, kept_rows_only)


@pytest.mark.parametrize("objective", REGRESSION_OBJECTIVES)
def test_regressor_refuses_a_fit_with_no_positive_weight(objective):
    X = _features()
    y = np.abs(X[:, 0]).astype(np.float32)
    model = bonsai.BonsaiRegressor(n_iters=2, objective=objective, n_threads=2)

    with pytest.raises(ValueError, match="positive weight"):
        model.fit(X, y, sample_weight=np.zeros(N_ROWS, dtype=np.float32))


@pytest.mark.parametrize("n_classes", [2, 3])
def test_classifier_refuses_a_fit_with_no_positive_weight(n_classes):
    X = _features()
    y = np.digitize(X[:, 0], np.linspace(-0.5, 0.5, n_classes - 1)).astype(np.int64)
    model = bonsai.BonsaiClassifier(n_iters=2, n_threads=2)

    with pytest.raises(ValueError, match="positive weight"):
        model.fit(X, y, sample_weight=np.zeros(N_ROWS, dtype=np.float32))
