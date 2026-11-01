"""The scikit-learn contract of BonsaiRegressor and BonsaiClassifier: the
shapes, refusals and fitted state sklearn's own estimators promise."""

from __future__ import annotations

import bonsai
import numpy as np
import pytest


def _binary(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A small separable binary problem with 0/1 labels."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    return X, (X[:, 0] - X[:, 1] > 0).astype(np.int64)


def _classifier(**kwargs) -> bonsai.BonsaiClassifier:
    return bonsai.BonsaiClassifier(n_iters=10, n_threads=2, **kwargs)


def test_classifier_score_reads_a_label_column_as_a_vector():
    X, y = _binary()
    clf = _classifier().fit(X, y)

    assert clf.score(X, y.reshape(-1, 1)) == clf.score(X, y)


def test_classifier_eval_set_reads_a_label_column_as_a_vector():
    X, y = _binary()
    column = y.reshape(-1, 1)

    clf = _classifier().fit(X, column, eval_set=(X[:100], column[:100]))

    assert len(clf.evals_result()["valid"]["logloss"]) == clf.n_iters_


def test_classifier_refuses_labels_wider_than_one_column():
    X, y = _binary()
    with pytest.raises(ValueError, match="y must be 1-dimensional or a single column"):
        _classifier().fit(X, np.column_stack([y, y]))


def _regression(n: int = 200, n_targets: int = 1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A small linear problem with ``n_targets`` columns of y."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    Y = np.column_stack([X[:, k] + 0.1 * rng.normal(size=n) for k in range(n_targets)])
    return X, Y[:, 0] if n_targets == 1 else Y


def test_score_refuses_y_whose_row_count_differs_from_X():
    Xr, yr = _regression()
    Xc, yc = _binary(200)
    reg = bonsai.BonsaiRegressor(n_iters=5, n_threads=2).fit(Xr, yr)
    clf = _classifier().fit(Xc, yc)

    for est, X, y in ((reg, Xr, yr), (clf, Xc, yc)):
        with pytest.raises(ValueError, match=r"len\(y\) must equal the row count of X"):
            est.score(X, y[:1])
        with pytest.raises(ValueError, match=r"len\(y\) must equal the row count of X"):
            est.score(X[:1], y)


def test_score_refuses_sample_weight_whose_row_count_differs_from_X():
    X, y = _binary(200)
    clf = _classifier().fit(X, y)

    with pytest.raises(ValueError, match=r"len\(sample_weight\) must equal the row count"):
        clf.score(X, y, sample_weight=np.ones(1))


def test_regressor_score_refuses_fewer_targets_than_the_model_predicts():
    X, Y = _regression(n_targets=3)
    reg = bonsai.BonsaiRegressor(n_iters=5, n_threads=2).fit(X, Y)

    with pytest.raises(ValueError, match="y has 2 targets but the model predicts 3"):
        reg.score(X, Y[:, :2])


def test_regressor_score_scores_the_targets_in_float64():
    X = np.zeros((40, 2), dtype=np.float32)
    y = np.where(np.arange(40) % 2 == 0, 2.0**24, 2.0**24 + 1)
    reg = bonsai.BonsaiRegressor(n_iters=2, n_threads=2).fit(X, y)
    y_pred = reg.predict(X).astype(np.float64)
    expected = 1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

    assert reg.score(X, y) == pytest.approx(expected)
    assert reg.score(X, y) < 0.0
