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
