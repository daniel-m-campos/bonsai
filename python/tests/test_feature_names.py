"""Tests for feature names: where they come from and what carries them."""

from __future__ import annotations

import bonsai
import numpy as np
import pytest


def _reg_data(n=3000, f=6, seed=0):
    """(X, y) float32 arrays with signal in the first two columns."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, f), dtype=np.float32)
    y = (X[:, 0] * 2 + X[:, 1] + rng.normal(0, 0.1, n)).astype(np.float32)
    return X, y


class _Frame:
    """The slice of a DataFrame bonsai duck-types: `columns` plus DLPack."""

    def __init__(self, values, columns):
        self._values = values
        self.columns = columns

    def __dlpack__(self, *args, **kwargs):
        return self._values.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        return self._values.__dlpack_device__()


def test_feature_names_come_from_the_keyword():
    X, y = _reg_data()
    names = ["age", "income", "tenure", "score", "visits", "spend"]
    ds = bonsai.Dataset(X, y, feature_names=names)
    assert list(ds.feature_names) == names
    text = bonsai.train({"booster.n_iters": "5", "tree.max_depth": "4"}, ds).dump()
    assert "income <=" in text
    assert "f1 <=" not in text


def test_feature_names_come_from_a_dataframe_columns():
    """A DataFrame-like X answers with `columns`; no pandas import anywhere."""
    X, y = _reg_data()
    names = ["age", "income", "tenure", "score", "visits", "spend"]
    ds = bonsai.Dataset(_Frame(X, names), y)
    assert list(ds.feature_names) == names
    # non-str column labels are named by their str(), the way a dump prints them
    ints = bonsai.Dataset(_Frame(X, list(range(6))), y)
    assert list(ints.feature_names) == ["0", "1", "2", "3", "4", "5"]
    # the fused array train path duck-types the same way
    text = bonsai.train({"booster.n_iters": "5"}, _Frame(X, names), y).dump()
    assert "income <=" in text
    assert "f0 <=" not in text


def test_feature_names_default_to_f0_fn():
    X, y = _reg_data()
    ds = bonsai.Dataset(X, y)
    assert list(ds.feature_names) == ["f0", "f1", "f2", "f3", "f4", "f5"]
    assert "f0" in bonsai.train({"booster.n_iters": "5"}, ds).dump()


def test_feature_names_reject_a_wrong_count_or_a_repeat():
    X, y = _reg_data()
    with pytest.raises(ValueError, match="one name per column"):
        bonsai.Dataset(X, y, feature_names=["a", "b", "c"])
    with pytest.raises(ValueError, match="one name per column"):
        bonsai.Dataset(_Frame(X, ["a", "b", "c"]), y)
    with pytest.raises(ValueError, match="must be unique"):
        bonsai.Dataset(X, y, feature_names=["a", "b", "c", "d", "e", "a"])


def test_feature_names_survive_a_reference_dataset():
    """A reference supplies the cuts, and the names travel with them."""
    X, y = _reg_data()
    names = ["age", "income", "tenure", "score", "visits", "spend"]
    train_ds = bonsai.Dataset(X[:2000], y[:2000], feature_names=names)
    valid_ds = bonsai.Dataset(X[2000:], y[2000:], reference=train_ds)
    assert list(valid_ds.feature_names) == names
