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


# Monotone constraints keyed by feature name =======================================================

NAMES = ["age", "income", "tenure", "score", "visits", "spend"]
MONO_PARAMS = {"dispatch.grower_name": "depthwise", "booster.n_iters": "25", "tree.max_depth": "5"}


def _mono(params, **extra):
    """MONO_PARAMS plus a tree.monotone_constraints value."""
    return {**MONO_PARAMS, "tree.monotone_constraints": params, **extra}


def test_named_monotone_matches_the_positional_list_on_both_train_paths(tmp_path):
    """The dict form is a spelling of the list, so it must produce the same
    model byte for byte, not merely the same predictions."""
    X, y = _reg_data()
    positional = [1, 0, -1, 0, 0, 0]
    named = {"age": 1, "tenure": -1}

    ds = bonsai.Dataset(X, y, feature_names=NAMES)
    by_list = bonsai.train(_mono(positional), ds)
    by_name = bonsai.train(_mono(named), ds)
    np.testing.assert_array_equal(np.asarray(by_list.predict(ds)), np.asarray(by_name.predict(ds)))
    assert by_list.config_toml == by_name.config_toml
    for model, stem in ((by_list, "list"), (by_name, "named")):
        model.save(str(tmp_path / f"{stem}.msgpack"))
    assert (tmp_path / "list.msgpack").read_bytes() == (tmp_path / "named.msgpack").read_bytes()

    # the fused array path names its columns the same way, through X.columns
    frame = _Frame(X, NAMES)
    array_list = bonsai.train(_mono(positional), frame, y)
    array_named = bonsai.train(_mono(named), frame, y)
    np.testing.assert_array_equal(
        np.asarray(array_list.predict(X)), np.asarray(array_named.predict(X))
    )


def test_named_monotone_resolves_against_synthesized_names():
    """f0..fN are real names as far as the model is concerned."""
    X, y = _reg_data()
    ds = bonsai.Dataset(X, y)
    by_list = bonsai.train(_mono([0, -1, 0, 0, 0, 1]), ds)
    by_name = bonsai.train(_mono({"f1": -1, "f5": 1}), ds)
    assert by_list.config_toml == by_name.config_toml
    np.testing.assert_array_equal(np.asarray(by_list.predict(ds)), np.asarray(by_name.predict(ds)))


def test_named_monotone_leaves_unlisted_features_free():
    """An unlisted feature is 0, not absent: a partial dict equals the explicit
    all-zeros-but-one list."""
    X, y = _reg_data()
    ds = bonsai.Dataset(X, y, feature_names=NAMES)
    explicit = bonsai.train(_mono([0, 0, 0, 0, 0, 0]), ds)
    empty = bonsai.train(_mono({}), ds)
    assert explicit.config_toml == empty.config_toml
    np.testing.assert_array_equal(np.asarray(explicit.predict(ds)), np.asarray(empty.predict(ds)))


def test_named_monotone_rejects_unknown_names_and_bad_values():
    X, y = _reg_data()
    ds = bonsai.Dataset(X, y, feature_names=NAMES)
    with pytest.raises(ValueError, match="does not have: 'salary'"):
        bonsai.train(_mono({"age": 1, "salary": -1}), ds)
    # at most five offenders are listed, then the count of the rest
    with pytest.raises(ValueError, match=r"and 2 more\).*carries 6 feature names"):
        bonsai.train(_mono({f"nope{i}": 1 for i in range(7)}), ds)
    with pytest.raises(ValueError, match="must be the int -1, 0, or 1"):
        bonsai.train(_mono({"age": 2}), ds)
    with pytest.raises(ValueError, match="must be the int -1, 0, or 1"):
        bonsai.train(_mono({"age": "1"}), ds)
    # the array path resolves against its own names, so a Dataset name is
    # unknown there when X carries none
    with pytest.raises(ValueError, match="does not have: 'age'"):
        bonsai.train(_mono({"age": 1}), X, y)


def test_named_monotone_flows_through_the_estimator_params():
    """The estimators hand params= to this same binding, so the dict form
    arrives with no estimator-side code. They coerce X to a numpy array first,
    so the names it resolves against are the synthesized ones."""
    X, y = _reg_data()

    def fit(constraints):
        return bonsai.BonsaiRegressor(
            n_iters=25, max_depth=5, grower="depthwise",
            params={"tree.monotone_constraints": constraints},
        ).fit(X, y)

    np.testing.assert_array_equal(
        fit({"f0": 1, "f2": -1}).predict(X), fit([1, 0, -1, 0, 0, 0]).predict(X)
    )
    with pytest.raises(ValueError, match="does not have: 'age'"):
        fit({"age": 1})
