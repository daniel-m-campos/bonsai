"""Tests for bonsai.params (the generated typed overrides) and the train
wrapper's params normalization."""

from __future__ import annotations

import dataclasses

import bonsai
import numpy as np
import pytest
from bonsai import _bonsai
from bonsai._train import _as_pairs
from bonsai.params import Booster, Params, Tree


def _reg_data(n=3000, f=8, seed=0):
    """(X, y) float32 arrays with signal in the first column."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, f), dtype=np.float32)
    y = (X[:, 0] * 2 + rng.normal(0, 0.1, n)).astype(np.float32)
    return X, y


# Params ===========================================================================================


def test_params_default_renders_no_overrides():
    assert Params().to_dict() == {}
    assert _as_pairs(Params()) == []


def test_params_mirrors_the_section_registry():
    """The generated dataclasses carry exactly the registry's sections and
    fields, so a registry change cannot drift past the generator silently."""
    schema = _bonsai._params_schema()
    assert [s["section"] for s in schema] == [
        f.name for f in dataclasses.fields(Params) if not f.name.startswith("_")
    ]
    for sec in schema:
        cls = Params._SECTION_TYPES[sec["section"]]
        assert [f["name"] for f in sec["fields"]] == [
            f.name for f in dataclasses.fields(cls)
        ]


def test_params_render_types_the_parser_reads():
    p = Params.from_dict({
        "tree.max_depth": 8,
        "booster.learning_rate": 0.1,
        "metrics.fit": ["rmse", "mae"],
        "data.header": True,
    })
    assert dict(_as_pairs(p)) == {
        "tree.max_depth": "8",
        "booster.learning_rate": "0.1",
        "metrics.fit": "rmse,mae",
        "data.header": "true",
    }


def test_params_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match=r"expected 'section\.name'"):
        Params.from_dict({"max_depth": 8})
    with pytest.raises(ValueError, match=r"\[tree\] has"):
        Params.from_dict({"tree.max_dept": 8})


def test_params_or_merges_right_side_wins():
    base = Params(tree=Tree(max_depth=8), booster=Booster(n_iters=100))
    swept = base | {"tree.max_depth": 10}
    assert swept.to_dict() == {"tree.max_depth": 10, "booster.n_iters": 100}
    # the base is untouched, and Params | Params merges the same way
    assert base.tree.max_depth == 8
    assert (base | Params(tree=Tree(max_depth=12))).tree.max_depth == 12


def test_params_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Params().tree = Tree()


def test_params_repr_shows_only_set_fields():
    assert repr(Params(tree=Tree(max_depth=4))) == "Params(tree=Tree(max_depth=4))"


# train wrapper ====================================================================================


def test_train_params_forms_are_bit_identical():
    """Params and dict (string or typed values) render to one wire format,
    so every form (and both data forms) produces the same model bit for
    bit."""
    X, y = _reg_data()
    ds = bonsai.Dataset(X, y)
    p = Params(booster=Booster(n_iters=15), tree=Tree(max_depth=5))
    ref = np.asarray(bonsai.train(p, ds).predict(X))
    for params in ({"booster.n_iters": "15", "tree.max_depth": "5"},
                   {"booster.n_iters": 15, "tree.max_depth": 5}):
        np.testing.assert_array_equal(
            ref, np.asarray(bonsai.train(params, ds).predict(X)))
    np.testing.assert_array_equal(
        ref, np.asarray(bonsai.train(p, X, y).predict(X)))


def test_train_rejects_the_retired_pairs_form():
    """The (key, value) pairs list is the internal wire format; the public
    wrapper names the two accepted forms instead of guessing."""
    X, y = _reg_data(n=500)
    with pytest.raises(TypeError, match="dict\\(pairs\\)"):
        bonsai.train([("booster.n_iters", "5")], bonsai.Dataset(X, y))


def test_train_accepts_none_params():
    X, y = _reg_data(n=500)
    assert bonsai.train(None, bonsai.Dataset(X, y)).n_iters == 100
