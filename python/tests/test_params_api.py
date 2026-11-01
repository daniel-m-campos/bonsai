"""Tests for bonsai.params (the generated typed overrides) and the params
forms the native train binding accepts."""

from __future__ import annotations

import dataclasses
import tempfile
from typing import Any

import bonsai
import numpy as np
import pytest
from bonsai import _bonsai
from bonsai.params import Booster, Params, Tree


def _reg_data(n=3000, f=8, seed=0):
    """(X, y) float32 arrays with signal in the first column."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, f), dtype=np.float32)
    y = (X[:, 0] * 2 + rng.normal(0, 0.1, n)).astype(np.float32)
    return X, y


# Params ===========================================================================================


def test_params_default_renders_no_overrides():
    """A default-constructed Params sets nothing, so the fit runs on the
    library defaults."""
    X, y = _reg_data(n=500)
    assert Params().to_dict() == {}
    assert bonsai.train(Params(), bonsai.Dataset(X, y)).n_iters == 100


def test_params_mirrors_the_section_registry():
    """The generated dataclasses carry exactly the registry's sections and
    fields, so a registry change cannot drift past the generator silently."""
    schema = _bonsai._params_schema()
    assert [s["section"] for s in schema] == [
        f.name for f in dataclasses.fields(Params) if not f.name.startswith("_")
    ]
    for sec in schema:
        cls = Params._SECTION_TYPES[sec["section"]]
        assert [f["name"] for f in sec["fields"]] == [f.name for f in dataclasses.fields(cls)]


def test_params_render_types_the_parser_reads():
    """The binding renders each typed value to the string the dotted-key
    parser reads back, so the round trip through a fit restores the type:
    ints, floats, string lists, and bools (which are not ints here)."""
    X, y = _reg_data(n=500)
    p = Params.from_dict(
        {
            "tree.max_depth": 8,
            "booster.learning_rate": 0.1,
            "metrics.fit": ["rmse", "mae"],
            "data.header": True,
        }
    )
    resolved = Params.from_model(bonsai.train(p, bonsai.Dataset(X, y)))
    assert resolved.tree is not None and resolved.booster is not None
    assert resolved.metrics is not None and resolved.data is not None
    assert resolved.tree.max_depth == 8
    assert resolved.booster.learning_rate == pytest.approx(0.1)
    assert resolved.metrics.fit == ["rmse", "mae"]
    assert resolved.data.header is True


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
    assert base.tree is not None and base.tree.max_depth == 8
    merged = base | Params(tree=Tree(max_depth=12))
    assert merged.tree is not None and merged.tree.max_depth == 12


def test_params_is_frozen():
    frozen: Any = Params()
    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen.tree = Tree()


def test_params_repr_shows_only_set_fields():
    assert repr(Params(tree=Tree(max_depth=4))) == "Params(tree=Tree(max_depth=4))"


def test_params_from_toml_carries_only_stated_keys():
    """from_toml is a sparse overrides object: only the file's keys, typed,
    strict on unknowns — the config= layering, now composable with |."""
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write("[tree]\nmax_depth = 4\nlambda_l2 = 0.5\n[booster]\nn_iters = 9\n")
        path = f.name
    p = Params.from_toml(path)
    assert p.to_dict() == {"tree.max_depth": 4, "tree.lambda_l2": 0.5, "booster.n_iters": 9}
    swept = p | {"booster.n_iters": 3}
    assert swept.booster is not None and swept.booster.n_iters == 3

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write("[tree]\nmax_dept = 4\n")
        bad = f.name
    with pytest.raises(Exception, match="max_dept"):
        Params.from_toml(bad)


def test_params_from_model_is_the_resolved_config():
    X, y = _reg_data(n=500)
    m = bonsai.train({"tree.max_depth": 3}, bonsai.Dataset(X, y))
    p = Params.from_model(m)
    assert p.tree is not None and p.booster is not None
    assert p.tree.max_depth == 3
    assert p.booster.n_iters == 100  # resolved default: every key is set


# train params ====================================================================================


def test_train_params_forms_are_bit_identical():
    """Params and dict (string or typed values) render to one wire format,
    so every form (and both data forms) produces the same model bit for
    bit."""
    X, y = _reg_data()
    ds = bonsai.Dataset(X, y)
    p = Params(booster=Booster(n_iters=15), tree=Tree(max_depth=5))
    ref = np.asarray(bonsai.train(p, ds).predict(X))
    for params in (
        {"booster.n_iters": "15", "tree.max_depth": "5"},
        {"booster.n_iters": 15, "tree.max_depth": 5},
    ):
        np.testing.assert_array_equal(ref, np.asarray(bonsai.train(params, ds).predict(X)))
    np.testing.assert_array_equal(ref, np.asarray(bonsai.train(p, X, y).predict(X)))


def test_train_rejects_the_retired_pairs_form():
    """The (key, value) pairs list is the internal wire format; the binding
    names the two accepted forms instead of guessing."""
    X, y = _reg_data(n=500)
    pairs: Any = [("booster.n_iters", "5")]
    with pytest.raises(TypeError, match="dict\\(pairs\\)"):
        bonsai.train(pairs, bonsai.Dataset(X, y))


def test_train_accepts_none_params():
    X, y = _reg_data(n=500)
    assert bonsai.train(None, bonsai.Dataset(X, y)).n_iters == 100


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            "booster.learning_rate",
            float("nan"),
            r"\[booster\]\.learning_rate: must be finite and > 0",
        ),
        ("booster.dart_drop_rate", 1.5, r"\[booster\]\.dart_drop_rate: must be in \[0, 1\]"),
        ("tree.lambda_l1", -500.0, r"\[tree\]\.lambda_l1: must be finite and >= 0"),
        ("tree.feature_fraction", 0.0, r"\[tree\]\.feature_fraction: must be in \(0, 1\]"),
        ("objective.quantile_alpha", 1.5, r"\[objective\]\.quantile_alpha: must be in \(0, 1\)"),
        ("bin_mapper.max_bin", 100_000, r"\[bin_mapper\]\.max_bin: must be in \[3, 65535\]"),
        ("tree.min_data_in_leaf", 256, r"\[tree\]\.min_data_in_leaf: must be at most 255"),
        (
            "booster.random_seed",
            2**32 + 42,
            r"\[booster\]\.random_seed: must be at most 4294967295",
        ),
        ("sampler.subsample", 1.5, r"sampler\.subsample must be in \(0, 1\]"),
    ],
)
def test_train_refuses_a_value_outside_its_key_domain(key, value, message):
    X, y = _reg_data(n=200, f=3)
    params = {"booster.n_iters": 2, "dispatch.sampler_name": "bernoulli", key: value}
    with pytest.raises(RuntimeError, match=message):
        bonsai.train(params, X, y)


def test_train_refuses_a_bad_positional_monotone_list():
    X, y = _reg_data(n=200, f=3)
    with pytest.raises(RuntimeError, match=r"item 0: must be in \[-1, 1\] \(got 2\)"):
        bonsai.train({"tree.monotone_constraints": [2, 0, 0]}, X, y)
    with pytest.raises(RuntimeError, match="lists 5 directions but the training data has 3"):
        bonsai.train({"tree.monotone_constraints": [1, 0, 0, -1, 1]}, X, y)
