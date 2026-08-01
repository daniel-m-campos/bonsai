"""Tests for bonsai.Dataset (the reusable pre-binned dataset)."""

from __future__ import annotations

import tempfile

import bonsai
import numpy as np
import pytest


def test_reusable_dataset_bit_identical_and_guard():
    rng = np.random.default_rng(0)
    X = rng.random((3000, 15), dtype=np.float32)
    y = (X[:, 0] + rng.normal(0, 0.1, 3000)).astype(np.float32)
    pairs = [("dispatch.grower_name", "depthwise"), ("booster.n_iters", "40"),
             ("tree.max_depth", "6")]
    ref = np.asarray(bonsai.train(pairs, X, y).predict(X))

    ds = bonsai.Dataset(X, y, max_bin=255)
    assert ds.n_rows == 3000 and ds.n_features == 15
    # bin-once reuse must equal fitting from (X, y) bit for bit
    got = np.asarray(bonsai.train(pairs, ds).predict(X))
    np.testing.assert_array_equal(ref, got)
    # reuse with different hyperparameters (no re-bin)
    assert np.asarray(bonsai.train([("booster.n_iters", "10")], ds).predict(X)).shape == (3000,)
    # binning is fixed by the Dataset — reject a bin_mapper param override
    with pytest.raises(Exception) as e:
        bonsai.train([("bin_mapper.max_bin", "63")], ds)
        assert "bin_mapper" in str(e.value)

    # ...and reject a config file that carries a [bin_mapper] section, which
    # would otherwise be silently ignored (binning comes from the Dataset).
    # The check is structural: even a section that restates the defaults
    # (max_bin = 255) is an explicit override and must be rejected.
    for section in ("[bin_mapper]\nmax_bin = 63\n", "[bin_mapper]\nmax_bin = 255\n"):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write(section)
            bad_cfg = f.name
        with pytest.raises(Exception) as e:
            bonsai.train([], ds, config=bad_cfg)
            assert "bin_mapper" in str(e.value)

    # a config file with only non-bin params must NOT false-positive, even when
    # the Dataset itself was binned with a non-default max_bin
    ds63 = bonsai.Dataset(X, y, max_bin=63, min_data_in_bin=3)
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write("[tree]\nmax_depth = 4\n")
        ok_cfg = f.name
    assert np.asarray(bonsai.train([], ds63, config=ok_cfg).predict(X)).shape == (3000,)


def test_dataset_bin_edges_carries_domain_bands_in_the_artifact():
    """Doc 18: explicit bin_edges travel inside the model artifact, so predict
    on raw values respects the domain bands with no external transform — the
    property pre-binning (the decision-67 emulation) structurally cannot have."""
    rng = np.random.default_rng(7)
    n = 5000
    age = rng.uniform(0.0, 100.0, n).astype(np.float32)
    noise = rng.random((n, 2), dtype=np.float32)
    X = np.column_stack([age, noise]).astype(np.float32)
    band = np.digitize(age, [18.0, 65.0]).astype(np.float32)
    y = (band * 2.0 + rng.normal(0, 0.05, n)).astype(np.float32)

    ds = bonsai.Dataset(X, y, bin_edges={0: np.array([18.0, 65.0], dtype=np.float32)})
    m = bonsai.train([("booster.n_iters", "30"), ("tree.max_depth", "4")], ds)

    # Within a band the model cannot distinguish raw values: the only cuts on
    # feature 0 are the two domain edges.
    probe = np.array([[5.0, 0.5, 0.5], [17.9, 0.5, 0.5],
                      [18.1, 0.5, 0.5], [64.0, 0.5, 0.5],
                      [66.0, 0.5, 0.5], [99.0, 0.5, 0.5]], dtype=np.float32)
    p = np.asarray(m.predict(probe))
    assert p[0] == p[1] and p[2] == p[3] and p[4] == p[5]
    # ...and across bands it must distinguish: the bands are the signal.
    assert p[0] < p[2] < p[4]

    # Edges are right-inclusive, the fitted-cut convention: 18.0 is minor.
    edge = np.asarray(m.predict(np.array([[18.0, 0.5, 0.5]], dtype=np.float32)))
    assert edge[0] == p[0]

    # The artifact round-trips: a reloaded model predicts byte-identically on
    # raw values with no external transform.
    with tempfile.NamedTemporaryFile(suffix=".bonsai", delete=False) as f:
        model_path = f.name
    m.save(model_path)
    np.testing.assert_array_equal(np.asarray(bonsai.load(model_path).predict(probe)), p)

    # Malformed edges are rejected at Dataset construction.
    for bad in ({0: np.array([65.0, 18.0], dtype=np.float32)},   # decreasing
                {0: np.array([], dtype=np.float32)},             # empty
                {9: np.array([1.0], dtype=np.float32)}):         # no such column
        with pytest.raises(Exception) as e:
            bonsai.Dataset(X, y, bin_edges=bad)
            assert "bin_edges" in str(e.value)


def test_dataset_eval_set_early_stopping():
    """train(params, dataset, eval_set=...) bins the valid set with the
    Dataset's mappers and enables early stopping — the MVP gap where
    early_stopping params silently did nothing in the Dataset path."""
    rng = np.random.default_rng(0)
    X = rng.random((4000, 10), dtype=np.float32)
    y = (X[:, 0] * 2 + rng.normal(0, 0.3, 4000)).astype(np.float32)
    Xt, yt, Xv, yv = X[:3000], y[:3000], X[3000:], y[3000:]

    ds = bonsai.Dataset(Xt, yt)
    pairs = [("booster.n_iters", "400"), ("booster.learning_rate", "0.3"),
             ("booster.early_stopping_rounds", "10")]
    m = bonsai.train(pairs, ds, eval_set=(Xv, yv))
    assert m.n_iters < 400, m.n_iters

    # same call without eval_set trains to completion (no valid, no stopping)
    m_full = bonsai.train([("booster.n_iters", "30")], ds)
    assert m_full.n_iters == 30

    # eval_set path must match the (X, y) path bit for bit under equal binning
    ref = bonsai.train(pairs, Xt, yt, eval_set=(Xv, yv))
    np.testing.assert_array_equal(
        np.asarray(ref.predict(Xv)), np.asarray(m.predict(Xv))
    )
