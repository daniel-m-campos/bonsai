"""Tests for bonsai.Dataset (the reusable pre-binned dataset)."""

from __future__ import annotations

import pickle
import tempfile

import bonsai
import numpy as np
import pytest
from bonsai import _bonsai


def _reg_data(n=4000, f=12, seed=0):
    """(X, y) float32 arrays with signal in the first two columns."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, f), dtype=np.float32)
    y = (X[:, 0] * 2 + X[:, 1] + rng.normal(0, 0.1, n)).astype(np.float32)
    return X, y


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


def test_dataset_device_hint_rejects_unknown_and_absent_devices():
    """A device hint is an explicit request, so an absent backend or device is
    an error — unlike the engine's own inference from a grower name, which
    degrades to the host silently."""
    X, y = _reg_data(n=500)
    assert bonsai.Dataset(X, y).device == "cpu"
    with pytest.raises(Exception, match='"cpu" or "cuda"'):
        bonsai.Dataset(X, y, device="gpu")
    if not bonsai.cuda_available():
        with pytest.raises(Exception, match="cuda_available"):
            bonsai.Dataset(X, y, device="cuda")


def test_dataset_is_not_picklable():
    """Binned columns (device memory under device="cuda") travel in no
    artifact; the failure must say so rather than write a broken one."""
    X, y = _reg_data(n=500)
    with pytest.raises(Exception, match="not picklable"):
        pickle.dumps(bonsai.Dataset(X, y))


def test_dataset_honors_n_threads():
    """Construction runs the binning pass under its own n_threads, the way a
    fit runs under parallel.n_threads."""
    X, y = _reg_data(n=2000)
    bonsai.Dataset(X, y, n_threads=3)
    assert _bonsai._n_threads() == 3
    bonsai.Dataset(X, y, n_threads=1)
    assert _bonsai._n_threads() == 1

    # binning is thread-count invariant, so the knob must not move the bits
    pairs = [("dispatch.grower_name", "depthwise"), ("booster.n_iters", "20")]
    one = bonsai.train(pairs, bonsai.Dataset(X, y, n_threads=1)).predict(X)
    many = bonsai.train(pairs, bonsai.Dataset(X, y, n_threads=4)).predict(X)
    np.testing.assert_array_equal(np.asarray(one), np.asarray(many))


def test_device_dataset_matches_the_fused_path():
    """A device-binned Dataset reaches the same device ingest the fused
    train(pairs, X, y) call takes; GPU histogram atomics make the comparison
    tolerance-equal, not bit-equal (docs/architecture/11)."""
    if not bonsai.cuda_available():
        pytest.skip("no CUDA build or no visible device")
    X, y = _reg_data(n=20000)
    pairs = [("dispatch.grower_name", "cuda_depthwise"), ("booster.n_iters", "30"),
             ("tree.max_depth", "6")]
    fused = np.asarray(bonsai.train(pairs, X, y).predict(X))

    ds = bonsai.Dataset(X, y, device="cuda")
    assert ds.device == "cuda"
    two_step = np.asarray(bonsai.train(pairs, ds).predict(X))
    np.testing.assert_allclose(fused, two_step, rtol=0, atol=1e-4)

    with pytest.raises(Exception, match="not picklable"):
        pickle.dumps(ds)


def test_device_dataset_materializes_host_bins_for_a_cpu_grower():
    """The mismatch policy is lazy materialization: the device plane fills
    host columns on first host consumer, bit-identical to the host fill, so a
    CPU fit from a device-binned Dataset equals one from a host Dataset."""
    if not bonsai.cuda_available():
        pytest.skip("no CUDA build or no visible device")
    X, y = _reg_data(n=20000)
    pairs = [("dispatch.grower_name", "depthwise"), ("booster.n_iters", "30"),
             ("tree.max_depth", "6")]
    host = np.asarray(bonsai.train(pairs, bonsai.Dataset(X, y)).predict(X))
    device = np.asarray(
        bonsai.train(pairs, bonsai.Dataset(X, y, device="cuda")).predict(X)
    )
    np.testing.assert_array_equal(host, device)


def test_device_dataset_rejects_a_device_id_mismatch():
    """The resident matrix cannot follow the fit to another device, so the
    disagreement is an error rather than a silent migration."""
    if not bonsai.cuda_available():
        pytest.skip("no CUDA build or no visible device")
    X, y = _reg_data(n=2000)
    ds = bonsai.Dataset(X, y, device="cuda", device_id=0)
    with pytest.raises(Exception, match="device_id=1"):
        bonsai.train([("dispatch.grower_name", "cuda_depthwise"),
                      ("parallel.device_id", "1")], ds)
    # a host Dataset carries no residency, so any device placement is fine
    bonsai.train([("booster.n_iters", "2"), ("parallel.device_id", "0")],
                 bonsai.Dataset(X, y))
