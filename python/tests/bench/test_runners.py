"""Tests for bonsai.bench.runners (worker paths, data cache)."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
from bonsai.bench import synth


def test_data_cache():
    import pathlib

    from bonsai.bench import runners

    cell = {"rows": 3000, "cols": 12, "seed": 7, "n_test": 500,
            "informative": 5}
    direct = synth.gen_data(cell["rows"], cell["cols"], cell["seed"],
                            cell["n_test"], cell["informative"])
    with tempfile.TemporaryDirectory() as td:
        first = runners.cached_gen_data(cell, td)
        files = sorted(p.name for p in pathlib.Path(td).iterdir())
        assert len(files) == 4 and not any(".tmp" in f for f in files)
        second = runners.cached_gen_data(cell, td)
        for d, a, b in zip(direct, first, second):
            assert np.array_equal(d, a) and np.array_equal(d, b)
        assert isinstance(second[0], np.memmap)
        assert second[0].dtype == np.float32


def test_binary_task_runners():
    """The shared runners serve airline's binary suite: task="binary" in the
    cell selects the logloss objective and AUC scoring, and the airline knob
    set carries every field the runners read (the row-shape contract).

    Also covers the ingest/train breakdown (issue #253): every runner now
    reports ingest_s (data-structure construction) and train_s (the fit call)
    alongside fit_s, which stays the outer wall clock and is not redefined as
    their sum.
    """
    from bonsai.bench import airline, runners

    need = {"depth", "iters", "lr", "bins", "seed", "min_data_in_leaf",
            "lambda_l2"}
    assert need <= set(airline.KNOBS)

    rng = np.random.default_rng(0)
    X = rng.random((600, 6), dtype=np.float32)
    yb = (X[:, 0] + 0.2 * rng.random(600) > 0.6).astype(np.float32)
    cell = dict(airline.KNOBS, iters=10, bins_effective=airline.KNOBS["bins"],
                task="binary")
    eps = 1e-3
    for variant in ("bonsai_depthwise", "lgbm_cpu"):
        run = runners.RUNNERS[runners.resolve(variant).lib]
        out = run({"cell": cell, "variant": variant, "threads": 2},
                  X[:500], yb[:500], X[500:], yb[500:])
        assert set(out) == {"fit_s", "ingest_s", "train_s", "predict_s",
                             "auc_test"}, (variant, out)
        assert 0.5 < out["auc_test"] <= 1.0, (variant, out["auc_test"])
        assert out["ingest_s"] > 0, (variant, out)
        assert out["train_s"] > 0, (variant, out)
        assert out["ingest_s"] + out["train_s"] <= out["fit_s"] + eps, (
            variant, out)


def test_bonsai_two_step_matches_monolithic():
    """The two-step Dataset(max_bin=...) + train(pairs, ds) form must predict
    identically to the monolithic train(pairs, X, y) form (issue #253): the
    Dataset constructor's other binning defaults (n_samples, seed,
    min_data_in_bin) must match what the engine used implicitly before."""
    import bonsai
    from bonsai.bench import params as rp

    rng = np.random.default_rng(0)
    X = rng.random((20_000, 20), dtype=np.float32)
    y = (X[:, :5].sum(axis=1) + 0.1 * rng.standard_normal(20_000)).astype(
        np.float32)

    full_pairs = rp.bonsai_core(
        learning_rate=0.1, max_depth=4, num_leaves=rp.num_leaves_full(4),
        min_data_in_leaf=20, lambda_l2=1.0, max_bin=63, seed=42, n_iters=10,
        n_threads=1, grower="depthwise", objective="mse")
    monolithic = bonsai.train(full_pairs, X, y)

    split_pairs = [p for p in full_pairs if not p[0].startswith("bin_mapper.")]
    ds = bonsai.Dataset(X, y, max_bin=63)
    two_step = bonsai.train(split_pairs, ds)

    np.testing.assert_array_equal(monolithic.predict(X), two_step.predict(X))


def test_variant_canonicalization_and_ts_guard():
    from bonsai.bench import runners
    from bonsai.bench import spec as spec_mod

    s = {"name": "t", "cells": [{"rows": 1000, "cols": 8}],
         "variants": ["bonsai_dw", "xgb"]}
    jobs = spec_mod.expand(s)
    assert [j["variant"] for j in jobs] == ["bonsai_depthwise", "xgb_hist"]
    cell = spec_mod.make_cell({}, rows=512, cols=4)
    with pytest.raises(RuntimeError) as e:
        runners.worker({"cell": cell, "variant": "bonsai_ts_depthwise",
                        "threads": 1})
        assert "airline" in str(e.value)
