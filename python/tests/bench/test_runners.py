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

    Also covers the ingest/train breakdown: a reference runner reports
    ingest_s (data-structure construction) and train_s (the fit call)
    alongside fit_s, which stays the outer wall clock and is not redefined as
    their sum. bonsai reports the pair as None, since every split of its
    fused call moves the measured path.
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
        if variant.startswith("bonsai_"):
            assert out["ingest_s"] is None and out["train_s"] is None, out
            continue
        assert out["ingest_s"] > 0, (variant, out)
        assert out["train_s"] > 0, (variant, out)
        assert out["ingest_s"] + out["train_s"] <= out["fit_s"] + eps, (
            variant, out)


def test_bonsai_runner_never_prebins(monkeypatch):
    """The bonsai runner must fit through the fused train(pairs, X, y) call.

    A prebuilt Dataset bins on the host whatever the grower is, so a cuda
    arm measured that way loses device binning: slower, and the host binned
    matrix stays resident. The guard is structural (Dataset construction is
    fatal here) because the cost is only visible on a GPU, which CI has
    none of.
    """
    import bonsai
    from bonsai.bench import runners

    def refuse(*args, **kwargs):
        raise AssertionError("the bonsai runner must not prebin a Dataset")

    monkeypatch.setattr(bonsai, "Dataset", refuse)
    rng = np.random.default_rng(0)
    X = rng.random((2000, 8), dtype=np.float32)
    y = (X[:, :3].sum(axis=1)).astype(np.float32)
    cell = {"lr": 0.1, "depth": 4, "bins": 63, "seed": 42, "iters": 5}
    out = runners.run_bonsai({"cell": cell, "variant": "bonsai_depthwise",
                              "threads": 1}, X[:1500], y[:1500], X[1500:],
                             y[1500:])
    assert out["fit_s"] > 0 and out["ingest_s"] is None


def test_standings_ab_knobs_match_the_standings_spec():
    """The A/B detector must fit the cell the standings fit.

    The detector restates the anchor knobs (the old arm would otherwise load
    the previous wheel's copy of the spec), so this guard fails the moment
    the standings spec moves and the detector does not. n_test is excluded:
    the A/B never predicts on a full test split.
    """
    import sys

    from bonsai.bench import spec as spec_mod

    sys.path.insert(0, "scripts")
    import standings_ab

    spec = spec_mod.load_spec("standings-rows")
    cell = spec_mod.make_cell(spec["defaults"], rows=16_000_000, cols=100)
    knobs = {k: v for k, v in standings_ab.ANCHOR_KNOBS.items()
             if k != "n_test"}
    assert knobs == {k: cell[k] for k in knobs}
    assert spec["threads"] == [16]


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
