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
    set carries every field the runners read (the row-shape contract)."""
    from bonsai.bench import airline, runners

    need = {"depth", "iters", "lr", "bins", "seed", "min_data_in_leaf",
            "lambda_l2"}
    assert need <= set(airline.KNOBS)

    rng = np.random.default_rng(0)
    X = rng.random((600, 6), dtype=np.float32)
    yb = (X[:, 0] + 0.2 * rng.random(600) > 0.6).astype(np.float32)
    cell = dict(airline.KNOBS, iters=10, bins_effective=airline.KNOBS["bins"],
                task="binary")
    for variant in ("bonsai_depthwise", "lgbm_cpu"):
        run = runners.RUNNERS[runners.resolve(variant).lib]
        out = run({"cell": cell, "variant": variant, "threads": 2},
                  X[:500], yb[:500], X[500:], yb[500:])
        assert set(out) == {"fit_s", "predict_s", "auc_test"}, (variant, out)
        assert 0.5 < out["auc_test"] <= 1.0, (variant, out["auc_test"])


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
