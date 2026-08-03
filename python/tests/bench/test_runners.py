"""Tests for bonsai.bench.runners (worker paths, data cache)."""

from __future__ import annotations

import json
import pathlib
import tempfile
import time

import numpy as np
import pytest
from bonsai.bench import synth


def test_data_cache():
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

    Also covers the ingest/train breakdown: every runner reports ingest_s
    (data-structure construction) and train_s (the fit call) alongside
    fit_s, which stays the outer wall clock and is not redefined as their
    sum. bonsai included: its Dataset carries the device hint that keeps
    the two-step form on the same binning path the fused call takes.
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


def test_bonsai_cuda_runner_prebins_with_a_device_hint(monkeypatch):
    """A cuda arm must ingest through a Dataset built with device="cuda".

    An unhinted Dataset bins on the host whatever grower follows it, so a
    cuda arm measured that way reports an ingest number for a pipeline it
    never runs and carries the host binned matrix for the whole fit:
    slower, and heavier. The hint is what makes the ingest/train split
    honest, and it is only visible on a GPU, which CI has none of, so the
    contract is guarded structurally here instead. Both failure modes are
    covered: fitting with no Dataset at all (the fused form), and building
    one without the hint.
    """
    import bonsai
    from bonsai.bench import runners

    class Built(Exception):
        """Raised in place of binning, once the call has been recorded."""

    seen = []

    def spy(X, y, **kwargs):
        seen.append(kwargs)
        raise Built

    monkeypatch.setattr(bonsai, "Dataset", spy)
    monkeypatch.setattr(bonsai, "cuda_available", lambda: True)
    rng = np.random.default_rng(0)
    X = rng.random((2000, 8), dtype=np.float32)
    y = (X[:, :3].sum(axis=1)).astype(np.float32)
    cell = {"lr": 0.1, "depth": 4, "bins": 63, "seed": 42, "iters": 5}
    try:
        # The fit itself is not the subject and cannot run here (no device),
        # so any failure past the recorded construction is discarded; the
        # assertions below read what the runner asked for.
        runners.run_bonsai({"cell": cell, "variant": "bonsai_cuda_depthwise",
                            "threads": 1}, X[:1500], y[:1500], X[1500:],
                           y[1500:])
    except Exception:
        pass
    assert seen, (
        "the bonsai runner fit without building a Dataset: the two-step "
        "Dataset + train(pairs, dataset) form is what reports ingest_s and "
        "train_s, and the fused call has no seam to report")
    assert seen[0].get("device") == "cuda", (
        "the bonsai runner built a Dataset for a cuda arm without "
        'device="cuda": that Dataset bins on the host, so ingest_s would '
        "describe host binning while the grower runs on the device "
        f"(kwargs seen: {seen[0]})")


def test_xgb_runner_never_builds_plain_dmatrix(monkeypatch):
    """The xgboost runner must ingest through QuantileDMatrix, not DMatrix.

    A plain DMatrix defers quantile-sketch construction to the first
    hist/gpu_hist iteration rather than building it at ingest, so a cuda
    arm measured through it loses device binning the same way bonsai's did
    when its runner was moved off its fused call. The guard is structural:
    DMatrix construction is fatal here, and the run must still complete
    without touching it.
    """
    xgb = pytest.importorskip("xgboost")
    from bonsai.bench import runners

    def refuse(*args, **kwargs):
        raise AssertionError(
            "the xgboost runner must ingest via QuantileDMatrix, not "
            "DMatrix: DMatrix does not build the on-device quantile sketch "
            "at ingest, so a cuda arm measured through it silently loses "
            "device binning")

    monkeypatch.setattr(xgb, "DMatrix", refuse)
    rng = np.random.default_rng(0)
    X = rng.random((2000, 8), dtype=np.float32)
    y = (X[:, :3].sum(axis=1)).astype(np.float32)
    cell = {"lr": 0.1, "depth": 4, "bins_effective": 63, "seed": 42, "iters": 5}
    out = runners.run_xgb({"cell": cell, "variant": "xgb_hist", "threads": 1},
                          X[:1500], y[:1500], X[1500:], y[1500:])
    assert out["fit_s"] > 0 and out["ingest_s"] > 0, out


def test_lgbm_runner_freezes_binning_knobs_and_constructs_in_ingest(monkeypatch):
    """LightGBM's binning knobs (max_bin, ...) freeze when Dataset() is
    constructed, not when train() later receives params: passing params
    only to train() silently ships the library's binning defaults instead
    of the shared knob, a real published bug once. The runner must also
    call Dataset.construct() itself, before the train() timer starts, so
    the binning pass lands in ingest_s rather than leaking into train_s
    (lgb.Dataset is otherwise lazy, and train() would construct it for
    free inside the timed call). Both properties are structural: the first
    is fatal if params is missing, the second is checked by planting a
    delay in construct() and confirming it lands in ingest_s, not train_s.
    """
    import lightgbm as lgb
    from bonsai.bench import runners

    real_dataset_cls = lgb.Dataset
    delay = 0.05

    class GuardedDataset(real_dataset_cls):
        def __init__(self, *args, **kwargs):
            if not kwargs.get("params"):
                raise AssertionError(
                    "lgb.Dataset must receive params at construction: "
                    "binning knobs like max_bin freeze then, and a later "
                    "train() call cannot change them (a real published "
                    "bug once)")
            super().__init__(*args, **kwargs)

        def construct(self, *args, **kwargs):
            # construct() is idempotent on an already-built Dataset (train()
            # calls it again internally), so the delay must land only on the
            # real first build, or a runner that skips the explicit call
            # would still show it landing in ingest_s via lgb.train's own
            # redundant call.
            if self._handle is None:
                time.sleep(delay)
            return super().construct(*args, **kwargs)

    monkeypatch.setattr(lgb, "Dataset", GuardedDataset)
    rng = np.random.default_rng(0)
    X = rng.random((2000, 8), dtype=np.float32)
    y = (X[:, :3].sum(axis=1)).astype(np.float32)
    cell = {"lr": 0.1, "depth": 4, "bins_effective": 63, "seed": 42, "iters": 5}
    out = runners.run_lgbm({"cell": cell, "variant": "lgbm_cpu", "threads": 1},
                           X[:1500], y[:1500], X[1500:], y[1500:])
    assert out["ingest_s"] >= delay, (
        "the lgbm runner must call Dataset.construct() itself, so the "
        "binning pass lands in ingest_s")
    assert out["train_s"] < delay, (
        "construct() ran inside the timed train() call instead of ingest_s")


def test_catboost_runner_builds_pool_outside_fit_timer(monkeypatch):
    """CatBoost quantizes at fit() time, not at Pool() construction: Pool()
    only wraps the raw arrays, so ingest_s under-reports relative to the
    other libraries' eager binning, a documented asymmetry rather than an
    equalized one. The Pool must still be built, and its construction
    timed, before the fit() call starts: the guard is structural and
    records call order, failing if fit() is ever invoked before Pool().
    """
    catboost = pytest.importorskip("catboost")
    from bonsai.bench import runners

    order = []
    real_pool = catboost.Pool
    real_fit = catboost.CatBoostRegressor.fit

    def tracking_pool(*args, **kwargs):
        order.append("pool")
        return real_pool(*args, **kwargs)

    def tracking_fit(self, *args, **kwargs):
        order.append("fit")
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(catboost, "Pool", tracking_pool)
    monkeypatch.setattr(catboost.CatBoostRegressor, "fit", tracking_fit)
    rng = np.random.default_rng(0)
    X = rng.random((2000, 8), dtype=np.float32)
    y = (X[:, :3].sum(axis=1)).astype(np.float32)
    cell = {"lr": 0.1, "depth": 4, "bins_effective": 63, "seed": 42, "iters": 5}
    out = runners.run_catboost(
        {"cell": cell, "variant": "catboost_cpu", "threads": 1}, X[:1500],
        y[:1500], X[1500:], y[1500:])
    assert order == ["pool", "fit"], order
    assert out["ingest_s"] > 0 and out["train_s"] > 0, out


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


def test_parity_gate_bands_the_two_step_form():
    """The refresh's parity gate must fail a two-step form that drifted.

    The gate runs on a GPU pod, so its arithmetic is what CI can check:
    an in-band pair passes, an out-of-band one fails and blocks the
    supersession, and an absent or skipped file passes with a caveat so a
    refresh measured without a device still lands.
    """
    import sys

    sys.path.insert(0, "scripts")
    import standings_refresh

    def rows(fused_s, two_step_s):
        return "\n".join(json.dumps(
            {"arm": arm, "rows": 16000000, "cols": 100,
             "grower": "cuda_depthwise", "fit_s": s, "peak_rss_gb": 6.9,
             "ingest_s": None if arm == "fused" else 1.1,
             "train_s": None if arm == "fused" else s - 1.1})
            for arm, s in (("fused", fused_s), ("two_step", two_step_s)))

    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "parity.jsonl"
        assert standings_refresh._parity(path)[1], "an absent file must pass"
        path.write_text('{"arm": "fused", "skipped": "no CUDA"}\n')
        assert standings_refresh._parity(path)[1], "a skipped run must pass"
        path.write_text(rows(12.0, 12.3))
        table, ok = standings_refresh._parity(path)
        assert ok and "PASS" in table, table
        path.write_text(rows(12.0, 15.0))
        table, ok = standings_refresh._parity(path)
        assert not ok and "FAIL" in table, table


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
