"""Tests for bonsai.bench.runners (worker paths, data cache)."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import tempfile
import time
from collections.abc import Callable

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
    """The shared runners serve binary suites: task="binary" in the cell
    selects the logloss objective and AUC scoring, and the SCALING knob set
    carries every field the runners read (the row-shape contract).

    Also covers the ingest/train breakdown: every runner reports ingest_s
    (data-structure construction) and train_s (the fit call) alongside
    fit_s, which stays the outer wall clock and is not redefined as their
    sum. bonsai included: its Dataset carries the device hint that keeps
    the two-step form on the same binning path the fused call takes.
    """
    from bonsai.bench import params, runners

    need = {"depth", "iters", "lr", "bins", "seed", "min_data_in_leaf",
            "lambda_l2"}
    assert need <= set(params.SCALING)

    rng = np.random.default_rng(0)
    X = rng.random((600, 6), dtype=np.float32)
    yb = (X[:, 0] + 0.2 * rng.random(600) > 0.6).astype(np.float32)
    cell = dict(params.SCALING, iters=10,
                bins_effective=params.SCALING["bins"], task="binary")
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


def test_xgb_runner_fails_a_silent_cpu_fallback(monkeypatch):
    """A cuda request that xgboost quietly drops to CPU must fail the row.

    xgboost 3.3 can warn "No visible GPU is found" and keep training on
    CPU instead of raising (issue #333); a row born from that fallback
    would carry a plausible CPU time under a cuda label. The guard reads
    booster.save_config() after the fit, which reflects the device the fit
    actually used, so this forces that field to "cpu" regardless of what
    the host under test actually has.
    """
    xgb = pytest.importorskip("xgboost")
    from bonsai.bench import runners

    real_train = xgb.train

    def fell_back_to_cpu(params, dtrain, *args, **kwargs):
        booster = real_train(params, dtrain, *args, **kwargs)
        cfg = json.loads(booster.save_config())
        cfg["learner"]["generic_param"]["device"] = "cpu"
        booster.save_config = lambda: json.dumps(cfg)
        return booster

    monkeypatch.setattr(xgb, "train", fell_back_to_cpu)
    rng = np.random.default_rng(0)
    X = rng.random((1500, 8), dtype=np.float32)
    y = X[:, :3].sum(axis=1).astype(np.float32)
    cell = {"lr": 0.1, "depth": 4, "bins_effective": 63, "seed": 42, "iters": 5}
    with pytest.raises(RuntimeError, match="fell back to CPU"):
        runners.run_xgb({"cell": cell, "variant": "xgb_cuda", "threads": 1},
                        X[:1000], y[:1000], X[1000:], y[1000:])


def test_xgb_runner_accepts_a_genuine_gpu_placement(monkeypatch):
    """The placement guard must not fire on a booster that reports the GPU.

    Companion to the fallback test above: forces save_config() to report a
    cuda device instead, so the guard is proven to discriminate rather than
    always raising for a cuda-labeled arm.
    """
    xgb = pytest.importorskip("xgboost")
    from bonsai.bench import runners

    real_train = xgb.train

    def landed_on_gpu(params, dtrain, *args, **kwargs):
        booster = real_train(params, dtrain, *args, **kwargs)
        cfg = json.loads(booster.save_config())
        cfg["learner"]["generic_param"]["device"] = "cuda:0"
        booster.save_config = lambda: json.dumps(cfg)
        return booster

    monkeypatch.setattr(xgb, "train", landed_on_gpu)
    rng = np.random.default_rng(0)
    X = rng.random((1500, 8), dtype=np.float32)
    y = X[:, :3].sum(axis=1).astype(np.float32)
    cell = {"lr": 0.1, "depth": 4, "bins_effective": 63, "seed": 42, "iters": 5}
    out = runners.run_xgb({"cell": cell, "variant": "xgb_cuda", "threads": 1},
                          X[:1000], y[:1000], X[1000:], y[1000:])
    assert out["fit_s"] > 0, out


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


def test_eval_split_carves_the_train_side_only():
    """The held-out eval rows come off the train side, never the test split.

    Carving from the test side would score the final metric on rows the
    library tuned its stop against. The "off" arm carves the same rows and
    hands back no eval pair at all: it is the denominator of the overhead
    ratio, so it must fit the same row count as the "eval" arm while having
    nothing an eval set could be built from.
    """
    from bonsai.bench import runners

    rng = np.random.default_rng(0)
    X = rng.random((1000, 4), dtype=np.float32)
    y = X[:, 0].copy()

    plain = runners.eval_split({}, X, y)
    assert plain[0].shape[0] == 1000 and plain[2] is None

    off = runners.eval_split({"eval_mode": "off"}, X, y)
    ev = runners.eval_split({"eval_mode": "eval"}, X, y)
    assert off[0].shape[0] == ev[0].shape[0] == 900
    assert off[2] is None and off[3] is None
    assert ev[2].shape[0] == 100 and ev[3].shape[0] == 100
    assert np.array_equal(ev[2], X[900:])

    assert runners.patience_of({"eval_mode": "eval"}) == 0
    assert runners.patience_of({"eval_mode": "stop"}) == 50
    assert runners.patience_of({"eval_mode": "stop", "patience": 7}) == 7
    with pytest.raises(ValueError, match="eval_mode"):
        runners.eval_split({"eval_mode": "sometimes"}, X, y)


# Early stopping ===================================================================================
#
# Every library arms early stopping through a different surface, and none of
# the four is visible in a timing number, so what the runner asked the
# library for is what these guards read. One probe per library installs the
# spies and says how to read that surface back; the arms themselves (off,
# eval, stop) are the same three for all four, and are swept once below.


@dataclasses.dataclass
class _StopProbe:
    """One library's early-stopping surface, as seen from outside the runner.

    ``patience`` returns the patience the library was handed, in that
    library's own spelling, or None when the arm never armed one.
    """

    run: Callable
    variant: str
    base: dict
    reset: Callable
    saw_eval_set: Callable
    patience: Callable
    armed_patience: object = 5
    tail: Callable | None = None


_REF_BASE = {"lr": 0.1, "depth": 4, "bins_effective": 63, "seed": 42,
             "iters": 20}


def _bonsai_stop_probe(monkeypatch) -> _StopProbe:
    """bonsai's surface is eval_set= on train plus the patience config key.

    Both halves are load-bearing: an eval_set with no patience runs the full
    cap, and a patience with no eval_set has nothing to watch.
    """
    import bonsai
    from bonsai.bench import runners

    seen = []
    real_train = bonsai.train

    def spy(pairs, data, *args, **kwargs):
        seen.append((dict(pairs), kwargs.get("eval_set")))
        return real_train(pairs, data, *args, **kwargs)

    monkeypatch.setattr(bonsai, "train", spy)
    key = "booster.early_stopping_rounds"
    return _StopProbe(
        run=runners.run_bonsai, variant="bonsai_depthwise",
        base={"lr": 0.1, "depth": 4, "bins": 63, "seed": 42, "iters": 20},
        reset=seen.clear,
        saw_eval_set=lambda: seen[0][1] is not None,
        patience=lambda: seen[0][0].get(key),
        armed_patience="5")


def _xgb_stop_probe(monkeypatch) -> _StopProbe:
    """xgboost takes evals and early_stopping_rounds at the xgb.train call.

    The validation matrix must be a QuantileDMatrix built with ref=dtrain
    AND the same max_bin: xgboost 3.3 rejects a ref matrix whose max_bin
    disagrees, and the failure is a fit-time exception on a rented pod. The
    tail check is what pins the matrix type; xgboost is also the one library
    that does not truncate on a stop, so the runner's own iteration_range is
    what keeps the metric off the overshot model.
    """
    xgb = pytest.importorskip("xgboost")
    from bonsai.bench import runners

    calls = []
    real_train = xgb.train

    def spy(params, dtrain, *args, **kwargs):
        calls.append(kwargs)
        return real_train(params, dtrain, *args, **kwargs)

    monkeypatch.setattr(xgb, "train", spy)
    return _StopProbe(
        run=runners.run_xgb, variant="xgb_hist", base=_REF_BASE,
        reset=calls.clear,
        saw_eval_set=lambda: "evals" in calls[0],
        patience=lambda: calls[0].get("early_stopping_rounds"),
        tail=lambda: isinstance(calls[0]["evals"][0][0], xgb.QuantileDMatrix))


def _lgbm_stop_probe(monkeypatch) -> _StopProbe:
    """lightgbm's documented mechanism is the lgb.early_stopping callback.

    Setting a params key instead would ride on an alias the library is free
    to retire, so the callback construction is what this pins, along with
    valid_sets appearing exactly when the arm has an eval set.
    """
    lgb = pytest.importorskip("lightgbm")
    from bonsai.bench import runners

    rounds, calls = [], []
    real_es, real_train = lgb.early_stopping, lgb.train

    def spy_es(*args, **kwargs):
        rounds.append(kwargs.get("stopping_rounds", args[0] if args else None))
        return real_es(*args, **kwargs)

    def spy_train(params, dtrain, *args, **kwargs):
        calls.append(kwargs)
        return real_train(params, dtrain, *args, **kwargs)

    def reset():
        rounds.clear()
        calls.clear()

    monkeypatch.setattr(lgb, "early_stopping", spy_es)
    monkeypatch.setattr(lgb, "train", spy_train)
    return _StopProbe(
        run=runners.run_lgbm, variant="lgbm_cpu", base=_REF_BASE, reset=reset,
        saw_eval_set=lambda: "valid_sets" in calls[0],
        patience=lambda: rounds[0] if rounds else None)


def _catboost_stop_probe(monkeypatch) -> _StopProbe:
    """CatBoost's patience is od_type="Iter" plus od_wait on the estimator.

    The tail check reads od_type, and params.catboost_early_stop carries the
    other half: CatBoost shrinks the model to its best iteration whenever an
    eval set is present, detector or not, so without use_best_model=False the
    "eval" arm would report a metric from a model shorter than the one whose
    fit was timed.
    """
    catboost = pytest.importorskip("catboost")
    from bonsai.bench import runners

    built, fits = [], []
    real_init = catboost.CatBoostRegressor.__init__
    real_fit = catboost.CatBoostRegressor.fit

    def spy_init(self, **kwargs):
        built.append(kwargs)
        real_init(self, **kwargs)

    def spy_fit(self, pool, *args, **kwargs):
        fits.append(kwargs)
        return real_fit(self, pool, *args, **kwargs)

    def reset():
        built.clear()
        fits.clear()

    monkeypatch.setattr(catboost.CatBoostRegressor, "__init__", spy_init)
    monkeypatch.setattr(catboost.CatBoostRegressor, "fit", spy_fit)
    return _StopProbe(
        run=runners.run_catboost, variant="catboost_cpu", base=_REF_BASE,
        reset=reset,
        saw_eval_set=lambda: fits[0]["eval_set"] is not None,
        patience=lambda: built[0].get("od_wait"),
        tail=lambda: built[0]["od_type"] == "Iter")


@pytest.mark.parametrize("probe_of", [
    pytest.param(_bonsai_stop_probe, id="bonsai"),
    pytest.param(_xgb_stop_probe, id="xgb"),
    pytest.param(_lgbm_stop_probe, id="lgbm"),
    pytest.param(_catboost_stop_probe, id="catboost"),
])
def test_runners_arm_early_stopping_on_the_stop_arm_only(probe_of, monkeypatch):
    """The eval set reaches the library on every arm but "off", and the
    patience on "stop" alone.

    The "off" arm must reach the fit with no eval set, or the overhead
    denominator is paying for eval too; the "eval" arm must reach it with no
    patience, or it stops before the fixed cap it exists to price.
    """
    p = probe_of(monkeypatch)
    rng = np.random.default_rng(0)
    X = rng.random((2000, 8), dtype=np.float32)
    y = X[:, :3].sum(axis=1).astype(np.float32)

    for mode, armed in (("off", False), ("eval", False), ("stop", True)):
        p.reset()
        cell = dict(p.base, eval_mode=mode, patience=5)
        out = p.run({"cell": cell, "variant": p.variant, "threads": 1},
                    X[:1500], y[:1500], X[1500:], y[1500:])
        assert p.saw_eval_set() == (mode != "off"), mode
        assert (p.patience() is not None) == armed, (mode, p.patience())
        assert out["eval_mode"] == mode
        assert (out["stopped_at"] is not None) == armed, out
    assert p.patience() == p.armed_patience
    assert p.tail is None or p.tail(), p.variant


def test_bonsai_contribs_phase_is_additive():
    """A contribs cell adds contribs_s and contribs_additivity to the row.
    mse's link is the identity, so predict()'s output IS the raw (pre-link)
    score pred_contribs sums to, and the decomposition must be exact up to
    floating-point noise.
    """
    from bonsai.bench import runners

    rng = np.random.default_rng(0)
    X = rng.random((600, 6), dtype=np.float32)
    y = X[:, :3].sum(axis=1).astype(np.float32)
    cell = {"lr": 0.1, "depth": 4, "bins": 63, "seed": 42, "iters": 5,
            "contribs": True}
    out = runners.run_bonsai(
        {"cell": cell, "variant": "bonsai_depthwise", "threads": 1},
        X[:500], y[:500], X[500:], y[500:])
    assert set(out) == {"fit_s", "ingest_s", "train_s", "predict_s",
                        "contribs_s", "contribs_additivity",
                        "r2_train", "r2_test"}, out
    assert out["contribs_s"] > 0, out
    assert out["contribs_additivity"] < 1e-6, out["contribs_additivity"]


def test_bonsai_contribs_off_by_default_carries_no_extra_fields():
    """A cell with no `contribs` key must reproduce the pre-SHAP row exactly
    (the same additive-schema rule eval_mode already follows)."""
    from bonsai.bench import runners

    rng = np.random.default_rng(0)
    X = rng.random((600, 6), dtype=np.float32)
    y = X[:, :3].sum(axis=1).astype(np.float32)
    cell = {"lr": 0.1, "depth": 4, "bins": 63, "seed": 42, "iters": 5}
    out = runners.run_bonsai(
        {"cell": cell, "variant": "bonsai_depthwise", "threads": 1},
        X[:500], y[:500], X[500:], y[500:])
    assert "contribs_s" not in out and "contribs_additivity" not in out


def test_legacy_cells_carry_no_eval_fields():
    """A cell without eval_mode must produce the pre-early-stopping row.

    Every committed perf row predates these fields, and the standings path
    still writes rows that must diff cleanly against them.
    """
    from bonsai.bench import runners

    rng = np.random.default_rng(0)
    X = rng.random((2000, 8), dtype=np.float32)
    y = X[:, :3].sum(axis=1).astype(np.float32)
    cell = {"lr": 0.1, "depth": 4, "bins": 63, "bins_effective": 63,
            "seed": 42, "iters": 5}
    out = runners.run_bonsai(
        {"cell": cell, "variant": "bonsai_depthwise", "threads": 1},
        X[:1500], y[:1500], X[1500:], y[1500:])
    assert set(out) == {"fit_s", "ingest_s", "train_s", "predict_s",
                        "r2_train", "r2_test"}, out


def test_early_stop_spec_expands_both_quantities():
    """The off and eval arms differ only in whether the library sees the eval
    set, which is what makes their ratio the eval overhead, so they must also
    be distinguishable to the resume key or the second arm is skipped as a
    duplicate of the first.
    """
    from bonsai.bench import driver
    from bonsai.bench import spec as spec_mod

    spec = spec_mod.load_spec("gpu-early-stop")
    off, ev, _stop = spec_mod.cells_of(spec)
    assert off["iters"] == ev["iters"] and off["lr"] == ev["lr"]

    jobs = spec_mod.expand(spec)
    keys = {driver._job_key(j, 0, "pod", "gpu-early-stop") for j in jobs}
    assert len(keys) == len(jobs)


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

    spec = spec_mod.load_spec("gpu-tall")
    cell = spec_mod.make_cell(spec["defaults"], rows=16_777_216, cols=128)
    knobs = {k: v for k, v in standings_ab.ANCHOR_KNOBS.items()
             if k != "n_test"}
    assert knobs == {k: cell[k] for k in knobs}
    assert spec["threads"] == [16]


def test_parity_gate_bands_the_two_step_form():
    """The refresh's parity gate must fail a two-step form that drifted.

    The gate runs on a GPU pod, so its arithmetic is what CI can check:
    an in-band pair passes, an out-of-band one fails and blocks the
    supersession, and a skipped file passes with a caveat so a refresh
    measured without a device still lands. An absent file is different:
    it means the check never ran, e.g. a lost scp or a pod that died
    before the parity phase, so it must fail the gate rather than pass
    it. The operator can only override that with the explicit
    ``allow_absent`` escape.
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
        note, ok = standings_refresh._parity(path)
        assert not ok, "an absent file must fail"
        assert "never ran" in note, note
        note, ok = standings_refresh._parity(path, allow_absent=True)
        assert ok, "allow_absent must accept an absent file"
        assert note == "Parity check absent (no parity.jsonl in this results dir)."
        path.write_text('{"arm": "fused", "skipped": "no CUDA"}\n')
        assert standings_refresh._parity(path)[1], "a skipped run must pass"
        path.write_text(rows(12.0, 12.3))
        table, ok = standings_refresh._parity(path)
        assert ok and "PASS" in table, table
        path.write_text(rows(12.0, 15.0))
        table, ok = standings_refresh._parity(path)
        assert not ok and "FAIL" in table, table


def test_variant_canonicalization():
    from bonsai.bench import spec as spec_mod

    s = {"name": "t", "cells": [{"rows": 1000, "cols": 8}],
         "variants": ["bonsai_dw", "xgb"]}
    jobs = spec_mod.expand(s)
    assert [j["variant"] for j in jobs] == ["bonsai_depthwise", "xgb_hist"]
