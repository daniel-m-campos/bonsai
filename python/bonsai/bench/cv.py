"""What a k-fold cross-validation loop costs, per library.

Every suite beside this one measures a single fit. This one measures the
loop: bin once, then fit k times over k row selections out of the same
data. That is a different question, and the libraries answer it very
differently.

The axis is how each library produces a fold's training set from an
already-binned dataset:

===============  ==========================================================
bonsai view      ``ds.subset(rows=tr)``: a row descriptor over the parent's
                 plane. Nothing is copied, so k folds cost one plane.
bonsai copy      ``Dataset(X[tr], y[tr], reference=ds)``: the same fit with
                 the fold materialized, which is what the view is measured
                 against.
lightgbm         ``Dataset.subset(idx)``: reuses the bin mappers but
                 scalar-gathers a fresh bin buffer per fold.
xgboost          ``DMatrix.slice(idx)``: copies the raw matrix AND discards
                 the cuts, so every fold re-quantizes.
xgboost qdm      ``QuantileDMatrix.slice(idx)``: raises. Recorded as a
                 refusal rather than omitted, because "cannot" is the
                 measurement.
catboost         ``Pool.slice(idx)``: range views over shared quantized
                 bins, the only other library that does not copy.
===============  ==========================================================

Three numbers come out. ``loop_s`` is the whole k-fold loop and is what a
tuning run actually waits on; ``fit_loop_s`` is the part of it that builds
each fold and trains on it, which is the part the fold mechanism decides,
and ``score_loop_s`` is the rest. They are split because every arm here
scores from the same raw test matrix but builds its folds differently, so
a sum could hide a fold mechanism winning while its predict path lost.
``peak_gb_pid`` (NVML, per process, sampled by the parent) is what decides
whether k folds fit on the card at all, which at large k is the binding
constraint rather than time.

The shared structure is built once per arm and timed as ``ingest_s``; a
runner that rebuilt it per fold would be measuring ingest k times and
would not be answering this suite's question.
"""

from __future__ import annotations

import time

import numpy as np

from bonsai.bench import params as rp
from bonsai.bench import runlog
from bonsai.bench.metrics import r2
from bonsai.bench.variants import Device, resolve


class Strategy:
    """How a fold's training set is produced. Carried by committed rows."""

    VIEW = "view"
    COPY = "copy"
    QUANTILE = "quantile"


# Which strategies each library actually offers. A cell asking for one a
# library does not have is a refusal row rather than a duplicate of its only
# behaviour, so a copy cell reads as "bonsai alone had a choice here" instead
# of quietly re-running lightgbm's single path under a second label.
_STRATEGIES = {
    "bonsai": (Strategy.VIEW, Strategy.COPY),
    "lgbm": (Strategy.VIEW,),
    "xgb": (Strategy.VIEW, Strategy.QUANTILE),
    "catboost": (Strategy.VIEW,),
}


def folds_of(cell: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """The (train, test) row-id pairs a cell names, as index arrays.

    Two schemes, because they stress opposite ends of the cost model. A
    walk-forward split gives contiguous folds, which is the shape temporal
    cross-validation has and the shape a row descriptor encodes as a
    range. A shuffled k-fold gives scattered ones, which is the worst case
    for every library here and the case a caller fixes by reordering once.
    """
    n, k = cell["rows"], cell["folds"]
    if cell.get("scheme", "walk_forward") == "walk_forward":
        edge = n // (k + 1)
        return [(np.arange(0, edge * (i + 1), dtype=np.int64),
                 np.arange(edge * (i + 1), edge * (i + 2), dtype=np.int64))
                for i in range(k)]
    rng = np.random.default_rng(cell["seed"])
    order = rng.permutation(n)
    blocks = np.array_split(order, k)
    return [(np.sort(np.concatenate(blocks[:i] + blocks[i + 1:])),
             np.sort(blocks[i])) for i in range(k)]


def run_cv(spec: dict, X, y) -> dict:
    """Run one arm's whole fold loop and report the two numbers."""
    v = resolve(spec[runlog.Row.VARIANT])
    cell = spec[runlog.Row.CELL]
    want = cell.get("strategy", Strategy.VIEW)
    if want not in _STRATEGIES[v.lib]:
        raise RuntimeError(f"unsupported: {v.lib} offers no {want!r} fold "
                           f"strategy, only {list(_STRATEGIES[v.lib])}")
    folds = folds_of(cell)
    runner = _RUNNERS[v.lib]
    t0 = time.perf_counter()
    ingest_s, scores, fit_loop_s, score_loop_s = runner(spec, X, y, folds)
    total = time.perf_counter() - t0
    return {
        runlog.Row.INGEST_S: round(ingest_s, 3),
        LOOP_S: round(total - ingest_s, 3),
        FIT_LOOP_S: round(fit_loop_s, 3),
        SCORE_LOOP_S: round(score_loop_s, 3),
        runlog.Row.FIT_S: round(total, 3),
        FOLDS: len(folds),
        runlog.Row.R2_TEST: round(float(np.mean(scores)), 4),
        R2_FOLDS: [round(float(s), 4) for s in scores],
    }


LOOP_S = "loop_s"
FIT_LOOP_S = "fit_loop_s"
SCORE_LOOP_S = "score_loop_s"
FOLDS = "folds"
R2_FOLDS = "r2_folds"


# Per-library runners =============================================================================
#
# Each returns (ingest_s, [r2 per fold], fit_s, score_s). The r2 list is not
# decoration: it is what proves the arms fit the same folds of the same data,
# without which a faster arm might simply be doing less. Fit and score are
# split because they answer different questions and a fold loop that reports
# only their sum can hide one moving inside the other.


def run_bonsai(spec, X, y, folds) -> tuple[float, list[float], float, float]:
    """bonsai, either strategy: one Dataset, then k views or k copies."""
    import bonsai

    cell = spec[runlog.Row.CELL]
    v = resolve(spec[runlog.Row.VARIANT])
    grower = v.name.removeprefix("bonsai_")
    device = "cuda" if grower.startswith("cuda") else "cpu"
    threads = spec[runlog.Row.THREADS]
    pairs = rp.bonsai_core(
        learning_rate=cell["lr"], max_depth=cell["depth"],
        num_leaves=rp.num_leaves_of(cell), **_knobs(cell),
        max_bin=cell["bins"], seed=cell["seed"], n_iters=cell["iters"],
        n_threads=threads, grower=grower, objective="mse",
        early_stopping_rounds=0)
    # The Dataset fixes the binning, and train(pairs, dataset) rejects any
    # bin_mapper.* that would claim otherwise.
    pairs = {k: v for k, v in dict(pairs).items() if not k.startswith("bin_mapper.")}

    t0 = time.perf_counter()
    ds = bonsai.Dataset(X, y, device=device, n_threads=threads)
    ingest_s = time.perf_counter() - t0

    view = cell.get("strategy", Strategy.VIEW) == Strategy.VIEW
    scores, t_fit, t_score = [], 0.0, 0.0
    for tr, te in folds:
        t = time.perf_counter()
        if view:
            train_ds = ds.subset(rows=tr)
        else:
            # The materialized baseline: reference= keeps the cuts, so the
            # only difference from the view arm is that the bins are copied.
            train_ds = bonsai.Dataset(np.ascontiguousarray(X[tr]), y[tr],
                                      reference=ds, device=device,
                                      n_threads=threads)
        model = bonsai.train(pairs, train_ds)
        t_fit += time.perf_counter() - t
        # Both strategies score from the raw test matrix. Scoring the view
        # arm through ds.subset(rows=te) would send it down the host bin
        # walk a device-resident view still falls back to, and the arms
        # would then differ in how they PREDICT rather than in how they
        # build a fold, which is the question this suite asks.
        t = time.perf_counter()
        pred = model.predict(np.ascontiguousarray(X[te]))
        t_score += time.perf_counter() - t
        scores.append(r2(y[te], pred))
    return ingest_s, scores, t_fit, t_score


def run_lgbm(spec, X, y, folds) -> tuple[float, list[float], float, float]:
    """lightgbm: Dataset.subset reuses the mappers, gathers a bin buffer."""
    import lightgbm as lgb

    cell = spec[runlog.Row.CELL]
    # The arm's own device, not a constant: a hard-coded cpu here would put a
    # CPU number in an lgbm_cuda row and nothing downstream could tell.
    p = {**rp.lgbm_core(learning_rate=cell["lr"], max_depth=cell["depth"],
                        num_leaves=rp.num_leaves_of(cell), **_knobs(cell),
                        max_bin=cell["bins_effective"], seed=cell["seed"]),
         "objective": "regression",
         "device_type": resolve(spec[runlog.Row.VARIANT]).device,
         "num_threads": spec[runlog.Row.THREADS], "verbose": -1}
    t0 = time.perf_counter()
    full = lgb.Dataset(X, label=y, free_raw_data=False,
                       params={"max_bin": cell["bins"], "verbose": -1})
    full.construct()
    ingest_s = time.perf_counter() - t0

    scores, t_fit, t_score = [], 0.0, 0.0
    for tr, te in folds:
        t = time.perf_counter()
        sub = full.subset(tr).construct()
        model = lgb.train(p, sub, num_boost_round=cell["iters"])
        t_fit += time.perf_counter() - t
        t = time.perf_counter()
        pred = model.predict(X[te])
        t_score += time.perf_counter() - t
        scores.append(r2(y[te], pred))
    return ingest_s, scores, t_fit, t_score


def run_xgb(spec, X, y, folds) -> tuple[float, list[float], float, float]:
    """xgboost: DMatrix.slice copies the rows and drops the cuts.

    QuantileDMatrix has no slice at all; a cell asking for one records the
    refusal, since being unable to subset without re-ingesting is the
    finding rather than a gap in the measurement.
    """
    import xgboost as xgb

    cell = spec[runlog.Row.CELL]
    device = resolve(spec[runlog.Row.VARIANT]).device
    p = {**rp.xgb_core(learning_rate=cell["lr"], max_depth=cell["depth"],
                       **_knobs(cell), max_bin=cell["bins_effective"],
                       seed=cell["seed"]),
         "objective": "reg:squarederror", "device": device,
         "nthread": spec[runlog.Row.THREADS]}
    quantile = cell.get("strategy") == Strategy.QUANTILE
    t0 = time.perf_counter()
    full = (xgb.QuantileDMatrix(X, label=y, max_bin=cell["bins"]) if quantile
            else xgb.DMatrix(X, label=y))
    ingest_s = time.perf_counter() - t0

    scores, t_fit, t_score = [], 0.0, 0.0
    for tr, te in folds:
        t = time.perf_counter()
        try:
            sub = full.slice(tr)
        except Exception as exc:
            raise RuntimeError(
                f"unsupported: {type(full).__name__}.slice refuses "
                f"({type(exc).__name__}: {exc})") from exc
        model = xgb.train(p, sub, num_boost_round=cell["iters"])
        t_fit += time.perf_counter() - t
        t = time.perf_counter()
        pred = model.predict(xgb.DMatrix(X[te]))
        t_score += time.perf_counter() - t
        scores.append(r2(y[te], pred))
    return ingest_s, scores, t_fit, t_score


def run_catboost(spec, X, y, folds) -> tuple[float, list[float], float, float]:
    """catboost: Pool.slice, the only other library that shares its bins."""
    from catboost import CatBoostRegressor, Pool

    cell = spec[runlog.Row.CELL]
    device = resolve(spec[runlog.Row.VARIANT]).device
    p = dict(rp.catboost_core(learning_rate=cell["lr"], max_depth=cell["depth"],
                              lambda_l2=_knobs(cell)["lambda_l2"],
                              max_bin=cell["bins_effective"], seed=cell["seed"],
                              device=device),
             iterations=cell["iters"], loss_function="RMSE",
             task_type=("GPU" if device == Device.CUDA else "CPU"), devices="0",
             thread_count=spec[runlog.Row.THREADS], verbose=False)
    t0 = time.perf_counter()
    full = Pool(X, label=y)
    ingest_s = time.perf_counter() - t0

    scores, t_fit, t_score = [], 0.0, 0.0
    for tr, te in folds:
        t = time.perf_counter()
        sub = full.slice(tr.tolist())
        model = CatBoostRegressor(**p)
        model.fit(sub, verbose=False)
        t_fit += time.perf_counter() - t
        t = time.perf_counter()
        pred = model.predict(X[te])
        t_score += time.perf_counter() - t
        scores.append(r2(y[te], pred))
    return ingest_s, scores, t_fit, t_score


def _knobs(c: dict) -> dict:
    """The cell's shared tree knobs, defaulted the way every suite does."""
    return {k: c.get(k, rp.SCALING[k]) for k in ("min_data_in_leaf", "lambda_l2")}


_RUNNERS = {
    "bonsai": run_bonsai,
    "lgbm": run_lgbm,
    "xgb": run_xgb,
    "catboost": run_catboost,
}
