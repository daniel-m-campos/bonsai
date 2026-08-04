"""Per-library runners and the scaling worker (child-process side).

Param mappings come from bonsai.bench.params only (a hand-derived knob has
produced a false conclusion twice; see params.py). The worker records the
reference-library versions it actually imported, so rows carry them.

A cell may carry an `eval_mode`, which selects one of three fit shapes and
is what the early-stopping suite sweeps:

    off   carve the eval fraction off the train side and discard it
    eval  carve it and hand it to the library, with no patience armed
    stop  carve it, hand it over, and arm the library's patience

`off` and `eval` differ only in whether the library sees the eval set, so
their fit_s ratio is the per-round eval overhead; `stop` measures the wall
clock to a stopped model. A cell with no `eval_mode` takes the legacy path
untouched: no carve, no eval set, no extra row fields.
"""

from __future__ import annotations

import os
import pathlib
import time

import numpy as np

from bonsai.bench import params as rp
from bonsai.bench import runlog
from bonsai.bench.metrics import auc, r2
from bonsai.bench.synth import gen_data
from bonsai.bench.variants import Device, Lib, resolve

# Row field and cell knob for the eval-mode discriminator, and the modes it
# admits. Committed rows carry these spellings.
EVAL_MODE = "eval_mode"
STOPPED_AT = "stopped_at"
EVAL_MODES = ("off", "eval", "stop")


# Public Functions =================================================================================

def eval_split(cell: dict, X, y):
    """Split the train side into a fit part and the held-out eval part.

    The eval rows come off the TRAIN side, never the test split, so the
    reported metric still lands on data no library saw. gen_data's rows are
    i.i.d., so a contiguous tail slice is a valid random holdout and, unlike
    fancy indexing, keeps the memmapped cache arrays as views.

    Parameters
    ----------
    cell : dict
        The cell; reads `eval_mode` and the optional `eval_frac`.
    X, y : array
        The full train side, as the worker generated it.

    Returns
    -------
    tuple
        (X_fit, y_fit, X_eval, y_eval). The eval pair is None both for a
        legacy cell (no eval_mode) and for the "off" arm, which must carve
        the same rows away but must never hand them to a library.

    Raises
    ------
    ValueError
        On an eval_mode outside EVAL_MODES.
    """
    mode = cell.get(EVAL_MODE)
    if mode is None:
        return X, y, None, None
    if mode not in EVAL_MODES:
        raise ValueError(f"unknown eval_mode {mode!r} (known: {EVAL_MODES})")
    cut = len(y) - eval_rows(cell, len(y))
    if mode == "off":
        return X[:cut], y[:cut], None, None
    return X[:cut], y[:cut], X[cut:], y[cut:]


def eval_rows(cell: dict, n_train: int) -> int:
    """How many rows the eval fraction carves off a train side of n_train."""
    if cell.get(EVAL_MODE) is None:
        return 0
    frac = cell.get("eval_frac", rp.EARLY_STOP["eval_frac"])
    return max(1, int(n_train * frac))


def patience_of(cell: dict) -> int:
    """The cell's early-stopping patience; 0 unless eval_mode is "stop".

    The "eval" arm exists to price per-round eval at a fixed iteration
    count, so it must never stop: returning 0 omits the mechanism entirely
    rather than setting a patience above the cap and trusting it not to fire.
    """
    if cell.get(EVAL_MODE) != "stop":
        return 0
    return int(cell.get("patience", rp.EARLY_STOP["patience"]))


def run_bonsai(spec, X, y, Xte, yte) -> dict:
    """Fit/predict one cell with bonsai's two-step Dataset + train form.

    The Dataset carries a device hint, "cuda" for a cuda grower and the
    host default otherwise, so binning happens exactly where the fused
    train(pairs, X, y) call would have put it and the ingest/train split
    describes the measured path rather than moving it. Without the hint a
    prebuilt Dataset bins on the host whatever grower follows, which is a
    pipeline no cuda arm runs. bin_mapper.* pairs are stripped because the
    Dataset fixes binning and train(pairs, dataset) rejects them.

    Every call builds its own Dataset: reusing one across repeats or
    variants amortizes ingest, and each row is charged its full ingest.
    fit_s stays the outer wall clock over both steps and is never
    redefined as their sum.

    A spec carrying fused=True takes the one-call form instead and
    reports no breakdown; that arm exists for the parity check in
    scripts/standings_ab.py, which prices the two forms against each
    other on a GPU host.

    An eval-mode cell passes eval_set=(Xv, yv) to train, which is what
    enables per-iter eval, and booster.early_stopping_rounds is what arms
    the stop (module docstring in src/python/module.cpp; docs/use/
    parameters.md). bonsai bins the eval set inside train, with the
    Dataset's own mappers, so that one-time cost lands in train_s where the
    reference libraries' lands in ingest_s; fit_s, the outer wall clock, is
    the fencepost-free number the overhead ratio is quoted on. On a stop the
    model is truncated to its best iteration, so n_iters is both the
    retained round count and the model the reported metric comes from.
    """
    import bonsai
    grower = spec[runlog.Row.VARIANT].removeprefix("bonsai_")
    if grower.startswith("cuda") and not bonsai.cuda_available():
        raise RuntimeError("unsupported: cuda grower without a CUDA device/build")
    c = spec[runlog.Row.CELL]
    task = c.get("task", "reg")
    threads = spec[runlog.Row.THREADS]
    X, y, Xev, yev = eval_split(c, X, y)
    rounds = patience_of(c)
    ev = (Xev, yev) if Xev is not None else None
    pairs = rp.bonsai_core(
        learning_rate=c["lr"], max_depth=c["depth"],
        num_leaves=rp.num_leaves_of(c),
        min_data_in_leaf=c.get("min_data_in_leaf", rp.SCALING["min_data_in_leaf"]),
        lambda_l2=c.get("lambda_l2", rp.SCALING["lambda_l2"]),
        max_bin=c["bins"], seed=c["seed"],
        n_iters=c["iters"], n_threads=threads, grower=grower,
        objective="logloss" if task == "binary" else "mse",
        early_stopping_rounds=rounds)
    if spec.get("fused"):
        fit_t0 = time.perf_counter()
        model = bonsai.train(pairs, X, y, eval_set=ev)
        fit_s, ingest_s, train_s = time.perf_counter() - fit_t0, None, None
    else:
        fit_t0 = t0 = time.perf_counter()
        ds = bonsai.Dataset(X, y, max_bin=c["bins"], n_threads=threads,
                            device="cuda" if grower.startswith("cuda") else "cpu")
        ingest_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        model = bonsai.train([p for p in pairs
                              if not p[0].startswith("bin_mapper.")], ds,
                             eval_set=ev)
        train_s = time.perf_counter() - t0
        fit_s = time.perf_counter() - fit_t0
    t0 = time.perf_counter()
    pred_te = np.asarray(model.predict(Xte))
    predict_s = time.perf_counter() - t0
    return _score(task, fit_s, predict_s, y, yte, pred_te,
                  lambda: np.asarray(model.predict(X)), ingest_s, train_s,
                  cell=c, stopped_at=model.n_iters if rounds else None)


def run_xgb(spec, X, y, Xte, yte) -> dict:
    """Fit/predict one cell with xgboost (QuantileDMatrix + train).

    An eval-mode cell adds evals=[(dval, "val")], and rounds arms
    early_stopping_rounds, both xgb.train keyword arguments (verified
    against xgboost 3.3). The validation matrix is a QuantileDMatrix built
    with ref=dtrain, the documented way to reuse the training quantiles, and
    it must repeat max_bin: xgboost 3.3 raises "Inconsistent `max_bin`"
    otherwise, because ref supplies the cuts but not the parameter.

    xgboost is the one library that does NOT truncate on a stop: the booster
    keeps the trailing patience rounds, and inplace_predict defaults to all
    of them. The predict calls therefore pass iteration_range explicitly, or
    the reported metric would be the overshot model's, not the stopped one's.
    """
    import xgboost as xgb
    c = spec[runlog.Row.CELL]
    task = c.get("task", "reg")
    device = resolve(spec[runlog.Row.VARIANT]).device
    X, y, Xev, yev = eval_split(c, X, y)
    rounds = patience_of(c)
    params = {**rp.xgb_core(learning_rate=c["lr"], max_depth=c["depth"],
                            min_data_in_leaf=c.get(
                                "min_data_in_leaf",
                                rp.SCALING["min_data_in_leaf"]),
                            lambda_l2=c.get("lambda_l2",
                                            rp.SCALING["lambda_l2"]),
                            max_bin=c["bins_effective"], seed=c["seed"]),
              "objective": ("binary:logistic" if task == "binary"
                            else "reg:squarederror"),
              "device": device, "nthread": spec[runlog.Row.THREADS]}
    fit_t0 = t0 = time.perf_counter()
    dtrain = xgb.QuantileDMatrix(X, label=y, max_bin=c["bins_effective"])
    fit_kwargs = {}
    if Xev is not None:
        dval = xgb.QuantileDMatrix(Xev, label=yev, ref=dtrain,
                                   max_bin=c["bins_effective"])
        fit_kwargs = {"evals": [(dval, "val")], "verbose_eval": False}
    ingest_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    booster = xgb.train(params, dtrain, num_boost_round=c["iters"],
                        **fit_kwargs, **rp.xgb_early_stop(rounds))
    train_s = time.perf_counter() - t0
    fit_s = time.perf_counter() - fit_t0
    # (0, 0) is xgboost's "every tree"; a stop needs the explicit range.
    span = (0, booster.best_iteration + 1) if rounds else (0, 0)
    # inplace_predict on a host array against a GPU booster may warn and
    # route through a slower device path; kept intentionally, since fit_s
    # and r2 are unaffected and a device-consistent path needs a new
    # dependency this bench does not otherwise carry.
    t0 = time.perf_counter()
    pred_te = booster.inplace_predict(Xte, iteration_range=span)
    predict_s = time.perf_counter() - t0
    return _score(task, fit_s, predict_s, y, yte, pred_te,
                  lambda: booster.inplace_predict(X, iteration_range=span),
                  ingest_s, train_s, cell=c,
                  stopped_at=booster.best_iteration + 1 if rounds else None)


def run_lgbm(spec, X, y, Xte, yte) -> dict:
    """Fit/predict one cell with lightgbm at the shared knobs.

    An eval-mode cell adds valid_sets=[dvalid], and rounds arms the
    documented lgb.early_stopping(...) callback (verified against lightgbm
    4.6). The validation Dataset is built with reference=dtrain so it reuses
    the training bin boundaries instead of fitting its own.

    lightgbm truncates on a stop, so best_iteration is the retained round
    count and predict() defaults to it; the predict calls stay unqualified.
    """
    import lightgbm as lgb
    c = spec[runlog.Row.CELL]
    task = c.get("task", "reg")
    device = resolve(spec[runlog.Row.VARIANT]).device
    X, y, Xev, yev = eval_split(c, X, y)
    rounds = patience_of(c)
    params = {**rp.lgbm_core(learning_rate=c["lr"], max_depth=c["depth"],
                             num_leaves=rp.num_leaves_of(c),
                             min_data_in_leaf=c.get(
                                 "min_data_in_leaf",
                                 rp.SCALING["min_data_in_leaf"]),
                             lambda_l2=c.get("lambda_l2",
                                             rp.SCALING["lambda_l2"]),
                             max_bin=c["bins_effective"], seed=c["seed"]),
              "objective": "binary" if task == "binary" else "regression",
              "device_type": device, "num_threads": spec[runlog.Row.THREADS]}
    fit_t0 = t0 = time.perf_counter()
    # lgb.Dataset is lazy; construct() forces the binning pass now so its
    # cost lands in ingest_s rather than leaking into the train() call. The
    # params must be present AT construction: binning knobs (max_bin) are
    # frozen then, and a later train() cannot change them.
    dtrain = lgb.Dataset(X, label=y, params=params).construct()
    fit_kwargs = {}
    if Xev is not None:
        fit_kwargs = {"valid_sets": [lgb.Dataset(Xev, label=yev,
                                                 reference=dtrain,
                                                 params=params).construct()]}
    stop_kwargs = rp.lgbm_early_stop(rounds)
    if stop_kwargs:
        fit_kwargs["callbacks"] = [lgb.early_stopping(**stop_kwargs)]
    ingest_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    model = lgb.train(params, dtrain, num_boost_round=c["iters"], **fit_kwargs)
    train_s = time.perf_counter() - t0
    fit_s = time.perf_counter() - fit_t0
    t0 = time.perf_counter()
    pred_te = model.predict(Xte)
    predict_s = time.perf_counter() - t0
    return _score(task, fit_s, predict_s, y, yte, pred_te,
                  lambda: model.predict(X), ingest_s, train_s, cell=c,
                  stopped_at=model.best_iteration if rounds else None)


def run_catboost(spec, X, y, Xte, yte) -> dict:
    """Fit/predict one cell with catboost at the shared knobs.

    An eval-mode cell passes eval_set to fit; the patience translation and
    CatBoost's shrink-on-eval-set behaviour both live in
    params.catboost_early_stop. On a stop the model is shrunk to its best
    iteration, so tree_count_ is the retained round count and predict()
    already uses it.
    """
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    c = spec[runlog.Row.CELL]
    task = c.get("task", "reg")
    device = resolve(spec[runlog.Row.VARIANT]).device
    X, y, Xev, yev = eval_split(c, X, y)
    rounds = patience_of(c)
    cls = CatBoostClassifier if task == "binary" else CatBoostRegressor
    model = cls(
        **rp.catboost_core(learning_rate=c["lr"], max_depth=c["depth"],
                           lambda_l2=c.get("lambda_l2",
                                           rp.SCALING["lambda_l2"]),
                           max_bin=c["bins_effective"], seed=c["seed"],
                           device=device),
        **rp.catboost_early_stop(rounds, has_eval_set=Xev is not None),
        iterations=c["iters"],
        loss_function="Logloss" if task == "binary" else "RMSE",
        task_type=("GPU" if device == Device.CUDA else "CPU"), devices="0",
        thread_count=spec[runlog.Row.THREADS], verbose=False)
    fit_t0 = t0 = time.perf_counter()
    # Pool() only wraps the raw arrays; CatBoost quantizes at fit() time, so
    # this ingest_s underestimates relative to the other libraries' eager
    # construction. That asymmetry is a finding, not a bug (issue #253).
    pool = Pool(X, label=y)
    eval_pool = Pool(Xev, label=yev) if Xev is not None else None
    ingest_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.fit(pool, eval_set=eval_pool)
    train_s = time.perf_counter() - t0
    fit_s = time.perf_counter() - fit_t0
    t0 = time.perf_counter()
    pred_te = (model.predict_proba(Xte)[:, 1] if task == "binary"
               else model.predict(Xte))
    predict_s = time.perf_counter() - t0
    return _score(task, fit_s, predict_s, y, yte, pred_te,
                  lambda: model.predict(X), ingest_s, train_s, cell=c,
                  stopped_at=model.tree_count_ if rounds else None)


RUNNERS = {Lib.BONSAI: run_bonsai, Lib.XGB: run_xgb, Lib.LGBM: run_lgbm,
           Lib.CATBOOST: run_catboost}


def cached_gen_data(cell: dict, cache_dir: str):
    """gen_data memoized to .npy files, loaded back as read-only memmaps.

    gen_data is byte-stable in its arguments (guard-tested), so the key is
    the argument tuple; at 2^31-cell campaigns every regeneration is 8GiB of
    avoidable work per (variant, rep). Writes go through os.replace so a
    half-written file never satisfies a later cache hit.
    """
    key = (f"{cell['rows']}x{cell['cols']}-s{cell['seed']}"
           f"-i{cell['informative']}-t{cell['n_test']}")
    root = pathlib.Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = [root / f"{key}-{n}.npy" for n in ("X", "y", "Xte", "yte")]
    if not all(p.exists() for p in paths):
        arrays = gen_data(cell["rows"], cell["cols"], cell["seed"],
                          cell["n_test"], cell["informative"])
        for p, a in zip(paths, arrays):
            tmp = p.with_name(p.name + ".tmp.npy")
            np.save(tmp, a)
            os.replace(tmp, p)
    return tuple(np.load(p, mmap_mode="r") for p in paths)


def worker(spec: dict) -> dict:
    """Child-process entry: generate data, dispatch the runner, add rates.

    Returns the payload dict the parent folds into the row; GPU arms pay
    an untimed micro-fit first so context/JIT cost stays out of fit_s.
    """
    c = spec[runlog.Row.CELL]
    cache_dir = os.environ.get("BONSAI_BENCH_DATA_CACHE")
    if cache_dir:
        X, y, Xte, yte = cached_gen_data(c, cache_dir)
    else:
        X, y, Xte, yte = gen_data(c["rows"], c["cols"], c["seed"], c["n_test"],
                                  c["informative"])
    v = resolve(spec[runlog.Row.VARIANT])
    if v.name.startswith("bonsai_ts_"):
        raise RuntimeError("unsupported: ordered-TS arms run in the airline "
                           "suite only (the encoder lives there)")
    run = RUNNERS[v.lib]
    if v.device == Device.CUDA:
        # Untimed micro-fit absorbs CUDA context creation (and, once per
        # session, the PTX JIT — disk-cached afterwards).
        micro = dict(spec, cell=dict(c, rows=8192, n_test=1024, iters=5))
        run(micro, X[:8192], y[:8192], Xte[:1024], yte[:1024])
    out = run(spec, X, y, Xte, yte)
    out[runlog.Row.PEAK_RSS_GB] = runlog.peak_rss_gb()
    # An eval-mode cell fits fewer rows than it names: the rate is charged
    # against the rows that were actually fit, not the cell's nominal count.
    fit_rows = c["rows"] - eval_rows(c, c["rows"])
    out["fit_rows_per_s"] = (round(fit_rows / out[runlog.Row.FIT_S])
                             if out[runlog.Row.FIT_S] else None)
    out["predict_rows_per_s"] = (round(c["n_test"] / out[runlog.Row.PREDICT_S])
                                 if out[runlog.Row.PREDICT_S] else None)
    for k in (runlog.Row.FIT_S, runlog.Row.INGEST_S, runlog.Row.TRAIN_S,
              runlog.Row.PREDICT_S):
        out[k] = round(out[k], 3) if out[k] is not None else None
    for k in (runlog.Row.R2_TRAIN, runlog.Row.R2_TEST):
        out[k] = round(out[k], 4)
    # The child is where the reference library was imported, so only the
    # child can report its version; the parent folds this into host.libs.
    out["libs"] = runlog.lib_versions()
    return out


# Private Helpers ==================================================================================

def _score(task: str, fit_s: float, predict_s: float, y, yte, pred_te,
           predict_train, ingest_s: float | None, train_s: float | None, *,
           cell: dict | None = None, stopped_at: int | None = None) -> dict:
    """The runner result dict; predict_train runs only for regression, so
    binary tasks never pay a full train-side predict.

    fit_s is the outer wall clock (raw-floats-to-model, unchanged protocol);
    ingest_s/train_s are the inner breakdown and are not summed back into
    it. Every arm reports the split; the pair is None only for a fused
    bonsai arm, which has no seam to report.

    eval_mode and stopped_at appear only for an eval-mode cell, so a legacy
    row's key set is exactly what it was before early stopping existed.
    stopped_at is the RETAINED round count in every library's spelling: the
    model the reported metric came from, which is the one comparable number
    across four different best-iteration conventions.
    """
    base = {runlog.Row.FIT_S: fit_s, runlog.Row.INGEST_S: ingest_s,
            runlog.Row.TRAIN_S: train_s, runlog.Row.PREDICT_S: predict_s}
    mode = (cell or {}).get(EVAL_MODE)
    if mode is not None:
        base[EVAL_MODE] = mode
        base[STOPPED_AT] = stopped_at
    if task == "binary":
        return {**base, runlog.Row.AUC_TEST: auc(yte, pred_te)}
    pred_tr = predict_train()
    return {**base, runlog.Row.R2_TRAIN: r2(y, pred_tr),
            runlog.Row.R2_TEST: r2(yte, pred_te)}
