"""Per-library runners and the scaling worker (child-process side).

Param mappings come from bonsai.bench.params only (a hand-derived knob has
produced a false conclusion twice; see params.py). The worker records the
reference-library versions it actually imported, so rows carry them.
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


def run_bonsai(spec, X, y, Xte, yte) -> dict:
    """Fit/predict one cell with bonsai's fused train(pairs, X, y) call.

    The fused call is the measured path because binning happens inside
    train(), where the grower name and the thread count are known: cuda
    growers bin on the device there, and the binning pass honors
    parallel.n_threads. A prebuilt Dataset knows neither, so timing its
    construction as ingest would measure a pipeline bonsai never runs on
    GPU. bonsai therefore reports no ingest/train breakdown: the measured
    path wins over the breakdown.
    """
    import bonsai
    grower = spec[runlog.Row.VARIANT].removeprefix("bonsai_")
    if grower.startswith("cuda") and not bonsai.cuda_available():
        raise RuntimeError("unsupported: cuda grower without a CUDA device/build")
    c = spec[runlog.Row.CELL]
    task = c.get("task", "reg")
    pairs = rp.bonsai_core(
        learning_rate=c["lr"], max_depth=c["depth"],
        num_leaves=rp.num_leaves_of(c),
        min_data_in_leaf=c.get("min_data_in_leaf", rp.SCALING["min_data_in_leaf"]),
        lambda_l2=c.get("lambda_l2", rp.SCALING["lambda_l2"]),
        max_bin=c["bins"], seed=c["seed"],
        n_iters=c["iters"], n_threads=spec[runlog.Row.THREADS], grower=grower,
        objective="logloss" if task == "binary" else "mse")
    t0 = time.perf_counter()
    model = bonsai.train(pairs, X, y)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_te = np.asarray(model.predict(Xte))
    predict_s = time.perf_counter() - t0
    return _score(task, fit_s, predict_s, y, yte, pred_te,
                  lambda: np.asarray(model.predict(X)), None, None)


def run_xgb(spec, X, y, Xte, yte) -> dict:
    """Fit/predict one cell with xgboost (QuantileDMatrix + train)."""
    import xgboost as xgb
    c = spec[runlog.Row.CELL]
    task = c.get("task", "reg")
    device = resolve(spec[runlog.Row.VARIANT]).device
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
    ingest_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    booster = xgb.train(params, dtrain, num_boost_round=c["iters"])
    train_s = time.perf_counter() - t0
    fit_s = time.perf_counter() - fit_t0
    # inplace_predict on a host array against a GPU booster may warn and
    # route through a slower device path; kept intentionally, since fit_s
    # and r2 are unaffected and a device-consistent path needs a new
    # dependency this bench does not otherwise carry.
    t0 = time.perf_counter()
    pred_te = booster.inplace_predict(Xte)
    predict_s = time.perf_counter() - t0
    return _score(task, fit_s, predict_s, y, yte, pred_te,
                  lambda: booster.inplace_predict(X), ingest_s, train_s)


def run_lgbm(spec, X, y, Xte, yte) -> dict:
    """Fit/predict one cell with lightgbm at the shared knobs."""
    import lightgbm as lgb
    c = spec[runlog.Row.CELL]
    task = c.get("task", "reg")
    device = resolve(spec[runlog.Row.VARIANT]).device
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
    ingest_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    model = lgb.train(params, dtrain, num_boost_round=c["iters"])
    train_s = time.perf_counter() - t0
    fit_s = time.perf_counter() - fit_t0
    t0 = time.perf_counter()
    pred_te = model.predict(Xte)
    predict_s = time.perf_counter() - t0
    return _score(task, fit_s, predict_s, y, yte, pred_te,
                  lambda: model.predict(X), ingest_s, train_s)


def run_catboost(spec, X, y, Xte, yte) -> dict:
    """Fit/predict one cell with catboost at the shared knobs."""
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    c = spec[runlog.Row.CELL]
    task = c.get("task", "reg")
    device = resolve(spec[runlog.Row.VARIANT]).device
    cls = CatBoostClassifier if task == "binary" else CatBoostRegressor
    model = cls(
        **rp.catboost_core(learning_rate=c["lr"], max_depth=c["depth"],
                           lambda_l2=c.get("lambda_l2",
                                           rp.SCALING["lambda_l2"]),
                           max_bin=c["bins_effective"], seed=c["seed"],
                           device=device),
        iterations=c["iters"],
        loss_function="Logloss" if task == "binary" else "RMSE",
        task_type=("GPU" if device == Device.CUDA else "CPU"), devices="0",
        thread_count=spec[runlog.Row.THREADS], verbose=False)
    fit_t0 = t0 = time.perf_counter()
    # Pool() only wraps the raw arrays; CatBoost quantizes at fit() time, so
    # this ingest_s underestimates relative to the other libraries' eager
    # construction. That asymmetry is a finding, not a bug (issue #253).
    pool = Pool(X, label=y)
    ingest_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.fit(pool)
    train_s = time.perf_counter() - t0
    fit_s = time.perf_counter() - fit_t0
    t0 = time.perf_counter()
    pred_te = (model.predict_proba(Xte)[:, 1] if task == "binary"
               else model.predict(Xte))
    predict_s = time.perf_counter() - t0
    return _score(task, fit_s, predict_s, y, yte, pred_te,
                  lambda: model.predict(X), ingest_s, train_s)


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
    out["fit_rows_per_s"] = (round(c["rows"] / out[runlog.Row.FIT_S])
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


def _score(task: str, fit_s: float, predict_s: float, y, yte, pred_te,
           predict_train, ingest_s: float | None,
           train_s: float | None) -> dict:
    """The runner result dict; predict_train runs only for regression, so
    binary tasks never pay a full train-side predict.

    fit_s is the outer wall clock (raw-floats-to-model, unchanged protocol);
    ingest_s/train_s are the inner breakdown and are not summed back into
    it. Both are None when a library offers no split that leaves the
    measured path intact.
    """
    base = {runlog.Row.FIT_S: fit_s, runlog.Row.INGEST_S: ingest_s,
            runlog.Row.TRAIN_S: train_s, runlog.Row.PREDICT_S: predict_s}
    if task == "binary":
        return {**base, runlog.Row.AUC_TEST: auc(yte, pred_te)}
    pred_tr = predict_train()
    return {**base, runlog.Row.R2_TRAIN: r2(y, pred_tr),
            runlog.Row.R2_TEST: r2(yte, pred_te)}
