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

from . import params as rp
from . import runlog
from .metrics import auc, r2
from .synth import gen_data
from .variants import resolve


def run_bonsai(spec, X, y, Xte, yte) -> dict:
    import bonsai
    grower = spec["variant"].removeprefix("bonsai_")
    if grower.startswith("cuda") and not bonsai.cuda_available():
        raise RuntimeError("unsupported: cuda grower without a CUDA device/build")
    c = spec["cell"]
    task = c.get("task", "reg")
    pairs = rp.bonsai_core(
        learning_rate=c["lr"], max_depth=c["depth"],
        num_leaves=rp.num_leaves_full(c["depth"]),
        min_data_in_leaf=c.get("min_data_in_leaf", rp.SCALING["min_data_in_leaf"]),
        lambda_l2=c.get("lambda_l2", rp.SCALING["lambda_l2"]),
        max_bin=c["bins"], seed=c["seed"],
        n_iters=c["iters"], n_threads=spec["threads"], grower=grower,
        objective="logloss" if task == "binary" else "mse")
    t0 = time.perf_counter()
    model = bonsai.train(pairs, X, y)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_te = np.asarray(model.predict(Xte))
    predict_s = time.perf_counter() - t0
    if task == "binary":
        return {"fit_s": fit_s, "predict_s": predict_s,
                "auc_test": auc(yte, pred_te)}
    pred_tr = np.asarray(model.predict(X))
    return {"fit_s": fit_s, "predict_s": predict_s,
            "r2_train": r2(y, pred_tr), "r2_test": r2(yte, pred_te)}


def run_xgb(spec, X, y, Xte, yte) -> dict:
    import xgboost as xgb
    c = spec["cell"]
    task = c.get("task", "reg")
    device = resolve(spec["variant"]).device
    params = {**rp.xgb_core(learning_rate=c["lr"], max_depth=c["depth"],
                            min_data_in_leaf=c.get(
                                "min_data_in_leaf",
                                rp.SCALING["min_data_in_leaf"]),
                            lambda_l2=c.get("lambda_l2",
                                            rp.SCALING["lambda_l2"]),
                            max_bin=c["bins_effective"], seed=c["seed"]),
              "objective": ("binary:logistic" if task == "binary"
                            else "reg:squarederror"),
              "device": device, "nthread": spec["threads"]}
    t0 = time.perf_counter()
    dtrain = xgb.QuantileDMatrix(X, label=y, max_bin=c["bins_effective"])
    booster = xgb.train(params, dtrain, num_boost_round=c["iters"])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_te = booster.inplace_predict(Xte)
    predict_s = time.perf_counter() - t0
    if task == "binary":
        return {"fit_s": fit_s, "predict_s": predict_s,
                "auc_test": auc(yte, pred_te)}
    pred_tr = booster.inplace_predict(X)
    return {"fit_s": fit_s, "predict_s": predict_s,
            "r2_train": r2(y, pred_tr), "r2_test": r2(yte, pred_te)}


def run_lgbm(spec, X, y, Xte, yte) -> dict:
    import lightgbm as lgb
    c = spec["cell"]
    task = c.get("task", "reg")
    device = resolve(spec["variant"]).device
    params = {**rp.lgbm_core(learning_rate=c["lr"], max_depth=c["depth"],
                             num_leaves=rp.num_leaves_full(c["depth"]),
                             min_data_in_leaf=c.get(
                                 "min_data_in_leaf",
                                 rp.SCALING["min_data_in_leaf"]),
                             lambda_l2=c.get("lambda_l2",
                                             rp.SCALING["lambda_l2"]),
                             max_bin=c["bins_effective"], seed=c["seed"]),
              "objective": "binary" if task == "binary" else "regression",
              "device_type": device, "num_threads": spec["threads"]}
    t0 = time.perf_counter()
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=c["iters"])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_te = model.predict(Xte)
    predict_s = time.perf_counter() - t0
    if task == "binary":
        return {"fit_s": fit_s, "predict_s": predict_s,
                "auc_test": auc(yte, pred_te)}
    pred_tr = model.predict(X)
    return {"fit_s": fit_s, "predict_s": predict_s,
            "r2_train": r2(y, pred_tr), "r2_test": r2(yte, pred_te)}


def run_catboost(spec, X, y, Xte, yte) -> dict:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    c = spec["cell"]
    task = c.get("task", "reg")
    device = resolve(spec["variant"]).device
    cls = CatBoostClassifier if task == "binary" else CatBoostRegressor
    model = cls(
        **rp.catboost_core(learning_rate=c["lr"], max_depth=c["depth"],
                           lambda_l2=c.get("lambda_l2",
                                           rp.SCALING["lambda_l2"]),
                           max_bin=c["bins_effective"], seed=c["seed"],
                           device=device),
        iterations=c["iters"],
        loss_function="Logloss" if task == "binary" else "RMSE",
        task_type=("GPU" if device == "cuda" else "CPU"), devices="0",
        thread_count=spec["threads"], verbose=False)
    t0 = time.perf_counter()
    pool = Pool(X, label=y)
    model.fit(pool)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_te = (model.predict_proba(Xte)[:, 1] if task == "binary"
               else model.predict(Xte))
    predict_s = time.perf_counter() - t0
    if task == "binary":
        return {"fit_s": fit_s, "predict_s": predict_s,
                "auc_test": auc(yte, pred_te)}
    pred_tr = model.predict(X)
    return {"fit_s": fit_s, "predict_s": predict_s,
            "r2_train": r2(y, pred_tr), "r2_test": r2(yte, pred_te)}


RUNNERS = {"bonsai": run_bonsai, "xgb": run_xgb, "lgbm": run_lgbm,
           "catboost": run_catboost}


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
    c = spec["cell"]
    cache_dir = os.environ.get("BONSAI_BENCH_DATA_CACHE")
    if cache_dir:
        X, y, Xte, yte = cached_gen_data(c, cache_dir)
    else:
        X, y, Xte, yte = gen_data(c["rows"], c["cols"], c["seed"], c["n_test"],
                                  c["informative"])
    v = resolve(spec["variant"])
    if v.name.startswith("bonsai_ts_"):
        raise RuntimeError("unsupported: ordered-TS arms run in the airline "
                           "suite only (the encoder lives there)")
    run = RUNNERS[v.lib]
    if v.device == "cuda":
        # Untimed micro-fit absorbs CUDA context creation (and, once per
        # session, the PTX JIT — disk-cached afterwards).
        micro = dict(spec, cell=dict(c, rows=8192, n_test=1024, iters=5))
        run(micro, X[:8192], y[:8192], Xte[:1024], yte[:1024])
    out = run(spec, X, y, Xte, yte)
    out["peak_rss_gb"] = runlog.peak_rss_gb()
    out["fit_rows_per_s"] = round(c["rows"] / out["fit_s"]) if out["fit_s"] else None
    out["predict_rows_per_s"] = (round(c["n_test"] / out["predict_s"])
                                 if out["predict_s"] else None)
    for k in ("fit_s", "predict_s"):
        out[k] = round(out[k], 3)
    for k in ("r2_train", "r2_test"):
        out[k] = round(out[k], 4)
    # The child is where the reference library was imported, so only the
    # child can report its version; the parent folds this into host.libs.
    out["libs"] = runlog.lib_versions()
    return out
