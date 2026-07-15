"""Airline delays speed rung (perf division): the benchm-ml real-data ladder,
bonsai vs xgboost / lightgbm / catboost at 0.1M / 1M / 10M rows.

    python -m bonsai.bench.airline out.jsonl --sizes 0.1m
    python -m bonsai.bench.airline out.jsonl --sizes 0.1m,1m,10m --variants all

Provenance: Szilard Pafka's benchm-ml / GBM-perf airline on-time dataset
(https://github.com/szilard/benchm-ml), train-{0.1m,1m,10m}.csv + test.csv
(100k rows) from the public S3 bucket. Binary target dep_delayed_15min;
features Month, DayofMonth, DayOfWeek, UniqueCarrier, Origin, Dest
(categorical) + DepTime, Distance (numeric). This is the standard real-data
GBM speed benchmark; it complements the synthetic scaling suite with mixed
categorical/numeric columns and a real class balance (~19% positive).

Encoding convention (decision 68, uniform across libraries): categoricals
become sorted-unique ordinal codes fit on the train split; test categories
unseen in train map to -1. This strips catboost's native categorical
machinery, the same documented trade the Grinsztajn suite makes; the
comparison measures histogram engines, not encoders.

Rows are labeled division="perf", timing_mode="in_memory" (schema v1,
bonsai.bench.runlog) with AUC as the quality column (metrics.auc, the
protocol's binary primary). Same-pod discipline applies: only rows from one
host compare (docs/method/benchmark-protocol.md).

Every (size, variant) runs in a child process (this file with --worker), the
scaling suite's OOM/segfault isolation pattern.
"""
import argparse
import json
import pathlib
import resource
import subprocess
import sys
import time
import urllib.request

import numpy as np

from . import params as rp
from . import runlog
from .datasets import data_root
from .metrics import auc

S3 = "https://s3.amazonaws.com/benchm-ml--main"
SIZES = {"0.1m": "train-0.1m.csv", "1m": "train-1m.csv", "10m": "train-10m.csv"}
CATEGORICAL = ("Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest")
NUMERIC = ("DepTime", "Distance")
TARGET = "dep_delayed_15min"

# The scaling suite's knob shape at the campaign-matched values; --depth and
# --iters exist so a pod session can also run Pafka's own protocol (depth 10)
# for cross-table comparability.
KNOBS = {"depth": 8, "iters": 100, "lr": 0.1, "bins": 255, "seed": 42,
         "min_data_in_leaf": 20, "lambda_l2": 1.0}

VARIANTS = {
    "bonsai_depthwise": ("bonsai", "cpu"),
    "bonsai_oblivious": ("bonsai", "cpu"),
    "bonsai_cuda_depthwise": ("bonsai", "cuda"),
    "bonsai_cuda_oblivious": ("bonsai", "cuda"),
    "xgb_hist": ("xgb", "cpu"),
    "xgb_cuda": ("xgb", "cuda"),
    "lgbm_cpu": ("lgbm", "cpu"),
    "catboost_cpu": ("catboost", "cpu"),
    "catboost_gpu": ("catboost", "cuda"),
}


def fetch(size: str) -> tuple[pathlib.Path, pathlib.Path]:
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for fname in (SIZES[size], "test.csv"):
        local = root / f"airline_{fname}" if fname == "test.csv" else root / fname
        if not local.exists():
            print(f"fetching {S3}/{fname} -> {local}", file=sys.stderr, flush=True)
            urllib.request.urlretrieve(f"{S3}/{fname}", local)
        out.append(local)
    return out[0], out[1]


def _encode(size: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse + ordinal-encode once per size; cached as npz beside the CSVs."""
    cache = data_root() / f"airline_{size}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["y"], z["Xte"], z["yte"]
    import pandas as pd
    train_path, test_path = fetch(size)
    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)
    cols = list(CATEGORICAL) + list(NUMERIC)
    x_tr = np.empty((len(tr), len(cols)), dtype=np.float32)
    x_te = np.empty((len(te), len(cols)), dtype=np.float32)
    for j, col in enumerate(cols):
        if col in CATEGORICAL:
            cats = np.sort(tr[col].astype(str).unique())
            code = {c: float(i) for i, c in enumerate(cats)}
            x_tr[:, j] = tr[col].astype(str).map(code).to_numpy(dtype=np.float32)
            x_te[:, j] = (te[col].astype(str).map(code).fillna(-1.0)
                          .to_numpy(dtype=np.float32))
        else:
            x_tr[:, j] = tr[col].to_numpy(dtype=np.float32)
            x_te[:, j] = te[col].to_numpy(dtype=np.float32)
    y_tr = (tr[TARGET] == "Y").to_numpy(dtype=np.float32)
    y_te = (te[TARGET] == "Y").to_numpy(dtype=np.float32)
    np.savez(cache, X=x_tr, y=y_tr, Xte=x_te, yte=y_te)
    return x_tr, y_tr, x_te, y_te


# ---- per-library runners (worker side): binary objective, AUC on raw scores ----
# Param mappings come from bonsai.bench.params (decision 68's lesson: never
# re-derive reference knobs by hand). AUC is rank-based, so each library's raw
# or probability output scores it identically.


def run_bonsai(spec, X, y, Xte, yte) -> dict:
    import bonsai
    grower = spec["variant"].removeprefix("bonsai_")
    if grower.startswith("cuda") and not bonsai.cuda_available():
        raise RuntimeError("unsupported: cuda grower without a CUDA device/build")
    k = spec["knobs"]
    pairs = [("dispatch.grower_name", grower),
             ("dispatch.objective_name", "logloss"),
             ("booster.n_iters", str(k["iters"])),
             ("booster.learning_rate", str(k["lr"])),
             ("booster.random_seed", str(k["seed"])),
             ("tree.max_depth", str(k["depth"])),
             ("tree.max_leaves", str(1 << k["depth"])),
             ("tree.min_data_in_leaf", str(k["min_data_in_leaf"])),
             ("tree.lambda_l2", str(k["lambda_l2"])),
             ("bin_mapper.max_bin", str(k["bins"])),
             ("parallel.n_threads", str(spec["threads"]))]
    t0 = time.perf_counter()
    model = bonsai.train(pairs, X, y)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = np.asarray(model.predict(Xte))
    predict_s = time.perf_counter() - t0
    return {"fit_s": fit_s, "predict_s": predict_s, "auc_test": auc(yte, pred)}


def run_xgb(spec, X, y, Xte, yte) -> dict:
    import xgboost as xgb
    k = spec["knobs"]
    device = VARIANTS[spec["variant"]][1]
    params = {**rp.xgb_core(learning_rate=k["lr"], max_depth=k["depth"],
                            min_data_in_leaf=k["min_data_in_leaf"],
                            lambda_l2=k["lambda_l2"],
                            max_bin=min(k["bins"], 254), seed=k["seed"]),
              "objective": "binary:logistic", "device": device,
              "nthread": spec["threads"]}
    t0 = time.perf_counter()
    dtrain = xgb.QuantileDMatrix(X, label=y, max_bin=min(k["bins"], 254))
    booster = xgb.train(params, dtrain, num_boost_round=k["iters"])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = booster.inplace_predict(Xte)
    predict_s = time.perf_counter() - t0
    return {"fit_s": fit_s, "predict_s": predict_s, "auc_test": auc(yte, pred)}


def run_lgbm(spec, X, y, Xte, yte) -> dict:
    import lightgbm as lgb
    k = spec["knobs"]
    params = {**rp.lgbm_core(learning_rate=k["lr"], max_depth=k["depth"],
                             num_leaves=1 << k["depth"],
                             min_data_in_leaf=k["min_data_in_leaf"],
                             lambda_l2=k["lambda_l2"],
                             max_bin=min(k["bins"], 254), seed=k["seed"]),
              "objective": "binary", "device_type": "cpu",
              "num_threads": spec["threads"]}
    t0 = time.perf_counter()
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=k["iters"])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(Xte)
    predict_s = time.perf_counter() - t0
    return {"fit_s": fit_s, "predict_s": predict_s, "auc_test": auc(yte, pred)}


def run_catboost(spec, X, y, Xte, yte) -> dict:
    from catboost import CatBoostClassifier, Pool
    k = spec["knobs"]
    device = VARIANTS[spec["variant"]][1]
    model = CatBoostClassifier(
        **rp.catboost_core(learning_rate=k["lr"], max_depth=k["depth"],
                           lambda_l2=k["lambda_l2"],
                           max_bin=min(k["bins"], 254), seed=k["seed"],
                           device=device),
        iterations=k["iters"], loss_function="Logloss",
        task_type=("GPU" if device == "cuda" else "CPU"), devices="0",
        thread_count=spec["threads"], verbose=False)
    t0 = time.perf_counter()
    pool = Pool(X, label=y)
    model.fit(pool)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict_proba(Xte)[:, 1]
    predict_s = time.perf_counter() - t0
    return {"fit_s": fit_s, "predict_s": predict_s, "auc_test": auc(yte, pred)}


RUNNERS = {"bonsai": run_bonsai, "xgb": run_xgb, "lgbm": run_lgbm,
           "catboost": run_catboost}


def worker(spec: dict) -> dict:
    X, y, Xte, yte = _encode(spec["size"])
    lib, device = VARIANTS[spec["variant"]]
    run = RUNNERS[lib]
    if device == "cuda":
        micro = dict(spec, knobs=dict(spec["knobs"], iters=5))
        run(micro, X[:8192], y[:8192], Xte[:1024], yte[:1024])
    out = run(spec, X, y, Xte, yte)
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out["peak_rss_gb"] = round(ru / (2**30 if sys.platform == "darwin" else 2**20), 2)
    out["fit_s"] = round(out["fit_s"], 3)
    out["predict_s"] = round(out["predict_s"], 3)
    out["auc_test"] = round(out["auc_test"], 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="airline.jsonl")
    ap.add_argument("--sizes", default="0.1m")
    ap.add_argument("--variants", default="all")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--depth", type=int, default=KNOBS["depth"])
    ap.add_argument("--iters", type=int, default=KNOBS["iters"])
    ap.add_argument("--host-name", default=None)
    ap.add_argument("--worker", action="store_true")
    args = ap.parse_args()

    if args.worker:
        spec = json.loads(sys.stdin.read())
        print("RESULT " + json.dumps(worker(spec)), flush=True)
        return 0

    variants = (list(VARIANTS) if args.variants == "all"
                else args.variants.split(","))
    host = runlog.detect_host(args.host_name)
    for size in args.sizes.split(","):
        _encode(size)  # fetch + parse once in the parent; workers hit the cache
        for variant in variants:
            knobs = dict(KNOBS, depth=args.depth, iters=args.iters)
            spec = {"size": size, "variant": variant, "threads": args.threads,
                    "knobs": knobs}
            proc = subprocess.run(
                [sys.executable, "-m", "bonsai.bench.airline", "--worker"],
                input=json.dumps(spec), capture_output=True, text=True,
                timeout=3600)
            line = next((ln for ln in proc.stdout.splitlines()
                         if ln.startswith("RESULT ")), None)
            if line is None:
                status = "unsupported" if "unsupported" in proc.stderr else "error"
                runlog.emit_row(args.out, division="perf", suite="airline",
                                knobs=knobs, host=host, timing_mode="in_memory",
                                size=size, variant=variant, status=status,
                                error=proc.stderr.strip()[-400:])
                print(f"{size} {variant}: {status}", flush=True)
                continue
            out = json.loads(line.removeprefix("RESULT "))
            runlog.emit_row(args.out, division="perf", suite="airline",
                            knobs=knobs, host=host, timing_mode="in_memory",
                            size=size, variant=variant, status="ok", **out)
            print(f"{size} {variant}: fit {out['fit_s']}s "
                  f"auc {out['auc_test']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
