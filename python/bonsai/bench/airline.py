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

The 10m train file plus test.csv are baked into the bonsai-ci image at
BAKED_DIR; fetch() prefers that copy and only reaches S3 when it is absent
(a repo checkout, a non-CI machine, or the 0.1m/1m sizes).

Encoding convention (decision 68, uniform across libraries): categoricals
become sorted-unique ordinal codes fit on the train split; test categories
unseen in train map to -1. This strips catboost's native categorical
machinery, the same documented trade the Grinsztajn suite makes; the
comparison measures histogram engines, not encoders.

The bonsai_ts_* variants are the labeled EXCEPTION to that convention: they
feed bonsai through OrderedTargetEncoder (decision 58) first, so the ordinal
code spaces become causal target-mean features. Rows carry encoding=
"ordered_ts" (vs "ordinal") and fit_s includes the encoding time (it is part
of that pipeline's wall clock; encode_s is also recorded separately). They
answer "what does the shipped categorical story buy on this data", not the
uniform engine comparison.

Rows are labeled division="perf", timing_mode="in_memory" (schema v1,
bonsai.bench.runlog) with AUC as the quality column (metrics.auc, the
protocol's binary primary). Same-pod discipline applies: only rows from one
host compare (docs/method/benchmark-protocol.md).

Every (size, variant) runs in a child process (this file with --worker), the
scaling suite's OOM/segfault isolation pattern.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import resource
import subprocess
import sys
import time
import urllib.request

import numpy as np

from bonsai.bench import runlog
from bonsai.bench import variants as vr
from bonsai.bench.datasets import data_root
from bonsai.bench.runners import RUNNERS

S3 = "https://s3.amazonaws.com/benchm-ml--main"
# bonsai-ci bakes the 10m train file plus the shared test file at this path
# (docker/ci.Dockerfile); a pod refresh on that image skips the download.
BAKED_DIR = pathlib.Path("/opt/bonsai-data/airline")
SIZES = {"0.1m": "train-0.1m.csv", "1m": "train-1m.csv", "10m": "train-10m.csv"}
CATEGORICAL = ("Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest")
NUMERIC = ("DepTime", "Distance")
TARGET = "dep_delayed_15min"

# The scaling suite's knob shape at the campaign-matched values; --depth and
# --iters exist so a pod session can also run Pafka's own protocol (depth 10)
# for cross-table comparability.
KNOBS = {"depth": 8, "iters": 100, "lr": 0.1, "bins": 255, "seed": 42,
         "min_data_in_leaf": 20, "lambda_l2": 1.0}

# variant -> (library, device), a view of the registry in bonsai.bench.variants.
VARIANTS = {n: (vr.resolve(n).lib, vr.resolve(n).device) for n in vr.AIRLINE}


def fetch(size: str) -> tuple[pathlib.Path, pathlib.Path]:
    """(train_csv, test_csv) for a size: the baked image copy if present,
    else downloaded from S3 once and cached under data_root()."""
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for fname in (SIZES[size], "test.csv"):
        baked = BAKED_DIR / fname
        if baked.exists():
            out.append(baked)
            continue
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


def worker(spec: dict) -> dict:
    """Child-process entry: encode, dispatch the shared runner, add AUC.

    bonsai_ts_* arms apply the ordered target encoder first; its cost folds
    into fit_s (a pipeline's wall clock includes its preprocessing) and is
    also recorded as encode_s so the split stays visible.
    """
    X, y, Xte, yte = _encode(spec["size"])
    lib, device = VARIANTS[spec[runlog.Row.VARIANT]]
    run = RUNNERS[lib]
    encode_s = 0.0
    # The shared runners read scaling-shaped cells; the knobs map on directly
    # and task="binary" selects the logloss objective and AUC scoring.
    if spec[runlog.Row.VARIANT].startswith("bonsai_ts_"):
        from bonsai.encoding import OrderedTargetEncoder
        t0 = time.perf_counter()
        enc = OrderedTargetEncoder(columns=range(len(CATEGORICAL)),
                                   seed=spec["knobs"]["seed"])
        X = np.ascontiguousarray(enc.fit_transform(X, y), dtype=np.float32)
        Xte = np.ascontiguousarray(enc.transform(Xte), dtype=np.float32)
        encode_s = time.perf_counter() - t0
        # run_bonsai reads the grower name off the variant; hand it the
        # plain spelling. The parent's emitted row keeps the ts_ name.
        spec = dict(spec, variant=spec[runlog.Row.VARIANT].replace("_ts_", "_", 1))
    k = spec["knobs"]
    child = {runlog.Row.CELL: dict(k, bins_effective=k["bins"], task="binary"),
             runlog.Row.VARIANT: spec[runlog.Row.VARIANT],
             runlog.Row.THREADS: spec[runlog.Row.THREADS]}
    if device == vr.Device.CUDA:
        micro = dict(child, cell=dict(child[runlog.Row.CELL], iters=5))
        run(micro, X[:8192], y[:8192], Xte[:1024], yte[:1024])
    out = run(child, X, y, Xte, yte)
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out[runlog.Row.PEAK_RSS_GB] = round(ru / (2**30 if sys.platform == "darwin" else 2**20), 2)
    # A pipeline's wall clock includes its preprocessing: encode_s folds into
    # fit_s and is also recorded on its own so the split stays visible.
    out[runlog.Row.FIT_S] = round(out[runlog.Row.FIT_S] + encode_s, 3)
    if encode_s:
        out["encode_s"] = round(encode_s, 3)
    out[runlog.Row.PREDICT_S] = round(out[runlog.Row.PREDICT_S], 3)
    out[runlog.Row.AUC_TEST] = round(out[runlog.Row.AUC_TEST], 4)
    # The child is where the reference library was imported, so only the
    # child can report its version; the parent folds this into host.libs.
    out["libs"] = runlog.lib_versions()
    return out


def main() -> int:
    """Parent loop: one worker child per (size, variant), rows to jsonl."""
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
            _run_variant(args.out, host, size, variant, knobs, args.threads)
    return 0


def _run_variant(out_path: str, host: dict, size: str, variant: str,
                 knobs: dict, threads: int):
    """One worker child for one (size, variant); emit its row, ok or not."""
    encoding = ("ordered_ts" if variant.startswith("bonsai_ts_")
                else "ordinal")
    spec = {"size": size, runlog.Row.VARIANT: variant,
            runlog.Row.THREADS: threads, "knobs": knobs}
    proc = subprocess.run(
        [sys.executable, "-m", "bonsai.bench.airline", "--worker"],
        input=json.dumps(spec), capture_output=True, text=True,
        timeout=3600)
    line = next((ln for ln in proc.stdout.splitlines()
                 if ln.startswith("RESULT ")), None)
    if line is None:
        status = "unsupported" if "unsupported" in proc.stderr else "error"
        runlog.emit_row(out_path, division="perf", suite="airline",
                        knobs=knobs, host=host, timing_mode="in_memory",
                        size=size, variant=variant, encoding=encoding,
                        status=status, error=proc.stderr.strip()[-400:])
        print(f"{size} {variant}: {status}", flush=True)
        return
    out = json.loads(line.removeprefix("RESULT "))
    libs = out.pop("libs", None)
    row_host = (dict(host, libs={**host.get("libs", {}), **libs})
                if libs else host)
    runlog.emit_row(out_path, division="perf", suite="airline",
                    knobs=knobs, host=row_host, timing_mode="in_memory",
                    size=size, variant=variant, encoding=encoding,
                    status="ok", **out)
    print(f"{size} {variant}: fit {out[runlog.Row.FIT_S]}s "
          f"auc {out[runlog.Row.AUC_TEST]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
