"""Scaling benchmark suite (perf division): synthetic regression swept over
rows / cols / bins / threads, bonsai vs xgboost / lightgbm / catboost.

    python -m bonsai.bench.scaling --smoke
    python -m bonsai.bench.scaling --axis all

Rows are labeled division="perf", timing_mode="in_memory" (schema v1,
bonsai.bench.runlog); the synthetic recipe and its provenance live in
bonsai.bench.synth. See docs/method/benchmark-protocol.md.

Methodology (decision 46): in-memory float32 numpy arrays through each
library's Python API — bonsai via the nanobind module (make python-cuda on GPU
hosts; the Makefile bench-scaling target sets PYTHONPATH). fit_s times each
library's own ingestion (bonsai ColumnBatch + binning, xgboost QuantileDMatrix,
lgb.Dataset, catboost Pool) plus training; predict_s times prediction from a
raw test matrix. Quality is R^2 on train and a held-out test split.

Every (cell, variant, repeat) runs in a child process (this same file with
--worker): OOM and segfaults become jsonl status lines instead of killing the
sweep, and bonsai's exit-time profilers flush per-run breakdowns to the
child's stderr where the parent captures them.

    make bench-scaling ARGS="--axis all"
    uv run scripts/bench_scaling.py --smoke            # tiny grid (Mac)
    uv run scripts/bench_scaling.py --dry-run --axis all

Results append to benchmarks/results/scaling.jsonl; analyze with
scripts/analyze_scaling.py. Grid corners the host cannot fit are recorded as
status="skipped" with the memory estimate — the feasibility frontier is data.
"""
import argparse
import datetime
import json
import os
import pathlib
import re
import resource
import subprocess
import sys
import time

import numpy as np

from . import params as rp
from . import runlog
from .synth import gen_data

# In-repo runs default next to the other results; wheel installs default to
# the working directory.
_repo = pathlib.Path(__file__).resolve().parents[3]
RESULTS = (_repo / "benchmarks" / "results" / "scaling.jsonl"
           if (_repo / "benchmarks").is_dir()
           else pathlib.Path("scaling.jsonl"))

BASE = {"rows": 1_000_000, "cols": 100, "bins": 255, "depth": 8, "iters": 100,
        "lr": 0.1, "informative": 20, "seed": 42}

AXES = {
    "rows": [(250_000, 100), (1_000_000, 100), (4_000_000, 100), (16_000_000, 100)],
    # (rows, cols): rows shrink past 4k cols to cap cells at 2^31.
    "cols": [(1_000_000, 16), (1_000_000, 64), (1_000_000, 256), (1_000_000, 1024),
             (1_000_000, 4096), (131_072, 16_384), (32_768, 65_536)],
    "bins": [15, 63, 255, 1023, 4095, 16_383, 65_535],
    "threads": [1, 4, 16, 64],
}

# variant -> (library, device). lgbm_cuda is declared but unsupported in v1:
# the pip wheel has no CUDA backend and a source build is deferred.
VARIANTS = {
    "bonsai_depthwise": ("bonsai", "cpu"),
    "bonsai_leafwise": ("bonsai", "cpu"),
    "bonsai_oblivious": ("bonsai", "cpu"),
    "bonsai_cuda_depthwise": ("bonsai", "cuda"),
    "bonsai_cuda_oblivious": ("bonsai", "cuda"),
    "xgb_hist": ("xgb", "cpu"),
    "xgb_cuda": ("xgb", "cuda"),
    "lgbm_cpu": ("lgbm", "cpu"),
    "lgbm_cuda": ("lgbm", "cuda"),
    "catboost_cpu": ("catboost", "cpu"),
    "catboost_gpu": ("catboost", "cuda"),
}

# GPU variants skip the widest cells: 16k+ cols exhausts consumer VRAM and
# kernel grids; the per-host VRAM estimator handles the rest.
GPU_MAX_COLS = 16_384


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ---- per-library runners (worker side) ----------------------------------------
# Param mappings mirror scripts/bench_gpu.py and scripts/compare.py — keep in
# sync. Shared tree config: depth-d full trees (lgbm num_leaves = 1 << depth,
# bonsai max_leaves likewise), min_data_in_leaf 20, lambda_l2 1.0.


def run_bonsai(spec, X, y, Xte, yte) -> dict:
    import bonsai
    grower = spec["variant"].removeprefix("bonsai_")
    if grower.startswith("cuda") and not bonsai.cuda_available():
        raise RuntimeError("unsupported: cuda grower without a CUDA device/build")
    c = spec["cell"]
    pairs = [("dispatch.grower_name", grower),
             ("dispatch.objective_name", "mse"),
             ("booster.n_iters", str(c["iters"])),
             ("booster.learning_rate", str(c["lr"])),
             ("booster.random_seed", str(c["seed"])),
             ("tree.max_depth", str(c["depth"])),
             ("tree.max_leaves", str(1 << c["depth"])),
             ("tree.min_data_in_leaf", "20"),
             ("tree.lambda_l2", "1.0"),
             ("bin_mapper.max_bin", str(c["bins"])),
             ("parallel.n_threads", str(spec["threads"]))]
    t0 = time.perf_counter()
    model = bonsai.train(pairs, X, y)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_te = np.asarray(model.predict(Xte))
    predict_s = time.perf_counter() - t0
    pred_tr = np.asarray(model.predict(X))
    return {"fit_s": fit_s, "predict_s": predict_s,
            "r2_train": r2(pred_tr, y), "r2_test": r2(pred_te, yte)}


def run_xgb(spec, X, y, Xte, yte) -> dict:
    import xgboost as xgb
    c = spec["cell"]
    device = VARIANTS[spec["variant"]][1]
    params = {**rp.xgb_core(learning_rate=c["lr"], max_depth=c["depth"],
                            min_data_in_leaf=20, lambda_l2=1.0,
                            max_bin=c["bins_effective"], seed=c["seed"]),
              "objective": "reg:squarederror", "device": device,
              "nthread": spec["threads"]}
    t0 = time.perf_counter()
    dtrain = xgb.QuantileDMatrix(X, label=y, max_bin=c["bins_effective"])
    booster = xgb.train(params, dtrain, num_boost_round=c["iters"])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_te = booster.inplace_predict(Xte)
    predict_s = time.perf_counter() - t0
    pred_tr = booster.inplace_predict(X)
    return {"fit_s": fit_s, "predict_s": predict_s,
            "r2_train": r2(pred_tr, y), "r2_test": r2(pred_te, yte)}


def run_lgbm(spec, X, y, Xte, yte) -> dict:
    import lightgbm as lgb
    c = spec["cell"]
    device = VARIANTS[spec["variant"]][1]
    if device == "cuda":
        raise RuntimeError("unsupported: pip lightgbm lacks CUDA; source build "
                           "deferred to a later round")
    params = {**rp.lgbm_core(learning_rate=c["lr"], max_depth=c["depth"],
                             num_leaves=1 << c["depth"], min_data_in_leaf=20,
                             lambda_l2=1.0, max_bin=c["bins_effective"],
                             seed=c["seed"]),
              "objective": "regression",
              "device_type": device, "num_threads": spec["threads"]}
    t0 = time.perf_counter()
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=c["iters"])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_te = model.predict(Xte)
    predict_s = time.perf_counter() - t0
    pred_tr = model.predict(X)
    return {"fit_s": fit_s, "predict_s": predict_s,
            "r2_train": r2(pred_tr, y), "r2_test": r2(pred_te, yte)}


def run_catboost(spec, X, y, Xte, yte) -> dict:
    from catboost import CatBoostRegressor, Pool
    c = spec["cell"]
    device = VARIANTS[spec["variant"]][1]
    model = CatBoostRegressor(
        **rp.catboost_core(learning_rate=c["lr"], max_depth=c["depth"],
                           lambda_l2=1.0, max_bin=c["bins_effective"],
                           seed=c["seed"], device=device),
        iterations=c["iters"], loss_function="RMSE",
        task_type=("GPU" if device == "cuda" else "CPU"), devices="0",
        thread_count=spec["threads"], verbose=False)
    t0 = time.perf_counter()
    pool = Pool(X, label=y)
    model.fit(pool)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_te = model.predict(Xte)
    predict_s = time.perf_counter() - t0
    pred_tr = model.predict(X)
    return {"fit_s": fit_s, "predict_s": predict_s,
            "r2_train": r2(pred_tr, y), "r2_test": r2(pred_te, yte)}


RUNNERS = {"bonsai": run_bonsai, "xgb": run_xgb, "lgbm": run_lgbm,
           "catboost": run_catboost}


def worker(spec: dict) -> dict:
    c = spec["cell"]
    X, y, Xte, yte = gen_data(c["rows"], c["cols"], c["seed"], c["n_test"],
                              c["informative"])
    lib, device = VARIANTS[spec["variant"]]
    run = RUNNERS[lib]
    if device == "cuda":
        # Untimed micro-fit absorbs CUDA context creation (and, once per
        # session, the PTX JIT — disk-cached afterwards).
        micro = dict(spec, cell=dict(c, rows=8192, n_test=1024, iters=5))
        run(micro, X[:8192], y[:8192], Xte[:1024], yte[:1024])
    out = run(spec, X, y, Xte, yte)
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss: bytes on macOS, KiB on Linux.
    out["peak_rss_gb"] = round(ru / (2**30 if sys.platform == "darwin" else 2**20), 2)
    out["fit_rows_per_s"] = round(c["rows"] / out["fit_s"]) if out["fit_s"] else None
    out["predict_rows_per_s"] = (round(c["n_test"] / out["predict_s"])
                                 if out["predict_s"] else None)
    for k in ("fit_s", "predict_s"):
        out[k] = round(out[k], 3)
    for k in ("r2_train", "r2_test"):
        out[k] = round(out[k], 4)
    return out


# ---- parent orchestration ------------------------------------------------------

PROFILE_RE = re.compile(r"(\w+)=([\d.]+)s")


def parse_profiles(stderr: str) -> dict:
    prof = {}
    for line in stderr.splitlines():
        if line.startswith(("cuda-profile:", "grow-profile:", "ingest-profile:",
                            "fit-profile:", "cuda-upload-decomp:")):
            prefix = line.split(":", 1)[0].removesuffix("-profile")
            for key, val in PROFILE_RE.findall(line):
                prof[f"{prefix}_{key}"] = float(val)
    return prof


def est_host_gb(rows: int, cols: int, n_test: int, lib: str) -> float:
    factor = 4.0 if lib == "catboost" else 3.0
    return (rows + n_test) * cols * 6 * factor / 2**30


def est_dev_gb(rows: int, cols: int) -> float:
    return (rows * cols * 2 + rows * 8 + 512 * 2**20) / 2**30


def make_cell(axis: str, rows: int, cols: int, bins: int) -> dict:
    n_test = min(rows // 5, 500_000)
    return dict(axis=axis, rows=rows, cols=cols, bins=bins, bins_effective=bins,
                depth=BASE["depth"], iters=BASE["iters"], lr=BASE["lr"],
                informative=BASE["informative"], n_test=n_test, seed=BASE["seed"])


def build_grid(axes: list[str], smoke: bool) -> list[tuple[dict, int]]:
    """Returns [(cell, threads)]; the base cell appears once with axis='base'."""
    if smoke:
        cells = [make_cell("rows", r, c, b)
                 for r in (50_000, 100_000) for c in (16, 64) for b in (15, 63)]
        for cell in cells:
            cell.update(iters=20, n_test=10_000)
        return [(c, 16) for c in cells]
    grid: list[tuple[dict, int]] = [(make_cell("base", BASE["rows"], BASE["cols"],
                                               BASE["bins"]), 16)]
    if "rows" in axes:
        grid += [(make_cell("rows", r, c, BASE["bins"]), 16)
                 for r, c in AXES["rows"] if r != BASE["rows"]]
    if "cols" in axes:
        grid += [(make_cell("cols", r, c, BASE["bins"]), 16)
                 for r, c in AXES["cols"] if c != BASE["cols"]]
    if "bins" in axes:
        grid += [(make_cell("bins", BASE["rows"], BASE["cols"], b), 16)
                 for b in AXES["bins"] if b != BASE["bins"]]
    if "threads" in axes:
        grid += [(make_cell("threads", BASE["rows"], BASE["cols"], BASE["bins"]), t)
                 for t in AXES["threads"] if t != 16]
    return grid


def load_resume(path: str) -> set[tuple]:
    """Keys already measured (pods die: funds, spot reaps). ok/unsupported are
    final; skipped/oom/timeout/error re-attempt — the new host may differ."""
    done = set()
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["status"] in ("ok", "unsupported"):
            c = r["cell"]
            done.add((r["variant"], r["threads"], r["repeat"], c["rows"],
                      c["cols"], c["bins"]))
    return done


def classify_error(message: str) -> str:
    low = message.lower()
    if any(s in low for s in ("out of memory", "memoryerror", "bad_alloc",
                              "cannot allocate", "oom")):
        return "oom"
    if any(s in low for s in ("unsupported", "max_bin", "border_count",
                              "invalid parameter", "must be")):
        return "unsupported"
    return "error"


def timeout_for(cell: dict) -> int:
    cells = cell["rows"] * cell["cols"]
    return int(min(3600, max(900, 90 * cells / 1e8)))


def run_one(spec: dict, timeout: int) -> dict:
    env = dict(os.environ)
    if spec["variant"].startswith("bonsai"):
        env.update(BONSAI_GROW_PROFILE="1", BONSAI_INGEST_PROFILE="1",
                   BONSAI_CUDA_PROFILE="1", BONSAI_FIT_PROFILE="1")
    try:
        proc = subprocess.run([sys.executable, "-m", "bonsai.bench.scaling", "--worker"],
                              input=json.dumps(spec), capture_output=True,
                              text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "message": f"exceeded {timeout}s"}
    result_line = next((ln for ln in proc.stdout.splitlines()
                        if ln.startswith("RESULT ")), None)
    if proc.returncode != 0 or result_line is None:
        if proc.returncode < 0:
            return {"status": "oom",
                    "message": f"killed by signal {-proc.returncode}"}
        tail = (proc.stderr.strip().splitlines() or ["no output"])[-1][:300]
        return {"status": classify_error(tail), "message": tail}
    out = json.loads(result_line.removeprefix("RESULT "))
    out["status"] = "ok"
    out["message"] = None
    prof = parse_profiles(proc.stderr)
    out["profile"] = prof or None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--axis", default="all",
                    help="rows|cols|bins|threads|all (comma-separable)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--host-name", default=None)
    ap.add_argument("--timeout-cap", type=int, default=3600)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", default=None,
                    help="jsonl of prior results; ok/unsupported runs are skipped")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    if args.worker:
        spec = json.loads(sys.stdin.read())
        print("RESULT " + json.dumps(worker(spec)))
        return 0

    host = runlog.detect_host(args.host_name)
    axes = (["rows", "cols", "bins", "threads"] if args.axis == "all"
            else args.axis.split(","))
    variants = args.variants.split(",")
    for v in variants:
        if v not in VARIANTS:
            ap.error(f"unknown variant {v}")
    grid = build_grid(axes, args.smoke)
    git_sha = runlog.git_sha()
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    knobs = dict(rp.SCALING, num_leaves_convention="full")

    def emit(cell, variant, threads, repeat, payload):
        line = {"schema": runlog.SCHEMA_VERSION,
                "ts": datetime.datetime.now(datetime.timezone.utc)
                .isoformat(timespec="seconds"),
                "git_sha": git_sha, "division": "perf", "suite": "scaling",
                "timing_mode": "in_memory", "knobs": knobs,
                "knobs_hash": runlog.knobs_hash(knobs),
                "dataset": "synthetic-friedman1", "task": "reg",
                "host": host, "cell": cell,
                "variant": variant, "threads": threads, "repeat": repeat,
                **payload}
        line.setdefault("profile", None)
        with out_path.open("a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"  {variant:>24} t={threads:<3} rows={cell['rows']:>8} "
              f"cols={cell['cols']:>5} bins={cell['bins']:>5} "
              f"-> {payload['status']}"
              + (f" fit={payload['fit_s']}s r2={payload['r2_test']}"
                 if payload["status"] == "ok" else f" ({payload['message']})"))

    done = load_resume(args.resume) if args.resume else set()

    warmed: set[str] = set()
    for cell, threads in grid:
        for variant in variants:
            lib, device = VARIANTS[variant]
            cell_v = dict(cell)
            if variant == "catboost_gpu":
                cell_v["bins_effective"] = min(cell["bins"], 254)
            repeats = max(args.repeats, 3) if (cell["axis"] == "base"
                                               and not args.smoke) else args.repeats
            skip = None
            if variant == "lgbm_cuda":
                skip = ("unsupported", "pip lightgbm lacks CUDA (deferred)")
            elif device == "cuda" and host["gpu"] is None:
                skip = ("skipped", "no CUDA device on host")
            elif device == "cuda" and cell["cols"] > GPU_MAX_COLS:
                skip = ("skipped", f"cols > {GPU_MAX_COLS} (GPU variant policy)")
            elif device == "cuda" and est_dev_gb(cell["rows"], cell["cols"]) > \
                    0.85 * (host["gpu_vram_gb"] or 0):
                skip = ("skipped", f"est {est_dev_gb(cell['rows'], cell['cols']):.1f}"
                                   f"GB > 0.85x{host['gpu_vram_gb']}GB VRAM")
            elif variant.startswith("bonsai") and cell["bins"] > 65_535:
                skip = ("unsupported", "bonsai bin_id_t is uint16 (max_bin <= 65535)")
            elif est_host_gb(cell["rows"], cell["cols"], cell["n_test"], lib) > \
                    0.8 * host["ram_gb"]:
                est = est_host_gb(cell["rows"], cell["cols"], cell["n_test"], lib)
                skip = ("skipped", f"est {est:.1f}"
                                   f"GB > 0.8x{host['ram_gb']}GB RAM")
            elif cell["axis"] == "threads" and threads > (host["n_vcpu"] or 1):
                skip = ("skipped", f"threads {threads} > {host['n_vcpu']} vcpus")
            if skip:
                already = (variant, threads, 0, cell["rows"], cell["cols"],
                           cell["bins"]) in done
                if not args.dry_run and not already:
                    emit(cell_v, variant, threads, 0,
                         {"status": skip[0], "message": skip[1]})
                else:
                    print(f"  {variant:>24} {cell['rows']}x{cell['cols']}x"
                          f"{cell['bins']} -> {skip[0]}: {skip[1]}")
                continue
            timeout = min(timeout_for(cell), args.timeout_cap)
            if args.dry_run:
                print(f"  {variant:>24} t={threads:<3} {cell['rows']}x"
                      f"{cell['cols']}x{cell['bins']} timeout={timeout}s "
                      f"repeats={repeats}")
                continue
            if device == "cuda" and lib not in warmed:
                warm_cell = make_cell("warmup", 32_768, 16, 63)
                warm_cell.update(iters=5, n_test=1024)
                run_one({"cell": warm_cell, "variant": variant, "threads": 4},
                        timeout=600)
                warmed.add(lib)
            for rep in range(repeats):
                if (variant, threads, rep, cell["rows"], cell["cols"],
                        cell["bins"]) in done:
                    print(f"  {variant:>24} t={threads:<3} {cell['rows']}x"
                          f"{cell['cols']}x{cell['bins']} rep={rep} -> resume-skip")
                    continue
                spec = {"cell": cell_v, "variant": variant, "threads": threads}
                emit(cell_v, variant, threads, rep, run_one(spec, timeout))
    return 0


if __name__ == "__main__":
    sys.exit(main())
