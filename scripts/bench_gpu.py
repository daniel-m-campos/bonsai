# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "xgboost", "lightgbm", "catboost"]
# ///
"""MSD benchmark ladder for the GPU perf loop (see benchmarks/README.md).

Runs bonsai and reference libraries with hyperparameters from
configs/year_prediction_msd.toml, prints one table with profile breakdowns
and the gap to xgboost-GPU, and appends one JSON line per variant to
benchmarks/results/gpu_msd.jsonl so regression tracking across commits is a
diff of that file. All fit_s time the FULL pipeline: CSV read + binning +
train (fair data-loading on every library).

    uv run scripts/bench_gpu.py [--threads 16] [--variants a,b,...]

Variants: bonsai_gpu (cuda_depthwise), bonsai_obl_gpu (cuda_oblivious),
bonsai_cpu (depthwise), bonsai_leaf_cpu (leafwise), xgb_cpu, xgb_gpu,
lgbm_cpu, lgbm_gpu (needs a CUDA source build), catboost_cpu, catboost_gpu.
Param mappings mirror scripts/compare.py — keep them in sync.
"""
import argparse
import datetime
import json
import pathlib
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import reference_params as rp
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[1]
BINARY = REPO / "build-cuda" / "src" / "bonsai"
CONFIG = REPO / "configs" / "year_prediction_msd.toml"
RESULTS = REPO / "benchmarks" / "results" / "gpu_msd.jsonl"

PROFILE_RE = re.compile(r"(\w+)=([\d.]+)s")


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def parse_profiles(stderr: str) -> dict:
    prof = {}
    for line in stderr.splitlines():
        if line.startswith(("cuda-profile:", "grow-profile:")):
            prefix = line.split(":", 1)[0].removesuffix("-profile")
            for key, val in PROFILE_RE.findall(line):
                prof[f"{prefix}_{key}"] = float(val)
    return prof


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


# The CLI binary (not the python module) is deliberate: only build-cuda/src/bonsai
# is CUDA-enabled (the `make python` extension is CPU-only), it reads the TOML
# config directly, and the profile breakdowns are env-gated stderr prints the
# subprocess boundary captures cleanly.
def run_bonsai(grower: str, threads: int, y_test: np.ndarray) -> dict:
    model, preds = "/tmp/bench_gpu_model.msgpack", "/tmp/bench_gpu_preds.csv"
    fit_cmd = [str(BINARY), "fit", "-c", str(CONFIG),
               "--set", f"dispatch.grower_name={grower}",
               "--set", f"parallel.n_threads={threads}",
               "--model", model]
    t0 = time.perf_counter()
    fit = subprocess.run(fit_cmd, check=True, capture_output=True, text=True)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    subprocess.run([str(BINARY), "predict", "-c", str(CONFIG), "--model", model,
                    "--out", preds], check=True, capture_output=True)
    predict_s = time.perf_counter() - t0
    pred = pd.read_csv(preds)["prediction"].to_numpy()
    return {"fit_s": round(fit_s, 2), "predict_s": round(predict_s, 2),
            "rmse": round(rmse(pred, y_test), 4), "profile": parse_profiles(fit.stderr)}


# fit_s times the whole ingest+train pipeline — CSV read + DMatrix (xgboost's
# binning) + train — to match bonsai's CLI, which reads the CSV and fits bin
# mappers inside its timed fit. Both measure data loading, so the comparison
# is apples-to-apples.
def run_xgb(device: str, threads: int, test, y_test) -> dict:
    import xgboost as xgb
    cfg = tomllib.loads(CONFIG.read_text())
    params = {**rp.xgb_core(learning_rate=cfg["booster"]["learning_rate"],
                            max_depth=cfg["tree"]["max_depth"],
                            min_data_in_leaf=cfg["tree"]["min_data_in_leaf"],
                            lambda_l2=cfg["tree"]["lambda_l2"],
                            max_bin=cfg["bin_mapper"]["max_bin"],
                            seed=cfg["booster"]["random_seed"]),
              "objective": "reg:squarederror",
              "device": device, **({"nthread": threads} if device == "cpu" else {})}
    t0 = time.perf_counter()
    train = pd.read_csv(REPO / cfg["data"]["train"])
    feats = [c for c in train.columns if c != "label"]
    dtrain = xgb.DMatrix(train[feats], label=train["label"])
    booster = xgb.train(params, dtrain, num_boost_round=cfg["booster"]["n_iters"])
    fit_s = time.perf_counter() - t0
    dtest = xgb.DMatrix(test[feats])
    t0 = time.perf_counter()
    pred = booster.predict(dtest)
    predict_s = time.perf_counter() - t0
    return {"fit_s": round(fit_s, 2), "predict_s": round(predict_s, 2),
            "rmse": round(rmse(pred, y_test), 4), "profile": {}}


# LightGBM full pipeline (CSV read + Dataset binning + train), same fair
# timing as bonsai/xgboost. device="cuda" for GPU, "cpu" otherwise.
def run_lgbm(device: str, threads: int, test, y_test) -> dict:
    import lightgbm as lgb
    cfg = tomllib.loads(CONFIG.read_text())
    depth = cfg["tree"]["max_depth"]
    params = {**rp.lgbm_core(learning_rate=cfg["booster"]["learning_rate"],
                             max_depth=depth,
                             num_leaves=1 << depth,  # full depth-d tree
                             min_data_in_leaf=cfg["tree"]["min_data_in_leaf"],
                             lambda_l2=cfg["tree"]["lambda_l2"],
                             max_bin=cfg["bin_mapper"]["max_bin"],
                             seed=cfg["booster"]["random_seed"]),
              "objective": "regression", "metric": "rmse",
              "device_type": device, "num_threads": threads}
    t0 = time.perf_counter()
    train = pd.read_csv(REPO / cfg["data"]["train"])
    feats = [c for c in train.columns if c != "label"]
    dtrain = lgb.Dataset(train[feats], label=train["label"])
    model = lgb.train(params, dtrain, num_boost_round=cfg["booster"]["n_iters"])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(test[feats])
    predict_s = time.perf_counter() - t0
    return {"fit_s": round(fit_s, 2), "predict_s": round(predict_s, 2),
            "rmse": round(rmse(pred, y_test), 4), "profile": {}}


# CatBoost full pipeline (CSV read + Pool binning + train). task_type="GPU"
# for GPU. CatBoost caps GPU border_count at 254.
def run_catboost(device: str, threads: int, test, y_test) -> dict:
    from catboost import CatBoostRegressor, Pool
    cfg = tomllib.loads(CONFIG.read_text())
    t0 = time.perf_counter()
    train = pd.read_csv(REPO / cfg["data"]["train"])
    feats = [c for c in train.columns if c != "label"]
    pool = Pool(train[feats], label=train["label"].to_numpy())
    model = CatBoostRegressor(
        **rp.catboost_core(learning_rate=cfg["booster"]["learning_rate"],
                           max_depth=cfg["tree"]["max_depth"],
                           lambda_l2=cfg["tree"]["lambda_l2"],
                           max_bin=cfg["bin_mapper"]["max_bin"],
                           seed=cfg["booster"]["random_seed"], device=device),
        iterations=cfg["booster"]["n_iters"], loss_function="RMSE",
        task_type=("GPU" if device == "cuda" else "CPU"), devices="0",
        thread_count=threads, verbose=False)
    model.fit(pool)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(test[feats])
    predict_s = time.perf_counter() - t0
    return {"fit_s": round(fit_s, 2), "predict_s": round(predict_s, 2),
            "rmse": round(rmse(pred, y_test), 4), "profile": {}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16,
                    help="host threads for bonsai / xgboost CPU "
                         "(never 0 on many-core hosts: issue #2)")
    ap.add_argument("--variants", default="bonsai_gpu,bonsai_cpu,xgb_cpu,xgb_gpu")
    args = ap.parse_args()

    cfg = tomllib.loads(CONFIG.read_text())
    test = pd.read_csv(REPO / cfg["data"]["test"])
    y_test = test["label"].to_numpy()

    sha = sh(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"])
    try:
        gpu = sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines()[0]
    except Exception:
        gpu = "none"

    runners = {"bonsai_gpu": lambda: run_bonsai("cuda_depthwise", args.threads, y_test),
               "bonsai_cpu": lambda: run_bonsai("depthwise", args.threads, y_test),
               "bonsai_obl_gpu": lambda: run_bonsai("cuda_oblivious", args.threads, y_test),
               "bonsai_leaf_cpu": lambda: run_bonsai("leafwise", args.threads, y_test),
               "xgb_cpu": lambda: run_xgb("cpu", args.threads, test, y_test),
               "xgb_gpu": lambda: run_xgb("cuda", args.threads, test, y_test),
               "lgbm_cpu": lambda: run_lgbm("cpu", args.threads, test, y_test),
               "lgbm_gpu": lambda: run_lgbm("cuda", args.threads, test, y_test),
               "catboost_cpu": lambda: run_catboost("cpu", args.threads, test, y_test),
               "catboost_gpu": lambda: run_catboost("cuda", args.threads, test, y_test)}

    results, ts = {}, datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    for name in args.variants.split(","):
        print(f"running {name} ...", flush=True)
        results[name] = runners[name]()
        with RESULTS.open("a") as f:
            f.write(json.dumps({"schema": 1, "division": "perf", "suite": "gpu_msd",
                                "timing_mode": "pipeline", "dataset": "year_msd",
                                "task": "reg",
                                "ts": ts, "git_sha": sha, "gpu": gpu, "threads": args.threads,
                                "variant": name, **results[name]}) + "\n")

    ref = results.get("xgb_gpu", {}).get("fit_s")
    print(f"\n{sha} on {gpu} ({args.threads} threads)")
    print(f"{'variant':<12} {'fit_s':>7} {'rmse':>8} {'xgb-gpu gap':>12}  profile")
    for name, r in results.items():
        gap = f"{r['fit_s'] / ref:.1f}x" if ref else "-"
        prof = " ".join(f"{k.split('_', 1)[1]}={v}" for k, v in r["profile"].items()) or "-"
        print(f"{name:<12} {r['fit_s']:>7} {r['rmse']:>8} {gap:>12}  {prof}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
