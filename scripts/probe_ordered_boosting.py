# PROVENANCE NOTE (decision 69): this probe is a completed experiment whose
# committed evidence was produced by this exact code; it stays as-run rather
# than being refactored onto bonsai.bench. NEW probes must import their knobs
# and reference-library mappings from bonsai.bench.params and their metrics
# from bonsai.bench.metrics.
#!/usr/bin/env python3
"""Decompose catboost's 16M accuracy edge — the CPU-reproducible half.

Runs the feature-admission ladder from benchmarks/catboost-scale-edge-2026-07.md:
  --door ordered : catboost boosting_type Ordered vs Plain vs bonsai (is it the scheme?)
  --door bins    : bonsai oblivious across bin_mapper.n_samples (is it bin quality?)
  --door isolate : bonsai vs catboost on CPU at 16M (does bonsai actually trail on CPU?)

The GPU half — cuda_oblivious 0.8638 (pre-fix) -> 0.8749 (decision 63) vs its own
CPU 0.8749 — needs a device and is reproduced with the standard bench harness on
build-cuda/python; see the study note. All matched: depth 8, lr 0.1, 100 iters.

    python scripts/probe_ordered_boosting.py --door ordered --out results/ord.jsonl
"""
import argparse
import json
import sys
import time

sys.path.insert(0, "scripts")
sys.path.insert(0, "build/python")
import bench_scaling as bs
import numpy as np
import reference_params as rp


def r2(pred, y):
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def catboost(X, y, Xte, yte, iters, boosting_type):
    from catboost import CatBoostRegressor, Pool
    m = CatBoostRegressor(
        **rp.catboost_core(learning_rate=0.1, max_depth=8, lambda_l2=1.0,
                           max_bin=254, seed=42, device="cpu"),
        iterations=iters, loss_function="RMSE", task_type="CPU",
        thread_count=8, boosting_type=boosting_type, verbose=False)
    t0 = time.perf_counter()
    m.fit(Pool(X, label=y))
    return round(time.perf_counter() - t0, 2), round(r2(np.asarray(m.predict(Xte)), yte), 4)


def bonsai(X, y, Xte, yte, iters, grower, n_samples=200_000):
    import bonsai as bn
    pairs = [("dispatch.grower_name", grower), ("dispatch.objective_name", "mse"),
             ("booster.n_iters", str(iters)), ("booster.learning_rate", "0.1"),
             ("booster.random_seed", "42"), ("tree.max_depth", "8"),
             ("tree.max_leaves", "256"), ("tree.min_data_in_leaf", "20"),
             ("tree.lambda_l2", "1.0"), ("bin_mapper.max_bin", "255"),
             ("bin_mapper.n_samples", str(n_samples)), ("parallel.n_threads", "8")]
    t0 = time.perf_counter()
    m = bn.train(pairs, X, y)
    return round(time.perf_counter() - t0, 2), round(r2(np.asarray(m.predict(Xte)), yte), 4)


def emit(out, rec):
    out.append(rec)
    print("ROW " + json.dumps(rec), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--door", choices=["ordered", "bins", "isolate"], required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out: list[dict] = []

    if args.door == "ordered":
        for rows in (200_000, 1_000_000):
            X, y, Xte, yte = bs.gen_data(rows, 100, 42, 50_000, 20)
            for iters in (100, 200):
                trials = [("catboost_ordered", catboost(X, y, Xte, yte, iters, "Ordered")),
                          ("catboost_plain", catboost(X, y, Xte, yte, iters, "Plain")),
                          ("bonsai_oblivious", bonsai(X, y, Xte, yte, iters, "oblivious"))]
                for name, (fs, r) in trials:
                    emit(out, {"door": "ordered", "rows": rows, "iters": iters,
                               "learner": name, "fit_s": fs, "r2_test": r})
    elif args.door == "bins":
        for rows in (1_000_000, 4_000_000):
            X, y, Xte, yte = bs.gen_data(rows, 100, 42, 50_000, 20)
            fs, r = catboost(X, y, Xte, yte, 100, "Plain")
            emit(out, {"door": "bins", "rows": rows, "learner": "catboost_plain",
                       "n_samples": None, "fit_s": fs, "r2_test": r})
            for ns in (200_000, 1_000_000, rows):
                if ns > rows:
                    continue
                fs, r = bonsai(X, y, Xte, yte, 100, "oblivious", ns)
                emit(out, {"door": "bins", "rows": rows, "learner": "bonsai_oblivious",
                           "n_samples": ns, "fit_s": fs, "r2_test": r})
    else:  # isolate
        X, y, Xte, yte = bs.gen_data(16_000_000, 100, 42, 100_000, 20)
        fs, r = catboost(X, y, Xte, yte, 100, "Plain")
        emit(out, {"door": "isolate", "rows": 16_000_000, "learner": "catboost_plain",
                   "fit_s": fs, "r2_test": r})
        for grower in ("oblivious", "depthwise"):
            fs, r = bonsai(X, y, Xte, yte, 100, grower)
            emit(out, {"door": "isolate", "rows": 16_000_000,
                       "learner": f"bonsai_{grower}", "fit_s": fs, "r2_test": r})

    if args.out:
        with open(args.out, "w") as fh:
            for rec in out:
                fh.write(json.dumps(rec) + "\n")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
