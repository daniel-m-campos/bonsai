#!/usr/bin/env python3
"""Issue #155 probe: does closing the missing-bin leak move any metric?

Fitted bins leak finite training values into the NaN sentinel in three ways
(stride top tail, greedy final group, rows above the 200k-sample max). The
branch adds a FLT_MAX top-band closer behind the BONSAI_BIN_CLOSER env
toggle; this probe A/Bs bonsai against itself, off vs on, at campaign knobs
(200 iters, lr 0.05, depth 6, 255 bins, seed 42) on both growers.

Cases:
- three targeted synthetics, one per leak mechanism (capped-max heavy value,
  rare top-tail signal, signal above the sampled max at 1M rows)
- four real datasets (california r2, amazon AUC, higgs-500k AUC which also
  exercises the sample-cap path, airline-0.1m AUC)

Each (case, grower, arm) runs in its own child process because the toggle is
read once per process. Verdict bar: decision-55 chance band (~+-0.001 AUC,
~+-0.002 r2 at these sizes); deltas inside the band = refuted.

    PYTHONPATH=build/python python scripts/probe_missing_bin.py
"""
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "tests" / "data"
KNOBS = [("booster.n_iters", "200"), ("booster.learning_rate", "0.05"),
         ("booster.random_seed", "42"), ("tree.max_depth", "6"),
         ("tree.max_leaves", "64"), ("tree.min_data_in_leaf", "20"),
         ("tree.lambda_l2", "1.0"), ("bin_mapper.max_bin", "255"),
         ("parallel.n_threads", "8")]


FILE_STEM = {"california": "california_housing"}


def load_csv(name: str):
    import pandas as pd
    stem = FILE_STEM.get(name, name)
    tr = pd.read_csv(DATA / f"{stem}_train.csv")
    te = pd.read_csv(DATA / f"{stem}_test.csv")
    X = tr.drop(columns=["label"]).to_numpy(dtype=np.float32)
    y = tr["label"].to_numpy(dtype=np.float32)
    Xte = te.drop(columns=["label"]).to_numpy(dtype=np.float32)
    yte = te["label"].to_numpy(dtype=np.float32)
    return np.ascontiguousarray(X), y, np.ascontiguousarray(Xte), yte


def synth_capped(rng):
    """Greedy-path leak: a capped sensor column whose cap run is heavy AND
    predictive. Off-arm, every capped row trains as missing."""
    n = 100_000
    raw = rng.lognormal(0.0, 1.0, n).astype(np.float32)
    cap = np.quantile(raw, 0.90).astype(np.float32)
    x0 = np.minimum(raw, cap)
    X = np.column_stack([x0] + [rng.random(n, dtype=np.float32) for _ in range(4)])
    y = (2.0 * (x0 >= cap) + x0 / 3.0 + rng.normal(0, 0.1, n)).astype(np.float32)
    return X.astype(np.float32), y


def synth_tail(rng):
    """Stride-path leak: the signal lives in the top 0.2% of a smooth column,
    inside the tail the stride never closes."""
    n = 100_000
    x0 = rng.random(n, dtype=np.float32)
    thresh = np.quantile(x0, 0.998)
    X = np.column_stack([x0] + [rng.random(n, dtype=np.float32) for _ in range(4)])
    y = (5.0 * (x0 > thresh) + x0 + rng.normal(0, 0.1, n)).astype(np.float32)
    return X.astype(np.float32), y


def synth_bigsample(rng):
    """Sample-cap leak: 1M rows > the 200k bin sample; signal on values
    beyond what the sample can see."""
    n = 1_000_000
    x0 = rng.lognormal(0.0, 1.5, n).astype(np.float32)
    thresh = np.quantile(x0, 0.9995)
    X = np.column_stack([x0, rng.random(n, dtype=np.float32)])
    y = (4.0 * (x0 > thresh) + np.log1p(x0) + rng.normal(0, 0.1, n)).astype(np.float32)
    return X.astype(np.float32), y


SYNTH = {"synth_capped": synth_capped, "synth_tail": synth_tail,
         "synth_bigsample": synth_bigsample}
REAL = {"california": "r2", "amazon": "auc", "higgs": "auc"}


def load_case(case: str):
    if case in SYNTH:
        rng = np.random.default_rng(42)
        X, y = SYNTH[case](rng)
        n_te = len(y) // 5
        return X[n_te:], y[n_te:], X[:n_te], y[:n_te], "r2"
    if case == "airline_0.1m":
        z = np.load(DATA / "airline_0.1m.npz")
        return z["X"], z["y"], z["Xte"], z["yte"], "auc"
    X, y, Xte, yte = load_csv(case)
    return X, y, Xte, yte, REAL[case]


def worker(case: str, grower: str) -> dict:
    sys.path.insert(0, str(REPO / "build" / "python"))
    import bonsai
    from bonsai.bench.metrics import auc, r2
    X, y, Xte, yte, metric = load_case(case)
    objective = "logloss" if metric == "auc" else "mse"
    pairs = [("dispatch.grower_name", grower),
             ("dispatch.objective_name", objective), *KNOBS]
    model = bonsai.train(pairs, X, y)
    pred = np.asarray(model.predict(Xte))
    score = auc(yte, pred) if metric == "auc" else r2(yte, pred)
    return {"case": case, "grower": grower, "metric": metric,
            "score": round(float(score), 5)}


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        print("RESULT " + json.dumps(worker(sys.argv[2], sys.argv[3])), flush=True)
        return 0

    cases = list(SYNTH) + list(REAL) + ["airline_0.1m"]
    rows = []
    for case in cases:
        for grower in ("depthwise", "oblivious"):
            scores = {}
            for arm in ("off", "on"):
                env = dict(os.environ)
                env.pop("BONSAI_BIN_CLOSER", None)
                if arm == "on":
                    env["BONSAI_BIN_CLOSER"] = "1"
                proc = subprocess.run(
                    [sys.executable, __file__, "--worker", case, grower],
                    capture_output=True, text=True, env=env, timeout=1800)
                line = next((ln for ln in proc.stdout.splitlines()
                             if ln.startswith("RESULT ")), None)
                if line is None:
                    print(f"{case}/{grower}/{arm} FAILED:\n{proc.stderr[-400:]}",
                          flush=True)
                    scores[arm] = None
                    continue
                scores[arm] = json.loads(line.removeprefix("RESULT "))["score"]
            delta = (None if None in scores.values()
                     else round(scores["on"] - scores["off"], 5))
            rows.append((case, grower, scores["off"], scores["on"], delta))
            print(f"{case:18s} {grower:10s} off={scores['off']} "
                  f"on={scores['on']} delta={delta}", flush=True)

    print("\n| case | grower | off | on | delta |")
    print("|---|---|--:|--:|--:|")
    for case, grower, off, on, delta in rows:
        print(f"| {case} | {grower} | {off} | {on} | {delta} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
