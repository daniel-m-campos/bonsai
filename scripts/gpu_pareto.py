#!/usr/bin/env python3
"""GPU accuracy-vs-time frontier at a single large cell.

Sweeps each library's iteration count at a fixed (rows, cols, depth, lr) and
records (fit_s, test r2), so the frontier separates "fast per round" from
"converges in fewer rounds" — the distinction a fixed-iteration table hides.
Reuses bench_scaling's data generator and per-library runners so configs match
the scaling suite exactly. Run all variants in ONE process on ONE pod: only
same-pod points compare (identical GPUs measure up to ~25% apart).

    PYTHONPATH=build-cuda/python python scripts/gpu_pareto.py --out frontier.jsonl

Produces benchmarks/results/gpu-pareto-16M-2026-07.jsonl; the analysis lives in
benchmarks/gpu-pareto-16M-2026-07.md.
"""
import argparse
import json
import sys
import time

sys.path.insert(0, "scripts")
import bench_scaling as bs

# variant -> iteration ladder. Reference libs get a longer ladder because they
# converge past bonsai's accuracy and the frontier needs their far points.
LADDERS = {
    "bonsai_cuda_depthwise": [60, 80, 100, 130],
    "bonsai_cuda_oblivious": [60, 80, 100, 130, 160],
    "xgb_cuda": [100, 150, 200, 300],
    "catboost_gpu": [100, 150, 200, 300, 450],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=16_000_000)
    ap.add_argument("--cols", type=int, default=100)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--bins", type=int, default=255)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-test", type=int, default=100_000)
    ap.add_argument("--informative", type=int, default=20)
    ap.add_argument("--variants", default=",".join(LADDERS))
    ap.add_argument("--out", default="benchmarks/results/gpu-pareto-16M-2026-07.jsonl")
    args = ap.parse_args()

    print("generating data...", file=sys.stderr, flush=True)
    X, y, Xte, yte = bs.gen_data(args.rows, args.cols, args.seed, args.n_test,
                                 args.informative)

    def cell(iters: int) -> dict:
        # run_bonsai reads c["bins"]; the reference runners read bins_effective
        # (their GPU 254-bin cap), exactly as bench_scaling's parent sets it.
        return {"rows": args.rows, "cols": args.cols, "seed": args.seed,
                "n_test": args.n_test, "informative": args.informative,
                "iters": iters, "lr": args.lr, "depth": args.depth,
                "bins": args.bins, "bins_effective": min(args.bins, 254)}

    results = []
    for variant in args.variants.split(","):
        lib = bs.VARIANTS[variant][0]
        run = bs.RUNNERS[lib]
        # one untimed warmup absorbs CUDA context creation + PTX JIT.
        warm = {"variant": variant, "threads": 16,
                "cell": dict(cell(5), rows=8192, n_test=1024)}
        try:
            run(warm, X[:8192], y[:8192], Xte[:1024], yte[:1024])
        except Exception as exc:
            print(f"warm {variant} failed: {exc}", file=sys.stderr, flush=True)
        for iters in LADDERS[variant]:
            spec = {"variant": variant, "threads": 16, "cell": cell(iters)}
            try:
                out = run(spec, X, y, Xte, yte)
            except Exception as exc:
                print(f"{variant}@{iters} FAILED: {exc}", file=sys.stderr, flush=True)
                continue
            rec = {"variant": variant, "iters": iters,
                   "fit_s": round(out["fit_s"], 2),
                   "r2_test": round(out["r2_test"], 4),
                   "r2_train": round(out["r2_train"], 4)}
            results.append(rec)
            print("ROW " + json.dumps(rec), flush=True)
            time.sleep(0)  # keep flushing under buffering

    with open(args.out, "w") as fh:
        for rec in results:
            fh.write(json.dumps(rec) + "\n")
    print(f"wrote {len(results)} rows to {args.out}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
