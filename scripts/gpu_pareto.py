#!/usr/bin/env python3
"""GPU accuracy-vs-time frontier at a single large cell.

Sweeps each library's iteration count at a fixed (rows, cols, depth, lr) and
records (fit_s, test r2), so the frontier separates "fast per round" from
"converges in fewer rounds", the distinction a fixed-iteration table hides.
The cell and per-variant iteration ladders live in the bundled spec
gpu-pareto-16M (bench/specs/); this script is a thin shim that drives
the unified CLI (python -m bonsai.bench run), which appends to --out and
resumes instead of truncating. Run all variants in ONE process on ONE pod:
only same-pod points compare (identical GPUs measure up to 25% apart).

    PYTHONPATH=build-cuda/python python scripts/gpu_pareto.py

The 2026-07 campaign rows are benchmarks/results/gpu-pareto-16M-2026-07.jsonl;
the analysis lives in benchmarks/gpu-pareto-16M-2026-07.md.
"""
import argparse
import datetime
import sys

from bonsai.bench import cli


def main() -> int:
    day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=f"benchmarks/results/gpu-pareto-16M-{day}.jsonl")
    ap.add_argument("--host-name", default=None)
    args = ap.parse_args()
    argv = ["run", "--spec", "gpu-pareto-16M", "--out", args.out]
    if args.host_name:
        argv += ["--host-name", args.host_name]
    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main())
