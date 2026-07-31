#!/usr/bin/env python3
"""One anchor-cell fit for the standings refresh A/B (decision 92).

    python scripts/standings_ab.py --rows N --cols N --grower G --arm old|new

The refresh workflow interleaves invocations of the previous release's wheel
(no PYTHONPATH) and the HEAD build (PYTHONPATH=build-cuda/python) on ONE pod;
same-pod interleaving is the only timing comparison the fleet-spread rule
allows. Prints one JSON line; the workflow aggregates medians per (cell,
grower, arm) and calls a move beyond +-5%.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--grower", default="cuda_depthwise")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    import bonsai
    from bonsai.bench.synth import gen_data

    X, y, _, _ = gen_data(args.rows, args.cols, 42, 1000, 20)
    pairs = [("dispatch.grower_name", args.grower),
             ("dispatch.objective_name", "mse"),
             ("booster.n_iters", str(args.iters)),
             ("booster.learning_rate", "0.1"),
             ("booster.random_seed", "42"),
             ("tree.max_depth", "8"), ("tree.max_leaves", "256"),
             ("tree.min_data_in_leaf", "20"), ("tree.lambda_l2", "1.0"),
             ("bin_mapper.max_bin", "255"),
             ("parallel.n_threads", str(args.threads))]
    # Warm fit absorbs CUDA context + JIT so the timed fit is steady state.
    bonsai.train([*pairs[:2], ("booster.n_iters", "3"), *pairs[3:]],
                 X[:8192], y[:8192])
    t0 = time.perf_counter()
    bonsai.train(pairs, X, y)
    fit_s = time.perf_counter() - t0
    print(json.dumps({"arm": args.arm, "rows": args.rows, "cols": args.cols,
                      "grower": args.grower, "fit_s": round(fit_s, 3),
                      "version": getattr(bonsai, "__version__", "source")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
