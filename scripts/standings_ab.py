#!/usr/bin/env python3
"""One anchor-cell fit for the standings refresh A/B (decision 92).

    python scripts/standings_ab.py --rows N --cols N --grower G --arm old|new

The refresh workflow interleaves invocations of the previous release's wheel
(no PYTHONPATH) and the HEAD build (PYTHONPATH=build-cuda/python) on ONE pod;
same-pod interleaving is the only timing comparison the fleet-spread rule
allows. Prints one JSON line; the workflow aggregates medians per (cell,
grower, arm) and calls a move beyond +-5%.

The fit runs through bonsai.bench.runners.worker, the same code the standings
execute, so the detector cannot measure a path the standings do not: a change
to the harness call form moves the verdict exactly as a change to the library
would. Each arm resolves that runner from its own build, so a harness change
between vintages is visible too. Both fit_s and peak RSS are reported: an RSS
regression is a perf regression.
"""

from __future__ import annotations

import argparse
import json
import sys

# The standings-rows anchor knobs. Restated rather than loaded from the spec
# because the old arm would load the wheel's copy of it; a drift guard lives
# in python/tests/bench/test_runners.py.
ANCHOR_KNOBS = {"lr": 0.1, "depth": 8, "bins": 255, "seed": 42,
                "min_data_in_leaf": 20, "lambda_l2": 1.0, "informative": 20,
                "n_test": 1000}


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
    from bonsai.bench import runners

    cell = dict(ANCHOR_KNOBS, rows=args.rows, cols=args.cols,
                iters=args.iters)
    out = runners.worker({"cell": cell, "variant": f"bonsai_{args.grower}",
                          "threads": args.threads})
    print(json.dumps({"arm": args.arm, "rows": args.rows, "cols": args.cols,
                      "grower": args.grower, "fit_s": out["fit_s"],
                      "peak_rss_gb": out.get("peak_rss_gb"),
                      "r2_test": out.get("r2_test"),
                      "version": getattr(bonsai, "__version__", "source"),
                      "module": bonsai.__file__}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
