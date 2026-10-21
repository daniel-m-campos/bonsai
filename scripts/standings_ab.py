#!/usr/bin/env python3
"""One anchor-cell fit for the standings refresh A/B (decision 92).

    python scripts/standings_ab.py --rows N --cols N --grower G --arm anchor|old|new

The refresh pod interleaves invocations of the previous release's wheel
(old), one fixed anchor release's wheel (anchor) and the HEAD build (new)
on ONE pod; same-pod interleaving is the only timing comparison the
fleet-spread rule allows. Prints one JSON line; check_standings.py takes
the min per (cell, grower, arm), the statistic a one-sided noise leaves
honest, and calls a move beyond +-5% against old or +-2% against the
anchor, the band that catches a loss compounding inside the first.

The fit runs through bonsai.bench.runners.worker, the same code the standings
execute, so the detector cannot measure a path the standings do not: a change
to the harness call form moves the verdict exactly as a change to the library
would. Each arm resolves that runner from its own build, so a harness change
between vintages is visible too. Both fit_s and peak RSS are reported: an RSS
regression is a perf regression.

`--fused` fits the same cell through bonsai's one-call train(pairs, X, y)
instead of the two-step Dataset + train form the runner otherwise uses. Two
arms differing only in that flag are the parity check behind the published
ingest/train split: they must agree, because a two-step Dataset that lost its
device hint would bin on the host and post a plausible ingest number for a
pipeline no cuda grower runs. One process per arm, so peak RSS is that arm's
own.

Every row names the CPU it was fitted on: a cell whose arms were measured on
two hosts compares pods, not builds, and check_standings.py refuses it. The
read is local rather than bonsai.bench.runlog's because the old and anchor
arms import the wheel's own bench, which may predate that helper.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import sys

# The standings-rows anchor knobs. Restated rather than loaded from the spec
# because the old arm would load the wheel's copy of it; a drift guard lives
# in python/tests/bench/test_runners.py.
ANCHOR_KNOBS = {
    "lr": 0.1,
    "depth": 8,
    "bins": 255,
    "seed": 42,
    "min_data_in_leaf": 20,
    "lambda_l2": 1.0,
    "informative": 20,
    "n_test": 1000,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--grower", default="cuda_depthwise")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument(
        "--fused", action="store_true", help="fit through train(pairs, X, y) (parity arm)"
    )
    ap.add_argument(
        "--skip-without-cuda",
        action="store_true",
        help="print a skipped row instead of failing off-GPU",
    )
    args = ap.parse_args()

    import bonsai
    from bonsai.bench import runners

    if args.skip_without_cuda and args.grower.startswith("cuda") and not bonsai.cuda_available():
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "grower": args.grower,
                    "skipped": "no CUDA build or no visible device",
                }
            )
        )
        return 0
    cell = dict(ANCHOR_KNOBS, rows=args.rows, cols=args.cols, iters=args.iters)
    out = runners.worker(
        {
            "cell": cell,
            "variant": f"bonsai_{args.grower}",
            "threads": args.threads,
            "fused": args.fused,
        }
    )
    print(
        json.dumps(
            {
                "arm": args.arm,
                "rows": args.rows,
                "cols": args.cols,
                "grower": args.grower,
                "fit_s": out["fit_s"],
                "ingest_s": out.get("ingest_s"),
                "train_s": out.get("train_s"),
                "peak_rss_gb": out.get("peak_rss_gb"),
                "r2_test": out.get("r2_test"),
                "version": getattr(bonsai, "__version__", "source"),
                "module": bonsai.__file__,
                "host": {"cpu_model": cpu_model()},
            }
        )
    )
    return 0


def cpu_model() -> str:
    """The CPU's marketing name, as /proc/cpuinfo or sysctl spells it."""
    if platform.system() == "Darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        return out.stdout.strip()
    for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return ""


if __name__ == "__main__":
    sys.exit(main())
