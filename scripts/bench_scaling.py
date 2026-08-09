# /// script
# requires-python = "==3.12.*"
# dependencies = ["numpy>=1.26", "xgboost>=2.0", "lightgbm>=4.3", "catboost>=1.2"]
# ///
"""Shim: the bench CLI lives in bonsai.bench.cli (decision 69).

Run with the built module on the path:
    PYTHONPATH=build/python uv run scripts/bench_scaling.py run --spec scaling-rows
or, equivalently: python -m bonsai.bench run --spec scaling-rows

The scaling ladders are bundled specs (`specs` lists them): scaling-rows,
scaling-cols, scaling-bins, scaling-threads.
"""
from __future__ import annotations

import sys

from bonsai.bench.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    # The pod runbook and the pod-validate skill drive the child protocol as
    # `--worker`; the CLI spells it as a subcommand.
    if argv == ["--worker"]:
        argv = ["worker"]
    sys.exit(main(argv))
