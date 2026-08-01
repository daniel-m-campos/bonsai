"""Tests for bonsai.bench.cli (plan laziness)."""

from __future__ import annotations

import subprocess
import sys


def test_cli_plan_is_lazy():
    """Planning must not import the reference libraries.

    Runs in a fresh interpreter so the check cannot depend on what other
    tests imported first (the old in-process check was order-fragile).
    """
    code = """
import json, pathlib, sys, tempfile
from bonsai.bench import cli
with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td) / "s.json"
    p.write_text(json.dumps(
        {"name": "t", "cells": [{"rows": 1000, "cols": 10}],
         "variants": ["bonsai_depthwise", "xgb_hist"]}))
    assert cli.main(["plan", "--spec", str(p)]) == 0
for heavy in ("xgboost", "lightgbm", "catboost"):
    assert heavy not in sys.modules, f"{heavy} imported by plan"
"""
    subprocess.run([sys.executable, "-c", code], check=True)
