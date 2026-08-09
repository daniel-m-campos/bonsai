#!/usr/bin/env python3
"""Fetch a benchmark dataset by registry name.

The fetch logic lives in the dataset registry (decision 69); this loads
bonsai/bench/datasets.py by file path so it works before the native module is
built. `fetch` itself validates the name, so this is only the argv plumbing.

    python3 scripts/fetch.py higgs
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

_p = pathlib.Path(__file__).resolve().parents[1] / "python/bonsai/bench/datasets.py"
_spec = importlib.util.spec_from_file_location("bench_datasets", _p)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["bench_datasets"] = _mod
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: fetch.py <a9a|covtype|higgs|year_msd>", file=sys.stderr)
        sys.exit(2)
    for path in _mod.fetch(sys.argv[1]):
        print(path)
