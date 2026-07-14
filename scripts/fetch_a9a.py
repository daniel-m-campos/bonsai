#!/usr/bin/env python3
"""Shim: fetch logic lives in the dataset registry (decision 69).
Loads bonsai/bench/datasets.py by file path so it works before the native
module is built. Run: python3 scripts/fetch_a9a.py"""
import importlib.util
import pathlib
import sys

_p = pathlib.Path(__file__).resolve().parents[1] / "python/bonsai/bench/datasets.py"
_spec = importlib.util.spec_from_file_location("bench_datasets", _p)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["bench_datasets"] = _mod
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    for path in _mod.fetch("a9a"):
        print(path)
