#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=1.26", "pandas>=2.2", "scikit-learn>=1.4"]
# ///
"""Fetch a benchmark dataset by registry name.

The fetch logic lives in the dataset registry (decision 69); this loads
bonsai/bench/datasets.py by file path so it works before the native module is
built, which is what the C++ test targets need for the test-pin datasets.
`fetch` itself validates the name, so this is only the argv plumbing.

    uv run scripts/fetch.py california     # the toy dataset the C++ tests pin
    uv run scripts/fetch.py amazon         # the categorical pin
    uv run scripts/fetch.py higgs          # a perf-scale ladder input

The two test-pin fetchers need scikit-learn and pandas, which the inline
script metadata above supplies to `uv run`; the rest are stdlib streams.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

_p = pathlib.Path(__file__).resolve().parents[1] / "python/bonsai/bench/datasets.py"
_spec = importlib.util.spec_from_file_location("bench_datasets", _p)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["bench_datasets"] = _mod
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        names = "|".join(sorted(_mod.fetchable()))
        print(f"usage: fetch.py <{names}>", file=sys.stderr)
        sys.exit(2)
    for path in _mod.fetch(sys.argv[1]):
        print(path)
