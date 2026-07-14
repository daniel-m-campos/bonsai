# /// script
# requires-python = "==3.12.*"
# dependencies = ["numpy>=1.26", "xgboost>=2.0", "lightgbm>=4.3", "catboost>=1.2"]
# ///
"""Shim: the scaling suite lives in bonsai.bench.scaling (decision 69).

Run with the built module on the path, exactly as before:
    PYTHONPATH=build/python uv run scripts/bench_scaling.py --smoke
or, equivalently: python -m bonsai.bench.scaling --smoke
"""
import sys

from bonsai.bench.scaling import (  # noqa: F401  (re-exports for importers)
    RUNNERS,
    VARIANTS,
    gen_data,
    main,
)

if __name__ == "__main__":
    sys.exit(main())
