"""Shim: the knob mappings live in bonsai.bench.params (decision 69).

Requires the built module on the path (PYTHONPATH=build/python), which every
harness that trains bonsai already needs.
"""
from bonsai.bench.params import (  # noqa: F401
    catboost_core,
    lgbm_core,
    xgb_core,
)
