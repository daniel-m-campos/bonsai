"""Tests for bonsai.bench.variants (the registry pins)."""

from __future__ import annotations

import pytest
from bonsai.bench import grinsztajn, variants


def test_variant_registry():
    # Committed rows pin these exact spellings forever.
    assert variants.SCALING == (
        "bonsai_depthwise", "bonsai_leafwise", "bonsai_levelwise",
        "bonsai_cuda_depthwise", "bonsai_cuda_levelwise", "xgb_hist",
        "xgb_cuda", "lgbm_cpu", "lgbm_cuda", "catboost_cpu", "catboost_gpu")
    assert grinsztajn.VARIANTS == variants.GRINSZTAJN == (
        "bonsai_dw", "bonsai_lw", "bonsai_obl", "xgb", "lgbm", "catboost")
    for n in variants.SCALING:
        assert variants.resolve(n).name == n
    # Historical alias spellings resolve to the intended canonical arm.
    for alias, canon in {"bonsai_dw": "bonsai_depthwise",
                         "bonsai_lw": "bonsai_leafwise",
                         "bonsai_obl": "bonsai_levelwise",
                         "xgb": "xgb_hist", "lgbm": "lgbm_cpu",
                         "catboost": "catboost_cpu",
                         "bonsai_cpu": "bonsai_depthwise",
                         "bonsai_leaf_cpu": "bonsai_leafwise",
                         "bonsai_gpu": "bonsai_cuda_depthwise",
                         "bonsai_obl_gpu": "bonsai_cuda_levelwise",
                         "xgb_cpu": "xgb_hist",
                         "xgb_gpu": "xgb_cuda"}.items():
        assert variants.resolve(alias).name == canon
    # Registered outside the scaling suite: the device leafwise arm runs
    # from specs only (issue #268), so the suite tuple above stays fixed.
    assert variants.resolve("bonsai_cuda_leafwise").device == variants.Device.CUDA
    # No alias shadows a canonical name.
    aliases = {a for v in variants.REGISTRY.values() for a in v.aliases}
    assert not aliases & set(variants.REGISTRY)
    # The (lib, device) pairs survive the derivation.
    assert (variants.resolve("bonsai_cuda_levelwise").lib,
            variants.resolve("bonsai_cuda_levelwise").device) == ("bonsai", "cuda")
    assert (variants.resolve("lgbm_cuda").lib,
            variants.resolve("lgbm_cuda").device) == ("lgbm", "cuda")
    with pytest.raises(KeyError):
        variants.resolve("nope")
    # The retired spellings are gone with no alias behind them.
    for retired in ("bonsai_oblivious", "bonsai_cuda_oblivious",
                    "bonsai_ts_depthwise", "bonsai_ts_cuda_depthwise"):
        with pytest.raises(KeyError):
            variants.resolve(retired)
