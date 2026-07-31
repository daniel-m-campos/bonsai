"""One variant registry for every benchmark suite.

Variant names are the row keys committed results carry forever, so the
canonical names, the per-suite membership tuples, and the historical alias
spellings live here in one table; the guard test pins every string.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Variant:
    name: str
    lib: str  # bonsai | xgb | lgbm | catboost
    device: str  # cpu | cuda
    aliases: tuple[str, ...] = ()


# Aliases are the spellings older suites committed: grinsztajn's short names
# and the retired gpu_msd track's *_gpu/*_cpu scheme.
_TABLE = (
    Variant("bonsai_depthwise", "bonsai", "cpu", ("bonsai_dw", "bonsai_cpu")),
    Variant("bonsai_leafwise", "bonsai", "cpu", ("bonsai_lw", "bonsai_leaf_cpu")),
    Variant("bonsai_oblivious", "bonsai", "cpu", ("bonsai_obl",)),
    Variant("bonsai_cuda_depthwise", "bonsai", "cuda", ("bonsai_gpu",)),
    Variant("bonsai_cuda_oblivious", "bonsai", "cuda", ("bonsai_obl_gpu",)),
    Variant("bonsai_ts_depthwise", "bonsai", "cpu"),
    Variant("bonsai_ts_oblivious", "bonsai", "cpu"),
    Variant("bonsai_ts_cuda_depthwise", "bonsai", "cuda"),
    Variant("bonsai_ts_cuda_oblivious", "bonsai", "cuda"),
    Variant("xgb_hist", "xgb", "cpu", ("xgb", "xgb_cpu")),
    Variant("xgb_cuda", "xgb", "cuda", ("xgb_gpu",)),
    Variant("lgbm_cpu", "lgbm", "cpu", ("lgbm",)),
    Variant("lgbm_cuda", "lgbm", "cuda"),
    Variant("catboost_cpu", "catboost", "cpu", ("catboost",)),
    Variant("catboost_gpu", "catboost", "cuda"),
)

REGISTRY = {v.name: v for v in _TABLE}
_BY_ALIAS = {a: v for v in _TABLE for a in v.aliases}

# Per-suite membership, in each suite's historical order and spelling.
SCALING = ("bonsai_depthwise", "bonsai_leafwise", "bonsai_oblivious",
           "bonsai_cuda_depthwise", "bonsai_cuda_oblivious", "xgb_hist",
           "xgb_cuda", "lgbm_cpu", "lgbm_cuda", "catboost_cpu", "catboost_gpu")
AIRLINE = ("bonsai_depthwise", "bonsai_oblivious", "bonsai_cuda_depthwise",
           "bonsai_cuda_oblivious", "bonsai_ts_depthwise", "bonsai_ts_oblivious",
           "bonsai_ts_cuda_depthwise", "bonsai_ts_cuda_oblivious", "xgb_hist",
           "xgb_cuda", "lgbm_cpu", "catboost_cpu", "catboost_gpu")
GRINSZTAJN = ("bonsai_dw", "bonsai_lw", "bonsai_obl", "xgb", "lgbm", "catboost")


def resolve(name: str) -> Variant:
    """Canonical name or alias -> Variant; raises KeyError on unknowns."""
    v = REGISTRY.get(name) or _BY_ALIAS.get(name)
    if v is None:
        raise KeyError(f"unknown variant {name!r}")
    return v


def names(device: str | None = None) -> list[str]:
    return [v.name for v in _TABLE if device in (None, v.device)]
