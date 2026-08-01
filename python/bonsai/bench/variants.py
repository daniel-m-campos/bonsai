"""One variant registry for every benchmark suite.

Variant names are the row keys committed results carry forever, so the
canonical names, the per-suite membership tuples, and the historical alias
spellings live here in one table; the guard test pins every string.
"""

from __future__ import annotations

import dataclasses
from typing import Final


class Lib:
    """Library identifiers; values are carried by committed result rows."""

    BONSAI: Final = "bonsai"
    XGB: Final = "xgb"
    LGBM: Final = "lgbm"
    CATBOOST: Final = "catboost"


class Device:
    """Execution devices; values are carried by committed result rows."""

    CPU: Final = "cpu"
    CUDA: Final = "cuda"


@dataclasses.dataclass(frozen=True)
class Variant:
    """One benchmark arm: canonical name, library, device, old spellings."""

    name: str
    lib: str  # Lib values
    device: str  # Device values
    aliases: tuple[str, ...] = ()


# Aliases are the spellings older suites committed: grinsztajn's short names
# and the retired gpu_msd track's *_gpu/*_cpu scheme.
_TABLE = (
    Variant("bonsai_depthwise", Lib.BONSAI, Device.CPU, ("bonsai_dw", "bonsai_cpu")),
    Variant("bonsai_leafwise", Lib.BONSAI, Device.CPU, ("bonsai_lw", "bonsai_leaf_cpu")),
    Variant("bonsai_oblivious", Lib.BONSAI, Device.CPU, ("bonsai_obl",)),
    Variant("bonsai_cuda_depthwise", Lib.BONSAI, Device.CUDA, ("bonsai_gpu",)),
    Variant("bonsai_cuda_oblivious", Lib.BONSAI, Device.CUDA, ("bonsai_obl_gpu",)),
    Variant("bonsai_ts_depthwise", Lib.BONSAI, Device.CPU),
    Variant("bonsai_ts_oblivious", Lib.BONSAI, Device.CPU),
    Variant("bonsai_ts_cuda_depthwise", Lib.BONSAI, Device.CUDA),
    Variant("bonsai_ts_cuda_oblivious", Lib.BONSAI, Device.CUDA),
    Variant("xgb_hist", Lib.XGB, Device.CPU, ("xgb", "xgb_cpu")),
    Variant("xgb_cuda", Lib.XGB, Device.CUDA, ("xgb_gpu",)),
    Variant("lgbm_cpu", Lib.LGBM, Device.CPU, ("lgbm",)),
    Variant("lgbm_cuda", Lib.LGBM, Device.CUDA),
    Variant("catboost_cpu", Lib.CATBOOST, Device.CPU, ("catboost",)),
    Variant("catboost_gpu", Lib.CATBOOST, Device.CUDA),
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
    """Canonical variant names, optionally filtered by device."""
    return [v.name for v in _TABLE if device in (None, v.device)]
