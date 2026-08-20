"""Typed parameter overrides: ``Params`` and one dataclass per config section.

The stable import path over the generated module: the dataclasses in
``bonsai._params`` are rendered at build time from the C++ section registry
(one field per dotted config key, ``None`` = leave the library default), so
their fields track the registry with no hand-written mirror.

Examples
--------
>>> from bonsai.params import Params, Tree, Booster
>>> base = Params(tree=Tree(max_depth=8), booster=Booster(n_iters=200))
>>> bonsai.train(base, ds)                                # doctest: +SKIP
>>> bonsai.train(base | {"tree.max_depth": 10}, ds)       # doctest: +SKIP
"""

from __future__ import annotations

from bonsai._params import (
    BinMapper,
    Booster,
    Data,
    Dispatch,
    Metrics,
    Objective,
    Parallel,
    Params,
    Sampler,
    Tree,
)

__all__ = [
    "BinMapper",
    "Booster",
    "Data",
    "Dispatch",
    "Metrics",
    "Objective",
    "Parallel",
    "Params",
    "Sampler",
    "Tree",
]
