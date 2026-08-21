"""bonsai: histogram gradient-boosted trees, C++23 core.

Two API layers, both over the same native module:

- ``train(params, X, y, ...)`` / ``load(path)`` return a ``Model`` — the
  thin, explicit layer. ``params`` is a ``Params`` (typed overrides, one
  dataclass per config section: ``bonsai.params``) or a
  ``{"tree.max_depth": 8}`` dict — the keys the CLI accepts via ``--set``
  (see ``default_config_toml()`` for all of them).
  ``Model`` carries ``predict / staged_predict / predict_leaf /
  pred_contribs (TreeSHAP) / feature_importance / dump / save``.
- ``BonsaiRegressor`` / ``BonsaiClassifier`` — sklearn-style estimators
  wrapping the same booster for pipelines and quick experiments. They speak
  bonsai's parameter names only; ``bonsai.interop`` translates a parameter
  dict written for XGBoost, LightGBM, or CatBoost into a ``Params``.

GPU training: pass ``device="cuda"`` (or pick a grower directly with
``dispatch.grower_name = "cuda_leafwise"`` / ``"cuda_depthwise"`` /
``"cuda_levelwise"``); ``device="cuda"`` maps by prefix, so the default
``leafwise`` grower runs ``cuda_leafwise``.
``cuda_available()`` reports whether this build and machine can honor it.
Models trained on GPU predict everywhere.

The zero-to-hero walk-through lives in ``docs/guide/`` (start with
chapter 0, one boosting round traced by hand on eight rows).

This file is the public surface only; the implementation lives in
``estimators`` (the sklearn layer), ``interop`` (the cross-library parameter
mapping), ``_coerce`` (input coercion), ``encoding`` (the categorical
encoder), and ``bench`` (the benchmark harness).
"""

from __future__ import annotations

import importlib.metadata

from bonsai import interop
from bonsai._bonsai import (
    Dataset,
    Model,
    cuda_available,
    default_config_toml,
    load,
    train,
)
from bonsai.encoding import OrderedTargetEncoder
from bonsai.estimators import BonsaiClassifier, BonsaiRegressor
from bonsai.params import Params

try:
    # CMake stamps this from pyproject.toml into the build tree only
    # (build/python/bonsai, build-cuda/python/bonsai); it is never
    # installed, so wheels never carry it. Its presence means this
    # process imported the source tree, not an installed distribution.
    from bonsai._version import __version__ as _build_version
except ImportError:
    _build_version = None

if _build_version is not None:
    # A source build imported through PYTHONPATH may sit next to an
    # unrelated installed wheel (e.g. a pod's baked release next to a
    # newer source checkout): report what actually built, tagged so it
    # reads distinctly from a wheel's plain version.
    __version__ = f"{_build_version}+source"
else:
    try:
        __version__ = importlib.metadata.version("bonsai-gbt")
    except importlib.metadata.PackageNotFoundError:
        __version__ = "source"

__all__ = [
    "BonsaiClassifier",
    "BonsaiRegressor",
    "Dataset",
    "Model",
    "OrderedTargetEncoder",
    "Params",
    "__version__",
    "cuda_available",
    "default_config_toml",
    "interop",
    "load",
    "train",
]
