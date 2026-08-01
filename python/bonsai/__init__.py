"""bonsai: histogram gradient-boosted trees, C++23 core.

Two API layers, both over the same native module:

- ``train(params, X, y, ...)`` / ``load(path)`` return a ``Model`` — the
  thin, explicit layer. ``params`` is a list of ``(dotted.key, value)``
  pairs using exactly the keys the CLI accepts via ``--set`` (see
  ``default_config_toml()`` for all of them). ``Model`` carries
  ``predict / staged_predict / predict_leaf / pred_contribs (TreeSHAP) /
  feature_importance / dump / save``.
- ``BonsaiRegressor`` / ``BonsaiClassifier`` — sklearn-style estimators
  wrapping the same booster for pipelines and quick experiments.

GPU training: pass ``device="cuda"`` (or pick a grower directly with
``dispatch.grower_name = "cuda_depthwise"`` / ``"cuda_oblivious"``);
``cuda_available()`` reports whether this build and machine can honor it.
Models trained on GPU predict everywhere.

The zero-to-hero walk-through lives in ``docs/guide/`` (start with
chapter 0, one boosting round traced by hand on eight rows).

This file is the public surface only; the implementation lives in
``estimators`` (the sklearn layer), ``_compat`` (the xgboost-compatibility
translation tables), ``_coerce`` (input coercion), ``encoding`` (the
categorical encoder), and ``bench`` (the benchmark harness).
"""

from __future__ import annotations

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

__all__ = [
    "BonsaiClassifier",
    "BonsaiRegressor",
    "Dataset",
    "Model",
    "OrderedTargetEncoder",
    "cuda_available",
    "default_config_toml",
    "load",
    "train",
]
