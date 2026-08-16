"""Parameter translation between bonsai and the reference libraries.

This module is the repo's single source of truth for cross-library knob
mapping. One table per library drives both directions, so a rename can never
drift between the estimator layer, the benchmark harness, and the docs.

``from_xgboost`` / ``from_lightgbm`` / ``from_catboost`` turn a reference
library's parameter dict into the ``(key, value)`` string pairs
``bonsai.train`` takes. ``to_xgboost`` / ``to_lightgbm`` / ``to_catboost``
go the other way, returning that library's parameter dict.

    >>> import bonsai
    >>> pairs = bonsai.interop.from_lightgbm(
    ...     {"num_leaves": 63, "max_depth": -1, "learning_rate": 0.05})
    >>> sorted(pairs)
    [('booster.learning_rate', '0.05'), ('tree.max_depth', '255'), ('tree.max_leaves', '63')]

Translation is not equivalence. Every mapping whose two sides mean different
things carries a note at its table row, and the module-level notes below
collect the ones that change what a fit does rather than what it is called.
Read them before trusting a translated config to reproduce a number.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Final

from bonsai._coerce import _to_config_str

__all__ = [
    "from_catboost",
    "from_lightgbm",
    "from_xgboost",
    "to_catboost",
    "to_lightgbm",
    "to_xgboost",
]

# LightGBM spells an uncapped depth -1; bonsai has no sentinel, so the cap
# travels as a depth no histogram tree reaches. Under leafwise growth the
# leaf budget (tree.max_leaves, LightGBM's num_leaves) is what binds anyway.
UNCAPPED_DEPTH: Final = 255


# Mapping Table ====================================================================================

@dataclasses.dataclass(frozen=True)
class _Knob:
    """One reference-library parameter and its bonsai counterpart.

    Parameters
    ----------
    foreign
        The reference library's parameter name.
    native
        The bonsai dotted config key it maps to.
    to_native, to_foreign
        Value transforms for each direction; ``None`` passes the value
        through unchanged.
    implies
        Extra bonsai pairs this parameter turns on, applied underneath any
        explicitly translated key.
    alias
        True for a second spelling of an already-mapped parameter. Aliases
        translate inbound and are skipped when building the reverse index,
        so each bonsai key has exactly one outbound spelling.
    """

    foreign: str
    native: str
    to_native: Callable[[Any], Any] | None = None
    to_foreign: Callable[[Any], Any] | None = None
    implies: dict | None = None
    alias: bool = False


@dataclasses.dataclass(frozen=True)
class _Library:
    """A reference library's whole translation surface.

    ``dropped_foreign`` and ``dropped_native`` name keys that are recognized
    and deliberately discarded, each with the reason; everything else that
    is unrecognized is reported by ``strict=True``.
    """

    name: str
    knobs: tuple[_Knob, ...]
    dropped_foreign: dict[str, str]
    dropped_native: dict[str, str]


# Every library implies its row sampler from the subsample fraction, and none
# of them treats device placement as a parameter bonsai can honor: in bonsai
# the device is the grower name (cuda_leafwise and friends).
_DROPPED_NATIVE_SHARED: Final = {
    "dispatch.sampler_name": "implied by the subsample fraction",
    "parallel.device_id": "device placement is not a reference-library knob",
}


# XGBoost ==========================================================================================

# Reverse lookup keeps the first spelling of each bonsai objective, so
# softmax comes back as multi:softprob rather than multi:softmax.
_XGB_OBJECTIVES: Final = {
    "reg:squarederror": "mse",
    "reg:linear": "mse",
    "reg:absoluteerror": "mae",
    "reg:quantileerror": "quantile",
    # bonsai's huber is the exact Huber loss, not XGBoost's pseudo-Huber
    # approximation: same shape, different curvature near the elbow.
    "reg:pseudohubererror": "huber",
    "count:poisson": "poisson",
    "binary:logistic": "logloss",
    "multi:softprob": "softmax",
    "multi:softmax": "softmax",
}

_XGB_GROW_POLICY: Final = {"depthwise": "depthwise", "lossguide": "leafwise"}

_XGBOOST: Final = _Library(
    name="xgboost",
    knobs=(
        _Knob("n_estimators", "booster.n_iters"),
        # xgb.train() spells the same count num_boost_round, at the call
        # site rather than in the params dict.
        _Knob("num_boost_round", "booster.n_iters", alias=True),
        _Knob("learning_rate", "booster.learning_rate"),
        _Knob("eta", "booster.learning_rate", alias=True),
        _Knob("max_depth", "tree.max_depth"),
        _Knob("max_leaves", "tree.max_leaves"),
        # NOT a row count. Both sides are a minimum summed hessian, so under
        # squared error the value is a row count and under logloss it is far
        # more rows than its face value (the correction of decision 68).
        _Knob("min_child_weight", "tree.min_child_hess"),
        _Knob("reg_lambda", "tree.lambda_l2"),
        _Knob("lambda", "tree.lambda_l2", alias=True),
        _Knob("reg_alpha", "tree.lambda_l1"),
        _Knob("alpha", "tree.lambda_l1", alias=True),
        _Knob("gamma", "tree.min_gain_to_split"),
        _Knob("min_split_loss", "tree.min_gain_to_split", alias=True),
        # Per-tree only. XGBoost's colsample_bylevel and colsample_bynode
        # have no bonsai counterpart and are reported as unmappable.
        _Knob("colsample_bytree", "tree.feature_fraction"),
        _Knob("max_bin", "bin_mapper.max_bin"),
        _Knob("subsample", "sampler.subsample",
              implies={"dispatch.sampler_name": "bernoulli"}),
        _Knob("seed", "booster.random_seed"),
        _Knob("random_state", "booster.random_seed", alias=True),
        _Knob("n_jobs", "parallel.n_threads"),
        _Knob("nthread", "parallel.n_threads", alias=True),
        # XGBoost keeps every tree it grew on a stop and leaves truncation to
        # the caller's iteration_range; bonsai returns the best-iteration
        # model, so a translated config predicts from a shorter ensemble.
        _Knob("early_stopping_rounds", "booster.early_stopping_rounds"),
        _Knob("objective", "dispatch.objective_name",
              to_native=lambda v: _lookup(_XGB_OBJECTIVES, v, "objective"),
              to_foreign=lambda v: _lookup(_XGB_OBJECTIVES_REV, v, "objective")),
        _Knob("quantile_alpha", "objective.quantile_alpha"),
        # XGBoost has no symmetric-tree policy, so bonsai's levelwise grower
        # is reported as unmappable rather than flattened to depthwise.
        _Knob("grow_policy", "dispatch.grower_name",
              to_native=lambda v: _lookup(_XGB_GROW_POLICY, v, "grow_policy"),
              to_foreign=lambda v: _lookup(_XGB_GROW_POLICY_REV,
                                           _cpu_grower(v), "grow_policy")),
        _Knob("rate_drop", "booster.dart_drop_rate"),
        _Knob("monotone_constraints", "tree.monotone_constraints"),
        _Knob("interaction_constraints", "tree.interaction_constraints"),
    ),
    dropped_foreign={
        "tree_method": "bonsai is always histogram-based",
        "device": "in bonsai the device is the grower name, e.g. cuda_leafwise",
        "verbosity": "logging only",
        "verbose": "logging only",
        "silent": "logging only",
        "validate_parameters": "bonsai always validates",
    },
    dropped_native=_DROPPED_NATIVE_SHARED,
)


# LightGBM =========================================================================================

_LGBM_OBJECTIVES: Final = {
    "regression": "mse",
    "regression_l2": "mse",
    "l2": "mse",
    "mean_squared_error": "mse",
    "mse": "mse",
    "regression_l1": "mae",
    "l1": "mae",
    "mean_absolute_error": "mae",
    "mae": "mae",
    "huber": "huber",
    "quantile": "quantile",
    "poisson": "poisson",
    "binary": "logloss",
    "multiclass": "softmax",
    "softmax": "softmax",
}

_LIGHTGBM: Final = _Library(
    name="lightgbm",
    knobs=(
        _Knob("n_estimators", "booster.n_iters"),
        _Knob("num_iterations", "booster.n_iters", alias=True),
        _Knob("num_boost_round", "booster.n_iters", alias=True),
        _Knob("num_round", "booster.n_iters", alias=True),
        _Knob("num_trees", "booster.n_iters", alias=True),
        _Knob("learning_rate", "booster.learning_rate"),
        _Knob("shrinkage_rate", "booster.learning_rate", alias=True),
        _Knob("eta", "booster.learning_rate", alias=True),
        # max_depth=-1 is LightGBM's uncapped growth. bonsai has no sentinel,
        # so it travels as UNCAPPED_DEPTH and num_leaves stays the binding
        # budget, which is what governs a leafwise fit either way. The
        # reverse never re-invents the sentinel: 255 comes back as 255.
        _Knob("max_depth", "tree.max_depth", to_native=lambda v: _lgbm_depth(v)),
        _Knob("num_leaves", "tree.max_leaves"),
        _Knob("max_leaves", "tree.max_leaves", alias=True),
        # Same name, different gate. LightGBM rejects a candidate whose
        # child would hold fewer rows than this; bonsai only refuses to
        # split a node holding fewer than twice it, so a translated fit can
        # still leave one row in a leaf. tree.min_child_hess is the
        # per-candidate floor, and counts rows under squared error.
        _Knob("min_data_in_leaf", "tree.min_data_in_leaf"),
        _Knob("min_child_samples", "tree.min_data_in_leaf", alias=True),
        _Knob("min_data", "tree.min_data_in_leaf", alias=True),
        _Knob("min_sum_hessian_in_leaf", "tree.min_child_hess"),
        _Knob("min_child_weight", "tree.min_child_hess", alias=True),
        _Knob("lambda_l1", "tree.lambda_l1"),
        _Knob("reg_alpha", "tree.lambda_l1", alias=True),
        _Knob("lambda_l2", "tree.lambda_l2"),
        _Knob("reg_lambda", "tree.lambda_l2", alias=True),
        _Knob("min_gain_to_split", "tree.min_gain_to_split"),
        _Knob("min_split_gain", "tree.min_gain_to_split", alias=True),
        _Knob("feature_fraction", "tree.feature_fraction"),
        _Knob("colsample_bytree", "tree.feature_fraction", alias=True),
        _Knob("bagging_fraction", "sampler.subsample",
              implies={"dispatch.sampler_name": "bernoulli"}),
        _Knob("subsample", "sampler.subsample", alias=True,
              implies={"dispatch.sampler_name": "bernoulli"}),
        _Knob("max_bin", "bin_mapper.max_bin"),
        _Knob("min_data_in_bin", "bin_mapper.min_data_in_bin"),
        _Knob("seed", "booster.random_seed"),
        _Knob("random_state", "booster.random_seed", alias=True),
        _Knob("random_seed", "booster.random_seed", alias=True),
        _Knob("num_threads", "parallel.n_threads"),
        _Knob("n_jobs", "parallel.n_threads", alias=True),
        _Knob("nthread", "parallel.n_threads", alias=True),
        # LightGBM arms patience through a callback, not this key; the key
        # is accepted here because its params-dict spelling is documented.
        _Knob("early_stopping_round", "booster.early_stopping_rounds"),
        _Knob("early_stopping_rounds", "booster.early_stopping_rounds",
              alias=True),
        _Knob("objective", "dispatch.objective_name",
              to_native=lambda v: _lookup(_LGBM_OBJECTIVES, v, "objective"),
              to_foreign=lambda v: _lookup(_LGBM_OBJECTIVES_REV, v, "objective")),
        _Knob("application", "dispatch.objective_name", alias=True,
              to_native=lambda v: _lookup(_LGBM_OBJECTIVES, v, "objective")),
        _Knob("drop_rate", "booster.dart_drop_rate"),
        _Knob("monotone_constraints", "tree.monotone_constraints"),
        _Knob("interaction_constraints", "tree.interaction_constraints"),
    ),
    dropped_foreign={
        "verbose": "logging only",
        "verbosity": "logging only",
        "device": "in bonsai the device is the grower name, e.g. cuda_leafwise",
        "device_type": "in bonsai the device is the grower name",
        "boosting": "DART is booster.dart_drop_rate; growth is the grower name",
        "boosting_type": "DART is booster.dart_drop_rate; growth is the grower name",
        # LightGBM overloads one key for two objectives' parameters, so it
        # cannot be translated without knowing which objective is active.
        "alpha": "set objective.huber_delta or objective.quantile_alpha directly",
    },
    dropped_native={
        **_DROPPED_NATIVE_SHARED,
        "dispatch.grower_name": "LightGBM grows leaf-wise only; num_leaves is the budget",
    },
)


# CatBoost =========================================================================================

# CatBoost's parametrized losses (Huber:delta=..., Quantile:alpha=...) are
# deliberately absent: parsing the suffix and dropping it would silently
# change the loss, so those two are reported and must be set by hand as
# dispatch.objective_name plus objective.huber_delta / objective.quantile_alpha.
# RMSE maps to mse because both minimize squared error; only the reported
# metric differs, and the root is monotone.
_CATBOOST_OBJECTIVES: Final = {
    "RMSE": "mse",
    "MAE": "mae",
    "Poisson": "poisson",
    "Logloss": "logloss",
    "MultiClass": "softmax",
}

_CATBOOST_GROW_POLICY: Final = {
    "SymmetricTree": "levelwise",
    "Depthwise": "depthwise",
    "Lossguide": "leafwise",
}

_CATBOOST: Final = _Library(
    name="catboost",
    knobs=(
        _Knob("iterations", "booster.n_iters"),
        _Knob("n_estimators", "booster.n_iters", alias=True),
        _Knob("num_boost_round", "booster.n_iters", alias=True),
        _Knob("num_trees", "booster.n_iters", alias=True),
        _Knob("learning_rate", "booster.learning_rate"),
        _Knob("eta", "booster.learning_rate", alias=True),
        _Knob("depth", "tree.max_depth"),
        _Knob("max_depth", "tree.max_depth", alias=True),
        _Knob("max_leaves", "tree.max_leaves"),
        _Knob("num_leaves", "tree.max_leaves", alias=True),
        _Knob("l2_leaf_reg", "tree.lambda_l2"),
        _Knob("reg_lambda", "tree.lambda_l2", alias=True),
        # A node gate on bonsai's side, a child gate on CatBoost's; see the
        # LightGBM table's note.
        _Knob("min_data_in_leaf", "tree.min_data_in_leaf"),
        _Knob("min_child_samples", "tree.min_data_in_leaf", alias=True),
        # The fencepost: border_count counts SPLITS where max_bin counts
        # BINS, so the two differ by one in every direction. Getting this
        # wrong shortchanged two published suites by a bin once.
        _Knob("border_count", "bin_mapper.max_bin",
              to_native=lambda v: int(v) + 1, to_foreign=lambda v: int(v) - 1),
        _Knob("max_bin", "bin_mapper.max_bin", alias=True,
              to_native=lambda v: int(v) + 1),
        # rsm samples features per level, tree.feature_fraction per tree, so
        # the same fraction decorrelates differently.
        _Knob("rsm", "tree.feature_fraction"),
        _Knob("colsample_bylevel", "tree.feature_fraction", alias=True),
        _Knob("subsample", "sampler.subsample",
              implies={"dispatch.sampler_name": "bernoulli"}),
        _Knob("random_seed", "booster.random_seed"),
        _Knob("random_state", "booster.random_seed", alias=True),
        _Knob("thread_count", "parallel.n_threads"),
        # CatBoost's patience is od_wait under od_type="Iter"; the
        # constructor's early_stopping_rounds sets the same detector.
        _Knob("early_stopping_rounds", "booster.early_stopping_rounds"),
        _Knob("od_wait", "booster.early_stopping_rounds", alias=True),
        _Knob("loss_function", "dispatch.objective_name",
              to_native=lambda v: _lookup(_CATBOOST_OBJECTIVES, v, "loss_function"),
              to_foreign=lambda v: _lookup(_CATBOOST_OBJECTIVES_REV, v,
                                           "loss_function")),
        _Knob("objective", "dispatch.objective_name", alias=True,
              to_native=lambda v: _lookup(_CATBOOST_OBJECTIVES, v, "objective")),
        _Knob("grow_policy", "dispatch.grower_name",
              to_native=lambda v: _lookup(_CATBOOST_GROW_POLICY, v, "grow_policy"),
              to_foreign=lambda v: _lookup(_CATBOOST_GROW_POLICY_REV,
                                           _cpu_grower(v), "grow_policy")),
        _Knob("monotone_constraints", "tree.monotone_constraints"),
    ),
    dropped_foreign={
        "task_type": "in bonsai the device is the grower name, e.g. cuda_levelwise",
        "devices": "in bonsai the device is the grower name",
        "verbose": "logging only",
        "logging_level": "logging only",
        "allow_writing_files": "CatBoost training artifacts have no bonsai counterpart",
        "train_dir": "CatBoost training artifacts have no bonsai counterpart",
        "od_type": "bonsai's only detector is patience on the eval loss",
        # CatBoost shrinks a model to its best iteration whenever an eval set
        # is present, detector armed or not. bonsai truncates only when
        # early stopping is armed, so this flag has nothing to translate to.
        "use_best_model": "bonsai truncates only when early stopping is armed",
    },
    dropped_native=_DROPPED_NATIVE_SHARED,
)


# Public Functions =================================================================================

def from_xgboost(params: Mapping[str, Any], *,
                 strict: bool = True) -> list[tuple[str, str]]:
    """Translate an XGBoost parameter dict into bonsai config pairs.

    Parameters
    ----------
    params
        XGBoost parameter names to values, as passed to ``XGBRegressor`` or
        ``xgb.train``.
    strict
        True raises on any parameter with no bonsai counterpart, naming
        every one. False drops them and translates the rest, which loses
        whatever they configured.

    Returns
    -------
    list[tuple[str, str]]
        ``(dotted.key, value)`` pairs ready for ``bonsai.train``.

    Raises
    ------
    ValueError
        Under ``strict``, when any parameter has no bonsai counterpart.
        Always, when a mapped parameter carries a value bonsai cannot
        express (an unknown objective string, say).

    Examples
    --------
    >>> import bonsai
    >>> bonsai.interop.from_xgboost({"n_estimators": 300, "reg_lambda": 2.0})
    [('booster.n_iters', '300'), ('tree.lambda_l2', '2.0')]
    """
    return _from_library(_XGBOOST, params, strict)


def from_lightgbm(params: Mapping[str, Any], *,
                  strict: bool = True) -> list[tuple[str, str]]:
    """Translate a LightGBM parameter dict into bonsai config pairs.

    ``max_depth=-1`` (LightGBM's uncapped growth) becomes
    ``tree.max_depth = 255``: bonsai has no sentinel, and under leafwise
    growth ``num_leaves`` is the binding budget either way.

    Parameters
    ----------
    params
        LightGBM parameter names to values, including the documented
        aliases (``num_iterations``, ``min_child_samples``, ...).
    strict
        True raises on any parameter with no bonsai counterpart, naming
        every one. False drops them and translates the rest, which loses
        whatever they configured.

    Returns
    -------
    list[tuple[str, str]]
        ``(dotted.key, value)`` pairs ready for ``bonsai.train``.

    Raises
    ------
    ValueError
        Under ``strict``, when any parameter has no bonsai counterpart.
        Always, when a mapped parameter carries a value bonsai cannot
        express.

    Examples
    --------
    >>> import bonsai
    >>> bonsai.interop.from_lightgbm({"num_leaves": 63, "max_depth": -1})
    [('tree.max_leaves', '63'), ('tree.max_depth', '255')]
    """
    return _from_library(_LIGHTGBM, params, strict)


def from_catboost(params: Mapping[str, Any], *,
                  strict: bool = True) -> list[tuple[str, str]]:
    """Translate a CatBoost parameter dict into bonsai config pairs.

    ``border_count`` counts splits where ``bin_mapper.max_bin`` counts bins,
    so the value gains one crossing over.

    Parameters
    ----------
    params
        CatBoost parameter names to values, as passed to ``CatBoostRegressor``
        or ``CatBoostClassifier``.
    strict
        True raises on any parameter with no bonsai counterpart, naming
        every one. False drops them and translates the rest, which loses
        whatever they configured.

    Returns
    -------
    list[tuple[str, str]]
        ``(dotted.key, value)`` pairs ready for ``bonsai.train``.

    Raises
    ------
    ValueError
        Under ``strict``, when any parameter has no bonsai counterpart.
        Always, when a mapped parameter carries a value bonsai cannot
        express, including CatBoost's parametrized losses
        (``Huber:delta=...``, ``Quantile:alpha=...``).

    Examples
    --------
    >>> import bonsai
    >>> bonsai.interop.from_catboost({"depth": 6, "border_count": 254})
    [('tree.max_depth', '6'), ('bin_mapper.max_bin', '255')]
    """
    return _from_library(_CATBOOST, params, strict)


def to_xgboost(pairs: Iterable[tuple[str, Any]], *,
               strict: bool = True) -> dict[str, Any]:
    """Translate bonsai config pairs into an XGBoost parameter dict.

    Parameters
    ----------
    pairs
        ``(dotted.key, value)`` pairs, or a mapping of the same. Values pass
        through with their Python type intact unless the mapping transforms
        them.
    strict
        True raises on any bonsai key with no XGBoost counterpart, naming
        every one. False drops them.

    Returns
    -------
    dict[str, Any]
        XGBoost parameter names to values.

    Raises
    ------
    ValueError
        Under ``strict``, when any key has no XGBoost counterpart. Always,
        when a value has none (bonsai's ``levelwise`` grower, say).

    Examples
    --------
    >>> import bonsai
    >>> bonsai.interop.to_xgboost([("tree.lambda_l2", 2.0)])
    {'reg_lambda': 2.0}
    """
    return _to_library(_XGBOOST, pairs, strict)


def to_lightgbm(pairs: Iterable[tuple[str, Any]], *,
                strict: bool = True) -> dict[str, Any]:
    """Translate bonsai config pairs into a LightGBM parameter dict.

    Parameters
    ----------
    pairs
        ``(dotted.key, value)`` pairs, or a mapping of the same.
    strict
        True raises on any bonsai key with no LightGBM counterpart, naming
        every one. False drops them.

    Returns
    -------
    dict[str, Any]
        LightGBM parameter names to values.

    Raises
    ------
    ValueError
        Under ``strict``, when any key has no LightGBM counterpart. Always,
        when a value has none.

    Examples
    --------
    >>> import bonsai
    >>> bonsai.interop.to_lightgbm([("tree.max_leaves", 63)])
    {'num_leaves': 63}
    """
    return _to_library(_LIGHTGBM, pairs, strict)


def to_catboost(pairs: Iterable[tuple[str, Any]], *,
                strict: bool = True) -> dict[str, Any]:
    """Translate bonsai config pairs into a CatBoost parameter dict.

    ``bin_mapper.max_bin`` loses one crossing over: CatBoost's
    ``border_count`` counts splits, not bins.

    Parameters
    ----------
    pairs
        ``(dotted.key, value)`` pairs, or a mapping of the same.
    strict
        True raises on any bonsai key with no CatBoost counterpart, naming
        every one. False drops them.

    Returns
    -------
    dict[str, Any]
        CatBoost parameter names to values.

    Raises
    ------
    ValueError
        Under ``strict``, when any key has no CatBoost counterpart. Always,
        when a value has none, including bonsai's ``huber`` and ``quantile``
        objectives, whose CatBoost spellings carry a parameter.

    Examples
    --------
    >>> import bonsai
    >>> bonsai.interop.to_catboost([("bin_mapper.max_bin", 255)])
    {'border_count': 254}
    """
    return _to_library(_CATBOOST, pairs, strict)


# Private Functions ================================================================================

def _invert(table: Mapping[str, str]) -> dict[str, str]:
    """Reverse a value table, keeping the first spelling of each result."""
    out: dict[str, str] = {}
    for foreign, native in table.items():
        out.setdefault(native, foreign)
    return out


_XGB_OBJECTIVES_REV: Final = _invert(_XGB_OBJECTIVES)
_XGB_GROW_POLICY_REV: Final = _invert(_XGB_GROW_POLICY)
_LGBM_OBJECTIVES_REV: Final = _invert(_LGBM_OBJECTIVES)
_CATBOOST_OBJECTIVES_REV: Final = _invert(_CATBOOST_OBJECTIVES)
_CATBOOST_GROW_POLICY_REV: Final = _invert(_CATBOOST_GROW_POLICY)


def _lookup(table: Mapping[str, str], value: Any, knob: str) -> str:
    """A value-table hit, or a ValueError naming what the table accepts."""
    key = str(value)
    if key not in table:
        raise ValueError(
            f"{knob} value {value!r} has no counterpart; "
            f"accepted values are {sorted(table)}"
        )
    return table[key]


def _lgbm_depth(value: Any) -> int:
    """LightGBM's depth cap, with its -1 sentinel resolved."""
    depth = int(value)
    return UNCAPPED_DEPTH if depth < 0 else depth


def _cpu_grower(value: Any) -> str:
    """A grower name without its device prefix.

    Device placement is the grower name in bonsai and a separate parameter
    everywhere else, so a ``cuda_`` grower translates to its growth policy
    and the reference library's own device knob is the caller's to set.
    """
    return str(value).removeprefix("cuda_")


def _from_library(lib: _Library, params: Mapping[str, Any],
                  strict: bool) -> list[tuple[str, str]]:
    """Foreign dict to bonsai pairs; the shared engine behind ``from_*``."""
    index = {knob.foreign: knob for knob in lib.knobs}
    implied: dict[str, Any] = {}
    mapped: dict[str, Any] = {}
    unmappable = []
    for key, value in dict(params).items():
        if key in lib.dropped_foreign:
            continue
        knob = index.get(key)
        if knob is None:
            unmappable.append(key)
            continue
        mapped[knob.native] = value if knob.to_native is None else knob.to_native(value)
        implied.update(knob.implies or {})
    _reject_unmappable(unmappable, strict, lib.name, "bonsai")
    merged = {**implied, **mapped}
    return [(key, _to_config_str(value)) for key, value in merged.items()]


def _to_library(lib: _Library, pairs: Iterable[tuple[str, Any]],
                strict: bool) -> dict[str, Any]:
    """bonsai pairs to a foreign dict; the shared engine behind ``to_*``."""
    index = {knob.native: knob for knob in lib.knobs if not knob.alias}
    out: dict[str, Any] = {}
    unmappable = []
    for key, value in dict(pairs).items():
        if key in lib.dropped_native:
            continue
        knob = index.get(key)
        if knob is None:
            unmappable.append(key)
            continue
        out[knob.foreign] = value if knob.to_foreign is None else knob.to_foreign(value)
    _reject_unmappable(unmappable, strict, "bonsai", lib.name)
    return out


def _reject_unmappable(keys: list[str], strict: bool, source: str, target: str):
    """Under ``strict``, one error naming every key that did not translate."""
    if not keys or not strict:
        return
    raise ValueError(
        f"{len(keys)} {source} parameter(s) have no {target} counterpart: "
        f"{sorted(keys)}. Pass strict=False to drop them, or set the "
        f"{target} key yourself."
    )
