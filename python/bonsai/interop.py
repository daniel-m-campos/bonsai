"""Parameter translation between bonsai and the reference libraries.

This module is the repo's single source of truth for cross-library knob
mapping. One table per library drives both directions, so a rename can never
drift between the estimator layer, the benchmark harness, and the docs.

``from_xgboost`` / ``from_lightgbm`` / ``from_catboost`` turn a reference
library's parameter dict into a ``bonsai.Params``, ready for
``bonsai.train``. ``to_xgboost`` / ``to_lightgbm`` / ``to_catboost``
go the other way, returning that library's parameter dict.

    >>> import bonsai
    >>> p = bonsai.interop.from_lightgbm(
    ...     {"num_leaves": 63, "max_depth": -1, "learning_rate": 0.05})
    >>> p
    Params(tree=Tree(max_depth=255, max_leaves=63), booster=Booster(learning_rate=0.05))

Translation is not equivalence. Every mapping whose two sides mean different
things carries a note at its table row, and the module-level notes below
collect the ones that change what a fit does rather than what it is called.
Read them before trusting a translated config to reproduce a number.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Final

from bonsai._params_ops import ParamsOps
from bonsai.params import Params

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

class _Values:
    """A closed value table for one knob, translated both ways.

    The reverse keeps the first spelling of each bonsai value, so softmax
    comes back as multi:softprob rather than multi:softmax.

    Parameters
    ----------
    knob
        The reference library's name for the knob, for the error a value
        outside the table raises.
    table
        Reference-library spellings to bonsai values.
    """

    def __init__(self, knob: str, table: Mapping[str, str]):
        self.knob = knob
        self.table = dict(table)
        self.reverse: dict[str, str] = {}
        for foreign, native in table.items():
            self.reverse.setdefault(native, foreign)

    def inbound(self, value: Any) -> str:
        return self._hit(self.table, value)

    def outbound(self, value: Any) -> str:
        return self._hit(self.reverse, value)

    def _hit(self, table: Mapping[str, str], value: Any) -> str:
        key = str(value)
        if key not in table:
            raise ValueError(
                f"{self.knob} value {value!r} has no counterpart; "
                f"accepted values are {sorted(table)}"
            )
        return table[key]


class _Policies(_Values):
    """Growth policies, which bonsai's device prefix is no part of.

    Device placement is the grower name in bonsai and a separate parameter
    everywhere else, so a ``cuda_`` grower translates to its growth policy
    and the reference library's own device knob is the caller's to set.
    """

    def outbound(self, value: Any) -> str:
        return super().outbound(str(value).removeprefix("cuda_"))


class _Knob:
    """One bonsai config key under a reference library's spellings.

    The first spelling is the one a bonsai key translates out to; the rest
    are accepted inbound. Every spelling shares the value transforms and the
    implied pairs, which is what makes it a spelling rather than a knob.

    Parameters
    ----------
    native
        The bonsai dotted config key.
    *spellings
        The reference library's names for it, outbound spelling first.
    values
        A closed value table; its two directions become the transforms.
    to_native, to_foreign
        Value transforms for each direction; None passes the value through.
    implies
        Extra bonsai pairs this knob turns on, applied underneath any
        explicitly translated key.
    """

    def __init__(self, native: str, *spellings: str, values: _Values | None = None,
                 to_native: Callable[[Any], Any] | None = None,
                 to_foreign: Callable[[Any], Any] | None = None,
                 implies: Mapping[str, Any] | None = None):
        self.native = native
        self.spellings = spellings
        self.foreign = spellings[0]
        self.to_native = values.inbound if values is not None else to_native
        self.to_foreign = values.outbound if values is not None else to_foreign
        self.implies = dict(implies or {})

    def inbound(self, value: Any) -> Any:
        return value if self.to_native is None else self.to_native(value)

    def outbound(self, value: Any) -> Any:
        return value if self.to_foreign is None else self.to_foreign(value)


@dataclasses.dataclass(frozen=True)
class _Library:
    """A reference library's whole translation surface.

    Parameters
    ----------
    name
        The reference library's name as this module spells it, ``"xgboost"``.
    title
        The name as the library spells it, ``"XGBoost"``.
    knobs
        The parameter mapping table for this library.
    dropped_foreign
        Foreign parameter names that are recognized and deliberately
        discarded, each with the reason; everything else unrecognized is
        reported by ``strict=True``.
    dropped_native
        bonsai dotted config keys that are recognized and deliberately
        discarded when translating the other direction, each with the
        reason.
    """

    name: str
    title: str
    knobs: tuple[_Knob, ...]
    dropped_foreign: dict[str, str]
    dropped_native: dict[str, str]


# A subsample fraction implies row sampling everywhere; bonsai makes the
# sampler explicit, so the fraction turns it on.
_ROW_SAMPLING: Final = {"dispatch.sampler_name": "bernoulli"}

# Every library implies its row sampler from the subsample fraction, and none
# of them treats device placement as a parameter bonsai can honor: in bonsai
# the device is the grower name (cuda_leafwise and friends).
_DROPPED_NATIVE_SHARED: Final = {
    "dispatch.sampler_name": "implied by the subsample fraction",
    "parallel.device_id": "device placement is not a reference-library knob",
}


# XGBoost ==========================================================================================

_XGB_OBJECTIVES: Final = _Values("objective", {
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
})

_XGB_GROW_POLICY: Final = _Policies("grow_policy", {
    "depthwise": "depthwise",
    "lossguide": "leafwise",
})

_XGBOOST: Final = _Library(
    name="xgboost",
    title="XGBoost",
    knobs=(
        # xgb.train() spells the same count num_boost_round, at the call
        # site rather than in the params dict.
        _Knob("booster.n_iters", "n_estimators", "num_boost_round"),
        _Knob("booster.learning_rate", "learning_rate", "eta"),
        _Knob("tree.max_depth", "max_depth"),
        _Knob("tree.max_leaves", "max_leaves"),
        # NOT a row count. Both sides are a minimum summed hessian, so under
        # squared error the value is a row count and under logloss it is far
        # more rows than its face value (the correction of decision 68).
        _Knob("tree.min_child_hess", "min_child_weight"),
        _Knob("tree.lambda_l2", "reg_lambda", "lambda"),
        _Knob("tree.lambda_l1", "reg_alpha", "alpha"),
        _Knob("tree.min_gain_to_split", "gamma", "min_split_loss"),
        # Per-tree only. XGBoost's colsample_bylevel and colsample_bynode
        # have no bonsai counterpart and are reported as unmappable.
        _Knob("tree.feature_fraction", "colsample_bytree"),
        _Knob("bin_mapper.max_bin", "max_bin"),
        _Knob("sampler.subsample", "subsample", implies=_ROW_SAMPLING),
        _Knob("booster.random_seed", "seed", "random_state"),
        _Knob("parallel.n_threads", "n_jobs", "nthread"),
        # XGBoost keeps every tree it grew on a stop and leaves truncation to
        # the caller's iteration_range; bonsai returns the best-iteration
        # model, so a translated config predicts from a shorter ensemble.
        _Knob("booster.early_stopping_rounds", "early_stopping_rounds"),
        _Knob("dispatch.objective_name", "objective", values=_XGB_OBJECTIVES),
        _Knob("objective.n_classes", "num_class"),
        _Knob("objective.quantile_alpha", "quantile_alpha"),
        # XGBoost has no symmetric-tree policy, so bonsai's levelwise grower
        # is reported as unmappable rather than flattened to depthwise.
        _Knob("dispatch.grower_name", "grow_policy", values=_XGB_GROW_POLICY),
        _Knob("booster.dart_drop_rate", "rate_drop"),
        _Knob("tree.monotone_constraints", "monotone_constraints"),
        _Knob("tree.interaction_constraints", "interaction_constraints"),
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

_LGBM_OBJECTIVES: Final = _Values("objective", {
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
})


def _lgbm_depth(value: Any) -> int:
    """LightGBM's depth cap, with its -1 sentinel resolved."""
    depth = int(value)
    return UNCAPPED_DEPTH if depth < 0 else depth


_LIGHTGBM: Final = _Library(
    name="lightgbm",
    title="LightGBM",
    knobs=(
        _Knob("booster.n_iters", "n_estimators", "num_iterations", "num_boost_round",
              "num_round", "num_trees"),
        _Knob("booster.learning_rate", "learning_rate", "shrinkage_rate", "eta"),
        # max_depth=-1 is LightGBM's uncapped growth. bonsai has no sentinel,
        # so it travels as UNCAPPED_DEPTH and num_leaves stays the binding
        # budget, which is what governs a leafwise fit either way. The
        # reverse never re-invents the sentinel: 255 comes back as 255.
        _Knob("tree.max_depth", "max_depth", to_native=_lgbm_depth),
        _Knob("tree.max_leaves", "num_leaves", "max_leaves"),
        # Same name, different gate. LightGBM rejects a candidate whose
        # child would hold fewer rows than this; bonsai only refuses to
        # split a node holding fewer than twice it, so a translated fit can
        # still leave one row in a leaf. tree.min_child_hess is the
        # per-candidate floor, and counts rows under squared error.
        _Knob("tree.min_data_in_leaf", "min_data_in_leaf", "min_child_samples", "min_data"),
        _Knob("tree.min_child_hess", "min_sum_hessian_in_leaf", "min_child_weight"),
        _Knob("tree.lambda_l1", "lambda_l1", "reg_alpha"),
        _Knob("tree.lambda_l2", "lambda_l2", "reg_lambda"),
        _Knob("tree.min_gain_to_split", "min_gain_to_split", "min_split_gain"),
        _Knob("tree.feature_fraction", "feature_fraction", "colsample_bytree"),
        _Knob("sampler.subsample", "bagging_fraction", "subsample", implies=_ROW_SAMPLING),
        _Knob("bin_mapper.max_bin", "max_bin"),
        _Knob("bin_mapper.min_data_in_bin", "min_data_in_bin"),
        _Knob("booster.random_seed", "seed", "random_state", "random_seed"),
        _Knob("parallel.n_threads", "num_threads", "n_jobs", "nthread"),
        # LightGBM arms patience through a callback, not this key; the key
        # is accepted here because its params-dict spelling is documented.
        _Knob("booster.early_stopping_rounds", "early_stopping_round",
              "early_stopping_rounds"),
        _Knob("dispatch.objective_name", "objective", "application",
              values=_LGBM_OBJECTIVES),
        _Knob("objective.n_classes", "num_class", "num_classes"),
        _Knob("booster.dart_drop_rate", "drop_rate"),
        _Knob("tree.monotone_constraints", "monotone_constraints"),
        _Knob("tree.interaction_constraints", "interaction_constraints"),
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
_CATBOOST_OBJECTIVES: Final = _Values("loss_function", {
    "RMSE": "mse",
    "MAE": "mae",
    "Poisson": "poisson",
    "Logloss": "logloss",
    "MultiClass": "softmax",
})

_CATBOOST_GROW_POLICY: Final = _Policies("grow_policy", {
    "SymmetricTree": "levelwise",
    "Depthwise": "depthwise",
    "Lossguide": "leafwise",
})

_CATBOOST: Final = _Library(
    name="catboost",
    title="CatBoost",
    knobs=(
        _Knob("booster.n_iters", "iterations", "n_estimators", "num_boost_round", "num_trees"),
        _Knob("booster.learning_rate", "learning_rate", "eta"),
        _Knob("tree.max_depth", "depth", "max_depth"),
        _Knob("tree.max_leaves", "max_leaves", "num_leaves"),
        _Knob("tree.lambda_l2", "l2_leaf_reg", "reg_lambda"),
        # A node gate on bonsai's side, a child gate on CatBoost's; see the
        # LightGBM table's note.
        _Knob("tree.min_data_in_leaf", "min_data_in_leaf", "min_child_samples"),
        # The fencepost: border_count counts SPLITS where max_bin counts
        # BINS, so the two differ by one in every direction. Getting this
        # wrong shortchanged two published suites by a bin once.
        _Knob("bin_mapper.max_bin", "border_count", "max_bin",
              to_native=lambda v: int(v) + 1, to_foreign=lambda v: int(v) - 1),
        # rsm samples features per level, tree.feature_fraction per tree, so
        # the same fraction decorrelates differently.
        _Knob("tree.feature_fraction", "rsm", "colsample_bylevel"),
        _Knob("sampler.subsample", "subsample", implies=_ROW_SAMPLING),
        _Knob("booster.random_seed", "random_seed", "random_state"),
        _Knob("parallel.n_threads", "thread_count"),
        # CatBoost's patience is od_wait under od_type="Iter"; the
        # constructor's early_stopping_rounds sets the same detector.
        _Knob("booster.early_stopping_rounds", "early_stopping_rounds", "od_wait"),
        _Knob("dispatch.objective_name", "loss_function", "objective",
              values=_CATBOOST_OBJECTIVES),
        _Knob("objective.n_classes", "classes_count"),
        _Knob("dispatch.grower_name", "grow_policy", values=_CATBOOST_GROW_POLICY),
        _Knob("tree.monotone_constraints", "monotone_constraints"),
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


# Docstrings =======================================================================================

_FROM_DOC = """Translate {title} parameters into bonsai config pairs.
{notes}
Parameters
----------
params
    {title} parameter names to values, under any spelling the library
    documents.
strict
    True raises on any parameter with no bonsai counterpart, naming
    every one. False drops them and translates the rest, which loses
    whatever they configured.

Returns
-------
Params
    The translated overrides, ready for ``bonsai.train`` (compose with
    ``|`` to layer more).

Raises
------
ValueError
    Under ``strict``, when any parameter has no bonsai counterpart.
    Always, when a mapped parameter carries a value bonsai cannot
    express.

Examples
--------
>>> import bonsai
>>> bonsai.interop.from_{name}({call})
{result}
"""

_TO_DOC = """Translate bonsai config pairs into {title} parameters.
{notes}
Parameters
----------
pairs
    ``(dotted.key, value)`` pairs, a mapping of the same, or a
    ``bonsai.Params``. Values pass through with their Python type intact
    unless the mapping transforms them.
strict
    True raises on any bonsai key with no {title} counterpart, naming
    every one. False drops them.

Returns
-------
dict[str, Any]
    {title} parameter names to values.

Raises
------
ValueError
    Under ``strict``, when any key has no {title} counterpart. Always,
    when a value has none.

Examples
--------
>>> import bonsai
>>> bonsai.interop.to_{name}({call})
{result}
"""


def _doc(template: str, lib: _Library, call: str, result: str, notes: str = "") -> str:
    """One public function's docstring: the shared contract around its
    own example and translation notes."""
    if notes:
        notes = f"\n{notes}\n"
    return template.format(title=lib.title, name=lib.name, notes=notes,
                           call=call, result=result)


# Public Functions =================================================================================

def from_xgboost(params: Mapping[str, Any], *,
                 strict: bool = True) -> Params:
    return _from_library(_XGBOOST, params, strict)


from_xgboost.__doc__ = _doc(
    _FROM_DOC, _XGBOOST,
    call='{"n_estimators": 300, "reg_lambda": 2.0}',
    result="Params(tree=Tree(lambda_l2=2.0), booster=Booster(n_iters=300))",
)


def from_lightgbm(params: Mapping[str, Any], *,
                  strict: bool = True) -> Params:
    return _from_library(_LIGHTGBM, params, strict)


from_lightgbm.__doc__ = _doc(
    _FROM_DOC, _LIGHTGBM,
    notes="``max_depth=-1`` (LightGBM's uncapped growth) becomes\n"
          "``tree.max_depth = 255``: bonsai has no sentinel, and under leafwise\n"
          "growth ``num_leaves`` is the binding budget either way.",
    call='{"num_leaves": 63, "max_depth": -1}',
    result="Params(tree=Tree(max_depth=255, max_leaves=63))",
)


def from_catboost(params: Mapping[str, Any], *,
                  strict: bool = True) -> Params:
    return _from_library(_CATBOOST, params, strict)


from_catboost.__doc__ = _doc(
    _FROM_DOC, _CATBOOST,
    notes="``border_count`` counts splits where ``bin_mapper.max_bin`` counts bins,\n"
          "so the value gains one crossing over. CatBoost's parametrized losses\n"
          "(``Huber:delta=...``, ``Quantile:alpha=...``) raise: set\n"
          "``dispatch.objective_name`` and the loss parameter by hand.",
    call='{"depth": 6, "border_count": 254}',
    result="Params(bin_mapper=BinMapper(max_bin=255), tree=Tree(max_depth=6))",
)


def to_xgboost(pairs: ParamsOps | Iterable[tuple[str, Any]], *,
               strict: bool = True) -> dict[str, Any]:
    return _to_library(_XGBOOST, pairs, strict)


to_xgboost.__doc__ = _doc(
    _TO_DOC, _XGBOOST,
    notes="bonsai's ``levelwise`` grower has no XGBoost policy and raises.",
    call='[("tree.lambda_l2", 2.0)]',
    result="{'reg_lambda': 2.0}",
)


def to_lightgbm(pairs: ParamsOps | Iterable[tuple[str, Any]], *,
                strict: bool = True) -> dict[str, Any]:
    return _to_library(_LIGHTGBM, pairs, strict)


to_lightgbm.__doc__ = _doc(
    _TO_DOC, _LIGHTGBM,
    call='[("tree.max_leaves", 63)]',
    result="{'num_leaves': 63}",
)


def to_catboost(pairs: ParamsOps | Iterable[tuple[str, Any]], *,
                strict: bool = True) -> dict[str, Any]:
    return _to_library(_CATBOOST, pairs, strict)


to_catboost.__doc__ = _doc(
    _TO_DOC, _CATBOOST,
    notes="``bin_mapper.max_bin`` loses one crossing over: CatBoost's\n"
          "``border_count`` counts splits, not bins. bonsai's ``huber`` and\n"
          "``quantile`` objectives raise, since their CatBoost spellings carry a\n"
          "parameter.",
    call='[("bin_mapper.max_bin", 255)]',
    result="{'border_count': 254}",
)


# Private Functions ================================================================================

def _from_library(lib: _Library, params: Mapping[str, Any],
                  strict: bool) -> Params:
    """Foreign dict to a ``Params``; the shared engine behind ``from_*``."""
    index = {spelling: knob for knob in lib.knobs for spelling in knob.spellings}
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
        mapped[knob.native] = knob.inbound(value)
        implied.update(knob.implies)
    _reject_unmappable(unmappable, strict, lib.name, "bonsai")
    return Params.from_dict({**implied, **mapped})


def _to_library(lib: _Library, pairs: ParamsOps | Iterable[tuple[str, Any]],
                strict: bool) -> dict[str, Any]:
    """bonsai pairs to a foreign dict; the shared engine behind ``to_*``."""
    index = {knob.native: knob for knob in lib.knobs}
    out: dict[str, Any] = {}
    unmappable = []
    items = pairs.to_dict() if isinstance(pairs, ParamsOps) else dict(pairs)
    if items.get("dispatch.objective_name") != "softmax":
        # Every objective carries the registry default n_classes=3 in a full
        # config dump; only softmax consumes it, and the reference libraries
        # reject a class count on non-multiclass objectives.
        items.pop("objective.n_classes", None)
    for key, value in items.items():
        if key in lib.dropped_native:
            continue
        knob = index.get(key)
        if knob is None:
            unmappable.append(key)
            continue
        out[knob.foreign] = knob.outbound(value)
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
