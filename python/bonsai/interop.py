"""Parameter translation between bonsai and the reference libraries.

This module is the repo's single source of truth for cross-library knob
mapping. One table per library drives both directions, so a rename can never
drift between the estimator layer, the benchmark harness, and the docs.

``from_xgboost`` / ``from_lightgbm`` / ``from_catboost`` turn a reference
library's parameter dict into a ``bonsai.Params``, ready for
``bonsai.train``. ``to_xgboost`` / ``to_lightgbm`` / ``to_catboost``
go the other way, returning that library's parameter dict.

    >>> import bonsai
    >>> p = bonsai.interop.from_lightgbm({"num_leaves": 63, "max_depth": -1})
    >>> p
    Params(tree=Tree(max_depth=255, max_leaves=63), dispatch=Dispatch(grower_name='leafwise'))

Translation is not equivalence. Every mapping whose two sides mean different
things carries a note at its table row, and the module-level notes below
collect the ones that change what a fit does rather than what it is called.
Read them before trusting a translated config to reproduce a number.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import re
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

# LightGBM spells an uncapped depth -1 (any value <= 0) and XGBoost spells it
# 0; bonsai has no sentinel, so the cap travels as a depth no histogram tree
# reaches. Under leafwise growth the leaf budget (tree.max_leaves, LightGBM's
# num_leaves) is what binds anyway.
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
        return _closed_lookup(self.knob, table, value)


def _closed_lookup(name: str, table: Mapping[str, Any], value: Any) -> Any:
    """``table[str(value)]``, or a ValueError naming the knob and the
    values it accepts."""
    key = str(value)
    if key not in table:
        raise ValueError(
            f"{name} value {value!r} has no counterpart; accepted values are {sorted(table)}"
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


@functools.cache
def _registry_default(key: str) -> Any:
    """The value bonsai resolves a dotted config key to when it is unset."""
    from bonsai import _bonsai

    section, leaf = key.split(".", 1)
    for entry in _bonsai._params_schema():
        if entry["section"] != section:
            continue
        for field in entry["fields"]:
            if field["name"] == leaf:
                return field["default"]
    raise KeyError(key)


@dataclasses.dataclass(frozen=True)
class _ReadUnder:
    """The bonsai selector a config key is read under.

    bonsai ignores the key unless ``selector`` resolves, stated or at its
    registry default, to one of ``values``. Outbound such a key is dropped,
    since the reference library would enforce a value bonsai never read.
    Inbound it is reported as unmappable, since under that selector bonsai
    has nothing to translate it to. Knobs that share a foreign spelling
    under disjoint selectors split one overloaded key by selector.
    """

    selector: str
    values: frozenset[str]

    def holds(self, native: Mapping[str, Any]) -> bool:
        return str(native.get(self.selector, _registry_default(self.selector))) in self.values


class _Switch:
    """A reference-library key whose closed values each stand for bonsai
    pairs rather than for one key's value, such as LightGBM's ``boosting``.

    Inbound, the stated value's pairs apply underneath every translated
    key, and a value outside the table raises. Outbound, a switch is
    written by the knobs that need it (see ``_When``).

    Parameters
    ----------
    *spellings
        The reference library's names for the key, outbound spelling first.
    table
        Each accepted value, as a string, to the bonsai pairs it stands for.
    default
        The value the reference library uses when the key is unset.
    """

    def __init__(self, *spellings: str, table: Mapping[str, Mapping[str, Any]], default: Any):
        self.spellings = spellings
        self.foreign = spellings[0]
        self.table = {key: dict(pairs) for key, pairs in table.items()}
        self.default = default

    def value(self, params: Mapping[str, Any]) -> str:
        stated = (params[spelling] for spelling in self.spellings if spelling in params)
        return str(next(stated, self.default))

    def implied(self, value: Any) -> Mapping[str, Any]:
        return _closed_lookup(self.foreign, self.table, value)


@dataclasses.dataclass(frozen=True)
class _When:
    """The switch value under which a reference library reads a key.

    Inbound, the key translates only when ``switch`` takes ``value``; under
    any other value it is dropped, since the library ignores it, or with
    ``reported`` named as unmappable, since the library reads it there in a
    way bonsai cannot. Outbound, the switch is written beside the key.
    """

    switch: _Switch
    value: Any
    reported: bool = False

    def holds(self, params: Mapping[str, Any]) -> bool:
        return self.switch.value(params) == str(self.value)

    def pair(self) -> dict[str, Any]:
        return {self.switch.foreign: self.value}


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
    forces
        bonsai pairs this knob sets over everything else the parameters
        translate to, for a value the reference library enforces where the
        translated selector would leave it unread.
    read_under
        The bonsai selector the key is read under; None reads it always.
    when
        The reference-library switch value the key is read under; None
        reads it always.
    foreign_default
        The reference library's value when the key is unset, where it
        differs from bonsai's. Inbound, an unset key translates as this
        value; outbound, an unset bonsai key goes out at bonsai's default.
    foreign_pairs
        Closed bonsai values to the reference-library pairs each stands
        for, for a key with no one foreign counterpart; such a knob has no
        spellings and translates outbound only.
    inert
        A value that configures nothing on either side, such as a disabled
        sentinel one library refuses; it translates to no pair at all.
    """

    def __init__(
        self,
        native: str,
        *spellings: str,
        values: _Values | None = None,
        to_native: Callable[[Any], Any] | None = None,
        to_foreign: Callable[[Any], Any] | None = None,
        implies: Mapping[str, Any] | None = None,
        forces: Mapping[str, Any] | None = None,
        read_under: _ReadUnder | None = None,
        when: _When | None = None,
        foreign_default: Any = None,
        foreign_pairs: Mapping[str, Mapping[str, Any]] | None = None,
        inert: Any = None,
    ):
        self.native = native
        self.spellings = spellings
        self.foreign = spellings[0] if spellings else native
        self.to_native = values.inbound if values is not None else to_native
        self.to_foreign = values.outbound if values is not None else to_foreign
        self.implies = dict(implies or {})
        self.forces = dict(forces or {})
        self.read_under = read_under
        self.when = when
        self.foreign_default = foreign_default
        self.foreign_pairs = foreign_pairs
        self.inert = inert

    def is_inert(self, value: Any) -> bool:
        return self.inert is not None and value == self.inert

    def is_read(self, native: Mapping[str, Any]) -> bool:
        return self.read_under is None or self.read_under.holds(native)

    def inbound(self, value: Any) -> Any:
        return value if self.to_native is None else self.to_native(value)

    def outbound(self, value: Any) -> Any:
        return value if self.to_foreign is None else self.to_foreign(value)

    def outbound_pairs(self, value: Any) -> dict[str, Any]:
        if self.foreign_pairs is not None:
            return dict(_closed_lookup(self.native, self.foreign_pairs, value))
        switch = self.when.pair() if self.when is not None else {}
        return {**switch, self.foreign: self.outbound(value)}


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
    switches
        The library's keys whose values stand for bonsai pairs.
    implies
        bonsai pairs every translation from this library carries, because
        the library's own default differs from bonsai's; applied underneath
        everything the parameters translate to.
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
    switches: tuple[_Switch, ...]
    implies: dict[str, Any]
    dropped_foreign: dict[str, str]
    dropped_native: dict[str, str]


# A subsample fraction implies row sampling everywhere; bonsai makes the
# sampler explicit, so the fraction turns it on.
_ROW_SAMPLING: Final = {"dispatch.sampler_name": "bernoulli"}

_UNDER_SOFTMAX: Final = _ReadUnder("dispatch.objective_name", frozenset({"softmax"}))
_UNDER_BERNOULLI: Final = _ReadUnder("dispatch.sampler_name", frozenset({"bernoulli"}))
_UNDER_GOSS: Final = _ReadUnder("dispatch.sampler_name", frozenset({"goss"}))
_UNDER_QUANTILE: Final = _ReadUnder("dispatch.objective_name", frozenset({"quantile"}))
_UNDER_HUBER: Final = _ReadUnder("dispatch.objective_name", frozenset({"huber"}))
_UNDER_LEAFWISE: Final = _ReadUnder(
    "dispatch.grower_name", frozenset({"leafwise", "cuda_leafwise"})
)

# No library treats device placement as a parameter bonsai can honor: in
# bonsai the device is the grower name (cuda_leafwise and friends).
_DROPPED_NATIVE_SHARED: Final = {
    "parallel.device_id": "device placement is not a reference-library knob",
}


def _uncapped_depth(value: Any) -> int:
    """A depth cap with the no-limit sentinel (0 and below) resolved."""
    depth = int(value)
    return UNCAPPED_DEPTH if depth <= 0 else depth


def _all_cores_as_auto(value: Any) -> int:
    """A thread count with the all-cores sentinel (any negative) resolved
    to bonsai's auto-detect, 0."""
    return max(int(value), 0)


def _monotone_directions(value: Any) -> list[int] | dict[str, int]:
    """Monotone directions as a list, from a list, an array, bonsai's
    ``"1,0,-1"`` or the reference libraries' ``"(1,0,-1)"``; a mapping
    keyed by feature name passes through, since every side takes one."""
    if isinstance(value, Mapping):
        return dict(value)
    parts = value.strip("()[] ").split(",") if isinstance(value, str) else value
    try:
        return [int(part) for part in parts if str(part).strip()]
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"monotone_constraints value {value!r} is not a sequence of -1/0/+1 directions"
        ) from err


def _xgb_monotone_directions(value: Any) -> tuple[int, ...] | dict[str, int]:
    """Monotone directions as XGBoost takes them: a tuple or a mapping,
    never a list, which its Booster reads as a mapping and crashes on."""
    directions = _monotone_directions(value)
    return directions if isinstance(directions, dict) else tuple(directions)


def _feature_indices(group: Any) -> list[int]:
    """One interaction group's feature indices, from a sequence or from
    bonsai's ``"0,1"`` / ``"0+1"`` spelling."""
    parts = re.split(r"[,+]", group) if isinstance(group, str) else group
    try:
        return [int(part) for part in parts if str(part).strip()]
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"interaction_constraints group {group!r} is not a list of feature indices; "
            "bonsai takes indices, not feature names"
        ) from err


def _interaction_groups(value: Any) -> list[list[int]]:
    """bonsai's interaction groups (``["0,1", "2,3"]`` or ``"0+1,2+3"``)
    as the reference libraries' lists of feature indices."""
    entries = value.split(",") if isinstance(value, str) else value
    return [_feature_indices(entry) for entry in entries]


def _xgb_interaction_groups(value: Any) -> str:
    """Interaction groups as XGBoost's index form, a JSON string: its
    Booster reads a list of lists as feature names."""
    return json.dumps(_interaction_groups(value))


def _interaction_strings(value: Any) -> list[str]:
    """The reference libraries' lists of feature indices, or XGBoost's
    JSON string of one, as bonsai's group strings."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"interaction_constraints value {value!r} is not a JSON list of index lists"
            ) from err
    return [",".join(str(index) for index in _feature_indices(group)) for group in value]


# XGBoost ==========================================================================================

_XGB_OBJECTIVES: Final = _Values(
    "objective",
    {
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
    },
)

_XGB_GROW_POLICY: Final = _Policies(
    "grow_policy",
    {
        "depthwise": "depthwise",
        "lossguide": "leafwise",
    },
)

_XGB_BOOSTER: Final = _Switch("booster", table={"gbtree": {}, "dart": {}}, default="gbtree")

_XGBOOST: Final = _Library(
    name="xgboost",
    title="XGBoost",
    knobs=(
        # xgb.train() spells the same count num_boost_round, at the call
        # site rather than in the params dict.
        _Knob("booster.n_iters", "n_estimators", "num_boost_round"),
        _Knob("booster.learning_rate", "learning_rate", "eta"),
        # max_depth=0 is XGBoost's uncapped growth; see the LightGBM row.
        _Knob("tree.max_depth", "max_depth", to_native=_uncapped_depth),
        # XGBoost enforces max_leaves under depthwise growth too, where
        # bonsai's depthwise grower ignores it, so a binding cap selects
        # leafwise growth: the leaf budget matches, the split order becomes
        # best-first. bonsai grows depthwise anyway when the cap cannot bind.
        # 0 is XGBoost's no-cap default and translates to nothing.
        _Knob(
            "tree.max_leaves",
            "max_leaves",
            forces={"dispatch.grower_name": "leafwise"},
            read_under=_UNDER_LEAFWISE,
            inert=0,
        ),
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
        _Knob(
            "sampler.subsample",
            "subsample",
            implies=_ROW_SAMPLING,
            read_under=_UNDER_BERNOULLI,
        ),
        _Knob("booster.random_seed", "seed", "random_state"),
        _Knob("parallel.n_threads", "n_jobs", "nthread", to_native=_all_cores_as_auto),
        # XGBoost keeps every tree it grew on a stop and leaves truncation to
        # the caller's iteration_range; bonsai returns the best-iteration
        # model, so a translated config predicts from a shorter ensemble.
        _Knob("booster.early_stopping_rounds", "early_stopping_rounds"),
        _Knob("dispatch.objective_name", "objective", values=_XGB_OBJECTIVES),
        _Knob("objective.n_classes", "num_class", read_under=_UNDER_SOFTMAX),
        _Knob("objective.quantile_alpha", "quantile_alpha", read_under=_UNDER_QUANTILE),
        # XGBoost has no symmetric-tree policy, so bonsai's levelwise grower
        # is reported as unmappable rather than flattened to depthwise.
        _Knob("dispatch.grower_name", "grow_policy", values=_XGB_GROW_POLICY),
        _Knob("booster.dart_drop_rate", "rate_drop", when=_When(_XGB_BOOSTER, "dart"), inert=0.0),
        # The subsample fraction carries bernoulli; XGBoost has no goss.
        _Knob("dispatch.sampler_name", foreign_pairs={"all_rows": {}, "bernoulli": {}}),
        _Knob(
            "tree.monotone_constraints",
            "monotone_constraints",
            to_native=_monotone_directions,
            to_foreign=_xgb_monotone_directions,
        ),
        # bonsai spells a group "0,1"; XGBoost takes lists of indices, as a
        # JSON string, because its Booster reads a list as feature names.
        _Knob(
            "tree.interaction_constraints",
            "interaction_constraints",
            to_native=_interaction_strings,
            to_foreign=_xgb_interaction_groups,
        ),
    ),
    switches=(_XGB_BOOSTER,),
    implies={},
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

_LGBM_OBJECTIVES: Final = _Values(
    "objective",
    {
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
    },
)


# LightGBM bags only when bagging_freq is above 0; bonsai's bernoulli sampler
# draws every tree, which is bagging_freq=1, and has no every-k-trees form.
_LGBM_BAGGING_FREQ: Final = _Switch(
    "bagging_freq",
    "subsample_freq",
    table={"0": {}, "1": {"dispatch.sampler_name": "bernoulli"}},
    default=0,
)

# rf, random-forest mode, has no bonsai counterpart and raises.
_LGBM_BOOSTING: Final = _Switch(
    "boosting",
    "boosting_type",
    "boost",
    table={"gbdt": {}, "gbrt": {}, "dart": {}, "goss": {"dispatch.sampler_name": "goss"}},
    default="gbdt",
)

_LGBM_SAMPLE_STRATEGY: Final = _Switch(
    "data_sample_strategy",
    table={"bagging": {}, "goss": {"dispatch.sampler_name": "goss"}},
    default="bagging",
)

_LIGHTGBM: Final = _Library(
    name="lightgbm",
    title="LightGBM",
    knobs=(
        _Knob(
            "booster.n_iters",
            "n_estimators",
            "num_iterations",
            "num_boost_round",
            "num_round",
            "num_trees",
        ),
        _Knob("booster.learning_rate", "learning_rate", "shrinkage_rate", "eta"),
        # max_depth <= 0 is LightGBM's uncapped growth. bonsai has no
        # sentinel, so it travels as UNCAPPED_DEPTH and num_leaves stays the
        # binding budget, which is what governs a leafwise fit either way.
        # The reverse never re-invents the sentinel: 255 comes back as 255.
        _Knob("tree.max_depth", "max_depth", to_native=_uncapped_depth),
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
        _Knob(
            "sampler.subsample",
            "bagging_fraction",
            "subsample",
            read_under=_UNDER_BERNOULLI,
            when=_When(_LGBM_BAGGING_FREQ, 1),
        ),
        _Knob("sampler.top_rate", "top_rate", read_under=_UNDER_GOSS),
        _Knob("sampler.other_rate", "other_rate", read_under=_UNDER_GOSS),
        _Knob(
            "dispatch.sampler_name",
            foreign_pairs={
                "all_rows": {},
                "bernoulli": {"bagging_freq": 1},
                "goss": {"data_sample_strategy": "goss"},
            },
        ),
        _Knob("bin_mapper.max_bin", "max_bin"),
        _Knob("bin_mapper.min_data_in_bin", "min_data_in_bin"),
        _Knob("booster.random_seed", "seed", "random_state", "random_seed"),
        _Knob(
            "parallel.n_threads", "num_threads", "n_jobs", "nthread", to_native=_all_cores_as_auto
        ),
        # LightGBM arms patience through a callback, not this key; the key
        # is accepted here because its params-dict spelling is documented.
        _Knob("booster.early_stopping_rounds", "early_stopping_round", "early_stopping_rounds"),
        _Knob("dispatch.objective_name", "objective", "application", values=_LGBM_OBJECTIVES),
        _Knob("objective.n_classes", "num_class", "num_classes", read_under=_UNDER_SOFTMAX),
        # LightGBM overloads alpha: the target quantile under quantile, the
        # Huber delta under huber, 0.9 for both when unset, where bonsai
        # defaults to 0.5 and 1.0.
        _Knob(
            "objective.quantile_alpha",
            "alpha",
            read_under=_UNDER_QUANTILE,
            foreign_default=0.9,
        ),
        _Knob("objective.huber_delta", "alpha", read_under=_UNDER_HUBER, foreign_default=0.9),
        _Knob(
            "booster.dart_drop_rate",
            "drop_rate",
            when=_When(_LGBM_BOOSTING, "dart"),
            foreign_default=0.1,
            inert=0.0,
        ),
        _Knob(
            "tree.monotone_constraints",
            "monotone_constraints",
            to_native=_monotone_directions,
            to_foreign=_monotone_directions,
        ),
        # bonsai spells a group "0,1"; LightGBM takes lists of indices and
        # silently ignores the string form.
        _Knob(
            "tree.interaction_constraints",
            "interaction_constraints",
            to_native=_interaction_strings,
            to_foreign=_interaction_groups,
        ),
    ),
    # LightGBM grows leaf-wise only, and bonsai's default grower is
    # depthwise, which ignores tree.max_leaves; without this num_leaves
    # would never bind.
    switches=(_LGBM_BAGGING_FREQ, _LGBM_BOOSTING, _LGBM_SAMPLE_STRATEGY),
    implies={"dispatch.grower_name": "leafwise"},
    dropped_foreign={
        "verbose": "logging only",
        "verbosity": "logging only",
        "device": "in bonsai the device is the grower name, e.g. cuda_leafwise",
        "device_type": "in bonsai the device is the grower name",
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
_CATBOOST_OBJECTIVES: Final = _Values(
    "loss_function",
    {
        "RMSE": "mse",
        "MAE": "mae",
        "Poisson": "poisson",
        "Logloss": "logloss",
        "MultiClass": "softmax",
    },
)

_CATBOOST_GROW_POLICY: Final = _Policies(
    "grow_policy",
    {
        "SymmetricTree": "levelwise",
        "Depthwise": "depthwise",
        "Lossguide": "leafwise",
    },
)

# CatBoost's MVS and Bayesian bootstraps weight rows by gradient, which no
# bonsai sampler does, so they raise; Bernoulli is bonsai's uniform draw.
_CATBOOST_BOOTSTRAP: Final = _Switch(
    "bootstrap_type",
    table={
        "Bernoulli": {"dispatch.sampler_name": "bernoulli"},
        "No": {"dispatch.sampler_name": "all_rows"},
    },
    default="MVS",
)

_CATBOOST: Final = _Library(
    name="catboost",
    title="CatBoost",
    knobs=(
        _Knob("booster.n_iters", "iterations", "n_estimators", "num_boost_round", "num_trees"),
        _Knob("booster.learning_rate", "learning_rate", "eta"),
        _Knob("tree.max_depth", "depth", "max_depth"),
        # CatBoost reads max_leaves under Lossguide only, bonsai's leafwise.
        _Knob("tree.max_leaves", "max_leaves", "num_leaves", read_under=_UNDER_LEAFWISE),
        _Knob("tree.lambda_l2", "l2_leaf_reg", "reg_lambda"),
        # A node gate on bonsai's side, a child gate on CatBoost's; see the
        # LightGBM table's note.
        _Knob("tree.min_data_in_leaf", "min_data_in_leaf", "min_child_samples"),
        # The fencepost: border_count counts SPLITS where max_bin counts
        # BINS, so the two differ by one in every direction. Getting this
        # wrong shortchanged two published suites by a bin once.
        _Knob(
            "bin_mapper.max_bin",
            "border_count",
            "max_bin",
            to_native=lambda v: int(v) + 1,
            to_foreign=lambda v: int(v) - 1,
        ),
        # rsm samples features per level, tree.feature_fraction per tree, so
        # the same fraction decorrelates differently.
        _Knob("tree.feature_fraction", "rsm", "colsample_bylevel"),
        # Under CatBoost's default bootstrap subsample is an MVS rate, so
        # only a Bernoulli one translates; Bernoulli's own default is 0.66.
        _Knob(
            "sampler.subsample",
            "subsample",
            read_under=_UNDER_BERNOULLI,
            when=_When(_CATBOOST_BOOTSTRAP, "Bernoulli", reported=True),
            foreign_default=0.66,
        ),
        _Knob(
            "dispatch.sampler_name",
            foreign_pairs={
                "all_rows": {"bootstrap_type": "No"},
                "bernoulli": {"bootstrap_type": "Bernoulli"},
            },
        ),
        _Knob("booster.random_seed", "random_seed", "random_state"),
        _Knob("parallel.n_threads", "thread_count", to_native=_all_cores_as_auto),
        # CatBoost's patience is od_wait under od_type="Iter"; the
        # constructor's early_stopping_rounds sets the same detector. bonsai
        # disables patience with 0, which CatBoost refuses, so 0 stays unset.
        _Knob("booster.early_stopping_rounds", "early_stopping_rounds", "od_wait", inert=0),
        _Knob("dispatch.objective_name", "loss_function", "objective", values=_CATBOOST_OBJECTIVES),
        _Knob("objective.n_classes", "classes_count", read_under=_UNDER_SOFTMAX),
        _Knob("dispatch.grower_name", "grow_policy", values=_CATBOOST_GROW_POLICY),
        _Knob(
            "tree.monotone_constraints",
            "monotone_constraints",
            to_native=_monotone_directions,
            to_foreign=_monotone_directions,
        ),
    ),
    # CatBoost's default grow_policy is SymmetricTree, bonsai's levelwise.
    switches=(_CATBOOST_BOOTSTRAP,),
    implies={"dispatch.grower_name": "levelwise"},
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
    documents. A None value leaves that parameter unset, the way the
    library's scikit-learn wrapper means it, so an estimator's
    ``get_params()`` translates directly.
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
    return template.format(title=lib.title, name=lib.name, notes=notes, call=call, result=result)


# Public Functions =================================================================================


def from_xgboost(params: Mapping[str, Any], *, strict: bool = True) -> Params:
    return _from_library(_XGBOOST, params, strict)


from_xgboost.__doc__ = _doc(
    _FROM_DOC,
    _XGBOOST,
    notes="A ``max_leaves`` above 0 selects the ``leafwise`` grower whatever the\n"
    "``grow_policy``: XGBoost enforces the cap under depthwise growth too, and\n"
    "bonsai's depthwise grower ignores it. The leaf budget matches; the split\n"
    "order becomes best-first.",
    call='{"n_estimators": 300, "reg_lambda": 2.0}',
    result="Params(tree=Tree(lambda_l2=2.0), booster=Booster(n_iters=300))",
)


def from_lightgbm(params: Mapping[str, Any], *, strict: bool = True) -> Params:
    return _from_library(_LIGHTGBM, params, strict)


from_lightgbm.__doc__ = _doc(
    _FROM_DOC,
    _LIGHTGBM,
    notes="The translation always selects the ``leafwise`` grower, LightGBM's only\n"
    "growth policy, so ``num_leaves`` binds. ``max_depth <= 0`` (LightGBM's\n"
    "uncapped growth) becomes ``tree.max_depth = 255``: bonsai has no\n"
    "sentinel, and under leafwise growth ``num_leaves`` is the binding budget\n"
    "either way. ``bagging_fraction`` translates only with ``bagging_freq=1``:\n"
    "LightGBM bags only above frequency 0, and bonsai draws for every tree.\n"
    "``boosting='goss'`` selects the goss sampler, ``'dart'`` sets\n"
    "``booster.dart_drop_rate`` (LightGBM's 0.1 unless ``drop_rate`` is\n"
    "given), and ``'rf'`` raises. ``alpha`` becomes\n"
    "``objective.quantile_alpha`` or ``objective.huber_delta`` by objective,\n"
    "at LightGBM's 0.9 when unset.",
    call='{"num_leaves": 63, "max_depth": -1}',
    result="Params(tree=Tree(max_depth=255, max_leaves=63), "
    "dispatch=Dispatch(grower_name='leafwise'))",
)


def from_catboost(params: Mapping[str, Any], *, strict: bool = True) -> Params:
    return _from_library(_CATBOOST, params, strict)


from_catboost.__doc__ = _doc(
    _FROM_DOC,
    _CATBOOST,
    notes="Without a ``grow_policy`` the translation selects the ``levelwise``\n"
    "grower, CatBoost's default SymmetricTree policy. ``border_count`` counts\n"
    "splits where ``bin_mapper.max_bin`` counts bins, so the value gains one\n"
    "crossing over. CatBoost's parametrized losses\n"
    "(``Huber:delta=...``, ``Quantile:alpha=...``) raise: set\n"
    "``dispatch.objective_name`` and the loss parameter by hand.\n"
    "``subsample`` translates only under ``bootstrap_type='Bernoulli'``, at\n"
    "CatBoost's 0.66 when unset: CatBoost's default MVS and Bayesian\n"
    "bootstraps weight rows by gradient, which no bonsai sampler does.",
    call='{"depth": 6, "border_count": 254}',
    result="Params(bin_mapper=BinMapper(max_bin=255), tree=Tree(max_depth=6), "
    "dispatch=Dispatch(grower_name='levelwise'))",
)


def to_xgboost(
    pairs: ParamsOps | Mapping[str, Any] | Iterable[tuple[str, Any]], *, strict: bool = True
) -> dict[str, Any]:
    return _to_library(_XGBOOST, pairs, strict)


to_xgboost.__doc__ = _doc(
    _TO_DOC,
    _XGBOOST,
    notes="bonsai's ``levelwise`` grower has no XGBoost policy and raises.\n"
    "A key bonsai reads only under one selector, such as ``tree.max_leaves``\n"
    "under leafwise growth or ``sampler.subsample`` under the bernoulli\n"
    "sampler, is left out elsewhere, since bonsai ignores it there. DART\n"
    "goes out with ``booster='dart'``, the switch that turns ``rate_drop``\n"
    "on, and the goss sampler, which XGBoost lacks, raises.",
    call='[("tree.lambda_l2", 2.0)]',
    result="{'reg_lambda': 2.0}",
)


def to_lightgbm(
    pairs: ParamsOps | Mapping[str, Any] | Iterable[tuple[str, Any]], *, strict: bool = True
) -> dict[str, Any]:
    return _to_library(_LIGHTGBM, pairs, strict)


to_lightgbm.__doc__ = _doc(
    _TO_DOC,
    _LIGHTGBM,
    notes="The bernoulli sampler goes out with ``bagging_freq=1`` and DART with\n"
    "``boosting='dart'``, the switches that turn their rates on. The goss\n"
    "sampler goes out as ``data_sample_strategy='goss'``. The quantile and\n"
    "huber objectives carry their parameter as ``alpha``, stated or at\n"
    "bonsai's default, since LightGBM's own default is 0.9.",
    call='[("tree.max_leaves", 63)]',
    result="{'num_leaves': 63}",
)


def to_catboost(
    pairs: ParamsOps | Mapping[str, Any] | Iterable[tuple[str, Any]], *, strict: bool = True
) -> dict[str, Any]:
    return _to_library(_CATBOOST, pairs, strict)


to_catboost.__doc__ = _doc(
    _TO_DOC,
    _CATBOOST,
    notes="``bin_mapper.max_bin`` loses one crossing over: CatBoost's\n"
    "``border_count`` counts splits, not bins. bonsai's ``huber`` and\n"
    "``quantile`` objectives raise, since their CatBoost spellings carry a\n"
    "parameter. The bernoulli sampler goes out as\n"
    "``bootstrap_type='Bernoulli'`` and all_rows as ``'No'``; the goss\n"
    "sampler, which CatBoost lacks, raises.",
    call='[("bin_mapper.max_bin", 255)]',
    result="{'border_count': 254}",
)


# Private Functions ================================================================================


def _from_library(lib: _Library, params: Mapping[str, Any], strict: bool) -> Params:
    """Foreign dict to a ``Params``; the shared engine behind ``from_*``."""
    stated = {
        key: value
        for key, value in dict(params).items()
        if value is not None and key not in lib.dropped_foreign
    }
    knobs: dict[str, list[_Knob]] = {}
    for knob in lib.knobs:
        for spelling in knob.spellings:
            knobs.setdefault(spelling, []).append(knob)
    switches = {spelling: switch for switch in lib.switches for spelling in switch.spellings}
    translation = _Translation(dict(lib.implies))
    unmappable = []
    for key, value in stated.items():
        if key in switches:
            translation.implied.update(switches[key].implied(value))
        elif key in knobs:
            unmappable += translation.add(key, knobs[key], stated)
        else:
            unmappable.append(key)
    translated = translation.resolved()
    unmappable += _drop_unread(translation.read, translated)
    _fill_foreign_defaults(lib, stated, translated)
    _reject_unmappable(unmappable, strict, lib.name, "bonsai")
    return Params.from_dict(translated)


@dataclasses.dataclass
class _Translation:
    """The bonsai pairs a foreign dict translates to, by precedence:
    implied underneath, then mapped, then forced over both."""

    implied: dict[str, Any]
    mapped: dict[str, Any] = dataclasses.field(default_factory=dict)
    forced: dict[str, Any] = dataclasses.field(default_factory=dict)
    read: dict[str, list[_Knob]] = dataclasses.field(default_factory=dict)

    def add(self, key: str, candidates: list[_Knob], stated: Mapping[str, Any]) -> list[str]:
        """Translate one stated key through every knob it spells; return
        it if it must be reported."""
        reported = False
        for knob in candidates:
            if knob.when is not None and not knob.when.holds(stated):
                reported = reported or knob.when.reported
                continue
            if knob.is_inert(stated[key]):
                continue
            self.mapped[knob.native] = knob.inbound(stated[key])
            self.implied.update(knob.implies)
            self.forced.update(knob.forces)
            self.read.setdefault(key, []).append(knob)
        return [key] if reported else []

    def resolved(self) -> dict[str, Any]:
        return {**self.implied, **self.mapped, **self.forced}


def _drop_unread(read: Mapping[str, list[_Knob]], translated: dict[str, Any]) -> list[str]:
    """Remove each bonsai key the translated selectors leave unread, and
    name every foreign key left with none."""
    unread = []
    for key, candidates in read.items():
        dropped = [knob for knob in candidates if not knob.is_read(translated)]
        for knob in dropped:
            translated.pop(knob.native, None)
        if len(dropped) == len(candidates):
            unread.append(key)
    return unread


def _fill_foreign_defaults(lib: _Library, stated: Mapping[str, Any], translated: dict[str, Any]):
    """Translate the reference library's default for each unset key whose
    default differs from bonsai's, where both sides read it."""
    for knob in lib.knobs:
        if knob.foreign_default is None or any(s in stated for s in knob.spellings):
            continue
        if knob.when is not None and not knob.when.holds(stated):
            continue
        if knob.is_read(translated):
            translated.setdefault(knob.native, knob.inbound(knob.foreign_default))


def _to_library(
    lib: _Library, pairs: ParamsOps | Mapping[str, Any] | Iterable[tuple[str, Any]], strict: bool
) -> dict[str, Any]:
    """bonsai pairs to a foreign dict; the shared engine behind ``to_*``."""
    index = {knob.native: knob for knob in lib.knobs}
    out: dict[str, Any] = {}
    unmappable = []
    items = pairs.to_dict() if isinstance(pairs, ParamsOps) else dict(pairs)
    for key, value in items.items():
        if key in lib.dropped_native:
            continue
        knob = index.get(key)
        if knob is None:
            unmappable.append(key)
            continue
        if knob.is_inert(value) or not knob.is_read(items):
            continue
        out.update(knob.outbound_pairs(value))
    _reject_unmappable(unmappable, strict, "bonsai", lib.name)
    return {**out, **_bonsai_defaults(lib, items)}


def _bonsai_defaults(lib: _Library, items: Mapping[str, Any]) -> dict[str, Any]:
    """bonsai's default, translated, for each unset key the reference
    library defaults differently, where both sides read it."""
    out: dict[str, Any] = {}
    for knob in lib.knobs:
        if knob.foreign_default is None or knob.native in items or not knob.is_read(items):
            continue
        default = _registry_default(knob.native)
        if not knob.is_inert(default):
            out.update(knob.outbound_pairs(default))
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
