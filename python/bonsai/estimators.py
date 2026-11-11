"""The scikit-learn estimator layer over the native booster.

``BonsaiRegressor`` and ``BonsaiClassifier`` are what users construct; the
shared ``_BonsaiEstimator`` base holds the sklearn contract (get_params /
set_params / pickling / fitted-state hooks) without ever importing sklearn
at module scope. The public import path is the package root
(``bonsai.BonsaiRegressor``); this module is the implementation home.

Constructor arguments are bonsai's own names, and only those. A parameter
dict written for another library goes through ``bonsai.interop`` first,
which says in one place what each knob becomes and where the two meanings
differ.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bonsai._bonsai import Dataset, Model, _params_from_toml, default_config_toml, load, train
from bonsai._coerce import _as_f32, _as_targets
from bonsai._params import Params
from bonsai._params_ops import ParamsOps

__all__ = ["BonsaiClassifier", "BonsaiRegressor"]


# Shared Base ======================================================================================


@dataclasses.dataclass(frozen=True)
class _FittedTargets:
    """The fitted attributes a ``y`` determines, set on the estimator only
    once ``train`` has returned, so a fit that raises leaves the previous
    fit's attributes paired with the previous fit's model."""

    def attributes(self) -> dict[str, object]:
        """The attributes by their sklearn names, ``classes_`` and the like."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


@dataclasses.dataclass(frozen=True)
class _RegressionTargets(_FittedTargets):
    n_outputs_: int


@dataclasses.dataclass(frozen=True)
class _ClassTargets(_FittedTargets):
    classes_: np.ndarray
    n_classes_: int


class _BonsaiEstimator:
    """Shared sklearn-contract machinery for ``BonsaiRegressor`` and
    ``BonsaiClassifier``.

    Duck-types the scikit-learn estimator contract (``get_params`` /
    ``set_params`` / ``score`` / ``_estimator_type``) without subclassing
    ``sklearn.base.BaseEstimator``: sklearn is never a runtime dependency of
    this package, so ``clone``, ``Pipeline``, ``GridSearchCV`` and
    ``cross_val_score`` all work, but ``import bonsai`` never needs sklearn
    installed.

    Not part of the public API; the subclasses are what users construct and
    their docstrings carry the constructor parameters. A subclass supplies
    the objective, how targets are encoded for the native booster, how
    predictions are decoded back, and its own ``score``.
    """

    _estimator_type: ClassVar[str]
    n_iters: int
    learning_rate: float
    max_depth: int
    max_leaves: int
    grower: str
    sampler: str
    early_stopping_rounds: int
    n_threads: int
    random_seed: int
    max_bin: int | None
    subsample: float | None
    device: str | None
    params: dict | ParamsOps | None
    _fit_eval_history: list[float]
    _fit_early_stopping_rounds: int

    def __repr__(self) -> str:
        """sklearn-style: class name plus the non-default parameters."""
        defaults = _init_defaults(type(self))
        shown = {k: v for k, v in self.get_params().items() if v != defaults.get(k)}
        args = ", ".join(f"{k}={v!r}" for k, v in shown.items())
        return f"{type(self).__name__}({args})"

    def get_params(self, deep: bool = True) -> dict:
        """Read back every constructor parameter, unchanged since construction.

        Part of the sklearn estimator contract.

        Parameters
        ----------
        deep
            Accepted for API compatibility; these estimators have no nested
            estimators to recurse into, so it has no effect.

        Returns
        -------
        dict
            One entry per ``__init__`` parameter name.
        """
        return {name: getattr(self, name) for name in _constructor_names(type(self))}

    def set_params(self, **params) -> Self:
        """Set constructor attributes in place.

        Part of the sklearn estimator contract.

        Parameters
        ----------
        **params
            Constructor parameter names to new values.

        Returns
        -------
        _BonsaiEstimator
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If a name is not a constructor parameter (sklearn's own
            estimators do the same).
        """
        valid = self.get_params(deep=False)
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {type(self).__name__}. "
                    f"Valid parameters are: {sorted(valid)}."
                )
            setattr(self, key, value)
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Report fitted state exactly, without relying on attribute naming.

        sklearn's ``check_is_fitted`` (used by ``Pipeline`` etc.) looks for
        instance ``__dict__`` attributes ending in ``_``; ``n_iters_`` etc.
        are properties backed by ``_model``, so they never show up there.

        Returns
        -------
        bool
            True once ``fit`` or ``from_file`` has set ``_model``.
        """
        return self._model is not None

    def __sklearn_tags__(self):
        """Build the sklearn tag object by hand, without subclassing sklearn.

        Only needed because the installed sklearn (>=1.6) requires
        ``__sklearn_tags__`` on any estimator passed through ``clone``,
        ``Pipeline``, ``cross_val_score``, or ``GridSearchCV``, even
        duck-typed ones that never subclass ``BaseEstimator``. Mirrors what
        ``RegressorMixin``/``ClassifierMixin``/``BaseEstimator`` produce, so
        sklearn stays import-only-in-tests; empirically verified against the
        installed sklearn (see test suite). ``_estimator_type`` is what
        picks the per-type tag. Both estimators route NaN features as
        missing, so ``allow_nan`` is set and sklearn's meta-estimators pass
        NaN through; the regressor fits several targets natively, which is
        ``MultiOutputMixin``'s ``multi_output``.

        Returns
        -------
        sklearn.utils.Tags
            The tag object sklearn's tooling introspects.
        """
        from sklearn.utils import (
            ClassifierTags,
            InputTags,
            RegressorTags,
            Tags,
            TargetTags,
        )

        kind = self._estimator_type
        is_regressor = kind == "regressor"
        per_type = (
            {"regressor_tags": RegressorTags()}
            if is_regressor
            else {"classifier_tags": ClassifierTags()}
        )
        tags: Any = Tags
        target_tags: Any = TargetTags
        input_tags: Any = InputTags
        return tags(
            estimator_type=kind,
            target_tags=target_tags(required=True, multi_output=is_regressor),
            input_tags=input_tags(allow_nan=True),
            **per_type,
        )

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sample_weight: npt.ArrayLike | None = None,
        eval_set: tuple[npt.ArrayLike, npt.ArrayLike] | Dataset | None = None,
        init_model: str | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> Self:
        """Fit to training data.

        ``BonsaiClassifier`` reads its classes off ``y`` here and picks
        ``logloss`` for two of them or ``softmax`` (with
        ``objective.n_classes`` set) for more.

        Parameters
        ----------
        X
            Feature matrix.
        y
            Regression targets, shape ``(n_rows,)`` or ``(n_rows, n_targets)``
            for a multi-target fit (one tree per target per round, and
            ``predict`` returns one column per target); for
            ``BonsaiClassifier`` class labels, any
            hashable/orderable values (ints, strings, ...), encoded to
            ``0..K-1`` internally and decoded back to the original
            ``classes_`` values by ``predict``.
        sample_weight
            Per-row weights, finite and non-negative with at least one
            positive (sklearn's convention). A weight scales its row's
            gradient and hessian and counts the row in the base score and
            in the leaf values MAE, Huber and Quantile renew, so the label
            of a zero-weight row never reaches the fit. None weights every
            row equally.
        eval_set
            One bare ``(X, y)`` tuple, Dataset, or None. A ``bonsai.Dataset``
            must be built with ``reference=`` the training dataset, which
            bins the validation rows once and reuses them across fits; for a
            classifier its labels are already the encoded ``0..K-1``.
            bonsai tracks a single validation set, so a list of them is
            rejected rather than silently reduced to its last entry.
        init_model
            Path to a saved ``.msgpack`` model to continue training from
            (warm start); binning reuses the loaded model's cut points. None
            trains from scratch.
        feature_names
            One name per column, overriding any string column names X
            carries. Fitted names land in ``feature_names_in_``, ``dump``
            prints them, and ``tree.monotone_constraints`` may be keyed by
            them. None with an unnamed X leaves the columns ``f0``..``fN``.
            Rejected with ``init_model``: a warm start keeps the loaded
            model's names. Predict-time inputs and the eval_set's X are
            checked against the fitted names, sklearn's rule: names that go
            missing or appear warn, names that disagree raise.

        Returns
        -------
        _BonsaiEstimator
            ``self``, fitted.

        Raises
        ------
        ValueError
            If ``eval_set`` is a list, or given at all with
            ``dart_drop_rate > 0``; if ``X`` has no rows or no columns,
            ``y`` holds a non-finite value, or ``sample_weight`` a
            non-finite or negative one or no positive one; for a classifier
            also if ``y`` contains NaN (in any dtype), holds floats that are
            not finite integers (a continuous target), or has fewer than 2
            classes, or an eval-set label was not seen in the training
            classes.
        """
        _reject_eval_set_list(eval_set)
        y_enc, targets = self._encode_targets(y)
        overrides = (
            self._warm_start_overrides(targets)
            if init_model is not None
            else self._build_overrides(targets)
        )
        names = self._fit_names(X, feature_names, init_model)
        ev = None
        if eval_set is not None:
            if init_model is None:
                self._check_names_against(names, _eval_set_features(eval_set))
            ev = self._encode_eval_set(eval_set, targets)
        sw = None if sample_weight is None else _as_f32(sample_weight, 1, "sample_weight")
        Xa = _as_f32(X, 2, "X")
        model = train(
            overrides,
            Xa,
            y_enc,
            ev,
            init_model,
            sample_weight=sw,
            feature_names=names,
        )
        self.__dict__.update(targets.attributes())
        self._model = model
        self._record_fit_inputs(Xa)
        self._record_eval(model, overrides)
        return self

    def predict(self, X: npt.ArrayLike | Dataset, num_iteration: int = 0) -> np.ndarray:
        """Predict with the fitted model.

        Parameters
        ----------
        X
            Feature matrix, or a ``bonsai.Dataset``: one carrying the
            model's cuts routes in bin space (device-resident builds
            included), others read the matrix they retained.
        num_iteration
            Truncates the ensemble to its first ``n`` trees, as on
            ``Model.predict``; 0 uses every tree. A prefix always starts at
            the head, because a boosted sum has no meaning without it.

        Returns
        -------
        numpy.ndarray
            Predictions, one per row of ``X``, or ``(n_rows, n_targets)`` for
            a multi-target regressor; ``BonsaiClassifier`` returns
            the original class labels (from ``classes_``), not the encoded
            ``0..K-1`` ids the native booster works in.
        """
        raw = np.asarray(self._fitted().predict(self._predict_input(X), num_iteration))
        return self._decode_predictions(raw)

    def staged_predict(self, X: npt.ArrayLike | Dataset) -> np.ndarray:
        """Predict after each boosting iteration.

        Parameters
        ----------
        X
            As for ``predict``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_iters, n_rows)``, or ``(n_iters, n_rows, n_targets)`` for
            a multi-target regressor: predictions after each boosting
            iteration, each stage what ``predict`` would return for that
            prefix (class labels for ``BonsaiClassifier``).
        """
        raw = np.asarray(self._fitted().staged_predict(self._predict_input(X)))
        return self._decode_predictions(raw)

    def predict_leaf(self, X: npt.ArrayLike | Dataset) -> np.ndarray:
        """Predict per-tree leaf indices, for feature engineering / embedding.

        Parameters
        ----------
        X
            As for ``predict``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_rows, n_trees)``. One column per tree in training
            order, so a softmax model (one tree per class per round) has
            ``n_iters * n_classes`` columns and column ``t`` is round
            ``t // n_classes``, class ``t % n_classes``.
        """
        return np.asarray(self._fitted().predict_leaf(self._predict_input(X)))

    def dump(self) -> str:
        """Dump the model as text.

        Returns
        -------
        str
            Every tree as indented text.
        """
        self._check_fitted()
        return self._fitted().dump()

    def pred_contribs(self, X: npt.ArrayLike | Dataset) -> np.ndarray:
        """Compute TreeSHAP contributions.

        Parameters
        ----------
        X
            As for ``predict``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_rows, n_features + 1)``; the last column is the bias.
            Rows sum to the raw (pre-link) prediction exactly.
        """
        return np.asarray(self._fitted().pred_contribs(self._predict_input(X)))

    def importance(self, type: str = "gain") -> np.ndarray:
        """Compute raw per-feature importance.

        Parameters
        ----------
        type
            ``"gain"`` for total split gain, ``"split"`` for split count.

        Returns
        -------
        numpy.ndarray
            One value per feature, unnormalized.
        """
        self._check_fitted()
        return np.asarray(self._fitted().feature_importance(type))

    @property
    def feature_importances_(self) -> np.ndarray:
        """numpy.ndarray: gain importance normalized to sum to 1 (sklearn convention)."""
        raw = self.importance("gain")
        total = raw.sum()
        return raw / total if total > 0 else raw

    def save(self, path: str):
        """Serialize the fitted model to a ``.msgpack`` file.

        Parameters
        ----------
        path
            Destination file path.
        """
        self._check_fitted()
        self._fitted().save(path)

    @classmethod
    def from_file(cls, path: str) -> Self:
        """Load a fitted model from a saved ``.msgpack`` file.

        The loaded model carries its feature count and its column names, so
        ``n_features_in_`` and ``feature_names_in_`` come back with it. Its
        resolved config becomes the estimator's parameters, so
        ``get_params`` describes the model and ``clone(...).fit`` trains the
        same configuration.

        Parameters
        ----------
        path
            Path to a model saved with ``save``.

        Returns
        -------
        _BonsaiEstimator
            A fresh estimator instance wrapping the loaded model.
        """
        model = load(path)
        out = cls(**cls._arguments_from_model(model))
        out._model = model
        names = _model_feature_names(out._model)
        out.n_features_in_ = len(out._fitted().feature_names)
        out._set_feature_names(names)
        return out

    @property
    def n_iters_(self) -> int:
        self._check_fitted()
        return self._fitted().n_iters

    def evals_result(self) -> dict:
        """Read back the per-round eval-set loss history.

        Requires a fit with an ``eval_set``; empty after loading from a file.
        The metric is the objective's own loss under its bonsai name and its
        own units: the ``mse`` objective reports squared error, not its
        root. After an ``init_model`` warm start only the continuation's
        measured rounds are reported (the warm-start rounds were never
        evaluated; ``best_iteration`` still counts absolute rounds).

        Returns
        -------
        dict
            ``{"valid": {objective_name: [...]}}``. The outer key names the
            eval set, spelled the way the CLI labels its metric column and
            ``data.valid`` names the file. Empty if there is no eval-set
            history.
        """
        self._check_fitted()
        name = self._fitted().objective_name
        hist = self._fit_eval_history
        start = next((i for i, v in enumerate(hist) if not np.isnan(v)), len(hist))
        hist = hist[start:]
        if not hist:
            return {}
        return {"valid": {name: hist}}

    @property
    def best_iteration(self) -> int:
        """int: 0-based absolute model round with the best eval-set loss.

        Defined only when fit ran with early stopping and an ``eval_set``: it
        is the round early stopping kept. After an ``init_model`` warm start
        the index counts the warm-start rounds too, so it lines up with
        ``n_iters_`` and ``predict(num_iteration=best_iteration + 1)``.

        Raises
        ------
        AttributeError
            If fit did not run with ``eval_set`` and ``early_stopping_rounds``.
        """
        # nanargmin: warm-start rounds are unmeasured NaN placeholders.
        return int(np.nanargmin(self._early_stopping_history("best_iteration")))

    @property
    def best_score(self) -> float:
        """float: eval-set loss at ``best_iteration``.

        In the objective's own units (the same values ``evals_result``
        reports).

        Raises
        ------
        AttributeError
            If fit did not run with ``eval_set`` and ``early_stopping_rounds``.
        """
        return float(np.nanmin(self._early_stopping_history("best_score")))

    def __getstate__(self) -> dict:
        """Pickle support: the native model rides along as msgpack bytes."""
        state = self.__dict__.copy()
        model = state.pop("_model", None)
        if model is not None:
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "m.msgpack"
                model.save(str(path))
                state["_model_bytes"] = path.read_bytes()
        return state

    def __setstate__(self, state: dict):
        """Pickle support: restore the native model from its msgpack bytes.

        The eval history rides in the state beside the bytes, since the
        saved model carries none; a pickle written before the history was
        kept comes back without one.
        """
        model_bytes = state.pop("_model_bytes", None)
        self._forget_eval()
        self.__dict__.update(state)
        if model_bytes is not None:
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "m.msgpack"
                path.write_bytes(model_bytes)
                self._model = load(str(path))
        else:
            self._model = None

    # Every dispatch or objective key a constructor argument writes, with all
    # the arguments that write it and every params key that does: `device`
    # rewrites the grower for a device, a subsample fraction turns the
    # sampler on. A warm start may inherit the model's value for a key only
    # when all of the arguments are still at their defaults and params names
    # none of the keys.
    _WARM_START_KNOBS = (
        ("dispatch.grower_name", ("grower", "device"), ("dispatch.grower_name",)),
        (
            "dispatch.sampler_name",
            ("sampler", "subsample"),
            ("dispatch.sampler_name", "sampler.subsample"),
        ),
    )

    # Each constructor argument with the config key it writes: fit writes
    # the key from the argument, and from_file reads the argument back from
    # a model's resolved config. Keys a fit derives from its data, or never
    # stores, stay out of the loaded estimator's params.
    _MODEL_ARGUMENTS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("n_iters", "booster.n_iters"),
        ("learning_rate", "booster.learning_rate"),
        ("max_depth", "tree.max_depth"),
        ("max_leaves", "tree.max_leaves"),
        ("grower", "dispatch.grower_name"),
        ("sampler", "dispatch.sampler_name"),
        ("early_stopping_rounds", "booster.early_stopping_rounds"),
        ("n_threads", "parallel.n_threads"),
        ("random_seed", "booster.random_seed"),
        ("max_bin", "bin_mapper.max_bin"),
    )
    _DERIVED_KEYS: ClassVar[tuple[str, ...]] = ("parallel.device_id",)

    @classmethod
    def _arguments_from_model(cls, model: Model) -> dict[str, object]:
        """Constructor arguments that reproduce ``model``'s resolved config.

        An argument whose default is a value takes the model's value, since
        the estimator's defaults are not the library's. An argument whose
        default is None (leave the library default) is set only where the
        model differs from the library. Every other key that differs goes to
        ``params``, except the ``data`` section, which names input files,
        and the keys a fit derives from ``y``.

        Parameters
        ----------
        model
            A loaded model.

        Returns
        -------
        dict
            Keyword arguments for ``cls``.
        """
        resolved = _shortest_spellings(Params.from_model(model).to_dict())
        library = _shortest_spellings(_params_from_toml(default_config_toml()))
        defaults = _init_defaults(cls)
        changed = {k: v for k, v in resolved.items() if v != library.get(k)}
        arguments: dict[str, object] = {}
        for name, key in cls._MODEL_ARGUMENTS:
            if defaults[name] is not None:
                arguments[name] = resolved[key]
            elif key in changed:
                arguments[name] = changed[key]
        params = cls._params_from_model(changed, resolved)
        if params:
            arguments["params"] = params
        return arguments

    @classmethod
    def _params_from_model(
        cls, changed: dict[str, object], resolved: dict[str, object]
    ) -> dict[str, object]:
        """The ``params`` of a loaded estimator: the changed keys no
        constructor argument carries. A ``sampler.subsample`` brings the
        model's sampler with it, so a fraction the ``all_rows`` sampler
        ignored does not turn on ``bernoulli`` at refit."""
        mapped = {key for _, key in cls._MODEL_ARGUMENTS}
        params = {
            k: v
            for k, v in changed.items()
            if k not in mapped and k not in cls._DERIVED_KEYS and not k.startswith("data.")
        }
        if "sampler.subsample" in params:
            params["dispatch.sampler_name"] = resolved["dispatch.sampler_name"]
        return params

    def _store(self, arguments: dict):
        """Keep every constructor argument raw; ``get_params``/``clone`` read
        them back. The subclass signatures stay explicit and complete, since
        sklearn reads them with ``inspect.signature`` and ``**kwargs`` there
        would erase the parameters."""
        for name in _constructor_names(type(self)):
            setattr(self, name, arguments[name])
        self._model: Model | None = None
        self._forget_eval()

    def _objective_pairs(self, targets: _FittedTargets) -> dict[str, object]:
        """Config keys the objective derives from ``targets``, beyond the
        constructor arguments: ``BonsaiClassifier`` chooses its objective
        by the class count, ``BonsaiRegressor`` adds its target count."""
        raise NotImplementedError

    def _encode_targets(self, y: npt.ArrayLike) -> tuple[np.ndarray, _FittedTargets]:
        """The training targets as the native booster takes them, and the
        fitted attributes they determine."""
        raise NotImplementedError

    def _encode_eval_targets(self, y: npt.ArrayLike, targets: _FittedTargets) -> np.ndarray:
        """Eval-set targets, encoded against what ``_encode_targets`` saw."""
        raise NotImplementedError

    def _decode_predictions(self, raw: np.ndarray) -> np.ndarray:
        """What ``predict`` returns for the native booster's output."""
        return raw

    def _encode_eval_set(
        self, eval_set, targets: _FittedTargets
    ) -> tuple[np.ndarray, np.ndarray] | Dataset:
        """A pre-binned ``Dataset`` passes through: it was built before
        ``fit`` saw any class, so a classifier's labels there are already the
        encoded ids and the native layer range-checks them."""
        if isinstance(eval_set, Dataset):
            return eval_set
        X_eval = _as_f32(eval_set[0], 2, "X")
        y_shape = np.shape(eval_set[1])
        if len(y_shape) == 2 and y_shape[0] != X_eval.shape[0]:
            raise ValueError(
                f"eval_set y has {y_shape[0]} rows and eval_set X has {X_eval.shape[0]}; "
                "a 2-D y is n_rows x n_targets"
            )
        return (X_eval, self._encode_eval_targets(eval_set[1], targets))

    def _build_overrides(self, targets: _FittedTargets) -> dict[str, object]:
        """Translate the first-class kwargs and ``params`` into the dotted
        config keys the native ``train()`` expects. Kept out of ``__init__``
        so constructor args stay raw attributes (required for
        ``get_params``/``clone``).

        Precedence (lowest to highest): first-class kwargs, then the
        optional ones that carry a ``None`` "leave it alone" default, then
        ``params`` (the power-user escape hatch always has the final
        word)."""
        merged: dict[str, object] = {}
        for name, key in self._MODEL_ARGUMENTS:
            value = getattr(self, name)
            if value is not None:
                merged[key] = value
        merged.update(self._objective_pairs(targets))
        if self.subsample is not None:
            merged["sampler.subsample"] = self.subsample
        if self.device is not None:
            merged.update(_device_pairs(str(merged["dispatch.grower_name"]), self.device))
        explicit = self._explicit_params()
        merged.update(explicit)
        if self._subsample_implies_bernoulli(explicit):
            merged["dispatch.sampler_name"] = "bernoulli"
        return merged

    def _subsample_implies_bernoulli(self, explicit: dict[str, object]) -> bool:
        """Whether a subsample fraction turns row sampling on.

        bonsai makes the sampler explicit, and the default ``all_rows``
        sampler ignores ``sampler.subsample``. A fraction, from the
        ``subsample`` argument or from ``params``, therefore switches to
        ``bernoulli`` while the sampler is still at its default. An explicit
        ``sampler`` argument or ``dispatch.sampler_name`` keeps its own
        machinery.
        """
        fraction_given = self.subsample is not None or "sampler.subsample" in explicit
        return (
            fraction_given
            and self.sampler == "all_rows"
            and "dispatch.sampler_name" not in explicit
        )

    def _explicit_params(self) -> dict[str, object]:
        """The ``params`` escape hatch as a dict, whichever form it was given in."""
        if isinstance(self.params, ParamsOps):
            return self.params.to_dict()
        return self.params or {}

    def _warm_start_overrides(self, targets: _FittedTargets) -> dict[str, object]:
        """Overrides for a warm start, with the untouched knobs dropped.

        The native reconcile reads any key it is handed as user-chosen, so
        a key still carrying its constructor default has to be withheld for
        the loaded model's own value to be inherited rather than refused.
        A key survives when any constructor argument that writes it was
        passed, when ``params`` names it, or when the value comes from the
        data (``BonsaiClassifier`` derives its objective from ``y``, which
        a warm start should still be checked against).
        """
        overrides = self._build_overrides(targets)
        explicit = self._explicit_params()
        defaults = _init_defaults(type(self))
        for key, args, params_keys in self._WARM_START_KNOBS:
            if any(k in explicit for k in params_keys):
                continue
            if all(getattr(self, a) == defaults[a] for a in args):
                overrides.pop(key, None)
        return overrides

    def _check_fitted(self):
        """One guard for every fitted-state precondition (the exception type
        is part of the API)."""
        self._fitted()

    def _fitted(self) -> Model:
        """The native model, once ``fit`` or ``from_file`` has set it."""
        if self._model is None:
            raise _not_fitted_error()(
                f"This {type(self).__name__} instance is not fitted yet: "
                "call fit() or from_file() first"
            )
        return self._model

    def _predict_input(self, X: npt.ArrayLike | Dataset) -> np.ndarray | Dataset:
        """What the predict family hands the native model, once the model
        exists and X's names agree with the fitted ones: a host-built
        ``Dataset`` passes straight through (the native reader takes the
        matrix it retained), anything else coerces to a float32 matrix."""
        self._check_fitted()
        self._check_predict_names(X)
        return X if isinstance(X, Dataset) else _as_f32(X, 2, "X")

    def _early_stopping_history(self, what: str) -> list[float]:
        """The eval history, when fit ran with early stopping; ``what`` names
        the property asking, for the error."""
        self._check_fitted()
        hist = self._fit_eval_history
        if not self._fit_early_stopping_rounds or not hist:
            raise AttributeError(f"{what} needs fit(eval_set=...) with early_stopping_rounds set")
        return hist

    def _check_predict_names(self, X: object):
        """Compare the predict input's names with the fitted ones.

        The predict half of sklearn's feature-name contract, matching
        ``sklearn.utils.validation._check_feature_names(reset=False)``: a
        disagreement about whether there are names at all is a warning, since
        the columns may still be in the right order, but two sets of names
        that differ is an error.

        Parameters
        ----------
        X
            The predict input, before coercion.

        Raises
        ------
        ValueError
            If both the fit and X have names and they differ.
        """
        fitted = getattr(self, "feature_names_in_", None)
        self._check_names_against(None if fitted is None else [str(name) for name in fitted], X)

    def _check_names_against(self, fitted: list[str] | None, X: object):
        """Compare X's names with ``fitted``, the names a fit trains or
        trained under, by the rule ``_check_predict_names`` documents."""
        given = _input_feature_names(X)
        if fitted is None and given is None:
            return
        if given is None or fitted is None:
            self._warn_names_on_one_side(fitted_has_names=given is None)
            return
        if fitted != given:
            raise ValueError(_names_disagreement(fitted, given))

    def _warn_names_on_one_side(self, fitted_has_names: bool):
        """Names on only one side of fit and predict: a warning, since the
        columns may still be in the right order."""
        estimator = type(self).__name__
        if fitted_has_names:
            message = (
                f"X does not have valid feature names, but {estimator} was "
                "fitted with feature names"
            )
        else:
            message = f"X has feature names, but {estimator} was fitted without feature names"
        warnings.warn(message, stacklevel=4)

    def _fit_names(
        self, X: object, feature_names: Sequence[str] | None, init_model: str | None
    ) -> list[str] | None:
        """The names to train under, read before X is coerced to an array.

        A warm start keeps the loaded model's names, so names merely
        discovered on X are dropped rather than forwarded: they were never an
        override, and passing them on would trip the native guard against
        naming a warm start's columns.

        Parameters
        ----------
        X
            The fit input, still the caller's own object.
        feature_names
            The ``fit`` keyword, or None.
        init_model
            The ``fit`` keyword, or None.

        Returns
        -------
        list of str or None
            Names for ``train``; None leaves the columns unnamed (or, under a
            warm start, named by the loaded model).

        Raises
        ------
        ValueError
            If names are given explicitly alongside ``init_model``.
        """
        explicit = feature_names is not None
        names = list(feature_names) if explicit else _feature_names_of(X)
        if init_model is None:
            return names
        if explicit:
            raise ValueError(
                "feature_names cannot be given with init_model: a warm start "
                "keeps the loaded model's feature names"
            )
        return None

    def _record_fit_inputs(self, Xa: np.ndarray):
        """Record what the fit saw, the way sklearn's estimators do.

        The names come back off the fitted model rather than from the fit
        arguments, so the attribute reports what the model actually carries;
        that is the only answer a warm start has, since it keeps the loaded
        model's names.

        ``feature_names_in_`` is absent rather than None when the columns are
        unnamed, which is what ``hasattr`` checks (sklearn's own convention)
        and what a refit on a bare array has to restore.

        Parameters
        ----------
        Xa
            The coerced feature matrix the fit ran on.
        """
        self.n_features_in_ = Xa.shape[1]
        self._set_feature_names(_model_feature_names(self._fitted()))

    def _forget_eval(self):
        """The eval record of an estimator no fit has run on."""
        self._fit_eval_history = []
        self._fit_early_stopping_rounds = 0

    def _record_eval(self, model: Model, overrides: dict[str, object]):
        """Keep the fit's eval history and the early stopping it ran under.

        The rounds come from the overrides the fit resolved, where
        ``params`` has the last word over the ``early_stopping_rounds``
        argument. The history is copied off the model because a saved or
        pickled model carries none.
        """
        self._fit_eval_history = [float(v) for v in model.eval_history]
        self._fit_early_stopping_rounds = int(
            cast("int", overrides["booster.early_stopping_rounds"])
        )

    def _set_feature_names(self, names: list[str] | None):
        """Set ``feature_names_in_``, or remove it when there are no names."""
        if names is None:
            self.__dict__.pop("feature_names_in_", None)
        else:
            self.feature_names_in_ = np.asarray(names, dtype=object)


# Regressor ========================================================================================


class BonsaiRegressor(_BonsaiEstimator):
    """sklearn-style wrapper around the native booster.

    Every argument is a bonsai name. First-class arguments cover the common
    knobs; anything else can be set through ``params`` using dotted config
    keys, e.g. ``params={"tree.lambda_l1": 0.5, "sampler.top_rate": 0.2}``,
    the same keys the CLI accepts via ``--set``. A parameter dict written
    for another library goes through ``bonsai.interop`` first, which is
    where the knob mapping and its caveats live.

    A TOML file routes through ``params``
    (``params=Params.from_toml(path).to_dict()``) and therefore wins over
    the keyword arguments, the way ``params`` always has. For file-as-base
    layering below explicit overrides, use ``train()`` with
    ``Params.from_toml(path) | overrides``.

    Parameters
    ----------
    n_iters
        Number of boosting rounds.
    learning_rate
        Shrinkage applied to each tree's contribution.
    max_depth
        Maximum tree depth.
    max_leaves
        Maximum leaves per tree, the leafwise growth budget.
    grower
        Growth strategy name (``"leafwise"``, ``"depthwise"``,
        ``"levelwise"``, ...); combined with ``device`` to pick the concrete
        grower.
    sampler
        Row sampler name; ``"all_rows"`` trains on every row.
    objective
        One of bonsai's own: ``"mse"``, ``"mae"``, ``"huber"``,
        ``"quantile"`` (with ``quantile_alpha``), or ``"poisson"``.
    early_stopping_rounds
        Patience in rounds before stopping on eval-set loss; 0 disables
        early stopping.
    n_threads
        Worker thread count; 0 uses every available core.
    random_seed
        Seed for every draw the fit makes: row sampling, DART drops and,
        with ``tree.feature_seed``, the feature_fraction draws.
    max_bin
        Histogram bin cap; None leaves the native default.
    subsample
        Row-sampling fraction; switches the sampler to ``"bernoulli"`` when
        ``sampler`` is still at its default, as ``sampler.subsample`` in
        ``params`` does. None leaves sampling off.
    device
        ``"cpu"`` or ``"cuda"`` (or ``"cuda:N"``); ``"cuda"`` picks the CUDA
        grower matching the chosen ``grower``, moving compute without
        changing the growth strategy, and ``"cuda:N"`` also sets
        ``parallel.device_id`` to ``N``. None leaves the grower name as given.
    quantile_alpha
        The quantile to fit when ``objective="quantile"``.
    params
        Overrides beyond the first-class kwargs: a ``bonsai.Params`` or a
        dotted config-key dict; the power-user escape hatch, applied after
        the first-class kwargs and always wins.
    """

    _estimator_type = "regressor"
    n_outputs_: int
    objective: str
    quantile_alpha: float | None

    def __init__(
        self,
        n_iters: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        max_leaves: int = 31,
        grower: str = "leafwise",
        sampler: str = "all_rows",
        objective: str = "mse",
        early_stopping_rounds: int = 0,
        n_threads: int = 0,
        random_seed: int = 42,
        max_bin: int | None = None,
        subsample: float | None = None,
        device: str | None = None,
        quantile_alpha: float | None = None,
        params: dict | ParamsOps | None = None,
    ):
        self._store(locals())

    def score(
        self,
        X: npt.ArrayLike | Dataset,
        y: npt.ArrayLike,
        sample_weight: npt.ArrayLike | None = None,
    ) -> float:
        """Compute R² (coefficient of determination) on held-out data.

        Matches sklearn's ``RegressorMixin.score``, computed by hand with no
        sklearn import.

        Parameters
        ----------
        X
            Feature matrix.
        y
            True regression targets.
        sample_weight
            Per-row weights for the R² computation. None weights every row
            equally.

        Returns
        -------
        float
            The R² score.

        Raises
        ------
        ValueError
            If ``y`` or ``sample_weight`` disagrees with the predictions in
            row count, or ``y`` in target count.
        """
        y_true = _as_targets(y, "y", np.float64)
        y_pred = np.asarray(self.predict(X), dtype=np.float64)
        w = (
            None
            if sample_weight is None
            else _as_f32(sample_weight, 1, "sample_weight").astype(np.float64)
        )
        _check_scored_shapes(y_true, y_pred, w)
        if y_true.ndim == 1:
            return _r2(y_true, y_pred, w)
        per_target = [_r2(y_true[:, k], y_pred[:, k], w) for k in range(y_true.shape[1])]
        return float(np.mean(per_target))

    _WARM_START_KNOBS = (
        *_BonsaiEstimator._WARM_START_KNOBS,
        ("dispatch.objective_name", ("objective", "quantile_alpha"), ("dispatch.objective_name",)),
    )

    _MODEL_ARGUMENTS = (
        *_BonsaiEstimator._MODEL_ARGUMENTS,
        ("objective", "dispatch.objective_name"),
        ("quantile_alpha", "objective.quantile_alpha"),
    )
    _DERIVED_KEYS = (*_BonsaiEstimator._DERIVED_KEYS, "objective.n_targets")

    @classmethod
    def from_file(cls, path: str) -> BonsaiRegressor:
        """Load a saved regressor; ``n_outputs_`` is read off the model."""
        out = super().from_file(path)
        out.n_outputs_ = int(out._fitted().n_targets)
        return out

    def _objective_pairs(self, targets: _FittedTargets) -> dict[str, object]:
        if isinstance(targets, _RegressionTargets) and targets.n_outputs_ > 1:
            return {"objective.n_targets": targets.n_outputs_}
        return {}

    def _encode_targets(self, y: npt.ArrayLike) -> tuple[np.ndarray, _FittedTargets]:
        arr = _as_targets(y, "y")
        return arr, _RegressionTargets(n_outputs_=int(arr.shape[1]) if arr.ndim == 2 else 1)

    def _encode_eval_targets(self, y: npt.ArrayLike, targets: _FittedTargets) -> np.ndarray:
        return _as_targets(y, "y").reshape(-1)


def _r2(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray | None) -> float:
    """R2 of one target, weighted when ``w`` is given; 1.0 for a perfect
    fit of a constant target and 0.0 for any other constant target."""
    if w is None:
        avg_true = y_true.mean()
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - avg_true) ** 2)
    else:
        avg_true = np.average(y_true, weights=w)
        ss_res = np.sum(w * (y_true - y_pred) ** 2)
        ss_tot = np.sum(w * (y_true - avg_true) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


# Classifier =======================================================================================


class BonsaiClassifier(_BonsaiEstimator):
    """sklearn-style classifier wrapping the native booster's ``logloss``
    (binary) and ``softmax`` (multiclass) objectives.

    Same first-class knobs as ``BonsaiRegressor`` except there is no
    ``objective`` argument: ``fit`` picks ``logloss`` for two classes or
    ``softmax`` (with ``objective.n_classes`` set) for more, based on
    ``np.unique(y)``. Labels may be any hashable/orderable values (ints,
    strings, ...); they're encoded to ``0..K-1`` internally and decoded back
    to the original ``classes_`` values by ``predict``.

    ``predict_proba`` covers both cases: binary from the native ``logloss``
    P(class 1), multiclass from the ``softmax`` booster's per-class
    probabilities; see its docstring.

    Parameters
    ----------
    n_iters
        Number of boosting rounds.
    learning_rate
        Shrinkage applied to each tree's contribution.
    max_depth
        Maximum tree depth.
    max_leaves
        Maximum leaves per tree, the leafwise growth budget.
    grower
        Growth strategy name (``"leafwise"``, ``"depthwise"``,
        ``"levelwise"``, ...); combined with ``device`` to pick the concrete
        grower.
    sampler
        Row sampler name; ``"all_rows"`` trains on every row.
    early_stopping_rounds
        Patience in rounds before stopping on eval-set loss; 0 disables
        early stopping.
    n_threads
        Worker thread count; 0 uses every available core.
    random_seed
        Seed for every draw the fit makes: row sampling, DART drops and,
        with ``tree.feature_seed``, the feature_fraction draws.
    max_bin
        Histogram bin cap; None leaves the native default.
    subsample
        Row-sampling fraction; switches the sampler to ``"bernoulli"`` when
        ``sampler`` is still at its default, as ``sampler.subsample`` in
        ``params`` does. None leaves sampling off.
    device
        ``"cpu"`` or ``"cuda"`` (or ``"cuda:N"``); ``"cuda"`` picks the CUDA
        grower matching the chosen ``grower``, moving compute without
        changing the growth strategy, and ``"cuda:N"`` also sets
        ``parallel.device_id`` to ``N``. None leaves the grower name as given.
    params
        Overrides beyond the first-class kwargs: a ``bonsai.Params`` or a
        dotted config-key dict; the power-user escape hatch, applied after
        the first-class kwargs and always wins.
    """

    _estimator_type = "classifier"
    n_classes_: int
    classes_: np.ndarray

    def __init__(
        self,
        n_iters: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        max_leaves: int = 31,
        grower: str = "leafwise",
        sampler: str = "all_rows",
        early_stopping_rounds: int = 0,
        n_threads: int = 0,
        random_seed: int = 42,
        max_bin: int | None = None,
        subsample: float | None = None,
        device: str | None = None,
        params: dict | ParamsOps | None = None,
    ):
        self._store(locals())

    def predict_proba(self, X: npt.ArrayLike | Dataset) -> np.ndarray:
        """Predict class probabilities with the fitted model.

        A ``logloss`` model's P(class 1) becomes ``[1 - p, p]``; a
        ``softmax`` model, two classes included, gives its per-class
        probabilities (a row-wise softmax of the class logits).

        Parameters
        ----------
        X
            As for ``predict``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_rows, n_classes)``, columns in ``classes_`` order.
        """
        X = self._predict_input(X)
        if self._is_logloss():
            p = np.asarray(self._fitted().predict(X), dtype=np.float64)
            return np.column_stack([1.0 - p, p])
        return np.asarray(self._fitted().predict_proba(X), dtype=np.float64)

    def score(
        self,
        X: npt.ArrayLike | Dataset,
        y: npt.ArrayLike,
        sample_weight: npt.ArrayLike | None = None,
    ) -> float:
        """Compute accuracy on held-out data.

        Matches sklearn's ``ClassifierMixin.score``, computed by hand with no
        sklearn import.

        Parameters
        ----------
        X
            Feature matrix.
        y
            True class labels.
        sample_weight
            Per-row weights for the accuracy computation. None weights every
            row equally.

        Returns
        -------
        float
            The accuracy score.

        Raises
        ------
        ValueError
            If ``sample_weight`` sums to zero, or ``y`` or ``sample_weight``
            disagrees with the predictions in row count.
        """
        y_true = _as_labels(y, "y")
        y_pred = np.asarray(self.predict(X))
        w = (
            None
            if sample_weight is None
            else _as_f32(sample_weight, 1, "sample_weight").astype(np.float64)
        )
        _check_scored_shapes(y_true, y_pred, w)
        correct = (y_true == y_pred).astype(np.float64)
        if w is None:
            return float(correct.mean())
        if w.sum() == 0.0:
            raise ValueError("sample_weight sums to zero; accuracy is undefined")
        return float(np.average(correct, weights=w))

    _DERIVED_KEYS = (
        *_BonsaiEstimator._DERIVED_KEYS,
        "dispatch.objective_name",
        "objective.n_classes",
    )

    @classmethod
    def from_file(cls, path: str) -> BonsaiClassifier:
        """Load a saved ``.msgpack`` classifier.

        The native format stores only the booster, so ``classes_`` comes back
        as the encoded ids ``0..K-1``, and ``predict`` returns those ids
        rather than the label values passed to ``fit``. Pickle the estimator
        to preserve original labels.

        Raises
        ------
        ValueError
            If the saved model's objective is not ``logloss`` or ``softmax``.
        """
        out = super().from_file(path)
        objective = out._fitted().objective_name
        if objective == "logloss":
            out.n_classes_ = 2
        elif objective == "softmax":
            out.n_classes_ = int(out._fitted().n_classes)
        else:
            raise ValueError(
                f"{path!r} was trained with objective {objective!r}; "
                "BonsaiClassifier.from_file needs a logloss or softmax model"
            )
        out.classes_ = np.arange(out.n_classes_)
        return out

    def _objective_pairs(self, targets: _FittedTargets) -> dict[str, object]:
        n_classes = _class_targets(targets).n_classes_
        if n_classes == 2:
            return {"dispatch.objective_name": "logloss"}
        return {
            "dispatch.objective_name": "softmax",
            "objective.n_classes": n_classes,
        }

    def _encode_targets(self, y: npt.ArrayLike) -> tuple[np.ndarray, _FittedTargets]:
        y_arr = _as_labels(y, "y")
        if _has_nan_label(y_arr):
            raise ValueError("Input y contains NaN.")
        _reject_continuous_labels(y_arr)
        classes = np.unique(y_arr)
        if len(classes) < 2:
            raise ValueError(
                f"BonsaiClassifier needs at least 2 classes, got {len(classes)} ({classes!r})"
            )
        encoded = np.searchsorted(classes, y_arr).astype(np.float32)
        return encoded, _ClassTargets(classes_=classes, n_classes_=len(classes))

    def _encode_eval_targets(self, y: npt.ArrayLike, targets: _FittedTargets) -> np.ndarray:
        """A label the training fold never saw cannot be encoded; letting
        searchsorted guess silently corrupts the eval metric and early
        stopping, so reject it."""
        classes = _class_targets(targets).classes_
        y_arr = _as_labels(y, "y")
        ids = np.clip(np.searchsorted(classes, y_arr), 0, len(classes) - 1)
        bad = classes[ids] != y_arr
        if bad.any():
            raise ValueError(
                f"eval_set labels {np.unique(y_arr[bad])!r} are not in the "
                f"training classes {classes!r}"
            )
        return _as_f32(ids, 1, "y")

    def _decode_predictions(self, raw: np.ndarray) -> np.ndarray:
        if self._is_logloss():
            return self.classes_[(raw >= 0.5).astype(np.int64)]
        return self.classes_[raw.astype(np.int64)]

    def _is_logloss(self) -> bool:
        """Whether the model predicts P(class 1) rather than class ids.

        Read off the model, not the class count: a two-class ``softmax``
        model, loaded or forced through ``params``, predicts ids.
        """
        return self._fitted().objective_name == "logloss"


# Private Functions ================================================================================


def __getattr__(name: str) -> type[Exception]:
    """Resolve ``NotFittedError`` by name, so the error pickles across processes."""
    if name == "NotFittedError":
        return _not_fitted_error()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@functools.cache
def _not_fitted_error() -> type[Exception]:
    """The error an unfitted estimator raises, built on first use.

    It is sklearn's ``NotFittedError`` when sklearn is installed, which is
    what sklearn's tooling and estimator checks catch, and stays a
    ``RuntimeError`` for callers written against the earlier error. sklearn
    is imported here rather than at module scope, so ``import bonsai`` never
    needs it; without sklearn the error keeps ``NotFittedError``'s own bases.

    Returns
    -------
    type
        One class per process, named ``bonsai.estimators.NotFittedError``.
    """
    try:
        from sklearn.exceptions import NotFittedError
    except ImportError:
        bases: tuple[type, ...] = (ValueError, AttributeError, RuntimeError)
    else:
        bases = (NotFittedError, RuntimeError)
    return type("NotFittedError", bases, {"__module__": __name__})


def _constructor_names(cls) -> list[str]:
    """The ``__init__`` parameter names, which are the sklearn parameters."""
    return [name for name in inspect.signature(cls.__init__).parameters if name != "self"]


def _init_defaults(cls) -> dict[str, object]:
    """Constructor-signature defaults, keyed by parameter name."""
    sig = inspect.signature(cls.__init__)
    return {name: prm.default for name, prm in sig.parameters.items() if name != "self"}


def _shortest_spellings(config: dict[str, object]) -> dict[str, object]:
    """A config dict with each float in its shortest float32 spelling.

    The config stores its floats as float32, so a learning rate of 0.3
    reads back as 0.30000001192092896; the shortest spelling that rounds
    to the same float32 is the value the caller wrote.
    """
    return {k: float(str(np.float32(v))) if isinstance(v, float) else v for k, v in config.items()}


def _names_disagreement(fitted: list[str], given: list[str]) -> str:
    """The error for two name lists that differ: which names are new, which
    are gone, or that only the order changed."""
    unseen = [name for name in given if name not in set(fitted)]
    missing = [name for name in fitted if name not in set(given)]
    detail = (f" Unseen at fit time: {unseen}." if unseen else "") + (
        f" Seen at fit time, now missing: {missing}." if missing else ""
    )
    return "The feature names should match those that were passed during fit." + (
        detail or " The same names arrived in a different order."
    )


def _eval_set_features(eval_set: tuple[npt.ArrayLike, npt.ArrayLike] | Dataset) -> object:
    """The feature half of an eval set, before any coercion."""
    return eval_set if isinstance(eval_set, Dataset) else eval_set[0]


def _input_feature_names(X: object) -> list[str] | None:
    """Names a predict input carries.

    A ``Dataset`` names every column, so its synthesized ``f0``..``fN`` are
    read as unnamed the way a model's are; anything else is a frame or an
    array, where only string ``.columns`` count.

    Parameters
    ----------
    X
        Any predict-family input.

    Returns
    -------
    list of str or None
        The names, or None when the input carries none.
    """
    if isinstance(X, Dataset):
        return _model_feature_names(X)
    return _feature_names_of(X)


def _model_feature_names(model: Model | Dataset) -> list[str] | None:
    """The model's own names, unless they are the synthesized f0..fN.

    Also reads a ``Dataset``, which names its columns the same way.

    Parameters
    ----------
    model
        Anything carrying a ``feature_names`` sequence.

    Returns
    -------
    list of str or None
        The names, or None when they are the synthesized ones.
    """
    names = list(model.feature_names)
    # A caller who really names columns "f0", "f1", ... reads as unnamed here:
    # nothing in the artifact distinguishes named from synthesized.
    if names == [f"f{i}" for i in range(len(names))]:
        return None
    return names


def _feature_names_of(X: object) -> list[str] | None:
    """X's column names, when it has string ones.

    A DataFrame with the default integer columns has no names, which is
    sklearn's rule too.

    Parameters
    ----------
    X
        Any fit input; only a ``.columns`` attribute is read.

    Returns
    -------
    list of str or None
        The column names, or None when X has none or they are not all
        strings.
    """
    columns = getattr(X, "columns", None)
    names = [] if columns is None else list(columns)
    if not names or not all(isinstance(name, str) for name in names):
        return None
    return names


def _class_targets(targets: _FittedTargets) -> _ClassTargets:
    """The class attributes a classifier's ``_encode_targets`` returned."""
    if not isinstance(targets, _ClassTargets):
        raise TypeError(f"expected class targets, got {type(targets).__name__}")
    return targets


def _as_labels(y: npt.ArrayLike, name: str) -> np.ndarray:
    """Class labels as a vector, one per row.

    A single-column 2-D array is read as one label per row, the way the
    regressors read a single-column target and sklearn's classifiers read
    a column vector.

    Parameters
    ----------
    y
        Labels of any hashable, orderable dtype.
    name
        The argument's name, for the error message.

    Returns
    -------
    numpy.ndarray
        Rank 1, in the caller's dtype.

    Raises
    ------
    ValueError
        When ``y`` is neither a vector nor a single column.
    """
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional or a single column, got shape {arr.shape}")
    return arr


def _check_scored_shapes(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray | None):
    """Refuse a ``score`` whose inputs would broadcast instead of pairing up.

    numpy would stretch a single row against many, or score only the
    targets ``y`` happens to share with the predictions, and report a
    number for data that does not line up.

    Parameters
    ----------
    y_true
        The caller's targets or labels, a single column already flattened.
    y_pred
        What ``predict`` returned for ``X``.
    w
        The sample weights, or None.

    Raises
    ------
    ValueError
        If the row counts differ, or the target counts do.
    """
    n_rows = len(y_pred)
    if len(y_true) != n_rows:
        raise ValueError(f"len(y) must equal the row count of X: got {len(y_true)} and {n_rows}")
    if y_true.shape[1:] != y_pred.shape[1:]:
        raise ValueError(
            f"y has {_target_count(y_true)} targets but the model predicts {_target_count(y_pred)}"
        )
    if w is not None and len(w) != n_rows:
        raise ValueError(
            f"len(sample_weight) must equal the row count of X: got {len(w)} and {n_rows}"
        )


def _target_count(a: np.ndarray) -> int:
    """Columns of a target array; a vector is one target."""
    return 1 if a.ndim == 1 else int(a.shape[1])


def _device_pairs(grower: str, device: str) -> dict[str, object]:
    """The config keys ``device`` writes: the grower, and which GPU.

    Every growth strategy has a CUDA grower, so the grower mapping is a
    prefix: the device moves compute and never substitutes the structural
    choice. ``"cuda:N"`` also places the fit on device ``N``.

    Parameters
    ----------
    grower
        The grower name before the device is applied.
    device
        ``"cpu"``, ``"cuda"`` (or ``"gpu"``), or ``"cuda:N"``.

    Returns
    -------
    dict
        ``dispatch.grower_name``, and ``parallel.device_id`` for ``"cuda:N"``.

    Raises
    ------
    ValueError
        If ``device`` is none of those, ``"cuda:abc"`` included.
    """
    dev = device.lower()
    if dev == "cpu":
        return {"dispatch.grower_name": grower.removeprefix("cuda_")}
    cuda_grower = grower if grower.startswith("cuda_") else f"cuda_{grower}"
    if dev in ("cuda", "gpu"):
        return {"dispatch.grower_name": cuda_grower}
    kind, _, index = dev.partition(":")
    if kind != "cuda" or not index.isdigit():
        raise ValueError(f"device must be 'cpu', 'cuda' or 'cuda:N', got {device!r}")
    return {"dispatch.grower_name": cuda_grower, "parallel.device_id": int(index)}


def _has_nan_label(y: np.ndarray) -> bool:
    """NaN among labels of any dtype; an object array tests each entry
    against itself, since NaN is the one value unequal to itself."""
    if y.dtype.kind == "f":
        return bool(np.isnan(y).any())
    if y.dtype.kind != "O":
        return False
    return bool((y != y).any())


def _reject_continuous_labels(y: np.ndarray):
    """Refuse sklearn's ``continuous`` target type: float labels that are
    not all finite integers, a regression target handed to a classifier."""
    if y.dtype.kind != "f":
        return
    discrete = np.isfinite(y) & (y == np.floor(y))
    if discrete.all():
        return
    raise ValueError(
        f"Unknown label type: continuous. BonsaiClassifier needs discrete class "
        f"labels, and y holds the float {float(y[~discrete].flat[0])!r}, which is not an "
        f"integer; use BonsaiRegressor for a continuous target"
    )


def _reject_eval_set_list(eval_set):
    """One validation set means one tuple; a list is an error, not a hint."""
    if not isinstance(eval_set, list):
        return
    raise ValueError(
        "eval_set takes one (X, y) tuple, not a list of them: bonsai tracks "
        "a single validation set. Pass eval_set=(X_valid, y_valid)."
    )
