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

import inspect
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt

from bonsai._bonsai import Dataset, Model, load, train
from bonsai._coerce import _as_f32, _to_config_str
from bonsai._params_ops import ParamsOps

__all__ = ["BonsaiClassifier", "BonsaiRegressor"]


# Shared Base ======================================================================================

class _BonsaiEstimator:
    """Shared sklearn-contract machinery for ``BonsaiRegressor`` and
    ``BonsaiClassifier``.

    Duck-types the scikit-learn estimator contract (``get_params`` /
    ``set_params`` / ``score`` / ``_estimator_type``) without subclassing
    ``sklearn.base.BaseEstimator`` — sklearn is never a runtime dependency of
    this package, so ``clone``, ``Pipeline``, ``GridSearchCV`` and
    ``cross_val_score`` all work, but ``import bonsai`` never needs sklearn
    installed.

    Not part of the public API — subclasses (``BonsaiRegressor``,
    ``BonsaiClassifier``) are what users construct. Holds only what's
    identical between them: parameter bookkeeping, config-pair building
    (minus the objective, which each subclass supplies), pickling, and the
    fitted/tag hooks sklearn's tooling looks for. ``_estimator_type``,
    ``score``, and ``fit``/``predict`` stay per-subclass since binary/
    multiclass/regression targets and outputs genuinely differ.

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
        Seed for the booster's random state.
    max_bin
        Histogram bin cap; None leaves the native default.
    subsample
        Row-sampling fraction; switches the sampler to ``"bernoulli"`` when
        ``sampler`` is still at its default. None leaves sampling off.
    device
        ``"cpu"`` or ``"cuda"`` (or ``"cuda:N"``); picks the grower that runs
        the chosen ``grower``'s strategy on that device. None leaves the
        grower name as given.
    params
        Overrides beyond the first-class kwargs: a ``bonsai.Params`` or a
        dotted config-key dict, e.g. ``{"tree.lambda_l1": 0.5}``; the
        power-user escape hatch, applied after the first-class kwargs and
        always wins.
    """

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
        """Store every argument raw; ``get_params``/``clone`` read them back."""
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.grower = grower
        self.sampler = sampler
        self.early_stopping_rounds = early_stopping_rounds
        self.n_threads = n_threads
        self.random_seed = random_seed
        self.max_bin = max_bin
        self.subsample = subsample
        self.device = device
        self.params = params
        self._model: Model | None = None

    def __repr__(self) -> str:
        """sklearn-style: class name plus the non-default parameters."""
        sig = inspect.signature(type(self).__init__)
        defaults = {name: prm.default for name, prm in sig.parameters.items()
                    if name != "self"}
        shown = {k: v for k, v in self.get_params().items()
                 if v != defaults.get(k)}
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
        names = [
            p.name
            for p in inspect.signature(self.__init__).parameters.values()
            if p.name != "self"
        ]
        return {name: getattr(self, name) for name in names}

    def set_params(self, **params) -> _BonsaiEstimator:
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
        ``Pipeline``, ``cross_val_score``, or ``GridSearchCV`` — even
        duck-typed ones that never subclass ``BaseEstimator``. Mirrors what
        ``RegressorMixin``/``ClassifierMixin``/``BaseEstimator`` produce, so
        sklearn stays import-only-in-tests; empirically verified against the
        installed sklearn (see test suite). ``_estimator_type`` is what
        picks the per-type tag.

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
        per_type = ({"regressor_tags": RegressorTags()} if kind == "regressor"
                    else {"classifier_tags": ClassifierTags()})
        return Tags(
            estimator_type=kind,
            target_tags=TargetTags(required=True),
            input_tags=InputTags(),
            **per_type,
        )

    def predict(self, X: npt.ArrayLike | Dataset,
                num_iteration: int = 0) -> np.ndarray:
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
            Predictions, one per row of ``X``.
        """
        self._check_fitted("fit() or load first")
        return np.asarray(self._model.predict(_as_x(X), num_iteration))

    def staged_predict(self, X: npt.ArrayLike | Dataset) -> np.ndarray:
        """Predict after each boosting iteration.

        Parameters
        ----------
        X
            Feature matrix, or a ``bonsai.Dataset``: one carrying the
            model's cuts routes in bin space (device-resident builds
            included), others read the matrix they retained.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_iters, n_rows)``: predictions after each boosting
            iteration.
        """
        self._check_fitted()
        return np.asarray(self._model.staged_predict(_as_x(X)))

    def predict_leaf(self, X: npt.ArrayLike | Dataset) -> np.ndarray:
        """Predict per-tree leaf indices, for feature engineering / embedding.

        Parameters
        ----------
        X
            Feature matrix, or a ``bonsai.Dataset``: one carrying the
            model's cuts routes in bin space (device-resident builds
            included), others read the matrix they retained.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_rows, n_trees)``. One column per tree in training
            order, so a softmax model (one tree per class per round) has
            ``n_iters * n_classes`` columns and column ``t`` is round
            ``t // n_classes``, class ``t % n_classes``.
        """
        self._check_fitted()
        return np.asarray(self._model.predict_leaf(_as_x(X)))

    def dump(self) -> str:
        """Dump the model as text.

        Returns
        -------
        str
            Every tree as indented text.
        """
        self._check_fitted()
        return self._model.dump()

    def pred_contribs(self, X: npt.ArrayLike | Dataset) -> np.ndarray:
        """Compute TreeSHAP contributions.

        Parameters
        ----------
        X
            Feature matrix, or a ``bonsai.Dataset``: one carrying the
            model's cuts routes in bin space (device-resident builds
            included), others read the matrix they retained.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_rows, n_features + 1)``; the last column is the bias.
            Rows sum to the raw (pre-link) prediction exactly.
        """
        self._check_fitted()
        return np.asarray(self._model.pred_contribs(_as_x(X)))

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
        return np.asarray(self._model.feature_importance(type))

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
        self._check_fitted("fit() before save()")
        self._model.save(path)

    @classmethod
    def from_file(cls, path: str) -> _BonsaiEstimator:
        """Load a fitted model from a saved ``.msgpack`` file.

        Parameters
        ----------
        path
            Path to a model saved with ``save``.

        Returns
        -------
        _BonsaiEstimator
            A fresh estimator instance wrapping the loaded model.
        """
        out = cls()
        out._model = load(path)
        return out

    @property
    def n_iters_(self) -> int:
        self._check_fitted()
        return self._model.n_iters

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
        name = self._model.objective_name
        hist = [float(v) for v in self._model.eval_history]
        start = next(
            (i for i, v in enumerate(hist) if not np.isnan(v)), len(hist)
        )
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
        self._check_fitted()
        hist = self._model.eval_history
        if not self.early_stopping_rounds or not len(hist):
            raise AttributeError(
                "best_iteration needs fit(eval_set=...) with "
                "early_stopping_rounds set"
            )
        # nanargmin: warm-start rounds are unmeasured NaN placeholders.
        return int(np.nanargmin(hist))

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
        self._check_fitted()
        hist = self._model.eval_history
        if not self.early_stopping_rounds or not len(hist):
            raise AttributeError(
                "best_score needs fit(eval_set=...) with "
                "early_stopping_rounds set"
            )
        return float(np.nanmin(hist))

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
        """Pickle support: restore the native model from its msgpack bytes."""
        model_bytes = state.pop("_model_bytes", None)
        self.__dict__.update(state)
        if model_bytes is not None:
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "m.msgpack"
                path.write_bytes(model_bytes)
                self._model = load(str(path))
        else:
            self._model = None

    def _objective_pairs(self) -> dict[str, str]:
        """Config keys the objective needs. ``BonsaiRegressor`` exposes a
        fixed ``objective`` kwarg; ``BonsaiClassifier`` derives it from
        ``classes_`` at fit time. Overridden per-subclass."""
        raise NotImplementedError

    def _build_overrides(self) -> dict[str, object]:
        """Translate the first-class kwargs and ``params`` into the dotted
        config keys the native ``train()`` expects. Kept out of ``__init__``
        so constructor args stay raw attributes (required for
        ``get_params``/``clone``).

        Precedence (lowest to highest): first-class kwargs, then the
        optional ones that carry a ``None`` "leave it alone" default, then
        ``params`` (the power-user escape hatch always has the final
        word)."""
        merged = {
            "booster.n_iters": self.n_iters,
            "booster.learning_rate": self.learning_rate,
            "booster.early_stopping_rounds": self.early_stopping_rounds,
            "booster.random_seed": self.random_seed,
            "tree.max_depth": self.max_depth,
            "tree.max_leaves": self.max_leaves,
            "dispatch.grower_name": self.grower,
            "dispatch.sampler_name": self.sampler,
            "parallel.n_threads": self.n_threads,
            **self._objective_pairs(),
        }
        if self.max_bin is not None:
            merged["bin_mapper.max_bin"] = self.max_bin
        if self.subsample is not None:
            # A subsample fraction implies row sampling; bonsai makes the
            # sampler explicit, so setting the fraction turns it on when the
            # sampler is still at its default (an explicit sampler= choice
            # keeps its machinery).
            merged["sampler.subsample"] = self.subsample
            if self.sampler == "all_rows":
                merged["dispatch.sampler_name"] = "bernoulli"
        if self.device is not None:
            merged["dispatch.grower_name"] = _grower_for_device(
                str(merged["dispatch.grower_name"]), self.device
            )
        overrides = (self.params.to_dict()
                     if isinstance(self.params, ParamsOps) else self.params)
        merged.update(overrides or {})
        return {k: _to_config_str(v) for k, v in merged.items()}

    @staticmethod
    def _reject_dart_eval_set(overrides: dict[str, object]):
        """Eval history shares early stopping's incremental valid-loss
        accumulation, which DART's per-round tree rescaling invalidates;
        the native layer would silently record nothing (and
        ``evals_result()`` would come back empty), so fail loudly at fit
        time, mirroring the native early-stopping + DART rejection."""
        for key, value in overrides.items():
            if key == "booster.dart_drop_rate" and float(value) > 0.0:
                raise ValueError(
                    "eval_set is unsupported with dart_drop_rate > 0: eval "
                    "history relies on incremental valid-loss bookkeeping "
                    "that DART's per-round tree rescaling invalidates"
                )

    def _check_fitted(self, message: str = "fit() first"):
        """One guard for every fitted-state precondition; message names the
        remedy per call site (exception type is part of the API)."""
        if self._model is None:
            raise RuntimeError(message)


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
        Seed for the booster's random state.
    max_bin
        Histogram bin cap; None leaves the native default.
    subsample
        Row-sampling fraction; switches the sampler to ``"bernoulli"`` when
        ``sampler`` is still at its default. None leaves sampling off.
    device
        ``"cpu"`` or ``"cuda"`` (or ``"cuda:N"``); ``"cuda"`` picks the CUDA
        grower matching the chosen ``grower``, moving compute without
        changing the growth strategy. None leaves the grower name as given.
    quantile_alpha
        The quantile to fit when ``objective="quantile"``.
    params
        Overrides beyond the first-class kwargs: a ``bonsai.Params`` or a
        dotted config-key dict; the power-user escape hatch, applied after
        the first-class kwargs and always wins.
    """

    _estimator_type = "regressor"

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
        """Same storage contract as the base; adds the regression objective."""
        super().__init__(**_shared_args(locals()))
        self.objective = objective
        self.quantile_alpha = quantile_alpha

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike,
            sample_weight: npt.ArrayLike | None = None,
            eval_set: tuple[npt.ArrayLike, npt.ArrayLike] | Dataset | None = None,
            init_model: str | None = None) -> BonsaiRegressor:
        """Fit the regressor to training data.

        Parameters
        ----------
        X
            Feature matrix.
        y
            Regression targets.
        sample_weight
            Per-row weights that scale each row's gradient and hessian
            (sklearn's convention). None weights every row equally.
        eval_set
            One bare ``(X, y)`` tuple, Dataset, or None. A ``bonsai.Dataset``
            must be built with ``reference=`` the training dataset, which
            bins the validation rows once and reuses them across fits.
            bonsai tracks a single validation set, so a list of them is
            rejected rather than silently reduced to its last entry.
        init_model
            Path to a saved ``.msgpack`` model to continue training from
            (warm start); binning reuses the loaded model's cut points. None
            trains from scratch.

        Returns
        -------
        BonsaiRegressor
            ``self``, fitted.

        Raises
        ------
        ValueError
            If ``eval_set`` is a list, or (with ``dart_drop_rate > 0``) if
            ``eval_set`` is given at all.
        """
        _reject_eval_set_list(eval_set)
        overrides = self._build_overrides()
        ev = None
        if eval_set is not None:
            self._reject_dart_eval_set(overrides)
            ev = (eval_set if isinstance(eval_set, Dataset)
                  else (_as_f32(eval_set[0], 2, "X"), _as_f32(eval_set[1], 1, "y")))
        sw = None if sample_weight is None else _as_f32(sample_weight, 1, "sample_weight")
        Xa = _as_f32(X, 2, "X")
        self._model = train(
            overrides, Xa, _as_f32(y, 1, "y"), ev, init_model, sample_weight=sw,
        )
        self.n_features_in_ = Xa.shape[1]
        return self

    def score(self, X: npt.ArrayLike | Dataset, y: npt.ArrayLike,
              sample_weight: npt.ArrayLike | None = None) -> float:
        """Compute R² (coefficient of determination) on held-out data.

        Matches sklearn's ``RegressorMixin.score`` — computed by hand, no
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
        """
        y_true = _as_f32(y, 1, "y").astype(np.float64)
        y_pred = np.asarray(self.predict(X), dtype=np.float64)
        w = (None if sample_weight is None
             else _as_f32(sample_weight, 1, "sample_weight").astype(np.float64))

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

    def _objective_pairs(self) -> dict[str, str]:
        pairs = {"dispatch.objective_name": self.objective}
        if self.quantile_alpha is not None:
            pairs["objective.quantile_alpha"] = self.quantile_alpha
        return pairs


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
        Seed for the booster's random state.
    max_bin
        Histogram bin cap; None leaves the native default.
    subsample
        Row-sampling fraction; switches the sampler to ``"bernoulli"`` when
        ``sampler`` is still at its default. None leaves sampling off.
    device
        ``"cpu"`` or ``"cuda"`` (or ``"cuda:N"``); ``"cuda"`` picks the CUDA
        grower matching the chosen ``grower``, moving compute without
        changing the growth strategy. None leaves the grower name as given.
    params
        Overrides beyond the first-class kwargs: a ``bonsai.Params`` or a
        dotted config-key dict; the power-user escape hatch, applied after
        the first-class kwargs and always wins.
    """

    _estimator_type = "classifier"

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
        """Same storage contract as the base; the objective is derived from
        the number of classes at fit time, so there is none to store."""
        super().__init__(**_shared_args(locals()))

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike,
            sample_weight: npt.ArrayLike | None = None,
            eval_set: tuple[npt.ArrayLike, npt.ArrayLike] | Dataset | None = None,
            init_model: str | None = None) -> BonsaiClassifier:
        """Fit the classifier to training data.

        ``fit`` picks ``logloss`` for two classes or ``softmax`` (with
        ``objective.n_classes`` set) for more, based on ``np.unique(y)``.

        Parameters
        ----------
        X
            Feature matrix.
        y
            Class labels; any hashable/orderable values (ints, strings,
            ...). Encoded to ``0..K-1`` internally and decoded back to the
            original ``classes_`` values by ``predict``.
        sample_weight
            Per-row weights that scale each row's gradient and hessian
            (sklearn's convention). None weights every row equally.
        eval_set
            One bare ``(X, y)`` tuple, Dataset, or None. A ``bonsai.Dataset``
            must be built with ``reference=`` the training dataset and carry
            labels already encoded to ``0..K-1``. bonsai tracks a single
            validation set, so a list of them is rejected.
        init_model
            Path to a saved ``.msgpack`` model to continue training from
            (warm start); binning reuses the loaded model's cut points. None
            trains from scratch.

        Returns
        -------
        BonsaiClassifier
            ``self``, fitted.

        Raises
        ------
        ValueError
            If ``y`` contains NaN, has fewer than 2 classes, or ``eval_set``
            is a list; also if ``eval_set`` is given with
            ``dart_drop_rate > 0``, or if an eval-set label was not seen in
            the training classes.
        """
        _reject_eval_set_list(eval_set)
        y_arr = np.asarray(y)
        if y_arr.dtype.kind == "f" and np.isnan(y_arr).any():
            raise ValueError("Input y contains NaN.")
        self.classes_ = np.unique(y_arr)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError(
                f"BonsaiClassifier needs at least 2 classes, got {self.n_classes_} "
                f"({self.classes_!r})"
            )
        y_enc = np.searchsorted(self.classes_, y_arr).astype(np.float32)

        overrides = self._build_overrides()
        ev = None
        if eval_set is not None:
            self._reject_dart_eval_set(overrides)
            ev = self._encode_eval_set(eval_set)
        sw = None if sample_weight is None else _as_f32(sample_weight, 1, "sample_weight")
        Xa = _as_f32(X, 2, "X")
        self._model = train(
            overrides, Xa, y_enc, ev, init_model, sample_weight=sw,
        )
        self.n_features_in_ = Xa.shape[1]
        return self

    def predict(self, X: npt.ArrayLike | Dataset,
                num_iteration: int = 0) -> np.ndarray:
        """Predict class labels with the fitted model.

        Parameters
        ----------
        X
            Feature matrix, or a ``bonsai.Dataset``: one carrying the
            model's cuts routes in bin space (device-resident builds
            included), others read the matrix they retained.
        num_iteration
            Truncates the ensemble to its first ``n`` trees; 0 uses every
            tree.

        Returns
        -------
        numpy.ndarray
            Original class labels (from ``classes_``), not the encoded
            ``0..K-1`` ids the native booster works in.
        """
        self._check_fitted("fit() or load first")
        raw = np.asarray(self._model.predict(_as_x(X), num_iteration))
        if self.n_classes_ == 2:
            idx = (raw >= 0.5).astype(np.int64)
        else:
            idx = raw.astype(np.int64)
        return self.classes_[idx]

    def predict_proba(self, X: npt.ArrayLike | Dataset) -> np.ndarray:
        """Predict class probabilities with the fitted model.

        Binary uses the ``logloss`` objective's P(class 1) directly
        (``[1 - p, p]``); multiclass uses the ``softmax`` booster's per-class
        probabilities (a row-wise softmax of the class logits).

        Parameters
        ----------
        X
            Feature matrix, or a ``bonsai.Dataset``: one carrying the
            model's cuts routes in bin space (device-resident builds
            included), others read the matrix they retained.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_rows, n_classes)``, columns in ``classes_`` order.
        """
        self._check_fitted("fit() or load first")
        if self.n_classes_ == 2:
            p = np.asarray(self._model.predict(_as_x(X)), dtype=np.float64)
            return np.column_stack([1.0 - p, p])
        return np.asarray(self._model.predict_proba(_as_x(X)), dtype=np.float64)

    def score(self, X: npt.ArrayLike | Dataset, y: npt.ArrayLike,
              sample_weight: npt.ArrayLike | None = None) -> float:
        """Compute accuracy on held-out data.

        Matches sklearn's ``ClassifierMixin.score`` — computed by hand, no
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
            If ``sample_weight`` sums to zero.
        """
        y_true = np.asarray(y)
        y_pred = np.asarray(self.predict(X))
        correct = (y_true == y_pred).astype(np.float64)

        if sample_weight is None:
            return float(correct.mean())
        w = _as_f32(sample_weight, 1, "sample_weight").astype(np.float64)
        if w.sum() == 0.0:
            raise ValueError("sample_weight sums to zero; accuracy is undefined")
        return float(np.average(correct, weights=w))

    @classmethod
    def from_file(cls, path: str) -> BonsaiClassifier:
        """Load a saved ``.msgpack`` classifier.

        The native format stores only the booster, so ``classes_`` comes back
        as the encoded ids ``0..K-1``, and ``predict`` returns those ids
        rather than the label values passed to ``fit``. Pickle the estimator
        to preserve original labels.

        Parameters
        ----------
        path
            Path to a model saved with ``save``.

        Returns
        -------
        BonsaiClassifier
            A fresh classifier instance wrapping the loaded model.

        Raises
        ------
        ValueError
            If the saved model's objective is not ``logloss`` or ``softmax``.
        """
        out = super().from_file(path)
        objective = out._model.objective_name
        if objective == "logloss":
            out.n_classes_ = 2
        elif objective == "softmax":
            out.n_classes_ = int(out._model.n_classes)
        else:
            raise ValueError(
                f"{path!r} was trained with objective {objective!r}; "
                "BonsaiClassifier.from_file needs a logloss or softmax model"
            )
        out.classes_ = np.arange(out.n_classes_)
        return out

    def _objective_pairs(self) -> dict[str, str]:
        if self.n_classes_ == 2:
            return {"dispatch.objective_name": "logloss"}
        return {
            "dispatch.objective_name": "softmax",
            "objective.n_classes": self.n_classes_,
        }

    def _encode_eval_set(self, eval_set) -> tuple[np.ndarray, np.ndarray] | Dataset:
        """Encode eval-set labels against ``classes_``.

        A label the training fold never saw cannot be encoded; letting
        searchsorted guess silently corrupts the eval metric and early
        stopping, so reject it. A pre-binned ``Dataset`` was built before
        ``fit`` saw any class, so its labels are already the encoded ids and
        the native layer range-checks them.
        """
        if isinstance(eval_set, Dataset):
            return eval_set
        ev_y_arr = np.asarray(eval_set[1])
        ev_y = np.clip(
            np.searchsorted(self.classes_, ev_y_arr), 0, self.n_classes_ - 1
        )
        bad = self.classes_[ev_y] != ev_y_arr
        if bad.any():
            raise ValueError(
                f"eval_set labels {np.unique(ev_y_arr[bad])!r} are not in the "
                f"training classes {self.classes_!r}"
            )
        return (_as_f32(eval_set[0], 2, "X"), _as_f32(ev_y, 1, "y"))


# Private Functions ================================================================================

def _shared_args(scope: dict) -> dict:
    """The arguments a subclass ``__init__`` hands straight to the base.

    Reads the base signature and picks those names out of the caller's
    locals, so a shared knob is spelled once there instead of three times
    per subclass. The subclass signatures stay explicit and complete:
    sklearn's ``get_params`` and ``clone`` read them with
    ``inspect.signature``, so a ``**kwargs`` there would erase the
    parameters.
    """
    names = [name for name in
             inspect.signature(_BonsaiEstimator.__init__).parameters
             if name != "self"]
    return {name: scope[name] for name in names}


def _as_x(X: npt.ArrayLike | Dataset) -> np.ndarray | Dataset:
    """Predict-family input: a host-built ``Dataset`` passes straight through
    (the native reader takes the matrix it retained), anything else coerces
    to a float32 matrix."""
    return X if isinstance(X, Dataset) else _as_f32(X, 2, "X")


def _grower_for_device(grower: str, device: str) -> str:
    """The grower that runs ``grower``'s strategy on ``device``.

    Every growth strategy has a CUDA grower, so the mapping is a prefix: the
    device moves compute and never substitutes the structural choice.
    """
    dev = device.lower()
    if dev in ("cuda", "gpu") or dev.startswith("cuda:"):
        return grower if grower.startswith("cuda_") else f"cuda_{grower}"
    if dev == "cpu":
        return grower.removeprefix("cuda_")
    raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")


def _reject_eval_set_list(eval_set):
    """One validation set means one tuple; a list is an error, not a hint."""
    if not isinstance(eval_set, list):
        return
    raise ValueError(
        "eval_set takes one (X, y) tuple, not a list of them: bonsai tracks "
        "a single validation set. Pass eval_set=(X_valid, y_valid)."
    )
