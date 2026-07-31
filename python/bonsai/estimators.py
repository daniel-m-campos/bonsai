"""The scikit-learn estimator layer over the native booster.

``BonsaiRegressor`` and ``BonsaiClassifier`` are what users construct; the
shared ``_BonsaiEstimator`` base holds the sklearn contract (get_params /
set_params / pickling / fitted-state hooks) without ever importing sklearn
at module scope. The public import path is the package root
(``bonsai.BonsaiRegressor``); this module is the implementation home.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np

from ._bonsai import Model, load, train
from ._coerce import _as_1d_f32, _as_2d_f32, _to_config_str
from ._compat import (
    _EVAL_METRIC,
    _XGB_OBJECTIVES,
    _grower_for_device,
    _normalize_eval_set,
)

__all__ = ["BonsaiClassifier", "BonsaiRegressor"]


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
        n_estimators: int | None = None,
        num_leaves: int | None = None,
        random_state: int | None = None,
        n_jobs: int | None = None,
        reg_lambda: float | None = None,
        reg_alpha: float | None = None,
        max_bin: int | None = None,
        min_child_samples: int | None = None,
        colsample_bytree: float | None = None,
        min_child_weight: float | None = None,
        gamma: float | None = None,
        subsample: float | None = None,
        device: str | None = None,
        params: dict | None = None,
        config: str | None = None,
    ):
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.grower = grower
        self.sampler = sampler
        self.early_stopping_rounds = early_stopping_rounds
        self.n_threads = n_threads
        self.random_seed = random_seed
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.max_bin = max_bin
        self.min_child_samples = min_child_samples
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.device = device
        self.params = params
        self.config = config
        self._model: Model | None = None

    def _objective_pairs(self) -> dict[str, str]:
        """Config keys the objective needs. ``BonsaiRegressor`` exposes a
        fixed ``objective`` kwarg; ``BonsaiClassifier`` derives it from
        ``classes_`` at fit time. Overridden per-subclass."""
        raise NotImplementedError

    # xgboost/lightgbm-style constructor aliases -> bonsai dotted config key.
    # `min_child_weight` maps to `min_child_hess` (both are the minimum
    # hessian mass a child may hold, same 1.0 default; for MSE that is a row
    # count). `subsample` and `device` need more than a key rename and are
    # handled in `_build_pairs`.
    _ALIAS_TO_KEY: ClassVar[dict[str, str]] = {
        "n_estimators": "booster.n_iters",
        "num_leaves": "tree.max_leaves",
        "random_state": "booster.random_seed",
        "n_jobs": "parallel.n_threads",
        "reg_lambda": "tree.lambda_l2",
        "reg_alpha": "tree.lambda_l1",
        "max_bin": "bin_mapper.max_bin",
        "min_child_samples": "tree.min_data_in_leaf",
        "colsample_bytree": "tree.feature_fraction",
        "min_child_weight": "tree.min_child_hess",
        "gamma": "tree.min_gain_to_split",
    }

    def _build_pairs(self) -> list[tuple[str, str]]:
        """Translate the first-class kwargs + aliases + ``params`` into the
        dotted config keys the native ``train()`` expects. Kept out of
        ``__init__`` so constructor args stay raw attributes (required for
        ``get_params``/``clone``).

        Precedence (lowest to highest): canonical first-class kwargs, then
        xgboost/lightgbm-style aliases (only those set, i.e. not ``None``),
        then ``params`` (the power-user escape hatch always has the final
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
        for alias, key in self._ALIAS_TO_KEY.items():
            value = getattr(self, alias)
            if value is not None:
                merged[key] = value
        if self.subsample is not None:
            # xgboost's subsample implies row sampling; bonsai makes the
            # sampler explicit, so the alias turns it on when it is still at
            # the default (an explicit sampler= choice keeps its machinery).
            merged["sampler.subsample"] = self.subsample
            if self.sampler == "all_rows":
                merged["dispatch.sampler_name"] = "bernoulli"
        if self.device is not None:
            merged["dispatch.grower_name"] = _grower_for_device(
                str(merged["dispatch.grower_name"]), self.device
            )
        merged.update(self.params or {})
        return [(k, _to_config_str(v)) for k, v in merged.items()]

    @staticmethod
    def _reject_dart_eval_set(pairs: list[tuple[str, str]]) -> None:
        """Eval history shares early stopping's incremental valid-loss
        accumulation, which DART's per-round tree rescaling invalidates;
        the native layer would silently record nothing (and
        ``evals_result()`` would come back empty), so fail loudly at fit
        time, mirroring the native early-stopping + DART rejection."""
        for key, value in pairs:
            if key == "booster.dart_drop_rate" and float(value) > 0.0:
                raise ValueError(
                    "eval_set is unsupported with dart_drop_rate > 0: eval "
                    "history relies on incremental valid-loss bookkeeping "
                    "that DART's per-round tree rescaling invalidates"
                )

    def get_params(self, deep: bool = True) -> dict:
        """sklearn contract: one entry per ``__init__`` parameter, unchanged
        since construction (``deep`` is accepted for API compatibility;
        these estimators have no nested estimators to recurse into)."""
        names = [
            p.name
            for p in inspect.signature(self.__init__).parameters.values()
            if p.name != "self"
        ]
        return {name: getattr(self, name) for name in names}

    def set_params(self, **params):
        """sklearn contract: set constructor attributes in place, return
        self. Unknown names raise (sklearn's own estimators do the same)."""
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
        """sklearn's ``check_is_fitted`` (used by ``Pipeline`` etc.) looks
        for instance ``__dict__`` attributes ending in ``_``; ``n_iters_``
        etc. are properties backed by ``_model``, so they never show up
        there. This makes fitted-state detection exact instead of relying
        on that naming convention."""
        return self._model is not None

    def __sklearn_tags__(self):
        """Only needed because the installed sklearn (>=1.6) requires
        ``__sklearn_tags__`` on any estimator passed through ``clone``,
        ``Pipeline``, ``cross_val_score``, or ``GridSearchCV`` — even
        duck-typed ones that never subclass ``BaseEstimator``. Built by hand
        (mirroring what ``RegressorMixin``/``ClassifierMixin``/
        ``BaseEstimator`` produce) so sklearn stays import-only-in-tests;
        empirically verified against the installed sklearn (see test
        suite). Subclasses fill in the estimator-type-specific tag."""
        raise NotImplementedError

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        model = state.pop("_model", None)
        if model is not None:
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "m.msgpack"
                model.save(str(path))
                state["_model_bytes"] = path.read_bytes()
        return state

    def __setstate__(self, state: dict) -> None:
        model_bytes = state.pop("_model_bytes", None)
        self.__dict__.update(state)
        if model_bytes is not None:
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "m.msgpack"
                path.write_bytes(model_bytes)
                self._model = load(str(path))
        else:
            self._model = None

    def predict(self, X, num_iteration: int = 0,
                iteration_range: tuple[int, int] | None = None) -> np.ndarray:
        """``iteration_range`` is xgboost's spelling of the same thing:
        ``(0, n)`` means predict with the first ``n`` trees. Ranges must
        start at 0 (a boosted sum has no meaning without its head)."""
        if self._model is None:
            raise RuntimeError("fit() or load first")
        if iteration_range is not None:
            if iteration_range[0] != 0:
                raise ValueError(
                    f"iteration_range must start at 0, got {iteration_range!r}"
                )
            num_iteration = iteration_range[1]
        return np.asarray(self._model.predict(_as_2d_f32(X), num_iteration))

    def staged_predict(self, X) -> np.ndarray:
        """(n_iters, n_rows): predictions after each boosting iteration."""
        if self._model is None:
            raise RuntimeError("fit() first")
        return np.asarray(self._model.staged_predict(_as_2d_f32(X)))

    def predict_leaf(self, X) -> np.ndarray:
        """(n_rows, n_iters): per-tree leaf indices (feature engineering /
        embedding trick)."""
        if self._model is None:
            raise RuntimeError("fit() first")
        return np.asarray(self._model.predict_leaf(_as_2d_f32(X)))

    def dump(self) -> str:
        """Every tree as indented text."""
        if self._model is None:
            raise RuntimeError("fit() first")
        return self._model.dump()

    def pred_contribs(self, X) -> np.ndarray:
        """(n_rows, n_features + 1) TreeSHAP contributions; last column is
        the bias. Rows sum to the raw (pre-link) prediction exactly."""
        if self._model is None:
            raise RuntimeError("fit() first")
        return np.asarray(self._model.pred_contribs(_as_2d_f32(X)))

    def importance(self, type: str = "gain") -> np.ndarray:
        """Raw per-feature importance: total split gain or split count."""
        if self._model is None:
            raise RuntimeError("fit() first")
        return np.asarray(self._model.feature_importance(type))

    @property
    def feature_importances_(self) -> np.ndarray:
        """Gain importance normalized to sum to 1 (sklearn convention)."""
        raw = self.importance("gain")
        total = raw.sum()
        return raw / total if total > 0 else raw

    def apply(self, X) -> np.ndarray:
        """xgboost's name for per-tree leaf indices; same as
        ``predict_leaf``."""
        return self.predict_leaf(X)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("fit() before save()")
        self._model.save(path)

    def save_model(self, path: str) -> None:
        """xgboost's name for ``save``."""
        self.save(path)

    def load_model(self, path: str):
        """xgboost's in-place loader: replaces this estimator's fitted state
        with the saved model and returns ``self``. Same caveat as
        ``from_file``: the native format stores only the booster, so a
        classifier comes back with encoded ``0..K-1`` class ids."""
        loaded = type(self).from_file(path)
        self.__dict__.update(loaded.__dict__)
        return self

    @classmethod
    def from_file(cls, path: str):
        out = cls()
        out._model = load(path)
        return out

    @property
    def n_iters_(self) -> int:
        if self._model is None:
            raise RuntimeError("fit() first")
        return self._model.n_iters

    def _eval_presentation(self) -> tuple[str, list[float]]:
        """(xgboost metric name, transformed per-round eval history). The
        history is indexed by absolute model round: after an ``init_model``
        warm start the pre-existing rounds are NaN placeholders (the
        transforms map NaN to NaN)."""
        name, transform = _EVAL_METRIC.get(
            self._model.objective_name, (self._model.objective_name, float)
        )
        return name, [transform(v) for v in self._model.eval_history]

    def evals_result(self) -> dict:
        """Per-round eval-set metric history, xgboost's shape:
        ``{"validation_0": {metric_name: [...]}}``. Requires a fit with an
        ``eval_set``; empty after loading from a file (as in xgboost). The
        mse objective is presented as rmse (the exact root; monotone, so
        early stopping saw the same ordering). After an ``init_model`` warm
        start only the continuation's measured rounds are reported (the
        warm-start rounds were never evaluated; ``best_iteration`` still
        counts absolute rounds)."""
        if self._model is None:
            raise RuntimeError("fit() first")
        name, hist = self._eval_presentation()
        start = next(
            (i for i, v in enumerate(hist) if not np.isnan(v)), len(hist)
        )
        hist = hist[start:]
        if not hist:
            return {}
        return {"validation_0": {name: hist}}

    @property
    def best_iteration(self) -> int:
        """0-based absolute model round with the best eval-set loss, defined
        (as in xgboost) only when fit ran with early stopping and an
        eval_set. After an ``init_model`` warm start the index counts the
        warm-start rounds too, so it lines up with ``n_iters_`` and
        ``predict(num_iteration=best_iteration + 1)``."""
        if self._model is None:
            raise RuntimeError("fit() first")
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
        """Eval-set metric at ``best_iteration`` (same presentation as
        ``evals_result``)."""
        if self._model is None:
            raise RuntimeError("fit() first")
        hist = self._model.eval_history
        if not self.early_stopping_rounds or not len(hist):
            raise AttributeError(
                "best_score needs fit(eval_set=...) with "
                "early_stopping_rounds set"
            )
        _, presented = self._eval_presentation()
        return float(np.nanmin(presented))


class BonsaiRegressor(_BonsaiEstimator):
    """sklearn-style wrapper around the native booster.

    First-class arguments cover the common knobs; anything else can be set
    through ``params`` using dotted config keys, e.g.
    ``params={"tree.lambda_l1": 0.5, "sampler.top_rate": 0.2}`` — the same
    keys the CLI accepts via ``--set``.

    xgboost/lightgbm-style aliases (``n_estimators``, ``num_leaves``,
    ``random_state``, ``n_jobs``, ``reg_lambda``, ``reg_alpha``, ``max_bin``,
    ``min_child_samples``, ``colsample_bytree``, ``min_child_weight``,
    ``gamma``, ``subsample``, ``device``) are accepted so calls copied from
    those libraries work unchanged; they default to ``None`` and, when set,
    override the matching canonical kwarg (e.g. ``n_estimators`` wins over
    ``n_iters``). ``subsample`` switches the sampler to ``bernoulli`` when
    ``sampler`` is at its default; ``device="cuda"`` picks the CUDA grower
    matching the chosen grower. xgboost objective strings
    (``reg:squarederror``, ``reg:absoluteerror``, ``reg:quantileerror`` with
    ``quantile_alpha``, ``reg:pseudohubererror``, ``count:poisson``) are
    accepted as spellings of the bonsai objectives; note bonsai's ``huber``
    is the exact Huber loss, not xgboost's pseudo-Huber. Anything else goes
    through ``params``.

    ``config`` names a TOML file used as the base config (the CLI's ``-c``).
    Keyword arguments and ``params`` always win over the file — including the
    first-class kwargs at their defaults, which are always emitted. To defer
    a knob to the file, set it there and leave it out of ``params``, or use
    ``train()`` directly.
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
        n_estimators: int | None = None,
        num_leaves: int | None = None,
        random_state: int | None = None,
        n_jobs: int | None = None,
        reg_lambda: float | None = None,
        reg_alpha: float | None = None,
        max_bin: int | None = None,
        min_child_samples: int | None = None,
        colsample_bytree: float | None = None,
        min_child_weight: float | None = None,
        gamma: float | None = None,
        subsample: float | None = None,
        device: str | None = None,
        quantile_alpha: float | None = None,
        params: dict | None = None,
        config: str | None = None,
    ):
        super().__init__(
            n_iters=n_iters,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_leaves=max_leaves,
            grower=grower,
            sampler=sampler,
            early_stopping_rounds=early_stopping_rounds,
            n_threads=n_threads,
            random_seed=random_seed,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            random_state=random_state,
            n_jobs=n_jobs,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            max_bin=max_bin,
            min_child_samples=min_child_samples,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            subsample=subsample,
            device=device,
            params=params,
            config=config,
        )
        self.objective = objective
        self.quantile_alpha = quantile_alpha

    def _objective_pairs(self) -> dict[str, str]:
        pairs = {
            "dispatch.objective_name": _XGB_OBJECTIVES.get(
                self.objective, self.objective
            )
        }
        if self.quantile_alpha is not None:
            pairs["objective.quantile_alpha"] = self.quantile_alpha
        return pairs

    def score(self, X, y, sample_weight=None) -> float:
        """R² (coefficient of determination), matching sklearn's
        ``RegressorMixin.score`` — computed by hand, no sklearn import."""
        y_true = _as_1d_f32(y).astype(np.float64)
        y_pred = np.asarray(self.predict(X), dtype=np.float64)
        w = None if sample_weight is None else _as_1d_f32(sample_weight).astype(np.float64)

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

    def __sklearn_tags__(self):
        from sklearn.utils import InputTags, RegressorTags, Tags, TargetTags

        return Tags(
            estimator_type="regressor",
            target_tags=TargetTags(required=True),
            regressor_tags=RegressorTags(),
            input_tags=InputTags(),
        )

    def fit(self, X, y, sample_weight=None,
            eval_set: tuple | list | None = None,
            init_model: str | None = None, verbose=None) -> BonsaiRegressor:
        """`sample_weight` scales each row's gradient and hessian (sklearn's
        convention). `eval_set` takes the xgboost list-of-tuples form or a
        bare (X, y) tuple; with a list, the last entry drives the eval
        history and early stopping (xgboost's own convention). init_model
        continues training from a saved .msgpack (warm start); binning
        reuses the loaded model's cut points. `verbose` is accepted for
        xgboost call compatibility and ignored (bonsai prints one line on
        early stop, nothing per round)."""
        del verbose
        pairs = self._build_pairs()
        eval_set = _normalize_eval_set(eval_set)
        ev = None
        if eval_set is not None:
            self._reject_dart_eval_set(pairs)
            ev = (_as_2d_f32(eval_set[0]), _as_1d_f32(eval_set[1]))
        sw = None if sample_weight is None else _as_1d_f32(sample_weight)
        Xa = _as_2d_f32(X)
        self._model = train(
            pairs, Xa, _as_1d_f32(y), ev, init_model, self.config,
            sample_weight=sw,
        )
        self.n_features_in_ = Xa.shape[1]
        return self


class BonsaiClassifier(_BonsaiEstimator):
    """sklearn-style classifier wrapping the native booster's ``logloss``
    (binary) and ``softmax`` (multiclass) objectives.

    Same first-class knobs as ``BonsaiRegressor`` except there is no
    ``objective`` argument — ``fit`` picks ``logloss`` for two classes or
    ``softmax`` (with ``objective.n_classes`` set) for more, based on
    ``np.unique(y)``. Labels may be any hashable/orderable values (ints,
    strings, ...); they're encoded to ``0..K-1`` internally and decoded back
    to the original ``classes_`` values by ``predict``.

    Same xgboost/lightgbm-style aliases as ``BonsaiRegressor``
    (``n_estimators``, ``num_leaves``, ``random_state``, ``n_jobs``,
    ``reg_lambda``, ``reg_alpha``, ``max_bin``, ``min_child_samples``,
    ``colsample_bytree``, ``min_child_weight``, ``gamma``, ``subsample``,
    ``device``), and an ``objective`` accepted for xgboost call
    compatibility (``binary:logistic`` / ``multi:softprob`` /
    ``multi:softmax``) — the real objective is derived from the number of
    classes at fit time.

    ``predict_proba`` covers both cases: binary from the native ``logloss``
    P(class 1), multiclass from the ``softmax`` booster's per-class
    probabilities — see its docstring.
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
        objective: str | None = None,
        early_stopping_rounds: int = 0,
        n_threads: int = 0,
        random_seed: int = 42,
        n_estimators: int | None = None,
        num_leaves: int | None = None,
        random_state: int | None = None,
        n_jobs: int | None = None,
        reg_lambda: float | None = None,
        reg_alpha: float | None = None,
        max_bin: int | None = None,
        min_child_samples: int | None = None,
        colsample_bytree: float | None = None,
        min_child_weight: float | None = None,
        gamma: float | None = None,
        subsample: float | None = None,
        device: str | None = None,
        params: dict | None = None,
        config: str | None = None,
    ):
        """``objective`` is accepted purely for xgboost call compatibility
        (``binary:logistic`` / ``multi:softprob`` / ``multi:softmax``); the
        real objective is derived from the number of classes at fit time."""
        super().__init__(
            n_iters=n_iters,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_leaves=max_leaves,
            grower=grower,
            sampler=sampler,
            early_stopping_rounds=early_stopping_rounds,
            n_threads=n_threads,
            random_seed=random_seed,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            random_state=random_state,
            n_jobs=n_jobs,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            max_bin=max_bin,
            min_child_samples=min_child_samples,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            subsample=subsample,
            device=device,
            params=params,
            config=config,
        )
        self.objective = objective

    def _objective_pairs(self) -> dict[str, str]:
        if self.n_classes_ == 2:
            return {"dispatch.objective_name": "logloss"}
        return {
            "dispatch.objective_name": "softmax",
            "objective.n_classes": self.n_classes_,
        }

    def score(self, X, y, sample_weight=None) -> float:
        """Accuracy, matching sklearn's ``ClassifierMixin.score`` — computed
        by hand, no sklearn import."""
        y_true = np.asarray(y)
        y_pred = np.asarray(self.predict(X))
        correct = (y_true == y_pred).astype(np.float64)

        if sample_weight is None:
            return float(correct.mean())
        w = _as_1d_f32(sample_weight).astype(np.float64)
        if w.sum() == 0.0:
            raise ValueError("sample_weight sums to zero; accuracy is undefined")
        return float(np.average(correct, weights=w))

    def __sklearn_tags__(self):
        from sklearn.utils import ClassifierTags, InputTags, Tags, TargetTags

        return Tags(
            estimator_type="classifier",
            target_tags=TargetTags(required=True),
            classifier_tags=ClassifierTags(),
            input_tags=InputTags(),
        )

    def fit(self, X, y, sample_weight=None,
            eval_set: tuple | list | None = None,
            init_model: str | None = None, verbose=None) -> BonsaiClassifier:
        """`sample_weight` scales each row's gradient and hessian (sklearn's
        convention). `eval_set` takes the xgboost list-of-tuples form or a
        bare (X, y) tuple; with a list, the last entry drives the eval
        history and early stopping. init_model continues training from a
        saved .msgpack (warm start); binning reuses the loaded model's cut
        points. `verbose` is accepted for xgboost call compatibility and
        ignored."""
        del verbose
        if self.objective is not None and self.objective not in (
            "binary:logistic", "binary:logitraw",
            "multi:softprob", "multi:softmax",
        ):
            raise ValueError(
                f"BonsaiClassifier objective {self.objective!r} is not a "
                "recognized xgboost classification objective; the objective "
                "is otherwise derived from the number of classes"
            )
        eval_set = _normalize_eval_set(eval_set)
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

        pairs = self._build_pairs()
        ev = None
        if eval_set is not None:
            self._reject_dart_eval_set(pairs)
            # A label the training fold never saw cannot be encoded; letting
            # searchsorted guess silently corrupts the eval metric and early
            # stopping, so reject it.
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
            ev = (_as_2d_f32(eval_set[0]), _as_1d_f32(ev_y))
        sw = None if sample_weight is None else _as_1d_f32(sample_weight)
        Xa = _as_2d_f32(X)
        self._model = train(
            pairs, Xa, y_enc, ev, init_model, self.config,
            sample_weight=sw,
        )
        self.n_features_in_ = Xa.shape[1]
        return self

    def predict(self, X, num_iteration: int = 0) -> np.ndarray:
        """Original class labels (from ``classes_``), not the encoded
        ``0..K-1`` ids the native booster works in."""
        if self._model is None:
            raise RuntimeError("fit() or load first")
        raw = np.asarray(self._model.predict(_as_2d_f32(X), num_iteration))
        if self.n_classes_ == 2:
            idx = (raw >= 0.5).astype(np.int64)
        else:
            idx = raw.astype(np.int64)
        return self.classes_[idx]

    def predict_proba(self, X) -> np.ndarray:
        """(n_rows, n_classes) class probabilities, columns in ``classes_``
        order.

        Binary uses the ``logloss`` objective's P(class 1) directly
        (``[1 - p, p]``); multiclass uses the ``softmax`` booster's per-class
        probabilities (a row-wise softmax of the class logits).
        """
        if self._model is None:
            raise RuntimeError("fit() or load first")
        if self.n_classes_ == 2:
            p = np.asarray(self._model.predict(_as_2d_f32(X)), dtype=np.float64)
            return np.column_stack([1.0 - p, p])
        return np.asarray(self._model.predict_proba(_as_2d_f32(X)), dtype=np.float64)

    @classmethod
    def from_file(cls, path: str) -> BonsaiClassifier:
        """Load a saved ``.msgpack`` classifier.

        The native format stores only the booster, so ``classes_`` comes back
        as the encoded ids ``0..K-1`` (xgboost's ``load_model`` convention) —
        ``predict`` then returns those ids, not the label values passed to
        ``fit``. Pickle the estimator to preserve original labels.
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
