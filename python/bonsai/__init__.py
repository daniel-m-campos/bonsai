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

GPU training: pass ``dispatch.grower_name = "cuda_depthwise"`` (or
``cuda_oblivious``); ``cuda_available()`` reports whether this build and
machine can honor it. Models trained on GPU predict everywhere.

The zero-to-hero walk-through lives in ``docs/guide/`` (start with
chapter 0, one boosting round traced by hand on eight rows).
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np

from ._bonsai import Model, cuda_available, default_config_toml, load, train
from .encoding import OrderedTargetEncoder

__all__ = [
    "BonsaiClassifier",
    "BonsaiRegressor",
    "Model",
    "OrderedTargetEncoder",
    "cuda_available",
    "default_config_toml",
    "load",
    "train",
]


def _as_2d_f32(X) -> np.ndarray:
    a = np.ascontiguousarray(X, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {a.shape}")
    return a


def _as_1d_f32(y) -> np.ndarray:
    a = np.ascontiguousarray(y, dtype=np.float32)
    if a.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got shape {a.shape}")
    return a


def _to_config_str(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


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
        self.params = params
        self.config = config
        self._model: Model | None = None

    def _objective_pairs(self) -> dict[str, str]:
        """Config keys the objective needs. ``BonsaiRegressor`` exposes a
        fixed ``objective`` kwarg; ``BonsaiClassifier`` derives it from
        ``classes_`` at fit time. Overridden per-subclass."""
        raise NotImplementedError

    # xgboost/lightgbm-style constructor aliases -> bonsai dotted config key.
    # Deliberately excludes `subsample` (bonsai's row-subsampling needs a
    # `sampler` choice, not a 1:1 key — `sampler.subsample` is a no-op under
    # the default `all_rows` sampler) and `min_child_weight` (bonsai's
    # `min_child_hess` is close but not identical semantics). Both, and any
    # other knob, go through `params=`.
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
        merged.update(self.params or {})
        return [(k, _to_config_str(v)) for k, v in merged.items()]

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

    def predict(self, X, num_iteration: int = 0) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit() or load first")
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

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("fit() before save()")
        self._model.save(path)

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


class BonsaiRegressor(_BonsaiEstimator):
    """sklearn-style wrapper around the native booster.

    First-class arguments cover the common knobs; anything else can be set
    through ``params`` using dotted config keys, e.g.
    ``params={"tree.lambda_l1": 0.5, "sampler.top_rate": 0.2}`` — the same
    keys the CLI accepts via ``--set``.

    xgboost/lightgbm-style aliases (``n_estimators``, ``num_leaves``,
    ``random_state``, ``n_jobs``, ``reg_lambda``, ``reg_alpha``, ``max_bin``,
    ``min_child_samples``, ``colsample_bytree``) are accepted so calls copied
    from those libraries work unchanged; they default to ``None`` and, when
    set, override the matching canonical kwarg (e.g. ``n_estimators`` wins
    over ``n_iters``). ``subsample`` and ``min_child_weight`` are deliberately
    **not** aliased — bonsai's row-subsampling needs a ``sampler`` choice
    rather than a 1:1 key, and ``min_child_hess`` isn't identical semantics
    to ``min_child_weight``; set those (or anything else) through ``params``.

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
            params=params,
            config=config,
        )
        self.objective = objective

    def _objective_pairs(self) -> dict[str, str]:
        return {"dispatch.objective_name": self.objective}

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

    def fit(self, X, y, sample_weight=None, eval_set: tuple | None = None,
            init_model: str | None = None) -> BonsaiRegressor:
        """`sample_weight` scales each row's gradient and hessian (sklearn's
        convention). init_model continues training from a saved .msgpack (warm
        start); binning reuses the loaded model's cut points."""
        pairs = self._build_pairs()
        ev = None
        if eval_set is not None:
            ev = (_as_2d_f32(eval_set[0]), _as_1d_f32(eval_set[1]))
        sw = None if sample_weight is None else _as_1d_f32(sample_weight)
        self._model = train(
            pairs, _as_2d_f32(X), _as_1d_f32(y), ev, init_model, self.config,
            sample_weight=sw,
        )
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

    Same xgboost/lightgbm-style aliases as ``BonsaiRegressor`` (``n_estimators``,
    ``num_leaves``, ``random_state``, ``n_jobs``, ``reg_lambda``, ``reg_alpha``,
    ``max_bin``, ``min_child_samples``, ``colsample_bytree``); ``subsample`` and
    ``min_child_weight`` are not aliased for the same reason — see
    ``BonsaiRegressor``'s docstring — use ``params`` for those.

    ``predict_proba`` returns calibrated-ish probabilities only for the
    binary case (the native ``logloss`` objective predicts P(class 1)
    directly); multiclass ``predict_proba`` is not yet available — see its
    docstring.
    """

    _estimator_type = "classifier"

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
        return float(np.average(correct, weights=w))

    def __sklearn_tags__(self):
        from sklearn.utils import ClassifierTags, InputTags, Tags, TargetTags

        return Tags(
            estimator_type="classifier",
            target_tags=TargetTags(required=True),
            classifier_tags=ClassifierTags(),
            input_tags=InputTags(),
        )

    def fit(self, X, y, sample_weight=None, eval_set: tuple | None = None,
            init_model: str | None = None) -> BonsaiClassifier:
        """`sample_weight` scales each row's gradient and hessian (sklearn's
        convention). init_model continues training from a saved .msgpack (warm
        start); binning reuses the loaded model's cut points."""
        y_arr = np.asarray(y)
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
            ev_y = np.searchsorted(self.classes_, np.asarray(eval_set[1]))
            ev = (_as_2d_f32(eval_set[0]), _as_1d_f32(ev_y))
        sw = None if sample_weight is None else _as_1d_f32(sample_weight)
        self._model = train(
            pairs, _as_2d_f32(X), y_enc, ev, init_model, self.config,
            sample_weight=sw,
        )
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
        """(n_rows, n_classes) class probabilities.

        Binary only for now: the native ``logloss`` objective's
        ``Model.predict`` already returns P(class 1) directly, so this is
        just ``[1 - p, p]``. Multiclass ``softmax`` currently has
        ``Model.predict`` return argmax class ids rather than per-class
        scores — ``predict()`` works for multiclass, but per-class
        probabilities need a booster-side change (a follow-up to #84;
        tracking a raw-score / per-class-scores accessor on the multiclass
        booster).
        """
        if self._model is None:
            raise RuntimeError("fit() or load first")
        if self.n_classes_ != 2:
            raise NotImplementedError(
                "predict_proba is only implemented for binary classification. "
                "The native softmax objective's Model.predict returns argmax "
                "class ids, not per-class scores, so multiclass probabilities "
                "aren't available yet (predict() still works). This needs a "
                "booster-side accessor for per-class raw scores — tracked as "
                "a follow-up to #84."
            )
        p = np.asarray(self._model.predict(_as_2d_f32(X)), dtype=np.float64)
        return np.column_stack([1.0 - p, p])
