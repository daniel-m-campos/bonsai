"""bonsai: histogram gradient-boosted trees, C++23 core.

Two API layers, both over the same native module:

- ``train(params, X, y, ...)`` / ``load(path)`` return a ``Model`` — the
  thin, explicit layer. ``params`` is a list of ``(dotted.key, value)``
  pairs using exactly the keys the CLI accepts via ``--set`` (see
  ``default_config_toml()`` for all of them). ``Model`` carries
  ``predict / staged_predict / predict_leaf / pred_contribs (TreeSHAP) /
  feature_importance / dump / save``.
- ``BonsaiRegressor`` — an sklearn-style estimator wrapping the same
  booster for pipelines and quick experiments.

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

import numpy as np

from ._bonsai import Model, cuda_available, default_config_toml, load, train
from .encoding import OrderedTargetEncoder

__all__ = [
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


class BonsaiRegressor:
    """sklearn-style wrapper around the native booster.

    First-class arguments cover the common knobs; anything else can be set
    through ``params`` using dotted config keys, e.g.
    ``params={"tree.lambda_l1": 0.5, "sampler.top_rate": 0.2}`` — the same
    keys the CLI accepts via ``--set``.

    ``config`` names a TOML file used as the base config (the CLI's ``-c``).
    Keyword arguments and ``params`` always win over the file — including the
    first-class kwargs at their defaults, which are always emitted. To defer
    a knob to the file, set it there and leave it out of ``params``, or use
    ``train()`` directly.

    Duck-types the scikit-learn estimator contract (``get_params`` /
    ``set_params`` / ``score`` / ``_estimator_type``) without subclassing
    ``sklearn.base.BaseEstimator`` — sklearn is never a runtime dependency of
    this package, so ``clone``, ``Pipeline``, ``GridSearchCV`` and
    ``cross_val_score`` all work, but ``import bonsai`` never needs sklearn
    installed.
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
        params: dict | None = None,
        config: str | None = None,
    ):
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.grower = grower
        self.sampler = sampler
        self.objective = objective
        self.early_stopping_rounds = early_stopping_rounds
        self.n_threads = n_threads
        self.random_seed = random_seed
        self.params = params
        self.config = config
        self._model: Model | None = None

    def _build_pairs(self) -> list[tuple[str, str]]:
        """Translate the first-class kwargs + ``params`` into the dotted
        config keys the native ``train()`` expects. Kept out of ``__init__``
        so constructor args stay raw attributes (required for
        ``get_params``/``clone``)."""
        merged = {
            "booster.n_iters": self.n_iters,
            "booster.learning_rate": self.learning_rate,
            "booster.early_stopping_rounds": self.early_stopping_rounds,
            "booster.random_seed": self.random_seed,
            "tree.max_depth": self.max_depth,
            "tree.max_leaves": self.max_leaves,
            "dispatch.grower_name": self.grower,
            "dispatch.sampler_name": self.sampler,
            "dispatch.objective_name": self.objective,
            "parallel.n_threads": self.n_threads,
            **(self.params or {}),
        }
        return [(k, _to_config_str(v)) for k, v in merged.items()]

    def get_params(self, deep: bool = True) -> dict:
        """sklearn contract: one entry per ``__init__`` parameter, unchanged
        since construction (``deep`` is accepted for API compatibility;
        ``BonsaiRegressor`` has no nested estimators to recurse into)."""
        names = [
            p.name
            for p in inspect.signature(self.__init__).parameters.values()
            if p.name != "self"
        ]
        return {name: getattr(self, name) for name in names}

    def set_params(self, **params) -> BonsaiRegressor:
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
        (mirroring what ``RegressorMixin``/``BaseEstimator`` produce) so
        sklearn stays import-only-in-tests; empirically verified against
        sklearn 1.8.0 (see test suite)."""
        from sklearn.utils import InputTags, RegressorTags, Tags, TargetTags

        return Tags(
            estimator_type="regressor",
            target_tags=TargetTags(required=True),
            regressor_tags=RegressorTags(),
            input_tags=InputTags(),
        )

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
    def from_file(cls, path: str) -> BonsaiRegressor:
        out = cls()
        out._model = load(path)
        return out

    @property
    def n_iters_(self) -> int:
        if self._model is None:
            raise RuntimeError("fit() first")
        return self._model.n_iters


def _to_config_str(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)
