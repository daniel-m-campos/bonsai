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
    """

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
        self._params = {
            "booster.n_iters": n_iters,
            "booster.learning_rate": learning_rate,
            "booster.early_stopping_rounds": early_stopping_rounds,
            "booster.random_seed": random_seed,
            "tree.max_depth": max_depth,
            "tree.max_leaves": max_leaves,
            "dispatch.grower_name": grower,
            "dispatch.sampler_name": sampler,
            "dispatch.objective_name": objective,
            "parallel.n_threads": n_threads,
            **(params or {}),
        }
        self._config = config
        self._model: Model | None = None

    def fit(self, X, y, eval_set: tuple | None = None,
            init_model: str | None = None) -> "BonsaiRegressor":
        """init_model continues training from a saved .msgpack (warm start);
        binning reuses the loaded model's cut points."""
        pairs = [(k, _to_config_str(v)) for k, v in self._params.items()]
        ev = None
        if eval_set is not None:
            ev = (_as_2d_f32(eval_set[0]), _as_1d_f32(eval_set[1]))
        self._model = train(
            pairs, _as_2d_f32(X), _as_1d_f32(y), ev, init_model, self._config
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
    def from_file(cls, path: str) -> "BonsaiRegressor":
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
