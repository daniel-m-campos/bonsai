"""The xgboost-compatibility translation layer: objective-string and
metric-name tables plus the device/eval_set argument shapes, so a canonical
XGBRegressor/XGBClassifier call runs against the Bonsai estimators with only
the class name swapped. The estimator classes consume this; nothing here
touches the native module."""

from __future__ import annotations

# xgboost objective strings accepted as spellings of bonsai objectives.
_XGB_OBJECTIVES: dict[str, str] = {
    "reg:squarederror": "mse",
    "reg:absoluteerror": "mae",
    "reg:quantileerror": "quantile",
    "reg:pseudohubererror": "huber",
    "count:poisson": "poisson",
}


def _grower_for_device(grower: str, device: str) -> str:
    """xgboost's device= mapped onto bonsai's grower dispatch. CUDA has
    depthwise and oblivious growers; leafwise (the CPU default) runs as
    cuda_depthwise, the nearest CUDA grower."""
    dev = device.lower()
    if dev in ("cuda", "gpu") or dev.startswith("cuda:"):
        if grower.startswith("cuda_"):
            return grower
        return "cuda_oblivious" if grower == "oblivious" else "cuda_depthwise"
    if dev == "cpu":
        return grower.removeprefix("cuda_") if grower.startswith("cuda_") else grower
    raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")


def _normalize_eval_set(eval_set: tuple | list | None) -> tuple | None:
    """Accept both the xgboost form (a list of (X, y) tuples; the LAST one
    drives the eval history and early stopping, xgboost's own convention)
    and the bare (X, y) tuple."""
    if eval_set is None:
        return None
    if not isinstance(eval_set, list):
        return eval_set
    if not eval_set:
        return None
    return eval_set[-1]


# evals_result()/best_score presentation per objective: xgboost metric name
# and the exact transform from bonsai's eval value (mse is presented as its
# root; monotone, so best_iteration and early stopping are unaffected).
_EVAL_METRIC: dict[str, tuple[str, object]] = {
    "mse": ("rmse", lambda v: float(v) ** 0.5),
    "mae": ("mae", float),
    "huber": ("huber", float),
    "quantile": ("quantile", float),
    "poisson": ("poisson", float),
    "logloss": ("logloss", float),
    "softmax": ("mlogloss", float),
}
