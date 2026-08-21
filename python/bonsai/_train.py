"""The public ``train``: the native call plus params normalization.

The native module speaks one wire format, ``(dotted.key, str)`` pairs; this
wrapper renders a ``Params`` or a dotted-key dict to it. The pairs form is
that internal wire format, not part of this contract. Everything else
passes through untouched, so both native overloads (arrays and Dataset)
keep their exact semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import overload

import numpy.typing as npt

from bonsai import _bonsai
from bonsai._bonsai import Dataset, Model
from bonsai._coerce import _to_config_str
from bonsai._params_ops import ParamsOps


@overload
def train(params: ParamsOps | Mapping[str, object] | None,
          dataset: Dataset,
          eval_set: tuple[npt.ArrayLike, npt.ArrayLike] | Dataset | None = ...,
          init_model: str | None = ...) -> Model: ...


@overload
def train(params: ParamsOps | Mapping[str, object] | None,
          X: npt.ArrayLike, y: npt.ArrayLike,
          eval_set: tuple[npt.ArrayLike, npt.ArrayLike] | Dataset | None = ...,
          init_model: str | None = ...,
          sample_weight: npt.ArrayLike | None = ...) -> Model: ...


def train(params, *args, **kwargs) -> Model:
    """Train a booster; see ``bonsai._bonsai.train`` for the full signature.

    Parameters
    ----------
    params : Params, Mapping, or None
        Overrides over the library defaults. A ``bonsai.Params`` or a
        ``{"tree.max_depth": 8}`` dict; values are rendered to config
        strings. ``None`` means no overrides. A TOML base composes here
        too: ``Params.from_toml(path) | overrides``.
    *args, **kwargs
        Passed to the native ``train`` unchanged: ``(X, y)`` arrays or a
        ``Dataset``, then ``eval_set``, ``init_model``, and (array form
        only) ``sample_weight``.

    Returns
    -------
    Model
        The trained booster.
    """
    return _bonsai.train(_as_pairs(params), *args, **kwargs)


def _as_pairs(
        params: ParamsOps | Mapping[str, object] | None) -> list[tuple[str, str]]:
    """Render an accepted params form to the native pairs wire format."""
    if params is None:
        return []
    if isinstance(params, ParamsOps):
        params = params.to_dict()
    if isinstance(params, Mapping):
        return [(key, _to_config_str(value)) for key, value in params.items()]
    raise TypeError(
        f"params must be a bonsai.Params, a mapping of dotted keys, or None; "
        f"got {type(params).__name__}. For legacy (key, value) pairs, pass "
        f"dict(pairs).")
