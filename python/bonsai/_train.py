"""The public ``train``: the native call plus params normalization.

The native module speaks one wire format, ``(dotted.key, str)`` pairs; this
wrapper lets callers hand over a ``Params``, a dotted-key dict, or pairs,
and renders each to that format. Everything else passes through untouched,
so both native overloads (arrays and Dataset) keep their exact semantics.
"""

from __future__ import annotations

from collections.abc import Mapping

from bonsai import _bonsai
from bonsai._coerce import _to_config_str
from bonsai._params_ops import ParamsOps


def train(params, *args, **kwargs):
    """Train a booster; see ``bonsai._bonsai.train`` for the full signature.

    Parameters
    ----------
    params : Params, Mapping, Sequence of (str, value), or None
        Overrides over the library defaults (and over ``config=`` when
        given). Accepts a ``bonsai.Params``, a ``{"tree.max_depth": 8}``
        dict, or the native ``[("tree.max_depth", "8")]`` pairs; values are
        rendered to config strings. ``None`` means no overrides.
    *args, **kwargs
        Passed to the native ``train`` unchanged: ``(X, y)`` arrays or a
        ``Dataset``, then ``eval_set``, ``init_model``, ``config``, and
        (array form only) ``sample_weight``.

    Returns
    -------
    Model
        The trained booster.
    """
    return _bonsai.train(_as_pairs(params), *args, **kwargs)


def _as_pairs(params) -> list[tuple[str, str]]:
    """Render any accepted params form to the native pairs wire format."""
    if params is None:
        return []
    if isinstance(params, ParamsOps):
        return params.to_pairs()
    if isinstance(params, Mapping):
        return [(key, _to_config_str(value)) for key, value in params.items()]
    return [(key, _to_config_str(value)) for key, value in params]
