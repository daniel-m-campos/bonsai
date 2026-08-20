"""Behavior for the generated Params dataclasses (``bonsai/_params.py``).

The generated module carries data only: one frozen dataclass per config
section plus ``Params``, every field defaulting to ``None`` ("leave the
library default"). Everything behavioral lives here, on a mixin the
generated ``Params`` inherits, so the generator stays a renderer.

Values are kept as the caller gave them; rendering to the dotted-key
string wire format happens in the ``train`` wrapper, and range/enum
validation stays in the C++ config layer, told once.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import ClassVar


class SparseRepr:
    """Repr showing only set fields: ``Tree(max_depth=8)``, not eleven Nones.

    The generated dataclasses opt out of the default repr (``repr=False``)
    and inherit this one, because unset-means-default makes the default
    repr all noise.
    """

    def __repr__(self) -> str:
        shown = ", ".join(
            f"{f.name}={getattr(self, f.name)!r}"
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None)
        return f"{type(self).__name__}({shown})"


class ParamsOps(SparseRepr):
    """Mixin for the generated ``Params``: round-trips and dict-style merge.

    ``to_dict`` walks only the fields that are set, so a
    default-constructed ``Params`` renders to no overrides at all.
    ``|`` merges the way dicts do (right side's set leaves win), which is
    the sweep idiom: ``train(BASE | {"tree.max_depth": d}, ds)``.
    """

    # Supplied by the generated subclass: section name -> section dataclass.
    _SECTION_TYPES: ClassVar[dict[str, type]]

    def to_dict(self) -> dict[str, object]:
        """The set overrides as ``{dotted.key: value}`` with Python types.

        The form optuna/MLflow log well, and the inverse of ``from_dict``.
        """
        out: dict[str, object] = {}
        for section_field in dataclasses.fields(self):
            section = getattr(self, section_field.name)
            if section is None:
                continue
            for leaf in dataclasses.fields(section):
                value = getattr(section, leaf.name)
                if value is not None:
                    out[f"{section_field.name}.{leaf.name}"] = value
        return out

    @classmethod
    def from_toml(cls, path: str) -> ParamsOps:
        """Build a ``Params`` from the keys a TOML config file explicitly sets.

        The C++ config layer does the parsing (no TOML dependency, any
        Python), strict like the CLI's ``-c``: unknown sections or keys
        raise. Only stated keys are carried, so the result stays a sparse
        overrides object: ``train(Params.from_toml(p) | sweep, ds)`` is the
        file-base-plus-overrides layering ``config=`` used to express.
        """
        from bonsai import _bonsai
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(_bonsai._params_from_toml(fh.read()))

    @classmethod
    def from_model(cls, model) -> ParamsOps:
        """The config a trained ``Model`` resolved, as a fully-set ``Params``.

        Every key is set (a model's config is fully determined), so this is
        for inspection and reproduction, not sparse layering; drop sections
        with ``dataclasses.replace(p, bin_mapper=None)`` where a consumer
        rejects them.
        """
        from bonsai import _bonsai
        return cls.from_dict(_bonsai._params_from_toml(model.config_toml))

    @classmethod
    def from_dict(cls, mapping: Mapping[str, object]) -> ParamsOps:
        """Build a ``Params`` from ``{dotted.key: value}``.

        Parameters
        ----------
        mapping : Mapping[str, object]
            Dotted keys exactly as ``train()`` pairs spell them
            (``"tree.max_depth"``); values keep their Python types.

        Raises
        ------
        ValueError
            On a key with no dot, an unknown section, or an unknown leaf,
            named in the message next to the legal choices.
        """
        section_types = cls._SECTION_TYPES
        by_section: dict[str, dict[str, object]] = {}
        for key, value in mapping.items():
            section, dot, leaf = key.partition(".")
            if not dot or section not in section_types:
                raise ValueError(
                    f"unknown params key {key!r}: expected 'section.name' with "
                    f"section one of {sorted(section_types)}")
            legal = {f.name for f in dataclasses.fields(section_types[section])}
            if leaf not in legal:
                raise ValueError(
                    f"unknown params key {key!r}: [{section}] has {sorted(legal)}")
            by_section.setdefault(section, {})[leaf] = value
        return cls(**{s: section_types[s](**kv) for s, kv in by_section.items()})

    def __or__(self, other: object) -> ParamsOps:
        if isinstance(other, ParamsOps):
            updates = other.to_dict()
        elif isinstance(other, Mapping):
            updates = dict(other)
        else:
            return NotImplemented
        return type(self).from_dict({**self.to_dict(), **updates})
