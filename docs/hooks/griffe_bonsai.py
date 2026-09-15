"""Two things the static reader misses about bonsai, supplied at site build.

The reference page is rendered by mkdocstrings reading the package statically:
``python/bonsai/*.py`` as source and the compiled module through the stub the
build writes beside it. Two places in the package say more at runtime than
their text says on disk, and each gets one hook here.

``bonsai._bonsai``: nanobind declares an overloaded function as ``@overload``
signatures and nothing else, which is what a type checker wants and what
griffe drops: an overload is filed under the name it waits for, and a function
that never arrives leaves nothing in the module's members. Six names are
declared that way (``train`` and the five ``Model`` prediction methods), each
carrying its docstring on the overloads. ``OverloadOnlyFunctions``
materializes the missing function from its first overload and attaches every
overload to it, so the page renders each signature.

``bonsai.interop``: the six translation functions build their docstrings from
the mapping tables at import (``from_xgboost.__doc__ = _doc(...)``), the one
place the cross-library names live, so the text on disk is a template.
``LiveDocstrings`` imports the module and copies each function's runtime
docstring onto its static twin. The site build has the extension built, so
the import is available; anywhere it is not, the functions render undocumented
rather than failing the build.
"""

from __future__ import annotations

import importlib

import griffe

LIVE_DOC_MODULES = ("bonsai.interop",)


class OverloadOnlyFunctions(griffe.Extension):
    """Materialize every function a stub declares only through overloads."""

    def on_module_members(self, *, mod: griffe.Module, **kwargs: object):
        _materialize(mod)

    def on_class_members(self, *, cls: griffe.Class, **kwargs: object):
        _materialize(cls)


class LiveDocstrings(griffe.Extension):
    """Copy import-time docstrings onto functions whose source has none."""

    def on_module_members(self, *, mod: griffe.Module, **kwargs: object):
        if mod.path not in LIVE_DOC_MODULES:
            return
        try:
            live = importlib.import_module(mod.path)
        except ImportError:
            return
        for name, member in mod.members.items():
            if isinstance(member, griffe.Alias) or not member.is_function or member.docstring:
                continue
            text = getattr(getattr(live, name, None), "__doc__", None)
            if text:
                member.docstring = griffe.Docstring(text, parent=member)


def _materialize(owner: griffe.Module | griffe.Class):
    for name, overloads in list(owner.overloads.items()):
        if name in owner.members:
            continue
        first = overloads[0]
        function = griffe.Function(
            name,
            parameters=first.parameters,
            returns=first.returns,
            decorators=[],
            docstring=first.docstring,
            lineno=first.lineno,
            endlineno=overloads[-1].endlineno,
            parent=owner,
        )
        function.overloads = overloads
        owner.set_member(name, function)
        del owner.overloads[name]
