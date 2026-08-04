"""bonsai.__version__ must self-report honestly for a source build."""

from __future__ import annotations

import pathlib
import re

import bonsai
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


def _pyproject_version() -> str:
    """The ``version`` field from the repo's pyproject.toml."""
    text = (REPO / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"$', text, flags=re.MULTILINE)
    assert match is not None, "pyproject.toml has no top-level version field"
    return match.group(1)


def test_source_build_reports_pyproject_version():
    """A build-tree import carries its own +source-tagged version.

    ``bonsai._version`` only exists in a CMake build tree (build/python,
    build-cuda/python); an installed wheel never ships it. Skip when it
    is absent so this stays silent against a plain `pip install` env.
    """
    try:
        from bonsai._version import __version__ as build_version
    except ImportError:
        pytest.skip("no build-time version stamp (not a source build)")
    assert build_version == _pyproject_version()
    assert bonsai.__version__ == f"{build_version}+source"
