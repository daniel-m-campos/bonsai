"""Hold the docs theme to the Material release it is installed against.

The theme reaches into mkdocs-material's internals, not only its config:
the override templates extend blocks of the base template, and the
stylesheet targets Material's own class names. A Material release that
renames either does not fail `mkdocs build --strict`; the hero would
vanish or the tables would lose their styling on the deployed site. This
check runs in `make site` before the build and fails on the first block
or class the installed Material no longer defines.

Usage::

    python3 scripts/theme_check.py
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Final

REPO: Final = pathlib.Path(__file__).resolve().parents[1]
OVERRIDES: Final = REPO / "overrides"
STYLESHEET: Final = REPO / "docs" / "stylesheets" / "bonsai.css"
BLOCK: Final = re.compile(r"{%-?\s*block\s+(\w+)")
MD_CLASS: Final = re.compile(r"\.(md-[a-z0-9_-]+)")


# Public Functions =================================================================================


def main() -> int:
    """Report every override block and stylesheet class Material lacks."""
    templates = _material_templates()
    missing_blocks = sorted(override_blocks(OVERRIDES) - base_blocks(templates / "base.html"))
    missing_classes = sorted(stylesheet_classes(STYLESHEET) - material_classes(templates))
    for kind, names in (("block", missing_blocks), ("class", missing_classes)):
        for name in names:
            print(f"theme-check: installed mkdocs-material has no {kind} `{name}`")
    if missing_blocks or missing_classes:
        return 1
    print("theme-check: every override block and .md- class exists in the installed theme")
    return 0


def base_blocks(base_html: pathlib.Path) -> set[str]:
    """The block names Material's base template defines."""
    return set(BLOCK.findall(base_html.read_text()))


def override_blocks(overrides: pathlib.Path) -> set[str]:
    """The block names the override templates extend."""
    names: set[str] = set()
    for template in overrides.glob("*.html"):
        names.update(BLOCK.findall(template.read_text()))
    return names


def stylesheet_classes(stylesheet: pathlib.Path) -> set[str]:
    """Every `.md-*` class the theme stylesheet selects."""
    return set(MD_CLASS.findall(stylesheet.read_text()))


def material_classes(templates: pathlib.Path) -> set[str]:
    """Every `.md-*` class Material's bundled stylesheets and templates name."""
    names: set[str] = set()
    for path in templates.rglob("*"):
        if path.suffix in {".css", ".html"} and path.is_file():
            names.update(MD_CLASS.findall(path.read_text(errors="ignore")))
            names.update(re.findall(r"\b(md-[a-z0-9_-]+)", path.read_text(errors="ignore")))
    return names


# Private Functions ================================================================================


def _material_templates() -> pathlib.Path:
    try:
        import material
    except ImportError:
        sys.exit("theme-check: mkdocs-material is not installed in this interpreter")
    return pathlib.Path(material.__file__).resolve().parent / "templates"


if __name__ == "__main__":
    sys.exit(main())
