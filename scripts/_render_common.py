"""Shared helpers for the docs generators (render_*.py).

Each generator runs by exact path (Makefile / CI), so this module is a
sibling import resolved from the scripts/ directory, not a package.
"""

from __future__ import annotations


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    """One GitHub-flavored markdown table; the generators own all styling."""
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])
