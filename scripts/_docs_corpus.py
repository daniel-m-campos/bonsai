"""The published-docs corpus rules shared by docs_lint and run_docs_examples.

mkdocs.yml's `exclude_docs:` block is the single source of truth for what
publishes; both consumers must agree with it, so the parser lives once here,
and `corpus_files` builds the lint's file list on top of it.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

# The frozen historical record: first-class for agents and deep divers, but
# out of the main line, so the prose lint leaves it as written.
FROZEN_PAGE = "decisions.md"
FROZEN_TREES = ("architecture/",)


def excluded_patterns(mkdocs_text: str) -> list[str]:
    """The `exclude_docs:` block-scalar patterns, docs-relative."""
    lines = mkdocs_text.splitlines()
    pats: list[str] = []
    for idx, line in enumerate(lines):
        if not re.match(r"^exclude_docs:\s*\|", line):
            continue
        for follow in lines[idx + 1:]:
            if follow.strip() == "":
                continue
            if not follow.startswith((" ", "\t")):
                break  # dedent: next top-level key
            pats.append(follow.strip())
        break
    return pats


def is_excluded(rel: str, pats: list[str]) -> bool:
    """Whether a docs-relative path matches any exclude pattern."""
    name = rel.rsplit("/", 1)[-1]
    for p in pats:
        if p.endswith("/"):
            if rel == p.rstrip("/") or rel.startswith(p):
                return True
        elif rel == p or name == p:  # bare name matches at any depth
            return True
    return False


def corpus_files() -> list[pathlib.Path]:
    """The published prose files the lint covers: README plus every docs page
    mkdocs publishes, minus the frozen historical record."""
    pats = excluded_patterns((REPO / "mkdocs.yml").read_text())
    files = [REPO / "README.md"]
    for path in sorted(DOCS.rglob("*.md")):
        rel = path.relative_to(DOCS).as_posix()
        if (is_excluded(rel, pats) or rel == FROZEN_PAGE
                or rel.startswith(FROZEN_TREES)):
            continue
        files.append(path)
    return files
