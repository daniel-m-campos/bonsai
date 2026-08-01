"""The published-docs corpus rules shared by docs_lint and run_docs_examples.

mkdocs.yml's `exclude_docs:` block is the single source of truth for what
publishes; both consumers must agree with it, so the parser lives once here.
"""

from __future__ import annotations

import re


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
