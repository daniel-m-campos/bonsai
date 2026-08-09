"""Shared helpers for the docs generators (render_*.py).

Each generator runs by exact path (Makefile / CI), so this module is a
sibling import resolved from the scripts/ directory, not a package.
"""

from __future__ import annotations

import pathlib
import sys


def write_or_check(out: pathlib.Path, text: str, *, repo: pathlib.Path,
                   script: str, label: str, detail: str) -> int:
    """Write the rendered page, or verify it under `--check`.

    Every generator ends the same way: `--check` (what CI runs) compares the
    committed page against the render and fails when they differ, and a plain
    run writes it.

    Parameters
    ----------
    out
        The page to write or verify.
    text
        The rendered page body.
    repo
        Repository root, for spelling `out` relatively in the messages.
    script
        The generator's path, named in the stale-page message.
    label
        The generator's name in the in-sync line, e.g. "timeline".
    detail
        What was counted, e.g. "23 milestones".

    Returns
    -------
    int
        Process exit code: 0 written or in sync, 1 stale.
    """
    rel = out.relative_to(repo)
    if "--check" in sys.argv:
        if not out.exists() or out.read_text() != text:
            print(f"ERROR: {rel} is stale; run python3 {script}",
                  file=sys.stderr)
            return 1
        print(f"{label}: in sync ({detail})")
        return 0
    out.write_text(text)
    print(f"wrote {rel} ({detail})")
    return 0


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    """One GitHub-flavored markdown table; the generators own all styling."""
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])
