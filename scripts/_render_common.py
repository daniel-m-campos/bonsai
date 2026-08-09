"""Shared helpers for the docs generators (render_*.py).

Each generator runs by exact path (Makefile / CI), so this module is a
sibling import resolved from the scripts/ directory, not a package.
"""

from __future__ import annotations

import pathlib
import sys


def write_or_check(outputs: dict[pathlib.Path, str], *, repo: pathlib.Path,
                   script: str, label: str, detail: str) -> int:
    """Write the rendered files, or verify them under `--check`.

    Every generator ends the same way: `--check` (what CI runs) compares the
    committed files against the render and fails when any differ, and a plain
    run writes them, creating parent directories as needed.

    Parameters
    ----------
    outputs
        Path to rendered content, one entry per file. Most generators write
        one page; the results ledger writes pages, charts, and the README.
    repo
        Repository root, for spelling the paths relatively in the messages.
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
    if "--check" in sys.argv:
        stale = [p for p, text in outputs.items()
                 if not p.exists() or p.read_text() != text]
        if stale:
            names = ", ".join(str(p.relative_to(repo)) for p in stale)
            print(f"ERROR: {names} is stale; run python3 {script}",
                  file=sys.stderr)
            return 1
        print(f"{label}: in sync ({detail})")
        return 0
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    written = (str(next(iter(outputs)).relative_to(repo))
               if len(outputs) == 1 else label)
    print(f"wrote {written} ({detail})")
    return 0


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    """One GitHub-flavored markdown table; the generators own all styling."""
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])
