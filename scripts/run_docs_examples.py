"""Execute the runnable examples in the docs so they cannot rot.

A fenced block opts into execution with the superfences attribute form:

    ```{.python .run}
    import bonsai
    ...
    ```

On the site it renders as an ordinary python block (attr_list + superfences
are enabled in mkdocs.yml). This script extracts every `.run` block from the
published pages, writes each to a temp file, and runs it with this
interpreter. A nonzero exit fails the run, naming the page and the block's
index on that page. Each block must be self-contained per docs/STYLE.md:
imports included, synthetic data generated inline, no downloads, small enough
to finish in seconds.

Blocks that need a GPU use `{.python .run-gpu}`. They are extracted and listed
but skipped unless `--gpu` is passed, so the default CI run stays CPU-only.

    PYTHONPATH=build/python python3 scripts/run_docs_examples.py
    PYTHONPATH=build/python python3 scripts/run_docs_examples.py --gpu

Stdlib only. The block runs in a scratch working directory, so a `save(...)`
call lands nowhere real. The interpreter and its PYTHONPATH are inherited, so
`import bonsai` resolves against whatever the caller set up (build/python).
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

# Each block runs in a scratch cwd, so any relative PYTHONPATH the caller set
# (build/python) is absolutized here against the invocation directory before
# it is handed to the child.
_INVOKED_FROM = os.getcwd()


def _child_env() -> dict:
    env = os.environ.copy()
    pp = env.get("PYTHONPATH")
    if pp:
        env["PYTHONPATH"] = os.pathsep.join(
            os.path.join(_INVOKED_FROM, part) if part and not os.path.isabs(part) else part
            for part in pp.split(os.pathsep))
    return env

OPEN_RE = re.compile(r"^(\s*)(`{3,}|~{3,})\{([^}]*)\}\s*$")


def excluded_patterns(mkdocs_text: str) -> list[str]:
    lines = mkdocs_text.splitlines()
    pats: list[str] = []
    for idx, line in enumerate(lines):
        if not re.match(r"^exclude_docs:\s*\|", line):
            continue
        for follow in lines[idx + 1:]:
            if follow.strip() == "":
                continue
            if not follow.startswith((" ", "\t")):
                break
            pats.append(follow.strip())
        break
    return pats


def is_excluded(rel: str, pats: list[str]) -> bool:
    name = rel.rsplit("/", 1)[-1]
    for p in pats:
        if p.endswith("/"):
            if rel == p.rstrip("/") or rel.startswith(p):
                return True
        elif rel == p or name == p:
            return True
    return False


def published_files() -> list[pathlib.Path]:
    pats = excluded_patterns((REPO / "mkdocs.yml").read_text())
    files = [REPO / "README.md"]
    for path in sorted(DOCS.rglob("*.md")):
        rel = path.relative_to(DOCS).as_posix()
        if not is_excluded(rel, pats):
            files.append(path)
    return files


def extract(text: str) -> list[tuple[str, int, str]]:
    """(kind, opener_line_no, code) for each .run / .run-gpu block; kind in
    {"cpu", "gpu"}."""
    lines = text.splitlines()
    blocks: list[tuple[str, int, str]] = []
    i = 0
    while i < len(lines):
        m = OPEN_RE.match(lines[i])
        if not m:
            i += 1
            continue
        fence = m.group(2)
        attrs = m.group(3).split()
        close_re = re.compile(r"^\s*" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$")
        body: list[str] = []
        j = i + 1
        while j < len(lines) and not close_re.match(lines[j]):
            body.append(lines[j])
            j += 1
        if ".run-gpu" in attrs:
            blocks.append(("gpu", i + 1, "\n".join(body)))
        elif ".run" in attrs:
            blocks.append(("cpu", i + 1, "\n".join(body)))
        i = j + 1
    return blocks


def run_block(code: str) -> tuple[bool, str, float]:
    with tempfile.TemporaryDirectory() as td:
        script = pathlib.Path(td) / "example.py"
        script.write_text(code + "\n")
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=td, capture_output=True, text=True, env=_child_env())
        dt = time.perf_counter() - t0
    ok = proc.returncode == 0
    return ok, proc.stderr.strip(), dt


def main() -> int:
    want_gpu = "--gpu" in sys.argv
    failures = 0
    ran = 0
    skipped_gpu = 0
    total_time = 0.0

    for path in published_files():
        rel = path.relative_to(REPO).as_posix()
        blocks = extract(path.read_text())
        for index, (kind, lineno, code) in enumerate(blocks):
            tag = ".run-gpu" if kind == "gpu" else ".run"
            if kind == "gpu" and not want_gpu:
                skipped_gpu += 1
                print(f"SKIP {rel} block {index} ({tag}, line {lineno}): needs --gpu")
                continue
            ok, err, dt = run_block(code)
            total_time += dt
            ran += 1
            if ok:
                print(f"ok   {rel} block {index} ({tag}, line {lineno})  {dt:.2f}s")
            else:
                failures += 1
                print(f"FAIL {rel} block {index} ({tag}, line {lineno})  {dt:.2f}s")
                print("     " + err.replace("\n", "\n     "))

    print()
    print(f"docs-examples: {ran} blocks run in {total_time:.1f}s, "
          f"{failures} failed, {skipped_gpu} gpu block(s) skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
