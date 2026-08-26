"""Enforce the comment policy over implementation code.

Implementation code carries no comments. src/ is implementation by
definition, wherever the file extension puts it: the templated hot path
lives in headers, so a .cpp-only rule would exempt two thirds of the mass.
Contract comments live in include/ above the declarations they bind, where
review judges them; this lint does not reach include/.

Three tagged exceptions. A tag is legitimate only when it has a mechanical
admission test, because a sweep that relied on judgment missed the same
comments three times: "this one is genuine" is not a test, it is the thing
that failed. Adding a fourth tag means designing its test first.

`// perf:` carries a measurement or a hardware/build fact. The block or the
declaration under it must contain a number, since the number may BE the
constant being documented. A hardware claim with no number is not evidence:
measure it or drop it.

`// sync:` carries a concurrency or ordering constraint whose violation is a
race, undefined behaviour, or a silently wrong answer. The block must NAME a
construct from SYNC_VOCAB and that construct must appear in the file's code.
A comment that cannot name the primitive it guards is prose, not a sync
constraint.

`// ffi:` carries a Python boundary constraint (GIL, capsule ownership,
DLPack lifetime). Only src/python/ may carry it, and it must name a term
from FFI_VOCAB that the file uses.

Structural beats tagged, always. If the constraint can be made impossible
(hoist the throw out of the region, funnel placement through one call site),
do that instead and write no comment.

Structural lines are not comments in the policy's sense and always pass:
NOLINT directives, clang-format toggles, and closing-brace namespace labels.
"""

from __future__ import annotations

import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
EXTENSIONS = {".cpp", ".cu", ".hpp", ".cuh"}

STRUCTURAL = re.compile(r"NOLINT|clang-format (on|off)|^//\s*namespace\b")
PERF = re.compile(r"^// perf:")
SYNC = re.compile(r"^// sync:")
FFI = re.compile(r"^// ffi:")
DIGIT = re.compile(r"\d")

# A sync comment must name one of these, and the file must really use it.
SYNC_VOCAB = (
    "for_each_index", "parallel::", "pragma omp", "call_once", "once_flag",
    "atomic", "mutex", "lock", "thread", "cudaSetDevice", "cudaStream",
    "cudaEvent", "cudaMemcpy", "Async", "__syncthreads", "fence", "barrier",
    "gil",
)

# An ffi comment must name one of these, and only src/python/ may carry one.
FFI_VOCAB = ("nb::", "capsule", "gil", "PyObject", "DLPack", "keep_alive",
             "NB_MODULE", "dlpack", "__dlpack__")


def _names(block: list[str], code_text: str, vocab: tuple[str, ...]) -> bool:
    """True when the block names a vocabulary term the file's code really uses.

    Both halves are the test. Naming a term proves the comment is about a
    concrete construct rather than prose, and requiring the term in the code
    stops a tag being pinned to something that is not there.
    """
    return any(any(w in b for b in block) and w in code_text for w in vocab)


def block_findings(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return (line, text) findings for illegal comments in one file."""
    findings = []
    lines = path.read_text().splitlines()
    code_text = "\n".join(ln for ln in lines
                          if not ln.strip().startswith("//"))
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if line.startswith("//"):
            j = i
            while j < len(lines) and lines[j].strip().startswith("//"):
                j += 1
            block = [lines[k].strip() for k in range(i, j)]
            first = block[0]
            if PERF.match(first):
                # The number may be the declaration itself: a measured
                # constant states its measurement by being that value.
                decl = lines[j] if j < len(lines) else ""
                if not any(DIGIT.search(b) for b in [*block, decl]):
                    findings.append((i + 1, "perf: carries no number, so it is"
                                            f" not a measurement: {first[:52]}"))
            elif SYNC.match(first):
                if not _names(block, code_text, SYNC_VOCAB):
                    findings.append((i + 1, "sync: names no construct this file"
                                            f" uses: {first[:56]}"))
            elif FFI.match(first):
                if "python" not in path.parts:
                    findings.append((i + 1, "ffi: only src/python/ may carry"
                                            f" this tag: {first[:56]}"))
                elif not _names(block, code_text, FFI_VOCAB):
                    findings.append((i + 1, "ffi: names no boundary construct"
                                            f" this file uses: {first[:56]}"))
            else:
                for k in range(i, j):
                    if not STRUCTURAL.search(lines[k].strip()):
                        findings.append((k + 1, lines[k].strip()[:80]))
            i = j
            continue
        code, sep, trail = raw.partition("//")
        if sep and code.strip() and '"' not in code and "'" not in code:
            comment = "// " + trail.strip()
            tagged = trail.strip().startswith(("perf:", "sync:", "ffi:"))
            if not STRUCTURAL.search(comment) and not tagged:
                findings.append((i + 1, comment[:80]))
        i += 1
    return findings


def main() -> int:
    total = 0
    for path in sorted(SRC.rglob("*")):
        if path.suffix not in EXTENSIONS:
            continue
        for line, text in block_findings(path):
            print(f"{path.relative_to(REPO)}:{line}: {text}")
            total += 1
    if total:
        print(f"comment-lint: {total} comment lines in src/ outside the policy")
        return 1
    print("comment-lint: src/ clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
