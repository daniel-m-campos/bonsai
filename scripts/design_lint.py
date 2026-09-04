"""Ratchet the design properties review cannot hold by hand.

A style preference becomes enforceable the moment it has a number and a
direction. This lint measures a few such numbers over production code and
refuses any change that moves a gated one the wrong way against the pinned
baseline. The baseline moves down freely, whenever a change improves a
number, and moves up only through ``--update-baseline`` in a commit whose
message says why, so a regression is a decision rather than a drift.

Gated numbers, each standing for one sentence of the house style:

``clone_windows``
    Redundant copies of six-line windows with literals normalized. One
    concept has one home; the second copy is the one that gets fixed
    without the first. Include and import blocks and rows made only of
    literals (a string table, a knob map, an initializer) are outside any
    narrative and are skipped; the first two were 38% of the count before
    they were, and the tables another 19%.
``vocabulary_singletons``
    Words that appear in exactly one public name under include/. The
    conceptual model is a small shared vocabulary; a word used once is
    either a concept nobody else reaches for or a metaphor that should
    have been a plain noun.
``contract_prose_lines``
    Comment lines under include/. A contract carried by prose is one that
    a name, a type, or a test has not carried yet.
``binding_prose_lines``
    Lines of src/python/module.cpp made only of string literals, which is
    to say docstrings and signatures written in C++. Documentation belongs
    beside the surface it documents.

Tracked but not gated:

``shape_clone_windows``
    The same window count with identifiers normalized too, so a block
    copied and renamed still matches. It is the more honest duplication
    measure and the noisier one: a table of bindings or knobs repeats a
    shape on purpose, and gating it would fail a change for adding one
    legitimate row. It is reported so the trend is visible.
``hardware_leaks``
    Hardware vocabulary (cuda, warp, pinned, occupancy) on code lines above
    the engine seam: everything under include/bonsai and src outside the
    cuda, metal, and registry directories. Above the seam the narrative is
    statistical or layout; a caller asking ``starts_with("cuda")`` is
    guessing at the machine below it, and the seam should answer instead.
    Tracked because each one is a placement decision rather than a slip.

None of these is a quality score. Each is a direction, and the ratchet
turns a preference the reviewer would have to re-argue every round into a
gate the change has to pass. A total can hide a swap (six words admitted
and six retired reads as +0) and a fall can be a divergence rather than a
consolidation (a fourth copy drifting away from three), so ``--against``
prints the sites and words that moved, not only the sums.

Example
-------
    $ python3 scripts/design_lint.py
    $ python3 scripts/design_lint.py --against /tmp/main-export
    $ python3 scripts/design_lint.py --update-baseline
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Final

ROOT: Final = pathlib.Path(__file__).resolve().parents[1]
BASELINE: Final = ROOT / "scripts" / "design_baseline.json"
PRODUCTION: Final = ("include", "src", "python/bonsai", "scripts")
SUFFIXES: Final = (".cpp", ".hpp", ".cu", ".cuh", ".py")
GENERATED: Final = ("_params.py", "_version.py")
WINDOW: Final = 6
MIN_WINDOW_CHARS: Final = 120
GATED: Final = frozenset(
    {"clone_windows", "vocabulary_singletons", "contract_prose_lines", "binding_prose_lines"}
)

SKIP_WORDS: Final = frozenset(
    {"if", "for", "while", "return", "switch", "sizeof", "static_cast", "operator"}
)
KEYWORDS: Final = frozenset(
    """if else for while return throw try catch const auto void int float double bool
    size_t class struct template typename namespace using static inline constexpr new
    delete this nullptr true false switch case default public private protected virtual
    override final operator sizeof static_cast reinterpret_cast dynamic_cast std
    def import from as in not and or is None True False lambda with pass raise yield
    break continue self cls""".split()
)
BELOW_SEAM: Final = frozenset({"cuda", "metal", "registry"})
HARDWARE_WORDS: Final = re.compile(r"\b(?:cuda\w*|warp\w*|pinned|occupancy|nvml\w*)\b", re.I)
TYPE_DECLARATION: Final = re.compile(
    r"\b(?:class|struct|enum(?: class)?|using)\s+([A-Za-z_]\w*)\b(?!::)"
)
FUNCTION_DECLARATION: Final = re.compile(r"^\s*((?:[\w:<>,\*&]+\s+)+)([a-z_]\w*)\s*\(")
NOT_A_NAME: Final = KEYWORDS | {"decltype", "requires", "noexcept", "alignof", "static_assert"}
CALL_LEADERS: Final = frozenset({"return", "throw", "co_return"})
MIN_ALIAS_CHARS: Final = 3
LITERAL_ROW: Final = re.compile(r"^\s*(?:\"[^\"]*\"|'[^']*'|\d+(?:\.\d+)?[fF]?|[\s:,()\[\]{}])+$")


@dataclass(frozen=True)
class Reading:
    """One metric's current value with the sites that dominate it."""

    value: int
    top_sites: list[tuple[str, int]]


def measure(root: pathlib.Path) -> dict[str, Reading]:
    """Compute every metric over the production tree under ``root``.

    Parameters
    ----------
    root
        Repository root.

    Returns
    -------
    dict[str, Reading]
        Metric name to its reading, in report order.
    """
    files = _production_files(root)
    headers = [f for f in files if f.is_relative_to(root / "include")]
    return {
        "clone_windows": _clone_windows(files, root, blind_to_identifiers=False),
        "shape_clone_windows": _clone_windows(files, root, blind_to_identifiers=True),
        "vocabulary_singletons": _vocabulary_singletons(headers),
        "contract_prose_lines": _comment_lines(headers, root),
        "binding_prose_lines": _string_literal_lines(root / "src/python/module.cpp", root),
        "hardware_leaks": _hardware_leaks(files, root),
    }


def compare(readings: dict[str, Reading], baseline: dict[str, int]) -> bool:
    """Print the scoreboard; return True when no gated metric exceeds its baseline."""
    ok = True
    print(f"{'metric':<24}{'baseline':>10}{'now':>8}{'delta':>8}  verdict")
    for name, reading in readings.items():
        pinned = baseline.get(name)
        if pinned is None:
            print(f"{name:<24}{'-':>10}{reading.value:>8}{'':>8}  unpinned")
            continue
        delta = reading.value - pinned
        gated = name in GATED
        verdict = "tracked" if not gated else "ok" if delta <= 0 else "REGRESSED"
        ok = ok and (not gated or delta <= 0)
        print(f"{name:<24}{pinned:>10}{reading.value:>8}{delta:>+8}  {verdict}")
        if gated and delta > 0:
            for site, n in reading.top_sites[:5]:
                print(f"    {n:6d}  {site}")
    return ok


def churn(before: dict[str, Reading], after: dict[str, Reading]):
    """Print, per metric, every site or word whose count moved between two trees."""
    for name, reading in after.items():
        was = dict(before[name].top_sites)
        now = dict(reading.top_sites)
        moved = {k: now.get(k, 0) - was.get(k, 0) for k in was.keys() | now.keys()}
        moved = {k: d for k, d in moved.items() if d}
        if not moved:
            continue
        print(f"{name} {reading.value - before[name].value:+d}")
        if name == "vocabulary_singletons":
            print("    admitted: " + ", ".join(sorted(k for k, d in moved.items() if d > 0)))
            print("    retired:  " + ", ".join(sorted(k for k, d in moved.items() if d < 0)))
            continue
        for site, d in sorted(moved.items(), key=lambda kv: -abs(kv[1])):
            print(f"    {d:+5d}  {site}")


def main() -> int:
    """Entry point; exit 1 on a regression against the baseline."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--update-baseline", action="store_true",
                        help="pin the current values as the new baseline")
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--against", type=pathlib.Path,
                        help="a second tree (an export of origin/main) whose readings "
                             "to diff site by site")
    args = parser.parse_args()

    readings = measure(args.root)
    if args.update_baseline:
        BASELINE.write_text(
            json.dumps({k: r.value for k, r in readings.items()}, indent=2) + "\n"
        )
        print(f"design-lint: baseline pinned to {BASELINE.relative_to(ROOT)}")
        return 0
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    ok = compare(readings, baseline)
    if args.against:
        print(f"churn against {args.against}")
        churn(measure(args.against), readings)
    print("design-lint: ok" if ok else "design-lint: a gated metric moved the wrong way")
    return 0 if ok else 1


def _production_files(root: pathlib.Path) -> list[pathlib.Path]:
    out = []
    for sub in PRODUCTION:
        for f in (root / sub).rglob("*"):
            if f.suffix in SUFFIXES and f.name not in GENERATED and "build" not in f.parts:
                out.append(f)
    return sorted(set(out))


def _lines(f: pathlib.Path) -> list[str]:
    return [line for line in f.read_text(errors="replace").splitlines() if line.strip()]


def _normalize(line: str, blind_to_identifiers: bool) -> str:
    line = re.sub(r'"[^"]*"', '"S"', line)
    line = re.sub(r"\b\d+(\.\d+)?[fF]?\b", "N", line)
    if blind_to_identifiers:
        line = re.sub(
            r"\b[A-Za-z_]\w*\b",
            lambda m: m.group(0) if m.group(0) in KEYWORDS else "id",
            line,
        )
    return re.sub(r"\s+", " ", line).strip()


def _clone_windows(
    files: list[pathlib.Path], root: pathlib.Path, blind_to_identifiers: bool
) -> Reading:
    windows: dict[str, list[pathlib.Path]] = collections.defaultdict(list)
    for f in files:
        raw = [line for line in _lines(f) if not _is_outside_the_narrative(line)]
        normalized = [_normalize(line, blind_to_identifiers) for line in raw]
        for i in range(len(raw) - WINDOW + 1):
            text = "".join(raw[i : i + WINDOW])
            if len(text) < MIN_WINDOW_CHARS or text.count("}") >= 4:
                continue
            windows["\n".join(normalized[i : i + WINDOW])].append(f)
    per_file: collections.Counter[str] = collections.Counter()
    for sites in windows.values():
        for f in sites[1:]:
            per_file[str(f.relative_to(root))] += 1
    return Reading(sum(per_file.values()), per_file.most_common())


def _split_words(name: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return [w.lower() for w in spaced.split("_") if w]


def _vocabulary_singletons(headers: list[pathlib.Path]) -> Reading:
    names = {
        name for f in headers for line in _lines(f) if (name := _declared_name(line))
    }
    counts: collections.Counter[str] = collections.Counter()
    for name in names:
        counts.update(_split_words(name))
    singles = sorted(w for w, c in counts.items() if c == 1)
    return Reading(len(singles), [(w, 1) for w in singles])


def _declared_name(line: str) -> str | None:
    code = line.split("//", 1)[0]
    if typed := TYPE_DECLARATION.search(code):
        name = typed.group(1)
        local_alias = code.lstrip().startswith("using") and len(name) < MIN_ALIAS_CHARS
        return None if name in NOT_A_NAME or local_alias else name
    called = FUNCTION_DECLARATION.match(code)
    if not called or called.group(1).split()[0] in CALL_LEADERS:
        return None
    name = called.group(2)
    return None if name in NOT_A_NAME or name in SKIP_WORDS else name


def _comment_lines(headers: list[pathlib.Path], root: pathlib.Path) -> Reading:
    per_file: collections.Counter[str] = collections.Counter()
    for f in headers:
        n = sum(1 for line in _lines(f) if line.strip().startswith("//"))
        if n:
            per_file[str(f.relative_to(root))] = n
    return Reading(sum(per_file.values()), per_file.most_common())


def _hardware_leaks(files: list[pathlib.Path], root: pathlib.Path) -> Reading:
    per_file: collections.Counter[str] = collections.Counter()
    for f in files:
        rel = str(f.relative_to(root))
        if any(part in BELOW_SEAM for part in f.parts) or not rel.startswith(("include", "src")):
            continue
        n = sum(
            len(HARDWARE_WORDS.findall(line))
            for line in _lines(f)
            if not line.strip().startswith(("//", "#include"))
        )
        if n:
            per_file[rel] = n
    return Reading(sum(per_file.values()), per_file.most_common())


def _is_bare_literal(line: str) -> bool:
    return re.search(r"[\"'\d]", line) is not None and LITERAL_ROW.match(line) is not None


def _is_outside_the_narrative(line: str) -> bool:
    return _is_bare_literal(line) or line.lstrip().startswith(("#include", "import ", "from "))


def _string_literal_lines(module: pathlib.Path, root: pathlib.Path) -> Reading:
    if not module.exists():
        return Reading(0, [])
    n = sum(1 for line in _lines(module) if _is_bare_literal(line))
    return Reading(n, [(str(module.relative_to(root)), n)])


if __name__ == "__main__":
    sys.exit(main())
