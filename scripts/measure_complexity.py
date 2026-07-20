"""Measure bonsai's own tree: size, cyclomatic complexity, and surface
counts, written to benchmarks/results/code-metrics-2026-07.jsonl (the code
division of the results ledger; rules in docs/method/benchmark-protocol.md).

    make python                              # extension for the API count
    .venv/bin/python scripts/measure_complexity.py

The plane map (each tracked .hpp/.cpp/.cu/.cuh/.py file lands in exactly
one plane; an unmapped code file is a hard error so tree drift fails loudly):

    core_headers   include/bonsai/** except cuda/ and cli/
    engine_impl    src/** except cuda/, cli/, python/
    cuda_plane     src/cuda/**, include/bonsai/cuda/**
    bindings_cli   src/python/**, src/cli/**, include/bonsai/cli/**,
                   python/bonsai/** except bench/
    bench_tooling  python/bonsai/bench/**, scripts/*.py, benchmarks/*.cpp
    tests          tests/**, python/tests/**

Complexity comes from lizard, pinned via uvx exactly like ruff in the
Makefile; the version is recorded in the meta row. LOC is wc -l over the
plane's code files; NLOC/CCN are lizard's. The five highest-CCN functions
across core_headers + engine_impl are published by name, deliberately.

Determinism: files come sorted from `git ls-files`, planes and offenders
have fixed orderings, and the recorded date is HEAD's commit date, so the
output is a pure function of the tree at one SHA (run twice, byte-equal).
Re-measurement supersedes this file in place (decision 69).
"""

from __future__ import annotations

import csv
import io
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "benchmarks" / "results" / "code-metrics-2026-07.jsonl"

LIZARD_PIN = "lizard@1.23.0"
CODE_EXTS = (".hpp", ".cpp", ".cu", ".cuh", ".py")
SCOPES = ["include", "src", "python", "tests", "scripts", "benchmarks"]

# Adjustments to the prescribed map, forced by the tree (flagged in review):
# include/bonsai/{detail,io}/ and src/{config,io,registry}/ exist and belong
# to the core planes; include/bonsai/cli/ pairs with src/cli/; python/tests/
# holds the binding tests; benchmarks/*.cpp are Catch2 micro-benchmarks.
PLANE_MAP = {
    "core_headers": ["include/bonsai/*.hpp", "include/bonsai/config/**",
                     "include/bonsai/registry/**", "include/bonsai/detail/**",
                     "include/bonsai/io/**"],
    "engine_impl": ["src/*.hpp", "src/*.cpp", "src/config/**", "src/io/**",
                    "src/registry/**"],
    "cuda_plane": ["src/cuda/**", "include/bonsai/cuda/**"],
    "bindings_cli": ["src/python/**", "src/cli/**", "include/bonsai/cli/**",
                     "python/bonsai/** except bench/"],
    "bench_tooling": ["python/bonsai/bench/**", "scripts/*.py",
                      "benchmarks/*.cpp"],
    "tests": ["tests/**", "python/tests/**"],
}


def plane_of(path: str) -> str | None:
    """First matching rule wins; order mirrors PLANE_MAP's exceptions."""
    if path.startswith(("src/cuda/", "include/bonsai/cuda/")):
        return "cuda_plane"
    if path.startswith(("src/python/", "src/cli/", "include/bonsai/cli/")):
        return "bindings_cli"
    if path.startswith("python/bonsai/bench/"):
        return "bench_tooling"
    if path.startswith("python/bonsai/"):
        return "bindings_cli"
    if path.startswith(("tests/", "python/tests/")):
        return "tests"
    if path.startswith("include/bonsai/"):
        return "core_headers"
    if path.startswith("src/"):
        return "engine_impl"
    if path.startswith(("scripts/", "benchmarks/")) and path.count("/") == 1:
        return "bench_tooling"
    return None


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.strip()


def plane_files() -> dict[str, list[str]]:
    files = [f for f in git("ls-files", "--", *SCOPES).splitlines()
             if f.endswith(CODE_EXTS)]
    planes: dict[str, list[str]] = {name: [] for name in PLANE_MAP}
    unmapped = []
    for f in sorted(files):
        plane = plane_of(f)
        (planes[plane] if plane else unmapped).append(f)
    if unmapped:
        sys.exit(f"ERROR: code files outside the plane map: {unmapped}\n"
                 "Extend PLANE_MAP/plane_of or exclude them explicitly.")
    return planes


# ---- lizard ------------------------------------------------------------------


def run_lizard(files: list[str], *flags: str) -> str:
    proc = subprocess.run(["uvx", LIZARD_PIN, *flags, *files], cwd=REPO,
                          capture_output=True, text=True)
    # lizard exits 1 when functions exceed its default CCN threshold; that
    # is a finding here, not a failure.
    if proc.returncode not in (0, 1):
        sys.exit(f"ERROR: uvx {LIZARD_PIN} failed:\n{proc.stderr}")
    return proc.stdout


def lizard_functions(files: list[str]) -> list[dict]:
    """--csv rows: nloc,ccn,token,param,length,location,file,name,sig,start,end."""
    out = []
    for row in csv.reader(io.StringIO(run_lizard(files, "--csv"))):
        out.append({"nloc": int(row[0]), "ccn": int(row[1]), "file": row[6],
                    "name": row[7], "start": int(row[9])})
    return out


def lizard_file_nloc(files: list[str]) -> dict[str, int]:
    """Per-file NLOC from the plain report's file table (6 fixed columns)."""
    wanted = set(files)
    nloc = {}
    for line in run_lizard(files).splitlines():
        parts = line.split()
        if len(parts) == 6 and parts[5] in wanted and parts[0].isdigit():
            nloc[parts[5]] = int(parts[0])
    missing = wanted - set(nloc)
    if missing:
        sys.exit(f"ERROR: lizard reported no NLOC for {sorted(missing)}")
    return nloc


# ---- surface facts -----------------------------------------------------------


def count_leaves(node) -> int:
    if isinstance(node, dict):
        return sum(count_leaves(v) for v in node.values())
    return 1


def parameter_count() -> int:
    """The count the parameters reference publishes, via its own merge logic.

    render_params.sections_from_src() folds the hand-declared optional knobs
    into the extracted ones, so this stays identical to the page's "N knobs"
    and cannot drift from it."""
    sys.path.insert(0, str(REPO / "scripts"))
    import render_params

    sections, _ = render_params.sections_from_src()
    return sum(len(rows) for rows in sections.values())


def dispatch_factors() -> dict[str, int]:
    """Count the three registry typelists textually (balanced-angle scan).

    FRAGILE by design: a rewrite of typelists.hpp away from
    `using X = TypeList<...>;` breaks this parse, which then fails loudly
    below instead of miscounting."""
    text = (REPO / "include" / "bonsai" / "registry" / "typelists.hpp").read_text()
    factors = {}
    for name in ("Objectives", "Growers", "Samplers"):
        marker = f"using {name} ="
        start = text.index(marker) + len(marker)
        start = text.index("TypeList<", start) + len("TypeList<")
        depth, count = 1, 1
        for ch in text[start:]:
            if ch == "<":
                depth += 1
            elif ch == ">":
                depth -= 1
                if depth == 0:
                    break
            elif ch == "," and depth == 1:
                count += 1
        factors[name.lower()] = count
    if any(n < 1 for n in factors.values()):
        sys.exit(f"ERROR: typelist parse degenerated: {factors}")
    return factors


def python_public_api() -> int:
    """Import the built package (PYTHONPATH=build/python after `make python`)
    and count its declared public names: __all__ minus modules, each
    getattr-verified so a stale __all__ fails here."""
    code = ("import types, bonsai\n"
            "names = [n for n in bonsai.__all__\n"
            "         if not isinstance(getattr(bonsai, n), types.ModuleType)]\n"
            "print(len(names))\n")
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                          capture_output=True, text=True,
                          env={"PYTHONPATH": str(REPO / "build" / "python"),
                               "PATH": "/usr/bin:/bin"})
    if proc.returncode != 0:
        sys.exit("ERROR: cannot import bonsai from build/python; run "
                 f"`make python` first.\n{proc.stderr}")
    return int(proc.stdout)


def runtime_deps() -> dict:
    """The dependency rule: Python runtime deps are pyproject's [project]
    dependencies; C++ third-party code is compiled in from CMake's
    FetchContent pins (header-only), so shipped artifacts link only the
    toolchain runtime (OpenMP, static cudart on CUDA builds). Catch2 sits
    inside the BONSAI_TESTS block and is excluded as test-only."""
    import tomllib
    py = tomllib.loads((REPO / "pyproject.toml").read_text())
    py_deps = [d.split(">")[0].split("=")[0].split("<")[0].strip()
               for d in py["project"]["dependencies"]]
    cpp_deps, in_tests = [], 0
    for line in (REPO / "CMakeLists.txt").read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("if(BONSAI_TESTS)"):
            in_tests += 1
        elif stripped.startswith("endif") and in_tests:
            in_tests -= 1
        elif stripped.startswith("FetchContent_Declare(") and not in_tests:
            cpp_deps.append(stripped.removeprefix("FetchContent_Declare(").strip())
    if not cpp_deps or "Catch2" in cpp_deps:
        sys.exit(f"ERROR: FetchContent parse degenerated: {cpp_deps}")
    return {"python_runtime_deps": len(py_deps),
            "python_runtime_dep_names": sorted(py_deps),
            "cpp_compiled_deps": len(cpp_deps),
            "cpp_compiled_dep_names": sorted(cpp_deps)}


# ---- assembly ----------------------------------------------------------------


def main() -> int:
    version = run_lizard([], "--version").strip()
    planes = plane_files()
    all_files = sorted(f for fs in planes.values() for f in fs)
    functions = lizard_functions(all_files)
    nloc = lizard_file_nloc(all_files)

    rows: list[dict] = [{
        "kind": "meta", "schema": "code-metrics-v1", "division": "code",
        "git_sha": git("rev-parse", "HEAD"),
        "date": git("show", "-s", "--format=%cs", "HEAD"),
        "tool": "lizard", "tool_version": version,
        "tool_pin": f"uvx {LIZARD_PIN}",
        "plane_map": PLANE_MAP,
        "notes": {
            "date": "HEAD commit date, so the file is a pure function of the tree",
            "loc": "wc -l over each plane's tracked .hpp/.cpp/.cu/.cuh/.py files",
            "dispatch": "textual typelist count; see dispatch_factors()",
            "deps": "rule documented in runtime_deps()",
        },
    }]

    for name in PLANE_MAP:
        files = planes[name]
        fns = [f for f in functions if f["file"] in set(files)]
        loc = sum((REPO / f).read_bytes().count(b"\n") for f in files)
        rows.append({
            "kind": "plane", "plane": name, "files": len(files), "loc": loc,
            "nloc": sum(nloc[f] for f in files), "functions": len(fns),
            "ccn_mean": round(sum(f["ccn"] for f in fns) / len(fns), 2),
            "ccn_max": max(f["ccn"] for f in fns),
        })

    core = {f for n in ("core_headers", "engine_impl") for f in planes[n]}
    offenders = sorted((f for f in functions if f["file"] in core),
                       key=lambda f: (-f["ccn"], f["file"], f["start"]))
    for rank, f in enumerate(offenders[:5], 1):
        rows.append({"kind": "offender", "rank": rank, "function": f["name"],
                     "file": f["file"], "ccn": f["ccn"], "nloc": f["nloc"]})

    factors = dispatch_factors()
    combos = 1
    for n in factors.values():
        combos *= n
    rows.append({
        "kind": "surface", "parameters": parameter_count(),
        "dispatch_combinations": combos, "dispatch_factors": factors,
        "python_public_api": python_public_api(), **runtime_deps(),
    })

    OUT.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(f"wrote {OUT.relative_to(REPO)} ({len(rows)} rows, "
          f"{len(all_files)} files, lizard {version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
