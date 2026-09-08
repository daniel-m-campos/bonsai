#!/usr/bin/env python3
"""Standings freshness gates (results-lifecycle policy, decision 92).

Two hard-fail chokepoints plus the refresh planner, stdlib only:

    python3 scripts/check_standings.py --decisions
        Claim-time gate (runs in make docs-check). A decisions-log entry that
        claims a perf change on an axis carries a `Standings: <axis>[, ...]`
        line; the gate fails while any tagged entry is newer than the axis's
        registered as_of_decision, forcing refresh-or-demote before the claim
        ships, and while a registered release A/B holds a moved cell that
        no tagged entry cites.

    python3 scripts/check_standings.py --release <version>
        Release-time gate (runs in the wheels publish job). Fails unless every
        registry entry was refreshed for exactly this release (and has a sha),
        bounding worst-case staleness at one release even for untagged
        changes, with one mechanical exception: the hash-unchanged skip below.

    python3 scripts/check_standings.py --stale
        The refresh driver's planner: one axis name per line, for the axes a
        refresh must actually measure. Everything else is provably current.

The hash-unchanged skip. An axis may pass the release gate without a fresh
refresh when its committed `hash_set` digest still matches its plane's current
digest AND none of its `refs` reference libraries has bumped a major version
since. Quality axes lean on decision 40's contract (CPU-only builds stay
bit-identical, so unmoved model-hash inputs cannot move a quality standing).
Perf axes lean on the plane split: a CUDA-only change cannot move a CPU-plane
wall clock, so the cpu axes stay current across it and the refresh skips them,
which is the whole point of registry v2's `plane` field.

Carried-forward stamps. An entry's `carried_forward` block says its `hash_set`
was moved to the current digest by an equivalence argument rather than a new
measurement (`update_standings.py --restamp-verified`). Such an entry clears
the gates like any other skip, so every report that clears it labels it
carried-forward rather than a run.

The release A/B. Every refresh fits three arms at each anchor cell, same pod,
interleaved: `anchor` (the ANCHOR_VERSION wheel, one fixed release every
refresh fits again), `old` (the previous release wheel) and `new` (HEAD).
The statistic is the min over reps, because noise on a fixed workload only
adds time. `new` vs `old` is read inside AB_BAND_PCT and `new` vs `anchor`
inside ANCHOR_BAND_PCT; the anchor is what makes the comparison cumulative,
since a 1.5% loss that hides inside the release band every release compounds
to 35% over twenty of them and shows against the anchor by the second. The
math lives here, the lowest module, so the driver, the renderer and the gate
read one answer.

The quality drift. A `quality-*-gpu` axis sweeps the same suite as its
`quality-*` partner with bonsai's CUDA growers alone, so every device row has
one CPU row to be read against: same suite, dataset, seed, and growing
strategy. The device plane's fit is not the host's (fixed-point cells, its
own tie order), so the metrics differ in the last places, and by how much
depends on the task: the scores are read on held-out rows of tasks as small
as 2043 training rows, where a different split at one near-tie cut moves the
score by more than 1e-3, the same thing that moves it between seeds. The
gate therefore reads each pair against its task: the gap holds inside the
host's seed-to-seed spread of that (suite, dataset, strategy), plus
QUALITY_DRIFT_FLOOR for the single-seed case, and the quality page tables
the mean, the worst pair and the spread it sits against. A gap past its
task's spread is a device-plane bug until shown otherwise.

Reference-library majors are compared against the installed package in this
environment when available (`importlib.metadata`, no import needed); most
release-gate runs have no bench extras installed, so this falls back to
`benchmarks/reference_versions.json`, a hand-maintained ledger of the latest
known major per library. The bench extras are unpinned in `pyproject.toml`,
so there is no lockfile to read statically instead.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata as metadata
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = REPO / "benchmarks" / "standings.json"
RESULTS = REPO / "benchmarks" / "results"
DECISIONS = REPO / "docs" / "decisions.md"
REF_VERSIONS = REPO / "benchmarks" / "reference_versions.json"
MODEL_HASH_SCRIPT = REPO / "scripts" / "model_hash.py"

ENTRY_RE = re.compile(r"^## (\d+)\. ", re.M)
TAG_RE = re.compile(r"^Standings: (.+)$", re.M)

# Axes the scenario redesign retired (decision 103). Frozen decisions-log
# entries still carry `Standings:` tags naming them and the record is never
# rewritten, so the claim-time gate accepts the name and has nothing left to
# compare it against.
RETIRED_AXES = frozenset({"rows", "width", "shape", "frontier", "airline"})

PLANE_CPU = "cpu"
PLANE_GPU = "gpu"
# The device plane's sources, by repo-relative prefix: everything under them
# can only move a device timing, so a cpu-plane axis is current across a
# change confined here.
CUDA_PREFIXES = ("src/cuda/", "include/bonsai/cuda/")

AB_BAND_PCT = 5
ANCHOR_BAND_PCT = 2
# Re-pinned only by the PR that carries the `Standings:` decision entry
# explaining the cumulative move it would otherwise hide.
ANCHOR_VERSION = "1.15.0"
AB_ARMS = ("anchor", "old", "new")
AB_FIT = "fit_s"
AB_RSS = "peak_rss_gb"
AB_TABLE_HEADER = (
    "| cell | grower | anchor | old | new | vs old | vs anchor "
    "| old RSS | new RSS | RSS delta |",
    "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|",
)
AB_NA = "n/a"

# Metric units (r2 or AUC): the host-vs-device prediction tolerance, added
# to a task's host seed spread so a single-seed task still has an allowance.
QUALITY_DRIFT_FLOOR = 1e-4
QUALITY_GPU_SUFFIX = "-gpu"
# The CPU spelling each device grower is read against: the short names are
# the ones the CPU suite's committed rows carry.
QUALITY_PARTNERS = {"bonsai_cuda_depthwise": "bonsai_dw",
                    "bonsai_cuda_leafwise": "bonsai_lw",
                    "bonsai_cuda_levelwise": "bonsai_obl"}
QUALITY_KEY = ("suite", "dataset", "seed")


@dataclasses.dataclass(frozen=True)
class Drift:
    """One device grower read against its CPU partner over the paired rows.

    ``worst`` is the pair furthest past (or nearest to) its allowance, the
    host seed spread of its task plus QUALITY_DRIFT_FLOOR, not the largest
    gap: a large gap on a task whose seeds spread wider is inside the noise.
    """

    grower: str
    pairs: int
    mean: float
    worst: float
    worst_spread: float
    worst_task: str

    @property
    def held(self) -> bool:
        return self.worst <= self.worst_spread + QUALITY_DRIFT_FLOOR


def load_registry() -> dict:
    reg = json.loads(REGISTRY.read_text())
    reg.pop("_", None)
    return reg


def check_decisions(reg: dict) -> list[str]:
    errors = []
    for n, axes, _ in tagged_entries(DECISIONS.read_text()):
        for axis in axes:
            if axis in RETIRED_AXES:
                continue
            if axis not in reg:
                errors.append(f"decision {n}: unknown standings axis "
                              f"{axis!r} (known: {sorted(reg)})")
            elif n > reg[axis]["as_of_decision"]:
                errors.append(
                    f"decision {n} supersedes the {axis!r} standings "
                    f"(as_of_decision {reg[axis]['as_of_decision']}): "
                    "refresh the axis and bump the registry, or drop the "
                    "tag if the claim does not move the standings")
    return errors


def check_ab(reg: dict) -> list[str]:
    """A moved release A/B needs an entry tagged with its axis citing it."""
    cited = tagged_bodies(DECISIONS.read_text())
    errors = []
    for axis, e in reg.items():
        name = e.get("ab")
        if not name or not (RESULTS / name).exists():
            continue
        moves = ab_moves(ab_rows(RESULTS / name))
        if not moves or name in cited.get(axis, ""):
            continue
        errors.append(
            f"{axis}: the release A/B {name} moved ({'; '.join(moves)}); "
            f"a decision entry tagged `Standings: {axis}` must cite {name} "
            "and say why the move ships, or the axis is re-measured")
    return errors


def check_drift(reg: dict) -> list[str]:
    """Every device quality axis holds its CPU partner inside the host spread."""
    errors = []
    for gpu_axis, cpu_axis in drift_pairs(reg):
        for d in quality_drift(quality_rows(RESULTS / reg[cpu_axis]["file"]),
                               quality_rows(RESULTS / reg[gpu_axis]["file"])):
            if d.held:
                continue
            errors.append(
                f"{gpu_axis}: {d.grower} drifts {d.worst:.2e} from "
                f"{QUALITY_PARTNERS[d.grower]} on {d.worst_task}, past that "
                f"task's host seed spread {d.worst_spread:.2e} by more than "
                f"{QUALITY_DRIFT_FLOOR:.0e} ({d.pairs} pairs); the device "
                "plane moved a quality standing, so it is a bug or a decision")
    return errors


def drift_pairs(reg: dict) -> list[tuple[str, str]]:
    """(device axis, CPU partner) for every pair the tree holds rows of."""
    pairs = []
    for axis, e in reg.items():
        if not axis.endswith(QUALITY_GPU_SUFFIX):
            continue
        partner = reg.get(axis[:-len(QUALITY_GPU_SUFFIX)], {})
        if not (e.get("file") and partner.get("file")):
            continue
        pairs.append((axis, axis[:-len(QUALITY_GPU_SUFFIX)]))
    return pairs


def quality_rows(path: pathlib.Path) -> list[dict]:
    """The scored rows of one quality file."""
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    return [r for r in rows
            if r.get("status") == "ok" and quality_value(r) is not None]


def quality_value(row: dict) -> float | None:
    """Metric value across schema generations (v1 `value`, pre-schema `metric`)."""
    v = row.get("value")
    if isinstance(v, (int, float)):
        return v
    m = row.get("metric")
    return m if isinstance(m, (int, float)) else None


def quality_drift(cpu_rows: list[dict], gpu_rows: list[dict]) -> list[Drift]:
    """One Drift per device grower, over the (suite, dataset, seed) it pairs on."""
    cpu = {(r["variant"], *(r[k] for k in QUALITY_KEY)): quality_value(r)
           for r in cpu_rows}
    spread = host_seed_spread(cpu_rows)
    gaps: dict[str, list[tuple[float, float, str]]] = {
        g: [] for g in QUALITY_PARTNERS}
    for r in gpu_rows:
        partner = QUALITY_PARTNERS.get(r["variant"])
        base = cpu.get((partner, *(r[k] for k in QUALITY_KEY)))
        if base is None:
            continue
        gaps[r["variant"]].append(
            (quality_value(r) - base, spread[(partner, r["suite"], r["dataset"])],
             f"{r['dataset']} s{r['seed']}"))
    drifts = []
    for grower, deltas in gaps.items():
        if not deltas:
            continue
        worst, worst_spread, task = max(deltas, key=lambda d: abs(d[0]) - d[1])
        drifts.append(Drift(grower, len(deltas),
                            sum(d for d, _, _ in deltas) / len(deltas),
                            abs(worst), worst_spread, task))
    return drifts


def host_seed_spread(cpu_rows: list[dict]) -> dict[tuple[str, int, str], float]:
    """Max minus min of the metric over seeds, per (variant, suite, dataset)."""
    values: dict[tuple[str, int, str], list[float]] = {}
    for r in cpu_rows:
        values.setdefault((r["variant"], r["suite"], r["dataset"]), []).append(
            quality_value(r))
    return {k: max(v) - min(v) for k, v in values.items()}


def tagged_entries(text: str) -> list[tuple[int, list[str], str]]:
    """(number, tagged axes, body) of every decision entry with a tag."""
    entries = list(ENTRY_RE.finditer(text))
    tagged = []
    for i, m in enumerate(entries):
        end = entries[i + 1].start() if i + 1 < len(entries) else len(text)
        body = text[m.end():end]
        axes = [a.strip() for tag in TAG_RE.finditer(body)
                for a in tag.group(1).split(",")]
        if axes:
            tagged.append((int(m.group(1)), axes, body))
    return tagged


def tagged_bodies(text: str) -> dict[str, str]:
    """The joined bodies of the entries tagging each axis, by axis."""
    bodies: dict[str, list[str]] = {}
    for _, axes, body in tagged_entries(text):
        for axis in axes:
            bodies.setdefault(axis, []).append(body)
    return {axis: "\n".join(b) for axis, b in bodies.items()}


def ab_rows(path: pathlib.Path) -> list[dict]:
    """The measured rows of one A/B file."""
    return measured([json.loads(ln) for ln in path.read_text().splitlines()
                     if ln.strip()])


def measured(rows: list[dict]) -> list[dict]:
    """The rows that carry a cell; a skipped arm carries none."""
    return [r for r in rows if not r.get("skipped")]


def ab_best(rows: list[dict]) -> dict[tuple, float]:
    """Min per (rows, cols, grower, arm, metric) over the A/B rows."""
    best: dict[tuple, float] = {}
    for r in rows:
        for metric in (AB_FIT, AB_RSS):
            if r.get(metric) is None:
                continue
            key = (r["rows"], r["cols"], r["grower"], r["arm"], metric)
            best[key] = min(best.get(key, float("inf")), r[metric])
    return best


def ab_cells(rows: list[dict]) -> list[tuple]:
    return sorted({(r["rows"], r["cols"], r["grower"]) for r in rows})


def ab_versions(rows: list[dict]) -> dict[str, str]:
    """The wheel or build each arm actually fitted, by arm."""
    return {r["arm"]: r["version"] for r in rows if "version" in r}


def ab_arm_line(rows: list[dict]) -> str:
    """`anchor 1.15.0, old 2.0.0, new 2.1.0+source`, the arms that ran."""
    versions = ab_versions(rows)
    return ", ".join(f"{arm} {versions[arm]}" for arm in AB_ARMS
                     if arm in versions)


def pct_delta(base: float, new: float) -> float:
    return 100 * (new - base) / base


def ab_moves(rows: list[dict]) -> list[str]:
    """One line per (cell, comparison) outside its band; empty means held."""
    best = ab_best(rows)
    moves = []
    for rw, c, g in ab_cells(rows):
        for arm, metric, band, what in _AB_COMPARISONS:
            base = best.get((rw, c, g, arm, metric))
            new = best.get((rw, c, g, "new", metric))
            if base is None or new is None:
                continue
            d = pct_delta(base, new)
            if abs(d) > band:
                moves.append(f"{rw}x{c} {g}: {what} {d:+.1f}% (band {band}%)")
    return moves


def ab_table(rows: list[dict]) -> str:
    """The verdict table: one row per cell, mins per arm, deltas flagged."""
    best = ab_best(rows)
    lines = list(AB_TABLE_HEADER)
    for rw, c, g in ab_cells(rows):
        fit = {arm: best.get((rw, c, g, arm, AB_FIT)) for arm in AB_ARMS}
        rss = {arm: best.get((rw, c, g, arm, AB_RSS)) for arm in ("old", "new")}
        cells = [
            f"{rw}x{c}", g,
            *(_fmt_value(fit[arm], "s") for arm in AB_ARMS),
            _fmt_delta(fit["old"], fit["new"], AB_BAND_PCT),
            _fmt_delta(fit["anchor"], fit["new"], ANCHOR_BAND_PCT),
            _fmt_value(rss["old"], "GB"), _fmt_value(rss["new"], "GB"),
            _fmt_delta(rss["old"], rss["new"], AB_BAND_PCT),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


_AB_COMPARISONS = (
    ("old", AB_FIT, AB_BAND_PCT, "fit vs the previous release"),
    ("anchor", AB_FIT, ANCHOR_BAND_PCT, f"fit vs the {ANCHOR_VERSION} anchor"),
    ("old", AB_RSS, AB_BAND_PCT, "RSS vs the previous release"),
)


def _fmt_value(v: float | None, unit: str) -> str:
    return AB_NA if v is None else f"{v:.2f}{unit}"


def _fmt_delta(base: float | None, new: float | None, band: float) -> str:
    if base is None or new is None:
        return AB_NA
    d = pct_delta(base, new)
    flag = " **moved**" if abs(d) > band else ""
    return f"{d:+.1f}%{flag}"


def check_release(reg: dict, version: str) -> list[str]:
    errors = []
    for axis, e in reg.items():
        if not e.get("sha"):
            errors.append(f"{axis}: never refreshed (no measurement sha); "
                          "run the standings refresh before releasing")
            continue
        if e.get("refreshed_for") == version and not e.get("carried_forward"):
            print(f"{axis}: fresh refresh for {version!r}")
            continue
        ok, reason = hash_skip(axis, e)
        if ok:
            label = ("carried-forward stamp" if e.get("carried_forward")
                     else "hash-unchanged skip")
            print(f"{axis}: {label} ({reason})")
        else:
            errors.append(
                f"{axis}: refreshed for {e.get('refreshed_for')!r}, not "
                f"{version!r}, and the hash-unchanged skip does not apply "
                f"({reason}); run the standings refresh before releasing")
    return errors


def stale_axes(reg: dict) -> list[str]:
    """Axes a refresh must measure: never refreshed, or their plane moved.

    Parameters
    ----------
    reg : dict
        The loaded registry.

    Returns
    -------
    list[str]
        Registry keys in registry order.
    """
    return [axis for axis, e in reg.items()
            if not e.get("sha") or not hash_skip(axis, e)[0]]


def is_quality_axis(axis: str) -> bool:
    """Whether a registry key names a quality axis, by naming convention."""
    return axis.startswith("quality")


def plane_digest(plane: str) -> str:
    """sha256 over the sources that can move one plane's measured numbers.

    The gpu plane is the whole C++ implementation (`src/`, `include/`) plus
    the fixed-model harness `scripts/model_hash.py` it drives: that set is
    what the cross-arch CI hash jobs build and run, and any of it can move a
    device timing. The cpu plane is the same set minus the CUDA subtrees.
    `update_standings.py` computes this the same way when stamping an axis's
    `hash_set`, so the two are directly comparable.

    Parameters
    ----------
    plane : str
        `PLANE_CPU` or `PLANE_GPU`.

    Returns
    -------
    str
        A 16-character hex digest.
    """
    digest = hashlib.sha256()
    for path in _plane_paths(plane):
        digest.update(path.relative_to(REPO).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def hash_skip(axis: str, entry: dict) -> tuple[bool, str]:
    """Whether an axis may skip refresh, and why.

    Parameters
    ----------
    axis : str
        The registry key, which decides the plane when the entry names none.
    entry : dict
        The axis's registry entry; needs `hash_set` and `refs`.

    Returns
    -------
    tuple[bool, str]
        `(True, reason)` when the skip applies, `(False, reason)` otherwise;
        `reason` is always a human-readable audit line.
    """
    plane = entry.get("plane") or (PLANE_GPU if is_quality_axis(axis) else None)
    if plane is None:
        return False, "no plane registered, so nothing proves the axis current"
    current = plane_digest(plane)
    if entry.get("hash_set") != current:
        return False, f"{plane}-plane sources changed since this axis's refresh"
    bumped = bumped_ref_majors(entry.get("refs", {}))
    if bumped:
        return False, f"reference-library major bumped: {', '.join(bumped)}"
    carried = entry.get("carried_forward")
    if carried:
        kind = carried.get("evidence", {}).get("kind", "unrecorded")
        return True, (f"{plane} hash set {current} carried forward at "
                      f"{carried.get('stamped_at')} on {kind} evidence, still "
                      f"measured at {carried.get('measured_at')}: "
                      f"{carried.get('reason')}")
    return True, f"{plane} hash set {current} unchanged, refs current"


def bumped_ref_majors(refs: dict) -> list[str]:
    """Reference libraries whose current major version outruns `refs`.

    Parameters
    ----------
    refs : dict
        Library name to the version measured against, e.g.
        `{"xgboost": "3.2.0"}`.

    Returns
    -------
    list[str]
        Sorted names of libraries that have bumped a major version since.
    """
    recorded = recorded_ref_versions()
    bumped = []
    for name, measured in sorted(refs.items()):
        current = installed_ref_version(name) or recorded.get(name, measured)
        if _major(current) > _major(measured):
            bumped.append(name)
    return bumped


def installed_ref_version(name: str) -> str | None:
    """The installed version of a reference library, via package metadata.

    Reads distribution metadata without importing the package, so this stays
    cheap even when the heavy bench extras are not installed.
    """
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def recorded_ref_versions() -> dict:
    """The hand-maintained `reference_versions.json` ledger, sans its doc key."""
    versions = json.loads(REF_VERSIONS.read_text())
    versions.pop("_", None)
    return versions


def _plane_paths(plane: str) -> list[pathlib.Path]:
    """The digested files for one plane, in a stable order."""
    paths = sorted(p for p in (REPO / "src").rglob("*") if p.is_file())
    paths += sorted(p for p in (REPO / "include").rglob("*") if p.is_file())
    paths.append(MODEL_HASH_SCRIPT)
    if plane == PLANE_CPU:
        return [p for p in paths
                if not p.relative_to(REPO).as_posix().startswith(CUDA_PREFIXES)]
    return paths


def _major(version: str) -> int:
    return int(str(version).split(".")[0])


def main() -> int:
    reg = load_registry()
    if "--decisions" in sys.argv:
        errors = check_decisions(reg) + check_ab(reg) + check_drift(reg)
        label = "decision gate"
    elif "--release" in sys.argv:
        version = sys.argv[sys.argv.index("--release") + 1]
        errors = check_release(reg, version) + check_drift(reg)
        label = f"release gate ({version})"
    elif "--stale" in sys.argv:
        # One name per line and nothing else: the refresh driver reads this.
        for axis in stale_axes(reg):
            print(axis)
        for axis, e in reg.items():
            if not e.get("carried_forward"):
                continue
            ok, reason = hash_skip(axis, e)
            if ok:
                print(f"note: carried-forward stamp, not a run: {axis}: "
                      f"{reason}", file=sys.stderr)
        return 0
    else:
        print("usage: check_standings.py --decisions | --release <version> "
              "| --stale", file=sys.stderr)
        return 2
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"standings {label}: ok ({len(reg)} axes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
