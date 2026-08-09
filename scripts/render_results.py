"""Render docs/method/results.md (the results ledger) from every committed
data file under benchmarks/results/.

The ledger is the use-it-or-remove-it contract for results data: the script
discovers committed data files via `git ls-files` and FAILS if any file is
neither rendered below nor explicitly listed as plot output, so a new
committed file must be wired in (or not committed) and a removed file must
take its section with it.

    python3 scripts/render_results.py           # rewrite the page
    python3 scripts/render_results.py --check   # CI: fail on drift

Stdlib only (no pandas): the CI python job runs the check with no extra
deps. Numbers are formatted, never recomputed differently per run, so the
output is byte-stable for a given input set.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import statistics
import subprocess
import sys
from collections import defaultdict
from typing import Final

from _render_common import md_table


class Axis:
    """Standings axes: registry keys in benchmarks/standings.json."""

    GPU_TALL: Final = "gpu-tall"
    GPU_WIDE: Final = "gpu-wide"
    GPU_EXTREME: Final = "gpu-extreme"
    CPU_TALL: Final = "cpu-tall"
    CPU_WIDE: Final = "cpu-wide"
    GPU_EARLY_STOP: Final = "gpu-early-stop"
    GRINSZTAJN: Final = "quality-grinsztajn"
    CODE: Final = "code"


class K:
    """Row keys read from the results jsonl (values are the runlog schema)."""

    FIT_S: Final = "fit_s"
    INGEST_S: Final = "ingest_s"
    TRAIN_S: Final = "train_s"
    R2_TEST: Final = "r2_test"
    VARIANT: Final = "variant"
    STATUS: Final = "status"
    RUN: Final = "run"
    DATASET: Final = "dataset"
    PEAK_RSS_GB: Final = "peak_rss_gb"
    DEV_MEM: Final = "dev_mem"
    GIT_SHA: Final = "git_sha"


REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
OUT = REPO / "docs" / "method" / "results.md"
ASSETS = REPO / "docs" / "method" / "assets"

consumed: set[str] = set()
charts: dict[str, str] = {}  # filename -> svg, written to docs/method/assets/


def load_jsonl(name: str) -> list[dict]:
    """Rows of a results jsonl by bare filename."""
    consumed.add(name)
    text = (RESULTS / name).read_text()
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def fmt(v: object, nd: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if v != v:  # NaN
            return "-"
        return f"{v:.{nd}f}"
    return str(v)


_STANDINGS_REG = json.loads(
    (REPO / "benchmarks" / "standings.json").read_text())


def standings_file(axis: str) -> str:
    """Standings sections load by registry entry, not literal name, so a
    refresh can supersede the file (new dated name) without generator edits."""
    return _STANDINGS_REG[axis]["file"]


def measured_stamp(rows: list[dict]) -> str:
    """One-sha standings stamp computed from the rows themselves, so the
    reader always sees the vintage (results-lifecycle policy, decision 92)."""
    shas = sorted({r[K.GIT_SHA] for r in rows if r.get(K.GIT_SHA)})
    if len(shas) != 1:
        return ""
    dates = sorted({r["ts"][:10] for r in rows if r.get("ts")})
    hosts = sorted({(r["host"].get("name") if isinstance(r.get("host"), dict)
                     else r.get("host")) for r in rows if r.get("host")})
    when = f" ({dates[-1]}" if dates else " ("
    where = f", {hosts[0]})" if len(hosts) == 1 else ")"
    return f" Measured at `{shas[0]}`{when}{where}."


def provenance(files: list[str], note: str) -> str:
    """The source-attribution line every section ends with."""
    links = ", ".join(
        f"[`{f}`](../../benchmarks/results/{f})" for f in files)
    return f"*Source: {links}. {note}*"


# SVG charts =======================================================================================
# Hand-rolled SVG (stdlib only) so the images are byte-stable and the CI drift
# check covers them exactly like the tables. One color per library everywhere.
# Text and grid in mid-grays so the site's light and dark themes both read.

LIB_COLOR = {
    "bonsai": "#2e7d32",
    "xgboost": "#1f77b4",
    "lightgbm": "#8e6bbf",
    "catboost": "#e08f1a",
}
LIB_COLOR["xgb"] = LIB_COLOR["xgboost"]
LIB_COLOR["lgbm"] = LIB_COLOR["lightgbm"]
TEXT = "#8a8a8a"
FONT = "font-family='system-ui,sans-serif'"


def _svg(width: int, height: int, body: list[str]) -> str:
    """Wrap body fragments in an svg element."""
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' "
            f"height='{height}' viewBox='0 0 {width} {height}'>\n"
            + "\n".join(body) + "\n</svg>\n")


def _text(x: float, y: float, s: str, size: int = 11, anchor: str = "start",
          color: str = TEXT, weight: str = "normal") -> str:
    return (f"<text x='{x}' y='{y}' font-size='{size}' fill='{color}' "
            f"{FONT} text-anchor='{anchor}' font-weight='{weight}'>{s}</text>")


def bar_chart(fname: str, title: str, rows: list[tuple[str, float, str]],
              x_max: float, note: str):
    """Horizontal bars: rows = (label, value, annotation); lower is better."""
    w, h, left, top = 720, 60 + 34 * len(rows) + 30, 110, 42
    plot_w = w - left - 150
    body = [_text(left, 22, title, size=13, weight="bold")]
    for i, (label, value, note_txt) in enumerate(rows):
        y = top + i * 34
        bw = round(plot_w * value / x_max, 1)
        color = LIB_COLOR.get(label)
        if color is None:  # variant names: color by library prefix
            for lib in ("bonsai", "xgb", "lgbm", "catboost"):
                if label.startswith(lib):
                    color = LIB_COLOR[lib]
                    break
            color = color or TEXT
        body.append(_text(left - 8, y + 14, label, anchor="end"))
        body.append(f"<rect x='{left}' y='{y}' width='{bw}' height='20' "
                    f"rx='3' fill='{color}' fill-opacity='0.85'/>")
        body.append(_text(left + bw + 8, y + 14,
                          f"{value:.2f}&#160;&#160;{note_txt}"))
    body.append(_text(left, top + len(rows) * 34 + 18, note, size=10))
    charts[fname] = _svg(w, h, body)


# Quality: Grinsztajn standings ====================================================================


def _value(r: dict):
    """Metric value across schema generations (v1 `value`, pre-schema `metric`)."""
    v = r.get("value")
    if isinstance(v, (int, float)):
        return v
    m = r.get("metric")
    return m if isinstance(m, (int, float)) else None


def _standings(rows: list[dict]):
    """Replicates bonsai.bench.grinsztajn.report(): mean value over seeds per
    (suite, dataset, variant); best variant per library; average-rank ties;
    a win is rank exactly 1.0."""
    acc: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r.get(K.STATUS) != "ok":
            continue
        v = _value(r)
        if v is None:
            continue
        acc[(r["suite"], r[K.DATASET], r[K.VARIANT])].append(v)
    lib_best: dict[tuple, float] = {}
    for (suite, ds, variant), vals in acc.items():
        lib = "bonsai" if variant.startswith("bonsai") else variant
        key = (suite, ds, lib)
        mean = sum(vals) / len(vals)
        lib_best[key] = max(lib_best.get(key, float("-inf")), mean)
    by_ds: dict[tuple, list[tuple[str, float]]] = defaultdict(list)
    for (suite, ds, lib), v in lib_best.items():
        by_ds[(suite, ds)].append((lib, v))
    ranks: dict[str, list[float]] = defaultdict(list)
    suite_ranks: dict[tuple, list[float]] = defaultdict(list)
    wins: dict[str, int] = defaultdict(int)
    for (suite, _ds), entries in by_ds.items():
        order = sorted(entries, key=lambda e: -e[1])
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
                j += 1
            rank = (i + j) / 2 + 1  # average rank for ties, 1-indexed
            for k in range(i, j + 1):
                lib = order[k][0]
                ranks[lib].append(rank)
                suite_ranks[(suite, lib)].append(rank)
                if rank == 1.0:
                    wins[lib] += 1
            i = j + 1
    n_datasets = len(by_ds)
    table = sorted(
        ((lib, sum(rs) / len(rs), wins.get(lib, 0)) for lib, rs in ranks.items()),
        key=lambda t: t[1])
    return table, suite_ranks, n_datasets


def grinsztajn_section() -> str:
    """The Grinsztajn standings section of the quality page."""
    main = load_jsonl(standings_file(Axis.GRINSZTAJN))
    table, suite_ranks, n = _standings(main)
    rows = [[lib, fmt(mean, 2), str(w)] for lib, mean, w in table]
    campaign = md_table(["library", "mean rank", "outright wins"], rows)
    bar_chart(
        "grinsztajn-rank.svg",
        f"Grinsztajn suite: mean rank across {n} tasks (lower is better)",
        [(lib, mean, f"{w} outright wins") for lib, mean, w in table],
        x_max=4.0,
        note="55 OpenML tasks, 3 seeds, campaign knobs, best variant per library (decision 68)")

    suites = sorted({s for s, _ in suite_ranks})
    libs = [lib for lib, _, _ in table]
    per_suite = md_table(
        ["library", *suites],
        [[lib, *[fmt(sum(v) / len(v), 2) if (v := suite_ranks.get((s, lib))) else "-"
                 for s in suites]] for lib in libs])

    return f"""### External standings: the Grinsztajn suite

The [Grinsztajn et al. tabular benchmark](https://arxiv.org/abs/2207.08815) at the paper-medium protocol: {n} OpenML tasks, three seeds, campaign knobs for every library (decision 68). Best variant per library, average rank across tasks, lower is better.

![Grinsztajn mean rank by library](assets/grinsztajn-rank.svg)

{campaign}

Per-suite mean rank:

{per_suite}

XGBoost's campaign mapping sets `min_child_weight=20` (hessian-weighted, the knob-translation bracket recorded in decision 68). The other end of that bracket, the same suite with XGBoost at `min_child_weight=1`, was measured: bonsai keeps the lead at mean rank 1.78 and XGBoost rises to second. Those rows are closed evidence now, listed in the archive.

Reproduce: `pip install bonsai-gbt[bench]`, then `python -m bonsai.bench.grinsztajn out.jsonl` to run the suite (hours; datasets fetch from OpenML), then `--report` on the same file to render the standings from the jsonl.

{provenance([standings_file(Axis.GRINSZTAJN)], "As-run; evidence narrative in [benchmarks/grinsztajn-2026-07.md](../../benchmarks/grinsztajn-2026-07.md), ruling in decision 68.")}
"""


# Perf: the scenario panels ========================================================================
# One page, one panel per grower, identical column sets: the reader compares a
# grower against its closest rival on one plane, then jumps plane or grower
# without re-orienting. Every section renders empty-but-valid while its axis is
# unmeasured, so the page is complete the moment the first refresh lands.

PLANE_GPU: Final = "gpu"
PLANE_CPU: Final = "cpu"

GPU_AXES: Final = (Axis.GPU_TALL, Axis.GPU_WIDE, Axis.GPU_EXTREME)
CPU_AXES: Final = (Axis.CPU_TALL, Axis.CPU_WIDE)
PERF_AXES: Final = (*GPU_AXES, *CPU_AXES, Axis.GPU_EARLY_STOP)

PENDING: Final = ("*Measurement pending the first redesigned refresh "
                  "(decision 103): the axis files land with it and this "
                  "section fills in unchanged.*")


class Col:
    """Panel table columns; the same set across every panel and both planes."""

    INGEST: Final = "ingest_s"
    TRAIN: Final = "train_s"
    RSS: Final = "peak RSS"
    VRAM: Final = "peak VRAM"
    METRIC: Final = "test r2"


GPU_COLUMNS: Final = (Col.INGEST, Col.TRAIN, Col.RSS, Col.VRAM, Col.METRIC)
CPU_COLUMNS: Final = (Col.INGEST, Col.TRAIN, Col.RSS, Col.METRIC)
HIGHER_IS_BETTER: Final = frozenset({Col.METRIC})

METRIC_COLUMNS: Final = frozenset({Col.METRIC})

# A margin the protocol cannot resolve is a tie, not a win, and the two kinds
# of column are unresolvable for different reasons. Time and memory wander
# between repeats on a rented host: where a scenario repeats an arm, that
# arm's own spread is the measured resolution, and where a cell ran once
# TIE_MARGIN stands in for it. Accuracy does not wander that way, so a metric
# column takes the quality work's chance band as an absolute difference
# instead of a relative one.
TIE_MARGIN: Final = 0.05
CHANCE_BAND: Final = 0.001
TIE_NOTE: Final = "(tie)"


@dataclasses.dataclass(frozen=True)
class Panel:
    """One grower and the reference library it is measured against."""

    grower: str
    rival: str
    gpu_arms: tuple[str, str]
    cpu_arms: tuple[str, str]

    def arms(self, plane: str) -> tuple[str, str]:
        """(bonsai variant, rival variant) on one plane."""
        return self.gpu_arms if plane == PLANE_GPU else self.cpu_arms

    def labels(self) -> tuple[str, str]:
        """The two arm labels, in the same order as `arms`."""
        return (f"bonsai {self.grower}", self.rival)


PANELS: Final = (
    Panel("depthwise", "XGBoost",
          ("bonsai_cuda_depthwise", "xgb_cuda"),
          ("bonsai_depthwise", "xgb_hist")),
    Panel("leafwise", "LightGBM",
          ("bonsai_cuda_leafwise", "lgbm_cuda"),
          ("bonsai_leafwise", "lgbm_cpu")),
    Panel("levelwise", "CatBoost",
          ("bonsai_cuda_levelwise", "catboost_gpu"),
          ("bonsai_levelwise", "catboost_cpu")),
)

# The ingest/train split (issue #301): committed rows carry both for every
# arm, bonsai included since the two-step runner refresh.
BONSAI_SPLIT_NOTE = (
    "bonsai's split comes from the two-step `Dataset(..., device=...)` plus "
    "`train(pairs, ds)` form, which for a cuda arm bins on the device "
    "exactly where the fused call does; every refresh fits the anchor cell "
    "both ways, interleaved on the same pod, and the supersession is gated "
    "on their agreement, so the seam these columns report belongs to the "
    "same pipeline the total measures.")
CATBOOST_INGEST_NOTE = (
    "CatBoost's `Pool()` step only wraps the raw arrays; it quantizes "
    "inside `fit`, so its ingest column reads low and that cost sits in "
    "train instead (issue #253). Its total is the only number directly "
    "comparable to the other libraries' split.")


def human(n: int) -> str:
    """250000 -> 250k, 16000000 -> 16M."""
    return f"{n // 1_000_000}M" if n >= 1_000_000 else f"{n // 1000}k"


def cell_label(rows: int, cols: int) -> str:
    """A rows x cols cell label with human row counts."""
    return f"{human(rows)} x {cols}"


def synthetic_input_gib(cell: dict) -> float:
    """The host bytes `bonsai.bench.synth.gen_data` holds resident for one
    synthetic cell, in GiB (matching `peak_rss_gb`'s own unit: Linux
    `ru_maxrss` is KiB, divided by 2**20).

    Train and test are numpy views into one `(rows + n_test) x cols` float32
    buffer (`X[:rows]`, `X[rows:]`), so the held-out rows stay resident
    through fit whether or not the runner still references them by name;
    only the size of that one shared buffer determines what can be freed.
    """
    return (cell["rows"] + cell["n_test"]) * cell["cols"] * 4 / 2**30


def rss_headroom_str(r: dict) -> str:
    """`6.9GB (+0.8)`: peak host RSS beside its headroom above the shared
    synthetic input array for that row's cell (see `synthetic_input_gib`)."""
    rss = r[K.PEAK_RSS_GB]
    headroom = rss - synthetic_input_gib(r["cell"])
    return f"{rss:.1f}GB ({headroom:+.1f})"


def axis_rows(axis: str) -> list[dict]:
    """Rows of a perf axis, or an empty list while the axis is unmeasured.

    Parameters
    ----------
    axis : str
        A registry key from `benchmarks/standings.json`.

    Returns
    -------
    list[dict]
        The measured rows, or `[]` when the registry entry carries no file
        yet (registry v2's placeholder for a never-measured axis).
    """
    name = _STANDINGS_REG[axis].get("file")
    return load_jsonl(name) if name else []


def best_row(rows: list[dict], variant: str) -> dict | None:
    """The fastest finished row for one arm, else a failed row for it.

    A failed row is a published result, not a gap: the extreme scenario runs
    every arm precisely so an OOM or an unsupported cell prints as itself.
    """
    ok = [r for r in rows if r[K.VARIANT] == variant and r[K.STATUS] == "ok"]
    if ok:
        return min(ok, key=lambda r: r[K.FIT_S])
    failed = [r for r in rows if r[K.VARIANT] == variant]
    return failed[0] if failed else None


def vram_gb(r: dict) -> float | None:
    """Per-process peak VRAM, or None when the sampler could not attribute it.

    `nvidia-smi` cannot see the child pid from inside a container, so a
    degraded row carries only a device total. Printing that total as a
    per-process number would credit bonsai with every other process on the
    card, so the column prints `-` instead.
    """
    dev = r.get(K.DEV_MEM) or {}
    if dev.get("source") != "nvml":
        return None
    return dev.get("peak_gb_pid")


def summary_matrix() -> str:
    """The headline: fit totals at the tall scenario, grower by plane."""
    body = [[panel.grower,
             _summary_cell(panel, Axis.GPU_TALL, PLANE_GPU),
             _summary_cell(panel, Axis.CPU_TALL, PLANE_CPU)]
            for panel in PANELS]
    if all(cell == "-" for row in body for cell in row[1:]):
        return PENDING
    return md_table(["grower", "GPU (gpu-tall)", "CPU (cpu-tall)"], body)


def panel_table(panel: Panel, axes: tuple[str, ...], plane: str) -> str:
    """One plane's scenarios for one grower, both arms per scenario row.

    Empty while none of the plane's axes is measured, which is what lets the
    page state the pending case once per panel instead of once per table.
    """
    columns = GPU_COLUMNS if plane == PLANE_GPU else CPU_COLUMNS
    body: list[list[str]] = []
    for axis in axes:
        rows = axis_rows(axis)
        if not rows:
            continue
        label = _scenario_label(axis, rows)
        variants = panel.arms(plane)
        cells = [_arm_cells(best_row(rows, v), columns) for v in variants]
        spreads = [_arm_spreads(rows, v, columns) for v in variants]
        for i, arm in enumerate(panel.labels()):
            j = 1 - i
            body.append([label, arm,
                         *[_compare(c, cells[i][c], cells[j][c],
                                    _tie_floor(spreads[i][c], spreads[j][c]))
                           for c in columns]])
    if not body:
        return ""
    return md_table(["scenario", "arm", *columns], body)


def fused_anchor_line() -> str:
    """The fused fit total at the tall cell, published once, from parity.

    The refresh already fits the anchor cell through bonsai's one-call form
    to gate the ingest/train split; this publishes that arm's median rather
    than measuring anything new. Empty until the companion file lands.
    """
    name = _STANDINGS_REG[Axis.GPU_TALL].get("companion")
    if not name or not (RESULTS / name).exists():
        return ""
    rows = [r for r in load_jsonl(name) if not r.get("skipped")]
    fused = [r[K.FIT_S] for r in rows
             if r.get("arm") == "fused" and r.get(K.FIT_S) is not None]
    split = [r[K.INGEST_S] + r[K.TRAIN_S] for r in rows
             if r.get("arm") == "two_step" and r.get(K.INGEST_S) is not None
             and r.get(K.TRAIN_S) is not None]
    if not fused or not split:
        return ""
    return (f"\nThe two columns are one call taken apart. bonsai's fused "
            f"`train(X, y)` form fits the same cell in "
            f"{statistics.median(fused):.1f}s against the split's "
            f"{statistics.median(split):.1f}s, medians of the refresh's "
            f"interleaved parity arm, which is what makes reporting the "
            f"seam honest.\n")


def early_stop_section() -> str:
    """Early stopping as its own axis: the eval overhead and the stop arm."""
    rows = axis_rows(Axis.GPU_EARLY_STOP)
    body: list[list[str]] = []
    for panel in PANELS:
        for variant, arm in zip(panel.arms(PLANE_GPU), panel.labels()):
            off = _mode_row(rows, variant, "off")
            evaluated = _mode_row(rows, variant, "eval")
            stopped = _mode_row(rows, variant, "stop")
            body.append([panel.grower, arm, _overhead(off, evaluated),
                         _fit_text(stopped),
                         fmt(stopped.get("stopped_at") if stopped else None)])
    table = md_table(["grower", "arm", "eval overhead", "stop-arm fit",
                      "stopped at"], body) if rows else PENDING
    return f"""## Early stopping

Three arms at the tall cell: `off` fits the round budget blind, `eval` scores a held-out split every round without acting on it, and `stop` acts on it (patience 50, cap 2000 rounds, learning rate 0.05). The overhead column is what watching costs against not watching, on train time; the last two columns are what watching buys, in fit time and the round the arm actually stopped at. A blank stop column means the arm ran to the cap without triggering.

{table}
"""


# Perf: private panel helpers ======================================================================


def _scenario_label(axis: str, rows: list[dict]) -> str:
    """`gpu-tall (16M x 128)`: the axis name and the cell it measured."""
    cell = rows[0]["cell"]
    return f"{axis} ({cell_label(cell['rows'], cell['cols'])})"


def _arm_cells(r: dict | None,
               columns: tuple[str, ...]) -> dict[str, tuple[str, float | None]]:
    """(cell text, comparable value) per column for one arm of one scenario.

    A row that did not finish carries its status word into every column: an
    OOM at the extreme cell is the measurement, so it prints where the
    numbers would have been rather than as a dash.
    """
    if r is None:
        return {c: ("-", None) for c in columns}
    if r.get(K.STATUS) != "ok":
        return {c: (str(r.get(K.STATUS)), None) for c in columns}
    vram = vram_gb(r)
    rss = r.get(K.PEAK_RSS_GB)
    values = {
        Col.INGEST: (_secs(r.get(K.INGEST_S)), r.get(K.INGEST_S)),
        Col.TRAIN: (_secs(r.get(K.TRAIN_S)), r.get(K.TRAIN_S)),
        Col.RSS: (rss_headroom_str(r) if rss is not None else "-", rss),
        Col.VRAM: ("-" if vram is None else f"{vram:.1f}GB", vram),
        Col.METRIC: (fmt(r.get(K.R2_TEST), 3), r.get(K.R2_TEST)),
    }
    return {c: values[c] for c in columns}


def _secs(v: float | None) -> str:
    """A seconds cell, or the house absent-value marker."""
    return "-" if v is None else f"{v:.1f}s"


def _wins(column: str, mine: float | None, theirs: float | None) -> bool:
    """Whether one arm holds the better value of a comparable pair."""
    if mine is None or theirs is None or mine == theirs:
        return False
    return mine > theirs if column in HIGHER_IS_BETTER else mine < theirs


def _gap(mine: float, theirs: float) -> float:
    """The relative margin between two comparable values, against the
    smaller of the two, so a margin and a spread read on the same scale."""
    smaller = min(abs(mine), abs(theirs))
    return abs(mine - theirs) / smaller if smaller else float("inf")


def _spread(values: list[float | None]) -> float | None:
    """One arm's observed relative spread across a scenario's repeats, or
    None when the cell ran once and no spread is observable."""
    seen = [v for v in values if v is not None]
    if len(seen) < 2:
        return None
    return _gap(max(seen), min(seen))


def _tie_floor(mine: float | None, theirs: float | None) -> float:
    """The margin a comparison must clear to count as a win: the wider of the
    two arms' observed spreads, or TIE_MARGIN when neither arm repeated. An
    arm whose repeats came out identical has a spread of zero, which is a
    resolution and not a missing one, so it never reaches the fallback."""
    seen = [s for s in (mine, theirs) if s is not None]
    return max(seen) if seen else TIE_MARGIN


def _is_tie(mine: float | None, theirs: float | None, floor: float) -> bool:
    """Whether a comparable pair sits inside the measurement's resolution."""
    if mine is None or theirs is None:
        return False
    return _gap(mine, theirs) <= floor


def _in_chance_band(mine: float | None, theirs: float | None) -> bool:
    """Whether an accuracy pair sits inside the chance band, the margin the
    quality work reads as threshold-placement luck rather than a difference."""
    if mine is None or theirs is None:
        return False
    return abs(mine - theirs) <= CHANCE_BAND


def _arm_spreads(rows: list[dict], variant: str,
                 columns: tuple[str, ...]) -> dict[str, float | None]:
    """One arm's observed spread per column, across the scenario's repeats."""
    finished = [_arm_cells(r, columns) for r in rows
                if r[K.VARIANT] == variant and r[K.STATUS] == "ok"]
    return {c: _spread([cells[c][1] for cells in finished]) for c in columns}


def _compare(column: str, mine: tuple[str, float | None],
             theirs: tuple[str, float | None], floor: float) -> str:
    """One arm's cell, marked against the other arm: bold when it holds the
    better value, noted as a tie when the margin sits inside the column's own
    resolution, `floor` for a measured cost and the chance band for accuracy.
    """
    text, value = mine
    other = theirs[1]
    tied = (_in_chance_band(value, other) if column in METRIC_COLUMNS
            else _is_tie(value, other, floor))
    if tied:
        return f"{text} {TIE_NOTE}"
    return _bold(text, _wins(column, value, other))


def _summary_cell(panel: Panel, axis: str, plane: str) -> str:
    """`bonsai 6.9s vs XGBoost 8.1s` for one grower on one plane."""
    rows = axis_rows(axis)
    if not rows:
        return "-"
    bonsai, rival = (best_row(rows, v) for v in panel.arms(plane))
    if bonsai is None or rival is None:
        return "-"
    ours, theirs = _fit_value(bonsai), _fit_value(rival)
    if _fit_is_tie(rows, panel, plane, ours, theirs):
        return (f"bonsai {_fit_text(bonsai)} vs "
                f"{panel.rival} {_fit_text(rival)} {TIE_NOTE}")
    return (f"{_bold('bonsai ' + _fit_text(bonsai), _wins('fit', ours, theirs))}"
            f" vs "
            f"{_bold(panel.rival + ' ' + _fit_text(rival), _wins('fit', theirs, ours))}")


def _fit_is_tie(rows: list[dict], panel: Panel, plane: str,
                ours: float | None, theirs: float | None) -> bool:
    """Whether a fit-total pair sits inside the two arms' own repeat spreads."""
    spreads = (_spread([_fit_value(r) for r in rows if r[K.VARIANT] == v])
               for v in panel.arms(plane))
    return _is_tie(ours, theirs, _tie_floor(*spreads))


def _fit_value(r: dict | None) -> float | None:
    """A row's fit total, or None when the row did not finish."""
    if r is None or r.get(K.STATUS) != "ok":
        return None
    return r.get(K.FIT_S)


def _fit_text(r: dict | None) -> str:
    """A row's fit total as text, or its status word."""
    if r is None:
        return "-"
    if r.get(K.STATUS) != "ok":
        return str(r.get(K.STATUS))
    return _secs(r.get(K.FIT_S))


def _mode_row(rows: list[dict], variant: str, mode: str) -> dict | None:
    """The best row of one arm under one `eval_mode`."""
    return best_row([r for r in rows if _eval_mode(r) == mode], variant)


def _eval_mode(r: dict) -> str | None:
    """A row's eval mode, from the row or the cell that produced it."""
    return r.get("eval_mode") or (r.get("cell") or {}).get("eval_mode")


def _overhead(off: dict | None, evaluated: dict | None) -> str:
    """What scoring every round costs against not scoring at all."""
    base, watched = _train_time(off), _train_time(evaluated)
    if not base or watched is None:
        return "-"
    return _pct(watched / base - 1)


def _train_time(r: dict | None) -> float | None:
    """The variable half of a row's time, falling back to the total."""
    if r is None or r.get(K.STATUS) != "ok":
        return None
    return r.get(K.TRAIN_S) if r.get(K.TRAIN_S) is not None else r.get(K.FIT_S)


def _bold(text: str, on: bool) -> str:
    """One table cell, bolded when it holds the best value in its column."""
    return f"**{text}**" if on else text


def _pct(x: float) -> str:
    """A signed percentage, the eval-overhead unit."""
    return f"{x * 100:+.1f}%"


# Perf: page sections ==============================================================================


def perf_summary_section() -> str:
    """The panels page opening: what the columns mean, then the headline."""
    return f"""Six scenarios, three growers, two planes, one page. Each grower is measured against the reference library closest to it: depthwise against XGBoost, leafwise against LightGBM, levelwise against CatBoost. The tall and wide scenarios hold the cell count constant (2^31 cells on the GPU plane, 2^28 on the CPU plane) so the pair separates shape from volume, and the extreme scenario sizes the input to most of the card's memory, where a row that runs out of it is published as such.

The columns are the same everywhere. `ingest_s` is the fixed cost of turning a float32 matrix into bins, paid once; `train_s` is the variable cost of the boosting rounds, the half that grows with the round budget. {BONSAI_SPLIT_NOTE} {CATBOOST_INGEST_NOTE} Peak RSS is host memory, with its headroom above the resident input array in parentheses; peak VRAM is device memory attributed to the training process by NVML, and a row whose sampler could not attribute it prints `-` rather than a device total that would count every other process on the card.

Bold marks the better value of a pair, but only where the measurement can tell the two arms apart. The tie rule governs the cost columns: `ingest_s`, `train_s`, the fit totals they add up to, peak RSS and peak VRAM. Those are host measurements that wander between repeats. Where a scenario repeats an arm, the spread across that arm's own repeats is what the host resolved on the day, and a margin inside the wider of the two arms' spreads prints as a tie. Where such a cell ran once and no spread is observable, a stated {TIE_MARGIN:.0%} margin stands in for it. Accuracy is read on its own scale, because a fit at a fixed seed repeats exactly and a {TIE_MARGIN:.0%} band on r2 would bury differences the suite can resolve. Two r2 values are a tie when they sit within {CHANCE_BAND} of each other, the chance band the quality work treats as threshold-placement luck. Both values of a tied pair print plain, with `{TIE_NOTE}` beside them.

## The headline: the tall scenario

Fit totals (ingest plus train) at the tall cell of each plane, bold to the faster arm, `{TIE_NOTE}` where the margin sits inside the measurement's own spread.

{summary_matrix()}
"""


def perf_panels_section() -> str:
    """One panel per grower: its GPU table, then its CPU table."""
    blocks = [_panel_block(panel) for panel in PANELS]
    return "\n".join(blocks) + _perf_provenance()


def _panel_block(panel: Panel) -> str:
    """One grower's two tables, with the fused anchor under the first."""
    head = f"## {panel.grower.capitalize()} against {panel.rival}"
    gpu = panel_table(panel, GPU_AXES, PLANE_GPU)
    cpu = panel_table(panel, CPU_AXES, PLANE_CPU)
    if not gpu and not cpu:
        return f"{head}\n\n{PENDING}\n"
    anchor = fused_anchor_line() if panel.grower == "depthwise" else ""
    return f"""{head}

### GPU

{gpu or PENDING}
{anchor}
### CPU

{cpu or PENDING}
"""


def _perf_provenance() -> str:
    """The source line for whichever perf axes have been measured."""
    files = [f for f in (_STANDINGS_REG[a].get("file") for a in PERF_AXES) if f]
    if not files:
        return ""
    stamp = measured_stamp(axis_rows(Axis.GPU_TALL))
    return "\n" + provenance(
        files,
        "As-run under the redesigned scenario matrix (decision 103), best of "
        "the session's repeats per arm; the whole matrix runs on one pod so "
        "the arms compare." + stamp) + "\n"


# The archive ======================================================================================
# Closed campaigns and probes whose data files were deleted from
# benchmarks/results/ once their decisions froze (results-lifecycle policy,
# decision 92). The refs are pinned rather than resolved at render time: CI
# checks out shallow, where `git log` cannot see the commit, and the page
# would drift between the two environments.


@dataclasses.dataclass(frozen=True)
class Retired:
    """One closed record: what it found, and where its data still reads."""

    record: str
    finding: str
    decisions: str
    files: tuple[str, ...]
    ref: str


ARCHIVE: Final = (
    Retired("The retired row and width standings",
            "The last measurement on the rows and width axes the scenario "
            "redesign replaced: 16M x 100 and the column sweep to 65k "
            "features, both planes, on one pod.",
            "103", ("rebaseline-2026-08.jsonl",
                    "cols-rebaseline-2026-08.jsonl"), "9221cf7"),
    Retired("The retired shape and frontier standings",
            "The iso-volume aspect sweep at 2^31 cells and the 16M "
            "accuracy-versus-time frontier, both replaced by the tall, wide, "
            "and extreme scenarios.",
            "91, 103", ("iso-volume-2026-08.jsonl",
                        "gpu-pareto-16M-2026-08.jsonl"),
            "a32eccf"),
    Retired("The retired airline standings",
            "The one real-data perf axis, removed with the redesign: its "
            "10M-row CSV bake answered a protocol question that no longer "
            "has a scenario.",
            "103", ("airline-2026-08.jsonl",), "9221cf7"),
    Retired("The retired early-stop axis",
            "Early stopping measured at 4M rows, superseded by the "
            "gpu-early-stop axis at the tall cell.",
            "103", ("early-stop-4M-2026-08.jsonl",), "89814a0"),
    Retired("The XGBoost 3.3 recheck",
            "3.3 matched 3.2 at every GPU cell and did not move host RSS; "
            "its one real gain was wide-CPU histogram time at 1M x 4096, "
            "and no published cell flipped.",
            "87", ("xgb33-recheck-2026-07.jsonl",), "563879a"),
    Retired("The CPU 16M round",
            "Software prefetch closed the 16M CPU gap to XGBoost-hist on "
            "that pod.",
            "61", ("cpu-prefetch-round-2026-07.jsonl",), "4b40fd9"),
    Retired("Ordered boosting at scale (the CatBoost door)",
            "CatBoost's Ordered and Plain modes against bonsai levelwise as "
            "rows grow: the door that closed the small-data-edge question.",
            "62 to 64", ("catboost-scale-edge-2026-07.jsonl",), "7b0e1ac"),
    Retired("The leafwise recheck",
            "Decision 42's reading, bonsai's CPU leafwise ahead of "
            "LightGBM's CUDA leaf-wise, inverts monotonically with scale to "
            "5.3x in LightGBM's favor at 16M.",
            "95", ("leafwise-recheck-2026-08.jsonl",), "54497c1"),
    Retired("The device-leafwise cadence probe",
            "The round skeleton's fixed cost measured 30.3 us against a "
            "100 us budget, which is what let the one-node-per-round design "
            "proceed to a build.",
            "issue #268", ("leafwise-cadence-2026-08.json",), "267b705"),
    Retired("The device-leafwise admission ladder",
            "The pre-registered kill criterion met: `cuda_leafwise` beat "
            "LightGBM's CUDA leaf-wise at 16M x 100 on the same pod, and "
            "beat its own CPU arm at every cell.",
            "97", ("leafwise-ladder-2026-08.jsonl",), "db74fb3"),
    Retired("The closing leafwise ladder",
            "Stage 3's two levers, the device-resident objective and the "
            "round's pinned staging, cut leafwise fit at every cell; a third "
            "was refuted and reverted.",
            "98", ("leafwise-stage3-2026-08.jsonl",), "a5a2e9d"),
    Retired("The leafwise correction",
            "Both campaign ladders re-measured on the fixed ingest path: the "
            "harness had binned on the host, so every published bonsai CUDA "
            "time was slow for a reason that was never the library.",
            "100", ("leafwise-correction-2026-08.jsonl",), "e4eb2b9"),
    Retired("The wide-CPU fill",
            "The row-wise u8 fill missed cache at 16k features; routing "
            "feature-parallel fixed the symptom and one column-block-tiled "
            "pass retired the strategy pair.",
            "88, 89", ("wide-cpu-hist-2026-07.jsonl",), "228e2e4"),
    Retired("The CUDA wide recheck",
            "The recorded wide-GPU gap to XGBoost no longer existed on "
            "current main, so the campaign closed at stage 0 and the stale "
            "reading was corrected wherever it appeared.",
            "90", ("cuda-wide-recheck-2026-07.jsonl",), "b60cecf"),
    Retired("The single-card ceiling",
            "A 500M x 100 float32 matrix trained end to end on one 80GB "
            "card: 60 rounds in 8.5 minutes at 69.9 GiB peak device memory.",
            "engine chapter 5", ("single-card-ceiling-2026-07.jsonl",),
            "7e9c8fd"),
    Retired("Campaign smoke: ten datasets at matched knobs",
            "The fast local regression check behind the quality campaign's "
            "three model-changing fixes.",
            "56, 57", ("quality-campaign-2026-07.jsonl",), "1f8eb20"),
    Retired("Probe: per-feature bin budgets",
            "No bin-budget policy moved a standing outside the chance band, "
            "so the 255-bin default stayed.",
            "67", ("binning-probe-2026-07.json",), "98fab49"),
    Retired("Probe: categorical machinery",
            "The categorical question resolved as an encoder rather than "
            "per-split machinery in the core.",
            "58", ("cat-tradeoff-2026-07.json",), "7465ede"),
    Retired("Probe: ranking objectives",
            "The stable NDCG@10 gap is to listwise losses only, which is how "
            "the ranking issue is scoped.",
            "issue #58", ("ranking-tradeoff-2026-07.jsonl",), "95f7eca"),
    Retired("Probe: CatBoost's categorical toggle",
            "CatBoost's own toggle prices its categorical machinery at 68% "
            "of its remaining cat-heavy lead over bonsai's encoder.",
            "80", ("tabarena-cat-probe-2026-07.jsonl",), "0efa5ce"),
    Retired("Probe: ordered boosting",
            "Ordered beat Plain beyond the chance band on 0 of 12 datasets, "
            "at 3.9x the train time.",
            "81", ("ordered-boosting-probe-2026-07.jsonl",), "885db11"),
    Retired("Probe: static K-permutation target statistics",
            "K-averaged ordered statistics recover a negative share of the "
            "gap: the substance is per-split, not preprocessing.",
            "82", ("static-k-encoder-probe-2026-07.jsonl",), "66f587f"),
    Retired("Probe: a per-dataset learning-rate rule",
            "A validation-selected oracle over eight rates wins validation "
            "and loses test, and CatBoost's automatic rate transplants to a "
            "no-op around the shipped 0.05.",
            "83", ("lr-rule-probe-2026-07.jsonl",), "974f5c6"),
    Retired("Probe: the bagged-protocol randomization interaction",
            "CatBoost's small-data lead is not a bagging interaction: "
            "8-fold data-bagging already gives bonsai the decorrelation.",
            "85", ("bagging-interaction-probe-2026-07.jsonl",), "9c3ceaa"),
    Retired("Probe: honest shadow-feature selection",
            "The shadow selector sits inside the chance band against plain "
            "top-k on 26 of 27 grower-dataset cells, under all three "
            "growers.",
            "86", ("feature-selection-probe-2026-07.jsonl",), "981dc1c"),
    Retired("The selection-method survey",
            "Ten selection methods, one shared judge, refit down a budget "
            "ladder on an untouched holdout; the worked example in guide "
            "chapter 14.",
            "86", ("selection-survey-2026-07.jsonl",), "fa6838a"),
    Retired("The Grinsztajn min_child_weight bracket",
            "XGBoost's ambiguous knob translation measured at both ends; the "
            "standings order holds under either convention.",
            "68", ("grinsztajn-2026-07-xgb-mcw1.jsonl",), "5e45b14"),
)


def archive_section() -> str:
    """The pointer table that replaces every deleted evidence section."""
    table = md_table(
        ["record", "what it found", "decisions", "data files",
         "last present at"],
        [[r.record, r.finding, r.decisions,
          ", ".join(f"`{f}`" for f in r.files), f"`{r.ref}`"]
         for r in ARCHIVE])
    return f"""Every campaign and probe below is closed. Its verdict is frozen in [the decisions log](../decisions.md), its narrative is in [`benchmarks/`](../../benchmarks), and its data file was deleted from `benchmarks/results/` rather than kept beside current evidence, which is the results-lifecycle policy (decision 92): the ledger renders what is current, whole, and git history keeps what is not.

The last column is the last commit where the file was present, so the rows still read:

```
git show <ref>:benchmarks/results/<file>
```

{table}
"""


# The code division ================================================================================


def code_metrics_section() -> str:
    """The code-division standings page body."""
    rows = load_jsonl(standings_file(Axis.CODE))
    meta = next(r for r in rows if r["kind"] == "meta")
    planes = [r for r in rows if r["kind"] == "plane"]
    offenders = [r for r in rows if r["kind"] == "offender"]
    s = next(r for r in rows if r["kind"] == "surface")

    plane_table = md_table(
        ["plane", "files", "LOC", "NLOC", "functions", "mean CCN", "max CCN"],
        [[p["plane"], str(p["files"]), str(p["loc"]), str(p["nloc"]),
          str(p["functions"]), fmt(p["ccn_mean"], 2), str(p["ccn_max"])]
         for p in planes]
        + [["all", *[str(sum(p[k] for p in planes))
                     for k in ("files", "loc", "nloc", "functions")], "-", "-"]])

    offender_table = md_table(
        ["function", "file", "CCN", "NLOC"],
        [[f"`{o['function']}`", f"`{o['file']}`", str(o["ccn"]), str(o["nloc"])]
         for o in offenders])

    d = s["dispatch_factors"]
    py_dep = "dependency" if s["python_runtime_deps"] == 1 else "dependencies"
    surface_line = (
        f"Surface counts: {s['parameters']} config parameters, "
        f"{s['dispatch_combinations']} registered dispatch combinations "
        f"({d['objectives']} objectives x {d['growers']} growers x {d['samplers']} samplers), "
        f"and {s['python_public_api']} public Python names. "
        f"Dependencies: {s['python_runtime_deps']} Python runtime {py_dep} "
        f"({', '.join(s['python_runtime_dep_names'])}) and {s['cpp_compiled_deps']} "
        f"compiled-in C++ libraries ({', '.join(s['cpp_compiled_dep_names'])}), "
        f"the rule stated in the protocol.")

    return f"""## The code division

Self-measurement of the bonsai tree, no comparison: line counts and lizard complexity per plane at one SHA. LOC is `wc -l`; NLOC is lizard's non-blank, non-comment count; CCN is cyclomatic complexity (independent paths through a function). The plane map and the non-claims: [the benchmark protocol](benchmark-protocol.md#the-code-division).

{plane_table}

The five highest-CCN functions across `core_headers` + `engine_impl`, published by name; a curated offender list would be marketing.

{offender_table}

{surface_line}

{provenance([standings_file(Axis.CODE)], f"lizard {meta['tool_version']} (`{meta['tool_pin']}`) at `{meta['git_sha'][:12]}`, {meta['date']}; regenerate with [scripts/measure_complexity.py](../../scripts/measure_complexity.py); superseded in place on re-measurement (decision 69).")}
"""


# assembly =========================================================================================

GEN_NOTE = ("<!-- GENERATED by scripts/render_results.py. "
            "Edit the generator, not this page. -->")

HEADER = GEN_NOTE + """

# The results ledger

Every results file behind a published claim is rendered across the pages below, generated straight from the data in [`benchmarks/results/`](../../benchmarks/results): `python3 scripts/render_results.py` rewrites them and CI fails on drift. Rows are as-run records under the [benchmark protocol](benchmark-protocol.md): quality division numbers never cite timing, perf division numbers name their timing mode, and superseded files are deleted rather than kept beside their replacements, so what is here is the current evidence, whole.
"""

# Per-suite pages under docs/method/results/, perf division first.
PAGES: list[tuple[str, str, str, list]] = [
    ("perf.md", "The scenario panels",
     "Every grower against its closest rival, both planes, at the "
     "redesigned scenarios.",
     [perf_summary_section, perf_panels_section, early_stop_section]),
    ("quality-grinsztajn.md", "Grinsztajn standings",
     "The only citable standings: 55 third-party tasks.",
     [grinsztajn_section]),
    ("code-metrics.md", "The code division",
     "Self-measurement of the tree: lines, complexity, surface counts.",
     [code_metrics_section]),
    ("archive.md", "The archive",
     "Closed campaigns and probes: what each found, and the git ref where "
     "its data still reads.",
     [archive_section]),
]

# Suite pages live one level below docs/method/, so their relative links
# gain one step. Replacement order matters: the two-step rule cannot touch
# ../guide/ and runs first.
_REROOT = [("](../../", "](../../../"), ("](../guide/", "](../../guide/"),
           ("](../architecture/", "](../../architecture/"),
           ("](../decisions.md", "](../../decisions.md"),
           ("](assets/", "](../assets/"), ("](benchmark-protocol.md", "](../benchmark-protocol.md")]


def _reroot(body: str) -> str:
    """Repo-relative links rewritten for the per-suite page depth."""
    for old, new in _REROOT:
        body = body.replace(old, new)
    return body


def _division_summaries() -> tuple[str, str]:
    """(perf, quality) headline paragraphs, digits computed from standings."""
    table, _, n = _standings(load_jsonl(standings_file(Axis.GRINSZTAJN)))
    lead_lib, lead_mean, lead_wins = table[0]
    quality = (
        f"{lead_lib} leads the {n}-task Grinsztajn standings at mean rank "
        f"{lead_mean:.2f} with {lead_wins} outright wins, and keeps the "
        f"lead under either reading of the one knob that translates "
        f"ambiguously between libraries. Every feature declined by "
        f"measurement is recorded in the archive.")
    return _perf_summary(), quality


def _perf_summary() -> str:
    """The perf paragraph, read off the summary matrix the panels publish."""
    gpu = _headline_clauses(Axis.GPU_TALL, PLANE_GPU)
    cpu = _headline_clauses(Axis.CPU_TALL, PLANE_CPU)
    if not gpu and not cpu:
        return ("The perf standings are being re-measured on the redesigned "
                "scenario matrix (decision 103): tall and wide scenarios at "
                "constant cell count plus a memory-maxout extreme, on both "
                "planes, with early stopping as its own axis. The panels "
                "page fills in with the first refresh; the campaigns that "
                "led here are in the archive.")
    parts = []
    if gpu:
        parts.append("On GPU at the tall scenario, fit totals run "
                     + "; ".join(gpu) + ".")
    if cpu:
        parts.append("On CPU at the tall scenario: " + "; ".join(cpu) + ".")
    parts.append("The wide and extreme scenarios, the host and device memory "
                 "columns, and the early-stopping axis are on the panels "
                 "page.")
    return " ".join(parts)


def _headline_clauses(axis: str, plane: str) -> list[str]:
    """`depthwise 6.9s vs XGBoost 8.1s`, one clause per grower, with the
    same tie note the tables carry so the prose cannot claim more."""
    rows = axis_rows(axis)
    out = []
    for panel in PANELS:
        if not rows:
            continue
        bonsai, rival = (best_row(rows, v) for v in panel.arms(plane))
        ours, theirs = _fit_value(bonsai), _fit_value(rival)
        if ours is None or theirs is None:
            continue
        note = (f" {TIE_NOTE}"
                if _fit_is_tie(rows, panel, plane, ours, theirs) else "")
        out.append(f"{panel.grower} {ours:.1f}s vs {panel.rival} "
                   f"{theirs:.1f}s{note}")
    return out


def _page_table(entries: list[tuple[str, str, str, list]]) -> str:
    """The landing page's navigation table."""
    return md_table(
        ["page", "what it holds"],
        [[f"[{title}](results/{rel})", desc] for rel, title, desc, _ in entries])


def _page(rel: str) -> tuple[str, str, str, list]:
    """One PAGES entry by file name."""
    return next(p for p in PAGES if p[0] == rel)


def landing_page() -> str:
    """The results landing: division summaries plus navigation."""
    perf, quality = _division_summaries()
    return f"""{HEADER}
## Perf division

{perf}

{_page_table([_page("perf.md")])}

## Quality division

{quality}

{_page_table([_page("quality-grinsztajn.md")])}

## The code division

Self-measurement of the bonsai tree, no comparative claim: [code metrics](results/code-metrics.md).

## The archive

Closed campaigns and probes, deleted from `benchmarks/results/` once their decisions froze: [the archive](results/archive.md) names what each one found and the git ref where its data still reads.
"""


README_PATH = REPO / "README.md"
MARK_BEGIN = "<!-- standings:begin (generated by scripts/render_results.py) -->"
MARK_END = "<!-- standings:end -->"

_SITE = "https://daniel-m-campos.github.io/bonsai"
_LIB_NAMES = {"lgbm": "lightgbm", "xgb": "xgboost"}


def readme_standings_block() -> str:
    """The README's digit surface, generated so it cannot drift (decision
    92): division summaries plus the quality table."""
    perf = _perf_summary()

    table, _, n_tasks = _standings(load_jsonl(standings_file(Axis.GRINSZTAJN)))
    qlines = ["| library | mean rank | outright wins |", "|---|--:|--:|"]
    for i, (lib, mean, wins) in enumerate(table):
        row = [_LIB_NAMES.get(lib, lib), f"{mean:.2f}", str(wins)]
        if i == 0:
            row = [f"**{x}**" for x in row]
        qlines.append("| " + " | ".join(row) + " |")
    quality_table = "\n".join(qlines)
    lead = _LIB_NAMES.get(table[0][0], table[0][0])

    return f"""### Perf

{perf}

The panels, and the closed campaigns behind them, are in [the ledger]({_SITE}/method/results/).

### Quality

On the [Grinsztajn et al. tabular benchmark](https://arxiv.org/abs/2207.08815) ({n_tasks} OpenML tasks selected by third parties, three seeds, matched knobs, best variant per library), {lead} takes the best mean rank with {table[0][2]} outright wins:

{quality_table}"""


def spliced_readme() -> str:
    """README with the generated standings block spliced between markers."""
    text = README_PATH.read_text()
    if MARK_BEGIN not in text or MARK_END not in text:
        raise SystemExit("README.md is missing the standings markers")
    i = text.index(MARK_BEGIN) + len(MARK_BEGIN)
    j = text.index(MARK_END)
    return text[:i] + "\n\n" + readme_standings_block() + "\n\n" + text[j:]


def render_pages() -> dict[pathlib.Path, str]:
    """Every generated page body keyed by output path."""
    out: dict[pathlib.Path, str] = {}
    for rel, title, _desc, fns in PAGES:
        body = _reroot("\n".join(fn() for fn in fns))
        head = "" if body.startswith("## ") else f"# {title}\n\n"
        out[REPO / "docs" / "method" / "results" / rel] = (
            f"{GEN_NOTE}\n\n{head}{body}")
    out[OUT] = landing_page()
    return out


def check_registry(consumed_files: set[str]) -> list[str]:
    """Standings registry invariants (decision 92): each registered file
    exists, is rendered, and its rows carry exactly the registered sha
    (sha_partial entries tolerate provenance-less rows, never a WRONG sha).
    An axis's companion evidence file, if it has one, is held to the same
    exists-and-is-rendered rule.

    An axis whose `file` is null has never been measured (registry v2's
    placeholder), so there is nothing to check yet; the release gate in
    scripts/check_standings.py is what refuses to ship one.
    """
    reg = json.loads((REPO / "benchmarks" / "standings.json").read_text())
    reg.pop("_", None)
    errors = []
    for axis, e in reg.items():
        if not e.get("file"):
            continue
        for kind, name in (("", e["file"]), ("companion ", e.get("companion"))):
            if not name:
                continue
            if not (RESULTS / name).exists():
                errors.append(f"standings {axis}: {kind}{name} does not exist")
            elif name not in consumed_files:
                errors.append(f"standings {axis}: {kind}{name} is not rendered")
        if not (RESULTS / e["file"]).exists() or not e.get("sha"):
            continue
        rows = [json.loads(ln)
                for ln in (RESULTS / e["file"]).read_text().splitlines()
                if ln.strip()]
        shas = {r.get(K.GIT_SHA) for r in rows}
        wrong = {s for s in shas if s and not s.startswith(e["sha"])
                 and not e["sha"].startswith(s)}
        if wrong:
            errors.append(f"standings {axis}: rows carry {sorted(wrong)}, "
                          f"registry says {e['sha']}")
        if None in shas and not e.get("sha_partial"):
            errors.append(f"standings {axis}: rows without git_sha in a "
                          "full-provenance standings file")
    return errors


def committed_data_files() -> set[str]:
    """Data files under benchmarks/results/ that pages must consume."""
    out = subprocess.run(
        ["git", "ls-files", "benchmarks/results"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout
    files = set()
    for line in out.splitlines():
        name = pathlib.PurePosixPath(line).name
        files.add(name)
    return files


def main() -> int:
    """Write (or --check) every generated page, chart, and the README block."""
    outputs = render_pages()
    manifest = committed_data_files()
    missed = manifest - consumed
    if missed:
        print(f"ERROR: committed results files not rendered: {sorted(missed)}\n"
              "Wire them into scripts/render_results.py or remove them "
              "(use it or remove it).", file=sys.stderr)
        return 1
    outputs[README_PATH] = spliced_readme()
    registry_errors = check_registry(consumed)
    if registry_errors:
        for e in registry_errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    for fname, svg in charts.items():
        outputs[ASSETS / fname] = svg
    # Use it or remove it, page-side: a suite page nothing generates is stale.
    pages_dir = REPO / "docs" / "method" / "results"
    orphans = [p for p in pages_dir.glob("*.md")
               if p not in outputs] if pages_dir.is_dir() else []
    if orphans:
        names = ", ".join(str(p.relative_to(REPO)) for p in orphans)
        print(f"ERROR: orphaned generated pages: {names}; remove them or wire "
              "them into PAGES.", file=sys.stderr)
        return 1
    if "--check" in sys.argv:
        stale = [p for p, content in outputs.items()
                 if not p.exists() or p.read_text() != content]
        if stale:
            names = ", ".join(str(p.relative_to(REPO)) for p in stale)
            print(f"ERROR: stale generated files: {names}; run "
                  "python3 scripts/render_results.py", file=sys.stderr)
            return 1
        print(f"results ledger: in sync ({len(PAGES)} pages + landing, "
              f"{len(charts)} charts)")
        return 0
    ASSETS.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    for p, content in outputs.items():
        p.write_text(content)
    print(f"wrote the ledger landing + {len(PAGES)} suite pages "
          f"({len(consumed)} data files consumed) + {len(charts)} charts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
