"""Fill the landing page's hero readout and door counts at build time.

The hero on docs/index.md is a server-rendered readout of the fit times at
the tall, wide and extreme scenarios, GPU and CPU side by side, drawn from
the same `summary_rows()` the scenario panel renders from, so the two can
never disagree. The scenario buttons are CSS radio inputs, so the switch
needs no script and survives instant navigation. The three
door cards carry a count each (chapters, parameters, scenarios) read from
the nav, the parameters extraction and the standings registry. Every number
is HTML text, so fetchers and screen readers read what the bars draw.

A missing row raises, so the hero never renders stale or empty.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from typing import Final

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import render_results  # noqa: E402

LANDING: Final = "index.md"
COUNT_SPAN: Final = re.compile(
    r'<span class="bz-count" data-bz-count="(learn|api|results)"></span>'
)
CHAPTER_TITLE: Final = re.compile(r"^(\d+|E\d+|C\d+)\. ")
VERSION_LINE: Final = re.compile(r'^version = "([^"]+)"', re.M)
GPU: Final = render_results.PLANE_GPU
CPU: Final = render_results.PLANE_CPU
Axis = render_results.Axis
SCENARIOS: Final = (
    ("tall", ((GPU, Axis.GPU_TALL, "GPU"), (CPU, Axis.CPU_TALL, "CPU"))),
    ("wide", ((GPU, Axis.GPU_WIDE, "GPU"), (CPU, Axis.CPU_WIDE, "CPU"))),
    ("extreme", ((GPU, Axis.GPU_EXTREME, "GPU"),)),
)
UNITS: Final = {"learn": "chapters", "api": "parameters", "results": "scenarios"}


class ReadoutError(RuntimeError):
    """The hero's data is incomplete; mkdocs aborts the build on it."""


# Public Functions =================================================================================


def on_page_markdown(markdown: str, page, config, files) -> str:
    """Attach the readout to the landing page and fill its door counts."""
    if page.file.src_uri != LANDING:
        return markdown
    page.meta["readout"] = readout()
    counts = door_counts(config["nav"])
    return COUNT_SPAN.sub(lambda m: _count_span(m.group(1), counts[m.group(1)]), markdown)


def readout() -> dict:
    """The hero's data: the tall claim, then one table per scenario."""
    scenarios = [_scenario(key, planes) for key, planes in SCENARIOS]
    tall = render_results.summary_rows(tuple((plane, axis) for plane, axis, _ in SCENARIOS[0][1]))
    return {"meta": _meta_line(), "claim": _claim(tall), "scenarios": scenarios}


def door_counts(nav: list) -> dict[str, int]:
    """Chapters in the Learn tab, knobs in the registry, perf scenarios."""
    titles = [t for t in _nav_titles(nav) if CHAPTER_TITLE.match(t)]
    registry = json.loads((REPO / "docs" / "api" / "parameters.src.json").read_text())
    return {
        "learn": len(titles),
        "api": sum(len(keys) for keys in registry.values()),
        "results": len(render_results.PERF_AXES),
    }


# Private Functions ================================================================================


def _scenario(key: str, planes: tuple) -> dict:
    rows = render_results.summary_rows(tuple((plane, axis) for plane, axis, _ in planes))
    keys = [plane for plane, _, _ in planes]
    tops = {plane: _column_top(key, rows, plane) for plane in keys}
    return {
        "key": key,
        "planes": [{"key": plane, "label": f"{label} ({axis})"} for plane, axis, label in planes],
        "caption": _caption(key, planes),
        "rows": [_row(row, keys, tops) for row in rows],
        "provenance": _provenance(planes),
    }


def _column_top(key: str, rows, plane: str) -> float:
    """The slowest finished arm in one plane column: every bar in that
    column is drawn against it, so bars compare across growers too."""
    finished = []
    for row in rows:
        cell = row.cells[plane]
        if not cell.present:
            raise ReadoutError(f"home readout: {row.grower} has no {plane} {key} row")
        finished.extend(s for s in (cell.bonsai_s, cell.rival_s) if s is not None)
    if not finished:
        raise ReadoutError(f"home readout: no arm finished the {plane} {key} cell")
    return max(finished)


def _row(row, planes: list[str], tops: dict[str, float]) -> dict:
    cells = []
    for plane in planes:
        cell = row.cells[plane]
        cells.append(
            {
                "tied": cell.tied,
                "arms": [
                    _arm("bonsai", cell.bonsai_s, cell.bonsai_text, tops[plane], cell.bonsai_wins),
                    _arm(row.rival, cell.rival_s, cell.rival_text, tops[plane], cell.rival_wins),
                ],
            }
        )
    return {"grower": row.grower, "cells": cells}


def _arm(who: str, seconds: float | None, status: str, top: float, wins: bool) -> dict:
    """One bar as its share of the column's slowest arm; an arm that did not
    finish carries its status word and no bar."""
    if seconds is None:
        return {"who": who, "w": "0.000", "text": status, "wins": False, "finished": False}
    return {
        "who": who,
        "w": f"{seconds / top:.3f}",
        "text": f"{seconds:.1f} s",
        "wins": wins,
        "finished": True,
    }


def _caption(key: str, planes: tuple) -> str:
    sizes = ", ".join(f"{label} {_cell_label(axis)}" for _, axis, label in planes)
    return f"{key} scenario · {sizes} · fit = ingest + train · seconds, lower is better"


def _cell_label(axis: str) -> str:
    cell = render_results.axis_rows(axis)[0]["cell"]
    return render_results.cell_label(cell["rows"], cell["cols"])


def _claim(rows) -> str:
    gpu = render_results.PLANE_GPU
    ours = [r.cells[gpu].bonsai_s for r in rows]
    theirs = [r.cells[gpu].rival_s for r in rows]
    if any(v is None for v in (*ours, *theirs)):
        raise ReadoutError("home readout: a GPU tall row is missing")
    label = _cell_label(Axis.GPU_TALL)
    rivals = ", ".join(r.rival for r in rows[:-1]) + f" and {rows[-1].rival}"
    return (
        f"Tall scenario, same pod, {label}: on one GPU bonsai fits in "
        f"{_span(ours)} s. {rivals} take {_span(theirs)} s."
    )


def _span(values: list[float]) -> str:
    lo, hi = min(values), max(values)
    return f"{lo:.1f}" if f"{lo:.1f}" == f"{hi:.1f}" else f"{lo:.1f} to {hi:.1f}"


def _meta_line() -> str:
    version = VERSION_LINE.search((REPO / "pyproject.toml").read_text())
    if version is None:
        raise ReadoutError("home readout: pyproject.toml carries no version line")
    return (
        f"bonsai-gbt {version.group(1)} · Python 3.9 to 3.13 · Linux x86_64, aarch64 · macOS arm64"
    )


def _provenance(planes: tuple) -> str:
    parts = []
    for _, axis, label in planes:
        host = render_results.axis_rows(axis)[0]["host"]
        device = host.get("gpu") if label == "GPU" else host.get("cpu_model")
        meta = render_results.standings_meta(axis)
        parts.append(f"{label} {device or host['name']}, refreshed for {meta['refreshed_for']}")
    return " · ".join(parts) + " · same pod, interleaved"


def _nav_titles(nav: list) -> list[str]:
    titles: list[str] = []
    for item in nav:
        if not isinstance(item, dict):
            continue
        for title, target in item.items():
            titles.append(title)
            if isinstance(target, list):
                titles.extend(_nav_titles(target))
    return titles


def _count_span(door: str, count: int) -> str:
    return (
        f'<span class="bz-count" data-bz-count="{door}">{count}<small>{UNITS[door]}</small></span>'
    )
