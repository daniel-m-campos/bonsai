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

import json
import math
import pathlib
import subprocess
import sys
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
OUT = REPO / "docs" / "method" / "results.md"
ASSETS = REPO / "docs" / "method" / "assets"

consumed: set[str] = set()
charts: dict[str, str] = {}  # filename -> svg, written to docs/method/assets/


def load_jsonl(name: str) -> list[dict]:
    consumed.add(name)
    path = RESULTS / name
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def load_json(name: str) -> dict:
    consumed.add(name)
    return json.loads((RESULTS / name).read_text())


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def fmt(v, nd=4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if v != v:  # NaN
            return "-"
        return f"{v:.{nd}f}"
    return str(v)


def provenance(files: list[str], note: str) -> str:
    links = ", ".join(
        f"[`{f}`](../../benchmarks/results/{f})" for f in files)
    return f"*Source: {links}. {note}*"


# ---- SVG charts ---------------------------------------------------------------
# Hand-rolled SVG (stdlib only) so the images are byte-stable and the CI drift
# check covers them exactly like the tables. One color per library everywhere;
# solid strokes are GPU variants, dashed are CPU. Text and grid in mid-grays so
# the site's light and dark themes both read.

LIB_COLOR = {
    "bonsai": "#2e7d32",
    "xgboost": "#1f77b4",
    "lightgbm": "#8e6bbf",
    "catboost": "#e08f1a",
}
LIB_COLOR["xgb"] = LIB_COLOR["xgboost"]
LIB_COLOR["lgbm"] = LIB_COLOR["lightgbm"]
TEXT = "#8a8a8a"
GRID = "#808080"
FONT = "font-family='system-ui,sans-serif'"


def _svg(width: int, height: int, body: list[str]) -> str:
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' "
            f"height='{height}' viewBox='0 0 {width} {height}'>\n"
            + "\n".join(body) + "\n</svg>\n")


def _text(x, y, s, size=11, anchor="start", color=TEXT, weight="normal") -> str:
    return (f"<text x='{x}' y='{y}' font-size='{size}' fill='{color}' "
            f"{FONT} text-anchor='{anchor}' font-weight='{weight}'>{s}</text>")


def _legend(items: list[tuple[str, str, str]], x: int, y: int) -> list[str]:
    out = []
    for i, (label, color, dash) in enumerate(items):
        yy = y + i * 17
        out.append(f"<line x1='{x}' y1='{yy - 4}' x2='{x + 22}' y2='{yy - 4}' "
                   f"stroke='{color}' stroke-width='2.5'{dash}/>")
        out.append(_text(x + 28, y + i * 17, label))
    return out


def _dash(cpu: bool) -> str:
    return " stroke-dasharray='6 4'" if cpu else ""


def bar_chart(fname: str, title: str, rows: list[tuple[str, float, str]],
              x_max: float, note: str) -> None:
    """Horizontal bars: rows = (label, value, annotation); lower is better."""
    w, h, left, top = 720, 60 + 34 * len(rows) + 30, 110, 42
    plot_w = w - left - 150
    body = [_text(left, 22, title, size=13, weight="bold")]
    for i, (label, value, note_txt) in enumerate(rows):
        y = top + i * 34
        bw = round(plot_w * value / x_max, 1)
        color = LIB_COLOR.get(label, TEXT)
        body.append(_text(left - 8, y + 14, label, anchor="end"))
        body.append(f"<rect x='{left}' y='{y}' width='{bw}' height='20' "
                    f"rx='3' fill='{color}' fill-opacity='0.85'/>")
        body.append(_text(left + bw + 8, y + 14,
                          f"{value:.2f}&#160;&#160;{note_txt}"))
    body.append(_text(left, top + len(rows) * 34 + 18, note, size=10))
    charts[fname] = _svg(w, h, body)


def line_chart(fname: str, title: str, y_label: str,
               series: list[tuple[str, str, bool, list[tuple[float, float]]]],
               x_ticks: list[tuple[float, str]], log_x=True, log_y=True,
               width=760, height=400, y_ticks=None,
               point_labels=None) -> None:
    """series = (label, color, is_cpu, [(x, y)]). Log-log by default."""
    left, right, top, bottom = 64, 170, 42, 44
    pw, ph = width - left - right, height - top - bottom

    def tx(v):
        return math.log10(v) if log_x else v

    def ty(v):
        return math.log10(v) if log_y else v

    xs = [tx(x) for _, _, _, pts in series for x, _ in pts]
    ys = [ty(y) for _, _, _, pts in series for _, y in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    y0 -= (y1 - y0) * 0.06 or 0.05
    y1 += (y1 - y0) * 0.06 or 0.05

    def px(v):
        return round(left + (tx(v) - x0) / (x1 - x0) * pw, 1)

    def py(v):
        return round(top + ph - (ty(v) - y0) / (y1 - y0) * ph, 1)

    body = [_text(left, 22, title, size=13, weight="bold")]
    if y_ticks is None and log_y:
        lo, hi = math.floor(y0), math.ceil(y1)
        y_ticks = [(10.0 ** e, f"{10 ** e:g}") for e in range(lo, hi + 1)
                   if y0 <= e <= y1]
    for v, label in y_ticks or []:
        yy = py(v)
        body.append(f"<line x1='{left}' y1='{yy}' x2='{left + pw}' y2='{yy}' "
                    f"stroke='{GRID}' stroke-opacity='0.22'/>")
        body.append(_text(left - 8, yy + 4, label, anchor="end"))
    for v, label in x_ticks:
        xx = px(v)
        body.append(f"<line x1='{xx}' y1='{top}' x2='{xx}' y2='{top + ph}' "
                    f"stroke='{GRID}' stroke-opacity='0.12'/>")
        body.append(_text(xx, top + ph + 18, label, anchor="middle"))
    body.append(_text(left - 44, top - 12, y_label, size=10))
    for _label, color, cpu, pts in series:
        path = " ".join(f"{px(x)},{py(y)}" for x, y in pts)
        body.append(f"<polyline points='{path}' fill='none' stroke='{color}' "
                    f"stroke-width='2.5'{_dash(cpu)}/>")
        for x, y in pts:
            body.append(f"<circle cx='{px(x)}' cy='{py(y)}' r='3.2' "
                        f"fill='{color}'/>")
    if point_labels:
        for x, y, s in point_labels:
            body.append(_text(px(x) + 6, py(y) - 7, s, size=9))
    body.extend(_legend([(lbl, c, _dash(cpu)) for lbl, c, cpu, _ in series],
                        left + pw + 18, top + 12))
    charts[fname] = _svg(width, height, body)


# ---- Quality: Grinsztajn standings ------------------------------------------


def _value(r: dict):
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
        if r.get("status") != "ok":
            continue
        v = _value(r)
        if v is None:
            continue
        acc[(r["suite"], r["dataset"], r["variant"])].append(v)
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
    main = load_jsonl("grinsztajn-2026-07.jsonl")
    mcw1 = load_jsonl("grinsztajn-2026-07-xgb-mcw1.jsonl")
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

    sens = [r for r in main if not str(r.get("variant", "")).startswith("xgb")]
    sens += mcw1
    stable, _, _ = _standings(sens)
    sens_rows = [[lib, fmt(mean, 2), str(w)] for lib, mean, w in stable]
    sensitivity = md_table(["library", "mean rank", "outright wins"], sens_rows)

    return f"""## Quality division

### External standings: the Grinsztajn suite

The [Grinsztajn et al. tabular benchmark](https://arxiv.org/abs/2207.08815) at the paper-medium protocol: {n} OpenML tasks, three seeds, campaign knobs for every library (decision 68). Best variant per library, average rank across tasks, lower is better.

![Grinsztajn mean rank by library](assets/grinsztajn-rank.svg)

{campaign}

Per-suite mean rank:

{per_suite}

Sensitivity: xgboost's campaign mapping sets `min_child_weight=20` (hessian-weighted, the knob-translation bracket recorded in decision 68); replacing its rows with the `min_child_weight=1` run gives the other end of the bracket:

{sensitivity}

Reproduce: `pip install bonsai-gbt[bench]`, then `python -m bonsai.bench.grinsztajn out.jsonl --report`.

{provenance(["grinsztajn-2026-07.jsonl", "grinsztajn-2026-07-xgb-mcw1.jsonl"], "As-run; evidence narrative in [benchmarks/grinsztajn-2026-07.md](../../benchmarks/grinsztajn-2026-07.md), ruling in decision 68.")}
"""


# ---- Quality: campaign smoke -------------------------------------------------


def campaign_section() -> str:
    raw = load_jsonl("quality-campaign-2026-07.jsonl")
    latest: dict[str, dict] = {}
    for row in raw:
        latest[row["run"]] = row["results"]  # file is chronological; last wins

    def lib_of(label: str) -> str:
        return "bonsai" if label.startswith("bonsai") else label

    out_rows = []
    for run in sorted(latest):
        res = latest[run]
        sample = next(iter(res.values()))
        aucs = [v.get("auc") for v in res.values()]
        use_auc = any(isinstance(a, float) and a == a for a in aucs)
        metric = "auc" if use_auc else ("acc" if any(
            isinstance(v.get("acc"), float) and v["acc"] == v["acc"]
            for v in res.values()) else "rmse")
        best: dict[str, float] = {}
        for label, m in res.items():
            v = m.get(metric)
            if not isinstance(v, (int, float)) or v != v:
                continue
            lib = lib_of(label)
            keep = max if metric in ("auc", "acc") else min
            best[lib] = keep(best.get(lib, v), v)
        if not best:
            continue
        pick = max if metric in ("auc", "acc") else min
        winner = pick(best, key=best.get)
        out_rows.append([
            run, metric,
            *[fmt(best.get(lib)) for lib in ("bonsai", "xgboost", "lightgbm", "catboost")],
            winner])
        del sample
    table = md_table(
        ["dataset", "metric", "bonsai", "xgboost", "lightgbm", "catboost", "best"],
        out_rows)
    return f"""### Campaign smoke: ten datasets at matched knobs

The internal quality campaign (`scripts/compare.py`, campaign knobs, best variant per library, latest run per dataset):

{table}

{provenance(["quality-campaign-2026-07.jsonl"], "Aggregate record; narrative in [benchmarks/quality-campaign-2026-07.md](../../benchmarks/quality-campaign-2026-07.md), decisions 56 and 57.")}
"""


# ---- Quality: probes ---------------------------------------------------------


def probes_section() -> str:
    binning = load_json("binning-probe-2026-07.json")
    bin_rows = []
    keys = ["bonsai_uniform255", "bonsai_importance", "bonsai_inverse",
            "bonsai_headroom", "lgbm_uniform255", "lgbm_importance",
            "xgb_uniform255"]
    for ds in sorted(binning):
        d = binning[ds]
        bin_rows.append([ds, *[fmt(d.get(k)) for k in keys]])
    bin_table = md_table(["dataset", *keys], bin_rows)

    cats = load_json("cat-tradeoff-2026-07.json")
    setups = sorted({k for d in cats.values() for k in d if not k.startswith("_")})
    datasets = sorted(cats)
    cat_table = md_table(
        ["setup", *datasets],
        [[s, *[fmt(cats[ds].get(s)) for ds in datasets]] for s in setups])

    rank = load_jsonl("ranking-tradeoff-2026-07.jsonl")
    # Two row shapes: synthetic regimes carry "regime", the MQ2008 gate "data".
    for r in rank:
        r["regime"] = r.get("regime") or r["data"]
    regimes = sorted({r["regime"] for r in rank})
    learners = []
    for r in rank:
        if r["learner"] not in learners:
            learners.append(r["learner"])
    cell = {(r["learner"], r["regime"]): r["ndcg_at_10"] for r in rank}
    rank_table = md_table(
        ["learner", *[f"{g} (NDCG@10)" for g in regimes]],
        [[ln, *[fmt(cell.get((ln, g))) for g in regimes]] for ln in learners])

    return f"""### Probe: per-feature bin budgets (declined, decision 67)

Test r² under per-feature bin-budget policies at a 255-bin default; no policy moved standings outside the chance band.

{bin_table}

{provenance(["binning-probe-2026-07.json"], "Probe: [scripts/probe_binning.py](../../scripts/probe_binning.py); evidence [benchmarks/binning-tradeoff-2026-07.md](../../benchmarks/binning-tradeoff-2026-07.md).")}

### Probe: categorical machinery (resolved as an encoder, decision 58)

AUC by setup: each reference library's own categorical toggle against ordinal codes, and bonsai's ordered-target-statistics preprocessing.

{cat_table}

{provenance(["cat-tradeoff-2026-07.json"], "Probe: [scripts/probe_categorical.py](../../scripts/probe_categorical.py); evidence [benchmarks/categorical-tradeoff-2026-07.md](../../benchmarks/categorical-tradeoff-2026-07.md).")}

### Probe: ranking objectives (gated, issue #58)

NDCG@10 by regime; the stable gap is to listwise losses only, so issue #58 is scoped listwise-first.

{rank_table}

{provenance(["ranking-tradeoff-2026-07.jsonl"], "Probe: [scripts/probe_ranking.py](../../scripts/probe_ranking.py); evidence [benchmarks/ranking-tradeoff-2026-07.md](../../benchmarks/ranking-tradeoff-2026-07.md).")}
"""


# ---- Perf: re-baseline -------------------------------------------------------

REBASE_VARIANTS = [
    ("bonsai_cuda_depthwise", "bonsai cuda dw"),
    ("bonsai_cuda_oblivious", "bonsai cuda obl"),
    ("xgb_cuda", "xgb cuda"),
    ("catboost_gpu", "catboost gpu"),
    ("lgbm_cpu", "lgbm cpu"),
    ("bonsai_oblivious", "bonsai cpu obl"),
]

# Chart styling: (display label, color, is_cpu). bonsai's two CUDA growers get
# two greens; CPU variants are dashed everywhere.
VARIANT_STYLE = {
    "bonsai_cuda_depthwise": ("bonsai cuda dw", "#1b5e20", False),
    "bonsai_cuda_oblivious": ("bonsai cuda obl", "#4caf50", False),
    "xgb_cuda": ("xgb cuda", LIB_COLOR["xgboost"], False),
    "catboost_gpu": ("catboost gpu", LIB_COLOR["catboost"], False),
    "lgbm_cpu": ("lgbm cpu", LIB_COLOR["lightgbm"], True),
    "bonsai_oblivious": ("bonsai cpu obl", "#4caf50", True),
}


def _cell_best(rows: list[dict]) -> dict[tuple, dict]:
    best: dict[tuple, dict] = {}
    for r in rows:
        c = r["cell"]
        key = (c["rows"], c["cols"], r["variant"])
        if key not in best or r["fit_s"] < best[key]["fit_s"]:
            best[key] = r
    return best


def _fmt_cell(best, rows, cols, variant) -> str:
    r = best.get((rows, cols, variant))
    if r is None:
        return "-"
    return f"{r['fit_s']:.1f}s ({fmt(r.get('r2_test'), 3).lstrip('0')})"


def rebaseline_section() -> str:
    rows = load_jsonl("rebaseline-2026-07.jsonl")
    best = _cell_best(rows)
    host = rows[0]["host"]
    row_axis = sorted({r["cell"]["rows"] for r in rows if r["cell"]["cols"] == 100})
    col_axis = sorted({r["cell"]["cols"] for r in rows if r["cell"]["rows"] == 1_000_000})

    def human(n):
        return f"{n // 1_000_000}M" if n >= 1_000_000 else f"{n // 1000}k"

    rows_table = md_table(
        ["rows", *[lbl for _, lbl in REBASE_VARIANTS]],
        [[human(n), *[_fmt_cell(best, n, 100, v) for v, _ in REBASE_VARIANTS]]
         for n in row_axis])
    cols_table = md_table(
        ["cols", *[lbl for _, lbl in REBASE_VARIANTS]],
        [[str(c), *[_fmt_cell(best, 1_000_000, c, v) for v, _ in REBASE_VARIANTS]]
         for c in col_axis])

    def series_for(cells: list[tuple[int, int]]):
        out = []
        for variant, (label, color, cpu) in VARIANT_STYLE.items():
            pts = [(x, best[(r, c, variant)]["fit_s"])
                   for x, (r, c) in cells if (r, c, variant) in best]
            if pts:
                out.append((label, color, cpu, pts))
        return out

    line_chart(
        "rebaseline-rows.svg",
        "Fit seconds vs rows (100 features, log-log; lower is better)",
        "fit seconds",
        series_for([(n, (n, 100)) for n in row_axis]),
        x_ticks=[(n, human(n)) for n in row_axis])
    line_chart(
        "rebaseline-cols.svg",
        "Fit seconds vs features (1M rows, log-log; lower is better)",
        "fit seconds",
        series_for([(c, (1_000_000, c)) for c in col_axis]),
        x_ticks=[(c, str(c)) for c in col_axis])

    return f"""## Perf division

### The re-baseline: fit seconds at scale

Same-pod sweep ({host['cpu_model']}, {host['gpu']}), synthetic regression, `fit()` timed end to end including each library's own ingest, best of repeats, test r² in parentheses.

![Fit seconds vs rows](assets/rebaseline-rows.svg)

Scaling rows (100 features):

{rows_table}

![Fit seconds vs features](assets/rebaseline-cols.svg)

Scaling features (1M rows):

{cols_table}

{provenance(["rebaseline-2026-07.jsonl"], "Runner: [scripts/bench_scaling.py](../../scripts/bench_scaling.py) (`python -m bonsai.bench.scaling`); README Performance derives from the same file.")}
"""


# ---- Perf: the remaining tracks ---------------------------------------------


def perf_tracks_section() -> str:
    scaling = load_jsonl("scaling.jsonl")
    hosts = sorted({r["host"]["name"] for r in scaling})
    axes = sorted({r["cell"]["axis"] for r in scaling})
    scaling_note = (
        f"{len(scaling)} runs across {len(hosts)} hosts and the axes "
        f"{', '.join(axes)}; regenerate exponents and the committed log-log plots under "
        f"[`benchmarks/results/scaling/`](../../benchmarks/results/scaling) with "
        f"[scripts/analyze_scaling.py](../../scripts/analyze_scaling.py).")

    msd = load_jsonl("gpu_msd.jsonl")
    latest: dict[str, dict] = {}
    for r in msd:
        if r["gpu"] not in latest or r["ts"] > latest[r["gpu"]]["ts"]:
            latest[r["gpu"]] = r
    msd_table = md_table(
        ["GPU", "as of", "fit_s", "predict_s", "rmse"],
        [[g, latest[g]["ts"][:10], fmt(latest[g]["fit_s"], 2),
          fmt(latest[g].get("predict_s"), 2), fmt(latest[g].get("rmse"), 4)]
         for g in sorted(latest)])

    pre = load_jsonl("cpu-prefetch-round-2026-07.jsonl")
    pre_best = _cell_best(pre)
    pre_rows = sorted({k[0] for k in pre_best})
    pre_variants = sorted({k[2] for k in pre_best})
    pre_table = md_table(
        ["rows", *pre_variants],
        [[f"{n:,}", *[_fmt_cell(pre_best, n, 100, v) for v in pre_variants]]
         for n in pre_rows])

    pareto = load_jsonl("gpu-pareto-16M-2026-07.jsonl")
    par_variants = []
    for r in pareto:
        if r["variant"] not in par_variants:
            par_variants.append(r["variant"])
    par_table = md_table(
        ["variant", "iters", "fit_s", "test r2"],
        [[r["variant"], str(r["iters"]), fmt(r["fit_s"], 2), fmt(r["r2_test"], 4)]
         for r in pareto])
    par_series = []
    par_labels = []
    for v in par_variants:
        label, color, cpu = VARIANT_STYLE.get(v, (v, TEXT, False))
        pts = sorted((r["fit_s"], r["r2_test"]) for r in pareto if r["variant"] == v)
        par_series.append((label, color, cpu, pts))
        for r in pareto:
            if r["variant"] == v:
                par_labels.append((r["fit_s"], r["r2_test"], str(r["iters"])))
    line_chart(
        "gpu-pareto-16M.svg",
        "16M rows: accuracy vs fit time by iteration count (up-left is better)",
        "test r2",
        par_series,
        x_ticks=[(s, f"{s}s") for s in (15, 30, 45, 60)],
        log_x=False, log_y=False, height=420,
        y_ticks=[(v, f"{v:.2f}") for v in (0.84, 0.86, 0.88)],
        point_labels=par_labels)

    edge = load_jsonl("catboost-scale-edge-2026-07.jsonl")

    def detail(r):
        if "iters" in r:
            return f"iters={r['iters']}"
        if "n_samples" in r:
            return f"bin samples={r['n_samples']}"
        return "-"

    edge_table = md_table(
        ["door", "rows", "learner", "knob", "fit_s", "test r2"],
        [[r["door"], f"{r['rows']:,}", r["learner"], detail(r),
          fmt(r["fit_s"], 2), fmt(r["r2_test"], 4)] for r in edge])

    return f"""### The scaling study

{scaling_note}

{provenance(["scaling.jsonl"], "The full-history perf ledger behind [benchmarks/README.md](../../benchmarks/README.md); decision 46.")}

### GPU year-MSD track

Latest run per device on the YearPredictionMSD pipeline benchmark (full history in the file):

{msd_table}

{provenance(["gpu_msd.jsonl"], "Runner: [scripts/bench_gpu.py](../../scripts/bench_gpu.py); pipeline timing mode.")}

### CPU 16M round (the prefetch tie)

{pre_table}

{provenance(["cpu-prefetch-round-2026-07.jsonl"], "Decision 61: software prefetch closed the 16M CPU gap to xgboost-hist on this pod.")}

### GPU accuracy-vs-time frontier at 16M

![Accuracy vs fit time at 16M rows](assets/gpu-pareto-16M.svg)

{par_table}

{provenance(["gpu-pareto-16M-2026-07.jsonl"], "Post-campaign re-run (2026-07-15, decision 72): bonsai is first to every accuracy up to r2 ~0.895, ties catboost through the plateau, and holds the 0.8981 ceiling; the marginal round fell 155 to 104 ms. Evidence: [benchmarks/gpu-pareto-16M-2026-07.md](../../benchmarks/gpu-pareto-16M-2026-07.md).")}

### Ordered boosting at scale (catboost door)

The probe behind decisions 62 to 64: catboost's Ordered vs Plain modes against bonsai oblivious as rows grow.

{edge_table}

{provenance(["catboost-scale-edge-2026-07.jsonl"], "Evidence: [benchmarks/catboost-scale-edge-2026-07.md](../../benchmarks/catboost-scale-edge-2026-07.md).")}
"""


# ---- assembly ----------------------------------------------------------------

HEADER = """<!-- GENERATED by scripts/render_results.py. Edit the generator, not this page. -->

# The results ledger

Every committed data file under [`benchmarks/results/`](../../benchmarks/results) is rendered on this page, which is generated straight from the data: `python3 scripts/render_results.py` rewrites it and CI fails if the page drifts from the files. Rows are as-run records under the [benchmark protocol](benchmark-protocol.md): quality division numbers never cite timing, perf division numbers name their timing mode, and superseded files are deleted rather than kept beside their replacements, so what is here is the current evidence, whole.
"""


def render() -> str:
    parts = [
        HEADER,
        grinsztajn_section(),
        campaign_section(),
        probes_section(),
        rebaseline_section(),
        perf_tracks_section(),
    ]
    return "\n".join(parts)


def committed_data_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "benchmarks/results"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout
    files = set()
    for line in out.splitlines():
        name = pathlib.PurePosixPath(line).name
        if line.endswith(".png"):
            continue  # plot outputs of analyze_scaling.py, linked in-page
        files.add(name)
    return files


def main() -> int:
    text = render()
    manifest = committed_data_files()
    missed = manifest - consumed
    if missed:
        print(f"ERROR: committed results files not rendered: {sorted(missed)}\n"
              "Wire them into scripts/render_results.py or remove them "
              "(use it or remove it).", file=sys.stderr)
        return 1
    outputs: dict[pathlib.Path, str] = {OUT: text}
    for fname, svg in charts.items():
        outputs[ASSETS / fname] = svg
    if "--check" in sys.argv:
        stale = [p for p, content in outputs.items()
                 if not p.exists() or p.read_text() != content]
        if stale:
            names = ", ".join(str(p.relative_to(REPO)) for p in stale)
            print(f"ERROR: stale generated files: {names}; run "
                  "python3 scripts/render_results.py", file=sys.stderr)
            return 1
        print(f"results ledger: in sync ({len(charts)} charts)")
        return 0
    ASSETS.mkdir(parents=True, exist_ok=True)
    for p, content in outputs.items():
        p.write_text(content)
    print(f"wrote {OUT.relative_to(REPO)} ({len(text.splitlines())} lines, "
          f"{len(consumed)} data files consumed) + {len(charts)} charts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
