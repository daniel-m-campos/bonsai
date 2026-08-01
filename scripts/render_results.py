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
from typing import Final

from _render_common import md_table


class Axis:
    """Standings axes: registry keys in benchmarks/standings.json."""

    ROWS: Final = "rows"
    WIDTH: Final = "width"
    SHAPE: Final = "shape"
    FRONTIER: Final = "frontier"
    AIRLINE: Final = "airline"
    GRINSZTAJN: Final = "quality-grinsztajn"
    CODE: Final = "code"


class K:
    """Row keys read from the results jsonl (values are the runlog schema)."""

    FIT_S: Final = "fit_s"
    R2_TEST: Final = "r2_test"
    VARIANT: Final = "variant"
    STATUS: Final = "status"
    DATASET: Final = "dataset"
    PEAK_RSS_GB: Final = "peak_rss_gb"
    GIT_SHA: Final = "git_sha"


class Evidence:
    """Frozen evidence files (dated, decision-linked; never superseded)."""

    BINNING_PROBE: Final = "binning-probe-2026-07.json"
    CAT_TRADEOFF: Final = "cat-tradeoff-2026-07.json"
    BAGGING_INTERACTION: Final = "bagging-interaction-probe-2026-07.jsonl"
    CATBOOST_SCALE_EDGE: Final = "catboost-scale-edge-2026-07.jsonl"
    CPU_PREFETCH: Final = "cpu-prefetch-round-2026-07.jsonl"
    CUDA_WIDE_RECHECK: Final = "cuda-wide-recheck-2026-07.jsonl"
    FEATURE_SELECTION: Final = "feature-selection-probe-2026-07.jsonl"
    GRINSZTAJN_MCW1: Final = "grinsztajn-2026-07-xgb-mcw1.jsonl"
    LR_RULE: Final = "lr-rule-probe-2026-07.jsonl"
    ORDERED_BOOSTING: Final = "ordered-boosting-probe-2026-07.jsonl"
    QUALITY_CAMPAIGN: Final = "quality-campaign-2026-07.jsonl"
    RANKING_TRADEOFF: Final = "ranking-tradeoff-2026-07.jsonl"
    SELECTION_SURVEY: Final = "selection-survey-2026-07.jsonl"
    SINGLE_CARD_CEILING: Final = "single-card-ceiling-2026-07.jsonl"
    STATIC_K_ENCODER: Final = "static-k-encoder-probe-2026-07.jsonl"
    TABARENA_CAT: Final = "tabarena-cat-probe-2026-07.jsonl"
    WIDE_CPU_HIST: Final = "wide-cpu-hist-2026-07.jsonl"
    XGB33_RECHECK: Final = "xgb33-recheck-2026-07.jsonl"
    LEAFWISE_RECHECK: Final = "leafwise-recheck-2026-08.jsonl"
    LEAFWISE_CADENCE: Final = "leafwise-cadence-2026-08.json"
    LEAFWISE_LADDER: Final = "leafwise-ladder-2026-08.jsonl"


REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
OUT = REPO / "docs" / "method" / "results.md"
ASSETS = REPO / "docs" / "method" / "assets"

consumed: set[str] = set()
charts: dict[str, str] = {}  # filename -> svg, written to docs/method/assets/


def load_jsonl(name: str) -> list[dict]:
    """Rows of a results jsonl by bare filename."""
    consumed.add(name)
    path = RESULTS / name
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def load_json(name: str) -> dict:
    """One results json by bare filename."""
    consumed.add(name)
    return json.loads((RESULTS / name).read_text())


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
    """Wrap body fragments in an svg element."""
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' "
            f"height='{height}' viewBox='0 0 {width} {height}'>\n"
            + "\n".join(body) + "\n</svg>\n")


def _text(x: float, y: float, s: str, size: int = 11, anchor: str = "start",
          color: str = TEXT, weight: str = "normal") -> str:
    return (f"<text x='{x}' y='{y}' font-size='{size}' fill='{color}' "
            f"{FONT} text-anchor='{anchor}' font-weight='{weight}'>{s}</text>")


def _legend(items: list[tuple[str, str, str]], x: int, y: int) -> list[str]:
    """Legend fragments for a chart."""
    out = []
    for i, (label, color, dash) in enumerate(items):
        yy = y + i * 17
        out.append(f"<line x1='{x}' y1='{yy - 4}' x2='{x + 22}' y2='{yy - 4}' "
                   f"stroke='{color}' stroke-width='2.5'{dash}/>")
        out.append(_text(x + 28, y + i * 17, label))
    return out


def _dash(cpu: bool) -> str:
    """CPU arms draw dashed; GPU solid."""
    return " stroke-dasharray='6 4'" if cpu else ""


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


def line_chart(fname: str, title: str, y_label: str,
               series: list[tuple[str, str, bool, list[tuple[float, float]]]],
               x_ticks: list[tuple[float, str]], log_x=True, log_y=True,
               width=760, height=400, y_ticks=None,
               point_labels=None, x_label: str = ""):
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
    if x0 == x1:
        raise ValueError(f"{fname}: a line chart needs two distinct x values")
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
    if x_label:
        body.append(_text(left + pw / 2, top + ph + 34, x_label, size=10,
                          anchor="middle"))
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
    """The Grinsztajn standings section of the quality landing."""
    main = load_jsonl(standings_file(Axis.GRINSZTAJN))
    mcw1 = load_jsonl(Evidence.GRINSZTAJN_MCW1)
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

    sens = [r for r in main if not str(r.get(K.VARIANT, "")).startswith("xgb")]
    sens += mcw1
    stable, _, _ = _standings(sens)
    sens_rows = [[lib, fmt(mean, 2), str(w)] for lib, mean, w in stable]
    sensitivity = md_table(["library", "mean rank", "outright wins"], sens_rows)

    return f"""### External standings: the Grinsztajn suite

The [Grinsztajn et al. tabular benchmark](https://arxiv.org/abs/2207.08815) at the paper-medium protocol: {n} OpenML tasks, three seeds, campaign knobs for every library (decision 68). Best variant per library, average rank across tasks, lower is better.

![Grinsztajn mean rank by library](assets/grinsztajn-rank.svg)

{campaign}

Per-suite mean rank:

{per_suite}

Sensitivity: XGBoost's campaign mapping sets `min_child_weight=20` (hessian-weighted, the knob-translation bracket recorded in decision 68); replacing its rows with the `min_child_weight=1` run gives the other end of the bracket:

{sensitivity}

Reproduce: `pip install bonsai-gbt[bench]`, then `python -m bonsai.bench.grinsztajn out.jsonl` to run the suite (hours; datasets fetch from OpenML), then `--report` on the same file to render the standings from the jsonl.

{provenance([standings_file(Axis.GRINSZTAJN), Evidence.GRINSZTAJN_MCW1], "As-run; evidence narrative in [benchmarks/grinsztajn-2026-07.md](../../benchmarks/grinsztajn-2026-07.md), ruling in decision 68.")}
"""


# Quality: campaign smoke ==========================================================================


def campaign_section() -> str:
    """The ten-dataset campaign smoke section."""
    raw = load_jsonl(Evidence.QUALITY_CAMPAIGN)
    latest: dict[str, dict] = {}
    for row in raw:
        latest[row["run"]] = row["results"]  # file is chronological; last wins

    def lib_of(label: str) -> str:
        return "bonsai" if label.startswith("bonsai") else label

    out_rows = []
    for run in sorted(latest):
        res = latest[run]
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
    table = md_table(
        [K.DATASET, "metric", "bonsai", "xgboost", "lightgbm", "catboost", "best"],
        out_rows)
    return f"""### Campaign smoke: ten datasets at matched knobs

The internal quality campaign (`scripts/compare.py`, campaign knobs, best variant per library, latest run per dataset):

{table}

{provenance([Evidence.QUALITY_CAMPAIGN], "Aggregate record; narrative in [benchmarks/quality-campaign-2026-07.md](../../benchmarks/quality-campaign-2026-07.md), decisions 56 and 57.")}
"""


# Quality: probes ==================================================================================


def _probe_binning_table() -> str:
    """Per-feature bin-budget probe table (decision 67)."""
    binning = load_json(Evidence.BINNING_PROBE)
    bin_rows = []
    keys = ["bonsai_uniform255", "bonsai_importance", "bonsai_inverse",
            "bonsai_headroom", "lgbm_uniform255", "lgbm_importance",
            "xgb_uniform255"]
    for ds in sorted(binning):
        d = binning[ds]
        bin_rows.append([ds, *[fmt(d.get(k)) for k in keys]])
    return md_table([K.DATASET, *keys], bin_rows)


def _probe_cat_table() -> str:
    """Categorical-machinery probe table (decision 58)."""
    cats = load_json(Evidence.CAT_TRADEOFF)
    setups = sorted({k for d in cats.values() for k in d if not k.startswith("_")})
    datasets = sorted(cats)
    return md_table(
        ["setup", *datasets],
        [[s, *[fmt(cats[ds].get(s)) for ds in datasets]] for s in setups])


def _probe_rank_table() -> str:
    """Ranking-objectives probe table (issue #58)."""
    rank = load_jsonl(Evidence.RANKING_TRADEOFF)
    # Two row shapes: synthetic regimes carry "regime", the MQ2008 gate "data".
    for r in rank:
        r["regime"] = r.get("regime") or r["data"]
    regimes = sorted({r["regime"] for r in rank})
    learners = []
    for r in rank:
        if r["learner"] not in learners:
            learners.append(r["learner"])
    cell = {(r["learner"], r["regime"]): r["ndcg_at_10"] for r in rank}
    return md_table(
        ["learner", *[f"{g} (NDCG@10)" for g in regimes]],
        [[ln, *[fmt(cell.get((ln, g))) for g in regimes]] for ln in learners])


def _probe_tabarena_table() -> str:
    """CatBoost categorical-toggle probe table (decision 80)."""
    tap = load_jsonl(Evidence.TABARENA_CAT)
    tap_ch = [r for r in tap if r["subset"] == "cat_heavy"]
    return md_table(
        [K.DATASET, "cat native", "cat ablated", "bonsai_ts", "categorical share",
         "remaining gap"],
        [[r[K.DATASET], fmt(r["cat_native"]), fmt(r["cat_ablated"]),
          fmt(r["bonsai_ts"]), fmt(r["categorical_share"]), fmt(r["remaining_gap"])]
         for r in sorted(tap_ch, key=lambda r: r["categorical_share"])])


def _probe_ordered_boosting_table() -> str:
    """Ordered-vs-Plain probe table (decision 81)."""
    ob = load_jsonl(Evidence.ORDERED_BOOSTING)
    return md_table(
        [K.DATASET, "metric", "CatBoost Ordered (matched)", "CatBoost Plain (matched)"],
        [[r[K.DATASET], r["metric"], fmt(r.get("cat_ordered_matched")),
          fmt(r.get("cat_plain_matched"))]
         for r in sorted(ob, key=lambda r: r[K.DATASET])])


def _probe_static_k_table() -> str:
    """Static K-permutation target-statistics probe table (decision 82)."""
    sk = load_jsonl(Evidence.STATIC_K_ENCODER)
    return md_table(
        [K.DATASET, "bonsai_ts (K=1)", "K=4", "K=8", "CatBoost native"],
        [[r[K.DATASET], fmt(r.get("ts_k1")), fmt(r.get("ts_k4")), fmt(r.get("ts_k8")),
          fmt(r.get("cat_native"))]
         for r in sorted(sk, key=lambda r: r[K.DATASET])])


def _probe_lr_rule_table() -> str:
    """Per-dataset learning-rate rule probe table (decision 83)."""
    lr = load_jsonl(Evidence.LR_RULE)
    return md_table(
        [K.DATASET, "default (0.05)", "oracle", "oracle lr", "CatBoost auto-lr"],
        [[r[K.DATASET], fmt(r["bonsai_default"]["test"]),
          fmt(r["bonsai_oracle"]["test"]), fmt(r["bonsai_oracle"]["chosen_lr"], 2),
          fmt(r["bonsai_cat_rule"]["transplanted_lr"], 3)]
         for r in sorted(lr, key=lambda r: r[K.DATASET])])


def _probe_bagging_table() -> str:
    """Bagged-protocol randomization-interaction probe table (decision 85)."""
    bi = load_jsonl(Evidence.BAGGING_INTERACTION)
    return md_table(
        [K.DATASET, "metric", "bag gain bonsai", "bag gain cat", "interaction",
         "randomization share", "in band"],
        [[r[K.DATASET], r["metric"], fmt(r.get("bagging_gain_bonsai")),
          fmt(r.get("bagging_gain_cat")), fmt(r.get("interaction")),
          fmt(r.get("randomization_share")),
          "yes" if r.get("interaction_in_band") else "no"]
         for r in sorted(bi, key=lambda r: r[K.DATASET])])


def _probe_selection_survey() -> dict:
    """Draw the selection-survey chart; per-dataset baseline errors back."""
    sv = load_jsonl(Evidence.SELECTION_SURVEY)
    sv_base = {r[K.DATASET]: r["error"] for r in sv
               if r["row_type"] == "baseline"}
    sv_curve: dict = defaultdict(dict)
    for r in sv:
        if r["row_type"] == "curve":
            sv_curve[(r[K.DATASET], r["method"])][r["k"]] = r["error"]
    SV_COLORS = {
        "corr": "#9e9e9e", "mutual_info": "#607d8b", "gain": "#2e7d32",
        "split": "#8bc34a", "shap_train": "#1f77b4", "shap_val": "#01579b",
        "perm_val": "#8e6bbf", "rfe_gain": "#e08f1a", "forward": "#c62828",
        "rfe_val": "#7b3f00",
    }
    sc = "superconductivity"
    series = []
    for m, color in SV_COLORS.items():
        pts = sorted(sv_curve[(sc, m)].items())
        if pts:
            series.append((m, color, m in ("corr", "mutual_info", "split"),
                           [(float(k), v) for k, v in pts]))
    line_chart(
        "selection-survey.svg",
        "Selection-method survey: holdout rmse vs features kept "
        "(superconductivity, 81 features)",
        "holdout rmse",
        series,
        x_ticks=[(4, "4"), (8, "8"), (16, "16"), (32, "32"), (64, "64")],
        x_label="features kept",
        log_x=True, log_y=False,
        y_ticks=[(v, f"{v:.1f}") for v in (10.0, 11.0, 12.0, 13.0)])
    return sv_base


def _probe_feature_selection_table() -> str:
    """Shadow-feature selection probe table (decision 86)."""
    fs = load_jsonl(Evidence.FEATURE_SELECTION)
    return md_table(
        [K.DATASET, "grower", "regime", "metric", "k / total", "bonsai all",
         "bonsai shadow", "bonsai top-k", "cat select", "shadow vs top-k",
         "shadow beats all"],
        [[r[K.DATASET], r["grower"], r["regime"], r["metric"],
          f"{r['k']} / {r['total_features']}", fmt(r.get("bonsai_all"), 5),
          fmt(r.get("bonsai_shadow"), 5), fmt(r.get("bonsai_topk_gain"), 5),
          fmt(r.get("cat_select"), 5), fmt(r.get("shadow_vs_topk"), 5),
          "yes" if r.get("shadow_vs_all", 0) > r.get("band", 0) else "no"]
         for r in fs])


def probes_section() -> str:
    """The probe evidence tables and readings, one subsection per probe."""
    bin_table = _probe_binning_table()
    cat_table = _probe_cat_table()
    rank_table = _probe_rank_table()
    tap_table = _probe_tabarena_table()
    ob_table = _probe_ordered_boosting_table()
    sk_table = _probe_static_k_table()
    lr_table = _probe_lr_rule_table()
    bi_table = _probe_bagging_table()
    sv_base = _probe_selection_survey()
    fs_table = _probe_feature_selection_table()

    return f"""### Probe: per-feature bin budgets (declined, decision 67)

Test r² under per-feature bin-budget policies at a 255-bin default; no policy moved standings outside the chance band.

{bin_table}

{provenance([Evidence.BINNING_PROBE], "Probe: [scripts/probe_binning.py](../../scripts/probe_binning.py); evidence [benchmarks/binning-tradeoff-2026-07.md](../../benchmarks/binning-tradeoff-2026-07.md).")}

### Probe: categorical machinery (resolved as an encoder, decision 58)

AUC by setup: each reference library's own categorical toggle against ordinal codes, and bonsai's ordered-target-statistics preprocessing.

{cat_table}

{provenance([Evidence.CAT_TRADEOFF], "Probe: [scripts/probe_categorical.py](../../scripts/probe_categorical.py); evidence [benchmarks/categorical-tradeoff-2026-07.md](../../benchmarks/categorical-tradeoff-2026-07.md).")}

### Probe: ranking objectives (gated, issue #58)

NDCG@10 by regime; the stable gap is to listwise losses only, so issue #58 is scoped listwise-first.

{rank_table}

{provenance([Evidence.RANKING_TRADEOFF], "Probe: [scripts/probe_ranking.py](../../scripts/probe_ranking.py); evidence [benchmarks/ranking-tradeoff-2026-07.md](../../benchmarks/ranking-tradeoff-2026-07.md).")}

### Probe: CatBoost's categorical machinery priced by its own toggle (reopener predicate, decision 80)

On the cat-heavy TabArena subset, CatBoost native vs the same model with categoricals ordinal-encoded: the toggle prices the machinery at 68% of CatBoost's remaining lead over bonsai_ts (mean share -0.0099 of -0.0147); the pure-numeric control is bit-identical in both arms. Where the price is largest, ablated CatBoost loses to bonsai_ts outright.

{tap_table}

{provenance([Evidence.TABARENA_CAT], "Probe: [scripts/probe_tabarena_cat.py](../../scripts/probe_tabarena_cat.py); evidence [benchmarks/tabarena-cat-probe-2026-07.md](../../benchmarks/tabarena-cat-probe-2026-07.md). Lower is better for every metric column (error/log-loss form).")}

### Probe: ordered boosting priced by CatBoost's own toggle (declined, decision 81)

On 12 small pure-numeric datasets at matched knobs, Ordered beats Plain beyond the chance band on 0 of 12 and is distinctly worse where the toggle moves most, at 3.9x the train time. The mechanism is not the small-data edge; the campaign died at its pre-registered first stage. Lower is better in both metric columns.

{ob_table}

{provenance([Evidence.ORDERED_BOOSTING], "Probe: [scripts/probe_ordered_boosting_rung0.py](../../scripts/probe_ordered_boosting_rung0.py); evidence [benchmarks/ordered-boosting-probe-2026-07.md](../../benchmarks/ordered-boosting-probe-2026-07.md).")}

### Probe: static K-permutation target statistics (declined, decision 82)

K-averaged ordered target statistics as plain preprocessing recover a negative share of the gap to native CatBoost on the cat-heavy pool (K=8 pool mean -0.026): the average converges toward leave-one-out statistics while the single ordering's noise was implicit regularization. The categorical substance is the per-split machinery. Lower is better in every metric column.

{sk_table}

{provenance([Evidence.STATIC_K_ENCODER], "Probe: [scripts/probe_static_k_encoder.py](../../scripts/probe_static_k_encoder.py); evidence [benchmarks/static-k-encoder-probe-2026-07.md](../../benchmarks/static-k-encoder-probe-2026-07.md).")}

### Probe: a per-dataset learning-rate rule (declined, decision 83)

Even a validation-selected oracle over eight rates gains nothing on the pool (it wins validation 10 of 12 but test only 6 of 12, the overfit signature), and CatBoost's own automatic rate transplants to a no-op in a tight band around the shipped 0.05. Lower is better in the metric columns.

{lr_table}

{provenance([Evidence.LR_RULE], "Probe: [scripts/probe_lr_rule.py](../../scripts/probe_lr_rule.py); evidence [benchmarks/lr-rule-probe-2026-07.md](../../benchmarks/lr-rule-probe-2026-07.md).")}

### Probe: the bagged-protocol randomization interaction (declined, decision 85)

Decision 81's last reopener: is CatBoost's small-data lead a bagged-protocol interaction, its randomization defaults decorrelating ensemble members where bonsai's deterministic ones cannot? Priced at zero core cost on the 12-dataset pure-numeric pool. The headline interaction, (cat single minus cat bag8) minus (bonsai single minus bonsai bag8), is negative in both pool means and inside the chance band on 7 of 12; of the 5 out-of-band cases 4 favor bonsai, and the neutralization arm shows stock CatBoost randomization is null under bagging. 8-fold data-bagging already gives bonsai the decorrelation. Lower is better; positive share means the lever lowers error.

{bi_table}

{provenance([Evidence.BAGGING_INTERACTION], "Probe: [scripts/probe_bagging_interaction.py](../../scripts/probe_bagging_interaction.py); evidence [benchmarks/bagging-interaction-probe-2026-07.md](../../benchmarks/bagging-interaction-probe-2026-07.md).")}

### Probe: honest shadow-feature selection (declined, decision 86)

Does a refit-based shadow-feature selector (append a permuted copy of every column, keep only real features that beat the shadow importances) buy accuracy over plain top-k-by-gain, and how does it price against CatBoost select_features? Priced at zero core cost on two regimes (REAL-WIDE up to 1024 features, NOISE-INJECTED with shuffled-copy noise equal to each set's feature count), with the bonsai arms run under all three growers. The shadow arm's `shadow vs top-k` delta is inside the chance band on 26 of 27 grower-dataset cells and favors truncation in the 27th, so every beyond-band win it scores is a win plain truncation also scores at the same k, and it loses beyond the band on the same two low-dimensional real sets under every grower (leafwise is bit-identical to depthwise, the 63-leaf budget never binds at depth 6). Selection is not an accuracy lever on this pool; where it recovers accuracy it removes injected junk a one-line top-k already removes. The grower-independent reference arms (cat select, XGBoost) live on the depthwise rows. Lower is better in every metric column; positive `shadow vs top-k` means the shadow machinery beats truncation.

{fs_table}

{provenance([Evidence.FEATURE_SELECTION], "Probe: [scripts/probe_feature_selection.py](../../scripts/probe_feature_selection.py); evidence [benchmarks/feature-selection-probe-2026-07.md](../../benchmarks/feature-selection-probe-2026-07.md).")}

#### The selection-method survey (guide 14 worked example)

Ten selection methods, one shared judge: each method produces a feature ranking, every ranking is refit at matched knobs on its top-k down a budget ladder, and error is read on an untouched holdout (superconductivity, 81 real features, baseline rmse {fmt(sv_base.get("superconductivity"), 5)}; wide-short contrast on QSAR-TID-11, baseline rmse {fmt(sv_base.get("QSAR-TID-11"), 5)}). The tables, the pairwise top-16 overlap matrix and the readings live in [guide chapter 14](../guide/14-feature-selection.md); they add no verdict weight to decision 86.

![Selection-method survey](assets/selection-survey.svg)

{provenance([Evidence.SELECTION_SURVEY], "Survey: [scripts/probe_selection_survey.py](../../scripts/probe_selection_survey.py); readings: [guide chapter 14](../guide/14-feature-selection.md).")}
"""


# Perf: re-baseline ================================================================================

# Chart styling: (display label, color, is_cpu). bonsai's two CUDA growers get
# two greens; CPU variants are dashed everywhere. The four GPU arms are shared
# between the re-baseline and iso-volume style maps; the (variant, label)
# column lists derive from the maps, one order per table.
_GPU_STYLE = {
    "bonsai_cuda_depthwise": ("bonsai cuda dw", "#1b5e20", False),
    "bonsai_cuda_oblivious": ("bonsai cuda obl", "#4caf50", False),
    "xgb_cuda": ("xgb cuda", LIB_COLOR["xgboost"], False),
    "catboost_gpu": ("catboost gpu", LIB_COLOR["catboost"], False),
}

VARIANT_STYLE = {
    **_GPU_STYLE,
    "lgbm_cuda": ("lgbm cuda", LIB_COLOR["lightgbm"], False),
    "lgbm_cpu": ("lgbm cpu", LIB_COLOR["lightgbm"], True),
    "bonsai_oblivious": ("bonsai cpu obl", "#4caf50", True),
}

REBASE_VARIANTS = [(k, v[0]) for k, v in VARIANT_STYLE.items()]


def human(n: int) -> str:
    """250000 -> 250k, 16000000 -> 16M."""
    return f"{n // 1_000_000}M" if n >= 1_000_000 else f"{n // 1000}k"


def cell_label(rows: int, cols: int) -> str:
    """A rows x cols cell label with human row counts."""
    return f"{human(rows)} x {cols}"


def fit_r2_str(r: dict) -> str:
    """The `12.3s (.876)` cell text for one row."""
    return f"{r['fit_s']:.1f}s ({fmt(r.get('r2_test'), 3).lstrip('0')})"


def best_fit_by(rows: list[dict], key_fn) -> dict[tuple, dict]:
    """Min-fit_s row per key (best of repeats), for flat-schema rows."""
    best: dict[tuple, dict] = {}
    for r in rows:
        k = key_fn(r)
        if k not in best or r[K.FIT_S] < best[k][K.FIT_S]:
            best[k] = r
    return best


def _cell_best(rows: list[dict]) -> dict[tuple, dict]:
    """Best-fit row per (rows, cols, variant)."""
    best: dict[tuple, dict] = {}
    for r in rows:
        c = r["cell"]
        key = (c["rows"], c["cols"], r[K.VARIANT])
        if key not in best or r[K.FIT_S] < best[key][K.FIT_S]:
            best[key] = r
    return best


def _fmt_cell(best: dict, rows: int, cols: int, variant: str) -> str:
    r = best.get((rows, cols, variant))
    if r is None:
        return "-"
    return fit_r2_str(r)


def rebaseline_section() -> str:
    """The rows-scaling section of the scale page."""
    rows = load_jsonl(standings_file(Axis.ROWS))
    best = _cell_best(rows)
    host = rows[0]["host"]
    row_axis = sorted({r["cell"]["rows"] for r in rows if r["cell"]["cols"] == 100})

    rows_table = md_table(
        ["rows", *[lbl for _, lbl in REBASE_VARIANTS]],
        [[human(n), *[_fmt_cell(best, n, 100, v) for v, _ in REBASE_VARIANTS]]
         for n in row_axis])

    def series_for(cells: list[tuple[int, int]]):
        out = []
        for variant, (label, color, cpu) in VARIANT_STYLE.items():
            pts = [(x, best[(r, c, variant)][K.FIT_S])
                   for x, (r, c) in cells if (r, c, variant) in best]
            if pts:
                out.append((label, color, cpu, pts))
        return out

    line_chart(
        "rebaseline-rows.svg",
        "Fit seconds vs rows (100 features, log-log; lower is better)",
        "fit seconds",
        series_for([(n, (n, 100)) for n in row_axis]),
        x_ticks=[(n, human(n)) for n in row_axis], x_label="rows")

    return f"""### The re-baseline: fit seconds at scale

Same-pod sweep ({host['cpu_model']}, {host['gpu']}), synthetic regression, `fit()` timed end to end including each library's own ingest, best of repeats, test r² in parentheses.

![Fit seconds vs rows](assets/rebaseline-rows.svg)

Scaling rows (100 features):

{rows_table}

Width scaling has its own standings axis on [Width and shape](perf-shape.md).

{provenance([standings_file(Axis.ROWS)], "Runner: [scripts/bench_scaling.py](../../scripts/bench_scaling.py) (`python -m bonsai.bench.scaling`); README Performance derives from the same file." + measured_stamp(rows))}
"""


def xgb33_recheck_table() -> str:
    """The xgboost-3.3 recheck section (decision 87)."""
    rc = load_jsonl(Evidence.XGB33_RECHECK)
    best: dict = {}
    for r in rc:
        if r[K.STATUS] != "ok":
            continue
        ver = r["xgboost"] if r[K.VARIANT].startswith("xgb") else None
        key = (r[K.VARIANT], ver, r["cell"]["rows"], r["cell"]["cols"])
        cur = best.get(key)
        best[key] = {
            K.FIT_S: min(r[K.FIT_S], cur[K.FIT_S]) if cur else r[K.FIT_S],
            "rss": max(r[K.PEAK_RSS_GB], cur["rss"]) if cur else r[K.PEAK_RSS_GB],
        }

    CELLS = [("gpu", 1_000_000, 100), ("gpu", 4_000_000, 100),
             ("gpu", 16_000_000, 100), ("gpu", 1_000_000, 256),
             ("gpu", 1_000_000, 1024), ("cpu", 16_000_000, 100),
             ("cpu", 1_000_000, 1024), ("cpu", 1_000_000, 4096)]
    ARMS = {"gpu": ("bonsai_cuda_depthwise", "xgb_cuda"),
            "cpu": ("bonsai_depthwise", "xgb_hist")}
    body = []
    for dev, nr, nc in CELLS:
        bv, xv = ARMS[dev]
        b = best[(bv, None, nr, nc)]
        x2 = best[(xv, "3.2.0", nr, nc)]
        x3 = best[(xv, "3.3.0", nr, nc)]
        body.append([dev, human(nr), str(nc), f"{b['fit_s']:.1f}s",
                     f"{x2['fit_s']:.1f}s", f"{x3['fit_s']:.1f}s",
                     f"{x3['fit_s'] / x2['fit_s']:.2f}x",
                     f"{b['rss']:.1f}GB", f"{x3['rss']:.1f}GB"])
    table = md_table(
        ["device", "rows", "cols", "bonsai fit", "xgb 3.2 fit", "xgb 3.3 fit",
         "3.3 vs 3.2", "bonsai RSS", "xgb 3.3 RSS"], body)

    return f"""### The XGBoost 3.3 recheck (decision 87)

XGBoost 3.3 (2026-07-21) claimed lower GPU quantile-sketching memory and wide-data CPU histogram tiling, both aimed at cells bonsai competes in, so the claims above were rechecked on one pod (L40S) with three same-pod arms: bonsai at main, XGBoost 3.2.0, XGBoost 3.3.0. Every published standing survives. On GPU, 3.3 matches 3.2 within noise at every cell and host RSS does not move (22.1GB at 16M against bonsai's 6.9GB, the README's 3x memory claim reproduced on a second host). On CPU at 16M bonsai sits 6% behind `xgboost-hist`, inside the published "within ~8%, host-dependent" band. The one real improvement: 3.3 halves wide-CPU hist time at 1M x 4096 (nothing at 1024), narrowing bonsai's lead at that cell from 2.4x to 1.19x. No published cell flips; the wide standings above are GPU, where 3.3 changes nothing. Fit is best of repeats; RSS is the worst repeat; this pod's absolutes do not compare to the re-baseline table per the fleet-spread caveat.

{table}

{provenance([Evidence.XGB33_RECHECK], "Driver: `python -m bonsai.bench.scaling --worker` per cell, three arms on one pod; verdict recorded as decision 87.")}
"""


def wide_cpu_hist_table() -> str:
    """The wide-CPU histogram study section (decisions 88/89)."""
    wc = load_jsonl(Evidence.WIDE_CPU_HIST)
    ladder = {(r["tag"], r[K.VARIANT], r["rows"], r["cols"]): r for r in wc
              if r["run"] == "pod-ladder"}

    def cell(tag, variant, rows, cols, key=K.FIT_S):
        r = ladder.get((tag, variant, rows, cols))
        return f"{r[key]:.0f}s" if r and key in r else "-"

    body = []
    for variant in ("bonsai_depthwise", "bonsai_leafwise"):
        b = cell("before", variant, 131072, 16384)
        a = cell("after", variant, 131072, 16384)
        lg = cell("ref", "lgbm_cpu", 131072, 16384)
        body.append([variant.removeprefix("bonsai_"), b, a, lg])
    table = md_table(
        ["grower", "row-wise fill (before)",
         "feature-parallel (decision 88, retired by 89)", "lgbm_cpu"], body)

    return f"""### The wide-CPU fill: from a 2-6x wall to one tiled pass (decisions 88, 89)

Ultra-wide selections broke the row-wise u8 fill: its per-row scatter targets the whole selected histogram footprint (33.6MB at 16k features x 255 bins), so every add missed cache (decision 88 routed those levels feature-parallel; same-pod at 131k x 16384 the 7x leafwise deficit against LightGBM collapsed to 1.2x). Decision 89 then retired the strategy pair: the mirror moved to a column-block-tiled layout and the fill runs tiles outer, rows inner, so the live scatter target is one block's histograms at any width. The tiled fill beat both prior strategies at their own best cells in an interleaved same-pod A/B (326 vs 369s at 1M x 4096 against the row path; 442 vs 514s at 131k x 16384 against feature-parallel; a wash at 16M x 100) and produces bit-identical models at every width. The origin is a production field report (issue #217).

{table}

{provenance([Evidence.WIDE_CPU_HIST], "Evidence: [benchmarks/wide-cpu-hist-2026-07.md](../../benchmarks/wide-cpu-hist-2026-07.md); verdict recorded as decision 88.")}
"""


def cuda_wide_recheck_table() -> str:
    """The stale-wide-reading correction section (decision 90)."""
    cw = load_jsonl(Evidence.CUDA_WIDE_RECHECK)
    body = []
    for r in sorted(cw, key=lambda r: (-r["cols"], r[K.VARIANT])):
        body.append([r[K.VARIANT], human(r["rows"]), str(r["cols"]),
                     f'{r[K.FIT_S]:.1f}s', fmt(r[K.R2_TEST]),
                     f'{r[K.PEAK_RSS_GB]:.1f}GB'])
    table = md_table([K.VARIANT, "rows", "cols", "fit", "test r²", "peak RSS"], body)

    return f"""### The CUDA wide recheck: the wall was already gone (decision 90)

A campaign to close the recorded ~5x wide-GPU gap to XGBoost closed at stage 0: the gap no longer exists. The recorded numbers dated to 2026-07-08 code, before the device-resident line landed; on current main, one pod, bonsai's CUDA growers lead every wide cell against both references at 3-4x less host memory. The stale reading ("CatBoost keeps the wide lead") is corrected wherever it appeared; the six-variant cols re-baseline below completes the recorded follow-up.

{table}

{provenance([Evidence.CUDA_WIDE_RECHECK], "Same pod (L40S, US-NC-1, 2026-07-30), SCALING knobs; verdict recorded as decision 90.")}
"""


def cols_rebaseline_table() -> str:
    """The width-standings section of the shape page."""
    cr = [r for r in load_jsonl(standings_file(Axis.WIDTH)) if r[K.STATUS] == "ok"]
    best = best_fit_by(
        cr, lambda r: (r["cell"]["rows"], r["cell"]["cols"], r[K.VARIANT]))
    cells = sorted({(r["cell"]["rows"], r["cell"]["cols"]) for r in cr},
                   key=lambda rc: rc[1])
    label = cell_label

    def cell_fmt(rows, cols, variant):
        r = best.get((rows, cols, variant))
        return "-" if r is None else fit_r2_str(r)

    fit_table = md_table(
        ["cell", *[lbl for _, lbl in REBASE_VARIANTS]],
        [[label(nr, nc), *[cell_fmt(nr, nc, v) for v, _ in REBASE_VARIANTS]]
         for nr, nc in cells])
    rss_table = md_table(
        ["cell", *[lbl for _, lbl in REBASE_VARIANTS]],
        [[label(nr, nc),
          *[f"{best[(nr, nc, v)]['peak_rss_gb']:.1f}GB"
            if (nr, nc, v) in best else "-" for v, _ in REBASE_VARIANTS]]
         for nr, nc in cells])

    series = []
    for variant, (lbl, color, cpu) in VARIANT_STYLE.items():
        pts = [(nc, best[(nr, nc, variant)][K.FIT_S])
               for nr, nc in cells if (nr, nc, variant) in best]
        if pts:
            series.append((lbl, color, cpu, pts))
    line_chart(
        "cols-rebaseline.svg",
        "Fit seconds vs features (1M rows; 16384-col cell 131k rows; log-log)",
        "fit seconds",
        series,
        x_ticks=[(nc, f"{nc}" if nc < 16_384 else "16384*") for _, nc in cells],
        x_label="features")

    return f"""### The cols re-baseline: wide standings on current main (decision 90 follow-up)

The six-variant cols-axis re-baseline promised by decision 90. bonsai's CUDA growers hold the fastest slot at every measured width, at a fraction of the reference libraries' peak host memory; the tables below carry the current numbers. The widest cell drops to 131k rows to hold total cells at 2^31, so its column is not comparable to the 1M-row columns (starred in the chart). The CPU reference arms bound the GPU advantage: bonsai CPU and LightGBM trade the widest-cell lead within a rep's noise while the GPU growers are several times faster than either.

![Fit seconds vs features, re-baseline](assets/cols-rebaseline.svg)

Fit seconds (test r²), best of reps:

{fit_table}

Peak host RSS, worst rep:

{rss_table}

{provenance([standings_file(Axis.WIDTH)], "One pod, SCALING knobs, GPU arms 2 reps / CPU arms 1; supersedes the July 8 study's wide cells." + measured_stamp(cr))}
"""


# Perf: iso-volume =================================================================================

ISO_STYLE = {
    **_GPU_STYLE,
    "bonsai_depthwise": ("bonsai cpu dw", "#1b5e20", True),
    "xgb_hist": ("xgb hist", LIB_COLOR["xgboost"], True),
}

ISO_VARIANTS = [(k, v[0]) for k, v in ISO_STYLE.items()]

ISO_HOST = _STANDINGS_REG["shape"]["host"]


def _iso_cells_for(best: dict, run: str) -> list[tuple[int, int]]:
    """The (rows, cols) cells measured in one run, narrowest first."""
    return sorted({(k[1], k[2]) for k in best if k[0] == run},
                  key=lambda rc: rc[1])


def _iso_tables(best: dict, run: str, variants: list) -> tuple[str, str]:
    """(fit table, device-memory table) for one iso-volume run."""
    def fit_cell(nr, nc, v):
        r = best.get((run, nr, nc, v))
        return "-" if r is None else fit_r2_str(r)

    def vram_cell(nr, nc, v):
        r = best.get((run, nr, nc, v))
        dm = (r or {}).get("dev_mem") or {}
        gb = dm.get("peak_gb_pid")
        return f"{gb:.1f}GB" if gb is not None else "-"

    cs = _iso_cells_for(best, run)
    fit = md_table(["cell", *[lbl for _, lbl in variants]],
                   [[cell_label(nr, nc),
                     *[fit_cell(nr, nc, v) for v, _ in variants]]
                    for nr, nc in cs])
    vram = md_table(["cell", *[lbl for v, lbl in variants
                               if not ISO_STYLE[v][2]]],
                    [[cell_label(nr, nc),
                      *[vram_cell(nr, nc, v) for v, _ in variants
                        if not ISO_STYLE[v][2]]]
                     for nr, nc in cs])
    return fit, vram


def _iso_charts(best: dict, run31: str):
    """The two iso-volume charts over the 2^31 iso-line of one run."""
    iso_cells = [(nr, nc) for nr, nc in _iso_cells_for(best, run31)
                 if nr * nc == 1 << 31]
    series_fit, series_vram = [], []
    for v, (lbl, color, cpu) in ISO_STYLE.items():
        pts = [(nc, best[(run31, nr, nc, v)][K.FIT_S])
               for nr, nc in iso_cells if (run31, nr, nc, v) in best]
        if pts:
            series_fit.append((lbl, color, cpu, pts))
        if not cpu:
            mpts = [(nc, (best[(run31, nr, nc, v)].get("dev_mem") or {})
                     .get("peak_gb_pid"))
                    for nr, nc in iso_cells if (run31, nr, nc, v) in best]
            mpts = [(x, y) for x, y in mpts if y is not None]
            if mpts:
                series_vram.append((lbl, color, cpu, mpts))
    ticks = [(nc, str(nc)) for _, nc in iso_cells]
    line_chart("iso-volume-fit.svg",
               "Fit seconds vs cols at constant rows x cols = 2^31 (log-log)",
               "fit seconds", series_fit, x_ticks=ticks,
               x_label="features (rows x cols = 2^31)")
    line_chart("iso-volume-vram.svg",
               "Measured peak device memory vs cols at 2^31 cells (log-log)",
               "peak GB", series_vram, x_ticks=ticks,
               x_label="features (rows x cols = 2^31)")


def iso_volume_section() -> str:
    """The decision-91 shape-frontier section of the width/shape page."""
    rows = load_jsonl(standings_file(Axis.SHAPE))
    pod = [r for r in rows if r["host"]["name"] == ISO_HOST]
    errors = [r for r in pod if r[K.STATUS] != "ok"]
    best = best_fit_by(
        [r for r in pod if r[K.STATUS] == "ok"],
        lambda r: (r["run"], r["cell"]["rows"], r["cell"]["cols"],
                   r[K.VARIANT]))
    label = cell_label
    run31, run33 = "iso-volume-2026-08-pod", "iso-volume-33-2026-08-pod"
    fit31, vram31 = _iso_tables(best, run31, ISO_VARIANTS)
    fit33, vram33 = _iso_tables(best, run33, ISO_VARIANTS[:4])
    _iso_charts(best, run31)

    err_note = ""
    if errors:
        e = errors[0]
        gb = (e.get("dev_mem") or {}).get("peak_gb_pid")
        c = e["cell"]
        err_note = (f" The one failure is data: xgb_cuda died at "
                    f"{label(c['rows'], c['cols'])} on both attempts, the "
                    f"sampler recording {gb:.1f}GB of device memory at death.")

    return f"""### The iso-volume shape frontier (decision 91)

Constant data volume, swept aspect ratio: every cell of the primary ladder holds rows x cols at 2^31 (an 8GiB float32 matrix) while cols runs 128 to 65536, so costs that scale with total cells stay flat and whatever rises is paying for width. Measured peak device memory (`dev_mem`, NVML-sampled while the child runs, gates off) is an output, not an estimate. One pod: RTX PRO 6000 Blackwell Workstation Edition (96GB, 64 vCPU, 1.1TB RAM, sync probe 4.5us/op), threads 16.

bonsai's CUDA growers are fastest at every cell of both ladders and their fit time is nearly flat across the tall half of the iso-line (7.2s at 16M x 128 to 9.3s at 1M x 2048) where both references vary 1.5-2x; every arm rises together past 8192 cols as histogram cost (cols x bins) takes over. Device memory separates harder than time: bonsai peaks at 3.4GB where XGBoost holds 18.9GB, and CatBoost allocates 90.2GB (the whole card) at every cell including 1M x 100, so it never fails but never shares the device.{err_note} At the widest aspect (32k x 65536, where p is 2x n) the oblivious grower keeps test r2 at .873 while depthwise falls to .815, the symmetric tree's regularization showing at extreme width. On the 2^33 stretch (a 32GiB matrix, GPU arms only) bonsai leads 4.1x over XGBoost at 67M x 128 (27.9 vs 113.8s) at 6.3x less device memory (11.7 vs 73.6GB).

![Fit seconds vs cols, iso-volume](assets/iso-volume-fit.svg)

![Peak device memory vs cols, iso-volume](assets/iso-volume-vram.svg)

Fit seconds (test r2), best of reps, 2^31 cells plus the 1M x 100 anchor:

{fit31}

Measured peak device memory (per-process, worst rep is within sampling noise of best), 2^31 cells plus the 1M x 100 anchor:

{vram31}

The 2^33 stretch, GPU arms:

{fit33}

{vram33}

{provenance([standings_file(Axis.SHAPE)], "Specs: bundled in [bench/specs/](../../python/bonsai/bench/specs/); driver: [scripts/pod_bench_driver.sh](../../scripts/pod_bench_driver.sh); evidence: [benchmarks/iso-volume-2026-08.md](../../benchmarks/iso-volume-2026-08.md); verdict recorded as decision 91." + measured_stamp(pod))}
"""


# Perf: the remaining tracks =======================================================================


def prefetch_section() -> str:
    """The CPU prefetch-tie section (decision 61)."""
    pre = load_jsonl(Evidence.CPU_PREFETCH)
    pre_best = _cell_best(pre)
    pre_rows = sorted({k[0] for k in pre_best})
    pre_variants = sorted({k[2] for k in pre_best})
    pre_table = md_table(
        ["rows", *pre_variants],
        [[f"{n:,}", *[_fmt_cell(pre_best, n, 100, v) for v in pre_variants]]
         for n in pre_rows])

    return f"""### CPU 16M round (the prefetch tie)

{pre_table}

{provenance([Evidence.CPU_PREFETCH], "Decision 61: software prefetch closed the 16M CPU gap to XGBoost-hist on this pod.")}
"""


def _pareto_chart(pareto: list[dict], par_variants: list[str]):
    """The accuracy-vs-time chart; ticks derive from the data so a refresh
    with faster fits or higher accuracy cannot strand them outside the
    plotted range."""
    par_series = []
    par_labels = []
    for v in par_variants:
        label, color, cpu = VARIANT_STYLE.get(v, (v, TEXT, False))
        pts = sorted((r[K.FIT_S], r[K.R2_TEST]) for r in pareto if r[K.VARIANT] == v)
        par_series.append((label, color, cpu, pts))
        for r in pareto:
            if r[K.VARIANT] == v:
                par_labels.append((r[K.FIT_S], r[K.R2_TEST],
                                   str(r["cell"]["iters"])))
    fmin, fmax = (min(r[K.FIT_S] for r in pareto), max(r[K.FIT_S] for r in pareto))
    x_step = 10 if fmax - fmin > 25 else 5
    r2min, r2max = (min(r[K.R2_TEST] for r in pareto),
                    max(r[K.R2_TEST] for r in pareto))
    line_chart(
        "gpu-pareto-16M.svg",
        "16M rows: accuracy vs fit time by iteration count (up-left is better)",
        "test r2",
        par_series,
        x_ticks=[(s, f"{s}s") for s in range(0, int(fmax) + 1, x_step)
                 if fmin <= s <= fmax],
        x_label="fit seconds",
        log_x=False, log_y=False, height=420,
        y_ticks=[(v / 100, f"{v / 100:.2f}")
                 for v in range(math.ceil(r2min * 100), int(r2max * 100) + 1, 2)],
        point_labels=par_labels)


def leafwise_recheck_section() -> str:
    """The decision-42 recheck: CPU leafwise vs LightGBM CUDA at scale."""
    rows = [r for r in load_jsonl(Evidence.LEAFWISE_RECHECK)
            if r[K.STATUS] == "ok"]
    best = _cell_best(rows)
    order = ["bonsai_leafwise", "lgbm_cuda", "lgbm_cpu",
             "bonsai_cuda_depthwise"]
    labels = {"bonsai_leafwise": "bonsai leafwise (cpu)",
              "lgbm_cuda": "lgbm cuda", "lgbm_cpu": "lgbm cpu",
              "bonsai_cuda_depthwise": "bonsai cuda dw (anchor)"}
    scales = sorted({k[0] for k in best})
    table = md_table(
        ["rows", *[labels[v] for v in order]],
        [[human(n), *[_fmt_cell(best, n, 100, v) for v in order]]
         for n in scales])
    return f"""### The leafwise recheck: decision 42's claim at scale (decision 95)

The decision-42-era reading ("bonsai CPU leafwise beats LightGBM's CUDA leaf-wise") was measured at 464k rows; on one pod at ladder scales it inverts, monotonically, to 5.3x in LightGBM's favor at 16M. The engineering conclusion stands the other way up: leaf-wise on the GPU is viable at scale (LightGBM ships it), and bonsai's leafwise grower is now the only grower without device support (issue #268). The anchor arm ties this pod to the ladders; LightGBM-CUDA's higher r2 column is the depth-cap artifact recorded in decision 95.

{table}

{provenance([Evidence.LEAFWISE_RECHECK], "One pod (L40S, US-MO-1, 2026-08-01), SCALING knobs at 100 iters, best of 2 reps; evidence for decision 95.")}
"""


def leafwise_cadence_section() -> str:
    """The device-leafwise stage 0 overhead probe (issue #268)."""
    cad = load_json(Evidence.LEAFWISE_CADENCE)
    configs = cad["configs"]
    table = md_table(
        ["config", "seconds", "us/round"],
        [[name, f"{c['seconds']:.3f}", f"{c['us_per_round']:.1f}"]
         for name, c in configs.items()])
    b = cad["buckets_us"]
    return f"""### The device-leafwise cadence probe (stage 0 of doc 20's admission)

Before building the device leafwise plane, its per-round fixed cost was priced by replaying the round skeleton ({cad["rounds"]:,} rounds: launches, two pinned syncs, staging, host heap ops) with trivial kernels on an L40S. The fixed cost F is the no-copyback config: **{cad["fixed_cost_us"]:.1f} us/round** against a {cad["budget_us"]:.0f} us budget and a {cad["kill_us"]:.0f} us kill line, so the one-node-per-round design proceeds. Buckets: launch and staging floor {b["launch_floor"]:.1f}, syncs {b["syncs"]:.1f}, grid width {b["grid_width"]:.1f} (the fixed cost does not track node size), and the data-proportional range copy-back {b["copyback"]:.1f} us/round excluded from F.

{table}

{provenance([Evidence.LEAFWISE_CADENCE], "One pod (L40S, US-MO-1, 2026-08-01); probe [experiments/leafwise_cadence/cadence.cu](../../experiments/leafwise_cadence/cadence.cu); evidence [benchmarks/leafwise-cadence-2026-08.md](../../benchmarks/leafwise-cadence-2026-08.md) for issue #268.")}
"""


_LADDER_LABELS = {"bonsai_cuda_leafwise": "bonsai cuda leafwise",
                  "lgbm_cuda": "lgbm cuda",
                  "bonsai_leafwise": "bonsai leafwise (cpu)",
                  "bonsai_cuda_depthwise": "bonsai cuda dw (anchor)"}

# The uncapped-depth arm records max_depth at the uint8 ceiling, which a
# 256-leaf budget can never reach; the row key separates it from the capped
# cell of the same shape.
UNCAPPED_DEPTH = 255


def leafwise_ladder_section() -> str:
    """The device-leafwise admission ladder, stage 2 of doc 20 (issue #268)."""
    rows = [r for r in load_jsonl(Evidence.LEAFWISE_LADDER)
            if r[K.STATUS] == "ok"]
    order = list(_LADDER_LABELS)
    capped = _cell_best([r for r in rows
                         if r["cell"]["depth"] != UNCAPPED_DEPTH])
    scales = sorted({k[0] for k in capped})

    def fit_of(n: int, variant: str) -> float:
        return capped[(n, 100, variant)][K.FIT_S]

    top = scales[-1]
    margin = 1 - fit_of(top, "bonsai_cuda_leafwise") / fit_of(top, "lgbm_cuda")
    over_cpu = [fit_of(n, "bonsai_leafwise") / fit_of(n, "bonsai_cuda_leafwise")
                for n in scales]
    table = md_table(
        ["rows", *[_LADDER_LABELS[v] for v in order]],
        [[human(n), *[_fmt_cell(capped, n, 100, v) for v in order]]
         for n in scales])
    return f"""### The device-leafwise ladder: admission by measurement (stage 2)

Same pod, four arms, best of two reps, interleaved. The kill criterion pre-registered in issue #268 was one number: beat LightGBM's CUDA leaf-wise at 16M x 100 on the same pod, or the grower does not register. It is met with {margin:.0%} to spare at 16M, and `cuda_leafwise` beats its own CPU arm at every cell, {min(over_cpu):.1f}x to {max(over_cpu):.1f}x. The anchor row prices what is left: the depthwise plane moves the same histogram volume in less time, because serializing the frontier to one node per round costs launches that the level plane batches away.

{table}

{_uncapped_table(rows)}

{provenance([Evidence.LEAFWISE_LADDER], "One pod (L40S, US-NC-1, 2026-08-01), SCALING knobs at 100 iters and 256 leaves, best of 2 reps; the ladder that admitted `cuda_leafwise` under [docs/architecture/20-cuda-leafwise.md](../../docs/architecture/20-cuda-leafwise.md). Absolute times run above the 2026-08-01 recheck, which used a different pod with a wider CPU; only same-pod comparisons are meaningful.")}
"""


def _uncapped_table(rows: list[dict]) -> str:
    """The uncapped-depth arm, where leaf-wise structurally differs."""
    best = _cell_best([r for r in rows
                       if r["cell"]["depth"] == UNCAPPED_DEPTH])
    order = ["bonsai_cuda_leafwise", "lgbm_cuda", "bonsai_leafwise"]
    table = md_table(
        ["16M x 100, no depth cap", K.FIT_S, "test r2"],
        [[_LADDER_LABELS[v], f"{best[(16_000_000, 100, v)][K.FIT_S]:.1f}s",
          fmt(best[(16_000_000, 100, v)][K.R2_TEST], 4)] for v in order])
    return f"""The capped ladder above hides the strategy's point: at a 256-leaf budget and a depth-8 cap the leaf budget and the full tree are the same tree, so every arm returns the depthwise accuracy. Lifting the cap is where leaf-wise differs, and it is also the only comparison against LightGBM that is honest, since its CUDA learner ignores `max_depth` outright (decision 95, confirmed in its source: `CUDASingleGPUTreeLearner::Train` never calls the one site that enforces depth). Uncapped, bonsai's leaf-wise growers reach LightGBM's accuracy to four decimals, which is the reading the capped rows' r2 column was never measuring. LightGBM is faster here: uncapped best-first must find a split for every leaf it creates, doubling bonsai's per-round count from 25,500 to 51,100, and that is the optimization target the admission leaves open rather than a quality difference.

{table}
"""


def frontier_section() -> str:
    """The accuracy-vs-time frontier page body."""
    pareto = [r for r in load_jsonl(standings_file(Axis.FRONTIER))
              if r[K.STATUS] == "ok"]
    par_variants = []
    for r in pareto:
        if r[K.VARIANT] not in par_variants:
            par_variants.append(r[K.VARIANT])
    par_table = md_table(
        [K.VARIANT, "iters", K.FIT_S, "test r2"],
        [[r[K.VARIANT], str(r["cell"]["iters"]), fmt(r[K.FIT_S], 2),
          fmt(r[K.R2_TEST], 4)] for r in pareto])
    _pareto_chart(pareto, par_variants)

    edge = load_jsonl(Evidence.CATBOOST_SCALE_EDGE)

    def detail(r):
        if "iters" in r:
            return f"iters={r['iters']}"
        if "n_samples" in r:
            return f"bin samples={r['n_samples']}"
        return "-"

    edge_table = md_table(
        ["door", "rows", "learner", "knob", K.FIT_S, "test r2"],
        [[r["door"], f"{r['rows']:,}", r["learner"], detail(r),
          fmt(r[K.FIT_S], 2), fmt(r[K.R2_TEST], 4)] for r in edge])

    return f"""### GPU accuracy-vs-time frontier at 16M

![Accuracy vs fit time at 16M rows](assets/gpu-pareto-16M.svg)

{par_table}

{provenance([standings_file(Axis.FRONTIER)], "bonsai is first to every measured accuracy across the grid (terminal accuracies tie within the noise band) and its marginal round cost stays below CatBoost's on the same pod. Evidence: [benchmarks/gpu-pareto-16M-2026-07.md](../../benchmarks/gpu-pareto-16M-2026-07.md)." + measured_stamp(pareto))}

### Ordered boosting at scale (CatBoost door)

The probe behind decisions 62 to 64: CatBoost's Ordered vs Plain modes against bonsai oblivious as rows grow.

{edge_table}

{provenance([Evidence.CATBOOST_SCALE_EDGE], "Evidence: [benchmarks/catboost-scale-edge-2026-07.md](../../benchmarks/catboost-scale-edge-2026-07.md).")}
"""


# assembly =========================================================================================

GEN_NOTE = ("<!-- GENERATED by scripts/render_results.py. "
            "Edit the generator, not this page. -->")

HEADER = GEN_NOTE + """

# The results ledger

Every results file behind a published claim is rendered across the pages below, generated straight from the data in [`benchmarks/results/`](../../benchmarks/results): `python3 scripts/render_results.py` rewrites them and CI fails on drift. Rows are as-run records under the [benchmark protocol](benchmark-protocol.md): quality division numbers never cite timing, perf division numbers name their timing mode, and superseded files are deleted rather than kept beside their replacements, so what is here is the current evidence, whole.
"""



def airline_section() -> str:
    """The airline (real-data, binary) page body."""
    rows = [r for r in load_jsonl(standings_file(Axis.AIRLINE)) if r[K.STATUS] == "ok"]

    def cell(variant, size, depth):
        m = [r for r in rows if r[K.VARIANT] == variant and r["size"] == size
             and r["knobs"]["depth"] == depth]
        return f"{m[0]['fit_s']:.1f}s / {m[0]['auc_test']:.4f}" if m else "-"

    variants = []
    for r in rows:
        if r[K.VARIANT] not in variants:
            variants.append(r[K.VARIANT])

    tables = []
    for depth, label in ((8, "campaign knobs (depth 8)"),
                         (10, "Pafka protocol (depth 10)")):
        tables.append(f"**{label}**, fit seconds / test AUC:\n\n" + md_table(
            [K.VARIANT, "0.1m", "1m", "10m"],
            [[v, cell(v, "0.1m", depth), cell(v, "1m", depth),
              cell(v, "10m", depth)] for v in variants]))

    def lib_of(v):
        return ("bonsai" if v.startswith("bonsai")
                else "xgb" if v.startswith("xgb")
                else "lgbm" if v.startswith("lgbm") else "catboost")

    bars = sorted(((r[K.VARIANT], r[K.FIT_S], f"AUC {r['auc_test']:.4f}")
                   for r in rows
                   if r["size"] == "10m" and r["knobs"]["depth"] == 8),
                  key=lambda t: t[1])
    bar_chart("airline-10m.svg", "airline 10M rows: fit seconds (depth 8, one pod)",
              [(v, s, note) for v, s, note in bars],
              max(s for _v, s, _n in bars),
              "bonsai_ts_* = OrderedTargetEncoder pipeline (encode time included); "
              "all rows same-pod L40S")
    return f"""## Airline delays: the real-data speed ladder

The benchm-ml airline ladder (0.1M/1M/10M rows, mixed categorical/numeric, AUC), both the campaign knob shape and Pafka's depth-10 protocol, all rows one pod. `bonsai_ts_*` rows are the labeled exception to the uniform ordinal-code convention (OrderedTargetEncoder pipeline; `fit_s` includes the encode).

![airline 10M fit seconds](assets/airline-10m.svg)

{tables[0]}

{tables[1]}

{provenance([standings_file(Axis.AIRLINE)], "A bonsai variant has the best AUC in every cell under both protocols, and bonsai CUDA depthwise is also the fastest fit from 1M rows up under ordinal encoding; XGBoost-GPU keeps only the smallest cell. Evidence: [benchmarks/airline-2026-07.md](../../benchmarks/airline-2026-07.md)." + measured_stamp(rows))}
"""



def ceiling_section() -> str:
    """The single-card ceiling section."""
    raw = load_jsonl(Evidence.SINGLE_CARD_CEILING)
    meta = raw[0]["meta"]
    rows = raw[1:]
    table = md_table(
        ["rows", "train() wall", "peak device mem", "throughput", "train r2 (1M sample)"],
        [[f"{r['rows'] // 1_000_000}M", f"{r['fit_s']:.0f}s",
          f"{r['peak_dev_mib'] / 1024:.1f} GiB",
          f"{r['rows_per_s'] / 1e6:.1f}M rows/s", fmt(r["r2_train_1m"], 4)]
         for r in rows])
    prov = provenance(
        [Evidence.SINGLE_CARD_CEILING],
        "Single-pod ladder (2026-07-18, " + meta["gpu"] + "): a 500M x 100 "
        "float32 matrix trains end to end on one 80GB card at 69.9 GiB peak, "
        "60 rounds in 8.5 minutes, with the device-resident objective keeping "
        "the fit loop bus-free. Evidence: "
        "[benchmarks/single-card-ceiling-2026-07.md]"
        "(../../benchmarks/single-card-ceiling-2026-07.md).")
    return "## The single-card ceiling\n\n" + table + "\n\n" + prov + "\n"

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


# Per-suite pages under docs/method/results/, perf division first. Suite pages
# keep their historical section headings verbatim so anchor slugs survive.
PAGES: list[tuple[str, str, str, list]] = [
    ("perf-scale.md", "Fit at scale",
     "Row-scale standings: the re-baseline, the XGBoost 3.3 recheck, and "
     "the CPU prefetch round.",
     [rebaseline_section, xgb33_recheck_table, prefetch_section,
      leafwise_recheck_section, leafwise_cadence_section,
      leafwise_ladder_section]),
    ("perf-shape.md", "Width and shape",
     "The wide-data arc: the CPU fill, the CUDA recheck, the cols "
     "re-baseline, and the iso-volume shape frontier with measured VRAM.",
     [wide_cpu_hist_table, cuda_wide_recheck_table, cols_rebaseline_table,
      iso_volume_section]),
    ("perf-frontier.md", "The accuracy-time frontier",
     "Accuracy versus fit time at 16M rows, plus the ordered-boosting door.",
     [frontier_section]),
    ("perf-airline.md", "Airline delays",
     "The benchm-ml real-data speed ladder at 0.1M, 1M, and 10M rows.",
     [airline_section]),
    ("perf-ceiling.md", "The single-card ceiling",
     "A 500M x 100 matrix trained end to end on one 80GB card.",
     [ceiling_section]),
    ("quality-grinsztajn.md", "Grinsztajn standings",
     "The only citable standings: 55 third-party tasks, both knob brackets.",
     [grinsztajn_section]),
    ("quality-campaign.md", "Campaign smoke",
     "Ten datasets at matched knobs, the fast local regression check.",
     [campaign_section]),
    ("quality-probes.md", "Probes",
     "Every feature admitted or declined by measurement, with its evidence.",
     [probes_section]),
    ("code-metrics.md", "The code division",
     "Self-measurement of the tree: lines, complexity, surface counts.",
     [code_metrics_section]),
]

# Suite pages live one level below docs/method/, so their relative links
# gain one step. Replacement order matters: the two-step rule cannot touch
# ../guide/ and runs first.
_REROOT = [("](../../", "](../../../"), ("](../guide/", "](../../guide/"),
           ("](assets/", "](../assets/"), ("](benchmark-protocol.md", "](../benchmark-protocol.md")]


def _reroot(body: str) -> str:
    """Repo-relative links rewritten for the per-suite page depth."""
    for old, new in _REROOT:
        body = body.replace(old, new)
    return body


def _division_summaries() -> tuple[str, str]:
    """(perf, quality) headline paragraphs, digits computed from standings."""
    rb = load_jsonl(standings_file(Axis.ROWS))
    fit = {}
    rss = {}
    for r in rb:
        c = r["cell"]
        if c["rows"] != 16_000_000 or r.get(K.FIT_S) is None:
            continue
        v = r[K.VARIANT]
        fit[v] = min(fit.get(v, r[K.FIT_S]), r[K.FIT_S])
        rss[v] = max(rss.get(v, 0.0), r[K.PEAK_RSS_GB])
    b_fit = min(fit[v] for v in fit if v.startswith("bonsai_cuda"))
    b_rss = min(rss[v] for v in rss if v.startswith("bonsai_cuda"))
    iso = load_jsonl(standings_file(Axis.SHAPE))
    dev = {}
    for r in iso:
        c = r["cell"]
        if (c["rows"], c["cols"]) != (16_777_216, 128) or r[K.STATUS] != "ok":
            continue
        gb = (r.get("dev_mem") or {}).get("peak_gb_pid")
        if gb is not None:
            dev[r[K.VARIANT]] = max(dev.get(r[K.VARIANT], 0.0), gb)
    perf = (
        f"bonsai's CUDA growers hold the fastest slot at every measured row "
        f"scale ({b_fit:.1f}s at 16M rows against XGBoost-GPU's "
        f"{fit['xgb_cuda']:.1f}s) at {b_rss:.1f}GB peak host memory against "
        f"XGBoost's {rss['xgb_cuda']:.1f}GB and CatBoost's "
        f"{rss['catboost_gpu']:.1f}GB. On the narrow airline shape bonsai "
        f"holds both best AUC and fastest fit from 1M rows up. "
        f"The 2026-07-30 studies hold every "
        f"width and aspect ratio, with measured device memory that sizes to "
        f"the problem: {dev['bonsai_cuda_depthwise']:.1f}GB at 16M x 128 at "
        f"constant 2^31-cell volume against XGBoost's "
        f"{dev['xgb_cuda']:.1f}GB and CatBoost's "
        f"{dev['catboost_gpu']:.1f}GB. Every number is same-pod; "
        f"identical-model GPUs across the rental fleet measure up to "
        f"~25% apart.")
    table, _, n = _standings(load_jsonl(standings_file(Axis.GRINSZTAJN)))
    lead_lib, lead_mean, lead_wins = table[0]
    quality = (
        f"{lead_lib} leads the {n}-task Grinsztajn standings at mean rank "
        f"{lead_mean:.2f} with {lead_wins} outright wins; the one knob that "
        f"translates ambiguously is bracketed in both directions on the "
        f"standings page. The campaign smoke is the fast local regression "
        f"check, and the probe archive records every feature declined by "
        f"measurement.")
    return perf, quality


def _page_table(entries: list[tuple[str, str, str, list]]) -> str:
    """The landing page's navigation table."""
    return md_table(
        ["page", "what it holds"],
        [[f"[{title}](results/{rel})", desc] for rel, title, desc, _ in entries])


def landing_page() -> str:
    """The results landing: division summaries plus navigation."""
    perf, quality = _division_summaries()
    perf_pages = [p for p in PAGES if p[0].startswith("perf-")]
    quality_pages = [p for p in PAGES if p[0].startswith("quality-")]
    return f"""{HEADER}
## Perf division

{perf}

{_page_table(perf_pages)}

## Quality division

{quality}

{_page_table(quality_pages)}

## The code division

Self-measurement of the bonsai tree, no comparative claim: [code metrics](results/code-metrics.md).
"""


README_PATH = REPO / "README.md"
MARK_BEGIN = "<!-- standings:begin (generated by scripts/render_results.py) -->"
MARK_END = "<!-- standings:end -->"

_SITE = "https://daniel-m-campos.github.io/bonsai"
_LIB_NAMES = {"lgbm": "lightgbm", "xgb": "xgboost"}


def readme_standings_block() -> str:
    """The README's digit surface, generated so it cannot drift (decision
    92): division summaries plus the two tables, fastest per row in bold."""
    perf, _quality = _division_summaries()

    rb = load_jsonl(standings_file(Axis.ROWS))
    best = _cell_best(rb)
    scales = sorted({r["cell"]["rows"] for r in rb if r["cell"]["cols"] == 100})
    lines = ["| rows | " + " | ".join(lbl for _, lbl in REBASE_VARIANTS) + " |",
             "|---|" + "--:|" * len(REBASE_VARIANTS)]
    for n in scales:
        cells, fits = [], []
        for v, _lbl in REBASE_VARIANTS:
            r = best.get((n, 100, v))
            fits.append(r[K.FIT_S] if r else float("inf"))
            cells.append(_fmt_cell(best, n, 100, v))
        k = fits.index(min(fits))
        secs, rest = cells[k].split("s (", 1)
        cells[k] = f"**{secs}s** ({rest}"
        lines.append(f"| {human(n)} | " + " | ".join(cells) + " |")
    rows_table = "\n".join(lines)

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

Same-pod re-baseline ladder, best of repeats, test r² in parentheses, fastest per row in bold.{measured_stamp(rb)}

{rows_table}

The width, shape, and accuracy-time frontier tables live in [the ledger]({_SITE}/method/results/).

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
    (sha_partial entries tolerate provenance-less rows, never a WRONG sha)."""
    reg = json.loads((REPO / "benchmarks" / "standings.json").read_text())
    reg.pop("_", None)
    errors = []
    for axis, e in reg.items():
        path = RESULTS / e["file"]
        if not path.exists():
            errors.append(f"standings {axis}: {e['file']} does not exist")
            continue
        if e["file"] not in consumed_files:
            errors.append(f"standings {axis}: {e['file']} is not rendered")
        if not e.get("sha"):
            continue
        rows = [json.loads(ln) for ln in path.read_text().splitlines()
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
