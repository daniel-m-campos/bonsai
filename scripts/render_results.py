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
import pathlib
import subprocess
import sys
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
OUT = REPO / "docs" / "method" / "results.md"

consumed: set[str] = set()


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
    return f"""## Perf division

### The re-baseline: fit seconds at scale

Same-pod sweep ({host['cpu_model']}, {host['gpu']}), synthetic regression, `fit()` timed end to end including each library's own ingest, best of repeats, test r² in parentheses.

Scaling rows (100 features):

{rows_table}

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

{par_table}

{provenance(["gpu-pareto-16M-2026-07.jsonl"], "Evidence: [benchmarks/gpu-pareto-16M-2026-07.md](../../benchmarks/gpu-pareto-16M-2026-07.md) (banner-annotated; pre-dates the decision 63 oblivious fix).")}

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
    if "--check" in sys.argv:
        if OUT.read_text() != text:
            print("ERROR: docs/method/results.md is stale; run "
                  "python3 scripts/render_results.py", file=sys.stderr)
            return 1
        print("results ledger: in sync")
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO)} ({len(text.splitlines())} lines, "
          f"{len(consumed)} data files consumed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
