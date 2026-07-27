#!/usr/bin/env python3
"""Rebuild benchmarks/selection-research-2026-07.md from the raw jsonl rows.

    python3 scripts/render_np_report.py

The report is a reading document: a research line on when feature selection
can be trusted, run in-session (2026-07-26/27) and deliberately kept out of
the guide. Narrative lives here; every number is recomputed from
benchmarks/results/*.jsonl on each run, so the tables cannot drift from the
data. Stdlib only.
"""
import json
import pathlib
import statistics as st
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
OUT = REPO / "benchmarks" / "selection-research-2026-07.md"


def load(name):
    return [json.loads(x) for x in (RESULTS / name).read_text().splitlines()
            if x.strip()]


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("--" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


S = []


def para(*lines):
    S.extend(lines)
    S.append("")


para("# When can you trust feature selection? A research record (2026-07)")
para("Four linked measurements on one question: when does picking the top-k "
     "features by an importance ranking actually work, and what breaks it? "
     "The prompt was a practical setting: market microstructure data with a "
     "markout target, thousands of features, millions of rows that are far "
     "from independent, and relationships that decay. This is a reading "
     "document, not guide material; the probe "
     "([`scripts/probe_np_crossover.py`](../scripts/probe_np_crossover.py)) "
     "carries each study's predictions, written down before the runs, and "
     "this page is regenerated from the raw rows by "
     "[`scripts/render_np_report.py`](../scripts/render_np_report.py).")
para("The through-line, up front: selection instruments fail exactly where "
     "you would want them most. Scarce independent rows and weak signal "
     "corrupt the importance ranking faster than they create headroom for "
     "removal, changing relationships punish every backward-looking score, "
     "and the defenses that work (rank per time period, screen for "
     "distribution shift, cluster before ranking, concentrate under drift) "
     "are structural, not statistical refinements.")

# ---- study 1 -----------------------------------------------------------------
rows = load("np-crossover-2026-07.jsonl")
cells = defaultdict(dict)
for r in rows:
    if r["dataset"] in ("superconductivity", "QSAR-TID-11"):
        cells[(r["dataset"], r["n_train"], r["draw"])][r["arm"]] = r

para("## Study 1: does shrinking the dataset turn selection into a "
     "regularizer?")
para("Setup: superconductivity's 81 real features plus 81 shuffled copies "
     "(ground-truth junk), so an oracle refit on the real 81 measures "
     "exactly what the junk costs. Rows are subsampled from 11,340 down to "
     "498 (5 draws each below full size); an untouched holdout judges every "
     "arm; knobs match the chapter-14 survey. QSAR-TID-11 runs unaugmented "
     "as the real wide-short contrast. Deltas are holdout rmse against the "
     "keep-everything baseline; negative means the arm wins; `nk` counts "
     "junk features kept in the top 81.")
body = []
for ds in ("superconductivity", "QSAR-TID-11"):
    for n in sorted({k[1] for k in cells if k[0] == ds}, reverse=True):
        draws = [v for k, v in cells.items() if k[0] == ds and k[1] == n]
        row = [ds, n, f"{st.mean(d['baseline']['error'] for d in draws):.3f}"]
        for arm in ("oracle", "gain_topk", "shap_val_topk"):
            if arm in draws[0]:
                dl = st.mean(d[arm]["error"] - d["baseline"]["error"]
                             for d in draws)
                nk = st.mean(d[arm]["noise_kept"] for d in draws)
                row.append(f"{dl:+.3f}" + (f" (nk {nk:.0f})" if nk >= 0
                                           else ""))
            else:
                row.append("-")
        body.append(row)
S.append(table(["dataset", "n train", "baseline rmse", "oracle",
                "gain top-k", "shap_val top-k"], body))
S.append("")
para("Finding: the hoped-for crossover does not exist. Removing the junk "
     "pays at every size (the oracle column), but the rankings recover only "
     "half of that at full size and nothing at n=498, where roughly a third "
     "of their picks are shuffled copies. The same row scarcity that makes "
     "junk expensive inflates its measured importance, so the instrument "
     "degrades exactly as fast as the problem grows. Both pre-registered "
     "predictions for this table (junk free at full n; rankings tracking "
     "the oracle) were refuted. QSAR agrees from the other side: shrinking "
     "rows turns its free k=256 tie into a small loss, never a win.")

scale = [r for r in rows if r["dataset"] == "synthetic"]
para("## Study 1b: the memory recipe at production scale")
para("Setup: synthetic 3.2M training rows by 2,048 features, 64 informative, "
     "single L40S GPU. The question a memory-bound n x p forces: can the "
     "ranking be fitted on a small row slice so the full matrix never has "
     "to exist?")
S.append(table(["arm", "holdout rmse", "fit wall (s)", "junk kept of 64"],
               [[r["arm"], r["error"], r["fit_wall_s"],
                 r["noise_kept"] if r["noise_kept"] >= 0 else "-"]
                for r in scale]))
S.append("")
para("Finding: a six-way accuracy tie (1,984 pure-noise columns cost 0.001 "
     "rmse at this row count), and every ranking, including the ones fitted "
     "on a 250k-row slice (6% of the data, 2 GB instead of 26 GB), "
     "recovered the informative set exactly. Rank on the largest slice that "
     "fits, refit on all rows times only the kept columns: measured safe, "
     "15x faster and 32x smaller on device than the full-width fit.")

# ---- study 2 -----------------------------------------------------------------
rows = load("hft-selection-trust-2026-07.jsonl") + \
    load("hft-selection-trust-ext-2026-07.jsonl")
cells = defaultdict(dict)
for r in rows:
    cells[(r["r2_target"], r["H"], r["lam"], r["draw"])][r["arm"]] = r
R2S = sorted({k[0] for k in cells}, reverse=True)


def mcell(r2t, h, lam, arm, f):
    return st.mean(cells[(r2t, h, lam, d)][arm][f] for d in (0, 1))


para("## Study 2: the trust map on market-shaped data")
para("Setup: synthetic data with the three difficulties of a markout "
     "target. Features are persistent AR(1) series; the target's noise is a "
     "forward moving average of width H, the overlap structure of an H-row "
     "markout, so 200k rows carry roughly 200k/H independent examples; "
     "signal weights optionally rotate over time (the drifting case). 32 of "
     "128 features carry signal with decaying weights; selection keeps 32; "
     "each cell reports junk kept and how many of the 10 strongest true "
     "signals were found. Splits are temporal with an H-row gap.")
for lam, tag in ((1.0, "fixed relationships"), (0.7, "drifting relationships")):
    S.append(f"### Gain ranking, {tag}")
    S.append("")
    body = []
    for r2t in R2S:
        row = [f"{r2t * 100:g}%"]
        for h in (1, 20, 100):
            row.append(f"{mcell(r2t, h, lam, 'gain_topk', 'noise_kept'):.1f} "
                       f"junk, {mcell(r2t, h, lam, 'gain_topk', 'sig_strong'):.1f}/10")
        body.append(row)
    S.append(table(["signal", "H=1", "H=20", "H=100"], body))
    S.append("")
S.append("### Era-averaged ranking, drifting relationships")
S.append("")
body = []
for r2t in R2S:
    row = [f"{r2t * 100:g}%"]
    for h in (1, 20, 100):
        row.append(f"{mcell(r2t, h, 0.7, 'era_stability_topk', 'noise_kept'):.1f} "
                   f"junk, {mcell(r2t, h, 0.7, 'era_stability_topk', 'sig_strong'):.1f}/10")
    body.append(row)
S.append(table(["signal", "H=1", "H=20", "H=100"], body))
S.append("")
para("Findings. First, trust collapses along both axes, and the first "
     "guess at a combined rule (independent examples times R-squared as a "
     "single budget) broke in the forgiving direction: settings with the "
     "same product differ threefold in junk kept, because signal strength "
     "rescues overlap far more than the sample arithmetic predicts. "
     "Second, drift imposes a floor that signal strength cannot remove: "
     "about 7 junk in 32 even at 25% signal with no overlap. Third, the "
     "era-averaged ranking (rank per time period, average the ranks) is "
     "the one defense that works under drift, finding two to three times "
     "as many strong signals as the plain ranking at the hard settings; "
     "scoring importance on a future data block, predicted to help, "
     "measurably did not. Fourth, at the hardest settings even the oracle "
     "cannot produce reliably positive out-of-sample R-squared: below a "
     "threshold the missing resource is independent data, and no selection "
     "method substitutes for it.")

# ---- study 3 -----------------------------------------------------------------
rows = load("pipeline-race-2026-07.jsonl")
cells = defaultdict(dict)
for r in rows:
    cells[(r["r2_target"], r["H"], r["lam"], r["draw"])][r["arm"]] = r
ARMS = ["corr_filter", "naive_gain", "naive_shap", "era_only",
        "pipe_no_era", "pipe_no_drift", "pipe_full", "oracle", "baseline"]

para("## Study 3: a staged pipeline raced against naive selectors")
para("Setup: 200 features with full ground truth. 24 independent signals, "
     "each with 3 near-copies (about 0.9 correlated), 16 broken features "
     "(signal-correlated until late in training, then the relationship "
     "dies and their distribution shifts), 88 pure noise. Every selector "
     "keeps 24. The staged pipeline: drop features whose recent "
     "distribution shifted (a train-recent classifier's importance), group "
     "correlated features, rank by importance averaged over 8 time "
     "periods, keep one representative per group. Ablations remove one "
     "stage each. Grading is against the answer key (signal groups "
     "covered, duplicate slots, broken and noise kept) and by refit "
     "R-squared on a purged, strictly later holdout.")
for lam, tag in ((1.0, "fixed relationships"), (0.7, "drifting relationships")):
    keys = [k for k in cells if k[2] == lam]
    S.append(f"### Composition, {tag} (mean over 12 cells)")
    S.append("")
    body = []
    for arm in ARMS:
        vals = [st.mean(cells[k][arm][f] for k in keys)
                for f in ("clusters_covered", "dup_slots", "broken_kept",
                          "noise_kept")]
        body.append([arm, *[f"{v:.1f}" if v >= 0 else "-" for v in vals]])
    S.append(table(["arm", "signal groups / 24", "duplicate slots",
                    "broken kept", "noise kept"], body))
    S.append("")
    S.append(f"### Holdout R-squared x100 by condition, {tag} (mean of 2 draws)")
    S.append("")
    body = []
    for r2t in (0.25, 0.10, 0.05):
        for h in (1, 20):
            row = [f"{int(r2t * 100)}% H={h}"]
            for arm in ARMS:
                v = st.mean(cells[(r2t, h, lam, d)][arm]["r2_hold"]
                            for d in (0, 1))
                row.append(f"{100 * v:.2f}")
            body.append(row)
    S.append(table(["condition", *ARMS], body))
    S.append("")
para("Findings. With fixed relationships, the best selector is the "
     "pipeline with the era stage removed (drift screen, clusters, plain "
     "full-data ranking per group): 19.3 of 24 signal groups against 15 to "
     "17 for the naive rankings, almost no duplicates or broken features, "
     "and the first selector in this whole line to beat keep-everything on "
     "accuracy, mildly but in 5 of 6 conditions. The era stage drags here "
     "because each period's fit has too few rows to rank 200 features. "
     "With drifting relationships everything reverses: keep-everything and "
     "naive top-k go negative (the model anti-predicts, having learned "
     "dead relationships and the broken features), the era stage becomes "
     "the thing that saves the pipeline, and the crude correlation filter, "
     "owner of the objectively worst feature list, produces the best model "
     "of all, beating even the oracle refit at every drift condition. Its "
     "duplicate-heavy list amounts to a concentrated bet on the few "
     "strongest, most persistent signal groups, and under rotating "
     "weights concentration transfers better than coverage.")
para("Scoring the written predictions: correlation-filter flooding "
     "confirmed; the drift screen's value confirmed (naive keeps 4 to 5 "
     "broken features under drift, the screen cuts that to about zero, the "
     "no-screen ablation lands between); the pipeline's predicted coverage "
     "of 22 or more was missed (best was 19.3); the ablation ordering held "
     "under drift and inverted under static, which study 2 had "
     "foreshadowed and the design under-weighted. The first run of this "
     "race carried a clustering index bug, caught because the pipeline "
     "showed duplicate slots that are impossible by construction; it was "
     "fixed and the race rerun in full.")

# ---- closing -----------------------------------------------------------------
para("## What a practitioner should take away")
para("1. Before believing any importance ranking, estimate your effective "
     "sample (rows divided by label-overlap width, discounted for "
     "cross-sectional correlation) and your realized R-squared, and place "
     "yourself on the study-2 tables. Strong signal forgives a lot; weak "
     "signal is forgiven nothing.")
para("2. Always: purged temporal splits, a distribution-shift screen, and "
     "correlation clustering with one representative per group. These are "
     "cheap, never hurt in any measured cell, and remove the two failure "
     "modes (redundancy, broken features) that importance scores cannot "
     "see.")
para("3. Do not commit to one ranking. Rank once on all data and once "
     "averaged per time period; which one is right depends on whether "
     "relationships are currently stable, which is observable on a rolling "
     "out-of-sample basis and not assumable in advance.")
para("4. Under suspected drift, prefer concentration over coverage, and "
     "judge any selection policy by its bad draw, not its average: the "
     "same drifting configuration produced +5% and -11% R-squared across "
     "two runs for the naive approaches, while the concentrated and "
     "era-averaged arms never went meaningfully negative.")
para("5. Selection earns accuracy only at the edges (removing planted or "
     "broken features under drift); its reliable payoff everywhere else "
     "is size: memory, latency, and pipeline surface, bought at a "
     "measured accuracy tie.")
para("## Caveats")
para("Two draws per cell throughout; drift cells have large draw-to-draw "
     "spread, so means there are indicative and the risk framing above is "
     "the honest one. All generators are synthetic and milder than real "
     "microstructure. One shape per study (feature counts, persistence, "
     "era count are fixed); the thresholds are properties of these shapes, "
     "not universal constants. A ratio metric (share of oracle R-squared) "
     "was retired mid-analysis after it misled on cells with tiny "
     "denominators; absolute numbers are reported instead.")

OUT.write_text("\n".join(S) + "\n")
print(f"wrote {OUT.relative_to(REPO)}")
