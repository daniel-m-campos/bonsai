# GPU accuracy-vs-time frontier at 16M rows (2026-07, post marginal-round campaign)

> **Supersedes** the 2026-07-14 measurement (git history is the archive; decision 69 convention). That run recorded the frontier after the decision-63 accuracy fix and found catboost owning the deep end through a 2x cheaper marginal round (77 vs 155 ms); it named the marginal round as the next perf target. This run, 2026-07-15 on current main, measures the frontier after that campaign landed (PR #148, decision 72). Decision 72 records the verdict.

A benchmark without its losses is advertising. For the first time in this file's lineage, the loss column is nearly empty; the honest residues are named at the bottom.

## What and why

A fixed-iteration table hides the distinction between "fast per round" and "converges in fewer rounds". Sweeping each library's iteration count at one large cell (16M rows x 100 cols, depth 8, lr 0.1, 255 bins, seed 42) and recording (fit seconds, test r²) draws the frontier that answers the question users actually have: how long to a given accuracy?

Protocol: `scripts/gpu_pareto.py`, all variants in one process on one RunPod L40S (SECURE, US-NC-1, driver 570.195.03), one untimed warmup per library, `fit()` timed end to end including each library's ingest. Raw rows: `results/gpu-pareto-16M-2026-07.jsonl` (22 points). Only same-pod points compare.

## The frontier, decomposed

Fitting each ladder as fixed cost plus marginal cost per round (least squares over all points):

| library | fixed cost | marginal ms/round | accuracy ceiling (as measured) |
|---|--:|--:|--:|
| bonsai `cuda_depthwise` | ~3.1s | 121 | 0.8934 @ 300 iters |
| bonsai `cuda_oblivious` | ~3.4s | 104 | **0.8981** @ 450 iters |
| catboost GPU | ~12.4s | 76 | 0.8980 @ 450 iters |
| xgboost GPU | ~21.6s | **65** | 0.8934 @ 300 iters |

Against the superseded table: bonsai's marginal round dropped from 155 to 104 ms (the decision-72 levers: identity contract, device root sums, final-level skip) and its fixed cost from ~4.6 to ~3.4s, while catboost and xgboost moved only with fleet variance (77→76, 58→65 ms across hosts). Accuracy is untouched everywhere: the shared r² points reproduce to four decimals (oblivious @100 = 0.8749 on both pods, ceiling 0.8981 on both), which is the campaign's behavior-preservation contract measured from the outside.

## Reading it

- **bonsai owns the fast end, further out than before.** Fixed cost is now a quarter of catboost's and a sixth of xgboost's: 0.8749 in 13.9s (catboost needs 19.7s for the same point), 0.8861 in 17.0s, 0.8948 in 24.5s (catboost: 0.8944 in 27.9s).
- **The crossover moved from ~100 to ~320 rounds.** The superseded run's curves crossed at the standing 100-round operating point; with the marginal round cut 33%, equal wall clock now happens near round 320, which is where both libraries are already inside their plateaus. At the measured 300-iter points the ladders are a statistical tie: bonsai 0.8974 in 35.3s, catboost 0.8973 in 35.1s.
- **The deep end is no longer catboost's.** Above r² 0.885 the superseded run had catboost reaching every target first; now bonsai arrives first at 0.889 (~19.7s interpolated vs 23.9s) and 0.894 (24.5 vs 27.9s), and the plateau race is inside fleet noise. catboost's remaining edge is 3.1s to its own ceiling point (46.6 vs 49.6s at 450 iters), against a ceiling one rounding digit lower.
- **The accuracy ceiling stays bonsai's**: 0.8981 vs catboost 0.8980 and xgboost 0.8934, unchanged from the superseded run, as it must be for a speed-only campaign.
- **xgboost is dominated on this frontier**: bonsai and catboost both reach every xgboost accuracy sooner; its 21.6s fixed cost prices it out of the fast end entirely.

## Verdict

The 16M frontier no longer splits by budget. bonsai is the fastest route to every measured accuracy up to r² ~0.895, ties catboost through the 0.897-0.898 plateau, and holds the ceiling. The honest residues: catboost's marginal round is still cheaper (76 vs 104 ms), so a workload living strictly at 450+ rounds of this cell reaches its plateau ~3s sooner on catboost; and the remaining 104ms round is ~72ms histogram-kernel compute, which is kernel-engineering territory (decision 72 records why it stays unspent).
