# GPU accuracy-vs-time frontier at 16M rows (2026-07, post device-resident objective)

> **Supersedes** the 2026-07-15 measurement (git history is the archive; decision 69 convention). That run recorded the frontier after the marginal-round campaign (decision 72): bonsai first to every accuracy up to r² ~0.895, tied through the plateau, with one residue named honestly, catboost's cheaper marginal round (76 vs 104 ms) winning workloads at 450+ rounds. This run, 2026-07-18 on current main, measures the frontier after the device-resident objective landed (PR #172, decision 77). The residue is gone.

A benchmark without its losses is advertising. This file's loss column is now empty at every measured point; the honest caveats are at the bottom.

## What and why

A fixed-iteration table hides the distinction between "fast per round" and "converges in fewer rounds". Sweeping each library's iteration count at one large cell (16M rows x 100 cols, depth 8, lr 0.1, 255 bins, seed 42) and recording (fit seconds, test r²) draws the frontier that answers the question users actually have: how long to a given accuracy?

Protocol: `scripts/gpu_pareto.py`, all variants in one process on one RunPod L40S (SECURE), one untimed warmup per library, `fit()` timed end to end including each library's ingest. Raw rows: `results/gpu-pareto-16M-2026-07.jsonl` (22 points). Only same-pod points compare; this pod is a different host than the superseded run's, so the reference libraries act as the same-pod controls (they moved only with fleet variance, below).

## The frontier, decomposed

Fitting each ladder as fixed cost plus marginal cost per round (least squares over all points):

| library | fixed cost | marginal ms/round | accuracy ceiling (as measured) |
|---|--:|--:|--:|
| bonsai `cuda_oblivious` | ~3.8s | **64** | 0.8979 @ 450 iters |
| bonsai `cuda_depthwise` | ~2.8s | 88 | 0.8934 @ 300 iters |
| catboost GPU | ~11.7s | 78 | 0.8980 @ 450 iters |
| xgboost GPU | ~20.5s | 59 | 0.8934 @ 300 iters |

Against the superseded table: bonsai `cuda_oblivious` dropped from 104 to 64 ms/round (38%) and `cuda_depthwise` from 121 to 88, while the controls moved only with fleet variance (catboost 76 to 78, xgboost 65 to 59). The change is the device-resident objective: per tree, the gradient upload, the interleave pass, the values and leaf-id downloads, the host objective loop, and the host score update no longer exist (decision 77, `benchmarks/resident-objective-2026-07.md`). Accuracy is untouched: the resident models are bit-identical to the host-objective GPU models at full data, and the ladder's r² values match the superseded run's shape point for point.

## Reading it

- **bonsai's marginal round is now the cheapest among the accuracy leaders.** 64 vs catboost's 78 ms on the same pod. The superseded run's one residue (a workload at 450+ rounds reaching its plateau sooner on catboost) is gone: at 450 iterations bonsai lands 0.8979 in 31.9s, catboost 0.8980 in 46.4s, a 4th-decimal tie at 45% more wall clock.
- **First to every accuracy, at every horizon.** r² 0.875 in 10.2s (catboost 19.0s), 0.885 in 12.5s (catboost ~22s interpolated), 0.894 in 17.1s (catboost 27.5s), 0.8973 in 23.4s (catboost 35.0s). There is no crossover anywhere on the measured ladders; the curves no longer intersect.
- **The fixed-cost advantage compounds.** Roughly 3.8s vs 11.7s vs 20.5s: bonsai has finished a 100-round fit before xgboost's ingest completes.
- **xgboost is dominated on this frontier**: bonsai and catboost both reach every xgboost accuracy sooner. Its 59 ms marginal round, the best on the board, cannot recover a 20.5s fixed cost inside this cell's useful range.
- **The accuracy ceiling is a statistical tie** (0.8979 vs 0.8980 at 450; the superseded same-pod run measured the same pair as 0.8981 vs 0.8980). Fourth-decimal ceiling ordering is fleet noise, claimed by neither side.

## Verdict

The 16M frontier belongs to bonsai unconditionally on this ladder: fastest fixed cost, cheapest marginal round among the accuracy leaders, first arrival at every measured accuracy, tied ceiling. The honest caveats: this holds for the eligible regime (MSE objective, no DART, no sample weights, uniform or no row sampling); ineligible fits pay the superseded run's frontier, whose one residue was already only a plateau-depth effect. Cross-pod absolutes carry ~25% fleet spread; the transferable facts are the decomposition shape and the same-pod orderings above.
