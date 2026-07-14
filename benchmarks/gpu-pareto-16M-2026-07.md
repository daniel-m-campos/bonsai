# GPU accuracy-vs-time frontier at 16M rows (2026-07, post-fix)

> **Supersedes** the 2026-07-12 pre-fix measurement (git history is the archive; decision 69 convention). That run motivated the investigation that found the GPU oblivious split-selection defect (missing issue-#60 port, decision 63) and preceded the populate/prefetch performance rounds; this one, 2026-07-14 on current main, measures the frontier with both resolved. Decision 71 records the verdict.

A benchmark without its losses is advertising. This one still has a loss to report, but it is a different and smaller one than its predecessor's.

## What and why

A fixed-iteration table hides the distinction between "fast per round" and "converges in fewer rounds". Sweeping each library's iteration count at one large cell (16M rows x 100 cols, depth 8, lr 0.1, 255 bins, seed 42) and recording (fit seconds, test r²) draws the frontier that answers the question users actually have: how long to a given accuracy?

Protocol: `scripts/gpu_pareto.py`, all variants in one process on one RunPod L40S (SECURE, driver 570.195.03), one untimed warmup per library, `fit()` timed end to end including each library's ingest. Raw rows: `results/gpu-pareto-16M-2026-07.jsonl` (22 points). Only same-pod points compare.

## The frontier, decomposed

Fitting each ladder as fixed cost plus marginal cost per round makes the shape legible:

| library | fixed cost | marginal ms/round | accuracy ceiling (as measured) |
|---|--:|--:|--:|
| bonsai `cuda_depthwise` | ~3.1s | 187 | 0.8934 @ 300 iters |
| bonsai `cuda_oblivious` | ~4.6s | 155 | **0.8981** @ 450 iters |
| catboost GPU | ~11.8s | **77** | 0.8980 @ 450 iters |
| xgboost GPU | ~22.3s | 58 | 0.8934 @ 300 iters |

## Reading it

- **bonsai owns the fast end.** Its fixed cost is a quarter of catboost's and a seventh of xgboost's, so below roughly r² 0.877 (the ~20s region) bonsai reaches any accuracy first: 0.8455 in 13.8s, 0.8666 in 17.9s, 0.8776 in 21.8s, points where catboost has barely amortized its startup and xgboost has not started producing.
- **The crossover sits near 100 rounds.** bonsai `cuda_oblivious` at 100 iters (20.1s, r² 0.8749) and catboost at 100 iters (19.5s, r² 0.8751) are the same point to fleet precision, which is the operating point of the standing scaling tables and their measured tie.
- **catboost owns the deep end, now for an honest reason.** Its marginal round is 2x cheaper (77 vs 155 ms), so above r² ~0.885 it reaches accuracy targets first: 0.8944 in 27.7s vs bonsai's 0.8948 in 36.3s, and 0.8973 in 35.1s vs 0.8974 in 51.9s. Pre-fix this looked like an accuracy gap; post-fix it is purely a marginal-throughput gap, the per-round kernel residue of decision 62 compounded by catboost's cheaper late rounds.
- **The accuracy ceiling is no longer catboost's.** bonsai `cuda_oblivious` plateaus at 0.8981, a hair above catboost's 0.8980 and well above xgboost's 0.8934. Pre-fix, bonsai could not reach this region at all (0.8638 at 100 iters, defect-limited); the ceiling itself was the recovered casualty of decision 63.
- **xgboost is dominated on this frontier**: catboost reaches every xgboost accuracy sooner, and its 22s fixed cost prices it out of the fast end entirely.

Against the superseded run at the shared points: r² values reproduce exactly where no fix applied (the depthwise ladder is identical to four decimals, the determinism contract across pods), the oblivious accuracy defect's disappearance is visible point by point (100 iters: 0.8638 to 0.8749), and bonsai's fit times dropped roughly 28% from the performance rounds while the reference libraries moved only with fleet variance.

## Verdict

The 16M frontier splits by budget: bonsai is the fastest route to good accuracy (everything up to roughly r² 0.88), catboost is the fastest route to maximum accuracy (0.89 and above), and the maximum itself now belongs to bonsai by a rounding digit. The remaining catboost edge is one number, its 77 vs bonsai's 155 ms marginal round at this scale, which is the next perf target if the deep end ever becomes load-bearing.
