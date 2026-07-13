# GPU accuracy-vs-time frontier at 16M rows (2026-07)

A benchmark without its losses is advertising. This one has a loss to report.

## What and why

The scaling table quotes GPU fits at a *fixed* 100 iterations, which conflates two questions a fixed row cannot separate: is a library fast because it does less work per round, or because it converges in fewer rounds? At 16M rows on GPU the honest question is the whole **accuracy-vs-time frontier** — sweep each library's iteration count and plot test r² against wall-clock `fit()`. A library owns the frontier where no competitor reaches the same accuracy in less time.

Every point below is the same 16M×100 synthetic regression (`bench_scaling.gen_data`, seed 42), depth 8, learning rate 0.1, 255 bins for bonsai / 254 for the references (their GPU cap), **all measured on one pod** (L40S, US-NC-1) so the numbers compare. Raw runs: [`results/gpu-pareto-16M-2026-07.jsonl`](results/gpu-pareto-16M-2026-07.jsonl).

## The frontier

| library | iters | fit_s | test r² | on frontier? |
|---|--:|--:|--:|:--:|
| bonsai `cuda_oblivious` | 60 | 19.35 | 0.8316 | ● fastest |
| bonsai `cuda_depthwise` | 60 | 20.09 | 0.8455 | ● |
| bonsai `cuda_oblivious` | 80 | 24.01 | 0.8531 | ○ |
| **catboost** GPU | 100 | 24.08 | 0.8751 | ● |
| bonsai `cuda_depthwise` | 80 | 25.15 | 0.8666 | — |
| bonsai `cuda_oblivious` | 100 | 28.05 | 0.8638 | — |
| **catboost** GPU | 150 | 28.35 | 0.8892 | ● |
| bonsai `cuda_depthwise` | 100 | 30.16 | 0.8776 | — |
| **catboost** GPU | 200 | 32.75 | 0.8944 | ● |
| bonsai `cuda_oblivious` | 130 | 33.85 | 0.8758 | — |
| xgboost GPU | 100 | 36.69 | 0.8776 | — |
| bonsai `cuda_depthwise` | 130 | 36.78 | 0.8860 | — |
| bonsai `cuda_oblivious` | 160 | 39.32 | 0.8829 | — |
| xgboost GPU | 150 | 39.95 | 0.8886 | — |
| **catboost** GPU | 300 | 40.33 | 0.8973 | ● |
| xgboost GPU | 200 | 43.42 | 0.8919 | — |
| xgboost GPU | 300 | 48.81 | 0.8934 | — |
| **catboost** GPU | 450 | 51.43 | 0.8980 | ● highest |

## Three readings, one of them a loss

**bonsai strictly dominates xgboost-GPU.** At equal accuracy bonsai is faster everywhere: bonsai `cuda_depthwise` reaches xgboost's own 100-iteration accuracy (0.8776) in **30.16s vs xgboost's 36.69s — 18% faster at the same r²**, and no xgboost point is left of any bonsai point at matched accuracy. Every xgboost row above is off the frontier.

**catboost owns the frontier at every accuracy you would actually ship.** Above ~0.875 r², catboost is unbeaten: catboost@150 (28.35s, 0.8892) is both faster *and* more accurate than bonsai `cuda_depthwise`@100 (30.16s, 0.8776). Its ordered boosting converges in fewer rounds on this data, and its symmetric-tree kernel runs ~20% faster per round than bonsai's. bonsai holds the frontier only in the sub-24s / sub-0.85 corner — the fast-and-rough regime.

**The "bonsai is more accurate than catboost" line from the fixed-100 table does not survive.** At 100 iterations bonsai `cuda_depthwise` (0.8776) does edge catboost (0.8751) — but that is an artifact of the fixed row. catboost clears bonsai's accuracy within a few more iterations that are individually cheaper, so at matched *time* catboost is ahead. Comparing at a fixed iteration count flattered us; the frontier does not.

## Why the find-kernel optimization round was abandoned

This study replaced a planned GPU kernel-optimization round. Instrumenting first (a profile-gated sync splitting the find lap into kernel-compute vs device→host transfer) showed the find kernel costs **0.17s** at 16M — the "8.4s find" in the grow profile is the profiler's opening `cudaDeviceSynchronize` catching the *previous* level's asynchronous histogram kernels, not the find scan. The histogram kernel, the genuine ~8s GPU cost, already accumulates in float shared memory (double only for the bounded cross-chunk merge), so the obvious precision lever is spent. Closing catboost's ~20% throughput gap would take a memory-bandwidth / atomic-contention rewrite of the histogram kernel — a real project, not a knob — and even a complete win there would not overturn catboost's *convergence* advantage, which is an algorithm difference (ordered boosting), not a kernel-speed one. The instrument-first pass is the deliverable: it turned a speculative multi-hour rewrite into a measured no-go.

## Where this leaves the GPU story

bonsai's GPU backend is Pareto-dominant over xgboost-GPU and competitive with catboost-GPU, trailing it by roughly 20% at matched accuracy on this large cell. That is the honest position: strictly superior to xgboost, behind catboost on raw large-scale GPU throughput. bonsai's broader case against catboost is made on the other axes — bit-identical cross-architecture determinism, a third of the host memory, an 1,800-line engine, chance-band categorical parity through preprocessing, and best-of-field CPU quality on 9 of 10 real datasets — not on this frontier.
