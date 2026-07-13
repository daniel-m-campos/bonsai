# Catboost's 16M edge, decomposed: a bug, not an algorithm (2026-07)

The GPU frontier study ([gpu-pareto-16M-2026-07.md](gpu-pareto-16M-2026-07.md)) found catboost ahead of bonsai at 16M rows and attributed it (decision 62) to *two* gaps: per-round kernel speed and per-round convergence. This note runs the feature-admission ladder on that "convergence" gap — cheapest decisive experiment first — and finds it was never catboost's algorithm at all. It was a bonsai bug: the GPU oblivious grower carried a split-selection defect that its CPU twin had already had fixed. Patched, bonsai's GPU oblivious matches catboost's accuracy exactly; only a genuine ~19% kernel-speed gap remains.

All CPU runs are a Mac (16 threads); all GPU runs are one L40S pod (same pod per comparison — identical GPUs measure up to 25% apart across the rental fleet). Matched config throughout: depth 8, lr 0.1, 255/254 bins, synthetic Friedman regression (`bench_scaling.gen_data`, seed 42). Probe scripts: `scripts/probe_ordered_boosting.py`; raw runs in `results/`.

## Door 1 — ordered boosting is not the edge

Hypothesis: catboost's ordered boosting (predecessor-only gradients, correcting prediction shift) buys the per-round accuracy. Toggling catboost's own `boosting_type` isolates it:

| rows | iters | catboost Ordered | catboost Plain | bonsai oblivious |
|--:|--:|--:|--:|--:|
| 200k | 100 | 0.8722 (17.1s) | 0.8720 (2.5s) | **0.8747** (3.9s) |
| 200k | 200 | 0.8932 (30.6s) | 0.8937 (4.7s) | **0.8940** (7.3s) |
| 1M | 100 | 0.8754 (63.2s) | 0.8760 (8.4s) | **0.8766** (10.0s) |

Ordered vs Plain is a wash — identical r², sometimes Ordered is *worse*, always ~7× slower. Ordered boosting fights prediction shift, which bites on small/noisy/categorical data, not clean high-signal regression; catboost itself defaults `boosting_type=Plain` past ~50k rows, so at 16M it was never ordered to begin with. And bonsai's oblivious grower *beats both catboost variants* here. **Refuted.**

## Door 2 — bin quality is not the edge

Hypothesis: bonsai bins from a 200k-row subsample (`bin_mapper.n_samples`), so at 16M that is 1.25% of the data and its cuts are coarse. Sweeping the sample count:

| rows | catboost | bonsai @200k | bonsai @1M | bonsai @full |
|--:|--:|--:|--:|--:|
| 1M | 0.8760 | 0.8766 | 0.8766 | — |
| 4M | 0.8762 | 0.8763 | 0.8766 | 0.8764 |

Sample count moves r² by ≤0.0003 — noise. 200k rows already pin the quantiles. **Refuted.**

## The tell — bonsai beats catboost on CPU, loses only on GPU

On CPU, bonsai ties-or-beats catboost at every scale, *including 16M*:

| 16M, 100 iters | catboost | bonsai oblivious | bonsai depthwise |
|--|--:|--:|--:|
| **CPU** r² | 0.8744 | **0.8749** | **0.8782** |
| **GPU** r² | 0.8751 | 0.8638 ✗ | 0.8776 |

catboost's GPU matches its CPU (0.8751 ≈ 0.8744). bonsai's *depthwise* GPU matches its CPU (0.8776 ≈ 0.8782). But bonsai's *oblivious* GPU (0.8638) is **0.011 below its own CPU oblivious** (0.8749). The deficit is not scale, not the reference library, and not precision — GPU histogram cells are *double* (float chunks merged in double), more precise than the CPU's float cells. It is oblivious-specific and device-specific: a logic divergence between the two level-find implementations.

## The bug — the GPU level-find kept a veto the CPU had dropped

The CPU level-find (`update_best_for_feature_for_level`, split.cpp) was fixed in the quality campaign (issue #60, decision 56): a frontier node whose cut would be infeasible (`child hess < min_child_hess`) contributes its *parent* score — zero gain — instead of vetoing the whole level candidate. The comment records why: *"at depth ≥ 5 some frontier node is always near-empty, so every good cut was rejected and oblivious trailed catboost by 3-26%."*

The device `level_find_kernel` never got that fix. It still computed `warp_all(feasible)` across the frontier and dropped any candidate with a single infeasible node. At 16M/depth-8, deep levels always have near-empty nodes among the 256, so good deep cuts were vetoed — the exact pathology issue #60 named, silently re-introduced on the GPU. Small-data GPU-vs-CPU oblivious tests passed because they never produce a near-empty node; the divergence only surfaces at depth.

## The fix and the result

The port is a direct mirror of the CPU: an infeasible node contributes its parent score, the veto is gone (decision 63). A new regression test (`CudaObliviousGrower matches CPU when deep nodes go infeasible`) forces infeasibility with a high `min_child_hess` at depth 7 — it fails on the old kernel (GPU −0.019 vs CPU 0.324) and passes on the new one; the full `[cuda]` suite stays green (125,864 assertions).

The 16M accuracy recovers exactly to the CPU value, same pod:

| @100 iters, 16M, one L40S | fit_s | r² |
|--|--:|--:|
| catboost GPU | 23.81 | 0.8751 |
| **bonsai cuda_oblivious — fixed** | 29.21 | **0.8749** |
| bonsai cuda_oblivious — before | ~28 | 0.8638 |
| bonsai cuda_depthwise | 30.37 | 0.8776 |

## What this settles, and what honestly remains

The "convergence gap" decision 62 attributed to catboost's ordered boosting was bonsai's own oblivious veto bug. Fixed, bonsai's GPU oblivious matches catboost's accuracy per round to the fourth decimal (0.8749 vs 0.8751), and bonsai's depthwise is *more* accurate (0.8776). Every user of `cuda_oblivious` at depth ≥ 5 was getting quietly worse models at scale; that is the real value here, benchmark aside.

What remains is a single, honest, genuine gap: catboost's symmetric-tree GPU kernel is **~19% faster per round** (0.238 vs 0.292 s/iter at 16M), so on the time-vs-accuracy frontier it is still ahead at matched accuracy. That is kernel efficiency, not algorithm — a bounded optimization target for a later round, not a disadvantage that requires changing what bonsai is.
