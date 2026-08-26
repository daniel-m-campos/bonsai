# A row-subset view over the device bin plane is free

Covert probe, 2026-08-22/23. Not for merge. Worktree `seg-view-probe`, base `ccb57e1`.

## Question

Cross-validation and feature selection currently force a copy: every library materializes a new dataset per fold. Can bonsai instead train on a *view* of an already-binned device-resident `Dataset`, and what does the view cost against the copy?

## Method

Every arm does identical arithmetic: the same row count `m`, the same depth, the same iteration count. They differ only in how the kernel reaches the bins.

| arm | what it is |
|---|---|
| `copy` | `Dataset(X[:m])`: m rows, root iota over a compact matrix. What every library does today. |
| `seg1` | full n-row Dataset, sampler emits m rows as one contiguous run. A walk-forward fold. |
| `seg160` | 160 runs. Triple-barrier purging at horizon 1000, measured. |
| `seg10000` | 10,000 runs, about 32 rows each. Far past any real scheme. |
| `scatter` | bernoulli draw: m ascending but randomly gapped indices. Upper bound on fragmentation. |

The arms ride the existing `BernoulliSampler` seam, env-switched by `BONSAI_PROBE_SEGMENTS` (probe-only, ~20 lines in `src/sampler.cpp`). Every other code path sees an identical ascending list of identical length, so row count, gradient gather, and node bookkeeping are held constant and only index locality varies.

Timing is the device-side histogram spans from `BONSAI_CUDA_PROFILE`, not wall clock, so host sampling cost and the per-tree row upload are excluded by construction. Arms interleave within each rep. `root_hist` is the controlled figure: one node, all m rows, depth 0, no tree structure. `adv_hist` also reflects tree shape, which the arms do not share.

## Result

**L40S, 16M x 128, depth 8, 30 iters, 3 reps** (`pod-l40s-2026-08-23.log`):

| arm | root_hist | vs copy | adv_hist | vs copy |
|---|---|---|---|---|
| copy | 0.2500s | +0.0% | 1.2900s | +0.0% |
| seg1 | 0.2500s | +0.0% | 1.2900s | +0.0% |
| seg160 | 0.2500s | +0.0% | 1.2900s | +0.0% |
| seg10000 | 0.2500s | +0.0% | 1.3000s | +0.8% |
| scatter | 0.2500s | +0.0% | 1.3700s | +6.2% |

4M x 128 on the same card: identical pattern, scatter +5.7% on `adv_hist`.

Jetson Orin, 400k x 64: same ordering, but scatter costs +26%. The Orin is bandwidth-starved enough that sector over-fetch surfaces directly; the L40S absorbs it. This is the second campaign in which the nano has misled on a bandwidth or occupancy axis, after tiled-small-nodes.

## Reading

**A view costs nothing.** Contiguous and realistically-purged views are indistinguishable from a materialized copy at both scales on both machines. The copy therefore buys no speed at all, while costing a full matrix copy and its memory.

**Fragmentation is not the cost axis; run length is.** Even 32-row runs land within 1%. The mechanism is sector coverage: each row's bin strip is 8 bytes, a warp reads 32 consecutive slots, and any run longer than a sector keeps the read coalesced no matter how many runs there are. Real temporal folds have runs of thousands to millions of rows.

**Only near-random access costs, and only a little on a real card.** 6% on the L40S, from reading about 1/0.8 the sectors at 80% density.

## Resolution caveat

The profile timer quantizes to 10ms. `adv_hist` at 1.29s is 129 ticks and resolves well. `root_hist` at 0.25s is 25 ticks, so the honest claim at root is "no difference above about 4%", not "identical".

## What this implies

The expensive part is already built. bonsai's histogram kernel *always* reads bins through an arbitrary row-index array (`nrows[k]`), for every fit, at every level; the full-data root just builds an iota. So a range or segment view needs **no kernel work**. The remaining work is an API that constructs the rows array and a `Dataset` that shares its parent's device plane.

Decision 103 rejected row reordering because "it is the only option here that changes what a row id means". A view changes nothing about what a row id means; it reuses the existing contract exactly.

## Round 2: the three open questions, answered

L40S, `pod2-l40s-2026-08-23.log`.

### A. The scatter penalty is density, and it is zero at the root

4M x 128 at 300 iterations, so `root_hist` accumulates 74 ticks instead of 7 and resolves to about 1.4%. `root_hist` is one node at depth 0: tree shape cannot reach it.

| arm | root_hist | vs copy | adv_hist | vs copy |
|---|---|---|---|---|
| copy | 0.7400s | +0.0% | 1.8400s | +0.0% |
| seg1 | 0.7400s | +0.0% | 1.8600s | +1.1% |
| seg160 | 0.7400s | +0.0% | 1.8400s | +0.0% |
| scatter | 0.7400s | +0.0% | 1.9500s | +6.0% |

At the root every arm reads all m rows in one sweep and nothing separates them, scatter included. The 6% is therefore a *depth* effect, and it is locality rather than tree shape: at depth a node's rows are scattered within their bounding span, and the copy and the contiguous view share the same span `[0, m)` while a bernoulli draw spans `[0, n)` at 80% density. The penalty is the reciprocal of the row set's density within its bounding span, which is 1.0 for any contiguous or segmented view and 0.8 for a random subsample.

### B. Column views want tile-aligned features, and the difference is large

Same feature count in both modes; only the distribution across the width-8 tiles changes. A tile with no selected feature returns early, so a packed selection empties tiles while a spread selection of the same size touches all of them. `adv_hist`, 4M x 128, 60 iters:

| features kept | spread | packed | packed vs spread | packed vs all-features |
|---|---|---|---|---|
| 100% | 0.87s | 0.88s | +1% | 1.00x |
| 50% | 0.64s | 0.41s | **-36%** | 0.47x (ideal 0.50) |
| 25% | 0.42s | 0.31s | **-26%** | 0.35x (ideal 0.25) |
| 12.5% | 0.29s | 0.23s | **-21%** | 0.26x (ideal 0.125) |

Packed selection scales nearly linearly at 50% and beats spread at every fraction. Spread selection at 50% saves only 26%, not 50%, because every tile is still visited. So a feature-selection loop should pack the surviving features into tile-aligned groups, either by choosing them that way or by reordering the plane's tiles once; scattered survivors collect roughly half the available benefit.

### C. Fold memory, the CPCV wall

16M x 128, k=5, per-process NVML:

| | device memory | vs one plane |
|---|---|---|
| one Dataset | 2474 MiB | 1.0x |
| 5 materialized folds | 10314 MiB | 4.2x |
| 5 index-array views | 2718 MiB | 1.1x |

3.8x less device memory at k=5, and a *range* view carries no index array at all, just two integers, so it stays flat at 1.0x. At the k=15 that combinatorial purged CV wants, the materialized form needs roughly 32 GB of plane against this card's 46 GB, while views stay at 2474 MiB.

### D. The tile reorder pays back in under three iterations

Derived from B, no extra run. A reorder reads the plane once and writes a smaller one, with no atomics and no shared staging, so it is strictly cheaper than one root histogram build over the same plane: 2.7 ms at 4M x 128 (`root_hist` 0.16s over 60 iters). Against the per-iteration saving packing buys:

| features kept | saving per iteration | payback |
|---|---|---|
| 50% | 3.83 ms | 0.7 iterations |
| 25% | 1.83 ms | 1.5 iterations |
| 12.5% | 1.00 ms | 2.7 iterations |

Against refits of hundreds of iterations this is a rounding error, so a selection loop should reorder and keep the exact feature set rather than approximating the selection to whole tiles to avoid the reorder. Two consequences: order the operations columns first then rows, since a reorder makes a new plane and row views must be taken over the packed one; and memory peaks at 1+fraction while parent and packed plane coexist.

Caveat: at 12.5% the root buckets are 5 to 7 ticks, at the 10 ms floor, and packed reads higher than spread there, which is quantization rather than an effect. The payback arithmetic uses `adv_hist` alone, where counts run 23 to 64 ticks.

## Round 3: the CPU plane tells a different story, and it is fixable

Local, 4 performance cores, 1M x 32, depthwise, `populate` bucket of `BONSAI_GROW_PROFILE` (the host counterpart of the CUDA histogram spans). `cpu-probe-2026-08-23.log`, `cpu-probe-sparse.log`.

**First: the CPU plane needs no engine work.** Both CPU fills already index bins through an arbitrary row list, `bins[rows[k]]` in the column fill and `rm_ptr + r * width` in the row fill, and grad, hess, labels, and weights are all full-length and globally indexed. A view gathers nothing.

### Dense regime, 80% of rows, both arms take the column fill

| arm | populate | vs copy | spread |
|---|---|---|---|
| copy | 0.3900s | +0.0% | 2.6% |
| seg1 | 0.4600s | +17.9% | 4.3% |
| seg160 | 0.4700s | +20.5% | 4.3% |
| seg10000 | 0.4700s | +20.5% | 8.5% |
| scatter | 0.4600s | +17.9% | 2.2% |

**Every view costs the same 18 to 20%, and locality is irrelevant**: a perfectly contiguous view costs what a fully scattered one does. This is not the CUDA density effect. It is the dense fast path at `src/grower.cpp:42`, `bool const dense = n == ds.n_rows()`, which gives a materialized copy `bins[k]` and every view `bins[rows[k]]`. On CPU that one extra load per element is the entire cost.

### Sparse regime, 10% of rows, the view drops to the row fill

| arm | populate | vs copy | spread |
|---|---|---|---|
| copy | 0.3700s | +0.0% | 27.0% |
| seg1 | 0.3500s | -5.4% | 31.4% |
| seg160 | 0.3600s | -2.7% | 19.4% |
| seg10000 | 0.3800s | +2.7% | 2.6% |
| scatter | 0.4300s | +16.2% | 2.3% |

Spreads of 19 to 31% on the first three arms make those deltas noise on this machine; only `scatter` separates. Read as: contiguous and segmented views are indistinguishable from a copy here, scattered costs about 16%. Caveat: the fill cutoff `rows.size() * 4 >= ds.n_rows()` (`src/grower.cpp:68,126`) measures against the dataset rather than the view, so at 10% the view takes the row fill while the copy takes the column fill. The arms run different algorithms, so this is not a controlled locality comparison the way the dense regime is.

### The consequence: the row descriptor earns its place differently on each plane

The 18% is not inherent. A contiguous view satisfies something as strong as the dense test: if the rows are `[a, a+m)` then `bins[rows[k]] == bins[a+k]`, so the fill can offset the base and index directly with no indirection. Segments get the same treatment per run. What blocks that today is discovery, not arithmetic: you cannot cheaply learn contiguity from a `std::vector<row_id_t>` without scanning it, but a **`Range` descriptor knows it statically**.

So the RLE is not bookkeeping. On CUDA it removes a per-tree row-list upload; on CPU it is what lets the fill keep its fast path, worth about 18% on exactly the dense-fold case cross-validation produces. A CPU view without a range-aware fill is not free, and the API should not claim it is.

Two latent items this regime confirms, both from `src/grower.cpp`:

- `dense = n == ds.n_rows()` silently reads `bins[k]` for any same-cardinality non-identity row list. A strict subset is safe; a permutation is not, which is exactly what `reorder` would produce.
- The fill cutoff's denominator should be the view, not the dataset, or small views always route to the row fill.

### x86-64 confirms it: the penalty is architectural, not Apple silicon

EPYC 9554 (Zen 4, AVX-512), 2M x 32, frac 0.8, 60 iters, 5 reps, thread count swept. Every pool capped (`OMP_NUM_THREADS`, the `Dataset` constructor's own `n_threads`, and the fit config), which is what finally produced `throttled during: 0` at every point.

| threads | copy | seg1 | seg160 | seg10000 | scatter | worst spread |
|---|---|---|---|---|---|---|
| 2 | 3.47s | +18.7% | +25.6% | +20.5% | +25.1% | 28% |
| 4 | 1.88s | +19.7% | +43.1% | +61.2% | +62.8% | 34% |
| 18 | 1.58s | +12.7% | +17.7% | +12.7% | +19.0% | 16% |

The 18-thread row is both the cleanest (spreads 9.5 to 16%) and the deployment-relevant one. Read it as: **a contiguous view costs about 13% and a scattered one about 19% on a many-core x86 box**, with the arms not cleanly separated from each other at this resolution. The 4-thread row's spreads (24 to 34%) make its ordering meaningless despite zero throttling; the cgroup caps CPU, not memory bandwidth, so co-tenants still move these numbers.

Matched-thread comparison against Apple silicon, 2 threads, 1M x 32: M2 copy 1.15s with every view arm at +24 to +25% and spreads under 5%. So both architectures pay a real indirection penalty, and it **shrinks as threads rise** (M2 +25% at 2 and +18% at 4; EPYC +19% at 2 and +13% at 18), consistent with memory-level parallelism hiding the extra load rather than bandwidth being the binding constraint.

The one architectural difference: on the M2 all four view arms are identical to within 1%, so locality contributes nothing; on x86 scatter sits about 6 points above contiguous, which is within the combined spreads but points the same way at both 2 and 18 threads. Treat that as suggestive, not established.

### route_unsampled costs 19% of the fill, so a view should skip it

`route_unsampled` (`src/grower_impl.hpp:576`, inside the `finalize_s` bucket) walks every row NOT in the row list through the finished tree each iteration so its training score stays current. It no-ops when the list covers the dataset, so a materialized copy never pays it and a view always does. M2, 1M x 32, frac 0.8, 4 threads, 5 reps:

| arm | populate | finalize | finalize vs copy |
|---|---|---|---|
| copy | 0.7000s | 0.0400s | +0.0% |
| seg1 | 0.8300s | 0.2000s | +400% |
| scatter | 0.8400s | 0.2400s | +500% |

The complement is only a fifth of the rows at this fraction, yet routing it costs four times what leaf finalization does: +0.16s against a 0.83s fill, about **19% of the histogram work**. That matches the pre-registered arithmetic estimate of ~20% and is large enough to decide the semantics rather than defer them.

**A view should skip it.** A view means this fit is about these rows; keeping the complement's scores current is work spent on rows the fit does not concern. The cost is a genuine divergence from sampling, where the complement's scores must stay live because the next tree resamples: out-of-view scores go stale, and that belongs in the docstring rather than in a footnote.

### Ops lessons from this round

- **A GPU pod can time the CPU plane, but only if every thread pool is capped.** Three earlier runs on the same pod class threw `seg1` at +18.3%, +31.0%, and +4.1% with `nr_throttled` climbing throughout. The cause was not co-tenants: `bonsai.Dataset(...)` carries its own `n_threads` and defaults to every advertised core, so binning ran 256-wide against a 27-core quota while only the fit's thread count was being tuned. Cap `OMP_NUM_THREADS`, the constructor, and the fit together and throttling goes to zero.
- RunPod CPU pods have a real cpuset (`cpu.max` reads `max`) and are the correct instrument, but the MCP tooling cannot request a vCPU count and defaults to 2, which is too small to be representative.
- Deploy the tree and start the job before polishing the script. A pod billed for eight minutes this round while its run script was being edited locally.

### Resolution and noise

The dense regime resolves well (2 to 9% spread against an 18% effect). The sparse regime does not (up to 31%) and would need a rented CPU pod at one vCPU per thread with a real cpuset to settle, per the CPU-plane rule that a shared machine cannot time this plane.

## Still open

- View lifetime: a view must keep its parent's plane alive. Refcounting question, not a measurement.
- Bin-edge provenance: quantizing once over a whole series lets cut points see the future. Unmeasured in the literature; a separate probe found 13.7% of global bin edges falling outside the training range on a deliberately non-stationary series.
