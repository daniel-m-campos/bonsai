# 7: Parallelism

> **Status:** done (decision 32, tag `v0.2.0`). This doc replaced the planned "ParallelBackend concept with OpenMP + std::execution impls"; what shipped is deliberately smaller. Concept-level walkthrough with the war stories: [guide/9-parallelism-and-determinism.md](../guide/9-parallelism-and-determinism.md).

## The seam

All parallelism flows through one function ([`include/bonsai/parallel.hpp`](../../include/bonsai/parallel.hpp)):

```cpp
template <typename F> void parallel::for_each_index(size_t n, F &&f);
```

Runs `f(i)` for `i in [0, n)`. OpenMP body when built with `BONSAI_OPENMP` (default), plain loop otherwise. Callers never see the difference. The worker count comes from `[parallel] n_threads` (0 = auto: hardware threads capped at 16 and at any cgroup CPU quota), applied process-wide by `resolve_config` / the Python module.

Auto is capped because the per-level parallel sections are short: on hosts where the core count far exceeds per-level parallelism, OpenMP spin-wait at the section barriers dominates useful work: a 60-vCPU host ran the MSD fit 10× slower at 60 threads than at 16 (issue #2, decision 44). An explicit `n_threads = N` passes through uncapped; on oversubscribed many-core hosts `OMP_WAIT_POLICY=passive` (or `KMP_BLOCKTIME=0`) is the operator knob that makes idle workers sleep instead of spin.

Auto also clamps to the cgroup CPU bandwidth quota when the process runs under one (`cpu.max` on cgroup v2, `cpu.cfs_quota_us` over `cpu.cfs_period_us` on v1, floor of the ratio). OpenMP's max-thread count follows the cpuset affinity mask, which a Kubernetes CPU limit or `docker --cpus` leaves at the host's core count, so an unclamped pool burns the quota early in each period and the scheduler freezes the cgroup for the rest of it: on a container advertising 128 CPUs with a 13.6-CPU quota, 16 threads left 97% of periods throttled against XGBoost's 1.2% (issue #359). The clamp reads the cgroup paths a container sees for itself; a process in a non-root cgroup of the host namespace reads unlimited and sizes as before, since resolving that case needs the relative path from `/proc/self/cgroup` joined to the mount point. `OMP_WAIT_POLICY=passive` stays the mitigation for whatever oversubscription remains: on that container it removed 22% of the CPU-seconds, 86% of the throttled time, and 5 to 11% of train. An explicit `n_threads = N` above the quota is still honored: the fixed-N contract keys the model bytes to the resolved count, so clamping it would make one configuration produce different models in different containers. It draws a one-time stderr warning instead, naming the quota and pointing at the lower count and `OMP_WAIT_POLICY=passive` (XGBoost clamps explicit counts as well, making no equivalent reproducibility promise).

Why not the proposed `ParallelBackend` concept dispatched like objectives and growers? One implementation doesn't earn a typelist dimension. The `std::execution` backend was dropped: it would have added a TBB dependency and a second code path to keep deterministic, for a benchmark-only payoff. The free-function seam keeps the door open: a second backend would slot in behind the same signature (see `6-dispatch.md` §"Backend placement").

## Scheduling

`schedule(dynamic, chunk)` with `chunk = max(1, n / (n_threads * 4))`:

- Feature loops (`n ≈ 90`) degrade to chunk 1: dynamic stealing keeps asymmetric performance/efficiency cores (Apple M-series) busy instead of letting E-cores drag the static-partition barrier.
- Row loops (`n ≈ 500k`) get a few thousand indices per chunk: scheduler lock traffic stays negligible.

## Names

One noun, one meaning, for the CPU fill's work (`src/fill/`). The entity nouns it fills into (cell, histogram, arena, slice, feature stride) are defined in [`2-histogram.md`](2-histogram.md).

| Term | What it names |
| --- | --- |
| row-major mirror | the dataset's bins stored row by row, so one row's bins are contiguous bytes (`Dataset::row_major_bins`; [guide 2](../guide/2-binning-and-histograms.md) owns the definition) |
| mirror tile | one column block of that mirror, 2048 features wide (`Dataset::mirror_tile_width`), so one tile's histograms stay cache-resident on wide data |
| row chunk | a contiguous range of one node's rows, the unit of thread work, `RowChunk` |
| partition | the cut of a level's chunk list into one contiguous range per thread |
| partials | the per-(thread, node) scratch histograms, one row of cells per partition slot |
| fill | to accumulate cells from rows |
| reduce | to sum a node's partials into its arena, in ascending thread order |
| route | to send a node to the dense column fill or to the sparse row-chunk fill |
| lone node | a node with no frontier beside it: every node the leaf plane fills, and the root |
| feature range | a contiguous run of the selection one worker owns, the lone-node fill's unit |
| row block | a contiguous share of a lone node's rows, so more than one worker can fill the same features |
| carve | to start an arena run's cells and hand the feature its slot, `carve_run` |

## What is parallel

Unit-parallel histogram fill for u8 (max_bin ≤ 255) data: a level's sparse nodes are cut into fixed-grain row chunks over the row-major mirror, and that node-major chunk list is split into contiguous per-thread ranges (`CpuHistogramEngine::populate_many`, decisions 49 and 105). Each range reads its rows' bins as contiguous bytes and accumulates into either the node's own cells (the lowest-numbered range touching that node) or one private partial reduced afterwards in ascending range order.

Feature-parallel: histogram fill for u16 data and for dense nodes (grad/hess gathered into node-row order first so every feature scan reads sequentially), the levelwise finder's split scan (per-feature bests merged serially in feature order to preserve the tie-break), histogram subtraction, the partial reduce, binning, mapper fitting. Node-parallel: the per-node split scan, one worker per frontier node (`LevelStep::host_find`), each node's scan serial over its features so the tie-break is the serial walk's. Row-parallel: predict (both tree types), objective grad/hess, score updates, CSV row parsing, out-of-bag routing.

## The lone node's fill

Best-first growth expands one node per round, so the leaf plane has no frontier to cut into per-thread ranges and the fill has to cut the node itself (`fill_lone`, `plan_lone_fill`, issue #367). The root is the same shape: it is the one lone node the level growers fill too, and it keeps their path.

Two decompositions, routed by the same density cutoff the level plane uses. A dense node (`rows >= n_rows / 4`) or u16 data takes the column fill unchanged: one worker per feature, its histogram L1-resident, bit-identical at any thread count. A sparse u8 node takes the row-major mirror, cut two ways at once.

**Feature ranges by row blocks.** A feature range is a contiguous run of the selection, cut inside the mirror tiles it spans, so the arena cells a worker touches are its own and no two workers collide. A row block is a contiguous share of the node's rows, which lets more than one worker fill the same features at the price of one partial arena for every block past the first. Block 0 accumulates straight into the node's arena and the rest are reduced into it in ascending block order, the same rule the level plane's ranges follow. The work list is the cross product, one index per (range, block), on a team of exactly `ranges * blocks`: every other region in a fit runs on the configured team, and a region of a different size makes the runtime rebuild the team the fills around it are keeping hot.

**Sizing the team, then spending it.** The work is the arena a fill zeroes plus the accumulates it makes, `n_sel * 256 + n_rows * n_sel` cells, one worker per 32768 of them and never more than the configured count. Counting the zero fill is what keeps a wide node from looking small: a node holding few rows still has a full arena to start. That team goes to row blocks while the mirror row is too narrow to give every worker its own 64-byte cache line and while each block still owns 8192 rows to earn the partial it will be charged, and to feature ranges after that. A fully selected tile is measured in cache lines; a sampled one is measured in selected features, since its ranges scatter over the whole tile and are line-private however narrow the selection is.

**The carve and the subtraction ride the fill.** Starting the arena's cells and subtracting the finished child from its sibling used to be parallel regions on either side of the fill. They are items in the fill's own work list now: the worker holding a feature range carves its runs before filling them and subtracts them from the sibling before it leaves. A split's whole histogram step is one region, two when row blocks leave partials, and then the reduce owns the subtraction instead, because a feature's arena is not final until its partials are in. Debug builds assert the fused carve reached every run. The failure that needs catching is a work list that drops a range, which drops that range's carve and its fill together: no read touches the missing run, the slot stays the placeholder, and the split scan skips the feature without a word.

**The partition team is two-valued.** A leaf-plane parent scatters its rows into two children with the level plane's blocked partition, but on the whole configured team or on one worker, never in between (`partition_workers`). The reason is the same team-rebuild cost: a partition region sized to the parent, sitting between two fills sized to the configured count, cost more than the partition saved when it was measured at 12 threads. A parent that cannot give every worker 4096 rows takes the serial scan. When it does spread, it takes four blocks per worker rather than an equal cut, so the dynamic schedule can absorb asymmetric cores instead of stranding the slowest one with a whole share.

## The determinism contract

**Models and predictions are bit-identical at a fixed configured thread count on the host plane**, decision 7's contract. From v0.2.0 to decision 49 the codebase held a stronger any-thread-count guarantee (no parallel site performed a cross-thread FP reduction); the row-wise fill spends it deliberately: a node that more than one thread range touches accumulates one partial per extra range, and their reduce in ascending range order makes its sums a function of the partition. Everything else (nodes a single range covers whole, the u16 feature-parallel fill, the dense-node fill, split scans, predictions) still matches the serial iteration order exactly, so the contract's dependence on thread count enters only through those reduced sums. The partition depends on the row counts of every node at the level, the fixed row grain, and the configured thread count, never on scheduling or timing. That first dependency is wider than the per-node scheme it replaced: the chunk list is cut level-wide, so one node's row count moves another node's cut points and therefore its summation order. A fixed `n_threads` is still reproducible across runs and machines with the same core count under auto. Set `parallel.n_threads` explicitly when reproducibility across machines matters.

The leaf plane lands in the same contract by a different route. It has no level-wide partition, but its fill plan reads `n_threads()` to decide how many row blocks to cut, and every block past the first sends its cells through a partial that the reduce folds back in block order. A one-block plan still matches the serial order exactly however many feature ranges it holds, since a cell takes all its rows from one worker; a multi-block plan does not, and that is the whole of what keys leafwise model bytes to the configured count. Two nodes with the same rows and the same selection therefore fill identically at one thread count and need not at another, which is why the campaign's byte-identity gate pins `parallel.n_threads` at 1, 4, and 8 rather than trusting one of them.

The plane qualifier is load-bearing: this whole doc is about host parallelism, and the contract does not extend to the CUDA growers. Device histogram cells are accumulated by thousands of threads under atomics, so the add order is whatever the scheduler produced and the same device fit writes different model bytes on every run (measured: four repetitions of one `cuda_depthwise` fit at one commit, four different model hashes). The device plane's claim is tolerance equality, not bit equality ([`11-gpu-resident.md`](11-gpu-resident.md)), and comparisons across commits there are argued against the measured run-to-run spread rather than against a hash.

Emitting exactly `n_threads` ranges is what bounds the partials. The ranges touching a node are contiguous, so the runs telescope and a level needs at most `n_threads - 1` partial buffers however many nodes it holds. The price is the schedule: `for_each_index` over `n == n_threads` indices degrades to chunk 1 with nothing left to steal, so this one site takes the static partition that §Scheduling above says dynamic stealing exists to avoid on asymmetric cores. The partial bound and the load balance are the same dial and the fill picks the bound; cutting more ranges per thread for balance would hand the partial count back (decision 105).

For example, a level holds three sparse nodes: A (2500 rows), B (900), C (3100). At grain 1024 each node's rows cut into row chunks of at most 1024, laid node-major in one flat list, eight chunks in all. Four threads take contiguous ranges of that list:

```
chunks: A0 A1 A2 | B0 | C0 C1 C2 C3
thread: T0 T0 T1 | T1 | T2 T2 T3 T3
```

The threads touching each node: A {T0, T1}, B {T1}, C {T2, T3}, always a run of adjacent threads because the chunk list is node-major. The lowest thread of each run writes straight into the node's arena (T0 for A, T1 for B, T2 for C); only the non-lowest members of a run write partials (T1 for A, T3 for C), so this level's reduce merges exactly two partials. B's run has one thread and no non-lowest member, so it needs no partial at all. Partials are bounded by `n_threads - 1` per level because runs overlap only at thread boundaries, and a level has at most `n_threads - 1` of those.

## The CPU fill's four constants

All four are measured, none is tunable, and each is a constant because its reason is structural rather than host-specific (`src/fill/`, decisions 49 and 105).

**Row grain, 1024 rows per chunk.** A chunk is streamed once and never re-walked, so its size gates no cache reuse; it trades only load balance against per-chunk setup, and lazy per-range zeroing plus a histogram-base rebuild only on node change make a chunk nearly free to begin.

**Density cutoff, `rows >= n_rows / 4`.** Above it a node routes to the feature-parallel column fill: per-feature sequential scans into an L1-resident 2KB target, no partials and no merge, and bit-identical at any thread count. Below it the row chunks win, because a row's 128B of bins amortize the fetch at any sparsity.

**Prefetch distance, 16 rows.** Below the root a node's rows are an ascending subset, so one row's bins and the next sit at irregular strides the hardware prefetcher cannot follow: the populate ledger showed the row loop DRAM-latency-bound at depth (the 16M cell, 78s of a 107s fit). The loop pulls the row's bins (two lines at 100 u8 features) and the grad/hess pair a fixed distance ahead. These are reads only, so results are bit-identical. The lookahead reads the *node's* row list rather than the current chunk's, so a chunk boundary costs no dead zone however fine the chunks are cut; only the node's last rows go unprefetched, and they are peeled out so the hot loop carries no per-row bound test. The column fill's gathered arm prefetches too, at 64 rows: it reads one column at a time, so its lookahead is measured in rows of that column and each row costs it only one L1-resident add, where the mirror's row carries a whole strip of them. LightGBM's column fill prefetches the same pattern at the same distance (`dense_bin.hpp`).

**Partials, grow-only and reused.** The storage reaches its high-water mark within the first levels and is kept for the rest of the fit, so its zero fill on the calling thread (and the NUMA page homing that implies) is paid a handful of times per process rather than per level. Workers re-zero a slot on first touch anyway, since it carries stale sums from the previous slice.

## Two libomp gotchas (hard-won)

1. **`thread_local` inside a parallel region** resolves to each *worker's* variable: a buffer sized on the main thread is empty (size 0) on every worker. Take a view or a pointer over the storage before entering the region (every fill in `src/fill/` does this; the original version segfaulted 74 tests).
2. **The Python extension must link libomp statically** and export only its init symbol (`BONSAI_OPENMP_STATIC=ON`). Linking Homebrew's `libomp.dylib` deadlocked the process the moment xgboost built a DMatrix: one OpenMP call stack spanned two libomp images (decision 36).

## Tests

Verified empirically rather than with a dedicated determinism suite: `parallel.n_threads=1` vs default produce byte-identical predictions on YearPredictionMSD (checked at every optimization step of the v0.2.0 round), and the whole 316-test suite runs under the parallel build.
