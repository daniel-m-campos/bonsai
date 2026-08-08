# 7: Parallelism

> **Status:** done (decision 32, tag `v0.2.0`). This doc replaced the planned
> "ParallelBackend concept with OpenMP + std::execution impls"; what shipped
> is deliberately smaller. Concept-level walkthrough with the war stories:
> [guide/9-parallelism-and-determinism.md](../guide/9-parallelism-and-determinism.md).

## The seam

All parallelism flows through one function ([`include/bonsai/parallel.hpp`](../../include/bonsai/parallel.hpp)):

```cpp
template <typename F> void parallel::for_each_index(size_t n, F &&f);
```

Runs `f(i)` for `i in [0, n)`. OpenMP body when built with `BONSAI_OPENMP`
(default), plain loop otherwise. Callers never see the difference. The
worker count comes from `[parallel] n_threads` (0 = auto: hardware threads
capped at 16), applied process-wide by `resolve_config` / the Python module.

Auto is capped because the per-level parallel sections are short: on hosts
where the core count far exceeds per-level parallelism, OpenMP spin-wait at
the section barriers dominates useful work: a 60-vCPU host ran the MSD fit
10× slower at 60 threads than at 16 (issue #2, decision 44). An explicit
`n_threads = N` passes through uncapped; on oversubscribed many-core hosts
`OMP_WAIT_POLICY=passive` (or `KMP_BLOCKTIME=0`) is the operator knob that
makes idle workers sleep instead of spin.

Why not the proposed `ParallelBackend` concept dispatched like objectives
and growers? One implementation doesn't earn a typelist dimension. The
`std::execution` backend was dropped: it would have added a TBB dependency
and a second code path to keep deterministic, for a benchmark-only payoff.
The free-function seam keeps the door open: a second backend would slot in
behind the same signature (see `6-dispatch.md` §"Backend placement").

## Scheduling

`schedule(dynamic, chunk)` with `chunk = max(1, n / (n_threads * 4))`:

- Feature loops (`n ≈ 90`) degrade to chunk 1: dynamic stealing keeps
  asymmetric performance/efficiency cores (Apple M-series) busy instead of
  letting E-cores drag the static-partition barrier.
- Row loops (`n ≈ 500k`) get a few thousand indices per chunk: scheduler
  lock traffic stays negligible.

## Names

One noun, one meaning, for the CPU fill's work (`src/grower.cpp`). The
entity nouns it fills into (cell, histogram, arena, slice, feature stride)
are defined in [`2-histogram.md`](2-histogram.md).

| Term | What it names |
| --- | --- |
| row chunk | a contiguous range of one node's rows, the unit of thread work, `RowChunk` |
| partition | the cut of a level's chunk list into one contiguous range per thread |
| partials | the per-(thread, node) scratch histograms, one row of cells per partition slot |
| fill | to accumulate cells from rows |
| reduce | to sum a node's partials into its arena, in ascending thread order |
| route | to send a node to the dense column fill or to the sparse row-chunk fill |

## What is parallel

Unit-parallel histogram fill for u8 (max_bin ≤ 255) data: a level's sparse
nodes are cut into fixed-grain row chunks over the dataset's row-major
mirror, and that node-major chunk list is split into contiguous per-thread
ranges (`CpuHistogramEngine::populate_many`, decisions 49 and 105). Each
range reads its rows' bins as contiguous bytes and accumulates into either
the node's own cells (the lowest-numbered range touching that node) or one
private partial reduced afterwards in ascending range order.
Feature-parallel: histogram fill for u16 data and for dense nodes (grad/hess
gathered into node-row order first so every feature scan reads
sequentially), the levelwise finder's split scan (per-feature bests merged
serially in feature order to preserve the tie-break), histogram
subtraction, the partial reduce, binning, mapper fitting. Node-parallel:
the per-node split scan, one worker per frontier node
(`LevelStep::host_find`), each node's scan serial over its features so the
tie-break is the serial walk's. Row-parallel: predict (both tree types), objective grad/hess,
score updates, CSV row parsing, out-of-bag routing.

## The determinism contract

**Models and predictions are bit-identical at a fixed configured thread
count**, decision 7's contract. From v0.2.0 to decision 49 the codebase
held a stronger any-thread-count guarantee (no parallel site performed a
cross-thread FP reduction); the row-wise fill spends it deliberately: a node
that more than one thread range touches accumulates one partial per extra
range, and their reduce in ascending range order makes its sums a function
of the partition. Everything else (nodes a single range covers whole, the
u16 feature-parallel fill, the dense-node fill, split scans, predictions)
still matches the serial iteration order exactly, so the contract's
dependence on thread count enters only through those reduced sums. The
partition depends on the row counts of every node at the level, the fixed
row grain, and the configured thread count, never on scheduling or timing.
That first dependency is wider than the per-node scheme it replaced: the
chunk list is cut level-wide, so one node's row count moves another node's
cut points and therefore its summation order. A fixed `n_threads` is still
reproducible across runs and machines with the same core count under auto.
Set `parallel.n_threads` explicitly when reproducibility across machines
matters.

Emitting exactly `n_threads` ranges is what bounds the partials. The ranges
touching a node are contiguous, so the runs telescope and a level needs at
most `n_threads - 1` partial buffers however many nodes it holds. The price
is the schedule: `for_each_index` over `n == n_threads` indices degrades to
chunk 1 with nothing left to steal, so this one site takes the static
partition that §Scheduling above says dynamic stealing exists to avoid on
asymmetric cores. The partial bound and the load balance are the same dial
and the fill picks the bound; cutting more ranges per thread for balance
would hand the partial count back (decision 105).

For example, a level holds three sparse nodes: A (2500 rows), B (900), C
(3100). At grain 1024 each node's rows cut into row chunks of at most 1024,
laid node-major in one flat list, eight chunks in all. Four threads take
contiguous ranges of that list:

```
chunks: A0 A1 A2 | B0 | C0 C1 C2 C3
thread: T0 T0 T1 | T1 | T2 T2 T3 T3
```

The threads touching each node: A {T0, T1}, B {T1}, C {T2, T3}, always a run
of adjacent threads because the chunk list is node-major. The lowest thread
of each run writes straight into the node's arena (T0 for A, T1 for B, T2
for C); only the non-lowest members of a run write partials (T1 for A, T3
for C), so this level's reduce merges exactly two partials. B's run has one
thread and no non-lowest member, so it needs no partial at all. Partials are
bounded by `n_threads - 1` per level because runs overlap only at thread
boundaries, and a level has at most `n_threads - 1` of those.

## The CPU fill's four constants

All four are measured, none is tunable, and each is a constant because its
reason is structural rather than host-specific
(`src/grower.cpp`, decisions 49 and 105).

**Row grain, 1024 rows per chunk.** A chunk is streamed once and never
re-walked, so its size gates no cache reuse; it trades only load balance
against per-chunk setup, and lazy per-range zeroing plus a histogram-base
rebuild only on node change make a chunk nearly free to begin.

**Density cutoff, `rows >= n_rows / 4`.** Above it a node routes to the
feature-parallel column fill: per-feature sequential scans into an
L1-resident 2KB target, no partials and no merge, and bit-identical at any
thread count. Below it the row chunks win, because a row's 128B of bins
amortize the fetch at any sparsity.

**Prefetch distance, 16 rows.** Below the root a node's rows are an
ascending subset, so one row's bins and the next sit at irregular strides
the hardware prefetcher cannot follow: the populate ledger showed the row
loop DRAM-latency-bound at depth (the 16M cell, 78s of a 107s fit). The
loop pulls the row's bins (two lines at 100 u8 features) and the grad/hess
pair a fixed distance ahead. These are reads only, so results are
bit-identical. The lookahead reads the *node's* row list rather than the
current chunk's, so a chunk boundary costs no dead zone however fine the
chunks are cut; only the node's last rows go unprefetched, and they are
peeled out so the hot loop carries no per-row bound test.

**Partials, grow-only and reused.** The storage reaches its high-water
mark within the first levels and is kept for the rest of the fit, so its
zero fill on the calling thread (and the NUMA page homing that implies) is
paid a handful of times per process rather than per level. Workers re-zero a
slot on first touch anyway, since it carries stale sums from the previous
slice.

## Two libomp gotchas (hard-won)

1. **`thread_local` inside a parallel region** resolves to each *worker's*
   variable: a buffer sized on the main thread is empty (size 0) on every
   worker. Capture raw pointers before entering the region
   (`populate_from_rows` does this; the original version segfaulted 74 tests).
2. **The Python extension must link libomp statically** and export only its
   init symbol (`BONSAI_OPENMP_STATIC=ON`). Linking Homebrew's
   `libomp.dylib` deadlocked the process the moment xgboost built a DMatrix:
   one OpenMP call stack spanned two libomp images (decision 36).

## Tests

Verified empirically rather than with a dedicated determinism suite:
`parallel.n_threads=1` vs default produce byte-identical predictions on
YearPredictionMSD (checked at every optimization step of the v0.2.0 round),
and the whole 316-test suite runs under the parallel build.
