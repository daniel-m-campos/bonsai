# 7 — Parallelism

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
(default), plain loop otherwise — callers never see the difference. The
worker count comes from `[parallel] n_threads` (0 = auto: hardware threads
capped at 16), applied process-wide by `resolve_config` / the Python module.

Auto is capped because the per-level parallel sections are short: on hosts
where the core count far exceeds per-level parallelism, OpenMP spin-wait at
the section barriers dominates useful work — a 60-vCPU host ran the MSD fit
10× slower at 60 threads than at 16 (issue #2, decision 44). An explicit
`n_threads = N` passes through uncapped; on oversubscribed many-core hosts
`OMP_WAIT_POLICY=passive` (or `KMP_BLOCKTIME=0`) is the operator knob that
makes idle workers sleep instead of spin.

Why not the proposed `ParallelBackend` concept dispatched like objectives
and growers? One implementation doesn't earn a typelist dimension. The
`std::execution` backend was dropped: it would have added a TBB dependency
and a second code path to keep deterministic, for a benchmark-only payoff.
The free-function seam keeps the door open — a second backend would slot in
behind the same signature (see `6-dispatch.md` §"Backend placement").

## Scheduling

`schedule(dynamic, chunk)` with `chunk = max(1, n / (n_threads * 4))`:

- Feature loops (`n ≈ 90`) degrade to chunk 1 — dynamic stealing keeps
  asymmetric performance/efficiency cores (Apple M-series) busy instead of
  letting E-cores drag the static-partition barrier.
- Row loops (`n ≈ 500k`) get a few thousand indices per chunk — scheduler
  lock traffic stays negligible.

## What is parallel

Unit-parallel histogram fill for u8 (max_bin ≤ 255) data: a level's nodes
become row-block work units over the dataset's row-major mirror
(`CpuHistogramEngine::populate_many`, decision 49) — each unit reads its
rows' bins as contiguous strips and accumulates into either the node's own
cells (single-block nodes) or a private partial slab merged in fixed block
order. Feature-parallel: histogram fill for u16 data (grad/hess gathered
into node-row order first so every feature scan reads sequentially), split
scans (per-feature bests merged serially in feature order to preserve the
tie-break), histogram subtraction, partial-slab merges, binning, mapper
fitting. Row-parallel: predict (both tree types), objective grad/hess,
score updates, CSV row parsing, out-of-bag routing.

## The determinism contract

**Models and predictions are bit-identical at a fixed configured thread
count** — decision 7's contract. From v0.2.0 to decision 49 the codebase
held a stronger any-thread-count guarantee (no parallel site performed a
cross-thread FP reduction); the row-wise fill spends it deliberately: nodes
large enough to split into multiple row blocks accumulate per-block partial
histograms whose fixed-order merge makes sums a function of the block
count, which derives from `n_threads`. Everything else — single-block
nodes, the u16 feature-parallel fill, split scans, predictions — still
matches the serial iteration order exactly, so the contract's dependence on
thread count enters only through multi-block node sums. Block counts depend
on node size, selection width, total selected bins, and the configured
thread count — never on scheduling or timing — so a fixed `n_threads` is
reproducible across runs and machines with the same core count under auto.
Set `parallel.n_threads` explicitly when reproducibility across machines
matters.

## Two libomp gotchas (hard-won)

1. **`thread_local` inside a parallel region** resolves to each *worker's*
   variable — a buffer sized on the main thread is empty (size 0) on every
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
