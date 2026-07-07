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

Feature-parallel: histogram fill (`populate_from_rows`, with grad/hess
gathered into node-row order first so every feature scan reads
sequentially), split scans (per-feature bests merged serially in feature
order to preserve the tie-break), histogram subtraction, binning, mapper
fitting. Row-parallel: predict (both tree types), objective grad/hess,
score updates, CSV row parsing, out-of-bag routing.

## The determinism contract

**Models and predictions are bit-identical to a serial run at any thread
count.** This is stronger than the fixed-thread-count contract the proposal
committed to (decision 7), and it holds for one reason: no parallel site
performs a cross-thread floating-point reduction. Each index — a feature's
histogram, a row's prediction — is computed by exactly one thread, in the
same iteration order a serial loop would use. FP non-associativity never
gets a chance to matter.

The trade documented for the future: row-parallel histogram building with
per-thread partial histograms + ordered merge (the xgboost/LightGBM shape,
and the main remaining fit-speed lever for few-feature datasets) would
reintroduce merge-order sensitivity and relax the contract back to
decision 7's fixed-N form. Take it consciously.

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
