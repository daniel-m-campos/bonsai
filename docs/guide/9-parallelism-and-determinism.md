# 9 — Parallelism & determinism

## The idea

Histogram GBT parallelizes naturally — features are independent during
histogram building, rows are independent during predict — but the naive
version silently sacrifices something valuable: **reproducibility**.
Floating-point addition is not associative — $(a+b)+c \ne a+(b+c)$ in the
last bit, so any parallel scheme that lets thread scheduling change the
*order* of additions produces models that differ run to run. The reference
libraries mostly accept this (fixed thread count → same model, different
count → close-but-different). bonsai holds a stronger line: **the model is
bit-identical to a serial run at any thread count.**

## The math (of non-determinism)

A histogram cell is a sum of $\sim 10^5$ doubles. Change the addition order and
the result moves in the last ulp; a split whose gain ties within an ulp
can then flip; one flipped split changes a subtree; 200 trees later the
models disagree visibly. Determinism is not about the average case — it's
about ties, and gradient data produces exact ties routinely (symmetric
gradients, duplicated rows).

The design rule that buys any-thread-count determinism: **no
cross-thread floating-point reductions.** Every accumulator must be
written by exactly one thread, in the same order a serial loop would use.
Parallelism then only chooses *which thread* computes each independent
piece — never the order of additions within one.

## In bonsai

- The seam: `parallel::for_each_index(n, f)` in
  [`include/bonsai/parallel.hpp`](../../include/bonsai/parallel.hpp) — an
  OpenMP `parallel for` (dynamic schedule, size-scaled chunks so
  asymmetric P/E cores stay busy) with a serial fallback. Worker count:
  `[parallel] n_threads`, 0 = all cores. Design notes in
  [architecture/7-parallel.md](../architecture/7-parallel.md).
- Feature-parallel histogram fill (`populate_from_rows`,
  [`src/grower.cpp`](../../src/grower.cpp)): each feature's histogram is
  owned by one thread and filled in row order — determinism by ownership,
  not by locks.
- Parallel split scan ([`src/split.cpp`](../../src/split.cpp)): each
  feature finds its own best independently; the winners merge **serially
  in feature order**, so gain ties break exactly as a serial scan would.
- Row-parallel predict / gradients / scatter: one row, one thread, no
  shared accumulator.

What bonsai does *not* do — and the price: row-parallel histogram
building with per-thread partial histograms merged at the end (the
xgboost/lightgbm shape). That's the remaining fit-speed lever for
few-feature datasets, and taking it would relax the contract back to
"deterministic at fixed thread count" because the merge is a cross-thread
FP reduction. The trade is documented, not taken by accident.

## Try it

```bash
bonsai fit -c configs/year_prediction_msd.toml --set parallel.n_threads=1 --model /tmp/t1.msgpack
bonsai fit -c configs/year_prediction_msd.toml --set parallel.n_threads=8 --model /tmp/t8.msgpack
bonsai predict -c configs/year_prediction_msd.toml --model /tmp/t1.msgpack --out /tmp/p1.csv
bonsai predict -c configs/year_prediction_msd.toml --model /tmp/t8.msgpack --out /tmp/p8.csv
cmp /tmp/p1.csv /tmp/p8.csv && echo "bit-identical"
```

Try the same with xgboost's `nthread` and diff the dumped models.

## Gotchas & war stories

- **`thread_local` inside a parallel region is per-worker.** The ordered
  grad/hess gather buffer is a main-thread `thread_local`; naming it
  inside the OpenMP lambda resolved to each *worker's own empty vector*
  and out-of-bounds writes segfaulted 74 tests at once. Fix: capture the
  data pointers before entering the region. If you use OpenMP with any
  TLS, this one is waiting for you.
- **Two OpenMP runtimes in one process deadlock.** The Python extension
  originally linked Homebrew's `libomp.dylib`; the moment xgboost (which
  bundles its own libomp) built a DMatrix in the same process, one OpenMP
  call stack spanned *two different libomp images* and parked forever at a
  join barrier. Fix, standard for wheels: link libomp **statically** into
  the module and export only the module-init symbol
  (`BONSAI_OPENMP_STATIC=ON`; decision 36). bonsai, xgboost, and lightgbm
  now interleave in one process.
- **User time lies under OpenMP.** Idle workers spin-wait (blocktime), so
  `user` CPU time looks saturated even when threads are starved at
  barriers. Profile with a sampler and read the *stacks*, not the totals —
  that's how the join-barrier bottlenecks in the v0.2.0 round were found.
