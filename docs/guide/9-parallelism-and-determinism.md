# 9 — Parallelism & determinism

## The idea

Histogram GBT parallelizes naturally — features are independent during
histogram building, rows are independent during predict — but the naive
version silently sacrifices something valuable: **reproducibility**.
Floating-point addition is not associative — $(a+b)+c \ne a+(b+c)$ in the
last bit, so any parallel scheme that lets thread scheduling change the
*order* of additions produces models that differ run to run. The reference
libraries mostly accept this (fixed thread count → same model, different
count → close-but-different). bonsai's contract is the same, held
deliberately rather than by accident: **the model is bit-identical across
runs at a fixed thread count** — and for everything *except* the u8
histogram fill, bit-identical to a serial run at any count. The one
exception was bought, measured, and is this chapter's closing story.

## The math (of non-determinism)

A histogram cell is a sum of $\sim 10^5$ doubles. Change the addition order and
the result moves in the last ulp; a split whose gain ties within an ulp
can then flip; one flipped split changes a subtree; 200 trees later the
models disagree visibly. Determinism is not about the average case — it's
about ties, and gradient data produces exact ties routinely (symmetric
gradients, duplicated rows).

The design rule that buys determinism: **no unordered cross-thread
floating-point reductions.** Every accumulator is either written by
exactly one thread in serial order, or assembled from per-block partials
merged in a *fixed* block order — so the addition order is a pure function
of configuration, never of scheduling.

## In bonsai

- The seam: `parallel::for_each_index(n, f)` in
  [`include/bonsai/parallel.hpp`](../../include/bonsai/parallel.hpp) — an
  OpenMP `parallel for` (dynamic schedule, size-scaled chunks so
  asymmetric P/E cores stay busy) with a serial fallback. Worker count:
  `[parallel] n_threads`, 0 = all cores. Design notes in
  [architecture/7-parallel.md](../architecture/7-parallel.md).
- Histogram fill ([`src/grower.cpp`](../../src/grower.cpp)): u16 bins
  keep the feature-parallel shape (`fill_feature_parallel`) — each
  feature's histogram owned by one thread, filled in row order,
  determinism by ownership. u8 bins use the row-wise fill
  (`populate_many`/`run_fill`, decision 49): row *blocks* own partial
  histograms over a row-major mirror, merged in fixed block order — the
  block count derives from the configured thread count, which is exactly
  where the fixed-count contract comes from.
- Parallel split scan ([`src/split.cpp`](../../src/split.cpp)): each
  feature finds its own best independently; the winners merge **serially
  in feature order**, so gain ties break exactly as a serial scan would.
- Row-parallel predict / gradients / scatter: one row, one thread, no
  shared accumulator.

An earlier revision of this chapter ended: "row-parallel histogram
building with per-thread partials is the remaining fit-speed lever, and
taking it would relax the contract — documented, not taken by accident."
The lever **was** then taken, deliberately (decision 49): ground-truth
instrumentation showed deep sparse nodes filling at a fifth of the dense
rate — a cache problem no feature-parallel scan could fix — and the
row-wise fill bought a measured 1.6–1.7× on real cells for the narrowed
contract above. The models are still exactly reproducible; you just have
to hold `n_threads` fixed, like every reference library. What did NOT
survive contact with measurement: a per-parity two-way split promising
1.6× in a microbenchmark delivered nothing in the real loop (the
microbenchmark's arrays were cache-resident; the streaming loop's are
not), and an LLC-size-based automatic layout choice was rejected because
it would have made *models* hardware-dependent.

## Try it

```bash
bonsai fit -c configs/year_prediction_msd.toml --set parallel.n_threads=1 --model /tmp/t1.msgpack
bonsai fit -c configs/year_prediction_msd.toml --set parallel.n_threads=8 --model /tmp/t8.msgpack
bonsai predict -c configs/year_prediction_msd.toml --model /tmp/t1.msgpack --out /tmp/p1.csv
bonsai predict -c configs/year_prediction_msd.toml --model /tmp/t8.msgpack --out /tmp/p8.csv
cmp /tmp/p1.csv /tmp/p8.csv && echo "bit-identical"
```

With u16 bins (`max_bin > 255`) this compares equal at *any* thread pair;
with u8 bins, re-run either side at the same `n_threads` twice and compare
those — run-to-run identity is the contract. Try the same with xgboost's
`nthread` and diff the dumped models.

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
- **NUMA first-touch can halve your fill rate silently.** The row-wise
  fill's partial-histogram slabs were first zeroed by the main thread —
  which *homed every page on one socket* of a dual-socket EPYC; workers on
  the other socket then paid remote-memory latency for every add (2× fill
  penalty, 3.1× thread scaling). Per-block zeroing into
  `make_unique_for_overwrite` storage fixed both (5.8× scaling). Where
  memory is *touched first* decides where it lives.
- **User time lies under OpenMP.** Idle workers spin-wait (blocktime), so
  `user` CPU time looks saturated even when threads are starved at
  barriers. Profile with a sampler and read the *stacks*, not the totals —
  that's how the join-barrier bottlenecks in the v0.2.0 round were found.
