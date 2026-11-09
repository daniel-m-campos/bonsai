# 9. Parallelism & determinism

## The idea

Histogram GBT parallelizes naturally (features are independent during histogram building, rows are independent during predict), but the naive version silently sacrifices something valuable: **reproducibility**. Floating-point addition is not associative: $(a+b)+c \ne a+(b+c)$ in the last bit, so any parallel scheme that lets thread scheduling change the *order* of additions produces models that differ run to run. The reference libraries mostly accept this (fixed thread count → same model, different count → a different model). bonsai's contract is the same, held deliberately rather than by accident: **when training on the CPU, the model is bit-identical across runs at a fixed thread count**, and for everything *except* the u8 histogram fill (the fill for data stored as one-byte bin indices, at most 256 bins per feature), bit-identical to a serial run at any count. The one exception is the row-wise fill for sparse u8 nodes: it gives up bit-identity across thread counts (a fixed thread count still reproduces exactly) and makes a fit 1.6 to 1.7× faster on an M2 with 8 threads (measured cells below). The paragraph after the list in [In bonsai](#in-bonsai) tells that story.

The "on the CPU" is not a hedge, it is the boundary. GPU training adds each row's gradient and hessian into shared histogram cells with atomic adds (memory updates no other thread can interrupt), and the order of those adds depends on how the GPU schedules its threads, not on the configuration. The cells are int64 fixed point, so the sum is one integer in any order and **the same GPU fit produces the same model bytes every run** on one device and one build. The device does not promise the rest of the CPU rule. A device fit can differ across devices and toolkits, because the gradient kernels' exp and log (transcendental functions) and fused multiply-adds round differently on different hardware and compilers. It also differs from the host fit, which sums cells in float: one tree grown on the host and on the device from the same gradients gives every row a value within an absolute 1e-4 of the other ([chapter 10](10-gpu-training.md)).

## The math (of non-determinism)

A histogram cell is a sum of $\sim 10^5$ floats. Change the addition order and the result can differ in its last bit (one ulp, the gap between two adjacent floating-point numbers); a split whose gain ties within one ulp can then flip; one flipped split changes a subtree; 200 trees later the models disagree visibly. Determinism is not about the average case: it's about ties, and gradient data produces exact ties routinely (symmetric gradients, duplicated rows).

The design rule that buys determinism: **no unordered cross-thread floating-point reductions.** Every accumulator is either written by exactly one thread in serial order, or assembled from per-thread partials merged in a *fixed* order, so the addition order is a pure function of configuration, never of scheduling.

## In bonsai

- The seam: `parallel::for_each_index(n, f)` in [`include/bonsai/parallel.hpp`](../../include/bonsai/parallel.hpp): an OpenMP `parallel for` (dynamic schedule, size-scaled chunks so asymmetric P/E cores stay busy) with a serial fallback. Worker count: `[parallel] n_threads`, 0 = auto (hardware threads, capped at 16 and at any cgroup CPU quota, so in a container with a CPU quota, auto mode uses at most the number of threads the quota allows, not the number of cores the machine reports). An explicit count over that quota is left alone and warns once on stderr, because the resolved count is part of the model's identity and a clamp would change the model bytes a container produces. Design notes in [the invariants](../invariants.md).
- Histogram fill ([`src/grower.cpp`](../../src/grower.cpp) and [`src/fill/`](../../src/fill/)): the routing is per node, not per bin width. u16 bins keep the feature-parallel shape (`fill_columns`): each feature's histogram owned by one thread, filled in row order, determinism by ownership. u8 bins take that same fill whenever the node holds at least a quarter of the rows (`k_col_fill_den = 4`), and only sparser u8 nodes take the row-wise fill (`fill_sparse`, reached through the `populate_many` router; decisions 49 and 105). *Row chunks* stream a row-major mirror, tiled into 2048-feature blocks and walked in runs of 512 features (decision 151). For each run, a thread zeroes and fills the histogram of the child with fewer rows, then gets the sibling's histogram by subtracting that one from the parent's ([the subtraction trick](2-binning-and-histograms.md)). The runs are kept small so both histograms for a run stay in cache. A node touched by more than one thread range owns a partial per extra range, reduced in fixed range order. The leafwise grower takes that same level fill when its leaf budget cannot bind (`max_leaves` 0, or at least 2^`max_depth`) and the node's histogram arena is under 2 MiB (decision 147), and otherwise fills one node at a time, feature-parallel first and row-blocked only when the row is too narrow to give every worker its own cache line, with the blocks' partials reduced in ascending block order. The partition derives from the configured thread count, which is exactly where the fixed-count contract comes from. So the dense nodes near the root stay bit-identical at any thread count, and the contract narrows only for the sparse nodes deeper in the tree.
- Split scan ([`src/split.cpp`](../../src/split.cpp)): the levelwise finder scans eight features at a time in lockstep on the same vector lanes (decision 157), one task per group, and merges the groups' winners **serially in feature order**, so gain ties break exactly as a serial scan would; a host without the vector lanes scans one feature per task and merges the same way. The per-node finder likewise scans eight features at a time on vector lanes, every lane doing the scalar scan's operations in the scalar scan's order, and merges the lanes with the same strict-greater rule, so it is bit-identical to the serial walk it replaced; a lane under a monotone constraint carries its direction and the node's leaf bounds, and the prefetch of the next group's cells changes no arithmetic and no order. How the lanes work is [case E7](../learn/engine/7-the-lane-finder.md).
- Node-parallel split scan ([`src/step/level.hpp`](../../src/step/level.hpp)): the level step passes every node of the current level (the frontier: all nodes waiting for a split) to `find_parallel`. It splits each node's features into ranges across all workers, so every worker has features to scan even at the top levels, where one node holds all the rows. Each node's range bests then merge in feature order, so the result is the serial scan's at any thread count.
- Row-parallel predict / gradients / scatter: one row, one thread, no shared accumulator.

Relaxing the contract was a deliberate trade, not an accident ([decision 49](../decisions.md#49-row-wise-histogram-fill-over-a-row-major-u8-mirror-determinism-relaxes-to-fixed-thread-count)): the profiler showed deep sparse nodes filling at about 0.7G adds/s against 3.3G for large dense nodes (a fifth of the dense rate), a cache problem no feature-parallel scan could fix, and the row-wise fill cut fit time 1.6 to 1.7× on an M2 with 8 threads (1M x 100, 4M x 100 and 1M x 512 cells, u8 bins) for the narrowing above. The models are still exactly reproducible; you just have to hold `n_threads` fixed, like every reference library. Two designs were dropped. Splitting the feature-parallel fill across two accumulators (one for even rows, one for odd) promised 1.6× in a microbenchmark but gained nothing in the real loop, because the microbenchmark's arrays fit in cache and the streaming loop's do not. An automatic layout choice based on last-level cache size was rejected because it would have made *models* hardware-dependent.

## Try it

```python {.run}
import numpy as np
import bonsai

rng = np.random.default_rng(0)
X = rng.normal(size=(5000, 10)).astype(np.float32)
y = (X[:, 0] * 2.0 + X[:, 1] + rng.normal(0, 0.1, 5000)).astype(np.float32)


def fit(n_threads):
    # continuous features at max_bin=511 take u16 bins: bit-identical at any thread count.
    return bonsai.BonsaiRegressor(n_iters=50, max_bin=511, n_threads=n_threads).fit(X, y)


p1 = np.asarray(fit(1).predict(X))
p8 = np.asarray(fit(8).predict(X))
print("bit-identical:", np.array_equal(p1, p8))
```

With u16 bins (some feature has more than 256 bins, which needs `max_bin > 256` and enough distinct values) this compares equal at *any* thread pair; with u8 bins, re-run either side at the same `n_threads` twice and compare those: run-to-run identity is the contract. Try the same with XGBoost's `nthread` and diff the dumped models.

## Gotchas & war stories

- **`thread_local` inside a parallel region is per-worker.** The ordered grad/hess gather buffer is a main-thread `thread_local`; naming it inside the OpenMP lambda resolved to each *worker's own empty vector* and out-of-bounds writes segfaulted 74 tests at once. Fix: capture the data pointers before entering the region. Any OpenMP code that uses thread-local storage (TLS) can hit this bug.
- **Two OpenMP runtimes in one process deadlock.** The Python extension originally linked Homebrew's `libomp.dylib`; the moment XGBoost (which bundles its own libomp) built a DMatrix in the same process, one OpenMP call stack spanned *two different libomp images* and parked forever at a join barrier. Fix, standard for wheels: link libomp **statically** into the module and export only the module-init symbol (`BONSAI_OPENMP_STATIC=ON`; decision 36). bonsai, XGBoost, and LightGBM now interleave in one process.
- **NUMA first-touch can halve your fill rate silently.** The row-wise fill's partial histograms were first zeroed by the main thread, which *homed every page on one socket* of a dual-socket EPYC; workers on the other socket then paid remote-memory latency for every add (2× fill penalty, 3.1× thread scaling). Per-block zeroing into `make_unique_for_overwrite` storage fixed both (5.8× scaling). Where memory is *touched first* decides where it lives.
- **User CPU time overstates useful work under OpenMP.** Idle workers spin-wait for a set period (the blocktime) before sleeping, so `user` time looks saturated even when threads are stuck waiting at barriers. Profile with a sampler and read the *stacks*, not the totals: that's how the join-barrier bottlenecks in the v0.2.0 round were found.
