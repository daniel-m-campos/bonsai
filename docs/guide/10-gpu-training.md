# 10 — GPU training

## The idea

Everything expensive in histogram GBT is a *data-parallel reduction or scatter over rows*: bin the raw values, sum gradients into cells, scan cells for the best split, route rows to children. A GPU has ~10× the memory bandwidth of a CPU socket, and bandwidth is exactly what these loops are starved for. What a GPU does **not** have is cheap access to the sequential, branchy part — deciding which nodes to split, bookkeeping the tree, enforcing constraints.

So the design question is not "port the algorithm to CUDA" but "draw the boundary": the device owns every per-row loop, the host owns every per-node decision, and the two exchange the smallest possible messages. In bonsai that boundary has a name — the **level transaction** ([architecture/14](../architecture/14-engine-narrative.md)) — and the CPU and CUDA engines implement the *same* transactions, so a grower cannot tell which backend it is running on. That is the whole trick; the rest of this chapter is the three problems that make the device side interesting: atomics and precision, keeping data resident, and knowing what to move next.

## The math

**Histogram accumulation is a race by construction.** Thousands of threads add `(g_i, h_i)` into 255 shared cells, so the adds must be atomic, and the order they land in is whatever the scheduler produced. Floating-point addition is not associative:

```math
\Big(\sum_i g_i\Big)_{\text{GPU}} \;=\; \Big(\sum_i g_i\Big)_{\text{exact}} + \varepsilon, \qquad |\varepsilon| \lesssim n\,u\,\max|g_i|
```

with $u$ the unit roundoff. Two consequences you must design for rather than wish away: GPU histograms match CPU histograms **to tolerance, not bit-exactly**, and two GPU runs of the same fit can differ in the last ulps.

**The precision scheme** keeps $\varepsilon$ small without paying for 64-bit atomics (which real hardware serializes through a compare-and-swap loop): accumulate in **float** within a chunk of at most 32k rows — shared-memory float atomics are native and fast — then merge the per-chunk partials in **double**. The float stage bounds $n$ in the error term at 32k; the double merge makes the cross-chunk sum effectively exact. Split *scoring* then happens in double on cells that are already sums of ≤32k floats.

**The subtraction trick survives intact** ([chapter 2](2-binning-and-histograms.md)): children of a split partition the parent's rows, so the device builds the smaller child's histogram and derives the larger by a cell-wise subtract kernel — in double, on resident buffers, without the host ever seeing a histogram.

## In bonsai

The entire backend is one CUDA C++ translation unit, [`src/cuda/histogram_engine.cu`](../../src/cuda/histogram_engine.cu) plus its kernel header [`src/cuda/detail/kernels.cuh`](../../src/cuda/detail/kernels.cuh), compiled by the project's own clang (`-x cuda`), not nvcc — same C++23, same libc++ as the rest of the build. Builds without CUDA link a stub that throws; `bonsai::cuda_available()` is the runtime predicate and the `cuda_*` growers are registered everywhere, so a GPU-trained model predicts fine on a CPU-only binary.

Follow one fit through the transactions:

- **Ingest** — [`cuda_ingest`](../../src/cuda/histogram_engine.cu) (decision 54): raw feature values stream to the device in ~64MB chunks and a kernel bins them with a `lower_bound` that reproduces the host `transform` *exactly* — same cuts, same comparisons, bit-identical bin ids. The product, an `IngestPlane`, rides on the `Dataset` as an opaque receipt; host binned columns are never materialized unless a host consumer asks. At 16M×100 this replaced 4.6s of host binning plus a 1.6GB upload with ~0.5–0.9s of transfer+kernel (host-dependent).
- **`begin_tree`** — the per-tree gradient upload, interleaved into `(g,h)` pairs on device.
- **`open_level`** (find) — level histograms live in a slot-indexed device buffer that ping-pongs between parent and child levels. The find kernel gives each (node, feature) pair one **warp**: a shuffle-based prefix scan over the cells, per-lane gain scoring, and a shuffle argmax that reduces to the best split with a fixed tie-break. Only the per-node *decisions* (feature, bin, gain, child sums — a few hundred bytes) cross back to the host.
- **`apply_level`** (partition) — route/count/scan/scatter kernels move each split node's row segment into stable left/right children entirely on device; the host receives two integers per split (the child row counts). Stability matters: it is what keeps the device partition semantically identical to the host one.
- **Histogram build** — the shared-memory chunked kernel for large children, a direct-to-global kernel for small ones (a full shared-memory pipeline per 40-row node is all overhead), then the subtract kernel for every larger sibling.
- **`end_tree`** — the epilogue (decision 53): the host sends the finished tree's node-value table (a few KB); a kernel maps every row's resident leaf assignment to its training value; values and leaf ids come home in two bulk copies. Before this transaction existed, the host looped over 16M rows per tree.

The host control plane between transactions is [`src/level_step.hpp`](../../src/level_step.hpp) and the growers — the same `plan_level`/`commit_children` logic the CPU path uses, because it *is* the CPU path's logic. When the device declines (a feature's bins exceed the shared-memory budget), the same grower runs the same tree on the host engine mid-fit.

What this buys, measured (same-pod L40S, 16M rows × 100 features × 100 trees, `fit()` timed end-to-end including binning): **bonsai 26.9s vs xgboost-GPU 28.9s, at a third of the host memory (7.3GB vs 22.1GB)**. The path from 3× slower to ahead is [chapter 11](11-performance-engineering.md).

## Try it

No CUDA device on your machine? The [RunPod runbook](../ops/runpod-runbook.md) gets you a validated GPU session for well under a dollar.

```bash
# The device suite: SKIPs without a GPU, exercises real kernels with one.
./build-cuda/tests/bonsai_tests "[cuda]"

# Any fit, with the device profile lines:
BONSAI_CUDA_PROFILE=1 BONSAI_GROW_PROFILE=1 \
  bonsai fit --config configs/california_housing.toml \
  --set dispatch.grower_name=cuda_depthwise
```

Read the `cuda-profile` line first: `upload` vs `gpu` vs `cpu_fallback` tells you whether the device path even ran (a large `cpu_fallback` means `max_bin` pushed past the shared-memory ceiling). Then `cuda-upload-decomp` splits every transfer by transaction — this is the line every optimization in chapter 11 was priced against.

```bash
# CPU vs GPU on the same data — expect tolerance-level agreement, not bit-equality:
uv run scripts/compare.py --config configs/california_housing.toml \
    --growers depthwise,cuda_depthwise
```

## Gotchas & war stories

- **The allocator can cost more than the kernels.** On GeForce drivers, the default CUDA mempool returns freed memory to the OS at every sync, and the resulting alloc/free churn synchronizes the whole process — a measured **11–14 seconds per fit** on an RTX 5090 before decision 48 pinned the pool's release threshold. If a GPU fit is inexplicably slow, suspect memory management before kernels.
- **Some rented GPUs are just broken.** A whole class of hosts showed ~300µs per device synchronization (healthy: ~4µs) — invariant to every software knob, an ASPM/IRQ-level defect. Multiply by ~25k syncs per fit and the machine is unusable. The bench harness front-loads a 30-second sync probe and rejects hosts above 50µs (decision 48).
- **A demoted split is not the same thing on both planes.** `demote_empty_splits` un-splits a node whose child would be empty. On the host that is bookkeeping; on the device the rows were *already scattered* and stamped, so demoting orphaned their leaf assignments — one mega-leaf per tree at the rightmost spine, quality collapsing only at 16M rows where double-precision sums stopped masking it (PR #29). Control-plane operations must know which plane's invariants they touch.
- **"First use is single-threaded" is an assumption, not a law.** The lazy host materialization of device-binned columns was first guarded by nothing, on the precedent of another lazy member — but its first consumer runs inside a parallel loop, and the race corrupted the heap *intermittently* (one clean validation pass meant nothing). It is now a `call_once` (PR #37). On the same theme: two GPU fits of identical inputs differ in atomic order, so GPU-vs-GPU test tolerances belong at the suite's 1e-4, not tighter.
- **Byte-identity is still available where it matters.** Device *binning* is bit-identical to host binning (same cuts, same `lower_bound` — integer outputs, no atomics), which is what lets a device-binned dataset train the same model as a host-binned one and lets the test suite say so exactly.
