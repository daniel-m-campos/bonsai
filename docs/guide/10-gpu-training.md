# 10. GPU training

## The idea

Training time in histogram gradient boosted trees (GBT) goes to four parallel loops: binning raw values, summing gradients into histogram cells, scanning cells for the best split, and routing rows to children. Memory bandwidth limits these loops, and a GPU has ~10× the memory bandwidth of a CPU socket. What a GPU does **not** have is cheap access to the sequential part with many conditional branches: deciding which nodes to split, bookkeeping the tree, enforcing constraints.

So the design question is not "port the algorithm to CUDA" but "draw the boundary": the device owns every per-row loop, the host owns every per-node decision, and the two exchange the smallest possible messages. bonsai has a CUDA grower for each of its three growth strategies, `cuda_depthwise`, `cuda_leafwise`, and `cuda_levelwise`, and each draws that boundary in its own way. `depthwise` and `levelwise` share a boundary called the **level transaction** ([decision 53](../decisions.md)): the CPU and CUDA engines implement the *same* transactions, so a grower cannot tell which backend it is running on. `leafwise` draws an analogous boundary, one node per round instead of one level, through its own `LeafStep` seam (decisions 95 to 98). This chapter walks through the level transaction first, since it covers two of the three CUDA growers, then the leaf step. The problems that make the device side interesting, atomics and precision, keeping data resident, and knowing what to move next, apply to all three.

## The math

**Histogram accumulation is a race by construction.** Thousands of threads add `(g_i, h_i)` into 255 shared cells, so the adds must be atomic, and the order they land in is whatever the scheduler produced. Floating-point addition is not associative, so a float cell carries an order-dependent error:

```math
\Big(\sum_i g_i\Big)_{\text{GPU}} \;=\; \Big(\sum_i g_i\Big)_{\text{exact}} + \varepsilon, \qquad |\varepsilon| \lesssim n\,u\,\max|g_i|
```

with $u$ the unit roundoff, and two runs of one fit differ in the last ulps. Integer addition is associative, so the cells are **int64 fixed point**. Once per tree the fill reads $M = n \max_i |g_i|$ and picks the power-of-two scale $s = 2^{61 - \lfloor \log_2 M \rfloor}$; each visit adds $\mathrm{round}(g_i s)$, and any subset of the tree's rows sums to under $2^{62} + n/2$, inside int64 whatever order the adds land in. A power-of-two scale makes the multiply exact, so the one rounding per row is the integer conversion, below $\max|g| \cdot n / 2^{62}$. Two consequences to design for: a device fit is **bit-identical to itself** on one device and build, and it matches the CPU fit, which accumulates float cells, **to tolerance, not bit-exactly**.

**What the scheme costs.** Less than the float fill it replaced. A 64-bit atomic on shared memory (the on-chip memory the threads of a block share) lowers to a compare-and-swap loop, so the shared stage adds each cell as two native 32-bit atomics with the carry taken from the first; the tile's shared footprint doubles from 4 to 8 bytes a cell, and the tile fill block is sized so each streaming multiprocessor (SM, one of the GPU's parallel compute units) stays fully occupied. The same-pod A/B in decision 124 prices the whole scheme at a 7.6 to 8.6% faster fit on an L40S. The float-per-chunk plus double-merge scheme it replaced bounded the float error term at 32k rows a chunk; the integer sum has no such error term, so the 32k-row chunk size is no longer an accuracy bound. The chunk count is still one chunk per 32k rows of the largest node being filled, raised when too few blocks would fill the SMs and capped at 64. Split *scoring* happens in double on cells dequantised once, by one multiply, at the start of the scan.

**The subtraction trick survives intact** ([chapter 2](2-binning-and-histograms.md)): children of a split partition the parent's rows, so the device builds the smaller child's histogram and derives the larger by cell-wise integer subtraction, exact, on resident buffers (the split finder does it in its own sweep), without the host ever seeing a histogram.

## In bonsai: the level transaction

The backend is five CUDA C++ translation units under `src/cuda/` ([`histogram_engine.cu`](../../src/cuda/histogram_engine.cu), [`detail/device_context.cu`](../../src/cuda/detail/device_context.cu), `mapper_fit.cu`, `predict.cu` and `shap.cu`), sharing the kernel header [`src/cuda/detail/kernels.cuh`](../../src/cuda/detail/kernels.cuh) and other headers in `src/cuda/detail/`, compiled by the project's own clang (`-x cuda`), not nvcc, same C++23, same libc++ as the rest of the build. Builds without CUDA link a stub that throws; `bonsai::cuda_available()` is the runtime predicate and the `cuda_*` growers are registered everywhere, so a GPU-trained model predicts fine on a CPU-only binary.

The device context splits its resident memory into six planes, each a struct with its own lifetime ([`src/cuda/detail/device_context.cuh`](../../src/cuda/detail/device_context.cuh)):

| Plane | Holds | Lifetime |
|---|---|---|
| `DeviceData` | the binned matrix | once per fit |
| `GradientPlane` | per-tree gradients, interleaved to `float2` | once per tree |
| `LevelPipeline` | rows, histograms, staging buffers | once per level |
| `LeafPipeline` | the leafwise histogram slot pool and segment map | once per tree |
| `ResidentPlane` | labels, scores, and resident-objective state | once per fit |
| `EvalPlane` | the validation set's bins, labels and scores, plus the loss partials | once per fit |

Dividing memory by lifetime keeps an upload made once per fit from being repeated for every tree, because each buffer's lifetime is part of its type. Naming the planes also makes the boundary crossings countable, which is what [chapter 11](11-performance-engineering.md)'s pricing rests on.

Follow one fit through the transactions:

- **Ingest**: [`cuda_ingest`](../../src/cuda/histogram_engine.cu) (decision 54): raw feature values stream to the device in ~64MB chunks and a kernel bins them with a `lower_bound` that reproduces the host `transform` *exactly*: same cuts, same comparisons, bit-identical bin ids. The product, an `IngestPlane`, rides on the `Dataset` as an opaque receipt; host binned columns are never materialized unless a host consumer asks. At 16M×100 this replaced 4.6s of host binning plus a 1.6GB upload with ~0.5–0.9s of transfer+kernel (host-dependent).
- **`begin_tree`**: the per-tree gradient upload, interleaved into `(g,h)` pairs on device.
- **`open_level`** (find): level histograms live in a slot-indexed device buffer that ping-pongs between parent and child levels. The find kernel gives each (node, feature) pair one **warp** (a group of 32 threads that execute together): a shuffle-based prefix scan over the cells, per-lane gain scoring, and a shuffle argmax that reduces to the best split with a fixed tie-break. Only the per-node *decisions* (feature, bin, gain, child sums, a few hundred bytes) cross back to the host.
- **`apply_level`** (partition): route/count/scan/scatter kernels move each split node's row segment into stable left/right children entirely on device; the host receives two integers per split (the child row counts). Stability matters: it is what keeps the device partition semantically identical to the host one.
- **Histogram build**: the shared-memory chunked kernel for large children, a direct-to-global kernel for small ones (a full shared-memory pipeline per 40-row node is all overhead), and the finder derives each larger sibling's cells by integer subtraction (parent minus smaller child) as it sweeps them; a separate derive kernel runs only for narrow levelwise levels, where too few features are selected to fill the SMs.
- **`end_tree`**: the epilogue (decision 53): the host sends the finished tree's node-value table (a few KB); a kernel maps every row's resident leaf assignment to its training value; values and leaf ids come home in two bulk copies. Before this transaction existed, the host looped over 16M rows per tree.

The host control plane between transactions is [`src/step/level.hpp`](../../src/step/level.hpp) and the growers, the same `plan_level`/`commit_children` logic the CPU path uses, because it *is* the CPU path's logic. There is no fallback across that boundary, per node or per tree: an engine grows a whole tree resident or refuses it in `begin_root`, with an error naming the limit it hit (a feature's bins past the shared-memory budget, say) and the `device="cpu"` remedy. A fit that asked for a device never quietly trains somewhere else, which is the same standard the [benchmark protocol](../results/benchmark-protocol.md) holds the reference libraries to.

What stays on the host stays there on purpose ([decision 41](../decisions.md)). Split decisions cross the bus down every level, because the grow loop must observe each level's outputs before opening the next; that pins one small device-to-host sync per level, about 800 per fit, each 10 to 20 microseconds on a healthy host. The mapper fit runs on the device, but the seeded row-sample draw behind it stays on the host: the sort and the cuts run on the device and reproduce the host's cut bytes exactly ([decision 140](../decisions.md)). The host-device split forbids two things: a fallback to the CPU partway through a fit, and any move of work between host and device that changes accumulation order or precision. In return, every model bonsai ships is either bit-reproducible or within `1e-4` of one that is ([determinism](../learn/determinism-as-a-contract.md)).

What this buys is measured on [the perf standings](../results/results/perf.md), which own the digits and are refreshed with each release. At the gpu-tall scenario (16M x 128) every bonsai device plane fits several times faster than the reference GPU trainer it is paired with, in a fraction of its peak host memory. Most of the gap is ingest: bonsai bins on the device, while XGBoost sketches on the host and ships the result over the bus, and the standings' ingest column shows that step alone costing XGBoost more than bonsai's whole fit. The path from 3× slower to ahead is [chapter 11](11-performance-engineering.md).

## In bonsai: the leaf step

`cuda_leafwise` grows the `leafwise` grower's best-first strategy fully on device: histograms live in a per-tree slot pool instead of the level plane's ping-pong buffers, and the gain heap that picks which leaf to expand next stays on the host, the same structure the CPU leafwise grower already uses. The design record is decisions 95, 97 and 98 in [the decision log](../decisions.md), with the corrected measurements in decision 100; the slot-pool layout is documented in [`include/bonsai/cuda/histogram_engine.hpp`](../../include/bonsai/cuda/histogram_engine.hpp); the adopt/refuse reading of LightGBM's learner is a [performance engineering case study](11-performance-engineering.md).

It is what `device="cuda"` runs by default: the Python estimators default to `grower="leafwise"`, and the device mapping is a prefix (`leafwise` becomes `cuda_leafwise`) that preserves the growth strategy rather than substituting one. The exception is a leaf budget that cannot bind (`max_leaves` of 0, or at least 2^`max_depth`): `cuda_leafwise` is then grown by `cuda_depthwise`, which builds the same tree in level rounds instead of one round per leaf ([decision 132](../decisions.md)). The default `max_leaves=31` at `max_depth=6` binds, so default fits are unaffected. Pass `grower="depthwise"` (CPU) or `grower="cuda_depthwise"` (GPU) explicitly to select the fastest plane at matched knobs instead.

Leaf-wise and depthwise agree structurally at a capped depth with a full leaf budget: the tree that comes out is the same tree either way. The two strategies differ only at uncapped depth, or when `max_leaves` is nonzero and below `2^max_depth`, because best-first order then decides which nodes get split at all.

[The perf standings](../results/results/perf.md) do not measure the leaf plane on the GPU: their leafwise arm runs `num_leaves = 2^depth`, a budget that cannot bind, so it grows through the depthwise plane (decision 132). On those cells bonsai's ingest plus train beats LightGBM's CUDA learner at all three published GPU scenarios (gpu-tall at 16M x 128, gpu-wide at 131k x 16384, gpu-extreme at 16M x 1024), by an order of magnitude at the two wider shapes, and peaks at less host memory in all three. LightGBM peaks at less device memory at gpu-tall, while bonsai uses less at gpu-wide and gpu-extreme, and LightGBM scores a slightly higher test r2 on every one of those cells; the panel carries the digits. The leaf plane itself is measured elsewhere. Under a 63-leaf budget that binds, it meets LightGBM's CUDA learner on the [leaf-capped Grinsztajn axis](../results/results/quality-grinsztajn.md) ([decision 130](../decisions.md)). That axis ranks test quality, not speed. Uncapped depth is the cell to read carefully, because best-first there must find a split for every leaf it creates and so pays about twice the rounds; the last measurement of it still put bonsai ahead (decision 100), which carries the reasoning.

The device-resident objective (MSE, LogLoss, Poisson, MAE, Huber, or Quantile; no DART; all-rows or Bernoulli sampling) arms for `cuda_leafwise` the same way it arms for `cuda_depthwise` and `cuda_levelwise`. There is nothing to configure: an eligible fit keeps labels and scores on the device for the whole fit, and `BONSAI_HOST_OBJECTIVE=1` forces the host path. The three constant-hessian objectives renew each leaf from its residuals after the tree is built; on the device that renewal is a radix selection of each leaf's k-th residual over the rows, four counting passes with no sort, so it stays on the device too. The one fit that still renews on the host is `cuda_levelwise` under a monotone constraint, whose leaf table is reprojected after renewal.

## Try it

No CUDA device on your machine? The [RunPod runbook](https://github.com/daniel-m-campos/bonsai/blob/main/docs/ops/runpod-runbook.md) gets you a validated GPU session for well under a dollar.

```bash
# The device suite: SKIPs without a GPU, exercises real kernels with one.
./build-cuda/tests/bonsai_tests "[cuda]"

# Any fit, with the device profile lines:
BONSAI_CUDA_PROFILE=1 BONSAI_GROW_PROFILE=1 \
  bonsai fit --config configs/california_housing.toml \
  --set dispatch.grower_name=cuda_depthwise
```

Read the `cuda-profile` line first: `upload` vs `gpu` tells you where the fit's time went. A fit whose `max_bin` pushes a feature's histogram past the device's shared-memory ceiling does not appear here at all, because it does not run: the fit stops with an error naming the limit, and the remedy is a lower `max_bin` or `device="cpu"`. Then `cuda-upload-decomp` splits every transfer by transaction: this is the line every optimization in chapter 11 was priced against.

```bash
# CPU vs GPU on the same data: expect tolerance-level agreement, not bit-equality:
uv run scripts/compare.py --config configs/california_housing.toml \
    --growers depthwise,cuda_depthwise
```

## Gotchas & war stories

- **The allocator can matter.** The default CUDA mempool release threshold is 0, so every free returns memory to the OS at the next sync, and on GeForce drivers the resulting `cudaMalloc`/`cudaFree` churn synchronizes the whole process. bonsai raises the threshold so freed blocks stay in the pool.
- **Some rented GPUs are just broken.** A whole class of hosts showed ~300µs per device synchronization (healthy: ~4µs), invariant to every software knob, an ASPM/IRQ-level defect. At ~25k syncs per fit, that adds about 7.5 seconds per fit, against 0.1 seconds on a healthy host. The benchmark protocol (decision 48) requires every rented pod to pass a 30-second sync-latency probe and rejects hosts above 50µs. The harness does not run it, so it is a manual step in the RunPod runbook.
- **A demoted split is not the same thing on both planes.** `demote_empty_splits` un-splits a node whose child would be empty. On the host that is bookkeeping; on the device the rows were *already scattered* and stamped, so demoting orphaned their leaf assignments, one mega-leaf per tree at the rightmost spine, quality collapsing only at 16M rows where double-precision sums stopped masking it (PR #29). Control-plane operations must know which plane's invariants they touch.
- **"First use is single-threaded" is an assumption, not a law.** The lazy host materialization of device-binned columns was first guarded by nothing, on the precedent of another lazy member, but its first consumer runs inside a parallel loop, and the race corrupted the heap *intermittently* (one clean validation pass meant nothing). It is now a `call_once` (PR #37).
- **Float accumulation order made a parity test flaky.** While the histogram cells were float, two GPU fits of identical inputs summed in different atomic orders. The leafwise parity test failed one run in five at its 1e-4 bound because of that spread. A tolerance should not hide a spread like that, so decision 124 moved the cells to fixed point, which removed it.
- **Byte-identity is available where it matters.** Device *binning* is bit-identical to host binning (same cuts, same `lower_bound`, integer outputs, no atomics), which is what lets a device-binned dataset train the same model as a host-binned one and lets the test suite say so exactly. A device *fit* is bit-identical to itself run to run (integer cells), so the suite compares model bytes, not predictions within a bound.
