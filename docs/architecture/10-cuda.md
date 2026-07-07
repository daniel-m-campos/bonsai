# 10 — CUDA histogram backend

> **Status:** working (Jetson Orin Nano sm_87; validated on A100 sm_80/x86_64). Registered as the `cuda_depthwise` grower. Perf story and measured ladders: [reviews/2026-07-03-design-review-cuda.md](../reviews/2026-07-03-design-review-cuda.md).
>
> **The grower-side seam was redesigned by [`12-grower-backend.md`](12-grower-backend.md) (decision 41, landed):** the `LevelStep` compile-time strategy replaced the optional-hook dispatch, and `populate_many` + the GPU copy-back path were retired for a CPU fallback. The kernel, precision scheme, and registration story below are current.

## The seam

Histogram construction is an engine policy on the growers ([`include/bonsai/grower.hpp`](../../include/bonsai/grower.hpp)):

```cpp
template <HistogramEngine EngineT = CpuHistogramEngine,
          NodeSplitFinder SplitterT = HistogramNodeSplitFinder>
class DepthwiseGrower;

using CudaDepthwiseGrower = DepthwiseGrower<CudaHistogramEngine>;
```

There is exactly one grow-loop implementation ([`src/grower_impl.hpp`](../../src/grower_impl.hpp)); the CUDA grower is an instantiation, not a fork. A `HistogramEngine` supplies `begin_tree` (per-tree staging: the CUDA engine uploads gradients there) and `populate` (fill one node's per-feature histograms — the CUDA engine delegates this to its CPU member; it exists as the fallback for trees `begin_root` declines). The grow loop reaches the engine through the `LevelStep` data plane ([`12-grower-backend.md`](12-grower-backend.md)): the host plane drives `populate` + the sibling-subtraction trick; a `GPULevelEngine` selects the device plane, where whole levels build, partition, and find splits on the device.

## Level batching

The depthwise and oblivious growers process a tree level through the shared `LevelStep`: control-plane bookkeeping (`plan_level`), level-wide row partitioning (`partition_rows` on the host plane, one node per worker thread — each partition touches only its own parent's rows, so results are scheduling-independent; `partition_level` on the device), then every smaller sibling's histograms in one pass with the larger siblings derived by subtraction (`finish_split` on the host; `advance_level` on the device). Leafwise keeps per-node `split_node` (the single-node data plane in [`src/level_step.hpp`](../../src/level_step.hpp)): its gain-ordered heap is inherently sequential, so there is no level to batch.

Level batching exists because per-node GPU round trips dominated: ~185 launches per tree collapsed to one per level (~9), which is where most of the phase-2 speedup came from; phase 3 then made the level state device-resident (doc 11).

## The kernel

One TU, [`src/cuda/histogram_builder.cu`](../../src/cuda/histogram_builder.cu), compiled as CUDA C++ by the project's own clang (`-x cuda --offload-arch`, cache var `BONSAI_CUDA_ARCH`) — same C++23, same libc++ as every other TU, so it uses `Dataset`/`Histogram`/`SplitInput` directly. No nvcc, no second standard library, no C ABI. The kernels and the `DeviceBuffer` helper live in `src/cuda/detail/{kernels,device_buffer}.cuh`, `#include`d into that one TU's anonymous namespace — a readability split, not a second compilation unit.

Device residency: the binned matrix uploads once per dataset (uint8 when every feature fits 256 bins — the `max_bin = 255` default — uint16 otherwise), gradients once per tree as a packed `float2` array, and per level one concatenated row-index upload. A gather kernel reorders `(grad, hess)` into level order once so the histogram kernel reads them sequentially per feature instead of re-gathering through the row indirection `n_features` times.

The histogram kernel's grid is (feature, node, row-chunk). Each block accumulates its ≤32k-row chunk into a shared-memory histogram (two copies split by warp parity to spread atomic contention), then merges into the (node, feature) slice of the pre-zeroed output with global double atomics. Precision scheme: shared accumulation is **float** (native shared-memory atomics; double atomics emulate via CAS loops), the cross-chunk merge is **double** — rounding stays bounded per chunk, and results match the CPU builder to tolerance, not bit-exactly, because atomics add in arbitrary order.

`k_min_gpu_rows = 512` routes each resident child to the right kernel (below it, the direct-global small-node kernel beats the shared-memory one — the fixed zero+merge cost is bin-proportional, small nodes are row-proportional). The 48 KiB/block shared-memory budget (`max_bin` ≳ 6k) is checked once per tree in `begin_root`, which declines the resident path and lets the `LevelStep` fall back to CPU histogram building — the launch never fails at runtime.

`BONSAI_CUDA_PROFILE=1` prints a per-fit wall-clock breakdown (upload / gpu / unpack / cpu-fallback) when the engine is destroyed.

## Always registered, capability at runtime

There is no `BONSAI_USE_CUDA` macro. `cuda_depthwise` sits in the registry typelist in every build — the builder header is CUDA-free (pimpl) — so the registry is identical across configurations and no config define has to keep TUs in ODR agreement. `BONSAI_CUDA=OFF` builds link a stub ([`src/cuda/histogram_builder_stub.cpp`](../../src/cuda/histogram_builder_stub.cpp)): construction succeeds, training throws with a message naming the fix. Consequences:

- Models trained with `cuda_depthwise` load and **predict on any build** — trees are ordinary `DenseTree`s; only training touches the builder.
- `bonsai::cuda_available()` is the runtime predicate. `bonsai info` marks growers that can't train on the current host ("predict-only here"); the parametric test suites run the cuda combos where a device exists and SKIP them where not (Catch2 exit code 4, mapped via ctest `SKIP_RETURN_CODE`).
- `BONSAI_USE_OPENMP` deliberately remains the one config macro: it guards an `#include <omp.h>` and a `#pragma`, which no language construct can, and is confined to [`parallel.hpp`](../../include/bonsai/parallel.hpp).

## Cross-platform validation

The same source builds and passes the full suite on Jetson (sm_87, aarch64, CUDA 12.6) and A100 (sm_80, x86_64, CUDA 12.6), with GPU-trained models reproducing identical RMSE across both. x86_64 needs `_ALLOW_UNSUPPORTED_LIBCPP` (NVIDIA's own bypass for their libc++-on-x86 header `#error`; set in the CUDA block, no-op elsewhere) — clang cannot target CUDA 13 yet, so hosts shipping it need a 12.x side-install pointed at via `CUDAToolkit_ROOT`.

## Deferred

Device-resident row partitioning and split finding (the full gpu_hist architecture — the measured remainder of the gap to xgboost-GPU: on A100 our kernels take ~2s of a 12s MSD fit while xgboost-GPU completes entirely in 1.8s, so host orchestration is now the whole gap), a `[cuda]` config section for the cutoff constants, a `Dataset` version id to replace the pointer-identity upload cache, CUDA variants of the oblivious/leafwise growers (one typelist line each, when wanted).
