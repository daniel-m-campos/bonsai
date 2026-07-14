# 11: GPU-resident growing

> **Status:** landed (decisions 40–42). Training for `cuda_depthwise` and `cuda_oblivious` is fully device-resident: histograms, rows, and split finding live on the GPU, and only split decisions and child counts cross the bus per level. The grower-side seam is the `LevelStep` strategy of [`12-grower-backend.md`](12-grower-backend.md) (decision 41); this doc owns the device plane: buffers, kernels, precision, and measured results. Historical design narrative: [reviews/2026-07-03-design-review-cuda.md](https://github.com/daniel-m-campos/bonsai/blob/main/docs/reviews/2026-07-03-design-review-cuda.md).

## Why device-resident

After phase 2 (level-batched histogram kernels, [`10-cuda.md`](10-cuda.md)), profiling on an A100 showed the kernels already matched xgboost-GPU's entire fit while the total ran 6.8× slower. The gap was host orchestration: per-level row uploads, histogram copy-back and unpack, host split finding, host partitioning. Phase 3 moved that data plane onto the device in stages, each gated on the MSD ladder (`make bench-gpu`), taking the A100 fit 13.1 → 5.0 s with RMSE unchanged at every step.

## The device plane

`CudaHistogramEngine` (`src/cuda/histogram_engine.cu`, one clang CUDA C++ TU; kernels in `src/cuda/detail/kernels.cuh`, included into that TU's anonymous namespace) keeps per-tree state resident:

- **Binned matrix**: uploaded once per dataset; uint8 when every feature fits 256 bins, uint16 otherwise.
- **Gradients**: raw grad/hess uploaded once per tree, interleaved to `float2` on-device.
- **Level histograms**: ping-pong slot buffers, slot = the node's index in the grower's frontier; larger children derive on-device by subtraction (parent − smaller).
- **Rows**: ping-pong segment buffers (decision 17's per-node lists, device-side); partitioning is route-count → per-segment scan → stable scatter, carrying the ordered gradients along.
- **Leaf assignment**: a persistent per-row array; leaves stamp their segments as they finalize, one ~2 MB download per tree feeds `values`/`leaf_ids` (host `route_unsampled` and leaf renewal unchanged).

What crosses the bus per level: node sums/bounds up (~16 B/node, plus interaction masks only when constrained), split decisions down (~56 B/node), child row counts down (8 B/node). That is the entire per-level traffic.

## The kernels

| kernel | grid | job |
|---|---|---|
| `interleave_kernel` / `gather_gh_kernel` | 1-D | raw grad/hess → `float2`; reorder into segment order |
| `hist_kernel` | (feature, node, row-chunk) | shared-memory float accumulation (two warp-parity sub-histograms), double global merge; skip-zero write-back |
| `hist_small_kernel` | node | sub-512-row nodes accumulate straight into global slots; cost scales with rows, not bins |
| `route_count_kernel` → `seg_scan_kernel` → `scatter_kernel` | (chunk, split) | stable device partition, hand-rolled (no CUB); child counts are the only D2H |
| `find_kernel` → `reduce_kernel` | (feature, node), warp each | warp-tiled prefix scan + per-lane scoring + warp argmax with the CPU scan's exact tie-break (max gain, lowest bin, `default_left` first); per-node winner reduced in feature order |
| `level_find_kernel` → `reduce_kernel` → `level_child_sums_kernel` | (feature), warp each | oblivious: one split for the whole frontier: per-cut child scores summed across all nodes (32-node chunks into per-feature scratch, any frontier width), feasibility required on every node; the fused chain pays one sync per level and returns each node's child sums to seed the next level |

The gain math (`score`, `bounded_leaf_weight`, `split_sums_dev` mirroring `split_sums_at`) is shared with the CPU finders; `constexpr` functions in `split.hpp` are device-callable under clang, so there is one definition of the splitting semantics.

## Precision, determinism, parity

Shared-memory accumulation is float (native atomics; double atomics CAS-loop), cross-chunk and subtraction arithmetic is double; rounding stays bounded per ≤32k-row chunk. The `cuda_*` contract is **tolerance-equal, not bit-equal**: atomic order and warp-tiled prefix association differ from the serial CPU scan, so equal-within-rounding gains may pick different splits (structurally different, equally good trees). Parity tests assert prediction/RMSE tolerance, never tree equality; CPU-only builds remain bit-identical and untouched. Small nodes (<512 rows) use the direct-global kernel because the shared-memory kernel's fixed per-(node,feature) cost is bin-proportional; there is no CPU fallback on the resident path: `begin_root` declines wholesale (oversized `max_bin`) and the host `LevelStep` plane takes over.

## Measured results

A100 (MSD 464 k × 90, 200 iters, depth 8; RMSE 8.9911 at every stage): 13.1 s phase-2 baseline → 7.5 (device find + resident histograms) → 5.4 (device partition + resident rows) → 5.0 s (device gradient interleave). The remaining fit time is dominated by CSV parse + binning outside the grow loop.

RTX 5090, cross-library, every `fit` timing the full pipeline: CSV read + binning + train (`scripts/bench_gpu.py`; catboost/lightgbm warm, first sm_120 run pays PTX JIT):

| strategy | bonsai | reference | RMSE (bonsai / ref) |
|---|--:|--:|--|
| depthwise | `cuda_depthwise` **3.0 s** | xgboost-GPU 4.8–7.1 s | 8.9911 / 8.9924 |
| oblivious | `cuda_oblivious` **3.9–4.1 s** | CatBoost-GPU 7.3–8.2 s | 9.1745 / 9.1403 |
| leaf-wise | `leafwise` (CPU) **11.1–11.5 s** | LightGBM-GPU (CUDA build) 12.0–12.9 s | 9.0871 / 8.9457 |

bonsai wins each structure-matched row. The leafwise row is deliberately honest: best-first growth expands one node at a time, which the level-batched resident plane cannot serve, so there is no `cuda_leafwise` (decision 42: a working registration was built, benchmarked, and withdrawn for computing CPU histograms under a GPU name); CPU leafwise still beats LightGBM's CUDA backend on this card. LightGBM's PyPI wheel is CPU-only: its GPU number is a CUDA source build (`pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON`); CatBoost caps GPU `border_count` at 254 vs 255 bins elsewhere.

Blackwell notes (sm_120): clang-21 `--offload-arch=native` compiles it clean; consumer FP64 (1/64 rate) made the original one-lane find scan dominate, fixed by the warp-parallel scan; the FP64 global-atomic merge remains the visible consumer-silicon cost. An isolated nvcc-vs-clang codegen comparison (`experiments/nvcc_vs_clang/`) found clang marginally faster; the single-TU clang design leaves no kernel performance on the table.

## The benchmark loop

Development iterates against the MSD ladder on a rented GPU node; workflow, node setup, and JSONL regression tracking live in [`benchmarks/README.md`](../../benchmarks/README.md). A limited-context session needs only that section and this document.

## What's not here

Device leafwise (needs a non-swapping advance), device-side objective gradients, multi-GPU, CUDA graphs, a `[cuda]` config section for the cutoff constants, faster CSV/binning (now the largest block of end-to-end fit): deliberate deferrals. The engine policy's origin story is [`10-cuda.md`](10-cuda.md); the grower seam is [`12-grower-backend.md`](12-grower-backend.md); ratifying entries are decisions 40, 41, and 42.
