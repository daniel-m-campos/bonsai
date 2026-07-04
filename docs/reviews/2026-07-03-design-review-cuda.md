# Design Review: CUDA histogram backend (phase 1)

> **Date**: 2026-07-03 **Scope**: the uncommitted `cuda_depthwise` diff on `fabel` — build seam (`BONSAI_CUDA`), C ABI (`include/bonsai/cuda/api.h`), kernel TU (`src/cuda/histograms.cu`), `CudaHistogramBuilder`, the `HistogramBuilder` policy refactor of the growers (`src/grower_impl.hpp`), registry additions, and `tests/unit/test_cuda_grower.cpp`. **Method**: SOLID → DOD, house style per the 2026-05-19 review. **Audience**: pre-commit review; findings ordered by severity.

## Scoping

- **Extensibility pressure**: **medium**. One new axis (histogram construction) turned into a policy; future backends (fixed-point GPU, SYCL, thread-pool CPU) slot in without touching the grow loops.
- **Performance pressure**: **high but not yet delivered** — phase 1 is correctness-first. Beats 1-thread CPU on wide data (16.5 s vs 23.6 s, 64 feat × 132 k rows); loses to 6-thread OpenMP (9.4 s). Bottlenecks known: shared-memory double-atomic contention, per-node sync round trips.
- **Correctness bar**: met — double accumulation on both paths; bit-identical California Housing RMSE (0.47382435); 392/392 tests.

## Per-Entity Analysis

### Build seam (CMakeLists.txt, src/CMakeLists.txt)

| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ✅ | `bonsai_gpu` INTERFACE target mirrors `bonsai_parallel`: empty when OFF, kernels + cudart + define when ON. OFF-build is byte-for-byte the pre-CUDA build. |
| D | ✅ | Kernel TU isolated in its own static lib; nvcc never sees C++23 flags (`$<COMPILE_LANGUAGE:CXX>` scoping on `-stdlib=libc++`). |
| O | ⚠️ | `add_compile_definitions(BONSAI_USE_CUDA)` is global — correct (typelists must agree across every TU) but means a CUDA and non-CUDA build of the same tree share no object files. Acceptable; documented in-line. |
| — | ⚠️ | `bonsai_cuda_kernels` does not get `bonsai_warnings` (clang flags don't apply to nvcc), so the .cu file compiles with no warning flags at all. **Nit**: add `-Xcompiler=-Wall,-Wextra` or equivalent. |

### C ABI + kernel (api.h, histograms.cu)

| Principle | Rating | Analysis |
|-----------|--------|----------|
| S/I | ✅ | Four functions: create / destroy / set_gradients / build_histograms. POD-only signatures; static error strings. Correct answer to the clang-libc++ / nvcc-libstdc++ split. |
| D2 | ✅ | Feature-major device layout matches `Dataset::feature_bins`; upload once per dataset, gradients once per tree; per-node traffic is rows + histograms only. |
| D3 | ❌ | **Shared-memory budget unchecked.** Shared bytes = `2 * max_bins * 8`; the 48 KB/block default caps `max_bins` at ~3 070, but `bin_mapper.max_bin` has no upper bound (`assert(> 2)` only, src/bin_mapper.cpp:77). A config with `max_bin = 4000` launches a kernel that fails at runtime with an opaque CUDA error mid-fit. **Should-fix**: fall back to the CPU builder (or fail loud at create) when the histogram exceeds the shared budget. |
| D4 | ✅ | Chunked grid (feature × row-chunk, global-atomic merge) was added in response to a measured occupancy problem, not speculatively. `ensure_capacity` geometric growth is minimal. |
| — | ⚠️ | `api.h` lives under `include/bonsai/` next to genuinely public headers, but it is an internal ABI consumed only by two src/ TUs. **Nit**: `src/cuda/api.h` would state the intent. |

### CudaHistogramBuilder (histogram_builder.hpp/.cpp)

| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ✅ | Owns exactly the host↔device choreography: identity-cached dataset upload, per-tree gradient upload, staging buffers, CPU fallback under `k_min_gpu_rows`. |
| L | ✅ | Satisfies `HistogramBuilder` with the same placeholder contract as the CPU builder (zero-binned unselected features); verified by the row/feature-subset parity test. |
| D | ⚠️ | Dataset identity is a heuristic (ds address + first-column buffer address + shape). A freed-then-reallocated Dataset aliasing all four is vanishingly unlikely but not impossible. Fine for phase 1; the clean fix is a version/id counter on `Dataset` — note for the categorical work, which touches `Dataset` anyway. |
| D3 | ⚠️ | `k_min_gpu_rows = 16384` is a constant, not config. Right call for phase 1 (a config knob would have to round-trip through saved models); revisit when a `[cuda]` config section earns its keep. |
| D4 | ✅ | pimpl keeps CUDA types out of the public header; move-only semantics are correct for a device-owning type (Booster never copies growers). |

### Grower policy refactor (grower.hpp, grower_impl.hpp, grower.cpp)

| Principle | Rating | Analysis |
|-----------|--------|----------|
| S/O | ✅ | The load-bearing decision of this diff: histogram construction became a policy (`HistogramBuilder` concept, defaulted second template param), so there is exactly one grow-loop implementation and the CUDA grower is an instantiation, not a fork. Oblivious/leafwise get CUDA variants for free later (one typelist line each). |
| L | ✅ | Default args preserve every existing spelling (`DepthwiseGrower<>`, explicit one-arg). CPU-build object code and all 343 pre-existing tests unchanged. |
| — | ⚠️ | `src/grower_impl.hpp` is a 600-line implementation header born from moving grower.cpp wholesale. The move is faithful (diff shows relocation + builder threading only), but the file now serves two masters: shared helpers (`grower_detail`) and the grower method templates. Acceptable; if a third instantiation TU appears, consider splitting helpers from method templates. |
| — | ✅ | `builder_.begin_tree()` placement (once per `grow`, before `make_root`) is correct for all three growers, including the multiclass booster's per-class grows (each class's gradients re-upload — necessary, since gradients differ per class). |

### Registry + model portability (typelists.hpp, names.hpp)

| Principle | Rating | Analysis |
|-----------|--------|----------|
| O | ✅ | Two-edit extension as advertised; dispatch, `bonsai info`, parametric tests, and model I/O expanded automatically (45→60 combos, 343→392 tests). |
| — | ⚠️ | **Model portability is one-way**: a model saved with `dispatch.grower_name = "cuda_depthwise"` will not load on a non-CUDA build (unknown impl name). The error is loud, which is the house convention (`k_format_version` philosophy), but a user may reasonably expect a *tree* model to predict anywhere — the trees are plain `DenseTree`s; only the name differs. **Consider** (phase 2): a load-time alias mapping `cuda_depthwise → depthwise` when the CUDA grower is unavailable, since predict has no builder dependency. |

### Tests (test_cuda_grower.cpp)

| Principle | Rating | Analysis |
|-----------|--------|----------|
| — | ✅ | Right shape: cell-level parity (all rows; row+feature subsets with placeholder check), full-grow prediction parity, cache-reuse exercise. Seeded scenarios, NaN column covers the missing bin. |
| — | ❌ | **Flakiness hazard**: "consecutive trees" asserts `first.values[r] == second.values[r]` — exact float equality across two GPU grows. Global/shared atomic ordering is nondeterministic, so histogram sums can differ in the last ulp between runs, which can flip a near-tie split and break exact equality. It passes today; it is not guaranteed to. **Should-fix**: assert with the same 1e-5 tolerance as the CPU/GPU comparison, or reframe the test to assert only shape/finiteness plus the dataset-switch behavior it exists to cover. |
| — | ❌ | **Coverage regression from the cutoff**: `random_scenario` has 512 rows — below `k_min_gpu_rows = 16384` — so every `populate` call in this file, including the direct cell-level parity tests, now silently delegates to the CPU builder. All three parity tests compare CPU with CPU. They validated the real kernel when first written (pre-cutoff) and kept passing after, which is exactly how this class of gap hides. **Should-fix**: grow the scenario past the cutoff (or make the threshold injectable) so the parity tests exercise the GPU again. |

## Verdict

The architecture is right: policy-parameterized growers (one grow loop), a C ABI quarantining the second toolchain, `DenseTree` reuse keeping serialization untouched, and honest hybrid CPU/GPU dispatch. The three findings that should land before commit are all small:

1. **Shared-memory overflow** (`max_bin` ≳ 3 070 → runtime kernel failure): fall back to CPU populate when the histogram exceeds the budget.
2. **Test flakiness**: exact-equality assertion over nondeterministic atomics.
3. **Test coverage gap**: after the 16 384-row cutoff, the parity tests no longer exercise the GPU path at all — this is the most important one, since it means the kernel is currently validated only indirectly (by the 392-test dispatch suite and the bit-identical CLI RMSE, both of which run large enough nodes).

Deferred with eyes open: kernel throughput (atomic contention, level batching, device partitioning — recorded as phase-2 next steps), `cuda_depthwise → depthwise` load alias, `Dataset` version id, `api.h` placement, nvcc warning flags.

## Post-review updates (same day)

All three should-fixes landed, alongside a kernel precision/perf change prompted by the MSD benchmark:

1. **Shared-memory overflow** — `CudaHistogramBuilder::populate` now falls back to the CPU builder when `2 * max_bins * sizeof(float)` exceeds the 48 KiB budget (`k_max_shared_bytes`), instead of failing the launch.
2. **Flaky exact-equality test** — the consecutive-grows assertion uses a 1e-4 tolerance with a comment on atomic-order nondeterminism.
3. **Coverage gap** — the test scenario grew to 4 096 rows, above the (now 512-row) cutoff, so the parity tests exercise the real kernel; tolerances widened (WithinRel 1e-4 ∥ WithinAbs 1e-5) for float accumulation.
4. **Kernel: float shared accumulation, double merge** (supersedes the review's "double on both paths" description). Shared-memory float atomics are native where double atomics CAS-loop; each ≤ 32 k-row chunk sums in float and chunks merge in double. Measured on Year Prediction MSD (463 715 × 90, 200 iters, depth 8): 254.0 s CPU 1-thread → 71.7 s CPU 6-thread → **54.1 s GPU** (cutoff 512), with test RMSE 8.9948 vs 8.9911 CPU (+0.04 %). California Housing RMSE unchanged at 4 decimals (0.47382432 vs 0.47382435).
5. **C ABI replaced by clang CUDA C++** (resolves the `api.h` placement nit by deleting the file). The kernel TU is now `src/cuda/histogram_builder.cu`, compiled by the project's clang with `-x cuda --offload-arch` — same C++23/libc++ as every other TU, so kernels and builder share one file, use `Dataset`/`Histogram`/`SplitInput` directly, RAII-wrap device buffers, and inherit `bonsai_warnings` (`-Werror`). nvcc and the second (libstdc++) toolchain are gone; `BONSAI_CUDA` now requires the CXX compiler to be clang, which the project mandates anyway. clang's codegen also benches slightly faster: **51.5 s** on the MSD fit. Trade-off accepted: kernel-TU compilation is now tied to clang's CUDA support rather than nvcc's.

## Phase 2 (same day): level batching, profile-guided kernel work, xgboost-GPU reference

Profiling (`BONSAI_CUDA_PROFILE=1`, new env-gated chrono counters in the builder) attributed phase 1's time to per-node launch overhead: ~185 populate round trips per tree, with sub-4k-row nodes averaging ~0.5 ms of almost pure launch+sync cost. Changes, in the order they earned their keep:

1. **Level batching** — the growers now hand a whole level to the builder at once. `grower_detail::populate_nodes` calls the builder's `populate_many` when present (detected via `if constexpr` + `requires`, so the CPU builder needs no batched method); `update_nodes` (depthwise) and the oblivious level loop were restructured into partition → batch-populate → subtract passes, with `split_node` retained for the leafwise grower's inherently sequential heap order. Launches per fit dropped from ~21,600 to 1,800 (one per tree level).
2. **Across-node parallelism for the CPU-fallback set** inside `populate_many` (inner feature-loops degrade to a team of one inside the active OpenMP region — bit-identical per node). Contended fallback time fell ~3.5×.
3. **Parallel row partitioning** — pass 1b of `update_nodes` partitions each split node's rows on its own worker (each partition touches only its own rows; child row order is scheduling-independent). This also sped up the *CPU* build: 6-thread MSD fit 71.7 s → 62.4 s.
4. **Kernel micro-opts** — uint8 device bins when every feature fits 256 bins, float2-packed gradients, a level-order gradient pre-gather kernel, and warp-parity sub-histograms. Collectively worth ~2.5 s; the kernel is DRAM-latency-bound on the bins gather, so these hit diminishing returns quickly.

**Reference bar** (user-directed): xgboost built from source for sm_87 (v3.0.2 — 3.1's CUDA floor of 12.9 exceeds this JetPack's 12.6; one two-line `WQSummary::Entry operator==` patch for CUDA 12.6's older CCCL) runs `device="cuda"` natively on this GPU.

**Final MSD ladder** (464,715 × 90, 200 iters, depth 8, this Jetson Orin Nano):

| configuration | fit (s) | test RMSE |
|---|--:|--:|
| bonsai CPU 1 thread | 254.0 | 8.9911 |
| bonsai CPU 6 threads (phase 2) | 62.4 | 8.9911 |
| bonsai cuda_depthwise phase 1 | 51.5 | 8.9948 |
| **bonsai cuda_depthwise phase 2** | **38.2** | **8.9911** |
| xgboost CPU 6 threads | 34.6 | 8.9849 |
| xgboost GPU (sm_87 build) | 26.2 | 8.9924 |

Smaller datasets (132k rows): 64-feature GPU now edges 6-thread CPU (7.4 s vs 7.7 s); 8-feature still favors CPU (2.7 s vs 2.2 s) — expected, and the hybrid picks reasonable paths automatically.

**Remaining gap analysis.** Of the 38.2 s: ~20 s kernels (DRAM-latency-bound bins gather), ~3 s rows upload, ~2 s host unpack, ~2 s CPU-fallback populate, ~11 s CPU-side growing (split finding, subtraction, bookkeeping, booster). Closing on xgboost-GPU's 26 s means going device-resident like it does: rows partitioned on device (no per-level upload), split finding on device (no histogram copy-back). That is the full gpu_hist architecture — deferred as its own phase with this measured justification.

## Post-phase-2 update: BONSAI_USE_CUDA eliminated

The registry `#ifdef` (and its global `add_compile_definitions` crutch, flagged above as an ODR-coupling smell) is gone. `cuda_depthwise` is registered unconditionally — the builder header is CUDA-free, so CPU-only builds compile the grower against a stub TU (`src/cuda/histogram_builder_stub.cpp`) whose `begin_tree` throws with a message naming the fix. Availability became a runtime predicate, `bonsai::cuda_available()`. Consequences, all verified:

- The registry typelists are identical in every configuration — the ODR hazard is structurally impossible now, and CMake lost the global define.
- **The model-portability finding above is resolved**: a cuda_depthwise model trained by the CUDA binary evaluates on the CPU-only binary (CH RMSE 0.4738 cross-binary); only re-training throws.
- `bonsai info` annotates growers that can't train on the current host ("predict-only here"); the registered combo set is 72 everywhere.
- The parametric test suites run the cuda combos where a device exists and SKIP them (Catch2 `SKIP`, exit code 4 mapped via ctest `SKIP_RETURN_CODE`) where it doesn't: 392/392 on both builds, 48 skipped on CPU-only.
- `BONSAI_USE_OPENMP` stays, deliberately: it guards an `#include <omp.h>` and a `#pragma`, which no language construct can, and it is confined to the one 70-line `parallel.hpp`. That is the documented exception; config macros elsewhere should follow the stub/runtime-capability pattern.
