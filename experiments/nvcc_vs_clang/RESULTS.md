# Results: nvcc vs clang kernel codegen (RTX 5090, sm_120)

Run 2026-07-06 on a RunPod RTX 5090 (Blackwell sm_120, driver 570.124, CUDA 12.8.93, Ubuntu 24.04), clang 21.1.8 vs nvcc 12.8. Same `bench.cu`, `-O3 -std=c++20`.

## Compilation

**nvcc requires `--expt-relaxed-constexpr`; clang does not.** Out of the box nvcc rejects calling the `constexpr` gain math (`score`, `bounded_leaf_weight`) from `__global__` code — it treats `constexpr` as host-only. clang treats `constexpr` as implicitly `__host__ __device__`, so the real kernels compile unmodified. This is a concrete instance of the friction behind the project's single-TU clang design (decision in `docs/architecture/10-cuda.md`): the engine's gain math is shared host/device precisely because clang allows it.

## Timing (mean of 6 runs, ±0.0002 ms)

| kernel | clang | nvcc | delta |
|---|--:|--:|--:|
| `hist_kernel` (double-atomic merge) | 0.0941 ms | 0.0975 ms | clang **3.6% faster** |
| `find_kernel` (one-lane double scan) | 0.5094 ms | 0.5096 ms | identical (noise) |

## Codegen detail

| | clang | nvcc |
|---|--:|--:|
| `hist_kernel` registers | 28 | 28 |
| `find_kernel` registers | 82 | 80 |
| `hist_kernel` SASS instrs | 144 | 144 |
| `hist_kernel` FADD / ATOM / LDS+STS | 3 / 2 / 5 | 3 / 2 / 5 |
| `hist_kernel` branches | 14 | 15 |
| module SASS lines | 4136 | 4380 |

The FP64 double-atomic merge lowers to the **same** atomic instructions in both (ATOM=2, RED=1) — the compiler doesn't change the atomic strategy. `hist_kernel` is instruction-for-instruction near-identical (144 each); nvcc has one extra branch and slightly different scheduling around the hot shared-memory atomic loop, which accounts for the small, reproducible ~3.6% gap.

## Conclusion

nvcc offers **no performance advantage** on these kernels — it matches clang on the scan and is marginally slower on the histogram merge, at the cost of requiring `--expt-relaxed-constexpr` and (for the full engine, not tested here) losing the C++23/libc++/single-TU/no-C-ABI design. The experiment validates the existing clang `-x cuda` choice: staying on clang leaves no kernel performance on the table.

Caveat: isolated device-codegen only. End-to-end fit time is dominated by host orchestration and CSV/binning, which no compiler swap here would touch.
