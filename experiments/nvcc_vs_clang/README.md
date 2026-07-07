# nvcc vs clang: isolated kernel codegen comparison

The bonsai CUDA backend is compiled as one clang `-x cuda` translation unit (decision in `docs/architecture/10-cuda.md`) — deliberately not nvcc. This experiment asks a narrow, fair question: **for the two FP64-bound kernels, does nvcc's device codegen differ in speed from clang's?** It does *not* attempt a full nvcc build of the engine — that C++23 host code cannot pass through nvcc, which is why the backend chose clang.

## What it measures

`bench.cu` `#include`s the real kernels from `src/cuda/detail/{device_buffer,kernels}.cuh` (a minimal `shim/bonsai/split.hpp` supplies the constexpr gain math, so no TreeConfig/SplitInput/C++23 comes along). It times two kernels at representative MSD-level dimensions (464k rows, 90 features, 256 bins, 32 nodes):

- **`hist_kernel`** — the double-atomic cross-chunk merge (the Blackwell FP64 penalty: 0.64→1.09s A100→5090).
- **`find_kernel`** — the one-lane double prefix scan.

The same source compiles under clang and nvcc, so any difference is codegen.

## Running (on a CUDA node — the macOS host can't)

```
bash run.sh sm_120      # 5090 (Blackwell); omit or use `native` to autodetect
```

Reports, for each compiler: per-kernel ms/launch, ptxas register usage, and a rough SASS FP64/atomic instruction count. Raw output lands in `out/`.

## Expectations to check against

Phase-1 memory had a confounded data point (clang 51.5s vs old nvcc+C-API 54.1s — different ABI too). A clean read has never been taken. Hypotheses:

- If clang and nvcc emit near-identical SASS for these kernels, the timings converge and the compiler choice is perf-neutral (clang wins on the architecture/single-TU grounds already documented).
- If they differ, the likely axis is the FP64 double-atomic lowering in `hist_kernel` (CAS-loop vs native `atom.add.f64` depending on arch) and register allocation on the long `find_kernel` scan.

## Caveats

- Kernel-isolated: this is device-codegen only. End-to-end fit time is dominated by host orchestration (~2.5s grow loop) and CSV/binning, which this doesn't touch.
- `-std=c++20` (the kernels use designated initializers); nvcc 12.8 supports it. The full engine needs C++23, which nvcc does not.
- Synthetic inputs sized to MSD, not a real dataset — fine for codegen timing, not for correctness.
