# Building from source

You only need this page if a prebuilt wheel does not cover you ([Install](install.md)), you want the `bonsai` command-line binary, or you are developing bonsai itself.

## What a build produces

Three artifacts come out of this repo, and they land in separate build trees so the plain CPU tree stays pristine while sanitizer and CUDA variants live beside it:

| artifact | command | lands at | who needs it |
|---|---|---|---|
| `bonsai` CLI binary | `make build` | `build/src/bonsai` | anyone using the command line; the wheel does not include it |
| C++ test binary | `make test` | `build/tests/` (run via ctest) | contributors |
| `_bonsai` Python module (dev tree) | `make python` | `build/python/bonsai/` | developing against the Python API |
| `_bonsai` Python module (installed) | `pip install .` | your environment's site-packages | using a source-built package like a wheel |
| CUDA variants of all of the above | `make build-cuda` / `python-cuda` | `build-cuda/...` | GPU training from a source build |

The trees: `build/` (CPU), `build-asan/` (ASan + UBSan), `build-tsan/` (ThreadSanitizer), `build-cuda/` (GPU). `make clean` removes `build/`, `build-asan/`, and `build-cuda/`, but not `build-tsan/`, so delete that one by hand.

## Toolchain

bonsai is C++23 built with LLVM clang and libc++. You need LLVM 20 or newer. bonsai uses `std::print`, `std::mdspan` and float `std::from_chars`, and libc++ 20 is the first release that has all three. clang 18 also rejects OpenMP capture of structured bindings, which clang 19 accepts. CI and development use LLVM 21, the tested configuration. libc++ rather than libstdc++ because Ubuntu 22.04's system libstdc++ (GCC 11) predates the C++23 pieces above. You also need CMake 3.25 or newer and Ninja. CMake downloads the four third-party C++ dependencies (Catch2, CLI11, tomlplusplus, nlohmann/json) when you configure the build, so you do not install any of them yourself.

On Ubuntu, install CMake from PyPI, because the apt cmake on Ubuntu 22.04 is 3.22, below the 3.25 minimum:

```bash
wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh && sudo bash /tmp/llvm.sh 21 && \
sudo apt-get install -y libc++-21-dev libc++abi-21-dev libomp-21-dev \
  clang-format-21 clang-tidy-21 clang-tools-21 ninja-build && \
pip install 'cmake>=3.25'
```

On macOS:

```bash
brew install llvm@21 libomp ninja cmake
```

The Makefile finds the toolchain at `/usr/lib/llvm-21/bin` (Linux) or homebrew's `llvm@21` keg, falling back to plain `llvm` (macOS); point it elsewhere with `make LLVM_BIN=/path/to/llvm/bin ...`.

## The make-target map

Run `make help` to list the targets with a one-line description of each, taken from the Makefile.

## OpenMP is required, loudly

Configure stops with an error if OpenMP is missing. A serial fallback once shipped builds that trained different models from the parallel builds: both models were valid, but a serial build could not reproduce the bits of a parallel build. This looked like a bug that appeared only on some CPU architectures ([decision 60](../decisions.md)). On macOS, homebrew's `libomp` is keg-only and CMake is given the hint automatically; on Linux, `libomp-21-dev` comes with the toolchain block above. A deliberately serial build is `-DBONSAI_OPENMP=OFF`. `BONSAI_OPENMP_STATIC=ON` (what wheels use) links the runtime statically so the Python module cannot deadlock against the OpenMP runtime that XGBoost and LightGBM bundle; on Linux runners no static libomp exists, so the wheel pipeline vendors a private dynamic copy instead ([#134](https://github.com/daniel-m-campos/bonsai/issues/134)). The ASan tree runs in CI only on Linux; homebrew LLVM's ASan deadlocks on current macOS.

## The Python module, two ways

**Installed, wheel-style.** `pip install .` drives scikit-build-core through the same CMake project; pass CMake options with `-C`:

```bash
pip install . -C cmake.define.BONSAI_CUDA=ON -C cmake.define.BONSAI_CUDA_ARCH=sm_120
```

That example is a local CUDA wheel for one GPU architecture. Requires `scikit-build-core>=0.10`, `nanobind>=2,<3`, `numpy` and `typing_extensions` at build time (pip fetches them as build requirements automatically). With `--no-build-isolation`, install them yourself and keep nanobind below 3, which fails on Python 3.9.

**Dev tree.** `make python` builds the extension into `build/python/bonsai` without installing anything; use it with `PYTHONPATH=build/python python ...`. `make python-cuda` is the same into `build-cuda/python`. The build uses `.venv/bin/python` by default; override with `make python PYTHON=/path/to/python`. That interpreter needs `nanobind<3` and `numpy` installed, plus `typing_extensions` below Python 3.11 (the stub step needs it), for example `uv venv .venv && uv pip install --python .venv/bin/python 'nanobind<3' numpy typing_extensions`. Use the dev tree when you edit C++ and re-run Python tests: `make python` rebuilds only the changed files into the existing `build/` directory, while `pip install .` configures and compiles the whole project from scratch in a temporary directory each time.

## CUDA builds

The CUDA backend needs a CUDA 12.x toolkit to build (clang cannot target CUDA 13; hosts shipping 13 need a 12.x side-install pointed at with `-DCUDAToolkit_ROOT=...`). A default build links cudart dynamically, so the CUDA 12 runtime library must also be present at run time; `-DBONSAI_CUDA_STATIC_RUNTIME=ON` (what release wheels use) removes that dependency. `make build-cuda` compiles with `BONSAI_CUDA_ARCH=native`, which probes the local device, so build on the machine with the GPU or configure by hand with an explicit architecture (see the failure table below). The knobs, all CMake cache variables:

| variable | meaning | default |
|---|---|---|
| `BONSAI_CUDA_ARCH` | `--offload-arch` target(s); a semicolon list builds one binary holding native code for each listed architecture | `native` |
| `BONSAI_CUDA_PTX_ARCH` | embed PTX (NVIDIA's portable intermediate code) for this one architecture only; the driver compiles it when the library loads, so GPUs of that architecture or newer can run the build even without native code for them | clang's per-arch default |
| `BONSAI_CUDA_STATIC_RUNTIME` | link cudart statically (what release wheels do) | `OFF` |

How the backend works (device-resident training, kernels compiled as CUDA C++ by the project's own clang) is [guide chapter 10](../guide/10-gpu-training.md); running the scaling suite against a CUDA build is [Reproducing the results](../results/benchmarks.md).

## When the build fails

| symptom | cause | fix |
|---|---|---|
| clang 18: errors in `<charconv>` or OpenMP capture errors | libc++ < 20 lacks float `from_chars`; clang < 19 rejects OpenMP capture of structured bindings | use LLVM 20+, ideally 21 |
| CUDA kernel translation unit (the `.cu` files under `src/cuda/`) fails with texture-header errors | CUDA 13 toolkit; clang cannot target it | install a CUDA 12.x toolkit, set `CUDAToolkit_ROOT` |
| macOS link errors about string hash symbols | SDK libc++ older than the compiler's headers | handled in CMakeLists (links the keg's libc++); update homebrew llvm if it recurs |
| configure error: OpenMP not found | intentional, see above | install libomp, or `-DBONSAI_OPENMP=OFF` deliberately |
| CUDA configure fails probing `native` | no GPU visible on the build machine | run the configure step by hand with the architecture (`sm_80` or your target), then build: `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$LLVM_BIN/clang++ -DBONSAI_CUDA=ON -DBONSAI_CUDA_ARCH=sm_80 -G Ninja -S . -B build-cuda && make build-cuda`, where `$LLVM_BIN` is your LLVM bin directory such as `/usr/lib/llvm-21/bin` (the Makefile skips its own configure once `build-cuda/build.ninja` exists) |

## Extending bonsai

Extending requires a source tree; the recipes below say which files a new component touches.

A grower decides how each tree is built (the `depthwise`, `leafwise` and `levelwise` strategies), and a sampler decides which rows each tree sees. Adding a stateless one takes **two edits**: add its type to the registry typelist (the compile-time list of components) in `include/bonsai/registry/typelists.hpp`, and add a `name_of<T>` row to `include/bonsai/registry/names.hpp`. The dispatch table, the `bonsai info` listing and the parametric registry tests then expand automatically.

A new objective is about **six file touches**: the typelist entry (`Objectives` in `include/bonsai/registry/typelists.hpp`), the `name_of<T>` row in `include/bonsai/registry/names.hpp`, three trait specializations that static assertions demand by name (link function, task kind, default metrics), the implementation itself, and a config field if it carries parameters (plus its TOML section). The parametric tests then cover every `(objective, grower, sampler)` combination without new test code.

Stateful components implement the same concepts as classes constructed from `Config`. A new tree type (beyond `DenseTree` and `ObliviousTree`) costs more than the other extensions: it changes what a model file contains, so the model format version (`k_format_version` in `src/io/model.cpp`) must be bumped, and the loader rejects files with any other version. The typelist registry that dispatches components is described in [decision 26](../decisions.md).
