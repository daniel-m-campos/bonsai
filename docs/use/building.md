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

The trees: `build/` (CPU), `build-asan/` (sanitizers), `build-cuda/` (GPU). `make clean` removes all three.

## Toolchain

bonsai is C++23 built with LLVM clang and libc++. The floor is LLVM 20 (`std::print`, `std::mdspan`, and float `std::from_chars` land in libc++ 20; clang 19 relaxed an OpenMP restriction on capturing structured bindings, so clang 18 fails twice over); development and CI run LLVM 21, which is the tested configuration. libc++ rather than libstdc++ because Ubuntu 22.04's system libstdc++ (GCC 11) predates the C++23 pieces above. CMake 3.25+ and Ninja round out the requirements; every C++ dependency (Catch2, CLI11, tomlplusplus, nlohmann/json) is vendored by CMake FetchContent, so there is nothing else to install.

On Ubuntu:

```bash
wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh && sudo bash /tmp/llvm.sh 21 && \
sudo apt-get install -y libc++-21-dev libc++abi-21-dev libomp-21-dev \
  clang-format-21 clang-tidy-21 clang-tools-21 ninja-build
```

On macOS:

```bash
brew install llvm@21 libomp ninja cmake
```

The Makefile finds the toolchain at `/usr/lib/llvm-21/bin` (Linux) or homebrew's `llvm@21` keg, falling back to plain `llvm` (macOS); point it elsewhere with `make LLVM_BIN=/path/to/llvm/bin ...`.

## The make-target map

| target | what it does | tree |
|---|---|---|
| `make build` | configure + compile the CLI, library, and tests | `build/` |
| `make test` | build, fetch the small pinned test datasets, run ctest | `build/` |
| `make rebuild` | clean + build | all |
| `make configure` | CMake configure only (needed once after adding new Python files) | `build/` |
| `make python` | build the `_bonsai` extension for the dev tree | `build/python/` |
| `make python-test` | python + run the binding, bench, encoding, and doc-snippet suites | `build/` |
| `make build-cuda` / `test-cuda` | the same with `-DBONSAI_CUDA=ON` | `build-cuda/` |
| `make python-cuda` | the CUDA-enabled Python extension | `build-cuda/python/` |
| `make test-asan` | ASan + UBSan build and test run (CI-only on macOS, see below) | `build-asan/` |
| `make format` / `lint` / `lint-python` | clang-format, clang-tidy, ruff | |
| `make fit-benchmark` | run the reference-library comparison harness (needs [uv](https://docs.astral.sh/uv/)) | `build/` |
| `make help` | the full list | |

## OpenMP is required, loudly

Configure fails hard if OpenMP is missing. This is deliberate: a silent serial fallback once shipped builds that trained different (valid, but not reproducible) model bits than parallel ones and manufactured a phantom cross-architecture bug ([decision 60](../decisions.md)). On macOS, homebrew's `libomp` is keg-only and CMake is given the hint automatically; on Linux, `libomp-21-dev` comes with the toolchain block above. A deliberately serial build is `-DBONSAI_OPENMP=OFF`. `BONSAI_OPENMP_STATIC=ON` (what wheels use) links the runtime statically so the Python module cannot deadlock against xgboost/lightgbm's bundled OpenMP; on Linux runners no static libomp exists, so the wheel pipeline vendors a private dynamic copy instead ([#134](https://github.com/daniel-m-campos/bonsai/issues/134)). The ASan tree runs in CI only on Linux; homebrew LLVM's ASan deadlocks on current macOS.

## The Python module, two ways

**Installed, wheel-style.** `pip install .` drives scikit-build-core through the same CMake project; pass CMake options with `-C`:

```bash
pip install . -C cmake.define.BONSAI_CUDA=ON -C cmake.define.BONSAI_CUDA_ARCH=sm_120
```

That example is a local CUDA wheel for one GPU architecture. Requires `nanobind` and `numpy` at build time (pip fetches them as build requirements automatically).

**Dev tree.** `make python` builds the extension into `build/python/bonsai` without installing anything; use it with `PYTHONPATH=build/python python ...`. `make python-cuda` is the same into `build-cuda/python`. Iterating on C++ then re-running Python tests is fastest this way; `pip install .` re-resolves the whole build each time.

## CUDA builds

The CUDA backend needs a CUDA 12.x toolkit at build time only (clang cannot target CUDA 13; hosts shipping 13 need a 12.x side-install pointed at with `-DCUDAToolkit_ROOT=...`). `make build-cuda` compiles with `BONSAI_CUDA_ARCH=native`, which probes the local device, so build on the machine with the GPU or pass an explicit architecture. The knobs, all CMake cache variables:

| variable | meaning | default |
|---|---|---|
| `BONSAI_CUDA_ARCH` | `--offload-arch` target(s); a semicolon list builds a fat binary | `native` |
| `BONSAI_CUDA_PTX_ARCH` | embed PTX for this single arch only (forward-JIT floor) | clang's per-arch default |
| `BONSAI_CUDA_STATIC_RUNTIME` | link cudart statically (what release wheels do) | `OFF` |

How the backend works (device-resident training, the one kernel TU compiled by the project's own clang) is [guide chapter 10](../guide/10-gpu-training.md) and [the architecture note](../architecture/11-gpu-resident.md); the wider scaling study lives in the [benchmarks README](../../benchmarks/README.md).

## When the build fails

| symptom | cause | fix |
|---|---|---|
| clang 18: errors in `<charconv>` or OpenMP capture errors | libc++ < 20 lacks float `from_chars`; clang < 19 rejects OpenMP capture of structured bindings | use LLVM 20+, ideally 21 |
| CUDA TU fails with texture-header errors | CUDA 13 toolkit; clang cannot target it | install a CUDA 12.x toolkit, set `CUDAToolkit_ROOT` |
| macOS link errors about string hash symbols | SDK libc++ older than the compiler's headers | handled in CMakeLists (links the keg's libc++); update homebrew llvm if it recurs |
| configure error: OpenMP not found | intentional, see above | install libomp, or `-DBONSAI_OPENMP=OFF` deliberately |
| CUDA configure fails probing `native` | no GPU visible on the build machine | pass `-DBONSAI_CUDA_ARCH=sm_80` (or your target) |

## Extending bonsai

Extending requires a source tree; the recipes below say which files a new component touches.

A stateless grower or sampler is **two edits**: add the type to the registry typelist and a string to the name table, and the dispatch table, `bonsai info` listing, and parametric registry tests all expand automatically.

A new objective is about **six file touches**: the typelist entry, the name, three trait specializations that static assertions demand by name (link function, task kind, default metrics), the implementation itself, and a config field if it carries parameters (plus its TOML section). The parametric tests then cover every `(objective, grower, sampler)` combination without new test code.

Stateful components implement the same concepts as classes constructed from `Config`; a new tree *type* (beyond `DenseTree`/`ObliviousTree`) is the one extension that ripples, since model IO versioning is involved. The machinery that makes all this work is [the dispatch note](../architecture/6-dispatch.md).
