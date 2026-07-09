<p align="center">
  <img src="docs/assets/bonsai-logo.png" alt="bonsai" width="640">
</p>

<p align="center">
  <b>A histogram gradient-boosted tree library and CLI in modern C++23.</b>
</p>

<p align="center">
  <a href="https://github.com/daniel-m-campos/bonsai/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/daniel-m-campos/bonsai/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <img alt="C++23" src="https://img.shields.io/badge/C%2B%2B-23-00599C?logo=cplusplus&logoColor=white">
  <img alt="CMake" src="https://img.shields.io/badge/CMake-%E2%89%A5%203.28-064F8C?logo=cmake&logoColor=white">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <img alt="tests" src="https://img.shields.io/badge/tests-443%20passing-success">
</p>

<p align="center">
  <a href="docs/proposal.md">Proposal</a> &nbsp;·&nbsp;
  <a href="docs/architecture/">Architecture</a> &nbsp;·&nbsp;
  <a href="docs/decisions.md">Decisions</a> &nbsp;·&nbsp;
  <a href="docs/report.md">Retrospective</a>
</p>

---

## What is bonsai?

bonsai is a from-scratch, histogram-based gradient boosted trees (GBT) library and command-line tool written in C++23. It pairs a small, concept-checked component API (objectives, growers, split finders, samplers) with compile-time dispatch in the training hot path, and ships a benchmark harness that pits it against xgboost, lightgbm, and catboost on real data. The aim is a readable, thoroughly documented GBT — a reference-grade implementation that still lands within RMSE tolerance of the production libraries.

## Highlights

- **Compile-time dispatch where it counts.** Components are C++ concepts; the runtime TOML config is resolved to a monomorphized `Booster<Objective, Grower, Splitter, Sampler>` exactly once at construction. Everything inside the training loop is statically dispatched — no virtual calls in the hot path.
- **Concept-checked components.** Contract violations are caught at compile time, not runtime. Adding an objective, grower, split finder, or sampler is two edits and the dispatch table, CLI listing, and parametric tests expand automatically.
- **A guide, not just docs.** [docs/guide/](docs/guide/) explains how gradient boosting works chapter by chapter — concept, math, then the ~50 real lines that implement it here, then an experiment. Written against this codebase because it's small enough to actually read.
- **Reference-library parity.** RMSE within tolerance of xgboost / lightgbm on the California Housing and Year Prediction MSD regression benchmarks, driven by a Python sidecar that runs all three reference libraries on the same config.
- **Three growers.** `depthwise` (level-wise, XGBoost-style), `leafwise` (best-first with a `max_leaves` budget, LightGBM-style), and `oblivious` (symmetric, CatBoost-style) — selectable per run from config.
- **Deterministic parallelism.** OpenMP across features and rows with no cross-thread reductions: models and predictions are bit-identical to a serial run at any thread count (`[parallel] n_threads`, 0 = all cores).
- **CLI-first, config-driven.** CatBoost-style subcommands, a strict TOML config, and inline `--set key=value` overrides — no Python bindings required.
- **One-command build, no system dependencies.** CMake + FetchContent vendors every dependency; a clean checkout builds with a single command.

## Quick start

Train, predict, and compare against xgboost/lightgbm/catboost on the California Housing dataset in one command:

```
make fit-benchmark
```

Writes `benchmarks/results/california_housing.{md,json}`.

### CLI

```
bonsai fit      -c CONFIG --model OUT.msgpack
bonsai predict  -c CONFIG --model IN.msgpack [--data CSV] --out PREDS.csv
bonsai eval     -c CONFIG --model IN.msgpack [--data CSV]
bonsai bench    -c CONFIG          # time fit + predict
bonsai info                        # list (objective, grower, sampler) combos
bonsai params                      # dump the default config as TOML
```

`-c CONFIG` is a TOML file; see `configs/california_housing.toml` for the canonical shape. Any key can be overridden inline:

```
bonsai fit -c config.toml \
    --set tree.max_depth=8 \
    --set dispatch.grower_name=oblivious \
    --set dispatch.sampler_name=bernoulli \
    --set sampler.subsample=0.8 \
    --model out.msgpack
```

Append `--dump-config` to any subcommand to print the resolved config (after `-c` + `--set`) as TOML and exit, useful for verifying overrides before a long fit.

### Python

A nanobind extension wraps the same training pipeline the CLI uses — numpy in, numpy out, models interchangeable with the CLI's `.msgpack` format:

```python
import bonsai

model = bonsai.BonsaiRegressor(
    n_iters=200, learning_rate=0.05, grower="leafwise",
    early_stopping_rounds=20,
    params={"tree.lambda_l1": 0.5},   # any dotted config key the CLI accepts
)
model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
pred = model.predict(X_test)
model.save("model.msgpack")           # loadable by `bonsai predict` and vice versa
```

Install with `pip install .` (scikit-build-core builds the extension), or for development `cmake -B build -DBONSAI_PYTHON=ON && cmake --build build --target _bonsai` and set `PYTHONPATH=build/python`. Requires `nanobind` and `numpy` at build time. `scripts/compare.py` automatically adds in-process "native" rows to the benchmark table when the module is importable, timed the same way as the reference libraries.

`BonsaiRegressor(config="cfg.toml")` / `train(..., config=...)` load a TOML file as the base config (the CLI's `-c`); kwargs and `params` override it. For GPU training from Python, `make python-cuda` builds the extension in the CUDA tree (use `PYTHONPATH=build-cuda/python`), or `pip install . -C cmake.define.BONSAI_CUDA=ON -C cmake.define.BONSAI_CUDA_ARCH=sm_120`; `bonsai.cuda_available()` reports whether `cuda_*` growers can train on this machine.

`bonsai info` lists every `(objective, grower, sampler)` triple the binary knows how to dispatch to (currently 6×4×3 = 72 combos; growers that can't train on the current machine, like `cuda_depthwise` without a GPU, are marked predict-only).

## Build

Requires:
- LLVM ≥ 20: clang + libc++ (C++23: `std::print`, `std::mdspan`; libc++ gains float `std::from_chars` in 20, and clang 19 relaxed an over-broad OpenMP restriction on capturing structured bindings — clang 18 fails on both)
- CMake ≥ 3.28
- Ninja (recommended)

Supported on macOS (homebrew LLVM) and Linux (apt.llvm.org packages; built against LLVM's libc++, not libstdc++). The Makefile defaults to `/opt/homebrew/opt/llvm/bin` on macOS and `/usr/lib/llvm-21/bin` on Linux; override with `make LLVM_BIN=/path/to/llvm/bin ...`. On Ubuntu:

```
wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh && sudo bash /tmp/llvm.sh 21 && \
sudo apt-get install -y libc++-21-dev libc++abi-21-dev libomp-21-dev \
  clang-format-21 clang-tidy-21 clang-tools-21 ninja-build
```

```
make build           # configure + compile (build/)
make test            # ctest
make rebuild         # clean + build
make format          # clang-format src/ + include/
make lint            # clang-tidy
make help            # list all targets
```

The `make fit-benchmark` target additionally needs [uv](https://docs.astral.sh/uv/) for the Python sidecar that runs the reference libraries.

### CUDA (optional)

A GPU histogram backend (needs the CUDA toolkit ≥ 12) behind the `cuda_depthwise` grower. The grower is registered in every build — models trained with it load and predict anywhere, and `bonsai info` marks it predict-only where no device is available — but training needs a binary built with the backend, which lives in its own tree so the CPU-only `build/` stays pristine:

```
make build-cuda      # configure + compile with -DBONSAI_CUDA=ON (build-cuda/)
make test-cuda       # ctest against the CUDA build
./build-cuda/src/bonsai fit -c config.toml --set dispatch.grower_name=cuda_depthwise ...
```

Training is device-resident ([docs/architecture/11-gpu-resident.md](docs/architecture/11-gpu-resident.md)): histograms, rows, and split finding all live on the GPU, with only split decisions and child counts crossing the bus per level (float shared-memory accumulation merged in double — RMSE matches the CPU grower to library-comparison precision). The host grow loop remains the single algorithm narrative and the decision-maker; saved models are ordinary `DenseTree`s, identical in format to `depthwise`, and predict on any build. The kernel TU ([src/cuda/histogram_builder.cu](src/cuda/histogram_builder.cu)) is CUDA C++ compiled by the project's own clang (`-x cuda`) — same C++23, same libc++, no nvcc — so kernels use bonsai types and the shared gain math directly. Set `BONSAI_CUDA_ARCH` if `native` detection is unavailable; `BONSAI_CUDA_PROFILE=1` / `BONSAI_GROW_PROFILE=1` print per-fit breakdowns. Two growers run on the GPU: `cuda_depthwise` (fully device-resident) and `cuda_oblivious` (device level-find choosing one split per level across the whole frontier, CatBoost-style symmetric trees).

Measured on an RTX 5090 against each reference library, Year Prediction MSD (464 k × 90, 200 iters, depth 8), every `fit` timing the full pipeline — CSV read + binning + train — via [scripts/bench_gpu.py](scripts/bench_gpu.py):

| matchup (same tree strategy) | bonsai | reference | test RMSE (bonsai / ref) |
|---|--:|--:|--|
| `cuda_depthwise` vs **xgboost-GPU** (depthwise hist) | **2.5–2.7 s** | 4.8–7.1 s | 8.99 / 8.99 |
| `cuda_oblivious` vs **CatBoost-GPU** (oblivious) | **3.5 s** | 7.3–8.2 s | 9.17 / 9.14 |
| `leafwise` (CPU) vs **LightGBM-GPU** (leaf-wise CUDA) | **11.1–11.5 s** | 12.0–12.9 s | 9.09 / 8.95 |

bonsai wins each structure-matched comparison; the leafwise row is deliberately honest — bonsai has no device leafwise (best-first growth doesn't fit the level-batched resident plane), yet its CPU leafwise still beats LightGBM's CUDA backend on this card. On an A100, `cuda_depthwise` fits the same benchmark in 5.0 s vs 14.7 s for 16-thread CPU.

Beyond the head-to-head table, a synthetic scaling study sweeps rows (to 16M), cols (to 65k), and bins (to 65535) against all three libraries across five GPU generations — methodology in decision 46, data and log-log exponent fits in [benchmarks/results/scaling.md](benchmarks/results/scaling.md). The optimization rounds it seeded (decisions 47–48) cut the Python module's peak memory to 1.8× of the input matrix with byte-identical models, made 16k-bin CPU fits 5× faster on Linux, and moved the GPU histogram cliff from 3k to ~6k bins (17× on the 4095-bin cell). The same study caught a rentable GPU host with a 300µs sync round-trip — benchmark pods now pass a latency probe before any number is trusted (decision 48).

## Extending bonsai

The minimum to add a new objective / grower / sampler is two edits:

1. Add the type to the matching list in [include/bonsai/registry/typelists.hpp](include/bonsai/registry/typelists.hpp).
2. Specialize `impl_name<T>` in [include/bonsai/registry/names.hpp](include/bonsai/registry/names.hpp).

The cartesian-product dispatch table, the `bonsai info` listing, and the parametric tests all expand automatically.

Anything beyond a stateless impl needs a bit more:

- **Stateful impl** (e.g. a sampler with a tunable parameter): add a `FooConfig` struct + a declarative section in [include/bonsai/config/sections/](include/bonsai/config/sections/), and an `NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE` line in [src/io/model.cpp](src/io/model.cpp) so it round-trips through saved models.
- **New Tree type** (a grower whose output isn't `DenseTree` / `ObliviousTree`): add `tree_to_json` / `tree_from_json<NewTree>` in [src/io/model.cpp](src/io/model.cpp) alongside the existing pair. The shared registry helpers find them automatically.
- **Model-envelope schema change** (any of the above touches what's serialized): bump `k_format_version` in [src/io/model.cpp](src/io/model.cpp) so old models fail loud.

See [docs/architecture/6-dispatch.md](docs/architecture/6-dispatch.md) for how the typelist machinery and JSON serialization fit together.

## Project layout

```
include/bonsai/   public headers (Booster, Tree, Grower, Sampler, …)
src/              implementation + CLI (src/cli/)
tests/unit/       Catch2 unit + parity tests (ctest)
benchmarks/       Catch2 microbenchmarks (bonsai_bench)
scripts/          uv-managed Python: compare.py, fetch_toy.py
configs/          example TOML configs
docs/             design + decision logs
```

## Documentation

- **[docs/guide/](docs/guide/): the bonsai guide** — gradient boosting from math to code. Each chapter takes one concept (histograms, split finding, GOSS, DART, feature importance, determinism…) from intuition through the actual implementing code to an experiment you can run against the reference libraries. This is the differentiator: reference libraries document parameters; the guide documents *mechanics*, against readable code.
- [docs/report.md](docs/report.md): project retrospective (what was built, performance vs reference libraries, reflections, deferred work) + 2026-07 addendum.
- [docs/proposal.md](docs/proposal.md): initial project proposal.
- [docs/architecture/](docs/architecture/): per-component design notes.
- [docs/context.md](docs/context.md): project context and roadmap.
- [docs/decisions.md](docs/decisions.md): decision log.

## License

MIT © 2026 Daniel M Campos. See [LICENSE](LICENSE).
