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
  <img alt="tests" src="https://img.shields.io/badge/tests-498%20passing-success">
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
- **Concept-checked components.** Contract violations are caught at compile time, not runtime. Adding a stateless grower or sampler is two edits; objectives add three trait specializations (link, task, default metrics) that static assertions demand by name. Either way the dispatch table, CLI listing, and parametric tests expand automatically — see [Extending bonsai](#extending-bonsai).
- **A guide, not just docs.** [docs/guide/](docs/guide/) explains how gradient boosting works chapter by chapter — concept, math, then the ~50 real lines that implement it here, then an experiment. Written against this codebase because it's small enough to actually read.
- **Reference-library performance, measured.** Same-pod sweeps against xgboost, lightgbm, and catboost at matched settings: the CUDA grower owns the fastest slot at every row scale — its `oblivious` grower edges catboost-GPU and beats xgboost-GPU at 16M (18.4 / 18.5 / 19.9s, matched accuracy) at ~3× less host memory (7 vs 19–22GB); CPU growers beat lightgbm at 16M and on wide data, and tie xgboost-hist at 16M. Numbers and caveats in [Performance](#performance); raw runs in `benchmarks/results/`.
- **Three growers.** `depthwise` (level-wise, XGBoost-style), `leafwise` (best-first with a `max_leaves` budget, LightGBM-style), and `oblivious` (symmetric, CatBoost-style) — selectable per run from config.
- **Deterministic parallelism.** OpenMP across features and rows with ordered merges only: models and predictions are bit-identical across runs at a fixed thread count (`[parallel] n_threads`, 0 = capped auto), and bit-identical to serial at any count outside the u8 histogram fill (decision 49).
- **CLI-first, config-driven.** CatBoost-style subcommands, a strict TOML config, and inline `--set key=value` overrides; Python bindings (`make python`) share the exact same seams.
- **One-command build, no system dependencies.** CMake + FetchContent vendors every dependency; a clean checkout builds with a single command.

## Performance

From the 2026-07-13 re-baseline (`benchmarks/results/rebaseline-2026-07.jsonl`): synthetic regression, `fit()` timed end-to-end **including each library's own binning/ingest**, a 16-thread dual-EPYC-9554 host with an L40S, every variant on the same pod per cell. Test-set r² in parentheses; bonsai's two GPU growers are shown separately (`depthwise` trades a little speed for accuracy, `oblivious` is the symmetric-tree match to catboost).

Scaling rows (100 features, 255 bins, 100 trees, depth 8):

| rows | bonsai cuda dw | bonsai cuda obl | xgb cuda | catboost gpu | lgbm cpu | bonsai cpu obl |
|---|--:|--:|--:|--:|--:|--:|
| 250k | **0.5s** (.871) | 1.0s (.875) | 0.8s (.872) | 1.6s (.875) | 2.5s (.872) | 5.2s (.875) |
| 1M | **1.1s** (.876) | 1.5s (.876) | 1.7s (.876) | 2.3s (.876) | 5.1s (.877) | 7.8s (.876) |
| 4M | 4.5s (.878) | **4.4s** (.875) | 5.3s (.878) | 5.0s (.877) | 19.9s (.879) | 20.2s (.875) |
| 16M | 20.5s (.879) | **18.4s** (.876) | 19.9s (.880) | 18.5s (.876) | 111.3s (.879) | 73.3s (.876) |

Scaling features (1M rows):

| cols | bonsai cuda dw | bonsai cuda obl | xgb cuda | catboost gpu | lgbm cpu |
|---|--:|--:|--:|--:|--:|
| 256 | **2.8s** | 2.9s | 3.6s | 3.5s | 11.5s |
| 1024 | 11.2s | 10.6s | 12.5s | **9.7s** | 59.2s |
| 4096 | 44.2s | 41.9s | 50.6s | **35.8s** | 256.2s |

Honest caveats, because benchmarks without them are advertising: identical-model GPUs across the rental fleet measure up to ~25% apart, so only same-pod columns compare. bonsai owns the fastest slot at every row scale — `depthwise` up to 1M, and `oblivious` at 4M/16M, where it edges catboost (16M: **18.4s vs 18.5s**, both .876) and beats xgboost-GPU (19.9s); its `depthwise` runs slightly hotter (20.5s) for slightly more accuracy (.879). On wide data catboost keeps the lead (1024/4096 cols), with bonsai second and ahead of xgboost-GPU. bonsai's peak host RSS at 16M is **7.0GB vs xgboost's 22.2GB and catboost's 19.4GB** (a ~3× edge), and its predict is ~3× faster. Two earlier apparent gaps against catboost were bonsai bugs, since fixed — the oblivious accuracy defect (decision 63; note the old table's .864 CPU-oblivious r² was that bug, now .876) and a per-feature binning pass catboost didn't pay (decision 64). The path from 3× behind to this table is [guide chapter 11](docs/guide/11-performance-engineering.md); the cut-quality residual vs xgboost (+0.001 r²) is decision 55.

## Claims and proofs

Every performance or quality claim bonsai makes links to a reproducible run and the decision that records it — the point of a small, measured library is that you can check it.

| Claim | Evidence |
|---|---|
| **Bit-identical models across CPU architectures** (arm64 == x86-64) at a fixed thread count — no reference library offers this | decisions [59](docs/decisions.md)/60; asserted per-commit by [`cross-arch.yml`](.github/workflows/cross-arch.yml) via [`scripts/model_hash.py`](scripts/model_hash.py) |
| **Ties xgboost-hist at 16M rows on CPU** (75.8 vs 75.7s, same pod) | [decision 61](docs/decisions.md); [`benchmarks/results/cpu-prefetch-round-2026-07.jsonl`](benchmarks/results/cpu-prefetch-round-2026-07.jsonl) |
| **Fastest GPU slot at every row scale**; at 16M `oblivious` edges catboost (18.4 vs 18.5s) and beats xgboost-GPU (19.9s) at matched accuracy | [rebaseline jsonl](benchmarks/results/rebaseline-2026-07.jsonl), [scale-edge](benchmarks/catboost-scale-edge-2026-07.md), decisions 62–64; [`scripts/gpu_pareto.py`](scripts/gpu_pareto.py) |
| **Categorical parity with catboost within chance-band**, via preprocessing not an engine feature | [decision 58](docs/decisions.md); [categorical-tradeoff](benchmarks/categorical-tradeoff-2026-07.md); [`python/bonsai/encoding.py`](python/bonsai/encoding.py) |
| **Best library on 9 of 10 real datasets** (CPU quality campaign) | [quality-campaign](benchmarks/quality-campaign-2026-07.md), decisions 56–57 |
| **~3× less host memory than xgboost** at 16M (7.3 vs 22.2GB) and ~3× faster predict | [`benchmarks/results/rebaseline-2026-07.jsonl`](benchmarks/results/rebaseline-2026-07.jsonl) |
| **Ranking is a measured, scoped gap** — a modest ~+0.015 NDCG@10 to a *listwise* loss, not pairwise LambdaRank | [ranking-tradeoff](benchmarks/ranking-tradeoff-2026-07.md); [`scripts/probe_ranking.py`](scripts/probe_ranking.py) |
| **Every feature earns its place by measurement** — refutations are recorded too | the [feature-admission gate](.claude/skills/feature-admission/SKILL.md); declines in decisions 58/62 |

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

`bonsai info` lists every `(objective, grower, sampler)` triple the binary knows how to dispatch to (currently 7×5×3 = 105 combos; growers that can't train on the current machine, like `cuda_depthwise` without a GPU, are marked predict-only).

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

Training is device-resident ([docs/architecture/11-gpu-resident.md](docs/architecture/11-gpu-resident.md)): histograms, rows, and split finding all live on the GPU, with only split decisions and child counts crossing the bus per level (float shared-memory accumulation merged in double — RMSE matches the CPU grower to library-comparison precision). The host grow loop remains the single algorithm narrative and the decision-maker; saved models are ordinary `DenseTree`s, identical in format to `depthwise`, and predict on any build. The kernel TU ([src/cuda/histogram_engine.cu](src/cuda/histogram_engine.cu)) is CUDA C++ compiled by the project's own clang (`-x cuda`) — same C++23, same libc++, no nvcc — so kernels use bonsai types and the shared gain math directly. Set `BONSAI_CUDA_ARCH` if `native` detection is unavailable; `BONSAI_CUDA_PROFILE=1` / `BONSAI_GROW_PROFILE=1` print per-fit breakdowns. Two growers run on the GPU: `cuda_depthwise` (fully device-resident) and `cuda_oblivious` (device level-find choosing one split per level across the whole frontier, CatBoost-style symmetric trees).

GPU-vs-reference numbers live in [Performance](#performance) (same-pod, matched settings). A wider synthetic scaling study — rows to 16M, cols to 65k, bins to 65535, across five GPU generations — is in [benchmarks/results/scaling.md](benchmarks/results/scaling.md) (methodology in decision 46); the optimization rounds it seeded are the story of [guide chapter 11](docs/guide/11-performance-engineering.md).

## Extending bonsai

The minimum to add a new grower or sampler is two edits:

1. Add the type to the matching list in [include/bonsai/registry/typelists.hpp](include/bonsai/registry/typelists.hpp).
2. Specialize `impl_name<T>` in [include/bonsai/registry/names.hpp](include/bonsai/registry/names.hpp).

The cartesian-product dispatch table, the `bonsai info` listing, and the parametric tests all expand automatically.

A new **objective** takes the same two edits plus three trait specializations that static assertions will demand by name (so forgetting one is a compile error, not a runtime surprise): `link_inverse_of<T>` and `default_metrics_of<T>` in [include/bonsai/objective_traits.hpp](include/bonsai/objective_traits.hpp) (implementations in `src/objective_traits.cpp`), and its `task_of<T>` kind. Budget ~6 file touches end to end for a configured objective; the checklist order is typelist → name → traits → impl → config field → TOML section.

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
