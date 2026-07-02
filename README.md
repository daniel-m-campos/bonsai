<p align="center">
  <img src="docs/assets/bonsai-logo.png" alt="bonsai" width="640">
</p>

<p align="center">
  <b>A histogram gradient-boosted tree library and CLI in modern C++23.</b>
</p>

<p align="center">
  <img alt="C++23" src="https://img.shields.io/badge/C%2B%2B-23-00599C?logo=cplusplus&logoColor=white">
  <img alt="CMake" src="https://img.shields.io/badge/CMake-%E2%89%A5%203.28-064F8C?logo=cmake&logoColor=white">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <img alt="tests" src="https://img.shields.io/badge/tests-316%20passing-success">
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

`bonsai info` lists every `(objective, grower, sampler)` triple the binary knows how to dispatch to (currently 5×3×3 = 45 combos).

## Build

Requires:
- clang ≥ 18 (C++23: `std::print`, concepts, ranges)
- CMake ≥ 3.28
- Ninja (recommended)

```
make build           # configure + compile (build/)
make test            # ctest
make rebuild         # clean + build
make format          # clang-format src/ + include/
make lint            # clang-tidy
make help            # list all targets
```

The `make fit-benchmark` target additionally needs [uv](https://docs.astral.sh/uv/) for the Python sidecar that runs the reference libraries.

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
