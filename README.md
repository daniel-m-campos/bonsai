# bonsai

A histogram gradient-boosted tree library and CLI, written in C++23.

## Usage

Bonsai is driven by a TOML config plus per-call `--set` overrides.

### Quick demo

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

`bonsai info` lists every `(objective, grower, sampler)` triple the binary knows how to dispatch to (currently 2×2×2 = 8 combos).

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

## Layout

```
include/bonsai/   public headers (Booster, Tree, Grower, Sampler, …)
src/              implementation + CLI (src/cli/)
tests/unit/       Catch2 unit + parity tests (ctest)
benchmarks/       Catch2 microbenchmarks (bonsai_bench)
scripts/          uv-managed Python: compare.py, fetch_toy.py
configs/          example TOML configs
docs/             design + decision logs
```

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

## Docs

- [docs/report.md](docs/report.md): project retrospective (what was built, performance vs reference libraries, reflections, deferred work).
- [docs/proposal.md](docs/proposal.md): initial project proposal.
- [docs/architecture/](docs/architecture/): per-component design notes.
- [docs/context.md](docs/context.md): project context and roadmap.
- [docs/decisions.md](docs/decisions.md): decision log.
