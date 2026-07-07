# Architecture

Per-component design docs. Numbered roughly in build order. Source of truth for choices is [`../decisions.md`](../decisions.md); when a doc disagrees, decisions wins.

## Contents

| # | Doc | Status |
|---|---|---|
| 1 | [`1-dataset.md`](1-dataset.md) — Dataset, BinMapper, readers | done |
| 2 | [`2-histogram.md`](2-histogram.md) — gradient/hessian sums, subtraction, parallel reduce | done |
| 3 | [`3-tree.md`](3-tree.md) — Tree concept, `DenseTree` + `ObliviousTree`, depth-wise + oblivious growers, histogram splitter | done |
| 4 | [`4-objective.md`](4-objective.md) — Objective concept, MSE, logloss | done |
| 5 | [`5-booster.md`](5-booster.md) — Booster, training loop, `update_one_iter` | done |
| 6 | [`6-dispatch.md`](6-dispatch.md) — registry, runtime → static boundary | done |
| 7 | `7-parallel.md` — parallelism seam, determinism contract | done (decision 32) |
| 8 | [`8-config.md`](8-config.md) — Config, TOML, CLI overrides | done |
| 9 | [`9-cli.md`](9-cli.md) — subcommands, overrides, fit-time output | done |
| 10 | [`10-cuda.md`](10-cuda.md) — GPU histogram backend: builder policy, level batching, runtime capability | done |

## Cross-cutting concerns

**Dispatch.** Static poly inside `Booster`, runtime at config boundary. Flat table over `cartesian_product_t<...>`; one vcall at boundary, zero inside `update_one_iter`. See [`6-dispatch.md`](6-dispatch.md) + decision 26.

**Threading.** Shipped as a single seam, not a backend concept (decision 32): `parallel::for_each_index` in `bonsai/parallel.hpp`, OpenMP body with serial fallback, `[parallel] n_threads` config. Every parallel site assigns each index to exactly one thread with no cross-thread reductions. Details in [`7-parallel.md`](7-parallel.md).

**Errors.** Component constructors validate their config slice, throw `ConfigError` with key path. No central validator. CLI top-level catches.

**Determinism contract** — now *stronger* than originally specified (decision 7 promised fixed-thread-count byte equality). Because no parallel site performs a cross-thread floating-point reduction, models and predictions are **bit-identical to a serial run at any thread count** (decision 32). The weaker fixed-N contract returns only if row-parallel histograms with per-thread merges ever land; `7-parallel.md` spells out the trade.

**Precision.** Float storage, double accumulators. Matches xgb/lgbm.

## Doc conventions

- Open with status line pointing at ratifying `decisions.md` entries.
- Code sketches show shape, not impl.
- Close with "What's not here" + cross-references.
- Filename: `N-<name>.md`.

## Test naming

Catch2 `TEST_CASE` names follow `"<Component>: <behavior under condition>"`: PascalCase component, colon, behavioral phrase starting with a present-tense verb, condition trailing after `when` / `if` / `for`. One `TEST_CASE` per behavior — use `SECTION` for parameter variations within a behavior.

Tags stack a component tag with one or more behavioral tags drawn from a fixed set: `[fit]`, `[transform]`, `[ctor]`, `[edge]`, `[nan]`, `[perf]`, `[smoke]`. This makes `--tags [nan]` and `--tags [fit]` useful filters across files.

```cpp
TEST_CASE("BinMapper: reserves a missing bin when column contains NaN",
          "[bin_mapper][fit][nan]")
```

Stubs get a single `"<Component>: smoke"` case tagged `[smoke]`, deleted once a real behavior test lands.
