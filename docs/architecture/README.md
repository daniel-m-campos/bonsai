# Architecture

Per-component design docs. Numbered roughly in build order. Source of
truth for choices is [`../decisions.md`](../decisions.md); when a doc
disagrees, decisions wins.

## Contents

| # | Doc | Status |
|---|---|---|
| 1 | [`1-dataset.md`](1-dataset.md) — Dataset, BinMapper, readers | done |
| 2 | [`2-histogram.md`](2-histogram.md) — gradient/hessian sums, subtraction, parallel reduce | done |
| 3 | [`3-tree.md`](3-tree.md) — Tree concept, `DenseTree` + `ObliviousTree`, depth-wise + oblivious growers, histogram splitter | done |
| 4 | [`4-objective.md`](4-objective.md) — Objective concept, MSE, logloss | done |
| 5 | [`5-booster.md`](5-booster.md) — Booster, training loop, `update_one_iter` | done |
| 6 | [`6-dispatch.md`](6-dispatch.md) — registry, runtime → static boundary | done |
| 7 | `7-parallel.md` — ParallelBackend, OpenMP, std::execution | TBD |
| 8 | [`8-config.md`](8-config.md) — Config, TOML, CLI overrides | partial (data slice only) |
| 9 | `9-cli.md` — subcommands, progress bars, logging | TBD |

## Cross-cutting concerns

**Dispatch.** Static poly inside `Booster`, runtime at config boundary.
Flat table over `cartesian_product_t<...>`; one vcall at boundary,
zero inside `update_one_iter`. See [`6-dispatch.md`](6-dispatch.md) +
decision 26.

**Threading.** Two backends behind `ParallelBackend` concept (OpenMP,
std::execution). Determinism is per-thread-count, not cross-thread
(decision 7). Forces per-thread local hists (no atomic FP adds);
final-merge order doesn't need to be `tid`-pinned. Detailed in
`7-parallel.md`.

**Errors.** Component constructors validate their config slice, throw
`ConfigError` with key path. No central validator. CLI top-level catches.

**Determinism contract** (decision 7). Same seed + data + config +
**same thread count** → same model bytes. Different thread counts:
predictions within numerical tolerance, bytes may differ. Tests in
`tests/integration/test_determinism.cpp` cover both: byte-equality at
fixed `n_threads`, prediction-tolerance across thread counts.

**Precision.** Float storage, double accumulators. Matches xgb/lgbm.

## Doc conventions

- Open with status line pointing at ratifying `decisions.md` entries.
- Code sketches show shape, not impl.
- Close with "What's not here" + cross-references.
- Filename: `N-<name>.md`.

## Test naming

Catch2 `TEST_CASE` names follow `"<Component>: <behavior under condition>"`:
PascalCase component, colon, behavioral phrase starting with a present-tense
verb, condition trailing after `when` / `if` / `for`. One `TEST_CASE` per
behavior — use `SECTION` for parameter variations within a behavior.

Tags stack a component tag with one or more behavioral tags drawn from a
fixed set: `[fit]`, `[transform]`, `[ctor]`, `[edge]`, `[nan]`, `[perf]`,
`[smoke]`. This makes `--tags [nan]` and `--tags [fit]` useful filters
across files.

```cpp
TEST_CASE("BinMapper: reserves a missing bin when column contains NaN",
          "[bin_mapper][fit][nan]")
```

Stubs get a single `"<Component>: smoke"` case tagged `[smoke]`, deleted
once a real behavior test lands.
