# Architecture

Per-component design docs. Numbered roughly in build order. Source of
truth for choices is [`../decisions.md`](../decisions.md); when a doc
disagrees, decisions wins.

## Contents

| # | Doc | Status |
|---|---|---|
| 1 | [`1-dataset.md`](1-dataset.md) — Dataset, BinMapper, readers | done |
| 2 | `2-histogram.md` — gradient/hessian sums, subtraction, parallel reduce | TBD |
| 3 | `3-tree.md` — Tree, Node, depth-wise grower, histogram splitter | TBD |
| 4 | `4-objective.md` — Objective concept, MSE, logloss | TBD |
| 5 | `5-booster.md` — Booster, training loop, `update_one_iter` | TBD |
| 6 | `6-dispatch.md` — registry, runtime → static boundary | TBD |
| 7 | `7-parallel.md` — ParallelBackend, OpenMP, std::execution | TBD |
| 8 | [`8-config.md`](8-config.md) — Config, TOML, CLI overrides | partial (data slice only) |
| 9 | `9-cli.md` — subcommands, progress bars, logging | TBD |

## Cross-cutting concerns

**Dispatch.** Static poly inside `Booster`, runtime at config boundary.
Shape is open — see `6-dispatch.md` when written.

**Threading.** Two backends behind `ParallelBackend` concept (OpenMP,
std::execution). Determinism required: same seed + data → same model
across thread counts. Forces per-thread local hists with fixed-order
merge. Detailed in `7-parallel.md`.

**Errors.** Component constructors validate their config slice, throw
`ConfigError` with key path. No central validator. CLI top-level catches.

**Determinism contract.** Same seed + data + config → same model bytes,
regardless of thread count or backend. Test in
`tests/integration/test_determinism.cpp`.

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
