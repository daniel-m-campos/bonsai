# Design Review: bonsai

> **Date**: 2026-05-19 **Scope**: Spine + CLI + config + registry + I/O, as of the Phase 2 milestone (commit `a1dfafb`, "docs: spine complete; relax AI policy; insert Phase 2.5"). ~4.2 KLOC across 47 headers and 25 sources. **Method**: `/design-review` skill (SOLID → DOD → supplementary principles), cross-referenced against the reference GBT libraries (XGBoost, LightGBM, CatBoost, sklearn). **Audience**: future-me / refactoring agents driving Phase 2.5 cleanup before Phase 3 parallelism.

## Project Overview

- **Repository**: `bonsai`
- **Language**: C++23
- **Description**: From-scratch histogram-based gradient-boosted-tree library. Spine is end-to-end working on California Housing as of 2026-05-18 (decision 28).

## Scoping

- **Extensibility pressure**: **medium-high**. Pluggable `Objective`, `TreeGrower`, `SplitFinder`, `Sampler`, plus future `ParallelBackend` and categorical handlers. Bounded by a closed compile-time plugin set (proposal §3.4) — not a third-party plugin ecosystem like XGBoost.
- **Performance pressure**: **high inside `update_one_iter`** (histogram build, split scoring, tree walk over ~500K rows), **low everywhere else** (CLI, config parsing, model I/O, CSV).
- **Data clarity**: **clear**. `(CSV rows, labels) → ColumnBatch → Dataset(binned, column-major) → per-iter (scores, grad, hess) → tree → updated scores`. Documented in [docs/architecture/2-histogram.md](../architecture/2-histogram.md) and the §6 sketch.

This is exactly the **"high extensibility + high perf" cell**: SOLID at boundaries, DOD inside the hot path. The proposal commits to this split explicitly; this review judges the code on that basis.

## Per-Entity Analysis

### Dataset / BinMapper / BinMappers / Histogram

| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ✅ | `BinMapper` fits cuts; `BinMappers` aggregates; `Dataset::bin` does the column-major transform; `Histogram` is per-(node,feature) cell array. Each changes for one reason. Two-stage `fit`/`bin` (decision 2) is the right split — distinct from LightGBM's coupled `Dataset(reference=)`. |
| I | ✅ | `BinMappers` exposes 3 read methods + factories; `Histogram` 7 narrow ops (`add`, `clear`, `operator-=`, four view accessors). No god-class surface. |
| D | ✅ | `Dataset::bin` takes `(batch, mappers, cfg)` — pure transform, no globals or implicit I/O. |
| O | ⚠️ | Adding categorical features will touch `Dataset` (categorical is a placeholder bool; categorical handlers are Phase 4). The seam isn't designed yet. **Mention only**, not a current issue. |
| L | N/A | No subtype hierarchy. |
| D1 | ✅ | `Dataset` is *the* data-flow center; everything else is a view over it. Crystal clear. |
| D2 | ✅ | Column-major `std::vector<std::vector<bin_id_t>>`, AoS `HistCell{sum_grad, sum_hess}`. AoS is correct here — scatter-write hot loop (`hist[col[i]] += grad[i]`) keeps grad+hess on one cache line per indexed update (rationale in [2-histogram.md](../architecture/2-histogram.md)). One nit: `features_` is a vector-of-vectors — 90 pointer indirections to locate columns. Outside the inner loop, so noise. |
| D3 | ⚠️ | `bin_id_t = uint16_t` uniform, but `max_bin=255` default → 50% wasted bandwidth in binned storage (decision 4 explicitly accepted this). Sklearn's `HistGradientBoosting` uses 8-bit bins; XGBoost matches you (uint16). Acceptable since decision documents the trade. |
| D4 | ✅ | `Histogram::operator-=` is the subtraction trick; not speculative — used from day one per decision 19. `BinMapper::fit` is sampling + nth_element, ~30 LOC, no over-engineering. |

### SplitNode / Split / HistogramSplitFinder

| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ✅ | `find()` is one static method. Missing-bin routing logic is one well-bounded loop in [src/split.cpp:13-58](../../src/split.cpp#L13-L58). |
| I | ✅ | `SplitFinder` concept is one method. Minimal. |
| D | ✅ | Takes `(SplitNode, TreeConfig)` — no hidden state. |
| O | ✅ | `DepthwiseGrower<SplitterT = HistogramSplitFinder>` is the seam. Exact splitter / quantile sketch slot in as new template args. |
| D1 | ✅ | Input `(per-feature histograms, parent gain)`, output `(feature, bin, gain, default_left)`. Clean transform. |
| D3 | ✅ | Inner two loops are straight-line over `n_bins ≤ 256`; both default-direction branches enumerated explicitly (`for (bool default_left : {true, false})`) — readable and the optimizer flattens it. |

### DenseTree / ObliviousTree

| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ✅ | Each owns its node layout + predict kernel. `DenseTree::walk_row` is variant-dispatch over `InternalNode`/`LeafNode`; `ObliviousTree::walk_row` is the branchless fixed-depth gather. |
| O | ✅ | Both satisfy the `Tree` concept ([include/bonsai/tree.hpp:14](../../include/bonsai/tree.hpp#L14)) without sharing a base — exactly the "concepts over inheritance" claim from the proposal. |
| L | ✅ | Both honor the same predict contract: `predict(X, out)` accumulates into `out`. |
| D2 | ✅ | `DenseTree::Nodes = vector<variant<InternalNode, LeafNode>>` — variant adds 8B tag per node; for depth 6 that's ~64 nodes × 24B ≈ 1.5KB per tree. Predict walks `O(depth)` per row. Fine. `ObliviousTree::leaf_values_` is `2^depth` floats — branchless gather works directly. |
| D3 | ✅ | Branchless NaN routing (decision 13) compiles to `cmov`/select on x86-64; preserves vectorizability for `ObliviousTree`. Verified by inspection of [src/tree.cpp:27-30](../../src/tree.cpp#L27-L30). |

**Drift flag:** Decision 8 says oblivious grower ships in Phase 1 alongside `ObliviousTree`. The tree type exists; **no `ObliviousGrower` exists in headers and the registry's `Growers` typelist contains only `DepthwiseGrower<HistogramSplitFinder>`**. The CLI cannot fit an oblivious tree today. Either the decision is stale or the grower is half-landed. Reconcile before reporting.

### Objective / Sampler / Metric

| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ✅ | `MSEObjective::{compute, eval, init_score}`, three thin static functions in [src/objective.cpp](../../src/objective.cpp). `LogLossObjective::eval` uses the numerically stable softplus form (`max(0,score) + log1p(exp(-|score|)) - y*score`) — best-in-class detail that XGBoost and LightGBM both also use. |
| I | ✅ | `Objective` concept has 3 required statics. External traits (`impl_name`, `task_of`, `link_inverse_of`, `default_metrics_of`) keep optional metadata out of the concept — clean **interface segregation by trait**, not by inheritance. |
| O | ⚠️ | Adding a third objective requires touching: (1) `objective.hpp` (struct), (2) `objective.cpp` (impls), (3) `registry/typelists.hpp` (typelist), (4) `registry/names.hpp` (impl_name), (5) `objective_traits.hpp` (`task_of`, `link_inverse_of`, `default_metrics_of`), (6) `objective_traits.cpp` (defaults_of impl). That's **6 spots**. The architecture doc claims "one new file plus one entry in the typelist." Consider folding the four trait specializations into a single per-impl `metadata<O>` trait struct, or co-locating them with the objective definition. |
| L | N/A | Concepts, not subtypes. |
| D | ✅ | Static methods; everything injected through call args. |
| D1 | ✅ | `(preds, labels) → (grad, hess) \| scalar \| scalar`. Transform-shaped, perfect for static dispatch. |
| D3 | ✅ | `Metric` is a POD `{name, task, function pointer}` (40B). The function-pointer-over-`std::function`-or-virtual choice is explained in a tight comment ([include/bonsai/metric.hpp:13-19](../../include/bonsai/metric.hpp#L13-L19)) — exactly the right idiom for `inline constexpr` registry entries. |

### Booster<Obj, Gr, Sa> + IBooster + Registry

| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ✅ | `Booster` orchestrates one iteration in 6 steps ([src/booster.cpp:23-65](../../src/booster.cpp#L23-L65)); each step is one logical statement. `IBooster` is the 4-method boundary erasure. |
| I | ✅ | `IBooster`: `update_one_iter`, `eval`, `predict`, `n_iters`. Four methods, each used by the CLI. No fat interface. The save/load accessors (`trees()`, `init_score()`, `load_state()`) are on the concrete class only — the I/O module can't befriend the templated type, so they're public-but-undocumented-API ([include/bonsai/booster.hpp:49-61](../../include/bonsai/booster.hpp#L49-L61)). Pragmatic. |
| D | ✅ | Booster takes `Config const&` (POD) at construction; weights via `Dataset`. RNG owned by Booster, seeded from config. Testable end-to-end. |
| O | ✅ | Adding a fourth `Sa` is one typelist entry + one `impl_name`. The cartesian-product table regenerates mechanically. |
| L | ✅ | `Booster<O,G,Sa> final : IBooster` — single concrete shape, all overrides honor the contract. |
| D1 | ✅ | The training loop's data flow is exactly what the architecture doc claims it is. |
| D2 | ⚠️ | Per-iter buffers (`grad_`, `hess_`, `scores_`, `row_indices_`) are class members — reused across iterations, not reallocated. Good. **But** `SplitNode` allocates `vector<Histogram>` per split (decision 18 explicitly rejected a histogram pool). At Phase 1 scale fine; verify in Phase 3 profiling. |
| D3 | ✅ | **One vcall per `update_one_iter`** at the CLI boundary, zero inside (decision 26). The architectural centerpiece. Verified: `train_in_memory` calls `booster->update_one_iter(train)` → dispatches into the monomorphized `Booster<MSE,Depth,AllRows>::update_one_iter` → calls `objective_type::compute(...)` → all subsequent calls are direct. |
| D4 | ✅ | The flat-table dispatch is non-trivial machinery (cartesian product, `for_each_type`, name-keyed linear scan) — but it's *required* by the architectural claim, not speculative. The decision matrix in [6-dispatch.md](../architecture/6-dispatch.md) shows candidates B/C were considered and rejected on call-site shape, not on a knee-jerk preference. |

### TypeList / cartesian_product_t / for_each_type

| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ✅ | Pure type-level algorithms. Each metafunction has one job (`size`, `type_at`, `concat`, `prepend_each`, etc.). |
| O | ✅ | New algorithms are new structs; existing ones don't change. |
| D4 | ✅ | ~30 LOC of fold-and-recurse, comparable to Boost.MP11's `mp_product` but hand-written deliberately as the project's metaprogramming showcase. Single-purpose; no speculative parameters. Builds a typelist `cartesian_product_t` on top of the basic typelist machinery. Worth a one-paragraph explainer comment for readers new to the pattern; the test file `test_typelist.cpp` provides one form of that. |

### Config system (TOML + section descriptors + FieldCodec)

| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ✅ | `Section` describes a TOML section; `FieldCodec<T>` parses/dumps one type; `load_section` / `apply_leaf` / `dump` are template functions that fold over fields. Each piece has one job. |
| O | ✅ | Adding a config field = one `field<&Sub::name>()` line in the section header (`include/bonsai/config/sections/*.hpp`). Adding a new type = one `FieldCodec<T>` specialization. The architecture vision matches reality here. |
| D | ✅ | Codecs return `std::expected<T, string>` (C++23) — no exceptions in the parser hot path; errors carry the key path. |
| D3 | N/A | Cold path. |
| D4 | ⚠️ | `field_name<MemPtr>()` parses `std::source_location::function_name()` to extract the member name — clever, but it relies on `__PRETTY_FUNCTION__` format that's compiler-pinned with `static_assert` ([include/bonsai/config/internal/field_name.hpp](../../include/bonsai/config/internal/field_name.hpp)). The alternative — `field("max_depth", &TreeConfig::max_depth)` — is one extra string literal per field, lower cleverness, no dependency on `function_name()` format. **Trade-off acknowledged**: the current form means *rename a struct member and the TOML key follows automatically*. If a future GCC/Clang version changes the format the build fails loudly, which is the right behavior. P2996 reflection retires this seamlessly per the file's own comment. Defensible call. |

### CLI / pipeline / IO

| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ✅ | Each subcommand handler is one short function. Reusable logic lives in `cli/pipeline.cpp` and `cli/common.cpp`. |
| I | ✅ | Subcommand handlers see only their own `*Opts` struct. |
| D | ⚠️ | `run_fit` / `run_predict` / `run_eval` print directly to `stdout`/`stderr` via `std::print` — testable as integration tests via CLI invocation, less so as unit tests. The `pipeline.hpp` layer (callbacks, `train_with_progress(FitTickFn)`) is the testable seam, and it is properly designed. **Acceptable for a CLI.** |
| D4 | ⚠️ | `io/model.cpp` uses `dynamic_cast` in `try_save_as<B>` / `try_load_into<B>` and walks a hard-coded `BoosterTypes = TypeList<Booster<MSE, ...>, Booster<LogLoss, ...>>` ([src/io/model.cpp:187-189](../../src/io/model.cpp#L187-L189)). **This duplicates the cartesian product** that `registry/typelists.hpp` already maintains. A third combo silently fails to save unless this list is updated too. Derive it from the same `Configurations` instead. |

## Supplementary Principles (cross-cutting)

| Principle | Rating | Analysis |
|-----------|--------|----------|
| DRY | ⚠️ | (1) `make_booster.cpp` and `objective_dispatch.cpp` repeat the `for_each_type<typelist>([&]<typename T>{ static_assert(HasName<T>); out[i++] = Entry{...}; })` shape **four times** — extractable to `make_named_table<Typelist, Entry, MakeEntry>()`. (2) `choose_metric_names` is **near-identical** in [src/cli/fit.cpp:26-42](../../src/cli/fit.cpp#L26-L42) and [src/cli/eval.cpp:27-44](../../src/cli/eval.cpp#L27-L44) — should live in `cli/common.hpp`. (3) `BoosterTypes` in `io/model.cpp` duplicates the cartesian product — derive from `Configurations`. (4) Linear `for (auto const &e : table) if (e.name == n) ...` lookup is reimplemented in three dispatch functions; small enough to leave, but a `find_in_table` helper would be 3 lines. |
| CoI | ✅ | One inheritance edge in the entire library: `Booster<...> final : IBooster`. One level deep, purely for boundary erasure. Everywhere else: composition (Booster holds Obj/Gr/Sa as template args; Grower holds SplitFinder; Dataset owns BinMappers). Concepts substitute for abstract bases throughout. **Best-in-class** application of the principle for a project of this scope. |
| LoD | ✅ | A few 3-step chains on POD aggregates: `ds.mappers()[split.feature_id].cuts()[split.bin_id]` (grower.cpp), `cfg.booster_config.learning_rate`. These are fine — LoD applies to behavioral objects, not config structs and view containers. No problematic chains found. |

## Summary

| Principle | Grade | Highlights |
|-----------|-------|------------|
| S | A− | Each component owns one concern; the BinMapper/BinMappers/Dataset/Histogram decomposition is textbook. |
| I | A | Concepts + external traits keep interfaces narrow; objectives don't expose `transform` they wouldn't use. |
| D | A | Functions take POD config + spans; no hidden globals; CLI handlers are the only stdout/stderr coupling. |
| O | B+ | Extension surfaces are clean *except* the objective-adds-touch-six-files friction and the `BoosterTypes` duplication in I/O. |
| L | A | Single boundary impl; concepts replace hierarchies that would otherwise raise LSP risk. |
| D1 | A | Data flow is explicit, documented, and matches the code. |
| D2 | A− | Column-major bins + AoS histograms + per-thread-local design = textbook DOD. One nit: variant-per-feature for uint8/uint16 bins was rejected but is reversible (decision 4). |
| D3 | A | One vcall per iter (boundary), zero inside (monomorphized). The architectural claim is real and provable with the planned dispatch microbenchmark. |
| D4 | B+ | Cartesian product machinery is justified by the closed-set claim. `field_name` via `source_location` is clever-but-justified (one trade-off, locked-in defensibly). Minor: four exception classes where one with a kind enum would do. |
| DRY | B− | Four duplicated for_each_type tables; `choose_metric_names` copy in two CLI files; `BoosterTypes` in model.cpp duplicates the registry's cartesian product. |
| CoI | A+ | Exactly one inheritance edge; concepts everywhere else. |
| LoD | A | Clean throughout. |

## Top 3 Recommendations

### 1. Derive `BoosterTypes` in `src/io/model.cpp` from `Configurations`

…the registry's cartesian product — so adding a combo to the typelist auto-extends save/load. As of today, `io::save_booster` will throw `"concrete Booster type not in registered set"` for any combo that lives in `make_booster`'s table but not in `model.cpp`'s hand-listed `BoosterTypes`. This is the highest-impact fix: a latent correctness bug, not just code smell. Replace the `dynamic_cast` chain with name-keyed dispatch (you already know the dispatch triple from the on-disk magic header) — saves N RTTI calls per save/load and removes the duplication.

Concrete sketch:

```cpp
// In io/model.cpp, replace BoosterTypes + save_dispatch/load_dispatch with:

using Configurations = cartesian_product_t<Objectives, Growers, Samplers>;

template <typename Combo>
using BoosterFor = Booster<type_at_t<0, Combo>,
                           type_at_t<1, Combo>,
                           type_at_t<2, Combo>>;

bool save_dispatch(IBooster const& booster, DispatchConfig const& disp, json& out) {
    bool ok = false;
    for_each_type<Configurations>([&]<typename Combo>{
        if (ok) return;
        using O  = type_at_t<0, Combo>;
        using G  = type_at_t<1, Combo>;
        using Sa = type_at_t<2, Combo>;
        if (disp.objective_name != impl_name<O>::value) return;
        if (disp.grower_name    != impl_name<G>::value) return;
        if (disp.sampler_name   != impl_name<Sa>::value) return;
        ok = try_save_as<BoosterFor<Combo>>(booster, out);
    });
    return ok;
}
```

Name-keyed early-out means at most one `dynamic_cast` per call instead of N. `save_booster` already takes `DispatchConfig const&`, so the triple is in scope. `load_booster` reads the triple from the on-disk magic before reconstructing the booster — same shape.

### 2. Reconcile the oblivious-grower drift

Decision 8 says `ObliviousGrower → ObliviousTree` ships in Phase 1; the tree class and its tests exist, but no `ObliviousGrower` is in [include/bonsai/grower.hpp](../../include/bonsai/grower.hpp) and the `Growers` typelist contains only `DepthwiseGrower<HistogramSplitFinder>`. Either land the grower (Phase 2.5 is the natural window) or update [decisions.md](../decisions.md) and [3-tree.md](../architecture/3-tree.md) to reflect what actually shipped. As-is, the codebase and the design docs disagree.

If landing the grower: same shape as `DepthwiseGrower<SplitterT>`, returns `GrowResult<ObliviousTree>`, holds a `vector<LevelSplit>` under construction and a `vector<float>` leaf table sized `2^depth`. The per-parent subtraction-trick protocol from decision 19 carries over; the per-level fold from decision 17 is the new piece.

### 3. Consolidate per-objective metadata

Adding a third objective touches six files (struct, impl, typelist, `impl_name`, `task_of`, `link_inverse_of`, `default_metrics_of`). Fold the four traits into a single `objective_metadata<O>` specialization co-located with the objective definition — keeps the concept narrow but cuts the friction from six edits to three (`objective.hpp`, `objective.cpp`, `typelist`). The four `_table` builders in `objective_dispatch.cpp` collapse into one. Strictly an ergonomics fix, but the architecture doc claims this is already the case; the recommendation is to make the doc true.

Concrete sketch:

```cpp
// include/bonsai/objective_traits.hpp
template <typename O> struct objective_metadata;  // primary

template <> struct objective_metadata<MSEObjective> {
    static constexpr std::string_view name = "mse";
    static constexpr TaskKind task = TaskKind::regression;
    static void apply_link(floats_out) {}
    static std::span<std::string_view const> default_metrics() {
        static constexpr std::array names{std::string_view{"rmse"}};
        return names;
    }
};
// LogLossObjective: same shape, different bodies.
```

Then `impl_name<O>::value`, `task_of<O>::value`, `link_inverse_of<O>::apply`, and `default_metrics_of<O>::value()` all forward to `objective_metadata<O>`. Old specializations deprecate cleanly; the registry's `for_each_type` loop in `objective_dispatch.cpp` collapses to a single pass.

## Key Takeaways

**The architectural claim is real and the code backs it up.** The proposal's centerpiece — *static polymorphism inside the hot path, dynamic at the configuration boundary* — is honestly implemented: monomorphized `Booster<O, G, Sa>` for the training loop, one `IBooster` vcall at the CLI seam, concepts (not virtual bases) for the four extension points. The cartesian-product flat table is moderate metaprogramming, but it's justified by what it buys (invalid-combo elimination, mechanical `--help` enumeration, future 5D promotion when `ParallelBackend` lands), and the design doc walks through three alternatives with a decision matrix before picking. This is what "informed divergence from the reference libraries" looks like.

**Modern C++ techniques are used coherently.** Typelists, pack expansion, fold expressions, and concepts are all present and load-bearing. CRTP is not used — the design picks template parameter lists instead, which is the right call for this dispatch shape. The abstract-factory pattern is explicitly rejected in [6-dispatch.md](../architecture/6-dispatch.md) — the rejection is reasoned, not ignorance. Concurrency primitives are not yet visible in bonsai because `ParallelBackend` is deferred to Phase 3. The one technique that stands out is `std::source_location`-based field reflection — defensible because the file pins the compiler format with `static_assert` and labels itself as a P2996 stand-in.

**Reference-library comparison is honest.** XGBoost / LightGBM / CatBoost all use macro-based string registries with vtables in the inner loop, because their plugin sets are open across third-party binaries. bonsai's closed set lets it monomorphize — a *real* architectural improvement, not a stylistic preference. The microbenchmark with/without virtualization (proposal §5) is the falsification test for this claim; design it carefully because that table is the centerpiece of the final report. LightGBM's `CheckParamConflict` (hundreds of lines of sequential validation) is replaced in bonsai by typelist-construction-level filtering ([6-dispatch.md](../architecture/6-dispatch.md) §"Encoding compatibility") — that section is the strongest single design contribution and would make a great paragraph in the writeup.

**The actionable items are not architectural.** Top three are: derive `BoosterTypes` from `Configurations` (latent correctness), reconcile oblivious-grower drift (doc-vs-code), and consolidate the six-touch-points for adding an objective (ergonomics that the doc claims you already have). None of these threaten the core architectural claims; all three are achievable in Phase 2.5 without spine changes, which is exactly what Phase 2.5 was inserted for.

---

## Refactoring tracking

When tackling these from this file, link each PR/commit back here so follow-up reviews can diff against the baseline. Suggested attribution:

- `refactor(io): derive BoosterTypes from Configurations (review §Rec 1)`
- `feat(grower): land ObliviousGrower (review §Rec 2 / decision 8)`
- `refactor(objective): consolidate trait specializations (review §Rec 3)`
