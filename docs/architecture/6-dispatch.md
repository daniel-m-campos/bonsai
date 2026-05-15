# 6. Dispatch — runtime → static boundary

> **Status:** Ratified by decision 26. Author: <user>. AI drafted
> outline + reference survey; design choices and ratification user-authored.

## What this doc decides

The mechanism that takes four runtime strings (objective, grower,
splitter, sampler — plus optionally backend) parsed from TOML and
produces one statically-typed `Booster<Obj, Gr, Sp, Sa>` instance,
without a four-deep nested-lambda tower.

Out of scope: the `Booster` class itself (`5-booster.md`), the
`Registry<Base, Config, Impls...>` machinery once its shape is fixed
(brief sketch lives here, full impl notes in `5-booster.md` or its own
`registry` section).

## Constraints

- **No macros.** Whole point of the typelist approach.
- **No static-init / linker-stripping.** Names live in typelists, not
  global registration objects.
- **Concepts enforce contract at compile time.** A bad impl in the
  typelist fails to compile, with a concept diagnostic.
- **Cartesian product is bounded.** ~15–90 `Booster` variants over the
  life of the project. Fine for compile time + binary size.
- **One runtime decision point.** `make_booster(config)` is the only
  place a string becomes a type. After that, everything is static.
- **Single canonical name per impl.** No aliases (decision: no registry
  aliases).
- **Reflection branch is optional.** Typelist version is the production
  path; reflection (`P2996`) is a showcase variant.

## What the reference libraries do (and why we don't copy them)

| Library | Mechanism | Why we don't copy |
|---|---|---|
| **xgboost** | `ObjFunction::Create(name)` factory backed by a global registry populated by `XGBOOST_REGISTER_OBJECTIVE` macros. Returns `unique_ptr<Base>`; vtable in inner loop. | Macros + global init + dynamic dispatch in hot path. We're explicitly trying to do the opposite. |
| **LightGBM** | `Objective::CreateObjectiveFunction(name, config)` switch-style factory; same vtable-in-hotpath story. | Same reasons as xgb. |
| **CatBoost** | Heavier OOP factories; runtime dispatch throughout. | Same reasons. |

Common pattern across all three: **type-erased base + dynamic
dispatch everywhere** because they have to support open plugin sets
across third-party binaries. bonsai does not — the impl set is closed
at our compile time.

The architectural improvement to claim in the writeup: **closed plugin
set + concept contracts → we can monomorphize the whole training loop**.

## The shape question

Given the constraint set above, the open question is purely
**ergonomics**: how do you write the function that takes four strings
and produces one typed object, when each string independently selects
a type from its own typelist, and the final type is the cartesian
product?

The naive form is the four-deep nested-lambda tower:

```cpp
auto booster = obj_registry.create(cfg.objective_name, cfg, [&](auto obj) {
    return gr_registry.create(cfg.grower_name, cfg, [&](auto gr) {
        return sp_registry.create(cfg.splitter_name, cfg, [&](auto sp) {
            return sa_registry.create(cfg.sampler_name, cfg, [&](auto sa) {
                return Booster<decltype(obj), decltype(gr),
                               decltype(sp), decltype(sa)>{cfg, ...};
            });
        });
    });
});
```

It works. The proposal flagged it as ugly. Realistic alternatives below.

## Type-level builder (a notation we use throughout)

A small refinement that cleans up the *body* of any candidate, but
does not by itself flatten the call site. Worth fixing notation
upfront because both candidates lean on it.

```cpp
struct Unset {};

template <typename Obj = Unset, typename Gr = Unset,
          typename Sp = Unset, typename Sa = Unset>
struct BoosterBuilder {
    template <Objective O>
    constexpr auto with_objective() const {
        return BoosterBuilder<O, Gr, Sp, Sa>{};
    }
    template <TreeGrower G>
    constexpr auto with_grower() const {
        return BoosterBuilder<Obj, G, Sp, Sa>{};
    }
    // ... with_splitter, with_sampler ...

    auto build(Config const& cfg) const
        requires (!std::same_as<Obj, Unset> && !std::same_as<Gr, Unset>
               && !std::same_as<Sp, Unset> && !std::same_as<Sa, Unset>)
    {
        return Booster<Obj, Gr, Sp, Sa>{cfg};
    }
};
```

Each `with_X` returns a new `BoosterBuilder<...>` with one more type
slot filled. `build()` is constrained to fully-resolved instantiations
via `requires`. Order-independent, self-documenting, zero runtime cost.

This pattern (also called a **type-level builder** or **compile-time
type builder**) is well-known — Boost.Hana and the embedded HAL register
libraries (Kvasir, Modm) generalize it.

What it does **not** do: pick types from runtime strings. That still
requires registry callbacks. The builder cleans up the *type assembly*
inside whatever dispatch shape we choose.

## Candidate A — flat dispatch table over the cartesian product

**Idea.** Enumerate the cartesian product at compile time. Build a
`std::array` keyed on `tuple<string_view, string_view, string_view,
string_view>` whose values are factory function pointers, each one
already monomorphized for its specific four-tuple of types.

**Sketch (using the builder for the body, `cartesian_product_t` for the loop):**

```cpp
using Objectives = TypeList<MSE, Logloss>;
using Growers    = TypeList<Depthwise, Oblivious>;
using Splitters  = TypeList<Hist>;
using Samplers   = TypeList<NoSampler, GOSS>;

// All combos as one flat typelist of 4-typelists:
//   TypeList<TypeList<MSE, Depthwise, Hist, NoSampler>,
//            TypeList<MSE, Depthwise, Hist, GOSS>, ...>
using Combos = cartesian_product_t<Objectives, Growers, Splitters, Samplers>;

inline constexpr auto kTable = []{
    std::array<Entry, size_v<Combos>> out{};
    std::size_t i = 0;
    for_each_type<Combos>([&]<typename Combo>{
        using O  = type_at_t<0, Combo>;
        using G  = type_at_t<1, Combo>;
        using S  = type_at_t<2, Combo>;
        using Sa = type_at_t<3, Combo>;
        out[i++] = {
            {O::name, G::name, S::name, Sa::name},
            +[](Config const& cfg) -> std::unique_ptr<IBooster> {
                return std::make_unique<Booster<O, G, S, Sa>>(cfg);
            }
        };
    });
    return out;
}();

auto make_booster(Config const& cfg) -> std::unique_ptr<IBooster> {
    auto key = std::tuple{cfg.objective_name, cfg.grower_name,
                          cfg.splitter_name, cfg.sampler_name};
    return find(kTable, key)(cfg);
}
```

`cartesian_product_t<Ls...>` is a small typelist algorithm we
hand-write (~30 LOC of fold + concat) — same shape as Python's
`itertools.product` or Boost.MP11's `mp_product`. Hand-writing it is
deliberate: it's exactly the kind of fold-expression exercise the
project is showcasing, and it lives in `include/bonsai/typelist.h`
alongside `for_each_type`, `type_at_t`, `size_v`.

`for_each_type` is a fold-expression / pack-expansion helper; the
"loop" is mechanical compile-time expansion, not a runtime `for`.

Reading the call site: one flat iteration over the cartesian product
typelist, no nesting.

**Pros.**
- Genuinely flat at the call site. One lookup, one call.
- Cartesian product is materialized, so the cost is visible
  (`kTable.size()`).
- Easy to print "available combinations" for `--help` output.
- Easy to gate combinations: skip the cell when generating the table
  (e.g., `oblivious + leafwise` is invalid).
- Typelists are the single source of truth — adding `Logloss` to
  `Objectives` regenerates every combo using it automatically.

**Cons.**
- Requires an `IBooster` virtual base — `std::array` needs a single
  value type. **One vcall per `update_one_iter`**, zero inside it.
- Every cell instantiates `Booster<...>`, used or not. Compile-time
  cost scales multiplicatively with typelist sizes.
- `for_each_type` machinery + cartesian fold is moderate
  metaprogramming.

## Candidate B — nested registry callbacks (no erasure)

**Idea.** Stay fully static everywhere. The runtime → type bridge is
fundamentally callback-shaped given typelist registries: the only way
to "return a type" picked at runtime is to call back into a template.

```cpp
ObjRegistry::create(cfg.objective_name, [&]<Objective O>{
  GrRegistry::create(cfg.grower_name, [&]<TreeGrower G>{
    SpRegistry::create(cfg.splitter_name, [&]<SplitFinder S>{
      SaRegistry::create(cfg.sampler_name, [&]<Sampler Sa>{
        auto booster = BoosterBuilder<>{}
            .with_objective<O>().with_grower<G>()
            .with_splitter<S>().with_sampler<Sa>()
            .build(cfg);
        booster.train(data);
        booster.save(out_path);
        // entire training + save runs monomorphized
      });
    });
  });
});
```

**Pros.**
- Fully static — no virtual base anywhere. Zero vcalls in or out of
  the hot path.
- Only the combination the user actually picks gets instantiated.
  Compile-time cost scales with usage, not with config space.
- Easy to add a fifth component (Backend) without re-generating a
  five-tuple table.

**Cons.**
- Rightward drift at the call site. The nesting *is* the price of
  "runtime string → static type" without erasure; no language feature
  removes it (until reflection lands).
- The whole training run lives inside the innermost lambda
  (continuation-passing style). Restructuring `main()` around this is
  intrusive.
- Untested combinations get no compile coverage. Tests must enumerate
  the combos they care about explicitly.

## Candidate C — keep the nested form, dress it up

Worth naming so we can reject it explicitly. Naive nested lambdas with
the bodies extracted into named static functions to flatten the
*visual* nesting while keeping the structural nesting. Same
compile-time profile as B (only the picked combo instantiates), worse
ergonomics than B with the type-level builder, no advantages over A.

Reject unless A and B are both unworkable.

## A hybrid that doesn't work: flat table + generic callback

Tempting middle ground: flat `std::array` table whose entries take a
generic callback, so the typed `Booster<...>` never escapes. Doesn't
work — function pointers in a `std::array` cannot be generic over the
callback's type. Either (a) the callback type is fixed (e.g.,
`std::function<void(IBooster&)>`, which *is* the erasure A already
admits), or (b) the table becomes a `constexpr` search over the
typelists rather than a `std::array` lookup, at which point it
collapses back into Candidate B.

Recording this so we don't re-derive it later.

## Decision matrix

| Criterion | A: dispatch table | B: nested callbacks | C: dressed-up nested |
|---|---|---|---|
| Static *inside* hot path (`update_one_iter` body) | yes | yes | yes |
| Static *outside* hot path too | no (1 vcall per iter) | yes (zero vcalls) | yes (zero vcalls) |
| Flat call site | yes | no | no |
| Compile-time cost | full cartesian product | only used combos | only used combos |
| `--help` enumeration of combos | trivial | requires separate enumeration | same as B |
| Extending to 5 components | re-generate table | add one nesting level | add one nesting level |
| Gating invalid pairs | omit table entry | runtime check or concept | runtime check |
| Final boundary type | `unique_ptr<IBooster>` | typed `Booster<...>` (in callback) | typed `Booster<...>` |
| Implementation complexity | high (cartesian fold) | medium (callback chain) | low (just lambdas) |

## Where to draw the static/dynamic line

Two tenable boundaries; pick one before resolving A vs B.

1. **"Static inside `update_one_iter`."** The proposal's stated rule.
   `update_one_iter` is the boundary. One vcall per iter is dwarfed by
   the histogram pass inside it. **Compatible with A.**
2. **"Static everywhere after `make_booster` returns."** Stricter goal.
   Zero erasure anywhere. Forces continuation-passing at the boundary.
   **Requires B.**

These are both honest interpretations of "static in hot paths, dynamic
at config boundary" — the question is whether `update_one_iter` is part
of the hot path or part of the boundary.

## Other open sub-decisions

- **Where erasure happens, if at all.** Falls out of A vs B above.
- **`Backend` placement — deferred to `7-parallel.md`.** Dispatch is
  4D for now (Obj, Gr, Sp, Sa). When Backend lands, two reversible
  options stay open: (a) compose separately
  (`booster.train<Backend>(data)`), zero dispatch changes; or
  (b) promote to 5D by adding one typelist to `cartesian_product_t`,
  table regenerates mechanically.
- **Invalid combinations — resolved.** Pre-filter at typelist
  construction. See "Encoding compatibility" below.

## Encoding compatibility (invalid combinations)

Some four-tuples in the cartesian product may be incompatible (none
known in MVP; plausible future case: an oblivious grower paired with a
sampler that assumes per-leaf decisions). Three places this can be
caught:

| Where | How | Cost |
|---|---|---|
| Runtime | factory throws `ConfigError` | latest signal, simplest |
| Compile-time, per cell | `requires compatible<O,G,S,Sa>` on `Booster` or a `filter` over the cartesian product | extra predicate machinery, hard error on bad list |
| **Typelist construction** | **build several products over compatible sub-typelists, `concat` them** | **compatibility lives in plain typelist syntax — no new machinery** |

**Chosen: typelist construction.**

```cpp
// Hypothetical: oblivious growers incompatible with GOSS sampler.
using DepthwiseSamplers = TypeList<NoSampler, GOSS>;
using ObliviousSamplers = TypeList<NoSampler>;        // no GOSS

using DepthwiseCombos = cartesian_product_t<
    Objectives, TypeList<Depthwise>, Splitters, DepthwiseSamplers>;
using ObliviousCombos = cartesian_product_t<
    Objectives, TypeList<Oblivious>, Splitters, ObliviousSamplers>;

using Combos = concat_t<DepthwiseCombos, ObliviousCombos>;
```

Properties:
- Invalid combos never appear in the table — no runtime check, no
  predicate, no instantiation.
- Compatibility logic lives in *how typelists are constructed*, in
  plain typelist syntax. No new metaprogramming layer.
- Adding a combo = add to the right sub-product or split a new one.
- `concat_t<L1, L2, ...>` is ~5 LOC, ships alongside
  `cartesian_product_t` in `include/bonsai/typelist.h`.

**Cost.** With many incompatibility rules this fragments. For bonsai's
expected scale (handful of combos with rare exclusions) it's fine. If
fragmentation becomes a problem, fall back to a `filter` predicate
over a single product.

### Why this is an architectural win, not just an ergonomics choice

The reference libraries can't do this and instead carry hand-written
runtime validators:

- **xgboost** — `LearnerImpl::Configure()` validates strings after
  parsing; bad combo throws at training start.
- **LightGBM** — `Config::CheckParamConflict()` is one big function
  with sequential `if` checks (~hundreds of lines), sometimes
  auto-correcting with a warning.
- **CatBoost** — `TCatBoostOptions::Validate()` plus per-component
  validators; same pattern, more layers.

They can't do better because their plugin sets are open across
third-party binaries — no closed cartesian product to enumerate, and
components are `unique_ptr<Base>` with no type info to constrain.

bonsai's closed plugin set + concept contracts let invalid combos be
**proven absent from the binary**. This is the kind of architectural
improvement §3.4 promises.

## Sketch of the registry that any candidate uses

Independent of A vs B, the per-component registry shape is the same:

```cpp
template <typename Concept, typename Config, typename... Impls>
    requires (Concept<Impls> && ...)
struct Registry {
    template <typename Callback>
    static auto create(std::string_view name, Config const& cfg, Callback&& cb) {
        // Fold over Impls: if Impl::name == name, instantiate Impl
        // and pass it to cb. Throws ConfigError if no match.
    }
    static constexpr auto names() {
        return std::array{Impls::name...};
    }
};
```

`Concept<Impls>` enforces the contract at typelist-construction time —
a misnamed or incomplete impl fails to compile with a concept
diagnostic, not a runtime "unknown name."

## Reflection branch (optional, showcase only)

C++26 `P2996` static reflection lets the typelist itself be derived
from a namespace or attribute scan rather than hand-listed. The
dispatch shape doesn't change; only how the typelist is populated. If
included, lives behind a `BONSAI_REFLECTION` build flag, parallel to
the typelist version, not replacing it.

## What's not here

- The `Booster` class shape itself — see `5-booster.md`.
- `ParallelBackend` integration — see `7-parallel.md`.
- TOML parsing and `Config` schema — see `8-config.md`.

## Cross-references

- `proposal.md` §3.4 — the open question this doc closes.
- `proposal.md` §3.5 — metaprogramming techniques used.
- `decisions.md` — closed decisions on no-macros, no-aliases,
  static-in-hot-path.
