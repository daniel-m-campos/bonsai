# 5. Booster

> **Status:** Ratified by decision 27.

## What this doc settles

- The `Booster<Obj, Gr, Sp, Sa>` class shape: members, ctor, the training loop, `update_one_iter` semantics.
- Where the **score accumulator** lives, how it's initialized (initial bias), and how `Objective`'s link inverse plugs into the predict path.
- Where the **sampler** runs in each iteration (decision 12).
- Where **sample weights** are applied (decision 25).
- The `IBooster` interface: the minimal virtual surface used at the config boundary (decision 26).
- Predict / save / load surface.

Out of scope:
- Dispatch / registry shape: done in `6-dispatch.md` + decision 26.
- ParallelBackend integration: `7-parallel.md` (deferred).
- TOML / Config schema: `8-config.md`.

## Why a class template, not a base class

Decision 26 fixed dispatch as a flat table over the cartesian product of (Obj, Gr, Sp, Sa). The table values are `unique_ptr<IBooster>`-returning factories that monomorphize `Booster<O,G,S,Sa>` per cell. So:

- `IBooster` exists *only* as the boundary erasure type. Its surface is the minimal set of operations the CLI calls after construction (`update_one_iter`, `predict`, `save`, `load`, `n_iters`, etc.). No virtual training internals.
- `Booster<O,G,S,Sa>` is the real class. Every member of the inner loop is a direct, inlinable call against `O`, `Gr`, `Sp`, `Sa`.
- Per decision 26: one vcall per `update_one_iter` (the `IBooster`-level call from CLI), zero vcalls inside it.

## Class shape (sketch)

```cpp
class IBooster {
public:
    virtual ~IBooster() = default;
    virtual void update_one_iter(Dataset const& train) = 0;
    virtual float eval(Dataset const& data) const = 0;
    virtual void predict(Dataset const& data, std::span<float> out) const = 0;
    virtual void save(std::filesystem::path const&) const = 0;
    virtual std::size_t n_iters() const = 0;
    // ... minimal CLI-facing surface ...
};

template <Objective Obj, TreeGrower Gr, SplitFinder Sp, Sampler Sa>
class Booster final : public IBooster {
public:
    Booster(Config const& cfg);
    // ... overrides ...

private:
    Config         cfg_;
    Obj            objective_;     // see "Objective: instance or static?"
    Gr             grower_;
    Sp             split_finder_;
    Sa             sampler_;

    typename Gr::Tree                    /* one tree per iter */;
    std::vector<typename Gr::Tree> trees_;

    // score accumulator, raw scores, length n_train_rows
    std::vector<float> scores_;
    // per-iter scratch: grad, hess (length n_train_rows)
    std::vector<float> grad_;
    std::vector<float> hess_;
    // initial bias (decision 22 — booster owns it)
    float init_score_;
};
```

Resolved:
- **`Obj` is purely-static.** `4-objective.md` pins `T::compute` / `T::eval` as static functions; no member stored. If a future objective needs config, revisit.
- **Booster borrows the `Dataset`.** Lifetime sits with the CLI; `Booster::update_one_iter` takes a `Dataset const&`. Saved model is `BinMappers` + trees + init_score, no Dataset.
- **Ctor takes only `Config`; first `update_one_iter` does data- dependent init.** Sizing `scores_`/`grad_`/`hess_` to `n_train_rows` and computing `init_score_` from labels both need a `Dataset`, but pushing them into the ctor would force every dispatch-table factory (see `6-dispatch.md`) to thread a `Dataset` through, and would give `load_booster` no symmetric construction path (loaded boosters have no training data; they restore `init_score_` and `trees_` from disk). Cost is one `if (scores_.empty())` guard at the top of `update_one_iter`; predicted-taken on every iter after the first. `load_booster` writes the state directly post-ctor and the guard short-circuits.
- **Booster does not own `BinMappers`.** Decision 3: trees store raw float thresholds, predict path doesn't need `BinMappers`.

## `update_one_iter` semantics

One iteration = one tree appended. Order of operations:

```
0. if scores_.empty():               # first iter only
       scores_.assign(n_rows, 0.0F)
       grad_.resize(n_rows); hess_.resize(n_rows)
       init_score_ = compute_init_score(Obj, train)   # see below
       scores_.assign(n_rows, init_score_)
   # data-dependent init, deferred from ctor.

1. objective.compute(scores_, labels, grad_, hess_)
   # writes raw-score grad/hess (decision 24, overwritten not accumulated)

2. if dataset.weights non-empty: grad_ *= w; hess_ *= w
   # booster-side multiply (decision 25)

3. row_indices = sampler.sample(grad_, hess_, rng)
   # sampler runs here (decision 12); identity-sample for NoSampler

4. tree = grower.grow(dataset, grad_, hess_, row_indices, split_finder, cfg)
   # one tree, statically-typed `typename Gr::Tree`

5. scores_ += learning_rate * tree.predict_train(...)
   # additive raw-score update; learning_rate lives in cfg, applied here
   # (decision per 3-tree.md: tree leaf values are raw, not pre-scaled)

6. trees_.push_back(std::move(tree))
```

Notes:
- Steps 1–2 always touch the full row set; sampling (step 3) only affects what the **grower** sees. Sampling-first would save the
  grad/hess pass on dropped rows, but (a) GOSS samples *on* `|grad|`
  and needs it computed first, (b) the grad/hess pass is ~5–10% of `update_one_iter` while histogram building (which sampling actually targets) is 60–80%, (c) keeping order grad-then-sample avoids a "does this sampler need grad?" flag on the `Sampler` concept. xgb + lgbm do the same.
- Step 5: applying `learning_rate` at score-update time (not at leaf-write time) keeps `Tree::predict` honest at predict time. Saved trees carry raw leaf values, the booster reapplies the rate. This matches xgb's "shrinkage = booster concern" model.
- Step 5 calls `tree.predict(train_rows)`, re-walking the tree per training row. Simple, clean, reuses the predict path. Cost is one extra `O(n_rows × depth)` walk per iter; at MVP scale (depth ≤ 8, n_rows ~500K) ~4M comparisons, small compared to histogram building.
- **Phase 2 optimization (deferred until benchmarking justifies it).** All three reference libraries (xgb `position_`, lgbm `data_partition_`, catboost `leaf_indices`) already have the row → leaf mapping in the grower as a byproduct of growing. They reuse it for the score update instead of calling predict. If profiling shows step 5 as a measurable fraction of `update_one_iter`, change grower return from `Tree` to `(Tree, std::vector<float> train_leaf_values)` and replace step 5 with a flat add. Pure additive change to the grower→booster boundary; `Tree` stays clean.

## Initial score (bias)

Per decision 22, the booster owns the initial bias. Three sources, in priority order:

1. `cfg.init_score` if user-set.
2. Else, an objective-appropriate default computed from labels:
   - **MSE**: mean(labels).
   - **Logloss**: `log(p / (1-p))` where `p = mean(labels)`, clamped to avoid `log(0)`.
3. Else (e.g., already-fit booster being continued): the value stored in the saved model.

Computation lives in `Booster::compute_init_score(Obj, Dataset)` (static-dispatch on `Obj` via overload or `if constexpr`). **Not** a member of `Objective`: that's the rejection in decision 22.

Called from the lazy-init path (step 0 of `update_one_iter`), not the ctor: the ctor takes only `Config` (see "Class shape" resolved bullets), and `load_booster` writes `init_score_` directly from disk without going through this function.

## Predict path

Two cases:

1. **Raw-score predict**: sums tree predictions plus `init_score`, applies `learning_rate` per tree. Returns raw scores.
2. **User-facing predict**: calls raw-score predict, then applies the objective's link inverse:
   - **MSE**: identity.
   - **Logloss**: sigmoid.

Inverse link is computed inline in `Booster::predict` via `if constexpr` on `std::is_same_v<Obj, LogLossObjective>`, or via a tiny `transform` helper specialized per objective. Either way it lives in the booster (decision 22), not in `Objective::compute` (decision 24).

## Save / load

Save and load are **I/O concerns, not booster methods.** Both are free functions in a separate I/O module:

```cpp
void save_booster(IBooster const&, std::string const& path,
                  BinMappers const&, Config const&);
struct LoadedBooster { std::unique_ptr<IBooster> booster;
                       BinMappers mappers; Config cfg; };
LoadedBooster load_booster(std::string const& path);
```

Rationale: `load` needs the objective/grower/splitter/sampler types *before* it has a `Booster` to dispatch on. It reads the names from disk, then calls into the same registry path `make_booster` uses. That's structurally the dispatch boundary, not a member function. `save` is the symmetric counterpart and lives next to `load`.

What gets written: the full training `Config` (six sub-structs, decision 29) under `"config"`, `BinMappers` (diagnostic only per decision 3), `init_score`, trees in their tree-specific serialization. On-disk format is msgpack-encoded JSON; the JSON shape comes from `NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE` macros in `src/io/model.cpp` (decision 29 for the why-not-codec rationale).

`IBooster` exposes whatever minimal accessors `save_booster` needs (e.g., `for_each_tree`, `init_score()`, `learning_rate()`). These are the same accessors the CLI uses for inspection commands, not save- specific surface.

## Sampler

Folded into this doc; spin out into `5b-sampler.md` if it grows. Per decision 12, the sampler is the **booster's** responsibility, not the grower's. Step 3 of `update_one_iter` is where it runs.

### `Sampler` concept (Phase 1)

```cpp
template <typename T>
concept Sampler = requires(floats_view grad, floats_view hess,
                           std::mt19937& rng,
                           std::span<size_t> out_indices) {
    { T::sample(grad, hess, rng, out_indices) } -> std::same_as<std::size_t>;
};
```

Static members, same shape as `Objective` and `SplitFinder`. Returns the count of selected indices (written into the head of `out_indices`); the buffer itself is owned by the booster and reused across iters.

### `NoSampler` (Phase 1)

Identity sampler: writes `0..n_rows-1`, returns `n_rows`. Compiles out trivially under inlining; the booster's call site is the same shape regardless of sampler.

### `BernoulliSampler`, `GossSampler` (shipped)

Both landed (decisions 27, 34). GOSS forced one concept change: `sample()` takes **mutable** grad/hess spans so the sampler can amplify the retained small-gradient rows in place. The booster recomputes both from the objective every iteration, so the scaling never outlives its tree. GOSS benchmarking also exposed the out-of-bag stale-score bug (decision 34): every row must receive the tree's contribution in `GrowResult.values`, sampled or not (`route_unsampled`).

### Instance samplers and objectives

Samplers are Config-constructed instances (subsample rate, GOSS rates). Objectives followed in decision 35 for the same reason (huber_delta, quantile_alpha); the booster holds `objective_` and calls through it. Statics still satisfy the instance-call syntax, so MSE/LogLoss kept their static methods. The booster also implements DART (decision 35: per-iteration tree dropout with bin-space routing) and the incremental early-stopping hooks (`score_base` / `accumulate_last_tree` / `truncate`, decision 34), plus `feature_importance` (decision 37).

## Threading touchpoints (foreshadowing `7-parallel.md`)

`update_one_iter` calls `objective.compute`, the weighted multiply, the grower (which calls `split_finder` and `sampler`), and the score accumulator update. Each of those is a candidate parallel region. Once `ParallelBackend` lands:

- `objective.compute` and the weighted multiply: trivially parallel per row.
- Grower / split finder: already designed around per-thread local histograms (see `2-histogram.md`).
- Score accumulator update: parallel per row.

`Backend` is **not** part of the dispatch cartesian product right now (decision 26 + `6-dispatch.md`). When it lands, options stay open: template `Booster<...>::update_one_iter<Backend>(...)`, or promote `Backend` to the cartesian product.

## What's not here

- `Objective` impls: `4-objective.md`.
- `Tree`, growers, splitters: `3-tree.md`.
- Sampler concept + `NoSampler` impl: TBD doc; sketched in decision 12.
- Dispatch / registry: `6-dispatch.md`.
- Threading: `7-parallel.md`.
- Model file format: TBD.
- CLI plumbing: `9-cli.md`.

## Resolved choices in this doc

1. `Obj` is purely-static; no instance member.
2. Step 5 re-predicts via `tree.predict(train_rows)`. Caching the row → leaf mapping deferred to Phase 2 (after benchmarking).
3. `learning_rate` applied at score-update time. Saved trees carry raw leaf values; booster reapplies the rate.
4. `save` / `load` are free functions in an I/O module, not `IBooster` methods.
5. `Sampler` concept folded into this doc (above). Spin out into `5b-sampler.md` only if it grows.

## Open for the user's call

- Ratify everything above into a single `decisions.md` entry (likely
  27) once the user is happy with the doc.
- Section header / sketch refinements: wording, member ordering, whether `IBooster`'s minimal surface is exactly what's listed.

## Cross-references

- Decisions 12, 22, 23, 24, 25, 26.
- `proposal.md` §3.4 (static/dynamic boundary).
- `3-tree.md`, `4-objective.md`, `6-dispatch.md`.
