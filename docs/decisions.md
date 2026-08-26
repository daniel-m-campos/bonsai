# Decisions

This log is the historical engineering record, kept for reference and for agents working in the codebase. The curated design pages live in the [Design](design/determinism.md) section.

Append-only log. Order = decision order. Caveman style. New entries at bottom.

---

## 1. Binning: quantile, with low-cardinality fallback

`BinMapper::fit` per feature.

- Equal-frequency cuts at `k/max_bin`-th quantiles. `max_bin = 255` default (`uint8` indices).
- If `n_distinct < max_bin`: one cut between each pair of consecutive distinct values. Bucket count = `n_distinct`.
- Dedupe cut collisions (sentinel values like `0.0`). Actual count `<= max_bin`, never exact.
- Sampling from the start. Default sample 200K rows uniform random, fixed seed. Configurable. If column has `<= sample_size` rows, use full column.
- Bin 0 reserved for missing. NaN + user-configured sentinel short-circuit to bin 0. Real values bins `1..n_bins-1`. Quantile skips NaNs.
- `BinMapper` serializable. Round-trip through model file. Predict on new data reuses train boundaries exact.

Rejected: equal-width (skew kills it). Quantile sketch (overkill, swap in later). xgb per-node default direction (complicates split scoring).

Knock-on: bin count varies per feature, histogram reads `n_bins[fid]`. Bin 0 special, split scoring skips it for real-valued cuts. `BinMapper` ownership vs `Dataset` is next decision.

Defer: `min_data_in_bin` knob.

---

## 2. `BinMapper` independent of `Dataset`. Two-stage API.

```
auto mappers = BinMappers::fit(train_source, cfg);
auto train   = Dataset::bin(train_source, mappers, cfg);
auto val     = Dataset::bin(val_source,   mappers, cfg);
auto test    = Dataset::bin(test_source,  mappers, cfg);
```

- `BinMappers` is `std::vector<BinMapper>` plus minimal wrapper (count, serialize). Built once on train, immutable thereafter.
- `Dataset::bin` is pure transform: takes source + mappers, returns binned column-major storage. No "training Dataset" vs "val Dataset" distinction.
- Model file serializes `BinMappers`. Predict-time `Dataset` builds fresh from them.

Rejected: lgbm-style `Dataset::from_csv(..., reference=train_ds)`. Couples mapper lifetime to a Dataset, awkward serialization, "training Dataset" becomes special.

Knock-on: train path is two calls instead of one. Trivial. `bin` is single-pass; `fit` does its own sampling + sort internally.

---

## 3. Trees store raw float thresholds, not bin indices

Tree node split = `(feature_id, threshold: float)`. Predict reads raw `float` from input row, compares directly. No binning at predict time.

`TreeGrower` finds the best split as `(fid, bin_idx)` during training, then converts to `threshold = cuts[bin_idx]` when writing the node. Conversion is one lookup per finalized split, free.

xgb + catboost do this. lgbm stores bin indices in tree nodes (and re-bins at predict, which is why lgbm forces the reference-Dataset dance).

Knock-on:
- Predict path doesn't need `BinMappers`. Single tree walk over raw floats.
- Model file: trees serialize directly, `BinMappers` optional in model file (kept for diagnostics + reproducibility, not load-bearing for predict).
- Training-time histogram code unchanged: still bins, still works on bin indices internally.
- Float threshold means tree comparison is `<` on float, not `<=` on int. Watch for off-by-one when comparing parity vs lgbm (different convention).

---

## 4. `Dataset` storage layout

Column-major. Per-feature `std::vector<uint16_t>` (uniform width). Labels + weights owned by `Dataset` (weights empty if uniform). `BinMappers` held by value (not `shared_ptr`); ~30KB copy is trivial, no shared mutable state.

```cpp
class Dataset {
    std::vector<std::vector<uint16_t>> features_;
    std::vector<float>                  labels_, weights_;
    BinMappers                          mappers_;
    std::vector<bool>                   is_categorical_;  // Phase 4 placeholder
    // n_rows, n_features
};
```

Public API: `n_rows()`, `n_features()`, `labels()`, `weights()`, `mappers()`, `n_bins(fid)`, `is_categorical(fid)`, `feature_bins(fid) -> span<bin_id_t const>`.

Rejected: `std::variant<vector<uint8_t>, vector<uint16_t>>` per feature to save ~50% on binned column memory. Saves ~45MB on YearPredictionMSD, ~308MB on Higgs. Neither pressure-tests modern hardware. Cost was variant dispatch complexity at every column scan via a `visit_column` wrapper. Rejected for MVP; reversible if a future dataset makes memory the bottleneck.

Group columns (ranking) deferred (non-goal).

---

## 5. *(reserved)*

Originally `visit_column` for variant-aware column access. Dropped when decision 4 collapsed to uniform `uint16_t` storage. Renumbering decisions breaks references; left as a placeholder.

---

## 6. Readers: free function per format, returning `Dataset`

```cpp
Dataset read_csv    (const std::string& path, const DataConfig&, const BinMappers&);
Dataset read_parquet(const std::string& path, const DataConfig&, const BinMappers&);  // Phase 4+
Dataset read_libsvm (const std::string& path, const DataConfig&, const BinMappers&);  // later

BinMappers fit_from_csv(const std::string& path, const Config&);
```

CLI dispatches on `cfg.data.format` string: `if "csv" call read_csv else if "parquet" ...`. Each reader its own translation unit. Adding a new format = new file + new branch in CLI dispatch.

No `Reader` concept, no abstract base, no template plumbing. File loading is once-per-program, not hot-path; concepts buy nothing here. Internal shared helper per reader (`CsvReader::columns(path, cfg) -> ColumnBatch`) keeps `read_csv` and `fit_from_csv` from duplicating logic.

**Phase 1 ships CSV only**, hand-rolled (~50 LOC). Numeric-only is enough for YearPredictionMSD. No Arrow, no parquet.

**Phase 4+: Arrow optional**, gated by `BONSAI_PARQUET=ON` CMake flag. Arrow handles CSV multi-threaded + parquet + feather + IPC. Heavier dep (transitive thrift/snappy), so kept optional. Arrow is the *reader* layer only. `Dataset` (binned storage + `BinMappers` + labels/weights) is bonsai's; Arrow's `Table` is raw column data we'd copy or borrow from.

Rejected: `Reader` concept + `Reader auto&` template params for `fit` and `bin`. Over-engineered for once-per-program file loading. Free functions are the simpler shape.

---

## 7. Determinism contract: fixed thread count, not cross-thread

Same seed + same data + **same thread count** → same model bytes. Different thread counts: predictions within numerical tolerance, but bytes may differ.

What this rules in:
- Per-thread local histograms (no atomic FP adds: those are bit-unstable even at fixed thread count).
- Deterministic chunking (e.g., OpenMP `schedule(static)`).
- `random_seed` carries through samplers / shufflers.

What this rules out (relative to earlier framing):
- Promising cross-thread bit-exactness. The earlier draft demanded fixed-order merge (`tid` outer, bin inner) so the per-thread reduction shape didn't depend on thread count. Dropped: costs design constraints on `ParallelBackend` (must expose ordered reduction primitive) and forecloses `OpenMP reduction(+:...)` and `std::execution` reduce shapes.

Field check: XGBoost and CatBoost don't promise cross-thread determinism. LightGBM offers it behind `deterministic=true` +
`force_col_wise|row_wise`, and its own maintainers describe the
guarantee as fragile (RFC #6731). The pragmatic, industry-standard contract is "thread count is part of the reproducibility input."

Test contract:
- `test_determinism_fixed_threads`: two runs at `n_threads=k` for `k ∈ {1, 4, 8}` produce identical model files. Required to pass.
- `test_determinism_cross_threads`: predictions across different thread counts agree to numerical tolerance (e.g., max abs diff
  < 1e-5 on YearPredictionMSD). Required to pass.

Knock-on:
- `ParallelBackend` does not need to expose "ordered reduction" as a primitive. `parallel_for` + thread-local accumulators is enough.
- The histogram's parallel-build description in [`architecture/2-histogram.md`](https://github.com/daniel-m-campos/bonsai/blob/06fab232a3b156a7c9f155fbbdf41fe14d45f1af/docs/architecture/2-histogram.md) §"Parallel construction" reflects this: no fixed-tid-order requirement.
- Atomic FP adds remain forbidden, but for the bit-stability-at-fixed-N reason, not the cross-N reason.

---

## 8. Two trees in Phase 1: depth-wise + oblivious

`DepthwiseGrower` → `DenseTree` and `ObliviousGrower` → `ObliviousTree` both ship in Phase 1. Proposal puts oblivious in Phase 4; pulled forward to force the `Tree` concept and `TreeGrower::Tree` associated type to be honest from day one.

The two tree types have structurally different on-disk shapes (flat node array vs per-level splits + leaf table) and structurally different predict kernels (walk-until-leaf vs fixed-depth branchless gather). With only depth-wise shipping, the second-tree-type machinery would be aspirational. Same rationale as logloss alongside MSE in Phase 1 (proposal §1).

Cost: one extra grower + one extra tree type of spine code, plus a second parity target (depth-wise vs xgboost/LightGBM, oblivious vs CatBoost). Both targets share YearPredictionMSD.

---

## 9. `Tree` is a concept; minimum surface = predict ×2 + diagnostics

```cpp
template <typename T>
concept Tree = requires(T const t,
                         std::span<float const> row,
                         std::span<float const> rows, size_t n_features,
                         std::span<float> out) {
    { t.predict(row) }                       -> std::same_as<float>;
    { t.predict(rows, n_features, out) }     -> std::same_as<void>;
    { t.n_leaves() }                         -> std::convertible_to<size_t>;
    { t.depth() }                            -> std::convertible_to<size_t>;
};
```

Concept, not abstract base. `Booster<Gr, ...>::trees_` is `std::vector<typename Gr::Tree>`, monomorphized. No vtable on predict.

Two `predict` overloads under one name (single-row returns float; batch fills `out`). Disambiguation by arity. Row-major batch input, matches xgb / lgbm. CatBoost's column-major fast path is motivated by predict- time rebinarization, which we don't do (decision 3: float thresholds). Oblivious can transpose-on-demand internally if profiling justifies.

Rejected: `leaf_index(row)` (DenseTree and ObliviousTree leaf-index spaces aren't unified; defer to Phase 4 if SHAP / leaf-output predict is wanted); `walk(visitor)` (no shared node shape between the two impls); serialization on the concept (lives in `bonsai::io` per decision 6).

---

## 10. Shrinkage is baked into leaf values at tree construction

Grower receives `learning_rate` as a constructor argument (in `TreeConfig`) and writes `lr · -G/(H + λ_l2)` into leaves. Trees are pure functions of input rows; predict has no learning-rate knowledge. Matches xgboost, LightGBM.

Rejected: per-iteration `learning_rate` argument to `grow()` (Phase 1 doesn't need decay schedules; non-breaking to add later as a `grow` overload).

Knock-on: `Tree` concept doesn't carry a `set_shrinkage` mutator; trees are immutable post-construction.

---

## 11. `ObliviousTree`: per-level `default_left`, not per-node

Every node at level `d` of an oblivious tree shares the same `(feature_id, threshold, default_left)` triple: that's the symmetric- tree contract. Relaxing `default_left` to per-node-at-level recovers a small accuracy edge in heavily-missing data but breaks the branchless predict kernel and isn't what CatBoost does.

`LevelSplit { uint32_t feature_id; float threshold; bool default_left; }` × depth. Leaf table of size `2^depth`.

When a level's feature has no missing rows in training data, `default_left` is don't-care; the splitter records whichever orientation scored higher (arbitrary if no missing rows existed) and predict honors it without thinking.

---

## 12. Sampler is the booster's responsibility, not the grower's

`grow(ds, grad, hess, row_indices)`. The grower receives sampled row indices; it doesn't know what sampler produced them. "Use all rows" is just `row_indices = 0..n_rows-1`.

Keeps `TreeGrower` concept narrow. Sampler swappable via `Booster<Obj, Gr, Sa, Backend>` independent of grower choice. Determinism contract testable on the grower without re-wiring the sampler.

Rejected: grower owns sampler as a member (lgbm-style). Couples grower template to sampler template; bigger cartesian product on `Booster` instantiations for no compositional gain.

---

## 13. Branchless NaN routing in predict

```cpp
bool is_nan  = std::isnan(v);
bool less    = !is_nan && (v < threshold);
bool go_left = less | (is_nan & default_left);
```

Compiles to mask / select on x86-64 and ARM. Preserves vectorizability for `ObliviousTree`'s batched predict; also keeps `DenseTree`'s single-row predict tight.

Rejected: branchful `if (isnan(v)) ... else ...` (loses oblivious's SIMD story); pre-cleaning predict input (caller can't know per-feature default direction; rules itself out).

Knock-on: predict-time sentinels (e.g. `-999`) declared in `BinMapperConfig` are *not* honored at predict, only `std::isnan`. Caller contract: convert sentinels to NaN before predict. Matches xgb / lgbm.

---

## 14. Splitter is a template parameter on the grower; one `SplitFinder` concept

```cpp
template <SplitFinder Sp = HistogramSplitFinder> class DepthwiseGrower;
template <SplitFinder Sp = HistogramSplitFinder> class ObliviousGrower;
```

Static dispatch through to the splitter; inlined into `grow()`. Default makes the common case ergonomic; explicit `Sp` makes the extension API real.

```cpp
struct SplitCandidate {
    uint32_t feature_id;
    uint16_t bin_idx;        // grower converts to threshold via cuts[bin_idx]
    bool     default_left;
    double   gain;
    bool     valid;
};

template <typename T>
concept SplitFinder = requires(T const f,
                                std::span<Histogram const> hists,
                                Dataset const& ds,
                                double sum_grad,
                                double sum_hess) {
    { f.find(hists, ds, sum_grad, sum_hess) }
        -> std::same_as<SplitCandidate>;
};
```

Both growers consume the same `vector<Histogram>` shape and produce the same `SplitCandidate`. Depth-wise calls `find` once per frontier node with that node's histograms; oblivious calls `find` once per level with the folded level histograms. The splitter doesn't know or care whether its input is per-node or level-pooled.

Earlier draft of this entry split this into `PerNodeSplitFinder` / `LevelSplitFinder` to make mismatched grower/splitter pairs a compile error. Rejected: histogram-based scoring collapses the two signatures to the same shape, so the "compile-time rejection" was illusory. Phase 4 splitters that don't fit this shape (e.g. an exact splitter scanning raw rows) earn their own concept when written.

Rejected: type-erased `unique_ptr<SplitFinder>` member (the dynamic- dispatch shape we explicitly chose against in proposal §3.4).

---

## 15. Splitter returns one best candidate per call

`find(...)` returns a single `SplitCandidate`. `valid = false` if no positive-gain split exists or no candidate clears `min_gain_to_split`.

Rejected: returning all per-feature candidates and letting the grower pick max. No caller wants this; flexibility nobody's asking for.

---

## 16. Splitter tie-break: lowest `fid`, then lowest `bin_idx`

When two candidates have equal gain (within bit-exact equality, not tolerance), prefer the one with the lower `feature_id`; if those tie, prefer the lower `bin_idx`. Stable, deterministic at fixed thread count. Matches lgbm's tie-break order (xgb's is implementation- dependent in their `hist` updater).

Knock-on: same convention applies to the smaller-sibling choice in the subtraction-trick wiring: when `n_left == n_right`, left wins.

---

## 17. Partitioning: per-node row-index lists (strategy A)

Each live `FrontierNode` carries its own `std::vector<uint32_t>` of row indices. At root: one list with the booster-supplied `row_indices`. At each split: partition the parent's list into `(left_rows, right_rows)`, replace parent in frontier with two children carrying the new lists.

Both growers use this strategy (oblivious folds per-node histograms into level histograms at scoring time; ~1.5% overhead on YearPredictionMSD-scale data).

Rejected: single `row_to_node` array of length `n_rows` rebinned on each split (xgb's `hist` updater shape; lgbm's voting parallel mode). Beats per-node lists on cache locality at very shallow depth, loses at typical `max_depth = 6`. More importantly: the subtraction trick wires naturally onto per-node histograms. A single rebinned position vector either fights subtraction (the all-live-nodes-in-one-pass kernel doesn't know to skip the larger sibling) or imports per-node branching back into the kernel.

Knock-on: oblivious grower needs **per-parent gain summation** across the frontier: for each candidate `(feature, bin)`, sum `score(left, λ) + score(right, λ) − score(parent, λ)` over every parent (CatBoost's symmetric-tree gain; see [decision 30](#30-obliviousgrower-fold-then-score-was-wrong-revert-and-re-spec)).
Same `O(n_features · n_bins · |frontier|)` order as a fold; the
difference is in *what* is accumulated (gain, not histogram cells). An earlier version of this knock-on said "needs a fold step (`level_hists = sum_per_feature(...)`) before split scoring"; that was wrong because `score(g, h)` is non-additive; corrected 2026-05-22.

---

## 18. Frontier holds histograms inline (no histogram pool)

`FrontierNode { rows, sum_grad, sum_hess, hists }`. `vector<Histogram>` per node, sized `n_features`. The grower carries `std::vector<FrontierNode> frontier` and rotates it level-by-level.

Lists and histograms are local to `grow()`; the grower is stateless across calls (one boosting iteration → one allocation cycle). Matches lgbm's `SerialTreeLearner`.

Rejected: histogram pool with slot-id indirection (xgb's `hist` updater). Recycles allocations across the tree, but Phase 1 isn't allocation-bound. The pool refactor is contained if profiling later shows the allocator dominating.

---

## 19. Subtraction trick from day one, both growers

Build smaller child by row-scan; derive larger by `larger_hist = parent_hist - smaller_hist`. Halves histogram-build work across the tree (per [`2-histogram.md`](https://github.com/daniel-m-campos/bonsai/blob/06fab232a3b156a7c9f155fbbdf41fe14d45f1af/docs/architecture/2-histogram.md) §"Why subtraction halves it"). Implemented from day one because retrofitting means restructuring the grower's per-node memory.

Per-parent protocol:
1. Splitter scores `parent.hists`; commit candidate.
2. Partition `parent.rows` into `(left_rows, right_rows)`.
3. Pick smaller (left wins ties).
4. Build smaller's hists by row-scan.
5. Derive larger's hists by `parent_hist - smaller_hist`. `Histogram` carries its own `(total_grad, total_hess)` so `operator-=` subtracts cells and totals together.
6. Push left, right into new frontier in left-then-right order (frontier order is structural, independent of build order).
7. Parent's hists released when parent goes out of scope.

For oblivious: same per-parent protocol, run once per parent at each level. Cross-level subtraction (deriving level `d+1`'s level histogram from level `d`'s) doesn't help: different levels score on different features, so per-feature histograms for level `d+1` have to be built regardless.

---

## 20. Phase 1 regularization knobs

`TreeConfig` fields:

| Knob | Default | Meaning |
|---|---|---|
| `max_depth`               | 6     | Hard cap on tree depth. |
| `min_data_in_leaf`        | 20    | Node row-count floor (and child row-count floor before splitting). |
| `min_sum_hessian_in_leaf` | 1e-3  | Effective row-count floor under non-MSE objectives. |
| `lambda_l2`               | 1.0   | L2 reg on leaf weights, in gain formula and leaf value. |
| `min_gain_to_split`       | 0.0   | Minimum gain to accept a candidate. |

Validated in grower constructors; `ConfigError` with key path on bad values. Section is `[tree]` in TOML.

Rejected for Phase 1: `max_leaves` (leaf-wise concept; depth-wise's natural cap is `max_depth`, oblivious's leaf count is `2^depth` exactly).

Leaf value: `leaf_value = learning_rate · -G / (H + lambda_l2)`, applied at finalization (decision 10).

## 21. `Objective` is a concept; static methods, no instance state

Matches the `SplitFinder` shape (decision 14). Two static functions required: `compute(preds, labels, grad, hess)` writes per-row gradients and hessians; `eval(preds, labels)` returns a scalar mean loss. Dispatch is at the `Booster<Gr, Obj, ...>` template parameter, fixed at compile time. No vtable; no shared mutable state across calls.

Rejected: virtual base class (loses compile-time dispatch); concept-with-instance-methods (no Phase 1 objective needs instance state).

## 22. `Objective` does **not** own initial score or link inverse

The first-tree bias prediction (mean for MSE, log-odds for logloss) and the predict-time link inverse (sigmoid for logloss) live in the booster, not the objective. Rationale: both are score-accumulator concerns. The booster maintains the running raw-score prediction array, decides whether the bias comes from config or labels, and is the sole owner of the predict path. Pushing them into `Objective` would either force every objective to expose a `transform` and an `initial_score` it might not need (e.g. MSE: `transform` is identity), or invite a fragmenting set of optional methods on the concept.

See `5-booster.md` (TBD) for where these land.

## 23. Phase 1 objectives: MSE + binary logloss, single-output

Two MVP impls satisfy the `Objective` concept: `MSEObjective` (regression) and `LogLossObjective` (binary classification). Both consume 1D per-row `floats_view` for `preds` and `labels`, write 1D `floats_out` for `grad` and `hess`.

Rejected for Phase 1:

- **Multi-class / softmax.** K-output extension; touches `Objective`, `Booster`, `Tree` (K-output leaves). Phase 4.
- **Quantile, Huber, Tweedie, Cox.** Out of scope; satisfy the concept when added.
- **Custom user objectives.** No registry needed: anyone satisfying the concept can drop in as a `Booster` template parameter.

## 24. `compute` writes raw-score grad/hess; output is overwritten, not accumulated

`Objective::compute` writes `grad` and `hess` outright; callers don't zero buffers first. `preds` are raw scores throughout (logloss does **not** apply sigmoid inside `compute` or `eval`); the booster keeps an additive raw-score accumulator across iterations and applies the link only at the outermost predict call. Matches xgboost / LightGBM; keeps boosting math additive.

`hess` is always non-negative (MSE: 1; logloss: `p·(1−p) ∈ (0, 0.25]`); the splitter's `min_child_hess` (decision 20) catches near-zero hessians before they propagate.

## 25. Sample weights are applied by the booster, not the objective

When `Dataset.weights` is non-empty, the booster multiplies the `grad` and `hess` buffers by the weight vector immediately after `Objective::compute` returns and before handing them to the grower. Keeps every `Objective` impl focused on loss math; the multiplication is a 2-line buffer loop that doesn't belong duplicated across objectives.

Rejected: a `WeightedObjective<T>` wrapper that satisfies `Objective` by composing with weights. Adds a template layer with no semantic content; same effect as the booster-side multiply.

## 26. Dispatch: flat table over cartesian product, `IBooster` at boundary

*Status 2026-08-26: the shipped registry is three axes, `cartesian_product_t<Objectives, Growers, Samplers>`; a `Splitters` axis never shipped and `DispatchConfig` carries three names (`include/bonsai/registry/typelists.hpp`). The Candidate A deliberation this entry cites lives at the [pinned archive](https://github.com/daniel-m-campos/bonsai/blob/06fab232a3b156a7c9f155fbbdf41fe14d45f1af/docs/architecture/6-dispatch.md).*


Runtime → static boundary uses Candidate A from `architecture/6-dispatch.md`: a `constexpr std::array` keyed on a name-tuple, generated by `for_each_type` over `cartesian_product_t<Objectives, Growers, Splitters, Samplers>`. Each cell is a monomorphized `Booster<O,G,S,Sa>` factory. Lookup at the config boundary returns `unique_ptr<IBooster>`.

Cost: **one virtual call per `update_one_iter`**. Acceptable. The hot path is histogram building inside `update_one_iter`, not the call itself; per-iteration vcall is dwarfed by the per-iteration histogram pass. Static-everywhere is preserved inside the iteration body, which is what the proposal §3.4 rule actually targets.

Rejected:

- **Nested registry callbacks (Candidate B).** Fully static, zero vcalls anywhere, but forces continuation-passing at the boundary and drags the whole training run into the innermost lambda.
- **Dressed-up nested lambdas (Candidate C).** No advantage over A or B once the type-level builder is in play.
- **Hybrid flat-table + generic callback.** Doesn't compile: `std::array` of function pointers can't be generic over the callback type.

Invalid combinations (none in MVP) are pre-filtered at typelist construction by building sub-products over compatible sub-typelists and concatenating. No runtime check, no concept predicate, no instantiation of bad cells.

`Backend` placement deferred to `7-parallel.md`; dispatch stays 4D for now; promotion to 5D or separate composition both stay open.

## 27. Booster shape, training loop, and Sampler concept

*Status 2026-08-26: the code has outrun the ratified shape: `Booster` takes three template parameters, `grow()` returns leaf values so the loop re-walks nothing, and the identity sampler is `AllRowsSampler`. `include/bonsai/booster.hpp` is the reference; the ratified doc lives at the [pinned archive](https://github.com/daniel-m-campos/bonsai/blob/06fab232a3b156a7c9f155fbbdf41fe14d45f1af/docs/architecture/5-booster.md).*


Ratifies `architecture/5-booster.md`.

**Class shape.** `IBooster` is the boundary erasure type with a minimal CLI-facing virtual surface (`update_one_iter`, `eval`, `predict`, `n_iters`, accessors for save/load). `Booster<Obj, Gr, Sp, Sa>` is the real class, monomorphized per cell of the dispatch table (decision 26). One vcall per `update_one_iter` from the CLI; zero inside.

**Training loop (`update_one_iter`).** Six steps in order:

1. `Obj::compute(scores_, labels, grad_, hess_)`: full row set.
2. Apply `Dataset.weights` to `grad_`, `hess_` if present (decision 25).
3. `Sa::sample(grad_, hess_, rng, out_indices)` → row indices for the grower.
4. `grower.grow(dataset, grad_, hess_, row_indices, split_finder, cfg)` → tree.
5. `scores_ += learning_rate * tree.predict(train_rows)`: re-walks the tree per training row.
6. `trees_.push_back(std::move(tree))`.

Sampling runs *after* grad/hess (not before) because GOSS samples on
`|grad|`, the grad pass is small (~5-10%) compared to histogram
building (60-80%, what sampling actually targets), and reordering would force a "does this sampler need grad?" flag on the concept.

**`learning_rate`.** Applied at score-update time (step 5), not pre-scaled into leaf values. Saved trees carry raw leaf values; the booster reapplies the rate at predict and at score-update. Matches xgb's "shrinkage = booster concern" model.

**Score update via re-predict, not cached leaf values.** Step 5 calls `tree.predict(train_rows)` rather than reading a precomputed `(row → leaf_value)` array from the grower. Cost is one extra `O(n_rows × depth)` walk per iter, small at MVP scale. xgb / lgbm / catboost all cache the row→leaf mapping as a byproduct of growing and reuse it for the score update; bonsai defers that optimization to Phase 2 (after benchmarking). When justified, change grower return from `Tree` to `(Tree, std::vector<float> train_leaf_values)` and replace step 5 with a flat add. Pure additive change to the grower→booster boundary; `Tree` stays clean.

**Initial score (bias).** Booster owns it (decision 22). Three sources in priority: `cfg.init_score`; objective-appropriate default from labels (mean for MSE, `log(p/(1-p))` for logloss); value loaded from disk for a continued booster.

**Predict path.** Raw-score predict sums tree predictions plus `init_score`, applies `learning_rate` per tree. User-facing predict applies the objective's link inverse (identity for MSE, sigmoid for logloss) via `if constexpr` on `Obj`. Inverse link lives in the booster (decision 22), not in `Objective::compute` (decision 24).

**`Obj` is purely-static.** No instance member; `T::compute` and `T::eval` are static (`4-objective.md` / decision 21).

**Booster borrows `Dataset`.** Lifetime sits with the CLI; `update_one_iter` takes `Dataset const&`. Saved model is `BinMappers` + trees + init_score, no `Dataset`. Booster does not own `BinMappers` (decision 3: predict path doesn't need them).

**Save / load are I/O, not booster methods.** Free functions `save_booster(IBooster const&, path)` and `load_booster(path) -> unique_ptr<IBooster>`. Rationale: `load` needs the four component types *before* it has a `Booster` to dispatch on, so it reads names from disk and calls into the same registry path `make_booster` uses: that's structurally the dispatch boundary, not a member function. `save` is the symmetric counterpart.

**`Sampler` concept (Phase 1).** Static members, same shape as `Objective` and `SplitFinder`:

```cpp
template <typename T>
concept Sampler = requires(floats_view grad, floats_view hess,
                           std::mt19937& rng,
                           std::span<size_t> out_indices) {
    { T::sample(grad, hess, rng, out_indices) } -> std::same_as<std::size_t>;
};
```

Returns the count of selected indices written into the head of `out_indices`; the buffer is owned by the booster and reused across iterations. RNG is passed in by the booster, which owns determinism (decision 7).

`NoSampler` is the Phase 1 identity impl (writes `0..n_rows-1`, returns `n_rows`). `GOSS` and `BernoulliSampler` are Phase 4. The sub-product machinery from `6-dispatch.md` handles any sampler-grower incompatibilities. Sampler doc folds into `5-booster.md` for now; spin out into `5b-sampler.md` if it grows.

Rejected:

- **`Obj` as instance member.** No state needed in MVP; revisit if a future objective needs config.
- **Cache row→leaf mapping in MVP.** Adds grower→booster API surface before benchmarking justifies it.
- **`IBooster::save` / `IBooster::load` as virtual methods.** Load has no `Booster` to dispatch on; both belong in an I/O module.
- **Sample-before-grad ordering.** GOSS dependency + small grad cost + concept-flag avoidance.

## 28. Spine complete; insert Phase 2.5

**Milestone (2026-05-18).** Phase 1 (Serial MVP) and Phase 2 (benchmark harness) are complete. The spine is end-to-end working on California Housing: `Dataset`, `BinMapper`, `BinMappers`, `Histogram`, depth-wise + oblivious `TreeGrower`, `DenseTree`, `ObliviousTree`, histogram `SplitFinder`, `Objective` concept + `MSEObjective` + `LogLossObjective`, `Sampler` concept + `AllRowsSampler`, `Booster<O,G,Sp,Sa>` + `IBooster`, registry / dispatch flat table, dispatch resolution doc (`6-dispatch.md`). The Python sidecar runs bonsai vs xgboost / lightgbm / catboost on the same TOML config. Eval baseline pinned at `rmse=0.7175214` (regression net via `tests/unit/test_eval_baseline.cpp`).

**Phase 2.5 inserted.** Between Phase 2 and Phase 3, before turning on parallelism, the next focus is a CLI / config usability and design pass plus the small items glossed over during the Phase 1/2 sprints. No new spine, no parallel backends. Items captured as commits during the work rather than pre-listed. Phase 3 (Parallelism) follows; YearPredictionMSD becomes the perf benchmark there.

**Parallel backends come after a design pass.** The two remaining spine items (`ParallelBackend` concept + first impl) wait on `architecture/7-parallel.md`, which fixes the threading-model design calls before any backend code is written.

## 29. Model file serializes the full `Config` via NLOHMANN macros, not the existing TOML codec

**On-disk format (2026-05-20).** `save_booster(IBooster const&, path, BinMappers const&, Config const&)`. Full Config rides along in the msgpack envelope under `"config"`. `load_booster` reads it straight into `LoadedBooster::cfg` and calls `make_booster(out.cfg)`: no synthesized-Config indirection (was `{dispatch, learning_rate}` torn apart at save and stitched back at load). Format version bumped 1 → 2; no v1 artifacts in the repo, fail-loud on stale files is correct.

**Mechanism.** `NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE` per Config sub-struct (Data, BinMapper, Tree, Booster, Dispatch, Metrics) plus one for `Config` itself. Same macros are used for the tree-node POD records (`DenseTree::InternalNode` / `LeafNode` / `Params`), replacing ~65 lines of hand-typed JSON-key mirror code. One-shot `adl_serializer<std::optional<T>>` added in the same TU: nlohmann v3.11 doesn't ship optional support. Every macro lives in `src/io/model.cpp` (no public-header nlohmann leakage).

**Why not the existing TOML codec.** The TOML path uses `Section` descriptors + `field_name<MemPtr>()` (`source_location`-based extraction) to drive serialization with zero per-struct boilerplate. The JSON path could reuse those Section tuples the same way: that was the originally-proposed approach. Rejected in favor of the macros: less code (~10 macro lines vs ~30 lines of generic fold + two new files in `bonsai::config`), one fewer abstraction layer to read when debugging the on-disk format, and the macro form lives next to the only consumer (`model.cpp`). Trade-off explicitly accepted: the macro field-list duplicates the existing Section field-list. A member added to a struct but not to one of the two enumerations silently drops from that serializer.

**Why not modern C++ reflection.** P2996 (static reflection, C++26) collapses every NLOHMANN macro in this file to one template per direction. Not available in our toolchain. When it lands, this decision is the natural retirement point: delete the macros, ship the reflection-based serializer, format version stays at 2. Boost.PFR is positional-only (no field names) and brings a dependency for no JSON-key benefit. Rejected.

**Inspection.** Model files are now `jq`-able as a structured tree: `nlohmann::json::from_msgpack(read_file(path)).dump(2)` yields `{magic, version, config: {data, bin_mapper, tree_config, booster_config, dispatch, metrics}, bin_mappers, init_score, trees}`. This was a load-bearing factor in the encoding choice: a TOML-string-in-JSON alternative (~2 lines using `dump_toml`/`parse_toml`) would have hidden Config behind one escaped-string blob and broken `jq` access to individual fields.

**Knock-on.** Every Config sub-struct gained `bool operator==(...) const = default;` so round-trip tests can assert `loaded.cfg == cfg` in one line. Also widened the existing `[model_io][config]` test to populate every Config leaf with a non-default value (covers each leaf-type's nlohmann conversion in one shot).

---

## 30. `ObliviousGrower`: fold-then-score was wrong, revert and re-spec

**Date.** 2026-05-21 (landed), 2026-05-22 (reverted).

### What landed (2026-05-21)

`ObliviousGrower<SplitFinder SplitterT>` in [`include/bonsai/grower.hpp`](../include/bonsai/grower.hpp) + [`src/grower.cpp`](../src/grower.cpp). The level-scoring step folded per-feature histograms across the frontier into one summed level histogram, then called the existing single-node `HistogramSplitFinder::find` on that fold. Registered as `impl_name = "oblivious"` in the `Growers` typelist; the registry's cartesian product picked up `{mse, logloss} × oblivious × all_rows` automatically. Model I/O was generalized to be tree-type-polymorphic (`try_save_as<B>` / `try_load_into<B>` use `typename B::tree_type`; new `tree_to_json` overload + `tree_from_json<TreeT>` specialization per tree type). `Histogram::operator+=` added to drive the fold.

### What went wrong

The gain function `score(g, h) = g²/(h + λ)` is non-additive:

```
score(Σ g_i_L, Σ h_i_L) + score(Σ g_i_R, Σ h_i_R) − score(Σ g_i, Σ h_i)
  ≠
Σ_i [ score(g_i_L, h_i_L) + score(g_i_R, h_i_R) − score(g_i, h_i) ]
```

Folding histograms before scoring gives the first expression. The gain induced by applying one split to every parent in the frontier is the second expression. Bonsai's fold-then-score therefore did not compute the right gain.

The bug surfaced during depth=2 testing as a "fold equals root histogram" property: because rows are partitioned (not removed) across the frontier, the fold at level `k+1` reconstructs the root histogram exactly, so the splitter re-picked the same `(feature, bin)` at every level and produced degenerate trees where `2^depth − 2` leaves were empty. I documented this as inherent to oblivious + basic gain. It isn't; it's an artifact of the wrong gain function.

### Verification against CatBoost (2026-05-22)

Inspected [catboost/private/libs/algo/greedy_tensor_search.cpp](https://github.com/catboost/catboost/blob/master/catboost/private/libs/algo/greedy_tensor_search.cpp). The symmetric-tree path calls `CalcBestScore` → `CalcStatsAndScores`, which builds per-leaf histograms across the current depth's leaves and aggregates gains via `SetBestScore`. Per-parent gain summation confirmed. The fold-then-score approach has no CatBoost analog.

### Resolution (2026-05-22)

Reverted the broken implementation in one commit:

- `ObliviousGrower<SplitterT>` declaration removed from [`include/bonsai/grower.hpp`](../include/bonsai/grower.hpp).
- `make_level_node` helper + `ObliviousGrower::grow` impl + explicit instantiation removed from [`src/grower.cpp`](../src/grower.cpp).
- `Histogram::operator+=` removed from [`include/bonsai/histogram.hpp`](../include/bonsai/histogram.hpp).
- `Growers` typelist reverted to `TypeList<DepthwiseGrower<...>>` in [`include/bonsai/registry/typelists.hpp`](../include/bonsai/registry/typelists.hpp); `impl_name<ObliviousGrower<...>>` removed from [`include/bonsai/registry/names.hpp`](../include/bonsai/registry/names.hpp).
- Two `Booster<..., ObliviousGrower<...>, ...>` explicit instantiations removed from [`src/booster.cpp`](../src/booster.cpp).
- `tests/unit/test_oblivious_grower.cpp` deleted; CMake entry removed; oblivious cases removed from `test_make_booster.cpp` and `test_model_io.cpp`. The `ObliviousTree`-specific `tree_to_json` overload and `tree_from_json<ObliviousTree>` specialization in [`src/io/model.cpp`](../src/io/model.cpp) were also removed because `-Werror=unused-function` would otherwise fire.

### What was kept

Infrastructure that is independently valid (does not assume the broken impl):

- `ObliviousTree::splits()` / `leaf_values()` accessors on [`include/bonsai/tree.hpp`](../include/bonsai/tree.hpp). Needed by I/O once the correct grower lands.
- `using tree_type = typename Gr::Tree;` public alias on `Booster` in [`include/bonsai/booster.hpp`](../include/bonsai/booster.hpp).
- Tree-type-polymorphic `try_save_as<B>` / `try_load_into<B>` in [`src/io/model.cpp`](../src/io/model.cpp) (uses `typename B::tree_type`), and the `NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE` macros for `ObliviousTree::LevelSplit` / `Params`.
- [`tests/unit/test_grower_helpers.hpp`](../tests/unit/test_grower_helpers.hpp) shared fixtures, grower-agnostic, still used by `test_grower.cpp`.

### Outstanding

Decision 8's Phase-1 commitment (depth-wise + oblivious) is not honored. The design-review drift flag at [`reviews/2026-05-19-design-review.md`](https://github.com/daniel-m-campos/bonsai/blob/main/docs/reviews/2026-05-19-design-review.md) §"DenseTree / ObliviousTree" remains accurate. The correct algorithm is documented as a design target in [`architecture/3-tree.md` §"Oblivious grow loop"](https://github.com/daniel-m-campos/bonsai/blob/06fab232a3b156a7c9f155fbbdf41fe14d45f1af/docs/architecture/3-tree.md); implementation pending in Phase 2.5 (user-authored).

### Lesson

Mathematical primitives need a sanity check before being trusted across an aggregation boundary. `score(g, h)` looks superficially linear in `g`, but the `g²` term kills additivity. Two more checkpoints that should have caught the bug earlier:

1. **Cross-reference against the reference library before implementing.** CatBoost's source explicitly aggregates gains per-leaf. A 10-minute read of `greedy_tensor_search.cpp` would have surfaced the right shape before any code was written.
2. **Design tests that distinguish the right answer from a plausibly-wrong one.** The original `depth=2` test was content with "4 leaves, structurally correct" and even rationalized the degenerate `[2 non-empty, 2 empty]` outcome. A test that demanded "all 4 leaves carry rows" or "level-1 split differs from level-0 on some non-trivial fixture" would have failed loudly.

## 31. `LeafwiseGrower`: best-first growth on a gain-keyed heap, `max_leaves` primary

Third grower, `dispatch.grower_name = "leafwise"` (LightGBM's default strategy). A `std::vector<Candidate>` maintained with `std::push_heap`/`std::pop_heap` holds every expandable leaf (`{SplitInput, SplitOutput, depth}`) keyed on split gain; each pop converts one leaf into two children, so `live_leaves` counts up and growth stops at `TreeConfig::max_leaves` (new field; `0` = unbounded, `max_depth` stays as the cap). Reuses `make_root` / `split_node` / `finalize_as_leaf` / `HistogramNodeSplitFinder` unchanged and emits a `DenseTree`, so registration is just the typelist + `impl_name` edits.

- **`std::vector` + heap algorithms over `std::priority_queue`**: `top()` returns `const&`, which fights moving the histogram-heavy `SplitInput` out; `pop_heap` + `std::move(heap.back())` doesn't.
- **Tie-break**: equal gains resolve to the lower node id (FIFO-ish), so trees are deterministic.
- **Semantics**: with `max_leaves = 2^max_depth` and a separable dataset, leafwise reproduces depthwise's tree exactly (covered by unit test).

## 32. Parallelism: OpenMP behind a one-function seam, determinism at any thread count

`bonsai/parallel.hpp` exposes `parallel::for_each_index(n, f)`: an OpenMP `parallel for` (dynamic schedule, chunk `n/(threads*4)`) with a serial fallback when OpenMP is absent, plus `set_n_threads` fed from a new `[parallel] n_threads` config section (0 = all cores). Every parallel site assigns each index to exactly one thread and performs **no cross-thread reductions**: per-feature histogram fill, per-feature split scans (per-feature bests merged serially in feature order, preserving the tie-break), row-wise predict, objective grad/hess, score updates, CSV row parsing, binning, mapper fitting.

Consequence: models and predictions are **bit-identical to a serial run at any thread count**: stronger than the proposal's fixed-thread-count contract (decision 7), because the row-parallel-within-feature + per-thread-histogram-merge design that motivates the weaker contract hasn't been needed yet. If it ever is (single-feature datasets), the contract degrades to fixed-N as originally specified.

Rejected for now: the `ParallelBackend` concept as a 5th dispatch dimension (proposal §3.4, `7-parallel.md` TBD). One free function covers every call site today; promoting it to a dispatched component adds a typelist dimension with a single implementation. The seam keeps the door open: `std::execution`/TBB would slot in behind the same signature.

## 33. Hot-path perf: ordered gradients, stable scatter, node totals out of `add`

Three measured wins on Year Prediction MSD (M2, 8 threads), all bit-identical outputs:

- **Ordered gradients** (LightGBM trick): `populate_from_rows` gathers grad/hess into node-row order once, so each of the 90 per-feature scans reads them sequentially instead of re-walking two full arrays with scattered indices.
- **Stable split scatter**: `split_node` replaces `std::partition` + two `assign`s with a two-pass exact-size stable scatter. Stability keeps every node's rows ascending (root's are iota), so bin lookups walk memory near-sequentially at every depth.
- **Node totals once**: `Histogram::add` no longer maintains running totals (2 redundant double-adds per row×feature, duplicated per feature); node totals are one O(n_bins) cell sum over `hists[0]`, hoisted per node in the split finders.

CSV load: whole-file read + line index + row-parallel `from_chars` parse straight into column-major storage, and the train file is parsed once (mapper fit + binning share the batch) instead of twice. Load 7.4s → 1.3s; 200-iter depthwise fit 73s → ~27s; leafwise ~12s.

## 34. Feature-parity round: colsample, GOSS, early stopping, L1, and the OOB score bug

Four reference-library features landed in one pass, each benchmarked A/B on
Year Prediction MSD with the equivalent knob enabled in xgboost / lightgbm /
catboost (protocol + tables in [feature_gap.md](https://github.com/daniel-m-campos/bonsai/blob/main/docs/feature_gap.md)):

- **`tree.feature_fraction`**: per-tree feature subsample drawn from a
  grower-owned rng (`tree.feature_seed`), histograms built for selected
  features only; unselected slots are zero-binned placeholders the finders
  skip. Node totals moved to `SplitInput::totals()` (first populated hist).
- **`sampler_name = "goss"`**: LightGBM's gradient one-side sampling; the
  `Sampler` concept now takes mutable grad/hess so the sampler can amplify
  the small-gradient sample in place.
- **`booster.early_stopping_rounds`**: incremental valid eval
  (`IBooster::score_base` / `accumulate_last_tree`, one single-tree predict
  per iteration) + `truncate` to the best iteration.
- **`tree.lambda_l1`**: XGBoost-style soft threshold on the gradient sum in
  both the gain score and the leaf value.

**The GOSS benchmark exposed a latent correctness bug** in every subsampled
path: `GrowResult.values` was only stamped for sampled rows, so out-of-bag
rows' entries stayed 0 and the booster's score accumulator silently
diverged from the real model for those rows: their gradients were computed
against predictions missing whole trees. Bernoulli had been quietly paying
~2% RMSE for this (9.1873 → 8.9916 on MSD after the fix); GOSS diverged
outright (RMSE 24.7, worse than predicting the mean) because it re-selects
by |grad| every iteration and fed on its own staleness. Fix: growers now
route unsampled rows through the finished tree *in bin space*
(`route_unsampled`, split bins recorded during growth), which is exact with
respect to the float-threshold predict path: `bin(v) <= b  ⟺  v <=
cuts[b]` under the right-inclusive binner, missing bin routed by
`default_left` on both paths.

Lesson (rhymes with decision 30): the booster-side `values` shortcut was
only ever validated with `all_rows`. A contract as easy to state as "every
row's train value equals the tree's prediction for that row" deserved a
test the day the first subsampler landed.

## 35. Remaining-gap round: objectives, monotone, interaction, DART

Second feature-parity pass (protocol and A/B tables in
[feature_gap.md](https://github.com/daniel-m-campos/bonsai/blob/main/docs/feature_gap.md)); sparse/EFB explicitly stays out of scope
until the harness has a sparse dataset to measure against.

- **Objectives became Config-constructed instances** (like `Sampler`) so
  parameterized losses carry state: `[objective] huber_delta /
  quantile_alpha`. Statics satisfy instance-call syntax, so MSE/LogLoss kept
  their static methods and only gained trivial ctors. MAE / Huber / Quantile
  land with sign/clamped/pinball gradients and median/quantile init scores.
  Known limitation: no leaf-renewal pass, worth ~10% MAE vs
  lightgbm/xgboost's renewed leaves on YearMSD; bonsai matches lightgbm on
  huber, where renewal matters less.
- **Monotone constraints**: candidate splits on a constrained feature are
  rejected when bounded child weights violate the direction, and children
  inherit midpoint-fenced leaf bounds via `SplitInput::lo/hi`; whole paths
  are provably monotone. Costs every library the same ~2% RMSE on CH.
- **Interaction constraints**: `SplitInput::allowed/path` carries the
  permitted feature set down the tree; a feature may split only where some
  group covers the whole path (or alone). Same ~9% RMSE cost as xgb/lgbm on
  a two-group CH config.
- **DART**: dropout of existing trees with bin-space routing to recover
  dropped train contributions (no per-tree caches), rescaling with
  xgboost's `normalize_type="tree"` factors: the DART paper's 1/(k+1)
  starves the new tree by ~1/lr under shrinkage and measurably tanked RMSE
  until replaced. bonsai's DART now degrades less than xgb/lgbm's at the
  same settings. Incompatible with early stopping by construction (throws).
- Oblivious grower rejects monotone/interaction constraints at
  construction rather than silently ignoring them.

## 36. Python bindings: nanobind over the CLI's own seams, static libomp

`_bonsai` (nanobind, `python/bonsai` package, `pip install .` via
scikit-build-core) wraps exactly the seams the CLI uses: `config::
apply_overrides` for params (dotted keys, same codec as `--set`),
`cli::train_with_progress` for fit (so early stopping and valid sets come
for free), `io::save_booster`/`load_booster` for model files interchangeable
with the CLI. No training or prediction logic lives in the binding. The
sklearn-ish `BonsaiRegressor` accepts first-class knobs plus a `params`
dict of dotted config keys. Parity test: same config through the module and
the CLI agrees to atol 2e-4 on California Housing predictions.

**The libomp lesson.** Linking the extension against Homebrew's
`libomp.dylib` deadlocked the process the moment xgboost built a DMatrix:
a `sample` trace showed one OpenMP call stack spanning *two different
libomp images* (ours and xgboost's bundled copy): classic duplicate-runtime
interposition. Fix, standard for wheels: `BONSAI_OPENMP_STATIC=ON` links
`libomp.a` into the module and `-Wl,-exported_symbol,_PyInit__bonsai`
strips every other export (1015 leaked `kmp` symbols before). Verified by
interleaving bonsai / xgboost / lightgbm training in one process.

**Native benchmark rows.** compare.py adds in-process "(native)" rows when
`build/python` is importable, timed like the reference libraries (no
subprocess, CSV, or model-save overhead, though bonsai's `train()` still
includes binning, which xgb's timed `train` does not). Re-baselined:
RMSE identical to the CLI rows; predict drops 0.20s -> 0.08s on
YearPredictionMSD, landing between xgboost (0.017) and lightgbm (0.105).

## 37. Feature importance recorded at grow time; the guide series

**Importance.** Split-count and gain importance ship behind
`IBooster::feature_importance(ImportanceType)`, `bonsai importance`, and the
Python `feature_importances_` / `importance(type)` surfaces. The one design
decision: gain is **stamped when the split is created** (`split_gains` per
node on `DenseTree`, `level_gains` on `ObliviousTree`, serialized, format
v5) because it is not reconstructible from a stored tree. Accumulation is a
20-line walk in `booster.hpp`. Verified by cross-library agreement on
California Housing: bonsai and lightgbm agree on both types *including
their disagreement with each other*: gain crowns MedInc, split-count
crowns Longitude/Latitude (many fine-grained, individually-small splits),
the textbook argument for gain as the default and a finding the agreement
test now pins.

**The guide.** `docs/guide/` is a nine-chapter pedagogical series (concept
→ math → the actual implementing code → runnable experiment → war story)
positioned as a deliberate differentiator: reference libraries document
parameters, the guide documents mechanics against a codebase small enough
to read. War stories are real ones from this log (OOB stale scores §34,
DART's k+1 trap §35, the two-libomp deadlock §36, the split-vs-gain
disagreement above). Stale narrative docs were refreshed in the same pass
(context, report addendum, architecture 2–8, new 7-parallel.md), and
milestones are now git-tagged (the MVP submission tag, `v0.2.0`–`v0.5.0`).

## 38. Completing the non-categorical gap: rows 10-17 in one push

Every remaining non-categorical row of [feature_gap.md](https://github.com/daniel-m-campos/bonsai/blob/main/docs/feature_gap.md)
landed (tables and datasets per row in that doc):

- **Leaf renewal** (10): `GrowResult::leaf_ids` + an objective `renew_leaf`
  hook; the booster regroups rows by leaf and replaces Newton steps with
  loss-optimal values (residual median / alpha-quantile / clamped-mean
  huber). Closed the recorded ~10% MAE gap outright: bonsai now ties
  lightgbm on mae/quantile and leads on huber.
- **Prediction extras** (13) + **warm start** (14): predict_at / staged /
  pred_leaf / dump; --init-model continuation that rebuilds training scores
  by bin-space routing and reuses the loaded mappers.
- **Classification benchmark** (11): streamed HIGGS subset, AUC in both the
  C++ metric registry (rank-sum) and compare.py. The logloss path's first
  live outing landed between xgboost and lightgbm.
- **TreeSHAP** (15): Algorithm 2 over per-node covers (stamped at grow
  time, format v6); verified against a brute-force Shapley reference to
  1e-9 plus the efficiency property at every level.
- **Multiclass** (16): `BoosterFor` became a trait so {softmax, G, Sa}
  routes to a dedicated `MulticlassBooster`, the one objective whose
  K-output shape the 1-D `Objective` concept can't express. Covertype:
  bonsai depthwise leads the field on accuracy.
- **Sparse input** (17): LIBSVM reader behind `data.format`, densified,
  with the boundary stated plainly: input parity yes, sparse compute no.
  a9a AUC within 0.2% of xgboost.

Recurring lesson, third occurrence (after §30, §34): the benchmark is the
strongest test. lightgbm's multiclass metric rejection, catboost's
regressor/classifier split, and a9a's short test-split feature space were
all caught by running the harness, not by unit tests.

## 39. Categorical stage 1: measure before building

**Decision.** Before implementing native categorical splits (gap row 12),
add a genuinely categorical dataset and measure what native handling
buys. Amazon employee access (OpenML 4135): 9 integer-ID features,
RESOURCE at 7.5k distinct values: the regime one-hot cannot reach and
arbitrary ID orderings hurt most.

**Design.** The measurement isolates technique from library: lightgbm
runs twice, once with the IDs as plain numerics and once with
`categorical_feature` declared. That within-library delta (+0.0144 AUC,
and faster fits) is what Fisher set splits are worth, independent of
engine differences. bonsai contributes its two practical options today
(raw IDs: 0.8476; K-fold target encoding: 0.8462) and the other
references their native modes (xgboost 0.8498, catboost 0.8812).

**Findings.** (a) The stage-2 prize is real but modest: ~+0.007 AUC from
bonsai's best to lightgbm-native. (b) catboost's +0.026 lead over
lightgbm-native does not come from target encoding per se (plain K-fold
target encoding was no better than raw IDs) but from the ordered
scheme plus feature combinations. (c) bonsai beats lightgbm when both
are denied categorical machinery, so the gap is the feature, not the
engine. Stage 2 (set splits) proceeds with a measured ceiling instead
of a hope.

---

## 40. GPU-resident growing: widen the builder seam to an optional level backend

**Decision.** Phase 3 (device-resident partitioning and split finding, design in [architecture/11-gpu-resident.md](https://github.com/daniel-m-campos/bonsai/blob/06fab232a3b156a7c9f155fbbdf41fe14d45f1af/docs/architecture/11-gpu-resident.md)) extends the builder policy with three *optional* hooks (`find_splits_many`, `partition_many`, `finalize_rows`) detected via `if constexpr` + `requires`, the same idiom as phase 2's `populate_many`. The host grow loop remains the single algorithm narrative and the decision-maker (leaf-vs-split, smaller-child pairing, constraint propagation); the device executes the data plane. `SplitInput` gains `sums` and `row_count` so it degrades to node metadata when histograms and rows stay device-resident.

**Rejected.** A device splitter as a fourth typelist dimension: there is one device implementation and it is coupled to the CUDA builder's state, so a registry axis buys combinations nobody can instantiate (the same restraint as decision 32 for threading). A `row_to_node` map (xgboost's shape): decision 17's reasons hold on the device too: per-node segments pair naturally with the subtraction trick and stable per-node row order. CUB/Thrust for the partition scan: a hand-rolled three-kernel scan keeps the backend a single self-contained TU with no new dependency.

**Consequences.** In resident mode `SplitInput.hists`/`rows` are empty; the inspect path for tests and debugging is an explicit `download_histograms`. The `cuda_depthwise` determinism contract is formally tolerance-equal (prediction/RMSE tolerance + split-agreement rate in parity tests, never tree equality); CPU-only builds stay bit-identical. Copy-back mode is retained as the degrade path (deep trees, oversized `max_bin`) and for the oblivious/leafwise growers.

---

## 41. Grower data-plane as the `LevelStep` strategy; retire the copy-back ladder

**Decision.** Reframe decision 40's "escalating optional hooks" as a single compile-time Strategy: a `LevelStep<SplitterT, Engine>` (primary template = host data plane; partial specialization for `GPULevelEngine` = device data plane) selected by engine type, so `grow()` reads as one control-plane narrative with the host/device fork localized to one specialization instead of smeared across six `if constexpr`/`if (resident())` sites. `resident()` is removed: the per-tree mode is captured once from `begin_root`'s bool into the `LevelStep`. The concept ladder collapses to two, renamed to shed the GoF-`Builder` connotation (the type is the pluggable compute substrate, and the CUDA one supplies the whole device data plane): `HistogramEngine` (host, from `HistogramBuilder`) and `GPULevelEngine` (the device-resident vocabulary, from `ResidentHistogramBuilder`). The `populate` → `populate_many` → resident progression was a dev-research ladder (phases 1–3, preserved in git and doc 11's measured stages); only device-resident is kept: `populate_many`, `BatchHistogramBuilder`, and the GPU **copy-back** histogram path are retired, and the rare decline (oversized `max_bin`, or level buffers that won't fit) falls back to **CPU** histogram building via the engine's existing `cpu` member. Design in [architecture/12-grower-backend.md](https://github.com/daniel-m-campos/bonsai/blob/06fab232a3b156a7c9f155fbbdf41fe14d45f1af/docs/architecture/12-grower-backend.md). Refines, not overturns, decision 40: host-owns-decisions, one-grow-loop, no-new-typelist-dimension, and the all-or-nothing device coupling all carry forward.

**Rejected.** Runtime strategy objects (a `unique_ptr<LevelPlane>` or `std::variant` handed to the loop): the dynamic-dispatch shape decisions 14/26/32 explicitly chose against: it would need a benchmark to *prove* zero-cost rather than guarantee it, add type erasure the grow loop avoids, and thread into every grower. Keeping the three-tier concept ladder: `populate_many` is a host-plane batch optimization, not a device peer, so the middle tier documented a distinction that no longer earns its name. Keeping GPU copy-back as the decline fallback: it keeps a whole kernel path alive for a rare case (default `max_bin=255` always goes resident); the CPU fallback trades a slice of that rare path's throughput for a materially smaller CUDA backend.

**Consequences.** `update_nodes` (182 lines, 18 params) decomposes into `plan_level` + `LevelStep` methods + `commit_children`, each ≤~40 lines; the CPU path becomes branch-free (a reader never meets a GPU concept). Oblivious and Leafwise route through the shared `LevelStep`, so their CUDA variants become a one-line alias + registry entry (not registered here: a later pass; leafwise's sequential gain-heap limits its device value). Docs 10/11's grower-side *seam* narrative is superseded by doc 12; their device-kernel content stands. Zero perf cost is a release gate, measured on a Thunder 4×A100 against xgboost (`scripts/bench_gpu.py`, MSD, before/after) in addition to CPU bit-identity and cuda tolerance-equal parity. Landed (commits b4d223c → a4764ba): every phase held 392/392 both configs, sha256-identical CPU models for all three growers, and unchanged resident-path launch counts; the leafwise open call resolved *against* singleton-frontier unification (a per-heap-pop LevelPlan buys ceremony, not shared code: `split_node` lives in level_step.hpp as the single-node data plane) and the oblivious unsampled routing stayed its own (leaf-table indices, not DenseTree node ids).

---

## 42. GPU oblivious via a device level-find; no `cuda_leafwise`

**Decision.** Register `cuda_oblivious` (`ObliviousGrower<CudaHistogramEngine>`), completing the pass decision 41 deferred. The resident plane is reused wholesale (an oblivious level is a depthwise level where every node takes the same split), so the only new device piece is the level-find: `find_level_split` sums each candidate cut's child scores across the whole frontier (32-node chunks into per-feature global scratch, any frontier width) and requires per-node `min_child_hess` feasibility, mirroring `update_best_for_feature_for_level`; a fused find → reduce → child-sums launch chain pays one sync per level. `GPULevelEngine` gains the method; `LevelStep` routes `LevelSplitFinder` growers to it. Measured (RTX 5090, fair full-pipeline timing): 3.9–4.1 s vs CatBoost-GPU 7.3–8.2 s on MSD, RMSE matching CPU oblivious exactly.

**Rejected: registering `cuda_leafwise`.** A working `LeafwiseGrower<CudaHistogramEngine>` was built and benchmarked, then withdrawn: best-first growth expands one node at a time, which the level-batched resident plane cannot serve (advance swaps whole levels), so every histogram was computed by the engine's CPU member: a `cuda_` registry name executing CPU work misleads `bonsai info`, serializes into model metadata, and mislabels benchmarks. The comparison it existed for holds without it: CPU `leafwise` (11.1–11.5 s) beats LightGBM's CUDA leaf-wise backend (12.0–12.9 s) on the 5090. A true device leafwise needs a non-swapping advance: phase-4.

**Consequences.** `trains_here` keys on the `cuda` name prefix (registry convention) instead of one hardcoded name; `skip_without_cuda` keys on the grower's `Engine` alias; the oblivious grow loop consumes per-node child sums from the level-find (children's `SplitInput.sums` are otherwise unknowable: device histograms are not host-scannable), and host/device leaf finalize is symmetric via `LevelStep::finalize_leaves`.

## 43. CI: GitHub Actions gates every PR; sanitizers as a build option, not a preset

**Decision.** One workflow (`.github/workflows/ci.yml`, ubuntu-24.04, clang-21 from apt.llvm.org so CI uses the Makefile's exact Linux toolchain): `build-test` (Release + ctest), `sanitize` (`BONSAI_SANITIZE=ON`, ASan+UBSan, RelWithDebInfo), `format` (`clang-format --dry-run --Werror`), `tidy` (`make lint` off a configure-only tree: clang-tidy needs just `compile_commands.json`), `python` (nanobind module + bindings tests), and `cuda-compile`. Sanitizers are a CMake option (`BONSAI_SANITIZE`) applied globally so FetchContent'd Catch2 carries the same instrumentation; a preset file would be a second configuration surface for a Makefile-driven repo. This discharges the proposal's sanitizer-in-CI commitment.

**CUDA is compile-only.** GitHub has no GPU runners, but the kernel TU is the code path a CPU-only CI would otherwise never touch. The job installs only `cuda-nvcc-12-6` + `cuda-cudart-dev-12-6` (headers, ptxas, libdevice, no driver) and builds with `-DBONSAI_CUDA_ARCH=sm_80`, since `native` requires a device to probe. Runtime GPU validation stays where it has always been: `make test-cuda` on a GPU host.

**Consequences.** LSan is off in the sanitize job (`detect_leaks=0`): libomp's pool allocations outlive `main` and drown real leaks; ASan+UBSan remain fatal (`-fno-sanitize-recover=all`). `BONSAI_SANITIZE` + `BONSAI_CUDA` is a configure error: sanitizing device code is not supported and a half-instrumented binary would imply coverage that isn't there. FetchContent `_deps` and the toy CSVs are cached per lockfile-ish keys; a cold run needs the network, warm runs don't.

**The first honest lint run.** `make lint` had been discarding run-clang-tidy's stderr, so tool failures on macOS printed "no findings"; the Linux CI job surfaced 128 findings the local gate never saw. Response: fix the real bugs (a use-after-move in the `for_each_type` fold, two unchecked optional accesses, an exception escaping `main`, plus small `google-*`/naming/enum-size items) and curate `.clang-tidy` down to checks the codebase actually satisfies: the disabled block documents each family (pointer arithmetic at C/CUDA/codec boundaries, reference members, callback `F&&` APIs, concept archetypes, established long functions). The Makefile now fails lint when the tool itself fails. Re-enabling a curated-out check is a deliberate cleanup PR, not a silent flip.

## 44. `n_threads = 0` means capped auto, not all hardware threads

**Decision.** Auto worker count is `min(omp_get_max_threads(), 16)`. Explicit `n_threads = N` passes through untouched, including N above the cap or above the core count. Measured on a 60-vCPU host (issue #2, MSD 463k×90, 200 iters, depth 8): the uncapped default ran CPU `depthwise` 10× slower than 16 threads (167 s vs 15.9 s) and `cuda_depthwise` 4× slower (51 s vs 12.2 s): per-level parallel sections are short (≤ ~250 nodes), so with many workers OpenMP barrier spin-wait dominates useful work, burning CPU-*hours* per fit. 16 was near-optimal there and is at or above the core count of every dev machine this project targets. The cap does not change the determinism contract: no cross-thread reductions exist at any thread count (decision 32).

**Rejected.** Work-scaled formulas (`clamp(n_rows/K, 1, hw)`): the pathology is barrier latency as a function of frontier width, not row count; a formula would be false precision. Setting `OMP_WAIT_POLICY` programmatically: `setenv` only works before runtime init, and inside the Python module another extension may have initialized OpenMP first. `kmp_set_blocktime`: libomp-specific with no header guarantee. The honest version of both is documentation: `OMP_WAIT_POLICY=passive` is the operator knob for oversubscribed many-core hosts (`7-parallel.md`).

## 45. Ingest profiled, then one optimization kept: sort-based quantile cuts

**Decision.** `BONSAI_INGEST_PROFILE=1` breaks the CSV-to-Dataset pipeline into read / index / parse / mapper-fit / bin / buffer (the GrowProfiler pattern; accumulates across train + valid, prints at exit). Profiling MSD (463k×90) on an M-series laptop attributed half of ingest to mapper-fit: `create_cuts` ran ~253 `nth_element` calls per feature over a shrinking 200k suffix. Replaced with one `std::sort` + stride reads: every prior `nth_element(lo, begin+k, end)` call selected the absolute k-th order statistic (its prefix already held the smaller partitions), so the sorted array read is value-identical, verified by model-hash equality on MSD. mapper-fit 0.53 s → 0.25 s; total ingest 1.06 s → 0.76 s. End-to-end on the RTX 5090 benchmark (fair full-pipeline timing): `cuda_depthwise` 2.98 s → 2.61 s, `cuda_oblivious` 3.91 s → 3.50 s (medians of 3), RMSE unchanged.

**Measured and rejected.** `memchr` field splitting: parse 0.26 s → 0.25 s; MSD fields are ~7 chars, too short for SIMD scanning to beat the char loop; reverted rather than kept as complexity without payoff. mmap read and a batched `transform` were not attempted: read (0.11 s), bin (0.09 s), and buffer (0.03 s) are each under the 100 ms bar the plan set for touching a stage. `from_chars` already dominates parse and libc++ 21's implementation is Eisel-Lemire-class; no fast_float dependency.

**Deferred: binned-dataset cache.** A sidecar binary (source size+mtime+bin-config hash as the invalidation key, raw column dumps for the uint16 matrix) would cut warm reloads to ~0.1 s, like LightGBM's binary datasets. It is a product feature, not a benchmark lever: cross-library comparisons stay cold-parse for fairness (decision on `bench_gpu.py` timing), so it waits for its own PR.

## 46. Scaling suite: in-memory sweeps, synthetic Friedman target, frontier-as-data

**Decision.** The scaling study (`scripts/bench_scaling.py`) measures fit/predict complexity in rows, cols, bins, and threads against xgboost/lightgbm/catboost from **in-memory numpy float32 through each library's Python API** (bonsai via the CUDA-enabled nanobind module), not CSV+CLI like `bench_gpu.py`. Corner CSVs would be 15–20 GB of text whose parse time is orthogonal to the question; the CLI's old justification (module was CPU-only) died with the previous PR. Fairness holds because fit is timed from raw arrays and includes each library's own ingestion (ColumnBatch+binning / QuantileDMatrix / lgb.Dataset / Pool); predict is timed from a raw test matrix; quality is scale-free R² train/test. Synthetic target is a generalized Friedman-1 (blocks of 5 informative features with decaying weights over ~20 of the columns, uniform features so bins populate evenly, noise for best-achievable R² ≈ 0.9); `make_regression` was rejected as purely linear: trees would neither differentiate on bins nor separate libraries. Data is deterministic in (rows, cols, seed) only, so bins/threads sweeps reuse identical matrices.

**Grid.** Base 1M×100×255 (depth 8, 100 iters, lr 0.1, 16 threads; 3 repeats at base for variance) with per-axis sweeps: rows ×4 to 16M, cols to 65k with rows shrinking past 4k cols (cells ≤ 2^31: a joint 2D log-fit over rows+cols cells de-confounds the tail), bins to 65535 (bonsai's `bin_id_t` cap; other libraries swept to their own refusals, recorded not assumed), threads {1,4,16,64} at base. A full cross-product at the user-set extremes is 2^40 cells; per-axis is the honest affordable shape.

**Frontier as data.** Every (cell, variant) runs in a child process with a size-scaled timeout; status ∈ ok/oom/timeout/error/unsupported/skipped with a message, and RAM/VRAM estimators pre-skip hopeless corners per host. What each GPU tier can and cannot fit is a first-class result (the reason for running 5090 + A100-80GB + 4090). The child-per-run design also delivers per-run profile capture for free: bonsai's exit-time profilers flush to the child's stderr. `lgbm_cuda` is declared `unsupported` in v1 (pip wheel lacks CUDA; source build deferred); catboost-GPU's 254-bin cap is applied and recorded as `bins_effective`.

## 47. Ingest and histogram-allocation rounds: measure on the platform that has the disease

**Decision.** Rounds 1–2 of the optimization campaign (PRs #8/#9/#10): the Python module ingests via a parallel cache-blocked transpose, then (round 1b) bins directly from the row-major numpy matrix through new `BinMappers::fit`/`Dataset::bin` overloads: peak RSS fell from 4.8× to 1.8× of raw X with byte-identical models; and histogram cell blocks recycle through a size-class pool (`HistBlockPool`) with the oblivious level-finder's prefix scratch hoisted to per-worker storage. The pool's win is invisible on macOS (its allocator already recycles; paired Mac A/B was deliberately flat) and decisive on Linux, where fresh >mmap-threshold blocks page-fault per populate: G1 measured CPU depthwise bins 4095 at 3.1×, 16383 at 5.2×, and the 65535 cell went from timeout to completing. Lesson recorded: allocator-behavior optimizations must be validated on the deployment platform; the Mac can prove only non-regression.

## 48. The "5090 anomaly" was a defective host; benchmark pods now pass a sync-latency probe

**Decision.** The scaling study's ~11–14s per-cuda-fit overhead on the 5090 host class was fully diagnosed in G1: one specific rentable machine has a GPU sync round-trip of ~300µs (healthy: 4µs measured on a second 5090), invariant to schedule flags (spin/yield/blocking), with perfect PCIe Gen5 x16 and 23.7GB/s pinned bandwidth, consistent with host-level ASPM/IRQ misconfiguration, unfixable from a container. bonsai's ~20–30k synchronizing ops per fit × 300µs reproduces the excess exactly. The allocator hypothesis was refuted by A/B on that host (async vs sync alloc identical); the stream-ordered allocator, pinned batched bins upload, and dynamic shared-memory opt-in (PR #11) stand on their own merits: the smem opt-in moved the 4095-bins cell from CPU fallback to GPU for a 17.4× win and exposed issue #12 (cuda_oblivious fallback trains garbage above the cliff, pre-existing). Benchmark protocol change: every rented pod must pass a 30-second sync-latency probe (>50µs → reject the pod), and the round-1 fleet's 5090 rows are annotated as defective-host data.

## 49. Row-wise histogram fill over a row-major u8 mirror; determinism relaxes to fixed thread count

*Status 2026-08-26: `Dataset::row_major_bins` was renamed; the mirror is reached as `Dataset::mirror()` (`include/bonsai/dataset.hpp`). The canonical determinism statement moved to [invariants.md](invariants.md) and [design/determinism.md](design/determinism.md).*


**Decision.** The scaling study's flat 2.5–4× CPU fit gap vs LightGBM was populate-bound (50–85% of fit) and, on deep small nodes, gather-bound: the feature-parallel fill reads one binned column per feature with `bins[rows[k]]` scatter, so a sparse node's every access misses cache, measured at ~0.7G adds/s against ~3.3G adds/s for large dense nodes (M2, adds counted by the profiler). `CpuHistogramEngine::populate_many` now fills u8 (max_bin ≤ 255) data row-wise over a lazily built row-major mirror (`Dataset::row_major_bins`, +n_rows×n_features bytes, never built by CUDA or predict-only paths): a level's nodes become row-block work units; each reads its rows' bins as contiguous 1×n_features strips regardless of node sparsity, streams grad/hess once total instead of once per feature, and accumulates into private per-block partial histograms merged in fixed block order. Nodes below a work threshold (fill ≥ ~16× the partial zero+merge cost) run as one block writing the node's cells directly (byte-identical to the old order), and the whole level shares one parallel section, so 128 deep nodes are 128 units instead of 128 serial parallel-section spawns. Base cell (1M×100×255, M2, 8 threads): fit 26.7→16.9s, populate 20.1→9.6s; 4M rows 43.0→25.3s; 1M×512 35.9→21.8s; identical R². LightGBM's same-cell fit on the M2 is 18.0s: first cell where bonsai CPU leads it.

**The contract spend (user-approved).** Multi-block nodes' sums depend on the block count, a pure function of node size, selection width, total selected bins, and the configured thread count, so models are bit-identical at a fixed `n_threads` (decision 7's original contract) but no longer across thread counts. Single-block nodes, the u16 feature-parallel fallback, split scans, and predictions still match serial order exactly. `docs/architecture/7-parallel.md` §"The determinism contract" is the canonical statement.

**Measured and rejected.** Two-accumulator parity splitting of the feature-parallel fill (microbenchmark promised 1.6×): exactly flat in the real loop at 1 and 8 threads: the microbench's arrays were cache-resident, isolating FP-chain latency the real streaming loop hides; kept the microbench-lies lesson, dropped the code. Block-count oversubscription beyond 4× threads: flat on M2. The row-major mirror as eager dual storage in `Dataset::bin`: rejected, CUDA runs would pay RSS for a mirror they never read.

## 50. Float histogram cells; double reductions; the empty-split demotion guard

**Decision.** `HistCell` is `{float, float}` (was double). Per-cell sums are bounded by node size and gradients/hessians arrive as float, so cell storage carries only per-cell rounding; every reduction that crosses cells (`Histogram::totals()`, `fill_prefix`, the node finder's running left sums (`split_sums_at` now takes doubles)) accumulates in double and converts once at the store. Measured (paired, M2 8 threads): base cell fit 15.3→13.2s, 8M×100 53.3→45.7s (populate 1.23–1.25×), **R² identical to six decimals at both scales**; California-Housing eval baseline moved 5.4e-6 (re-pinned). Histogram pool, partial slabs, and prefix scratch all halve. The CUDA engine had already proven the shape (float per ≤32k-row chunk, double merge, RMSE parity): this brings the CPU fill to the same discipline. User-approved trade: models change (one-time re-pin), quality doesn't.

**The bug it flushed out.** Sibling subtraction in f32 leaves noise ~O(cell_sum × eps) in the derived child's cells, far above the `gain > 0` gate, so a degenerate cut (every row one side) can score a tiny positive gain. In f64 the same mechanism existed at 1e-16 and never fired; in f32 it produced empty-child splits whose cover-0 nodes made SHAP contributions NaN (caught by test 60). Fix is structural, not precision-tuned: after partitioning, `demote_empty_splits` (depthwise plan) and the leafwise equivalent convert any split with an empty child back to a leaf: the partition's row counts are ground truth at any precision. Pre-allocated child nodes remain as unreachable placeholders; predict and SHAP walk from the root.

**Rejected.** A `hist_precision` runtime knob (f32 partials only, f64 default): measurement showed no quality cost, so the config surface buys nothing, and a compile-time macro would fork the ABI of a public-header type across builds and double the CI/wheel matrix. Keeping f64 cells with f32 only in partial slabs: leaves the direct-fill nodes, subtraction, and finder scans at 2× traffic for no accuracy benefit that the double-reduction discipline doesn't already provide.
## 51. quantile_step ceiling stride: cuts never exceed the budget

**Decision.** `quantile_step` uses the ceiling stride `ceil(n/(budget+1))` (was `floor(n/budget)`, floored at 1), guaranteeing at most `cut_budget` cuts for any subsample size. The floored stride overshot whenever the subsample wasn't comfortably above the budget (400 distinct values at the default `max_bin=255` produced 401 bins, 1000 produced ~335), giving small datasets more granularity than requested and silently disqualifying them from u8 storage and the row-wise fill (issue #17, found while writing decision-49 tests). Every model's cut positions shift (this is the second model-changing round after decision 50, landed back-to-back deliberately): synthetic 1M×100 R² identical at 0.8991; the California Housing eval baseline *improves* 0.7175214 → 0.7157657 (−0.24%), consistent with budget-respecting quantiles being no worse while small datasets regain the fast path.

**Rejected.** Keeping the floor and clamping cut *count* post-hoc (drop every k-th surplus cut): non-uniform quantile spacing for no benefit over the ceiling stride. Treating it as a documentation caveat: the silent u8 disqualification interacts with two performance features that key off `max_bin<=255`, which is too much surprise for a one-word fix.

## 52. Device residency: REFUTED by experiment; the lever is per-level staging latency

**Decision (revised twice, then measured).** The phase-A experiment (PR #28, per the review directive on the design PR: prove it before redesigning APIs) refutes device-resident gradients: skipping every per-tree grad/hess upload saved **1.6s of a 42.5s fit** (3.7%) at 16M×100 on an L40S, with `upload_s` unchanged at 7.5s. Both of this decision's prior premises misattributed that upload line: first to the one-time bins upload (~70ms), then to bulk gradient bytes (~1.6s at pinned rates). The line is actually hundreds of small per-level `Staged<>` syncs (node sums, bounds, row offsets: latency-bound pageable copies × ~800 tree-levels per fit), which residency does not touch. The next honest lever for the 42.5 vs 27.9s gap to xgboost-GPU: batch/pin the per-level staging and cut per-level host↔device round-trips. The CPU/GPU API-consistency redesign the experiment gated should target *that*: a level-transaction interface would make the batching natural.

**What the experiment paid for itself with.** A live mainline landmine: `demote_empty_splits` (decision 50) orphans device-plane row stamps when it fires after the device has scattered a demoted split's rows (one mega-leaf per tree at rightmost-spine node ids 2^k−2, feeding a score runaway), dormant under double-precision device sums, extracted as its own fix. Plus the diagnostic pattern that found it: per-tree device-state dumps diffed against the model's own predictions.

**Rejected.** Continuing to phase B (device binning) or the API redesign on the unmeasured premise; debugging the experiment's residual 16M-only quality gap (r² 0.79 vs 0.89) once the timing verdict had already killed the approach.

## 53. The level-transaction engine narrative (adopted)

*Status 2026-08-26: the narrative doc this entry ratified was dissolved by decision 114; its still-true residue, the plane composition, lives as comments in `src/cuda/detail/device_context.cuh`, and the concept sketch it drafted shipped under different names (see `include/bonsai/grower.hpp`).*


**Status.** Executed. Step 1 (transaction vocabulary on both planes, byte-identical) landed in PR #33; steps 2–3 landed as one change set: the identity row list cached on device and restored D2D per tree instead of re-uploaded (the avoidable share of the root line), and `end_tree` handing the node value table to the engine, which maps rows to leaf values on device and returns values/leaf ids in two bulk copies (replacing the finalize line's per-tree host stamping loop over every row). The Impl decomposition shipped in the same set. One deliberate deviation from the sketch below: `gh_ordered` lives in LevelPipeline, not GradientPlane; it is the level-row-ordered gather and ping-pongs with `gh_b`.

**Decision (proposed).** The CPU/GPU engine APIs unify around a level-transaction narrative (`begin_tree` / `open_level(LevelInputs) → LevelOutputs` / `apply_level` / `end_tree() → TreeEpilogue`) with the backend an implementation detail, per the review commission on the device-residency design. The shape is derived from the measured 16M ledger (PR #31), not from a lever bet: the root becomes an ordinary one-node level (dissolving `begin_root`'s bespoke per-tree staging, the ledger's largest line at ~14s), `open_level`'s single-struct input batches the per-level staging by construction, and `end_tree` owning the per-row epilogue is where finalize residency (8.4s of per-tree D2H + host stamping) can land with its real motivation. `CudaHistogramEngine::Impl` (41 buffers) decomposes into DeviceData / GradientPlane / LevelPipeline planes with ledger-exposed lifetimes. Migration in three bit-identical-gated PRs: host plane first, device plane onto the transactions, then the device epilogue. Full design: `docs/architecture/14-engine-narrative.md`.

**Rejected.** Redesigning around the refuted residency/staging levers; a single templated engine (hides the narrative); fusing the level phases into one transaction (the control plane must observe results between them: doc 12's host-control contract stands).

## 54. Device binning at ingest (adopted)

**Status.** Implemented. `cuda_ingest` (both arms), the `IngestPlane` receipt on `Dataset` with lazy host materialization, plane adoption in `ensure_dataset`, pipeline wiring by grower-name prefix, and the doc-16 instrumentation (`bins_upload`, `fin_wait`/`fin_d2h`, `dbin`, fit-profile in the bench harness) shipped as one change set. One correction from implementation: the module path bins from the borrowed row-major numpy view (`features_view`), CSV from the feature-major `ColumnBatch`; the doc-15 draft had the arms swapped.

**Decision (proposed).** Ingest joins the transaction narrative as the zeroth verb (`ingest(raw, mappers) → IngestPlane`) with the backend an implementation detail (doc 16 frames the narrative as a compute DAG; this is the last node outside the vocabulary). The host backend's ingest is today's `fill_binned`, untouched. The CUDA backend's ingest streams raw columns through double-buffered pinned staging and a `lower_bound`-exact bin kernel; its product is an opaque `IngestPlane` handle carried by `Dataset` as the transaction's receipt (defined in the CUDA TU, null in stub builds, the `row_major_` lazy-mirror precedent). The train pipelines select the ingest backend the way growers dispatch (`cuda` name prefix + `cuda_available()`, training dataset only); `ensure_dataset` adopts its own plane instead of copying and uploading host bins. Host binned columns are not materialized in device mode: the fallback-decline arm (oversized `max_bin`, decidable at ingest) keeps the host path outright, and `route_unsampled`'s `bin_at` triggers a one-time cached D2H materialization only when row sampling is on. Cuts stay host-fitted (identical `transform` semantics ⇒ bit-identical bins ⇒ before/after models must be equal, the gate). Motivation is the measured line, not a bet: host `bin` ~4.6s of 1.6G binary searches + an unlapped 1.6GB upload, replaced by ~0.5s of overlapped transfer+kernel at 16M×100 (same-pod L40S; priced by the measured 19GB/s gh edge, scripts/dag_model.py). Ships with the missing lap counters (`bins_upload`, finalize `fin_wait`/`fin_d2h`, ingest `dbin`): the PR #35 refutation was designed against an undecomposed finalize line, the fourth such miss. Full design: `docs/architecture/15-device-binning.md`.

**Rejected.** Engine-side rebinning (the host transform *is* the cost); retaining raw floats on `Dataset` (+6.4GB RSS at 16M); device mapper-fit this round (reservoir-RNG identity: model-changing, deferred to its own decision); a config knob for device binning (it is an implementation detail of the cuda growers, not a user choice); the first draft's `DeviceBins` side channel on `Dataset` (identical mechanics, but the API grows by exception instead of by the narrative's vocabulary).

## 55. The cut-quality gap is not sample size (study)

**Study (2026-07-12).** The residual test-r² gap to xgboost-GPU at the 16M×100 cell (bonsai 0.8791 vs xgb 0.8800) was hypothesized to come from fitting cuts on a 200k-per-feature subsample. Sweeping `bin_mapper.n_samples` over {200k, 1M, 4M, 16M (full column)} on one pod, same cell, same seed: r²_test moves 0.879083 → 0.879048 → 0.879194 → 0.879261 (**+0.0002 end to end, ~2% of the gap**), while fit cost is flat (the mapper's reservoir scan, not the sort, dominates and even that is noise at this scale). Verdict: sample size is a dead lever; the 200k default stands. **Follow-up (same day, issue #42) dissolved the gap entirely**: bonsai trained on xgboost's exact cut set (extracted via `DMatrix.get_quantile_cut()`) scores the same as on its own cuts (0.879034 vs 0.879083), and xgboost's own r² moves by 0.001 (the full size of the "gap") between `max_bin` 255 and 256 (0.879953 vs 0.878966, same pod, same seed). There is no sketch deficit; at this cell every library sits in a ±0.001 band governed by threshold-placement chance, and no research round is warranted.

**Rejected.** Raising the default `n_samples` (cost without benefit); treating the gap as a defect to chase this round (it is a documented trade at +0.0009 r², with bonsai holding a 3× host-memory and ~3× predict-speed advantage at that cell).

## 56. Quality-campaign fixes: the oblivious veto, exact duplicate cuts, the true softmax hessian (adopted)

**Decision.** Three model-changing fixes from the 2026-07 quality campaign (`benchmarks/quality-campaign-2026-07.md`), each probe-confirmed on real datasets before implementation: (1) the oblivious level scan no longer vetoes a candidate when one frontier node's children fall under `min_child_hess`: the infeasible node contributes zero gain and the broadcast split still applies (empty children are first-class since #57); this was worth 3–26% rmse on real data and moves bonsai-oblivious ahead of catboost. (2) `create_cuts` emits one right-inclusive cut per distinct value when the subsample's distinct count fits the budget (the stride+dedup previously collapsed house_sales' 13-value bedrooms column to 7 cuts); measured net-neutral on the campaign but objectively correct: the California pin moves 0.7157605 → 0.71625 (+0.07%). (3) the multiclass hessian is the true diagonal `p(1−p)`; the factor-2 variant halved every Newton step and cost exactly 2× the iterations to match lightgbm (letter 0.9515 → 0.9613 at the same budget). Post-fix, bonsai-depthwise is the best-scoring library on 8 of 9 campaign datasets.

**Rejected.** Keeping the veto with a lower default `min_child_hess` (the veto is wrong at any threshold: one node should never censor the level); weighting duplicate cuts by count above the budget (deferred to #63, where continuous placement is the live question); retaining the factor-2 hessian for "xgboost compatibility" (xgboost loses to both at matched budgets precisely because of it).

## 57. Count-weighted cuts for heavy-value columns only (adopted)

**Decision.** The #63 follow-up to decision 56's cut work: above the distinct-value budget, a column where some value's count reaches a mean-sized bin gets lightgbm-shaped greedy allocation: heavy values take a bin to themselves, the rest fill toward a running mean, cuts at midpoints between adjacent distinct values (`greedy_weighted_cuts` in `src/bin_mapper.cpp`). Columns with no heavy value keep the decision-51 quantile stride bit-identically: with every count below a mean bin, equal frequency already *is* the count-weighted allocation, so the greedy walk could only reshuffle thresholds inside the chance band. Measured on the campaign suite: house_sales 131,841 → 128,959 rmse (27% of the gap to xgboost), every other standing unchanged, bonsai-depthwise still best on 8 of 9; the California pin moves 0.71625 → 0.71719 (+0.13%, chance-band). Both cut placements were probed and are metric-equivalent (midpoint vs value cuts produce identical training bins when the sample covers the column); midpoint kept as the principled choice for unseen test values.

**Rejected.** Greedy allocation for every over-budget column (measured: house_sales gains more, 126,528, but magic_telescope drops below lightgbm and phoneme's depthwise drops below the best ref: a chance-band tax on every continuous dataset to overfit one; the per-column rule takes the heavy-value win without the tax); closing #63 (bonsai at 511 bins matches xgboost at 256 on house_sales: the refs extract more from an equal budget on dense-continuous columns, so per-column budget allocation stays an open research question, kept in tension with decision 55's synthetic verdict).

## 58. Categoricals resolved by measurement: an encoder, not an engine feature (adopted)

**Decision.** Categorical support ships as `OrderedTargetEncoder` in the Python package (causal/ordered target statistics, seeded and deterministic, `keep_codes` giving trees both the response-rate and identity views) plus guide chapter 13: the C++ core stays numeric. The call was bought with a probe, not taste (`scripts/probe_categorical.py`, evidence in `benchmarks/categorical-tradeoff-2026-07.md`): each reference library's categorical machinery toggled on/off at matched knobs on amazon/adult/kick measured native Fisher set splits (the doc-17 stage-2a design, which would have grown the ~1,400-line split/tree/SHAP/model-format core by roughly a third) at **+0.029 / +0.000 / −0.018 AUC by lightgbm's own toggle**: a coin flip whose complexity every user carries. Meanwhile ~100 lines of preprocessing hit 0.8590 on amazon, beating lightgbm-native (0.8572) and closing 48% of the gap to catboost-native (0.8894, whose ordinal baseline collapses to 0.779, so much of its celebrated categorical gain is recovering its own weak numeric path). On the repo's own amazon split the encoder is worth +0.049 AUC (0.811 → 0.860), pinned by `test_encoding.py`. Doc 17's design stays on the shelf, priced and declined; the probe method is codified as the `feature-admission` skill.

**Follow-up (same day).** Crossed pairs close the rest of the gap, still from preprocessing: `cross=2` on the encoder packs each pair of code columns into an int64 key and applies the same ordered TS: amazon 0.8604 → **0.8877** vs catboost-native 0.8897 (chance-band at this test size), with catboost's own toggle as the control: `max_ctr_complexity=1` (crosses off) drops it to 0.8587, *below* our singles line: the crosses were its entire remaining edge, and our single-column ordered TS already beats theirs. Triples measured 0.8859 (overfit, rejected as a default). Probe: `scripts/probe_crossed_ts.py`; pinned in `test_encoding.py`.

**Rejected.** Stage-2a native set splits (measured median gain ≈ 0, negative on kick, cost concentrated in the most-read files in the repo); plain or K-fold target encoding (stage 1 measured the leak: 0.8462 vs 0.8590 ordered: causality is load-bearing, guide 13 derives why); xgboost-style native handling (its own toggle loses on 2 of 3 datasets); engine-side stage 2b for now (per-tree permutations + crossed-category statistics are catboost's remaining amazon edge, +0.030, and crossed-TS *preprocessing* is the next cheap probe before any engine work is reconsidered).

## 59. Cross-architecture bit determinism: no fp contraction on the host plane (adopted)

**Decision.** `-ffp-contract=off` for all host C++ (CMake directory option; the CUDA kernel TU opts back to `fast`, the device plane's f32-chunk/f64-merge scheme owns its precision story and no cross-platform hash contract covers it). Found when decision 57's midpoint cuts split the California pin by platform (0.71719 arm64 vs 0.71725 x86-64, each internally deterministic): a temporary CI test dumping every cut bit-exactly proved the bin mappers identical across platforms, which cornered the divergence in training arithmetic: clang contracts `a*b+c` into single-rounded fma on targets that have the instruction (arm64) and cannot on baseline x86-64, and the new thresholds put one split decision inside that one-ulp window. Flipping `-ffp-contract=off` on the Mac reproduced the Linux value exactly, confirming the mechanism. Measured cost: nil (interleaved 5-rep California 2000-iter fits: 1.736 vs 1.738 s; the hot paths are adds and divides, not fused-multiply-add shapes). This upgrades the determinism contract from thread-count invariance (decision 32) to **the same model bits on any host architecture**: no reference library makes that claim. The eval pin and the `model_hash.py` baseline are now platform-independent by construction (baselines refresh with this change; the campaign quality tables were measured pre-flag on arm64 and move at rounding level only).

**Rejected.** Widening the pin margin to a platform band (surrenders the pin's one-ulp sensitivity and the reproducibility story); per-platform pin values (documents the symptom, keeps the disease); `-ffp-contract=off` on the CUDA TU too (fma is the native device op; nothing gates device bits across platforms, and the device/host boundary is already tolerance-checked where it must be).

## 60. Issue #72 resolved: the "cross-arch divergence" was OpenMP build variance (adopted)

**Decision.** Parallel training is **bit-identical across host architectures**, proven, not just claimed: with matched builds, arm64 and x86-64 produce byte-equal models at every thread count tried (t1 `f35340da495343c1`, t2 `9e2c081bd5e3fcfe`, t4 `fb5692c2aebea4fe`, t8 `afc31f746baaafb7`, sampled default `8c2c375a331a1bb7`; per-block root-histogram cell bits diffed to zero on a 2-thread minimal artifact). Issue #72's "architecture divergence" was a phantom with a real cause: `find_package(OpenMP)` fails on a stock Mac (homebrew libomp is keg-only, the llvm kegs ship no OpenMP), the failure was a silent `STATUS` fallback to a serial build, and serial builds train different (valid, but not build-reproducible) bits than parallel ones because the fill plan's block counts scale with `parallel::n_threads()` (the documented fixed-N contract). Worse, the Makefile's python configure sent its output to `/dev/null`, so which build you got was invisible. Three fixes: (1) CMake hints homebrew's keg-only libomp explicitly on macOS; (2) OpenMP-not-found is now a **hard configure error**: a build variant that changes model bits must never be silent; serial stays available as an explicit `-DBONSAI_OPENMP=OFF`; (3) the python configure's OpenMP/error lines are no longer swallowed. The cross-arch CI gate now asserts **both** serial and parallel hash equality across the runner pair. Model bits are henceforth a pure function of (input, config, configured thread count), on any machine.

**Rejected.** Keeping the silent serial fallback (it manufactured issue #72 and cost a night of ulp-level forensics (a fill-plan bit dump, thread/iteration ladders, and an on-pod x86 bisection) to exonerate arithmetic that was never guilty); making the fill plan thread-count-independent (a fixed global block count either starves big hosts or taxes small ones; the fixed-N contract is the documented, measured trade); chasing the third hash family observed on one destroyed local build and one opaque runner configure (the loudness fix eliminates the class, silent variants can no longer exist).

## 61. The populate round: software prefetch closes the 16M CPU gap (adopted)

**Decision.** One change, priced from the ledger before it was written: the fill row loop prefetches the mirror strip and grad/hess pair sixteen rows ahead (`run_fill`, `src/grower.cpp`). Below the root a node's rows are an ascending *subset*, so successive row-major strips sit at irregular strides the hardware prefetcher cannot follow; the 16M×100 ledger showed the loop DRAM-latency-bound: populate 82.1s of a 107.4s fit (row loop 78.3s), everything else ≤ 12s. After: row loop 45.5s (−42%), fit **75.8s, a dead tie with xgboost-hist's 75.7s, same pod, same session** (was 1.42× behind). Reads only, models byte-identical (hash gate `f35340…`/`8c2c37…` unchanged, r² to four decimals), 4M improves 30.2→24.3s. This closes the largest visible CPU loss on any chart.

**Rejected.** The same prefetch in the partition passes (measured: 75.85s vs 75.76: noise; refutation logged and the change reverted, per the doc-16 rule that unmeasured complexity does not ship); constant-hessian cell elision (the hess add lands on the already-loaded cache line: priced as ~free before any code); prefetch-distance tuning beyond the first guess (the first guess hit the ledger's floor; further squeezing trades maintenance for noise-level gains).

## 62. The GPU find-kernel round, abandoned by instrumentation; the 16M GPU frontier belongs to catboost (adopted)

**Decision.** No GPU kernel-optimization round ships, and the docs stop implying a large-scale GPU speed crown over catboost. Both conclusions are measurements, not opinions.

The round was scoped to speed up the device find kernel (all-double warp scan, suspected FP64-bound on an L40S). Instrumenting first (a profile-gated sync splitting the find lap into kernel-compute vs device→host transfer (`find_kern_s`/`find_d2h_s` in `histogram_engine.cu`, profile path only)) showed the find kernel costs **0.17s** at 16M. The "8.4s find" in the grow profile is the profiler's opening `cudaDeviceSynchronize` in `find_splits_many` catching the *previous* level's asynchronous histogram kernels; the find scan itself is negligible. The genuine ~8s GPU cost is the histogram accumulation, which already sums in float shared memory (double only for the bounded cross-chunk merge), so the obvious precision lever is spent. Every banked hypothesis was refuted by one measurement: the instrument-first pass is the deliverable, turning a speculative multi-hour rewrite into a measured no-go.

That reframed the question from kernel speed to the whole **accuracy-vs-time frontier** at 16M ([benchmarks/gpu-pareto-16M-2026-07.md](../benchmarks/gpu-pareto-16M-2026-07.md), all one pod). The honest result: bonsai *strictly dominates* xgboost-GPU (reaches xgboost's 100-iteration accuracy 0.8776 in 30.16s vs 36.69s, and beats it at every matched-accuracy point), but **catboost owns the frontier at every accuracy above ~0.875**: catboost@150 (28.35s, 0.8892) is faster *and* more accurate than bonsai `cuda_depthwise`@100 (30.16s, 0.8776). Decomposed against the structural match (bonsai's symmetric `cuda_oblivious`), catboost's lead is *two independent gaps*: a per-round **speed** gap (~15–20%, its tuned symmetric-tree kernel) and a per-round **convergence** gap (+0.011 test r² at 100 iters, same tree shape, its ordered boosting corrects the prediction-shift bias in ordinary gradients). The earlier "bonsai is more accurate than catboost" reading was a fixed-100-iteration artifact that the frontier dissolves. Closing this needs *both* a histogram-kernel rewrite *and* an ordered-boosting-style change to the core booster; winning either alone leaves the other gap standing.

**Consequence.** bonsai's crown over catboost rests on the other axes: bit-identical cross-architecture determinism (§59–60, no competitor has it), a third of the host memory, an ~1,800-line engine, chance-band categorical parity via preprocessing (§58), and best-of-field CPU quality on 9 of 10 real datasets, not on large-scale GPU throughput, where it is strictly ahead of xgboost and roughly 20% behind catboost.

**Rejected.** A histogram-kernel memory/atomic-contention rewrite under the promo deadline (real project, not a knob); quoting the fixed-100-iteration cell as evidence bonsai out-accuracies catboost (the frontier refutes it); any framing of "strictly superior to all three libraries on performance" (unsupported: catboost wins this cell on speed).

**Corrected by §63.** The "convergence gap" attributed to catboost's ordered boosting above was wrong. The feature-admission ladder refuted ordered boosting (a wash vs `boosting_type=Plain`, which catboost uses at scale anyway) and bin quality, then isolated the +0.011 to a *bonsai bug*: the GPU oblivious level-find still vetoed level candidates on infeasible nodes, a defect the CPU had fixed (issue #60). Patched, GPU `cuda_oblivious` matches its CPU twin and catboost's accuracy exactly. Only the ~19% per-round *kernel-speed* gap is real: a bounded optimization target, not an algorithmic disadvantage. Study: [benchmarks/catboost-scale-edge-2026-07.md](../benchmarks/catboost-scale-edge-2026-07.md).

## 63. GPU oblivious lost accuracy at scale to a missing issue-#60 port, not to catboost's algorithm (adopted)

**Decision.** Port the CPU level-find's issue-#60 behavior to the device `level_find_kernel`: an infeasible frontier node (`child hess < min_child_hess`) contributes its parent score (zero gain) instead of vetoing the whole level split. The kernel previously computed `warp_all(feasible)` across the frontier and dropped any candidate with one infeasible node: the *pre*-issue-#60 pathology (decision 56) that at depth ≥ 5, where some node is always near-empty, rejected every good deep cut. The veto, its device scratch (`level_feas`), and the now-unused `warp_all` helper are removed.

**Why it was found.** `cuda_oblivious` scored 0.8638 test r² at 16M vs its own CPU grower's 0.8749 (and catboost's 0.8751), while `cuda_depthwise` matched CPU (0.8776 vs 0.8782). Precision was ruled out (device histogram cells are double, *more* precise than the CPU's float); ordered boosting and bin-sample quality were ruled out by the feature-admission ladder. That localized the loss to the oblivious device level-find, and the code diff to the CPU was the veto. The bug hid because small-data GPU-vs-CPU oblivious tests never produce a near-empty node; the new test forces one with a high `min_child_hess` at depth 7 (fails pre-fix, GPU −0.019 vs CPU 0.324; passes post-fix).

**Consequence.** Same pod, 16M: `cuda_oblivious` 0.8638 → **0.8749**, matching CPU to the fourth decimal and catboost's 0.8751: the accuracy gap that read as a convergence disadvantage (§62) was ours all along, and every `cuda_oblivious` user at depth ≥ 5 was silently getting worse models at scale. The honest residual against catboost is now a single ~19% per-round kernel-speed gap (0.238 vs 0.292 s/iter), which is efficiency, not algorithm. Full `[cuda]` suite green (125,864 assertions); no CPU model bits touched, so the cross-arch hash gate is unaffected (the device plane is tolerance-equal, not bit-equal, by design).

**Rejected.** Closing the residual kernel-speed gap in the same change (a separate, bounded optimization: measure the oblivious kernel's occupancy and launch overhead first, per decision 62's instrument-first rule); leaving the veto with a comment (it silently degraded every deep oblivious GPU fit: a correctness bug, not a tuning choice).

## 64. One shared row sample for binning, not one reservoir pass per feature (adopted)

**Decision.** `BinMappers::fit` draws a single seeded row sample for the whole matrix and gathers each feature's values at those rows, instead of each feature independently reservoir-sampling its own column. The old `create_subsample` ran `std::ranges::sample` over a NaN-filtering view (an O(n) pass) once per feature; at 16M×100 that was 100 passes over 16M rows, and the `BONSAI_INGEST_PROFILE` lap put mapper-fit at **8.45s on the Mac / ~5.7s on an L40S**, pure serial host time before any training (and time catboost, which bins on device, does not pay). The shared sample runs the O(n) selection once, then each feature does an O(n_samples) gather: **mapper-fit 8.45s → 0.35s at 16M (24×)**, `r²` unchanged to four decimals (0.8782 both): the bin-sample study (§ the [scale-edge note](../benchmarks/catboost-scale-edge-2026-07.md)) already showed sample count is quality-neutral past ~200k rows, so a different-but-equivalent sample moves nothing.

**Determinism and scope.** Bit-identical for any dataset that fits the sample: `n_rows ≤ n_samples` takes the whole-column path unchanged, so every small-data test and the **serial** model hash (`model_hash.py` runs 500k rows with `n_samples=500000`) are untouched. Only fits above `n_samples` get new cuts: the *default* (sampled, 8-thread) hash changes, superseding decision 60's `8c2c375a331a1bb7`; the serial `f35340da495343c1` stands. `std::ranges::sample` over an `iota_view` uses selection sampling with `mt19937` (deterministic and architecture-independent) so the cross-arch gate (which compares arm==x86 dynamically, no hard-coded baseline) still holds, and the fixed-thread-count contract (decisions 59–60) is preserved. Two tests pin it: a >`n_samples` fit is reproducible cut-for-cut, and the `ColumnBatch`/`features_view` overloads agree.

**Consequence.** Measured same-pod at 16M (L40S, 100 iters): `cuda_oblivious` **19.65s vs catboost 18.85s** at matched accuracy (0.8749 vs 0.8751), a ~4% residual, down from the ~19% §63 flagged, with mapper-fit falling from ~5.7s to 0.77s of the fit. Combined with §63's accuracy fix, bonsai's GPU oblivious went from *slower and less accurate* to *even on accuracy, within 4% on speed*. Every large CPU fit gets the same ~8s ingest saving. It is the standard lightgbm/xgboost approach (sample rows once); bonsai had simply not adopted it.

**Rejected.** Preserving the exact per-feature reservoir bits (would keep the 100× O(n) cost for no quality gain, since the sample is quality-neutral); a bit-preserving fast path only for NaN-free columns (fragile: depends on `std::ranges::sample`'s implementation-defined selection matching a hand-rolled reservoir, and still O(n) per feature to detect NaN).

## 65. Reusable pre-binned Dataset with a sealed bin config (adopted)

**Decision.** PR #91 adds a Python-level `bonsai.Dataset(X, y, weight=None, max_bin=255, n_samples=200000, seed=0, min_data_in_bin=1)` that runs `BinMappers::fit` + `Dataset::bin` once at construction, then feeds the same `bonsai::Dataset` to every `train(params, dataset)` call. A hyperparameter sweep or CV loop skips the per-fit bin pass entirely. Verified bit-identical to fitting from `(X, y)` directly.

**Consequence.** Because the object address is now stable across fits, the CUDA resident-matrix upload-skip cache (decision 54's `ensure_dataset`) actually fires for the first time: previously every `train()` built a fresh `Dataset`, so the cache never hit. Lifetime is simple: the wrapper pins the numpy `X` buffer, and `FeatureBuffer` borrows it row-major with no float materialization.

**Sealing the bin config.** `bin_mapper.*` overrides via `params` are rejected by key-prefix at the Python boundary. The harder case is a config *file*: value comparison can't distinguish an absent section from one explicitly restating defaults, so a file setting `max_bin = 255` against a `Dataset` built with `max_bin=63` must still error rather than silently discard the Dataset's binning. Two follow-up commits on PR #91 closed this: `f899e52` added a structural check, `config::toml_has_section()`, that tests for section presence rather than value equality, and a further pass made the guard general and exposed `min_data_in_bin`.

**Rejected / deferred.** Disk persistence, row-subsetting for CV, and `fit(dataset)` on the sklearn estimators are deliberate non-goals of this MVP. The `eval_set`/early-stopping path in `train(dataset)` was a known gap in the MVP, tracked and closed separately (see branch `feat/dataset-eval-set`).

## 66. Prebuilt wheels from native runners, not manylinux containers (adopted)

**Decision.** PR #94 + #98, shipped in v1.2.0: 15 wheels + sdist attach to the GitHub Release automatically on `release: published`. Matrix is {ubuntu-22.04, ubuntu-22.04-arm, macos-14} × py{3.9–3.13}, CPU-only.

**Why not cibuildwheel/manylinux.** bonsai requires LLVM ≥ 20 with libc++ (C++23 `std::print`/`std::mdspan`). The official LLVM release binaries need a newer glibc than the `manylinux_2_28` container ships, so wheels build on plain runners with apt.llvm.org LLVM 21 (Linux) / brew `llvm@21` (macOS) instead. `auditwheel`/`delocate` vendor libc++ (+ libc++abi/libunwind) into each wheel; OpenMP is statically linked via the pre-existing `BONSAI_OPENMP_STATIC`, so zero C++ changes were needed to make this work. `auditwheel` tagged the result `manylinux_2_34` (dual-tagged `2_35`): actual glibc floor 2.34, so RHEL 9/Alma 9 are covered, a better floor than the Ubuntu 22.04+ design target implied.

**Python floor.** Lowered to 3.9 in PR #98 (concrete user use case, reversing an earlier decline). Four touchpoints, not one: `requires-python`, the CMake `find_package(Python)` floor (a pyproject-only audit missed this; the local 3.9 build caught it), the wheels matrix (now 15 legs), and `ruff target-version=py39` so 3.10+-only syntax can't regress in.

**Verification.** Every wheel installs into a clean venv and runs a smoke script: regressor fit, classifier `predict_proba`, `Dataset` train, and (added after PR #95 found `from_file` broke on Python 3.10 because `tomllib` is 3.11+ stdlib and the original smoke test never called it) a save/`from_file` round-trip across the whole version matrix.

**Consequence.** PyPI publish is deliberately deferred until trusted publishing is registered (`bonsai-gbt` name + pypi.org trusted publisher, an owner action); wheels ship via GitHub Releases in the meantime. CUDA wheels are out of scope here and tracked as issue #99: the design is a fat-arch linux x86_64 leg plus a rented-GPU validation gate before assets attach, since GitHub-hosted runners can build CUDA but never execute it.

**Rejected.** A `manylinux_2_28` container image to reach older distros (a documented follow-up, not blocking; no user has asked); PyPI publish in this PR (needs the owner-side trusted-publisher registration first).

## 67. Automatic per-feature bin budgets: declined by measurement; explicit user edges: open design question (partial)

**Decision.** No engine API for per-feature bin budgets or user-supplied edges (issue #63's residual; lightgbm ships `max_bin_by_feature`, catboost ships `per_float_feature_quantization`). Priced at zero core cost by `scripts/probe_binning.py` on five datasets at campaign knobs: budgets emulated exactly through the issue-#61 one-cut-per-distinct rule for bonsai, and priced natively via lightgbm's own toggle. Best observed gain anywhere is +0.0011 (synthetic with known signal structure, unlimited extra budget), at the decision-55 chance band; the importance-guided policy loses outright on adult (−0.0011 AUC) and kick (−0.0096), and lightgbm's own toggle is ≤ +0.001 at best and negative on kick. The inverse-allocation control costs up to −0.084 r², so budgets are causally live but asymmetric: uniform 255 sits at saturation and reallocation can only break things. The headroom policy on MSD (−0.0003) also kills the hypothesis that decision 55's +0.001 cut-quality residual is a resolution-allocation problem. Full table in `benchmarks/binning-tradeoff-2026-07.md`.

**Consequence.** What is declined is the ACCURACY-motivated feature: automatic importance-driven budget allocation buys nothing at the 255-bin default and misallocates on real data. What stays open is the CAPABILITY: user-supplied explicit bin edges are a deployment-artifact question the probe never measured. The pre-discretization emulation is bit-exact at train time but leaves the binning OUTSIDE the model artifact: every serving path must re-apply the transform forever, and predict on raw features silently produces garbage. A native `Dataset(bin_edges=...)` would bake edges into the per-feature BinMappers, which already serialize into the model, so predict round-trips on raw values; the design is specified in architecture doc 18 and admission-gated on a concrete workload. The default also gained evidence: uniform `max_bin = 255` is robust in a way per-feature schemes are not.

**Rejected.** An engine API "because lightgbm and catboost have one" (their own measurements above justify declining it); shipping ManualBinner as a package class (unlike the categorical encoder, the preprocessing won nothing, so it stays a documented recipe, not an API surface).

**Reopen the accuracy branch if:** a memory-constrained regime forces the global budget down (max_bin ≤ 64, where saturation no longer holds and allocation could matter; never measured). **Admit the capability branch when** a concrete workload needs domain-mandated bins in the model artifact (regulatory bands, clinical thresholds, reproducing an existing scheme); doc 18 prices it.
## 68. The Grinsztajn benchmark becomes the standings suite; bonsai has the best mean rank (adopted)

**Decision.** Adopt the 55-task Grinsztajn et al. (2022) benchmark (OpenML suites 297/298/299/304) as the external standings suite, replacing the hand-picked internal ten as the citable table (`scripts/run_tabular_suite.py`, `benchmarks/grinsztajn-2026-07.md`; the internal campaign stays as the fast smoke tier). At the paper's medium protocol and campaign-matched knobs across 990 fits: bonsai has the best library mean rank, 2.04 vs xgboost 2.11, lightgbm 2.53, catboost 3.33, with the most consistent profile in the field: second place or better on 44 of 55 datasets and last exactly once. xgboost keeps the most outright wins (26 vs bonsai's 10): it is the peak library at small-data knobs, bonsai the consistency library. bonsai leads categorical regression outright (mean rank 1.77).

**Why external.** A self-picked suite invites a selection-bias objection that no honesty of execution can answer; a published benchmark selected by third parties removes it. The claim format also improves: distributional (mean rank, head-to-head, rank distribution) instead of a win count over ten.

**Caveats recorded with the numbers.** The 10k-row cap is decision 55's regime (xgboost's cut-quality edge at its strongest); ordinal codes strip catboost's categorical machinery (uniform convention, undersells catboost on categorical tracks); depthwise and leafwise coincide at these knobs so library-level best-variant ranking is used.

**Rejected.** OpenML-CC18 (image-flattened datasets dilute the tabular story); TabZilla-176 (volume over curation); running references on GPU for wall-clock (GPU paths change reference numerics and reproducibility; accuracy standings are hardware-independent). TabArena submission (the living leaderboard with external protocol) is the post-promo follow-up, not rejected.

**Correction (same day).** The first run hardcoded xgboost's `min_child_weight = 1` instead of the campaign mapping (`= min_data_in_leaf = 20`, `scripts/reference_params.py`), giving xgboost ~20x smaller leaves than the other libraries were allowed; under that skew xgboost ranked 2.11 with 26 outright wins. Re-run at the campaign mapping: bonsai mean rank 1.73 with 27 outright wins, lightgbm 2.35, xgboost 2.73, catboost 3.20; bonsai >= xgboost on 42 of 55. Because min_child_weight is hessian-weighted (20 implies ~80+ rows per leaf on classification, harsher than the others' 20 rows), the two conventions bracket xgboost and the claim kept is the one that holds at both ends: bonsai has the best mean rank under either. The remaining named losses to xgboost, `year` (+0.0066 r²) and `yprop_4_1` (+0.0053), are the decision-55 residual with real datasets attached. Sensitivity rows preserved in `results/grinsztajn-2026-07-xgb-mcw1.jsonl`.

## 69. The benchmark charter: bonsai.bench in the wheel, two divisions, one row schema (adopted)

**Decision.** The benchmark harness moves inside the Python package (`bonsai.bench`: params, metrics, synth, runlog, datasets, and the grinsztajn/scaling suite runners), shipped in the wheel behind a `[bench]` extra so `pip install bonsai-gbt[bench]` reproduces the published tables; `import bonsai` stays numpy-only via lazy imports. Every result row is one of two closed divisions: quality (metric primary, timing never citable) or perf (timing_mode mandatory: `in_memory` vs `pipeline`), self-described by row schema v1 (division, suite, knobs + hash, metric/value, timing_mode, git sha, full host capture). Knob sets (CAMPAIGN, SCALING) and the two declared lightgbm leaf conventions live only in `bonsai.bench.params`: hand-re-deriving reference mappings caused decision 68's published correction and is now structurally prevented and test-pinned. The normative rules are `docs/method/benchmark-protocol.md`; migration verified by a byte-identical model-hash gate, byte-stable Friedman goldens, and a zero-mismatch replay of committed grinsztajn rows.

**Rulings.** Completed probes are provenance and stay as-run (annotated, never refactored onto the new library); `bench_categorical`'s deviating knobs stay annotated rather than rerun; benchmark data stays in `tests/data/` (moving it would touch configs, Makefile sentinels, and CI caches for zero mechanical gain) with the registry as machine truth and the test-pin fetchers kept build-independent for CI; superseded results files are deleted, git history being the archive (`rebaseline.jsonl`/`.md` removed, superseded by the dated re-baseline).

**Rejected.** A repo-local scripts/benchlib (users could not reproduce tables from an install); moving data out of tests/data; regenerating any committed row under the new schema (old and new rows coexist; readers tolerate both).

## 70. CUDA wheels: fold-in, static cudart, and a rented-GPU release gate (adopted)

**Decision.** The linux x86_64 release wheel ships the CUDA backend (issue #99): SASS for `sm_70;sm_75;sm_80;sm_86;sm_89;sm_90;sm_120` plus a compute_90 PTX forward-JIT floor (sm_100 JITs; `BONSAI_CUDA_ARCH` became a list, `BONSAI_CUDA_PTX_ARCH` pins the PTX), built with CUDA 12.8 (12.6's ptxas cannot emit sm_120; clang cannot target 13) and `CUDA::cudart_static` (`BONSAI_CUDA_STATIC_RUNTIME`), so the extension carries zero CUDA `DT_NEEDED`s, auditwheel grades manylinux_2_34, and GPU-less machines import a de-facto CPU wheel. Measured on the leg-0 probe before commitment: the whole backend costs **2.33MB of wheel** (control 1.27MB; xgboost's GPU wheel ~300MB) and **~5s of build** for six extra arches; no CUDA runtime symbols exported. Because GitHub runners can build CUDA but never execute it, releases gate through one rented L40S session that boots the candidate runtime image (`docker/runtime.Dockerfile`, wheel baked in, published as `ghcr.io/daniel-m-campos/bonsai:{<tag>-cuda,cuda}` after the gate): on-pod `wheel_smoke_cuda.py` (both cuda growers, pre-registered 1e-3/0.005 parity bands, byte-stable save/load round-trip) plus `model_hash.py`, whose output must equal CI's, extending the decisions 59/60 byte-identity contract to shipped artifacts on real GPU hardware; the wheels workflow's hash-compare job asserts the same triple equality (macos-arm, linux-arm, linux-x86-with-CUDA) on every wheel build. Teardown is unconditional with a loud leftover-pod sweep; a red gate withholds only the CUDA artifacts and the CPU release stands. Driver floor documented as R525 (CUDA 12 minor-version compatibility); the gate exercises 12.8-static-cudart on RunPod's 12.4-driver hosts every release by construction.

**Rejected.** Vendoring dynamic libcudart via auditwheel (a second mangled runtime in-process beside torch/xgboost); an `nvidia-cuda-runtime-cu12` pip dependency (mandatory NVIDIA download for CPU-only users); a separate `bonsai-gbt-cuda` distribution and `+cu12` local tags (pre-registered fallback if the wheel had exceeded 50MB, it measured 2.33; local tags are PyPI-illegal); building wheels inside the `bonsai-ci` image (its 12.4 pin is the RunPod container-start driver ceiling, its ptxas cannot emit sm_120, and decision 66 already rejected container wheel builds); toolkit 12.6 without sm_120 SASS (registered as the fallback if 12.8-on-R550 minor-version compatibility fails at the gate; it did not).

## 71. The post-fix 16M GPU frontier: split by budget, ceiling recovered (adopted)

**Decision.** The 16M accuracy-vs-time frontier was re-measured on current main (2026-07-14, same-pod L40S, `scripts/gpu_pareto.py` with ladders extended into the deep end), superseding the pre-fix 2026-07-12 run whose bonsai points carried both the decision-63 oblivious accuracy defect and pre-populate-round fit times. The post-fix frontier decomposes cleanly into fixed cost plus marginal cost per round: bonsai `cuda_oblivious` ~4.6s + 155ms/round, catboost ~11.8s + 77ms/round, xgboost ~22.3s + 58ms/round. Consequences, recorded as the standing verdict: **bonsai owns the fast end** (lowest fixed cost; fastest to every accuracy up to roughly r² 0.88), the curves cross at the ~100-round operating point (20.1s/0.8749 vs 19.5s/0.8751, the scaling tables' measured tie), **catboost owns time-to-accuracy in the deep end** (0.8973 in 35.1s vs bonsai's 0.8974 in 51.9s) through its 2x cheaper marginal round, and **the accuracy ceiling now belongs to bonsai** by a rounding digit (0.8981 vs 0.8980 at 450 iters), a region the pre-fix defect made unreachable. Validation of the supersession itself: the depthwise ladder reproduces the old run's r² to four decimals (no fix touched it; determinism across pods), the oblivious ladder shows the defect's removal point by point (0.8638 to 0.8749 at 100 iters), and bonsai times dropped ~28% while reference libraries moved only with fleet variance. Evidence: `benchmarks/gpu-pareto-16M-2026-07.md` + jsonl (replaced in place, git history is the archive per decision 69); the results-ledger chart now shows this run. The named next perf target if the deep end becomes load-bearing: bonsai's 155ms marginal round vs catboost's 77ms.

**Rejected.** Keeping the pre-fix frontier beside the new one (decision 69: superseded results are deleted, not attic'd); extending the README's fixed-iteration "fastest GPU slot" claim to the whole frontier (false above r² 0.885 and the claims table stays exact); chasing the marginal-round gap now (the 100-round operating point is the standing benchmark regime and is a tie; the deep end has no user yet, so the gate applies).

## 72. The marginal-round campaign: 155 to 104 ms, the frontier taken whole (adopted)

**Decision.** Decision 71's named target (bonsai's 155ms oblivious marginal round vs catboost's 77) was attacked instrument-first per doc 16 and closed in two stages (PR #148) plus this close-out. Stage 0 built the price list before any lever: a profile-only sync peel replayed the decision-62 misattribution (6.1s filed under `lfind_stage` was the previous level's async histogram kernels draining at the next sync), cudaEvent pairs priced the async build directly, and conservation flushed two residues nothing else explained (make_root's 64MB/tree host identity copy at 33ms/round; a final-level histogram build for children that are leaves at 22ms/round). Stage 1 landed the three levers the table priced above ~10ms: the **identity contract** (full-data fits pass empty rows + row_count; the engine iota-builds and caches identity on device), **deterministic device root sums** (fixed-grid two-pass reduce replacing a 16M-row host loop), and the **final-level skip** (`advance_layout_only()` keeps only the segment flip stamping needs). Three levers were killed by their pre-registered criteria at a combined price under a millisecond (epilogue sync scope 0.1ms, per-level memset 0.5ms, pinned gh staging break-even). Same-pod: round 181→125ms, fit 19.43→13.88s, r² four-decimal identical, CPU model hash byte-identical.

**Frontier verdict (2026-07-15 re-run, same-pod L40S US-NC-1, superseding in place per decision 69).** bonsai `cuda_oblivious` ~3.4s + 104ms/round, `cuda_depthwise` ~3.1s + 121ms (the levers were engine-level, so depthwise fell from 187ms too), catboost ~12.4s + 76ms, xgboost ~21.6s + 65ms. The bonsai-catboost crossover moved from ~100 rounds to ~320, which sits inside both libraries' accuracy plateaus: bonsai is now first to every measured accuracy up to r² ~0.895 (0.8749 in 13.9s vs 19.7s; 0.8948 in 24.5s vs 27.9s), the 300-iter points are a statistical tie (0.8974/35.3s vs 0.8973/35.1s), and the ceiling stays bonsai's (0.8981 vs 0.8980). Accuracy reproduced to four decimals at every shared point, the campaign's behavior-preservation contract measured from the outside. The ship bar (≤110ms) was met; the crown bar (≤77ms, kernel parity) was not, and did not need to be: its premise was that the crossover mattered inside the useful range, and the fixed-cost advantage plus the 33% marginal cut pushed the crossover past it.

**Stage 2 (histogram-kernel engineering): not spent.** The plan's own gate asked for honest histogram compute above ~80ms/round with parity otherwise unreachable; the event-timed build is ~72ms and parity turned out unnecessary for the frontier. The floor is recorded instead: the 104ms round is ~72ms histogram kernel + ~32ms partition, bus, and per-level residue. Reopen if a workload lives at 450+ rounds of this cell, where catboost still reaches its (lower) plateau about 3s sooner.

**Consequence.** `scripts/dag_model.py` and doc 16 were refreshed from this pod's profiled cells: the old `find 7.62s` node is retired (the find kernel is 0.13s; the weight was always the histogram build, now its own event-timed node at 7.9s/fit), conservation closes at fit level (est 14.9s vs measured 15.70s), and the all-device floor is ~13.9s with the histogram build at 70% of device compute. Guide 11 gained the campaign's rows and the one-round summary of the method. The results ledger regenerates from the superseded-in-place artifacts.

**Rejected.** Stage 2 now (gate not met, crown achieved without it); keeping the pre-campaign frontier beside the new one (decision 69); quoting the 104ms as a cross-pod absolute (fleet spread is ~25%; the number that transfers is the decomposition shape and the same-pod deltas).

## 73. Explicit bin edges ship as a Dataset capability (adopted)

**Decision.** Doc 18's design is built: `bonsai.Dataset(X, y, bin_edges={col: edges})` bins listed columns at user-supplied interior cut points; unlisted columns fit as usual. The plumbing is exactly the constructor-not-mechanism the doc predicted (`BinMapper::from_edges` beside the loader-trusted `from_cuts`; a `BinEdges` override on both `BinMappers::fit` overloads with edge mappers seeded before the parallel region so validation errors never cross it; one `bin_edges` argument on the Python `Dataset`), no model-format bump, no hot-path branches, no CUDA change. The admission gate is recorded as met by roadmap signal: decision 67 asked for a workload that needs domain-mandated bins inside the artifact, and the owner ranked the capability onto the roadmap; accuracy remains a non-claim (decision 67 measured saturation).

**The one thing the design missed, found by the acceptance test.** The first implementation appended only the `+inf` sentinel, and the acceptance test's cross-band assertion failed: bands above and below the LAST edge were inseparable. Root cause is an engine convention, not a bug: the split scan (`histogram.hpp cut_cells`) never offers the last real bin as a candidate because for fitted columns that cut is degenerate (the observed maximum defines it). For user edges the band above the last edge is a domain statement, so `from_edges` appends a `FLT_MAX` cut to close it as a real bin plus the `+inf` sentinel to keep the missing bin NaN-only: k edges give k+1 splittable bands, NaN routes to its own bin, and raw `+inf` inputs land in the missing bin exactly as they do for fitted columns. Edges must be finite and below `FLT_MAX` (reserved), strictly increasing, non-empty; bad column indices and duplicates throw `ConfigError`.

**Verification.** The acceptance test is the one the decision-67 emulation structurally cannot pass: fit with domain bands, predict on RAW values (within-band invariance, cross-band separation, right-inclusive edge membership), save/load round-trip predicting byte-identically, all with no external transform. Byte-identity gates: `model_hash.py` unchanged (serial f35340da495343c1, sampled 8c2c375a331a1bb7), full suite 38,296 assertions green, overridden-absent fits bit-identical by construction (the override path is never entered).

**Rejected.** Validating inside `from_cuts` (the loader's trusted path stays branch-free; user input is validated at its own named entry); a config-file/TOML surface for edges (arrays of floats per column do not fit the flat dotted-key override grammar; the Python `Dataset` is the artifact-construction door and the CLI can gain one later if a workload asks); exposing the emulation as a package class (decision 67's ruling stands, superseded by the native capability).

## 74. The FLT_MAX top-band closer: the missing bin is NaN-only on every path (adopted)

**Decision.** `create_cuts` now appends a `FLT_MAX` cut before the `+inf` sentinel on every fitting path (issue #155), so every finite value above the last placed cut bins into a real, splittable top band instead of the NaN sentinel. This closes three leaks the fitted paths carried: the stride path's top tail (~one mean-bin of rows), the greedy path's final group (an entire heavy run when the maximum is a capped value), and rows above the 200k-sample maximum on any path. It also removes a train/predict routing skew: leaked rows trained down the learned `default_left` branch but predicted by raw threshold comparison (right of everything), so the same row routed differently in training and deployment; the valid/early-stopping path scores raw and was already on the deployment side. The budget slot was already reserved (`from_sample`'s `max_bin - 2`), so finite cut placement is untouched and no bin count exceeds `max_bin`. The invariant now matches decision 73's edge columns: **no finite value ever bins as missing, anywhere**, and the mechanism (the closer cut) is the same one `from_edges` uses.

**Evidence (scripts/probe_missing_bin.py, campaign knobs, both growers).** The mechanism synthetics are decisive: capped-max heavy value r² 0.483→0.988 (the capped 10% carried signal and trained as missing), rare top-tail signal +0.16, signal above the sampled max at 1M rows +0.008. The real suite is chance-band flat: higgs ±0.0001, airline ±0.0006, california ±0.001 with opposite signs per grower, and amazon's −0.002 sits inside that dataset's own 0.0034 spread across `max_bin` 253-256 with no closer at all (the null-perturbation yardstick; threshold-placement churn on ID codes). Full table in `benchmarks/missing-bin-closer-2026-07.md`.

**Why adopt (contrast with decision 67).** The probe suite happens to contain no capped columns; the wild does (clipped sensors, capped amounts, 999-coded maxima), and there the failure mode is works-vs-doesn't, not a tuning delta. The fix costs nothing at runtime, spends a slot that was already reserved, removes a correctness asymmetry rather than chasing accuracy, and unifies the sentinel semantics across fitted and user-edge columns.

**Standings re-validation (the surprise).** The citable Grinsztajn table was re-run bonsai-arms-only against frozen reference rows (same protocol, seeds, and knobs; runner identity proven by the decision-69 replay). Expectation was chance-band; the result was not: mean rank 1.73 to **1.44**, outright wins 27 to **36** of 55, second-or-better 44 to **50**, last-place finishes 1 to **0**, head-to-head >= lightgbm 37 to 46, >= xgboost 42 to 48. The honest decomposition: per-dataset values mostly moved inside 0.001 (43 of 55; 8 improved beyond it, 4 regressed, worst house_16H −0.0040), but the 10k-row cap is the regime where the greedy path is nearly universal (mean bin ~39 rows, so 461 of 495 fits changed), the suite was full of photo-finish second places, and the closer's small nudges broke them systematically one way. Rank tables amplify near-ties; the durable movements are the head-to-head counts and never-last. The decision-55 residual also narrowed: `year` recovered +0.0036 of its +0.0066 gap to xgboost. The grinsztajn jsonl's bonsai rows are superseded in place; README and the ledger carry the new table.

**Consequence.** Models change wherever a leak existed: canonical hashes move to serial `09dbf47353033362` / sampled `ca7174cb1560221e` (data digests unchanged; the cross-arch CI gate is dynamic equality and unaffected), and the California pin improves 0.71725→0.7153 (−0.27%, inside the band; lineage in the test comment). New invariant test pins all three paths: observed max and out-of-sample larger values bin below the sentinel, NaN bins into it. An all-NaN column now emits `{FLT_MAX, inf}` (a real bin exists even with nothing observed). A tree may now split real values away from missing at the last cut, a candidate that previously did not exist; small-fixture tests were unaffected because the new candidate is infeasible or zero-gain there.

**Rejected.** Keeping the leak as a documented budget trade (the trade was never priced until now, and the price on capped columns is catastrophic); a config knob for the closer (invariants are not options); closing the leak by clamping transform to the last real bin instead (mimics lightgbm's convention but leaves the top band unsplittable and the skew intact for `default_left` trees).

## 75. Per-node/per-level feature subsampling: declined by measurement (adopted)

**Decision.** No `colsample_bylevel` / `colsample_bynode` equivalent in the grower (issue #45; xgboost ships both, lightgbm ships `feature_fraction_bynode`). bonsai already samples features once per tree (`tree.feature_fraction`), and the feature only earns its plumbing (the grower's selected-features path plus a CUDA find-staging change that today assumes a per-tree selection) if per-node/per-level beats per-tree. Priced at zero core cost by `scripts/probe_feature_subsample.py` on four datasets (8 to 200 features) at campaign knobs, toggling the reference libraries' own knobs. In isolation, per-node/per-level never leads: on california/higgs/year (<= 90 features) every arm sits in the decision-55 chance band or per-tree wins, and on the 200-feature synthetic where feature bagging matters most, per-tree at 0.5 (−4.73 rmse) dominates the best per-node arm (−1.18) by ~4x. Stacked on per-tree (their intended use) per-node adds a genuine −0.6 rmse on the 200-feature case, its one real signal, but a tuned per-tree fraction ALONE beats the stack: `bytree0.4` (89.34) is better than `bytree0.5+bynode0.5` (90.26). Full tables in `benchmarks/feature-subsample-tradeoff-2026-07.md`.

**Consequence.** The regularization per-node offers is reachable, and exceeded, by tuning the single knob bonsai already ships; the feature is a strictly dominated lever, so it changes no standings and unlocks no workload `feature_fraction` cannot already serve. The `max_bin = 255`-style outcome repeats: the existing simpler control is validated as sufficient. No core lines, no new config surface, no CUDA find-staging rework.

**Rejected.** Building it "because xgboost and lightgbm have it" (their own toggles above justify declining it, the categorical/bin-budget pattern of decisions 58/67). **Reopener:** a real many-feature workload where per-node/per-level beats a tuned `feature_fraction` beyond the chance band, none found across 8 to 200 features; bonsai's wide-data strength is the regime to watch, but even synthetic 200-feature data favours the existing knob.


## 76. Data-parallel multi-GPU: built, measured to parity, parked as an experiment (adopted)

**Decision.** The data-parallel multi-GPU engine (architecture doc 19: `MultiCudaHistogramEngine`, the `cuda_multi_*` growers, `parallel.device_ids`) is withdrawn from main and parked on the `experiment/multi-gpu` branch. The track was built through its full plan and validated end to end (correctness held everywhere: identical r2 across single, 2, and 4 GPUs at 16M and 64M rows, both peer and host-staged reduction regimes, on 2x A40 and 4x A100 NVLink hardware), then priced by five optimization levers across four same-pod ladders: parallel per-device fan-out, sliced per-shard finalize copies (which alone took the 64M fit from 184s to 63s), per-shard gradient upload slices, pinned staging, and a double-buffered reduction pipeline. The end state is END-TO-END PARITY with the single-GPU engine (16M: ~16.5s vs ~15.8s; 64M: 58.9s vs 57.5s), against a pre-registered bar of 1.3x at 2 GPUs and 1.8x at 4. The floor is architectural, not a tuning residue: gradients are host-computed per tree, so the gh stream must cross host memory every tree regardless of staging strategy (pinned equaled pageable), and the level reduction's cost is its correctness syncs, not transport (forced host-staged equaled peer on an NVLink mesh). What main keeps: `parallel.device_id` (single-device selection), the `CudaDeviceContext` extraction and header/implementation split (structural improvements to the single-GPU engine), and the tightened `GPULevelEngine` concept.

**Consequence.** The supported multi-GPU story is fit-parallelism: N GPUs run N independent fits (sweeps, CV folds, ensembles) via `parallel.device_id`, which scales linearly by construction and compounds bonsai's existing advantages (a fit's small memory footprint allows more concurrent fits per card; `bonsai.Dataset` bins once for the whole sweep). Capacity is likewise a weaker motivation for bonsai than for any competitor: u8-binned storage puts roughly 500M x 100 rows on one 80GB card, so vertical scaling covers the realistic range. The experiment cost ~$17 of pod time and every conclusion above is a measurement.

**Rejected.** Keeping the engine in main at parity (a checkbox feature that grows the core and the registry for no measured win is exactly what the feature-admission discipline exists to refuse; this track bypassed that gate as a personal-use experiment, and the outcome re-validates the gate); further lever rounds (the two remaining costs were each refuted by a targeted fix, which is the signature of an architectural floor). **Reopener:** a device-resident objective, where each GPU computes its shard's gradients from resident scores, eliminating the host gh stream. That redesign is also the single-GPU engine's own next frontier (the decision-72 marginal-round floor is the same host/device boundary), so it should be pursued for single-GPU wins first, with the parked branch inheriting the result.

## 77. The device-resident objective: the per-tree host round-trip deleted (adopted)

**Decision.** Decision 76's reopener was pursued single-GPU first, as its own campaign (issue #171), and shipped for the MSE objective. When a fit is eligible (an objective with a device gradient, no DART, no sample weights, a sampler that never reads gradient values), the booster arms a resident mode: labels and initial scores upload once per fit, each tree's packed (grad, hess) array is derived on the device from the resident scores by a two-line kernel, and the tree epilogue routes every row through the finished tree in bin space and fuses the leaf value into the resident scores. Per tree, nothing crosses the bus in either direction: the gradient upload, the interleave pass, the values and leaf-id downloads, the host objective loop, and the host score update all cease to exist. Ineligible fits take the host path untouched, `BONSAI_HOST_OBJECTIVE=1` forces it, and the CPU plane is untouched by construction (model hashes byte-identical; the eligibility seam is a compile-time no-op for CPU growers).

**Evidence (stage 0 priced, stage 1 measured; `benchmarks/resident-objective-2026-07.md`).** The instrument-first price list on existing counters put the reachable pool at 35.9ms/round at 16M rows (22% of the fit) and 128ms/round at 64M (20%), clearing the pre-registered kill bar (12ms and 10%) threefold. The same-pod interleaved A/B on the shipped branch then measured cuts LARGER than the pool: oblivious 16M 136.1 to 102.6ms/round (24.6%), depthwise 16M 20.1%, oblivious 64M 16.4%, depthwise 64M 14.9%, because the routing epilogue is also cheaper than the stamp-and-copy epilogue it replaced. r2 identical to every reported digit in all pairs; the full-data resident model is bit-identical to the host-objective GPU model on Jetson parity tests. The ship bar (15ms/round at 16M) was met at 2.2x; the same-share projection puts the decision-72 frontier round near the 77ms stretch bar, to be confirmed by a frontier re-run.

**Consequence.** The decision-72 marginal-round floor moves: the round is now histogram build plus partition plus find, with the objective boundary gone. The complexity ledger, measured per the panel this campaign introduced: 616 non-test lines (about 380 in the CUDA plane, 240 in generic headers), zero new config knobs, zero registry growth, and the fit loop keeps its host shape (the eligibility seam is one named call, `try_resident_round`; the capacity predicate is one shared function both `begin_root` and `resident_begin` apply, so the decline conditions cannot drift apart; the resident state is armed per Dataset and disarms with a sync when the Dataset or a runtime gate changes). One depth-scaling note: the routing epilogue is O(depth) dependent loads per row against the O(1) gather it replaced, measured cheaper at depth 8; if deep-tree fits ever regress there, the reopener is a hybrid epilogue (leaf gather for partitioned rows, routing only for out-of-bag rows). The decision-76 multi-GPU reopener is now live on its parked branch (per-shard gradients from resident shard scores are exactly this mechanism), and remains parked pending a workload. Stage 2 (LogLoss, Poisson, resident sample weights) is justified by these numbers and follows as its own step; Softmax stays host-side (per-class tree shape, its own campaign).

**Rejected.** Extending eligibility to GOSS (it reads and reweights host gradients; a device GOSS is its own design); renewal objectives (MAE, Huber, Quantile renew leaves from host residuals; resident mode would have to download what it just avoided uploading); persisting resident state in the artifact (scores are training state, not model).

## 78. The frontier re-run confirms decision 77: unconditional at 16M (adopted)

**Decision.** The 16M accuracy-vs-time frontier was re-measured on current main (2026-07-18, one L40S pod, `scripts/gpu_pareto.py`, 22 same-pod points), superseding the decision-72 run in place. bonsai `cuda_oblivious` decomposes to ~3.8s fixed + 64ms/round (was ~3.4s + 104), `cuda_depthwise` to ~2.8s + 88 (was ~3.1s + 121), while the same-pod controls moved only with fleet variance (catboost 76 to 78, xgboost 65 to 59), attributing the whole marginal cut to the device-resident objective. The stretch bar decision 77 projected is met on the measurement that matters: bonsai's marginal round is now cheaper than catboost's on the same pod, the decision-72 crossover no longer exists at any measured horizon, and the one residue that run named honestly (catboost reaching its plateau sooner at 450+ rounds) is gone: 0.8979 in 31.9s vs 0.8980 in 46.4s, a fourth-decimal tie at 45% more wall clock.

**Consequence.** The frontier verdict is unconditional for the eligible regime (MSE, no DART, no sample weights, uniform or no row sampling); ineligible fits keep the decision-72 frontier, whose residue was already plateau-depth only. The ledger chart and `benchmarks/gpu-pareto-16M-2026-07.md` carry the superseded-in-place evidence. The remaining round is histogram build plus partition plus find; kernel engineering stays unspent per decision 72's gate, now with a higher bar since no competitor pressure remains at this cell.

**Rejected.** Claiming the fourth-decimal ceiling (fleet noise, either direction); extending the claim beyond the eligible regime before stage 2 lands LogLoss and Poisson.

## 79. Resident LogLoss, Poisson, and sample weights: the eligible regime widens to the common case (adopted)

**Decision.** Stage 2 of the resident-objective campaign (issue #171) extends the device-resident objective from MSE to LogLoss and Poisson and admits per-row sample weights, closing the campaign. The gradient kernel is templated on the objective kind and a weighted flag (compile-time dispatch, no per-row branches), formulas mirror the host implementations exactly in float with IEEE expf, and the Poisson raw-score clamp moved to one shared header constant so host and device cannot disagree. Weights upload once per fit beside the labels under the same dataset identity key. Eligibility now reads: MSE, LogLoss, or Poisson; weights or not; no DART; uniform or Bernoulli row sampling. Everything else keeps the host path untouched, and BONSAI_HOST_OBJECTIVE=1 still forces it.

**Evidence.** Parity better than the tolerance budgeted for transcendentals: resident predictions are bit-identical to the host-objective GPU path on every tested case (LogLoss, Poisson including a clamp-stress fit that is exact by construction, weighted MSE, weighted LogLoss under Bernoulli). Same-pod interleaved A/B at 16M rows, 100 iterations, oblivious: LogLoss 15.4s to 11.5s (25.7% cut), Poisson 14.3s to 10.6s (25.8%), weighted MSE 22.2s to 10.4s (53%), metrics identical to six decimals in every pair.

**The surprise the A/B surfaced.** The weighted host path was paying a serial 16M-element weight multiply every tree, roughly 80ms/round of single-threaded work. The loop is elementwise with no reduction, so it is now parallel (bitwise-identical models by construction); the 53% weighted cut above was measured against the pre-fix host path, and the honest post-fix comparison will be nearer the unweighted cuts. Weighted CPU fits and ineligible weighted GPU fits inherit the same fix.

**Rejected.** Fast-math exp intrinsics (parity is worth more than nanoseconds in a membw-bound kernel); softmax residency (per-class trees, its own campaign if ever); GOSS residency (it reads and reweights host gradients by design).

## 80. The categorical reopener predicate is established; the build decision waits for launch strategy (adopted)

**Decision.** Decision 58's escape hatch named two conditions for reopening native categorical splits: crossed-TS preprocessing failing to close the catboost gap, and a workload where that gap is load-bearing. The first is now measured (scripts/probe_tabarena_cat.py, benchmarks/tabarena-cat-probe-2026-07.md): on the cat-heavy TabArena subset, catboost's own reference toggle (native categoricals vs the same model ordinal-encoded) prices its categorical machinery at 68% of its remaining lead over bonsai with the ordered target encoder (mean share -0.0099 of a -0.0147 remaining gap, 12 datasets), while the pure-numeric control is bit-identical in both arms. The distribution is honest: beyond the chance band on 8 of 12, inside it on 3, and the machinery hurts on 1; the leave-one-out ratio spans 47 to 81%. Where the machinery's price is largest (amazon, kddcup09, splice), ablated catboost loses to bonsai_ts outright: the encoder already matches everything except the native per-split ordered statistics and feature combinations. The machinery also costs catboost 4.2x train time on these datasets.

**Consequence.** The reopener predicate stands, but no build starts: the second condition (a load-bearing workload) is a launch-strategy judgment tied to whether TabArena standing is a product goal (issue #157). If it fires, the design starting point is architecture doc 17 (priced and shelved), now with a measured target: the per-split share on cat-heavy data, roughly one point of AUC-scale metric. The other two thirds of the catboost story (its pure-numeric small-data lead) belongs to a future ordered-boosting campaign and no categorical work will close it.

**Rejected.** Treating the TabArena Elo gap alone as the load-bearing workload (a leaderboard is evidence of capability, not of a user's workload); building the doc-17 engine speculatively ahead of the launch call (the admission discipline held once and the price list only got sharper).

## 81. Ordered boosting: declined at stage 0, the mechanism is not the small-data edge (adopted)

**Decision.** The ordered-boosting campaign died at its own stage 0, per the pre-registered kill. CatBoost's own `boosting_type` toggle, run at both its defaults and matched knobs on a 12-dataset pure-numeric small-data pool (scripts/probe_ordered_boosting_rung0.py, benchmarks/ordered-boosting-probe-2026-07.md), prices the mechanism at zero or below: Ordered beats Plain beyond the chance band on 0 of 12 at matched knobs (1 of 12 at defaults), the mean share is negative in both task families, and on the two datasets where the toggle moves most Ordered is distinctly worse while bonsai beats CatBoost outright. Ordered also costs about 3.9x Plain's train time, which is the sanity check that the toggle really engaged. The reachability prototype produced its own refutation with a named cause: a faithful 2-fold honest-gradient booster does not converge, because a fold's gradient never sees its own accumulator and therefore never vanishes; early stopping truncates it far short of usefulness, and the simpler two-booster form is bagging, not the mechanism.

**Consequence.** The long-standing attribution "CatBoost's small-data lead is ordered boosting" is withdrawn from the record. The residual decomposes elsewhere: the probe's single-split harness does not reproduce the bagged gauge's CatBoost lead at all (bonsai is competitive to better on 10 of 12 there), so what remains of the aggregate-leaderboard edge points at the bagged-ensemble protocol interaction and tuning defaults, plus the categorical machinery decision 80 already priced on cat-heavy data. No bonsai booster-math campaign follows from any of those. The what-to-use-when row softens its attribution accordingly.

**Rejected.** Building any ordered-gradient scheme into the core (the mechanism fails its own vendor's ablation on the target regime); extending the prototype to high fold counts (the non-convergence is structural at small k, and the compute grows k-fold against a mechanism already priced at zero). **Reopener:** a measured bagged-protocol interaction showing Ordered contributing under bagging where it does not single-split, or a user workload where CatBoost's small-data lead is load-bearing and survives matched-knobs Plain.

## 82. Static permutation averaging: declined, the categorical substance is per-split (adopted)

**Decision.** The cheapest route to CatBoost's measured categorical share, K-permutation-averaged ordered target statistics as plain preprocessing, was priced on the decision-80 cat-heavy pool at matched protocol (scripts/probe_static_k_encoder.py, benchmarks/static-k-encoder-probe-2026-07.md) and declined by its pre-registered WEAK criterion: K=8 recovers a negative share of the gap to native CatBoost (pool mean -0.026, TS-active -0.039), the K curve is non-monotone (small bump at 4, negative at 8, with an 0.018 swing on a 1000-row dataset between adjacent K values), and the two datasets where decision 80 priced the machinery largest are hurt or flat at both K. The mechanism is named: the K-average converges toward leave-one-out target statistics, the leaky mode decision 58 already measured, while the single ordering's noise acts as implicit regularization that averaging destroys. The reproduction gate held bit-for-bit (max delta 8.3e-17 against the cached gauge), which also corrected the record: the gauge wrapper runs the encoder at cross defaults, not cross=2.

**Consequence.** The categorical question now has both bounds measured: the encoder-side static ceiling is zero, and the native per-split share is 68 percent of the remaining cat-heavy lead (decision 80). If the doc-17 build ever proceeds, it must contain the dynamic machinery (per-split ordered statistics and path combinations), and it remains gated on the launch-strategy call. One cheap deployment lever is now visible and separate from any build: the TabArena wrapper never enabled the shipped cross=2 encodings that decision 58 measured well on exactly the highest-cardinality pool member; enabling it is integration configuration for issue #157, not research.

**Rejected.** Productizing K-averaging in the encoder (negative at the only K that matters); higher K (the trend is the wrong direction and the mechanism explains why); treating the K=4 bump as signal (inside small-data variance, non-monotone).

## 83. An automatic learning-rate default: declined, even the oracle buys nothing (adopted)

**Decision.** The hypothesis that a per-dataset learning-rate rule (CatBoost's automatic default being the exemplar) is a cheap slice of the small-data defaults residual was priced and declined (scripts/probe_lr_rule.py, benchmarks/lr-rule-probe-2026-07.md; the ordered-probe pool and protocol, reproduction bit-exact). The ceiling itself is empty: a validation-selected per-dataset oracle over eight learning rates gains nothing on the pool mean, and its signature is overfitting, winning the validation split on 10 of 12 datasets but the test split on only 6. On the two datasets where CatBoost-default leads beyond the band, the oracle closes 15 percent of one gap and worsens the other. CatBoost's own automatic values, transplanted by reading the resolved rate from its params, sit in a tight 0.024 to 0.075 band around bonsai's shipped 0.05 and are a no-op; its heuristic trends upward with rows while the noisy oracle trends downward, mildly anti-correlated. A flat 0.1 control is worse than 0.05. Caveats recorded honestly: low rates ride early stopping longer (correlation -0.79 with trees kept) and the 1000-tree cap binds at the grid's low end on most datasets.

**Consequence.** bonsai's fixed 0.05 default stands; no auto-default feature follows. The small-data defaults residual loses its last named single-knob suspect: what remains of the CatBoost aggregate story is the bagged-ensemble protocol and its non-rate defaults (per-split randomization), which interact with bagging and belong to decision 81's bagging-interaction reopener as one combined future probe. A prior worth recording as corrected: the automatic rate had been assessed as the likeliest cheap lever on the leaderboard; the measurement says the lever does not exist at this pool's sizes.

**Rejected.** Fitting a size rule anyway (the leave-one-out fit swings sign across folds, pure variance); a finer sweep (the ceiling is the problem, not the grid); raising the tree cap to un-truncate low rates (the reference protocol is the comparison and the mid-grid is cap-free on most datasets already).

## 84. A code-metrics division: the readable-core claim made falsifiable (adopted)

**Decision.** The results site gains a self-only code division (scripts/measure_complexity.py, the results-ledger section, the benchmark-protocol subsection). It measures the bonsai tree at one git SHA with lizard (pinned uvx lizard@1.23.0): per plane (core_headers, engine_impl, cuda_plane, bindings_cli, bench_tooling, tests) it records file count, LOC, NLOC, function count, and mean and max cyclomatic complexity, plus the five highest-CCN core functions by name and the surface counts (45 parameters, 105 dispatch combinations, 9 public Python names, dependency counts). The measurement is drift-gated in CI beside the other generated pages and superseded in place on re-measurement (decision 69). The baseline: core_headers 4,926 LOC at mean CCN 2.04 with a per-function ceiling of 15, and the worst function across the core planes is the CSV parser at CCN 29 (in the IO layer, not the split, tree, or booster headers).

**Why.** The code-quality pillar was the only one of the four project goals with no results representation; speed had the frontier, accuracy the standings, pedagogy the Learn tracks, and the readable core carried only unfalsifiable prose of the kind the style guide forbids for performance claims. A claim about code you can read must ship with counts you can check. The five worst functions are published by name deliberately; an offender list its author curates away is marketing, not measurement.

**Non-claims, recorded.** LOC alone is not quality and a small number is not an argument; the division makes the claim checkable, not proven. No comparative claim against any other library is made or implied: comparative measurement (competitors at pinned tags, with a published core-selection rule and a paired capability column) is deferred until this self-only methodology has lived in public. Pre-registered CI budgets on named core functions (a maximum CCN per function, the gate that would have caught the resident-objective seam before review did) are a possible later step, held until the baseline has history to budget against.

**Rejected.** Comparative from day one (the methodology earns trust on our own tree first, and an unfair competitor selection would poison the division's credibility); LOC-only reporting (complexity is the load-bearing metric, LOC the least informative); hiding the offender list (naming our worst functions is the credibility move).

## 85. The bagged-protocol interaction: declined, CatBoost's small-data lead is not a randomization interaction (adopted)

**Decision.** Decision 81's reopener named a measured bagged-protocol interaction, CatBoost's randomization defaults decorrelating its ensemble members under bagging where they do not single-split, as the one thing that could resurrect the small-data story. It was priced and refuted (scripts/probe_bagging_interaction.py, benchmarks/bagging-interaction-probe-2026-07.md; the ordered-boosting probe's pool, splits, and loader imported read-only, bonsai_single bit-identical to that probe's bonsai arm on all 12). The headline interaction, (cat_single minus cat_bag8) minus (bonsai_single minus bonsai_bag8), is negative in both pool means (regression -0.032, binary -0.0002) and strictly inside the chance band on 7 of 12; the pre-registered REFUTED wording asked for 10-plus-of-12 in band, so that wording is recorded as not literally met, but the sign analysis makes the refutation firmer, because 4 of the 5 out-of-band cases are NEGATIVE (bonsai gains more from bagging than CatBoost, the opposite of the hypothesis) and the lone CatBoost-favoring beyond-band case is one 460-row coin-flip. The named mechanism is null: randomization_share (neutralizing CatBoost's Bayesian bootstrap and random_strength under bagging) averages -0.0004 on regression and +0.0013 on binary, inside or at the band on 11 of 12, and on the single dataset where the interaction favors CatBoost it covers the whole of it. The plain cause sits in the bagging-gain columns: 8-fold data-bagging already gives bonsai the decorrelation (bagging_gain_bonsai mean +0.104 on regression against CatBoost's +0.072), so there is no residual headroom for CatBoost's per-tree randomization to exploit, and bonsai's shipped sampler knobs (arm 3, bernoulli plus feature subsampling) reach any randomization benefit where one exists.

**Consequence.** Decision 81's reopener closes on its first clause. The CatBoost small-data lead that does survive under bagging, reproduced directionally on 4 of the 5 gauge datasets by the library-default arm including both largest cached leads, is a LEVEL difference from CatBoost's defaults operating under bagging (arm 7 versus arm 2), not a bagging-specific interaction (arms 4 and 5) and not the randomization the hypothesis named (arm 6). No bonsai core change follows, and no wrapper randomization lever is recommended, because the zero-core-cost response already ships and buys nothing the deterministic bag lacks on this pool. What remains of the whole CatBoost aggregate story is now fully partitioned: the categorical per-split machinery priced on cat-heavy data (decision 80), and defaults-level tuning under bagging, with the booster-math suspects (ordered boosting decision 81, static K encoding 82, an auto learning rate 83, this bagging interaction) all measured to zero.

**Rejected.** Building or wiring any ensemble-randomization scheme for the bagged regime (the mechanism is null and bonsai's existing knobs already reach it); treating the one positive out-of-band dataset as signal (a single 460-row coin-flip fully absorbed by randomization_share, inside the noise the ordered-boosting and static-K probes both charted at these sizes); reading the gauge-reproduction magnitudes as a bit-match (the BAG8 here is an 8-fold inner bag on a single fold-0 test split, directionally faithful, not the AutoGluon protocol). **Reopener:** a bit-faithful AutoGluon-bagged reproduction showing a positive CatBoost interaction beyond the band on a pool majority (not one coin-flip), or a user workload where CatBoost's bagged small-data lead is load-bearing and survives matched-knobs Plain single-model, the same load-bearing-workload bar decision 80 set for the categorical build.

## 86. Honest shadow-feature selection: declined as an accuracy lever on all three growers, the gain ranking already has it (adopted)

**Decision.** A refit-based honest feature selector (a shadow-feature / Boruta prototype, append a permuted copy of every column and keep only real features that beat the 95th percentile of the shadow importances, over 5 seeds with a 3-of-5 vote) was prototyped at zero core cost and priced against CatBoost's in-library select_features and against plain top-k-by-gain truncation (scripts/probe_feature_selection.py, benchmarks/feature-selection-probe-2026-07.md; two regimes over 9 datasets from the ordered-boosting pool, REAL-WIDE up to 1024 features and NOISE-INJECTED with shuffled-copy noise equal to each set's feature count, the bonsai arms run under all three growers). It was declined on its pre-registered second clause, and the decline holds per grower: every beyond-band accuracy win the shadow arm produced is matched by plain top-k truncation at the same k. Under depthwise the shadow arm moved 3 datasets beyond the band (concrete_compressive_strength +0.736 rmse after 8 injected noise columns had cost bonsai_all 0.74, breast_cancer +0.00126, MagicTelescope +0.00172), but shadow_vs_topk is inside the chance band on all 9 datasets, and on concrete and MagicTelescope the two arms select a bit-identical kept set (shadow_vs_topk exactly 0.00000), because bonsai's own gain importance already ranks real features above the permuted ones, the same ranking the shadow threshold reads. Leafwise is bit-identical to depthwise on all 9 datasets (at the campaign shape the 63-leaf budget never binds, so the gain-ordered and level-ordered growers split the same node set; the dispatch was verified engaged with a binding budget) and inherits the verdict line for line. Oblivious, the grower whose coarser one-feature-per-level importance spectrum was the live threat to the top-k control, refutes that threat in the opposite direction: it adds a fourth beyond-band win (wind), every win is again a top-k win at the same k, and the only beyond-band shadow_vs_topk in the whole 27-row grid (pima_diabetes, -0.0072) has the shadow cutoff LOSING to plain truncation. The shadow vote contributes a cutoff rule at 6x the compute of one fit, not a better ranking, under any grower. The ADOPT-SIGNAL branch failed both ways on all three: the machinery also LOSES beyond the band on the same 2 low-dimensional informative sets everywhere (pima_diabetes -0.011, spambase -0.0012), so the no-loss condition is unmet, and it does not match cat_select pool-wide (differs beyond the band on 5 of 9) at half its wall time (depthwise 0.77, leafwise 0.74, oblivious 1.20 of cat_select's 126.9 s pool total, none half or less).

**Consequence.** Feature selection is not an accuracy lever on this pool. Where it recovers accuracy it is recovering from injected junk that a one-line top-k-by-gain removes just as well, and where the features are all real it costs accuracy. A bonsai.select module, if it ever ships, is a wall-time-and-interpretability tool (smaller models, a noise-detection report), explicitly not an accuracy feature, and no bonsai-core change follows: the whole prototype is Python around the shipped importance("gain") call, and the measurement says even that Python wrapper does not beat a sort of the same importances. The noise-recovery precision/recall table is retained as the deliverable exhibit for the guide chapter that follows this probe: it teaches that both selectors are perfect only when the noise is truly independent (concrete, both drop all 8 noise and keep all 8 real) and imperfect the moment the real features are correlated or the pool is high-dimensional (breast_cancer, the shadow arm drops 16 of 30 real features and cat_select drops 21 while keeping 8 noise).

**Rejected.** Shipping the shadow selector as an accuracy feature (its wins are top-k's wins on every grower and it loses where features are real); productizing it over plain top-k-by-gain (shadow_vs_topk is inside the band on 26 of the 27 grower-dataset cells and favors truncation in the 27th); reading the concrete recovery as a selection win for the machinery rather than for the ranking (top-k recovers the same accuracy from a bit-identical kept set); treating the oblivious grower as a separate selection story (its coarser spectrum changes the numbers, not the verdict, and its one beyond-band separation favors top-k). **Reopener:** a workload where shadow_vs_topk clears the band in the shadow's favor on 2 or more datasets under any grower (the shadow cutoff genuinely beating truncation at equal k), or a high-noise-fraction regime where the data-driven cutoff beats every fixed k a user would guess; the interpretability and model-size case is not refuted here and is a separate, non-accuracy decision.

## 87. The XGBoost 3.3 recheck: every published standing survives, one competitor improvement recorded (adopted)

**Decision.** XGBoost 3.3 (2026-07-21) claimed lower GPU quantile-sketching memory and CPU histogram tiling for wide datasets, both aimed at cells bonsai competes in, so the perf claims were rechecked on one pod (L40S, US-NC-1) with three same-pod arms: bonsai at main, XGBoost 3.2.0, and XGBoost 3.3.0 in separate venvs (benchmarks/results/xgb33-recheck-2026-07.jsonl, 41 rows; the recheck subsection of the results ledger). On GPU, 3.3 matches 3.2 within noise at every measured cell (rows 1M/4M/16M at 100 cols, cols 256 and 1024 at 1M), host RSS does not move (22.1GB at 16M against bonsai's 6.9GB, reproducing the README's 3x memory ratio on a second host), and the order never changes: bonsai_cuda_depthwise fits 3.0-3.3x faster than xgb_cuda at every cell, with zero cpu-fallback nodes in every bonsai profile. On CPU at 16M bonsai is 6% behind xgboost-hist, inside the published "within ~8%, host-dependent" band. The one real improvement in the release: 3.3 halves wide-CPU hist time at 1M x 4096 (783.7s to 387.1s; no change at 1024, the tiling engages by width), which narrows bonsai's CPU lead at that cell from 2.4x to 1.19x (323.9s vs 387.1s) but flips nothing: the ledger's wide-cols standings are GPU, where CatBoost leads and 3.3 changes nothing.

**Consequence.** README and ledger claims stand unedited; rebaseline-2026-07.jsonl remains the authoritative standings table. The recheck rows carry explicit per-row xgboost version tags, giving the perf claims a recorded version boundary (valid against 3.2.0 and 3.3.0) for the first time. One watch item is recorded rather than acted on: the two widest CPU cells (131k x 16384, 32k x 65536) were not re-measured, and the 4096-col tiling gain suggests 3.3 could pass bonsai's CPU growers there; if wide-CPU standings ever become reader-facing, re-measure those two cells first.

**Rejected.** Superseding the re-baseline table from this pod (only three of its six variants ran here, and cross-pod absolutes are invalid under the ~25% fleet spread); a README edit (no number cited there moved); treating the 1024-col null as the tiling failing (4096 is simply the first measured cell wide enough for it to engage); re-measuring the CatBoost and LightGBM arms (neither shipped a release since the table was measured).

## 88. The wide-CPU fill routing: feature-parallel past a 24MB footprint (adopted)

**Decision.** The first production field report (issue #217: 131k x 16384 on a Xeon 6521P, LightGBM-CPU ahead of every bonsai grower 2-3x, matching the committed multi-host scaling rows) traced to the row-wise u8 histogram fill: its per-row scatter targets the whole selected histogram footprint (total_cells x 8B, 33.6MB at 16k features x 255 bins), so past the last-level cache every add misses and the partial slabs add a zero+merge pass per block (populate was 84-88s of a ~100s fit at the stage-0 profile). u8 levels whose footprint exceeds 24MB now route through the existing feature-parallel fill, previously reserved for u16 data: one thread per feature, no partials, no merge, bit-identical at any thread count. Same-pod before/after at 131k x 16384 (t=16, SCALING knobs): depthwise 1019s to 379s (2.7x, LightGBM parity at 367s), leafwise 2591s to 445s (5.8x, the 7x deficit against LightGBM collapses to 1.2x), identical r2, peak RSS 18.8GB against LightGBM's 50.1GB. The threshold was set by an interleaved same-pod A/B at 1M x 4096 (alternating builds, twice each): the row path WINS mid-width on a big-L3 host (320/337s vs 547/547s all-feature-parallel), because the EPYC's L3 absorbs an 8.4MB scatter target while the feature-parallel fill pays scattered per-feature column reads on sparse child nodes. 24MB flips only shapes that lost on every measured host.

**Consequence.** bonsai's wide-CPU standings move from 2-3x behind LightGBM to parity (depthwise) at 2.7x less memory; the fastest-slot claims (<=256 cols) are untouched and the narrow path is code-identical (fixed-input model hash byte-identical to before; 526 C++ and 67 Python tests green). Above the threshold, model bytes change at identical accuracy, and gain thread-count invariance the row path never had. Evidence: benchmarks/wide-cpu-hist-2026-07.md, raw rows in results/wide-cpu-hist-2026-07.jsonl.

**Rejected.** A per-node rows bound (<=256k rows feature-parallel, built and withdrawn: the A/B showed sparse-node feature-parallel is the mid-width problem, not node size, and the bound kept the regression at 573s); shipping the M2-calibrated 6MB threshold (it flips 4096-col shapes that a big-L3 host wins on the row path); a config knob for the threshold (a measured constant until a real workload disputes it).

**Reopener / recorded follow-ups (issue #217).** A cache-size-aware threshold to capture the mid-width win on small-cache hosts (M2: 101.7s to 44.9s at 131k x 4096 goes untaken by the 24MB constant; XGBoost 3.3's aarch64 cache detection is the same lever); skipping the row-major mirror for always-wide fits (2.1GB at the 16k cell); the CUDA planes' own wide wall (~5x behind xgb_cuda at the 16k cell) is untouched by this change.

## 89. The tiled mirror: one fill retires the wide-CPU strategy pair (adopted)

**Decision.** Decision 88's per-width strategy pair (row-wise below a 24MB footprint, feature-parallel above) lasted one day: the recorded follow-up, XGBoost 3.3's column-tiling lever, was probed immediately and dominated. The u8 row-major mirror moves to a column-block-tiled layout (2048-feature blocks, each row-major on its own; one block at narrow widths reproduces the classic layout byte for byte) and the fill runs tiles outer, rows inner, so the live scatter target is one block's histograms (at most ~4MB) at any selection width while reads stay sequential inside each block. Per-feature accumulation order is unchanged from the untiled row path, so models are BIT-IDENTICAL at every width (verified: identical model sha256 from the main and tiled builds at a multi-block width), which retires the 24MB threshold, the u8 feature-parallel route, and the cache-size-aware-threshold follow-up in one move. Interleaved same-pod A/B (two reps each, single worker): tiled beats the row path at its best cell (326/321s vs 369/377s at 1M x 4096), beats feature-parallel at its best cell (442/448s vs 514/532s at 131k x 16384), and is a wash at 16M x 100 (114/116 vs 119/112); the M2 leg takes the mid-width win the fixed threshold had left on the table (44.8s vs 101.7 at 131k x 4096) and scales linearly to 8192 with no cliff. Identical r2 everywhere.

**Consequence.** One fill covers every u8 width with no host model, no threshold, and no strategy dichotomy; the feature-parallel fill remains only for u16 data, its original job. Same-machine reproducibility strengthens: the u8 fill's sums no longer depend on selection width routing, and wide models regain byte-stability across versions from here on. Peak RSS at the 16k cell is 21.0GB (against 18.8 for the strategy pair and 50.1 for LightGBM), the cost of per-slice partial stripes. Evidence: the superseding section of benchmarks/wide-cpu-hist-2026-07.md; raw rows tagged run=tiled-ab in the same jsonl.

**Rejected.** Keeping the strategy pair alongside the tiled fill (dead code with a threshold nobody can calibrate); runtime cache detection (mooted: the tile constant is cache-conservative everywhere measured, M2 through EPYC); a fit-time empirical trial (would trade same-machine byte reproducibility for a decision the tiling makes unnecessary). **Reopener:** a host or shape where a 2048-feature block's ~4MB histogram footprint is not cache-resident (the tile width becomes the knob to revisit, not the strategy); the CUDA planes' wide wall stays open on issue #217.

## 90. The CUDA wide recheck: the wall was already gone (adopted)

**Decision.** The campaign opened to close the recorded wide-GPU gap (bonsai_cuda ~355s at 131k x 16384 against xgb_cuda ~72s in the multi-host scaling study) and closed at stage 0, because the gap had already been closed by three weeks of unrelated work: the recorded rows date to 2026-07-07/08 code (git bd783e6/972652c), before the level-transaction and device-resident campaigns landed. On current main, one L40S pod, SCALING knobs (benchmarks/results/cuda-wide-recheck-2026-07.jsonl): bonsai_cuda_depthwise 54.9s at 131k x 16384 against catboost_gpu 71.2s and xgb_cuda 76.7s, and 37.7s at 1M x 4096 against catboost_gpu 50.2s and xgb_cuda 103.7s, at 3-4x less peak host RSS (8.8-16.4GB against 25-60GB). bonsai_cuda_oblivious (61.2s at the 16k cell) also beats both references. The wide-data standings prose ("CatBoost keeps the lead, bonsai second"), stale in README and the results door, is corrected to cite this recheck.

**Consequence.** No CUDA wide campaign runs; a refutation-by-progress is the deliverable, and the lesson is operational: perf standings quoted from a study must carry their git sha forward, because three weeks of engine work can invert them silently (this is the second stale-row correction this week, after the CPU field report). The stage-0 price list is banked for the future: even in the winning fit, the find stage is ~80% of grow (find_kern 17.3s plus gpu_wait 24.5s of a 54.9s fit at the 16k cell) and the host-side mapper fit costs 11.5s at 16k features, so a further ~2x of internal headroom is measured and recorded on issue #217 without being spent. The full six-variant cols-axis re-baseline (superseding the July 8 study's wide cells properly) is the recorded follow-up before any wide-standings chart ships.

**Rejected.** Running the campaign anyway against the internal headroom (the standings motivation is gone and the crown items rank higher); superseding the July 8 scaling study wholesale from this two-cell recheck (cross-pod absolutes and the narrow cells remain valid history); leaving the stale prose standing while the jsonl told a different story (the reader-facing claim is the product).

## 91. The iso-volume shape frontier: measured device memory becomes an output (adopted)

**Decision.** The first campaign on the redesigned bench tooling holds rows x cols constant and sweeps aspect ratio, with measured peak device memory (`dev_mem`, NVML-sampled per worker child) recorded in every row and the a-priori memory gates disabled by spec, because a measured failure at a shape is the experiment's output rather than a condition to pre-empt. Two committed ladders (benchmarks/specs/): 2^31 cells from 16M x 128 to 32k x 65536, six arms; 2^33 GPU-only to 262k x 32768. One pod, RTX PRO 6000 Blackwell Workstation Edition 96GB (sm_120 via CUDA 12.8 toolkit side-installed over the cuda12.4 image, clang-21 offload), main a907895 (benchmarks/results/iso-volume-2026-08.jsonl). bonsai's CUDA growers are fastest at every cell of both ladders: near-flat 6.8-9.3s across the tall half of the 2^31 line where both references vary 1.5-2x, 4.1x over XGBoost-GPU at 67M x 128 on the 2^33 ladder (27.9 vs 113.8s at 11.7 vs 73.6GB device memory). The instrument's first catches: XGBoost-GPU fails at 32k x 65536 having allocated only 33.4GB of the 96GB card (an internal limit, not exhaustion); CatBoost-GPU allocates 90.2GB at every cell including 1M x 100 (reserves the card, never sizes to the problem); bonsai's footprint tracks the problem, 0.8 to 58.5GB along the ladder. At extreme aspect (p about 2x n) the oblivious grower holds r2 .873 where every depthwise-family arm falls to .815-.817.

**Consequence.** Feasibility claims now cite measured rows, not estimator output: `GPU_MAX_COLS` stays as the default policy for unattended sweeps on consumer cards, but it is a policy knob a spec disables, and the 96GB class measurably runs cells the old skip assumed impossible. The ad-hoc campaign-driver era ends: this campaign produced zero hand-written schemas (every row schema v1 with sha, host, and run label stamped by the harness), and the committed spec plus scripts/pod_bench_driver.sh reproduce it in two commands. The work-rig replication path (make bench-iso, host-tagged) stands open for a second same-silicon point. Watch item: CUDA 13 stays blocked until clang can target it; the campaign runs on the 12.8 toolkit ceiling.

**Rejected.** Fixing a VRAM budget instead of a cell budget (device memory is library-dependent, so the budget would encode one library's allocator; constant logical volume isolates shape and lets memory be the measurement); nvcc for the sm_120 build to reach CUDA 13 (the campaign must measure the clang-built binary path the wheel ships); baking a cuda12.8 image (the side-install costs 2 minutes a session and the 12.4 image is the only one that boots fleet-wide).

## 92. The results lifecycle: standings supersede in place, evidence freezes, staleness hard-fails (adopted)

**Decision.** Results files conflated two roles and the conflation produced drift three times in one month (the scaling history's stale wide cells stood until decision 90; the adversarial sweep found four more stale reader-facing standings claims). The split: evidence files are the dated record behind a decision, frozen forever, append-only, corrected by banner plus log entry; standings files are the current claim on one published axis, listed in benchmarks/standings.json with the single sha their rows were measured at, superseded in place by re-measurement (decision 69's code-division rule generalized). The ledger stamps every standings caption with the measured sha computed from the rows. Two gates hard-fail via scripts/check_standings.py: a decisions entry claiming a perf change carries a `Standings: <axis>` line and docs-check fails while any tagged entry outruns the axis's registered state (this entry, 92, is the parse baseline); the wheels publish job fails unless every axis was refreshed for exactly the version being released. Reader-facing prose never restates standings digits. Deleted under the policy (git history archives): the multi-host scaling history and its exponent analysis, the retired year-MSD track, and the Catch2 microbenches, which measured kernels in isolation and were structurally blind to the memory-system effects that actually moved standings.

**Consequence.** The refresh becomes one rented pod session, automated as the standings-refresh workflow: a same-pod A/B of the previous release wheel against HEAD on anchor cells is the perf-change detector (the only rigorous one, since cross-pod and CI-runner timing comparisons are invalid at fleet spread), then the standings specs re-measure and the supersession lands as one reviewed PR whose body reports the A/B verdict; a moved verdict demands a tagged decision, which the claim gate then enforces. Releases inherit a fixed cost of roughly one pod session and gain standings never older than one release. The perf division sheds its history-file class entirely; repeatability questions are answered by reps within a standings run, not by cross-vintage archaeology.

**Rejected.** Recipe-only storage for perf (a recipe reproduces the procedure, not the numbers; the fleet-spread rule exists because hardware is not a controlled instrument, so the measured single-sha jsonl is the irreducible minimal artifact and it already carries the recipe in every row); CI microbench tripwires (shared-runner noise floors and kernel-scope blindness; instruction counting is stable but measures the wrong thing); banner annotations for stale standings (banners rot, deletion cannot); a calendar cron for refreshes (manual plus pre-release bounds staleness without surprise spend).

## 93. First automated standings refresh: stale-vintage supersession, airline speed flip (adopted)

**Decision.** The standings-refresh workflow's first successful run (2026-07-31, one L40S, sha `d3ffcd0`) supersedes the rows, width, frontier, and airline standings in place. The same-pod A/B against the 1.5.4 wheel read no code movement: all four anchor cells within ±0.5%, so HEAD and the shipped wheel are performance-identical.

Standings: rows, width, frontier, airline

**What moved and why.** The published numbers moved anyway, because the superseded files predated the late-July engine work that shipped in 1.5.4: the rows ladder was measured at `434a382` (2026-07-13), before the tiled CPU fill and the radix mapper sort. The refreshed 16M x 100 headline is 10.3s (CUDA oblivious) against XGBoost-GPU's 19.6s, where the stale file read 18.4s vs 19.9s. On the airline shape the speed standings flip: bonsai CUDA depthwise is now the fastest fit at 1M and 10M rows under ordinal encoding (1.9s vs XGBoost-GPU's 2.6s at 10M), retiring the long-standing "XGBoost-GPU owns raw speed on the narrow shape" reading; XGBoost keeps only the 100k cell. The frontier holds its shape: bonsai first to every measured accuracy, terminal accuracies tied within the noise band, marginal round below CatBoost's. Width holds: bonsai CUDA fastest at every width, with the CPU arms trading the widest cell inside a rep's noise.

**Consequence.** This is the policy working as designed: the drift was not unreleased speed but unrefreshed files, and the refresh caught it in one pod session. Narrative captions that restated standings digits (airline provenance, frontier provenance, the cols re-baseline prose) are rewritten digit-free; the tables and generator-computed stamps carry the numbers. Chart ticks on the frontier page now derive from the data after the hardcoded ones stranded outside the refreshed range.

## 94. LightGBM-CUDA joins the reference arms; August supersession (adopted)

**Decision.** The bonsai-ci image builds lightgbm from source with USE_CUDA=ON (PR #257), and `lgbm_cuda` joins the rows, width, and frontier suites. The 2026-08-01 refresh (one L40S, sha `0b077ad`) supersedes all four fast axes; the A/B against the 1.5.4 wheel read no code movement (four anchors within ±0.8%).

Standings: rows, width, frontier, airline

**What the new arm shows.** LightGBM-CUDA is real GPU performance at scale, not the small-data loser the decision-42-era snapshot suggested: at 16M x 100 it fits in 31.7s, 6.6x its own CPU (210.2s) and ahead of XGBoost-GPU on this pod, though 3x behind bonsai (10.5s). Its test r2 runs consistently higher at matched knobs (0.884-0.886 vs the field's 0.876-0.880), an implementation difference worth its own probe. At width it hits the histogram wall hardest of any arm: 563.7s at 131k x 16384, 11x bonsai's 50.3s, validating the decision-42 regime argument where it was made. bonsai keeps the fastest slot at every cell of every ladder.

**Fleet-variance caveat, recorded.** XGBoost-GPU's 16M anchor swung 19.6s to 36.9s between two same-model pods (identical CPU class) while bonsai's held within 2%; this exceeds the documented ~25% spread and flatters bonsai's published margin over XGBoost by pod luck. The same-pod ladder remains the standings per protocol, but the A/B detector anchors only bonsai arms today; adding one reference anchor to the A/B would catch reference-side pod pathology. Follow-up, not blocking.

**Also fixed.** The refresh workflow's month-rollover bug: a supersession across months deletes the old dated files, and the committed-files render gate (git ls-files) fired before the deletions were staged. Deletions now stage before rendering.

## 95. The leafwise recheck: decision 42's claim inverts at scale (adopted)

**Decision.** The decision-42-era reading "CPU `leafwise` beats LightGBM's CUDA leaf-wise" is superseded by measurement: on one pod (L40S, US-MO-1, 2026-08-01, `leafwise-recheck-2026-08.jsonl`) LightGBM-CUDA wins at every ladder scale, monotonically from 1.2x at 250k to 5.3x at 16M (24.4s vs 128.4s). The original claim was measured at 464k rows and was a small-n artifact. Doc 11's leafwise row carries the correction; the recheck table lives on the scale page.

**What stands, what changes.** The engineering conclusions of decision 42 stand: no `cuda_leafwise` registration that computes CPU histograms under a GPU name, and bonsai's existing CUDA growers dominate lgbm_cuda outright (same-pod anchor 17.2s vs 24.4s at 16M; frontier time-to-accuracy 2.2-3.1x, decision 94's data). What changes is the deferral's premise: LightGBM demonstrates leaf-wise-on-GPU is viable at scale, so the device-leafwise deferral converts from "structurally unpromising" to a tracked engine gap with a measured competitive target - issue #268, kill criterion pre-registered (beat lgbm_cuda at 16M x 100 same-pod or do not register).

**Also recorded.** The r2 pattern across all leafwise-recheck cells matches decision 94's depth-cap finding: lgbm_cuda scores .879-.886 while every depth-8-honoring arm (bonsai leafwise, lgbm_cpu included) sits at .872-.879, and lgbm_cpu with max_depth=-1 reproduces the CUDA scores exactly - LightGBM's CUDA learner does not enforce max_depth, so its quality column in the ladders is not at protocol knobs. Cross-pod CPU variance also logged: lgbm_cpu's 16M fit measured 104.7s here vs 210.2s on the August ladder pod, a 2x swing on a CPU arm while the GPU anchor moved 42%.

## 96. The standings-refresh CI workflow retires for a local driver (adopted)

**Decision.** The standings-refresh workflow's tail (supersede files, render, open the bot PR) failed on all four of its dispatches, each on a distinct one-shot bug found only by paying for the roughly three-hour pod run that reaches it: pip missing in the bonsai-ci image's uv venv, a rows-only schema `ZeroDivisionError` in the A/B verdict table, and the month-rollover staging bug decision 94 already logged. Every fix landed after the fact, on the next paid dispatch. The measurement half never failed: all four runs rented the pod, ran the standings specs, and produced correct jsonl, with the artifact-upload step (added after the first tail crash) recovering the data every time the tail crashed downstream. The structural problem is that the tail has no test harness short of a real pod rental, so the workflow's own CI never caught what it broke. `scripts/standings_refresh.py` replaces it: a `measure` phase (pod create, detached on-pod run, incremental scp pulls, teardown plus stray-pod sweep) and a `supersede` phase (registry update, staged `git add -A benchmarks/` before render, A/B verdict, branch, commit, `gh pr create`, with a `--no-pr` escape hatch) that runs on a developer machine and can be rerun without touching a pod. The supersede phase was dry-run validated against the 2026-08 measurement artifact before this decision.

**Consequence.** The decision-92 refresh ritual is unchanged: same two phases in the same order, same release ordering (bump PR merges first, then refresh with `prev_version` set, then tag), same `Standings:`-tagged-decision gate on a moved verdict, same freshness check at publish time. Only the vehicle moved, from a GitHub Actions runner to a local script invoked by hand per the runbook. The tail is now iterable at the cost of a supersede-phase rerun against an already-measured results directory, not a fresh pod rental, so the next tail bug (there will be one) costs minutes instead of hours. Still unexercised: the supersede phase's `gh pr create` step itself, since the dry run stopped at the commit; the first live run is that step's real test.

**Rejected.** Iterating on the CI tail in place (the four dispatches already show this costs one paid pod session per bug, with no way to shorten the loop); splitting responsibilities, CI for measurement and a local script for supersede (two systems for one ritual, and the CI half brings back the pip-in-uv-venv class of failure for no reliability gain now that the local driver also measures); a scheduled cron trigger (decision 92 already rejected this for the same workflow, and a flaky tail makes an unattended cadence worse, not better). **Reopener:** if the supersede phase accumulates its own run-scarce failure history, revisit whether a cheap local test harness (fixture results directory, no pod) can be built before deciding whether any part belongs back on a runner.

## 97. `cuda_leafwise` admitted by measurement (adopted)

**Decision.** The device leafwise plane of [`20-cuda-leafwise.md`](https://github.com/daniel-m-campos/bonsai/blob/06fab232a3b156a7c9f155fbbdf41fe14d45f1af/docs/architecture/20-cuda-leafwise.md) keeps its registration. Issue #268's kill criterion was pre-registered and is met: on one pod (L40S, US-NC-1, 2026-08-01, `leafwise-ladder-2026-08.jsonl`, sha `2a23f33`) `cuda_leafwise` fits 16M x 100 in 30.8s against lgbm_cuda's 32.4s. The secondary bar is met with room: it beats same-pod CPU `leafwise` 4.6x to 7.1x at every cell. Four arms, best of two reps, interleaved, SCALING knobs at 100 iters and 256 leaves.

| rows | cuda leafwise | lgbm cuda | leafwise (cpu) | cuda dw (anchor) |
|---|--:|--:|--:|--:|
| 250k | 2.9s | 6.7s | 13.5s | 0.7s |
| 1M | 4.4s | 7.7s | 29.3s | 1.6s |
| 4M | 10.2s | 12.2s | 66.9s | 6.0s |
| 16M | 30.8s | 32.4s | 219.8s | 22.6s |

**The uncapped-depth reading.** At the protocol knobs a 256-leaf budget under a depth-8 cap is the full tree, so every arm returns the same tree and the capped ladder cannot see what leaf-wise is for; it is also the regime where lgbm_cuda's r2 column reads high for the reason decision 95 identified, its CUDA learner ignoring `max_depth`. The extra 16M arm lifts the cap (explicit 256 leaves, no depth limit) and settles both. Uncapped, `cuda_leafwise` scores .8859 and CPU `leafwise` .8859 against lgbm_cuda's .8858: the quality gap the capped rows showed was the depth cap, not the engine, and LightGBM's fixed .8858 across both arms confirms the cap was never binding on its side. The time reading goes the other way and is recorded as-is: uncapped, `cuda_leafwise` takes 38.6s against lgbm_cuda's 31.6s, because best-first without a cap must find a split for every leaf it creates, doubling the round count from 25,500 to 51,100 where the capped tree skips the find at the cap. bonsai is faster at matched knobs and slower at matched accuracy, and the honest statement of the admission is that both hold.

**Where the time goes.** The profiled 16M rep prices the frontier serialization at 350 us per round: 19.70s of grow against the depthwise plane's 10.78s for identical histogram volume, over 25,500 rounds. That is 3.5x the design budget and 11x what the stage 0 skeleton probe measured, and the counters say why the skeleton was wrong rather than the design. 51,085 launches, 9.80s of find-stage host time against 3.46s of measured kernel time, `gpu_wait` at zero: the plane is launch-bound with the device idle, the exact inverse of the depthwise anchor's 7.96s `gpu_wait`. Trivial kernels in a cadence harness model neither the real kernels' occupancy on a frontier of one nor the staging around them. The admission stands on the ladder, not on the budget, and doc 20 now carries the decomposition and the ranked levers (launch batching, the 32 ms/tree `setup` residue, adaptive accumulator width).

**Consequence.** Every bonsai grower now has device support, closing the gap issue #268 opened; a leaf-wise user gets 30.8s at 16M where before the choice was 219.8s on bonsai's CPU arm or 32.4s by switching libraries. `bonsai_cuda_leafwise` joins the benchmark variant registry but deliberately not the SCALING suite tuple: the standings specs do not carry the leafwise arms, so no standings verdict moves and none of the four standings files is touched. Benchmark cells may now name an explicit `num_leaves`, which the uncapped arm needs and which nothing else uses.

**Rejected.** Withholding the registration until F reaches the 100 us budget (the kill criterion was pre-registered as a fit-time comparison against a named competitor, and moving the bar after seeing the data is the failure mode admission gates exist to prevent); reporting only the capped ladder (it is the flattering half, and the uncapped arm is where the strategy differs and where LightGBM wins); adding the arm to the standings sweep (it would expand every future refresh by a slow CPU arm and a device arm for a comparison the standings do not make). **Reopener:** if the launch-count levers land and the uncapped arm reaches parity with lgbm_cuda on time, the uncapped regime becomes publishable as a standings shape rather than a doc-20 footnote.

## 98. Device leafwise stage 3: two levers land, one is refuted (adopted)

**Decision.** Stage 3 of the device leafwise campaign closes here: two levers landed, a third was built, measured, and reverted, and the closing ladder is the record. On one pod (L40S, US-NC-1, 2026-08-02, `leafwise-stage3-2026-08.jsonl`, sha `703a78e`) `cuda_leafwise` fits 16M x 100 in 24.4s against lgbm_cuda's 31.9s, where decision 97 measured 30.8s against 32.4s. Three device arms, best of two reps, interleaved, unprofiled; the CPU `leafwise` arm is unchanged since decision 97 and is cited from it rather than re-measured.

| rows | cuda leafwise | lgbm cuda | cuda dw (anchor) | decision 97 leafwise |
|---|--:|--:|--:|--:|
| 250k | 2.1s | 6.6s | 0.7s | 2.9s |
| 1M | 3.2s | 7.6s | 1.6s | 4.4s |
| 4M | 7.6s | 12.3s | 5.9s | 10.2s |
| 16M | 24.4s | 31.9s | 23.0s | 30.8s |

The two ladders are different rentals of the same GPU model, so the reference arms carry the cross-ladder reading: lgbm_cuda and the depthwise anchor reproduce their decision 97 times within 2.4% at every cell, which licenses reading the leafwise column's 21% to 27% cut as the levers rather than the rental. Against decision 97's CPU arm the plane is now 6.4x to 9.2x faster, the margin over lgbm_cuda at 16M widens from 5% to 24%, and the fit gap to resident depthwise at 16M falls from 8.2s to 1.3s. `r2_test` is identical to decision 97 at every cell.

**Lever 1, the device-resident objective (landed).** The seam doc 20 put out of scope for stage 1 now arms for leafwise growth: eligible fits keep labels and scores on the device for the whole fit, so the per-tree gradient upload, the values and leaf-id downloads, and the host objective and score loops all go away. Same-pod A/B at 16M x 100, two reps each: median fit 30.23s to 24.60s, median grow 20.02s to 13.13s, `r2_test` and `r2_train` identical on every run rather than merely within tolerance. Arming is decided once per fit against the conservative bound (every feature selected, the widest feature sizing the stride) because a tree that declined to the host plane mid-fit would have no host gradients to fall back on; the whole decline predicate lives in one `leaf_budget_ok`.

**Lever 2, pinned and packed round staging (landed, over a gate it misses at one cell).** The fit-constant monotone vector uploads once per tree instead of once per split find, the histogram kernels' (offset, count, slot) triple becomes one packed upload behind three pointers into it, and what remains moves to pinned host memory and asynchronous copies: four per round against eight pageable ones. The saving is a fixed 0.37s of grow per fit, 14.5 us of the round against 25,500 rounds, and it is the same absolute number at every scale because it is per round and not per row. Read at the 16M gate cell that is 1.8% of fit against a 2% bar; it clears on grow (2.6%, ten interleaved blocks out of ten) and by 5x to 8x on fit at the other two cells of the same ladder (250k -17.1%, 1M -11.0%), and it ships on that basis. The deviation is recorded rather than hidden, and the rule it argues for is that a fixed-cost lever is gated across the ladder, not at the one cell where the most compute dilutes it.

**Lever 3, the partition chain (built, measured, reverted).** A parallel tiled segment scan replaced the single-threaded one, and the range copy-back moved to a non-blocking stream under the round's histogram, which read the smaller child from the scratch side the scatter had just written. Against its own parent it moves the paired median fit +0.03% at 16M, -1.2% at 1M, and +0.3% at 250k, with grow -0.8% at 16M, against the same 2% gate. Both halves are reverted rather than carried as a stream and an event nobody paid for, and the refutation is the deliverable: the 0.5s to 0.9s the stage 2 diagnosis attributed to the copy-back and the scan is real device time, but it overlaps work already in flight, so removing it buys back only the fraction that was on the critical path.

**The stage 2 diagnosis was profiler misattribution, and the correction is the reason stage 3 worked.** Decision 97 read the plane as launch-bound with the device idle at 350 us per round. Re-instrumented with per-kernel events, `find_stage`'s 9.80s turns out to be 84% device compute in disguise: `leaf_find`'s first pageable staging copy stream-syncs, so it absorbed the in-flight histogram and subtract kernels (8.2s) that the sibling planes peel into `gpu_wait`. The histogram runs at depthwise parity (8.99s against 8.77s of device time for the same volume), so there is no leafwise histogram penalty at this leaf budget and level batching would buy nothing. What decision 97's comparison actually measured was an uneven one: depthwise ran resident and leafwise could not, a seam a kill-switch A/B prices at 5.90s. The true leaf-plane penalty was 3.6s, and the ~200 us per round of host residue read off the profiled run costs 14.5 us unprofiled. Profiling stays the right instrument for attributing device time between kernels and the wrong one for pricing host residue on rounds this short; any future leaf-plane lever is gated on unprofiled wall clock at more than one scale.

**The uncapped arm, honestly.** Decision 97 recorded uncapped 16M at 38.6s against lgbm_cuda's 31.6s at equal accuracy and called it the admission's open number. The levers cut it 16% to 32.3s against 31.7s, with `r2_test` .8862 against .8858, so LightGBM's lead at matched accuracy falls from 22% to 2% and does not disappear. The reading stands as decision 97 wrote it, one step smaller: bonsai is faster at matched knobs and marginally slower at matched accuracy, and the remaining 0.6s is the same structural fact, that uncapped best-first must find a split for every leaf it creates, doubling the round count from 25,500 to 51,100.

**Consequence.** A leafwise GPU user gets 24.4s at 16M x 100 where decision 97 shipped 30.8s and the pre-registration bar was LightGBM's 32.4s. The registration, the dispatch grid, and the standings are untouched: no standings spec carries a leafwise arm, so nothing published moves and no axis needs a refresh. Three things stay open and are named rather than scheduled: the last 1.3s of fit to resident depthwise parity at 16M, which is now small enough that the next lever must be priced before it is built; small-node occupancy, which is real only when the mean node falls below roughly 150k rows and is therefore gated on that regime rather than on this ladder; and issue #278, the leafwise CPU-vs-GPU parity bound flaking at 1.06e-4 against its 1e-4 contract, which predates stage 3 and is a calibration decision on a contract surface (decision 40), not a drive-by widening.

**Rejected.** Carrying lever 3 anyway because its grow number was negative in every block (a lever that does not move fit is not a lever, and a stream plus an event is permanent complexity paid for a measurement inside the noise); holding lever 2 back on its missed 2% at the gate cell (the gate is a percentage and the saving is a fixed cost, so the gate cell is structurally the worst place to read it, and the same lever is 5x to 8x over the bar at the cells where the round is the fit); re-measuring CPU `leafwise` on this pod for a complete four-arm table (it is unchanged code, it costs more pod minutes than every device arm combined, and decision 97's numbers are the honest citation); widening the parity bound while the flake was in front of us (issue #278 exists so that the contract moves deliberately). **Reopener:** if a priced lever closes the last 1.3s to depthwise, or if a sub-150k-row regime makes small-node occupancy the top term, stage 4 opens with the same discipline: unprofiled wall clock, more than one scale, and a pre-registered gate read across the ladder.

## 99. `Dataset` takes a device hint, so the two-step form keeps device binning (adopted)

**Decision.** `bonsai.Dataset(X, y, ..., device="cuda", device_id=0)` bins on the device, and the two-step workflow reaches the same ingest path the fused `train(pairs, X, y)` call has used since decision 54. Until now it could not: device binning is chosen inside `make_labeled` from the grower name, and a standalone `Dataset` construction has no grower, so it always binned on the host. The cost was measured on one L40S at 4M x 100: fit 2.92s against 4.30s and peak host RSS 1.86GB against 2.51GB, identical r2, with the ingest profile reading `dbin=0.09s bin=0.00s` fused and `dbin=0.00s bin=1.14s` two-step. The benchmark harness hit exactly this and published it into a standings refresh before it was caught (issue #290 records the protocol response). Parity is the admission bar and it is met: 4M reads 3.207s fused against 3.209s hinted, 16M 12.056s against 12.005s, both inside interleaved-repeat noise on time and memory.

**Consequence.** A hyperparameter sweep binds its data once. The engine is rebuilt per `train` call, so a sweep over a host array pays mapper-fit plus device binning on every fit; over a device-hinted Dataset it pays once, measured at 3.63s against 0.74s over five fits at 16M, which is 5% of that sweep's wall clock and grows as the per-fit iteration count falls. Mismatch is resolved by materialization rather than refusal: a device-binned Dataset handed to a CPU grower fills its host bins on first host consumer, through the `call_once` path `Dataset` already had, and its model is byte-identical to a host Dataset's. A `parallel.device_id` that disagrees with the Dataset raises before any device work; `device="cuda"` without a build or a device raises, because it is an explicit request rather than an engine inference, while the oversized-`max_bin` ingest decline stays silent and reports `device == "cpu"` truthfully. Device-binned Datasets do not pickle. The constructor also gained `n_threads`, which it never honored before, so its binning ran outside `parallel.n_threads`.

**Rejected.** Binning both host and device eagerly (pays memory for a fallback most callers never take); refusing the CPU-grower combination outright (the materialization path already existed and refusing would make a reusable Dataset less reusable, which is the point of the feature); inferring the device from the first `train` call instead of taking a hint (binning would then happen at a moment the caller cannot see, and a sweep's first fit would pay a cost its siblings do not). **Reopener:** device-resident input (issue #289), which changes where the bytes come from rather than where they are binned, and shares this seam.

## 100. The device-leafwise campaign was measured on a host-binning harness (adopted)

**Decision.** Every bonsai CUDA number in decisions 95, 97, and 98 was measured through a benchmark harness that could not reach device binning, so the campaign's published absolutes overstate bonsai's fit time and understate the result it reports. The mechanism is decision 99's: at `4e035f0` the bench runner split its fit into `Dataset(X, y, max_bin)` then `train(pairs, ds)`, and a Dataset built before a grower is named always binned on the host, so every `cuda_` arm carried a host binned matrix, and about 2.5GB of extra host memory, for the whole fit (16M x 100 peak RSS reads 9.47GB in the affected ladders against 6.93GB on the fixed path). All three evidence files carry that commit and none carries the fix, `77633a6`: `leafwise-recheck-2026-08.jsonl` at `f5e7740`, `leafwise-ladder-2026-08.jsonl` at `2a23f33`, `leafwise-stage3-2026-08.jsonl` at `703a78e`. The bias ran one way, against bonsai: no reference arm touched the affected path. Decisions 95, 97, and 98 are not rewritten: this entry is the correction of record, in the pattern decision 95 used on decision 42. Evidence: `leafwise-correction-2026-08.jsonl` (one L40S, US-NC-1, 2026-08-03, the campaign's three device arms at the campaign's knobs, best of two reps, interleaved, unprofiled, on the fixed path, plus the uncapped arm and two instrument controls).

| rows | cuda leafwise | lgbm cuda | cuda dw (anchor) | decision 98 leafwise |
|---|--:|--:|--:|--:|
| 250k | 3.3s | 8.6s | 0.8s | 2.1s |
| 1M | 4.2s | 9.7s | 1.5s | 3.2s |
| 4M | 7.2s | 14.8s | 4.4s | 7.6s |
| 16M | 18.7s | 37.0s | 15.8s | 24.4s |

**The rental, stated before the readings that depend on it.** This is the slowest of the three L40S rentals the campaign has used: `lgbm_cuda`, which never touched the affected path, reads 16% to 29% above the times decision 98 published for it, largest at the small cells where the fit is host-side fixed cost. That is why the 250k and 1M leafwise cells read above their published numbers rather than below, and it is why no absolute here should be quoted against a campaign absolute without its reference arm. What survives the rental is the ratio, and a second reading of the load-bearing cell survives with it: the 2026-08-02 standings measured 16M x 100 on the fixed path on a rental that reproduces the campaign's reference times, at `cuda_leafwise` 13.8s against `lgbm_cuda` 31.8s. The distance between 18.7s and 13.8s is taken apart rather than left as pod luck: the bench driver's profile counters price at 0% to 2%, the `--data-cache` memmap every campaign ladder used costs another 12% at 16M (the standings run without it, and dropping it here reads 16.5s leafwise and 13.7s depthwise), and what remains is 19% of rental on identical protocol, in line with the reference arm's 16%.

**What moves.** The kill criterion pre-registered in issue #268 was met far more decisively than recorded: `cuda_leafwise` is 2.0x LightGBM's CUDA leaf-wise at 16M on this pod and 2.3x on the standings pod, where decision 97 published 5% of margin and decision 98 published 24%. The fit gap to the resident depthwise anchor at 16M is 2.9s here and 1.7s on the standings pod, not the 1.3s decision 98 recorded: the host binning pass sat on both planes and flattered their distance, so the leaf plane is not as close to depthwise parity as the record claims. Against decision 97's CPU `leafwise` column the plane is 4.1x to 11.8x rather than 6.4x to 9.2x, spread wider at both ends by the same rental effect. Decision 98's 21% to 27% cut on the stage 2 column stands as measured and is a floor: both ladders carried the same handicap, so removing it from both sides raises the percentage rather than lowering it.

**What does not move.** Every lever delta in the campaign was a same-path A/B with the handicap on both arms, so decision 98's levers 1 to 3 and doc 20's lever 4 stand exactly as measured. CPU `leafwise` binned on the host by design, in every run, so decision 95's headline (LightGBM's CUDA leaf-wise beats bonsai's CPU leafwise 5.3x at 16M) is untouched; only the `cuda_depthwise` anchor row of that recheck carried the handicap. No standings axis moves: the standings were re-measured on the fixed harness for 1.6.0 and already read the corrected numbers, which is how the drift was caught.

**The campaign's one negative finding inverts.** Decision 97 recorded the uncapped 16M arm at 38.6s against `lgbm_cuda`'s 31.6s and decision 98 at 32.3s against 31.7s, calling LightGBM's 2% lead at matched accuracy the honest half of the admission. On the fixed harness, same pod, same knobs, best of two reps: `cuda_leafwise` 27.5s at `r2_test` .8862 against `lgbm_cuda` 36.8s at .8858. bonsai leads the uncapped cell by 34% at matched accuracy. The structural fact decision 97 named is unchanged, that uncapped best-first must find a split for every leaf it creates and so pays 51,100 rounds against the capped tree's 25,500; what it does not do is cost the arm the comparison.

**Also recorded.** The bench driver forces `BONSAI_GROW_PROFILE`, `BONSAI_INGEST_PROFILE`, `BONSAI_CUDA_PROFILE`, and `BONSAI_FIT_PROFILE` on for every bonsai child, so both campaign ladders ran with the counters on despite decision 98 and the ledger describing the closing ladder as unprofiled; the ledger line is corrected here. Priced on this pod at every cell, the counters cost 0% to 2% of leafwise fit (16M: 18.8s against 18.7s), so nothing in the campaign turns on them, and `BONSAI_BENCH_NO_PROFILE=1` now exists so the question is measurable rather than assumed. The `--data-cache` memmap costs more than the instrument it was chosen over: 12% of leafwise fit at 16M, on a flag every campaign ladder passed and the standings do not, which is worth knowing before the next cross-ladder comparison is drawn. One further observation, not chased here: two of ten device-leafwise reps on the fixed path scored `r2_test` .8798 and .8781 where every other rep of the same cell scored .8793 and .8783, while the affected ladders and the standings reproduced their scores exactly across reps. That is run-to-run variation inside the fit, it is adjacent to issue #278's parity-bound flake, and it wants its own measurement rather than a sentence here.

**Rejected.** Deleting or rewriting the three superseded evidence files (they are dated records of what was measured, decisions 95, 97, and 98 cite them, and the archive moves by correction rather than by edit; the ledger sections that render them now say in prose that they carry the handicap and point here); re-measuring CPU `leafwise` (it binned on the host in every run, so nothing about it moved, and it costs more pod minutes than every device arm combined); publishing the correction from the standings alone (they carry one cell, not a ladder, and not the uncapped arm that inverts); renting a second pod for absolutes that match the campaign's rentals (three rentals already agree on the ratio, which is the claim, and decision 94's fleet-variance caveat is the standing answer for the rest). **Reopener:** issue #290 pins the ingest contract so that a device arm which silently bins on the host fails loudly instead of waiting for a reader to notice an odd table; if that lands and any published table still disagrees with it, this correction reopens rather than a fourth ladder being rented.

## 101. Perf rows report ingest and train for every arm (adopted)

*Status 2026-08-04: the committed standings behind this entry were measured before the two-step runner merged and carried no bonsai split; the 2026-08-04 refresh supersedes them with the split measured for every arm.*

**Decision.** Every perf row now carries `ingest_s` and `train_s` alongside `fit_s`, bonsai included. The bench runner fits through `Dataset(X, y, max_bin=..., device=...)` then `train(pairs, dataset)`; `fit_s` remains the outer wall clock over both and is never redefined as their sum. This was impossible until decision 99: a prebuilt `Dataset` without a device hint bins on the host whatever grower follows it, so its ingest number would describe a pipeline no GPU arm runs. That is not a hypothetical failure, it reached a published refresh and was withdrawn (decision 100).

**Consequence.** The split says something the total hides, and it says it against us. At 16M x 100 on one L40S, XGBoost trains in 7.0s against bonsai's 11.0s, 36% faster, while bonsai's ingest is 1.2s against XGBoost's 30.0s. The 3x margin in the total column is ingest, not boosting: bonsai bins on the device where the references sketch on the host and ship the result across the bus. Against LightGBM's CUDA leaf-wise and CatBoost's GPU oblivious, bonsai leads both halves (12.8s against 15.4s of training, 9.4s against 19.6s), so the deficit is specific to XGBoost's depth-wise kernels and is now a named target rather than an invisible one. The two halves also scale differently with hardware: on a second host with a faster card, bonsai's training halved while its ingest held flat and XGBoost's ingest got worse, which a single column cannot express.

**The standing evidence is a parity arm, not an assertion.** Every refresh fits the anchor cell through both the fused and two-step forms, interleaved on the same pod, banded at 5% on `fit_s` and on peak RSS, and a failure stops the supersession before it touches a file rather than annotating the result. This run read 12.01s fused against 11.99s two-step, 0.2% apart, with the split at ingest 1.05s and train 10.94s. Without that arm a regression in the device hint would silently reintroduce host binning and post a plausible ingest number, which is exactly how the withdrawn refresh passed unnoticed.

**Rejected.** Reporting the split for the reference libraries only, which is what the harness did and what made bonsai's column read as absent rather than fast; deriving bonsai's ingest from the profile counters, which perturb short rounds and are the wrong instrument (decision 98); and reusing one `Dataset` across repeats or variants, which amortizes a cost each row must be charged in full. The amortization is real and worth measuring, but as its own study: one device Dataset served three growers in 1.1s of ingest plus 25.3s of fits rather than three separate ingests.

## 102. Device-resident input bins in place (adopted)

**Decision.** Any CUDA array supporting DLPack is accepted as training data and binned where it already lives, so a caller whose data is on the GPU reaches a trained model without a host round trip. This closes the one regime where bonsai's ingest advantage inverted: for host input, device binning means the host never materializes a second full-size copy (decision 54, and the headroom column added under issue #291 prices it at roughly 8% of the input array against 2.5x to 3.5x for the reference libraries), but a caller already holding a device array had to copy down to host memory first, which XGBoost's `QuantileDMatrix` avoids by sketching in place. Measured on one L40S at 4M x 100 with the data already resident, 20 iterations: 1.30s and 1.28s to download and fit, against 1.04s and 0.96s binning in place, maximum prediction difference 9.5e-7.

**Consequence.** The bin mapper still cuts on a host sample, and deliberately so: sampling on the device would change the sampled set and therefore the model, which doc 15's phase-2 rule forbids. The device arm instead gathers exactly the rows `bin_sample_rows` names into a compact block, downloads that block, and runs the ordinary `BinMappers::fit`, which re-samples it and finds it already at sample size. Cuts are bit-identical to the host path by construction, and the download is the sample rather than the matrix: 80MB against 6.4GB at 16M x 100. Ownership is borrow-for-one-call: ingest copies into a plane owning its own device memory, so nothing bonsai holds points into the caller's allocation afterward. Stream ordering is the producer's under DLPack, which synchronizes at export, and bonsai reads on the default stream, so there is no handle to plumb and nothing to wait on. Labels and weights are accepted device-resident and downloaded once, because every consumer of them (the host objective, the eval loop, the resident uploader) reads them host-side; when that stops being true they can stay. Mismatch follows decision 99: a device-id disagreement raises before any device work, a device request without a device or a build raises, and `device="cpu"` with device input raises rather than silently copying back. The `cuda_ingest` bin-count decline inverts here, since there is no host copy to fall back to: the device arm always mints a plane, and a declining grower takes the lazy host materialization instead.

**Parity, stated in two parts because it is two claims.** For CPU growers the model artifact is byte-identical to the host path, which is the bin-identity claim: the grower reads the device's bins materialized on the host. For `cuda_*` growers the contract is tolerance-equal at 1e-4 on predictions and nothing stronger, because device float atomics are not reproducible run to run; three repeats of one `cuda_depthwise` fit on one unchanged host array produced three different model hashes on the same pod. That is the same phenomenon behind the r2 spread recorded in decision 100 and the parity-bound flake of issue #278, and it is a property of the device plane rather than of this feature.

**Rejected.** Sampling the bin mapper's cuts on the device (changes the model, and the compact-gather download costs 80MB to avoid that); retaining a borrowed device pointer past the call (an aliasing contract callers cannot see); accepting device input into the standings ladders (every arm is handed the same host array, and a device-input study is a separate measurement where every arm that supports such input is measured on it); parsing `__cuda_array_interface__` by hand (built first and replaced before ship: nanobind's DLPack import runs the same validation with no owned code, and the one producer class it does not reach, numba's device array, arrives through `cupy.asarray`). **Reopener:** keeping labels device-resident once a consumer can read them there.

## 103. The device bin plane is tile-blocked (adopted)

Standings: rows, width, frontier, airline

**Decision.** The CUDA binned matrix is written and read in feature tiles of 8: feature f lives in tile f / 8 at strip position f % 8, tile t starts at cell `n_rows * t * 8`, and one row's strip inside it is the tail-aware `min(8, n_feats - t * 8)` cells wide. This is the layout decision 89 adopted on the host (`Dataset::row_major_bins`), at the width device shared memory allows. It is the plane, not a copy of it: both ingest arms write it, the host upload stages it, `materialize` un-tiles on the way home because the host store is per-feature columns, and every device reader moved with it, the histogram build by a new kernel and the small-node build, the partition count and the resident epilogue by one index expression each. The layout arithmetic lives in exactly one function and the width is one constant. The histogram build owns one tile per block, loads a row's strip in a single aligned vector load, reads the row id and its gradient pair once per tile rather than once per feature, and keeps one sub-histogram per lane with no warp-parity duplication, so eight lanes at 255 bins ask 16 KiB and the whole build stays inside the static shared budget with no opt-in. A tree that subsamples features rides the same tiles through a per-feature slot map, and bin counts too wide for a tile's sub-histograms fall back to one feature per block on the same plane, which costs what the feature-major plane cost, so the wide-bin envelope is unchanged.

**Consequence.** Measured on one L40S (EU-NL-1, interleaved A/B against main, three reps, 16M x 100), where `tile 8` is what ships and `tile 16` is the width the depthwise build alone would have picked:

| cell | main | tile 16 | tile 8 |
|---|--:|--:|--:|
| depthwise depth 8, train | 11.02s | 6.85s | **7.23s** |
| depthwise depth 8, `adv_hist` | 7.95s | 2.49s | **3.07s** |
| depthwise depth 4, train | 3.58s | 3.70s | **3.61s** |
| leafwise 16M, train | 12.64s | 14.65s | **11.31s** |

The histogram kernel the whole gap lived in falls 61%, and the depth-8 fit falls 34%. The crown reading is the one that matters: at this cell bonsai's depthwise arm was 59% behind XGBoost's trainer and is now inside 4%, which is a statistical tie at this pod's spread. Leafwise gains 10.5% for free, having never been the target, depth 4 is neutral at 0.8%, and `r2_test` spans {0.8793, 0.8798} on both branches equally, the known flutter rather than a difference. The width is 8 and not 16 because one plane serves both growers and the choice is therefore joint: 16 is better for depthwise by 5% of fit and worse for leafwise by 15.9%, which fails the pre-registered 5% leaf-plane bar, and it fails it for the reason doc 20's lever 4 predicted, since a leafwise round histograms one node and a tile-narrowed grid stops filling the device. 8 passes every pre-registered bar and improves both planes, so it ships, and the depthwise width question is not closed but priced: **reopener,** an adaptive accumulator width (16-bit or quantised sub-histograms, named and never opened in doc 20) would halve the tile's shared cost and put 16 or wider back on the table for both planes at once.

**Rejected.** Feature grouping at fixed layout, refuted by measurement in PR #332: it varied the block's feature width while the plane stayed feature-major, so it paid a G-fold shared footprint and collected none of the gather benefit, which exists only when the bytes a block needs are adjacent, and its G=8 arm additionally kept the warp-parity duplication and so ran at half the occupancy before reading a byte differently. Row reordering into node order, which sounds free because the partition already computes the ordering and is not: the partition permutes the row list, not the matrix, so a contiguous gather would mean physically permuting the whole matrix once per level, priced at 25.6 GB per tree against the 107 GB it removes, and it is the only option here that changes what a row id means. Full row-major (ELLPACK shape), dominated at this feature count: a block cannot own 100 features of shared histogram, so it reads a partial unaligned strip out of each row and reintroduces exactly the partial-sector waste tiling removes, while `route_add` loses its coalescing and both the host upload and `materialize` need a transpose anyway. Per-plane widths, one for depthwise and one for leafwise, which is not an option without a second copy of the matrix and therefore the 1.6 GB the probe form was retired to avoid. The probe record, including the pre-registered criteria this measurement was read against, is PR #335; the shipping measurement is this PR's.

## 104. The standings are six scenarios on two gated planes (adopted)

Standings: gpu-tall, gpu-wide, gpu-extreme, cpu-tall, cpu-wide, gpu-early-stop

**Decision.** The perf standings are redesigned from four grid axes to six single-cell scenarios: an iso-volume tall/wide pair per plane (2^31 cells on GPU, 2^28 on CPU), one VRAM-ceiling extreme on GPU where an OOM is a published result, and early stopping as a standing behavior axis. Every scenario publishes the same dimensions, the fixed/variable split (`ingest_s`, `train_s`) beside peak host RSS and per-process device memory, with one fused wall clock at gpu-tall from the parity arm. Arms pair by growth strategy on one page, grower by grower, hardware never mixed in a table. The standings card moves to the RTX PRO 6000 Blackwell (96GB), the card the extreme scenario is sized to. Axes carry a plane, and the release gate skips any axis whose plane's sources are byte-unchanged since its refresh, so a one-plane change re-measures one plane.

**Consequence.** A full refresh is 48 jobs where the retired grids ran hundreds, and a routine one is smaller still under the plane gate; the three-hour refresh that motivated this (issue #318) becomes tens of minutes. The framing complaint is answered structurally: fixed and variable costs are separate columns rather than a footnote under a total, memory is two honest numbers rather than one host figure, and the tall/wide contrast replaces ladders whose interior rungs backed no claim. The grower rename rides along (decision at issue #305, shipped in the same train): the growth policy is levelwise; the tree shape remains oblivious where CatBoost's vocabulary is meant.

**Retired.** The rows, width, shape, frontier, and airline axes and their files; the accuracy-time frontier's unique content (the capacity sweep) and the airline suite's real-data story leave the standings entirely, by ruling, and twenty-three closed-campaign evidence files leave the tree with them. Git history is the archive (decision 92), and the generated archive page maps every retired record to its decisions and the ref where its data lives. The renderer fell by a fifth of the repo's bench tooling in the same stroke.

**Rejected.** Per-grower pages (one page with panels compares better); keeping the airline axis under the canonical Pafka protocol (ruled out rather than filled); a per-plane width or per-scenario knob surface (SCALING stays the one knob family); resolving archive refs at render time (shallow CI checkouts cannot see deleted-file history, so refs are pinned at generation). **Reopener:** doc 21's component axis joins this registry with `plane: gpu` when its design is built; the extreme scenario re-sizes if the standings card changes.

## 105. The sparse fill is one composition: buffer and reduce (adopted)

**Decision.** The CPU sparse-node histogram fill becomes a single decomposition, admitted as the default and as the only path (issue #360). A level's sparse nodes are cut into fixed 1024-row blocks; the node-major block list is split into `n_threads` contiguous ranges; the lowest range touching a node accumulates straight into that node's arena; every higher range owns one partial, zeroed on first touch and skipped entirely when never touched; a second pass over `(node, feature)` pairs sums the partials in ascending range order. The per-node block plan it replaces is deleted with it: block counts derived from node size, selection width and bin footprint and capped at four blocks per thread, one partial per block, a merge per multi-block node. So is `BONSAI_HIST_REDUCE`, the env toggle that selected between the two during the A/B. Dense-node routing to the column fill is untouched, and the grain is a constant because a block is streamed once and never re-walked, so its size gates no cache reuse and trades only balance against per-block setup.

**Measured.** The gain is small and one-sided, which is what carried it. Train falls 3% on the 12-thread EPYC, reads 0 to -2% on the 16-thread Xeon, and shows no separable change on the M2 against a rep-to-rep drift near 1s at 2M x 128, depth 8, 100 iterations, 8 threads. Across the interleaved paired reps on every host the new arm never lost one. Peak RSS falls, and that part is structural rather than tuned: because the block list is node-major and thread ranges are contiguous, the ranges touching a node form a run, the runs telescope, and a level needs at most `n_threads - 1` partial buffers however many nodes it holds. The old plan bounded blocks per node and paid a partial for each, so its slab grew as `O(n_threads x n_nodes)` where this one is `O(n_threads)`. At depth 8 that is orders of magnitude.

**Consequence.** Models shift. A node's summation order now depends on the level-wide partition, so at a fixed thread count and a fixed dataset the depthwise and levelwise CPU model bytes differ from every version before this one, and the next standings refresh re-baselines rather than compares. Determinism itself is unchanged and stated in `architecture/7-parallel.md`: buffers are keyed by the partition index rather than the OpenMP worker, so scheduling cannot reorder a sum, and a fixed `n_threads` reproduces. The contract's dependency list is wider than it was, because the block list is cut level-wide: one node's row count moves another node's cut points. The load-balance trade is written down in the same place, because the obvious future fix silently undoes the memory result. Emitting exactly `n_threads` work units is what buys the partial bound, and it degrades `for_each_index` to chunk 1 with nothing left to steal, which is the static partition that doc's scheduling rationale warns about on asymmetric cores. The bound and the balance are one dial and this fill picks the bound.

**Rejected.** Shipping the toggle, which the design review priced as the one outcome worse than both alternatives on every axis it could price: two schedules to keep correct, two determinism stories under a doc that describes one, roughly 175 lines dead on arrival, a knob whose value `1` did not mean 1, and a fill path no test could reach because the selector read the environment. Default or decline, and the measurement decided which. Also rejected: grafting one ingredient of the reference decomposition at a time, refuted twice before this composition was tried whole; and a work-proportional block count, which the old plan needed because a block cost a full slab zero plus a merge to start, and which this one does not, since lazy per-range zeroing and a rebuild of the histogram bases only on node change make a block nearly free to begin.

## 106. Per-candidate min_data_in_leaf: declined by measurement (adopted)

**Decision.** bonsai keeps its node-level row floor; the per-candidate child floor LightGBM and XGBoost apply is not adopted. Measured at zero core cost via the `min_child_hess` identity (MSE writes a per-row hessian of 1.0, so `min_child_hess = 20` is the feature for regression): on the 55-dataset quality suite over 3 seeds, the strict arm is net negative (20 datasets beyond the decision-55 band, 16 losses; AUC mean -0.008, rmse mean +0.65%), and XGBoost's own `min_child_weight` toggle loses the same way on the same datasets, so the floor is an over-aggressive regularization default rather than a bonsai deficiency. Sub-floor leaves are common (median 25% of leaves) but hold ~1% of training rows. Evidence in PR #379's record; the probe was deliberately not kept.

**Consequence.** The semantics live in one place: the interop tables document the non-equivalence at the LightGBM/CatBoost `min_data_in_leaf` rows, and the parameters page now states what the knob gates. `min_child_hess = 20` expresses the strict form today, exactly a 20-row floor under squared error. Side finding: mapping `min_data_in_leaf = 20` onto XGBoost's `min_child_weight = 20` (decision 68's matched-knob rule) costs XGBoost beyond the band on 23 of 55 datasets; the rule stays, but the handicap is measured now, not assumed. **Reopener:** a workload whose sub-floor leaves carry real mass, or a row-expressed classification floor beating the default beyond the band.

## 106. The hist chunk axis fills the card (adopted)

**Decision.** `launch_hist` owns the chunk policy for all four of its callers: `n_chunks` is the larger of the row-split term (`max_rows / 32768`) and a fill term, `ceil(sm_count * 4 / (grid_x * n_nodes))`, clamped to [1, 64]. Both histogram kernels return before their shared zero when the chunk starts past the node's row count, a block-uniform exit that makes overshooting the chunk axis free on small nodes. The SM count is read once beside the shared-limit probe; a failed query disables the fill term rather than the launch.

**Attribution.** The small-cell fit-time trade published with the tiled plane (decision 103, issue #340) was occupancy starvation, not per-block fixed cost. Tiling divides `grid.x` by the tile width, and a shallow level over small data launches `tiles x nodes` blocks: a fraction of a 142-SM L40S, while the Orin's 8 SMs never notice. The three-arm A/B on the Orin (feature-major plane, tiled plane + per-feature kernel, tiled plane + tiled kernel; 3 interleaved reps) has the tiled kernel WINNING the exact cells that regress on the L40S, 7.33s to 5.41s of level-hist at 1M x 128, which rules the per-block cost story out on the device where fixed costs bite hardest.

**Measured.** Same-pod interleaved medians on one L40S (US-TX-3, 3 reps, `r2_test` a single value per cell with one 4th-decimal flutter of the issue #278 class): train at 262k x 128 falls 30.4% depthwise and 22.0% levelwise; 1M falls 6.3% and 8.1%; leafwise at 1M falls 35.3% (4.20s to 2.72s), because its per-round build launches one node's tiles and was the most starved caller of all; 4M and 16M read -0.6% to -0.0%, the no-regression bar met exactly. The Orin guard pair is a no-op (5.41s vs 5.40s), as the fill term predicts for a narrow device.

**Rejected.** Raising the small-node cutoff: the 2026-08-17 sweep measured every cutoff above 512 worse at every cell, monotonically. Per-node chunk lists: they shrink the chunk axis where the mechanism wants it grown, and the early exit already prices uniform overshoot at a block launch. The early exit alone: measured a no-op at uniform cells on the Orin (workless chunks are rare without skew), kept solely as the guard the fill term composes with. Routing more nodes to `hist_small_kernel`: unchanged at 512, per the same sweep.

## 107. Typed params are generated from the section registry (adopted)

**Decision.** The Python surface gains `bonsai.Params`, one frozen dataclass per config section with every field defaulting to `None` ("leave the library default"), plus a `train()` wrapper accepting `Params` or a dotted-key dict, both rendered to the unchanged `(str, str)` pairs wire format (accepted publicly in the first cut, retired the same day; the Consequence records the turn). The dataclasses are not written by hand: a `_bonsai._params_schema()` binding folds the same `all_sections` tuple `dump_toml` folds, emitting name, C++ type, and default per field, and `scripts/gen_params_py.py` renders `bonsai/_params.py` from it at build time, ordered before the nanobind stub the way the stub is ordered after the extension. Behavior (`to_dict`/`from_dict`, `|` merge with right-side-wins, the sparse repr) lives on a committed mixin, so the generator stays a renderer.

**Consequence.** The registry stays the single source: a new `field<&SubConfig::name>()` line appears in `Params` on the next build with its real C++ type, with no Python mirror to drift (the TOML-inference alternative types `lambda_l1 = 0.0`'s whole-number rendering as int; the registry binding types it float). Misspelled knobs fail at `Params`/`from_dict` construction with the section's legal names in the message instead of at fit time in C++. `Params | {"tree.max_depth": d}` is the sweep idiom, and the dotted key doubling as the optuna trial name makes objectives one merge long. The pairs wire format is untouched, so every model-hash gate holds; `data.*` and CLI-only sections generate like the rest rather than maintaining a curated subset. The bench seating landed in the same change set: `interop.to_*` accepts a `Params` (the `dict(pairs)` normalization grows one isinstance), the `*_core` builders state their cells as `Params.from_dict` literals (retiring the `_BONSAI_KEY` hand mirror and `_translated`), `BONSAI_CAMPAIGN_PARAMS` derives its values from `CAMPAIGN` under registry validation, and a spec's `defaults` block rejects unknown keys at load. With bonsai's one production consumer being its own author, the pairs form then retired from the public wrapper in the same change set: `train` accepts `Params | dict | None` and raises on a pairs list with the `dict(pairs)` escape named, `interop.from_*` returns a `Params` with typed values (the round trip through `to_*` is now exact rather than stringified), and the pairs remain only as the native wire format under the wrapper. The `config=` argument then retired from `train` and the estimators (a mix of concerns, Daniel's call): `Params.from_toml(path)` carries only the keys a TOML file states — parsed by a `typed_overrides` walker in the C++ config layer, so no Python TOML dependency and no 3.11 floor — and `from_toml(path) | overrides` expresses the `-c` + `--set` layering through the one params argument. `Params.from_model` rides the same binding (the resolved config, every key set), closing that deferred item. The Dataset path's `[bin_mapper]` config-file check is subsumed: a TOML base arrives as ordinary pairs and hits the existing rejection. Still deferred: estimator re-seating (`params=` and `fit` accepting `Params`) and per-round callbacks for optuna pruning.

**Rejected.** A hand-written Python mirror (a second source of truth, the exact drift class `bench/params.py`'s docstring documents). Runtime TOML parsing of `default_config_toml()` (needs tomli on 3.9/3.10, and inferred types carry the int-for-float trap). A committed generated file with a CI staleness check (reviewable, but adds a render ritual and install plumbing the stub precedent already avoids; Daniel's call, build-time only). Flat fields on one class (`seed`, `random_seed`, and `feature_seed` collide, and section names carry meaning the flat space loses).

## 108. Prediction and TreeSHAP serve the resident Dataset; SHAP takes a crown (adopted)

*Status 2026-08-25: the oblivious half of the decline recorded here is lifted by decision 111; multiclass still declines.*

**Decision.** Every X-taking Model method accepts a Dataset, and the dispatch picks the strongest route the Dataset supports: the model's own cuts route bin space (exact, no raw rows, DLPack builds included), foreign cuts fall back to the retained host matrix, neither raises with both remedies. Width-1 dense-tree models additionally route two device planes: whole-ensemble predict (a route_add generalization over the resident bins, plan receipt keyed on a booster mutation epoch) and TreeSHAP (path decomposition into 8-byte u8 bin-interval elements evaluated by a division-free closed form, thread per row-path, fp32 walk under a double-precision host epilogue). Oblivious and multiclass models decline to the host bin walk. Design record: architecture/22.

**Measured** (RTX PRO 6000 Blackwell Server, same pod, interleaved arms, xgboost 3.3.0 on cuda:0 confirmed via save_config). The gpu-shap axis (500 trees, SHAP over the held-out test matrix per protocol, 2 repeats, sha ffce72d): bonsai beats xgboost's GPU engine in every cell, 0.57s vs 2.38s at 1M x 128 d6 (4.2x), 1.34s vs 5.61s at 4M x 128 (4.2x), 0.84s vs 2.57s at 1M x 512 (3.0x), 2.46s vs 7.46s at 1M x 128 d8 (3.0x); context arms: CatBoost CPU SHAP 3.2-8.4s, LightGBM CPU SHAP 173-378s with the d8 cell timing out at 900s. A deeper hand ladder over the full training matrix at the same shapes: 2.32s vs 11.44s, 8.59s vs 46.25s, 10.43s vs 44.02s (4.2-5.4x). Additivity residuals same order both arms (bonsai 8.2e-6 to 1.38e-5, xgboost 3.8e-6 to 5.8e-6); bonsai's closed form is exact in exact arithmetic where xgboost's QuadratureSHAP (its 3.3 replacement for GPUTreeShap) is a quadrature approximation, and the fp32 gap against bonsai's own fp64 host walk measured 7.9e-6 at 200 trees. Predict from the resident Dataset 0.011-0.038s against xgboost inplace_predict 0.40-1.75s from host numpy; that column measures the resident-loop workload (xgboost pays the host-to-device movement per call, which is the cost this design removes), not a kernel-versus-kernel claim. Device parity held on both sm_87 and Blackwell: predict bit-equal after spelling the fma rounding out, SHAP within two orders of its tolerance. Host-side wins ride along: the per-row bias walk deleted (leafwise pred_contribs -30% on a Mac at 300k x 32 x 200) and the oblivious re-densify cached per epoch.

**Consequence.** The select-then-refit loop runs end to end on the resident bins: bin once, fit, eval, explain, filter outside, refit, with the only recurring host traffic being the results. The u8 bin compare and the epoch-keyed plan cache are edges no competitor algorithm swap removes: xgboost dequantizes its ellpack per read and recompresses its model every ShapValues call. Watch items, recorded not hidden: the fp32 additivity residual grows with tree count and crosses 1e-5 at 500 trees depth 8 (the escape today is passing the raw matrix, which runs the fp64 host walk; an fp64 kernel instantiation is the fix if a workload needs the device at that fidelity), and the v1 kernel geometry left shared-memory row staging, atomic-contention mitigation, and K=32 register pressure unpriced. The gpu-shap axis publishes the category with a fidelity column beside the throughput race.

**Rejected.** Thread-per-row iterative DFS (local-memory bound, quadratic in depth, priced in architecture/22); warp-per-path GPUTreeShap geometry at bonsai's merged path lengths (shuffle machinery to parallelize six iterations, lanes idle); a rows= filter parameter (filtering happens outside in numpy against cheap predictions); a device predicate mini-language (same reason); folding the bias into the kernel (it is one fp64 scalar per model).

Standings: gpu-shap

## 109. The bin store is the sharing unit; the fit keys its own labels (adopted)

A per-member call-site census of `Dataset`'s nine change reasons, production separated from tests, preceded the cut (the census itself was working material, not kept). The cut: `BinStore` owns the binned matrix (columns, plane, lazy host bins, width flag, the row mirror) and the cuts, because bins are unreadable without them; `Dataset` remains the composite consumers hold: labels, weights, and which rows, over a `shared_ptr<BinStore const>`. The name does not invert: the census showed every production consumer reads bins and labels through one object, and the Python `Dataset` plus 175 doc mentions make the composite the public noun. Forwarders keep all existing call sites and keep `bin_at` header-inline.

The behavioral half re-keys the caches. Bins key by the store's address, corroborated by geometry, and on the adopted-plane path the key holds the plane alive, so that address cannot be recycled under it. Labels and the resident arming key by MINTED tokens (`LabelsId` per labels block, `FitId` per fit specification: new for every factory product and every view, shared by copies), never by address: a first cut keyed labels by the Meta block's address and review round 3 showed allocator reuse re-creates exactly the trap being removed, one level down, and that the host-side `resident_train_` pointer had the same defect with deterministic stack-slot reuse. Tokens are monotone and zero is never minted, so equality means identity with no reuse caveat: a view shares its parent's `LabelsId` (skips the label re-upload) but carries its own `FitId` (re-arms the resident state, whose row list is per-fit).

Measured: interleaved same-machine A/B (B,A,B,A blocks, 2M x 32 x 150 iters), `populate` min 3.45s on both arms, `finalize` 0.49-0.53s on both; the forwarder hop is paid per column because the fills hoist their spans. Wire identity `55c6fe308852d9bb` unmoved. Rejected: renaming the store `Dataset` and the residue `Fit` (inverts the public noun against every consumer's read pattern); moving the row view to a `grow()` argument (churns the IBooster seam for zero census evidence); `std::function` in the mirror's mint seam (type erasure on a minting path for one caller).

## 111. The device plan input owns what it lends (adopted)

**Decision.** `DevicePlanInput`, the one seam both device planes read, carries an optional owner beside its `span<DenseTree const>`. A dense booster lends a view of its own ensemble and leaves the owner null; a levelwise booster attaches the epoch-cached dense equivalent that its host TreeSHAP path already builds, and points the span at that. Nothing else changes: the packers take `DenseTree` whichever grower produced it, and they copy and upload at pack time, so the owner has only to outlive the pack call, which the caller's own local already does. Levelwise width-1 models therefore take device predict and device TreeSHAP from a resident Dataset with no change above the seam. Multiclass keeps declining, through the empty default and through the width gate above it.

**Why it was ever declined.** Not for a kernel reason. `dense_equivalent` has always expanded an oblivious tree into the shape TreeSHAP's cover-weighted walk is written against, which is how levelwise `pred_contribs` works on the host. What blocked the device was purely lifetime: the seam's contract was that the span stays valid until the booster mutates, which a booster lending its own vector can promise and one converting into a cache cannot. Recording the reason matters because the shape recurs: a borrow-versus-own mismatch at a type-erased seam reads like a missing capability from either side of it.

**Consequence.** Densification mints a perfect tree, so a depth-d levelwise tree packs 2^d paths whatever its live coverage. An oblivious tree splits one feature per level and may split the same feature at two levels, so the expansion contains corners asserting `f <= t_i` and `f > t_j` at once: the dead slots are unreachable by construction, not merely unvisited, and a measured fixture has 106 of its 256 leaves dead with no input of either distribution routing to one. What the device therefore meets is the unsatisfied case, a zero-cover element in a path the row does not follow, and it meets it constantly. The two walks reach it differently: the host form divides by the cover fraction and so has to guard the branch off, while the device closed form only ever multiplies by it and needs no guard. That they agree anyway is what the parity fixtures check, and a host case pins the structural claim so the device fixtures cannot quietly stop carrying dead leaves. Correctness is device-verified on the Orin, and the throughput question the 2^d raised is now measured and answered the other way. Same L40S, arms interleaved, 1M x 128 at 200 trees, TreeSHAP over the held-out matrix, the levelwise device path against the released wheel where it declines: 22.0x at depth 4, 50.8x at 6, 75.2x at 8, 83.4x at 10, the ratio growing with depth. Levelwise on the device also beats DEPTHWISE on the device from depth 6 up, 5.40s against 10.24s at depth 10, which is the opposite of what packing 2^d paths predicts. The cause is the structure that mints the dead slots: an oblivious tree repeats features across levels, a path merges one element per DISTINCT feature, so its merged paths are short where a depthwise path can carry one element per level. Merged length picks the kernel template and the polynomial degree, so levelwise packs more paths and pays less for each, and the per-path saving wins. The same shortness shows up in fidelity: worst additivity residual 7.8e-6 against depthwise's 1.0e-5 at depth 10, which is decision 108's watch item reached at 200 trees rather than 500. The control is depthwise, which takes the device path in both arms and reads -0.1% to -1.0% across the cells that resolve, so the levelwise effect is not pod drift. One host-side cost is charged here rather than left to be discovered: the plan input stopped being free for an oblivious booster, so `predict_on_device` and `contribs_on_device` now ask for it only after the gates that do not need it, or a levelwise model predicting from a host-binned Dataset would build and retain a dense ensemble it never packs.

**Rejected.** Making the input own always, with dense synthesizing a non-owning aliasing `shared_ptr` (dresses a borrow as ownership and churns every consumer for a distinction none of them reads); a `variant` of span and owner (forces visitation at each read site, and the owner is a lifetime anchor rather than state a reader branches on); pruning the dead paths at pack time (a row can route into one, and the contributions must still sum to the prediction).

## 112. Benchmark data is drawn in fixed blocks, and the recipe is numbered (adopted)

**Decision.** `bonsai.bench.synth.gen_data` draws its rows in `N_BLOCKS` fixed contiguous blocks, one spawned `SeedSequence` stream each, over a thread pool; a stream of its own carries the whole-matrix draws (the informative-feature choice, and the sigma the noise is scaled by). The block count is a constant in the module, never a core count, so the same cell produces the same bytes on a 27-core pod and a 128-core one; the worker count is sized from `runlog.cpu_quota()` and never reaches the output. Both properties are tests, not comments. `DATA_RECIPE` is 2, every emitted row carries it, and it is part of the `BONSAI_BENCH_DATA_CACHE` key.

**Why re-anchor rather than parallelize bit-exactly.** PCG64 exposes `advance`, so workers could jump to their block offsets and reproduce recipe 1 byte for byte. That version would hard-code how many raw draws numpy consumes per generated element, which is exactly the thing NEP 19 reserves the right to change: `RandomState` is frozen, `Generator` streams may move across feature releases. The archive's byte-identity already holds only within a numpy version, which is why every row records `libs.numpy`. A numbered recipe with a deliberate, recorded break is more honest than an implicit dependency that a routine numpy bump can shift underneath the goldens.

**Measured** (M2, 8 cores, interleaved arms, min of 3): 1M x 100 recipe 1 0.34s against recipe 2 0.08s (4.1x), 2M x 128 0.77s against 0.16s (4.7x). The block fill scales near-linearly because numpy's `Generator` releases the GIL in its bulk fill path and each block owns its own bit generator, so the per-generator lock is never contended: measured 1.86x at 2 threads, 5.57x at 8, flattening at 16 where the fill is memory-bandwidth-bound. What this buys is not the seconds: at the extreme cell the old recipe spent roughly 30 to 45 minutes of a 48-minute sweep drawing random numbers on one core, and a shorter sweep is a smaller window for a pod to die mid-run, which has cost a session once already.

**Consequence.** Recipe-1 and recipe-2 rows are not comparable and the protocol says so where the data is described. The registry does not gate on this: `plane_digest` covers `src/`, `include/`, and `scripts/model_hash.py`, so a recipe change staleness-flags nothing, and the discipline that keeps a results directory honest is landing the change with a full refresh rather than between refreshes. `scripts/model_hash.py`'s frozen linear variant is untouched by design; the wire-identity gate is unaffected and `55c6fe308852d9bb` is unmoved.

**Rejected.** Bit-exact parallelism via `advance` (couples the goldens to numpy's per-element draw count, above); deriving the block count from the host's cores (a 27-core pod and a 128-core pod would then generate different data for one cell, which is strictly worse than generating serially); processes instead of threads (the extreme cell's matrix is 64GiB and would have to cross a pickle boundary, and the GIL is already released where the time goes); blocking the train and test ranges separately to make the train rows independent of `n_test` (the old recipe did not have that property either, since `y.std()` spans both, and nothing reads it).

## 113. The refresh is re-anchored on one card and one draw per cell (adopted)

**Decision.** Three changes to the standings refresh, taken together because a full re-measure was already due and a comparability break is paid for once. **The extreme axis moves to a single card.** Its RAM floor was 320GB, sized to catboost's 196.3GB of ingest copies rather than to bonsai, which peaks at 66.7GB on the same 2^34-cell input; that floor forced a two-GPU draw at twice the rate. The floor is now 150GB, one 188GB card clears it, and catboost publishes an OOM beside xgboost's, which the protocol already treats as a result because capacity is a claim. **The data cache is exported, and loads whole.** It was refused entirely because it handed back memmaps, which fault pages inside `fit()` and so moved measured `fit_s` unequally across libraries, only some of which re-touch the raw arrays after ingest. A full read completes before `worker()` opens a timer, so the fit sees memory indistinguishable from freshly drawn arrays; a cell too large to hold twice declines to the generator, since the cache is an optimization and never a reason to run out of memory. **The pod waits are bounded.** `measure` polls for the DONE marker under a per-axis deadline and a consecutive-unreachable limit, and the create ladder is walked to a deadline with backoff instead of twice.

**Why now.** Every axis was stale after decision 111, so the release refresh re-measures all nine whatever else changes. Landing the recipe re-anchor (decision 112), the extreme axis's host of record, and the cache in the same sweep costs one break in comparability instead of three, and each of them alone would otherwise have had to wait for a refresh someone was already paying for.

**Consequence.** Rows before this refresh and after are not comparable, which was already true of the recipe change and is now true for one more reason on the extreme axis: its host of record is a 188GB single-card container rather than a 351GB two-card one, and catboost's row there changes from a finish to an OOM. The published extreme table keeps a throughput ratio, since lightgbm's 116.2GB still fits, so the axis reads as bonsai against a survivor plus two arms that cannot run rather than as bonsai alone. Cost falls roughly in half on that axis, and the per-worker regeneration the cache removes was the sweep's largest untimed cost after decision 112 halved it: on gpu-tall, twelve draws become one.

**Rejected.** Sizing the extreme cell to bonsai's own ceiling so every competitor OOMs (about 1.6 to 2.2x the current rows before host generation binds, and it costs the ratio: one number and three OOMs is a weaker table than a measured 7.2x against a survivor, and a cell chosen so competitors fail invites the objection that it was); keeping the 320GB floor and accepting the two-GPU rate (it buys nothing bonsai needs); a memmap-mode cache with a correction factor (the distortion is unequal across libraries, so no single factor exists).

---

## 114. The architecture directory is dissolved; claims are routed and linted (adopted)

**Decision.** `docs/architecture/` (23 files, 3,264 lines) is deleted. A five-way audit measured it 76% cuttable, found four docs asserting the opposite of the shipped code (a storage design recorded as rejected that shipped as `BinColumns`; a 230-line deliberation over a four-axis dispatch that shipped as three; an API sketch with zero matching symbols; stale "not here" lists naming shipped features), and mapped 19 claims restated across 94 sites. Content now routes by the table in `docs/STYLE.md` "Where things live": conventions to `CLAUDE.md`, single-file constraints to comments at the definition site, cross-file contracts to `docs/invariants.md` under a docs-check lint that fails when a cited path or symbol stops resolving, rationale here, pedagogy to guide/ and learn/. This supersedes the "design rationale lives in docs/architecture/, told once" convention.

**Consequence.** The two homes that stay must be kept honest by machine: invariants.md by the resolution lint, this log by append-only banners (this entry adds four). Historical citations of architecture docs resolve via links pinned to the last commit carrying the directory. The docs corpus drops by roughly a third; the failure mode where a normal PR strands three or more restatements of one claim now fails docs-check instead of waiting for a reader.

**Rejected.** Moving architecture content into docstrings: no pipeline renders docstrings to the site, so the move would hide the content, and the comment convention is constraint statements, not essays. Consolidating into this log: entries 26 and 27 were stale in lockstep with the docs they ratified, so the archive is a record, not a maintenance surface. Keeping the directory with a staleness lint: the lint would have flagged most of the directory, which is the same verdict as deletion with extra machinery.
