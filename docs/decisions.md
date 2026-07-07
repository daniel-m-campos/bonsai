# Decisions

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

Rejected: `std::variant<vector<uint8_t>, vector<uint16_t>>` per feature to save ~50% on binned column memory. Saves ~45MB on YearPredictionMSD, ~308MB on Higgs — neither pressure-tests modern hardware. Cost was variant dispatch complexity at every column scan via a `visit_column` wrapper. Rejected for MVP; reversible if a future dataset makes memory the bottleneck.

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
- Per-thread local histograms (no atomic FP adds — those are bit-unstable even at fixed thread count).
- Deterministic chunking (e.g., OpenMP `schedule(static)`).
- `random_seed` carries through samplers / shufflers.

What this rules out (relative to earlier framing):
- Promising cross-thread bit-exactness. The earlier draft demanded fixed-order merge (`tid` outer, bin inner) so the per-thread reduction shape didn't depend on thread count. Dropped — costs design constraints on `ParallelBackend` (must expose ordered reduction primitive) and forecloses `OpenMP reduction(+:...)` and `std::execution` reduce shapes.

Field check: XGBoost and CatBoost don't promise cross-thread determinism. LightGBM offers it behind `deterministic=true` +
`force_col_wise|row_wise`, and its own maintainers describe the
guarantee as fragile (RFC #6731). The pragmatic, industry-standard contract is "thread count is part of the reproducibility input."

Test contract:
- `test_determinism_fixed_threads`: two runs at `n_threads=k` for `k ∈ {1, 4, 8}` produce identical model files. Required to pass.
- `test_determinism_cross_threads`: predictions across different thread counts agree to numerical tolerance (e.g., max abs diff
  < 1e-5 on YearPredictionMSD). Required to pass.

Knock-on:
- `ParallelBackend` does not need to expose "ordered reduction" as a primitive. `parallel_for` + thread-local accumulators is enough.
- The histogram's parallel-build description in [`architecture/2-histogram.md`](architecture/2-histogram.md) §"Parallel construction" reflects this — no fixed-tid-order requirement.
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

Two `predict` overloads under one name (single-row returns float; batch fills `out`). Disambiguation by arity. Row-major batch input — matches xgb / lgbm. CatBoost's column-major fast path is motivated by predict- time rebinarization, which we don't do (decision 3 — float thresholds). Oblivious can transpose-on-demand internally if profiling justifies.

Rejected: `leaf_index(row)` (DenseTree and ObliviousTree leaf-index spaces aren't unified — defer to Phase 4 if SHAP / leaf-output predict is wanted); `walk(visitor)` (no shared node shape between the two impls); serialization on the concept (lives in `bonsai::io` per decision 6).

---

## 10. Shrinkage is baked into leaf values at tree construction

Grower receives `learning_rate` as a constructor argument (in `TreeConfig`) and writes `lr · -G/(H + λ_l2)` into leaves. Trees are pure functions of input rows; predict has no learning-rate knowledge. Matches xgboost, LightGBM.

Rejected: per-iteration `learning_rate` argument to `grow()` (Phase 1 doesn't need decay schedules; non-breaking to add later as a `grow` overload).

Knock-on: `Tree` concept doesn't carry a `set_shrinkage` mutator; trees are immutable post-construction.

---

## 11. `ObliviousTree`: per-level `default_left`, not per-node

Every node at level `d` of an oblivious tree shares the same `(feature_id, threshold, default_left)` triple — that's the symmetric- tree contract. Relaxing `default_left` to per-node-at-level recovers a small accuracy edge in heavily-missing data but breaks the branchless predict kernel and isn't what CatBoost does.

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

Knock-on: predict-time sentinels (e.g. `-999`) declared in `BinMapperConfig` are *not* honored at predict — only `std::isnan`. Caller contract: convert sentinels to NaN before predict. Matches xgb / lgbm.

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

Knock-on: same convention applies to the smaller-sibling choice in the subtraction-trick wiring — when `n_left == n_right`, left wins.

---

## 17. Partitioning: per-node row-index lists (strategy A)

Each live `FrontierNode` carries its own `std::vector<uint32_t>` of row indices. At root: one list with the booster-supplied `row_indices`. At each split: partition the parent's list into `(left_rows, right_rows)`, replace parent in frontier with two children carrying the new lists.

Both growers use this strategy (oblivious folds per-node histograms into level histograms at scoring time; ~1.5% overhead on YearPredictionMSD-scale data).

Rejected: single `row_to_node` array of length `n_rows` rebinned on each split (xgb's `hist` updater shape; lgbm's voting parallel mode). Beats per-node lists on cache locality at very shallow depth, loses at typical `max_depth = 6`. More importantly: the subtraction trick wires naturally onto per-node histograms. A single rebinned position vector either fights subtraction (the all-live-nodes-in-one-pass kernel doesn't know to skip the larger sibling) or imports per-node branching back into the kernel.

Knock-on: oblivious grower needs **per-parent gain summation** across the frontier — for each candidate `(feature, bin)`, sum `score(left, λ) + score(right, λ) − score(parent, λ)` over every parent (CatBoost's symmetric-tree gain; see [decision 30](#30-obliviousgrower-fold-then-score-was-wrong-revert-and-re-spec)).
Same `O(n_features · n_bins · |frontier|)` order as a fold; the
difference is in *what* is accumulated (gain, not histogram cells). An earlier version of this knock-on said "needs a fold step (`level_hists = sum_per_feature(...)`) before split scoring" — that was wrong because `score(g, h)` is non-additive; corrected 2026-05-22.

---

## 18. Frontier holds histograms inline (no histogram pool)

`FrontierNode { rows, sum_grad, sum_hess, hists }`. `vector<Histogram>` per node, sized `n_features`. The grower carries `std::vector<FrontierNode> frontier` and rotates it level-by-level.

Lists and histograms are local to `grow()`; the grower is stateless across calls (one boosting iteration → one allocation cycle). Matches lgbm's `SerialTreeLearner`.

Rejected: histogram pool with slot-id indirection (xgb's `hist` updater). Recycles allocations across the tree, but Phase 1 isn't allocation-bound. The pool refactor is contained if profiling later shows the allocator dominating.

---

## 19. Subtraction trick from day one, both growers

Build smaller child by row-scan; derive larger by `larger_hist = parent_hist - smaller_hist`. Halves histogram-build work across the tree (per [`2-histogram.md`](architecture/2-histogram.md) §"Why subtraction halves it"). Implemented from day one because retrofitting means restructuring the grower's per-node memory.

Per-parent protocol:
1. Splitter scores `parent.hists`; commit candidate.
2. Partition `parent.rows` into `(left_rows, right_rows)`.
3. Pick smaller (left wins ties).
4. Build smaller's hists by row-scan.
5. Derive larger's hists by `parent_hist - smaller_hist`. `Histogram` carries its own `(total_grad, total_hess)` so `operator-=` subtracts cells and totals together.
6. Push left, right into new frontier in left-then-right order (frontier order is structural, independent of build order).
7. Parent's hists released when parent goes out of scope.

For oblivious: same per-parent protocol, run once per parent at each level. Cross-level subtraction (deriving level `d+1`'s level histogram from level `d`'s) doesn't help — different levels score on different features, so per-feature histograms for level `d+1` have to be built regardless.

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
- **Custom user objectives.** No registry needed — anyone satisfying the concept can drop in as a `Booster` template parameter.

## 24. `compute` writes raw-score grad/hess; output is overwritten, not accumulated

`Objective::compute` writes `grad` and `hess` outright; callers don't zero buffers first. `preds` are raw scores throughout (logloss does **not** apply sigmoid inside `compute` or `eval`); the booster keeps an additive raw-score accumulator across iterations and applies the link only at the outermost predict call. Matches xgboost / LightGBM; keeps boosting math additive.

`hess` is always non-negative (MSE: 1; logloss: `p·(1−p) ∈ (0, 0.25]`); the splitter's `min_child_hess` (decision 20) catches near-zero hessians before they propagate.

## 25. Sample weights are applied by the booster, not the objective

When `Dataset.weights` is non-empty, the booster multiplies the `grad` and `hess` buffers by the weight vector immediately after `Objective::compute` returns and before handing them to the grower. Keeps every `Objective` impl focused on loss math; the multiplication is a 2-line buffer loop that doesn't belong duplicated across objectives.

Rejected: a `WeightedObjective<T>` wrapper that satisfies `Objective` by composing with weights. Adds a template layer with no semantic content; same effect as the booster-side multiply.

## 26. Dispatch: flat table over cartesian product, `IBooster` at boundary

Runtime → static boundary uses Candidate A from `architecture/6-dispatch.md`: a `constexpr std::array` keyed on a name-tuple, generated by `for_each_type` over `cartesian_product_t<Objectives, Growers, Splitters, Samplers>`. Each cell is a monomorphized `Booster<O,G,S,Sa>` factory. Lookup at the config boundary returns `unique_ptr<IBooster>`.

Cost: **one virtual call per `update_one_iter`**. Acceptable. The hot path is histogram building inside `update_one_iter`, not the call itself; per-iteration vcall is dwarfed by the per-iteration histogram pass. Static-everywhere is preserved inside the iteration body, which is what the proposal §3.4 rule actually targets.

Rejected:

- **Nested registry callbacks (Candidate B).** Fully static, zero vcalls anywhere, but forces continuation-passing at the boundary and drags the whole training run into the innermost lambda.
- **Dressed-up nested lambdas (Candidate C).** No advantage over A or B once the type-level builder is in play.
- **Hybrid flat-table + generic callback.** Doesn't compile — `std::array` of function pointers can't be generic over the callback type.

Invalid combinations (none in MVP) are pre-filtered at typelist construction by building sub-products over compatible sub-typelists and concatenating. No runtime check, no concept predicate, no instantiation of bad cells.

`Backend` placement deferred to `7-parallel.md` — dispatch stays 4D for now; promotion to 5D or separate composition both stay open.

## 27. Booster shape, training loop, and Sampler concept

Ratifies `architecture/5-booster.md`.

**Class shape.** `IBooster` is the boundary erasure type with a minimal CLI-facing virtual surface (`update_one_iter`, `eval`, `predict`, `n_iters`, accessors for save/load). `Booster<Obj, Gr, Sp, Sa>` is the real class, monomorphized per cell of the dispatch table (decision 26). One vcall per `update_one_iter` from the CLI; zero inside.

**Training loop (`update_one_iter`).** Six steps in order:

1. `Obj::compute(scores_, labels, grad_, hess_)` — full row set.
2. Apply `Dataset.weights` to `grad_`, `hess_` if present (decision 25).
3. `Sa::sample(grad_, hess_, rng, out_indices)` → row indices for the grower.
4. `grower.grow(dataset, grad_, hess_, row_indices, split_finder, cfg)` → tree.
5. `scores_ += learning_rate * tree.predict(train_rows)` — re-walks the tree per training row.
6. `trees_.push_back(std::move(tree))`.

Sampling runs *after* grad/hess (not before) because GOSS samples on
`|grad|`, the grad pass is small (~5-10%) compared to histogram
building (60-80%, what sampling actually targets), and reordering would force a "does this sampler need grad?" flag on the concept.

**`learning_rate`.** Applied at score-update time (step 5), not pre-scaled into leaf values. Saved trees carry raw leaf values; the booster reapplies the rate at predict and at score-update. Matches xgb's "shrinkage = booster concern" model.

**Score update via re-predict, not cached leaf values.** Step 5 calls `tree.predict(train_rows)` rather than reading a precomputed `(row → leaf_value)` array from the grower. Cost is one extra `O(n_rows × depth)` walk per iter, small at MVP scale. xgb / lgbm / catboost all cache the row→leaf mapping as a byproduct of growing and reuse it for the score update; bonsai defers that optimization to Phase 2 (after benchmarking). When justified, change grower return from `Tree` to `(Tree, std::vector<float> train_leaf_values)` and replace step 5 with a flat add. Pure additive change to the grower→booster boundary; `Tree` stays clean.

**Initial score (bias).** Booster owns it (decision 22). Three sources in priority: `cfg.init_score`; objective-appropriate default from labels (mean for MSE, `log(p/(1-p))` for logloss); value loaded from disk for a continued booster.

**Predict path.** Raw-score predict sums tree predictions plus `init_score`, applies `learning_rate` per tree. User-facing predict applies the objective's link inverse (identity for MSE, sigmoid for logloss) via `if constexpr` on `Obj`. Inverse link lives in the booster (decision 22), not in `Objective::compute` (decision 24).

**`Obj` is purely-static.** No instance member; `T::compute` and `T::eval` are static (`4-objective.md` / decision 21).

**Booster borrows `Dataset`.** Lifetime sits with the CLI; `update_one_iter` takes `Dataset const&`. Saved model is `BinMappers` + trees + init_score, no `Dataset`. Booster does not own `BinMappers` (decision 3 — predict path doesn't need them).

**Save / load are I/O, not booster methods.** Free functions `save_booster(IBooster const&, path)` and `load_booster(path) -> unique_ptr<IBooster>`. Rationale: `load` needs the four component types *before* it has a `Booster` to dispatch on, so it reads names from disk and calls into the same registry path `make_booster` uses — that's structurally the dispatch boundary, not a member function. `save` is the symmetric counterpart.

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

**On-disk format (2026-05-20).** `save_booster(IBooster const&, path, BinMappers const&, Config const&)`. Full Config rides along in the msgpack envelope under `"config"`. `load_booster` reads it straight into `LoadedBooster::cfg` and calls `make_booster(out.cfg)` — no synthesized-Config indirection (was `{dispatch, learning_rate}` torn apart at save and stitched back at load). Format version bumped 1 → 2; no v1 artifacts in the repo, fail-loud on stale files is correct.

**Mechanism.** `NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE` per Config sub-struct (Data, BinMapper, Tree, Booster, Dispatch, Metrics) plus one for `Config` itself. Same macros are used for the tree-node POD records (`DenseTree::InternalNode` / `LeafNode` / `Params`), replacing ~65 lines of hand-typed JSON-key mirror code. One-shot `adl_serializer<std::optional<T>>` added in the same TU — nlohmann v3.11 doesn't ship optional support. Every macro lives in `src/io/model.cpp` (no public-header nlohmann leakage).

**Why not the existing TOML codec.** The TOML path uses `Section` descriptors + `field_name<MemPtr>()` (`source_location`-based extraction) to drive serialization with zero per-struct boilerplate. The JSON path could reuse those Section tuples the same way — that was the originally-proposed approach. Rejected in favor of the macros: less code (~10 macro lines vs ~30 lines of generic fold + two new files in `bonsai::config`), one fewer abstraction layer to read when debugging the on-disk format, and the macro form lives next to the only consumer (`model.cpp`). Trade-off explicitly accepted: the macro field-list duplicates the existing Section field-list. A member added to a struct but not to one of the two enumerations silently drops from that serializer.

**Why not modern C++ reflection.** P2996 (static reflection, C++26) collapses every NLOHMANN macro in this file to one template per direction. Not available in our toolchain. When it lands, this decision is the natural retirement point — delete the macros, ship the reflection-based serializer, format version stays at 2. Boost.PFR is positional-only (no field names) and brings a dependency for no JSON-key benefit. Rejected.

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

The bug surfaced during depth=2 testing as a "fold equals root histogram" property: because rows are partitioned (not removed) across the frontier, the fold at level `k+1` reconstructs the root histogram exactly, so the splitter re-picked the same `(feature, bin)` at every level and produced degenerate trees where `2^depth − 2` leaves were empty. I documented this as inherent to oblivious + basic gain. It isn't — it's an artifact of the wrong gain function.

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
- [`tests/unit/test_grower_helpers.hpp`](../tests/unit/test_grower_helpers.hpp) shared fixtures — grower-agnostic, still used by `test_grower.cpp`.

### Outstanding

Decision 8's Phase-1 commitment (depth-wise + oblivious) is not honored. The design-review drift flag at [`reviews/2026-05-19-design-review.md`](reviews/2026-05-19-design-review.md) §"DenseTree / ObliviousTree" remains accurate. The correct algorithm is documented as a design target in [`architecture/3-tree.md` §"Oblivious grow loop"](architecture/3-tree.md); implementation pending in Phase 2.5 (user-authored).

### Lesson

Mathematical primitives need a sanity check before being trusted across an aggregation boundary. `score(g, h)` looks superficially linear in `g`, but the `g²` term kills additivity. Two more checkpoints that should have caught the bug earlier:

1. **Cross-reference against the reference library before implementing.** CatBoost's source explicitly aggregates gains per-leaf. A 10-minute read of `greedy_tensor_search.cpp` would have surfaced the right shape before any code was written.
2. **Design tests that distinguish the right answer from a plausibly-wrong one.** The original `depth=2` test was content with "4 leaves, structurally correct" and even rationalized the degenerate `[2 non-empty, 2 empty]` outcome. A test that demanded "all 4 leaves carry rows" or "level-1 split differs from level-0 on some non-trivial fixture" would have failed loudly.

## 31. `LeafwiseGrower`: best-first growth on a gain-keyed heap, `max_leaves` primary

Third grower, `dispatch.grower_name = "leafwise"` (LightGBM's default strategy). A `std::vector<Candidate>` maintained with `std::push_heap`/`std::pop_heap` holds every expandable leaf — `{SplitInput, SplitOutput, depth}` — keyed on split gain; each pop converts one leaf into two children, so `live_leaves` counts up and growth stops at `TreeConfig::max_leaves` (new field; `0` = unbounded, `max_depth` stays as the cap). Reuses `make_root` / `split_node` / `finalize_as_leaf` / `HistogramNodeSplitFinder` unchanged and emits a `DenseTree`, so registration is just the typelist + `impl_name` edits.

- **`std::vector` + heap algorithms over `std::priority_queue`**: `top()` returns `const&`, which fights moving the histogram-heavy `SplitInput` out; `pop_heap` + `std::move(heap.back())` doesn't.
- **Tie-break**: equal gains resolve to the lower node id (FIFO-ish), so trees are deterministic.
- **Semantics**: with `max_leaves = 2^max_depth` and a separable dataset, leafwise reproduces depthwise's tree exactly (covered by unit test).

## 32. Parallelism: OpenMP behind a one-function seam, determinism at any thread count

`bonsai/parallel.hpp` exposes `parallel::for_each_index(n, f)` — an OpenMP `parallel for` (dynamic schedule, chunk `n/(threads*4)`) with a serial fallback when OpenMP is absent — plus `set_n_threads` fed from a new `[parallel] n_threads` config section (0 = all cores). Every parallel site assigns each index to exactly one thread and performs **no cross-thread reductions**: per-feature histogram fill, per-feature split scans (per-feature bests merged serially in feature order, preserving the tie-break), row-wise predict, objective grad/hess, score updates, CSV row parsing, binning, mapper fitting.

Consequence: models and predictions are **bit-identical to a serial run at any thread count** — stronger than the proposal's fixed-thread-count contract (decision 7), because the row-parallel-within-feature + per-thread-histogram-merge design that motivates the weaker contract hasn't been needed yet. If it ever is (single-feature datasets), the contract degrades to fixed-N as originally specified.

Rejected for now: the `ParallelBackend` concept as a 5th dispatch dimension (proposal §3.4, `7-parallel.md` TBD). One free function covers every call site today; promoting it to a dispatched component adds a typelist dimension with a single implementation. The seam keeps the door open — `std::execution`/TBB would slot in behind the same signature.

## 33. Hot-path perf: ordered gradients, stable scatter, node totals out of `add`

Three measured wins on Year Prediction MSD (M2, 8 threads), all bit-identical outputs:

- **Ordered gradients** (LightGBM trick): `populate_from_rows` gathers grad/hess into node-row order once, so each of the 90 per-feature scans reads them sequentially instead of re-walking two full arrays with scattered indices.
- **Stable split scatter**: `split_node` replaces `std::partition` + two `assign`s with a two-pass exact-size stable scatter. Stability keeps every node's rows ascending (root's are iota), so bin lookups walk memory near-sequentially at every depth.
- **Node totals once**: `Histogram::add` no longer maintains running totals (2 redundant double-adds per row×feature, duplicated per feature); node totals are one O(n_bins) cell sum over `hists[0]`, hoisted per node in the split finders.

CSV load: whole-file read + line index + row-parallel `from_chars` parse straight into column-major storage, and the train file is parsed once (mapper fit + binning share the batch) instead of twice. Load 7.4s → 1.3s; 200-iter depthwise fit 73s → ~27s; leafwise ~12s.

## 34. Feature-parity round: colsample, GOSS, early stopping, L1 — and the OOB score bug

Four reference-library features landed in one pass, each benchmarked A/B on
Year Prediction MSD with the equivalent knob enabled in xgboost / lightgbm /
catboost (protocol + tables in [feature_gap.md](feature_gap.md)):

- **`tree.feature_fraction`** — per-tree feature subsample drawn from a
  grower-owned rng (`tree.feature_seed`), histograms built for selected
  features only; unselected slots are zero-binned placeholders the finders
  skip. Node totals moved to `SplitInput::totals()` (first populated hist).
- **`sampler_name = "goss"`** — LightGBM's gradient one-side sampling; the
  `Sampler` concept now takes mutable grad/hess so the sampler can amplify
  the small-gradient sample in place.
- **`booster.early_stopping_rounds`** — incremental valid eval
  (`IBooster::score_base` / `accumulate_last_tree`, one single-tree predict
  per iteration) + `truncate` to the best iteration.
- **`tree.lambda_l1`** — XGBoost-style soft threshold on the gradient sum in
  both the gain score and the leaf value.

**The GOSS benchmark exposed a latent correctness bug** in every subsampled
path: `GrowResult.values` was only stamped for sampled rows, so out-of-bag
rows' entries stayed 0 and the booster's score accumulator silently
diverged from the real model for those rows — their gradients were computed
against predictions missing whole trees. Bernoulli had been quietly paying
~2% RMSE for this (9.1873 → 8.9916 on MSD after the fix); GOSS diverged
outright (RMSE 24.7 — worse than predicting the mean) because it re-selects
by |grad| every iteration and fed on its own staleness. Fix: growers now
route unsampled rows through the finished tree *in bin space*
(`route_unsampled`, split bins recorded during growth), which is exact with
respect to the float-threshold predict path — `bin(v) <= b  ⟺  v <=
cuts[b]` under the right-inclusive binner, missing bin routed by
`default_left` on both paths.

Lesson (rhymes with decision 30): the booster-side `values` shortcut was
only ever validated with `all_rows`. A contract as easy to state as "every
row's train value equals the tree's prediction for that row" deserved a
test the day the first subsampler landed.

## 35. Remaining-gap round: objectives, monotone, interaction, DART

Second feature-parity pass (protocol and A/B tables in
[feature_gap.md](feature_gap.md)); sparse/EFB explicitly stays out of scope
until the harness has a sparse dataset to measure against.

- **Objectives became Config-constructed instances** (like `Sampler`) so
  parameterized losses carry state — `[objective] huber_delta /
  quantile_alpha`. Statics satisfy instance-call syntax, so MSE/LogLoss kept
  their static methods and only gained trivial ctors. MAE / Huber / Quantile
  land with sign/clamped/pinball gradients and median/quantile init scores.
  Known limitation: no leaf-renewal pass, worth ~10% MAE vs
  lightgbm/xgboost's renewed leaves on YearMSD; bonsai matches lightgbm on
  huber, where renewal matters less.
- **Monotone constraints**: candidate splits on a constrained feature are
  rejected when bounded child weights violate the direction, and children
  inherit midpoint-fenced leaf bounds via `SplitInput::lo/hi` — whole paths
  are provably monotone. Costs every library the same ~2% RMSE on CH.
- **Interaction constraints**: `SplitInput::allowed/path` carries the
  permitted feature set down the tree; a feature may split only where some
  group covers the whole path (or alone). Same ~9% RMSE cost as xgb/lgbm on
  a two-group CH config.
- **DART**: dropout of existing trees with bin-space routing to recover
  dropped train contributions (no per-tree caches), rescaling with
  xgboost's `normalize_type="tree"` factors — the DART paper's 1/(k+1)
  starves the new tree by ~1/lr under shrinkage and measurably tanked RMSE
  until replaced. bonsai's DART now degrades less than xgb/lgbm's at the
  same settings. Incompatible with early stopping by construction (throws).
- Oblivious grower rejects monotone/interaction constraints at
  construction rather than silently ignoring them.

## 36. Python bindings: nanobind over the CLI's own seams, static libomp

`_bonsai` (nanobind, `python/bonsai` package, `pip install .` via
scikit-build-core) wraps exactly the seams the CLI uses — `config::
apply_overrides` for params (dotted keys, same codec as `--set`),
`cli::train_with_progress` for fit (so early stopping and valid sets come
for free), `io::save_booster`/`load_booster` for model files interchangeable
with the CLI. No training or prediction logic lives in the binding. The
sklearn-ish `BonsaiRegressor` accepts first-class knobs plus a `params`
dict of dotted config keys. Parity test: same config through the module and
the CLI agrees to atol 2e-4 on California Housing predictions.

**The libomp lesson.** Linking the extension against Homebrew's
`libomp.dylib` deadlocked the process the moment xgboost built a DMatrix —
a `sample` trace showed one OpenMP call stack spanning *two different
libomp images* (ours and xgboost's bundled copy): classic duplicate-runtime
interposition. Fix, standard for wheels: `BONSAI_OPENMP_STATIC=ON` links
`libomp.a` into the module and `-Wl,-exported_symbol,_PyInit__bonsai`
strips every other export (1015 leaked `kmp` symbols before). Verified by
interleaving bonsai / xgboost / lightgbm training in one process.

**Native benchmark rows.** compare.py adds in-process "(native)" rows when
`build/python` is importable, timed like the reference libraries (no
subprocess, CSV, or model-save overhead — though bonsai's `train()` still
includes binning, which xgb's timed `train` does not). Re-baselined:
RMSE identical to the CLI rows; predict drops 0.20s -> 0.08s on
YearPredictionMSD, landing between xgboost (0.017) and lightgbm (0.105).

## 37. Feature importance recorded at grow time; the guide series

**Importance.** Split-count and gain importance ship behind
`IBooster::feature_importance(ImportanceType)`, `bonsai importance`, and the
Python `feature_importances_` / `importance(type)` surfaces. The one design
decision: gain is **stamped when the split is created** (`split_gains` per
node on `DenseTree`, `level_gains` on `ObliviousTree`, serialized — format
v5) because it is not reconstructible from a stored tree. Accumulation is a
20-line walk in `booster.hpp`. Verified by cross-library agreement on
California Housing: bonsai and lightgbm agree on both types *including
their disagreement with each other* — gain crowns MedInc, split-count
crowns Longitude/Latitude (many fine-grained, individually-small splits),
the textbook argument for gain as the default and a finding the agreement
test now pins.

**The guide.** `docs/guide/` is a nine-chapter pedagogical series — concept
→ math → the actual implementing code → runnable experiment → war story —
positioned as a deliberate differentiator: reference libraries document
parameters, the guide documents mechanics against a codebase small enough
to read. War stories are real ones from this log (OOB stale scores §34,
DART's k+1 trap §35, the two-libomp deadlock §36, the split-vs-gain
disagreement above). Stale narrative docs were refreshed in the same pass
(context, report addendum, architecture 2–8, new 7-parallel.md), and
milestones are now git-tagged (`mpcs-submission`, `v0.2.0`–`v0.5.0`).

## 38. Completing the non-categorical gap: rows 10-17 in one push

Every remaining non-categorical row of [feature_gap.md](feature_gap.md)
landed (tables and datasets per row in that doc):

- **Leaf renewal** (10): `GrowResult::leaf_ids` + an objective `renew_leaf`
  hook; the booster regroups rows by leaf and replaces Newton steps with
  loss-optimal values (residual median / alpha-quantile / clamped-mean
  huber). Closed the recorded ~10% MAE gap outright — bonsai now ties
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
  routes to a dedicated `MulticlassBooster` — the one objective whose
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
RESOURCE at 7.5k distinct values — the regime one-hot cannot reach and
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
lightgbm-native does not come from target encoding per se — plain K-fold
target encoding was no better than raw IDs — but from the ordered
scheme plus feature combinations. (c) bonsai beats lightgbm when both
are denied categorical machinery, so the gap is the feature, not the
engine. Stage 2 (set splits) proceeds with a measured ceiling instead
of a hope.

---

## 40. GPU-resident growing: widen the builder seam to an optional level backend

**Decision.** Phase 3 (device-resident partitioning and split finding, design in [architecture/11-gpu-resident.md](architecture/11-gpu-resident.md)) extends the builder policy with three *optional* hooks — `find_splits_many`, `partition_many`, `finalize_rows` — detected via `if constexpr` + `requires`, the same idiom as phase 2's `populate_many`. The host grow loop remains the single algorithm narrative and the decision-maker (leaf-vs-split, smaller-child pairing, constraint propagation); the device executes the data plane. `SplitInput` gains `sums` and `row_count` so it degrades to node metadata when histograms and rows stay device-resident.

**Rejected.** A device splitter as a fourth typelist dimension: there is one device implementation and it is coupled to the CUDA builder's state, so a registry axis buys combinations nobody can instantiate (the same restraint as decision 32 for threading). A `row_to_node` map (xgboost's shape): decision 17's reasons hold on the device too — per-node segments pair naturally with the subtraction trick and stable per-node row order. CUB/Thrust for the partition scan: a hand-rolled three-kernel scan keeps the backend a single self-contained TU with no new dependency.

**Consequences.** In resident mode `SplitInput.hists`/`rows` are empty; the inspect path for tests and debugging is an explicit `download_histograms`. The `cuda_depthwise` determinism contract is formally tolerance-equal (prediction/RMSE tolerance + split-agreement rate in parity tests, never tree equality); CPU-only builds stay bit-identical. Copy-back mode is retained as the degrade path (deep trees, oversized `max_bin`) and for the oblivious/leafwise growers.

---

## 41. Grower data-plane as the `LevelStep` strategy; retire the copy-back ladder

**Decision.** Reframe decision 40's "escalating optional hooks" as a single compile-time Strategy: a `LevelStep<SplitterT, Engine>` (primary template = host data plane; partial specialization for `GPULevelEngine` = device data plane) selected by engine type, so `grow()` reads as one control-plane narrative with the host/device fork localized to one specialization instead of smeared across six `if constexpr`/`if (resident())` sites. `resident()` is removed — the per-tree mode is captured once from `begin_root`'s bool into the `LevelStep`. The concept ladder collapses to two, renamed to shed the GoF-`Builder` connotation (the type is the pluggable compute substrate, and the CUDA one supplies the whole device data plane): `HistogramEngine` (host, from `HistogramBuilder`) and `GPULevelEngine` (the device-resident vocabulary, from `ResidentHistogramBuilder`). The `populate` → `populate_many` → resident progression was a dev-research ladder (phases 1–3, preserved in git and doc 11's measured stages); only device-resident is kept — `populate_many`, `BatchHistogramBuilder`, and the GPU **copy-back** histogram path are retired, and the rare decline (oversized `max_bin`, or level buffers that won't fit) falls back to **CPU** histogram building via the engine's existing `cpu` member. Design in [architecture/12-grower-backend.md](architecture/12-grower-backend.md). Refines, not overturns, decision 40: host-owns-decisions, one-grow-loop, no-new-typelist-dimension, and the all-or-nothing device coupling all carry forward.

**Rejected.** Runtime strategy objects (a `unique_ptr<LevelPlane>` or `std::variant` handed to the loop): the dynamic-dispatch shape decisions 14/26/32 explicitly chose against — it would need a benchmark to *prove* zero-cost rather than guarantee it, add type erasure the grow loop avoids, and thread into every grower. Keeping the three-tier concept ladder: `populate_many` is a host-plane batch optimization, not a device peer, so the middle tier documented a distinction that no longer earns its name. Keeping GPU copy-back as the decline fallback: it keeps a whole kernel path alive for a rare case (default `max_bin=255` always goes resident); the CPU fallback trades a slice of that rare path's throughput for a materially smaller CUDA backend.

**Consequences.** `update_nodes` (182 lines, 18 params) decomposes into `plan_level` + `LevelStep` methods + `commit_children`, each ≤~40 lines; the CPU path becomes branch-free (a reader never meets a GPU concept). Oblivious and Leafwise route through the shared `LevelStep`, so their CUDA variants become a one-line alias + registry entry (not registered here — a later pass; leafwise's sequential gain-heap limits its device value). Docs 10/11's grower-side *seam* narrative is superseded by doc 12; their device-kernel content stands. Zero perf cost is a release gate, measured on a Thunder 4×A100 against xgboost (`scripts/bench_gpu.py`, MSD, before/after) in addition to CPU bit-identity and cuda tolerance-equal parity. Landed (commits b4d223c → a4764ba): every phase held 392/392 both configs, sha256-identical CPU models for all three growers, and unchanged resident-path launch counts; the leafwise open call resolved *against* singleton-frontier unification (a per-heap-pop LevelPlan buys ceremony, not shared code — `split_node` lives in level_step.hpp as the single-node data plane) and the oblivious unsampled routing stayed its own (leaf-table indices, not DenseTree node ids).

---

## 42. GPU oblivious via a device level-find; no `cuda_leafwise`

**Decision.** Register `cuda_oblivious` (`ObliviousGrower<CudaHistogramEngine>`), completing the pass decision 41 deferred. The resident plane is reused wholesale — an oblivious level is a depthwise level where every node takes the same split — so the only new device piece is the level-find: `find_level_split` sums each candidate cut's child scores across the whole frontier (32-node chunks into per-feature global scratch, any frontier width) and requires per-node `min_child_hess` feasibility, mirroring `update_best_for_feature_for_level`; a fused find → reduce → child-sums launch chain pays one sync per level. `GPULevelEngine` gains the method; `LevelStep` routes `LevelSplitFinder` growers to it. Measured (RTX 5090, fair full-pipeline timing): 3.9–4.1 s vs CatBoost-GPU 7.3–8.2 s on MSD, RMSE matching CPU oblivious exactly.

**Rejected: registering `cuda_leafwise`.** A working `LeafwiseGrower<CudaHistogramEngine>` was built and benchmarked, then withdrawn: best-first growth expands one node at a time, which the level-batched resident plane cannot serve (advance swaps whole levels), so every histogram was computed by the engine's CPU member — a `cuda_` registry name executing CPU work misleads `bonsai info`, serializes into model metadata, and mislabels benchmarks. The comparison it existed for holds without it: CPU `leafwise` (11.1–11.5 s) beats LightGBM's CUDA leaf-wise backend (12.0–12.9 s) on the 5090. A true device leafwise needs a non-swapping advance — phase-4.

**Consequences.** `trains_here` keys on the `cuda` name prefix (registry convention) instead of one hardcoded name; `skip_without_cuda` keys on the grower's `Engine` alias; the oblivious grow loop consumes per-node child sums from the level-find (children's `SplitInput.sums` are otherwise unknowable — device histograms are not host-scannable), and host/device leaf finalize is symmetric via `LevelStep::finalize_leaves`.

## 43. CI: GitHub Actions gates every PR; sanitizers as a build option, not a preset

**Decision.** One workflow (`.github/workflows/ci.yml`, ubuntu-24.04, clang-21 from apt.llvm.org so CI uses the Makefile's exact Linux toolchain): `build-test` (Release + ctest), `sanitize` (`BONSAI_SANITIZE=ON`, ASan+UBSan, RelWithDebInfo), `format` (`clang-format --dry-run --Werror`), `tidy` (`make lint` off a configure-only tree — clang-tidy needs just `compile_commands.json`), `python` (nanobind module + bindings tests), and `cuda-compile`. Sanitizers are a CMake option (`BONSAI_SANITIZE`) applied globally so FetchContent'd Catch2 carries the same instrumentation; a preset file would be a second configuration surface for a Makefile-driven repo. This discharges the proposal's sanitizer-in-CI commitment.

**CUDA is compile-only.** GitHub has no GPU runners, but the kernel TU is the code path a CPU-only CI would otherwise never touch. The job installs only `cuda-nvcc-12-6` + `cuda-cudart-dev-12-6` (headers, ptxas, libdevice — no driver) and builds with `-DBONSAI_CUDA_ARCH=sm_80`, since `native` requires a device to probe. Runtime GPU validation stays where it has always been: `make test-cuda` on a GPU host.

**Consequences.** LSan is off in the sanitize job (`detect_leaks=0`): libomp's pool allocations outlive `main` and drown real leaks; ASan+UBSan remain fatal (`-fno-sanitize-recover=all`). `BONSAI_SANITIZE` + `BONSAI_CUDA` is a configure error — sanitizing device code is not supported and a half-instrumented binary would imply coverage that isn't there. FetchContent `_deps` and the toy CSVs are cached per lockfile-ish keys; a cold run needs the network, warm runs don't.

**The first honest lint run.** `make lint` had been discarding run-clang-tidy's stderr, so tool failures on macOS printed "no findings" — the Linux CI job surfaced 128 findings the local gate never saw. Response: fix the real bugs (a use-after-move in the `for_each_type` fold, two unchecked optional accesses, an exception escaping `main`, plus small `google-*`/naming/enum-size items) and curate `.clang-tidy` down to checks the codebase actually satisfies — the disabled block documents each family (pointer arithmetic at C/CUDA/codec boundaries, reference members, callback `F&&` APIs, concept archetypes, established long functions). The Makefile now fails lint when the tool itself fails. Re-enabling a curated-out check is a deliberate cleanup PR, not a silent flip.
