# Decisions

Append-only log. Order = decision order. Caveman style. New entries at bottom.

---

## 1. Binning: quantile, with low-cardinality fallback

`BinMapper::fit` per feature.

- Equal-frequency cuts at `k/max_bin`-th quantiles. `max_bin = 255` default
  (`uint8` indices).
- If `n_distinct < max_bin`: one cut between each pair of consecutive
  distinct values. Bucket count = `n_distinct`.
- Dedupe cut collisions (sentinel values like `0.0`). Actual count
  `<= max_bin`, never exact.
- Sampling from the start. Default sample 200K rows uniform random,
  fixed seed. Configurable. If column has `<= sample_size` rows, use full
  column.
- Bin 0 reserved for missing. NaN + user-configured sentinel short-circuit
  to bin 0. Real values bins `1..n_bins-1`. Quantile skips NaNs.
- `BinMapper` serializable. Round-trip through model file. Predict on new
  data reuses train boundaries exact.

Rejected: equal-width (skew kills it). Quantile sketch (overkill, swap
in later). xgb per-node default direction (complicates split scoring).

Knock-on: bin count varies per feature, histogram reads `n_bins[fid]`.
Bin 0 special, split scoring skips it for real-valued cuts. `BinMapper`
ownership vs `Dataset` is next decision.

Defer: `min_data_in_bin` knob.

---

## 2. `BinMapper` independent of `Dataset`. Two-stage API.

```
auto mappers = BinMappers::fit(train_source, cfg);
auto train   = Dataset::bin(train_source, mappers, cfg);
auto val     = Dataset::bin(val_source,   mappers, cfg);
auto test    = Dataset::bin(test_source,  mappers, cfg);
```

- `BinMappers` is `std::vector<BinMapper>` plus minimal wrapper (count,
  serialize). Built once on train, immutable thereafter.
- `Dataset::bin` is pure transform: takes source + mappers, returns
  binned column-major storage. No "training Dataset" vs "val Dataset"
  distinction.
- Model file serializes `BinMappers`. Predict-time `Dataset` builds
  fresh from them.

Rejected: lgbm-style `Dataset::from_csv(..., reference=train_ds)`. Couples
mapper lifetime to a Dataset, awkward serialization, "training Dataset"
becomes special.

Knock-on: train path is two calls instead of one. Trivial. `bin` is
single-pass; `fit` does its own sampling + sort internally.

---

## 3. Trees store raw float thresholds, not bin indices

Tree node split = `(feature_id, threshold: float)`. Predict reads raw
`float` from input row, compares directly. No binning at predict time.

`TreeGrower` finds the best split as `(fid, bin_idx)` during training, then
converts to `threshold = cuts[bin_idx]` when writing the node. Conversion
is one lookup per finalized split, free.

xgb + catboost do this. lgbm stores bin indices in tree nodes (and re-bins
at predict, which is why lgbm forces the reference-Dataset dance).

Knock-on:
- Predict path doesn't need `BinMappers`. Single tree walk over raw floats.
- Model file: trees serialize directly, `BinMappers` optional in model file
  (kept for diagnostics + reproducibility, not load-bearing for predict).
- Training-time histogram code unchanged: still bins, still works on bin
  indices internally.
- Float threshold means tree comparison is `<` on float, not `<=` on int.
  Watch for off-by-one when comparing parity vs lgbm (different convention).

---

## 4. `Dataset` storage layout

Column-major. Per-feature `std::vector<uint16_t>` (uniform width). Labels +
weights owned by `Dataset` (weights empty if uniform). `BinMappers` held by
value (not `shared_ptr`); ~30KB copy is trivial, no shared mutable state.

```cpp
class Dataset {
    std::vector<std::vector<uint16_t>> features_;
    std::vector<float>                  labels_, weights_;
    BinMappers                          mappers_;
    std::vector<bool>                   is_categorical_;  // Phase 4 placeholder
    // n_rows, n_features
};
```

Public API: `n_rows()`, `n_features()`, `labels()`, `weights()`,
`mappers()`, `n_bins(fid)`, `is_categorical(fid)`,
`feature_bins(fid) -> span<bin_id_t const>`.

Rejected: `std::variant<vector<uint8_t>, vector<uint16_t>>` per feature
to save ~50% on binned column memory. Saves ~45MB on YearPredictionMSD,
~308MB on Higgs — neither pressure-tests modern hardware. Cost was
variant dispatch complexity at every column scan via a `visit_column`
wrapper. Rejected for MVP; reversible if a future dataset makes memory
the bottleneck.

Group columns (ranking) deferred (non-goal).

---

## 5. *(reserved)*

Originally `visit_column` for variant-aware column access. Dropped when
decision 4 collapsed to uniform `uint16_t` storage. Renumbering decisions
breaks references; left as a placeholder.

---

## 6. Readers: free function per format, returning `Dataset`

```cpp
Dataset read_csv    (const std::string& path, const DataConfig&, const BinMappers&);
Dataset read_parquet(const std::string& path, const DataConfig&, const BinMappers&);  // Phase 4+
Dataset read_libsvm (const std::string& path, const DataConfig&, const BinMappers&);  // later

BinMappers fit_from_csv(const std::string& path, const Config&);
```

CLI dispatches on `cfg.data.format` string: `if "csv" call read_csv else if
"parquet" ...`. Each reader its own translation unit. Adding a new format =
new file + new branch in CLI dispatch.

No `Reader` concept, no abstract base, no template plumbing. File loading
is once-per-program, not hot-path; concepts buy nothing here. Internal
shared helper per reader (`CsvReader::columns(path, cfg) -> ColumnBatch`)
keeps `read_csv` and `fit_from_csv` from duplicating logic.

**Phase 1 ships CSV only**, hand-rolled (~50 LOC). Numeric-only is enough
for YearPredictionMSD. No Arrow, no parquet.

**Phase 4+: Arrow optional**, gated by `BONSAI_PARQUET=ON` CMake flag.
Arrow handles CSV multi-threaded + parquet + feather + IPC. Heavier dep
(transitive thrift/snappy), so kept optional. Arrow is the *reader* layer
only. `Dataset` (binned storage + `BinMappers` + labels/weights) is bonsai's;
Arrow's `Table` is raw column data we'd copy or borrow from.

Rejected: `Reader` concept + `Reader auto&` template params for `fit` and
`bin`. Over-engineered for once-per-program file loading. Free functions are
the simpler shape.

---

## 7. Determinism contract: fixed thread count, not cross-thread

Same seed + same data + **same thread count** → same model bytes.
Different thread counts: predictions within numerical tolerance, but
bytes may differ.

What this rules in:
- Per-thread local histograms (no atomic FP adds — those are bit-unstable
  even at fixed thread count).
- Deterministic chunking (e.g., OpenMP `schedule(static)`).
- `random_seed` carries through samplers / shufflers.

What this rules out (relative to earlier framing):
- Promising cross-thread bit-exactness. The earlier draft demanded
  fixed-order merge (`tid` outer, bin inner) so the per-thread reduction
  shape didn't depend on thread count. Dropped — costs design constraints
  on `ParallelBackend` (must expose ordered reduction primitive) and
  forecloses `OpenMP reduction(+:...)` and `std::execution` reduce
  shapes.

Field check: XGBoost and CatBoost don't promise cross-thread
determinism. LightGBM offers it behind `deterministic=true` +
`force_col_wise|row_wise`, and its own maintainers describe the
guarantee as fragile (RFC #6731). The pragmatic, industry-standard
contract is "thread count is part of the reproducibility input."

Test contract:
- `test_determinism_fixed_threads`: two runs at `n_threads=k` for
  `k ∈ {1, 4, 8}` produce identical model files. Required to pass.
- `test_determinism_cross_threads`: predictions across different
  thread counts agree to numerical tolerance (e.g., max abs diff
  < 1e-5 on YearPredictionMSD). Required to pass.

Knock-on:
- `ParallelBackend` does not need to expose "ordered reduction" as
  a primitive. `parallel_for` + thread-local accumulators is enough.
- The histogram's parallel-build description in
  [`architecture/2-histogram.md`](architecture/2-histogram.md) §"Parallel
  construction" reflects this — no fixed-tid-order requirement.
- Atomic FP adds remain forbidden, but for the bit-stability-at-fixed-N
  reason, not the cross-N reason.

---

## 8. Two trees in Phase 1: depth-wise + oblivious

`DepthwiseGrower` → `DenseTree` and `ObliviousGrower` → `ObliviousTree`
both ship in Phase 1. Proposal puts oblivious in Phase 4; pulled
forward to force the `Tree` concept and `TreeGrower::Tree` associated
type to be honest from day one.

The two tree types have structurally different on-disk shapes (flat
node array vs per-level splits + leaf table) and structurally different
predict kernels (walk-until-leaf vs fixed-depth branchless gather).
With only depth-wise shipping, the second-tree-type machinery would
be aspirational. Same rationale as logloss alongside MSE in Phase 1
(proposal §1).

Cost: one extra grower + one extra tree type of spine code, plus a
second parity target (depth-wise vs xgboost/LightGBM, oblivious vs
CatBoost). Both targets share YearPredictionMSD.

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

Concept, not abstract base. `Booster<Gr, ...>::trees_` is
`std::vector<typename Gr::Tree>`, monomorphized. No vtable on predict.

Two `predict` overloads under one name (single-row returns float; batch
fills `out`). Disambiguation by arity. Row-major batch input — matches
xgb / lgbm. CatBoost's column-major fast path is motivated by predict-
time rebinarization, which we don't do (decision 3 — float thresholds).
Oblivious can transpose-on-demand internally if profiling justifies.

Rejected: `leaf_index(row)` (DenseTree and ObliviousTree leaf-index
spaces aren't unified — defer to Phase 4 if SHAP / leaf-output predict
is wanted); `walk(visitor)` (no shared node shape between the two
impls); serialization on the concept (lives in `bonsai::io` per
decision 6).

---

## 10. Shrinkage is baked into leaf values at tree construction

Grower receives `learning_rate` as a constructor argument (in
`TreeConfig`) and writes `lr · -G/(H + λ_l2)` into leaves. Trees are
pure functions of input rows; predict has no learning-rate knowledge.
Matches xgboost, LightGBM.

Rejected: per-iteration `learning_rate` argument to `grow()` (Phase 1
doesn't need decay schedules; non-breaking to add later as a `grow`
overload).

Knock-on: `Tree` concept doesn't carry a `set_shrinkage` mutator; trees
are immutable post-construction.

---

## 11. `ObliviousTree`: per-level `default_left`, not per-node

Every node at level `d` of an oblivious tree shares the same
`(feature_id, threshold, default_left)` triple — that's the symmetric-
tree contract. Relaxing `default_left` to per-node-at-level recovers a
small accuracy edge in heavily-missing data but breaks the branchless
predict kernel and isn't what CatBoost does.

`LevelSplit { uint32_t feature_id; float threshold; bool default_left; }`
× depth. Leaf table of size `2^depth`.

When a level's feature has no missing rows in training data,
`default_left` is don't-care; the splitter records whichever orientation
scored higher (arbitrary if no missing rows existed) and predict honors
it without thinking.

---

## 12. Sampler is the booster's responsibility, not the grower's

`grow(ds, grad, hess, row_indices)`. The grower receives sampled row
indices; it doesn't know what sampler produced them. "Use all rows" is
just `row_indices = 0..n_rows-1`.

Keeps `TreeGrower` concept narrow. Sampler swappable via
`Booster<Obj, Gr, Sa, Backend>` independent of grower choice.
Determinism contract testable on the grower without re-wiring the
sampler.

Rejected: grower owns sampler as a member (lgbm-style). Couples grower
template to sampler template; bigger cartesian product on `Booster`
instantiations for no compositional gain.

---

## 13. Branchless NaN routing in predict

```cpp
bool is_nan  = std::isnan(v);
bool less    = !is_nan && (v < threshold);
bool go_left = less | (is_nan & default_left);
```

Compiles to mask / select on x86-64 and ARM. Preserves vectorizability
for `ObliviousTree`'s batched predict; also keeps `DenseTree`'s
single-row predict tight.

Rejected: branchful `if (isnan(v)) ... else ...` (loses oblivious's
SIMD story); pre-cleaning predict input (caller can't know per-feature
default direction; rules itself out).

Knock-on: predict-time sentinels (e.g. `-999`) declared in
`BinMapperConfig` are *not* honored at predict — only `std::isnan`. Caller
contract: convert sentinels to NaN before predict. Matches xgb / lgbm.

---

## 14. Splitter is a template parameter on the grower; one `SplitFinder` concept

```cpp
template <SplitFinder Sp = HistogramSplitFinder> class DepthwiseGrower;
template <SplitFinder Sp = HistogramSplitFinder> class ObliviousGrower;
```

Static dispatch through to the splitter; inlined into `grow()`. Default
makes the common case ergonomic; explicit `Sp` makes the extension API
real.

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

Both growers consume the same `vector<Histogram>` shape and produce the
same `SplitCandidate`. Depth-wise calls `find` once per frontier node
with that node's histograms; oblivious calls `find` once per level with
the folded level histograms. The splitter doesn't know or care whether
its input is per-node or level-pooled.

Earlier draft of this entry split this into `PerNodeSplitFinder` /
`LevelSplitFinder` to make mismatched grower/splitter pairs a compile
error. Rejected: histogram-based scoring collapses the two signatures
to the same shape, so the "compile-time rejection" was illusory.
Phase 4 splitters that don't fit this shape (e.g. an exact splitter
scanning raw rows) earn their own concept when written.

Rejected: type-erased `unique_ptr<SplitFinder>` member (the dynamic-
dispatch shape we explicitly chose against in proposal §3.4).

---

## 15. Splitter returns one best candidate per call

`find(...)` returns a single `SplitCandidate`. `valid = false` if no
positive-gain split exists or no candidate clears `min_gain_to_split`.

Rejected: returning all per-feature candidates and letting the grower
pick max. No caller wants this; flexibility nobody's asking for.

---

## 16. Splitter tie-break: lowest `fid`, then lowest `bin_idx`

When two candidates have equal gain (within bit-exact equality, not
tolerance), prefer the one with the lower `feature_id`; if those tie,
prefer the lower `bin_idx`. Stable, deterministic at fixed thread
count. Matches lgbm's tie-break order (xgb's is implementation-
dependent in their `hist` updater).

Knock-on: same convention applies to the smaller-sibling choice in
the subtraction-trick wiring — when `n_left == n_right`, left wins.

---

## 17. Partitioning: per-node row-index lists (strategy A)

Each live `FrontierNode` carries its own `std::vector<uint32_t>` of row
indices. At root: one list with the booster-supplied `row_indices`. At
each split: partition the parent's list into `(left_rows, right_rows)`,
replace parent in frontier with two children carrying the new lists.

Both growers use this strategy (oblivious folds per-node histograms
into level histograms at scoring time; ~1.5% overhead on
YearPredictionMSD-scale data).

Rejected: single `row_to_node` array of length `n_rows` rebinned on
each split (xgb's `hist` updater shape; lgbm's voting parallel mode).
Beats per-node lists on cache locality at very shallow depth, loses at
typical `max_depth = 6`. More importantly: the subtraction trick wires
naturally onto per-node histograms. A single rebinned position
vector either fights subtraction (the all-live-nodes-in-one-pass kernel
doesn't know to skip the larger sibling) or imports per-node branching
back into the kernel.

Knock-on: oblivious grower needs a fold step
(`level_hists = sum_per_feature(frontier[*].hists)`) before split
scoring. Cheap relative to histogram build.

---

## 18. Frontier holds histograms inline (no histogram pool)

`FrontierNode { rows, sum_grad, sum_hess, hists }`. `vector<Histogram>`
per node, sized `n_features`. The grower carries
`std::vector<FrontierNode> frontier` and rotates it level-by-level.

Lists and histograms are local to `grow()`; the grower is stateless
across calls (one boosting iteration → one allocation cycle). Matches
lgbm's `SerialTreeLearner`.

Rejected: histogram pool with slot-id indirection (xgb's `hist`
updater). Recycles allocations across the tree, but Phase 1 isn't
allocation-bound. The pool refactor is contained if profiling later
shows the allocator dominating.

---

## 19. Subtraction trick from day one, both growers

Build smaller child by row-scan; derive larger by
`larger_hist = parent_hist - smaller_hist`. Halves histogram-build
work across the tree (per [`2-histogram.md`](architecture/2-histogram.md)
§"Why subtraction halves it"). Implemented from day one because
retrofitting means restructuring the grower's per-node memory.

Per-parent protocol:
1. Splitter scores `parent.hists`; commit candidate.
2. Partition `parent.rows` into `(left_rows, right_rows)`.
3. Pick smaller (left wins ties).
4. Build smaller's hists by row-scan.
5. Derive larger's hists by `parent_hist - smaller_hist` (also subtract
   `sum_grad` and `sum_hess`).
6. Push left, right into new frontier in left-then-right order
   (frontier order is structural, independent of build order).
7. Parent's hists released when parent goes out of scope.

For oblivious: same per-parent protocol, run once per parent at each
level. Cross-level subtraction (deriving level `d+1`'s level histogram
from level `d`'s) doesn't help — different levels score on different
features, so per-feature histograms for level `d+1` have to be built
regardless.

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

Validated in grower constructors; `ConfigError` with key path on bad
values. Section is `[tree]` in TOML.

Rejected for Phase 1: `max_leaves` (leaf-wise concept; depth-wise's
natural cap is `max_depth`, oblivious's leaf count is `2^depth` exactly).

Leaf value: `leaf_value = learning_rate · -G / (H + lambda_l2)`,
applied at finalization (decision 10).

## 21. `Objective` is a concept; static methods, no instance state

Matches the `SplitFinder` shape (decision 14). Two static functions
required: `compute(preds, labels, grad, hess)` writes per-row
gradients and hessians; `eval(preds, labels)` returns a scalar mean
loss. Dispatch is at the `Booster<Gr, Obj, ...>` template parameter,
fixed at compile time. No vtable; no shared mutable state across
calls.

Rejected: virtual base class (loses compile-time dispatch);
concept-with-instance-methods (no Phase 1 objective needs instance
state).

## 22. `Objective` does **not** own initial score or link inverse

The first-tree bias prediction (mean for MSE, log-odds for logloss)
and the predict-time link inverse (sigmoid for logloss) live in the
booster, not the objective. Rationale: both are score-accumulator
concerns. The booster maintains the running raw-score prediction
array, decides whether the bias comes from config or labels, and is
the sole owner of the predict path. Pushing them into `Objective`
would either force every objective to expose a `transform` and an
`initial_score` it might not need (e.g. MSE: `transform` is
identity), or invite a fragmenting set of optional methods on the
concept.

See `5-booster.md` (TBD) for where these land.

## 23. Phase 1 objectives: MSE + binary logloss, single-output

Two MVP impls satisfy the `Objective` concept:
`MSEObjective` (regression) and `LogLossObjective` (binary
classification). Both consume 1D per-row `floats_view` for `preds`
and `labels`, write 1D `floats_out` for `grad` and `hess`.

Rejected for Phase 1:

- **Multi-class / softmax.** K-output extension; touches
  `Objective`, `Booster`, `Tree` (K-output leaves). Phase 4.
- **Quantile, Huber, Tweedie, Cox.** Out of scope; satisfy the
  concept when added.
- **Custom user objectives.** No registry needed — anyone satisfying
  the concept can drop in as a `Booster` template parameter.

## 24. `compute` writes raw-score grad/hess; output is overwritten, not accumulated

`Objective::compute` writes `grad` and `hess` outright; callers don't
zero buffers first. `preds` are raw scores throughout
(logloss does **not** apply sigmoid inside `compute` or `eval`); the
booster keeps an additive raw-score accumulator across iterations
and applies the link only at the outermost predict call. Matches
xgboost / LightGBM; keeps boosting math additive.

`hess` is always non-negative (MSE: 1; logloss: `p·(1−p) ∈ (0,
0.25]`); the splitter's `min_child_hess` (decision 20) catches
near-zero hessians before they propagate.

## 25. Sample weights are applied by the booster, not the objective

When `Dataset.weights` is non-empty, the booster multiplies the
`grad` and `hess` buffers by the weight vector immediately after
`Objective::compute` returns and before handing them to the grower.
Keeps every `Objective` impl focused on loss math; the multiplication
is a 2-line buffer loop that doesn't belong duplicated across
objectives.

Rejected: a `WeightedObjective<T>` wrapper that satisfies `Objective`
by composing with weights. Adds a template layer with no semantic
content; same effect as the booster-side multiply.

## 26. Dispatch: flat table over cartesian product, `IBooster` at boundary

Runtime → static boundary uses Candidate A from
`architecture/6-dispatch.md`: a `constexpr std::array` keyed on a
name-tuple, generated by `for_each_type` over
`cartesian_product_t<Objectives, Growers, Splitters, Samplers>`. Each
cell is a monomorphized `Booster<O,G,S,Sa>` factory. Lookup at the
config boundary returns `unique_ptr<IBooster>`.

Cost: **one virtual call per `update_one_iter`**. Acceptable. The hot
path is histogram building inside `update_one_iter`, not the call
itself; per-iteration vcall is dwarfed by the per-iteration histogram
pass. Static-everywhere is preserved inside the iteration body, which
is what the proposal §3.4 rule actually targets.

Rejected:

- **Nested registry callbacks (Candidate B).** Fully static, zero
  vcalls anywhere, but forces continuation-passing at the boundary
  and drags the whole training run into the innermost lambda.
- **Dressed-up nested lambdas (Candidate C).** No advantage over A or
  B once the type-level builder is in play.
- **Hybrid flat-table + generic callback.** Doesn't compile —
  `std::array` of function pointers can't be generic over the
  callback type.

Invalid combinations (none in MVP) are pre-filtered at typelist
construction by building sub-products over compatible sub-typelists
and concatenating. No runtime check, no concept predicate, no
instantiation of bad cells.

`Backend` placement deferred to `7-parallel.md` — dispatch stays 4D
for now; promotion to 5D or separate composition both stay open.

## 27. Booster shape, training loop, and Sampler concept

Ratifies `architecture/5-booster.md`.

**Class shape.** `IBooster` is the boundary erasure type with a
minimal CLI-facing virtual surface (`update_one_iter`, `eval`,
`predict`, `n_iters`, accessors for save/load). `Booster<Obj, Gr,
Sp, Sa>` is the real class, monomorphized per cell of the dispatch
table (decision 26). One vcall per `update_one_iter` from the CLI;
zero inside.

**Training loop (`update_one_iter`).** Six steps in order:

1. `Obj::compute(scores_, labels, grad_, hess_)` — full row set.
2. Apply `Dataset.weights` to `grad_`, `hess_` if present
   (decision 25).
3. `Sa::sample(grad_, hess_, rng, out_indices)` → row indices for
   the grower.
4. `grower.grow(dataset, grad_, hess_, row_indices, split_finder,
   cfg)` → tree.
5. `scores_ += learning_rate * tree.predict(train_rows)` — re-walks
   the tree per training row.
6. `trees_.push_back(std::move(tree))`.

Sampling runs *after* grad/hess (not before) because GOSS samples on
`|grad|`, the grad pass is small (~5-10%) compared to histogram
building (60-80%, what sampling actually targets), and reordering
would force a "does this sampler need grad?" flag on the concept.

**`learning_rate`.** Applied at score-update time (step 5), not
pre-scaled into leaf values. Saved trees carry raw leaf values; the
booster reapplies the rate at predict and at score-update. Matches
xgb's "shrinkage = booster concern" model.

**Score update via re-predict, not cached leaf values.** Step 5
calls `tree.predict(train_rows)` rather than reading a precomputed
`(row → leaf_value)` array from the grower. Cost is one extra
`O(n_rows × depth)` walk per iter, small at MVP scale. xgb / lgbm /
catboost all cache the row→leaf mapping as a byproduct of growing
and reuse it for the score update; bonsai defers that optimization
to Phase 2 (after benchmarking). When justified, change grower
return from `Tree` to `(Tree, std::vector<float> train_leaf_values)`
and replace step 5 with a flat add. Pure additive change to the
grower→booster boundary; `Tree` stays clean.

**Initial score (bias).** Booster owns it (decision 22). Three
sources in priority: `cfg.init_score`; objective-appropriate default
from labels (mean for MSE, `log(p/(1-p))` for logloss); value loaded
from disk for a continued booster.

**Predict path.** Raw-score predict sums tree predictions plus
`init_score`, applies `learning_rate` per tree. User-facing predict
applies the objective's link inverse (identity for MSE, sigmoid for
logloss) via `if constexpr` on `Obj`. Inverse link lives in the
booster (decision 22), not in `Objective::compute` (decision 24).

**`Obj` is purely-static.** No instance member; `T::compute` and
`T::eval` are static (`4-objective.md` / decision 21).

**Booster borrows `Dataset`.** Lifetime sits with the CLI;
`update_one_iter` takes `Dataset const&`. Saved model is
`BinMappers` + trees + init_score, no `Dataset`. Booster does not
own `BinMappers` (decision 3 — predict path doesn't need them).

**Save / load are I/O, not booster methods.** Free functions
`save_booster(IBooster const&, path)` and `load_booster(path) ->
unique_ptr<IBooster>`. Rationale: `load` needs the four component
types *before* it has a `Booster` to dispatch on, so it reads names
from disk and calls into the same registry path `make_booster` uses
— that's structurally the dispatch boundary, not a member function.
`save` is the symmetric counterpart.

**`Sampler` concept (Phase 1).** Static members, same shape as
`Objective` and `SplitFinder`:

```cpp
template <typename T>
concept Sampler = requires(floats_view grad, floats_view hess,
                           std::mt19937& rng,
                           std::span<size_t> out_indices) {
    { T::sample(grad, hess, rng, out_indices) } -> std::same_as<std::size_t>;
};
```

Returns the count of selected indices written into the head of
`out_indices`; the buffer is owned by the booster and reused across
iterations. RNG is passed in by the booster, which owns determinism
(decision 7).

`NoSampler` is the Phase 1 identity impl (writes `0..n_rows-1`,
returns `n_rows`). `GOSS` and `BernoulliSampler` are Phase 4. The
sub-product machinery from `6-dispatch.md` handles any sampler-grower
incompatibilities. Sampler doc folds into `5-booster.md` for now;
spin out into `5b-sampler.md` if it grows.

Rejected:

- **`Obj` as instance member.** No state needed in MVP; revisit if a
  future objective needs config.
- **Cache row→leaf mapping in MVP.** Adds grower→booster API
  surface before benchmarking justifies it.
- **`IBooster::save` / `IBooster::load` as virtual methods.** Load
  has no `Booster` to dispatch on; both belong in an I/O module.
- **Sample-before-grad ordering.** GOSS dependency + small grad
  cost + concept-flag avoidance.
