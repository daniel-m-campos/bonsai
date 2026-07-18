# 17: Native categorical features: Fisher set splits (stage 2a) + ordered target statistics sketch (stage 2b)

> **Status:** REOPENER PREDICATE ESTABLISHED (decision 80): catboost's own toggle prices its categorical machinery at 68% of its remaining TabArena lead over the encoder on cat-heavy data (`benchmarks/tabarena-cat-probe-2026-07.md`); the build decision waits on the launch-strategy call (issue #157). Previously priced and DECLINED by measurement (decision 58, PR #67): each reference library's own categorical toggle measured native set splits at +0.029/+0.000/−0.018 AUC across amazon/adult/kick, while ordered-TS *preprocessing* (`OrderedTargetEncoder`, guide 13) beat lightgbm-native from outside the core; see `benchmarks/categorical-tradeoff-2026-07.md`. This design is retained as the priced option; the decision entry names the evidence that would reopen it. Feature-gap row 12, stage 2 ([`../feature_gap.md`](https://github.com/daniel-m-campos/bonsai/blob/main/docs/feature_gap.md) §18).

## Motivation

Stage 1 measured the gap on Amazon employee access (32,769 rows, 9 integer-ID features, RESOURCE 7,518 distinct): native Fisher set splits are worth **+0.0144 AUC inside lightgbm** (0.8405 → 0.8549) and **~+0.007 against bonsai's best today** (0.8476), while fitting ~3× faster because one set split replaces many axis splits.

Ordered target statistics are the bigger prize (catboost leads the field at 0.8812, **+0.034 over bonsai today**) and plain K-fold target encoding does *not* reproduce it (0.8462), so the permutation scheme is load-bearing, not the encoding per se.

This doc designs 2a (set splits) fully and sketches 2b (ordered TS) enough to show 2a's plumbing is what 2b needs.

## Non-goals

- Feature *combinations* (catboost's crossed categoricals), a large chunk of catboost's Amazon edge, but a separate machine; revisit after 2b.
- String categories in CSV/libsvm ingest: categorical columns must arrive as non-negative integer codes in float cells (pandas users pass `.cat.codes`); parsing strings is an ingest feature, not a split feature.
- Auto-detection from dtype: declaration is explicit; the Python surface takes `float32` arrays and cannot see pandas dtypes anyway ([`../../python/bonsai/__init__.py`](../../python/bonsai/__init__.py)).
- One-hot expansion mode and a `max_cat_to_onehot` knob: the exact scan below already degenerates to one-hot behavior for tiny cardinalities.
- CUDA-native categorical kernels in 2a (see CUDA plan: documented decline to the host path).

## API / config surface

`[data] categorical_columns = ["MGR_ID", "RESOURCE"]`: strings, each a column name or a decimal index, matching the `interaction_constraints` string-list precedent; lands in `DataConfig` ([`../../include/bonsai/config/data_config.hpp`](../../include/bonsai/config/data_config.hpp)) next to `ignore_columns`.

Python: `categorical_features=[...]` kwarg on `BonsaiRegressor`/`BonsaiClassifier`, mapped to `data.categorical_columns` like every other first-class kwarg.

Two new `TreeConfig` knobs ([`../../include/bonsai/config/tree_config.hpp`](../../include/bonsai/config/tree_config.hpp)): `cat_smooth = 10.0` (hessian smoothing in the sort key) and `max_cat_threshold = 32` (cap on left-set size), lightgbm's names and defaults, so cross-library benches stay apples-to-apples.

Config validation rejects a monotone constraint on a declared categorical column (direction is meaningless over an unordered set) and rejects `categorical_columns` entries that name no column.

`Dataset::is_categorical(fid)` already exists ([`../../include/bonsai/dataset.hpp`](../../include/bonsai/dataset.hpp)) and is currently always false. This design wires it from `DataConfig` through `Dataset::bin`.

## Binning: category codes → bins 1:1

`BinMapper` gains a categorical mode ([`../../include/bonsai/bin_mapper.hpp`](../../include/bonsai/bin_mapper.hpp)): instead of quantile cuts it stores the sorted list of kept codes, and `transform(x)` is an exact-match binary search.

Bin layout: bins `0..K-1` = kept codes in ascending code order, bin `K` = overflow ("other", present only when cardinality exceeds the budget), last bin = missing, preserving the "last bin is the NaN slot" invariant every consumer relies on ([`../../src/bin_mapper.cpp`](../../src/bin_mapper.cpp), `Histogram::missing()`).

Overflow policy: when distinct codes exceed `max_bin - 2`, keep the most frequent `max_bin - 3` over the fit subsample (tie-break by ascending code, deterministic) and fold the rest into the overflow bin, which participates in the scan as an ordinary category.

Unseen codes at transform time map to the missing bin: "we know nothing" routes like NaN, by `default_left`.

Mapper fit validates the column: NaN allowed, otherwise values must be non-negative integers exactly representable in f32 (`< 2^24`); violations throw at `Dataset::bin`, not deep in training.

## Data flow

```mermaid
flowchart LR
    A[raw codes<br/>float column] -->|categorical BinMapper:<br/>exact code lookup| B[bins u8/u16<br/>0..K-1 + other + missing]
    B -->|histogram fill<br/>unchanged| C[per-bin HistCells]
    C -->|sort bins by G/(H+cat_smooth),<br/>prefix-scan sorted order| D[SplitOutput +<br/>winning bin set]
    D -->|finalize: bin set → raw code set<br/>via mapper| E[tree node: set index<br/>+ CatSets table]
    E -->|training: bin-space bitset test<br/>predict: code binary search| F[route left/right]
```

The histogram fill, `HistCell` layout, and the subtraction trick are untouched; a categorical feature is just a feature whose bin ids happen to carry no order.

## The Fisher sort: why scanning one order suffices

A split assigns each category (bin) `k` with sums `(G_k, H_k)` to the left or right child; the second-order objective for child weights `w_L, w_R` is separable per category:

```math
\mathcal{L} = \sum_{k \in L} \left( G_k w_L + \tfrac{1}{2} H_k w_L^2 \right) + \sum_{k \in R} \left( G_k w_R + \tfrac{1}{2} H_k w_R^2 \right) + \tfrac{\lambda}{2}(w_L^2 + w_R^2)
```

For any fixed pair `w_L < w_R`, category `k` prefers left iff `G_k w_L + ½H_k w_L² ≤ G_k w_R + ½H_k w_R²`, i.e. iff

```math
\frac{G_k}{H_k} \le -\tfrac{1}{2}(w_L + w_R)
```

, a threshold on `G_k/H_k`, so the optimal partition at the optimum is *contiguous* in `G_k/H_k` order (Fisher 1958's grouping result), and scanning the `2^{K-1}` partitions reduces to scanning `K-1` prefixes of the sorted order.

With leaf-level L2 the per-category argument is exact only for `λ = 0`; following lightgbm we sort by `G_k / (H_k + \text{cat\_smooth})`, where `cat_smooth` doubles as regularization against low-count categories dominating the order, near-optimal in practice, and the test plan checks exactness at `λ₂ = 0` by brute force.

## Split scan changes

In `update_best_for_feature_for_node` ([`../../src/split.cpp`](../../src/split.cpp)): for a categorical feature, gather the non-empty cut cells into thread-local scratch, sort by the smoothed ratio (tie-break by bin id, determinism), then run the *existing* prefix accumulation and `score_candidate` core (issue #50) over the sorted order, with two extra gates: prefix length ≤ `max_cat_threshold`, and both `default_left` arms as today.

The winning candidate records the sorted prefix as a bin set; `SplitOutput` grows an optional set payload (empty = numeric threshold, today's shape).

The oblivious level scan (`update_best_for_feature_for_level`) needs one order shared by the whole frontier since the split is broadcast: sum the per-parent cells into a level-aggregate histogram, sort *that*, and prefix-scan every parent in the aggregate order. Per-parent optimal orders differ, and the aggregate is the level-consistent choice, same spirit as the issue-#60 zero-gain-contribution rule.

Monotone constraints skip categorical features by construction (rejected at config validation); interaction constraints work unchanged (they gate features, not thresholds).

## Node representation + model format

`DenseTree::Node` stays 20 bytes ([`../../include/bonsai/tree.hpp`](../../include/bonsai/tree.hpp)): a set split sets the top bit of `feature_id` (`is_set_split(n)` / `split_feature(n)` helpers; construction asserts `n_features < 2^31 - 1`, and the flagged id can never equal `k_leaf_flag` under that cap), and `threshold_or_value` holds the set index as an exactly-representable integer (assert `< 2^24`).

The tree gains a `CatSets` side table: one flat `std::vector<uint32_t>` of sorted raw codes plus a `std::vector<uint32_t>` of offsets; a node's set is a slice, membership at predict is a binary search over ≤ `max_cat_threshold` codes.

Sets store raw *codes*, not bins, upholding decision 3 (trees store raw thresholds; no `BinMappers` at predict): tree finalization converts the winning bin set through the mapper's kept-code list once, at grow time.

`ObliviousTree::LevelSplit` gets the same top-bit + set-index treatment; `dense_equivalent` passes the flagged fields through verbatim, so oblivious SHAP inherits the DenseTree path.

Model format ([`../../src/io/model.cpp`](../../src/io/model.cpp)): `k_format_version` 7 → 8; tree objects gain `"cat_codes"` / `"cat_offsets"` (empty for numeric-only trees), mapper entries gain `"categorical": true` with codes in a `"codes"` array (`"cuts"` stays numeric-only); the loader accepts {7, 8}, with 7 implying no categorical fields, the `covers`-absent precedent.

Rejected: widening `Node` to 24 bytes (regresses the decision-49-era predict-path win for every model, categorical or not); a variant node (the 20-byte flat node exists precisely because the variant was measured slower); keying set-ness off model-level mapper flags at predict (trees must stay self-contained; the `Tree` concept has no mapper access).

## Routing + missing/unseen policy

Training-side routing is bin-space: `plan_level` carries, parallel to `split_bins`, a per-set-split bitset over the feature's bins (≤ `max_bin` bits of scratch); `route_unsampled` ([`../../src/grower_impl.hpp`](../../src/grower_impl.hpp)) and the leafwise partition test membership instead of `b <= split_bins[idx]`.

The missing bin keeps today's semantics exactly: `b == last` routes by `default_left`, and the scan's two-arm evaluation already prices both defaults: NaN, unseen-at-predict codes, and unseen-at-transform codes all land there.

The overflow "other" bin is an ordinary member candidate: it can be in or out of the set, chosen by gain like any category.

Predict-side routing is code-space: `goes_left` for a set node is `!is_nan && contains(cat_sets[idx], (uint32_t)v)`, else `default_left`.

## SHAP treatment

Verified: the TreeSHAP walk ([`../../src/shap.cpp`](../../src/shap.cpp)) touches split semantics in exactly one place: `hot_child`'s goes-left predicate; the path algebra (`extend`/`unwind`, cover fractions, the interventional `expected_value_impl`) reasons only about node ids, covers, and `feature_id` equality, all of which are predicate-agnostic.

There are currently three copies of the goes-left predicate (`DenseTree::leaf_for`, `ObliviousTree::leaf_for` in [`../../src/tree.cpp`](../../src/tree.cpp), `hot_child` in `shap.cpp`); a byte-identical prep PR extracts one shared `goes_left(node, X, row)` so the set branch lands once.

The `feature_id`-equality checks in SHAP must compare the *masked* id, so a numeric and a set split on the same feature condition the same path slot. The masked-accessor helpers make this mechanical.

## CUDA plan

Stage 2a is **CPU-only by design**: the cuda growers decline datasets with categorical columns at `begin_root` (the existing oversized-`max_bin` decline arm in [`../../src/level_step.hpp`](../../src/level_step.hpp), the ONE runtime fork the doc-12/13 design allows) and training proceeds on the host plane with a one-line log, exactly like the current fallback.

What the device path needs later, priced now so the decline is a decision and not an accident: `find_kernel` / `level_find_kernel` ([`../../src/cuda/histogram_engine.cu`](../../src/cuda/histogram_engine.cu)) gain a per-(node, cat-feature) key sort of ≤ `max_bin` cells (single-warp bitonic in shared memory) before the prefix walk; the winning sorted-prefix is materialized as a device bitset the apply/partition kernels test; the set rides the existing find-stage D2H alongside `SplitOutput`.

None of that changes the level-transaction vocabulary (decision 53); it is find/apply kernel work only, but it is real effort and ships only if 2a's measured host-side quality delta justifies it.

## Determinism / hash-gate implications

2a adds no RNG: the ratio sort tie-breaks by bin id, the frequency cap tie-breaks by code, and `reduce_in_feature_order` keeps its serial-order tie-break, so categorical models are bit-identical at fixed thread count, the decision-49 contract, with no new spend.

Numeric-only datasets must be **byte-identical** before/after every 2a PR (`scripts/model_hash.py` gate; the quality-gates ritual); the feature is additive and the numeric scan path must not move.

Categorical output has no baseline to hash against, so it is **quality-table validated**: `scripts/bench_categorical.py` on Amazon, target ≥ +0.007 AUC over bonsai-numeric (≈ 0.8549, the lightgbm-native line), recorded in [`../feature_gap.md`](https://github.com/daniel-m-campos/bonsai/blob/main/docs/feature_gap.md) §18 as the stage-2a row.

## Stage 2b: ordered target statistics (sketch)

Per tree `t`, draw a permutation `σ_t` of the training rows from `mt19937(hash(booster.random_seed, t))`; for each categorical feature, replace row `i`'s code with the running smoothed mean of the label over *earlier* rows in `σ_t` sharing its category:

```math
\text{TS}(i) = \frac{\sum_{j:\, \sigma(j) < \sigma(i),\, c_j = c_i} y_j + a \cdot p}{\left|\{j : \sigma(j) < \sigma(i),\, c_j = c_i\}\right| + a}
```

, causal by construction (row `i` never sees its own label), which is what plain K-fold encoding lacked and why it measured at 0.8462.

The TS column is numeric-on-the-fly: quantize it against a per-feature cut set fitted once at ingest (on the σ₀ encoding), so the histogram plane sees ordinary bins and only the fill's input column changes per tree.

Sampling interaction: TS is computed over the *full* training set per permutation, independent of the row sampler, so bagging/GOSS neither perturbs the encoding nor doubles the causality bookkeeping.

Determinism: the permutation is seeded, and the per-feature running mean is a serial prefix pass (parallel *across* features), bit-identical at any thread count, stronger than the fill's contract, so 2b spends nothing new either.

Predict uses the full-training-set statistics per category, serialized as a per-feature TS table in the model. This is the 2b model-format addition.

Cost note: one prefix pass per categorical feature per tree (O(n_rows) each) plus a per-tree re-fill of those columns; on Amazon-shaped data this is small, on 16M-row data it is why 2b is decision-gated rather than assumed.

## Risks

- **High-cardinality overfit / cap loss**: RESOURCE's 7,518 codes collapse to ~252 + other under `max_bin = 255`, and greedy set splits overfit rare categories; mitigated by `cat_smooth`, `max_cat_threshold`, and honestly: catboost's +0.026 says 2b is the real answer here; 2a's gate is beating the lightgbm-native line, which lives under the same cap.
- **Set-index-in-float fragility**: exact only below 2^24, asserted at both encode sites and covered by a construction test; documented as a deliberate 20-byte-node trade.
- **Top-bit feature-id flag**: collides with `k_leaf_flag` only at `fid = 2^31 - 1`; construction asserts `n_features < 2^31 - 1`; all readers go through the masked accessors from the prep PR, so no raw `feature_id` comparison survives.
- **Sort cost in the hot scan**: ≤ 255-element sort per categorical feature per node in thread-local scratch; numeric features pay nothing (branch on `is_categorical` per feature, outside the bin loop); confirm with the perf-round ledger on a mixed dataset before merge.
- **Aggregate-order suboptimality (oblivious)**: the shared level order can miss per-parent optima: accepted; it is the same aggregation compromise the level scan already makes, and issue #60's zero-gain rule bounds the damage.
- **Unseen-category distribution shift**: predict routes unseen codes by `default_left`, which training chose for NaN+unseen mass, documented in the guide; smarter fallbacks (route to "other") are a follow-up knob if it bites.

## Test plan

- Mapper unit tests: 1:1 code→bin mapping, frequency cap + overflow bin, NaN → last bin, unseen → last bin, non-integer/negative input throws, deterministic cap tie-break.
- Fisher exactness: for random histograms with K ≤ 12, `λ₂ = 0`, `cat_smooth = 0`, the scan's best equals exhaustive `2^{K-1}` partition enumeration; with `λ₂ > 0` the scan beats the best numeric-order axis split.
- Routing parity: every training row lands in the same leaf via bin-space routing (`route_unsampled`) and code-space routing (`predict`/`leaf_for`): the mapper round-trip test.
- Model I/O: v8 round-trip with byte-stable predictions; a pinned v7 fixture still loads; a v8 numeric-only model has empty cat fields.
- SHAP: `sum(phi) == predict` property test on categorical trees, dense and oblivious-via-`dense_equivalent`.
- Gates: `model_hash.py` unchanged on numeric-only datasets after every PR; Amazon quality row ≥ 0.853 AUC recorded in feature_gap §18.
- CUDA: a test asserting the cuda grower declines a categorical dataset and the fallback model equals the cpu grower's bit-for-bit.

## Staged delivery

| PR | scope | gate | effort |
|---|---|---|---|
| 2a-0 | unify the goes-left predicate (tree.cpp ×2, shap.cpp) + masked-id accessors | byte-identical (hash) | 0.5 d |
| 2a-1 | `DataConfig.categorical_columns`, Python kwarg, categorical `BinMapper`, `Dataset` wiring, validation | byte-identical for numeric configs | 1 d |
| 2a-2 | Fisher scan (node finder), set payload in `SplitOutput`, `CatSets` node repr, bin/code routing, model v8, SHAP branch | numeric hash + Amazon quality row | 2 d |
| 2a-3 | oblivious aggregate-order variant + `dense_equivalent`/SHAP + cuda decline test | numeric hash + oblivious Amazon row | 1 d |
| 2a-4 | docs (this doc → architecture/, decisions entry, guide section), feature_gap §18 update, perf spot-check | — | 0.5 d |
| 2b | ordered TS per the sketch, own design round first | decision-gated on 2a's measured delta | ~1 wk |

Stage 2b proceeds only if 2a's Amazon delta lands as measured and the remaining gap to catboost (~+0.026) still justifies the per-tree encoding cost. That call gets its own decision entry.
