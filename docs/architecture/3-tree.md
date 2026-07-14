# 3. Tree, Grower, Splitter

> **Status:** Phase 1 design, ratified in [`../decisions.md`](../decisions.md) entries 8–20.

## Why two trees from day one

Phase 1 ships **both** a depth-wise grower (`DepthwiseGrower` → `DenseTree`) and an oblivious grower (`ObliviousGrower` → `ObliviousTree`). The proposal puts oblivious in Phase 4 ([`../proposal.md` §7.1](https://github.com/daniel-m-campos/bonsai/blob/main/docs/proposal.md)); we pull it forward.

The reason: the oblivious tree has a fundamentally different on-disk shape than the depth-wise tree (per-level splits + leaf table vs flat node array), and the predict kernel is structurally different (fixed- depth branchless gather vs walk-until-leaf). If only depth-wise ships in Phase 1, the `Tree` concept and the `TreeGrower::Tree` associated type are aspirational — they accommodate a hypothetical second tree type but never get exercised. Pulling oblivious forward forces both the `Tree` concept and the `Booster<Gr, ...>` propagation of `typename Gr::Tree` to be honest from day one. Same rationale as landing logloss alongside MSE in Phase 1 ([`../proposal.md` §1](https://github.com/daniel-m-campos/bonsai/blob/main/docs/proposal.md)): two implementations of an open concept catch regression-only assumptions that one wouldn't.

The cost is one extra grower + one extra tree type of spine code, plus a second parity target (depth-wise vs xgboost/LightGBM, oblivious vs CatBoost). Both targets share the same regression dataset (California Housing for integration; YearPredictionMSD is held back as the post-parallelism perf benchmark).

## The three joints

This doc settles three things, in order of how they constrain each other:

1. **`Tree` as a concept** — the minimum surface every concrete tree type exposes. Two impls satisfy it: `DenseTree` and `ObliviousTree`.
2. **`TreeGrower` as a concept**, with associated `Tree` type. Two impls: `DepthwiseGrower` and `ObliviousGrower`. Both consume the `Dataset` (1) and per-row gradients/hessians (2), produce one `Tree` per call.
3. **`SplitFinder`** — split scoring. One concept; the splitter doesn't care whether the histograms it scores came from one frontier node (depth-wise) or from a level-wide fold (oblivious). Concrete impl: `HistogramSplitFinder`.

Each joint is described below. The grower's internal mechanics (partitioning, subtraction-trick wiring) follow.

## `Tree` — the concept

```cpp
template <typename T>
concept Tree = requires(T const t,
                         std::span<float const> row,
                         std::span<float const> rows, size_t n_features,
                         std::span<float> out) {
    { t.predict(row) }                         -> std::same_as<float>;
    { t.predict(rows, n_features, out) }       -> std::same_as<void>;
    { t.n_leaves() }                         -> std::convertible_to<size_t>;
    { t.depth() }                              -> std::convertible_to<size_t>;
};
```

Four members. Two `predict` overloads (single-row vs row-major batch), two diagnostics (`n_leaves`, `depth`).

- **No `BinMappers` at predict.** Trees store raw `float` thresholds (decision 3 in [`../decisions.md`](../decisions.md)). Predict consumes raw row data; no rebinning.
- **Row-major batch input.** The batch overload takes a flat `std::span<float const>` of `n_rows × n_features` row-major data and fills `out` with `n_rows` predictions. CatBoost's CPU evaluator uses a column-major fast path because it rebinarizes at predict; we don't, so row-major is the natural shape (matches xgboost, LightGBM). If benchmarks later show oblivious wants column-major to vectorize, an internal scratch transpose is a non-breaking implementation detail.
- **Shrinkage baked into leaves at construction.** The grower reads `learning_rate` from its `TreeConfig` and writes `lr · (-G/(H + λ))` into leaves. Trees are pure functions of input rows; predict has no learning-rate knowledge.
- **Concept, not abstract base.** `Booster<Gr, ...>::trees_` is `std::vector<typename Gr::Tree>`, monomorphized per grower. No vtable on the predict path.

What's deliberately *not* on the concept:

- No `leaf_index(row)`. The leaf-index space differs between `DenseTree` (positions in `nodes_`) and `ObliviousTree` (`0..2^depth - 1`). Phase 4 features that might want unified leaf indices (SHAP, leaf-output prediction) can add a tree-specific, non-concept method when needed.
- No `walk(visitor)`. The two impls don't share a node shape.
- No serialization on the concept. I/O lives in `bonsai::io` per [`1-dataset.md`](1-dataset.md) §"Serialization": free-function `write_binary(os, tree) / read_*_binary(is)` overloads per concrete tree type.

## `DenseTree` — flat node array (variant)

Output of `DepthwiseGrower`. Walks-until-leaf at predict.

```cpp
struct InternalNode {
    uint32_t feature_id;
    float    threshold;
    uint32_t left_child;     // index into nodes_
    uint32_t right_child;    // index into nodes_
    bool     default_left;   // missing routing
};

struct LeafNode {
    float    leaf_value;     // shrinkage already applied
};

class DenseTree {
public:
    float predict(std::span<float const> row) const;
    void  predict(std::span<float const> rows, size_t n_features,
                  std::span<float> out) const;

    size_t n_leaves() const { return n_leaves_; }
    size_t depth()      const { return depth_; }

    // Construction (used by grower; not part of concept).
    DenseTree(std::vector<std::variant<InternalNode, LeafNode>> nodes,
              size_t depth, size_t n_leaves);

private:
    std::vector<std::variant<InternalNode, LeafNode>> nodes_;
    size_t depth_;
    size_t n_leaves_;
};
```

`std::variant<InternalNode, LeafNode>` per slot. No wasted bytes, predict-time disambiguation is `std::holds_alternative` (or `std::visit`) — the same branch the simpler `feature_id < 0` sentinel would compile to. The variant form is honest about the two sub-shapes; serialization gets a clean tag-per-node story (`bonsai::io` writes the variant index then the active alternative).

**Predict kernel:**

```cpp
float DenseTree::predict(std::span<float const> row) const {
    uint32_t i = 0;
    while (auto const* n = std::get_if<InternalNode>(&nodes_[i])) {
        float v = row[n->feature_id];
        // Branchless missing routing (decision 13).
        bool is_nan   = std::isnan(v);
        bool less     = !is_nan && (v < n->threshold);
        bool go_left  = less | (is_nan & n->default_left);
        i = go_left ? n->left_child : n->right_child;
    }
    return std::get<LeafNode>(nodes_[i]).leaf_value;
}
```

The batch overload is a loop over rows. Single-row predict is the inner kernel; batch predict reuses it. Vectorization happens at the `predict_batch` boundary on `ObliviousTree`, where it pays.

## `ObliviousTree` — splits-per-level + leaf table

Output of `ObliviousGrower`. Branchless fixed-depth gather at predict.

```cpp
struct LevelSplit {
    uint32_t feature_id;
    float    threshold;
    bool     default_left;   // shared across all 2^d nodes at this level
};

class ObliviousTree {
public:
    float predict(std::span<float const> row) const;
    void  predict(std::span<float const> rows, size_t n_features,
                  std::span<float> out) const;

    size_t n_leaves() const { return leaf_values_.size(); }   // 2^depth
    size_t depth()      const { return splits_.size(); }

    ObliviousTree(std::vector<LevelSplit> splits,
                  std::vector<float>      leaf_values);

private:
    std::vector<LevelSplit> splits_;        // size = depth
    std::vector<float>      leaf_values_;   // size = 2^depth
};
```

Every node at level `d` shares `splits_[d]`. That's the symmetric-tree contract — feature_id, threshold, and `default_left` are level-wide, not per-node. (Decision 11.) Relaxing any of these to per-node breaks the kernel and is no longer "oblivious" in the CatBoost sense.

**Predict kernel:**

```cpp
float ObliviousTree::predict(std::span<float const> row) const {
    uint32_t idx = 0;
    for (auto const& s : splits_) {
        float v = row[s.feature_id];
        bool is_nan  = std::isnan(v);
        bool less    = !is_nan && (v < s.threshold);
        bool go_left = less | (is_nan & s.default_left);
        idx = (idx << 1) | (go_left ? 0u : 1u);
    }
    return leaf_values_[idx];
}
```

Fixed depth, branchless except for the NaN check (which the mask form turns into a select). The batch overload is where this representation earns its keep: the per-level inner work is a parallel comparison across all rows for one `(fid, threshold)` pair, foldable into a `depth × block_size` bit matrix that gathers `leaf_values_` at the end. Phase 1 ships a straightforward row-loop for `predict(rows, ...)`; the vectorized form is a profile-driven optimization. The kernel *shape* (no per-node branching) is what enables it.

## NaN handling — branchless mask

Both kernels use the same idiom (decision 13):

```cpp
bool is_nan   = std::isnan(v);
bool less     = !is_nan && (v < threshold);
bool go_left  = less | (is_nan & default_left);
```

NaN inputs short-circuit `less` to false, then `is_nan & default_left` selects the recorded missing-direction. This compiles to mask / select instructions on x86-64 and ARM, preserving vectorizability for `ObliviousTree`'s batched predict.

The `default_left` flag is what the splitter writes when scoring the missing-bin contribution at training time — see §"Missing rows" below.

## Missing rows — training to predict

The histogram's missing bin (`n_bins - 1`, populated by `BinMapper::transform` for NaN and configured sentinel inputs; [`1-dataset.md`](1-dataset.md), [`2-histogram.md`](2-histogram.md)) exists so the splitter can score "missing rows go left" vs "missing rows go right" and record the better orientation as `default_left`.

```
training:
  BinMapper bins NaN → bin n_bins-1
   → histogram cell accumulates (sum_grad, sum_hess) for missing rows
   → splitter scores both orientations, picks better
   → grower writes (fid, threshold, default_left) into the node

predict:
  raw float input → isnan check
   → if NaN, route by default_left
   → if not, compare to threshold
```

The chain is split into two halves joined at `default_left`. The predict path doesn't carry `BinMappers`; the missing-bin idea has done its training-time job by the time the tree is finalized.

**Predict-time sentinels.** `BinMapperConfig` allows declaring a non-NaN sentinel (e.g. `-999`) to mean missing. At training time, `BinMapper::transform` short-circuits both NaN and the configured sentinel into the missing bin. At predict time, the tree only checks `std::isnan(v)`. Predict-time inputs containing the sentinel are treated as the literal value, not as missing. Document this as a contract: callers must convert sentinels to NaN before predict. (Matches xgboost / LightGBM behavior; avoids per-feature sentinel plumbing in the model file and predict CLI.)

## `TreeGrower` — the concept

```cpp
template <typename T>
concept TreeGrower = requires(T g,
                               Dataset const& ds,
                               std::span<float const> grad,
                               std::span<float const> hess,
                               std::span<uint32_t const> row_indices) {
    typename T::Tree;
    requires Tree<typename T::Tree>;
    { g.grow(ds, grad, hess, row_indices) }
        -> std::same_as<typename T::Tree>;
};
```

One verb: `grow`. Returns a `Tree`. The grower instance is constructed once (with `TreeConfig`, learning rate, splitter) and `grow` is called once per boosting iteration.

- **Stateless across calls.** All per-tree scratch (frontier, row-index lists, histogram allocations) is local to `grow()`. Reusing one grower instance across boosting iterations is safe; reusing across *concurrent* boosting iterations is not, but Phase 1 has no concurrent-tree path.
- **Sampler is the booster's responsibility.** `row_indices` is the sampled set of rows for this iteration. The grower doesn't know what sampler produced them. (Decision 12.) "Use all rows" is just `row_indices = 0..n_rows-1`.
- **Learning rate is a constructor argument.** `TreeConfig` carries it, the grower bakes it into leaf values at tree construction time. Per-iteration schedules are a future `grow` overload.
- **Associated `Tree` type.** `typename T::Tree` is the grower's tree type. `Booster<Obj, Gr, Sa, Backend>` propagates it as `std::vector<typename Gr::Tree> trees_`.

## `DepthwiseGrower` and `ObliviousGrower`

Two concrete grower types. Each is a class template parameterized on a splitter; each names its `Tree`.

```cpp
template <SplitFinder Sp = HistogramSplitFinder>
class DepthwiseGrower {
public:
    using Tree = DenseTree;

    DepthwiseGrower(TreeConfig const& cfg, Sp splitter);

    DenseTree grow(Dataset const& ds,
                   std::span<float const> grad,
                   std::span<float const> hess,
                   std::span<uint32_t const> row_indices);
private:
    TreeConfig cfg_;
    Sp         splitter_;
};

template <SplitFinder Sp = HistogramSplitFinder>
class ObliviousGrower {
public:
    using Tree = ObliviousTree;
    /* same constructor signature */
    ObliviousTree grow(/* same args */);
};
```

Splitter is a template parameter (decision 14). Default is `HistogramSplitFinder`; both growers use the same concept and the same default impl. Phase 4 splitters with a different scoring shape (e.g. exact splitter, raw-row scan rather than histogram-based) will be introduced behind their own concept when needed.

The booster shape, restated:

```cpp
template <Objective Obj, TreeGrower Gr, Sampler Sa, ParallelBackend Backend>
class Booster {
    std::vector<typename Gr::Tree> trees_;
    /* ... */
};
```

`Gr` already carries its splitter. The booster doesn't see splitters directly.

## `SplitFinder` — one concept

Depth-wise and oblivious score different *contents* — depth-wise scans one node's per-feature histograms, oblivious scans a level's pooled per-feature histograms — but the *shape* of input and output is the same: `vector<Histogram>` of size `n_features` going in, one `SplitCandidate` coming out. The "node" the splitter scores is whatever the grower hands it; the splitter doesn't care whether those histograms came from one frontier node or from a level-wide fold.

```cpp
struct SplitCandidate {
    uint32_t feature_id;
    uint16_t bin_idx;        // grower converts to threshold via cuts[bin_idx]
    bool     default_left;
    double   gain;
    bool     valid;          // false if no positive-gain split found
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

One concrete `HistogramSplitFinder` impl satisfies it; both growers use the same instance. `DepthwiseGrower` calls it once per frontier node with that node's histograms; `ObliviousGrower` calls it once per level with the folded level histograms.

A two-concept variant (`PerNodeSplitFinder` / `LevelSplitFinder`) was considered to make mismatched grower/splitter pairs a compile error. Rejected: with histogram-based scoring the concept signatures collapse to the same shape, so the "compile-time rejection" property would be illusory. Phase 4 splitters that don't fit this shape (e.g. an exact splitter that scans raw rows rather than histograms) earn their own concept when they're written.

The splitter returns one best candidate per call (decision 15). If no split has positive gain or no candidate clears `min_gain_to_split` (see "Regularization" below), the candidate's `valid` is false and the grower stops growing along that path.

**Tie-breaks** (decision 16): when two candidates have equal `gain`, prefer lowest `feature_id`, then lowest `bin_idx`. Stable, deterministic at fixed thread count.

## Partitioning — list per node

Strategy A (decision 17): each live node carries its own `std::vector<uint32_t>` of row indices. At root: one node holding `row_indices` (the booster-supplied sampled set). At each split: partition the parent's list into `(left_rows, right_rows)`; replace the parent in the frontier with two children carrying the new lists.

Rejected: a single `row_to_node` array of length `n_rows` rebined on each split (LightGBM's voting parallel mode, xgboost's `hist` updater). At our typical depths (max_depth 6) the per-node lists win on cache locality once the tree gets past the root. More importantly: the subtraction trick wires naturally onto per-node histograms, where parent and child histograms are distinct objects ready for `operator-=`. A single rebuckected position vector either fights subtraction (the all-live-nodes-in-one-pass kernel doesn't know to skip the larger sibling) or imports per-node branching back into the kernel.

Lists are local to `grow()` — no grower-level pool. Phase 1 allocations are dominated by histogram building; pooling row-index storage is a future optimization that won't change the API.

## Frontier — one place node state lives

```cpp
struct FrontierNode {
    std::vector<uint32_t>   rows;
    double                  sum_grad;
    double                  sum_hess;
    std::vector<Histogram>  hists;       // [n_features]
};
```

The grower carries `std::vector<FrontierNode> frontier`. Each level's new frontier is built from the previous level's, using the subtraction-trick wiring described next.

The frontier holds histograms directly (decision 18) — not slot ids into a separate pool. xgboost's `hist` updater uses a histogram pool to recycle allocations across the tree; we don't, in Phase 1, because single-threaded allocation churn isn't a profile concern and the inline form is simpler to reason about. A pool refactor is contained if profiling later shows the allocator dominating.

## Subtraction-trick wiring

The mechanism (per [`2-histogram.md`](2-histogram.md) §"Why subtraction halves it"): build the *smaller* child by row-scan; derive the *larger* by `parent_hist - smaller_hist`. We commit to it from day one for both growers (decision 19).

**The protocol per parent split:**

```python
def split_parent(parent, candidate, ds, grad, hess):
    left_rows, right_rows = partition(parent.rows, candidate, ds)

    # Tie on size: left wins. Deterministic.
    if len(left_rows) <= len(right_rows):
        smaller_rows, larger_rows = left_rows, right_rows
        smaller_is_left = True
    else:
        smaller_rows, larger_rows = right_rows, left_rows
        smaller_is_left = False

    smaller = build_by_scan(smaller_rows, ds, grad, hess)
    larger  = build_by_subtraction(parent, smaller, larger_rows)

    # Frontier order is left-then-right, regardless of which was smaller.
    return (smaller, larger) if smaller_is_left else (larger, smaller)


def build_by_subtraction(parent, smaller, larger_rows):
    return FrontierNode(
        rows     = larger_rows,
        sum_grad = parent.sum_grad - smaller.sum_grad,
        sum_hess = parent.sum_hess - smaller.sum_hess,
        hists    = [p - s for p, s in zip(parent.hists, smaller.hists)],
    )
```

`build_by_scan` is `O(n_features × n_smaller_rows)`; `build_by_subtraction` is `O(n_features × n_bins)`. The work-savings come from `n_smaller_rows ≤ n_parent_rows / 2`.

**Root case.** No parent exists; build root histograms directly by row- scan over `row_indices`. One-time cost per `grow` call.

## Depth-wise grow loop

```python
def grow_depthwise(ds, grad, hess, row_indices, cfg, splitter, lr):
    tree = DenseTreeBuilder()
    frontier = [build_by_scan(row_indices, ds, grad, hess)]

    for _ in range(cfg.max_depth):
        new_frontier = []
        for parent in frontier:
            candidate = splitter.find(parent.hists, ds,
                                       parent.sum_grad, parent.sum_hess)
            if not candidate.valid or violates_regularization(parent, candidate, cfg):
                tree.add_leaf(leaf_value(parent, lr, cfg.lambda_l2))
                continue
            tree.add_internal(candidate)
            new_frontier.extend(split_parent(parent, candidate, ds, grad, hess))

        if not new_frontier:
            break
        frontier = new_frontier

    # Any nodes left in the frontier become leaves (max_depth reached).
    for node in frontier:
        tree.add_leaf(leaf_value(node, lr, cfg.lambda_l2))
    return tree.finish()


def leaf_value(node, lr, lambda_l2):
    return lr * -node.sum_grad / (node.sum_hess + lambda_l2)
```

Shrinkage is applied in `leaf_value` (decision 10). `tree.add_internal` records `(fid, threshold = cuts[bin_idx], default_left)` from the candidate; `tree.add_leaf` records the leaf value. Wiring node ids through the builder is straightforward bookkeeping, omitted here.

## Oblivious grow loop

> **Status:** implemented. The fold-then-score impl was reverted 2026-05-22 (decision 30) and re-landed with the correct per-parent gain summation described below; oblivious runs in every benchmark since.

Same outer shape as depth-wise. Differences: **all** nodes at the level share one chosen split, and the gain for a candidate split is the **sum of per-parent gains** across the frontier — not the gain of a single folded histogram.

```python
def grow_oblivious(ds, grad, hess, row_indices, cfg, lr):
    splits = []
    frontier = [build_by_scan(row_indices, ds, grad, hess)]

    for _ in range(cfg.max_depth):
        candidate = find_level_split(frontier, ds, cfg)
        if not candidate.valid:
            break  # every node at this level becomes a leaf

        splits.append(LevelSplit(candidate))

        new_frontier = []
        for parent in frontier:
            new_frontier.extend(split_parent(parent, candidate, ds, grad, hess))
        frontier = new_frontier

    leaf_values = [leaf_value(node, lr, cfg.lambda_l2) for node in frontier]
    return ObliviousTree(splits, leaf_values)
```

`find_level_split` iterates `(feature, bin, default_left)` candidates and, for each, sums per-parent gain across the frontier:

```python
def find_level_split(frontier, ds, cfg):
    best = invalid_candidate()
    for fid in range(ds.n_features()):
        for bin_id in range(ds.n_bins(fid) - 1):
            for default_left in (True, False):
                level_gain = 0.0
                feasible   = True
                for p in frontier:
                    gL, hL, gR, hR = split_sums(p.hists[fid], bin_id, default_left,
                                                p.grad, p.hess)
                    if hL < cfg.min_child_hess or hR < cfg.min_child_hess:
                        feasible = False  # oblivious is all-or-nothing
                        break
                    level_gain += (score(gL, hL, cfg.lambda_l2)
                                 + score(gR, hR, cfg.lambda_l2)
                                 - score(p.grad, p.hess, cfg.lambda_l2))
                if feasible and level_gain > best.gain and level_gain >= cfg.min_gain_to_split:
                    best = Candidate(fid, bin_id, default_left, level_gain)
    return best
```

**Why per-parent summation, not fold-then-score.** The gain function `score(g, h) = g²/(h + λ)` is non-additive:

```
score(Σ g_i_L, Σ h_i_L) + score(Σ g_i_R, Σ h_i_R) − score(Σ g_i, Σ h_i)
  ≠
Σ_i [ score(g_i_L, h_i_L) + score(g_i_R, h_i_R) − score(g_i, h_i) ]
```

Folding histograms before scoring gives the first expression — which is *not* the gain induced by applying one split to every parent in the frontier. The second expression is. CatBoost confirms this in [catboost/private/libs/algo/greedy_tensor_search.cpp](https://github.com/catboost/catboost/blob/master/catboost/private/libs/algo/greedy_tensor_search.cpp): the symmetric-tree path calls `CalcBestScore` → `CalcStatsAndScores`, which builds per-leaf histograms across the current depth's leaves and aggregates gains via `SetBestScore`.

**Constraint shape.** `min_child_hess` is enforced **per parent** — if any parent in the frontier would create an undersized child under the candidate, the entire candidate is rejected. This is what "all nodes at the level share the chosen split" implies: oblivious is all-or-nothing. `min_data_in_leaf` is not enforced for oblivious (matches CatBoost — that knob applies only to its Lossguide / Depthwise growing policies).

**Cost.** `O(n_features · n_bins · |frontier|)` per level for the
gain accumulation — same order as the original fold-then-score sketch; the difference is in *what* is accumulated (gain, not histogram cells). Build phase after `find_level_split` (the `split_parent` calls) is bit-for-bit identical to depth-wise's per-parent subtraction-trick loop. The two growers share that helper.

The output `ObliviousTree` is constructed from:
- `splits_` = the `LevelSplit` recorded at each level (one per level).
- `leaf_values_` = leaf values from the final frontier in left-to-right order, one per node.

## Regularization

`TreeConfig` (entry in [`8-config.md`](8-config.md)) carries the following knobs (decision 20):

| Knob | Default | Meaning |
|---|---|---|
| `max_depth`            | 6     | Hard cap on tree depth. |
| `min_data_in_leaf`     | 20    | A node with fewer than this many rows cannot be split. Children's row counts are also checked. |
| `min_sum_hessian_in_leaf` | 1e-3 | Minimum `sum_hess` for a node to be split. Hessian as effective row-count proxy under non-MSE objectives. |
| `lambda_l2`            | 1.0   | L2 regularization on leaf weights, in the gain formula and leaf-value computation. |
| `min_gain_to_split`    | 0.0   | Minimum gain for a candidate to be accepted. Defaults to "any positive gain." |

Validated in `DepthwiseGrower` / `ObliviousGrower` constructors. `ConfigError` thrown with key path on bad values.

`max_leaves` shipped with `LeafwiseGrower` (decision 31), which uses it as the primary stopping criterion; depth-wise and oblivious ignore it (their natural caps are `max_depth` and `2^depth`). Growers also enforce `feature_fraction`, `lambda_l1`, `monotone_constraints`, and `interaction_constraints` — see decisions 34–35 and the [guide](../guide/) for the concept-level treatment.

## Leaf values

For each finalized leaf:

```
leaf_value = learning_rate · -G / (H + lambda_l2)
```

where `G = sum_grad` and `H = sum_hess` over the leaf's rows. This matches xgboost's formulation (the same gain formula minimized analytically). Shrinkage is applied at construction (decision 10); trees are pure functions of input rows.

## Determinism

The contract from [`../decisions.md`](../decisions.md) §7 (same seed + same data + **same thread count** → same model bytes) is satisfied here by:

- **Deterministic tie-breaks** in the splitter (decision 16): ties in gain resolve by lowest `fid`, then lowest `bin_idx`.
- **Deterministic smaller-sibling choice**: when `n_left == n_right`, left wins. The build order doesn't affect bytes but does affect reduction shape under parallelism (Phase 3).
- **Frontier order is deterministic**: left-then-right per parent, inherited from input row order through partitioning.
- **No floating-point atomics** during histogram build (per [`2-histogram.md`](2-histogram.md) §"Parallel construction").

Cross-thread-count reproducibility is *not* promised; predictions agree within numerical tolerance.

## What's not here

- `Tree` and `Histogram` serialization → `bonsai::io` per [`1-dataset.md`](1-dataset.md) §"Serialization".
- Booster shape and training loop → `5-booster.md`.
- Backend dispatch for parallel histogram build, parallel splitter scoring → `7-parallel.md`.
- Sampler concept → `5-booster.md` (sampler is the booster's, not the grower's, responsibility).
- Categorical splits (Phase 4). Layout of the histogram stays the same; what changes is `SplitFinder`'s partition-search inner loop.
- `LeafwiseGrower` (Phase 4). Will satisfy `TreeGrower`, produce `DenseTree`, use the same `SplitFinder` concept. Its frontier shape is a priority queue, not a level-wise vector.

## Cross-references

- [`../decisions.md`](../decisions.md) entries 8–14 (this doc's ratifying decisions); entries 1, 3, 4, 7 (binning, float thresholds, column-major storage, determinism) for the inputs the grower consumes.
- [`../proposal.md` §3.1, §3.4](https://github.com/daniel-m-campos/bonsai/blob/main/docs/proposal.md) for the `TreeGrower` / `SplitFinder` extension points and the static- dispatch ethos.
- [`1-dataset.md`](1-dataset.md) for `Dataset` shape and missing-bin convention.
- [`2-histogram.md`](2-histogram.md) for `Histogram` shape, the subtraction trick, and the determinism contract for parallel build.
