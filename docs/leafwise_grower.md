# Adding a Leaf-Wise (Best-First) Grower

A guide for implementing `LeafwiseGrower` in bonsai. This is a learning map — it explains
what a leaf-wise grower is, why each piece exists, and the exact seams to plug into. The
code sketches are illustrative, not copy-paste.

## Context

bonsai currently ships two growers (`include/bonsai/grower.hpp`):

- **`DepthwiseGrower`** — level-wise. Splits *every* node at depth `d` before moving to
  `d+1`. Each node picks its own best feature/threshold. Stops at `max_depth`.
- **`ObliviousGrower`** — symmetric. One shared split per level applied to all frontier
  nodes.

Both are **breadth-first**: they expand by level. A **leaf-wise** grower (LightGBM's
default, a.k.a. best-first) is **priority-first**: at each step it expands the *single*
leaf whose split yields the largest gain, anywhere in the tree, regardless of depth. This
drives training loss down faster per node added, producing unbalanced trees that are more
accurate for a fixed leaf budget — at higher overfitting risk, which is why the leaf count
(`max_leaves`) is the primary regularizer.

**Goal:** add a third grower, `LeafwiseGrower`, selectable via
`DispatchConfig.grower_name = "leafwise"`, reusing the existing histogram split-finder and
`DenseTree` output.

**Stop criteria:** `max_leaves` primary + `max_depth` cap. Grow best-first until the leaf
budget is reached; never expand a leaf that would exceed `max_depth`; also stop when no
leaf has a valid positive-gain split.

## The core idea

Depthwise loop (today, `src/grower.cpp:157`):

```
for depth in 0..max_depth:
    for each node in current level:   # process ALL nodes at this level
        find best split; split or finalize
```

Leaf-wise loop (new):

```
push root's best-split candidate onto a max-heap keyed by gain
while heap not empty and n_leaves < max_leaves:
    pop the globally best candidate
    materialize its split -> left, right children
    for each child: if depth < max_depth, find its best split, push onto heap
                    else mark it a pending leaf
finalize every leaf still pending
```

The heap is what makes it "best-first": the next node split is always the highest-gain
opportunity in the *entire* tree, not the next one in level order.

## What you'll build

### 1. Config: add `max_leaves` — `include/bonsai/config/tree_config.hpp`

Add one field to `TreeConfig` (currently lines 7-16):

```cpp
uint32_t max_leaves = 31;   // LightGBM's default; 0 = unbounded (depth-capped only)
```

- Keep `max_depth` — it's now the depth cap, not the primary limit.
- `TreeConfig` already has `bool operator==(...) = default;` — a new field is picked up
  automatically.
- Wire `max_leaves` through wherever `TreeConfig` is parsed from CLI/JSON config
  (grep for `max_depth` to find every parse/serialize site).

### 2. The grower class — `include/bonsai/grower.hpp`

Mirror `DepthwiseGrower` (lines 32-42). It uses the **same** `NodeSplitFinder` concept and
produces the **same** `DenseTree`, so the class shape is nearly identical:

```cpp
template <NodeSplitFinder SplitterT = HistogramNodeSplitFinder>
class LeafwiseGrower
{
  public:
    using Tree = DenseTree;
    explicit LeafwiseGrower(TreeConfig const &cfg);
    GrowResult<Tree> grow(Dataset const &ds, floats_view grad, floats_view hess,
                          row_index_view row_indices);
  private:
    TreeConfig config_;
};
```

It satisfies the existing `TreeGrower` concept (lines 22-30) for free — no concept changes.

### 3. The grow loop + heap — `src/grower.cpp`

This is the heart of the exercise. Reuse the existing file-local helpers — they already do
everything except the scheduling:

- `make_root(...)` (line 53) — builds the root `SplitInput` with full histograms.
- `populate_from_rows(...)` (line 37) — builds per-feature histograms from row indices.
- `split_node(...)` (line 63) — partitions rows, builds child histograms (incl. the
  smaller-child + subtraction trick at lines 88-97). Returns `{left, right}` SplitInputs.
  Reuse as-is.
- `finalize_as_leaf(...)` (line 25) — writes a leaf value into `nodes` and `values`,
  increments `n_leaves`.
- `SplitterT::find(node, config_)` (`src/split.cpp:152`) — returns a `SplitOutput` whose
  `.gain` and `.valid` you key the heap on.

**Heap element.** You need to carry, for each pending split: the node's `SplitInput`
(histograms + rows + id), its best `SplitOutput`, and its `depth`. A `SplitInput` owns
heavyweight `std::vector<Histogram>` — so move it into the heap, never copy.

- `std::priority_queue::top()` returns `const&`, which fights with moving the element out.
- Prefer a `std::vector<Candidate>` maintained with `std::push_heap` / `std::pop_heap` and
  a custom comparator, which lets you `std::move` the popped element. **Recommended** here
  because `SplitInput` is large and move-only-friendly.

```cpp
struct Candidate {
    SplitInput  node;     // histograms + rows + id  (move-only payload)
    SplitOutput split;    // best split for this node; .gain is the heap key
    uint8_t     depth;
};
// comparator: a.split.gain < b.split.gain   (max-heap on gain)
```

**Node id allocation.** Unlike depthwise (which allocates ids level-by-level in
`update_nodes`, lines 118-121), here you allocate ids as you pop. Pattern, mirroring
`update_nodes`:

```cpp
// when expanding a popped candidate `c` whose split is valid:
node_id_t left_id  = nodes.size(); nodes.emplace_back(DenseTree::leaf(0.0F));
node_id_t right_id = nodes.size(); nodes.emplace_back(DenseTree::leaf(0.0F));
float threshold = ds.mappers()[c.split.feature_id].cuts()[c.split.bin_id];
nodes[c.node.id] = DenseTree::internal(c.split.feature_id, threshold,
                                       left_id, right_id, c.split.default_left);
auto [left, right] = split_node(ds, grad, hess, std::move(c.node), c.split,
                                left_id, right_id);
// for each child: find split; if valid && depth+1 < max_depth -> push; else pending leaf
```

**Counting leaves against `max_leaves`.** Every node in the heap is currently a leaf; each
expansion converts 1 leaf into 2 (net +1). So: start `live_leaves = 1` (the root). Each
expansion does `++live_leaves`. Stop expanding once `live_leaves >= max_leaves`, then
finalize everything remaining in the heap as leaves. This makes the final leaf count
exactly `max_leaves` (or fewer if the heap drains first).

**Loop skeleton:**

```cpp
auto cmp = [](Candidate const &a, Candidate const &b){ return a.split.gain < b.split.gain; };
std::vector<Candidate> heap;
std::vector<SplitInput> pending_leaves;

SplitInput root = make_root(ds, grad, hess, row_indices);
nodes.emplace_back(DenseTree::leaf(0.0F));         // root id 0, like depthwise line 154
SplitOutput root_split = SplitterT::find(root, config_);
size_t live_leaves = 1;
uint8_t max_observed_depth = 0;

if (root_split.valid && config_.max_depth > 0) {
    heap.push_back({std::move(root), root_split, 0});
    std::push_heap(heap.begin(), heap.end(), cmp);
} else {
    finalize_as_leaf(nodes, root, config_.lambda_l2, n_leaves, values);
}

while (!heap.empty() && live_leaves < config_.max_leaves) {
    std::pop_heap(heap.begin(), heap.end(), cmp);
    Candidate c = std::move(heap.back()); heap.pop_back();
    // allocate ids, set internal node, split_node(...) as above
    ++live_leaves;
    for (child in {left, right}) {
        uint8_t cd = c.depth + 1;
        SplitOutput cs = (cd < config_.max_depth) ? SplitterT::find(child, config_)
                                                   : SplitOutput{/*invalid*/};
        if (cs.valid) {
            heap.push_back({std::move(child), cs, cd});
            std::push_heap(heap.begin(), heap.end(), cmp);
        } else {
            pending_leaves.push_back(std::move(child));
        }
        max_observed_depth = std::max(max_observed_depth, cd);
    }
}
// budget hit or heap drained: every remaining heap entry + pending becomes a leaf
for (auto &c : heap)           finalize_as_leaf(nodes, c.node, config_.lambda_l2, n_leaves, values);
for (auto &n : pending_leaves) finalize_as_leaf(nodes, n,      config_.lambda_l2, n_leaves, values);
```

**Watch-outs:**

- `finalize_as_leaf` takes `SplitInput const&` and reads `node.id` / `node.rows` /
  `node.total_grad/hess()` — children get ids only when their parent is expanded, which
  the sketch does before pushing. Leaves never expanded still carry the id stamped at
  creation in `split_node`.
- Tree metadata: `GrowResult` returns
  `Tree(std::move(nodes), {.depth = ?, .n_leaves = ?})` (depthwise: line 179). For
  leaf-wise, `depth = max_observed_depth`, `n_leaves = live_leaves`. Confirm `DenseTree`'s
  metadata struct field names in `include/bonsai/tree.hpp`.
- Explicit template instantiation: add
  `template class LeafwiseGrower<HistogramNodeSplitFinder>;` at the bottom of
  `src/grower.cpp` (depthwise does this at line 183).

### 4. Register the grower (2 small edits)

The dispatch system is string-keyed and statically checked. Follow the depthwise pattern.

- **`include/bonsai/registry/typelists.hpp:16`** — add to the `Growers` typelist:
  ```cpp
  using Growers = TypeList<DepthwiseGrower<HistogramNodeSplitFinder>,
                           ObliviousGrower<HistogramLevelSplitFinder>,
                           LeafwiseGrower<HistogramNodeSplitFinder>>;
  ```
- **`include/bonsai/registry/names.hpp`** — add the name trait (mirrors lines 38-41):
  ```cpp
  template <> struct impl_name<LeafwiseGrower<HistogramNodeSplitFinder>> {
      static constexpr std::string_view value = "leafwise";
  };
  ```
  The `static_assert(all_named_v<Growers>)` at `typelists.hpp:35` fails to compile if you
  forget this — a good guardrail.
- No dispatcher/booster changes needed — selection flows through
  `DispatchConfig.grower_name` automatically.

## Files to touch (summary)

| File | Change |
|------|--------|
| `include/bonsai/config/tree_config.hpp` | add `max_leaves` field |
| `include/bonsai/grower.hpp` | declare `LeafwiseGrower` class |
| `src/grower.cpp` | implement `grow()` + heap loop; explicit instantiation |
| `include/bonsai/registry/typelists.hpp` | add to `Growers` |
| `include/bonsai/registry/names.hpp` | add `impl_name` specialization |
| (config parse sites) | wire `max_leaves` from CLI/JSON — grep `max_depth` |

No changes to `split.hpp` / `split.cpp` — the node split-finder is reused unchanged.

## Verification

1. **Build**: compile the project (the `all_named` static_assert confirms registration).
2. **Unit test** (mirror existing grower tests — grep `tests/` for `DepthwiseGrower`):
   - Tiny dataset, `max_leaves = 1` → tree is a single root leaf.
   - `max_leaves = N` on a separable dataset → final `n_leaves == N` exactly (until the
     heap drains).
   - `max_depth` cap: small `max_depth`, huge `max_leaves` → no leaf deeper than
     `max_depth`; verify the tree is *unbalanced* (depths differ across leaves), unlike
     depthwise.
   - Determinism: same inputs → identical tree (heap ties must break deterministically).
3. **Parity sanity**: with `max_leaves = 2^max_depth` on a small problem, leaf-wise
   training loss should be `<=` depthwise for the same leaf count.
4. **End-to-end**: run the CLI/benchmark with `grower_name = "leafwise"` on a real dataset;
   confirm it trains and metrics are reasonable vs. `depthwise`.

## Open questions you'll hit while coding

- **Heap tie-break**: equal gains need a stable order for reproducibility. Suggest a
  secondary key = node id (FIFO-ish). Decide before writing the comparator.
- **`max_leaves == 0` semantics**: treat as "unbounded" (depth-capped only), or reject in
  config validation. Pick one and document it on the field.
