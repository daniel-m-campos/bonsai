# 4 — Growing trees

## The idea

Chapters 2–3 explain how to split *one* node. A grower decides **which
node to split next**, and that scheduling choice is most of what
distinguishes the three reference libraries:

- **Depth-wise** (xgboost's default): split every node at depth `d`
  before touching depth `d+1`. Balanced trees, capped by `max_depth`.
- **Leaf-wise / best-first** (lightgbm's default): always split the single
  leaf — anywhere in the tree — whose best split has the highest gain.
  Deliberately *unbalanced*: for a fixed leaf budget (`max_leaves`), it
  drives training loss down fastest, at higher overfitting risk.
- **Oblivious / symmetric** (catboost): every node at a level shares *one*
  split. The tree is a perfect table — 2^depth leaves indexed by a
  bitstring — which makes predict extremely fast and acts as a strong
  regularizer.

## The math

There's no new math — only a queue discipline over the same gain scores:

```
depth-wise:  FIFO by level          — frontier is a vector, loop per depth
leaf-wise:   max-heap keyed on gain — pop best, split, push two children
oblivious:   argmax over the SUM of per-parent gains for one shared (feature, bin)
```

Leaf-wise's property worth internalizing: each expansion converts one leaf
into two (net +1), so `max_leaves` bounds the tree exactly, and the tree
is the greedy-optimal sequence of the first $\texttt{max\_leaves} - 1$ splits. A
depth-wise tree spends its budget evenly; a leaf-wise tree spends it where
gain lives.

## In bonsai

All three share every mechanism from chapters 2–3 — `make_root`,
`split_node`, `finalize_as_leaf`, the same split finders — and differ only
in their grow loop ([`src/grower.cpp`](../../src/grower.cpp)):

- `DepthwiseGrower::grow` — `current` / `next` frontier vectors,
  `update_nodes` splits or finalizes each node of a level, swap, repeat
  until `max_depth`.
- `LeafwiseGrower::grow` — a `std::vector<Candidate>` maintained with
  `std::push_heap`/`std::pop_heap` (chosen over `std::priority_queue`
  because `top()` returns const& and fights moving the histogram-heavy
  payload out). Gain is the key; ties break on lower node id so growth is
  deterministic. Stops at `max_leaves`, `max_depth`, or a drained heap.
  There's a full implementation walkthrough in
  [leafwise_grower.md](../leafwise_grower.md) — the guide this grower was
  actually built from.
- `ObliviousGrower::grow` — one shared split per level via the *level*
  finder (gain = sum of per-parent gains — see decision 30 for the
  fold-then-score mistake that summing per-parent fixes), children by
  index arithmetic, leaves in a $2^{\text{depth}}$ table.

Both node-splitting growers emit a `DenseTree` (flat node array); the
oblivious grower emits an `ObliviousTree` (level splits + leaf table) —
[`include/bonsai/tree.hpp`](../../include/bonsai/tree.hpp). Selection is a
config string: `dispatch.grower_name = depthwise | leafwise | oblivious`.

## Try it

```bash
# Same leaf budget, three disciplines:
uv run scripts/compare.py --config configs/year_prediction_msd.toml \
    --growers depthwise,leafwise,oblivious --samplers all_rows
```

Expect the ordering the benchmarks reproduce every time: leaf-wise is the
best accuracy-per-second (31 leaves ≈ depthwise-8's RMSE at half the
time), depth-wise squeezes out the last fraction of RMSE with 16× more
nodes, oblivious trades a little RMSE for the fastest predict.

## Gotchas & war stories

- **`max_leaves` vs `max_depth` govern different growers.** A leafwise
  config that only sets `max_depth=8` still stops at the default 31
  leaves — the budget, not the cap, is usually what binds.
- **Heap ties are a determinism hazard.** Equal gains happen (symmetric
  data produces them exactly); without the node-id tie-break, two runs
  could grow different trees and both look "correct".
- **The oblivious gain is a sum of per-parent gains, not the gain of
  summed histograms.** The score is quadratic in $G$, so folding histograms
  first computes the wrong thing — the original implementation did, and
  produced degenerate trees whose failure mode looked plausible enough to
  get rationalized (decision 30). The re-implementation matches catboost's
  per-leaf aggregation.
