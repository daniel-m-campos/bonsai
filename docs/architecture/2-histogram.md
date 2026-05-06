# 2. Histogram

> **Status:** Phase 1 design, ratified in
> [`../decisions.md`](../decisions.md) entries 1, 4, and 7. Spine
> component — initial impl reserved for hand-authoring (see
> [`../ai-usage.md`](../ai-usage.md)).

## Why

Histogram GBT scoring runs on a fixed shape: for each `(node, feature)`
pair, sum gradients and hessians into per-bin buckets, then walk the
bucket array once to score split candidates. This file pins down what
that bucket array looks like, who owns it, and how parallel building
keeps determinism without atomic adds.

Three properties drive the design:

1. **Hot loop is `hist[col[i]] += grad[i]`.** Layout must be flat,
   contiguous, and indexed by `uint16_t` bin without indirection.
2. **Subtraction trick halves work per split.** Sibling histogram =
   parent − built-sibling. Requires per-(node,feature) histograms to be
   addressable as a unit, and arithmetic over them to be elementwise.
3. **Determinism at fixed thread count** (decision 7). Per-thread local
   histograms, no atomic FP adds (bit-unstable even at fixed `n_threads`).
   Cross-thread-count bytes may differ; predictions match within
   tolerance.

## Algorithm context

This section is the "what is this object *for*" before the "what does
it look like." Skip if you already know histogram GBDT inside out;
otherwise the layout choices below won't motivate themselves.

### Gradients and hessians, per row

At each boosting iteration the booster has a vector of current scores
`s[i]` (one per training row, accumulated from prior trees). The
objective turns those into per-row first and second derivatives of the
loss:

```
g[i] = ∂L/∂s   evaluated at s[i], y[i]      // gradient
h[i] = ∂²L/∂s² evaluated at s[i], y[i]      // hessian
```

For MSE: `g[i] = s[i] − y[i]`, `h[i] = 1`. For binary logloss:
`p = σ(s)`; `g = p − y`; `h = p·(1 − p)`. Both are pointwise — no
cross-row dependence — so `g` and `h` are flat `vector<double>` of
length `n_rows`, recomputed once per iteration before the tree is
grown.

The grower's job: build *one* tree that, when its leaf values are
added to `s[i]`, reduces the next iteration's loss the most. Histogram
machinery exists to find that tree's splits cheaply.

### What a split-finder is doing

A node holds some subset of rows. To split it, we need to pick a
feature and a threshold such that routing rows left/right produces the
biggest drop in loss. The relevant XGBoost-style gain formula is:

```
gain(L, R) = G_L² / (H_L + λ)  +  G_R² / (H_R + λ)  −  G_P² / (H_P + λ)
```

where `G_L = Σ g[i]` over rows that go left, `H_L = Σ h[i]` likewise,
ditto `R` and parent `P`, and `λ` is L2 regularization on leaf
weights. Pick the `(feature, threshold)` that maximizes `gain`.
Negative gain → don't split.

So the split-finder needs `G_L, H_L, G_R, H_R` for every candidate
threshold on every feature. That's the work histograms exist to make
cheap.

### Why a histogram makes that cheap

Naively: for each candidate threshold `t` on feature `f`, scan all
node rows once, partition into left/right, sum gradients. That's
`O(n_rows · n_thresholds · n_features)` per node.

Pre-binning collapses thresholds: there are only `n_buckets` (≤256)
distinct cut positions per feature, given by the `BinMapper`. So:

1. **Build phase** (one pass over node rows per feature):

   ```
   for i in node_rows:
       bin = ds.column(f)[i]        # uint16_t
       hist[bin].sum_grad += g[i]
       hist[bin].sum_hess += h[i]
   ```

   `O(n_rows)` once per feature. After this, `hist[b]` holds the total
   gradient and hessian of every row whose feature `f` falls in bin
   `b`.

2. **Scoring phase** (one pass over bins per feature):

   ```
   // real-valued bins are 0 .. n_buckets - 2; bin n_buckets - 1 is missing.
   // G_real, H_real exclude the missing bin.
   G_L = 0; H_L = 0
   for b in 1 .. n_buckets - 2:
       G_L += hist[b - 1].sum_grad
       H_L += hist[b - 1].sum_hess
       G_R = G_real - G_L
       H_R = H_real - H_L
       score gain(L, R); track best
   ```

   `O(n_buckets)` per feature. Cumulative left-sums; right is the
   real-bin total minus left, no second pass over bins. `b` runs
   `1 .. n_buckets - 2` because the candidate cut sits *between*
   adjacent real-valued bins — `n_buckets - 2` real bins yield
   `n_buckets - 3` interior cuts plus the cut after bin 0, i.e.
   `b = 1 .. n_buckets - 2` putting bins `0..b-1` on the left.
   The missing bin (`n_buckets - 1`) is excluded from the sweep
   entirely; how its rows are routed is the splitter's call (see
   below).

   Total per node: `O(n_rows · n_features)` build + `O(n_buckets ·
   n_features)` score. The first term dominates. Going from
   `n_thresholds` candidates per feature (originally `O(n_rows)`) to
   `n_buckets` (≤256) is the whole point — and it's paid for in the
   scoring loop, not the build loop.

### Why subtraction halves it

When a node splits into children `L` and `R`, both children need
histograms before they themselves can be split. Naive cost: scan both
child row sets — total `n_parent_rows` work per feature, same as
building the parent.

But: `hist_parent[b] = hist_L[b] + hist_R[b]` exactly, bin by bin
(every parent row went somewhere). So if we already have `hist_parent`
cached and we scan only the smaller child to build its histogram, we
can compute the larger child's histogram by subtraction:

```
hist_R = hist_parent − hist_L          // elementwise, O(n_buckets)
```

The row-scan cost goes from `n_parent` to `min(n_L, n_R) ≤ n_parent /
2`. Halves histogram-build work across the tree. Free correctness
(it's exact arithmetic on the same accumulators). Implemented from
day one because retrofitting means restructuring the grower's
per-node memory.

### The missing bin

`BinMapper` reserves the **last** bucket (`n_buckets - 1`) for
missing/sentinel values. Real-valued bins occupy `0 .. n_buckets - 2`,
separated by right-edge cuts in `cuts_` (the final cut is a `+inf`
sentinel; NaN short-circuits straight to the missing slot). This
matches LightGBM's convention.

For real-valued split scoring, candidate thresholds live *between*
adjacent real-valued bins — cut positions `0|1, 1|2, ..., (n-3)|(n-2)`
where `n = n_buckets`. The scoring loop above sweeps exactly those.
The missing bin sits outside that sweep: its `(sum_grad, sum_hess)`
cell is honest accumulated data, but it doesn't participate in the
"left vs right of cut" partition.

How missing rows *are* routed at split time is a question for
`SplitFinder` — typical strategies: send all missing left, send all
missing right, score both and pick the better. Whichever strategy
the splitter uses, it reads the missing bin's totals separately and
folds them into one side's `(G, H)` before computing gain. The
histogram itself doesn't pick a strategy; it just keeps the missing
cell available.

### Tiny worked example

Node has 6 rows, feature `f` has 4 buckets (0–2 are real, 3 is
missing). Suppose for these 6 rows:

| row | bin | g    | h   |
|-----|-----|------|-----|
| 0   | 1   | +0.3 | 1.0 |
| 1   | 0   | −0.5 | 1.0 |
| 2   | 2   | +0.1 | 1.0 |
| 3   | 0   | −0.4 | 1.0 |
| 4   | 3   | +0.2 | 1.0 |
| 5   | 1   | +0.6 | 1.0 |

Build pass produces:

| bin | sum_grad | sum_hess |
|-----|----------|----------|
| 0   | −0.9     | 2.0      |
| 1   | +0.9     | 2.0      |
| 2   | +0.1     | 1.0      |
| 3   | +0.2     | 1.0      |  ← missing

Totals over real bins (0..2): `G_real = +0.1, H_real = 5.0`. Scoring
sweeps `b = 1 .. n_buckets - 2 = 2`, putting bins `0..b-1` on left:

- `b = 1` (cut at `0|1`): `G_L = −0.9, H_L = 2.0; G_R = +1.0, H_R = 3.0`
- `b = 2` (cut at `1|2`): `G_L = 0.0,  H_L = 4.0; G_R = +0.1, H_R = 1.0`

Plug into `gain(...)` with whatever `λ`, pick the bigger. The
splitter then decides whether the missing bin's `(+0.2, 1.0)` rides
left or right and re-scores. That's the whole inner game; the rest
is making it fast and parallel and deterministic.

## Cell layout

```cpp
struct HistCell {
    double sum_grad = 0.0;
    double sum_hess = 0.0;
};
```

- **`double` accumulators.** Float storage upstream, double here.
  Matches xgb/lgbm; bandwidth halved in `Dataset` and gradient buffers,
  precision preserved at the reduction.
- **AoS, not parallel arrays.** The build loop is the hot one
  (`O(n_rows)` per `(node, feature)`) and it scatters writes indexed
  by `bin`. AoS keeps grad+hess on one cache line per indexed update;
  SoA would force two cache lines in flight per row for no benefit,
  since the access pattern isn't sequential. Scoring is sequential
  but only `O(n_buckets ≤ 256)` and not the bottleneck. xgb/lgbm
  both AoS.
- **No count field.** `sum_hess` carries effective row count for MSE
  (`hess = 1` per row → `sum_hess == n_rows`); for logloss the count
  isn't needed by split scoring. If a future objective wants
  `min_data_in_leaf` independent of hessian, add a third lane then —
  not now.

Rejected: `float` accumulators (loses precision on long columns;
xgb/lgbm both use double); 32B padding to a cache line (wasteful —
4 bins per line is fine, the loop is sequential).

## `Histogram` — one feature, one node

```cpp
class Histogram {
public:
    explicit Histogram(size_t n_buckets);

    void add(uint16_t bin, double grad, double hess);
    void clear();                                      // zero cells, keep size

    HistCell const& operator[](size_t bin) const;
    size_t size() const;

    // Subtraction trick: *this -= other, in place. size() must match.
    Histogram& operator-=(Histogram const& other);

private:
    std::vector<HistCell> cells_;
};
```

- One `Histogram` per `(node, feature)`. The grower owns the matrix of
  these (see §"Ownership" below).
- `size() == n_buckets[fid]`, not `max_bin`. Variable per feature
  because `BinMapper::fit` deduplicates collisions and does
  low-cardinality fallback (decision 1 in [`../decisions.md`](../decisions.md)).
- `clear()` zeros the cells in place; `size()` is invariant. Histograms
  are allocated once per `(node-slot, feature)` and reused across
  iterations — `clear()` is the reset, not a destructor.
- `operator-=` is the subtraction trick primitive; precondition
  `size() == other.size()`. Asserted in debug, UB otherwise — this is
  a hot path called per `(child, feature)`.

The missing-bin cell (`cells_[n_buckets - 1]`) accumulates like any
other; the histogram doesn't know about missing semantics. Split
scoring is what excludes it from the real-valued sweep.

## Ownership

A node's histograms across all features = `vector<Histogram>` indexed
by `fid`. Per-feature, not flattened across features, because:

- Bucket count varies per feature (decision 1). A flattened
  `vector<HistCell>` would need a per-feature offset table — same
  indirection cost without the type-level clarity.
- Subtraction is per-(node, feature). `feat_hist[fid] -= sibling[fid]`
  reads one contiguous run.
- Parallel construction is feature-parallel in the OpenMP backend
  (Phase 3). Per-feature handles drop straight into a `parallel_for
  over fid`.

Lives in the grower (`3-tree.md`), not in `Dataset`. Histograms are
training-time scratch; `Dataset` is immutable input. Allocating in the
grower lets the grower reuse arenas across nodes.

## Subtraction trick — API

The algorithm is in §"Why subtraction halves it" above. The shape it
takes in the grower:

```cpp
auto small_hist = build_for(small_child, fid);    // row scan
auto large_hist = parent_hist;                     // copy
large_hist -= small_hist;                          // subtraction
```

The copy is `n_buckets * 16B` per feature — cheap relative to the row
scan it replaces. If profiling later shows the copy as hot, switch to
in-place: `parent_hist -= small_hist; large = parent_hist;` — but
parent is needed for both subtractions only on the *first* split of a
node, so the simple form is fine.

## Parallel construction

Determinism contract (decision 7, see
[`../decisions.md`](../decisions.md)): same seed + data + **same
thread count** → same model bytes. Different thread counts → predictions
within numerical tolerance, bytes may differ. This still rules out:

- **Atomic FP adds.** Floating-point add isn't associative; thread
  interleaving changes results bit-for-bit *even at fixed thread
  count*. So atomics break the fixed-N contract, not just cross-N.

It does *not* rule out:

- **OpenMP `reduction(+:...)`** or other parallel-reduce primitives,
  as long as the reduction shape is deterministic for a given thread
  count (which standard implementations are, per `schedule(static)` +
  fixed worker assignment).
- **Per-thread local histograms with any final-merge order**, as long
  as the merge order is the same on every run at that thread count.

The pattern:

1. Each thread owns a private `vector<HistCell>` sized for the feature
   (or for the full feature × bucket grid, for feature-parallel).
2. Threads scan their row ranges, accumulating into private cells
   only. No cross-thread writes during scan.
3. Final merge sums per-thread partials into the canonical histogram.
   Order doesn't have to be `tid`-major — it just has to be
   reproducible at fixed `n_threads`.

Cost: `n_threads * n_features * n_buckets * 16B` of scratch. On
YearPredictionMSD with 90 features, 255 bins, 8 threads: ~2.8MB. Fine.

`ParallelBackend` (concept, see [`7-parallel.md`](7-parallel.md)) wraps
the row-range scan + per-thread reduction so the same `Histogram::build`
template instantiates against `SerialBackend`, `OpenMPBackend`,
`StdExecBackend`. Histogram itself doesn't know which backend it's
running against — it exposes the `add` / `clear` / `-=` primitives and
trusts the backend to schedule.

## Hot loop notes

The build pseudocode is in §"Why a histogram makes that cheap." Two
implementation notes that don't fit there:

- **Root vs descendant.** At the root, `rows` is implicit `0..n_rows`
  and the loop is a straight sequential scan of `col` and
  `grad`/`hess`. At a descendant node, `rows` is the indirection list
  the grower maintains — same loop, different access pattern. SIMD
  opportunities differ between the two; phase 1 doesn't try to
  vectorize either.
- **Inlining.** `Histogram::add` must inline through to the cell
  write. Definition lives in the header.

## What's not here

- Where node-row partitioning lives (the `rows` span above) →
  [`3-tree.md`](3-tree.md).
- Split scoring — how the histogram is *consumed* — also
  [`3-tree.md`](3-tree.md).
- Backend dispatch for parallel build → [`7-parallel.md`](7-parallel.md).
- Categorical histograms (Phase 4). Layout is the same; what changes
  is split scoring (partition-based, not threshold-based).
- Serialization. Histograms are training-time scratch and don't
  round-trip to disk.

## Cross-references

- [`../decisions.md`](../decisions.md) entry 1 (binning, missing-bin
  reservation) and entry 4 (column-major, `uint16_t` storage).
- [`../proposal.md` §3.2](../proposal.md) for the
  performance-sensitive surface list (subtraction trick, deterministic
  reductions, double accumulators).
- [`1-dataset.md`](1-dataset.md) for the column input shape.
