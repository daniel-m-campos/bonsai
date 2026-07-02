# 2 — Binning & histograms

## The idea

To split a node you must ask, for every feature, "which threshold best
separates low residuals from high ones?" The exact answer requires sorting
every feature's values — expensive, repeatedly. The histogram trick answers
a slightly blurrier question much faster: discretize each feature into at
most ~255 quantile buckets **once**, up front; then a node's split search
is just "sum gradients into 255 cells, scan the cells". No sorting ever
again, and feature values shrink to one byte-ish integers that are kind to
caches.

The blur costs almost nothing: with 255 quantile bins, candidate
thresholds sit at every ~0.4th percentile of the feature. Splits between
them were statistically indistinguishable anyway — which is why xgboost
(`tree_method=hist`), lightgbm, and catboost all default to this.

## The math

Binning maps value to bucket via quantile cut points
$c_0 < c_1 < \cdots < c_{k-1}$: bin $b$ holds values in $(c_{b-1}, c_b]$
(right-inclusive). Per node and feature, the histogram accumulates
per-bin sums:

```math
\text{cell}[b] = \Big(\textstyle\sum_i g_i,\; \sum_i h_i\Big)
\quad\text{over rows } i \text{ in the node with } \operatorname{bin}(x_i) = b
```

A candidate split "$\le b$" needs the left-side sums $G_L, H_L$ — a prefix
sum over cells — and the right side is $G - G_L$ by subtraction from the
node totals. One O(bins) scan scores every threshold of a feature
([chapter 3](3-finding-splits.md)).

**The subtraction trick**: when a node splits, its two children partition
its rows, so $\text{hist}(\text{parent}) = \text{hist}(\text{left}) +
\text{hist}(\text{right})$ cell-by-cell. Build the histogram only for the
*smaller* child and get the larger one free:
$\text{hist}(\text{large}) = \text{hist}(\text{parent}) -
\text{hist}(\text{small})$. Since the smaller child has at
most half the rows, this halves histogram work at every level — the single
most important optimization in histogram GBT.

## In bonsai

- **Fitting cuts** — [`BinMapper::fit`](../../src/bin_mapper.cpp):
  reservoir-sample the column, pull quantiles with `nth_element` at a
  fixed stride, deduplicate, append `+inf` as a sentinel. The **last bin
  is reserved for missing values** (NaN); a plain sentinel value can also
  be mapped to missing via `data.missing_sentinel`.
- **Binning** — [`Dataset::bin`](../../src/dataset.cpp): per feature,
  `transform` is a `lower_bound` over the cuts. Storage is column-major
  (`feature_bins(fid)` is contiguous) because histogram building walks one
  feature across many rows.
- **The histogram** — [`include/bonsai/histogram.hpp`](../../include/bonsai/histogram.hpp):
  a `vector<HistCell>{sum_grad, sum_hess}`. `add(bin, g, h)` is the entire
  hot loop body. Note what's *not* there: running totals. They used to be
  maintained per `add` — two redundant double-adds per row×feature,
  duplicated across every feature of the node — and were hoisted to a
  once-per-node cell sweep (`totals()`; decision 33).
- **Building per node** — `populate_from_rows` in
  [`src/grower.cpp`](../../src/grower.cpp). Two details worth reading:
  grad/hess are first *gathered into node-row order* so each of the ~90
  per-feature scans streams memory sequentially instead of re-walking two
  full arrays with scattered indices; and the feature loop is parallel —
  each feature's histogram is owned by one thread ([chapter 9](9-parallelism-and-determinism.md)).
- **The subtraction trick** — the tail of `split_node`, same file:
  `populate_from_rows(small)`, then `large.hists[f] -= small.hists[f]`
  feature-parallel, with the parent's histograms *moved*, not copied.

## Try it

```bash
# Coarser bins: how much accuracy do 32 buckets really cost?
uv run scripts/compare.py --config configs/california_housing.toml \
    --hp bin_mapper.max_bin=32 --growers leafwise --samplers all_rows
```

On California Housing, dropping 255 → 32 bins moves RMSE by well under 1%
for every library — the blur really is cheap — while histogram scans get
8× shorter.

## Gotchas & war stories

- **The missing bin participates in every histogram** but is excluded from
  the threshold sweep; split scoring decides which side NaNs travel
  ([chapter 3](3-finding-splits.md)). Forgetting that the last bin is
  special is the classic off-by-one of this design.
- **Accumulate in double, store in float.** Cells are double pairs;
  gradients are float. Half a million float adds into a float accumulator
  loses real precision; into a double, it doesn't.
- **Row order = reproducibility.** FP addition isn't associative, so *the
  order rows enter `add` is part of the contract*. bonsai keeps each
  node's rows in ascending order (a stable split scatter), which is also
  why its histograms are bit-identical at any thread count.
