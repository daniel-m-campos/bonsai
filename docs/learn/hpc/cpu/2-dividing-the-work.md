# C2. Dividing the work

The bottom line first. The histogram fill can run its parallel loop over rows or over features. Each form streams one side of memory and scatters the other, and the winner is whichever scattered side fits.

## The job

To split a node, a booster needs one histogram per feature. Each histogram holds, for every bin, the sum of the gradients and hessians of the node's rows in that bin.

See [guide chapter 2](../../../guide/2-binning-and-histograms.md) for the definition and the subtraction trick that halves the work. The loop itself is three lines of intent:

> for each row in the node, for each selected feature: add that row's gradient and hessian into the cell at (feature, bin).

**Formula:** `adds_per_fill = n_node_rows * n_selected_features`

**Example:** A root fill at either running example makes 2^28 adds, about 268 million.

There is no arithmetic to remove here. The body is two additions of a float pair, and the bin id is one byte already in memory.

No ordering of the loop changes the answer either. So everything below is about where the bytes are, not about what is computed.

Two arrays are in play, and they pull in opposite directions:

- **The mirror.** The bins, stored row by row, so one row's bins are contiguous bytes. See [guide chapter 2](../../../guide/2-binning-and-histograms.md) for details.
- **The arena.** The node's histograms, stored feature by feature, so one feature's 256 cells are contiguous.

A row walk is sequential in the mirror and scattered in the arena. A feature walk is the reverse. There is no third option, because the two arrays cannot both be walked in order.

## What gets touched, how often

Each cell update is a read-modify-write, or RMW: load the cell's current sums, add, store back. Two facts follow. A cell updated thousands of times wants to stay resident between updates, and two workers cannot RMW the same cell without losing updates.

Only the smaller child of a split is ever filled. The larger child's histogram is the parent's minus the smaller child's, one subtraction per cell. A row is therefore re-scanned at a level only when its node is the smaller one.

| structure | written | read |
|---|---|---|
| mirror | once at construction, never again | one row-slice stream per fill; a row re-reads at most once per level |
| gradients and hessians | once per boosting round | once per row per fill, once more at leaf values |
| arena cells | zeroed once, then RMW about `n_node_rows / 256` times each | once by the subtraction, once by the split scan |
| row index lists | each row id moves once per level | once per fill, once per partition |

Read the asymmetry out of the table. The mirror is read-many and written never, so workers sharing its lines costs only bandwidth ([chapter C1](1-cache-awareness.md)'s refetch arithmetic).

The arena is written many times over, so workers sharing its lines is false sharing plus lost updates. Every division below therefore gives each worker exclusive arena cells and negotiates only over who re-reads the mirror.

## The two forms, and who pays

Parallelizing means handing each worker a piece. Within one node, a piece can be a range of rows or a range of features. Nothing else exists.

**Feature-parallel.** Each worker owns a contiguous range of features and reads every row. Its writes go only to its own features' arena runs, so no two workers ever write the same cell. Its reads cover the whole mirror, so every worker reads every row.

**Row-parallel.** Each worker owns a range of rows and touches every feature. Its reads are its own rows, so no two workers read the same mirror bytes. Its writes go to every feature, so workers collide on cells and cannot write the arena directly.

| | feature-parallel | row-parallel |
|---|---|---|
| reads | shared: every worker reads every row | private: each worker reads its own rows |
| writes | private: each worker owns its features' cells | conflicting: every worker writes every feature |
| extra machinery | none | one partial histogram per extra worker, plus a reduce |
| determinism | the serial order at any worker count | the order depends on the partition, so on the worker count |

The row-parallel form's extra machinery is the expensive half. The cost is not the addition, it is the memory.

**Formula:** `partial_cells = (n_workers_on_node - 1) * n_selected_features * 256`

Each of those cells is zeroed before the fill and read again during the reduce. Neither pass does any useful arithmetic.

**Example:** At the tall cell with 12 threads, the leaf plane's partial volume, its `partial_cells` summed over fills, measured 24x the depthwise plane's. That is 118 million cells per two boosting rounds.

The count also grows with the thread count: 11 partial sets per node at 12 threads against 3 at 4. So the row-parallel price rises exactly when you add workers to make it cheaper.

## The wide collapse and the tall refusal

The campaign's first lever made the lone-node fill feature-parallel. Each worker takes a contiguous feature range of one mirror tile and accumulates straight into the arena. Zero partials, zero partial zeroing, zero reduce.

The wide cell collapsed. Leaf-plane populate fell from 172.1 to 45.5 seconds on the same pod, below the depthwise plane's own 59 to 61 seconds at that cell.

Every lever in the campaign ran against a prediction written down before the measurement. If the result lands outside the predicted range, the theory behind the lever is wrong, no matter how fast the code got.

This lever's theory was that partial volume is the dominant cost, and the written prediction was a wide populate of 70 to 110 seconds and a tall one of 13 to 17. Wide beat its range outright.

The tall cell refused to move. Populate read 27.8 seconds before and 30.7 after, nowhere near the predicted 13 to 17.

The host was noisy, two byte-identical builds of the same code differed 9% on it, but no noise covers a miss that large. Partial volume was not what made tall slow, so the mechanism was somewhere else.

The line arithmetic from chapter C1 points at it. Slicing a 128-byte row among 12 workers gives each about 11 features, so all 12 refetch the same 2 lines.

Slicing a 16,384-byte row instead gives each worker its own ~21 lines, and the overlap at the boundaries is negligible.

So the feature-parallel form is a bandwidth trade. It removes the partials and adds the refetch multiplier from chapter C1, `max(1, W / L)`.

At 256 lines that trade is free. At 2 lines it costs `W / 2`: roughly 6x the read bytes at 12 workers.

A second experiment confirmed the same shape another way: it removed synchronization instead of memory traffic, and tall barely moved. [Chapter C3](3-coordination-costs.md) tells that story.

## The 2D form that serves both shapes

The answer reads as one sentence. Go feature-parallel, and split the rows into blocks only when the row has too few cache lines to give every worker its own.

**The rule.** Divide the selection into contiguous feature ranges. While the mirror row is too narrow to give every worker its own 64-byte line, split the node's rows into blocks as well. Run the cross product of (range, block) with exactly that many workers.

**The price.** Every block past the first owns one partial arena, reduced in ascending block order. A plan with more than one block therefore pays the row-parallel machinery again.

Blocks are taken only while each one still owns at least 8,192 rows. That is the work a block must do to earn the partial it is charged.

**The result at tall.** Populate fell from 33.5 to 21.7 seconds and train from 45.4 to 32.9 on one EPYC draw. The pre-registered falsifier bar was a 10% drop, and the lever cleared it three times over.

Mirror traffic per tree fell from 3.85GB to 1.45GB in the same change.

**The result at wide.** Unchanged, as predicted, once a self-inflicted regression was found and reverted. [Chapter C3](3-coordination-costs.md) tells that part, because its cause was how many workers were requested, not how the work was divided.

The attribution behind the rule is the campaign's best table. It is a 14-band histogram of node sizes at the tall cell, with the per-band cost of five forced decompositions.

It found three things. Parallelism stops paying below roughly 400 to 800 rows. The pure row-parallel form loses at every band on shared-cache hosts. The 2D form wins above roughly 16,000 rows.

The contract this shipped as, with the exact constants and the determinism consequence, is in [architecture doc 7](../../../architecture/7-parallel.md).

## The third axis: how the bins are stored
Both forms above assume the row-major mirror. Column-major storage is the other option, and LightGBM is built on it: each worker scans one feature's column into that feature's histogram.

A column scan gives every worker its own read stream, so the refetch multiplier disappears at any row width.

The cost is read traffic, the total bytes fetched from memory. Below the root a node owns a subset of rows, so a column scan reads scattered positions, and most bytes in each fetched line belong to rows the node does not own. At the tall cell that is roughly 20x bonsai's read traffic, about 30GB against 1.45GB per tree.

Which layout wins depends on the host's bandwidth. On a bandwidth-rich EPYC draw LightGBM's tall train beat bonsai's. On the local M2 bonsai trains tall 2.4x faster: 17.4 to 22.1 seconds against 42.4 to 70.7 at 4 and 8 threads.

The tempting conclusion, that bonsai should fill from columns on rich hosts too, was tested directly. bonsai stores both representations, so the campaign built a column-major fill arm (an arm is one variant in a comparison) for single nodes and a per-host selector.

On a rich EPYC draw, with three arms rotated on one machine, the forced column arm tied the forced mirror arm: tall populate 15.6 and 15.2 seconds against 14.9 and 16.2. The pre-registered falsifier required a 25% cut, so the lever was reverted.

Layout is therefore not what buys LightGBM its rich-host win. The selector itself was correct wherever it ran, at 0.7% cost: the premise failed, not the implementation.

Neither layout dominates. [Chapter C3](3-coordination-costs.md)'s ledger records what each one buys.

## Why best-first growth makes this hard

There is a third form, and it is the cheapest of the three: node-parallel, one worker per node.

The level growers expand a whole level at once, so a fill has many nodes in front of it. The work list is long and one worker usually owns whole nodes.

Both sides of the access pattern are then private: its own rows, its own nodes' arenas. Partials appear only where a worker's range straddles a node boundary, at most once per worker per level.

Best-first growth has no such list. It expands one node per round, so the frontier is a single node and there is no second node to hand anyone.

Every worker must therefore be paid out of one node's rows or one node's features. That is precisely the choice this chapter has been pricing.

That constraint is what the campaign's opening baseline priced. The leaf-wise grower trained the tall cell in 39.8 seconds against the depthwise grower's 14.4. At the wide cell it was 518.2 against 98.3 (issue #367).

Those were the same host at 12 threads. Both growers build the same complete tree to identical r2, so the trees, the arithmetic and the machine were all held fixed.

The entire gap was how the work was divided.

## What to carry forward

- The fill's arithmetic is fixed. Only the memory access pattern is a design choice.
- Feature-parallel has private writes and shared reads. Row-parallel has private reads, conflicting writes, and therefore partials.
- The partials grow with the worker count, so a row-parallel fill gets more expensive exactly when you add workers to make it cheaper.
- The refetch multiplier is `max(1, W / L)`, so feature-parallel is free at 256 lines and costs about 6x at 2 lines with 12 workers.
- The 2D form hands out feature ranges first, and adds row blocks only when the row has too few cache lines to give every worker its own.
- Column-major storage removes the multiplier and pays roughly 20x read traffic for it. Which one wins moves with the host.
- Node batching beats both forms and is unavailable to best-first growth, which is why this problem exists.
