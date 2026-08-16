# H2. Cutting the work

The bottom line first. The histogram fill can be cut two ways. Each one streams one side of memory and scatters the other, and the winner is whichever scattered side fits.

Chapter H1 gave you the arithmetic. This chapter spends it on the real loop.

## The job

To split a node, a booster needs one histogram per feature. Each histogram holds, for every bin, the sum of the gradients and hessians of the node's rows in that bin.

[Guide chapter 2](../../guide/2-binning-and-histograms.md) owns that definition and the subtraction trick that halves the work. The loop itself is three lines of intent:

> for each row in the node, for each selected feature: add that row's gradient and hessian into the cell at (feature, bin).

**Formula.** Adds per fill equals node rows times selected features.

**Example.** A root fill at either running example makes 2^28 adds, about 268 million.

There is no arithmetic to remove here. The body is two additions of a float pair, and the bin id is one byte already in memory.

No ordering of the loop changes the answer either. So everything below is about where the bytes are, not about what is computed.

Two arrays are in play, and they pull in opposite directions:

- **The mirror.** The bins, stored row by row, so one row's bins are contiguous bytes. [Guide chapter 2](../../guide/2-binning-and-histograms.md) owns this definition.
- **The arena.** The node's histograms, stored feature by feature, so one feature's 256 cells are contiguous.

A row walk is sequential in the mirror and scattered in the arena. A feature walk is the reverse. There is no third option, because the two arrays cannot both be walked in order.

## The two cuts, and who pays

Parallelizing means handing each worker a piece. Within one node there are exactly two pieces to hand out.

**The feature cut.** Each worker owns a contiguous range of features and reads every row. Its writes go only to its own features' arena runs, so no two workers ever write the same cell. Its reads cover the whole mirror, so every worker reads every row.

**The row cut.** Each worker owns a range of rows and touches every feature. Its reads are its own rows, so no two workers read the same mirror bytes. Its writes go to every feature, so workers collide on cells and cannot write the arena directly.

| | feature cut | row cut |
|---|---|---|
| reads | shared: every worker reads every row | private: each worker reads its own rows |
| writes | private: each worker owns its features' cells | conflicting: every worker writes every feature |
| extra machinery | none | one partial histogram per extra worker, plus a reduce |
| determinism | the serial order at any worker count | the order depends on the partition, so on the worker count |

The row cut's extra machinery is the expensive half. The cost is not the addition, it is the memory.

**Formula.** Partial cells equals workers touching a node, minus one, times selected features times 256.

Each of those cells is zeroed before the fill and read again during the reduce. Neither pass does any useful arithmetic.

**Example.** At the tall cell with 12 threads, the leaf plane's partial volume measured 24x the depthwise plane's. That is 118 million cells per two boosting rounds (round two).

The count also grows with the thread count: 11 partial sets per node at 12 threads against 3 at 4. So the row cut's price rises exactly when you add workers to make it cheaper.

## The wide collapse and the tall refusal

Round three cut the fill by feature. Each worker takes a contiguous feature range of one mirror tile and accumulates straight into the arena. Zero partials, zero partial zeroing, zero reduce.

The wide cell collapsed. Leaf-plane populate fell from 172.1 to 45.5 seconds on the same pod, below the depthwise plane's own 59 to 61 seconds at that cell.

The round's pre-registered prediction had been 70 to 110 seconds, so the lever beat its own bar.

The tall cell refused to move. Populate read 27.8 seconds before and 30.7 after, on a host whose hash-identical control arms differed 9%, against a prediction of 13 to 17 seconds.

The falsifier fired. Partial volume was not the tall term, and by elimination the mechanism was somewhere else.

Chapter H1 already told you where. Cutting a 128-byte row among 12 workers gives each about 11 features, so all 12 refetch the same 2 lines.

Cutting a 16,384-byte row gives each worker its own ~21 lines, and the overlap at the boundaries is negligible.

So the feature cut is a bandwidth trade. It removes the partials and adds a refetch multiplier of about min(workers, lines per row).

At 256 lines that trade is free. At 2 lines it is not, and the fill's memory traffic multiplies by roughly the thread count.

Round four confirmed the same shape a second way. It removed synchronization instead of memory traffic and watched tall barely move, which [chapter H3](3-coordination-costs.md) tells.

## The 2D cut that serves both shapes

Round five is the answer, and it reads as one sentence. Cut by feature, then give row blocks the workers the row's cache lines cannot make private.

**The rule.** Cut the selection into contiguous feature ranges. While the mirror row is too narrow to give every worker its own 64-byte line, split the node's rows into blocks as well. Run the cross product of (range, block) on a team of exactly that size.

**The price.** Every block past the first owns one partial arena, reduced in ascending block order. A plan with more than one block therefore pays the row cut's machinery again.

Blocks are taken only while each one still owns at least 8,192 rows. That is the work a block must do to earn the partial it is charged.

**The result at tall.** Populate fell from 33.5 to 21.7 seconds and train from 45.4 to 32.9 on one EPYC draw. The round's falsifier bar was a 10% drop, and the lever cleared it three times over.

Mirror traffic per tree fell from 3.85GB to 1.45GB in the same change.

**The result at wide.** Unchanged, as predicted, once a self-inflicted regression was found and reverted. [Chapter H3](3-coordination-costs.md) tells that part, because its cause was team sizing rather than the cut.

The attribution behind the rule is the campaign's best table. It is a 14-band histogram of node sizes at the tall cell, with the per-band cost of five forced decompositions.

It found three things. Parallelism stops paying below roughly 400 to 800 rows. The pure row cut loses at every band on shared-cache hosts. The 2D cut wins above roughly 16,000 rows.

The contract this shipped as, with the exact constants and the determinism consequence, is in [architecture doc 7](../../architecture/7-parallel.md).

## The third axis: how the bins are stored

Both cuts above assume the row-major mirror. That is a choice, and the other choice is a real engine's answer.

LightGBM stores bins column-major and fills by column. Each worker scans one feature's column top to bottom into that feature's histogram.

Its per-worker read streams are perfectly disjoint, because two workers never touch the same column. No refetch multiplier exists at any width.

It pays for that in volume. Below the root a node holds a subset of rows, so a column scan reads its column at irregular positions.

The lines it fetches then carry mostly rows the node does not own. Measured against bonsai at the tall cell, that is roughly 20x the read traffic, about 30GB against 1.45GB per tree.

Which side of the trade wins is a host property, and the campaign measured both signs. On a bandwidth-rich EPYC draw the purchase is cheap and LightGBM's tall train beat bonsai's.

On the local M2 it inverts, and bonsai trains tall 2.4x faster: 17.4 to 22.1 seconds against 42.4 to 70.7 at 4 and 8 threads.

The obvious inference from that is wrong, and round eight is how we know. bonsai stores both representations, so the campaign built a column-major fill arm for single nodes and calibrated the choice per host.

The test was the best one available: a rich EPYC draw, LightGBM at its fastest, three arms rotated on one pod. The forced column arm tied the forced mirror arm, at tall populate 15.6 and 15.2 seconds against 14.9 and 16.2.

The pre-registered falsifier required the column arm to cut populate by 25% on that draw. It did not, so layout is not what buys LightGBM its rich-draw win, and the lever was reverted.

The selector itself was correct wherever it ran, at 0.7% cost. That is what makes this a premise falsification rather than an implementation failure.

So the storage axis is two coherent packages, not a better and a worse one. [Chapter H3](3-coordination-costs.md) closes the ledger with what each package actually buys.

## Why this was a campaign at all

There is a third cut, and it is the cheapest of the three. Cut by node.

The level growers expand a whole level at once, so a fill has many nodes in front of it. The work list is long and one worker usually owns whole nodes.

Both sides of the access pattern are then private: its own rows, its own nodes' arenas. Partials appear only where a worker's range straddles a node boundary, at most once per worker per level.

Best-first growth has no such list. It expands one node per round, so the frontier is a single node and there is no second node to hand anyone.

Every worker must therefore be paid out of one node's rows or one node's features. That is precisely the choice this chapter has been pricing.

That is the structural reason the campaign existed. At the published baseline the leaf-wise grower trained the tall cell in 39.8 seconds against the depthwise grower's 14.4. At the wide cell it was 518.2 against 98.3 (issue #367).

Those were the same host at 12 threads. Both growers build the same complete tree to identical r2, so the trees, the arithmetic and the machine were all held fixed.

The entire gap was how the work was cut.

## What to carry forward

- The fill's arithmetic is fixed. Only the memory access pattern is a design choice.
- The feature cut has private writes and shared reads. The row cut has private reads, conflicting writes, and therefore partials.
- Partial volume grows with the worker count, which is why the row cut resists the obvious fix.
- The refetch multiplier is about min(workers, lines per row), so the feature cut is free at 256 lines and costs the worker count at 2.
- The 2D cut takes feature ranges first, and buys row blocks only with the workers the lines cannot make private.
- Column-major storage removes the multiplier and pays roughly 20x read traffic for it. Which one wins moves with the host.
- Node batching beats both cuts and is unavailable to best-first growth, which is why this problem exists.
