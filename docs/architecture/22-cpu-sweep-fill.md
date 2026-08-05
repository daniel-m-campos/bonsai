# 22. Deep-level sweep fill for the CPU histogram engine

Status: design, unbuilt. This note prices a candidate lever for issue #355 (the CPU populate campaign) and asks permission to overturn one clause of decision 17. Nothing here is admitted; the probe's kill criterion is at the end.

## The evidence this note answers

The per-level populate ledger (issue #355, step 0) measured the CPU histogram fill's throughput by fill depth. The root fill runs at 8-8.5 G adds/s on server parts, which is XGBoost's ballpark; fills below the root collapse in proportion to how small the cell is: 11x slower than root at 250k rows, 3x at 2M, 2.2x at 16M. The decay tracks the published train-time gap to XGBoost exactly (3.07x behind at 250k, 1.32x at 16M). Two cheap levers were refuted by A/B in step 1: gathering (g, h) into node order moved at most 4% (the scattered 128-byte strip reads dominate the g/h stream by 16x), and the prefetch distance is not the limiter at any tested cell. What remains is the strip stream itself: a deep node's rows are an ascending sparse subset, and every row costs two latency-bound cache-line fetches the prefetcher can only partly hide.

## The shape of the fix, and what it collides with

The root fill is fast because it walks rows 0..n sequentially. The only way a deep level gets that access pattern back is to walk the full row range once for the whole level, instead of walking each node's row list separately, and route each row to its node's histogram through a map. That is the `row_to_node` shape, and decision 17 explicitly rejected it for the frontier: it "beats per-node lists on cache locality at very shallow depth, loses at typical max_depth = 6," and "the subtraction trick wires naturally onto per-node histograms" while an all-live-nodes pass "doesn't know to skip the larger sibling."

Both rejection reasons were right for what they priced, and both have since changed at the specific place this note targets:

- The rejection priced replacing per-node row lists as the partitioning structure everywhere. This note keeps decision 17's partitioning untouched: per-node lists remain the source of truth, partition_rows and the leafwise grower are unchanged. The sweep is a fill-order strategy inside `populate_many` only, and the map it needs is derived from the per-node lists each level, not maintained in their place.
- The subtraction objection is answered by construction: the sweep's map routes ONLY the smaller children's rows and holds a skip sentinel for every other row. Larger siblings derive by subtraction exactly as today; `finish_split` does not change. The "per-node branching in the kernel" the rejection feared is one table lookup and one predictable compare per row.
- The "loses at typical depth" verdict was measured without the ledger. It is true at large n, where the ledger shows per-node fills approaching root rate anyway (2.1 vs 8.0 G adds/s at 16M, and the level's smaller children cover only ~27% of rows, so a full sweep reads 3.7x more strip bytes than it uses). It is false at small and mid n, where per-node throughput collapses to 0.4-1.6 G adds/s while the level's whole strip volume fits or nearly fits L3 and a sequential sweep re-reads it at cache bandwidth.

## The design

At each level, `populate_many` chooses one of two fill strategies for the level's batch of smaller children:

- **List fill** (today's path, unchanged): per-node row lists, the plan_fill unit machinery, direct or partial-merge accumulation.
- **Sweep fill** (new): build `slot[r]` (u16, one entry per dataset row; skip sentinel default) by writing each smaller child's list index over its rows, then for each feature tile of width W, walk rows 0..n-1 sequentially; a row with a live slot accumulates its W bins into that node's histogram cells for the tile's features. Reads are fully sequential (strip, slot, g, h); the only scattered traffic is the accumulate, whose live target is bounded by construction: live_nodes x W x 256 bins x 8B. W is chosen so that target sits in L3 (W=16 at 255 nodes is ~8MB; W=8 is ~4MB). The full 128-byte strip is re-read once per tile, which is the price of bounding the target; at the cells where the sweep wins, the level's strip volume is L3-resident and re-reads are nearly free.

The strategy choice is a per-level decision made from quantities the engine already has: n_live_rows (sum of smaller-children rows), n (dataset rows), and the level's node count. The probe hard-codes the switch behind an env toggle; the admitted version derives it from the measured crossover, and the heuristic must be a pure function of (n, n_live_rows, nodes, width) so runs are reproducible.

Accumulation order inside one node changes from "its list's order" to "ascending dataset row order", which is the same order, because partition_rows is stable and the root's list is iota: every node's list is already ascending. Parallelism splits the sweep's row range into blocks with per-block partials merged in block order, the same fixed-count reproducibility contract plan_fill's multi-block nodes carry today (docs/architecture/7-parallel.md). One contract narrowing must be named honestly: deep single-block nodes are bit-identical at ANY thread count today; under a parallel sweep they become fixed-thread-count reproducible like every multi-block node already is. The alternative (single-threaded sweep per level) keeps the stronger contract and may even win at small n where parallel overhead bites; the probe measures both.

## What this costs

- `slot` map: 2 bytes x n, thread-local, rebuilt per level by sequential writes over the smaller children's lists (~n_live_rows stores). At 2M rows that is 4MB and microseconds.
- Core lines: one new fill loop plus a strategy switch in populate_many, estimate 60-90 lines in src/grower.cpp. No engine signature changes, no config surface (the admitted form has zero knobs), no model-format impact.
- The u16 slot caps live smaller children per level at 65,534, which depth <= 16 satisfies by orders of magnitude; leafwise fills one node at a time and never takes this path.
- Guide impact: a paragraph in the histogram chapter's fill section, additive.

## Expected magnitude, from the ledger's own numbers

At 250k x 128, depth-8 levels run ~0.5 G adds/s over ~830M adds per level-set: the sub-root fills cost ~4.3 of populate's 5.1s. A sweep at even half of root rate cuts that to ~1.0s, taking the 250k train gap from 3.1x to roughly 1.9x. At 2M the same arithmetic takes 1.74x to roughly 1.35x. At 16M the sweep is predicted to LOSE to the list fill (3.7x read amplification against a 2.2x latency deficit), which is why the strategy switch exists and why the probe must find the crossover rather than assume it.

## The probe and its kill criterion

Probe: `BONSAI_HIST_SWEEP=<level>` forces sweep fill at fill depths >= level (0 disables), single-threaded sweep first, tile width fixed at 16. Same A/B discipline as step 1: same pod, interleaved arms, engagement line under BONSAI_GROW_PROFILE, r2 must be EXACTLY equal (add order is unchanged per the argument above; any drift kills the probe on the spot).

Kill criterion, pre-registered: at 250k and 1M x 128, the sweep arm must cut total populate time by >= 25% with the per-level ledger showing deep-level throughput at >= 2x the list fill's; and at 16M the strategy switch (sweep off above the crossover) must hold the regression to <= 2%. Miss either and this note joins doc 17 as priced-but-declined.

Decision 17's partitioning clause is not touched in any outcome; what the probe can overturn is only its fill-order corollary, and only for deep levels on the CPU plane.
