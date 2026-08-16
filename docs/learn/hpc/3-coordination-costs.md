# H3. Coordination costs

The bottom line first. Starting parallel work costs about 35 microseconds. That is nothing against a large job and everything against a small one.

Best-first growth makes small jobs by construction. So this chapter is about the fixed fee, and the four ways the campaign tried to stop paying it.

## What a region costs

A parallel region is one entry into the runtime. Hand a work list to a team of workers, wait for all of them to finish, continue.

bonsai has exactly one such primitive, and every parallel site in the engine goes through it. [Architecture doc 7](../../architecture/7-parallel.md) owns its contract.

Round four priced the entry directly: about 35 microseconds at 12 threads on the campaign's host class.

The same measurement killed the folk remedy that always gets proposed here. The remedy is a persistent region. Open one region for the whole tree, and synchronize inside it with a barrier, instead of paying entry per split.

It buys nothing. An in-region barrier on this runtime costs the same 35 microseconds as a fork and join.

The fee is the synchronization, not the region, and moving the boundary does not move the fee. That refutation is worth more than the lever would have been, because it closes a whole family of designs.

The second thing round four found is easier to act on. The phases on either side of the fill were separate regions of their own.

Those phases are starting the arena cells and subtracting the finished child from its sibling. They were paying 167 and 135 microseconds of entry each, to do less work than the entry cost.

They were fused into the fill's own work list. One worker now starts its feature range, fills it, and subtracts it before it leaves.

Regions per split fell from 4.16 to 2.16, which is 51,000 fewer entries across a fit.

## When the fee dominates, and when it disappears

Here is the count that makes the tall cell different. The leaf plane enters roughly 25,500 fill regions per fit against the level plane's 800.

The reason is structural. Best-first growth splits one node per round, and a 100-round fit at a 256-leaf budget makes 25,500 splits.

**The arithmetic.** 51,000 removed entries times 35 microseconds predicts 1.79 seconds off the fit.

**The measurement.** Tall populate fell from 24.73 to 22.95 seconds, which is 1.78 seconds.

That is as clean as attribution gets. The round still recorded it as a falsification.

The pre-registered bar was an 8% drop and the lever delivered 7.2%. By the round's own rule, synchronization count is not the dominant tall term.

A lever can be perfectly understood and still be the wrong lever. Keeping both facts is what makes the ledger at the end of this chapter trustworthy.

Now do the same division at each shape, using measured quantities.

| | cpu-tall | cpu-wide |
|---|--:|--:|
| splits per fit | 25,500 | 25,500 |
| regions per split, after round four | 2.16 | 2.16 |
| entry cost per split, at 35us each | ~76us | ~76us |
| work per fill | ~0.9ms | 7.5ms |
| entry as a share of the fill | ~8% | ~1% |

The counts are the same. The work is not.

A wide fill starts a 32MiB arena and reads 16,384 bytes of every row. A tall fill starts 256KiB and reads 128.

So the same 35 microseconds is a term worth chasing at tall and a rounding error at wide. The campaign's rounds moved between the two shapes accordingly.

The last two rows are derived from the campaign's measured seconds rather than measured directly. Tall's figure divides a 22.95-second populate by the split count, and wide's is the round-five reading of one full-team fill. Treat both as an order of magnitude.

## Sizing the team

If the fee is per region and per worker, the first defense is to stop asking for workers you cannot use.

**The floor.** A fill's work is the arena it starts plus the accumulates it makes, `n_sel * 256 + n_rows * n_sel` cells. It takes one worker per 32,768 of those cells, and never more than the configured count.

A team of one runs the loop inline and enters no region at all. The smallest nodes therefore pay nothing, which is the cheapest fix in this chapter.

Counting the arena is what keeps a wide node from looking small. A node holding a handful of rows still has a full arena to start, and at 16,384 features that is 32MiB of zeroing.

**The floor's own measurement.** Parallelism stops paying below roughly 400 to 800 rows. That comes from the 14-band node-size attribution [chapter H2](2-cutting-the-work.md) introduced.

Round five got the floor wrong in its first form, and the wide cell caught it. The work count included the accumulates but not the arena start.

So a wide node with few rows scored as tiny work, and zeroed its 32MiB arena on a team of one. Measured: 84.9 milliseconds per such fill against 7.5 on the full team.

That is an 11x self-inflicted penalty, on exactly the shape the round was not trying to change. The fix counts the start, and a full-team single-block plan then executes the previous round's function unchanged.

## The cost nobody prices: team-size churn

The floor's first form had a second defect, and it is the more surprising one.

Asking the runtime for a different team size is not free. OpenMP keeps a pool of workers hot between regions.

A region of a new size makes it rebuild the team that the regions around it were reusing. A rule that computes an exact size per node therefore requests a new size constantly.

Round five's floor asked for 8 distinct team sizes, at 129 switches per 431 fills. Round six then reproduced the same failure from the other side, and its own control caught it.

That round sized a partition's team to the parent node, which is textbook. Give every worker at least 4,096 rows, and use the largest team that satisfies it.

Partition time fell 22%, from 4.2 to 3.2 seconds. Populate rose 7%, absorbed the whole gain, and train came out flat.

The cause was churn. The rule requested a new team size on essentially every split, 265 switches per tree against the baseline's 78. The fills on either side paid for every rebuild.

The amendment is a construction rather than a tuning. The partition team is two-valued: the full team when every worker gets 4,096 rows, and one worker otherwise.

It takes four blocks per worker rather than an equal cut, so the dynamic schedule can still absorb asymmetric cores. Switch count returns exactly to the baseline's 78, so the amended rule cannot pay a churn cost the baseline does not.

The methodology note is worth as much as the fix. The regression was visible only because a byte-identical control arm ran beside the change on the same pod.

It also needed the arms rotated rather than merely interleaved. With arms in fixed order, a run's thermal drift lands entirely on the last arm.

## Choosing an arm at runtime, honestly

Once two arms exist and neither wins everywhere, something has to choose. There are two honest forms and one common failure.

**The priced gate.** Pick a constant, and only where its reason is structural rather than host-specific. bonsai's fill has four such constants, each with the argument that makes it a constant, in [architecture doc 7](../../architecture/7-parallel.md).

A constant tuned on one host is a claim about that host. That is how the engine track's [wide-data case](../engine/6-the-wide-data-wall.md) lost a threshold one day after shipping it.

**The deferred switch.** Measure both arms on real work, then commit for the rest of the fit. The discipline is in the word real.

The probe has to run on nodes that represent the fit, at enough size to resolve the difference, and after the caches are warm. A probe that violates any of the three measures the probe.

**The failure mode, measured.** LightGBM chooses between its own two fills by calibrating on one histogram, the root's. That is the single node where the column layout is always best.

On the local M2 at the tall cell, its row-major path beats its own column path by 1.9 to 2.1x. Its selector ships the losing arm.

That is not a hypothetical cost. On that host it is most of why bonsai trains the tall cell 2.4x faster than LightGBM, on identical data and knobs with matching r2.

bonsai's own attempt at the deferred switch is the third ledger entry below, and it was reverted. The selector was correct everywhere it ran, at 0.7% cost, and its pre-registered property caught a cold-rep bias in its own first draft.

What failed was the premise that the two arms differ on a rich host, which [chapter H2](2-cutting-the-work.md) tells.

## The trade-off ledger

The campaign closed with four entries. Each is a place where two designs are coherent and neither dominates, so the output is a named mechanism rather than a winner.

| trade-off | the mechanism | the number |
|---|---|---|
| partition floor against frontier batching | a partition's team is capped by the one node being split, and best-first forbids batching the frontier | 510 partition region entries per tree against the level plane's 16 |
| read traffic against line sharing | column storage buys disjoint per-worker streams; the mirror buys locality and shares lines | roughly 20x read traffic, about 30GB against 1.45GB per tree |
| pruning against contiguous selection | dropping dead features is work subtraction for a column fill, and forfeits the mirror fill's arithmetic address | +54% wide and +4% tall when built, against 0.5s of fill volume saved |
| region count | retired as a differentiator: the reference engine pays more entries per split and still wins some draws | 4 to 5 regions per split against bonsai's 2.16 |

The third entry is the one worth reading twice. The pruning lever was built completely and then measured.

At the wide cell, 23 to 27% of features stop producing any positive-gain split by depth 7, so the mechanism fires. It still cost 18.3 seconds at four threads.

The cost decomposes cleanly. 13.7 seconds was the arena becoming a unique size per node, so the block pool never hits. 1.7 was losing the arithmetic address that contiguous selection provides, and 2.6 was the census itself.

At the tall cell it barely fires at all, 3%. Two million rows over 256 leaves keeps every node near 8,000 rows, which is too shallow for features to die.

Quality was never at risk either way. With pruning on, single-thread models came out byte-identical, so every pruned feature was one that never won a split.

So bonsai's fill is fast because a mirror tile's selected features are contiguous. One address computation per add, and one arena size per tree.

Any per-node sub-selection forfeits both. That is why the same lever is nearly free for a column-major engine and profitable never for this one.

## What to carry forward

- Entry to a parallel region costs about 35 microseconds at 12 threads, and an in-region barrier costs the same.
- Persistent-region designs therefore buy nothing. The fee is synchronization, not the region boundary.
- Region count matters where work per region is small: about 8% of a tall fill and about 1% of a wide one.
- The cheapest team is a team of one, which enters no region. Size teams from a work count that includes the setup.
- Asking for a new team size is itself a cost, and a rule that computes an exact size per node pays it constantly.
- A selector calibrated on one unrepresentative node is worse than no selector. LightGBM's root-only calibration is the worked example.
- Where two coherent designs cross, ship the honest statement with its mechanism named, and put the number in the ledger.
