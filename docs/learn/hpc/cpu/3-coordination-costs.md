# C3. Coordination costs

The bottom line first. Coordinating parallel work charges fixed fees, about 35 microseconds per synchronization event, while only the fill itself is variable work. Best-first growth makes small fills by construction, so the fees show.

This chapter names the costs, sorts them into fixed and variable, and derives the team-sizing rule from the sorted list. The refutations that set each constant come with it.

## The cost model

A parallel region is one fork/join episode: a team of worker threads starts, each worker runs its share of the loop, and everyone waits at the join before the program continues.

bonsai has exactly one such primitive, and every parallel site in the engine goes through it. See [architecture doc 7](../../../architecture/7-parallel.md) for details.

Five costs cover everything the fill's coordination can pay. Each has a driver: the event or quantity that incurs it.

| cost | driver | size |
|---|---|---|
| region entry | one synchronization event, fork/join and in-region barrier alike | ~35us at 12 threads |
| team rebuild | a region requesting a different team size than the region before it | absorbed a 22% partition gain when triggered per split |
| partial arenas | each row block past the first ([chapter C2](2-dividing-the-work.md)'s machinery) | `n_sel * 256` cells zeroed and reduced per extra block |
| arena zeroing | each fill, however few rows the node holds | `n_sel * 256` cells: 256KiB tall, 32MiB wide |
| accumulates | each (row, selected feature) pair in the fill | `n_rows * n_sel` cells |

The first three are fixed: they price coordination events, and their unit cost does not shrink when the node does. The last two are variable: they are the work itself.

Two measurements pin down the entry row, and the first closes a whole family of designs.

The fee attaches to the synchronization event, not to the region boundary. An in-region barrier costs the same 35 microseconds as a fork and join. So the folk remedy, one persistent region held open for the whole tree with a barrier per split, saves nothing: it renames the events without changing their count.

Fusing phases does change the count. Arena zeroing and the sibling subtraction ran as regions of their own, paying 167 and 135 microseconds of entry each to do less work than the entry cost.

Folded into the fill's own work list, one worker now zeroes its feature range, fills it, and subtracts it before it leaves. Regions per split fell from 4.16 to 2.16, which is 51,000 fewer events across a fit.

## When the fee dominates, and when it disappears

Here is the count that makes the tall cell different. The leaf plane enters roughly 25,500 fill regions per fit against the level plane's 800.

The reason is structural. Best-first growth splits one node per round, and a 100-round fit at a 256-leaf budget makes 25,500 splits.

**The arithmetic.** 51,000 removed entries times 35 microseconds predicts 1.79 seconds off the fit.

**The measurement.** Tall populate fell from 24.73 to 22.95 seconds, which is 1.78 seconds.

That is as clean as attribution gets. The campaign still recorded it as a falsification.

The pre-registered bar was an 8% drop and the lever delivered 7.2%. By its own rule, synchronization count is not the dominant tall term.

A lever can be perfectly understood and still be the wrong lever. Keeping both facts is what makes the ledger at the end of this chapter trustworthy.

Now do the same division at each shape, using measured quantities.

| | cpu-tall | cpu-wide |
|---|--:|--:|
| splits per fit | 25,500 | 25,500 |
| regions per split, after the fusion above | 2.16 | 2.16 |
| entry cost per split, at 35us each | ~76us | ~76us |
| work per fill | ~0.9ms | 7.5ms |
| entry as a share of the fill | ~8% | ~1% |

The counts are the same. The work is not.

A wide fill zeroes a 32MiB arena and reads 16,384 bytes of every row. A tall fill zeroes 256KiB and reads 128.

So the same 35 microseconds is a term worth chasing at tall and a rounding error at wide. The campaign moved between the two shapes accordingly.

The last two rows are derived from the campaign's measured seconds rather than measured directly. Tall's figure divides a 22.95-second populate by the split count, and wide's is a measured reading of one full-team fill. Treat both as an order of magnitude.

## The decision rule

The sorted costs derive the sizing rule in three clauses, each set by a measured number.

**Count the variable work in cells, all of it.** A fill's work is `work_cells = n_sel * 256 + n_rows * n_sel`, the zeroing plus the accumulates.

Counting the zeroing is what keeps a wide node from looking small: a node holding a handful of rows still zeroes a 32MiB arena. The rule's first form counted only the accumulates, so exactly those nodes ran the 32MiB zero on one worker, at 84.9 milliseconds per fill against the full team's 7.5. An 11x penalty, self-inflicted on the shape the change was not trying to touch.

**Take one worker per 32,768 cells, never more than configured.** The constant is the amortization threshold from the 14-band attribution [chapter C2](2-dividing-the-work.md) introduced: parallelism stops paying below roughly 400 to 800 rows.

A team of one runs the loop inline and enters no region at all. The smallest nodes pay no fee, which is the cheapest fix in this chapter.

**Allow two team sizes: full or one.** An exact size per node is the textbook refinement, and it pays the rebuild fee constantly. OpenMP keeps a pool of workers hot between regions, and a region of a new size makes it rebuild the team the regions around it were reusing.

The exact-size floor requested 8 distinct sizes, at 129 switches per 431 fills. Sizing partition teams to their parent node reproduced the same failure from the other side: partition time fell 22%, from 4.2 to 3.2 seconds, but the rule switched sizes on essentially every split, 265 per tree against the baseline's 78. The fills on either side paid for every rebuild: populate rose 7%, absorbed the whole gain, and train came out flat.

The two-valued team, full when every worker gets 4,096 rows and one worker otherwise, returns the switch count exactly to the baseline's 78. A rule at the baseline's switch count cannot pay a churn cost the baseline does not. It hands each worker four blocks rather than one equal share, so the dynamic schedule still absorbs asymmetric cores.

The methodology note is worth as much as the rule. The churn regression was visible only because a byte-identical control arm ran beside the change on the same machine, with the arms rotated rather than merely interleaved. In a fixed order, a run's thermal drift lands entirely on the last arm.

## Choosing an arm at runtime

Once two arms exist and neither wins everywhere, something has to choose. There are two honest forms and one common failure.

**The priced gate.** Pick a constant, and only where its reason is structural rather than host-specific. bonsai's fill has four such constants, each with the argument that makes it a constant, in [architecture doc 7](../../../architecture/7-parallel.md).

A constant tuned on one host is a claim about that host. That is how the engine track's [wide-data case](../gpu/6-the-wide-data-wall.md) lost a threshold one day after shipping it.

**The deferred switch.** Measure both arms on real work, then commit for the rest of the fit. The discipline is in the word real.

The probe has to run on nodes that represent the fit, at enough size to resolve the difference, and after the caches are warm. Violate any of the three and the probe measures itself, not the fit.

**The failure mode, measured.** LightGBM chooses between its own two fills by calibrating on one histogram, the root's. That is the single node where the column layout is always best.

On the local M2 at the tall cell, its row-major path beats its own column path by 1.9 to 2.1x. Its selector ships the losing arm.

That is not a hypothetical cost. On that host it is most of why bonsai trains the tall cell 2.4x faster than LightGBM, on identical data and knobs with matching r2.

bonsai's own attempt at the deferred switch is the third ledger entry below, and it was reverted. The selector was correct everywhere it ran, at 0.7% cost, and its pre-registered property caught a cold-rep bias in its own first draft.

What failed was the premise that the two arms differ on a rich host: [chapter C2](2-dividing-the-work.md)'s rotated-arm test measured them tied.

## The trade-off ledger

The campaign closed with four entries. Each is a place where two designs are coherent and neither dominates, so the output is a named mechanism rather than a winner.

| trade-off | the mechanism | the number |
|---|---|---|
| partition floor against frontier batching | a partition's team is capped by the one node being split, and best-first forbids batching the frontier | 510 partition region entries per tree against the level plane's 16 |
| read traffic against line sharing | column storage buys disjoint per-worker streams; the mirror buys locality and shares lines | roughly 20x read traffic, about 30GB against 1.45GB per tree |
| pruning against contiguous selection | dropping dead features is work subtraction for a column fill, and forfeits the mirror fill's arithmetic address | +54% wide and +4% tall when built, against 0.5s of fill time saved |
| region count | retired as a differentiator: the reference engine pays more entries per split and still wins some draws | 4 to 5 regions per split against bonsai's 2.16 |

The third entry is the one worth reading twice. The pruning lever was built completely and then measured.

At the wide cell, 23 to 27% of features stop producing any positive-gain split by depth 7, so the mechanism fires. It still cost 18.3 seconds at four threads.

The cost decomposes cleanly. 13.7 seconds was the arena becoming a unique size per node, so the block pool never hits. 1.7 was losing the arithmetic address that contiguous selection provides, and 2.6 was the census itself.

At the tall cell it barely fires at all, 3%. Two million rows over 256 leaves keeps every node near 8,000 rows, which is too shallow for features to die.

Quality was never at risk either way. With pruning on, single-thread models came out byte-identical, so every pruned feature was one that never won a split.

So bonsai's fill is fast because a mirror tile's selected features are contiguous. One address computation per add, and one arena size per tree.

Any per-node sub-selection forfeits both. That is why the same lever is nearly free for a column-major engine and never profitable for this one.

## What to carry forward

- Entry to a parallel region costs about 35 microseconds at 12 threads, and an in-region barrier costs the same.
- Persistent-region designs therefore save nothing. The fee is synchronization, not the region boundary.
- Region count matters where work per region is small: about 8% of a tall fill and about 1% of a wide one.
- The cheapest team is a team of one, which enters no region. Size teams from a work count that includes the zeroing.
- Asking for a new team size is itself a cost, and a rule that computes an exact size per node pays it constantly.
- A selector calibrated on one unrepresentative node is worse than no selector. LightGBM's root-only calibration is the worked example.
- Where two coherent designs cross, ship the honest statement with its mechanism named, and put the number in the ledger.
