# E6. The wide-data wall

Every case so far ran on data shaped like the benchmarks: millions of rows, around a hundred features. This case is what happened when the first production user fitted the other shape, 131 thousand rows by 16,384 features, and LightGBM's CPU beat every bonsai grower by 2 to 3x. The fix was eleven lines, because the fast path already existed; the work was finding out why the slow path was slow, and the first two theories were wrong.

## Two fills, one choice

A histogram booster spends its life doing one thing: for every row in a node, add that row's gradient and hessian into one bin of every feature's histogram. bonsai has two loops that do it, and they differ only in which array they walk sequentially.

The **row-wise fill** walks rows. For each row it reads that row's bins out of the row-major mirror, one contiguous stretch of bytes (all 16,384 features' bin ids for that row, one byte each), and scatters 16,384 add-pairs into 16,384 different histograms. The reads are perfectly sequential; the writes go everywhere.

The **feature-parallel fill** walks columns. Each thread owns whole features: it scans one feature's column of bin ids top to bottom and accumulates into that one feature's histogram, 2KB that never leaves the fastest cache. The writes are perfectly local; on a node that covers a subset of rows, the reads go everywhere.

Neither loop is better. Each streams one side of the access pattern and scatters the other, so the winner is decided by which scattered side the cache can absorb. That is the whole case, and it is why the answer changed twice as the shapes changed.

## The arithmetic of the wall

The row-wise fill's scattered side is the histogram working set: every selected feature's histogram must be writable at once, because any row touches all of them.

| features | histogram working set (255 bins x 8B) |
|--:|--:|
| 100 | 0.2 MB |
| 1,024 | 2.1 MB |
| 4,096 | 8.4 MB |
| 16,384 | 33.6 MB |

At 100 features the whole target fits in a core's L2 cache and every add costs a few cycles: this is the regime every earlier case optimized, and the row path is excellent there. At 16,384 features the target is 33.6MB per row chunk filled, and with 16 threads each writing its own partial copy, over half a gigabyte of histogram scratch is live at once. No cache holds it. Every add becomes a DRAM round trip at roughly a hundred cycles instead of four, and the machine spends its time waiting, not adding. On top of that, the partial copies that make the row path deterministic must each be zeroed before the fill and merged after it, and both passes also scale with the footprint.

That is why the effect is worth multiples and not percent: the fill is 85% of a wide fit (the stage-0 profile read populate at 84 to 88s of a 100s fit), and the wall multiplies the cost of its innermost instruction by the L2-to-DRAM latency ratio. A fit does not get 20% slower when its hottest loop's cache hit rate collapses; it changes regime.

Leafwise was the worst grower on the report, at 7x behind LightGBM, for a structural reason from decision 42's file: best-first growth fills one node at a time, so it makes the most separate passes over that oversized machinery and amortizes the zero+merge cost the least.

## The first cut, and the first refutation

The fast path for this regime was already in the file. The feature-parallel fill had been reserved for high-bin-count data by a gate on bin width; the campaign's first commit made the gate also consider shape, flipping to feature-parallel past a footprint threshold measured on an M2 laptop: the row path won 131k x 2048 (20.6s vs 24.2s) and lost 131k x 4096 (101.7s vs 44.9s), so the threshold went between them.

The same-pod validation said no. At 131k x 16384 the routing was a triumph, 1019s to 379s depthwise and 2591s to 445s leafwise, LightGBM parity at 2.7x less memory. But 1M x 4096, which the M2 threshold also flipped, regressed from 348s to 523s on the pod's EPYC host. The M2 calibration did not transfer.

The first theory blamed row count: the feature-parallel fill re-reads the node's gradient and hessian arrays once per feature, free while they sit in cache, expensive once they stream. A per-node bound followed (small nodes flip, big nodes stay row-wise), and it made the cell worse, 574s. Plausible mechanism, wrong mechanism, and only a measurement said so.

## The A/B that found the real one

Pod timing had drifted across the session, so the decisive experiment interleaved the two builds on the same cell, alternating, twice each, one worker at a time: main 320s and 337s, all-feature-parallel 547s and 547s. Not noise. (The session also produced one batch of numbers that was pure noise: a launch believed dead had stayed alive, two workers split the host, and everything measured 2x slow until the process list explained it. A harness can lie, and the only defense is checking who else is running.)

The real mechanism is the feature-parallel fill's own scattered side. Below the root, a node covers a subset of rows, so each per-feature scan reads `bins[rows[k]]` at irregular positions in a million-entry column, for 4,096 columns. The row path's scattered side at that width is only 8.4MB, and the EPYC's huge shared L3 simply absorbs it, so the row path keeps its sequential reads and pays nothing for its writes. The M2's smaller cache system cannot absorb 8.4MB, which is why the same cell flipped the other way there. One wall, two hosts, two verdicts, both correct.

## What shipped

One constant: levels whose selected-histogram footprint exceeds 24MB take the feature-parallel fill. That flips only the ultra-wide shapes that lost on every measured host, and leaves 4,096-column shapes on the row path that a big L3 wins. The narrow path is code-identical below the threshold, and the fixed-input model hash proved it byte for byte. Above it, models change bytes at identical accuracy, and gain something the row path never had: the feature-parallel fill's sums are bit-identical at any thread count.

| cell (t=16, same pod) | grower | before | after | LightGBM CPU |
|--|--|--:|--:|--:|
| 1M x 4096 | depthwise | 348s | unchanged | 382s |

The 131k x 16384 rows of the same ladder, where the routing collapsed a 7x deficit to parity, are in [the archive](../../method/results/archive.md).

Peak memory at the wide cell: 18.8GB against LightGBM's 50.1GB.

What the constant deliberately leaves on the table is recorded on issue #217: the M2's mid-width win goes untaken because a fixed threshold cannot know the host's cache (a cache-size-aware threshold is the follow-up, and it is the same lever XGBoost 3.3 shipped that year as aarch64 cache detection); the row-major mirror is still built even when no level row-fills; and the CUDA planes' recorded wide deficit, which this case never touched, still rested on 2026-07-08 numbers (the epilogue returns to it).

## The lessons

The discipline held: the price list came before the lever, and the lever was routing, not new machinery. Two theories died in one day, each killed by a same-pod measurement, and the second death took the interleaved A/B because ordinary before/after had too much drift to be trusted. The deepest one is about caches: the two fills are the same arithmetic with the scatter on opposite sides, so "which is faster" is never a property of the code. It is a property of whether the scattered side fits, and that is decided by the data's shape and the host's silicon, which is why the honest answer shipped as a threshold with a recorded residual instead of a winner.

## Epilogue: the wall dissolves

The threshold lived one day. Its recorded residual, XGBoost 3.3's column-tiling lever, was probed the next morning: store the mirror in 2048-feature blocks, each row-major on its own, and run the fill tiles outer, rows inner. Now the scattered side always fits, because the live target is one block's histograms at any width, and the sequential side stays sequential inside each block. The choice between the two fills stopped existing, which is different from being made well.

The probe's interleaved A/B beat both strategies at their own best cells: 326s against the row path's 369 at 1M x 4096, 442s against feature-parallel's 514 at 131k x 16384, a wash at 16M x 100. And because tiling never changes a feature's accumulation order, the tiled fill produces bit-identical models to the classic row path at every width, so the determinism contract came out stronger than either strategy could make it. The chapter's closing lesson gains its final form: when a trade-off is real, ship the honest threshold and record what it leaves on the table, and then look for the restructuring that makes the trade-off imaginary. It existed here, one layout change away, published in a competitor's release notes all along.

The CUDA wide wall this case left as a residual got the same treatment on 2026-07-30, and it had already fallen: the recorded deficit dated to 2026-07-08 code, and on current main the CUDA growers lead every wide cell (refutation-by-progress, decision 90). The iso-volume frontier (decision 91) then measured the shape axis wholesale, holding rows x cols constant while width swept 128 to 65536 columns; both are in [the archive](../../method/results/archive.md).

Evidence: [benchmarks/wide-cpu-hist-2026-07.md](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/wide-cpu-hist-2026-07.md); decisions 88 to 91; the ratifying field report is issue #217.
