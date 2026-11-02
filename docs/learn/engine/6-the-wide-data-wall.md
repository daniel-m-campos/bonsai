# E6. The wide-data wall

Every case so far ran on data shaped like the benchmarks: millions of rows, around a hundred features. This case is what happened when the first production user fitted the other shape, 131 thousand rows by 16,384 features, and LightGBM's CPU beat every bonsai grower by 2 to 3x. The fix was eleven lines, because the fast path already existed; the work was finding out why the slow path was slow, and the first two theories were wrong.

## Two fills, one choice

A histogram booster spends its life doing one thing: for every row in a node, add that row's gradient and hessian into one bin of every feature's histogram. bonsai has two loops that do it, and they differ only in which array they walk sequentially.

The **row-wise fill** walks rows. For each row it reads that row's bins out of the row-major mirror (a second copy of the binned data, stored row by row), and scatters 16,384 add-pairs into 16,384 different histograms. At the time of this case the mirror was untiled, so a row's bins were one contiguous stretch of bytes: all 16,384 features' bin ids, one byte each. The epilogue covers the tiled layout on main. The reads are perfectly sequential; the writes go everywhere.

The **feature-parallel fill** walks columns. Each thread owns whole features: it scans one feature's column of bin ids top to bottom and accumulates into that one feature's histogram, 2KB that never leaves the fastest cache. The writes are perfectly local; on a node that covers a subset of rows, the reads go everywhere.

Neither loop is better. Each streams one side of the access pattern and scatters the other, so the winner is decided by which scattered side the cache can absorb. That is the whole case, and it is why the answer changed twice as the shapes changed.

## The arithmetic of the wall

The row-wise fill's scattered side is the histogram working set: every selected feature's histogram must be writable at once, because any row touches all of them. The wide-data wall is the width at which that working set no longer fits in cache.

| features | histogram working set (256 cells x 8B) |
|--:|--:|
| 100 | 0.2 MB |
| 1,024 | 2.1 MB |
| 4,096 | 8.4 MB |
| 16,384 | 33.6 MB |

At 100 features the whole target fits in a core's L2 cache and every add costs a few cycles: this is the shape every earlier case optimized, and the row path is excellent there. At 16,384 features the target is 33.6MB per row chunk filled, and with 16 threads each writing its own partial copy, over half a gigabyte of histogram scratch is live at once. No cache holds it. Every add becomes a DRAM round trip at roughly a hundred cycles instead of four, and the machine spends its time waiting, not adding. On top of that, the partial copies that make the row path deterministic must each be zeroed before the fill and merged after it, and both passes also scale with the footprint.

That is why the effect is worth multiples and not percent. The fill takes about 85% of a wide fit: in the first profile (131k x 4096 on an M2 with 8 threads), 84 to 88s of a fit of about 100s ([benchmarks/wide-cpu-hist-2026-07.md](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/wide-cpu-hist-2026-07.md)). The code calls this step populate. Past the wall each add costs a DRAM access instead of an L2 access (the 100-versus-4-cycle ratio above). A fit does not get 20% slower when the fill's cache hit rate collapses; it gets several times slower (1019s against 379s at 131k x 16384).

Leafwise growth (split the single best leaf next) was the slowest bonsai grower here, 7x slower than LightGBM before the fix (2591s vs 367s, decision 88). It fills one node at a time, so it runs the most fill passes and pays the zero and merge cost on every one.

## The first cut, and the first refutation

The fast path for this regime was already in the file. The feature-parallel fill had been reserved for high-bin-count data by a gate on bin width; the campaign's first commit made the gate also consider shape, flipping to feature-parallel past a footprint threshold measured on an M2 laptop: the row path won 131k x 2048 (20.6s vs 24.2s) and lost 131k x 4096 (101.7s vs 44.9s), so the threshold went between them.

The same-pod validation said no. At 131k x 16384 the routing cut depthwise from 1019s to 379s and leafwise from 2591s to 445s. Depthwise came within 4% of LightGBM CPU (366.9s) and leafwise stayed 1.2x behind it, at 2.7x less peak memory (18.8GB vs 50.1GB). But 1M x 4096, which the M2 threshold also flipped, regressed from 348s to 523s on the pod's EPYC host. The M2 calibration did not transfer.

The first theory blamed row count: the feature-parallel fill re-reads the node's gradient and hessian arrays once per feature, free while they sit in cache, expensive once they stream. A per-node bound followed (small nodes flip, big nodes stay row-wise), and it made that cell (one rows x features shape on one host, here 1M x 4096) slower still: 574s, against 348s for the row path and 523s for the M2 threshold. The mechanism sounded right and was wrong; only the measurement showed it.

## The A/B that found the real one

Pod timing had drifted across the session, so the decisive experiment interleaved the two builds on the same cell, alternating, twice each, one worker at a time: main 320s and 337s, all-feature-parallel 547s and 547s. Not noise. (The session also produced one batch of numbers that was pure noise: a launch believed dead had stayed alive, two workers split the host, and everything measured 2x slow until the process list explained it. A harness can lie, and the only defense is checking who else is running.)

The real mechanism is the feature-parallel fill's own scattered side. Below the root, a node covers a subset of rows, so each per-feature scan reads `bins[rows[k]]` at irregular positions in a million-entry column, for 4,096 columns. The row path's scattered side at that width is only 8.4MB. The EPYC's shared L3 is 32MiB per 8-core complex ([the cache chapter](../cpu/1-cache-awareness.md)), so the 8.4MB stays in cache: the row path keeps its sequential reads and its writes stay cheap. The M2's last level is an 8MiB system cache shared by everything, no larger than the working set itself, so the feature-parallel fill was faster there. The same cell has opposite best fills on the two hosts.

## What shipped

The constant no longer exists: the epilogue below retired it the next day, and there is no footprint threshold in the code today. Main routes each node by the density cutoff of chapter 9 in the guide: u16 bins always take the feature-parallel fill, and u8 bins take it whenever the node holds at least a quarter of the rows, at any width, while sparser u8 nodes take the tiled row fill. The constant sent levels whose selected-histogram footprint exceeded 24MB to the feature-parallel fill. The narrow path was code-identical below the threshold, and the fixed-input model hash proved it byte for byte. Above it, models changed bytes at identical accuracy, and gained something the row path never had: the feature-parallel fill's sums are bit-identical at any thread count.

| cell (t=16, same pod) | grower | before | after | LightGBM CPU |
|--|--|--:|--:|--:|
| 1M x 4096 | depthwise | 348s | unchanged | 382s |

The 131k x 16384 rows of the same ladder, where the routing cut a 7x leafwise deficit to 1.2x and brought depthwise to parity, are in [the archive](../../results/results/archive.md). That parity is a 2026-07 reading; the epilogue closes with where wide CPU stood on 2026-08-08.

Peak memory at 131k x 16384: 18.8GB against LightGBM's 50.1GB.

The constant deliberately left three limits, recorded on issue #217:

- It could not see the host's cache size, so it missed the M2's mid-width win. A cache-size-aware threshold was the follow-up, the same lever XGBoost 3.3 shipped that year as aarch64 cache detection.
- bonsai built the row-major mirror even when no level used the row fill.
- The CUDA planes' recorded wide deficit was untouched by this case and still rested on 2026-07-08 numbers (the epilogue returns to it).

## The lessons

Three lessons. First, measure what each candidate fix costs before writing code: the fix here was a routing change in one function (decision 88), not new code. Second, two theories were refuted in one day, each by a same-pod measurement. The second refutation needed an interleaved A/B, because ordinary before-and-after runs drifted too much to trust. Third, neither fill is faster in general. Both do the same arithmetic and scatter writes or reads on opposite sides. Which one wins depends on whether the scattered side fits in cache, and that depends on the data's shape and the host's cache. So the campaign shipped a threshold and recorded what the threshold leaves unfixed, not a single winner.

## Epilogue: the wall dissolves

The threshold lived one day. Its recorded residual, XGBoost 3.3's column-tiling lever, was tried the next morning and shipped that day (decision 89): store the mirror in 2048-feature blocks, each row-major on its own, and run the fill tiles outer, rows inner. That is the layout on main, `Dataset::mirror_tile_width`. Now the row fill's scattered side always fits, because the live target is one block's histograms at any width, and the sequential side stays sequential inside each block. The width-driven choice between the two fills stopped existing, which is different from being made well. Main later added a per-node choice by row density for u8 bins, described in chapter 9 of the guide.

The admitting interleaved A/B beat both strategies at their own best cells: 326s against the row path's 369 at 1M x 4096, 442s against feature-parallel's 514 at 131k x 16384, a wash at 16M x 100. And because tiling never changes a feature's accumulation order, the tiled fill produces bit-identical models to the classic row path at every width, so the determinism contract came out stronger than either strategy could make it. When a trade-off is real, ship a threshold and record what it leaves unfixed, then look for a layout that removes the trade-off. Here that layout was column tiling, which XGBoost 3.3 had already released.

The CUDA wide wall this case left as a residual got the same treatment on 2026-07-30, and it had already fallen: the recorded deficit dated to 2026-07-08 code, and on current main the CUDA growers lead every wide cell (refutation-by-progress, decision 90). The iso-volume frontier (decision 91) then measured the shape axis wholesale, holding rows x cols constant while width swept 128 to 65536 columns; both are in [the archive](../../results/results/archive.md).

**Where wide CPU stood on 2026-08-08.** The parity this case won is a 2026-07 reading and it did not carry to the scenario of that date. On the standings' cpu-wide cell (16k x 16384) depthwise trained in 96.8s against XGBoost's 134.9s, and leafwise trained in 516.7s against LightGBM's 57.2s. The depthwise lead held; the leaf plane's parity did not. That leaf-plane gap has since closed. Current numbers are on [the perf standings](../../results/results/perf.md); every number in this chapter is the campaign's.

Evidence: [benchmarks/wide-cpu-hist-2026-07.md](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/wide-cpu-hist-2026-07.md); decisions 88 to 91; the ratifying field report is issue #217.
