# C1. Cache awareness

The bottom line first. A processor does not read bytes, it reads 64-byte lines, and the central question in this track is which lines are already resident, meaning held in a cache close to the core.

The two running examples are cpu-tall (128 features per row) and cpu-wide (16,384 features per row), a 128x difference in row width. The next two chapters trace their results back to that one ratio.

This chapter assumes no hardware background. It does assume you know what a row, a feature and a histogram are, which [guide chapter 2](../../guide/2-binning-and-histograms.md) covers. If you already know the memory hierarchy, read the two tables and skip to the closing sections.

## Cache line

Ask a modern x86 or ARM core for one byte and it fetches the aligned 64-byte block containing it. That block is a cache line.

The line is the unit of every transfer between cache levels and between cache and memory. Three consequences follow, and they are most of cache-aware programming:

- Reading 64 adjacent bytes costs one fetch. Reading 64 bytes spread over 64 lines costs 64 fetches.
- A line you touch is a line you hold, so touching more lines than the cache holds evicts (pushes out of the cache) something you wanted.
- A line is also the unit of ownership between cores, which the sharing section below returns to.

**Formula:** `bytes_fetched = n_lines * 64`

**Example:** bonsai stores one bin per byte. A row of 128 features occupies 128 bytes and touches 2 lines. A row of 16,384 features occupies 16,384 bytes and touches 256.

**Formula:** `lines_per_row = ceil(n_features / 64)`

**Example:** a row of cpu-tall requires `ceil(128 / 64) = 2` lines. A row of cpu-wide requires `ceil(16384 / 64) = 256` lines.

Hold those two numbers. Every mechanism in this track is a consequence of 2 against 256.

## Cache Hierarchy

Caches are arranged in hierarchies: small and fast near the core, large and slow further out. Sizes and latencies are host properties, not universal constants.

Units in this series are binary for storage: 1KiB is 1,024 bytes, 1MiB is 1,024KiB, and every power-of-two size lands exactly (the wide arena, the memory holding every feature's histograms, is 2^25 bytes, so 32MiB, not "33.5MB"). Measured traffic totals keep the decimal GB the instruments report.

Here are two real topologies, both of them hosts this campaign (the performance work this track follows) measured on.

| level | Apple M2, 4 performance plus 4 efficiency cores | server EPYC Zen 4, up to 96 cores |
|---|---|---|
| L1 data | 128KiB per performance core, ~1ns | 32KiB per core, ~1ns |
| L2 | 16MiB shared by the 4 performance cores | 1MiB private per core, ~4ns |
| | 4MiB shared by the 4 efficiency cores, ~5ns | |
| last level | 8MiB system cache shared by everything | 32MiB per 8-core complex, ~15ns |
| DRAM | ~100ns | ~110ns |

These are representative figures from vendor documentation and published latency work. They are not measurements from this campaign, so read the ratios and not the digits.

The ratio that matters is the last row against the first. A DRAM access costs roughly 100 times an L1 hit.

That is why a loop whose working set (the data it touches repeatedly) falls out of cache does not get 20% slower. Each access now costs a DRAM latency instead of an L1 latency, roughly 100 times more. The engine track's [wide-data case](../engine/6-the-wide-data-wall.md) measures the effect on a full fit.

The second thing to read off the table is the sharing column. The M2 has no per-core L2 at all. Four performance cores share one 16MiB pool, and the efficiency cores share a different one.

The EPYC gives every core a private 1MiB L2. It then shares an L3 within each 8-core complex, and no level above that is cheap to cross.

So "the cache" is not one thing. Whether two workers help each other or evict each other depends on which level they share, and the two hosts answer differently.

The campaign's own attribution says exactly that. It measured the cost of sharing by thread count: not visible at 4 threads inside one M2 cluster, visible at 8 threads across the M2's two clusters, and dominant at 12 threads across an EPYC's core complexes.

## What fits where

A histogram cell is a gradient sum and a hessian sum, 8 bytes. A feature's histogram takes a 256-cell slot: at the default max_bin of 255 that is up to 255 bins (the reserved missing bin included) padded to a fixed stride. This holds for u8 bins; u16 bins pack to their exact width.

[Guide chapter 2](../../guide/2-binning-and-histograms.md) describes the binning and histogram definitions. This chapter only needs their sizes. The histogram arena is the memory holding every feature's histogram, 256 cells per feature. The row-major mirror is a second copy of the bin matrix, stored row by row, that the row fill reads.

### Arena Size
**Formula:** `n_features * n_bins * (2 * sizeof(float)) = n_features * 256 * 8` bytes.

**Example:** 
* cpu-tall gives `128 * 256 * 8 = 256KiB`
* cpu-wide gives `16,384 * 256 * 8 = 32MiB`

### Row-Major Mirror Size
**Formula:** `n_rows * n_features * 1` bytes.

**Example:**
* cpu-tall gives `2,097,152 * 128 = 256MiB`
* cpu-wide gives `16,384 * 16,384 = 256MiB`

Now place those working sets against the cache hierarchy table above. Three readings come out of it.

| working set | size | where it lives |
|---|--:|---|
| one tall row of bins | 128 bytes | any level |
| one wide row of bins | 16KiB | L1 on the M2, half of the EPYC's L1 |
| tall's histogram arena | 256KiB | L2 on either host |
| wide's histogram arena | 32MiB | past the M2's shared levels, past one EPYC complex's L3 |
| either mirror, whole | 256MiB | DRAM, on any host in this table |

Three readings come out of the table.

1. **The tall arena stays resident.** One worker filling 128 features holds a 256KiB arena, which its L2 keeps. Scattered writes into a resident arena are cheap, so the tall fill's real cost is the streaming read of the mirror.
2. **The wide arena stays resident nowhere.** 32MiB overflows one EPYC complex's L3 and is four times the M2's system cache. That is the wall the engine track's [wide-data case](../engine/6-the-wide-data-wall.md) hit. The fix was tiling the mirror into blocks of 2048 features, so one block's histograms fit in cache again. The fill then walks each block in runs of 512 features. A thread's run of the child's histograms and the matching run of its sibling's stay in cache while the fill zeroes, writes and subtracts them (decision 151).
3. **Neither mirror fits anywhere, and it does not matter.** A mirror is streamed once per fill and never re-walked, so it gates no reuse. Its cost is bandwidth, not residency. That is why the fill's row grain is a constant rather than a tuned number (decision 105).

## Sharing a line, two ways

Lines are also the unit of ownership between cores. Two distinct costs follow, and they are often confused.

**Shared-line refetch.** Two workers read different bytes of the same read-only line. Each fetches that line into its own cache, nothing is wrong, and the line is simply transferred more than once. The cost is bandwidth, and it scales with the number of workers sharing the line.

**False sharing.** Two workers write different bytes of the same line. Each write invalidates the other's copy, so the line ping-pongs between caches. The cost is coherence traffic: each write forces the other worker to refetch the line before it can write. The fill avoids this case by construction (see the end of this section), so this campaign has no measured per-event cost for it.

Now the arithmetic that drives [chapter C2](2-dividing-the-work.md).

Give each of `W` workers its own contiguous slice of one row's features. The row spans `L` cache lines, and a worker fetches every line its slice touches.

One picture per shape. Each box is one 64-byte cache line, and the brackets underneath mark whose slices land in it.

```text
cpu-tall row: 128 bytes = 2 lines, 12 workers at ~11 features each

|-------- line 0 ---------|-------- line 1 ----------|
 [w0][w1][w2][w3][w4][w5]  [w6][w7][w8][w9][w10][w11]

each line is fetched by ~6 workers: 12 fetches where 2 would do
```

```text
cpu-wide row: 16,384 bytes = 256 lines, 12 workers at ~21 lines each

|- lines 0..20 -|- lines 21..42 -| ... |- lines 235..255 -|
 [----- w0 ----] [----- w1 -----]  ...  [----- w11 ------]

each line is fetched by 1 worker (2 only at a slice boundary)
```
Now count fetches. A serial walk of the row fetches each of its `L` lines once: `L` fetches.

Split the row instead, and each line is fetched once by every worker whose slice overlaps it. Two regimes fall out:

- Slices wider than a line (`W < L`): each line has one owning worker, so the total stays about `L`.
- Slices narrower than a line (`W > L`): each worker's whole slice sits inside one line, so each of the `W` workers fetches a line, and the total is about `W`.

**Formula:** `line_fetches ~= max(W, L)`. Divide by the serial walk's `L` fetches: `traffic_multiplier = max(W / L, 1)`.

**Example, cpu-tall at 12 workers.** `max(1, 12 / 2) = 6`: roughly 6x the bytes of a single walk.

**Example, cpu-wide at 12 workers.** `max(1, 12 / 256) = 1`, plus boundary sharing: about 1.1.

The campaign measured the consequence rather than the multiplier. When the fill stopped slicing narrow rows across all 12 workers, mirror traffic per tree fell from 3.85GB to 1.45GB at the tall cell.

That is 2.7x fewer bytes moved. It came from changing how the work was divided, not what the loop computes.

The write side is handled by construction rather than by measurement. A worker owns a contiguous feature range. It zeroes and fills whole arena runs for those features, so the cells it writes sit on its own lines.

`NodeHistograms::carve` in `include/bonsai/histogram.hpp` does the zeroing and slot hand-out, one feature's arena run at a time. That feature-range ownership is also why false sharing never appears in the campaign's attribution tables.

## The winner depends on the host
Which library trains faster depends on the host. LightGBM reads 8x to 20x more memory per tree than bonsai. That loses it the race on the M2, yet at the campaign's start it won on a bandwidth-rich EPYC. Four measurements show it.

1. bonsai's mirror fill reads about 3.85GB per tree at the tall cell while its workers still share cache lines (the 6x multiplier above), and 1.45GB once the 2D split ends the sharing ([chapter C2](2-dividing-the-work.md)).
2. LightGBM's column fill reads about 30GB per tree at the same cell. Its workers share nothing: every read stream is private.
3. On the M2, bonsai trains tall 2.4x to 3.2x faster: 17.4 to 22.1 seconds against LightGBM's 42.4 to 70.7, across 4 and 8 threads on identical data, knobs, and r2.
4. On a bandwidth-rich EPYC draw (a draw is the specific machine a cloud rental hands you), at the campaign's start and before the optimizations of the next two chapters, LightGBM trained tall 1.4x faster: 15.6 seconds against bonsai's 21.6, both measured back to back on that one machine.

The same two binaries, the same data, and the winner flipped with the host. Only one variable moved: how much bandwidth the memory system supplies.

A read-traffic handicap of about 8x before those optimizations (30GB against 3.85GB), and 20x once the 2D split removed the line sharing, decides the race where bandwidth is scarce, and stops mattering where bandwidth is abundant. The fast loop is a property of the pair (loop, host), never of the loop alone.

The loop side of that pair is what the campaign then changed. After the optimizations in the next two chapters, bonsai trains tall faster than LightGBM on every server host the standings have measured since: 7.2 against 10.3 seconds on the EPYC 9535 in the 2.7.0 refresh (decision 155). The host lesson still holds, and those optimizations cut bonsai's side of the traffic gap.

## What to carry forward

- A line is 64 bytes, and `lines_per_row = ceil(n_features / 64)`: 2 for tall, 256 for wide.
- A DRAM access costs roughly 100 L1 hits, so a working set that stops fitting makes each access about 100 times slower, not a few percent slower.
- An arena is `n_features * 2KiB`: 256KiB at tall and 32MiB at wide, and only the first stays resident.
- Slicing a 2-line row among 12 workers multiplies its read bytes by roughly 6. Slicing a 256-line row does not.
- Which fill wins is a property of the host, so a threshold tuned on one machine is a claim about that machine.
