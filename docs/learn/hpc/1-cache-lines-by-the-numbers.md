# H1. Cache lines by the numbers

The bottom line first. A processor does not read bytes, it reads 64-byte lines, and almost every question in this track is which lines are already resident.

The two running examples differ by 128x in row width. That one ratio drives most of what the next two chapters measure.

This chapter assumes nothing. If you already know the memory hierarchy, read the two tables and skip to the closing sections.

## The line is 64 bytes

Ask a modern x86 or ARM core for one byte and it fetches the aligned 64-byte block containing it. That block is a cache line.

The line is the unit of every transfer between cache levels and between cache and memory. Three consequences follow, and they are most of cache-aware programming:

- Reading 64 adjacent bytes costs one fetch. Reading 64 bytes spread over 64 lines costs 64 fetches.
- A line you touch is a line you hold, so touching more lines than the cache holds evicts something you wanted.
- A line is also the unit of ownership between cores, which the sharing section below returns to.

**Formula.** Bytes a walk fetches equals lines it touches times 64.

**Example.** bonsai stores one bin per byte. A row of 128 features occupies 128 bytes and touches 2 lines. A row of 16,384 features occupies 16,384 bytes and touches 256.

**Formula.** Lines per row equals ceil(features / 64).

**Example.** cpu-tall gives ceil(128 / 64), which is 2 lines. cpu-wide gives ceil(16384 / 64), which is 256.

Hold those two numbers. Every mechanism in this track is a consequence of 2 against 256.

## What the levels cost

Caches come in levels: small and fast near the core, large and slow further out. Sizes and latencies are host properties, not universal constants.

Units in this series are binary for storage: 1KiB is 1,024 bytes, 1MiB is 1,024KiB, and every power-of-two size lands exactly (the wide arena is 2^25 bytes, so 32MiB, not "33.5MB"). Measured traffic totals keep the decimal GB the instruments report.

Here are two real topologies, both of them hosts this campaign measured on.

| level | Apple M2, 4 performance plus 4 efficiency cores | server EPYC Zen 4, up to 96 cores |
|---|---|---|
| L1 data | 128KiB per performance core, ~1ns | 32KiB per core, ~1ns |
| L2 | 16MiB shared by the 4 performance cores | 1MiB private per core, ~4ns |
| | 4MiB shared by the 4 efficiency cores, ~5ns | |
| last level | 8MiB system cache shared by everything | 32MiB per 8-core complex, ~15ns |
| DRAM | ~100ns | ~110ns |

These are representative figures from vendor documentation and published latency work. They are not measurements from this campaign, so read the ratios and not the digits.

The ratio that matters is the last row against the first. A DRAM access costs roughly 100 times an L1 hit.

That is why a loop whose working set falls out of cache does not get 20% slower. It changes regime, which the engine track's [wide-data case](../engine/6-the-wide-data-wall.md) shows end to end.

The second thing to read off the table is the sharing column. The M2 has no per-core L2 at all. Four performance cores share one 16MiB pool, and the efficiency cores share a different one.

The EPYC gives every core a private 1MiB L2. It then shares an L3 within each 8-core complex, and no level above that is cheap to cross.

So "the cache" is not one thing. Whether two workers help each other or evict each other depends on which level they share, and the two hosts answer differently.

The campaign's own attribution says exactly that. One mechanism was invisible at 4 threads inside one M2 cluster. It was visible at 8 threads across the M2's two clusters, and dominant at 12 threads across an EPYC's core complexes.

## What fits where

A histogram cell is a gradient sum and a hessian sum, 8 bytes. A feature's histogram is 256 cells, one per bin, including the reserved missing bin.

[Guide chapter 2](../../guide/2-binning-and-histograms.md) owns the binning and histogram definitions. This chapter only needs their sizes.

**Formula.** Arena bytes equals features times 256 cells times 8 bytes.

**Example.** cpu-tall gives 128 x 256 x 8, which is 256KiB. cpu-wide gives 16,384 x 256 x 8, which is 32MiB.

**Formula.** Mirror bytes equals rows times features times 1 byte.

**Example.** cpu-tall gives 2,097,152 x 128, which is 256MiB. cpu-wide gives 16,384 x 16,384, which is the same 256MiB.

Now place those working sets against the table above. Three readings come out of it.

| working set | size | where it lives |
|---|--:|---|
| one tall row of bins | 128 bytes | any level |
| one wide row of bins | 16KiB | L1 on the M2, half of the EPYC's L1 |
| tall's histogram arena | 256KiB | L2 on either host |
| wide's histogram arena | 32MiB | past the M2's shared levels, past one EPYC complex's L3 |
| either mirror, whole | 256MiB | DRAM, on any host in this table |

First, the tall fill's target is small enough to stay resident. One worker filling 128 features holds a 256KiB arena, which its L2 keeps.

The scattered writes a row-order walk makes are therefore cheap. What the tall fill costs is the streaming read of the mirror.

Second, the wide fill's target cannot stay resident anywhere. 32MiB is larger than one EPYC complex's L3 and four times the M2's system cache.

That is the wall the engine track's [wide-data case](../engine/6-the-wide-data-wall.md) hit. The fix there was tiling the mirror into 2048-feature blocks, so one block's histograms fit again.

Third, and this is the one that surprises people: neither mirror fits in any cache, and it does not matter.

A mirror is streamed once per fill and never re-walked, so it gates no reuse. Its cost is bandwidth, not residency.

That is why the fill's row grain is a constant rather than a tuned cache-fitting number. [Architecture doc 7](../../architecture/7-parallel.md) states the four constants and the reason each one is structural.

## Sharing a line, two ways

Lines are also the unit of ownership between cores. That creates two distinct costs, they are often confused, and the campaign paid both.

**Shared-line refetch.** Two workers read different bytes of the same read-only line. Each fetches that line into its own cache, nothing is wrong, and the line is simply transferred more than once. The cost is bandwidth, and it scales with the number of workers sharing the line.

**False sharing.** Two workers write different bytes of the same line. Each write invalidates the other's copy, so the line ping-pongs between caches. The cost is coherence traffic, and it is worse per event than a refetch.

Now the arithmetic that drives [chapter H2](2-cutting-the-work.md).

**Formula.** Cut a row of L lines among W workers, giving each a contiguous feature slice. Each worker fetches every line its slice touches. When L is smaller than W, several workers land in the same lines, and the row is fetched up to W times.

**Example, cpu-tall at 12 workers.** The row is 128 bytes, so 2 lines. Each slice is about 11 features, so all 12 slices sit inside those same 2 lines. Every worker fetches both, and the row's read traffic multiplies by roughly 12.

**Example, cpu-wide at 12 workers.** The row is 16,384 bytes, so 256 lines. Each slice is about 1,365 features and spans its own ~21 lines. At most one line at each end is shared with a neighbour, so the multiplier is about 1.1.

The campaign measured the consequence rather than the multiplier. When the fill stopped cutting narrow rows by feature alone, mirror traffic per tree fell from 3.85GB to 1.45GB at the tall cell (round five).

That is a 2.7x cut in bytes moved. It came from changing how the work was cut, not what the loop computes.

The write side is handled by construction rather than by measurement. A worker owns a contiguous feature range. It starts and fills whole arena runs for those features, so the cells it writes are its own lines.

That is what `carve` names in [architecture doc 7](../../architecture/7-parallel.md). It is also why false sharing never appears in the campaign's attribution tables.

## Topology decides winners

Everything above says one thing from different angles. The fast loop is the one whose scattered side fits, and what fits is a host property.

The clearest demonstration is an inversion between the two hosts in the table above.

On the local M2, bonsai trains the tall cell 2.4x faster than LightGBM. The readings are 17.4 to 22.1 seconds against 42.4 to 70.7 at 4 and 8 threads, on identical data and knobs with matching r2.

The mechanism is the one this chapter has been building. LightGBM's column-major fill buys disjoint per-worker streams at roughly 20x bonsai's read traffic, about 30GB against 1.45GB per tree. The M2's memory system cannot feed that purchase.

On a bandwidth-rich EPYC draw the same purchase is cheap, and the sign flips. There LightGBM's tall train read 15.6 seconds against bonsai's 21.6, roughly 1.35x the other way, with the arms rotated on one pod (round eight).

Both numbers are honest and neither is a standing claim. The M2 figures are local engineering numbers, never publishable under [the benchmark protocol](../../method/benchmark-protocol.md).

The EPYC figures are same-pod campaign evidence. What bonsai currently trains these cells in is on [the scenario panels](../../method/results/perf.md), which is the only page where a digit is a claim.

The lesson is not that bonsai wins or loses at tall. It is that the tall pairing sits near parity and is decided by the draw's memory subsystem.

That is a weaker statement than either measurement alone, and a more useful one.

## What to carry forward

- A line is 64 bytes, and lines per row is ceil(features / 64): 2 for tall, 256 for wide.
- A DRAM access costs roughly 100 L1 hits, so a working set that stops fitting changes regime rather than degrading.
- An arena is features times 2KiB: 256KiB at tall and 32MiB at wide, and only the first stays resident.
- Cutting a 2-line row among 12 workers multiplies its read traffic by roughly 12. Cutting a 256-line row does not.
- Which fill wins is a property of the host, so a threshold tuned on one machine is a claim about that machine.
