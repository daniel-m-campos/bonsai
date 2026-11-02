# The CPU track

## What this is

This track teaches how bonsai's CPU histogram fill (the loop that adds each row's gradient into per-feature bins) works, from the cache line (the 64-byte block in which memory moves into cache) up. On the M2, bonsai trains the cpu-tall cell 2.4x to 3.2x faster than LightGBM (17.4 to 22.1 seconds against 42.4 to 70.7). On one EPYC host at the campaign's start, LightGBM trained it 1.4x faster (15.6 seconds against 21.6, [chapter C1](1-cache-awareness.md#the-winner-depends-on-the-host)). The [engine track](../engine/README.md) covers the GPU (device) side, and its [case E7](../engine/7-the-lane-finder.md) covers the CPU (host) split finder. Both rest on the machine model taught here. You learn what a cache miss (a read the cache cannot serve, so it goes to slower memory) costs, and why one loop wins on one host and loses on another.

The material comes from one campaign on the CPU leaf-wise grower (issue #367, PR #369, August 2026). It ran nine measured rounds. Five shipped a lever and four refuted one. Where a later decision moved what the campaign built, the chapter says so in place and cites it; the arithmetic is the durable part.

Read the four refuted rounds closely. Each one ends with a named mechanism, and a mechanism applies to other code where a speedup you cannot explain does not.

Three claims the campaign closed, in order of how much they teach:

- A fill's speed is not a property of its code. It is a property of whether the bytes it scatters over fit in cache.
- The fill can loop over rows in parallel or over features in parallel. The two forms have opposite cache behavior, and which one wins flips with the data's shape.
- Starting a parallel region (one fork/join: the thread team starts, runs its loop, and waits at the end) costs about 35 microseconds at 12 threads. That is a rounding error for a wide fill and a real cost for a small one.

There are no code examples here. The engine's real lines are in [guide chapter 9](../../guide/9-parallelism-and-determinism.md), the constants in `src/fill/lone.hpp`, and the contracts in [the invariants](../../invariants.md).

## The two running examples

Every chapter uses the same two datasets, introduced here once. They are the standings' own CPU scenarios, defined in [the benchmark protocol](../../results/benchmark-protocol.md).

| description | formula | cpu-tall | cpu-wide |
|---|---|--:|--:|
| number of rows |`n_rows`|  2,097,152 | 16,384 |
| number of features |`n_features` | 128 | 16,384 |
| number of cells |`n_cells = n_rows * n_features` | 2^28 | 2^28 |
| size of one binned row |`n_features * 1` | 128 bytes | 16,384 bytes |
| size of one node's histogram arena |`n_features * bins_per_feature * 8` | 256KiB | 32MiB |

Both datasets hold 2^28 cells, so any difference you read in these chapters comes from shape (many rows or many features), not size. That is the entire reason the pair exists.

Both fits use the same parameters: 100 boosting rounds, depth 8, 255 bins, and 12 threads, unless a sentence says otherwise.

## The chapters

- **C1. [Cache awareness](1-cache-awareness.md).** What a line is, what the levels cost, and what fits where. Why a 128-byte row and a 16,384-byte row (a 128x difference) use the cache so differently.
- **C2. [Dividing the work](2-dividing-the-work.md).** The histogram fill as a worked problem: row-parallel against feature-parallel, who pays for each, and the storage orientation underneath both.
- **C3. [Coordination costs](3-coordination-costs.md).** The fixed and variable costs of a parallel fill, and the team-sizing rule they derive. How the rule's first forms went wrong, and the four-entry ledger the campaign closed with.

Read them in order: each chapter uses the numbers the previous chapter derives.
