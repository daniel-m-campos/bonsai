# The CPU track

## What this is

This track teaches how bonsai's CPU histogram fill got fast, from the cache line up. The [GPU track](../gpu/README.md) tells the device side's episodes; the machine model here underlies both. What a cache line is, what a miss costs, and why one loop wins on one host and loses on another.

The material comes from one campaign on the CPU leaf-wise grower (issue #367, PR #369). It ran nine measured rounds. Five shipped a lever and four refuted one.

The refutations are the curriculum. A mechanism you can name is worth more than a speedup you cannot explain.

Three claims the campaign closed, in order of how much they teach:

- A fill's speed is not a property of its code. It is a property of whether the bytes it scatters over fit in cache.
- The fill can loop over rows in parallel or over features in parallel. The two forms have opposite cache behavior, and which one wins flips with the data's shape.
- Starting a parallel region (one fork/join episode: the thread team starts, runs its loop, and waits at the end) is cheap in absolute terms and expensive against small work.

There are no code examples here. The engine's real lines are in [guide chapter 9](../../../guide/9-parallelism-and-determinism.md), and the contracts are in [architecture doc 7](../../../architecture/7-parallel.md).

## The two running examples

Every chapter uses the same two datasets, introduced here once. They are the standings' own CPU scenarios, defined in [the benchmark protocol](../../../method/benchmark-protocol.md).

| description | formula | cpu-tall | cpu-wide |
|---|---|--:|--:|
| number of rows |`n_rows`|  2,097,152 | 16,384 |
| number of features |`n_features` | 128 | 16,384 |
| number of cells |`n_cells = n_rows * n_features` | 2^28 | 2^28 |
| size of one binned row |`n_features * 1` | 128 bytes | 16,384 bytes |
| size of one node's histogram arena |`n_features * bins_per_feature * 8` | 256KiB | 32MiB |

The two datasets hold the same number of cells by construction, so nothing that differs between them is a size effect.

So every difference you read in these chapters is a shape effect. That is the entire reason the pair exists.

Both fits use the same knobs: 100 boosting rounds, depth 8, 255 bins, and 12 threads unless a sentence says otherwise.

## The chapters

- **C1. [Cache awareness](1-cache-awareness.md).** What a line is, what the levels cost, and what fits where. Why a 128-byte row and a 16,384-byte row are different machines.
- **C2. [Dividing the work](2-dividing-the-work.md).** The histogram fill as a worked problem: row-parallel against feature-parallel, who pays for each, and the storage orientation underneath both.
- **C3. [Coordination costs](3-coordination-costs.md).** What a parallel region costs and when that cost dominates. How team sizing goes wrong, and the four-entry ledger the campaign closed with.

Read them in order. Each one spends the arithmetic the one before it set up.
