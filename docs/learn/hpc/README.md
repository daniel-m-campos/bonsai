# The HPC track

This track teaches how bonsai's CPU histogram fill got fast, starting at the cache line and stopping at the trade-off ledger. <!-- HPC track should also cover GPU at some point, not just cpu. actually this seems like a naming issue engine vs hpc. It's all HPC, maybe gpu vs. cpu is better. -->

## What this is

The [algorithm track](../../guide/README.md) teaches how gradient boosting works. The [engine track](../engine/README.md) teaches how bonsai got fast on the GPU, one case at a time.

This track teaches the machine underneath both. What a cache line is, what a miss costs, and why one loop wins on one host and loses on another.

The material comes from one campaign on the CPU leaf-wise grower (issue #367, PR #369). It ran nine measured rounds. Five shipped a lever and four refuted one.

The refutations are the curriculum. A mechanism you can name is worth more than a speedup you cannot explain.

Three claims the campaign closed, in order of how much they teach:

- A fill's speed is not a property of its code. It is a property of whether the bytes it scatters over fit in cache.
- Two cuts of the same loop have opposite cache behavior. Which one wins flips with the data's shape. <!-- "cuts of the loop" refering to row or feature looping is unclear, you used this is various places and the "cut" slang should be avoided -->
- Synchronization is cheap in absolute terms and expensive against small work. Region count matters only where the work per region is small. <!-- region is vague-->

## Who it is for

A developer who can read a loop and has never priced one. You need no C++, no OpenMP, and no benchmarking experience.

You do need one habit. When your arithmetic and a measurement disagree, the measurement is right and the arithmetic was about the wrong thing. Four of the nine rounds ended that way.

There are no code examples here. The engine's real lines are in [guide chapter 9](../../guide/9-parallelism-and-determinism.md), and the contracts are in [architecture doc 7](../../architecture/7-parallel.md).

## The two running examples

Every chapter uses the same two datasets, introduced here once. They are the standings' own CPU scenarios, defined in [the benchmark protocol](../../method/benchmark-protocol.md).

| | cpu-tall | cpu-wide |
|---|--:|--:|
| rows | 2,097,152 | 16,384 |
| features | 128 | 16,384 |
| `n_cells = n_rows * n_features` | 2^28 | 2^28 |
| one row of bins, one byte each | 128 bytes | 16,384 bytes |
| one node's histogram arena | 256KiB | 32MiB |

The two cells hold the same number of bins by construction. Nothing that differs between them is a size effect.

So every difference you read in these chapters is a shape effect. That is the entire reason the pair exists.

Both fits use the same knobs: 100 boosting rounds, depth 8, 255 bins, and 12 threads unless a sentence says otherwise.

## The chapters

- **H1. [Cache lines by the numbers](1-cache-lines-by-the-numbers.md).** What a line is, what the levels cost, and what fits where. Why a 128-byte row and a 16,384-byte row are different machines.
- **H2. [Cutting the work](2-cutting-the-work.md).** The histogram fill as a worked problem. Two ways to cut it, who pays for each, and the storage orientation underneath both.
- **H3. [Coordination costs](3-coordination-costs.md).** What a parallel region costs and when that cost dominates. How team sizing goes wrong, and the four-entry ledger the campaign closed with.

Read them in order. Each one spends the arithmetic the one before it set up.

## What this track does not own

| topic | owner |
|---|---|
| the row-major mirror and the histogram | [guide chapter 2](../../guide/2-binning-and-histograms.md) |
| the determinism contract | [guide chapter 9](../../guide/9-parallelism-and-determinism.md) |
| the parallel primitives and the fill's constants | [architecture doc 7](../../architecture/7-parallel.md) |
| how any number here was measured | [the benchmark protocol](../../method/benchmark-protocol.md) |
| what bonsai currently trains these cells in | [the scenario panels](../../method/results/perf.md) |

Prose here never restates a standings digit. Campaign numbers appear with the round that measured them, and current numbers live on the panels page.

## The record

Issue #367 opened the campaign with its opening ledger. PR #369 carries the round-by-round evidence, including the pre-registered falsifier each round was read against.

[Decision 105](../../decisions.md) records the fill composition the level plane ships. [Decision 106](../../decisions.md) records the one competitor lever the campaign measured and declined.
