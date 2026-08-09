# The engine track

This track is how an engine whose planes are measured in [code metrics](../../method/results/code-metrics.md) became the fastest route to every measured accuracy at 16M rows on one GPU. It teaches that through the episodes that did it.

## What this is

The algorithm track teaches how gradient boosting works. This track teaches how bonsai got fast. It is HPC pedagogy told case-method, like a college class. Each chapter is one real engineering episode from the project record, built up slowly, with the discipline as the through-line. The losses and the refutations are the curriculum, not the footnotes.

## Who it is for

A developer who knows some C++ and has never priced a GPU kernel. You do not need CUDA experience. You need to be willing to trust a measurement over an intuition, which is the entire method.

## The method in brief

The rules are the compute-DAG method's, stated once in [guide chapter 11](../../guide/11-performance-engineering.md): price a move before betting on it, decompose a line before optimizing it, and treat whatever conservation cannot explain as the next target. Two working habits run alongside them in every case here. Every delta is a same-pod delta, because two identical GPUs measured 25% apart across the fleet. And a refutation is a deliverable: a measured no, with the conditions that would reopen it, is worth as much as a win.

## The chapters

- **E1. [The marginal round](1-the-marginal-round.md).** Instrumentation cancels a kernel rewrite, then a price list cuts the 16M round from 155 to 104 ms.
- **E2. [The missing bin](2-the-missing-bin.md).** An acceptance test fails, and the fix closes a train/predict skew hiding in every fitted model.
- **E3. [The parity verdict](3-the-parity-verdict.md).** A data-parallel multi-GPU engine, built and measured to parity, then parked as an experiment.
- **E4. [The resident objective](4-the-resident-objective.md).** Deleting the per-tree host round-trip, and the round falls again from 104 to 64 ms.
- **E5. [The ceiling](5-the-ceiling.md).** 500M rows by 100 features trained end to end on one 80GB card.
- **E6. [The wide-data wall](6-the-wide-data-wall.md).** A production field report at 16k features, the cache arithmetic behind a 2-6x cliff, two wrong theories killed by an interleaved A/B, and the tiled layout that dissolves the trade-off.
