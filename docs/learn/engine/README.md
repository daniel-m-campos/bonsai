# The engine track

This track follows the episodes that made bonsai fast. On the accuracy-versus-time frontier at 16M rows on one GPU, bonsai reached every accuracy measured against CatBoost and XGBoost up to r² 0.895 sooner than either, for example r² 0.8749 in 13.9s against CatBoost's 19.7s ([case E1](1-the-marginal-round.md)); decision 78 confirmed it on the 2026-07 re-run. The current [timings](../../results/results/perf.md) are the standing record.

## What this is

The algorithm track teaches how gradient boosting works. This track teaches how bonsai got fast, and the [CPU track](../cpu/README.md) teaches how the CPU path works, starting with the cache line (the 64-byte block the CPU loads from memory at once). This engine track teaches high-performance computing (making programs run fast on specific hardware) through case studies. Each chapter walks through one real engineering episode from the project record, step by step, and applies the same method each time. Failed experiments get as much space as successes, because a measured failure teaches as much as a win.

## Who it is for

A developer who knows some C++ and has never estimated how long a GPU kernel (a function that runs on the GPU) will take before running it. You do not need CUDA experience. You need to be willing to trust a measurement over an intuition.

## The method in brief

The rules are the compute-DAG method's (training modeled as a graph of steps, each with a measured cost), stated once in [guide chapter 11](../../guide/11-performance-engineering.md): estimate the time a change will save before you build it, split each profiler line into measured parts before you optimize it, and treat the gap between the measured total and the sum of the parts as the next target. Two working habits run alongside them in every case here. Compare timings only between runs on the same rented GPU machine (a pod), because two identical L40S pods measured 25% apart ([E1](1-the-marginal-round.md)). And a refutation is a deliverable: a measured no, with the conditions that would reopen it, is worth as much as a win.

## The chapters

- **E1. [The marginal round](1-the-marginal-round.md).** Timing each step shows that a planned kernel rewrite would not pay off, and a table that prices each candidate change from measured costs then cuts one boosting round (one tree's worth of training) at 16M rows from 155 to 104 ms.
- **E2. [The missing bin](2-the-missing-bin.md).** An acceptance test fails, and the fix closes a train/predict skew hiding in every fitted model.
- **E3. [The parity verdict](3-the-parity-verdict.md).** A data-parallel multi-GPU engine, built and measured to parity, then parked as an experiment.
- **E4. [The resident objective](4-the-resident-objective.md).** Deleting the per-tree host round-trip, and the round falls again from 104 to 64 ms.
- **E5. [The ceiling](5-the-ceiling.md).** 500M rows by 100 features trained end to end on one 80GB card.
- **E6. [The wide-data wall](6-the-wide-data-wall.md).** A production field report at 16k features: LightGBM trained 2 to 3x faster than every bonsai grower (7x for leafwise). Cache arithmetic shows why. Two wrong explanations fail an interleaved A/B (both builds timed alternately on the same machine). A tiled layout (the row-major copy of the data stored in 2048-feature blocks) then removes the need to choose between the two fill loops.
- **E7. [The lane finder](7-the-lane-finder.md).** A SIMD lesson (one instruction working on several numbers at once) told through the split finder. It scans eight features in lockstep, bit-identical to the one-feature-at-a-time scan. The chapter covers vector types, masks and selects, why eight lanes hide the latency of the slow divide, and why a first draft limited to the baseline x86 instruction set (SSE2) lost. It also covers the run-time check that uses the wider AVX2 instructions only where the CPU has them. On wide data the depthwise and leafwise fits run 13.5 to 27% shorter than with the scalar scan, same pod.
