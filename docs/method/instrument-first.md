# Instrument first

The rule: no optimization work begins until the cost being attacked has been decomposed and priced by instrumentation, and no optimization ships without a measurement showing the predicted win materialized. The full cycle is decompose, price, implement, validate, record ([architecture doc 16](https://github.com/daniel-m-campos/bonsai/blob/main/docs/architecture/16-compute-dag.md) is the canonical statement; [guide chapter 11](../guide/11-performance-engineering.md) tells it as a story).

The rule exists because profilers lie in exactly the places that matter.

## The round that instrumentation cancelled

At 16M rows, the GPU grow profile showed 8.4 seconds in the split-find lap, the largest single line. The obvious plan was a multi-hour rewrite of the find kernel (the warp scan runs in double precision; an L40S has weak FP64).

Instrumenting first, with a profile-gated sync that split the lap into kernel-compute versus transfer, showed the find kernel actually costs 0.17 seconds. The 8.4 seconds was the profiler's opening `cudaDeviceSynchronize` catching the previous level's asynchronous histogram kernels: async work billed to whoever synchronizes next. Every hypothesis banked for the round was refuted by one measurement, and the rewrite was cancelled before any kernel code was written ([decision 62](../decisions.md)).

The same discipline, run in the other direction, produced the project's best single win: an instruction-level cost ledger of the 16M CPU fit showed the histogram fill loop DRAM-latency-bound (82 of 107 seconds), predicted that a software prefetch would close most of it, and the one-line change landed within the ledger's estimate: 107 seconds down to 75.8, a dead tie with xgboost-hist's 75.7 in the same session ([decision 61](../decisions.md); on a later pod xgboost leads by ~7%, and the CPU order is recorded as host-dependent).

## The corollary: microbenchmarks lie too

A microbenchmark of the fill loop promised 1.6× from splitting the accumulator chain; the real loop was exactly flat at every thread count. The microbenchmark's arrays were cache-resident, isolating a floating-point dependency the real streaming loop hides behind memory latency. The code was dropped and the lesson recorded ([decision 49](../decisions.md), the "measured and rejected" section).

The general form: an isolated measurement inherits the bottleneck structure of the isolation, not of the system. Price changes in the real loop or not at all.

## Transferring it

Any system with asynchronous execution has the sync-boundary attribution problem: GPU pipelines, async I/O, message queues, JIT runtimes. The fix is always the same shape: add a profile-gated boundary that separates "work done here" from "work awaited here" before believing any per-stage number.

And the cheapest optimization is the one you never write. A cancelled round with a recorded refutation costs an hour; a shipped speculative rewrite costs review, maintenance, and the next person's confusion indefinitely.
