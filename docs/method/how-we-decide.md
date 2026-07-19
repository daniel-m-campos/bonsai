# How we decide

Two rules govern every change to bonsai. Price a change before you build it. Admit a feature only when measurement earns its place. Both replace argument with a number, and both record the refutations so nobody pays for the same idea twice.

## Price before you build

No optimization begins until instrumentation has decomposed and priced the cost it attacks. None ships without a measurement showing the predicted win arrived. The full cycle is decompose, price, implement, validate, record ([architecture doc 16](https://github.com/daniel-m-campos/bonsai/blob/main/docs/architecture/16-compute-dag.md) is the canonical statement; [guide chapter 11](../guide/11-performance-engineering.md) tells it as a story).

The rule exists because profilers lie in exactly the places that matter.

### The round that instrumentation cancelled

At 16M rows, the GPU grow profile showed 8.4 seconds in the split-find lap, the largest single line. The obvious plan was a multi-hour rewrite of the find kernel, whose warp scan runs in double precision on an L40S with weak FP64.

Instrumenting first told a different story. A profile-gated sync split the lap into kernel-compute versus transfer and showed the find kernel costs 0.17 seconds. The 8.4 seconds was the profiler's opening `cudaDeviceSynchronize` catching the previous level's asynchronous histogram kernels: async work billed to whoever synchronizes next. Every hypothesis banked for the round was refuted by one measurement, and the rewrite was cancelled before any kernel code was written ([decision 62](../decisions.md)).

The same discipline, run the other way, produced the best single win. An instruction-level cost ledger of the 16M CPU fit showed the histogram fill loop DRAM-latency-bound, 82 of 107 seconds. It predicted that a software prefetch would close most of the gap. The one-line change landed inside the estimate: 107 seconds down to 75.8, a tie with XGBoost-hist's 75.7 in the same session ([decision 61](../decisions.md)). On a later pod XGBoost leads by ~7%, and the CPU order is recorded as host-dependent.

### Microbenchmarks lie too

A microbenchmark of the fill loop promised 1.6x from splitting the accumulator chain. The real loop was flat at every thread count. The microbenchmark's arrays were cache-resident, which isolated a floating-point dependency that the real streaming loop hides behind memory latency. The code was dropped and the lesson recorded ([decision 49](../decisions.md)).

An isolated measurement inherits the bottleneck structure of the isolation, not of the system. Price a change in the real loop or not at all.

## Admit a feature by measurement

A feature enters the core only after three tests pass. Its benefit is shown by the cheapest possible prototype, at zero cost to the core. The benefit moves standings on a fixed evaluation suite, not a hand-picked example. The kill criteria were written down before the experiment ran. Declines are recorded with the same care as admissions.

The rule exists because the strongest pressure on a library is "the other libraries have it," and that pressure is answerable with measurement.

### Categoricals: the gate end to end

Native categorical splits were the most-requested structural feature, present in all three reference libraries. The cheapest prototype was not C++. It was toggling LightGBM's own categorical support on and off, and feeding bonsai a 40-line ordered-target-statistics preprocessing step, across three real categorical datasets.

The measurements settled it. LightGBM's native toggle hurts on one of the three datasets (kick, -0.018 AUC against plain ordinals). bonsai plus the preprocessing beats LightGBM-native on the hardest one (amazon, 0.8590 against 0.8572). CatBoost-native keeps a real lead that traces to machinery that is engine-side by nature.

The verdict wrote itself. The invasive C++ design, already fully drafted, was declined. The 40-line encoder shipped as `OrderedTargetEncoder`, and the engine core stayed untouched ([decision 58](../decisions.md), [the trade-off study](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/categorical-tradeoff-2026-07.md)). The declined design is preserved in [architecture doc 17](https://github.com/daniel-m-campos/bonsai/blob/main/docs/architecture/17-categorical-splits.md) with the measurements that would reopen it.

### Ordered boosting: a hypothesis killed for free

The hypothesis that CatBoost's accuracy at scale came from ordered boosting was killed without implementing anything. Benchmark CatBoost against itself, `Ordered` against `Plain`, same data and budget. The result was identical accuracy at roughly 7x the cost, and CatBoost defaults the feature off past ~50k rows anyway ([the study](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/catboost-scale-edge-2026-07.md)).

That is the gate's cheapest form. When the feature exists in a reference implementation, the reference is the prototype. Measure it there before building it here.

## Both rules keep their receipts

The output worth the most is the decline: a recorded, measured "no" with the conditions that would reopen it. It converts a recurring debate into a lookup. The named cost axes force the price into the open before anyone writes C++. Those axes are core lines of code, new configuration knobs, whether existing behavior stays bit-identical, and whether the documentation story survives.

The raw feed behind both rules is the [decisions log](../decisions.md): numbered, dated, with the rejected alternatives next to the adopted ones.
