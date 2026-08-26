# How we decide

Two rules govern every change to bonsai. Price a change before you build it. Admit a feature only when measurement earns its place. Both replace argument with a number, and both record the refutations so nobody pays for the same idea twice.

## Price before you build

No optimization begins until instrumentation has decomposed and priced the cost it attacks. None ships without a measurement showing the predicted win arrived. The full cycle is decompose, price, implement, validate, record ([guide chapter 11](../guide/11-performance-engineering.md) tells it as a story; `scripts/dag_model.py` is the living evaluator).

The rule exists because profilers lie in exactly the places that matter. [Case E1](../learn/engine/1-the-marginal-round.md) is the round instrumentation cancelled: the largest line in the GPU grow profile turned out to be the previous level's histograms draining at the profiler's own sync, and a multi-hour kernel rewrite was called off before any kernel code was written ([decision 62](../decisions.md)). Run the other way, the same ledger predicted the one-line prefetch that closed the largest CPU loss on any chart ([decision 61](../decisions.md)).

Isolated measurements mislead in the same shape: a microbenchmark inherits the bottleneck structure of the isolation, not of the system, which is why an accumulator split that looked like a win on cache-resident arrays was flat in the real streaming loop ([chapter 9](../guide/9-parallelism-and-determinism.md), [decision 49](../decisions.md)). Price a change in the real loop or not at all.

## Admit a feature by measurement

A feature enters the core only after three tests pass. Its benefit is shown by the cheapest possible prototype, at zero cost to the core. The benefit moves standings on a fixed evaluation suite, not a hand-picked example. The kill criteria were written down before the experiment ran. Declines are recorded with the same care as admissions.

The rule exists because the strongest pressure on a library is "the other libraries have it," and that pressure is answerable with measurement.

### Categoricals: the gate end to end

Native categorical splits were the most-requested structural feature, present in all three reference libraries. The cheapest prototype was not C++. It was toggling LightGBM's own categorical support on and off, and feeding bonsai a 40-line ordered-target-statistics preprocessing step, across three real categorical datasets.

The measurements settled it. LightGBM's native toggle hurts on one of the three datasets (kick, -0.018 AUC against plain ordinals). bonsai plus the preprocessing beats LightGBM-native on the hardest one (amazon, 0.8590 against 0.8572). CatBoost-native keeps a real lead that traces to machinery that is engine-side by nature.

The verdict wrote itself. The invasive C++ design, already fully drafted, was declined. The 40-line encoder shipped as `OrderedTargetEncoder`, and the engine core stayed untouched ([decision 58](../decisions.md), [the trade-off study](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/categorical-tradeoff-2026-07.md)). The declined design and the measurements that would reopen it are recorded in decisions 58 and 80; the drafted design lives in git history.

### Ordered boosting: a hypothesis killed for free

The hypothesis that CatBoost's accuracy at scale came from ordered boosting was killed without implementing anything. Benchmark CatBoost against itself, `Ordered` against `Plain`, same data and budget. The result was identical accuracy at roughly 7x the cost, and CatBoost defaults the feature off past ~50k rows anyway ([the study](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/catboost-scale-edge-2026-07.md)).

That is the gate's cheapest form. When the feature exists in a reference implementation, the reference is the prototype. Measure it there before building it here.

## Both rules keep their receipts

The output worth the most is the decline: a recorded, measured "no" with the conditions that would reopen it. It converts a recurring debate into a lookup. The named cost axes force the price into the open before anyone writes C++. Those axes are core lines of code, new configuration knobs, whether existing behavior stays bit-identical, and whether the documentation story survives.

The raw feed behind both rules is the [decisions log](../decisions.md): numbered, dated, with the rejected alternatives next to the adopted ones.
