# The feature-admission gate

The rule: a feature enters the core only after (1) its benefit is demonstrated by the cheapest possible prototype, at zero cost to the core, (2) the benefit moves standings on a fixed evaluation suite, not a hand-picked example, and (3) the kill criteria were written down before the experiment ran. Declines are recorded with the same care as admissions.

The rule exists because the strongest pressure on a library is "the other libraries have it," and that pressure is answerable with measurement instead of argument.

## Categoricals: the gate working end to end

Native categorical splits were the most-requested structural feature, present in all three reference libraries. The cheapest prototype was not C++: it was toggling lightgbm's own categorical support on and off, and feeding bonsai a 40-line ordered-target-statistics preprocessing step, across three real categorical datasets.

The measurements: lightgbm's own native toggle *hurts* on one of the three datasets (kick, −0.018 AUC vs plain ordinals); bonsai plus the preprocessing beats lightgbm-native on the hardest one (amazon, 0.8590 vs 0.8572); catboost-native keeps a real lead that traces to machinery that is engine-side by nature.

The verdict wrote itself: the invasive C++ design (already fully drafted) was declined, the 40-line encoder shipped as `OrderedTargetEncoder`, and the engine core stayed untouched ([decision 58](../decisions.md), [the trade-off study](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/categorical-tradeoff-2026-07.md)). The declined design is preserved in [architecture doc 17](https://github.com/daniel-m-campos/bonsai/blob/main/docs/architecture/17-categorical-splits.md) with the measurements that would reopen it.

## Ordered boosting: a hypothesis killed for free

The hypothesis that catboost's accuracy at scale came from ordered boosting was killed without implementing anything: benchmark catboost against itself, `Ordered` vs `Plain`, same data and budget. Identical accuracy, roughly 7× the cost, and catboost defaults the feature off past ~50k rows anyway ([the study](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/catboost-scale-edge-2026-07.md)).

That is the gate's cheapest form: when the feature exists in a reference implementation, the reference *is* the prototype. Measure it there before building it here.

## Transferring it

The gate generalizes to any system under feature pressure. The named cost axes (core lines of code, new configuration knobs, whether existing behavior stays bit-identical, whether the documentation story survives) force the price into the open, and the pre-registered kill criteria prevent the experiment from being re-argued after the result is in.

The output worth the most is the decline: a recorded, measured "no" with reopening conditions. It converts a recurring debate into a lookup.
