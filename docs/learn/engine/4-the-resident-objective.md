# E4. The resident objective

Case E3 parked a multi-GPU engine and named its reopener: a device-resident objective. This case pursues that reopener single-GPU first, and the 16M oblivious round falls from 104 to 64 milliseconds.

The move is not to optimize the host round-trip. It is to delete it. Every tree, the objective crossed host RAM for work the device could already do in place, and the fix makes the crossing cease to exist.

## The setting

The reopener came from the multi-GPU floor. That floor was the host-computed gradient stream, one boundary crossed every tree. The single-GPU engine crosses the same boundary, so it was the place to attack first (issue #171).

Here is the boundary, physically. Every tree, the host computes the gradients and hessians, then uploads a 128 MB packed array at 16M rows. It downloads the leaf values and leaf ids afterward, and updates the scores on the host. The device already holds the labels and the scores that all of it derives from.

## Price from existing counters

The price list needed no new instrumentation. It read from the `FitProfiler` and CUDA counters already on main.

The reachable pool is the objective, the score update, the gradient upload, and the finalize download. That summed to 35.9 ms/round at 16M, 22% of the fit, and 128 ms/round at 64M, 20% of the fit.

The pre-registered kill bar was 12 ms/round at 16M and 10% at 64M. The pool cleared it roughly threefold, so the campaign was licensed before a kernel was written.

The price list also flagged a bonus outside the pre-registered set. A full-data fit was refilling an identity vector that the engine ignores, at 4.4 ms/round. Rung 1 harvested it as an iota-once refill.

## The seam

The fix keeps three things resident and derives the rest on the card. Labels and initial scores upload once per fit. Each tree's packed gradient and hessian array is derived on the device from the resident scores by a two-line kernel.

The tree epilogue is the other half. A fused kernel routes every row through the finished tree in bin space and folds the leaf value straight into the resident scores. Per tree, nothing crosses the bus in either direction.

The seam is one named call, `try_resident_round`. Ineligible fits take the untouched host path, `BONSAI_HOST_OBJECTIVE=1` forces it, and the CPU plane is byte-identical by construction.

## The A/B measured more than the pool

The shipped branch was measured by a same-pod interleaved A/B. Interleaving the host and resident arms cancels drift. The env flag is the escape hatch that makes it a controlled experiment, because the same binary runs both arms.

| cell | host ms/round | resident ms/round | cut |
|---|--:|--:|--:|
| oblivious 16M | 136.1 | 102.6 | 24.6% |
| depthwise 16M | 148.7 | 118.7 | 20.1% |
| oblivious 64M | 545.2 | 455.7 | 16.4% |
| depthwise 64M | 606.3 | 515.8 | 14.9% |

The 16M cut is larger than that host's own pool of about 28 ms. The reason is the epilogue: routing each row through the tree in bin space is also cheaper than the stamp-and-copy epilogue it replaced. This host has a fast CPU, so weaker hosts should see larger cuts, not smaller.

## It proved identical

Speed means nothing here without parity, and parity was exact. The r2 was identical to every reported digit in all four pairs.

The stronger proof ran on a Jetson. Full-data resident predictions were bit-identical to the host-objective GPU model, at a maximum prediction difference of 0.00000. The change measures faster and proves identical.

The escape hatch made that parity checkable. `BONSAI_HOST_OBJECTIVE=1` runs the old path, so a reviewer diffs the two model files directly. The CPU plane never sees the seam, because the eligibility branch is a compile-time no-op for CPU growers.

## The frontier consequence

The frontier re-run then priced the whole effect from the outside (decision 78). On one L40S pod the oblivious marginal round fell from 104 to 64 ms, while the same-pod controls moved only with fleet variance.

That put bonsai's round below CatBoost's 78 ms on the same pod. The decision-72 crossover no longer exists at any measured horizon. The one honest residue that run had named, CatBoost reaching its plateau sooner, is also gone. bonsai reaches 0.8979 in 31.9s against CatBoost's 0.8980 in 46.4s, a fourth-decimal tie at 45% more wall clock.

The floor moved, but the discipline did not. The remaining round is histogram build plus partition plus find, with the objective boundary gone. That next floor is kernel engineering, which decision 72 left unspent, and no competitor pressure remains at this cell to spend it.

## The bonus the A/B exposed

Rung 2 widened eligibility to LogLoss, Poisson, and per-row sample weights (decision 79). The weighted A/B surfaced a bug two hops from the seam it was testing.

The weighted host path was paying a serial 16M-element weight multiply every tree, roughly 80 ms/round of single-threaded work. The loop is elementwise with no reduction, so it is now parallel and bitwise-identical. The weighted MSE cut read 22.2s to 10.4s, 53%, but a caveat travels with it. The 53% is against the pre-fix path, so the honest post-fix figure is nearer the unweighted cuts.

## The complexity ledger

The seam is cheap to carry, and the campaign measured that too. It added 616 non-test lines, about 380 in the CUDA plane and 240 in generic headers. It added zero config knobs and grew the registry by nothing.

The eligibility check is one shared function. Both `begin_root` and `resident_begin` apply the same capacity predicate, so the decline conditions cannot drift apart. The resident state arms per Dataset and disarms with a sync when the Dataset or a runtime gate changes.

One scaling caveat is recorded. The routing epilogue does O(depth) dependent loads per row, against the O(1) gather it replaced, and it measured cheaper at depth 8. If deep-tree fits ever regress there, the reopener is a hybrid epilogue.

## What stays on the host

Not every fit is eligible, and the boundaries are drawn by measurement. GOSS reads and reweights host gradients, so a device version is its own design. The renewal objectives, MAE, Huber, and Quantile, rebuild leaves from host residuals, so residency would download what it just avoided uploading.

Softmax stays host-side as well, because its per-class tree shape is a separate campaign. The eligible set is what the common case needs: MSE, LogLoss, or Poisson, weighted or not, with uniform or Bernoulli row sampling.

## What it teaches

- **Delete a boundary, do not optimize across it.** The campaign did not speed up the host round-trip. It removed the round-trip, and the cut exceeded its own price list because the replacement epilogue was cheaper too.
- **An env-flag A/B is a controlled experiment.** One binary ran both arms, interleaved on one pod to cancel drift, with `BONSAI_HOST_OBJECTIVE=1` as the escape hatch. That is what let the r2 parity and the weighted-loop bug both surface cleanly.
- **Instrument-first rails compound.** This campaign priced itself from counters an earlier campaign left in the code. No new instrumentation was written before the kill bar was cleared threefold.

## The record

- Decisions: [77](../../decisions.md) (the per-tree host round-trip deleted), [78](../../decisions.md) (the frontier re-run, unconditional at 16M), and [79](../../decisions.md) (LogLoss, Poisson, and sample weights).
- Evidence: [the resident-objective price list and A/B](../../../benchmarks/resident-objective-2026-07.md), and [the 16M GPU frontier](../../../benchmarks/gpu-pareto-16M-2026-07.md) superseded in place for the 104-to-64 ms round.
- Earlier: case [E3](3-the-parity-verdict.md) named this reopener, and case [E1](1-the-marginal-round.md) left the round at 104 ms.
