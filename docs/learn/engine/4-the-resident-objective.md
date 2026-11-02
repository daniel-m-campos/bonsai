# E4. The resident objective

Case [E3](3-the-parity-verdict.md) stopped work on a multi-GPU engine at its floor, and one cost in that floor was the host computing gradients and hessians every tree. The way to remove that cost was to move the objective (the step that computes each row's gradient and hessian from the labels and scores) onto the GPU, so labels and scores stay in GPU memory. This case does that on a single GPU first, and the 16M-row levelwise round falls from 104 to 64 milliseconds.

The move is not to optimize the host round-trip. It is to delete it. Every tree, the host sent data through RAM for work the GPU could do itself. Now nothing crosses per tree.

## The setting

The idea of moving the objective onto the GPU came from the multi-GPU floor. That floor was the host-computed gradient stream, one boundary crossed every tree. The single-GPU engine crosses the same boundary, so it was the place to attack first (issue #171).

Here is the boundary, physically. Every tree, the host computes the gradients and hessians, then uploads a 128 MB packed array at 16M rows. It downloads the leaf values and leaf ids afterward, and updates the scores on the host. The device already holds the labels and the scores that all of it derives from.

## Price from existing counters

The price list needed no new instrumentation. It read from the `FitProfiler` and CUDA counters already on main.

The reachable pool is the objective, the score update, the gradient upload, and the finalize download. That summed to 35.9 ms/round at 16M, 22% of the fit, and 128 ms/round at 64M, 20% of the fit.

The approval threshold, set before measuring, was 12 ms/round at 16M rows and 10% of the fit at 64M. The measured 35.9 ms/round at 16M is three times the first, and 20% of the fit at 64M is twice the second, so the work was approved before any kernel was written.

The price list also flagged a bonus outside the pre-registered set. A full-data fit was refilling an identity vector that the engine ignores, at 4.4 ms/round. Stage 1 harvested it as an iota-once refill.

## The seam

The fix keeps the labels and scores in GPU memory (called resident below) and computes each tree's gradients and hessians there. Labels and initial scores upload once per fit. Each tree's packed gradient and hessian array is derived on the device from the resident scores by a two-line kernel.

The tree epilogue is the other half. A fused kernel routes every row through the finished tree in bin space and folds the leaf value straight into the resident scores. Per tree, nothing crosses the bus in either direction.

The seam is one named call, `try_resident_round`. Ineligible fits take the untouched host path. Setting `BONSAI_HOST_OBJECTIVE=1` makes an eligible fit take it too, at the cost shown in the table below, and the CPU plane is byte-identical by construction.

## The A/B measured more than the pool

The shipped branch was measured by a same-pod interleaved A/B. Interleaving the host and resident arms cancels drift. The env flag is what makes it a controlled experiment, because the same binary runs both arms.

| cell | host ms/round | resident ms/round | cut |
|---|--:|--:|--:|
| levelwise 16M | 136.1 | 102.6 | 24.6% |
| depthwise 16M | 148.7 | 118.7 | 20.1% |
| levelwise 64M | 545.2 | 455.7 | 16.4% |
| depthwise 64M | 606.3 | 515.8 | 14.9% |

The 16M cut is larger than that host's own pool of about 28 ms. The reason is the epilogue: routing each row through the tree in bin space is also cheaper than the stamp-and-copy epilogue it replaced. This A/B host's CPU keeps the host share of the round small. A slower host CPU spends longer in the host path, so you should expect a larger cut there, but this page measured only that one host.

## It proved identical

Speed means nothing here without parity, and parity was exact. The r2 was identical to every reported digit in all four pairs.

The stronger proof ran on a Jetson. Full-data resident predictions were bit-identical to the host-objective GPU model, at a maximum prediction difference of 0.00000. The resident path cut 14.9 to 24.6% per round across the four cells above and gave bit-identical predictions on the Jetson.

The flag made that parity checkable. `BONSAI_HOST_OBJECTIVE=1` runs the old path, so a reviewer diffs the two model files directly. The CPU plane never sees the seam, because the eligibility branch is a compile-time no-op for CPU growers.

## The frontier consequence

The frontier re-run then priced the whole effect from the outside (decision 78). On one L40S pod the levelwise marginal round fell from 104 to 64 ms, while the same-pod controls moved only with fleet variance.

That put bonsai's round below CatBoost's 78 ms on the same pod. The decision-72 crossover no longer exists at any measured horizon. The one honest residue that run had named, CatBoost reaching its plateau sooner, is also gone. bonsai reaches 0.8979 in 31.9s against CatBoost's 0.8980 in 46.4s, a fourth-decimal tie at 45% more wall clock.

The floor moved, but the discipline did not. The remaining round is histogram build plus partition plus find, with the objective boundary gone. That next floor is kernel engineering, which decision 72 left unspent, and no competitor pressure remains at this cell to spend it.

## The bonus the A/B exposed

Stage 2 widened eligibility to LogLoss, Poisson, and per-row sample weights (decision 79). The weighted A/B surfaced a bug two hops from the seam it was testing.

The weighted host path was paying a serial 16M-element weight multiply every tree, roughly 80 ms/round of single-threaded work. The loop is elementwise with no reduction, so it is now parallel and bitwise-identical. The weighted MSE cut read 22.2s to 10.4s, 53%, but the figure needs a qualifier. The 53% compares against the host path before the parallel weight loop was added. No weighted A/B was re-run against the fixed host path, so the post-fix cut is unmeasured. The unweighted cuts in the same A/B at 16M rows were 25.7% (LogLoss) and 25.8% (Poisson), and the weighted cut against the fixed path is expected to be closer to those than to 53%.

## The complexity ledger

The change adds 616 non-test lines, about 380 in the CUDA plane and 240 in generic headers. It adds no configuration parameters and no registry entries.

The eligibility check is one shared function. Both `begin_root` and `resident_begin` apply the same capacity predicate, so the decline conditions cannot drift apart. The resident state arms per Dataset and disarms with a sync when the Dataset or a runtime gate changes.

One scaling caveat is recorded. The routing epilogue does O(depth) dependent loads per row, against the O(1) gather it replaced, and it measured cheaper at depth 8. If deep-tree fits ever regress there, the follow-up to try is a hybrid epilogue.

## What stays on the host

Not every fit is eligible, and the boundaries are drawn by measurement. GOSS reads and reweights host gradients, so a device version is its own design. At the time of this campaign the renewal objectives, MAE, Huber, and Quantile, stayed on the host, because they rebuild leaves from host residuals and residency would download what it just avoided uploading. Decision 153 later moved that renewal onto the device, so they are eligible now, except for an oblivious tree under a monotone constraint, whose leaf table the host reprojects.

Softmax stays host-side as well, because its per-class tree shape is a separate campaign. The eligible set today is MSE, LogLoss, Poisson, MAE, Huber, or Quantile, weighted or not, with uniform or Bernoulli row sampling and without DART. `BONSAI_HOST_OBJECTIVE=1` forces the host path.

## What it teaches

- **Delete a boundary, do not optimize across it.** The campaign did not speed up the host round-trip. It removed the round-trip, and the cut exceeded its own price list because the replacement epilogue was cheaper too.
- **An env-flag A/B is a controlled experiment.** One binary ran both arms, interleaved on one pod to cancel drift, with `BONSAI_HOST_OBJECTIVE=1` selecting the host arm. That is what let the r2 parity and the weighted-loop bug both surface cleanly.
- **Counters built for one campaign price the next.** This campaign used the existing `FitProfiler` and CUDA counters and wrote no new instrumentation before the pool cleared its 16M threshold threefold.

## The record

- Decisions: [77](../../decisions.md) (the per-tree host round-trip deleted), [78](../../decisions.md) (the frontier re-run, unconditional at 16M), and [79](../../decisions.md) (LogLoss, Poisson, and sample weights).
- Evidence: [the resident-objective price list and A/B](../../../benchmarks/resident-objective-2026-07.md), and [the 16M GPU frontier](../../../benchmarks/gpu-pareto-16M-2026-07.md) superseded in place for the 104-to-64 ms round.
- Earlier: case [E3](3-the-parity-verdict.md) named this follow-up, and case [E1](1-the-marginal-round.md) left the round at 104 ms.
