# E1. The marginal round

At 16M rows, one CatBoost boosting round took 77 milliseconds and ours took 155. This case is how we closed most of that gap without writing the kernel we set out to write. The discipline that did it is instrument-first: price a change from measurement before you build it, and treat every refutation as a result.

This round was one lap in a longer campaign. Earlier rounds had already taken the 16M fit from about 43 to 26.9 seconds and past XGBoost-GPU. What remained was the per-round cost, and that is where CatBoost still led.

## The setting

The cell is the largest in bonsai's suite: 16M rows by 100 features, depth 8, learning rate 0.1, 255 bins. Timing a single fixed iteration count hides the question users actually ask, which is how long to reach a given accuracy. So the frontier sweeps each library's iteration count and records (fit seconds, test r²) at every rung.

This lens is also honest about losses. A fixed-iteration table can hide a per-round deficit behind faster convergence, or hide slow convergence behind a cheap round. The frontier shows both, so you cannot dress up one axis by quoting the other.

Fit as a straight line, each library becomes a fixed cost plus a marginal cost per round. The fixed cost is ingest and setup, paid once. The marginal cost is one boosting round, paid per tree. On one L40S pod, bonsai's oblivious grower sat at ~4.6 seconds fixed plus 155 ms per round. CatBoost sat at ~11.8 seconds plus 77 ms.

The marginal round is the right target because it compounds. A fit spends it once per tree, so a third off the round is a third off every deep fit at this cell. That decomposition also tells you who wins where. bonsai owned the fast end outright: under half CatBoost's fixed cost, and first to every accuracy up to roughly r² 0.88. But CatBoost's round was roughly half the price, so any deep-enough workload went to CatBoost. Decision 71 named the target in one line: bonsai's 155 ms round against CatBoost's 77.

## The naive plan

The GPU grow profile pointed straight at a culprit. The split-find lap read 8.4 seconds, the largest single line in the profile. The find kernel runs an all-double warp scan, and double precision is weak on an L40S consumer-class card. The obvious plan was a multi-hour rewrite of that kernel in mixed precision.

The plan was reasonable. The profile was not lying about the wall clock; the lap really did take 8.4 seconds. It was lying about attribution, and attribution is the only thing an optimization can act on.

## Instrument first

We did not write the rewrite. We wrote a measurement first.

Here is one oblivious level, physically. The host queues the next level's memset, histogram, and subtract kernels, then returns. It commits the level's children and demotes empty nodes, host bookkeeping that overlaps the device work. Then it opens the next level and stages a kilobyte of node sums, a synchronous copy. That copy cannot start until the queued histogram build ahead of it finishes, so the host blocks there.

The wait belongs to the build, but the label said staging. Async work bills to whoever synchronizes next.

The instrument that separates them is a sync peel, compiled out unless a profile flag is set. Three instruments see this loop differently. A host lap times wall clock between two host points, including any wait it happens to absorb. A `cudaEvent` pair times only the kernels between two stream points, never the wait. The peel is one `cudaDeviceSynchronize`, lapped by the host, that converts "absorbed somewhere" into a named bucket.

So the division of labor: the peel tells you the wait, the events tell you the work, and production keeps neither.

## What the measurement said

With the peel in place, the numbers separated. The find kernel costs 0.17 seconds at 16M, not 8.4. The 8.4 seconds was the profiler's opening sync catching the previous level's histogram kernels as they drained.

The real GPU cost was the histogram build. It already sums in float shared memory, using double only for a bounded cross-chunk merge, so the obvious precision lever was already spent. Every banked hypothesis for the kernel rewrite was refuted by one measurement. The rewrite was cancelled before any kernel code existed.

This also fixed the target. With the histogram build as the real cost, the round has a floor. It is ingest transfer plus kernel compute plus one tiny per-level sync the control plane cannot avoid. When the levers that remain sit within noise of that floor, the placement game is over by arithmetic, and only kernel engineering is left.

So the question reframed. The find kernel was never the cost. The 155 ms round was, and it is a scheduling and boundary problem, not a math-precision one.

## The price list

The campaign then attacked the 155 ms round the same way. Rung 0 built a price list before touching a single lever.

A price is arithmetic, not a guess. Each candidate change moves or deletes an edge in the training DAG. An edge that crosses the host/device boundary costs its bytes over the measured bandwidth. State that price from same-pod constants first; play the lever only if it wins across the plausible range.

The DAG is small enough to make this exact. It has about ten node types, and at most six have a free host-or-device placement. So enumerating placements is trivial; the entire difficulty is honest constants.

Conservation is the rule that found the levers. The buckets must sum to the wall clock at every altitude. When they sum but a line is physically impossible, that line is mislabeled; when they do not sum, the gap is the next target.

The peel replayed the exact decision-62 misattribution on the oblivious plane: 6.1 seconds filed under find-staging was again the previous level's histogram kernels draining. With that wait relocated, two residues fell out that no label explained. One was a 64MB-per-tree host copy of the identity row list, at 33 ms per round. The other was a full histogram build for a final level whose children are all leaves, at 22 ms per round.

Six levers were priced from that one table, each with a kill criterion written down first. Three cleared the bar and landed.

- **The identity contract.** Full-data fits pass empty rows plus a count. The engine builds and caches the identity permutation on device, deleting the host copy. Priced at 33 ms.
- **Device root sums.** A 16M-row host reduction becomes a fixed-grid two-pass device kernel. Priced at ~12 ms, and the reduce stays deterministic.
- **The final-level skip.** The last level keeps only the bookkeeping that stamping needs, deleting a histogram build that nothing reads. Priced against a 22 ms cost.

Three levers were killed by their pre-registered criteria. The epilogue sync scope came in at 0.1 ms, a per-level memset at 0.5 ms, and pinned staging at break-even. The three kills cost a combined price under one millisecond to refute, because the table priced them before anyone wrote a kernel.

## The outcome

Same pod, the profiled round fell from 181 to 125 ms and the 16M fit from 19.43 to 13.88 seconds. Two gates guarded the change: r² identical to four decimals, and the CPU model hash byte-identical. A GPU-only change must not move the host plane, and this one did not.

Measured from the outside, the frontier's least-squares marginal round fell from 155 to 104 ms. That moved the bonsai-CatBoost crossover from about round 100 to about round 320, which sits inside both libraries' accuracy plateaus. bonsai became first to every measured accuracy up to r² ~0.895: 0.8749 in 13.9s against CatBoost's 19.7s, 0.8948 in 24.5s against 27.9s. The 300-iteration points were a statistical tie, 0.8974 in 35.3s against CatBoost's 0.8973 in 35.1s. The ceiling stayed bonsai's by a rounding digit, 0.8981 against 0.8980.

The frontier crossing had moved past the plateau, which is the whole point: the number that beat you no longer occurs inside the useful range. The remaining 104 ms round is about 72 ms of histogram kernel plus 32 ms of partition and bus. That is a kernel-engineering boundary, not a placement one. The campaign's last recorded act was not spending the kernel rung. The ship bar of 110 ms was met; the crown bar of 77 ms was not; and the frontier no longer needed it.

One caveat travels with these numbers. Two identical L40S pods measured 25% apart, so every delta above is same-pod. The cross-pod number that transfers is the decomposition shape, not the absolute milliseconds.

A pointer forward: decision 78 moved this frontier again. When the device-resident objective removed the per-tree host round-trip, the same oblivious round fell from 104 to 64 ms on a July 18 re-run. That is cheaper than CatBoost's round, and the crossover disappeared entirely. That story is case E4.

## Reading it yourself

You can watch this loop on any machine with a CUDA device. Set the grow, cuda, fit, and ingest profile flags on a fit. Then check conservation: does the fit total explain the wall clock, and does grow equal the sum of its buckets? A perfectly explained clock with a physically impossible line item is a misattribution; an unexplained gap is the next target.

## What it teaches

- **Profilers bill async work to the wrong line.** The 8.4-second find was a histogram build draining at the next sync. A number you cannot decompose is a number you cannot trust. A kilobyte copy that reads 61 ms is already telling you it is somebody else's time, because bytes over bandwidth cannot cost that.
- **A price list kills more ideas than it approves.** This round priced six levers and a whole kernel rewrite. It approved three levers, killed three more, and had already killed the kernel rewrite that started it. The refutation is the deliverable, and it is cheapest at the price-list stage.
- **Pre-registered kill criteria prevent sunk-cost shipping.** Each lever had its kill number written before it was measured. Three levers died for a combined cost under a millisecond, instead of after a day of implementation you would then feel obliged to justify.

## The record

- Decisions: [62](../../decisions.md) (the cancelled kernel round), [71](../../decisions.md) (the target named), and [72](../../decisions.md) (the campaign and its close-out).
- Evidence: [the 16M GPU frontier](../../../benchmarks/gpu-pareto-16M-2026-07.md), superseded in place, with git history holding the decision-72 measurement. [Guide chapter 11](../../guide/11-performance-engineering.md) walks the same round step by step.
- Later: [decision 78](../../decisions.md) re-ran the frontier after the device-resident objective and cut the round to 64 ms.
