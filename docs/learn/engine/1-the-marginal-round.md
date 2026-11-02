# E1. The marginal round

At 16M rows, one CatBoost boosting round took 77 milliseconds and one bonsai round took 155. This case cut the bonsai round to 104 milliseconds, after an earlier round had cancelled the kernel rewrite we set out to write. The discipline that did it is instrument-first: price a change from measurement before you build it, and treat every refutation as a result.

This round was one lap in a longer campaign. Earlier rounds had already cut the 16M x 100 fit from about 43 to 26.9 seconds, and XGBoost-GPU measured 27.9 seconds in [decision 52](../../decisions.md). What remained was the per-round cost, and that is where CatBoost still led.

## The setting

When this round ran, the cell was the largest in bonsai's suite: 16M rows by 100 features, depth 8, learning rate 0.1, 255 bins. Timing a single fixed iteration count hides the question users actually ask, which is how long to reach a given accuracy. So the frontier sweeps each library's iteration count and records (fit seconds, test r²) at each iteration count.

This lens is also honest about losses. A fixed-iteration table can hide a per-round deficit behind faster convergence, or hide slow convergence behind a cheap round. The frontier shows both, so you cannot dress up one axis by quoting the other.

Fit as a straight line, each library becomes a fixed cost plus a marginal cost per round. The fixed cost is ingest and setup, paid once. The marginal cost is one boosting round, paid per tree. On one L40S pod, bonsai's levelwise grower sat at ~4.6 seconds fixed plus 155 ms per round. CatBoost sat at ~11.8 seconds plus 77 ms.

The marginal round is the right target because it compounds. A fit spends it once per tree, so a third off the round is a third off every deep fit at this cell. That decomposition also tells you who wins where. bonsai was first to every accuracy up to roughly r² 0.88, and its fixed cost was under half of CatBoost's (4.6 s against 11.8 s). But CatBoost's round was roughly half the price (77 ms against 155 ms), so past about round 100 CatBoost was the faster fit. Decision 71 named the target in one line: bonsai's 155 ms round against CatBoost's 77.

## The naive plan

The GPU grow profile pointed straight at a culprit. The split-find lap read 8.4 seconds, the largest single line in the profile. The find kernel runs a warp scan (a running sum across the 32 threads of a GPU warp) entirely in double precision, and the L40S runs double precision at 1/64 of its single-precision rate. The obvious plan was a multi-hour rewrite of that kernel in mixed precision.

The plan was reasonable. The profile was not lying about the wall clock; the lap really did take 8.4 seconds. It was lying about attribution, and attribution is the only thing an optimization can act on.

## Instrument first

We did not write the rewrite. We wrote a measurement first.

Here is one levelwise level, physically. The host queues the next level's memset, histogram, and subtract kernels, then returns. It commits the level's children and demotes empty nodes, host bookkeeping that overlaps the device work. Then it opens the next level and stages a kilobyte of node sums, a synchronous copy. That copy cannot start until the queued histogram build ahead of it finishes, so the host blocks there.

The wait belongs to the histogram build, but the profile labeled it staging. Work queued on the device while the host moves on (asynchronous work) is timed at the next call that waits for it.

The instrument that separates them is a sync peel: one `cudaDeviceSynchronize` call that the host times, so the wait shows up as its own line. It ships in every CUDA build, disabled, and runs only when the `BONSAI_CUDA_PROFILE` environment variable is set. Three instruments see this loop differently. A host lap times wall clock between two host points, including any wait it happens to absorb. A `cudaEvent` pair times only the kernels between two stream points, never the wait. The peel times the wait alone, which turns time that was absorbed into some other label into a line of its own.

So the division of labor: the peel tells you the wait, the events tell you the work, and a production run leaves both switched off.

## What the measurement said

With the peel in place, the numbers separated. The find kernel costs 0.17 seconds at 16M, not 8.4. The 8.4 seconds was the profiler's opening sync catching the previous level's histogram kernels as they drained.

The real GPU cost was the histogram build. It already sums in float shared memory, using double only for a bounded cross-chunk merge, so the obvious precision lever was already spent. Every banked hypothesis for the kernel rewrite was refuted by one measurement. The rewrite was cancelled before any kernel code existed.

This also fixed the target. With the histogram build as the real cost, the round has a floor. It is ingest transfer plus kernel compute plus one per-level sync the control plane cannot avoid. When every remaining host-or-device placement saves less than the run-to-run noise of the round, moving work between host and device cannot help, and only faster kernels can.

So decision 62 reframed the question, from kernel speed to the whole accuracy-vs-time frontier. The find kernel was never the cost. Once decision 71 re-measured the frontier, the 155 ms marginal round was the target, and it is a scheduling and boundary problem, not a math-precision one.

## The price list

The campaign then attacked the 155 ms round the same way. Stage 0 built a price list before touching a single lever.

A price is arithmetic, not a guess. Each candidate change moves or deletes an edge in the training DAG (directed acyclic graph of steps). An edge that crosses the host/device boundary costs its bytes over the measured bandwidth. State that price from same-pod constants first; play the lever only if it wins across the plausible range.

The DAG is small enough to make this exact. It has about ten node types, and at most six have a free host-or-device placement. That gives at most 64 placements to enumerate (two choices for each of six nodes), so the work is measuring the constants, not searching the placements.

The sum check is the rule that found the levers: the timed parts must add up to the whole. The fit total must explain the wall clock, and the grow total must equal the sum of its buckets. When they sum but a line is physically impossible, that line is mislabeled; when they do not sum, the gap is the next target.

The peel replayed the exact decision-62 misattribution on the levelwise plane: 6.1 seconds timed as find-staging was again the previous level's histogram kernels draining. With that wait relocated, two residues fell out that no label explained. One was a 64MB-per-tree host copy of the identity row list, at 33 ms per round. The other was a full histogram build for a final level whose children are all leaves, at 22 ms per round.

Six levers were priced from that one table, each with a kill criterion written down first. Three cleared the bar and landed.

- **The identity contract.** Full-data fits pass empty rows plus a count. The engine builds and caches the identity permutation on device, deleting the host copy. Priced at 33 ms.
- **Device root sums.** A 16M-row host reduction becomes a fixed-grid two-pass device kernel. Priced at ~12 ms, and the reduce stays deterministic.
- **The final-level skip.** The last level keeps only the bookkeeping that stamping needs, deleting a histogram build that nothing reads. Priced against a 22 ms cost.

Three levers were killed by their pre-registered criteria. The epilogue sync scope came in at 0.1 ms, a per-level memset at 0.5 ms, and pinned staging at break-even. Together the three levers were worth under one millisecond, and the table showed that before anyone wrote a kernel.

## The outcome

Same pod, the profiled round fell from 181 to 125 ms and the 16M fit from 19.43 to 13.88 seconds. Two gates guarded the change: r² identical to four decimals, and the CPU model hash byte-identical. A GPU-only change must not move the host plane, and this one did not.

Measured from the outside, the frontier's least-squares marginal round fell from 155 to 104 ms. That moved the bonsai-CatBoost crossover from about round 100 to about round 320, which sits inside both libraries' accuracy plateaus. bonsai became first to every measured accuracy up to r² ~0.895: 0.8749 in 13.9s against CatBoost's 19.7s, 0.8948 in 24.5s against 27.9s. The 300-iteration points were a statistical tie, 0.8974 in 35.3s against CatBoost's 0.8973 in 35.1s. The ceiling stayed bonsai's by a rounding digit, 0.8981 against 0.8980.

CatBoost now overtakes bonsai only at about round 320, where both libraries have stopped improving. The remaining 104 ms round is about 72 ms of histogram kernel plus 32 ms of partition and bus. That is a kernel-engineering boundary, not a placement one. We stopped at 104 ms and did not start the kernel-engineering stage. The 110 ms ship bar was met, the 77 ms crown bar was not, and the frontier no longer needed it.

One caveat travels with these numbers. Two identical L40S pods measured 25% apart, so every delta above is same-pod. The cross-pod number that transfers is the decomposition shape, not the absolute milliseconds.

A pointer forward: decision 78 moved this frontier again. When the device-resident objective removed the per-tree host round-trip, the same levelwise round fell from 104 to 64 ms on a July 18 re-run. That is cheaper than CatBoost's round, and the crossover disappeared entirely. That story is case E4.

## Reading it yourself

You can watch this loop on any machine with a CUDA device. Set the environment variables `BONSAI_GROW_PROFILE`, `BONSAI_CUDA_PROFILE`, `BONSAI_FIT_PROFILE` and `BONSAI_INGEST_PROFILE` to `1` when you run a fit (the full command is in [guide chapter 11, Try it](../../guide/11-performance-engineering.md#try-it)). Then check that the buckets add up: does the fit total explain the wall clock, and does grow equal the sum of its buckets? A perfectly explained clock with a physically impossible line item is a misattribution; an unexplained gap is the next target.

## What it teaches

- **Profilers time asynchronous work at the wrong line.** The 8.4-second find was a histogram build draining at the next sync. A number you cannot decompose is a number you cannot trust. A kilobyte copy that reads 61 ms is already telling you it is somebody else's time, because bytes over bandwidth cannot cost that.
- **A price list kills more ideas than it approves.** This round priced six levers and a whole kernel rewrite. It approved three levers, killed three more, and had already killed, in the earlier decision 62 round, the kernel rewrite that started this line of work. The refutation is the deliverable, and it is cheapest at the price-list stage.
- **Pre-registered kill criteria prevent sunk-cost shipping.** Each lever had its kill number written before it was measured. Three levers died with a combined projected saving under a millisecond, instead of after a day of implementation you would then feel obliged to justify.

## The record

- Decisions: [62](../../decisions.md) (the cancelled kernel round), [71](../../decisions.md) (the target named), and [72](../../decisions.md) (the campaign and its close-out).
- Evidence: [the 16M GPU frontier](../../../benchmarks/gpu-pareto-16M-2026-07.md), superseded in place, with git history holding the decision-72 measurement. [Guide chapter 11](../../guide/11-performance-engineering.md) summarizes the same round among the campaign's moves.
- Later: [decision 78](../../decisions.md) re-ran the frontier after the device-resident objective and cut the round to 64 ms.
