# The device-resident objective: price list and same-pod A/B (2026-07)

Campaign issue #171, decision 77. Hypothesis: every tree, the objective round-trips host RAM (host grad/hess compute, 128MB gh upload at 16M rows, 128MB values and leaf-id download, host score update) when the device already holds everything needed to do it in place.

Protocol: campaign spec throughout (cols=100, bins=255, depth=8, lr=0.1, informative=20, n_test=500k, seed=42, threads=16), synthetic regression via `scripts/bench_scaling.py --worker`, all profile layers on. Same-pod comparisons only; cross-pod absolutes are not comparable (~25% fleet spread).

## Rung 0: the price list (L40S US-NC-1, EPYC 9354 host, pod bonsai-rung0-171)

Pool = objective + score + gh upload + finalize D2H, read from existing `FitProfiler` and CUDA counters on main at d9ede3f. 16M at 100 iters (oblivious twice, 1.4% fit spread, r2 identical), 64M at 60 iters.

| cell | fit_s | objective | score | gh H2D | fin D2H | pool ms/round | share of fit |
|---|---|---|---|---|---|---|---|
| obl 16M (mean of 2) | 16.14 | 5.0 | 3.6 | 14.6 | 12.8 | 35.9 | 22.2% |
| dw 16M | 17.55 | 4.5 | 3.8 | 14.1 | 11.6 | 34.0 | 19.4% |
| obl 64M | 39.29 | 24.3 | 11.5 | 46.3 | 46.0 | 128.2 | 19.6% |
| dw 64M | 42.55 | 24.8 | 11.2 | 43.8 | 45.2 | 125.0 | 17.6% |

Pre-registered kill (pool < 12ms/round at 16M and < 10% at 64M): not triggered, cleared roughly threefold. Bonus pool outside the pre-registered set: `sample` = 4.4ms/round at 16M (AllRowsSampler refilling an identity vector the engine ignores for full-data fits), harvested in rung 1 as the iota-once refill.

## Rung 1: the same-pod interleaved A/B (L40S US-MO-1, pod bonsai-rung1-ab-171)

Branch `perf/resident-objective` at 90e5a4a. Arms interleaved host/resident to cancel drift; host = `BONSAI_HOST_OBJECTIVE=1`, resident = default eligibility. 16M oblivious ran twice per arm.

| cell | host ms/round | resident ms/round | cut | cut % |
|---|---|---|---|---|
| obl 16M (mean of 2) | 136.1 | 102.6 | 33.5 | 24.6% |
| dw 16M | 148.7 | 118.7 | 29.9 | 20.1% |
| obl 64M | 545.2 | 455.7 | 89.6 | 16.4% |
| dw 64M | 606.3 | 515.8 | 90.5 | 14.9% |

Ship bar (>= 15ms/round cut at 16M): met at 2.2x. r2_train and r2_test identical to every reported digit in all four pairs (both 16M oblivious repeats included); peak RSS unchanged; the resident arm's fit-profile reads objective=0.00 score=0.00.

The cut exceeds this host's own pool (~28ms on its fat-CPU shape) because the routing epilogue (one kernel walking each row through the finished tree in bin space, fusing the score update) is also cheaper than the stamp-and-copy epilogue it replaced. This host's CPU minimizes the host-leg share, so weaker hosts should see larger cuts, not smaller.

## Parity evidence (Jetson Orin, sm_87)

Full-data resident vs host-objective GPU: bit-identical predictions (dr2 ~ 1e-9, max pred diff 0.00000) for depthwise, oblivious, warm start, and the escape hatch. Bernoulli sampling: dr2 1.1e-4 from the root-sum reduction-order gap (host reduction vs device two-pass over the gathered subset), equal accuracy, band documented in the test. CPU plane byte-identical (`model_hash.py` serial 09dbf47353033362, sampled ca7174cb1560221e, equal to main).

## Verdict

Adopted (decision 77). The same-share projection puts the decision-72 frontier round (104ms, L40S US-NC-1) near the 77ms stretch bar; confirming that requires a frontier ladder re-run on one pod, which supersedes the pareto artifacts in place per decision 69.

Raw session logs archived with the campaign issue; cells reproduce from the committed protocol above.
