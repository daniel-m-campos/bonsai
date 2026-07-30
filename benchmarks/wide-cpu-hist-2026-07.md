# The wide-CPU histogram wall: measured, routed, bounded (decision 88)

2026-07-30, issue #217, from Daniel's production field report (Xeon 6521P, leafwise CPU, 131k x 16384: LightGBM ahead 2-3x, matching the committed multi-host scaling rows). Raw rows: [`results/wide-cpu-hist-2026-07.jsonl`](results/wide-cpu-hist-2026-07.jsonl). Change: `src/grower.cpp` routes u8 levels whose selected-histogram footprint exceeds 24MB through the existing feature-parallel fill (one thread per feature, no partial slabs) instead of the row-wise fill.

## The price list (stage 0)

At 131k x 4096 on M2/8t, `populate` was 84-88s of a ~100s fit: the row-wise fill's per-row scatter targets `total_cells x 8B` of histogram (8.4MB at 4096 cols, 33.6MB at 16384), so past the cache every add misses, and the partial slabs add a zero+merge pass per block. The find, partition, and bookkeeping stages were noise by comparison.

## The measurements

**M2/8t crossover ladder** (131k rows, 20 iters, depthwise): row path wins at 1024 cols (8.4s vs 11.5s) and 2048 (20.6 vs 24.2); feature-parallel wins at 4096 (44.9 vs 101.7, with LightGBM at 53.7).

**Same-pod before/after** (L40S host, 128-core EPYC, t=16, 100 iters, SCALING knobs):

| cell | variant | before | after | lgbm_cpu | speedup |
|---|---|---|---|---|---|
| 131k x 16384 | depthwise | 1019.1s | 379.3s (repro 388.3, 399.8) | 366.9s | 2.6-2.7x |
| 131k x 16384 | leafwise | 2591.4s | 445.3s | 366.9s | 5.8x |
| 1M x 4096 | depthwise | 348.1s | unchanged (row path) | 381.9s | - |

Identical r² per cell in every run. Peak RSS at the 16k cell: bonsai 18.8GB vs LightGBM's 50.1GB (2.7x less). The 7x leafwise deficit against LightGBM collapses to 1.2x; depthwise reaches parity.

**The interleaved A/B that set the threshold** (1M x 4096, alternating builds, twice each, single worker): main 320.2/337.2s vs all-feature-parallel 547.1/547.5s. Mid-width on a big-L3 host belongs to the row path: the EPYC's L3 absorbs an 8.4MB scatter target, and the feature-parallel fill's scattered `bins[rows[k]]` column reads on sparse child nodes cost more than they save. A per-node rows bound (256k) was built and withdrawn on the same evidence (it kept the regression, 573.6s). One footprint threshold at 24MB flips only the ultra-wide shapes that lost on every measured host.

**No-regression gates**: the fixed-input model hash is byte-identical to main (the narrow path is code-identical below the threshold); 526/526 C++ tests; 67/67 Python tests. Above the threshold, models change bytes (a different, and now thread-count-invariant, summation order) at identical accuracy; the feature-parallel fill is bit-identical at any thread count, which the row path never was.

## Honest residuals

- The shipped 24MB threshold leaves the M2's mid-width win (44.9 vs 101.7s at 131k x 4096) on the table: small-cache hosts hit the wall earlier than the threshold assumes. A cache-size-aware threshold is the recorded follow-up on issue #217, the same lever XGBoost 3.3 shipped as aarch64 cache detection.
- The shipped-build 1M x 4096 confirmation read 366.7s against main's own 320-348 spread on the same pod hours apart; the path is code-identical at that width, and the pod's session drift covers the difference. Rows tagged `after-per-node` and the contaminated double-launch batch (two concurrent workers) are excluded from the tables; the A/B is the controlled comparison.
- The row-major mirror (2.1GB at 131k x 16384) is still built even when every level routes feature-parallel; skipping it for always-wide fits is a possible memory follow-up.
- The CUDA planes have their own wide wall (~5x behind xgb_cuda at the 16k cell) that this change does not touch.
