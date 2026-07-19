# The single-card ceiling: 500M rows x 100 features on one 80GB GPU (2026-07)

The u8-bin arithmetic has always implied that one 80GB card holds a half-billion-row dataset; this measures it. One RunPod A100-SXM4-80GB, one process, a 500M x 100 float32 Friedman-1 matrix (200GB raw) generated to a disk memmap, and an ascending ladder of `bonsai.train` calls on row-prefix views with the device-resident MSE objective (`cuda_oblivious`, 60 iterations, depth 8, 255 bins).

## The ladder

| rows | train() wall | peak device memory | throughput (rows x iters / s) | train r2 (1M sample) |
|--:|--:|--:|--:|--:|
| 300M | 328.5s | 42.2 GiB | 54.8M rows/s | 0.8305 |
| 400M | 414.2s | 56.0 GiB | 57.9M rows/s | 0.8303 |
| 450M | 465.1s | 63.0 GiB | 58.0M rows/s | 0.8340 |
| **500M** | **512.6s** | **69.9 GiB** | **58.5M rows/s** | 0.8329 |

Every rung passed; the ladder never hit the card. Wall time is the whole `train()` call: chunked host-to-device ingest, device binning, mapper fit, and all 60 boosting rounds. Device memory grows at ~13.9 GiB per 100M rows (1 byte per cell of binned matrix plus ~4.2 bytes per row of resident fit state), which extrapolates the true ceiling to roughly 570M rows x 100 on this card.

## Reading it

The 500M fit runs 8.5 minutes end to end on one card, and throughput RISES with scale (54.8M to 58.5M row-iterations per second) because the fixed costs amortize; there is no cliff anywhere on the ladder. The claim this measures: **a half-billion-row, hundred-feature dataset is a single-GPU workload for bonsai, with about 8 GiB to spare.** For contrast, xgboost's GPU histogram method wants roughly 8 bytes per cell of gradient-capable storage before ExtMemQuantile spilling, and catboost's GPU pool sizing is similarly float-dominated; neither library's documentation claims this cell on one 80GB device without external-memory modes. We state that as a documentation-level contrast, not a same-pod measurement; the same-pod frontier comparisons live in `gpu-pareto-16M-2026-07.md`.

The device-resident objective (decisions 77-79) is what keeps the fit loop bus-free at this scale: per tree, gradients derive on device from resident scores and labels, and nothing per-row crosses the PCIe bus. Host RAM held only the memmap page cache; the binned matrix never materializes host-side on the CUDA ingest path.

## Honest caveats

Synthetic data (the bench Friedman-1 recipe, chunked with a pilot-chunk noise scale), one pod, one run per rung; wall times carry the usual ~25% fleet spread and the r2 column is a 1M-row train-sample proxy, not a held-out test. The ~570M extrapolated ceiling is arithmetic beyond the last measured rung, not a measurement. Ingest dominates the wall time at this scale (the 60-round boosting portion is a minority of the 512.6s); workloads reusing a `bonsai.Dataset` across fits pay it once.

Protocol and raw rows: `results/single-card-ceiling-2026-07.jsonl`; the generator ladder script lives with the campaign records (issue #171 lineage).
