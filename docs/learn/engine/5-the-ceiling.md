# E5. The ceiling

The u8-bin arithmetic had always implied that one 80GB card holds a half-billion-row dataset. This case is one pod checking whether that was a promise or a wall.

The answer is a promise, and it comes with a table. A 500M-row by 100-feature matrix trained end to end on a single card, and the throughput rose the whole way up.

## The claim from arithmetic

bonsai stores one byte per binned cell, so a 500M by 100 matrix is about 50 GB of binned data. Add the resident fit state and it still fits inside 80 GB. The arithmetic said the cell was reachable; a measurement had to confirm it.

## The ladder

One RunPod A100-SXM4-80GB ran one process. A 500M by 100 float32 Friedman-1 matrix, 200 GB raw, was generated to a disk memmap. Then an ascending ladder of `bonsai.train` calls ran on row-prefix views with the device-resident MSE objective, at 60 iterations, depth 8, and 255 bins.

| rows | train() wall | peak device memory | throughput | train r2 (1M sample) |
|--:|--:|--:|--:|--:|
| 300M | 328.5s | 43.2 GiB | 54.8M rows/s | 0.8305 |
| 400M | 414.2s | 57.4 GiB | 57.9M rows/s | 0.8303 |
| 450M | 465.1s | 64.5 GiB | 58.0M rows/s | 0.8340 |
| **500M** | **512.6s** | **71.6 GiB** | **58.5M rows/s** | 0.8329 |

Every rung passed and the ladder never hit the card. Device memory grows at about 14.2 GiB per 100M rows. At 500M that is 71.6 GiB of the 80, roughly 8 GiB to spare.

## Reading it

The 500M fit runs 8.5 minutes end to end on one card. The wall time is the whole `train()` call: chunked host-to-device ingest, device binning, mapper fit, and all 60 boosting rounds.

Throughput rises with scale, from 54.8 to 58.5M row-iterations per second, because the fixed costs amortize across more rows. There is no cliff anywhere on the ladder. A rising curve with no cliff is the shape you want, and here it holds to the last measured rung.

What keeps the loop bus-free at this scale is the device-resident objective of case E4. Per tree, gradients derive on the card from resident scores and labels, and nothing per-row crosses the PCIe bus.

## The contrast

The comparison is not only against last week's bonsai. The reference libraries do not advertise this cell on one card.

XGBoost's GPU histogram method wants roughly 8 bytes per cell before its external-memory spilling, and CatBoost's GPU pool is similarly float-dominated. Neither library's documentation claims a 500M by 100 fit on one 80GB device without external-memory modes. That is stated as a documentation-level contrast, not a same-pod measurement, which lives in the frontier ledger.

## Honest caveats

The data is synthetic, one pod, one run per rung, so the walls carry the usual 25% fleet spread. The r2 column is a 1M-row train-sample proxy, not a held-out test score.

The 550M extrapolated ceiling is arithmetic beyond the last measured rung, not a measurement, so it is labeled as arithmetic. Ingest dominates the wall at this scale, and the 60-round boosting portion is the minority. A workload reusing one `bonsai.Dataset` across fits pays the ingest once.

## What it teaches

- **Capacity claims are measurements, not arithmetic.** The u8 storage math implied the half-billion-row cell for months. It became a claim only when a ladder trained it, and the extrapolated 550M is still flagged as arithmetic because nothing measured it.
- **Rising throughput means fixed costs amortize.** The curve climbed from 54.8 to 58.5M row-iterations per second with scale. When you see that shape, look for the cliff, and report honestly when there is none.

## The record

- Evidence: [the single-card ceiling ladder](../../../benchmarks/single-card-ceiling-2026-07.md), and its rows in [the results ledger](../../method/results.md#the-single-card-ceiling).
- Enabled by: the device-resident objective of case [E4](4-the-resident-objective.md), decisions [77 to 79](../../decisions.md).
