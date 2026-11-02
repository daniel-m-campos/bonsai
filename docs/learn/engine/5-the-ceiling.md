# E5. The ceiling

One 80GB card trained a 500M-row by 100-feature dataset end to end in 8.5 minutes (one pod, one run). The memory arithmetic had predicted it would fit, because bonsai stores each binned value in one byte; this case measures it.

Throughput rose at every step of the ladder, from 54.8M to 58.5M row-iterations per second.

## The claim from arithmetic

bonsai stores one byte per binned cell, so a 500M by 100 matrix is about 50 GB of binned data. Add the per-row labels, scores, and gradients that stay on the card during training and it still fits inside 80 GB. The arithmetic says a 500M by 100 fit is possible. One run on one card checks it.

## The ladder

One RunPod A100-SXM4-80GB ran one process. A 500M by 100 float32 matrix of Friedman-1 data (a standard synthetic regression benchmark), 200 GB raw, was generated to a disk memmap. Then an ascending ladder of `bonsai.train` calls fit the first 300M, 400M, 450M, and 500M rows of that file, each as a row-prefix view (a slice of the leading rows). Every call used the device-resident MSE objective (gradients computed on the card, case E4), 60 iterations, depth 8, and 255 bins.

| rows | train() wall | peak device memory | throughput (row-iterations/s) | train r2 (1M sample) |
|--:|--:|--:|--:|--:|
| 300M | 328.5s | 42.2 GiB | 54.8M | 0.8305 |
| 400M | 414.2s | 56.0 GiB | 57.9M | 0.8303 |
| 450M | 465.1s | 63.0 GiB | 58.0M | 0.8340 |
| **500M** | **512.6s** | **69.9 GiB** | **58.5M** | 0.8329 |

Every step finished, and none ran out of device memory. Device memory grows at about 13.9 GiB per 100M rows. At 500M that is 69.9 GiB of the card's 80, about 10 GiB below the limit. Extending that slope to the remaining memory puts the arithmetic ceiling on this card at roughly 570M rows by 100 features.

## Reading it

The 500M fit runs 8.5 minutes end to end on one card. The wall time is the whole `train()` call: chunked host-to-device ingest, device binning, mapper fit, and all 60 boosting rounds.

Throughput rose from 54.8M to 58.5M row-iterations per second as rows grew from 300M to 500M, and it did not fall at any step. The likely cause is that costs which do not grow with row count are spread over more rows at larger sizes, but the run did not measure that.

The device-resident objective of case E4 keeps the boosting loop off the PCIe bus at this scale: per tree, gradients derive on the card from resident scores and labels, and nothing per-row crosses the bus.

## The contrast

XGBoost and CatBoost do not document a 500M by 100 fit on one 80GB GPU without external-memory modes. XGBoost's GPU histogram method wants roughly 8 bytes per cell, 8 times bonsai's 1 byte, unless you turn on its external-memory mode, which keeps the data outside device memory. There is no per-cell figure for CatBoost's GPU pool, so this page makes no size comparison for it. This contrast comes from documentation, not a same-pod measurement; the same-pod comparisons at 16M rows live in [the 16M GPU frontier](../../../benchmarks/gpu-pareto-16M-2026-07.md).

## Honest caveats

The data is synthetic and each step ran once on a single pod. Identical rented pods have measured up to 25% apart, so read the times as approximate and do not compare them with times from another pod. The r2 column is a 1M-row train-sample proxy, not a held-out test score.

That 570M figure is arithmetic beyond the last measured step, not a measurement. The record does not split the 512.6s into ingest and boosting, so the throughput figures include ingest and understate boosting speed on a reused `bonsai.Dataset`, which pays ingest once.

## What it teaches

- **Capacity claims are measurements, not arithmetic.** The u8 storage math implied a half-billion-row fit for months. It became a claim only when a ladder trained it, and the 570M extrapolation is still arithmetic because nothing measured it.
- **Rising throughput points to costs that do not grow with row count.** The curve climbed from 54.8M to 58.5M row-iterations per second as rows grew from 300M to 500M. The run did not measure the cause, so check for a drop at each larger size and report it when there is none.

## The record

- Evidence: [the single-card ceiling ladder](../../../benchmarks/single-card-ceiling-2026-07.md), and its row in [the archive](../../results/results/archive.md).
- Enabled by: the device-resident objective of case [E4](4-the-resident-objective.md), decisions [77 to 79](../../decisions.md).
