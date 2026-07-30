# The iso-volume campaign: the shape frontier at constant data volume (decision 91)

2026-07-30. First campaign on the redesigned bench tooling (committed specs, `python -m bonsai.bench run`, `scripts/pod_bench_driver.sh`), and the first with measured peak device memory as an output. Raw rows: [results/iso-volume-2026-08.jsonl](results/iso-volume-2026-08.jsonl), every row schema v1 with git sha `a907895`, host, and run label stamped by the harness. One pod: RTX PRO 6000 Blackwell Workstation Edition (96GB VRAM, sm_120, CUDA 12.8 toolkit side-installed over the cuda12.4 image), 64 vCPU, 1.1TB RAM, EU-RO-1, sync-latency probe 4.5us/op (accept threshold 50). Total cost about $4.

## The design

Hold rows x cols constant and sweep the aspect ratio: costs that scale with total cells (binning, partition, gradient passes) are flat along the line, so anything that rises is paying for width (histogram footprint, find-split, mapper fit). Two ladders: 2^31 cells (8GiB float32 matrix, six arms: four GPU x 2 reps, two CPU context arms x 1 rep) from 16M x 128 to 32k x 65536, and a 2^33 stretch (32GiB matrix, GPU arms only) to 262k x 32768. Memory gates off by spec: a measured out-of-memory is the experiment's output, not a condition to avoid. `dev_mem` is NVML-sampled (250ms) while the worker child runs, recording per-process and whole-device peaks.

## The measurements

Fit seconds (test r2), best of reps, 2^31 cells:

| cell | bonsai cuda dw | bonsai cuda obl | xgb cuda | catboost gpu | bonsai cpu dw | xgb hist |
|---|---|---|---|---|---|---|
| 1M x 100 (anchor) | 0.5s (.877) | 1.1s (.876) | 1.6s (.876) | 1.8s (.876) | 5.2s (.877) | 3.5s (.876) |
| 16M x 128 | 7.2s (.879) | 6.9s (.876) | 28.1s (.879) | 26.0s (.875) | 70.8s (.879) | 65.1s (.878) |
| 4M x 512 | 6.8s (.878) | 6.2s (.875) | 24.7s (.878) | 14.8s (.877) | 62.4s (.878) | 60.2s (.878) |
| 1M x 2048 | 9.3s (.876) | 8.6s (.875) | 27.3s (.876) | 16.6s (.876) | 72.1s (.876) | 84.8s (.876) |
| 262k x 8192 | 20.5s (.867) | 21.4s (.874) | 34.5s (.867) | 35.5s (.873) | 125.5s (.867) | 127.1s (.867) |
| 65k x 32768 | 57.8s (.841) | 68.5s (.870) | 62.1s (.840) | 96.3s (.866) | 310.0s (.842) | 276.3s (.841) |
| 32k x 65536 | 105.5s (.815) | 130.4s (.873) | failed | 172.4s (.868) | 491.1s (.815) | 444.0s (.817) |

Measured peak device memory (per-process), GPU arms, 2^31 cells:

| cell | bonsai cuda dw | bonsai cuda obl | xgb cuda | catboost gpu |
|---|---|---|---|---|
| 16M x 128 | 3.4GB | 3.4GB | 18.9GB | 90.2GB |
| 1M x 2048 | 4.4GB | 4.4GB | 18.7GB | 90.2GB |
| 262k x 8192 | 9.7GB | 9.7GB | 18.8GB | 90.2GB |
| 65k x 32768 | 30.7GB | 30.7GB | 48.3GB | 90.2GB |
| 32k x 65536 | 58.5GB | 58.5GB | died at 33.4GB | 90.2GB |

The 2^33 stretch (GPU arms, 2 reps):

| cell | bonsai cuda dw | bonsai cuda obl | xgb cuda | catboost gpu |
|---|---|---|---|---|
| 67M x 128 | 27.9s, 11.7GB | 23.9s, 11.7GB | 113.8s, 73.6GB | 103.8s, 90.2GB |
| 4M x 2048 | 25.8s, 10.5GB | 22.6s, 10.5GB | 101.6s, 72.7GB | 47.7s, 90.2GB |
| 262k x 32768 | 85.3s, 36.7GB | 88.3s, 36.7GB | 160.7s, 73.2GB | 154.7s, 90.2GB |

## What the memory instrument caught

XGBoost-GPU failed at 32k x 65536 on both attempts (native backtrace, error row), and the sampler recorded 33.4GB of device memory at death: the failure is not simple exhaustion of the 96GB card but an internal limit hit mid-allocation, a fact invisible to the old estimate-and-skip design. CatBoost-GPU allocates 90.2GB at every cell including the 1M x 100 anchor: it reserves the card rather than sizing to the problem, which is why it never fails and also why it cannot share a device. bonsai's footprint tracks the problem: 0.8GB at the anchor, 3.4GB tall, 58.5GB at the widest aspect (histogram slots scale with cols x bins), leaving room on 96GB silicon for cells the old `GPU_MAX_COLS` policy assumed impossible.

## Findings on the iso-line

bonsai's CUDA growers are fastest at every cell of both ladders. Fit time is nearly flat across the tall half (7.2s at 16M x 128, 6.8s at 4M x 512, 9.3s at 1M x 2048) confirming rows-dominant costs are shape-invariant at constant volume; every arm rises together past 8192 cols as cols x bins histogram work takes over. The 4x lead over both references at the tall end narrows toward parity with XGBoost-GPU at 65k x 32768 (57.8 vs 62.1s) before XGBoost exits the frontier entirely. On CPU the tiled fill holds bonsai at XGBoost-hist parity out to the widest cells (491 vs 444s at 32k x 65536, 72 vs 85s at 1M x 2048). At extreme aspect the quality split is the story: oblivious holds r2 .873 at 32k x 65536 where depthwise (and both references' depthwise-family growers) fall to .815-.817, the symmetric tree's implicit regularization at p about 2x n, consistent with the cols re-baseline.

## Honest residuals

One pod, one silicon class: absolutes carry the same-host caveat as always, and the work-rig replication (`make bench-iso`, host-tagged) is the standing invitation to a second same-silicon point. The 250ms sampling interval bounds `dev_mem` precision; transient spikes shorter than the interval can be missed (the interval is recorded in every row). Test r2 falls with width by construction (informative feature count is fixed while noise features grow), so quality compares across arms within a cell, never across cells. CUDA 13 remains blocked at the toolchain (clang cannot target it); revisit when LLVM grows support.
