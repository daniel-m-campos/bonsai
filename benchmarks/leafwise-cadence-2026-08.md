# Stage 0 of device leafwise admission: pricing the per-round cost F

2026-08-01. The device leafwise design ([docs/architecture/20-cuda-leafwise.md](../docs/architecture/20-cuda-leafwise.md)) rests on one number: the fixed per-round cost F of its split cadence, since the model for fit time is the depthwise compute volume plus rounds times F. Stage 0 of that doc's admission section prices F before any engine code exists, so a wrong thesis is caught by a probe rather than by a plane. The probe is [experiments/leafwise_cadence/cadence.cu](../experiments/leafwise_cadence/cadence.cu), a standalone nvcc program with no bonsai headers that replays the round's launch, sync and staging skeleton 25,500 times (255 splits per tree x 100 trees, the issue #268 protocol) against a balanced 256-leaf schedule on 16M rows. Raw output: [results/leafwise-cadence-2026-08.json](results/leafwise-cadence-2026-08.json).

## Provenance

RunPod pod `vhgox7m8jqk69j`, SECURE cloud, US-MO-1, image `ghcr.io/daniel-m-campos/bonsai-ci:cuda12.4`, $0.99/hr, about 12 minutes of wall clock. GPU per `nvidia-smi`: NVIDIA L40S, driver 580.159.03, 142 SMs. Toolkit: CUDA 12.4 (nvcc V12.4.131), compiled `-O2 -arch=sm_89`; the runtime reports driver support for CUDA 13.0. Timed with `std::chrono::steady_clock` around the full 25,500 rounds after a 500-round warmup, one stream, pinned host staging on both directions.

## The measurements

| config | what it drops | seconds | us/round |
|---|---|---|---|
| full | nothing | 1.175 | 46.1 |
| no-copyback | the D2D range copy-back | 0.773 | **30.3** |
| no-syncs | copy-back and both pinned syncs | 0.640 | 25.1 |
| floor | copy-back, every grid at one block | 0.757 | 29.7 |

Bucket breakdown, in us per round:

| bucket | derivation | us/round |
|---|---|---|
| launch and staging floor | floor minus syncs | 24.5 |
| pinned syncs (2) | no-copyback minus no-syncs | 5.2 |
| grid width | no-syncs minus the launch floor | 0.6 |
| **F (fixed per-round cost)** | **no-copyback** | **30.3** |
| range copy-back | full minus no-copyback | 15.8 |

A repeat run on the same pod gave F = 30.1 us with buckets 23.8, 5.4 and 0.9, so the split is stable to about a microsecond.

## The verdict

F = 30.3 us, under the 100 us budget by a factor of three and an order of magnitude under the 300 us kill line, so stage 0 passes and the design proceeds to stage 1 rather than returning to the batched-K question. At this cost the cadence spends 0.77 s across a 100-iteration 256-leaf fit instead of the 2.6 s the design budgeted, and the full round including the copy-back spends 1.18 s. On the 17.2 s depthwise anchor that prices a fit near 18 s at 16M, against lgbm_cuda's 24.4 s and its inferred ~950 us per round. The buckets say where the cost sits: eight launches and four staged transfers dominate at 24.5 us, the two pinned syncs cost 5.2 us together (2.6 us each, the same order as the sync-latency probe the pod drivers report), and widening the grids from one block to the partition and histogram shapes costs 0.6 us, which is the finding that the fixed cost really is fixed and does not track node size. The copy-back is the one term that scales with rows rather than rounds: 15.8 us per round on average, or 1.5 GB moved per tree, and it is excluded from F because it is leafwise-specific data movement that the design already accounts for as compute, not overhead.

## Caveat

The kernels are trivial by construction: each launch touches one element, so what is measured is launch, sync and staging overhead, not compute, occupancy, or memory traffic inside the histogram and partition work. The known risk named in doc 20 is occupancy on mid-size nodes, and this probe cannot see it; a 31k-row node underfilling 142 SMs costs real time that a one-element kernel does not pay. Stage 2's ladder is the compute test, and stage 1's parity work is what makes that ladder possible. The copy-back number carries a second caveat: it moves untouched device memory in one contiguous range per round, which is the best case for the design's actual copy of rows plus ordered gradients.

Evidence chain: doc 20's admission section (stage 0), issue #268 (the target and the kill criterion), decision 42 (the withdrawal this design answers), decision 95 (the recheck that reopened it).
