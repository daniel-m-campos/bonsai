# E3. The parity verdict

A data-parallel multi-GPU engine passed every correctness test on real 2x and 4x hardware, then never shipped. This case is why passing correctness is not the same as earning a place in the core.

The engine was built through its full plan and validated end to end. Then five optimizations were applied and the engine was measured against one GPU, and the verdict was parity, not speedup. The feature-admission gate (a check that a feature shows a measured benefit before anyone writes core code) exists to refuse a feature that grows the core for no measured win.

## The setting

The design was architecture doc 19, now only on the `experiment/multi-gpu` branch: a `MultiCudaHistogramEngine` beside the single-GPU one, selected by `cuda_multi_*` grower names that main no longer registers. Rows shard across N devices. Each device builds partial histograms over its shard, and the partials reduce to a coordinator that finds the split.

The CUDA backend plugs into the engine through a C++ concept (a compile-time interface that a type must satisfy), not through a class hierarchy. A data-parallel backend is a new type that satisfies the same concept, added beside the old one. Adding it needed no new logic in the existing engine, grower, or dispatch code.

The one genuinely new operation is the histogram reduction between build and find. It reduces only the smaller children's partials, deriving each larger child as parent minus small, which roughly halves the cross-device traffic. Row partitioning stays per-shard, so no rows ever cross devices.

Because the engine was added beside the single-GPU engine instead of inside it, it went into main as a personal-use experiment and skipped the feature-admission gate. That gate prices a feature's benefit before anyone writes the code.

## Correctness held everywhere

Correctness was never in doubt, and that is the point. The tree comes out identical on every device, because every split is decided on the one reduced histogram.

Validation confirmed it on real hardware. The r2 (the share of target variance the model explains) was identical on one, two, and four GPUs, at both 16M and 64M rows. It held in both reduction regimes: peer (partial histograms copied GPU to GPU directly) and host-staged (copied through CPU memory), on 2x A40 and 4x A100 NVLink boxes.

The two interconnects were the experiment's independent variable. The plan estimated a level's reduction at 16M rows by 100 features at roughly 50 to 100 MB for the deepest levels: about 2 to 4 ms per level over PCIe, and about 0.1 ms over NVLink at 600 GB/s. That is why the sweeps ran on both a PCIe box and an NVLink mesh.

The reproducibility posture carried over unchanged. The GPU path was, at the time, a tolerance match and not bit-exact, because atomics accumulated in arbitrary order (the cells have since gone int64 fixed point, decision 124). Multi-GPU fixes the reduction order by ascending device id, so a given device count is deterministic run to run.

A passing correctness suite feels like progress. It is a precondition for value, not evidence of it.

## The price list

I then tried five optimizations (called levers below) and measured them in four sweeps, each timed on a single rented pod so its comparisons are not skewed by pod-to-pod variance. Before the runs I set the bar: a 1.3x speedup at 2 GPUs and 1.8x at 4.

- **Parallel per-device fan-out.** Each device runs its shard's work concurrently.
- **Sliced per-shard finalize copies.** This one lever took the 64M fit from 184 to 63 seconds.
- **Per-shard gradient upload slices.** Each device receives only its shard's gradients.
- **Pinned staging.** Page-locked host buffers for the cross-device transfers.
- **A double-buffered reduction pipeline.** Overlap the reduce with the next build.

The finalize-copy lever alone did most of the work, from 184 down to 63 seconds at 64M. The other four chased the residue.

## The verdict was parity

The end state matched one GPU, and did not beat it. At 16M the multi-GPU fit ran about 16.5 seconds against the single-GPU 15.8. At 64M it ran 58.9 seconds against 57.5.

That is parity, so the 1.3x and 1.8x bars were both missed. The engine did the same work on four cards that one card already did.

## The floor was architectural

Two costs remained after the levers, and the targeted fix for each changed the measured time by nothing. Both costs come from the design of the engine, not from settings you could tune.

Pinned staging equaled pageable (ordinary host memory that the driver must first copy through a staging buffer). Gradients are host-computed per tree, so the gradient stream must cross host memory every tree, whatever the staging strategy.

Forced host-staging equaled peer on an NVLink mesh. The reduction's time goes to the synchronization points where devices wait for each other so every split is decided on the same reduced histogram, not to moving data, so a 600 GB/s interconnect saved no time.

When a targeted fix moves a cost by nothing, the cost is structural. No further lever round was justified.

## Parked, and the gate re-validated

The engine was withdrawn to the `experiment/multi-gpu` branch. Main kept the parts that improve the single-GPU engine: single-device selection, the `CudaDeviceContext` extraction, and a tightened engine concept.

The whole episode cost about $17 of pod time, and every conclusion above is a measurement. The supported multi-GPU story is fit-parallelism: N cards run N independent fits, which scales linearly by construction.

Capacity is a weak reason to want the engine too. The u8 storage already puts roughly 500M rows by 100 features on one 80GB card, so vertical scaling covers the realistic range.

The single-GPU engine came through unchanged. The extraction that let the two engines share code was move-only and gated by the CUDA suite, with the CPU model hash unchanged.

The decision records why the admission gate would have caught this. The gate asks for a measured benefit at zero core cost before any C++ is written. Merge-as-you-go inverted that order, putting the complexity in main first and pricing the value last.

The reopener is a device-resident objective, where each GPU derives its shard's gradients on the card. That removes the host gradient stream, one of the two costs the floor is made of; the reduction's correctness syncs remain. It is also the single-GPU engine's own next frontier, so it was pursued there first. That is case E4.

## What it teaches

- **Correctness passing is not value.** The engine matched single-GPU r2 on 2 and 4 GPUs and still did not earn main, because parity is not a speedup. A passing suite is a precondition, measured against a bar it did not clear.
- **An architectural floor announces itself.** For two costs, the targeted fix changed the measured time by nothing: pinned equaled pageable, host-staged equaled peer. A fix that moves a number by nothing has found a structural wall.
- **Merge-as-you-go misorders the work.** Complexity landed in main before the value was priced. The admission gate prices benefit first for exactly this reason, and this experiment re-validated it.

## The record

- Decisions: [76](../../decisions.md) (built, measured to parity, parked as an experiment).
- Design: [architecture doc 19](https://github.com/daniel-m-campos/bonsai/blob/experiment/multi-gpu/docs/architecture/19-multi-gpu.md), the multi-GPU design plan, kept on the parked branch. The withdrawal record and measurements are decision 76, linked above.
- Reopener: [decision 77](../../decisions.md) pursued the device-resident objective single-GPU first, told as case [E4](4-the-resident-objective.md).
