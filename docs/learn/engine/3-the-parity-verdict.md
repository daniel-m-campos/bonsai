# E3. The parity verdict

A data-parallel multi-GPU engine passed every correctness test on real 2x and 4x hardware, then never shipped. This case is why passing correctness is not the same as earning a place in the core.

The engine was built through its full plan and validated end to end. Then five optimization levers priced it, and the verdict was parity with one GPU, not speedup. A feature that grows the core for no measured win is exactly what the admission discipline exists to refuse.

## The setting

The design is architecture doc 19: a `MultiCudaHistogramEngine` beside the single-GPU one, selected by the `cuda_multi_*` grower names. Rows shard across N devices. Each device builds partial histograms over its shard, and the partials reduce to a coordinator that finds the split.

The seam holds because the CUDA backend plugs in through a concept, not an inheritance tree. A data-parallel backend is a new type satisfying the same concept, added beside the old one. No existing engine, grower, or dispatch code changes shape.

The one genuinely new operation is the histogram reduction between build and find. It reduces only the smaller children's partials, deriving each larger child as parent minus small, which roughly halves the cross-device traffic. Row partitioning stays per-shard, so no rows ever cross devices.

That containment is real, and it is also the trap. The engine went into main as a personal-use experiment, so it bypassed the feature-admission gate. That gate prices a feature's benefit before anyone writes the code.

## Correctness held everywhere

Correctness was never in doubt, and that is the point. The tree comes out identical on every device, because every split is decided on the one reduced histogram.

Validation confirmed it on real hardware. The r2 was identical across single, 2, and 4 GPUs, at both 16M and 64M rows. It held in both reduction regimes, peer and host-staged, on 2x A40 and 4x A100 NVLink boxes.

The two interconnects were the experiment's independent variable. A level's reduction is small, so over PCIe it costs a few milliseconds per level, and over NVLink at 600 GB/s it is noise. That is why the ladder ran on both a PCIe box and an NVLink mesh.

The reproducibility posture carried over unchanged. The GPU path is a tolerance match, not bit-exact, because atomics accumulate in arbitrary order. Multi-GPU fixes the reduction order by ascending device id, so a given device count is deterministic run to run.

A passing correctness suite feels like progress. It is a precondition for value, not evidence of it.

## The price list

The track was then priced by five levers across four same-pod ladders. The pre-registered bar was 1.3x speedup at 2 GPUs and 1.8x at 4.

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

Two costs remained after the levers, and each refuted its own targeted fix cleanly. That clean refutation is the signature of an architectural floor, not a tuning residue.

Pinned staging equaled pageable. Gradients are host-computed per tree, so the gradient stream must cross host memory every tree, whatever the staging strategy.

Forced host-staging equaled peer on an NVLink mesh. The reduction's cost is its correctness syncs, not its transport, so a 600 GB/s interconnect bought nothing.

When a targeted fix moves a cost by nothing, the cost is structural. No further lever round was justified.

## Parked, and the gate re-validated

The engine was withdrawn to the `experiment/multi-gpu` branch. Main kept the parts that improve the single-GPU engine: single-device selection, the `CudaDeviceContext` extraction, and a tightened engine concept.

The whole episode cost about $17 of pod time, and every conclusion above is a measurement. The supported multi-GPU story is fit-parallelism: N cards run N independent fits, which scales linearly by construction.

Capacity is a weak reason to want the engine too. The u8 storage already puts roughly 500M rows by 100 features on one 80GB card, so vertical scaling covers the realistic range.

None of this disturbed the single-GPU crown regime. The extraction that let the two engines share code was behavior-neutral and hash-gated, so the single-GPU model bytes never moved.

The decision records why the admission gate would have caught this. The gate asks for a measured benefit at zero core cost before any C++ is written. Merge-as-you-go inverted that order, putting the complexity in main first and pricing the value last.

The reopener is a device-resident objective, where each GPU derives its shard's gradients on the card. That removes the host gradient stream, the one cost the floor is made of. It is also the single-GPU engine's own next frontier, so it was pursued there first. That is case E4.

## What it teaches

- **Correctness passing is not value.** The engine was bit-correct on 4 GPUs and still did not earn main, because parity is not a speedup. A passing suite is a precondition, measured against a bar it did not clear.
- **An architectural floor announces itself.** Two costs each refuted a targeted fix cleanly: pinned equaled pageable, host-staged equaled peer. A fix that moves a number by nothing has found a structural wall.
- **Merge-as-you-go misorders the work.** Complexity landed in main before the value was priced. The admission gate prices benefit first for exactly this reason, and this experiment re-validated it.

## The record

- Decisions: [76](../../decisions.md) (built, measured to parity, parked as an experiment).
- Design: [architecture doc 19](https://github.com/daniel-m-campos/bonsai/blob/main/docs/architecture/19-multi-gpu.md), the priced multi-GPU plan with its validation hardware.
- Reopener: [decision 77](../../decisions.md) pursued the device-resident objective single-GPU first, told as case [E4](4-the-resident-objective.md).
