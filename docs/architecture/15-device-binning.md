# 15: Device binning: ingest joins the transaction narrative

> **Status:** implemented (decision 54; this change set). The last big ingest lever that does not change the model, planned against same-pod ledgers (PR #34/#35 runs) and the pipeline facts below. Framing: doc 16 (ingest is the one compute node still outside the transaction vocabulary); this design moves it inside rather than growing a side channel.

## The ledger line

Post-decision-53, the 16M×100 `cuda_depthwise` fit spends (same-pod L40S, US-MO-1; fit 39.4s): find 7.8s, ingest 8.4–8.8s (mapper-fit ~3.9 + bin ~4.6), populate 5.2s, finalize ~3.8s, partition 1.6s. The `bin` line is host CPU work: `fill_binned` runs `BinMapper::transform` (a `lower_bound` over the feature's cuts) once per cell (1.6G binary searches), then `ensure_dataset` re-copies the result into pinned staging and uploads 1.6GB, unlapped, inside the first `begin_tree`. The device then reads *only the device copy* for the rest of the fit.

xgboost's 16M edge (27.9s vs 37.4s in the re-baseline) is device binning: raw values go over PCIe once and the quantized matrix is built where it is consumed.

## Pipeline facts the design must respect

- **Raw floats are not retained.** Both ingest paths (`features_view` borrowed from the Python module's numpy matrix, `ColumnBatch` from the CSV parser) are consumed by `Dataset::bin`; only binned columns survive. Device binning must therefore hook the ingest step itself: after `Dataset` construction the raw data is gone.
- **Cuts are tiny and host-fitted.** `BinMappers::fit` subsamples ≤200k values per feature (`n_samples`), sorts once, strides (decision 51). Cuts per feature ≤ `max_bin` floats. `transform` semantics: NaN → last bin, else `lower_bound(cuts, x)`.
- **In device mode, host bins have exactly two consumers.** (1) The fallback decline (`begin_root` refusing oversized `max_bin` → full CPU data plane); (2) `route_unsampled`'s `bin_at` random access, only when row sampling is on. The device plane partitions, finds, and stamps on device (stages A–D); host partition/populate arms run only in fallback mode.
- **The stub build must stay CUDA-free.** `Dataset` is built long before any engine exists and cannot name CUDA types.

## Proposed shape

### Ingest is the zeroth transaction

Doc 14 gave the backends one vocabulary (`begin_tree` / `open_level` / `apply_level` / `end_tree`), and the strain in this design's first draft (a CUDA pimpl sprouting on `Dataset`) came from ingest being the one compute node outside it. So ingest joins the narrative instead:

```
ingest(raw columns, fitted mappers) -> IngestPlane      // once per fit
begin_tree / open_level / apply_level / end_tree        // unchanged
```

The host backend's ingest **is** today's `fill_binned`: CPU growers are untouched byte for byte. The CUDA backend's ingest streams raw columns to the device and bins there; its product, the `IngestPlane`, is an opaque handle (declared host-pure, defined in the CUDA TU, null in stub builds: the `row_major_` lazy-mirror precedent) owning the feature-major device matrix, the per-feature bin counts, and the identity fields `ensure_dataset` checks today. `Dataset` carries the handle as the transaction's receipt; it never looks inside. `ensure_dataset` recognizes and adopts its own plane (no host read, no staging copy, no upload) and keeps the current upload path for host-binned datasets.

The pipeline seam: `Dataset::bin` cannot know the grower, but the train pipelines can: they select the ingest backend exactly the way growers dispatch (the `cuda` name prefix + `cuda_available()`) and hand `Dataset::bin` the hook. Plain `Dataset` construction takes the host default.

### The CUDA ingest transaction

```
cuda ingest(raw columns or row-major view, cuts tables) -> IngestPlane
```

- **Transfer**: raw floats stream through a double-buffered pinned staging pair (bounded, ~64MB each) with `cudaMemcpyAsync`: upload of chunk *k+1* overlaps the bin kernel of chunk *k*. Raw data never fully resides on device; total device footprint = staging + the binned matrix (same as today).
- **Kernel**: one thread per cell; binary search over the feature's cut table in shared memory (cuts ≤ 255 floats/feature fit easily; u16 datasets read the table from global). `ColumnBatch` chunks are feature-major (kernel is a straight map); `features_view` chunks are row-major (kernel writes transposed; coalesced on the read side, the write pattern is the same scatter `fill_binned` does today).
- **Exactness**: the kernel reproduces `transform` exactly: NaN → last bin, else `lower_bound`. Same cuts, same total order on floats ⇒ **bit-identical bins** to the host fill, which is the whole byte-identity argument: CPU-path models are untouched by construction, and device-path models must equal the before-models exactly (r² equality gate, as PR #34).

### Scope

Training dataset only: validation datasets never enter grow (eval predicts from raw features), so they keep the host ingest; tests, CPU workflows, and predict never see the device path.

### Host bins go lazy in device mode

When device binning ran, host binned columns are not materialized at ingest. The two consumers get them on demand:

- **Fallback decline**: decided by `max_bin` vs the shared-memory ceiling, both known at ingest (the ceiling probe is safe once `cuda_available()`). If the dataset would decline, bin on host eagerly as today and skip the device path entirely; no lazy machinery on this arm.
- **`route_unsampled` under row sampling**: first `bin_at` triggers a one-time D2H materialization of the host columns (1.6GB pageable, ~1.5s, once per fit), cached like `row_major_bins()`. Row sampling off (the bench and the common CUDA configuration) never pays it.

## What this buys (projection, same-pod discipline)

Replaced: host `bin` ~4.6s + unlapped 1.6GB upload ~0.5s. New cost: 6.4GB raw over PCIe, streamed and overlapped with the bin kernel, priced by the measured gh edge (12.8GB in 0.68s ⇒ ~19GB/s) at **~0.35–0.5s**, plus ~0.2s of kernel. (The first draft said 2.3–2.6s from stale intuition; `scripts/dag_model.py` corrected it: doc 16.) **Projected: fit 39.4 → ~34.8–35.3s on the US-MO-1 host class**, leaving mapper-fit as the dominant ingest line. Cross-pod absolutes are meaningless (~25% fleet spread measured between two L40S pods); the gate is the same-pod before/after delta.

## Instrumentation shipped with the round

- `ingest-profile` gains `dbin` (device-binning transfer+kernel) so the before/after decomposes.
- `cuda-upload-decomp` gains `bins_upload` (the `ensure_dataset` copy+upload, today's dark matter) and finalize lap counters (`fin_wait`/`fin_d2h`): the PR #35 refutation showed the finalize line is undecomposed and misleads design; never again.

## Phase 2 (not this round): mapper-fit

The remaining ~3.9s is `create_subsample`'s reservoir scan (`std::ranges::sample` over a filter view: 16M reads + per-element RNG, per feature). Any device or algorithmic change to sampling changes the sampled set → different cuts → **model-changing**; it needs its own decision with quality data, and is deferred until the byte-identical levers are exhausted.

## Rejected

- **A `DeviceBins` side channel on `Dataset` without the narrative verb** (this design's first draft): identical mechanics, but the API grows by exception instead of by vocabulary: the convolutedness the transaction narrative exists to prevent. Superseded by the ingest transaction.
- **Engine-side rebinning** (host bins → device rebin): saves nothing; the host `transform` cost is the line item.
- **Retaining raw floats on Dataset** so the engine can bin later: +6.4GB host RSS at 16M for a copy the ingest hook can stream through 128MB of staging.
- **Device mapper-fit** in this round: RNG-identical reservoir sampling on device is not worth inventing; see phase 2.
- ~~Transposing kernel only for CSV~~; corrected during implementation: the module path bins straight from the borrowed row-major numpy view (`features_view`), and CSV parses into the feature-major `ColumnBatch`; the row-major arm is therefore the primary (bench) arm. Both arms shipped.
