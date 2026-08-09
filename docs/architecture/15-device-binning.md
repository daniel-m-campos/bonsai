# 15: Device binning: ingest joins the transaction narrative

> **Status:** implemented (decision 54; this change set). The binned matrix both ingest arms write is tile-blocked as of decision 103, so "feature-major" below names the layout this document shipped with, not the one in the tree. Since then the Python `Dataset` takes a device hint of its own (`device="cuda"`), so the seam below reads "plain `Dataset` construction takes the host default, and an explicit hint takes the device arm". The last big ingest lever that does not change the model, planned against same-pod ledgers (PR #34/#35 runs) and the pipeline facts below. Framing: doc 16 (ingest is the one compute node still outside the transaction vocabulary); this design moves it inside rather than growing a side channel.

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

- **Transfer, as built**: raw floats stream through one device buffer of ~64MB (`k_ingest_chunk_bytes`, `src/cuda/histogram_engine.cu`). Each chunk is copied by a blocking `cudaMemcpy` from the caller's pageable memory, binned, and then the buffer is reused. The double-buffered pinned staging pair with `cudaMemcpyAsync` that this design proposed was not built: the copy already runs at bus rate and dominates the kernel, so the overlap machinery stayed unpriced, and the comment above the chunk constant records that `dbin` is the counter that would decide it. Raw data never fully resides on device; total device footprint = one chunk + the binned matrix.
- **Kernel, as built**: one thread per cell; binary search over the feature's cut table in **global** memory (`transform_bin`, `src/cuda/detail/ingest_kernels.cuh`), reading one flat cuts array with per-feature offsets. The proposed shared-memory cut table was not built either. `features_view` chunks are row-major (`bin_rows_kernel`, feature varies fastest across threads, so the raw reads coalesce); `ColumnBatch` chunks arrive one column at a time (`bin_col_kernel`, whose writes stride by the strip width). Both arms write the tile-blocked plane of decision 103.
- **Exactness**: the kernel reproduces `transform` exactly: NaN → last bin, else `lower_bound`. Same cuts, same total order on floats ⇒ **bit-identical bins** to the host fill, which is the whole byte-identity argument: CPU-path models are untouched by construction, and device-path models must equal the before-models exactly (r² equality gate, as PR #34).

### Scope

Training dataset only: validation datasets never enter grow (eval predicts from raw features), so they keep the host ingest; tests, CPU workflows, and predict never see the device path.

### Host bins go lazy in device mode

When device binning ran, host binned columns are not materialized at ingest. The two consumers get them on demand:

- **Fallback decline**: decided by `max_bin` vs the shared-memory ceiling, both known at ingest (the ceiling probe is safe once `cuda_available()`). If the dataset would decline, bin on host eagerly as today and skip the device path entirely; no lazy machinery on this arm.
- **`route_unsampled` under row sampling**: first `bin_at` triggers a one-time D2H materialization of the host columns (1.6GB pageable, ~1.5s, once per fit), cached like `row_major_bins()`. Row sampling off (the bench and the common CUDA configuration) never pays it.

## What this buys (projection, same-pod discipline)

Replaced: host `bin` ~4.6s + unlapped 1.6GB upload ~0.5s. New cost: 6.4GB raw over PCIe plus the bin kernel, priced at design time by the measured gh edge (12.8GB in 0.68s ⇒ ~19GB/s) at **~0.35–0.5s** of transfer and ~0.2s of kernel. (The first draft said 2.3–2.6s from stale intuition; `scripts/dag_model.py` corrected it: doc 16.) **Projected: fit 39.4 → ~34.8–35.3s on the US-MO-1 host class**, leaving mapper-fit as the dominant ingest line. Cross-pod absolutes are meaningless (~25% fleet spread measured between two L40S pods); the gate is the same-pod before/after delta.

**Measured after the round.** The `dbin` lap reads **0.55s** for transfer and kernel together at 16M×100 (L40S US-NC-1, 2026-07-15), and the fit moved 37.9 → 31.3s (doc 16's moves table). The projection held even though the transfer is blocking and pageable rather than overlapped and pinned, because 6.4GB at the measured pageable H2D rate is ~0.46s on its own. Doc 16 prices the placement floor on that measured line, not on the overlapped transport.

## Instrumentation shipped with the round

- `ingest-profile` gains `dbin` (device-binning transfer+kernel) so the before/after decomposes.
- `cuda-upload-decomp` gains `bins_upload` (the `ensure_dataset` copy+upload, today's dark matter) and finalize lap counters (`fin_wait`/`fin_d2h`): the PR #35 refutation showed the finalize line is undecomposed and misleads design; never again.

## Phase 2 (not this round): mapper-fit

The remaining ~3.9s is `create_subsample`'s reservoir scan (`std::ranges::sample` over a filter view: 16M reads + per-element RNG, per feature). Any device or algorithmic change to sampling changes the sampled set → different cuts → **model-changing**; it needs its own decision with quality data, and is deferred until the byte-identical levers are exhausted.

## Device-resident input: the same transaction, one step earlier

Everything above assumes the raw floats start on the host. When they do not, when the caller hands over a cupy array, a torch CUDA tensor, or a jax array through DLPack, the transaction is the same and one copy disappears: `cuda_ingest_device` launches the same `bin_rows_kernel` over the same host-fitted cuts, with the caller's buffer as the chunk pointer instead of a staging buffer. Bin ids are bit-identical to both the host fill and `cuda_ingest`, because it is the same kernel reading the same bytes.

The import itself is nanobind's: a `nb::ndarray<float const, nb::ndim<2>, nb::c_contig, nb::device::cuda>` accepts the capsule and refuses everything that is not a C-contiguous float32 CUDA array, so dtype, rank, contiguity, and device tag are validated by code bonsai does not own. What remains bonsai's is the policy below.

**The mapper still fits on the host, on a sample that is drawn on the device.** Cuts come from `BinMapper::from_sample` over a shared row sample (`bin_sample_rows`), and moving that to the device would change the sampled set and therefore the model (the phase-2 note above). So the device arm gathers exactly the rows the host sampler would have used, in the same order, into a compact block and downloads that block: `n_samples x n_features` floats rather than the whole matrix, which at 16M x 100 is 80MB against 6.4GB. The sample is then fed to the ordinary `BinMappers::fit`, which re-samples it and finds it already at sample size, so the cuts are bit-identical by construction. Below `n_samples` rows the sample is every row, so a small matrix does cross the bus once for its cuts; that is the same rule the host path follows, and it is bounded by `n_samples x n_features` however large the data gets.

**The caller's buffer is borrowed for the length of one call and never retained.** Ingest copies the bins it needs into a plane that owns its own device memory, so nothing bonsai holds points into the caller's allocation after the constructor returns. The Python `Dataset` keeps a host numpy array alive because its `FeatureBuffer` borrows it; it deliberately keeps no reference to a device buffer, because there is nothing left to borrow. The caller may free or overwrite the array as soon as the call returns.

**Mismatch is refused, not migrated,** on decision 99's rule. An array whose capsule names a device other than `parallel.device_id` raises before any device work, naming both ids; a device-resident array with no CUDA build or no visible device raises, because it is an explicit placement and not an engine inference. The one asymmetry with a host array is the `device="cpu"` hint, which raises for device-resident input instead of copying the matrix back: the copy is exactly what the caller avoided by handing over a pointer.

**Stream ordering is the producer's, by contract.** DLPack puts the synchronization at export: a producer whose writes are still in flight on its own stream must make them visible to the consumer's stream before it returns the capsule. bonsai's import asks for no particular stream and ingest reads on the legacy default one, which is the stream every well-behaved producer orders against, so there is nothing for bonsai to wait on and no handle to plumb. The refused alternative is the one the first draft built: reading a stream handle out of the array and calling `cudaStreamSynchronize` on it, which duplicates work the producer already owes and only works for producers that publish a handle at all.

**Labels and weights are downloaded, not plumbed through.** A device-resident `y` is accepted for symmetry, but `Dataset` stores labels and weights in host vectors, the host objective and eval loops read them there, and the device-resident objective uploads its own copy keyed by dataset identity. There is no consumer that could take a device pointer, so the honest implementation is a single D2H of `n` floats, which is `1/n_features` of the matrix. Eval sets stay host-side entirely, because the per-iteration eval predicts on the host.

**Fallback.** `cuda_ingest` declines when a dataset's total bins exceed the resident path's shared-memory ceiling, on the reasoning that grow would decline into the host plane anyway and the device bins would be wasted. That reasoning inverts for device-resident input: there is no host copy of the raw matrix to fall back to, so declining would mean downloading 4 bytes per cell instead of materializing 1 or 2. The device arm therefore never declines, and a grower that does declines into the plane's lazy host materialization, the path decision 99 already uses for a device-binned `Dataset` handed to a CPU grower.

**The benchmark contract does not move.** The standings hand every arm the same host numpy array, because that is the workflow the ladders measure and the only contract every arm accepts; nothing here changes that, and no ladder switches to device input. Measuring this feature is a separate study with its own spec, in which every arm that supports device-resident input (bonsai here, XGBoost's `QuantileDMatrix`, LightGBM where it applies) is measured on it, and the host-input standings stay the published comparison.

## Rejected

- **A `DeviceBins` side channel on `Dataset` without the narrative verb** (this design's first draft): identical mechanics, but the API grows by exception instead of by vocabulary: the convolutedness the transaction narrative exists to prevent. Superseded by the ingest transaction.
- **Engine-side rebinning** (host bins → device rebin): saves nothing; the host `transform` cost is the line item.
- **Retaining raw floats on Dataset** so the engine can bin later: +6.4GB host RSS at 16M for a copy the ingest hook can stream through one 64MB device chunk.
- **Device mapper-fit** in this round: RNG-identical reservoir sampling on device is not worth inventing; see phase 2.
- ~~Transposing kernel only for CSV~~; corrected during implementation: the module path bins straight from the borrowed row-major numpy view (`features_view`), and CSV parses into the feature-major `ColumnBatch`; the row-major arm is therefore the primary (bench) arm. Both arms shipped.
