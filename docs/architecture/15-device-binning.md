# 15 — Device binning: the ingest plane joins the backend

> **Status:** proposed (decision 54). The last big ingest lever that does not change the model, planned against same-pod ledgers (PR #34/#35 runs) and the pipeline facts below.

## The ledger line

Post-decision-53, the 16M×100 `cuda_depthwise` fit spends (same-pod L40S, US-MO-1; fit 39.4s): find 7.8s, ingest 8.4–8.8s (mapper-fit ~3.9 + bin ~4.6), populate 5.2s, finalize ~3.8s, partition 1.6s. The `bin` line is host CPU work: `fill_binned` runs `BinMapper::transform` (a `lower_bound` over the feature's cuts) once per cell — 1.6G binary searches — then `ensure_dataset` re-copies the result into pinned staging and uploads 1.6GB, unlapped, inside the first `begin_tree`. The device then reads *only the device copy* for the rest of the fit.

xgboost's 16M edge (27.9s vs 37.4s in the re-baseline) is device binning: raw values go over PCIe once and the quantized matrix is built where it is consumed.

## Pipeline facts the design must respect

- **Raw floats are not retained.** Both ingest paths (`ColumnBatch` from the Python module, `features_view` from CSV) are consumed by `Dataset::bin`; only binned columns survive. Device binning must therefore hook the ingest step itself — after `Dataset` construction the raw data is gone.
- **Cuts are tiny and host-fitted.** `BinMappers::fit` subsamples ≤200k values per feature (`n_samples`), sorts once, strides (decision 51). Cuts per feature ≤ `max_bin` floats. `transform` semantics: NaN → last bin, else `lower_bound(cuts, x)`.
- **In device mode, host bins have exactly two consumers.** (1) The fallback decline (`begin_root` refusing oversized `max_bin` → full CPU data plane); (2) `route_unsampled`'s `bin_at` random access, only when row sampling is on. The device plane partitions, finds, and stamps on device (stages A–D); host partition/populate arms run only in fallback mode.
- **The stub build must stay CUDA-free.** `Dataset` is built long before any engine exists and cannot name CUDA types.

## Proposed shape

### An opaque device plane on Dataset

`Dataset` gains a `std::shared_ptr<DeviceBins>` — an opaque handle declared in `bonsai/cuda/device_bins.hpp`, defined inside the CUDA TU, null everywhere else (mirrors the lazy `row_major_` mirror precedent). It owns the feature-major device matrix (`u8` or `u16`), the per-feature bin counts, and the identity fields `ensure_dataset` checks today.

```cpp
// dataset.hpp — no CUDA types
class DeviceBins;                       // opaque; defined in the CUDA TU
std::shared_ptr<DeviceBins> device_bins_;  // null unless device-binned
```

`CudaHistogramEngine::ensure_dataset` adopts the handle when present (no host read, no staging copy, no upload) and keeps its current path as the fallback for host-binned datasets.

### Binning on device at ingest

A free function in the CUDA TU, called from `Dataset::bin` behind the handle-request flag:

```
bin_on_device(raw columns or row-major view, cuts tables) -> DeviceBins
```

- **Transfer**: raw floats stream through a double-buffered pinned staging pair (bounded, ~64MB each) with `cudaMemcpyAsync` — upload of chunk *k+1* overlaps the bin kernel of chunk *k*. Raw data never fully resides on device; total device footprint = staging + the binned matrix (same as today).
- **Kernel**: one thread per cell; binary search over the feature's cut table in shared memory (cuts ≤ 255 floats/feature fit easily; u16 datasets read the table from global). `ColumnBatch` chunks are feature-major (kernel is a straight map); `features_view` chunks are row-major (kernel writes transposed — coalesced on the read side, the write pattern is the same scatter `fill_binned` does today).
- **Exactness**: the kernel reproduces `transform` exactly — NaN → last bin, else `lower_bound`. Same cuts, same total order on floats ⇒ **bit-identical bins** to the host fill, which is the whole byte-identity argument: CPU-path models are untouched by construction, and device-path models must equal the before-models exactly (r² equality gate, as PR #34).

### Who asks for it

`Dataset::bin` cannot know the grower. The train pipelines can: `bonsai::train` / the CLI pipeline request device binning when the config's grower name has the `cuda` prefix (the `trains_here` convention) **and** `cuda_available()`. Plain `Dataset` construction — tests, CPU workflows, predict — never touches the device path.

### Host bins go lazy in device mode

When device binning ran, host binned columns are not materialized at ingest. The two consumers get them on demand:

- **Fallback decline**: decided by `max_bin` vs the shared-memory ceiling — both known at ingest (the ceiling probe is safe once `cuda_available()`). If the dataset would decline, bin on host eagerly as today and skip the device path entirely; no lazy machinery on this arm.
- **`route_unsampled` under row sampling**: first `bin_at` triggers a one-time D2H materialization of the host columns (1.6GB pageable, ~1.5s, once per fit), cached like `row_major_bins()`. Row sampling off — the bench and the common CUDA configuration — never pays it.

## What this buys (projection, same-pod discipline)

Replaced: host `bin` ~4.6s + unlapped 1.6GB upload ~0.5s. New cost: 6.4GB raw over PCIe, overlapped with the bin kernel ≈ 2.3–2.6s (pinned, PCIe4). **Projected: fit 39.4 → ~36.8–37.3s on the US-MO-1 host class.** Cross-pod absolutes are meaningless (~25% fleet spread measured between two L40S pods); the gate is the same-pod before/after delta.

## Instrumentation shipped with the round

- `ingest-profile` gains `dbin` (device-binning transfer+kernel) so the before/after decomposes.
- `cuda-upload-decomp` gains `bins_upload` (the `ensure_dataset` copy+upload, today's dark matter) and finalize lap counters (`fin_wait`/`fin_d2h`) — the PR #35 refutation showed the finalize line is undecomposed and misleads design; never again.

## Phase 2 (not this round): mapper-fit

The remaining ~3.9s is `create_subsample`'s reservoir scan (`std::ranges::sample` over a filter view: 16M reads + per-element RNG, per feature). Any device or algorithmic change to sampling changes the sampled set → different cuts → **model-changing**; it needs its own decision with quality data, and is deferred until the byte-identical levers are exhausted.

## Rejected

- **Engine-side rebinning** (host bins → device rebin): saves nothing — the host `transform` cost is the line item.
- **Retaining raw floats on Dataset** so the engine can bin later: +6.4GB host RSS at 16M for a copy the ingest hook can stream through 128MB of staging.
- **Device mapper-fit** in this round: RNG-identical reservoir sampling on device is not worth inventing; see phase 2.
- **Uploading raw row-major and transposing on device for the module path**: `ColumnBatch` is already feature-major; the transposing kernel exists only for the CSV `features_view` arm.
