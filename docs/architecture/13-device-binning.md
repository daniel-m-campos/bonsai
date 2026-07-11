# 13 — Device binning

> **Status:** proposed (decision 52). The last big documented lever from the
> scaling study (decision 46): xgboost-GPU's rows-amortization edge.

## Why

The re-baseline (PR #24) leaves one structural GPU gap: at 16M×100,
`bonsai_cuda_depthwise` fits in 38.0s vs `xgb_cuda` 27.9s. The profile
decomposes bonsai's fit as:

| stage | s | device-binning target? |
|---|--:|---|
| ingest_fit (host quantile mappers) | 3.8 | phase 2 |
| ingest_bin (host float→bin transform) | 4.9 | **phase 1** |
| cuda_upload (binned columns → device) | 6.0 | **phase 1** |
| cuda_gpu (hist/partition/find kernels) | 2.7 | already fast |
| grow_find + grow_partition (host wall incl. sync) | 8.7 | — |

Phase 1 replaces "bin 6.4GB of floats on host (4.9s), then push 1.6GB of
binned columns through a cold path (6.0s)" with "push raw float chunks
through pinned staging (~2.5–3s at measured 23.7GB/s) and bin on device
(microseconds per SM pass)". Projected: **38 → ~30s**. Phase 2 moves the
quantile fit's sample+sort per feature onto the device: **~30 → ~26s ≈
xgboost parity**. CatBoost's rows exponent (0.48 vs our 0.91) says this is
how the rows axis is won.

## Phase 1 design

- **Cuts to device once.** `BinMappers` stays host-fit (phase 1). Upload the
  per-feature cut arrays (≤64K floats total at max_bin≤255×100 features)
  into one packed device buffer with per-feature offsets.
- **Chunked upload+bin pipeline.** Raw X arrives column-major from the
  ColumnBatch path or row-major from the Python module. Stream it in
  feature-group chunks (e.g. 8 features × n_rows floats) through the
  existing `PinnedBuffer` staging: H2D copy of chunk *k+1* overlaps the bin
  kernel of chunk *k* on two streams. Transient VRAM = 2 chunks, not the
  full float matrix — the 16M×100 case needs ~512MB transient, and the
  VRAM clamp arithmetic in `bench_scaling` stays valid.
- **Bin kernel.** One thread per row per feature in the chunk:
  `bins[f][r] = lower_bound(cuts[f], x)` as a branchless binary search over
  the packed cuts (≤8 steps at 255 bins). Semantics must match
  `BinMapper::transform` exactly — same right-inclusive cut convention,
  NaN → last bin — so device bins are **bit-identical** to host bins
  (integer output; no FP-accumulation caveats).
- **Output.** u8 device columns when `all_fit_u8` (the max_bin≤255 default),
  u16 otherwise — the same dual-width rule as `Dataset` (decision on
  visit_bins). The engine's `ensure_dataset` keeps its identity check; the
  device-binned buffers simply take the place of the uploaded host bins.
- **Host `Dataset` stays authoritative for CPU paths.** Device binning is an
  engine concern: when the CUDA engine owns binning, the *host* binned
  columns are never read by the GPU plane — predict-time routing and any
  CPU fallback still use the host columns, which remain bit-identical by
  construction.

## The API constraint phase 1 must solve first

Two facts bound the design more tightly than the pipeline above suggests:

1. **The engine never sees raw floats.** `CudaHistogramEngine` receives the
   already-binned `Dataset`; the float matrix lives upstream (the module's
   borrowed numpy span, the CLI's ColumnBatch). Device binning therefore
   needs `Dataset` to carry a non-owning `raw_features()` view, set by both
   `bin()` overloads — valid for the training call's duration under the
   same lifetime contract `FeatureBuffer::borrowed` already establishes.
2. **Host bins cannot be skipped, only their upload.** `route_unsampled`
   (booster leaf-value stamping for sampler-dropped rows) and the engine's
   CPU fallback read host binned columns during training. So phase 1's
   honest savings at 16M×100 is the 6.0s upload replaced by a ~2.5–3s raw
   stream, **not** the 4.9s host bin pass — projected 38 → ~33s, with the
   remaining ~3s recoverable only by making host binning lazy for rows the
   GPU path never touches (a follow-up with its own correctness argument,
   or by moving `route_unsampled` device-side alongside `finalize_rows`).

The 30s projection in the table above stands only if that follow-up lands;
review should treat ~33s as phase 1's committed number.

## Phase 2 sketch (own PR, after phase 1 numbers)

Device quantile fit: per feature, sample ≤n_samples rows on device (seeded
reservoir matching `create_subsample`'s selection exactly is the hard
determinism constraint — if infeasible at reasonable cost, fit stays host
and phase 2 is only worth the 3.8s when the sampling seed contract is
renegotiated), radix-sort the sample (CUB), read cuts at the ceiling
stride (decision 51). Cuts differing from host-fit cuts would change
models; **phase 2 does not proceed unless cuts are provably identical**.

## Testing

- Unit: device bins == host bins, element-exact, u8 and u16, NaN column,
  constant column, >255-distinct column (u16), chunk-boundary rows.
- Parity: existing CUDA grower tests run unchanged on device-binned data —
  identical bins mean identical histograms modulo the established float
  chunk merge tolerances.
- Bench gate: 16M×100 `bonsai_cuda_*` before/after on one pod; accept at
  ≥1.2× fit and no VRAM regression at the 131k×16384 wide cell.

## Rejected

- **Uploading the raw matrix whole** (6.4GB resident): simpler, but busts
  the VRAM clamps that let 16GB-class cards run the rows axis, for zero
  speed benefit over the chunked pipeline.
- **Binning in the histogram kernel on the fly** (no binned columns at
  all): re-does the binary search per (node, feature) visit instead of once
  per cell — the fill reads bins ~tree-depth × iters times each.
- **Device mapper fit in phase 1**: couples a models-change risk (cut
  reproducibility) to a pure-perf change; sequenced behind phase 1's
  numbers instead.
