# 13 — Full device residency: gradients first, binning second

> **Status:** REFUTED by measurement (decision 52, PR #28): skipping all
> per-tree gradient uploads saved 3.7% of the 16M fit — `upload_s` is
> per-level staging latency, not gradient bytes. Retained as the record of
> two successive misattributions and the experiment that caught them; the
> live lever is batching/pinning the per-level `Staged<>` syncs.

## Why, quantitatively

The re-baseline (PR #24) leaves one structural GPU gap: at 16M×100,
`bonsai_cuda_depthwise` fits in 38.0s vs `xgb_cuda` 27.9s. The profile says
`cuda_upload` is 6.0s — but the one-time binned-column upload is 1.6GB,
~70ms at measured pinned rates. The upload time is dominated by what
`begin_tree` does **every tree**: re-upload grad and hess (2 × 16M × 4B ×
100 iters = **12.8GB per fit**) plus the interleave pass, because the
booster computes objectives against host-resident scores.

The dependency is circular by design today: device trees finalize leaf
values → host booster updates host scores → host objective produces
grad/hess → engine re-uploads. Breaking it means making **scores, objective
gradients, and the per-tree score update device-resident**, with the host
receiving only what it observes (eval metrics, early-stop signals, final
model). That is the same architecture xgboost trains under, and it retires
the largest single line in the 16M profile.

## Phase A — device-resident gradients (the 12.8GB line)

- Device buffers for `scores`, `grad`, `hess` sized n_rows, owned by the
  engine across the fit (a new `begin_fit`/`end_fit` bracket on the engine
  concept, or lazily keyed on the Dataset identity like `ensure_dataset`).
- After each tree: the device already holds per-row leaf assignments
  (`finalize_rows`); a fused kernel applies `scores += lr * leaf_value[row]`
  and immediately computes next-iteration `grad/hess` from the objective —
  one pass over n_rows, no host round-trip.
- Objectives become device-instantiable: MSE and logloss are pointwise
  (architecture doc 4); the objective's `compute` gains a device twin under
  the same dispatch that selected the CUDA engine. Constant-hessian leaf
  renewal (`renew_leaves`) and DART's dropped-tree bookkeeping force a host
  fallback for those configs in phase A — documented, not silently wrong.
- `route_unsampled` and warm-start score rebuilds read scores at fit end:
  one D2H of n_rows floats per fit, not per tree.
- **Numerics:** device objective math in the same precision as host
  (float scores, float grad/hess) — per-row pointwise ops, no reductions,
  so CPU/GPU parity tolerances stay where the histogram merge set them.

Projected from the profile: retire most of `cuda_upload` (6.0 → ≲0.5s) and
the host objective share currently invisible inside the fit wall.
**38 → ~31–32s.**

## Phase B — device binning (demoted from lead)

Still worth doing, but honestly sized. Host binning cannot be skipped while
`route_unsampled` and the CPU fallback read host columns, and uploading raw
floats (6.4GB) to save a 1.6GB bins upload is a regression — so phase B is
only the **mapper-fit + ingest-bin tail** (3.8s + 4.9s at 16M), and only
via a lazy-host-bins or device-`route_unsampled` follow-up with its own
correctness argument. Sequenced after phase A because phase A's win is
2–3× larger for less contract risk. Phase A's device leaf assignments are
also exactly what a device `route_unsampled` needs — B gets cheaper after A.

## Testing

- Parity: existing CUDA grower suites, plus a multi-iteration booster
  parity case (10 trees, device vs host boosters, R² within the
  established tolerances) — the new risk is drift across iterations, which
  single-tree tests can't see.
- Fallback coverage: DART and renewal configs must route to the host path
  and match it exactly.
- Bench gate: 16M×100 `bonsai_cuda_*` before/after on one pod; accept at
  ≥1.15× fit; `cuda_upload` must drop below 1s.

## Rejected

- **Device binning first** (this doc's own first draft): the upload term it
  targeted was misattributed; the honest ingest tail it can reach is half
  the size of the gradients line and carries a raw-float-lifetime API
  question besides.
- **Keeping per-tree uploads but compressing gh** (fp16): halves a line
  that phase A deletes, and injects precision risk into the one place the
  f32-cells round deliberately kept double reductions.
- **Whole-booster device port** (eval, sampling, DART on device): the
  booster's control flow is cheap on host; only the O(n_rows)-per-tree data
  plane pays for residency.
