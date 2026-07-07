# 11 — GPU-resident growing

> **Status:** landed (decision 40; commits 556310b, a1a40c4, a644358 on cuda-phase2). A100 MSD ladder: 13.1 s (phase 2) → 7.5 (stage A) → 5.4 (stage B) → **5.0 s** (stage C) vs xgboost-GPU 1.8 s; RMSE 8.9911 at every stage; 392/392 both configs; CPU builds bit-identical. Grow-loop time is ~2.5 s — near xgboost's train phase — with CSV parse + binning (~2.5 s, outside this design's scope) now the largest remaining block. Baseline evidence: [reviews/2026-07-03-design-review-cuda.md](../reviews/2026-07-03-design-review-cuda.md).
>
> **Grower-side seam superseded by [`12-grower-backend.md`](12-grower-backend.md) (decision 41, landed):** the optional-hooks model this doc designed is replaced by the `LevelStep` compile-time strategy, `resident()` is retired, and the phase-2 copy-back path is deleted in favour of a CPU fallback. *The seam* below is kept as the phase-3 design record; the device-kernel stages (A/B/C) and the level state machine remain current.

## Why

On an A100, `cuda_depthwise` fits MSD in 12.2 s of which kernels are only 2.0 s — already at parity with xgboost-GPU's entire 1.8 s fit. The other 10 s is host orchestration: per-level row uploads (1.4 s), histogram copy-back + unpack (1.3 s), CPU-fallback populate (0.5 s), and ~7 s of host-side growing (split finding, partitioning, gh packing, bookkeeping — to be pinned by `BONSAI_GROW_PROFILE=1` before stage A starts). Phase 3 moves the per-level data plane onto the device and shrinks the bus traffic to decisions and counts.

## The seam

Phase 2 established the idiom: `populate_many` is an *optional* batched hook on the builder, detected via `if constexpr` + `requires`, so the `HistogramBuilder` concept stays two-method and the CPU builder never changes. Phase 3 adds three more optional hooks, detected the same way:

```cpp
// Best split per node from device-resident level histograms. Returns one
// SplitOutput per node plus the winning cut's child sums (2 HistCells per
// node) — everything propagate_monotone_bounds and finalize_as_leaf need,
// so histograms never come home. Host keeps the min_data_in_leaf pre-check.
void find_splits_many(..., std::span<SplitOutput> out, std::span<HistCell> child_sums);

// Partition every split node's device row segment; returns per-child row
// counts (covers, next-level eligibility, smaller-child pairing). The host
// remains the decision-maker; the device executes.
void partition_many(..., std::span<uint32_t> child_counts);

// End of tree: download the per-row leaf assignment (~2 MB). Host fills
// values[] from its leaf table and runs route_unsampled unchanged.
void finalize_rows(std::span<node_id_t> leaf_ids_out);
```

The grow loop survives as the single algorithm narrative. Each level step becomes a named `grower_detail` helper that dispatches host-or-device internally: `find_splits` → `partition_level` → `populate_level` → finalize. `SplitterT` remains the host fallback — used when the hooks are absent, when the resident path degrades (below), and by parity tests — so `CudaDepthwiseGrower = DepthwiseGrower<HistogramNodeSplitFinder, CudaHistogramBuilder>` keeps meaning what it says. No new typelist dimension (the restraint of decisions 26/32).

`SplitInput` gains `HistCell sums` and `size_t row_count`. In resident mode `hists` and `rows` stay empty — the struct degrades to node metadata (id, monotone bounds, interaction mask, path, sums, count); `totals()` reads `sums` and the finder's row-count guard reads `row_count`. The inspect path for tests and debugging is an explicit `download_histograms(node)` on the CUDA builder.

Copy-back mode (the phase-2 path) was retained through phase 3 as the degrade path, then retired by decision 41 once the resident path had proven out: today `begin_root` declining (oversized `max_bin`, buffers won't fit) falls back to **CPU** histogram building through the engine's own CPU member, and the oblivious/leafwise growers use the host `LevelStep` plane.

## Level state machine

What crosses the bus per level is the design's core; each stage deletes rows from this table.

| transfer | phase 2 | stage A | stage B |
|---|---|---|---|
| H2D rows (concat lists) | ~4 MB × levels | same | — (resident, root upload once per tree) |
| H2D gradients | once per tree | same | same (interleave on device in stage C) |
| D2H histograms | ~n_nodes × 90 × 256 × 16 B | — | — |
| H2D node metadata (lo/hi, masks) | — | ~16 B/node (+masks only if constrained) | same |
| D2H split results | — | ~56 B/node | same |
| D2H child row counts | — | — | 8 B/node |
| D2H leaf ids | — | — | ~2 MB once per tree |

## Stage A — device split finding + resident level histograms (measured: 13.1 → 7.5 s)

Landed (commit 556310b): A100 MSD fit 7.5 s vs the 13.1 s baseline, RMSE 8.9911 unchanged, 392/392 both configs. Profile after: upload 0.73 + device 0.86 + find-sync 1.34 + partition 1.6; unpack, CPU-fallback, and host find read zero. Two implementation lessons: small nodes need the direct-global kernel (row-proportional) rather than the shared-memory kernel (bin-proportional) once no CPU cutoff exists, and the one-lane FP64 scan is ~2 ms/node on the Jetson iGPU (1/32-rate FP64) while immaterial on the A100 — a warp-parallel scan is deferred until a target platform needs it.

Two ping-pong device buffers hold parent/child level histograms (≈94 MB each at depth 8 × 90 features × 256 bins; a `begin_tree` capacity check degrades the whole fit to copy-back mode when they would not fit). A subtraction kernel mirrors `finish_split` (larger child = parent − smaller, double precision, driven by per-level slot triples). The per-node CPU fallback disappears in resident mode — `k_min_gpu_rows` existed to amortize per-launch copy-back, which no longer exists; small nodes ride the batched launch.

The find kernel (grid: feature × node) runs a sequential double-precision prefix scan over the cut cells in the same order as the CPU scan — both `default_left` routings, `min_child_hess`, monotone feasibility via the header-inline `bounded_leaf_weight` (directly usable in the .cu TU; the single-toolchain design paying off), interaction mask, `min_gain_to_split`. A reduce kernel picks the per-node winner with the same lowest-feature-id tie-break as `reduce_in_feature_order`, structurally preserving the CPU's choice up to floating-point rounding.

## Stage B — device partitioning + resident rows (measured: 7.5 → 5.4 s)

Landed (commit a1a40c4): A100 MSD fit 5.41 s, RMSE 8.9911 unchanged, 392/392 both configs. Host partition fell 1.6 → 0.19 s and per-level row uploads are gone; remaining attributed time is the per-tree gradient upload (~0.6 s, stage C's device-interleave target) plus ~2 s of real device kernel work — and CSV parse + binning outside the grow loop is now the largest single block, a post-phase-3 candidate.

The device keeps decision 17's shape: one concatenated row array in level order plus per-node (offset, count) segments — exactly what `populate_many` already assembles per level, made persistent and double-buffered. Partitioning is three kernels per level: route flags (same `goes_left` logic: bin compare, last-bin routes by `default_left`), a hand-rolled exclusive scan (block scan → block-sums scan → offset add; no CUB/Thrust, keeping the TU self-contained), and a stable scatter that carries the ordered gradients along — which also deletes the per-level gather kernel. Stability preserves the CPU's ascending per-node row order. A leaf-assignment kernel stamps `leaf_ids` at end of tree; `route_unsampled` and leaf renewal stay host-side and unchanged.

## Stage C — residual trims (measured: 5.4 → 5.0 s)

Landed (commit a644358): gradient interleave moved on-device (raw grad/hess uploads feed an interleave kernel, deleting the serial per-tree float2 pack) and the find kernel assigns one selected feature per warp lane instead of one per 32-lane block — full utilization, identical per-feature scan and tie-break (find-sync 1.33 → 0.98 s). Pinned staging was evaluated and skipped: per-level transfers are already tens of microseconds. Syncs stand at two per level (split results, partition counts) plus one leaf download per tree.

## Determinism and parity

The `cuda_depthwise` contract is tolerance-equal, not bit-equal (established in phase 2: float shared-memory accumulation, atomic ordering). Device split finding preserves the CPU's scan order and tie-break structurally, but equal-within-rounding gains may pick different splits, changing tree *structure* while model quality is equivalent. Parity tests therefore assert prediction/RMSE tolerance and a high split-agreement rate — never tree equality. CPU-only builds remain bit-identical and untouched.

## The benchmark loop

Development happens against the MSD ladder on a rented GPU node; workflow, tooling, and regression tracking are documented in [`benchmarks/README.md`](../../benchmarks/README.md) (GPU node workflow). A limited-context session needs only that section and this document.

## What's not here

Device-side objective gradients, multi-GPU, CUDA graphs, a `[cuda]` config section for the cutoffs, device oblivious/leafwise variants — phase-4 candidates, deliberately excluded. Risks tracked for implementation: the ~7 s host attribution is estimated until the grow profiler pins it; the depth-driven memory degrade path must be tested, not just designed; the hand-rolled scan kernel gets a standalone unit test against `std::exclusive_scan` before it drives partitioning.
