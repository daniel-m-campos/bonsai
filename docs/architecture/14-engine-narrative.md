# 14: One engine narrative: the level transaction

> **Status:** executed (decision 53). Step 1 (transaction vocabulary) in
> PR #33; steps 2–3 (device rows cache, engine-owned tree epilogue) and the
> Impl decomposition in the follow-up change set. The API-consistency
> redesign the device-residency review commissioned, planned against the
> measured 16M ledger (PR #31) instead of the two premises measurement
> killed (doc 13).

## The commission

Review feedback on the device-residency design set the bar: CPU and GPU
engines must expose **consistent APIs that portray the same algorithmic
narrative, with the backend an implementation detail**, and
`CudaHistogramEngine::Impl` must stop being a god object (41 device/staged
buffers, 132 member references, every phase's scratch in one struct).

## What the ledger demands of the shape

The 16M×100 ledger (L40S, ~43s): GPU root path ~14s, ingest 9.4s, finalize
8.4s, find 8.2s (awaited kernel compute; true staging 0.03s), partition
1.7s. Two structural observations follow:

1. **The root is a special case that costs like a whole phase.** `begin_root`
   re-uploads the row list (64MB × 100 trees) and runs the root histogram
   through its own staging path. In the narrative, the root is just a level
   with one node: the special case exists only because the interface makes
   per-tree setup a different method from per-level work.
2. **Per-tree boundaries leak host round-trips.** finalize's 8.4s is a
   64MB D2H plus a 16M-row host stamping loop per tree, because "end of
   tree" isn't a first-class engine operation: the grow loop reaches into
   `finalize_rows` and does host bookkeeping the backend could own.

## Proposed shape

### The narrative types

One vocabulary, both planes (names bind to what docs 2/12 already say):

- `LevelInputs`: frontier metadata: per-node sums, bounds, constraint
  masks, selected features. One struct, one staging step.
- `LevelOutputs`: per-node split decisions + child sums.
- `TreeEpilogue`: per-row leaf assignment and train values, **produced by
  the engine** at tree end (host: the existing stamping loops; device: kept
  resident, materialized to host lazily or handed to the next consumer).

### The transaction interface

```cpp
concept GrowerEngine =
    begin_tree(ds, grad, hess)                    // per-tree data plane
 && open_level(LevelInputs) -> LevelOutputs        // find, batched staging
 && apply_level(partition ops, advance ops)        // rows -> children
 && end_tree() -> TreeEpilogue;                    // leaf ids/values
```

- The **root is `open_level` with one node**: `begin_root`'s bespoke
  staging (the 14s line) merges into the same batched path as every other
  level, and the row list uploads once per fit, not once per tree (rows
  are engine state across `apply_level` calls already on the device plane;
  the host plane's `SplitInput.rows` remain the working representation).
- `open_level` takes ONE struct, so the four `Staged<>` syncs per level
  become one transfer by construction: not an optimization pass later,
  a property of the interface.
- `end_tree` owning the epilogue is where finalize residency lives: the
  device plane can defer the 64MB D2H, fuse the score update
  (doc 13's fused kernel, now correctly motivated by the 8.4s finalize
  line rather than upload bandwidth), or hand `leaf_by_row` to
  `route_unsampled` device-side. The host plane's implementation is the
  code that exists today, relocated.

### Breaking up `Impl`

Four planes with the lifetimes the ledger exposed:

| plane | owns | lifetime |
|---|---|---|
| `DeviceData` | bins, cuts, n_bins, identity check | per dataset |
| `GradientPlane` | grad/hess/gh, later scores | per tree (per fit once resident) |
| `LevelPipeline` | hist slots, partition scratch, find scratch, staged inputs | per level, reused |
| `ProfileCounters` | (exists) | per process |

`CudaHistogramEngine::Impl` becomes composition of these; the host engine
grows the same seams as (thin) structs so the narrative reads identically
in both files.

## Migration

Three PRs, each green and rebaselined: (1) introduce the transaction types
and adapt the **host** plane (pure refactor, bit-identical); (2) move the
device plane onto them, folding `begin_root` into `open_level` (targets
the 14s line; parity + bench gates); (3) device `end_tree` epilogue
(targets the 8.4s line; the score-update fusion from doc 13 lands here if
its measured value justifies it). Each step keeps 449 green and the
existing parity suites as the contract.

## Rejected

- **Redesigning around residency or staging levers** (docs 13's two drafts):
  both premises died to measurement; the interface above *accommodates* the
  live levers without betting on any single one.
- **A single templated engine with `if constexpr` planes**: hides the
  narrative instead of unifying it; decision 41's host-primary/device-
  specialization split stays, with shared vocabulary.
- **Fusing find+partition+advance into one mega-transaction**: the control
  plane (leaf-vs-split decisions, constraint propagation) is host logic by
  design (doc 12); the transaction boundary sits exactly where control
  needs to observe results.
