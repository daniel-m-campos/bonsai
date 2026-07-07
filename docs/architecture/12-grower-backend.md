# 12 — Grower data-plane: the `LevelStep` strategy

> **Status:** landed (decision 41; commits b4d223c → a4764ba on cuda-phase2). Supersedes the grower-side *seam* of [`11-gpu-resident.md`](11-gpu-resident.md) — its device-kernel stages (A/B/C) stand unchanged; this doc replaces how the grow loop reaches them. Gates held at every phase: 392/392 both configs, CPU models bit-identical (sha256, all three growers), resident-path launch counts unchanged, A100 MSD fit within noise of the pre-refactor baseline.

## Why

`src/grower_impl.hpp` grew the CUDA path feature-by-feature without refactoring, and it shows. `grower_detail::update_nodes` is a 182-line, 18-parameter function that inlines Pass-1 bookkeeping, a device arm, a host arm, and the common tail. The host-vs-device seam is smeared across six sites as `if constexpr (ResidentHistogramBuilder<T>)` wrapping a runtime `if (builder.resident())` — in `make_root`, `find_splits`, `update_nodes`, and the tail of `DepthwiseGrower::grow`. `resident()` is opaque (it is re-queried six times though `grow()` already learns the mode from `begin_root`), and the three growers share no level vocabulary: Oblivious hand-inlines a depthwise-shaped level update and its own copy of `route_unsampled`. Doc 11 promised "each level step becomes a named helper that dispatches host-or-device internally"; the code never decomposed that way.

## Reframe: control plane vs data plane

The grow loop is the **control plane** — it owns every *decision*: leaf-vs-split, smaller-child pairing, monotone/interaction propagation, tree assembly. The mechanical *data plane* — find splits, partition rows, build children's histograms, finalize leaves — is what varies between CPU and GPU. Decision 40 already used these words ("the host grow loop remains the decision-maker; the device executes the data plane"); this design makes them structural by extracting the data plane into one object.

From the grower's point of view there are only **two** data planes, not three. CUDA copy-back is indistinguishable from pure-host: in both, the grower runs `SplitterT::find` on host histograms and `partition_rows` on host rows — the GPU-ness hides entirely inside `populate`. Only device-resident growing is genuinely different (histograms in device slot-buffers, rows in device segments, split-finding and partitioning on the device). So the data plane has exactly two implementations: **host** and **GPU**.

## `LevelStep` — a compile-time strategy

`LevelStep` groups the per-tree data-plane operations behind one interface, selected by engine *type* via partial specialization plus a concept. Selection happens at template instantiation, so it is zero runtime cost — no virtuals, no type erasure (the restraint of decisions 14/26/32). The name is doc 11's own phrase ("each level step becomes a named helper") and is deliberately distinct from the `*Grower` classes, which model `TreeGrower` and grow a whole tree.

```cpp
// Host data plane (src/level_step.hpp): serves every HistogramEngine — the CPU
// engine, and the CUDA engine's CPU fallback. Branch-free; no GPU concept appears.
// The constructor opens the tree (engine.begin_tree) and captures the per-tree
// context (engine&, ds, config, grad, hess, selected), so methods take only
// what varies per level.
template <HistogramEngine EngineT, typename SplitterT>
class LevelStep {
    SplitInput make_root(row_indices);                // populate + totals
    void find_splits(frontier, out&, child_sums&);    // per-node find, or one level find broadcast
    void partition(plan&);                            // partition_rows, one node per worker
    void build_children(plan&);                       // populate smaller + finish_split subtraction
    void finalize(frontier, nodes&, ...);             // host leaf stamping
};

// GPU data plane: only for a GPULevelEngine. Holds the per-tree mode; the ONE
// irreducible runtime fork (on-device vs CPU fallback) lives here and nowhere else.
template <GPULevelEngine EngineT, typename SplitterT>
class LevelStep<EngineT, SplitterT> {
    bool on_device;   // captured once inside make_root from begin_root's bool
    // each method: if (on_device) engine.<device op>  else  <host LevelStep op>
};
```

`grow()` then reads as pure control-plane narrative — no `if constexpr`, no `resident()` in sight:

```cpp
LevelStep<EngineT, SplitterT> step(engine, ds, config, grad, hess, selected);
current.push_back(step.make_root(row_indices));       // mode captured once, inside the step
while (depth < max_depth) {
    step.find_splits(current, splits, child_sums);
    auto plan = plan_level(current, splits, child_sums, nodes, ...);  // host bookkeeping only
    step.partition(plan);
    step.build_children(plan);
    commit_children(plan, covers, current, next);     // constraint propagation + hand-off
}
step.finalize(current, nodes, values, leaf_ids, row_indices);  // stamp leaves + pull device row ids
```

Two refinements over the original sketch, adopted at implementation: the step owns *root creation* (absorbing the old `make_root` fork — the mode never escapes the data plane), and it holds the per-tree context from its constructor — which, more than the decomposition itself, is what deleted the 18-parameter signatures. Find *granularity* resolves inside `find_splits`: a `LevelSplitFinder` runs once and broadcasts to every frontier node (the oblivious shape), a `NodeSplitFinder` loops — so the oblivious grower reuses the same step with zero new machinery.

`update_nodes` dissolves into `plan_level` (host bookkeeping) + the `LevelStep` methods + `commit_children` (constraint propagation and frontier hand-off) — each a helper of at most ~40 lines. The CPU path is fully branch-free; the GPU details are quarantined in the one specialization. And the design is extensible by construction: every grower builds `LevelStep<SplitterT, Engine>` and calls the same methods, so a future `CudaObliviousGrower` (one alias plus a registry line) gets the GPU plane for free — Oblivious broadcasts its single level split into the per-node `plan` the `LevelStep` consumes.

## Concept surface — two concepts

The three-tier ladder (`HistogramBuilder` → `BatchHistogramBuilder` → `ResidentHistogramBuilder`) was an artifact of the phased research (below). It collapses to two, and the survivors are renamed to shed the GoF-`Builder` connotation — a `*Builder` reads as incremental construction, but this is the pluggable compute substrate the `LevelStep` runs on (and the CUDA one supplies the whole device data plane, far more than "building"):

```cpp
concept HistogramEngine = requires { begin_tree; populate; };   // the host plane

// The GPU data plane: the device-resident vocabulary, taken all-or-nothing (there is one
// device implementation, coupled to the engine's state — decision 40's coupling still holds).
concept GPULevelEngine = HistogramEngine<T> && requires {
    typename T::LevelOp; typename T::PartitionOp; typename T::LeafStamp;
    begin_root; find_splits_many; partition_level; advance_level; stamp_leaves; finalize_rows;
};
```

`resident()` is gone: the mode is a value the grow loop captures once from `begin_root`'s `bool` return into the `LevelStep`, never a method re-queried on the engine. `BatchHistogramBuilder`/`populate_many` are gone (see below).

## How `LevelStep` bridges the two engine surfaces

The two concepts are a *refinement*, not two rival surfaces: `GPULevelEngine` is a `HistogramEngine` plus the device vocabulary, so `CudaHistogramEngine` satisfies both (it keeps `populate` for the CPU fallback and adds the device methods) while `CpuHistogramEngine` satisfies only the floor. That nesting is what drives selection — the `LevelStep` specialization constrained on `GPULevelEngine` is *more constrained* than the primary, so for a Cuda engine the compiler picks it, and for a Cpu engine only the primary is viable:

```cpp
template <NodeSplitFinder S, HistogramEngine E> struct LevelStep { ... };        // host (primary)
template <NodeSplitFinder S, GPULevelEngine  E> struct LevelStep<S, E> { ... };  // GPU (more constrained)
```

The grow loop never sees the difference — it calls `find_splits` / `partition` / `build_children` / `finalize` on one uniform interface, and `LevelStep` absorbs the surface gap below it. What each method reaches for:

| `LevelStep` method | host primary (drives `HistogramEngine`) | GPU spec (drives `GPULevelEngine`) |
|---|---|---|
| `find_splits` | `SplitterT::find` on host histograms | `on_device ? engine.find_splits_many : SplitterT::find` |
| `partition` | `partition_rows` (host) | `on_device ? engine.partition_level : partition_rows` |
| `build_children` | `engine.populate` (smaller) + `finish_split` | `on_device ? engine.advance_level : populate + finish_split` |
| `finalize` | host leaf stamping | `on_device ? engine.stamp_leaves + finalize_rows : host stamp` |

The surfaces differ because the two planes divide labor differently, and that division is not arbitrary. On the host, "find" is a *separate* pluggable policy (`SplitterT`) and "partition" is generic host code, so the engine only has to build histograms — the floor is exactly `begin_tree` + `populate`. On the GPU, find / partition / advance are device kernels coupled to resident state (slot buffers, row segments); they cannot live anywhere but inside the engine, so `GPULevelEngine` is larger precisely because device-residency fuses more of the data plane into the backend. The concept surface reports that fact rather than hiding it.

Two consequences worth naming. `SplitterT` looks vestigial on the GPU path because there it is the parity contract the device kernel must match plus the decline fallback, not the live finder (refinement 1 below). And the GPU specialization's `!on_device` column delegates to the *exact* host operations, which is why `CudaHistogramEngine` must still satisfy the floor: the refinement is load-bearing for the decline path, not cosmetic.

## Policy hierarchy — stress-tested across the use cases

`LevelStep` is a *derived* composition, not a new injection axis, so it is worth stating exactly what is injected and what falls out. Stress-testing against every growth × compute combination — ignoring which combinations the registry happens to build — settles the shape.

Only two axes vary independently: *growth* (the outer loop — depthwise / oblivious / leafwise) and *compute location* (CPU / GPU). Everything else is a function of those two — find *granularity* (per-node vs per-level) is fixed by growth (oblivious is level-wise, the others node-wise); histogram source and data residency are fixed by compute. So a good hierarchy injects two policies and derives the rest.

Filling in the matrix exposes two fractures in the naive "growth × compute is fully orthogonal" picture:

| | CPU | GPU |
|---|---|---|
| Depthwise | host `LevelStep`, node-find | GPU `LevelStep`, device node-find |
| Oblivious | host `LevelStep`, level-find | GPU `LevelStep`, device **level-find** |
| Leafwise | host `LevelStep`, node-find, heap loop | — no level to batch |

First, oblivious sums the whole frontier's histograms and picks *one* split, so oblivious/GPU needs a device *level*-find, not the per-node `find_splits_many` — find granularity threads into the device plane, not just the host. Second, the GPU plane's entire value is batching a frontier into one launch, so a leafwise frontier of one is the 185-launches anti-pattern phase 2 removed: **GPU-ness requires a level-batched frontier, which leafwise structurally lacks.**

The hierarchy that survives injects two policies, derives the plane, and makes the couplings type-level facts rather than conventions:

```cpp
Grower<Engine, Finder = Grower::DefaultFinder>   // Growth is the class; two injected deps
  //  using Frontier      = ...;   // level-vector | gain-heap
  //  using DefaultFinder = ...;   // node | level  — granularity fixed by Growth
  └─ LevelStep<Finder, Engine>    // data plane, DERIVED — not a registry axis
       // host: uses Finder + Engine::populate;  gpu: uses Engine device ops
```

Four refinements this surfaces over a bare `Grower<SplitterT, Engine>`:

1. The finder is grower-*declared* (`DefaultFinder`), overridable only for tests — the depthwise-with-a-level-finder mismatch becomes impossible by construction, and the finder stops reading as a vestigial parameter on the GPU path (it is a default, and it is the parity contract the device find must match).
2. Find *granularity* threads into `LevelStep` on both planes, so `GPULevelEngine` grows a level-find sibling to `find_splits_many` for oblivious/GPU.
3. Leafwise shares the data plane's *primitives*, not its level batching (resolved below).
4. The GPU `LevelStep` specialization is concept-constrained to level-batched growth, so leafwise/GPU is a compile-time non-selection (it stays host), not a silent slow path.

The result keeps the two-injected-policies / derived-plane / no-new-axis shape, but the couplings the current alias gets right by hand become constraints the compiler enforces. The only genuinely new capability the full matrix demands is the device level-find; everything else is the same primitives recomposed.

*Resolved at implementation (phase 5):* full singleton-frontier unification was attempted per the original lean and rejected at the gate — a per-heap-pop `LevelPlan` adds ceremony and allocations in the hot best-first loop for zero shared-code gain, because `split_node` already composes the same `level_step.hpp` primitives (`partition_rows` + `populate` + `finish_split`). Landed instead: leafwise opens its tree through the `LevelStep` (constructor + `make_root`), and `split_node` lives in `level_step.hpp` documented as the *single-node* data plane — `level_step.hpp` is the complete data-plane home, and honest naming won over maximal hierarchy.

## Retiring the copy-back research

The `populate` → `populate_many` → resident progression was a development ladder — its stages are the phase-1/2/3 story preserved in git history and in doc 11's measured ladder. Only the last rung is worth keeping. So this design retires the intermediate machinery: `populate_many`, `BatchHistogramBuilder`, and the GPU **copy-back** histogram path (build on the device, copy home to `SplitInput.hists`). When the resident path declines — a feature's bins overflow the shared-memory budget, or the level buffers would not fit — the CUDA engine falls back to **CPU** histogram construction, delegating through the `cpu` member it already owns for sub-cutoff nodes. The default `max_bin = 255` always goes resident; the fallback is a rare path, and building it on the CPU (rather than a bespoke GPU-copy-back path) trades a slice of fallback throughput for a much smaller CUDA backend.

## Invariants preserved

The host owns all decisions; the GPU executes the data plane. There is one grow-loop implementation, not a per-backend fork. CPU-only builds stay **bit-identical**; `cuda_depthwise` stays **tolerance-equal** (not tree-equal). No new typelist or dispatch dimension enters — `LevelStep` is selected off the grower's existing `Engine` parameter. Cost is zero: the strategy is chosen at compile time, the GPU specialization carries one well-predicted branch per level, and nothing indirects per row.

## Verification

Both configurations build clean and pass `ctest` (392/392) at every phase: the host tests prove bit-identical CPU behavior, the `[cuda]` parity tests prove tolerance-equal device behavior. `BONSAI_CUDA_PROFILE=1` launch counts stay put on the Jetson. The zero-perf claim is *measured*, not asserted: `scripts/bench_gpu.py` runs the MSD ladder against xgboost on a Thunder 4×A100 before and after, and `fit_s` plus the xgboost gap must be unchanged within noise (bonsai and xgboost run sequentially — the node has wedged under concurrent CUDA startup).

## What's not here

Device oblivious **landed** (decision 42): `find_level_split` is the device level-find this paragraph anticipated, `cuda_oblivious` is registered, and its parity tests ride the cuda suite — see [`11-gpu-resident.md`](11-gpu-resident.md). Device leafwise — its gain-heap is inherently sequential (no level to batch — [`10-cuda.md`](10-cuda.md)), so its device value is limited and none is planned; a `cuda_leafwise` registration was built, benchmarked, and withdrawn for running CPU histograms under a GPU name (decision 42). One correction to this doc's earlier draft: the oblivious grower's *unsampled-row routing* does not unify with `route_unsampled` (it routes to ObliviousTree leaf-table indices, not DenseTree node ids) — only its level update went through the `LevelStep`. The device kernels and their optimization stages are unchanged and remain in [`11-gpu-resident.md`](11-gpu-resident.md); the engine policy originates as the histogram builder in [`10-cuda.md`](10-cuda.md); the ratifying choice is decision 41, superseding the framing of decision 40.
