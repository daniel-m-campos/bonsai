# 11 — Performance engineering as a compute DAG

## The idea

Chapter 10 showed *where* the GPU boundary sits. This chapter is about **how you find out where it should sit** — the method behind the July 2026 campaign that took the 16M-row fit from ~43s to 26.9s and past xgboost-GPU, told through its real moves and, more instructively, its real refutations.

The reframing that made the campaign systematic: training is a small **compute DAG**. Nodes are the algorithmic steps (bin, gradients, per level: build/find/partition, epilogue, score update); each node has a measured cost per feasible placement (host or device); edges carry data, and an edge crossing the placement boundary costs `bytes / bandwidth(direction)`. Choosing an implementation *is* choosing a placement plus a schedule. General DAG placement is NP-hard; this DAG has ~10 node types and at most six with free placement, so **exhaustive enumeration is trivial** — the entire difficulty is honest constants. ([architecture/16](../architecture/16-compute-dag.md) is the reference; `scripts/dag_model.py` is the living evaluator.)

Three rules fall out, each purchased with a real mistake:

1. **Price before betting.** An edge move must state its model price from measured constants before anyone writes a kernel.
2. **A node you can't decompose is a node you can't optimize.** Aggregate profiler lines mislead; every line must split into wait vs work, transfer vs compute.
3. **Conservation flushes dark matter.** The node costs must sum to the measured total; when they don't, the gap *is* the next target.

## The math

The model is deliberately primitive — it exists to kill bad bets, not to simulate:

```math
T(\text{placement}) \;=\; \sum_{v \in \text{nodes}} c_v(\text{side}(v)) \;+\; \sum_{(u,w)\,:\,\text{side}(u)\neq\text{side}(w)} \frac{\text{bytes}(u,w)}{\beta_{\text{dir}}}\,(1 - \text{overlap})
```

with per-direction bandwidths $\beta$ measured on the same machine (H2D pageable measured ~14–19GB/s; D2H is the slow direction and host-dependent). Two honesty constraints govern its use: constants come from **same-pod** profile runs only (two identical-model GPUs measured 25% apart across the fleet), and a move is played only if it wins across the plausible constant range — *dominance over precision*.

The model also yields a **floor**: with every feasible node on device, $T \geq$ one-time ingest transfer + total kernel compute + the irreducible per-level sync (the host must observe each level's split decisions before opening the next — that is the control-plane contract, and it pins one tiny D2H per level forever). When enumeration says remaining placements sit within noise of the floor, the placement game is over *by arithmetic* and only kernel engineering remains.

## The campaign, replayed as moves

Every row is a real decision from [decisions.md](../decisions.md); the deltas are same-pod measurements.

| move | change on the graph | priced | measured |
|---|---|---|---|
| 49 row-wise fill | host node cost (cache behavior, not placement) | — | populate 1.6–1.7× |
| 53 §2 rows cache | delete a 64MB/tree H2D edge | ~0.4s | root staging 0.42→0.04s |
| 53 §3 epilogue | 16M-row host loop → device map + bulk D2H | several s | finalize 9.4→3.9s |
| **52 device gradients** | delete the 12.8GB/fit gh H2D edge | **~0.9s → NO-GO** | experiment confirmed ~1.6s; killed |
| **35 pinned epilogue D2H** | reroute D2H via pinned + memcpy | **unpriceable** — finalize was an aggregate | refuted: 3.78→4.45s (*worse*) |
| 54 device binning | delete 4.6s host bin node + 1.6GB edge; add a 6.4GB streamed edge | ~4.5s | fit 37.9→31.3s |
| 38 buffer recycling | delete a 12.8GB/fit host *memset* node | ~6s | fit 34.7→26.9s |

Four of these deserve the space:

- **Decision 52 is the cautionary tale for rule 1.** "Keep gradients device-resident" sounds obviously right — until you price the edge it deletes: 128MB/tree over a ~19GB/s bus is under a second of a 42s fit. A pod-day experiment confirmed what one line of arithmetic would have said. (The experiment still paid: it flushed out a live device-plane bug, PR #29.)
- **PR #35 is rule 2.** The finalize line read 3.9s and intuition said "pageable D2H is slow, pin it". But finalize was an *aggregate* — stamp kernels + map kernel + sync + copies — and the actual copy share was small; the pinned route added a 128MB/tree memcpy and measured **worse**. The counters that would have priced it correctly (`fin_wait`/`fin_d2h`) were added the next day, and no line has been designed against un-decomposed since.
- **Decision 54 is the canonical "min-bytes ≠ min-time".** Device binning *increases* boundary traffic 4× (6.4GB of raw floats instead of 1.6GB of binned bytes) and still wins big, because the edge is cheaper than the 4.6s host node it deletes — and the model, fed the measured gh-edge bandwidth, re-priced the design's own draft (which had guessed 2.4s for the transfer; it's ~0.5s) *while the design was being written*.
- **PR #38 is rule 3 twice over.** Conservation said fit − grow − ingest left **12s attributed to nothing**. New buckets named it in one pod run: the per-tree zero-initialization of the output vectors — 12.8GB of `memset` per fit that every profiler had filed under "grow". Deleting it (the booster recycles the buffers; every element is provably written before read) was worth more than any kernel this campaign — and its correctness proof is the byte-identical model hash with tree *n+1* starting from tree *n*'s garbage.

## In bonsai — what the abstraction looks like as C++

The DAG is not a diagram on the side; it is load-bearing in the code's shapes:

- **Transactions as the API** — [`src/level_step.hpp`](../../src/level_step.hpp): `make_root` / `open_level` / `apply_level` / `end_tree`, identical vocabulary on both planes, with `ingest` as the zeroth verb (decision 54). The transaction boundary *is* the DAG cut; `LevelOutputs` and the epilogue are its edge payloads. Backends are implementation details behind a concept, not a base class — static dispatch in the narrative.
- **Planes as structs with lifetimes** — `CudaHistogramEngine::Impl` is three nested structs (`DeviceData` per dataset, `GradientPlane` per tree, `LevelPipeline` per level) so a buffer's lifetime is its type's, not a comment's.
- **Opaque receipts + backend tags at TU firewalls** — [`IngestPlane`](../../include/bonsai/dataset.hpp) is the one sanctioned virtual interface besides `IBooster`: the host TU cannot name CUDA types, so compile-time dispatch is impossible *by construction* there — and even then, recognition is a TU-local tag address, not RTTI. Everywhere else: concepts, `if constexpr (requires ...)`, and typelists.
- **Profile-gated laps** — every phase brackets itself with a `Lap` that is a no-op unless the env var is set ([`include/bonsai/detail/perf.hpp`](../../include/bonsai/detail/perf.hpp)), including profile-only `cudaDeviceSynchronize` calls that split *wait* from *work* in async lines. The constants in the model are these lines; instrumentation ships **before** the optimization it prices, in the same PR.
- **Immovable buffers** — `DeviceBuffer` deliberately has no copy *or move*; aggregates holding one are filled by out-param. The compiler enforcing "device memory does not silently relocate" caught a bug in this very campaign at CI time.

## Try it

```bash
# The model, with the campaign's constants — evaluate placements and the floor:
uv run scripts/dag_model.py --floor

# Reproduce a ledger line yourself (any machine with a CUDA device):
BONSAI_GROW_PROFILE=1 BONSAI_CUDA_PROFILE=1 BONSAI_FIT_PROFILE=1 BONSAI_INGEST_PROFILE=1 \
  bonsai bench --config configs/california_housing.toml --hp dispatch.grower_name=cuda_depthwise
```

Then check conservation: does `fit-profile`'s total explain the wall clock? Does `grow` equal the sum of `grow-profile`'s buckets? If not, you have found the next chapter of this story.

## Gotchas & war stories

- **Fleet variance is not noise you can average away.** 39.5s vs 49.0s for identical code on two same-model pods. Every claim in this chapter is a same-pod delta; any cross-machine absolute in a benchmark table should make you reach for the raw jsonl.
- **Intuition has a shelf life of about a week.** The 2.4s-vs-0.5s transfer mis-estimate in decision 54's draft came from constants remembered from a *different bus generation*. The model beat its own author because it refuses to remember — it only reads measurements.
- **Refutations are deliverables.** Decisions 52 and 35 are written up with the same care as the wins, because the cheapest optimization is the one arithmetic kills before implementation. If your decision log contains only successes, your method isn't producing knowledge — just survivors.
