# 11. Performance engineering as a compute DAG

## The idea

Chapter 10 showed *where* the GPU boundary sits. This chapter is about **how you find out where it should sit**: the method behind the July 2026 campaign. The campaign cut the 16M x 100 fit from about 43s to 26.9s, and XGBoost-GPU measured 27.9s in [decision 52](../decisions.md). This chapter replays the campaign's moves and its refutations, because the refutations teach more.

The reframing that made the campaign systematic: training is a **compute DAG** (directed acyclic graph: steps are nodes, data flow between steps is edges). Nodes are the algorithmic steps (bin, gradients, per level: build/find/partition, epilogue, score update); each node has a measured cost per feasible placement (host or device); edges carry data, and an edge crossing the placement boundary costs `bytes / bandwidth(direction)`. Choosing an implementation *is* choosing a placement plus a schedule. General DAG placement is NP-hard; this DAG has ~10 node types and at most six with free placement, so **exhaustive enumeration covers at most 2^6 = 64 placements**: the entire difficulty is honest constants.

Three rules fall out, each purchased with a real mistake:

1. **Price before betting.** An edge move must state its model price from measured constants before anyone writes a kernel.
2. **A node you can't decompose is a node you can't optimize.** Aggregate profiler lines mislead; every line must split into wait vs work, transfer vs compute.
3. **Account for the whole total.** The node costs must sum to the measured total, and any unexplained gap is the next target.

## The math

The model is deliberately primitive. It exists to kill bad bets, not to simulate:

```math
T(\text{placement}) \;=\; \sum_{v \in \text{nodes}} c_v(\text{side}(v)) \;+\; \sum_{(u,w)\,:\,\text{side}(u)\neq\text{side}(w)} \frac{\text{bytes}(u,w)}{\beta_{\text{dir}}}\,(1 - \text{overlap})
```

with per-direction bandwidths $\beta$ measured on the same machine (host-to-device, H2D, copies from pageable (not pinned) host memory measured 14 to 19 GB/s; the device-to-host, D2H, rate depends on the host). Two honesty constraints govern its use: constants come from **same-pod** profile runs only (a pod is one rented GPU machine, and two identical-model GPUs measured 25% apart across the fleet), and a move is played only if it wins across the plausible constant range: *dominance over precision*.

The model also yields a **floor**: with every feasible node on device, $T \geq$ one-time ingest transfer + total kernel compute + the irreducible per-level sync (the host must observe each level's split decisions before opening the next; that is the control-plane contract, and it pins one tiny D2H per level forever). When enumeration says remaining placements sit within noise of the floor, the placement game is over *by arithmetic* and only kernel engineering remains.

## The campaign, replayed as moves

Every row is a real decision from [decisions.md](../decisions.md) or, where labelled PR, a pull request; the deltas are same-pod measurements.

| move | change on the graph | priced | measured |
|---|---|---|---|
| 49 row-wise fill | host node cost (cache behavior, not placement) | n/a | fit 1.6–1.7× (populate 2.1×) |
| 53 §2 rows cache | delete a 64MB/tree H2D edge | ~0.4s | root staging 0.42→0.04s |
| 53 §3 epilogue | 16M-row host loop → device map + bulk D2H | several s | finalize 9.4→3.9s |
| **52 device gradients** | delete the 12.8GB/fit gh H2D edge | **~0.9s → NO-GO** | experiment confirmed ~1.6s; killed |
| **PR #35 pinned epilogue D2H** | reroute D2H via pinned + memcpy | **unpriceable** (finalize was an aggregate) | refuted: 3.78→4.45s (*worse*) |
| 54 device binning | delete 4.6s host bin node + 1.6GB edge; add a 6.4GB streamed edge | ~4.5s | fit 37.9→31.3s |
| PR #38 buffer recycling | delete a 12.8GB/fit host *memset* node | ~6s | fit 34.7→26.9s |
| 72 identity contract | delete a 64MB/tree host identity copy + its staging | 33ms/round | grow/round 181→148ms |
| 72 device root sums | 16M-row host reduce → deterministic two-pass kernel | ~12ms/round | landed within price |
| 72 final-level skip | delete the last level's write-only histogram build | ~18ms/round | grow/round → 125ms |
| **72 epilogue sync, memset, pinned gh** | three more levers from the same table | **0.1ms / 0.5ms / break-even → killed** | never implemented |

Four of these deserve the space:

- **Decision 52 is the cautionary tale for rule 1.** "Keep gradients device-resident" sounds obviously right, until you price the edge it deletes: 128MB/tree over a ~19GB/s bus is under a second of a 42s fit. A pod-day experiment confirmed what one line of arithmetic would have said. (The experiment still paid: it flushed out a live device-plane bug, PR #29.)
- **PR #35 is rule 2.** The finalize line read 3.9s and intuition said "pageable D2H is slow, pin it". But finalize was an *aggregate* (stamp kernels + map kernel + sync + copies) and the actual copy share was small; the pinned route added a 128MB/tree memcpy and measured **worse**. The counters that would have priced it correctly (`fin_wait`/`fin_d2h`) were added the next day, and no line has been designed against un-decomposed since.
- **Decision 54 is the canonical "min-bytes ≠ min-time".** Device binning *increases* boundary traffic 4× (6.4GB of raw floats instead of 1.6GB of binned bytes) and still cuts the fit from 37.9s to 31.3s, because the 6.4GB transfer (~0.5s) costs less than the 4.6s host binning node it deletes, and the model, fed the measured gh-edge bandwidth, re-priced the design's own draft (which had guessed 2.4s for the transfer; it's ~0.5s) *while the design was being written*.
- **PR #38 is rule 3 twice over.** Summing the buckets showed fit − grow − ingest left **12s attributed to nothing**. New buckets named it in one pod run: the per-tree zero-initialization of the output vectors, 12.8GB of `memset` per fit that every profiler had filed under "grow". Deleting it (the booster recycles the buffers; every element is provably written before read) cut the fit from 34.7s to 26.9s (7.8s), more than device binning's 37.9s to 31.3s, and its correctness proof is the byte-identical model hash with tree *n+1* starting from tree *n*'s stale buffer contents.
- **The decision-72 rows are the method compressed into one round.** The target was the levelwise grower's 155ms marginal round (decision 71's residue). Stage 0 built the price list before any lever: a profile-only sync peel replayed the decision-62 lesson (6.1s filed under find staging was really the previous level's histogram kernels draining at the next sync), and the sum check found two residues nothing else explained, the identity copy and the final level's write-only build. Six levers were priced from that one table; three landed for ~63ms. The other three were killed before implementation, priced at 0.1ms, 0.5ms and break-even per round. Result: 181→125ms per round (fit 19.4→13.9s), r² four-decimal identical, CPU hash byte-identical. What remains is ~72ms of histogram kernel plus ~32ms of partition and bus, which is a *kernel engineering* boundary, not a placement one: the floor section's distinction, measured. The whole round is walked step by step, instrument by instrument, in [the marginal round](../learn/engine/1-the-marginal-round.md).

## In bonsai: what the abstraction looks like as C++

The DAG is not a diagram on the side; it is load-bearing in the code's shapes:

- **Transactions as the API**: [`src/step/level.hpp`](../../src/step/level.hpp): `make_root` / `open_level` / `apply_level` / `end_tree`, identical vocabulary on both planes, with `ingest` as the zeroth verb (decision 54), implemented as `Dataset::bin` on the host and `cuda_ingest` in [`include/bonsai/cuda/histogram_engine.hpp`](../../include/bonsai/cuda/histogram_engine.hpp) on the device. The transaction boundary *is* the DAG cut; `LevelOutputs` and the epilogue are its edge payloads. Backends are implementation details behind a concept, not a base class: static dispatch in the narrative.
- **Planes as structs with lifetimes**: the device context is six planes, each a struct with its own lifetime (`DeviceData` and `ResidentPlane` per fit, `GradientPlane` and `LeafPipeline` per tree, `LevelPipeline` per level, `EvalPlane` per fit for the validation set; the table is in [chapter 10](10-gpu-training.md)), so a buffer's lifetime is its type's, not a comment's.
- **Opaque receipts and backend tags at translation-unit boundaries**: [`IngestPlane`](../../include/bonsai/bin_store.hpp) is the one sanctioned virtual interface (runtime dispatch through a base class) besides `IBooster`. The host-side source file (translation unit) cannot name CUDA types, so compile-time dispatch is impossible *by construction* there. Even then, the backend is recognized by the address of a tag local to that source file, not by RTTI (C++ runtime type information, `dynamic_cast` and `typeid`). Everywhere else: concepts, `if constexpr (requires ...)`, and typelists.
- **Profile-gated laps**: every phase brackets itself with a `Lap` that is a no-op unless the env var is set ([`include/bonsai/detail/perf.hpp`](../../include/bonsai/detail/perf.hpp)), including profile-only `cudaDeviceSynchronize` calls that split *wait* from *work* in async lines. The constants in the model are these lines; instrumentation ships **before** the optimization it prices, in the same PR.
- **Immovable buffers**: `DeviceBuffer` deliberately has no copy *or move*; aggregates holding one are filled by out-param. The compiler enforcing "device memory does not silently relocate" caught a bug in this very campaign at CI time.

## Try it

```bash
# Reproduce a ledger line yourself (needs a CUDA device and a binary from `make build-cuda`):
BONSAI_GROW_PROFILE=1 BONSAI_CUDA_PROFILE=1 BONSAI_FIT_PROFILE=1 BONSAI_INGEST_PROFILE=1 \
  bonsai bench --config configs/california_housing.toml --set dispatch.grower_name=cuda_depthwise
```

Then check that the buckets add up: does `fit-profile`'s total explain the wall clock? Does `grow` equal the sum of `grow-profile`'s buckets? If either check fails, the unexplained gap is your next optimization target.

## Case study: reading a competitor's implementation

Before building the device leafwise plane, we read LightGBM's CUDA tree learner end to end and sorted what we found into adopt and refuse lists. The exercise generalizes: a competitor's shipped code is a free experiment log, and the discipline is deciding which of their choices answer *your* constraints.

Adopted, with their source sites: the flat histogram **slot pool** with in-place subtraction (`cuda_histogram_constructor.cu`, `SubtractHistogramKernel`), which removes the ping-pong memset instead of working around it; recomputing **only the two new children** per round, which bonsai gets free by keeping the split heap on the host; the **fat split record** (~120 B) that spares a second reduction; and adaptive accumulator width as a future lever, off by default even there.

Refused, because they answer constraints bonsai does not have: a minimum grid floor, device-wide syncs as ordering (8 per round where bonsai's round needs 2), 4-thread bookkeeping kernel launches (bonsai keeps node bookkeeping on the host), and a second device copy of the binned matrix in a different major order.

One finding worth more than the design notes: LightGBM's CUDA learner ignores `max_depth` entirely; its train loop never calls the one site that enforces depth, and `max_depth` appears nowhere under `src/treelearner/cuda/`. We found it by measurement first (a depth-capped comparison that refused to cap) and confirmed it in their source. Reading the competitor's code did not just donate design ideas, it explained a benchmark anomaly that would otherwise have cost a rerun.

## Gotchas & war stories

- **Fleet variance is not noise you can average away.** 39.5s vs 49.0s for identical code on two same-model pods. Every claim in this chapter is a same-pod delta; any cross-machine absolute in a benchmark table should make you reach for the raw jsonl.
- **Remembered constants go stale.** Decision 54's draft guessed 2.4s for the transfer. Priced from the measured 19GB/s host-to-device edge, it is about 0.5s.
- **Refutations are deliverables.** Decision 52 (device residency, refuted by experiment) and the PR #35 pinned-epilogue refutation (cited in decision 54) are written up with the same care as the wins, because the cheapest optimization is the one arithmetic kills before implementation. If your decision log contains only successes, your method isn't producing knowledge, just survivors.
