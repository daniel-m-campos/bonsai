# 20: Device leafwise: a slot-pool plane for best-first growth

> **Status:** design, not built (issue #268). Decision 42 withdrew a `cuda_leafwise` registration because the level-batched resident plane cannot serve a frontier of one, and deferred a true device leafwise pending "a non-swapping advance". The 2026-08 recheck (decision 95) reopened the question: LightGBM's CUDA leaf-wise is real GPU performance at scale and now beats bonsai's CPU leafwise 5.3x at 16M rows. This doc is the design for that non-swapping advance. Kill criterion, pre-registered in issue #268: beat lgbm_cuda's fit time at 16M x 100 on the same pod, or do not register.

## The claim

Device leafwise does not need new histogram, partition, or split-scan math. It needs a different histogram storage discipline (a per-tree slot pool instead of the per-level ping-pong), a persistent per-leaf segment map instead of the per-level layout vectors, and slot indirection in the one kernel that lacks it. The existing kernels are already frontier-width-agnostic where it matters: `hist_kernel` and `hist_small_kernel` take a slot table, the partition chain is segment-local (one op touches only its own `[offset, offset+count)` range), and `find_kernel` needs the same slot table `hist_kernel` already has. The new engine surface is a second concept beside `GPULevelEngine`, not a change to it; the depthwise and oblivious paths stay exactly as decisions 53/71/72 left them.

The performance thesis comes from reading LightGBM's CUDA learner (`src/treelearner/cuda/` at master, 2026-08): its 24.4 s at 16M x 100 x 100 iters x 256 leaves is not a bandwidth result but a per-split-round constant of roughly 950 us times 25,500 rounds (255 splits per tree x 100 trees). The constant is fixed overhead that does not shrink as leaves shrink: a hard-coded minimum histogram grid of 160 row-blocks per feature partition (`cuda_histogram_constructor.hpp`, `min_grid_dim_y_`), so every round pays ~8M global double atomics in the shared-to-global epilogue regardless of leaf size; 8 full `cudaDeviceSynchronize` per round; ~16 kernel launches per round, four of which run 6 to 20 threads; and a full-width histogram subtraction and split scan every round. bonsai's plane has none of these by construction: grids track node size, the per-level sync floor is one pinned D2H (doc 16), and the find scans only selected features. A leafwise plane that keeps per-round fixed cost near 100 us lands the same histogram volume as `cuda_depthwise` (the subtraction trick makes total work identical at matched knobs) plus ~2.5 s of serialization overhead at 16M, comfortably under 24.4 s against the 17.2 s depthwise anchor on the issue #268 pod.

## Lessons taken from LightGBM's implementation

Four things worth adopting, four worth refusing. File references are to the LightGBM repo.

**Adopt: the slot pool with in-place subtraction.** LightGBM allocates one flat histogram pool of `num_leaves` slots per tree, zeroed once per tree. Each split builds the smaller child's histogram into a fresh slot and derives the larger child by in-place subtraction in the parent's slot, which the larger child then inherits (`cuda_data_partition.cu`, `SplitTreeStructureKernel`; `cuda_histogram_constructor.cu`, `SubtractHistogramKernel`). Slot demand is exactly `num_leaves` (root plus one per split), allocation is a monotonic counter, and nothing is ever freed or recycled mid-tree. This is precisely the non-swapping advance decision 42 asked for: it does not work around the ping-pong's wholesale memset, it removes the ping-pong.

**Adopt: only the two new children are ever recomputed.** LightGBM caches each leaf's best split on device and rescans only the leaves the last split created. bonsai gets the same property for free by keeping the heap on the host, which is already the CPU leafwise structure (`grower_impl.hpp`) and already doc 12's contract (host owns all decisions). No device argmax over leaves is needed; the 56 B `FeatBest` per new child crossing the bus per round is the cache.

**Adopt: a fat split record.** LightGBM's ~120 B `CUDASplitInfo` carries both children's sums, counts, and values out of the split kernel so partition needs no second reduction. bonsai's `FeatBest` plus the host-side split output already carries what the heap and the next round's staging need; keep the round trip to two small transfers.

**Adopt: adaptive accumulator width, eventually.** LightGBM's quantized-gradient path picks 8/16/32-bit histogram accumulators from leaf row count, halving histogram traffic exactly where round counts are highest. It is off by default there and out of scope for stage 1 here, but it is the natural follow-up lever if the histogram build dominates after admission.

**Refuse: any minimum grid floor** (their `min_grid_dim_y_ = 160`), **device-wide syncs as ordering** (8 per round; bonsai's round needs 2), **bookkeeping kernels** (their 4-, 6-, and 20-thread launches; bonsai keeps node bookkeeping on the host where it already lives), and **a second device copy of the binned matrix** (LightGBM keeps row-major for histograms and column-major for partitioning; bonsai's partition chain reads the same feature-major matrix the histogram kernels read).

One finding for the ledger rather than the design: LightGBM's CUDA learner ignores `max_depth` entirely. `CUDASingleGPUTreeLearner::Train` overrides the serial learner's loop and never calls `BeforeFindBestSplit`, the only site that enforces depth; there are zero occurrences of `max_depth` under `src/treelearner/cuda/`. This is the code-level confirmation of decision 95's measurement. bonsai's device leafwise honors the depth cap exactly as the CPU leafwise does, because the cap is enforced by the host grower, not the plane.

## The plane

A new `LeafPipeline` beside `LevelPipeline` in the device context, sharing `DeviceData` (the binned matrix, per-dataset) and `GradientPlane` (the interleaved `float2`, per-tree) unchanged.

**Histogram pool.** One buffer of `num_leaves * n_selected * stride` doubles (`stride = 2 * max_sel_bins`, same layout as today's slots), memset once per tree. At the issue #268 knobs (256 leaves, 100 features, 255 bins) that is 104 MB, the same as LightGBM's pool. Slot assignment: root takes slot 0; each split builds the smaller child into the next free slot (monotonic counter) and subtracts in place in the parent's slot, which becomes the larger child's slot. A new `subtract_inplace_kernel` variant reads and writes the same buffer (today's `subtract_kernel` assumes parent and child in different buffers); the hist kernels need nothing, they already write through `out_slot[]` indirection onto zeroed memory.

**Segment map.** `slot_offsets` and `slot_counts` become persistent per-leaf host vectors keyed by slot id, updated one entry at a time, instead of per-level vectors rebuilt wholesale by `layout_children`. The row array keeps its invariant: a permutation of `[0, n)` where every live leaf owns a contiguous range. One split rewrites exactly one range into two adjacent subranges.

**Non-swapping partition.** The existing `route_count_kernel` / `seg_scan_kernel` / `scatter_kernel` chain runs with `n_ops = 1`, scattering the parent's range into the same offsets of the other row buffer, followed by a D2D copy of that range back into the primary buffer (rows plus the ordered `float2` gradients that travel with them). The b-side row and gradient buffers the level plane uses for ping-pong become this scratch; no new allocation. Cost: two extra passes over the split node's rows, zero sibling traffic, no buffer flip. `leaf_by_row` stamping and the tree epilogue (`finalize_rows`, `finalize_tree`) are reusable as is once the segment map is persistent.

**Find with slot indirection.** `find_kernel` currently addresses histograms as `node * n_sel + sel`, assuming the frontier occupies slots 0..n-1 contiguously. It gains the same `slot[]` table argument `hist_kernel` has, so a launch over the two new children names their actual pool slots. The scan math, tie-break, and `reduce_kernel` are untouched, preserving the single definition of splitting semantics shared with the CPU finders.

**Root.** The GPU `LevelStep::make_root` returns a root whose host-side histogram and row views are empty, so the CPU splitter would produce single-leaf trees; the leaf plane cannot reuse it. `leaf_begin_root` uploads the root segment, builds slot 0, and runs the device find on it, returning the root's `FeatBest` to seed the heap.

## The round

Per heap pop, in launch order on one stream:

1. Host stages one `PartOpDev` for the popped leaf (20 B H2D).
2. Partition chain (3 kernels) plus the range copy-back (1 D2D); child counts come down (8 B, sync 1). The host now knows smaller and larger.
3. Host stages the hist triple; `hist_kernel` or `hist_small_kernel` (by the existing 512-row policy) builds the smaller child into a fresh slot; `subtract_inplace_kernel` derives the larger in the parent's slot.
4. Host stages the two children's sums and bounds (32 B up, from the split output it already holds); `find_kernel` + `reduce_kernel` over the two children; two `FeatBest` come down (112 B, sync 2).
5. Host pushes both children into the gain heap and continues. Heap order, tie-break, monotone and interaction propagation, empty-child demotion, the leaf budget, and the depth cap all stay in the grower, unchanged from CPU leafwise.

That is 6 to 7 launches, 2 pinned syncs, and under 200 B of bus traffic per round. The budget that decides admission is the fixed per-round cost F: fit time is roughly the depthwise compute volume plus `rounds x F`, with `rounds = (leaves - 1) x iters` = 25,500 at the protocol knobs. At F = 100 us that is 2.6 s of overhead at any scale; against the 17.2 s depthwise anchor it prices a ~20 s fit at 16M, beating 24.4 s. At 250k rows, where lgbm_cuda fits in 5.7 s, the same 2.6 s of overhead sits on top of ~0.5 s of compute: still ahead, but it shows why F is the number to defend. LightGBM's F is ~950 us; the design goal is an order of magnitude below that, and stage 0 measures it before any plane code is written.

The known risk is occupancy, not sync latency. Deep in a 256-leaf tree the median split touches tens of thousands of rows; at 32k-row chunks a 31k-row node launches `n_selected` blocks total, underfilling an L40S's 142 SMs. Two mitigations, in order: shrink the chunk size as node size falls so the grid stays wide (more cross-chunk merge traffic, bounded and proportional to actual rows, never a fixed floor), and the small-node kernel already covers the tail. If admission shows the plane compute-bound on mid-size nodes anyway, the priced fallback is batched expansion: pop the top K heap entries and run one K-node launch through the same multi-node kernels. That is an algorithm change, not a tuning knob, because strict best-first order can differ when a child of the first pop would have outranked the second pop; it would ship, if ever, as a measured separate mode, never as a silent default, since the parity contract with CPU leafwise (tolerance-equal predictions, decision 40's contract extended) only absorbs rounding differences, not ordering differences. Speculative pipelining of round i+1's histogram under round i's partition is compatible with this plane (LightGBM's barriers foreclose it; ours don't) and is deferred alongside CUDA graphs to a post-admission optimization round.

## The seam

A new `GPULeafEngine` concept beside `GPULevelEngine` in `grower.hpp`: `leaf_begin_tree`, `leaf_begin_root` (returning the accept/decline bool and the root's split), `split_leaf` (partition + counts), `build_leaf` (hist + subtract + find on the two children), plus the shared `stamp_leaves` / `finalize_rows` / `finalize_tree` epilogue. `LeafwiseGrower<CudaHistogramEngine>` follows the level plane's one-runtime-fork pattern: `leaf_begin_root` declines wholesale (same `hist_budget_ok` predicate, plus a pool-budget check on `num_leaves * slot_doubles`) and the existing CPU `split_node` path takes over; there is no per-node CPU fallback on the resident path. Decision 41's finding stands: no `LevelPlan` unification, no new typelist dimension; the grower composes engine calls directly, exactly as the CPU leafwise composes `level_step.hpp` primitives today.

Out of scope for stage 1, wired later: the device-resident objective (`LeafwiseGrower::resident_begin` is a static false today; the resident plane's per-fit machinery is orthogonal and joins after admission), and multi-GPU (doc 19 is parked; nothing here forecloses it).

## Memory and decline gates

The pool is the only new allocation: `num_leaves * n_selected * stride * 8` bytes (104 MB at the protocol knobs; 2.1 GB at 2048 columns x 256 leaves x 255 bins). The decline predicate extends `hist_budget_ok` with the pool bound so oversized configurations fall back to CPU leafwise wholesale at `leaf_begin_root`, visible to both callers, per the existing single-predicate rule. Rows, gradients, flags, and the leaf assignment array are the level plane's buffers reused; peak device memory does not exceed the depthwise path's at equal knobs plus the pool delta (the level plane's transient 2n-slot histogram high-water mark at the widest level is comparable to the pool at matched leaf budgets).

## Profiling and measurement discipline

Per-round laps must replicate the sync peel decisions 62 and 72 were bitten by: a lap that opens without draining the previous round's kernels absorbs them. The `BONSAI_CUDA_PROFILE` accumulator pattern (event pairs recorded at launch, read at the next profile sync) extends with per-round buckets: partition, hist, subtract, find, and the host residue, so F is directly observable, not inferred.

## Admission

Feature-admission applies; the burden of proof is on the plane.

- **Stage 0, price F before building.** A harness that replays the round cadence (7 launches on representative grids, 2 pinned syncs, host staging) 25,500 times on the target L40S. If F lands above ~300 us, the thesis is wrong and the design returns to the batched-K question before any engine code exists.
- **Stage 1, parity.** The plane behind `cuda_leafwise`, tolerance-equal against CPU `leafwise` (prediction/RMSE tolerance, never tree equality), depth cap honored, decline paths exercised, `bonsai info` truthful about where histograms are computed. The decision 42 anti-goal is the test: no CPU histogram work under the `cuda_` name.
- **Stage 2, the ladder.** Same-pod L40S: `cuda_leafwise` vs `lgbm_cuda` vs CPU `leafwise` vs the `cuda_depthwise` anchor at 250k/1M/4M/16M x 100, protocol knobs, plus one uncapped-depth arm (explicit `num_leaves`, no depth cap) where leaf-wise structurally differs and where lgbm_cuda's ignored depth cap makes the comparison honest. The kill criterion is issue #268's: beat lgbm_cuda at 16M x 100 on the same pod, or do not register. `cuda_leafwise` must also beat same-pod CPU leafwise everywhere it registers; a device grower that loses to its own CPU arm does not ship.

Evidence chain: decision 42 (the withdrawal and the constraint), decision 95 (the recheck that reopened it), issue #268 (the target and kill criterion), `benchmarks/results/leafwise-recheck-2026-08.jsonl` (the ladder this design must move).
