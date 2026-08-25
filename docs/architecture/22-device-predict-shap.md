# 22: Device predict and device TreeSHAP: serving the resident Dataset

> **Status:** landed for every width-1 model, dense and levelwise (decision 111); performance ledger pending the first datacenter pod session, and the levelwise arm's throughput is unmeasured. Correctness is device-verified on a Jetson Orin (sm_87) for both arms: predict bit-equal to the host walk, SHAP worst element gap 1.17e-7 and worst additivity residual 3.22e-7 against the fp64 reference, over five fixtures.

## The problem this solves

The production loop this serves is: bin a Dataset on the device once, train, eval, run TreeSHAP for feature selection, adjust, refit, at millions of rows. Before this work every Model method wanted a raw host matrix, so a device-built Dataset could train but not predict or explain, and `pred_contribs` ran a minute-scale host walk whose output the caller immediately reduced. The device already held the bins; nothing could read them after the fit.

## The route ladder

Every X-taking Model method now accepts a Dataset, and the binding dispatch picks the best route the Dataset supports. Cuts match the model's own: route bins, which is exact (bin(v) <= split_bin iff v <= cuts[split_bin], the same equivalence the deferred eval switch trusts), needs no raw rows, and is what makes device-resident (DLPack) builds servable. Cuts differ but the host matrix was retained: the raw walk, unchanged. Neither: an error naming both remedies.

The bin-space walk lives once in `detail/bin_walk.hpp` and serves the booster's `*_binned` family (`predict_at`, `predict_staged`, `predict_leaf`, `predict_proba`, `pred_contribs`), each pinned bit-equal to its raw twin. Above that, two device planes serve the two hot calls; everything else rides the host bin walk deliberately, because staged and leaf outputs are transfer-dominated.

## Device predict

`cuda_predict_plan` packs the whole ensemble into the SoA node form the training epilogue already walks (`route_add_kernel`'s shape), thresholds inverted to bin ids under the model's mappers; `cuda_predict` runs one thread per row, looping trees, accumulating in a register, no atomics. The plan is a receipt cached under the booster's mutation epoch, so a sweep of predict calls uploads the trees once. The seam into the booster is a single virtual, `device_plan_input()`, returning trees, learning rate, init score, the epoch, and an optional owner for the trees; multiclass boosters return the empty default and decline to the host walk.

A levelwise model reaches both planes through that owner (decision 111). Its trees are oblivious, so it attaches the epoch-cached dense equivalent the host SHAP path already builds and points the span at that. Nothing else moves: the packers take `DenseTree` either way, and they copy and upload at pack time, so the owner has only to outlive the pack call. Densification mints a perfect tree, so a depth-d levelwise tree packs 2^d paths whatever its live coverage, and the slots with no evidence behind them pack verbatim carrying a zero cover fraction. Those slots are unreachable rather than merely unvisited: an oblivious tree can split one feature at two levels, so the expansion holds corners that assert `f <= t_i` and `f > t_j` at once. What the kernel meets is a zero-cover element in a path the row does not follow, where the host form divides by that fraction and guards the branch off while the closed form only multiplies and needs none. The plan input is also no longer free to ask for on this arm, so the callers in `module.cpp` ask for it after the gates that do not need it.

Bit-equality with the host walk holds row for row on device, but only after spelling the final composition as `__fadd_rn(init, __fmul_rn(lr, acc))`: the kernel TU compiles with contraction on, and the naive form fused one multiply-add, drifting 1 ulp on 5 percent of rows. The lesson generalizes: a device walk with no atomics can match the host bit for bit, and when it does not, the cause is a specific rounding, findable and fixable, not noise to tolerate.

## Device TreeSHAP

The host algorithm is Lundberg's Algorithm 2, whose recursion copies its path vector at every two-branch node. The device design replaces the recursion with per-leaf-path closed forms over a packed representation:

- **8-byte path elements.** Each merged element carries a u16 feature id (high bit: missing ok), an inclusive u8 bin interval, and an fp32 cover fraction. Merging happens at pack time by interval intersection. The missing bin cannot fold into the interval (a default-left split admits `[0,s]` plus the last bin, not an interval), so a per-feature `last_bin` sidecar completes the membership test: `bin == last ? missing_ok : lo <= bin && bin <= hi`.
- **Division-free evaluation.** Along a fixed path every one-fraction is 0 or 1, so the extend polynomial is `P(t) = prod (z_j + o_j t)` over merged non-root elements, with Shapley weights `w_i = 1/(n * C(n-1, i))`. All unsatisfied elements share one weighted sum; each satisfied element deflates its monic factor by synthetic division. No divisions in the loop, hence no cancellation site, hence fp32 is defensible on device. Zero-cover branches fall out of the arithmetic with no guards.
- **The bias never rides the walk.** The per-tree expected value is row-independent; the host was recomputing it per (row, tree). It is now computed once per tree (also a 17 to 30 percent host-side win on its own) and enters the device path as one fp64 scalar in the host epilogue.

The closed form was proven against `tree_shap` element-wise before any kernel existed: 2.0e-15 worst relative gap on cover-exact trees (summation-order roundoff only), 8.8e-9 on real ensembles, which is exactly the fp32 `zero_fraction` storage and nothing else. The kernel is one thread per (row, path), polynomial coefficients in registers under a merged-length template in {8, 16, 32}, `atomicAdd` into global fp32 phi, with lr and bias applied in a double-precision host epilogue that mirrors the host composition term for term. Measured on device: worst element gap 1.37e-7, additivity residual 2.1e-7 against a 1e-5 tolerance.

## What we race, and the edges

XGBoost 3.3 replaced GPUTreeShap with QuadratureSHAP: an 8-point Gauss-Legendre reformulation, node-native, O(depth x 8) per row per tree, on both CPU and GPU. The race is against that engine, and three structural edges are independent of which algorithm the competitor runs. First, bins: both competitors dequantize to float in the hot loop (xgboost's ellpack loader does a bin lookup and then a dequantize before every compare); bonsai compares u8 bin ids against u8 bounds. Second, caching: neither competitor caches its compressed model across calls, re-extracting per invocation; bonsai's plans are epoch-keyed receipts, so the repeated-explain loop pays packing once. Third, exactness: the closed form reproduces the Lundberg reference exactly, where quadrature is an approximation; the bench category carries a fidelity column beside the throughput race.

## Rejected: thread-per-row iterative DFS

A mechanical port of the recursion needs an explicit stack of path copies in local memory, about 900 bytes per thread at depth 6 growing quadratically, capping occupancy near 4.5 warps per SM and moving terabytes of local traffic at the target shape. It loses before it computes. It becomes relevant only for a future interventional SHAP (background-dataset-dependent recursion has no path decomposition), which is a separate admission.

## Decline gates

Every device path declines to the host bin walk, never errors, on capability shortfalls: no CUDA build or device, multiclass model (this cut), any feature over 255 bins (the 8-bit interval), merged path length over 32, foreign ingest plane, allocation failure. A missing-covers model stays a host-side error, because that is a model defect, not a capability gap.

## Pending the pod

The Orin validates correctness, not throughput. Open before any published number: the u16 dispatch arm has never run on a device; the async-mempool allocation path is untested off Tegra; the plan has no multi-device guard; the v1 SHAP geometry is deliberately unoptimized (no shared-memory row staging, atomic contention unmeasured, K=32 register pressure unpriced); and the pack-plus-upload amortization point is unmeasured. The gpu-shap standings axis is wired only when the first same-pod evidence file exists.
