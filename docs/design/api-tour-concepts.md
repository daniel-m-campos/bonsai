# Concepts to types

This page is the implementer's surface: the concepts you satisfy to add an objective, a grower, or a compute backend.
For the caller's surface (estimators, `train`, config keys), read [The API in one read](../use/api-tour.md) first; this page is its mirror image, one layer down.
Each section quotes the requires-clause, then cites why the concept is shaped that way and what a new implementation must honor.

## Dataset: bin once, train many

Histogram gradient boosting scans bins, not raw values, so binning happens once and every model reuses it.
`Dataset` is a concrete type, not a concept.
`Dataset::bin` is a static factory that runs the binning pass.
A later `train` over the same `Dataset` skips it, and stays bit-identical to training from the arrays.

```cpp
static Dataset bin(detail::ColumnBatch const &batch, BinMappers const &mappers,
                   DataConfig const &cfg,
                   std::shared_ptr<IngestPlane const> plane = nullptr);
```

Why this shape: column-major `u8`/`u16` storage keeps the histogram scan cache-friendly (decision 4, [dataset doc](../architecture/1-dataset.md)).
The optional `IngestPlane` lets a backend hand back binned columns it already produced on the device (decision 54).
Host columns then materialize only on first host read.
What a caller must honor: bin train, validation, and test through the same `BinMappers`.
Otherwise "feature 7 below bin 3" means different things across splits.
Lives in [`include/bonsai/dataset.hpp`](../../include/bonsai/dataset.hpp).

## Objective: compute, eval, init_score

A loss enters the engine only through its first two derivatives per row, so an objective is three functions.

```cpp
template <typename T>
concept Objective = std::constructible_from<T, Config const &> &&
    requires(T const &o, floats_view preds, floats_view targets,
             floats_out grad, floats_out hess) {
        { o.compute(preds, targets, grad, hess) } -> std::same_as<void>;
        { o.eval(preds, targets) } -> std::same_as<float>;
        { o.init_score(targets) } -> std::same_as<float>;
    };
```

Why constructible from `Config`: parameterized losses (Huber delta, quantile alpha) carry state.
Parameter-free objectives keep static methods, which still satisfy the instance call ([objective doc](../architecture/4-objective.md)).
The `renew_leaf` extension serves surrogate-hessian objectives (MAE, Huber, Quantile), where `h = 1` stands in for a second derivative that is zero or discontinuous. Their Newton step is a heuristic, so the booster replaces each leaf with the loss-optimal value over its residuals. MSE also has a constant hessian, exactly 1, but needs no renewal: its Newton step is the residual mean, already the exact minimizer of squared error.
They replace the Newton leaf value with the loss-optimal one.
The booster detects the method with `if constexpr (requires(std::span<float> r) { objective_.renew_leaf(r); })`, and skips the pass otherwise.
What a new objective must honor: `compute` overwrites grad and hess rather than accumulating (decision 24).
Add `renew_leaf` only when the Newton step differs from the loss-optimal leaf value, that is, when the hessian is a surrogate.
Lives in [`include/bonsai/objective.hpp`](../../include/bonsai/objective.hpp).

## TreeGrower: three growth strategies as types

Depthwise, leaf-wise, and oblivious are three answers to one question: which node to split next.

```cpp
template <typename T>
concept TreeGrower = requires(T g, Dataset const &ds, floats_view grad,
                              floats_view hess, row_index_view row_indices) {
    typename T::Tree;
    requires Tree<typename T::Tree>;
    { g.grow(ds, grad, hess, row_indices) }
        -> std::same_as<GrowResult<typename T::Tree>>;
};
```

The three strategies are three class templates: `DepthwiseGrower`, `ObliviousGrower`, and `LeafwiseGrower`.
Each is parameterized on its `HistogramEngine` and split finder.
Why this shape: growth is the outer loop, and compute location and find granularity fall out of it.
So a grower injects two policies and derives the rest ([grower-backend doc](../architecture/12-grower-backend.md)).
What a new grower must honor: expose a `Tree` type and return a `GrowResult`.
The booster owns sampling and scoring; the grower owns only the tree.
Lives in [`include/bonsai/grower.hpp`](../../include/bonsai/grower.hpp).

## HistogramEngine: two methods, one wide contract

Building per-bin gradient sums is the hot loop, and it is where CPU and GPU diverge.

```cpp
template <typename T>
concept HistogramEngine =
    requires(T b, Dataset const &ds, floats_view grad, floats_view hess,
             SplitInput &split_input, std::span<feature_id_t const> selected) {
        b.begin_tree(ds, grad, hess);
        b.populate(ds, grad, hess, split_input, selected);
    };
```

The concept checks two signatures, but the real contract is wider.
It lives in the comment above the concept.
`populate` must accumulate the node's rows into the mappers' bins.
It uses a fixed summation order, with missing values in the last bin.
That contract is enforced by the parity suite, not the compiler, which is the whole subject of [The HPC tension](the-hpc-tension.md).
What a new engine must honor: bin exactness and a deterministic reduction order.
A type that satisfies the syntax while bending either trains silently wrong models.
Lives in [`include/bonsai/grower.hpp`](../../include/bonsai/grower.hpp).

## GPULevelEngine: the device cluster, all or nothing

When histograms, rows, and split finding all stay on the GPU, only decisions and counts cross the bus.
`GPULevelEngine` refines `HistogramEngine` with the whole device vocabulary: `begin_root`, `find_splits_many`, `partition_level`, `advance_level`, `stamp_leaves`, `finalize_rows`, and the resident-objective seam.
Why one concept and not seven: the device data plane works whole or not at all.
So `begin_root` returns a single `bool` that the grow loop captures once as the per-tree mode ([grower-backend doc](../architecture/12-grower-backend.md)).
What a new device backend must honor: satisfy the floor `HistogramEngine` too, because the decline path falls back to the exact host operations.
Lives in [`include/bonsai/grower.hpp`](../../include/bonsai/grower.hpp).

## IBooster: the one type-erased boundary

The caller needs one runtime handle; the training loop needs zero runtime dispatch.
`Booster<Objective, TreeGrower, Sampler>` is monomorphized per cell, and `IBooster` is the single virtual base the factory returns.
The header calls `IBooster` "the one type-erased boundary" and keeps three client groups on it deliberately.
It does not split them into three interfaces.
Why deliberate: one vcall per `update_one_iter`, zero inside it (decision 26), so the whole inner loop stays static.
What a new client must honor: the minimal virtual surface of train, predict, introspect, and the CLI early-stopping seam.
Lives in [`include/bonsai/booster.hpp`](../../include/bonsai/booster.hpp).

## This page versus the user API

[The API in one read](../use/api-tour.md) is the surface you call; this page is the surface you implement.
The two meet at one seam: a config string names a cell in the dispatch table ([the system map](system-map.md)).
From there, every call is a static, inlinable dispatch against the types above.
