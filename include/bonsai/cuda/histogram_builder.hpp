#pragma once

#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include <memory>
#include <span>

namespace bonsai
{

// True when this build carries the CUDA backend AND a usable device is
// present. cuda_depthwise is registered in every build; only training
// needs this to be true. See docs/architecture/10-cuda.md.
bool cuda_available();

// HistogramBuilder that offloads histogram construction to the GPU
// (src/cuda/histogram_builder.cu; a throwing stub backs it when built
// without BONSAI_CUDA). GPU cells match the CPU builder to tolerance, not
// bit-exactly: atomics accumulate in arbitrary order. Design and precision
// scheme: docs/architecture/10-cuda.md.
class CudaHistogramBuilder
{
  public:
    CudaHistogramBuilder();
    ~CudaHistogramBuilder();
    CudaHistogramBuilder(CudaHistogramBuilder &&) noexcept;
    CudaHistogramBuilder &operator=(CudaHistogramBuilder &&) noexcept;
    CudaHistogramBuilder(CudaHistogramBuilder const &)            = delete;
    CudaHistogramBuilder &operator=(CudaHistogramBuilder const &) = delete;

    void begin_tree(Dataset const &ds, floats_view grad, floats_view hess);
    void populate(Dataset const &ds, floats_view grad, floats_view hess,
                  SplitInput &node, std::span<feature_id_t const> selected);
    // Batched variant: one kernel launch covers a whole tree level.
    void populate_many(Dataset const &ds, floats_view grad, floats_view hess,
                       split_input_refs nodes, std::span<feature_id_t const> selected);

    // --- Resident level backend (phase 3, docs/architecture/11-gpu-resident.md).
    // Level histograms stay on the device, keyed by the node's index in the
    // grower's frontier ("slot"); splits are found on the device and only
    // decisions and child sums cross the bus. The depthwise grower detects
    // these hooks via if constexpr + requires; the splitter policy remains
    // the host fallback when begin_root declines.

    // One child-level derivation: the smaller child's histogram builds from
    // its device row segment; the larger derives on-device as parent minus
    // smaller. Rows never cross the bus: partition_level wrote the segments.
    struct LevelOp
    {
        uint32_t parent_slot;
        uint32_t small_slot;
        uint32_t large_slot;
    };

    // One split's routing: partition the parent slot's row segment into the
    // children, stably (left rows keep ascending order, then right rows).
    struct PartitionOp
    {
        uint32_t     parent_slot;
        uint32_t     left_slot;
        uint32_t     right_slot;
        feature_id_t feature_id;
        bin_id_t     bin_id;
        bool         default_left;
    };

    struct LeafStamp
    {
        uint32_t  slot;
        node_id_t node_id;
    };

    // Starts the resident path for this tree: builds the root histogram into
    // slot 0 and fills root.sums/row_count. Returns false (leaving root
    // untouched) when the resident path cannot run — no device, or a feature's
    // bins exceed the shared-memory budget — and the caller uses populate.
    bool begin_root(Dataset const &ds, floats_view grad, floats_view hess,
                    SplitInput &root, std::span<feature_id_t const> selected);
    bool resident() const;
    // Records final leaf assignment for every row in the given slots'
    // segments (call before the level advances past them).
    void stamp_leaves(std::span<LeafStamp const> stamps);
    // Routes every split slot's rows into its children on the device;
    // child_counts receives (n_left, n_right) per op — the only partition
    // data that returns to the host.
    void partition_level(Dataset const &ds, std::span<PartitionOp const> ops,
                         std::span<uint32_t> child_counts);
    // Populates all smaller children and subtracts all larger ones, then
    // makes the child level current.
    void advance_level(Dataset const &ds, std::span<LevelOp const> ops);
    // End of tree: the per-row leaf assignment (indexed by row id; only rows
    // this tree trained on carry fresh values).
    void finalize_rows(std::span<node_id_t> leaf_by_row);
    // Best split per frontier node from the current level's device
    // histograms; child_sums receives the winning cut's (left, right) totals,
    // 2 cells per node. level[i] corresponds to slot i.
    void find_splits_many(Dataset const &ds, TreeConfig const &config,
                          std::span<SplitInput const> level, std::span<SplitOutput> out,
                          std::span<HistCell> child_sums);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

static_assert(HistogramBuilder<CudaHistogramBuilder>);

// Registered as "cuda_depthwise" (registry/typelists.hpp).
using CudaDepthwiseGrower =
    DepthwiseGrower<HistogramNodeSplitFinder, CudaHistogramBuilder>;

} // namespace bonsai
