#pragma once

#include "bonsai/config/tree_config.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace bonsai
{

// A data-parallel HistogramEngine over N CudaDeviceContexts, one per entry of
// cuda_selected_devices() (empty means the current device only). It satisfies
// the same HistogramEngine + GPULevelEngine concepts as CudaHistogramEngine
// and is selected by a distinct grower name, so the single-GPU path is
// untouched (docs/architecture/19-multi-gpu.md). N == 1 is total passthrough
// to context 0: single-device behavior is identical to CudaHistogramEngine by
// construction. For N > 1 the row space is sharded across contexts, each
// builds partial level histograms over its shard, the coordinator (context 0)
// reduces them (peer memcpy where peer access is real, a host-staged bounce
// buffer otherwise) and finds the split on the global histogram; row
// partitioning stays per-shard. Tolerance-match, never bit-exact, on the same
// terms as the single engine. Not thread-safe against concurrent train calls.
class MultiCudaHistogramEngine
{
  public:
    MultiCudaHistogramEngine();
    ~MultiCudaHistogramEngine();
    MultiCudaHistogramEngine(MultiCudaHistogramEngine &&) noexcept;
    MultiCudaHistogramEngine &operator=(MultiCudaHistogramEngine &&) noexcept;
    MultiCudaHistogramEngine(MultiCudaHistogramEngine const &)            = delete;
    MultiCudaHistogramEngine &operator=(MultiCudaHistogramEngine const &) = delete;

    // Reuse the single engine's op types so the GPULevelEngine concept's
    // typename T::LevelOp / PartitionOp / LeafStamp resolve identically.
    using LevelOp     = CudaHistogramEngine::LevelOp;
    using PartitionOp = CudaHistogramEngine::PartitionOp;
    using LeafStamp   = CudaHistogramEngine::LeafStamp;

    void begin_tree(Dataset const &ds, floats_view grad, floats_view hess);
    void populate(Dataset const &ds, floats_view grad, floats_view hess,
                  SplitInput &split_input, std::span<feature_id_t const> selected);

    bool begin_root(Dataset const &ds, floats_view grad, floats_view hess,
                    SplitInput &root, std::span<feature_id_t const> selected);
    void stamp_leaves(std::span<LeafStamp const> stamps);
    void partition_level(Dataset const &ds, std::span<PartitionOp const> ops,
                         std::span<uint32_t> child_counts);
    void advance_level(Dataset const &ds, std::span<LevelOp const> ops);
    void advance_layout_only();
    void finalize_rows(std::span<node_id_t> leaf_by_row);
    void finalize_tree(std::span<float const> node_values, std::span<float> values,
                       std::span<node_id_t> leaf_ids);
    void find_splits_many(Dataset const &ds, TreeConfig const &config,
                          std::span<SplitInput const> level, std::span<SplitOutput> out,
                          std::span<HistCell> child_sums);
    void find_level_split(Dataset const &ds, TreeConfig const &config,
                          std::span<SplitInput const> level, std::span<SplitOutput> out,
                          std::span<HistCell> child_sums);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

static_assert(HistogramEngine<MultiCudaHistogramEngine>);
static_assert(GPULevelEngine<MultiCudaHistogramEngine>);

} // namespace bonsai
