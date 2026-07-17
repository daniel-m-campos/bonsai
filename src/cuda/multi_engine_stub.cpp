// CUDA-less stand-in for MultiCudaHistogramEngine (BONSAI_CUDA=OFF builds):
// construction succeeds so load/predict work, training throws.

#include "bonsai/cuda/multi_engine.hpp"
#include <memory>
#include <span>
#include <stdexcept>

namespace bonsai
{

namespace
{

[[noreturn]] void throw_unavailable()
{
    throw std::runtime_error(
        "cuda_multi_depthwise requires a build with -DBONSAI_CUDA=ON (make "
        "build-cuda); this binary was built without the CUDA backend");
}

} // namespace

struct MultiCudaHistogramEngine::Impl
{
};

MultiCudaHistogramEngine::MultiCudaHistogramEngine() : impl_(std::make_unique<Impl>())
{
}
MultiCudaHistogramEngine::~MultiCudaHistogramEngine() = default;
MultiCudaHistogramEngine::MultiCudaHistogramEngine(
    MultiCudaHistogramEngine &&) noexcept = default;
MultiCudaHistogramEngine &
MultiCudaHistogramEngine::operator=(MultiCudaHistogramEngine &&) noexcept = default;

void MultiCudaHistogramEngine::begin_tree(Dataset const & /*ds*/, floats_view /*grad*/,
                                          floats_view /*hess*/)
{
    throw_unavailable();
}

void MultiCudaHistogramEngine::populate(Dataset const & /*ds*/, floats_view /*grad*/,
                                        floats_view /*hess*/,
                                        SplitInput & /*split_input*/,
                                        std::span<feature_id_t const> /*selected*/)
{
    throw_unavailable();
}

bool MultiCudaHistogramEngine::begin_root(Dataset const & /*ds*/, floats_view /*grad*/,
                                          floats_view /*hess*/, SplitInput & /*root*/,
                                          std::span<feature_id_t const> /*selected*/)
{
    throw_unavailable();
}

void MultiCudaHistogramEngine::stamp_leaves(std::span<LeafStamp const> /*stamps*/)
{
    throw_unavailable();
}

void MultiCudaHistogramEngine::partition_level(Dataset const & /*ds*/,
                                               std::span<PartitionOp const> /*ops*/,
                                               std::span<uint32_t> /*child_counts*/)
{
    throw_unavailable();
}

void MultiCudaHistogramEngine::advance_level(Dataset const & /*ds*/,
                                             std::span<LevelOp const> /*ops*/)
{
    throw_unavailable();
}

void MultiCudaHistogramEngine::advance_layout_only() {}

void MultiCudaHistogramEngine::finalize_rows(std::span<node_id_t> /*leaf_by_row*/)
{
    throw_unavailable();
}

void MultiCudaHistogramEngine::finalize_tree(std::span<float const> /*node_values*/,
                                             std::span<float> /*values*/,
                                             std::span<node_id_t> /*leaf_ids*/)
{
    throw_unavailable();
}

void MultiCudaHistogramEngine::find_splits_many(Dataset const & /*ds*/,
                                                TreeConfig const & /*config*/,
                                                std::span<SplitInput const> /*level*/,
                                                std::span<SplitOutput> /*out*/,
                                                std::span<HistCell> /*child_sums*/)
{
    throw_unavailable();
}

void MultiCudaHistogramEngine::find_level_split(Dataset const & /*ds*/,
                                                TreeConfig const & /*config*/,
                                                std::span<SplitInput const> /*level*/,
                                                std::span<SplitOutput> /*out*/,
                                                std::span<HistCell> /*child_sums*/)
{
    throw_unavailable();
}

} // namespace bonsai
