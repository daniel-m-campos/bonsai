// CUDA-less stand-in for CudaHistogramEngine (BONSAI_CUDA=OFF builds):
// construction succeeds so load/predict work, training throws.

#include "bonsai/cuda/histogram_engine.hpp"
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
        "cuda_depthwise requires a build with -DBONSAI_CUDA=ON (make "
        "build-cuda); this binary was built without the CUDA backend");
}

} // namespace

bool cuda_available()
{
    return false;
}

struct CudaHistogramEngine::Impl
{
};

CudaHistogramEngine::CudaHistogramEngine() : impl_(std::make_unique<Impl>()) {}
CudaHistogramEngine::~CudaHistogramEngine()                               = default;
CudaHistogramEngine::CudaHistogramEngine(CudaHistogramEngine &&) noexcept = default;
CudaHistogramEngine &
CudaHistogramEngine::operator=(CudaHistogramEngine &&) noexcept = default;

void CudaHistogramEngine::begin_tree(Dataset const & /*ds*/, floats_view /*grad*/,
                                     floats_view /*hess*/)
{
    throw_unavailable();
}

void CudaHistogramEngine::populate(Dataset const & /*ds*/, floats_view /*grad*/,
                                   floats_view /*hess*/, SplitInput & /*split_input*/,
                                   std::span<feature_id_t const> /*selected*/)
{
    throw_unavailable();
}

bool CudaHistogramEngine::begin_root(Dataset const & /*ds*/, floats_view /*grad*/,
                                     floats_view /*hess*/, SplitInput & /*root*/,
                                     std::span<feature_id_t const> /*selected*/)
{
    throw_unavailable();
}

void CudaHistogramEngine::stamp_leaves(std::span<LeafStamp const> /*stamps*/)
{
    throw_unavailable();
}

void CudaHistogramEngine::partition_level(Dataset const & /*ds*/,
                                          std::span<PartitionOp const> /*ops*/,
                                          std::span<uint32_t> /*child_counts*/)
{
    throw_unavailable();
}

void CudaHistogramEngine::advance_level(Dataset const & /*ds*/,
                                        std::span<LevelOp const> /*ops*/)
{
    throw_unavailable();
}

void CudaHistogramEngine::finalize_rows(std::span<node_id_t> /*leaf_by_row*/)
{
    throw_unavailable();
}

void CudaHistogramEngine::find_splits_many(Dataset const & /*ds*/,
                                           TreeConfig const & /*config*/,
                                           std::span<SplitInput const> /*level*/,
                                           std::span<SplitOutput> /*out*/,
                                           std::span<HistCell> /*child_sums*/)
{
    throw_unavailable();
}

void CudaHistogramEngine::find_level_split(Dataset const & /*ds*/,
                                           TreeConfig const & /*config*/,
                                           std::span<SplitInput const> /*level*/,
                                           std::span<SplitOutput> /*out*/,
                                           std::span<HistCell> /*child_sums*/)
{
    throw_unavailable();
}

} // namespace bonsai
