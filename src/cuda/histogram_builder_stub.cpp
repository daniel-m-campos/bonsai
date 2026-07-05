// CUDA-less stand-in for CudaHistogramBuilder (BONSAI_CUDA=OFF builds):
// construction succeeds so load/predict work, training throws.

#include "bonsai/cuda/histogram_builder.hpp"
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

struct CudaHistogramBuilder::Impl
{
};

CudaHistogramBuilder::CudaHistogramBuilder() : impl_(std::make_unique<Impl>()) {}
CudaHistogramBuilder::~CudaHistogramBuilder()                                = default;
CudaHistogramBuilder::CudaHistogramBuilder(CudaHistogramBuilder &&) noexcept = default;
CudaHistogramBuilder &
CudaHistogramBuilder::operator=(CudaHistogramBuilder &&) noexcept = default;

void CudaHistogramBuilder::begin_tree(Dataset const & /*ds*/, floats_view /*grad*/,
                                      floats_view /*hess*/)
{
    throw_unavailable();
}

void CudaHistogramBuilder::populate(Dataset const & /*ds*/, floats_view /*grad*/,
                                    floats_view /*hess*/, SplitInput & /*split_input*/,
                                    std::span<feature_id_t const> /*selected*/)
{
    throw_unavailable();
}

void CudaHistogramBuilder::populate_many(Dataset const & /*ds*/, floats_view /*grad*/,
                                         floats_view /*hess*/,
                                         split_input_refs /*nodes*/,
                                         std::span<feature_id_t const> /*selected*/)
{
    throw_unavailable();
}

bool CudaHistogramBuilder::begin_root(Dataset const & /*ds*/, floats_view /*grad*/,
                                      floats_view /*hess*/, SplitInput & /*root*/,
                                      std::span<feature_id_t const> /*selected*/)
{
    throw_unavailable();
}

bool CudaHistogramBuilder::resident() const
{
    return false;
}

void CudaHistogramBuilder::stamp_leaves(std::span<LeafStamp const> /*stamps*/)
{
    throw_unavailable();
}

void CudaHistogramBuilder::partition_level(Dataset const & /*ds*/,
                                           std::span<PartitionOp const> /*ops*/,
                                           std::span<uint32_t> /*child_counts*/)
{
    throw_unavailable();
}

void CudaHistogramBuilder::advance_level(Dataset const & /*ds*/,
                                         std::span<LevelOp const> /*ops*/)
{
    throw_unavailable();
}

void CudaHistogramBuilder::finalize_rows(std::span<node_id_t> /*leaf_by_row*/)
{
    throw_unavailable();
}

void CudaHistogramBuilder::find_splits_many(Dataset const & /*ds*/,
                                            TreeConfig const & /*config*/,
                                            std::span<SplitInput const> /*level*/,
                                            std::span<SplitOutput> /*out*/,
                                            std::span<HistCell> /*child_sums*/)
{
    throw_unavailable();
}

} // namespace bonsai
