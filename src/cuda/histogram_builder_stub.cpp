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
                                    floats_view /*hess*/, SplitInput & /*node*/,
                                    std::span<feature_id_t const> /*selected*/)
{
    throw_unavailable();
}

void CudaHistogramBuilder::populate_many(
    Dataset const & /*ds*/, floats_view /*grad*/, floats_view /*hess*/,
    split_input_refs /*nodes*/, std::span<feature_id_t const> /*selected*/)
{
    throw_unavailable();
}

} // namespace bonsai
