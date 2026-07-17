// CUDA-less stand-in for CudaHistogramEngine (BONSAI_CUDA=OFF builds):
// construction succeeds so load/predict work, training throws.

#include "bonsai/config/errors.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

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

void cuda_select_device(uint32_t device_id)
{
    // 0 is the ambient default everywhere; a nonzero id in a CUDA-less
    // build is a misconfiguration and must be loud (decision 60's rule).
    if (device_id != 0)
    {
        throw ConfigError("parallel.device_id " + std::to_string(device_id) +
                          " requires a CUDA build (-DBONSAI_CUDA=ON); this "
                          "binary was built without the CUDA backend");
    }
}

void cuda_select_devices(std::span<uint32_t const> ids)
{
    // Empty or {0} is the ambient default (a no-op); any other id in a
    // CUDA-less build is a misconfiguration and must be loud. Nothing is
    // stored: without a backend the selection can never be consumed, so the
    // stub stays stateless.
    for (uint32_t const id : ids)
    {
        if (id != 0)
        {
            throw ConfigError("parallel.device_ids entry " + std::to_string(id) +
                              " requires a CUDA build (-DBONSAI_CUDA=ON); this "
                              "binary was built without the CUDA backend");
        }
    }
}

std::vector<uint32_t> cuda_selected_devices()
{
    return {};
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

void CudaHistogramEngine::advance_layout_only() {}

void CudaHistogramEngine::finalize_rows(std::span<node_id_t> /*leaf_by_row*/)
{
    throw_unavailable();
}

void CudaHistogramEngine::finalize_tree(std::span<float const> /*node_values*/,
                                        std::span<float> /*values*/,
                                        std::span<node_id_t> /*leaf_ids*/)
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

std::shared_ptr<IngestPlane const> cuda_ingest(detail::ColumnBatch const & /*batch*/,
                                               BinMappers const & /*mappers*/)
{
    return nullptr;
}

std::shared_ptr<IngestPlane const> cuda_ingest(features_view /*X*/,
                                               BinMappers const & /*mappers*/)
{
    return nullptr;
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
