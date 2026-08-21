// CUDA-less stand-in for the device TreeSHAP plan (BONSAI_CUDA=OFF builds):
// the plan declines and every caller runs the host bin walk, as it does on a
// CUDA build with no device.

#include "bonsai/bin_mappers.hpp"
#include "bonsai/cuda/shap.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/tree.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace bonsai
{

std::shared_ptr<CudaShapPlan const> cuda_shap_plan(std::span<DenseTree const> /*trees*/,
                                                   BinMappers const & /*mappers*/,
                                                   float /*learning_rate*/,
                                                   float /*init_score*/)
{
    return nullptr;
}

bool cuda_pred_contribs(CudaShapPlan const & /*plan*/, IngestPlane const & /*plane*/,
                        size_t /*n_rows*/, size_t /*n_features*/,
                        std::span<double> /*out*/)
{
    return false;
}

} // namespace bonsai
