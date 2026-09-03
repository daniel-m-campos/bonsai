
#include "bonsai/bin_mappers.hpp"
#include "bonsai/cuda/predict.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/tree.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace bonsai
{

std::shared_ptr<CudaPredictPlan const>
cuda_predict_plan(std::span<DenseTree const> /*trees*/, BinMappers const & /*mappers*/,
                  float /*learning_rate*/, float /*init_score*/)
{
    return nullptr;
}

bool cuda_predict(CudaPredictPlan const & /*plan*/, IngestPlane const & /*plane*/,
                  size_t /*n_rows*/, size_t /*n_features*/, row_index_view /*rows*/,
                  size_t /*n_trees*/, std::span<float> /*out*/)
{
    return false;
}

} // namespace bonsai
