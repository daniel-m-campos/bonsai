#pragma once

#include "bonsai/bin_mappers.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/tree.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace bonsai
{

// Device predict: a whole-ensemble bin-space walk over a Dataset's resident
// device plane. The plan is the ensemble packed once (SoA node tables in bin
// space under the mappers' cuts); the walk reads the plane the ingest
// transaction already left on the device, so a prediction over a
// device-binned Dataset moves nothing but the scores.
//
// Opaque by design: the concrete type owns device buffers and is defined in
// the CUDA translation unit, which this header never names.
class CudaPredictPlan;

// Packs the ensemble for the device. nullptr declines and the caller runs the
// host bin walk: no CUDA build, no usable device, an empty ensemble, or a
// feature whose bins exceed what the plane's bin ids can express. Trees arrive
// already dense (an oblivious model owns its own densify), and the mappers must
// be the cuts the thresholds came from.
std::shared_ptr<CudaPredictPlan const>
cuda_predict_plan(std::span<DenseTree const> trees, BinMappers const &mappers,
                  float learning_rate, float init_score);

// Walks every tree (or the first n_trees when nonzero) for every row of the
// plane and writes init + lr * sum per row, one D2H into out. false declines at
// run time — the plane belongs to another backend, its shape disagrees with the
// plan's, or a device allocation failed — and the caller runs the host bin walk.
bool cuda_predict(CudaPredictPlan const &plan, IngestPlane const &plane, size_t n_rows,
                  size_t n_features, size_t n_trees, std::span<float> out);

} // namespace bonsai
