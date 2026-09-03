#pragma once

#include "bonsai/bin_mappers.hpp"
#include "bonsai/bin_store.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/tree.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace bonsai
{

// Device TreeSHAP: the packed leaf paths of shap_paths.hpp evaluated against a
// Dataset's resident device plane. The plan is the ensemble packed once (one
// flat element table over every root-to-leaf path, with the merged cover
// fractions already baked in); the walk reads the plane the ingest transaction
// already left on the device, so explaining a device-binned Dataset moves
// nothing but the contributions.
//
// Opaque by design: the concrete type owns device buffers and is defined in
// the CUDA translation unit, which this header never names.
class CudaShapPlan;

// Packs the ensemble's leaf paths for the device. nullptr declines and the
// caller runs the host bin walk: no CUDA build, no usable device, an empty
// ensemble, a model whose trees carry no covers, a split feature wider than
// the 8-bit packed interval, or a merged path longer than the widest kernel
// (32 elements). Trees arrive already dense and single-output, and the mappers
// must be the cuts the thresholds came from.
std::shared_ptr<CudaShapPlan const> cuda_shap_plan(std::span<DenseTree const> trees,
                                                   BinMappers const          &mappers,
                                                   float learning_rate,
                                                   float init_score);

// Evaluates every packed path for each plane row in rows (every row when
// empty) and writes one contribution vector per position (n_features + 1
// doubles, bias last) into out, one D2H. The device walk is fp32 where the host
// walk is fp64, so this is a tolerance-equal route, not a bit-equal one. false
// declines (another backend's plane, a shape off the plan's, an out not sized
// to the rows, or a failed allocation) and the caller runs the host bin walk.
bool cuda_pred_contribs(CudaShapPlan const &plan, IngestPlane const &plane,
                        size_t n_rows, size_t n_features, row_index_view rows,
                        std::span<double> out);

} // namespace bonsai
