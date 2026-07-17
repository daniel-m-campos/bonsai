#pragma once

// The CUDA ingest plane: the feature-major binned matrix a device-side ingest
// produces, plus the backend tag that proves its concrete type. Shared by the
// CUDA translation units: everything here has external linkage in namespace
// bonsai::cuda_detail and references no internal-linkage entity, so including
// it from more than one TU is ODR-clean.

#include "bonsai/dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "device_buffer.cuh"

namespace bonsai
{
namespace cuda_detail
{

// The CUDA ingest transaction's product: the feature-major binned matrix,
// resident from birth. Dataset carries it as an opaque receipt; ensure_dataset
// adopts it instead of uploading host columns; materialize() pulls host columns
// home once for the host consumers (fallback decline, route_unsampled under row
// sampling). The problem this solves: ensure_dataset receives a
// shared_ptr<IngestPlane> (the base type) and must prove it is really a
// CudaIngestPlane before downcasting, without RTTI. Every plane carries an
// opaque tag pointer, and this function is the only source of this backend's
// tag (the address of a function-local static, unique process-wide), so tag
// equality proves the concrete type and makes the static_cast sound. The
// inline function's local static is guaranteed to be one object across every
// translation unit that calls it, so the identity holds process-wide even when
// two TUs link against this header.
inline void const *cuda_backend_tag()
{
    static char const anchor = 0;
    return &anchor;
}

class CudaIngestPlane final : public IngestPlane
{
  public:
    CudaIngestPlane() : IngestPlane(cuda_backend_tag()) {}

    DeviceBuffer<uint8_t>  bins8;
    DeviceBuffer<uint16_t> bins16;
    DeviceBuffer<uint32_t> n_bins; // per-feature bin counts
    bool                   bins_are_u8 = false;
    size_t                 n_rows      = 0;
    size_t                 n_feats     = 0;

    void materialize(std::vector<std::vector<uint8_t>>  &u8,
                     std::vector<std::vector<uint16_t>> &u16) const override;
};

} // namespace cuda_detail
} // namespace bonsai
