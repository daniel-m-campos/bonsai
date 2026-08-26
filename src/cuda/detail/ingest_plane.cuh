#pragma once

#include "bonsai/bin_store.hpp"
#include "bonsai/dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "device_buffer.cuh"

namespace bonsai
{
namespace cuda_detail
{

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
    DeviceBuffer<uint32_t> n_bins;
    bool                   bins_are_u8 = false;
    size_t                 n_rows      = 0;
    size_t                 n_feats     = 0;
    uint32_t               tile_w      = k_bin_tile_width;

    void materialize(BinColumns &cols) const override;

    std::shared_ptr<IngestPlane const>
    select_columns(std::span<feature_id_t const> keep,
                   std::span<row_id_t const>     rows) const override;
};

} // namespace cuda_detail
} // namespace bonsai
