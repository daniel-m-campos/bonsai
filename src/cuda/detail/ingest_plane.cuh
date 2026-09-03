#pragma once

#include "bonsai/bin_store.hpp"
#include "bonsai/dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
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

    template <typename Fn> void with_bins(Fn &&fn) const
    {
        if (bins_are_u8)
        {
            fn(bins8.data());
        }
        else
        {
            fn(bins16.data());
        }
    }

    void materialize(BinColumns &cols) const override;

    std::shared_ptr<IngestPlane const>
    select_columns(std::span<feature_id_t const> keep,
                   std::span<row_id_t const>     rows) const override;
};

inline CudaIngestPlane const *matching_plane(IngestPlane const &plane, size_t n_rows,
                                             size_t n_features, size_t plan_feats)
{
    if (plane.backend_tag() != cuda_backend_tag() || n_rows == 0 ||
        n_rows > std::numeric_limits<uint32_t>::max())
    {
        return nullptr;
    }
    auto const &cp      = static_cast<CudaIngestPlane const &>(plane);
    bool const  matches = cp.n_rows == n_rows && cp.n_feats == n_features &&
                         n_features == plan_feats && cp.tile_w == k_bin_tile_width;
    return matches ? &cp : nullptr;
}

} // namespace cuda_detail
} // namespace bonsai
