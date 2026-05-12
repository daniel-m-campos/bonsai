#pragma once

#include <cstdint>
#include <span>

namespace bonsai
{

using bin_id_t     = uint16_t;
using feature_id_t = uint32_t;
using row_id_t     = uint32_t;
using node_id_t    = uint32_t;

using floats_view    = std::span<float const>;
using floats_out     = std::span<float>;
using row_index_view = std::span<row_id_t const>;

} // namespace bonsai
