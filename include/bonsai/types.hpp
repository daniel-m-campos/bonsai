#pragma once

#include <cstddef>
#include <cstdint>
#include <mdspan>
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
using row_index_out  = std::span<row_id_t>;

using features_view = std::mdspan<float const, std::dextents<size_t, 2>>;

} // namespace bonsai
