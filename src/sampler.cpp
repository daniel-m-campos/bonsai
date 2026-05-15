#include "bonsai/sampler.hpp"
#include "bonsai/types.hpp"
#include <cstddef>
#include <numeric>
#include <random>

namespace bonsai
{

size_t AllRowsSampler::sample(floats_view, floats_view, std::mt19937 &,
                              row_index_out out_indices)
{
    std::iota(out_indices.begin(), out_indices.end(), row_index_out::value_type{0});
    return out_indices.size();
}

} // namespace bonsai
