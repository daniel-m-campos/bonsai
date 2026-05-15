#pragma once

#include "bonsai/types.hpp"
#include <concepts>
#include <cstddef>
#include <random>
#include <span>
namespace bonsai
{

template <typename T>
concept Sampler = requires(floats_view grad, floats_view hess, std::mt19937 &rng,
                           row_index_out out_indices) {
    { T::sample(grad, hess, rng, out_indices) } -> std::same_as<size_t>;
};

struct AllRowsSampler
{
    static size_t sample(floats_view, floats_view, std::mt19937 &,
                         row_index_out out_indices);
};

} // namespace bonsai
