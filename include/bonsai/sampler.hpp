#pragma once

#include "bonsai/config/config.hpp"
#include "bonsai/types.hpp"
#include <concepts>
#include <cstddef>
#include <random>

namespace bonsai
{

template <typename T>
concept Sampler = std::constructible_from<T, Config const &> &&
                  requires(T const &s, floats_view grad, floats_view hess,
                           std::mt19937 &rng, row_index_out out_indices) {
                      {
                          s.sample(grad, hess, rng, out_indices)
                      } -> std::same_as<size_t>;
                  };

struct AllRowsSampler
{
    explicit AllRowsSampler(Config const &);

    static size_t sample(floats_view grad, floats_view hess, std::mt19937 &rng,
                         row_index_out out_indices);
};

struct BernoulliSampler
{
    explicit BernoulliSampler(Config const &cfg);

    size_t sample(floats_view grad, floats_view hess, std::mt19937 &rng,
                  row_index_out out_indices) const;

  private:
    float p_;
};

} // namespace bonsai
