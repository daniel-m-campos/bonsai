#pragma once

#include "bonsai/config/config.hpp"
#include "bonsai/types.hpp"
#include <concepts>
#include <cstddef>
#include <random>

namespace bonsai
{

// grad/hess are mutable: samplers that reweight the kept rows (GOSS) scale
// them in place; the booster recomputes both from the objective every
// iteration, so the scaling never outlives the tree it was drawn for.
template <typename T>
concept Sampler = std::constructible_from<T, Config const &> &&
                  requires(T const &s, floats_out grad, floats_out hess,
                           std::mt19937 &rng, row_index_out out_indices) {
                      {
                          s.sample(grad, hess, rng, out_indices)
                      } -> std::same_as<size_t>;
                  };

struct AllRowsSampler
{
    explicit AllRowsSampler(Config const &);

    static size_t sample(floats_out grad, floats_out hess, std::mt19937 &rng,
                         row_index_out out_indices);
};

struct BernoulliSampler
{
    explicit BernoulliSampler(Config const &cfg);

    size_t sample(floats_out grad, floats_out hess, std::mt19937 &rng,
                  row_index_out out_indices) const;

  private:
    float p_;
};

// Gradient one-side sampling (LightGBM): keep the top_rate fraction of rows
// by |grad|, sample other_rate of the total uniformly from the rest, and
// amplify the sampled rest's grad/hess by (1 - top_rate) / other_rate so
// histogram sums stay unbiased.
struct GossSampler
{
    explicit GossSampler(Config const &cfg);

    size_t sample(floats_out grad, floats_out hess, std::mt19937 &rng,
                  row_index_out out_indices) const;

  private:
    float top_rate_;
    float other_rate_;
};

} // namespace bonsai
