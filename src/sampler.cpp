#include "bonsai/sampler.hpp"

#include "bonsai/config/config.hpp"
#include "bonsai/config/errors.hpp"
#include "bonsai/types.hpp"

#include <cstddef>
#include <numeric>
#include <random>
#include <string>

namespace bonsai
{

AllRowsSampler::AllRowsSampler(Config const & /*cfg*/) {}

size_t AllRowsSampler::sample(floats_view /*grad*/, floats_view /*hess*/,
                              std::mt19937 & /*rng*/, row_index_out out_indices)
{
    std::iota(out_indices.begin(), out_indices.end(), row_index_out::value_type{0});
    return out_indices.size();
}

BernoulliSampler::BernoulliSampler(Config const &cfg) : p_(cfg.sampler.subsample)
{
    if (!(p_ > 0.0F))
    {
        throw ConfigError("sampler.subsample must be > 0 (got " + std::to_string(p_) +
                          ")");
    }
}

size_t BernoulliSampler::sample(floats_view /*grad*/, floats_view /*hess*/,
                                std::mt19937 &rng, row_index_out out_indices) const
{
    if (p_ >= 1.0F)
    {
        std::iota(out_indices.begin(), out_indices.end(), row_index_out::value_type{0});
        return out_indices.size();
    }

    std::bernoulli_distribution keep(p_);
    size_t                      n_selected = 0;
    for (size_t i = 0; i < out_indices.size(); ++i)
    {
        if (keep(rng))
        {
            out_indices[n_selected++] = static_cast<row_index_out::value_type>(i);
        }
    }
    return n_selected;
}

} // namespace bonsai
