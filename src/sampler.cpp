#include "bonsai/sampler.hpp"

#include "bonsai/config/config.hpp"
#include "bonsai/config/errors.hpp"
#include "bonsai/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace bonsai
{

AllRowsSampler::AllRowsSampler(Config const & /*cfg*/) {}

size_t AllRowsSampler::sample(floats_out /*grad*/, floats_out /*hess*/,
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

size_t BernoulliSampler::sample(floats_out /*grad*/, floats_out /*hess*/,
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

GossSampler::GossSampler(Config const &cfg)
    : top_rate_(cfg.sampler.top_rate), other_rate_(cfg.sampler.other_rate)
{
    if (!(top_rate_ > 0.0F) || top_rate_ > 1.0F)
    {
        throw ConfigError("sampler.top_rate must be in (0, 1] (got " +
                          std::to_string(top_rate_) + ")");
    }
    if (!(other_rate_ > 0.0F) || top_rate_ + other_rate_ > 1.0F)
    {
        throw ConfigError("sampler.other_rate must be > 0 with top_rate + "
                          "other_rate <= 1 (got " +
                          std::to_string(other_rate_) + ")");
    }
}

size_t GossSampler::sample(floats_out grad, floats_out hess, std::mt19937 &rng,
                           row_index_out out_indices) const
{
    size_t const n = out_indices.size();
    auto const   top_k =
        static_cast<size_t>(std::round(top_rate_ * static_cast<double>(n)));
    auto const other_k =
        static_cast<size_t>(std::round(other_rate_ * static_cast<double>(n)));
    if (top_k == 0 || top_k >= n)
    {
        std::iota(out_indices.begin(), out_indices.end(), row_index_out::value_type{0});
        return n;
    }

    // Rank rows by |grad|: the top_k largest are kept outright.
    std::vector<row_id_t> order(n);
    std::iota(order.begin(), order.end(), row_id_t{0});
    std::nth_element(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(top_k),
                     order.end(), [&](row_id_t a, row_id_t b)
                     { return std::abs(grad[a]) > std::abs(grad[b]); });

    std::vector<char> keep(n, 0);
    for (size_t i = 0; i < top_k; ++i)
    {
        keep[order[i]] = 1;
    }

    // Uniformly sample other_k rows from the rest and amplify them so the
    // histogram grad/hess sums stay unbiased estimates of the full data.
    std::vector<row_id_t> rest(order.begin() + static_cast<std::ptrdiff_t>(top_k),
                               order.end());
    std::vector<row_id_t> picked;
    picked.reserve(other_k);
    std::sample(rest.begin(), rest.end(), std::back_inserter(picked), other_k, rng);
    float const amplify = (1.0F - top_rate_) / other_rate_;
    for (row_id_t const r : picked)
    {
        keep[r] = 1;
        grad[r] *= amplify;
        hess[r] *= amplify;
    }

    // Emit in ascending row order for downstream scan locality.
    size_t n_selected = 0;
    for (size_t i = 0; i < n; ++i)
    {
        if (keep[i] != 0)
        {
            out_indices[n_selected++] = static_cast<row_index_out::value_type>(i);
        }
    }
    return n_selected;
}

} // namespace bonsai
