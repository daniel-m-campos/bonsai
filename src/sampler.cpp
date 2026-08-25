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
                              std::mt19937 & /*rng*/, row_index_view candidates,
                              row_index_out out)
{
    std::ranges::copy(candidates, out.begin());
    return candidates.size();
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
                                std::mt19937 &rng, row_index_view candidates,
                                row_index_out out) const
{
    if (p_ >= 1.0F)
    {
        std::ranges::copy(candidates, out.begin());
        return candidates.size();
    }

    std::bernoulli_distribution keep(p_);
    size_t                      n_selected = 0;
    for (row_id_t const r : candidates)
    {
        if (keep(rng))
        {
            out[n_selected++] = r;
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
                           row_index_view candidates, row_index_out out) const
{
    size_t const n = candidates.size();
    auto const   top_k =
        static_cast<size_t>(std::round(top_rate_ * static_cast<double>(n)));
    auto const other_k =
        static_cast<size_t>(std::round(other_rate_ * static_cast<double>(n)));
    if (top_k == 0 || top_k >= n)
    {
        std::ranges::copy(candidates, out.begin());
        return n;
    }

    // Rank candidates by the |grad| of the row each one names: the top_k
    // largest are kept outright. order holds positions into the candidate
    // list; the reweighting below turns them back into row ids.
    std::vector<row_id_t> order(n);
    std::iota(order.begin(), order.end(), row_id_t{0});
    std::nth_element(
        order.begin(), order.begin() + static_cast<std::ptrdiff_t>(top_k), order.end(),
        [&](row_id_t a, row_id_t b)
        { return std::abs(grad[candidates[a]]) > std::abs(grad[candidates[b]]); });

    std::vector<char> keep(n, 0);
    for (size_t i = 0; i < top_k; ++i)
    {
        keep[order[i]] = 1;
    }

    // Uniformly sample other_k candidates from the rest and amplify them so
    // the histogram grad/hess sums stay unbiased estimates of the full data.
    std::vector<row_id_t> rest(order.begin() + static_cast<std::ptrdiff_t>(top_k),
                               order.end());
    std::vector<row_id_t> picked;
    picked.reserve(other_k);
    std::sample(rest.begin(), rest.end(), std::back_inserter(picked), other_k, rng);
    float const amplify = (1.0F - top_rate_) / other_rate_;
    for (row_id_t const position : picked)
    {
        keep[position]   = 1;
        row_id_t const r = candidates[position];
        grad[r] *= amplify;
        hess[r] *= amplify;
    }

    // Emit in candidate order, which the view keeps ascending for the
    // downstream scan.
    size_t n_selected = 0;
    for (size_t i = 0; i < n; ++i)
    {
        if (keep[i] != 0)
        {
            out[n_selected++] = candidates[i];
        }
    }
    return n_selected;
}

} // namespace bonsai
