#include "bonsai/sampler.hpp"

#include "bonsai/config/config.hpp"
#include "bonsai/config/errors.hpp"
#include "bonsai/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
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

namespace
{

constexpr uint8_t k_on_top = 1U;
constexpr uint8_t k_picked = 2U;

void scale_row(floats_out grad, floats_out hess, row_id_t r, float factor)
{
    grad[r] *= factor;
    hess[r] *= factor;
}

struct KeptCopies
{
    float on_top = 0.0F;
    float picked = 0.0F;
};

void count_copies(std::unordered_map<row_id_t, KeptCopies> &copies,
                  row_index_view candidates, row_index_view positions,
                  float KeptCopies::*tally)
{
    for (row_id_t const position : positions)
    {
        if (auto const it = copies.find(candidates[position]); it != copies.end())
        {
            it->second.*tally += 1.0F;
        }
    }
}

void amplify_picked_rows(floats_out grad, floats_out hess, float amplify,
                         row_index_view candidates, row_index_view top_positions,
                         row_index_view picked_positions)
{
    std::vector<uint8_t> marks(grad.size(), 0);
    for (row_id_t const position : top_positions)
    {
        marks[candidates[position]] |= k_on_top;
    }
    std::unordered_map<row_id_t, KeptCopies> split_rows;
    for (row_id_t const position : picked_positions)
    {
        row_id_t const r = candidates[position];
        if ((marks[r] & k_picked) != 0)
        {
            continue;
        }
        marks[r] |= k_picked;
        if ((marks[r] & k_on_top) != 0)
        {
            split_rows.emplace(r, KeptCopies{});
            continue;
        }
        scale_row(grad, hess, r, amplify);
    }
    if (split_rows.empty())
    {
        return;
    }
    count_copies(split_rows, candidates, top_positions, &KeptCopies::on_top);
    count_copies(split_rows, candidates, picked_positions, &KeptCopies::picked);
    for (auto const &[r, kept] : split_rows)
    {
        scale_row(grad, hess, r,
                  (kept.on_top + (kept.picked * amplify)) /
                      (kept.on_top + kept.picked));
    }
}

} // namespace

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

    std::vector<row_id_t> rest(order.begin() + static_cast<std::ptrdiff_t>(top_k),
                               order.end());
    std::vector<row_id_t> picked;
    picked.reserve(other_k);
    std::sample(rest.begin(), rest.end(), std::back_inserter(picked), other_k, rng);
    for (row_id_t const position : picked)
    {
        keep[position] = 1;
    }
    amplify_picked_rows(grad, hess, (1.0F - top_rate_) / other_rate_, candidates,
                        {order.data(), top_k}, picked);

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
