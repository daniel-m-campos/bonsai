#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>

#include "bonsai/config/tree_config.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"

namespace bonsai::test
{

// The obvious cut: three bins, opposing gradients at bins 0 and 1, the missing
// bin empty. One valid cut (bin 0), whose gain under lambda_l2 = 1 is
//   score(-1, 1, 1) + score(+1, 1, 1) - score(0, 2, 1) = 0.5 + 0.5 - 0 = 1.0.
inline Histogram obvious_cut_histogram()
{
    Histogram h{3};
    h.add(0, -1.0, 1.0);
    h.add(1, +1.0, 1.0);
    return h;
}

inline SplitInput node_of(Histogram h)
{
    SplitInput node{.hists = {}, .rows = {}};
    node.hists.push_back(std::move(h));
    return node;
}

inline SplitInput obvious_cut_node()
{
    return node_of(obvious_cut_histogram());
}

// The node totals as the finder reads them: narrowed to the cell width.
inline HistCell cell_totals(SplitInput const &node)
{
    NodeTotals const t = node.totals();
    return {.sum_grad = static_cast<float>(t.sum_grad),
            .sum_hess = static_cast<float>(t.sum_hess)};
}

// The finder's best update: strictly greater, at or above min_gain_to_split,
// so the first feature and the first bin keep a tie.
inline void offer_split(SplitOutput &best, double gain, feature_id_t fid, bin_id_t bin,
                        bool default_left, TreeConfig const &cfg)
{
    if (gain > best.gain && gain >= cfg.min_gain_to_split)
    {
        best = {.gain         = gain,
                .feature_id   = fid,
                .bin_id       = bin,
                .default_left = default_left,
                .valid        = true};
    }
}

// Histograms of one node's rows: a quarter of the draws land in the first
// three bins, a fifth go missing when asked, a quarter carry zero hessian and
// half carry a gradient a hundred times wider. One draw per node keeps the
// normal distribution's cached second value across its features.
struct HistogramDraw
{
    std::mt19937                   &rng;
    size_t                          rows;
    bool                            with_missing;
    std::normal_distribution<float> grad{0.0F, 1.0F};

    Histogram operator()(size_t n_bins)
    {
        Histogram h{n_bins};
        for (size_t r = 0; r < rows && n_bins > 1; ++r)
        {
            size_t bin = rng() % 4 == 0 ? std::min<size_t>(rng() % 3, n_bins - 2)
                                        : rng() % (n_bins - 1);
            if (with_missing && rng() % 5 == 0)
            {
                bin = n_bins - 1;
            }
            float const hess = rng() % 4 == 0 ? 0.0F : 1.0F;
            h.add(static_cast<bin_id_t>(bin), grad(rng) * (rng() % 2 ? 1.0F : 100.0F),
                  hess);
        }
        return h;
    }
};

// A split rule drawn over the knobs the scan reads: L1 on or off, L2 on or
// off, a hessian floor on or off, a minimum gain on or off.
inline TreeConfig random_rule(std::mt19937 &rng)
{
    TreeConfig cfg{.min_data_in_leaf = 0};
    cfg.lambda_l1         = rng() % 2 ? 0.0F : 0.7F;
    cfg.lambda_l2         = rng() % 3 ? 1.0F : 0.0F;
    cfg.min_child_hess    = rng() % 2 ? 0.0F : 2.0F;
    cfg.min_gain_to_split = rng() % 3 ? 0.0F : 0.25F;
    return cfg;
}

// Equal to the bit, gain included.
inline bool same_split(SplitOutput const &a, SplitOutput const &b)
{
    return std::bit_cast<uint64_t>(a.gain) == std::bit_cast<uint64_t>(b.gain) &&
           a.feature_id == b.feature_id && a.bin_id == b.bin_id &&
           a.default_left == b.default_left && a.valid == b.valid;
}

} // namespace bonsai::test
