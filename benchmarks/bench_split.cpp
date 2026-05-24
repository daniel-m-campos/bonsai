#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "bonsai/config/tree_config.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"

using namespace bonsai; // NOLINT

namespace
{

std::vector<Histogram> make_workload(size_t n_features, size_t n_bins)
{
    std::vector<Histogram> hists;
    hists.reserve(n_features);
    for (size_t f = 0; f < n_features; ++f)
    {
        Histogram h{n_bins};
        for (size_t b = 0; b < n_bins; ++b)
        {
            // Deterministic but non-trivial; ensures the splitter finds
            // a real winner per feature instead of running the early-out
            // path everywhere.
            double const g =
                std::sin(static_cast<double>(f) + (0.13 * static_cast<double>(b)));
            double const hess = 1.0 + (0.01 * static_cast<double>(b));
            h.add(static_cast<bin_id_t>(b), g, hess);
        }
        hists.push_back(std::move(h));
    }
    return hists;
}

TreeConfig const k_cfg{
    .min_child_hess = 1.0F, .lambda_l2 = 1.0F, .min_data_in_leaf = 0};

// Totals (grad, hess) are now maintained inside Histogram by add(),
// so just hand the histograms over.
SplitInput make_node(std::vector<Histogram> hists)
{
    return SplitInput{.hists = std::move(hists), .rows = {}};
}

std::vector<SplitInput> make_frontier(size_t n_parents, size_t n_features,
                                      size_t n_bins)
{
    std::vector<SplitInput> frontier;
    frontier.reserve(n_parents);
    for (size_t p = 0; p < n_parents; ++p)
    {
        // Vary the workload per parent so different parents prefer different
        // cuts and the summed-argmax path is exercised.
        auto hists = make_workload(n_features, n_bins);
        for (size_t f = 0; f < n_features; ++f)
        {
            Histogram h{n_bins};
            for (size_t b = 0; b < n_bins; ++b)
            {
                double const g = std::sin(static_cast<double>((p * n_features) + f) +
                                          (0.13 * static_cast<double>(b)));
                double const hess = 1.0 + (0.01 * static_cast<double>(b));
                h.add(static_cast<bin_id_t>(b), g, hess);
            }
            hists[f] = std::move(h);
        }
        frontier.push_back(SplitInput{.hists = std::move(hists), .rows = {}});
    }
    return frontier;
}

} // namespace

TEST_CASE("HistogramSplitFinder: bench small (8 features x 64 bins)",
          "[bench][split][small]")
{
    auto const node = make_node(make_workload(8, 64));

    BENCHMARK("find: 8x64")
    {
        return HistogramNodeSplitFinder::find(node, k_cfg);
    };
}

TEST_CASE("HistogramSplitFinder: bench medium (64 features x 128 bins)",
          "[bench][split][medium]")
{
    auto const node = make_node(make_workload(64, 128));

    BENCHMARK("find: 64x128")
    {
        return HistogramNodeSplitFinder::find(node, k_cfg);
    };
}

TEST_CASE("HistogramSplitFinder: bench large (256 features x 256 bins)",
          "[bench][split][large]")
{
    auto const node = make_node(make_workload(256, 256));

    BENCHMARK("find: 256x256")
    {
        return HistogramNodeSplitFinder::find(node, k_cfg);
    };
}

TEST_CASE("HistogramSplitFinder: bench xlarge (1024 features x 512 bins)",
          "[bench][split][xlarge]")
{
    auto const node = make_node(make_workload(1024, 512));

    BENCHMARK("find: 1024x512")
    {
        return HistogramNodeSplitFinder::find(node, k_cfg);
    };
}

TEST_CASE(
    "HistogramLevelSplitFinder: bench shallow level (4 parents x 8 features x 64 bins)",
    "[bench][split][level][shallow]")
{
    auto const frontier = make_frontier(4, 8, 64);

    BENCHMARK("level find: 4p x 8f x 64b")
    {
        return HistogramLevelSplitFinder::find(frontier, k_cfg);
    };
}

TEST_CASE(
    "HistogramLevelSplitFinder: bench mid level (16 parents x 64 features x 128 bins)",
    "[bench][split][level][mid]")
{
    auto const frontier = make_frontier(16, 64, 128);

    BENCHMARK("level find: 16p x 64f x 128b")
    {
        return HistogramLevelSplitFinder::find(frontier, k_cfg);
    };
}

TEST_CASE(
    "HistogramLevelSplitFinder: bench deep level (64 parents x 256 features x 256 bins)",
    "[bench][split][level][deep]")
{
    auto const frontier = make_frontier(64, 256, 256);

    BENCHMARK("level find: 64p x 256f x 256b")
    {
        return HistogramLevelSplitFinder::find(frontier, k_cfg);
    };
}
