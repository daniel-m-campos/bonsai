#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

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

HistogramSplitFinder::Params const k_params{
    .parent_score = 0.0, .lambda_l2 = 1.0, .min_child_hess = 1.0};

} // namespace

TEST_CASE("HistogramSplitFinder: bench small (8 features x 64 bins)",
          "[bench][split][small]")
{
    auto const hists = make_workload(8, 64);
    auto const view  = histogram_view_t{hists};

    BENCHMARK("find: 8x64")
    {
        return HistogramSplitFinder::find(view, k_params);
    };
}

TEST_CASE("HistogramSplitFinder: bench medium (64 features x 128 bins)",
          "[bench][split][medium]")
{
    auto const hists = make_workload(64, 128);
    auto const view  = histogram_view_t{hists};

    BENCHMARK("find: 64x128")
    {
        return HistogramSplitFinder::find(view, k_params);
    };
}

TEST_CASE("HistogramSplitFinder: bench large (256 features x 256 bins)",
          "[bench][split][large]")
{
    auto const hists = make_workload(256, 256);
    auto const view  = histogram_view_t{hists};

    BENCHMARK("find: 256x256")
    {
        return HistogramSplitFinder::find(view, k_params);
    };
}

TEST_CASE("HistogramSplitFinder: bench xlarge (1024 features x 512 bins)",
          "[bench][split][xlarge]")
{
    auto const hists = make_workload(1024, 512);
    auto const view  = histogram_view_t{hists};

    BENCHMARK("find: 1024x512")
    {
        return HistogramSplitFinder::find(view, k_params);
    };
}
