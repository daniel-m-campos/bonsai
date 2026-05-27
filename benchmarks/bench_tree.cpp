#include <algorithm>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/config/tree_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"

using namespace bonsai; // NOLINT

namespace
{

struct Workload
{
    DenseTree          tree;
    std::vector<float> raw_features; // row-major, n_rows × n_features
    size_t             n_rows;
    size_t             n_features;

    features_view view() const
    {
        return features_view{raw_features.data(), n_rows, n_features};
    }
};

// Build a tree by training a DepthwiseGrower on synthetic random data so
// splits are real (not degenerate single-leaf). Returns the tree + the
// raw row-major feature matrix to predict against.
Workload make_workload(size_t n_rows, size_t n_features, uint8_t depth, uint32_t seed)
{
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> xdist(0.0F, 1.0F);

    detail::ColumnBatch batch;
    batch.features.resize(n_features);
    batch.labels.resize(n_rows);
    batch.feature_names.reserve(n_features);
    for (size_t f = 0; f < n_features; ++f)
    {
        batch.features[f].resize(n_rows);
        for (size_t r = 0; r < n_rows; ++r)
        {
            batch.features[f][r] = xdist(rng);
        }
        batch.feature_names.push_back("f" + std::to_string(f));
    }
    for (size_t r = 0; r < n_rows; ++r)
    {
        float y = 0.0F;
        for (size_t f = 0; f < n_features; ++f)
        {
            y += std::sin(batch.features[f][r] * static_cast<float>(f + 1));
        }
        batch.labels[r] = y;
    }

    BinMappers mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset    ds      = Dataset::bin(batch, mappers, {});

    std::vector<float> grad(n_rows);
    std::vector<float> hess(n_rows, 1.0F);
    float const        mean_y =
        std::accumulate(batch.labels.begin(), batch.labels.end(), 0.0F) /
        static_cast<float>(n_rows);
    for (size_t r = 0; r < n_rows; ++r)
    {
        grad[r] = mean_y - batch.labels[r];
    }

    std::vector<row_id_t> rows(n_rows);
    std::iota(rows.begin(), rows.end(), row_id_t{0});

    TreeConfig        cfg{.min_child_hess    = 0.0F,
                          .min_gain_to_split = 0.0F,
                          .lambda_l2         = 1.0F,
                          .max_depth         = depth,
                          .min_data_in_leaf  = 1};
    DepthwiseGrower<> grower{cfg};
    auto [tree, _] = grower.grow(ds, grad, hess, rows);

    std::vector<float> raw(n_rows * n_features);
    for (size_t r = 0; r < n_rows; ++r)
    {
        for (size_t f = 0; f < n_features; ++f)
        {
            raw[(r * n_features) + f] = batch.features[f][r];
        }
    }

    return Workload{
        .tree         = std::move(tree),
        .raw_features = std::move(raw),
        .n_rows       = n_rows,
        .n_features   = n_features,
    };
}

} // namespace

TEST_CASE("DenseTree::predict: small (1k rows x 4 features, depth 4)",
          "[bench][tree][dense]")
{
    auto               wl = make_workload(1'000, 4, 4, 42);
    std::vector<float> out(wl.n_rows);
    BENCHMARK("dense predict: 1k x 4f x d4")
    {
        std::fill(out.begin(), out.end(), 0.0F);
        wl.tree.predict(wl.view(), out);
        return out[0];
    };
}

TEST_CASE("DenseTree::predict: medium (10k rows x 8 features, depth 6)",
          "[bench][tree][dense]")
{
    auto               wl = make_workload(10'000, 8, 6, 42);
    std::vector<float> out(wl.n_rows);
    BENCHMARK("dense predict: 10k x 8f x d6")
    {
        std::fill(out.begin(), out.end(), 0.0F);
        wl.tree.predict(wl.view(), out);
        return out[0];
    };
}

TEST_CASE("DenseTree::predict: large (100k rows x 16 features, depth 8)",
          "[bench][tree][dense]")
{
    auto               wl = make_workload(100'000, 16, 8, 42);
    std::vector<float> out(wl.n_rows);
    BENCHMARK("dense predict: 100k x 16f x d8")
    {
        std::fill(out.begin(), out.end(), 0.0F);
        wl.tree.predict(wl.view(), out);
        return out[0];
    };
}

TEST_CASE("DenseTree::predict: deep (10k rows x 8 features, depth 10)",
          "[bench][tree][dense]")
{
    auto               wl = make_workload(10'000, 8, 10, 42);
    std::vector<float> out(wl.n_rows);
    BENCHMARK("dense predict: 10k x 8f x d10")
    {
        std::fill(out.begin(), out.end(), 0.0F);
        wl.tree.predict(wl.view(), out);
        return out[0];
    };
}
