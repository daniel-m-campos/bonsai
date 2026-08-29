// The ObliviousWalk pack must be indistinguishable from the per-tree walk
// bit for bit: predict is the serving surface and the eval baseline pins
// its output exactly, so "close" is not a passing grade here.

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <limits>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/split.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"

using namespace bonsai; // NOLINT

namespace
{

constexpr size_t k_cols = 6;

// Trees of deliberately unequal depth, thresholds inside the data range,
// default_left alternating so NaN routing exercises both directions.
std::vector<ObliviousTree> make_trees()
{
    std::mt19937                          rng(11);
    std::uniform_real_distribution<float> df(-1.0F, 1.0F);
    std::vector<ObliviousTree>            trees;
    for (size_t depth : {2, 4, 3, 4, 1})
    {
        ObliviousTree::LevelSplits splits;
        for (size_t d = 0; d < depth; ++d)
        {
            splits.push_back(
                {.feature_id   = static_cast<feature_id_t>((d + trees.size()) % k_cols),
                 .threshold    = df(rng),
                 .default_left = (d + trees.size()) % 2 == 0});
        }
        ObliviousTree::LeafTable leaves(size_t{1} << depth);
        for (auto &v : leaves)
        {
            v = df(rng);
        }
        trees.emplace_back(std::move(splits), std::move(leaves));
    }
    return trees;
}

// Random rows with NaN injected into every column somewhere.
std::vector<float> make_rows(size_t n)
{
    std::mt19937                          rng(5);
    std::uniform_real_distribution<float> df(-1.0F, 1.0F);
    std::vector<float>                    x(n * k_cols);
    for (auto &v : x)
    {
        v = df(rng);
    }
    for (size_t i = 0; i < n; i += 7)
    {
        x[(i * k_cols) + (i % k_cols)] = std::numeric_limits<float>::quiet_NaN();
    }
    return x;
}

} // namespace

TEST_CASE("ObliviousWalk: accumulate matches the per-tree walk bit for bit",
          "[ObliviousWalk][transform][nan]")
{
    auto const          trees = make_trees();
    size_t const        n     = 257;
    auto const          x     = make_rows(n);
    features_view const xv{x.data(), n, k_cols};

    std::vector<float> want(n, 0.0F);
    for (auto const &t : trees)
    {
        t.predict(xv, floats_out{want});
    }

    ObliviousWalk const walk{std::span<ObliviousTree const>{trees}};
    std::vector<float>  got(n, 0.0F);
    walk.accumulate(xv, trees.size(), floats_out{got});

    for (size_t i = 0; i < n; ++i)
    {
        REQUIRE(got[i] == want[i]);
    }
}

TEST_CASE("ObliviousWalk: a tree-count prefix stops where predict_at stops",
          "[ObliviousWalk][transform]")
{
    auto const          trees = make_trees();
    size_t const        n     = 64;
    auto const          x     = make_rows(n);
    features_view const xv{x.data(), n, k_cols};

    ObliviousWalk const walk{std::span<ObliviousTree const>{trees}};
    for (size_t k : {size_t{0}, size_t{1}, size_t{3}, trees.size()})
    {
        std::vector<float> want(n, 0.0F);
        for (size_t t = 0; t < k; ++t)
        {
            trees[t].predict(xv, floats_out{want});
        }
        std::vector<float> got(n, 0.0F);
        walk.accumulate(xv, k, floats_out{got});
        for (size_t i = 0; i < n; ++i)
        {
            REQUIRE(got[i] == want[i]);
        }
    }
}

namespace
{

// Ragged random dense trees: every depth from 1 to 6, NaN-routing flags
// mixed, 19 trees so the 8-wide lockstep sees two full groups plus a
// remainder, and a prefix can cut inside a group.
std::vector<DenseTree> make_dense_trees()
{
    std::mt19937                          rng(31);
    std::uniform_real_distribution<float> df(-1.0F, 1.0F);
    std::vector<DenseTree>                trees;
    for (size_t k = 0; k < 19; ++k)
    {
        size_t const     depth = 1 + (k % 6);
        DenseTree::Nodes nodes;
        auto             build = [&](auto &&self, size_t lvl) -> node_id_t
        {
            auto const id = static_cast<node_id_t>(nodes.size());
            if (lvl == depth || (lvl > 1 && (k + lvl) % 3 == 0))
            {
                nodes.push_back(DenseTree::leaf(df(rng)));
                return id;
            }
            nodes.push_back(DenseTree::leaf(0.0F));
            auto const l = self(self, lvl + 1);
            auto const r = self(self, lvl + 1);
            nodes[id] =
                DenseTree::internal(static_cast<feature_id_t>((k + lvl) % k_cols),
                                    df(rng), l, r, (k + lvl) % 2 == 0);
            return id;
        };
        build(build, 0);
        trees.emplace_back(std::move(nodes),
                           DenseTree::Params{.depth = depth, .n_leaves = 0});
    }
    return trees;
}

} // namespace

TEST_CASE("DenseWalk: accumulate matches the per-tree walk bit for bit",
          "[DenseWalk][transform][nan]")
{
    auto const          trees = make_dense_trees();
    size_t const        n     = 257;
    auto const          x     = make_rows(n);
    features_view const xv{x.data(), n, k_cols};

    std::vector<float> want(n, 0.0F);
    for (auto const &t : trees)
    {
        t.predict(xv, floats_out{want});
    }

    DenseWalk const    walk{std::span<DenseTree const>{trees}};
    std::vector<float> got(n, 0.0F);
    walk.accumulate(xv, trees.size(), floats_out{got});

    for (size_t i = 0; i < n; ++i)
    {
        REQUIRE(got[i] == want[i]);
    }
}

TEST_CASE("DenseWalk: a prefix cutting inside an interleave group is exact",
          "[DenseWalk][transform][edge]")
{
    auto const          trees = make_dense_trees();
    size_t const        n     = 64;
    auto const          x     = make_rows(n);
    features_view const xv{x.data(), n, k_cols};

    DenseWalk const walk{std::span<DenseTree const>{trees}};
    for (size_t k : {size_t{0}, size_t{5}, size_t{8}, size_t{13}, trees.size()})
    {
        std::vector<float> want(n, 0.0F);
        for (size_t t = 0; t < k; ++t)
        {
            trees[t].predict(xv, floats_out{want});
        }
        std::vector<float> got(n, 0.0F);
        walk.accumulate(xv, k, floats_out{got});
        for (size_t i = 0; i < n; ++i)
        {
            REQUIRE(got[i] == want[i]);
        }
    }
}

TEST_CASE("Booster: leafwise predict is unchanged by the dense walk pack",
          "[Booster][transform]")
{
    std::mt19937                          rng(29);
    std::uniform_real_distribution<float> df(-1.0F, 1.0F);
    size_t const                          n_rows = 4000;
    detail::ColumnBatch                   batch;
    batch.features.resize(k_cols);
    for (auto &col : batch.features)
    {
        col.resize(n_rows);
        for (auto &v : col)
        {
            v = df(rng);
        }
    }
    batch.labels.resize(n_rows);
    for (size_t r = 0; r < n_rows; ++r)
    {
        batch.labels[r] = batch.features[0][r] * batch.features[2][r];
    }
    batch.feature_names.assign(k_cols, "f");
    BinMappers const mappers = BinMappers::fit(batch, {});
    Dataset const    ds      = Dataset::bin(batch, mappers, {});

    Config cfg;
    cfg.tree_config.max_depth = 5;
    Booster<MSEObjective, LeafwiseGrower<CpuHistogramEngine>, AllRowsSampler> booster{
        cfg};
    for (int i = 0; i < 12; ++i)
    {
        booster.update_one_iter(ds);
    }

    size_t const        n = 300;
    auto const          x = make_rows(n);
    features_view const xv{x.data(), n, k_cols};

    std::vector<float> got(n);
    booster.predict(xv, floats_out{got});

    std::vector<float> want(n, 0.0F);
    for (auto const &t : booster.trees())
    {
        t.predict(xv, floats_out{want});
    }
    for (auto &v : want)
    {
        v = booster.init_score() + (v * cfg.booster_config.learning_rate);
    }

    for (size_t i = 0; i < n; ++i)
    {
        REQUIRE(got[i] == want[i]);
    }
}

TEST_CASE("Booster: oblivious predict is unchanged by the walk pack",
          "[Booster][transform]")
{
    std::mt19937                          rng(23);
    std::uniform_real_distribution<float> df(-1.0F, 1.0F);
    size_t const                          n_rows = 4000;
    detail::ColumnBatch                   batch;
    batch.features.resize(k_cols);
    for (auto &col : batch.features)
    {
        col.resize(n_rows);
        for (auto &v : col)
        {
            v = df(rng);
        }
    }
    batch.labels.resize(n_rows);
    for (size_t r = 0; r < n_rows; ++r)
    {
        batch.labels[r] = batch.features[0][r] - batch.features[1][r];
    }
    batch.feature_names.assign(k_cols, "f");
    BinMappers const mappers = BinMappers::fit(batch, {});
    Dataset const    ds      = Dataset::bin(batch, mappers, {});

    Config cfg;
    cfg.tree_config.max_depth = 4;
    Booster<MSEObjective,
            ObliviousGrower<CpuHistogramEngine, HistogramLevelSplitFinder>,
            AllRowsSampler>
        booster{cfg};
    for (int i = 0; i < 10; ++i)
    {
        booster.update_one_iter(ds);
    }

    size_t const        n = 300;
    auto const          x = make_rows(n);
    features_view const xv{x.data(), n, k_cols};

    std::vector<float> got(n);
    booster.predict(xv, floats_out{got});

    std::vector<float> want(n, 0.0F);
    for (auto const &t : booster.trees())
    {
        t.predict(xv, floats_out{want});
    }
    for (auto &v : want)
    {
        v = booster.init_score() + (v * cfg.booster_config.learning_rate);
    }

    for (size_t i = 0; i < n; ++i)
    {
        REQUIRE(got[i] == want[i]);
    }
}
