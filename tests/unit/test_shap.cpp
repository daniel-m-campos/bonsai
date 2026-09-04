#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/tree_config.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/multiclass_booster.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/shap.hpp"
#include "bonsai/types.hpp"
#include "test_grower_helpers.hpp"

using namespace bonsai;       // NOLINT
using namespace bonsai::test; // NOLINT

namespace
{

// Exact Shapley values by subset enumeration, attributing the same
// cover-conditioned expectation TreeSHAP uses. Feasible for tiny M.
std::vector<double> brute_force_shapley(DenseTree const &tree, features_view X,
                                        row_id_t row, size_t n_features)
{
    auto factorial = [](size_t k)
    {
        double f = 1.0;
        for (size_t i = 2; i <= k; ++i)
        {
            f *= static_cast<double>(i);
        }
        return f;
    };
    std::vector<double> phi(n_features, 0.0);
    size_t const        n_subsets = size_t{1} << n_features;
    for (size_t mask = 0; mask < n_subsets; ++mask)
    {
        std::vector<unsigned char> in(n_features, 0);
        size_t                     size = 0;
        for (size_t f = 0; f < n_features; ++f)
        {
            if ((mask >> f) & 1U)
            {
                in[f] = 1;
                ++size;
            }
        }
        std::span<bool const> const s{reinterpret_cast<bool const *>(in.data()),
                                      n_features};
        double const                ev_s = tree_expected_value(tree, X, row, s);
        double const                w =
            factorial(size) * factorial(n_features - size - 1) / factorial(n_features);
        for (size_t f = 0; f < n_features; ++f)
        {
            if ((mask >> f) & 1U)
            {
                continue; // need S without f
            }
            std::vector<unsigned char> with = in;
            with[f]                         = 1;
            std::span<bool const> const sw{reinterpret_cast<bool const *>(with.data()),
                                           n_features};
            phi[f] += w * (tree_expected_value(tree, X, row, sw) - ev_s);
        }
    }
    return phi;
}

// 3 informative features, non-trivial interactions.
detail::ColumnBatch shap_batch()
{
    return detail::ColumnBatch{
        .features      = {{0.0F, 0.1F, 0.2F, 0.9F, 1.0F, 1.1F, 1.2F, 1.3F},
                          {0.0F, 1.0F, 0.1F, 1.1F, 0.2F, 1.2F, 0.3F, 1.3F},
                          {0.5F, 0.0F, 1.0F, 0.6F, 0.1F, 1.1F, 0.7F, 1.2F}},
        .labels        = std::vector<float>(8, 0.0F),
        .weights       = {},
        .feature_names = {"a", "b", "c"},
    };
}

} // namespace

TEST_CASE("TreeSHAP: matches brute-force Shapley on a small tree", "[shap][exact]")
{
    auto               built = build(shap_batch());
    std::vector<float> grad{-3.0F, +1.0F, -2.0F, +2.0F, -1.0F, +3.0F, -2.5F, +1.5F};
    std::vector<float> hess(8, 1.0F);
    auto               rows = iota_rows(8);

    TreeConfig        cfg{.min_child_hess   = 0.0F,
                          .lambda_l2        = 1.0F,
                          .max_depth        = 3,
                          .min_data_in_leaf = 0};
    DepthwiseGrower<> grower{cfg};
    auto [tree, values, lids, bounds] = grower.grow(built.ds, grad, hess, rows);
    REQUIRE(tree.params().depth >= 2); // interactions present

    std::vector<float> const x{0.15F, 0.9F, 0.4F};
    features_view const      X{x.data(), 1, 3};

    std::vector<double> phi(4, 0.0);
    tree_shap(tree, X, 0, phi);

    auto const exact = brute_force_shapley(tree, X, 0, 3);
    for (size_t f = 0; f < 3; ++f)
    {
        CHECK(phi[f] == Catch::Approx(exact[f]).margin(1e-9));
    }
    // Efficiency: bias + contributions == the tree's prediction for x.
    std::array<float, 1> pred{0.0F};
    tree.predict(X, pred);
    CHECK(phi[0] + phi[1] + phi[2] + phi[3] ==
          Catch::Approx(static_cast<double>(pred[0])).margin(1e-9));
}

TEST_CASE("Booster: pred_contribs rows sum to the raw prediction", "[shap][booster]")
{
    auto                built = build(shap_batch());
    detail::ColumnBatch batch = shap_batch();
    std::vector<float>  labels{1.0F, -1.0F, 2.0F, -2.0F, 0.5F, 3.0F, -0.5F, 1.5F};

    Config cfg;
    cfg.tree_config.min_data_in_leaf = 0;
    cfg.tree_config.min_child_hess   = 0.0F;
    cfg.tree_config.max_depth        = 3;

    // Build a dataset with real labels.
    batch.labels             = labels;
    BinMappers const mappers = BinMappers::fit(batch, {});
    Dataset const    train   = Dataset::bin(batch, mappers, {});

    Booster<MSEObjective, LeafwiseGrower<>, AllRowsSampler> b{cfg};
    for (int i = 0; i < 8; ++i)
    {
        b.update_one_iter(train);
    }

    // Row-major raw feature matrix for the 8 train rows.
    std::vector<float> raw(8 * 3);
    for (size_t r = 0; r < 8; ++r)
    {
        for (size_t f = 0; f < 3; ++f)
        {
            raw[(r * 3) + f] = batch.features[f][r];
        }
    }
    features_view const X{raw.data(), 8, 3};

    std::vector<double> contribs(8 * 4);
    b.pred_contribs(X, contribs, 3);

    std::vector<float> pred(8);
    b.predict(X, pred);
    for (size_t r = 0; r < 8; ++r)
    {
        double sum = 0.0;
        for (size_t c = 0; c < 4; ++c)
        {
            sum += contribs[(r * 4) + c];
        }
        CHECK(sum == Catch::Approx(static_cast<double>(pred[r])).margin(1e-4));
    }
}

TEST_CASE("Oblivious pred_contribs: dense expansion is exact and rows sum to "
          "the raw prediction",
          "[shap][oblivious]")
{
    detail::ColumnBatch batch = shap_batch();
    batch.labels              = {1.0F, -1.0F, 2.0F, -2.0F, 0.5F, 3.0F, -0.5F, 1.5F};
    BinMappers const mappers  = BinMappers::fit(batch, {});
    Dataset const    train    = Dataset::bin(batch, mappers, {});

    Config cfg;
    cfg.tree_config.min_data_in_leaf = 0;
    cfg.tree_config.min_child_hess   = 0.0F;
    cfg.tree_config.max_depth        = 3;

    Booster<MSEObjective, ObliviousGrower<>, AllRowsSampler> b{cfg};
    for (int i = 0; i < 6; ++i)
    {
        b.update_one_iter(train);
    }

    std::vector<float> raw(8 * 3);
    for (size_t r = 0; r < 8; ++r)
    {
        for (size_t f = 0; f < 3; ++f)
        {
            raw[(r * 3) + f] = batch.features[f][r];
        }
    }
    features_view const X{raw.data(), 8, 3};

    // The expansion must be prediction-identical to the oblivious walk.
    for (auto const &tree : b.trees())
    {
        auto const         dense = dense_equivalent(tree);
        std::vector<float> a(8, 0.0F);
        std::vector<float> d(8, 0.0F);
        tree.predict(X, a);
        dense.predict(X, d);
        for (size_t r = 0; r < 8; ++r)
        {
            REQUIRE(a[r] == d[r]);
        }
    }

    // Efficiency: contributions + bias reproduce the ensemble prediction.
    std::vector<double> contribs(8 * 4);
    b.pred_contribs(X, contribs, 3);
    std::vector<float> pred(8);
    b.predict(X, pred);
    for (size_t r = 0; r < 8; ++r)
    {
        double sum = 0.0;
        for (size_t c = 0; c < 4; ++c)
        {
            sum += contribs[(r * 4) + c];
        }
        REQUIRE(sum == Catch::Approx(pred[r]).margin(1e-4));
    }

    // A cover-less tree (pre-recording model) explains itself.
    ObliviousTree const bare{ObliviousTree::LevelSplits{},
                             ObliviousTree::LeafTable{0.5F}};
    REQUIRE_THROWS_AS(dense_equivalent(bare), std::invalid_argument);
}

TEST_CASE("Multiclass pred_contribs: per-class slices vote like predict",
          "[shap][multiclass]")
{
    detail::ColumnBatch batch;
    batch.features.resize(1);
    batch.feature_names = {"x"};
    for (int rep = 0; rep < 4; ++rep)
    {
        for (int k = 0; k < 3; ++k)
        {
            batch.features[0].push_back(static_cast<float>(k) + 0.1F +
                                        (0.01F * static_cast<float>(rep)));
            batch.labels.push_back(static_cast<float>(k));
        }
    }
    BinMappers const mappers = BinMappers::fit(batch, {});
    Dataset const    train   = Dataset::bin(batch, mappers, {});

    Config cfg;
    cfg.objective.n_classes          = 3;
    cfg.tree_config.min_data_in_leaf = 0;
    cfg.tree_config.min_child_hess   = 0.0F;

    MulticlassBooster<DepthwiseGrower<>, AllRowsSampler> b{cfg};
    for (int i = 0; i < 10; ++i)
    {
        b.update_one_iter(train);
    }

    size_t const        n = batch.labels.size();
    std::vector<float>  raw(batch.features[0]);
    features_view const X{raw.data(), n, 1};

    std::vector<double> contribs(n * 3 * 2);
    b.pred_contribs(X, contribs, 1);
    std::vector<float> pred(n);
    b.predict(X, pred);
    for (size_t i = 0; i < n; ++i)
    {
        // Each class slice sums to that class's raw score; the argmax over
        // those sums must therefore agree with predict().
        size_t best      = 0;
        double best_swum = -1e30;
        for (size_t k = 0; k < 3; ++k)
        {
            double sum = 0.0;
            for (size_t c = 0; c < 2; ++c)
            {
                sum += contribs[(((i * 3) + k) * 2) + c];
            }
            if (sum > best_swum)
            {
                best_swum = sum;
                best      = k;
            }
        }
        REQUIRE(static_cast<float>(best) == pred[i]);
    }
}

TEST_CASE("Oblivious dense expansion: dead slots route identically and SHAP "
          "stays finite",
          "[shap][oblivious]")
{
    // depth 2, feature 0 then feature 1; the (left, left) slot got no
    // training rows. The expansion emits the dead slot verbatim, so every
    // x — including one walking into the dead branch — routes identically,
    // and tree_shap's zero-cover guard keeps phi finite with exact
    // efficiency.
    ObliviousTree const tree{
        ObliviousTree::LevelSplits{
            {.feature_id = 0, .threshold = 1.0F, .default_left = true},
            {.feature_id = 1, .threshold = 1.0F, .default_left = true}},
        ObliviousTree::LeafTable{0.0F, 10.0F, 20.0F, 30.0F},
        {},
        {0.0F, 4.0F, 3.0F, 5.0F}};
    auto const dense = dense_equivalent(tree);

    // Cover-positive corners agree exactly.
    std::vector<float> raw{0.5F, 2.0F, 2.0F, 0.5F, 2.0F, 2.0F};
    features_view      X{raw.data(), 3, 2};
    std::vector<float> a(3, 0.0F);
    std::vector<float> d(3, 0.0F);
    tree.predict(X, a);
    dense.predict(X, d);
    for (size_t r = 0; r < 3; ++r)
    {
        REQUIRE(a[r] == d[r]);
    }

    // The dead (left, left) corner: both walks land on the no-evidence
    // slot, and TreeSHAP reproduces its value exactly (sum(phi) == f(x)).
    std::vector<float> dead_raw{0.5F, 0.5F};
    features_view      Xd{dead_raw.data(), 1, 2};
    std::vector<float> ao(1, 0.0F);
    std::vector<float> ad(1, 0.0F);
    tree.predict(Xd, ao);
    dense.predict(Xd, ad);
    REQUIRE(ao[0] == 0.0F);
    REQUIRE(ad[0] == ao[0]);
    std::vector<double> phi(3, 0.0);
    tree_shap(dense, Xd, 0, phi);
    REQUIRE(std::isfinite(phi[0]));
    REQUIRE(std::isfinite(phi[1]));
    REQUIRE(phi[0] + phi[1] + phi[2] == Catch::Approx(0.0).margin(1e-9));
}

TEST_CASE("TreeSHAP: the ice-cream example is tree-shape invariant", "[shap][guide15]")
{
    // The worked example of guide chapter 15. Model B: revenue 90 on a sunny
    // weekend, 10 on a sunny weekday, else 0; 100 training days, 25 per
    // (sunny, weekend) quadrant. Feature 0 is sunny, feature 1 is weekend.
    // Two tree shapes realize the same function; the chapter's tables pin
    // phi = {sunny 35, weekend 30, bias 25} for the sunny-weekend day, and
    // the values must not depend on which feature the root splits on.
    DenseTree::Nodes shape_w;
    shape_w.emplace_back(
        DenseTree::internal(1, 0.5F, node_id_t{1}, node_id_t{2}, false));
    shape_w.emplace_back(
        DenseTree::internal(0, 0.5F, node_id_t{3}, node_id_t{4}, false));
    shape_w.emplace_back(
        DenseTree::internal(0, 0.5F, node_id_t{5}, node_id_t{6}, false));
    shape_w.emplace_back(DenseTree::leaf(0.0F));
    shape_w.emplace_back(DenseTree::leaf(10.0F));
    shape_w.emplace_back(DenseTree::leaf(0.0F));
    shape_w.emplace_back(DenseTree::leaf(90.0F));
    DenseTree const tree_w{std::move(shape_w),
                           DenseTree::Params{.depth = 2, .n_leaves = 4},
                           {},
                           {100.0F, 50.0F, 50.0F, 25.0F, 25.0F, 25.0F, 25.0F}};

    DenseTree::Nodes shape_s;
    shape_s.emplace_back(
        DenseTree::internal(0, 0.5F, node_id_t{1}, node_id_t{2}, false));
    shape_s.emplace_back(DenseTree::leaf(0.0F));
    shape_s.emplace_back(
        DenseTree::internal(1, 0.5F, node_id_t{3}, node_id_t{4}, false));
    shape_s.emplace_back(DenseTree::leaf(10.0F));
    shape_s.emplace_back(DenseTree::leaf(90.0F));
    DenseTree const tree_s{std::move(shape_s),
                           DenseTree::Params{.depth = 2, .n_leaves = 3},
                           {},
                           {100.0F, 50.0F, 50.0F, 25.0F, 25.0F}};

    // All four day types, one per row: the chapter's per-day table, its
    // dataset-level mean-absolute importance (sunny 25, weekend 20), and
    // the zero signed mean over the training distribution.
    std::array<float, 8> const days{0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
    features_view const        X{days.data(), 4, 2};
    std::array<std::array<double, 3>, 4> const expected{{{-15.0, -10.0, 25.0},
                                                         {15.0, -30.0, 25.0},
                                                         {-35.0, 10.0, 25.0},
                                                         {35.0, 30.0, 25.0}}};

    std::array<double, 2> mean_abs{};
    std::array<double, 2> mean_signed{};
    for (row_id_t row = 0; row < 4; ++row)
    {
        std::vector<double> phi_w(3, 0.0);
        std::vector<double> phi_s(3, 0.0);
        tree_shap(tree_w, X, row, phi_w);
        tree_shap(tree_s, X, row, phi_s);
        auto const exact = brute_force_shapley(tree_w, X, row, 2);
        for (size_t i = 0; i < 3; ++i)
        {
            CHECK(phi_w[i] == Catch::Approx(phi_s[i]).margin(1e-9));
            CHECK(phi_w[i] == Catch::Approx(expected[row][i]).margin(1e-9));
        }
        for (size_t f = 0; f < 2; ++f)
        {
            CHECK(phi_w[f] == Catch::Approx(exact[f]).margin(1e-9));
            mean_abs[f] += std::abs(phi_w[f]) / 4.0;
            mean_signed[f] += phi_w[f] / 4.0;
        }
    }
    CHECK(mean_abs[0] == Catch::Approx(25.0).margin(1e-9));
    CHECK(mean_abs[1] == Catch::Approx(20.0).margin(1e-9));
    CHECK(mean_signed[0] == Catch::Approx(0.0).margin(1e-9));
    CHECK(mean_signed[1] == Catch::Approx(0.0).margin(1e-9));
}

TEST_CASE("pred_contribs: the dense cache serves repeats and follows mutations",
          "[shap][oblivious]")
{
    detail::ColumnBatch batch = shap_batch();
    batch.labels              = {1.0F, -1.0F, 2.0F, -2.0F, 0.5F, 3.0F, -0.5F, 1.5F};
    BinMappers const mappers  = BinMappers::fit(batch, {});
    Dataset const    train    = Dataset::bin(batch, mappers, {});

    Config cfg;
    cfg.tree_config.min_data_in_leaf = 0;
    cfg.tree_config.min_child_hess   = 0.0F;
    cfg.tree_config.max_depth        = 3;

    Booster<MSEObjective, ObliviousGrower<>, AllRowsSampler> b{cfg};
    for (int i = 0; i < 6; ++i)
    {
        b.update_one_iter(train);
    }

    std::vector<float> raw(8 * 3);
    for (size_t r = 0; r < 8; ++r)
    {
        for (size_t f = 0; f < 3; ++f)
        {
            raw[(r * 3) + f] = batch.features[f][r];
        }
    }
    features_view const X{raw.data(), 8, 3};

    // A cache-served repeat is bit-identical to the call that filled it.
    std::vector<double> first(8 * 4);
    b.pred_contribs(X, first, 3);
    std::vector<double> again(8 * 4);
    b.pred_contribs(X, again, 3);
    REQUIRE(first == again);

    // Every mutation route must invalidate: contribs keep explaining the
    // CURRENT ensemble (efficiency against the current predict), never a
    // stale dense copy.
    auto const check_efficiency = [&]
    {
        std::vector<double> contribs(8 * 4);
        b.pred_contribs(X, contribs, 3);
        std::vector<float> pred(8);
        b.predict(X, pred);
        for (size_t r = 0; r < 8; ++r)
        {
            double sum = 0.0;
            for (size_t c = 0; c < 4; ++c)
            {
                sum += contribs[(r * 4) + c];
            }
            REQUIRE(sum == Catch::Approx(pred[r]).margin(1e-4));
        }
        return contribs;
    };

    b.truncate(3);
    REQUIRE(check_efficiency() != first);

    b.update_one_iter(train);
    check_efficiency();
}
