#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/multiclass_booster.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/types.hpp"

using namespace bonsai; // NOLINT

namespace
{

// 3 features over 8 rows; feature c carries NaN, so every walk under test
// routes through the missing bin at least once.
detail::ColumnBatch binned_batch()
{
    float const nan = std::numeric_limits<float>::quiet_NaN();
    return detail::ColumnBatch{
        .features      = {{0.0F, 0.1F, 0.2F, 0.9F, 1.0F, 1.1F, 1.2F, 1.3F},
                          {0.0F, 1.0F, 0.1F, 1.1F, 0.2F, 1.2F, 0.3F, 1.3F},
                          {0.5F, nan, 1.0F, 0.6F, nan, 1.1F, 0.7F, 1.2F}},
        .labels        = {1.0F, -1.0F, 2.0F, -2.0F, 0.5F, 3.0F, -0.5F, 1.5F},
        .weights       = {},
        .feature_names = {"a", "b", "c"},
    };
}

// The batch's own rows as the row-major matrix the raw calls take. Predicting
// the rows the mappers were fit on is what makes bin(v) the row's stored bin
// by construction, so any routing disagreement is the code's, not the data's.
std::vector<float> row_major(detail::ColumnBatch const &batch)
{
    size_t const       n = batch.labels.size();
    size_t const       p = batch.features.size();
    std::vector<float> raw(n * p);
    for (size_t r = 0; r < n; ++r)
    {
        for (size_t f = 0; f < p; ++f)
        {
            raw[(r * p) + f] = batch.features[f][r];
        }
    }
    return raw;
}

Config small_config()
{
    Config cfg;
    cfg.tree_config.min_data_in_leaf = 0;
    cfg.tree_config.min_child_hess   = 0.0F;
    cfg.tree_config.max_depth        = 3;
    return cfg;
}

// Every member of the binned family against its raw twin, bit for bit.
// contrib_slices is 1 for a width-1 objective and n_classes for softmax.
template <typename B>
void check_parity(B const &b, Dataset const &ds, features_view X, size_t n_features,
                  size_t contrib_slices)
{
    size_t const n = X.extent(0);

    // The whole ensemble and a truncated prefix.
    for (size_t const k : {size_t{0}, size_t{3}})
    {
        std::vector<float> raw(n);
        std::vector<float> bin(n);
        b.predict_at(X, raw, k);
        b.predict_at_binned(ds, bin, k);
        for (size_t i = 0; i < n; ++i)
        {
            REQUIRE(raw[i] == bin[i]);
        }
    }

    std::vector<float> raw_staged(n * b.n_iters());
    std::vector<float> bin_staged(n * b.n_iters());
    b.predict_staged(X, raw_staged);
    b.predict_staged_binned(ds, bin_staged);
    for (size_t i = 0; i < raw_staged.size(); ++i)
    {
        REQUIRE(raw_staged[i] == bin_staged[i]);
    }

    std::vector<node_id_t> raw_leaf(n * b.n_trees());
    std::vector<node_id_t> bin_leaf(n * b.n_trees());
    b.predict_leaf(X, raw_leaf);
    b.predict_leaf_binned(ds, bin_leaf);
    for (size_t i = 0; i < raw_leaf.size(); ++i)
    {
        REQUIRE(raw_leaf[i] == bin_leaf[i]);
    }

    size_t const        cols = n * contrib_slices * (n_features + 1);
    std::vector<double> raw_phi(cols);
    std::vector<double> bin_phi(cols);
    b.pred_contribs(X, raw_phi, n_features);
    b.pred_contribs_binned(ds, bin_phi, n_features);
    for (size_t i = 0; i < cols; ++i)
    {
        REQUIRE(raw_phi[i] == bin_phi[i]);
    }
}

} // namespace

TEST_CASE("Binned prediction: depthwise booster matches the raw walk bit for bit",
          "[predict][binned]")
{
    detail::ColumnBatch const batch   = binned_batch();
    BinMappers const          mappers = BinMappers::fit(batch, {});
    Dataset const             ds      = Dataset::bin(batch, mappers, {});

    Booster<MSEObjective, DepthwiseGrower<>, AllRowsSampler> b{small_config()};
    for (int i = 0; i < 8; ++i)
    {
        b.update_one_iter(ds);
    }

    std::vector<float> const raw = row_major(batch);
    features_view const      X{raw.data(), 8, 3};
    check_parity(b, ds, X, 3, 1);

    // A width-1 objective refuses probabilities on both paths.
    std::vector<double> proba(8 * 2);
    REQUIRE_THROWS_AS(b.predict_proba_binned(ds, proba), std::logic_error);
}

TEST_CASE("Binned prediction: oblivious booster matches the raw walk bit for bit",
          "[predict][binned][oblivious]")
{
    detail::ColumnBatch const batch   = binned_batch();
    BinMappers const          mappers = BinMappers::fit(batch, {});
    Dataset const             ds      = Dataset::bin(batch, mappers, {});

    Booster<MSEObjective, ObliviousGrower<>, AllRowsSampler> b{small_config()};
    for (int i = 0; i < 6; ++i)
    {
        b.update_one_iter(ds);
    }

    std::vector<float> const raw = row_major(batch);
    features_view const      X{raw.data(), 8, 3};
    check_parity(b, ds, X, 3, 1);
}

TEST_CASE("Binned prediction: multiclass matches the raw walk bit for bit",
          "[predict][binned][multiclass]")
{
    detail::ColumnBatch batch = binned_batch();
    batch.labels              = {0.0F, 1.0F, 2.0F, 0.0F, 1.0F, 2.0F, 0.0F, 1.0F};
    BinMappers const mappers  = BinMappers::fit(batch, {});
    Dataset const    ds       = Dataset::bin(batch, mappers, {});

    Config cfg              = small_config();
    cfg.objective.n_classes = 3;

    MulticlassBooster<DepthwiseGrower<>, AllRowsSampler> b{cfg};
    for (int i = 0; i < 5; ++i)
    {
        b.update_one_iter(ds);
    }

    std::vector<float> const raw = row_major(batch);
    features_view const      X{raw.data(), 8, 3};
    check_parity(b, ds, X, 3, 3);

    std::vector<double> raw_proba(8 * 3);
    std::vector<double> bin_proba(8 * 3);
    b.predict_proba(X, raw_proba);
    b.predict_proba_binned(ds, bin_proba);
    for (size_t i = 0; i < raw_proba.size(); ++i)
    {
        REQUIRE(raw_proba[i] == bin_proba[i]);
    }
}
