#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <vector>

#include "bonsai/config/tree_config.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/split.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include "grower_impl.hpp"

using namespace bonsai; // NOLINT
namespace gd = bonsai::grower_detail;

namespace
{

// A one-split plan mimicking plan_level's state: parent node 0 rewritten as
// internal with pre-allocated child leaves 1 and 2, and a partitioned
// PendingSplit ready for populate.
struct DemoteFixture
{
    TreeConfig             config;
    gd::LevelPlan          plan;
    DenseTree::Nodes       nodes;
    size_t                 n_leaves = 0;
    train_leaf_values      values;
    std::vector<node_id_t> leaf_ids;
    std::vector<float>     split_gains;

    DemoteFixture()
    {
        nodes.push_back(DenseTree::internal(0, 0.5F, 1, 2, true));
        nodes.push_back(DenseTree::leaf(0.0F));
        nodes.push_back(DenseTree::leaf(0.0F));
        values.assign(4, 0.0F);
        leaf_ids.assign(4, 0);
        split_gains.assign(3, 0.0F);
        split_gains[0] = 1.5F;

        gd::DeferredSplit d;
        d.parent.id = 0;
        d.left_id   = 1;
        d.right_id  = 2;
        // Parent histogram: one selected feature so finalize_as_leaf can
        // compute totals for the demoted leaf's value.
        Histogram h(3);
        h.add(0, -2.0F, 4.0F);
        d.p.parent_hists.push_back(std::move(h));
        plan.splits.push_back(std::move(d));
    }

    void run()
    {
        gd::demote_empty_splits(config, plan, nodes, n_leaves, values, leaf_ids,
                                split_gains);
    }
};

} // namespace

TEST_CASE("empty-child host split demotes the parent to a leaf", "[demote]")
{
    DemoteFixture fx;
    fx.plan.splits[0].p.left.rows  = {0, 1, 2, 3};
    fx.plan.splits[0].p.right.rows = {};
    fx.run();

    REQUIRE(fx.plan.splits.empty());
    REQUIRE(DenseTree::is_leaf(fx.nodes[0]));
    REQUIRE(fx.n_leaves == 1);
    REQUIRE(fx.split_gains[0] == 0.0F);
    // Survivor rows carry the demoted leaf's stamp.
    for (size_t r = 0; r < 4; ++r)
    {
        REQUIRE(fx.leaf_ids[r] == 0);
        REQUIRE(fx.values[r] == fx.nodes[0].threshold_or_value);
    }
}

TEST_CASE("device-plane empty child is not demoted", "[demote]")
{
    // Device-partitioned children carry row_count with empty rows; the
    // device has already scattered the parent's rows into the child slots,
    // so demotion would orphan their leaf stamps (the PR #29 bug). A 0-row
    // device child must pass through as an empty leaf instead.
    DemoteFixture fx;
    fx.plan.splits[0].p.left.rows.clear();
    fx.plan.splits[0].p.right.rows.clear();
    fx.plan.splits[0].p.left.row_count  = 4;
    fx.plan.splits[0].p.right.row_count = 0;
    fx.run();

    REQUIRE(fx.plan.splits.size() == 1);
    REQUIRE(!DenseTree::is_leaf(fx.nodes[0]));
    REQUIRE(fx.n_leaves == 0);
    REQUIRE(fx.split_gains[0] == 1.5F);
}

TEST_CASE("populated host split is left alone", "[demote]")
{
    DemoteFixture fx;
    fx.plan.splits[0].p.left.rows  = {0, 1};
    fx.plan.splits[0].p.right.rows = {2, 3};
    fx.run();

    REQUIRE(fx.plan.splits.size() == 1);
    REQUIRE(!DenseTree::is_leaf(fx.nodes[0]));
}
