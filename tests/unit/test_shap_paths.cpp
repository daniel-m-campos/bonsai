#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "bonsai/bin_mapper.hpp"
#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/multiclass_booster.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/shap.hpp"
#include "bonsai/shap_paths.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include "test_grower_helpers.hpp"

using namespace bonsai;       // NOLINT
using namespace bonsai::test; // NOLINT

namespace
{

// Relative gap, floored at one so near-zero elements compare absolutely.
// Catch's Approx keeps its own relative epsilon (1.2e-5) alongside .margin(),
// which is far looser than anything claimed here, so the comparison is
// spelled out.
void check_close(double got, double want, double tol)
{
    double const scale = std::max({1.0, std::abs(got), std::abs(want)});
    CHECK(std::abs(got - want) / scale <= tol);
}

// Algorithm 2's arithmetic is double throughout, and so is the closed form's,
// so a golden tree leaves only summation order between them: the worst gap
// measured is 2.0e-15, and this pin sits three orders above it.
constexpr double k_exact_tol = 1e-12;

// A grower tree adds the one rounding the packed form makes on purpose: the
// merged cover fraction is stored float in an 8-byte element where Algorithm
// 2 keeps it double. Worst gap measured is 8.8e-9.
constexpr double k_binned_tol = 1e-7;

// ---------------------------------------------------------------------------
// A hand-cut feature space: the edges are exactly the thresholds a golden
// tree may split on, so bin_of_threshold inverts them exactly and every bin
// has a nameable raw value.
// ---------------------------------------------------------------------------

std::vector<float> const k_edges{0.5F, 1.5F, 2.5F, 3.5F};

// from_edges appends the FLT_MAX top band and the +inf missing sentinel, so
// n_bins is 6: finite bins 0..4, missing bin 5.
constexpr size_t k_hand_bins = 6;
constexpr size_t k_hand_last = 5;

BinMappers hand_mappers(size_t n_features)
{
    std::vector<BinMapper>   mappers;
    std::vector<std::string> names;
    for (size_t f = 0; f < n_features; ++f)
    {
        mappers.push_back(BinMapper::from_edges(k_edges));
        names.push_back("f" + std::to_string(f));
    }
    return BinMappers::from_mappers(std::move(mappers), std::move(names));
}

// A raw value landing in `bin`.
float value_for_bin(size_t bin)
{
    if (bin == k_hand_last)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (bin < k_edges.size())
    {
        return k_edges[bin]; // lower_bound lands on this cut
    }
    return k_edges.back() + 1.0F; // the FLT_MAX top band
}

// Every combination of per-feature bins, row-major.
std::vector<float> all_bin_rows(size_t n_features)
{
    size_t rows = 1;
    for (size_t f = 0; f < n_features; ++f)
    {
        rows *= k_hand_bins;
    }
    std::vector<float> raw(rows * n_features);
    for (size_t r = 0; r < rows; ++r)
    {
        size_t rest = r;
        for (size_t f = 0; f < n_features; ++f)
        {
            raw[(r * n_features) + f] = value_for_bin(rest % k_hand_bins);
            rest /= k_hand_bins;
        }
    }
    return raw;
}

// The reference: tree_shap over the ensemble, bias included.
std::vector<double> reference_phi(std::span<DenseTree const> trees, features_view X,
                                  row_id_t row, size_t n_features)
{
    std::vector<double> phi(n_features + 1, 0.0);
    for (auto const &tree : trees)
    {
        tree_shap(tree, X, row, phi);
    }
    return phi;
}

// The closed form, plus the per-tree expected values the packer leaves out.
std::vector<double> packed_phi(ShapPaths const &paths, std::span<DenseTree const> trees,
                               std::span<bin_id_t const> row_bins, size_t n_features)
{
    std::vector<double> phi(n_features + 1, 0.0);
    eval_shap_paths(paths, row_bins, n_features + 1, phi);
    for (auto const &tree : trees)
    {
        phi[n_features] += tree_expected_value(tree);
    }
    return phi;
}

// Compare both forms over every bin combination of `n_features` features,
// missing bins included.
void cross_check(std::span<DenseTree const> trees, size_t n_features)
{
    auto const          mappers = hand_mappers(n_features);
    auto const          paths   = pack_shap_paths(trees, mappers, 1);
    auto const          raw     = all_bin_rows(n_features);
    size_t const        rows    = raw.size() / n_features;
    features_view const X{raw.data(), rows, n_features};

    std::vector<bin_id_t> bins(n_features);
    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t f = 0; f < n_features; ++f)
        {
            bins[f] = mappers[f].transform(raw[(r * n_features) + f]);
        }
        auto const want = reference_phi(trees, X, static_cast<row_id_t>(r), n_features);
        auto const got  = packed_phi(paths, trees, bins, n_features);
        for (size_t c = 0; c <= n_features; ++c)
        {
            check_close(got[c], want[c], k_exact_tol);
        }
    }
}

// ---------------------------------------------------------------------------
// Random golden trees. Covers are laid out so every fraction the reference
// forms is exact in float: a node's cover stays a power of two wherever it
// has children of its own, and a leaf sibling absorbs the remainder.
// ---------------------------------------------------------------------------

struct Rng
{
    uint64_t state = 0;
    uint32_t next()
    {
        state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
        return static_cast<uint32_t>(state >> 33U);
    }
    size_t below(size_t n)
    {
        return next() % n;
    }
};

struct Building
{
    DenseTree::Nodes   nodes;
    std::vector<float> covers;
    size_t             depth    = 0;
    size_t             n_leaves = 0;
};

// NOLINTNEXTLINE(misc-no-recursion)
node_id_t build_shape(Building &b, Rng &rng, size_t depth, size_t max_depth,
                      size_t n_features)
{
    auto const id = static_cast<node_id_t>(b.nodes.size());
    b.nodes.emplace_back();

    if (depth >= max_depth || (depth >= 2 && rng.below(4) == 0))
    {
        auto const value = (static_cast<float>(rng.below(41)) - 20.0F) / 4.0F;
        b.nodes[id]      = DenseTree::leaf(value);
        b.depth          = std::max(b.depth, depth);
        ++b.n_leaves;
        return id;
    }

    auto const f     = static_cast<feature_id_t>(rng.below(n_features));
    auto const s     = rng.below(k_edges.size());
    bool const dleft = rng.below(2) == 0;
    auto const left  = build_shape(b, rng, depth + 1, max_depth, n_features);
    auto const right = build_shape(b, rng, depth + 1, max_depth, n_features);
    b.nodes[id]      = DenseTree::internal(f, k_edges[s], left, right, dleft);
    return id;
}

// NOLINTNEXTLINE(misc-no-recursion)
void assign_covers(Building &b, Rng &rng, node_id_t id, uint32_t cover)
{
    b.covers[id] = static_cast<float>(cover);
    if (DenseTree::is_leaf(b.nodes[id]))
    {
        return;
    }
    auto const left  = b.nodes[id].left;
    auto const right = b.nodes[id].right;
    bool const li    = !DenseTree::is_leaf(b.nodes[left]);
    bool const ri    = !DenseTree::is_leaf(b.nodes[right]);

    uint32_t split = 0;
    if (cover == 0)
    {
        split = 0;
    }
    else if (rng.below(7) == 0)
    {
        split = rng.below(2) == 0 ? 0 : cover; // a starved sibling
    }
    else if (!li && !ri)
    {
        split = static_cast<uint32_t>(rng.below(cover + 1));
    }
    else if (li && ri)
    {
        split = cover / 2;
    }
    else
    {
        uint32_t const half = cover >> rng.below(4);
        split               = li ? half : cover - half;
    }
    assign_covers(b, rng, left, split);
    assign_covers(b, rng, right, cover - split);
}

DenseTree random_tree(Rng &rng, size_t max_depth, size_t n_features)
{
    Building b;
    build_shape(b, rng, 0, max_depth, n_features);
    b.covers.assign(b.nodes.size(), 0.0F);
    assign_covers(b, rng, 0, 4096);
    return DenseTree{std::move(b.nodes),
                     DenseTree::Params{.depth = b.depth, .n_leaves = b.n_leaves},
                     {},
                     std::move(b.covers)};
}

// The dead-slot oblivious tree of test_shap.cpp, its thresholds moved onto a
// k_edges cut so the packer can invert them, its covers left alone.
DenseTree dead_slot_tree()
{
    ObliviousTree const tree{
        ObliviousTree::LevelSplits{
            {.feature_id = 0, .threshold = 1.5F, .default_left = true},
            {.feature_id = 1, .threshold = 1.5F, .default_left = true}},
        ObliviousTree::LeafTable{0.0F, 10.0F, 20.0F, 30.0F},
        {},
        {0.0F, 8.0F, 4.0F, 4.0F}};
    return dense_equivalent(tree);
}

// ---------------------------------------------------------------------------
// Corpus helpers: the boosters test_shap.cpp exercises, cross-checked over an
// eval set that includes NaN rows and rows off the trained support.
// ---------------------------------------------------------------------------

detail::ColumnBatch shap_batch()
{
    return detail::ColumnBatch{
        .features      = {{0.0F, 0.1F, 0.2F, 0.9F, 1.0F, 1.1F, 1.2F, 1.3F},
                          {0.0F, 1.0F, 0.1F, 1.1F, 0.2F, 1.2F, 0.3F, 1.3F},
                          {0.5F, 0.0F, 1.0F, 0.6F, 0.1F, 1.1F, 0.7F, 1.2F}},
        .labels        = {1.0F, -1.0F, 2.0F, -2.0F, 0.5F, 3.0F, -0.5F, 1.5F},
        .weights       = {},
        .feature_names = {"a", "b", "c"},
    };
}

detail::ColumnBatch eval_batch()
{
    float const         nan = std::numeric_limits<float>::quiet_NaN();
    detail::ColumnBatch out = shap_batch();
    std::array<std::array<float, 3>, 7> const extra{{{nan, 0.5F, 0.5F},
                                                     {0.5F, nan, 0.5F},
                                                     {0.5F, 0.5F, nan},
                                                     {nan, nan, nan},
                                                     {-5.0F, -5.0F, -5.0F},
                                                     {9.0F, 9.0F, 9.0F},
                                                     {-5.0F, 9.0F, nan}}};
    for (auto const &row : extra)
    {
        for (size_t f = 0; f < 3; ++f)
        {
            out.features[f].push_back(row[f]);
        }
        out.labels.push_back(0.0F);
    }
    return out;
}

// Cross-check a booster's pred_contribs_binned against pack + eval, composing
// the learning rate, the per-tree expected values and init_score by hand.
void check_booster(std::span<DenseTree const> trees, BinMappers const &mappers,
                   Dataset const &eval, std::span<double const> contribs,
                   size_t n_features, size_t n_classes, float lr,
                   std::span<float const> init_scores)
{
    auto const   paths = pack_shap_paths(trees, mappers, n_classes);
    size_t const cols  = n_features + 1;
    REQUIRE(!paths.heads.empty());
    std::vector<bin_id_t> row_bins(n_features, 0);
    for (size_t r = 0; r < eval.plane_n_rows(); ++r)
    {
        std::vector<double> phi(n_classes * cols, 0.0);
        for (size_t f = 0; f < n_features; ++f)
        {
            row_bins[f] = eval.bin_at(f, static_cast<row_id_t>(r));
        }
        eval_shap_paths(paths, row_bins, cols, phi);
        for (size_t k = 0; k < n_classes; ++k)
        {
            for (size_t t = k; t < trees.size(); t += n_classes)
            {
                phi[(k * cols) + n_features] += tree_expected_value(trees[t]);
            }
            for (size_t c = 0; c < cols; ++c)
            {
                phi[(k * cols) + c] *= lr;
            }
            phi[(k * cols) + n_features] += init_scores.empty() ? 0.0F : init_scores[k];
        }
        for (size_t c = 0; c < n_classes * cols; ++c)
        {
            check_close(phi[c], contribs[(r * n_classes * cols) + c], k_binned_tol);
        }
    }
}

} // namespace

TEST_CASE("Shap paths: the packed element is eight bytes", "[shap][paths]")
{
    STATIC_REQUIRE(sizeof(ShapPathElem) == 8);
    STATIC_REQUIRE(alignof(ShapPathElem) == 4);
    STATIC_REQUIRE(sizeof(ShapPathHead) == 12);
    STATIC_REQUIRE(ShapPathElem::k_missing_ok == 0x8000U);
}

TEST_CASE("Shap paths: the weight table is 1 / (n * C(n-1, i))", "[shap][paths]")
{
    constexpr size_t k_max = 12;
    auto const       w     = shap_path_weights(k_max);
    REQUIRE(w.size() == k_max * (k_max + 1) / 2);

    std::vector<double> binom(k_max + 1, 0.0);
    for (size_t n = 1; n <= k_max; ++n)
    {
        binom[0] = 1.0;
        for (size_t i = n; i-- > 1;)
        {
            binom[i] += binom[i - 1];
        }
        // binom now holds C(n - 1, .).
        size_t const base = (n - 1) * n / 2;
        for (size_t i = 0; i < n; ++i)
        {
            CHECK(w[base + i] == 1.0 / (static_cast<double>(n) * binom[i]));
        }
        // Shapley normalization: C(n - 1, i) subsets each carry w_i, and the
        // row sums to one.
        double total = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            total += binom[i] * w[base + i];
        }
        CHECK(total == Catch::Approx(1.0).margin(1e-12));
    }
}

TEST_CASE("Shap paths: closed form reproduces tree_shap on golden trees",
          "[shap][paths]")
{
    SECTION("depth 1")
    {
        DenseTree::Nodes nodes;
        nodes.emplace_back(
            DenseTree::internal(0, k_edges[1], node_id_t{1}, node_id_t{2}, false));
        nodes.emplace_back(DenseTree::leaf(3.0F));
        nodes.emplace_back(DenseTree::leaf(-1.0F));
        std::vector<DenseTree> const trees{
            DenseTree{std::move(nodes),
                      DenseTree::Params{.depth = 1, .n_leaves = 2},
                      {},
                      {8.0F, 6.0F, 2.0F}}};
        cross_check(trees, 2);
    }

    SECTION("the ice-cream shape on binned features")
    {
        DenseTree::Nodes shape;
        shape.emplace_back(
            DenseTree::internal(1, k_edges[0], node_id_t{1}, node_id_t{2}, false));
        shape.emplace_back(
            DenseTree::internal(0, k_edges[0], node_id_t{3}, node_id_t{4}, false));
        shape.emplace_back(
            DenseTree::internal(0, k_edges[0], node_id_t{5}, node_id_t{6}, false));
        shape.emplace_back(DenseTree::leaf(0.0F));
        shape.emplace_back(DenseTree::leaf(10.0F));
        shape.emplace_back(DenseTree::leaf(0.0F));
        shape.emplace_back(DenseTree::leaf(90.0F));
        std::vector<DenseTree> const trees{
            DenseTree{std::move(shape),
                      DenseTree::Params{.depth = 2, .n_leaves = 4},
                      {},
                      {100.0F, 50.0F, 50.0F, 25.0F, 25.0F, 25.0F, 25.0F}}};
        cross_check(trees, 2);
    }

    SECTION("the dead-slot oblivious expansion")
    {
        std::vector<DenseTree> const trees{dead_slot_tree()};
        cross_check(trees, 2);
    }

    SECTION("random trees to depth 6")
    {
        Rng rng{.state = 0x5EEDU};
        for (int t = 0; t < 24; ++t)
        {
            std::vector<DenseTree> const trees{random_tree(rng, 6, 3)};
            cross_check(trees, 3);
        }
    }

    SECTION("a random ensemble packed in one go")
    {
        Rng                    rng{.state = 0xC0FFEEU};
        std::vector<DenseTree> trees;
        for (int t = 0; t < 8; ++t)
        {
            trees.push_back(random_tree(rng, 5, 3));
        }
        cross_check(trees, 3);
    }
}

TEST_CASE("Shap paths: repeated features merge into one interval", "[shap][paths]")
{
    // Feature 0 split twice on one root-to-leaf path. Node 4 keeps a second
    // feature so the merge is visible in max_path_len.
    auto const two_level = [](size_t outer, size_t inner)
    {
        DenseTree::Nodes nodes;
        nodes.emplace_back(
            DenseTree::internal(0, k_edges[outer], node_id_t{1}, node_id_t{4}, true));
        nodes.emplace_back(
            DenseTree::internal(0, k_edges[inner], node_id_t{2}, node_id_t{3}, false));
        nodes.emplace_back(DenseTree::leaf(1.0F));
        nodes.emplace_back(DenseTree::leaf(2.0F));
        nodes.emplace_back(
            DenseTree::internal(1, k_edges[1], node_id_t{5}, node_id_t{6}, false));
        nodes.emplace_back(DenseTree::leaf(-3.0F));
        nodes.emplace_back(DenseTree::leaf(4.0F));
        return DenseTree{std::move(nodes),
                         DenseTree::Params{.depth = 2, .n_leaves = 4},
                         {},
                         {16.0F, 8.0F, 4.0F, 4.0F, 8.0F, 2.0F, 6.0F}};
    };

    SECTION("tighter under looser")
    {
        std::vector<DenseTree> const trees{two_level(2, 0)};
        auto const                   paths = pack_shap_paths(trees, hand_mappers(2), 1);
        CHECK(paths.max_path_len == 2);
        cross_check(trees, 2);
    }

    SECTION("looser under tighter")
    {
        std::vector<DenseTree> const trees{two_level(0, 2)};
        auto const                   paths = pack_shap_paths(trees, hand_mappers(2), 1);
        CHECK(paths.max_path_len == 2);
        cross_check(trees, 2);
    }

    SECTION("an unsatisfiable merged interval")
    {
        // Left at cut 0 then right at cut 2: no finite bin is at or below
        // bin 0 and above bin 2, and default_left true on both splits rules
        // the missing bin out on the right leg.
        DenseTree::Nodes nodes;
        nodes.emplace_back(
            DenseTree::internal(0, k_edges[0], node_id_t{1}, node_id_t{4}, true));
        nodes.emplace_back(
            DenseTree::internal(0, k_edges[2], node_id_t{2}, node_id_t{3}, true));
        nodes.emplace_back(DenseTree::leaf(1.0F));
        nodes.emplace_back(DenseTree::leaf(2.0F));
        nodes.emplace_back(DenseTree::leaf(-1.0F));
        std::vector<DenseTree> const trees{
            DenseTree{std::move(nodes),
                      DenseTree::Params{.depth = 2, .n_leaves = 3},
                      {},
                      {16.0F, 8.0F, 8.0F, 0.0F, 8.0F}}};

        auto const paths         = pack_shap_paths(trees, hand_mappers(1), 1);
        bool       unsatisfiable = false;
        for (auto const &e : paths.elems)
        {
            unsatisfiable =
                unsatisfiable ||
                (e.lo > e.hi && (e.feature & ShapPathElem::k_missing_ok) == 0);
        }
        CHECK(unsatisfiable);
        cross_check(trees, 1);
    }
}

TEST_CASE("Shap paths: max_path_len counts merged elements", "[shap][paths]")
{
    // Four splits deep over three features; the deepest path merges to three
    // elements because feature 0 repeats.
    DenseTree::Nodes nodes;
    nodes.emplace_back(
        DenseTree::internal(0, k_edges[0], node_id_t{1}, node_id_t{8}, false));
    nodes.emplace_back(
        DenseTree::internal(1, k_edges[1], node_id_t{2}, node_id_t{7}, false));
    nodes.emplace_back(
        DenseTree::internal(2, k_edges[2], node_id_t{3}, node_id_t{6}, false));
    nodes.emplace_back(
        DenseTree::internal(0, k_edges[3], node_id_t{4}, node_id_t{5}, false));
    nodes.emplace_back(DenseTree::leaf(1.0F));
    nodes.emplace_back(DenseTree::leaf(2.0F));
    nodes.emplace_back(DenseTree::leaf(3.0F));
    nodes.emplace_back(DenseTree::leaf(4.0F));
    nodes.emplace_back(DenseTree::leaf(5.0F));
    std::vector<DenseTree> const trees{
        DenseTree{std::move(nodes),
                  DenseTree::Params{.depth = 4, .n_leaves = 5},
                  {},
                  {32.0F, 16.0F, 8.0F, 4.0F, 2.0F, 2.0F, 4.0F, 8.0F, 16.0F}}};

    auto const paths = pack_shap_paths(trees, hand_mappers(3), 1);
    CHECK(paths.max_path_len == 3);
    CHECK(paths.heads.size() == 5);
    CHECK(paths.elems.size() == 12);
    CHECK(paths.last_bin[0] == k_hand_last);

    std::vector<size_t> lens;
    for (auto const &h : paths.heads)
    {
        lens.push_back(h.n_elems);
    }
    std::ranges::sort(lens);
    CHECK(lens == std::vector<size_t>{1, 2, 3, 3, 3});
    cross_check(trees, 3);
}

TEST_CASE("Shap paths: leafwise dense booster matches pred_contribs_binned",
          "[shap][paths][booster]")
{
    auto const       batch   = shap_batch();
    BinMappers const mappers = BinMappers::fit(batch, {});
    Dataset const    train   = Dataset::bin(batch, mappers, {});
    Dataset const    eval    = Dataset::bin(eval_batch(), mappers, {});

    Config cfg;
    cfg.tree_config.min_data_in_leaf = 0;
    cfg.tree_config.min_child_hess   = 0.0F;
    cfg.tree_config.max_depth        = 3;

    Booster<MSEObjective, LeafwiseGrower<>, AllRowsSampler> b{cfg};
    for (int i = 0; i < 8; ++i)
    {
        b.update_one_iter(train);
    }

    std::vector<double> contribs(eval.plane_n_rows() * 4);
    b.pred_contribs_binned(eval, contribs, 3);
    std::array<float, 1> const init{b.init_score()};
    check_booster(b.trees(), mappers, eval, contribs, 3, 1,
                  cfg.booster_config.learning_rate, init);
}

TEST_CASE("Shap paths: oblivious densified booster matches pred_contribs_binned",
          "[shap][paths][booster]")
{
    auto const       batch   = shap_batch();
    BinMappers const mappers = BinMappers::fit(batch, {});
    Dataset const    train   = Dataset::bin(batch, mappers, {});
    Dataset const    eval    = Dataset::bin(eval_batch(), mappers, {});

    Config cfg;
    cfg.tree_config.min_data_in_leaf = 0;
    cfg.tree_config.min_child_hess   = 0.0F;
    cfg.tree_config.max_depth        = 3;

    Booster<MSEObjective, ObliviousGrower<>, AllRowsSampler> b{cfg};
    for (int i = 0; i < 6; ++i)
    {
        b.update_one_iter(train);
    }

    std::vector<double> contribs(eval.plane_n_rows() * 4);
    b.pred_contribs_binned(eval, contribs, 3);
    auto const                 dense = internal::densify(b.trees());
    std::array<float, 1> const init{b.init_score()};
    check_booster(dense, mappers, eval, contribs, 3, 1,
                  cfg.booster_config.learning_rate, init);
}

// The device pack is sized by this count, so it is the growth law behind
// cuda_shap_plan's allocation decline: a densified oblivious tree emits a head
// per expansion slot, not per slot carrying evidence.
TEST_CASE("Shap paths: a densified oblivious tree packs a head per slot",
          "[shap][paths][booster]")
{
    auto const       batch   = shap_batch();
    BinMappers const mappers = BinMappers::fit(batch, {});
    Dataset const    train   = Dataset::bin(batch, mappers, {});

    Config cfg;
    cfg.tree_config.min_data_in_leaf = 0;
    cfg.tree_config.min_child_hess   = 0.0F;
    cfg.tree_config.max_depth        = 3;

    Booster<MSEObjective, ObliviousGrower<>, AllRowsSampler> b{cfg};
    for (int i = 0; i < 6; ++i)
    {
        b.update_one_iter(train);
    }

    auto const dense = internal::densify(b.trees());
    auto const paths = pack_shap_paths(dense, mappers, 1);

    size_t leaves = 0;
    for (DenseTree const &tree : dense)
    {
        size_t const here = static_cast<size_t>(
            std::ranges::count_if(tree.nodes(), [](DenseTree::Node const &n)
                                  { return DenseTree::is_leaf(n); }));
        // Perfect by construction: every level splits, so the expansion is full
        // and holds corners no row can reach.
        CHECK(here == (size_t{1} << cfg.tree_config.max_depth));
        CHECK(tree.nodes().size() == (2 * here) - 1);
        leaves += here;
    }
    CHECK(paths.heads.size() == leaves);
}

TEST_CASE("Shap paths: multiclass paths carry their class id", "[shap][paths][booster]")
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

    detail::ColumnBatch eval_raw = batch;
    eval_raw.features[0].push_back(std::numeric_limits<float>::quiet_NaN());
    eval_raw.labels.push_back(0.0F);
    eval_raw.features[0].push_back(-9.0F);
    eval_raw.labels.push_back(0.0F);
    Dataset const eval = Dataset::bin(eval_raw, mappers, {});

    Config cfg;
    cfg.objective.n_classes          = 3;
    cfg.tree_config.min_data_in_leaf = 0;
    cfg.tree_config.min_child_hess   = 0.0F;

    MulticlassBooster<DepthwiseGrower<>, AllRowsSampler> b{cfg};
    for (int i = 0; i < 10; ++i)
    {
        b.update_one_iter(train);
    }

    auto const          paths = pack_shap_paths(b.trees(), mappers, 3);
    std::vector<size_t> per_class(3, 0);
    for (auto const &h : paths.heads)
    {
        REQUIRE(h.klass < 3);
        ++per_class[h.klass];
    }
    CHECK(per_class[0] > 0);
    CHECK(per_class[1] > 0);
    CHECK(per_class[2] > 0);

    std::vector<double> contribs(eval.plane_n_rows() * 3 * 2);
    b.pred_contribs_binned(eval, contribs, 1);
    check_booster(b.trees(), mappers, eval, contribs, 1, 3,
                  cfg.booster_config.learning_rate, b.init_scores());
}

TEST_CASE("Shap paths: the packer refuses what it cannot pack", "[shap][paths]")
{
    DenseTree::Nodes nodes;
    nodes.emplace_back(
        DenseTree::internal(0, k_edges[1], node_id_t{1}, node_id_t{2}, false));
    nodes.emplace_back(DenseTree::leaf(1.0F));
    nodes.emplace_back(DenseTree::leaf(2.0F));
    DenseTree::Nodes copy = nodes;

    std::vector<DenseTree> const no_covers{DenseTree{
        std::move(copy), DenseTree::Params{.depth = 1, .n_leaves = 2}, {}, {}}};
    REQUIRE_THROWS_AS(pack_shap_paths(no_covers, hand_mappers(1), 1),
                      std::invalid_argument);

    // 300 finite bins do not fit the packed 8-bit interval.
    std::vector<float> wide;
    for (int i = 0; i < 300; ++i)
    {
        wide.push_back(static_cast<float>(i));
    }
    std::vector<BinMapper> wide_mapper;
    wide_mapper.push_back(BinMapper::from_edges(wide));
    auto const wide_mappers = BinMappers::from_mappers(std::move(wide_mapper), {"w"});
    std::vector<DenseTree> const trees{
        DenseTree{std::move(nodes),
                  DenseTree::Params{.depth = 1, .n_leaves = 2},
                  {},
                  {8.0F, 4.0F, 4.0F}}};
    REQUIRE_THROWS_AS(pack_shap_paths(trees, wide_mappers, 1), std::invalid_argument);
}
