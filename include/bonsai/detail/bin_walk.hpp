#pragma once

#include "bonsai/dataset.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include <cstddef>
#include <vector>

namespace bonsai::detail
{

// Routing a tree in bin space, shared by the booster's prediction family and
// by TreeSHAP. Its own header so both callers include it without a cycle.

// A tree's splits in bin space, so a routing walk inverts bin_of_threshold
// once per tree instead of once per row: per split, the bin its threshold came
// from and its feature's missing bin. Indexed by node id (DenseTree) or by
// level (ObliviousTree).
struct SplitBins
{
    std::vector<bin_id_t> split;
    std::vector<bin_id_t> last;
};

inline SplitBins split_bins(DenseTree const &tree, Dataset const &ds)
{
    auto const &nodes = tree.nodes();
    SplitBins   sb{std::vector<bin_id_t>(nodes.size(), 0),
                 std::vector<bin_id_t>(nodes.size(), 0)};
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        if (!DenseTree::is_leaf(nodes[i]))
        {
            auto const f = nodes[i].feature_id;
            sb.split[i]  = ds.bin_of_threshold(f, nodes[i].threshold_or_value);
            sb.last[i]   = static_cast<bin_id_t>(ds.n_bins(f) - 1);
        }
    }
    return sb;
}

inline SplitBins split_bins(ObliviousTree const &tree, Dataset const &ds)
{
    auto const &splits = tree.splits();
    SplitBins   sb{std::vector<bin_id_t>(splits.size(), 0),
                 std::vector<bin_id_t>(splits.size(), 0)};
    for (size_t lvl = 0; lvl < splits.size(); ++lvl)
    {
        auto const f  = splits[lvl].feature_id;
        sb.split[lvl] = ds.bin_of_threshold(f, splits[lvl].threshold);
        sb.last[lvl]  = static_cast<bin_id_t>(ds.n_bins(f) - 1);
    }
    return sb;
}

// Row r's arrival node id (DenseTree) or leaf-table index (ObliviousTree),
// routed in bin space, which is what leaf_for returns for the raw row.
// `bin_of(fid)` yields that row's bin for one feature, which is the only thing
// the two bin layouts disagree on: the training columns answer with
// Dataset::bin_at, the eval path with the row-major mirror. Routing itself is
// routes_left either way, because bin(v) <= split_bin exactly when
// v <= cuts[split_bin] and the missing bin follows default_left.
template <typename BinOf>
node_id_t leaf_binned(DenseTree const &tree, SplitBins const &sb, BinOf const &bin_of)
{
    auto const &nodes = tree.nodes();
    node_id_t   idx   = 0;
    while (!DenseTree::is_leaf(nodes[idx]))
    {
        auto const &nd   = nodes[idx];
        bool const  left = routes_left(bin_of(nd.feature_id), sb.last[idx],
                                       sb.split[idx], nd.default_left);
        idx              = left ? nd.left : nd.right;
    }
    return idx;
}

template <typename BinOf>
node_id_t leaf_binned(ObliviousTree const &tree, SplitBins const &sb,
                      BinOf const &bin_of)
{
    auto const &splits = tree.splits();
    node_id_t   index  = 0;
    for (size_t lvl = 0; lvl < splits.size(); ++lvl)
    {
        auto const &s   = splits[lvl];
        bool const left = routes_left(bin_of(s.feature_id), sb.last[lvl], sb.split[lvl],
                                      s.default_left);
        index           = (index << 1U) | (left ? 0U : 1U);
    }
    return index;
}

// The same walk taken one step further: the leaf's contribution.
template <typename BinOf>
float value_binned(DenseTree const &tree, SplitBins const &sb, BinOf const &bin_of)
{
    return tree.nodes()[leaf_binned(tree, sb, bin_of)].threshold_or_value;
}

template <typename BinOf>
float value_binned(ObliviousTree const &tree, SplitBins const &sb, BinOf const &bin_of)
{
    return tree.leaf_table()[leaf_binned(tree, sb, bin_of)];
}

} // namespace bonsai::detail
