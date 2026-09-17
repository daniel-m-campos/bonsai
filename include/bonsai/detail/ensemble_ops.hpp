#pragma once

#include "bonsai/dataset.hpp"
#include "bonsai/detail/bin_walk.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/row_view.hpp"
#include "bonsai/shap.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <mdspan>
#include <memory>
#include <mutex>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace bonsai::detail
{

// The synthesized name a feature carries when its source gave it none.
inline std::string numbered_feature_name(size_t f)
{
    return std::format("f{}", f);
}

inline std::vector<std::string> numbered_feature_names(size_t n)
{
    return std::views::iota(size_t{0}, n) |
           std::views::transform(numbered_feature_name) |
           std::ranges::to<std::vector<std::string>>();
}

// Accumulate a tree's (unscaled-by-lr) contribution over a binned Dataset's
// rows, routing the columns the tree was grown on. Used by DART to subtract
// dropped trees without caching per-tree train predictions, and by warm start.
template <Tree T>
void accumulate_train_contribution(T const &tree, Dataset const &ds, floats_out out)
{
    auto const sb = split_bins(tree, ds);
    parallel::for_each_index(ds.plane_n_rows(),
                             [&](size_t r)
                             {
                                 out[r] += value_binned(tree, sb, [&](size_t f)
                                                        { return ds.bin_at(f, r); });
                             });
}

// accumulate_train_contribution's reader twin: one output per VIEW row, in the
// view's order, which is the shape every reader over a row view answers in.
// Identical to the twin above when the dataset is not a view.
template <Tree T>
void accumulate_view_contribution(T const &tree, Dataset const &ds, floats_out out)
{
    auto const     sb = split_bins(tree, ds);
    RowIndex const rows{ds.row_view()};
    parallel::for_each_index(rows.size(),
                             [&](size_t k)
                             {
                                 row_id_t const r = rows[k];
                                 out[k] += value_binned(tree, sb, [&](size_t f)
                                                        { return ds.bin_at(f, r); });
                             });
}

inline std::string feature_label(std::span<std::string const> names, size_t f)
{
    return f < names.size() ? names[f] : numbered_feature_name(f);
}

// Indented text dump, one line per node.
inline void dump_tree(DenseTree const &tree, std::span<std::string const> names,
                      std::string &out)
{
    auto const &nodes  = tree.nodes();
    auto const &gains  = tree.split_gains();
    auto const &covers = tree.covers();
    auto        into   = std::back_inserter(out);
    // NOLINTNEXTLINE(misc-no-recursion)
    auto walk = [&](auto const &self, node_id_t id, int depth) -> void
    {
        out.append(static_cast<size_t>(depth) * 2, ' ');
        auto const &n    = nodes[id];
        bool const  leaf = DenseTree::is_leaf(n);
        if (leaf)
        {
            std::format_to(into, "leaf={:f}", n.threshold_or_value);
        }
        else
        {
            std::format_to(into, "{} <= {:f} [nan->{}] gain={:f}",
                           feature_label(names, n.feature_id), n.threshold_or_value,
                           n.default_left ? "left" : "right",
                           id < gains.size() ? gains[id] : 0.0F);
        }
        if (id < covers.size())
        {
            std::format_to(into, " cover={}", static_cast<size_t>(covers[id]));
        }
        out += '\n';
        if (!leaf)
        {
            self(self, n.left, depth + 1);
            self(self, n.right, depth + 1);
        }
    };
    walk(walk, 0, 0);
}

inline void dump_tree(ObliviousTree const &tree, std::span<std::string const> names,
                      std::string &out)
{
    auto const &splits = tree.splits();
    auto const &gains  = tree.level_gains();
    auto        into   = std::back_inserter(out);
    for (size_t lvl = 0; lvl < splits.size(); ++lvl)
    {
        std::format_to(into, "level {}: {} <= {:f} [nan->{}] gain={:f}\n", lvl,
                       feature_label(names, splits[lvl].feature_id),
                       splits[lvl].threshold,
                       splits[lvl].default_left ? "left" : "right",
                       lvl < gains.size() ? gains[lvl] : 0.0F);
    }
    out += "leaves:";
    for (float const v : tree.leaf_table())
    {
        std::format_to(into, " {:f}", v);
    }
    out += '\n';
    if (!tree.leaf_covers().empty())
    {
        out += "covers:";
        for (float const c : tree.leaf_covers())
        {
            std::format_to(into, " {}", static_cast<size_t>(c));
        }
        out += '\n';
    }
}

inline float gain_at(std::span<float const> gains, size_t i)
{
    return i < gains.size() ? gains[i] : 0.0F;
}

inline auto feature_gain_of(DenseTree const &tree)
{
    return std::views::iota(size_t{0}, tree.nodes().size()) |
           std::views::filter([&tree](size_t i)
                              { return !DenseTree::is_leaf(tree.nodes()[i]); }) |
           std::views::transform(
               [&tree](size_t i)
               {
                   return std::pair{size_t{tree.nodes()[i].feature_id},
                                    gain_at(tree.split_gains(), i)};
               });
}

inline auto feature_gain_of(ObliviousTree const &tree)
{
    return std::views::iota(size_t{0}, tree.splits().size()) |
           std::views::transform(
               [&tree](size_t lvl)
               {
                   return std::pair{size_t{tree.splits()[lvl].feature_id},
                                    gain_at(tree.level_gains(), lvl)};
               });
}

// One tree's contribution to per-feature importance.
template <Tree T>
void accumulate_importance(T const &tree, ImportanceType type, std::vector<double> &out)
{
    for (auto const [f, gain] : feature_gain_of(tree))
    {
        if (out.size() <= f)
        {
            out.resize(f + 1, 0.0);
        }
        out[f] += type == ImportanceType::gain ? gain : 1.0;
    }
}

// The per-tree biases a contribs batch shares across its rows: the expected
// value is row-independent, so one walk per tree replaces one per (row, tree).
template <typename Trees> std::vector<double> shap_biases(Trees const &trees)
{
    return trees |
           std::views::transform([](auto const &tree)
                                 { return tree_expected_value(tree); }) |
           std::ranges::to<std::vector<double>>();
}

// Per-row, per-tree leaf indices; out is n_rows * trees.size(), row-major by
// row. Both boosters store trees flat (multiclass round-major), so the walk
// is the same one.
template <typename Trees>
void predict_leaf_over(Trees const &trees, features_view X, std::span<node_id_t> out)
{
    size_t const n       = X.extent(0);
    size_t const n_trees = trees.size();
    assert(out.size() == n * n_trees);
    auto const leaves = std::mdspan(out.data(), n, n_trees);
    parallel::for_each_index(n,
                             [&](size_t i)
                             {
                                 for (size_t t = 0; t < n_trees; ++t)
                                 {
                                     leaves[i, t] =
                                         trees[t].leaf_for(X, static_cast<row_id_t>(i));
                                 }
                             });
}

// The per-tree SplitBins a binned batch shares across its rows: the threshold
// inversion is row-independent, so one walk per tree replaces one per (row,
// tree). The bias hoist's counterpart for routing.
template <typename Trees>
std::vector<SplitBins> tree_split_bins(Trees const &trees, Dataset const &bins)
{
    return trees |
           std::views::transform([&](auto const &tree)
                                 { return split_bins(tree, bins); }) |
           std::ranges::to<std::vector<SplitBins>>();
}

// predict_leaf_over's twin over binned rows, same row-major-by-row output.
template <typename Trees>
void predict_leaf_over_binned(Trees const &trees, Dataset const &bins,
                              std::span<node_id_t> out)
{
    RowIndex const rows{bins.row_view()};
    size_t const   n_trees = trees.size();
    assert(out.size() == rows.size() * n_trees);
    auto const sb     = tree_split_bins(trees, bins);
    auto const leaves = std::mdspan(out.data(), rows.size(), n_trees);
    parallel::for_each_index(
        rows.size(),
        [&](size_t k)
        {
            row_id_t const r      = rows[k];
            auto const     bin_of = [&](size_t f) { return bins.bin_at(f, r); };
            for (size_t t = 0; t < n_trees; ++t)
            {
                leaves[k, t] = leaf_binned(trees[t], sb[t], bin_of);
            }
        });
}

// TreeSHAP's cover-weighted walk is written against the dense shape, so an
// oblivious ensemble is expanded once per tree (2^depth nodes) rather than
// once per row.
inline std::vector<DenseTree> densify(std::vector<ObliviousTree> const &trees)
{
    return trees | std::views::transform(dense_equivalent) |
           std::ranges::to<std::vector<DenseTree>>();
}

// Epoch-keyed derived-value cache, one home for every pack a booster mints
// from its trees (the dense SHAP equivalents, the predict walk packs).
// Readers are concurrent (the bindings release the GIL); a mutation only
// bumps the booster's epoch and never touches the cache. Epochs start at 1,
// so epoch_ = 0 is the never-filled state. The build callable runs under
// the lock, so one thread rebuilds per stale epoch, and the returned
// shared_ptr keeps a superseded value alive for a reader still walking it.
template <typename ValueT> class EpochCache
{
  public:
    template <typename BuildF>
    std::shared_ptr<ValueT const> get(uint64_t epoch, BuildF &&build) const
    {
        std::scoped_lock const lock(mutex_);
        if (epoch_ != epoch)
        {
            cache_ = std::make_shared<ValueT const>(build());
            epoch_ = epoch;
        }
        return cache_;
    }

  private:
    mutable std::mutex                    mutex_;
    mutable std::shared_ptr<ValueT const> cache_;
    mutable uint64_t                      epoch_ = 0;
};

// A value paired with its mutation counter. Readers take read(); every writer
// goes through mutate(), which bumps the epoch before handing the value over.
// Mutation is never concurrent with reads (the GIL-released concurrency is
// predict-only). An epoch-keyed reader samples epoch() before read(), which
// get(trees.epoch(), build) does by argument order, so even a torn
// interleaving rebuilds on the next call instead of caching a stale value
// under the final epoch. The counter is monotonic, so several mutations in
// one round are fine.
template <typename T> class Versioned
{
  public:
    T const &read() const
    {
        return value_;
    }
    uint64_t epoch() const
    {
        return epoch_;
    }
    T &mutate()
    {
        ++epoch_;
        return value_;
    }

  private:
    T        value_;
    uint64_t epoch_ = 1;
};

// TreeSHAP walks DenseTree; an oblivious ensemble hands over its cached dense
// equivalents and a dense ensemble hands over its own trees.
template <typename TreeT, typename Fn>
void with_dense_trees(EpochCache<std::vector<DenseTree>> const &dense,
                      Versioned<std::vector<TreeT>> const &trees, Fn &&fn)
{
    if constexpr (std::same_as<TreeT, ObliviousTree>)
    {
        auto const held =
            dense.get(trees.epoch(), [&] { return densify(trees.read()); });
        std::forward<Fn>(fn)(*held);
    }
    else
    {
        std::forward<Fn>(fn)(trees.read());
    }
}

} // namespace bonsai::detail
