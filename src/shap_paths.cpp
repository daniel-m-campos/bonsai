#include "bonsai/shap_paths.hpp"

#include "bonsai/bin_mapper.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace bonsai
{

namespace
{

constexpr size_t k_max_bins = 256;

struct Active
{
    feature_id_t feature    = 0;
    int          lo         = 0;
    int          hi         = 0;
    bool         missing_ok = true;
    double       zero       = 1.0;
};

ShapPathElem pack_one(Active const &a)
{
    int lo = a.lo;
    int hi = a.hi;
    if (hi < lo)
    {
        lo = 1;
        hi = 0;
    }
    auto feature = static_cast<uint16_t>(a.feature);
    if (a.missing_ok)
    {
        feature |= ShapPathElem::k_missing_ok;
    }
    return {.feature       = feature,
            .lo            = static_cast<uint8_t>(lo),
            .hi            = static_cast<uint8_t>(hi),
            .zero_fraction = static_cast<float>(a.zero)};
}

// NOLINTNEXTLINE(misc-no-recursion)
void walk(DenseTree const &tree, BinMappers const &mappers, uint16_t klass,
          node_id_t id, std::vector<Active> &active, ShapPaths &out)
{
    auto const &n = tree.nodes()[id];
    if (DenseTree::is_leaf(n))
    {
        out.heads.push_back({.first   = static_cast<uint32_t>(out.elems.size()),
                             .n_elems = static_cast<uint16_t>(active.size()),
                             .klass   = klass,
                             .value   = n.threshold_or_value});
        for (auto const &a : active)
        {
            out.elems.push_back(pack_one(a));
        }
        out.max_path_len = std::max(out.max_path_len, active.size());
        return;
    }

    feature_id_t const f = n.feature_id;
    if (f >= ShapPathElem::k_missing_ok)
    {
        throw std::invalid_argument(
            "pack_shap_paths: feature id " + std::to_string(f) +
            " does not fit the 15 bits beside the missing-bin flag");
    }
    auto const &mapper = mappers[f];
    if (mapper.n_bins() > k_max_bins)
    {
        throw std::invalid_argument("pack_shap_paths: feature " + std::to_string(f) +
                                    " has " + std::to_string(mapper.n_bins()) +
                                    " bins; the packed interval is 8-bit");
    }
    auto const split = static_cast<int>(mapper.bin_of_threshold(n.threshold_or_value));
    auto const last  = static_cast<int>(mapper.n_bins()) - 1;

    auto const  &covers = tree.covers();
    double const cover  = covers[id];
    auto const   frac   = [&](node_id_t child)
    { return cover > 0.0 ? covers[child] / cover : 0.0; };

    auto const   found = std::ranges::find(active, f, &Active::feature);
    bool const   fresh = found == active.end();
    size_t const idx =
        fresh ? active.size() : static_cast<size_t>(found - active.begin());
    if (fresh)
    {
        active.push_back({.feature = f, .lo = 0, .hi = last - 1});
    }
    Active const saved = active[idx];

    active[idx].hi         = std::min(saved.hi, split);
    active[idx].missing_ok = saved.missing_ok && n.default_left;
    active[idx].zero       = saved.zero * frac(n.left);
    walk(tree, mappers, klass, n.left, active, out);

    active[idx].lo         = std::max(saved.lo, split + 1);
    active[idx].hi         = saved.hi;
    active[idx].missing_ok = saved.missing_ok && !n.default_left;
    active[idx].zero       = saved.zero * frac(n.right);
    walk(tree, mappers, klass, n.right, active, out);

    if (fresh)
    {
        active.pop_back();
    }
    else
    {
        active[idx] = saved;
    }
}

} // namespace

ShapPaths pack_shap_paths(std::span<DenseTree const> trees, BinMappers const &mappers,
                          size_t klass_stride)
{
    ShapPaths out;
    out.last_bin.resize(mappers.size(), 0);
    for (size_t f = 0; f < mappers.size(); ++f)
    {
        size_t const n_bins = std::clamp(mappers[f].n_bins(), size_t{1}, k_max_bins);
        out.last_bin[f]     = static_cast<uint8_t>(n_bins - 1);
    }

    std::vector<Active> active;
    for (size_t t = 0; t < trees.size(); ++t)
    {
        auto const &tree = trees[t];
        if (tree.covers().size() != tree.nodes().size())
        {
            throw std::invalid_argument(
                "pack_shap_paths: tree carries no per-node covers (model predates "
                "format v6 or was hand-built)");
        }
        auto const klass =
            static_cast<uint16_t>(klass_stride == 0 ? 0 : t % klass_stride);
        active.clear();
        walk(tree, mappers, klass, 0, active, out);
    }
    return out;
}

std::vector<double> shap_path_weights(size_t max_len)
{
    std::vector<double> weights(max_len * (max_len + 1) / 2, 0.0);
    if (max_len == 0)
    {
        return weights;
    }
    std::vector<double> binom(max_len, 0.0);
    binom[0] = 1.0;
    for (size_t n = 1; n <= max_len; ++n)
    {
        size_t const r = n - 1;
        if (r > 0)
        {
            binom[r] = 0.0;
            for (size_t i = r + 1; i-- > 1;)
            {
                binom[i] += binom[i - 1];
            }
        }
        size_t const base = r * n / 2;
        for (size_t i = 0; i < n; ++i)
        {
            weights[base + i] = 1.0 / (static_cast<double>(n) * binom[i]);
        }
    }
    return weights;
}

void eval_shap_paths(ShapPaths const &paths, std::span<bin_id_t const> row_bins,
                     size_t cols, std::span<double> phi)
{
    size_t const max_n = paths.max_path_len;
    if (max_n == 0)
    {
        return;
    }
    auto const          weights = shap_path_weights(max_n);
    std::vector<double> poly(max_n + 1, 0.0);
    std::vector<double> deflated(max_n, 0.0);
    std::vector<char>   satisfied(max_n, 0);

    for (auto const &head : paths.heads)
    {
        size_t const n = head.n_elems;
        if (n == 0)
        {
            continue;
        }
        auto const elems = std::span{paths.elems}.subspan(head.first, n);

        poly[0] = 1.0;
        for (size_t j = 0; j < n; ++j)
        {
            uint16_t const     tag = elems[j].feature;
            feature_id_t const f   = tag & ~ShapPathElem::k_missing_ok;
            bin_id_t const     bin = row_bins[f];
            bool const         one = bin == paths.last_bin[f]
                                         ? (tag & ShapPathElem::k_missing_ok) != 0
                                         : bin >= elems[j].lo && bin <= elems[j].hi;
            satisfied[j]           = static_cast<char>(one);

            double const z = elems[j].zero_fraction;
            poly[j + 1]    = 0.0;
            for (size_t i = j + 2; i-- > 1;)
            {
                poly[i] = (z * poly[i]) + (one ? poly[i - 1] : 0.0);
            }
            poly[0] *= z;
        }

        double const *w = weights.data() + ((n - 1) * n / 2);

        double unsatisfied_sum = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            unsatisfied_sum += w[i] * poly[i];
        }

        auto const value = static_cast<double>(head.value);
        for (size_t k = 0; k < n; ++k)
        {
            feature_id_t const f   = elems[k].feature & ~ShapPathElem::k_missing_ok;
            double            &out = phi[(size_t{head.klass} * cols) + f];
            if (satisfied[k] == 0)
            {
                out -= value * unsatisfied_sum;
                continue;
            }
            double const z  = elems[k].zero_fraction;
            deflated[n - 1] = poly[n];
            for (size_t i = n - 1; i-- > 0;)
            {
                deflated[i] = poly[i + 1] - (z * deflated[i + 1]);
            }
            double sum = 0.0;
            for (size_t i = 0; i < n; ++i)
            {
                sum += w[i] * deflated[i];
            }
            out += value * (1.0 - z) * sum;
        }
    }
}

} // namespace bonsai
