#pragma once

#include "bonsai/dataset.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/row_view.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include <cstddef>
#include <span>
#include <vector>

namespace bonsai::fill_detail
{

struct GhView
{
    std::span<float const> g;
    std::span<float const> h;
};

inline constexpr size_t k_gather_region_rows = 1 << 14;

inline GhView ordered_gh(std::span<row_id_t const> rows, floats_view grad,
                         floats_view hess)
{
    static thread_local std::vector<float> ordered_grad;
    static thread_local std::vector<float> ordered_hess;
    size_t const                           n    = rows.size();
    bool const                             unit = hess.empty();
    ordered_grad.resize(n);
    ordered_hess.resize(unit ? 0 : n);
    std::span<float> const g{ordered_grad};
    std::span<float> const h{ordered_hess};
    parallel::for_each_index_on(n < k_gather_region_rows ? 1 : parallel::n_threads(), n,
                                [&, g, h, rows, unit](size_t k)
                                {
                                    g[k] = grad[rows[k]];
                                    if (!unit)
                                    {
                                        h[k] = hess[rows[k]];
                                    }
                                });
    return {.g = g, .h = h};
}

inline GhView node_gh(Dataset const & /*ds*/, SplitInput const &node, floats_view grad,
                      floats_view hess)
{
    if (node.shape.identity)
    {
        return {.g = grad, .h = hess};
    }
    if (node.shape.runs.size() == 1)
    {
        RowRun const &run = node.shape.runs.front();
        size_t const  at  = run.start;
        return {.g = grad.subspan(at, run.size()),
                .h = hess.empty() ? hess : hess.subspan(at, run.size())};
    }
    return ordered_gh(node.rows, grad, hess);
}

} // namespace bonsai::fill_detail
