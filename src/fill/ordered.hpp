#pragma once

// grad and hess in one node's row order: the gather every fill of a
// node reads from.
//
// Included into src/grower.cpp only: the fill's pieces stay in one
// translation unit, so nothing here crosses a call the optimizer cannot see.

#include "bonsai/dataset.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include <cstddef>
#include <span>
#include <vector>

namespace bonsai::fill_detail
{

// grad and hess in one node's row order. A fill that walks the node once per
// feature reads them sequentially from here instead of re-walking the full
// arrays with scattered indices (n_features x full-array traffic otherwise).
// An empty hess view is the unit hessian: the fills add the literal 1.0F.
struct GhView
{
    std::span<float const> g;
    std::span<float const> h;
};

// Below this the gather runs serially: one pass over a node this small costs
// less than the parallel region that would spread it.
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
    // Capture views over the thread_local storage: naming the vectors inside
    // the parallel region would resolve to each worker's own (empty) one.
    std::span<float> const g{ordered_grad};
    std::span<float> const h{ordered_hess};
    // A team of one runs the loop inline and enters no region, which is what
    // a node below the threshold wants.
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

// grad and hess as one node's fill reads them: in place when the node's rows
// are the identity (the root, absent row sampling), gathered into its row
// order otherwise. In place is indexing by position, so it needs identity,
// not merely full cardinality.
inline GhView node_gh(Dataset const & /*ds*/, SplitInput const &node, floats_view grad,
                      floats_view hess)
{
    return node.rows_identity ? GhView{.g = grad, .h = hess}
                              : ordered_gh(node.rows, grad, hess);
}

} // namespace bonsai::fill_detail
