#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

// TreeSHAP with the recursion flattened and the row taken out of the packing.
//
// Algorithm 2 (src/shap.cpp) walks the tree per row, growing and unwinding a
// "unique path" as it goes. Along one fixed root-to-leaf path the only thing
// the row decides is, per merged element, whether it satisfies every
// constraint that path places on that element's feature. So the tree can be
// packed once into its leaf paths, and a row evaluates each path in closed
// form:
//
//   P(t) = prod_j (z_j + o_j t)   over the path's merged elements j,
//
// with z_j the merged cover fraction and o_j in {0, 1} the row's answer. The
// Algorithm-2 permutation weights of a path of n elements are
// w_i = 1 / (n * C(n-1, i)), and element k contributes
//
//   value * (o_k - z_k) * sum_i w_i * [t^i] (P / (z_j + o_j t)).
//
// For an unsatisfied element the deflation is a scalar divide by z_k that the
// (o_k - z_k) prefactor cancels, so one weighted sum of P serves every
// unsatisfied element on the path. For a satisfied element the deflation is
// synthetic division by the monic factor (t + z_k). No divisions by the
// path-length ratios Algorithm 2 rebuilds per element, and no per-row
// bookkeeping: the packed form is what a device kernel consumes.

// One merged element of one leaf path: the intersected bin interval a row
// must fall in to satisfy every constraint the path places on this feature,
// and the product of cover fractions along it.
//
// The interval covers finite bins only. A feature's last bin holds missing
// values (BinMapper::transform sends NaN there), so the row satisfies this
// element iff
//
//   bin == last ? missing_ok : (lo <= bin && bin <= hi)
//
// where `last` comes from ShapPaths::last_bin. An empty interval is packed as
// lo = 1, hi = 0.
struct ShapPathElem
{
    // Set in `feature` when the feature's missing bin also satisfies.
    static constexpr uint16_t k_missing_ok = 0x8000U;

    uint16_t feature       = 0;
    uint8_t  lo            = 0;
    uint8_t  hi            = 0;
    float    zero_fraction = 0.0F;
};

static_assert(sizeof(ShapPathElem) == 8);
static_assert(alignof(ShapPathElem) == 4);

// One leaf path: where its elements live and what the leaf pays. `value` is
// the raw leaf contribution, NOT scaled by the learning rate and carrying no
// bias; a caller composes lr, the per-tree expected value, and init_score
// exactly as pred_contribs does.
struct ShapPathHead
{
    uint32_t first   = 0;
    uint16_t n_elems = 0;
    uint16_t klass   = 0;
    float    value   = 0.0F;
};

static_assert(sizeof(ShapPathHead) == 12);

// An ensemble's leaf paths, packed flat. Tree identity is dropped: the
// contributions of every path sum, so only the class the path scores for
// matters. `last_bin` is the per-feature missing bin, indexed by feature id
// and sized to the mappers, because the interval in ShapPathElem cannot
// express "the missing bin too" on its own.
struct ShapPaths
{
    std::vector<ShapPathElem> elems;
    std::vector<ShapPathHead> heads;
    std::vector<uint8_t>      last_bin;
    size_t                    max_path_len = 0;
};

// Pack every root-to-leaf path of every tree. `klass_stride` is the class
// count: trees are flat round-major, so tree t scores class t % klass_stride
// (pass 1 for single-output ensembles).
//
// Every leaf path is emitted, including paths through zero-cover branches.
// Algorithm 2 skips those branches to avoid dividing 0/0 while unwinding; the
// closed form has no such division, and a zero cover fraction either zeroes
// P outright (the branch carries nothing) or leaves a bare t factor (the row
// routed into a dead branch and still contributes). Both fall out of the
// arithmetic, so the packer needs no zero-cover rule.
//
// Throws std::invalid_argument if a tree carries no covers, if a split
// feature has more than 256 bins (lo/hi are 8-bit), or if a split feature id
// does not fit the 15 bits left beside k_missing_ok.
ShapPaths pack_shap_paths(std::span<DenseTree const> trees, BinMappers const &mappers,
                          size_t klass_stride);

// Algorithm 2's permutation weights, flattened: row n (for n in [1, max_len])
// holds w_i = 1 / (n * C(n-1, i)) for i in [0, n), at offset n * (n - 1) / 2.
std::vector<double> shap_path_weights(size_t max_len);

// Evaluate every packed path against one row's bins. `row_bins[f]` is the
// row's bin for feature f. Accumulates contributions into `phi`, which is
// n_classes slices of `cols` doubles each, class k at k * cols; feature f of
// class k lands at phi[(k * cols) + f]. The bias column phi[(k * cols) + cols
// - 1] is left alone: the per-tree expected values are not part of a path.
void eval_shap_paths(ShapPaths const &paths, std::span<bin_id_t const> row_bins,
                     size_t cols, std::span<double> phi);

// eval_shap_paths over a row of a Dataset, in pred_contribs' per-row layout
// (n_classes * (n_features + 1) doubles). Gathers the row's bins first, so it
// is the convenience form, not the batch form. Accumulates; the caller
// initializes phi and composes lr, the per-tree expected values, and
// init_score itself.
void shap_paths_one_row(ShapPaths const &paths, Dataset const &ds, row_id_t row,
                        size_t n_features, size_t n_classes, std::span<double> phi);

} // namespace bonsai
