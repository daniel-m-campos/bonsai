#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda.h>

#include <vector_types.h>

#include "bonsai/objective_traits.hpp"
#include "bonsai/split.hpp"

#include "device_buffer.cuh"

namespace bonsai
{
namespace
{

using namespace cuda_detail;

// NOLINTBEGIN(bugprone-easily-swappable-parameters,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,readability-function-cognitive-complexity,readability-identifier-naming)

constexpr __device__ size_t pair_off(uint32_t i)
{
    return 2 * static_cast<size_t>(i);
}

__global__ void interleave_kernel(float const *grad, float const *hess, uint32_t n,
                                  float2 *gh)
{
    uint32_t const span = gridDim.x * blockDim.x;
    for (uint32_t r = (blockIdx.x * blockDim.x) + threadIdx.x; r < n; r += span)
    {
        gh[r] = {.x = grad[r], .y = hess[r]};
    }
}

inline void interleave(float const *grad, float const *hess, uint32_t n, float2 *gh)
{
    interleave_kernel<<<dim3(std::clamp<uint32_t>(n / 256, 1, 1024)), dim3(256)>>>(
        grad, hess, n, gh);
    check(cudaGetLastError(), "interleave launch");
}

__global__ void gather_gh_kernel(float2 const *gh, uint32_t const *rows,
                                 uint32_t total_rows, float2 *gh_ordered)
{
    uint32_t const span = gridDim.x * blockDim.x;
    for (uint32_t k = (blockIdx.x * blockDim.x) + threadIdx.x; k < total_rows;
         k += span)
    {
        gh_ordered[k] = gh[rows[k]];
    }
}

inline void gather(float2 const *gh, uint32_t const *rows, uint32_t n,
                   float2 *gh_ordered)
{
    gather_gh_kernel<<<dim3(std::clamp<uint32_t>(n / 256, 1, 512)), dim3(256)>>>(
        gh, rows, n, gh_ordered);
    check(cudaGetLastError(), "gather launch");
}

inline __device__ hist_int_t quantise(float v, float scale)
{
    return __float2ll_rn(v * scale);
}

inline __device__ void hist_add(hist_int_t *cell, hist_int_t q)
{
    atomicAdd(reinterpret_cast<unsigned long long *>(cell),
              static_cast<unsigned long long>(q));
}

// perf: A 64-bit shared atomicAdd lowers to a compare-and-swap spin
// (ATOMS.CAST.SPIN.64 on sm_89); two native 32-bit ATOMS.ADD with the carry
// taken from the returned low word cut the 16M-row root fill from 1.09 s to
// 0.68 s per 100 trees on an L40S. Whole-word integer sums commute, so the
// split is exact whatever order the halves land in.
inline __device__ void hist_add_shared(hist_int_t *cell, hist_int_t q)
{
    auto *const    words = reinterpret_cast<uint32_t *>(cell);
    auto const     uq    = static_cast<unsigned long long>(q);
    uint32_t const lo    = static_cast<uint32_t>(uq);
    uint32_t const hi    = static_cast<uint32_t>(uq >> 32);
    uint32_t const old   = atomicAdd(words, lo);
    uint32_t const carry = old > (UINT32_MAX - lo) ? 1U : 0U;
    if (hi + carry != 0)
    {
        atomicAdd(words + 1, hi + carry);
    }
}

inline __device__ NodeRows node_rows(uint32_t const *rows, float2 const *gh_ordered,
                                     uint32_t const *row_offsets,
                                     uint32_t const *row_counts, uint32_t node)
{
    uint32_t const offset = row_offsets[node];
    return {rows + offset, gh_ordered + offset, row_counts[node]};
}

inline __device__ void zero_shared(hist_int_t *sh, uint32_t n)
{
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x)
    {
        sh[i] = 0;
    }
    __syncthreads();
}

__global__ void gh_absmax_kernel(float2 const *gh, uint32_t n, uint2 *absmax)
{
    float          mg   = 0.0F;
    float          mh   = 0.0F;
    uint32_t const span = gridDim.x * blockDim.x;
    for (uint32_t r = (blockIdx.x * blockDim.x) + threadIdx.x; r < n; r += span)
    {
        mg = fmaxf(mg, fabsf(gh[r].x));
        mh = fmaxf(mh, fabsf(gh[r].y));
    }
    for (int off = 16; off > 0; off >>= 1)
    {
        mg = fmaxf(mg, __shfl_down_sync(0xffffffffU, mg, off));
        mh = fmaxf(mh, __shfl_down_sync(0xffffffffU, mh, off));
    }
    if ((threadIdx.x & 31U) == 0)
    {
        atomicMax(&absmax->x, __float_as_uint(mg));
        atomicMax(&absmax->y, __float_as_uint(mh));
    }
}

inline __device__ float fixed_point_scale(uint32_t absmax_bits, uint32_t n_rows,
                                          double *inv)
{
    double const m = static_cast<double>(__uint_as_float(absmax_bits)) * n_rows;
    if (!(m > 0.0) || !isfinite(m))
    {
        *inv = 0.0;
        return 0.0F;
    }
    int const e = 61 - ilogb(m) < 126 ? 61 - ilogb(m) : 126;
    *inv        = ldexp(1.0, -e);
    return ldexpf(1.0F, e);
}

__global__ void gh_quant_kernel(uint2 const *absmax, uint32_t n_rows, GhQuant *out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }
    GhQuant q{};
    q.scale.x = fixed_point_scale(absmax->x, n_rows, &q.inv.x);
    q.scale.y = fixed_point_scale(absmax->y, n_rows, &q.inv.y);
    q.inv_f   = {static_cast<float>(q.inv.x), static_cast<float>(q.inv.y)};
    *out      = q;
}

inline void launch_gh_quant(float2 const *gh, uint32_t n, uint2 *absmax, GhQuant *out)
{
    check(cudaMemset(absmax, 0, sizeof(uint2)), "absmax zero");
    gh_absmax_kernel<<<dim3(std::clamp<uint32_t>(n / 256, 1, 1024)), dim3(256)>>>(
        gh, n, absmax);
    check(cudaGetLastError(), "absmax launch");
    gh_quant_kernel<<<dim3(1), dim3(1)>>>(absmax, n, out);
    check(cudaGetLastError(), "quant launch");
}

template <typename BinT>
__global__ void hist_kernel(BinT const *bins, float2 const *gh_ordered,
                            uint32_t const *rows, uint32_t const *row_offsets,
                            uint32_t const *row_counts, uint32_t const *features,
                            uint32_t const *n_bins, uint32_t n_rows, uint32_t n_feats,
                            uint32_t n_sel, hist_int_t *out, uint32_t stride,
                            uint32_t const *out_slot, GhQuant const *quant)
{
    extern __shared__ hist_int_t sh[];
    uint32_t const               f    = features[blockIdx.x];
    uint32_t const               node = blockIdx.y;
    NodeRows const seg = node_rows(rows, gh_ordered, row_offsets, row_counts, node);
    if (blockIdx.z * blockDim.x >= seg.count)
    {
        return;
    }
    uint32_t const nb = n_bins[f];
    zero_shared(sh, 2 * nb);
    float2 const   scale = quant->scale;
    uint32_t const span  = gridDim.z * blockDim.x;
    for (uint32_t k = (blockIdx.z * blockDim.x) + threadIdx.x; k < seg.count; k += span)
    {
        uint32_t const b = bins[tiled_cell(f, seg.rows[k], n_rows, n_feats)];
        float2 const   v = seg.gh[k];
        hist_add_shared(&sh[pair_off(b)], quantise(v.x, scale.x));
        hist_add_shared(&sh[pair_off(b) + 1], quantise(v.y, scale.y));
    }
    __syncthreads();
    uint32_t const oslot = out_slot != nullptr ? out_slot[node] : node;
    hist_int_t    *o =
        out + (((static_cast<size_t>(oslot) * n_sel) + blockIdx.x) * stride);
    for (uint32_t i = threadIdx.x; i < 2 * nb; i += blockDim.x)
    {
        if (sh[i] != 0)
        {
            hist_add(&o[i], sh[i]);
        }
    }
}

template <size_t Bytes>
inline __device__ void load_words(void const *sp, uint32_t *word)
{
    if constexpr (Bytes == 4)
    {
        word[0] = *static_cast<uint32_t const *>(sp);
    }
    else if constexpr (Bytes == 8)
    {
        uint2 const v = *static_cast<uint2 const *>(sp);
        word[0]       = v.x;
        word[1]       = v.y;
    }
    else
    {
        uint4 const v = *static_cast<uint4 const *>(sp);
        word[0]       = v.x;
        word[1]       = v.y;
        word[2]       = v.z;
        word[3]       = v.w;
    }
}

template <uint32_t W, typename BinT>
inline __device__ void load_strip(BinT const *sp, BinT *strip)
{
    constexpr size_t bytes = W * sizeof(BinT);
    if constexpr (bytes != 4 && bytes != 8 && bytes != 16)
    {
#pragma unroll
        for (uint32_t j = 0; j < W; ++j)
        {
            strip[j] = sp[j];
        }
    }
    else
    {
        constexpr uint32_t per_word = 4 / sizeof(BinT);
        constexpr uint32_t bin_bits = 8 * sizeof(BinT);
        uint32_t           word[bytes / 4];
        load_words<bytes>(sp, word);
#pragma unroll
        for (uint32_t j = 0; j < W; ++j)
        {
            strip[j] =
                static_cast<BinT>(word[j / per_word] >> (bin_bits * (j % per_word)));
        }
    }
}

template <uint32_t W> struct TileBlock
{
    uint32_t t;
    uint32_t node;
    uint32_t f0;
    uint32_t wt;
    uint32_t slot[W];
    bool     any;
};

template <uint32_t W>
inline __device__ TileBlock<W> tile_block(uint32_t const *sel_slot, uint32_t n_feats)
{
    TileBlock<W> tb{};
    tb.t    = blockIdx.x;
    tb.node = blockIdx.y;
    tb.f0   = tb.t * W;
    tb.wt   = tile_strip(tb.t, n_feats);
#pragma unroll
    for (uint32_t j = 0; j < W; ++j)
    {
        tb.slot[j] = j < tb.wt ? sel_slot[tb.f0 + j] : k_not_selected;
        tb.any     = tb.any || tb.slot[j] != k_not_selected;
    }
    return tb;
}

template <uint32_t W, typename BinT>
inline __device__ void load_partial_strip(BinT const *sp, uint32_t wt, BinT *strip)
{
#pragma unroll
    for (uint32_t j = 0; j < W; ++j)
    {
        strip[j] = j < wt ? sp[j] : BinT{};
    }
}

template <uint32_t W, bool Full, typename BinT, typename Visit>
inline __device__ void visit_tile_rows(TileBlock<W> const &tb, BinT const *bins,
                                       uint32_t n_rows, NodeRows const &seg,
                                       float2 scale, uint32_t first, uint32_t step,
                                       Visit visit)
{
    BinT const *tp = bins + (static_cast<size_t>(n_rows) * tb.t * W);
    for (uint32_t k = first; k < seg.count; k += step)
    {
        float2 const     v  = seg.gh[k];
        hist_int_t const qg = quantise(v.x, scale.x);
        hist_int_t const qh = quantise(v.y, scale.y);
        BinT             strip[W];
        if constexpr (Full)
        {
            load_strip<W>(tp + (static_cast<size_t>(seg.rows[k]) * W), strip);
        }
        else
        {
            load_partial_strip<W>(tp + (static_cast<size_t>(seg.rows[k]) * tb.wt),
                                  tb.wt, strip);
        }
#pragma unroll
        for (uint32_t j = 0; j < W; ++j)
        {
            if (tb.slot[j] != k_not_selected)
            {
                visit(j, pair_off(strip[j]), qg, qh);
            }
        }
    }
}

template <uint32_t W, typename BinT, typename Visit>
inline __device__ void visit_tile_rows(TileBlock<W> const &tb, BinT const *bins,
                                       uint32_t n_rows, NodeRows const &seg,
                                       float2 scale, uint32_t first, uint32_t step,
                                       Visit visit)
{
    if (tb.wt == W)
    {
        visit_tile_rows<W, true>(tb, bins, n_rows, seg, scale, first, step, visit);
    }
    else
    {
        visit_tile_rows<W, false>(tb, bins, n_rows, seg, scale, first, step, visit);
    }
}

template <uint32_t W, typename BinT>
__global__ void
hist_tile_kernel(BinT const *bins, float2 const *gh_ordered, uint32_t const *rows,
                 uint32_t const *row_offsets, uint32_t const *row_counts,
                 uint32_t const *sel_slot, uint32_t const *n_bins, uint32_t n_rows,
                 uint32_t n_feats, uint32_t n_sel, hist_int_t *out, uint32_t stride,
                 uint32_t const *out_slot, GhQuant const *quant)
{
    extern __shared__ hist_int_t sh[];
    TileBlock<W> const           tb = tile_block<W>(sel_slot, n_feats);
    if (!tb.any)
    {
        return;
    }
    NodeRows const seg = node_rows(rows, gh_ordered, row_offsets, row_counts, tb.node);
    if (blockIdx.z * blockDim.x >= seg.count)
    {
        return;
    }
    zero_shared(sh, tb.wt * stride);
    visit_tile_rows<W>(tb, bins, n_rows, seg, quant->scale,
                       (blockIdx.z * blockDim.x) + threadIdx.x, gridDim.z * blockDim.x,
                       [&](uint32_t j, uint32_t off, hist_int_t qg, hist_int_t qh)
                       {
                           hist_int_t *my = sh + (static_cast<size_t>(j) * stride);
                           hist_add_shared(&my[off], qg);
                           hist_add_shared(&my[off + 1], qh);
                       });
    __syncthreads();
    uint32_t const oslot = out_slot != nullptr ? out_slot[tb.node] : tb.node;
#pragma unroll
    for (uint32_t j = 0; j < W; ++j)
    {
        if (tb.slot[j] != k_not_selected)
        {
            hist_int_t const *base = sh + (static_cast<size_t>(j) * stride);
            hist_int_t       *o =
                out + (((static_cast<size_t>(oslot) * n_sel) + tb.slot[j]) * stride);
            uint32_t const nb = n_bins[tb.f0 + j];
            for (uint32_t i = threadIdx.x; i < 2 * nb; i += blockDim.x)
            {
                if (base[i] != 0)
                {
                    hist_add(&o[i], base[i]);
                }
            }
        }
    }
}

// perf: Small nodes skip the shared-memory stage: below ~512 rows the fixed
// per-(node,feature) zero+merge cost dominates, so row visits go straight into
// the node's global slot. One block per (tile, node): a single block per node
// measured 18.8 ms per launch at 131k x 16384 on an RTX PRO 6000, 7.6 s of a
// 27.4 s fit, from one SM issuing count x n_sel x 2 atomics.
template <uint32_t W, typename BinT>
__global__ void hist_small_kernel(BinT const *bins, float2 const *gh_ordered,
                                  uint32_t const *rows, uint32_t const *row_offsets,
                                  uint32_t const *row_counts, uint32_t const *sel_slot,
                                  uint32_t n_rows, uint32_t n_feats, uint32_t n_sel,
                                  hist_int_t *out, uint32_t stride,
                                  uint32_t const *out_slot, GhQuant const *quant)
{
    TileBlock<W> const tb = tile_block<W>(sel_slot, n_feats);
    if (!tb.any)
    {
        return;
    }
    hist_int_t    *o = out + (static_cast<size_t>(out_slot[tb.node]) * n_sel * stride);
    NodeRows const seg = node_rows(rows, gh_ordered, row_offsets, row_counts, tb.node);
    visit_tile_rows<W>(tb, bins, n_rows, seg, quant->scale, threadIdx.x, blockDim.x,
                       [&](uint32_t j, uint32_t off, hist_int_t qg, hist_int_t qh)
                       {
                           hist_int_t *cell =
                               o + (static_cast<size_t>(tb.slot[j]) * stride) + off;
                           hist_add(cell, qg);
                           hist_add(cell + 1, qh);
                       });
}

constexpr uint32_t k_part_rows_per_thread = 16;
constexpr uint32_t k_part_block           = 256;
constexpr uint32_t k_part_chunk           = k_part_block * k_part_rows_per_thread;

inline __device__ void block_scan(uint32_t *sh)
{
    __syncthreads();
    for (uint32_t step = 1; step < k_part_block; step *= 2)
    {
        uint32_t v = 0;
        if (threadIdx.x + 1 >= step + 1)
        {
            v = sh[threadIdx.x + 1 - step];
        }
        __syncthreads();
        sh[threadIdx.x + 1] += v;
        __syncthreads();
    }
}

inline __device__ bool goes_left_dev(uint32_t b, uint32_t last_bin, uint32_t bin,
                                     uint32_t dl)
{
    if (b == last_bin)
    {
        return dl != 0;
    }
    return b <= bin;
}

template <typename BinT>
__global__ void
route_count_kernel(BinT const *bins, uint32_t const *n_bins, uint32_t const *rows,
                   PartOpDev const *ops, uint32_t n_rows, uint32_t n_feats,
                   uint32_t max_chunks, uint8_t *flags, uint32_t *block_counts)
{
    __shared__ uint32_t sh[k_part_block];
    PartOpDev const     op    = ops[blockIdx.y];
    uint32_t const      chunk = blockIdx.x;
    uint32_t const      base  = chunk * k_part_chunk;
    uint32_t const      last  = n_bins[op.fid] - 1;
    uint32_t            mine  = 0;
    for (uint32_t j = 0; j < k_part_rows_per_thread; ++j)
    {
        uint32_t const i = base + (threadIdx.x * k_part_rows_per_thread) + j;
        if (i < op.count)
        {
            bool const l = goes_left_dev(
                bins[tiled_cell(op.fid, rows[op.offset + i], n_rows, n_feats)], last,
                op.bin, op.dl);
            flags[op.offset + i] = l ? 1 : 0;
            mine += l ? 1U : 0U;
        }
    }
    sh[threadIdx.x] = mine;
    __syncthreads();
    for (uint32_t step = k_part_block / 2; step > 0; step /= 2)
    {
        if (threadIdx.x < step)
        {
            sh[threadIdx.x] += sh[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        block_counts[(static_cast<size_t>(blockIdx.y) * max_chunks) + chunk] = sh[0];
    }
}

__global__ void seg_scan_kernel(uint32_t *block_counts, uint32_t max_chunks,
                                uint32_t *n_left)
{
    __shared__ uint32_t sh[k_part_block + 1];
    uint32_t *c     = block_counts + (static_cast<size_t>(blockIdx.x) * max_chunks);
    uint32_t  carry = 0;
    if (threadIdx.x == 0)
    {
        sh[0] = 0;
    }
    for (uint32_t base = 0; base < max_chunks; base += k_part_block)
    {
        uint32_t const k    = base + threadIdx.x;
        sh[threadIdx.x + 1] = k < max_chunks ? c[k] : 0;
        block_scan(sh);
        if (k < max_chunks)
        {
            c[k] = carry + sh[threadIdx.x];
        }
        carry += sh[k_part_block];
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        n_left[blockIdx.x] = carry;
    }
}

__global__ void scatter_kernel(uint32_t const *rows_in, float2 const *gh_in,
                               uint8_t const *flags, PartOpDev const *ops,
                               uint32_t const *block_counts, uint32_t const *n_left,
                               uint32_t max_chunks, uint32_t *rows_out, float2 *gh_out)
{
    __shared__ uint32_t sh[k_part_block + 1];
    PartOpDev const     op    = ops[blockIdx.y];
    uint32_t const      chunk = blockIdx.x;
    uint32_t const      base  = chunk * k_part_chunk;
    uint32_t            mine  = 0;
    for (uint32_t j = 0; j < k_part_rows_per_thread; ++j)
    {
        uint32_t const i = base + (threadIdx.x * k_part_rows_per_thread) + j;
        mine += (i < op.count && flags[op.offset + i] != 0) ? 1U : 0U;
    }
    sh[threadIdx.x + 1] = mine;
    if (threadIdx.x == 0)
    {
        sh[0] = 0;
    }
    block_scan(sh);
    uint32_t const nl_total = n_left[blockIdx.y];
    uint32_t const block_lefts =
        block_counts[(static_cast<size_t>(blockIdx.y) * max_chunks) + chunk];
    uint32_t lefts  = block_lefts + sh[threadIdx.x];
    uint32_t before = base + (threadIdx.x * k_part_rows_per_thread);
    for (uint32_t j = 0; j < k_part_rows_per_thread; ++j)
    {
        uint32_t const i = base + (threadIdx.x * k_part_rows_per_thread) + j;
        if (i >= op.count)
        {
            break;
        }
        uint32_t dst = 0;
        if (flags[op.offset + i] != 0)
        {
            dst = op.offset + lefts;
            ++lefts;
        }
        else
        {
            dst = op.offset + nl_total + (before + j - lefts);
        }
        rows_out[dst] = rows_in[op.offset + i];
        gh_out[dst]   = gh_in[op.offset + i];
    }
}

__global__ void stamp_kernel(uint32_t const *rows, PartOpDev const *segs,
                             uint32_t const *node_ids, uint32_t *leaf_by_row)
{
    PartOpDev const seg = segs[blockIdx.x];
    uint32_t const  id  = node_ids[blockIdx.x];
    for (uint32_t i = threadIdx.x; i < seg.count; i += blockDim.x)
    {
        leaf_by_row[rows[seg.offset + i]] = id;
    }
}

__global__ void zero_slots_kernel(hist_int_t *pool, uint32_t const *slot_ids,
                                  uint32_t id_stride, uint32_t slot_cells)
{
    uint32_t const slot = slot_ids[blockIdx.y * id_stride];
    hist_int_t    *out  = pool + (static_cast<size_t>(slot) * slot_cells);
    uint32_t const span = gridDim.x * blockDim.x;
    for (uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < slot_cells;
         i += span)
    {
        out[i] = 0;
    }
}

template <typename CellT>
inline __device__ CellT *strip_at(CellT *hists, uint32_t slot, uint32_t n_sel,
                                  uint32_t sel, uint32_t stride)
{
    return hists + (((static_cast<size_t>(slot) * n_sel) + sel) * stride);
}

struct CutStrip
{
    hist_int_t const *cells;
    hist_int_t const *small;
    hist_int_t       *store;
};

inline __device__ void derive_strip_range(CutStrip const &s, uint32_t from,
                                          uint32_t stride)
{
    for (uint32_t i = from + threadIdx.x; i < stride; i += blockDim.x)
    {
        s.store[i] = s.cells[i] - s.small[i];
    }
}

inline __device__ CutStrip open_strip(hist_int_t *hists, hist_int_t const *parents,
                                      SiblingDerive const &d, uint32_t slot,
                                      uint32_t n_sel, uint32_t sel, uint32_t stride,
                                      bool store_derived)
{
    hist_int_t *own = strip_at(hists, slot, n_sel, sel, stride);
    if (d.parent_slot == k_not_selected)
    {
        return {.cells = own, .small = nullptr, .store = nullptr};
    }
    return {.cells = strip_at(parents, d.parent_slot, n_sel, sel, stride),
            .small = strip_at(hists, d.small_slot, n_sel, sel, stride),
            .store = store_derived ? own : nullptr};
}

inline __device__ void derive_large_strip(hist_int_t *hists, hist_int_t const *parents,
                                          SiblingDerive const &d, uint32_t slot,
                                          uint32_t n_sel, uint32_t sel, uint32_t stride)
{
    if (d.parent_slot == k_not_selected)
    {
        return;
    }
    derive_strip_range(open_strip(hists, parents, d, slot, n_sel, sel, stride, true), 0,
                       stride);
    __syncwarp();
}

inline __device__ void derive_level_strips(hist_int_t *hists, hist_int_t const *parents,
                                           SiblingDerive const *derive,
                                           uint32_t n_nodes, uint32_t n_sel,
                                           uint32_t sel, uint32_t stride)
{
    if (derive == nullptr)
    {
        return;
    }
    for (uint32_t p = 0; p < n_nodes; ++p)
    {
        derive_large_strip(hists, parents, derive[p], p, n_sel, sel, stride);
    }
}

__global__ void derive_strips_kernel(hist_int_t *hists, hist_int_t const *parents,
                                     SiblingDerive const *derive, uint32_t n_sel,
                                     uint32_t stride)
{
    derive_large_strip(hists, parents, derive[blockIdx.y], blockIdx.y, n_sel,
                       blockIdx.x, stride);
}

struct SplitSumsDev
{
    double gL, hL, gR, hR;
};

inline __device__ SplitSumsDev split_sums_dev(double pg, double ph, double miss_g,
                                              double miss_h, double real_g,
                                              double real_h, int dl)
{
    double const rg = real_g - pg;
    double const rh = real_h - ph;
    return {.gL = dl != 0 ? pg + miss_g : pg,
            .hL = dl != 0 ? ph + miss_h : ph,
            .gR = dl != 0 ? rg : rg + miss_g,
            .hR = dl != 0 ? rh : rh + miss_h};
}

inline __device__ double nan_to_zero(double g)
{
    long long const magnitude = __double_as_longlong(g) & 0x7fffffffffffffffLL;
    return magnitude > 0x7ff0000000000000LL ? 0.0 : g;
}

template <bool k_l1>
inline __device__ double cut_score(double g, double h, double l1, double l2)
{
    double const t = k_l1 ? l1_thresholded(g, l1) : nan_to_zero(g);
    double const d = h + l2;
    return d > 0.0 ? (t * t) / d : 0.0;
}

inline __device__ hist_int_t warp_inclusive_scan(hist_int_t v)
{
    uint32_t const lane = threadIdx.x % 32;
    for (uint32_t o = 1; o < 32; o <<= 1)
    {
        hist_int_t const up = __shfl_up_sync(0xffffffffU, v, static_cast<int>(o));
        if (lane >= o)
        {
            v += up;
        }
    }
    return v;
}

inline __device__ uint32_t bit_reverse5(uint32_t i)
{
    return __brev(i) >> 27;
}

struct ShuffleTreeSum
{
    double p0, p1, p2, p3, p4;

    __device__ double push(uint32_t leaf, double v)
    {
        if ((leaf & 1U) == 0U)
        {
            p0 = v;
            return v;
        }
        v = p0 + v;
        if ((leaf & 2U) == 0U)
        {
            p1 = v;
            return v;
        }
        v = p1 + v;
        if ((leaf & 4U) == 0U)
        {
            p2 = v;
            return v;
        }
        v = p2 + v;
        if ((leaf & 8U) == 0U)
        {
            p3 = v;
            return v;
        }
        v = p3 + v;
        if ((leaf & 16U) == 0U)
        {
            p4 = v;
            return v;
        }
        return p4 + v;
    }
};

struct LevelNode
{
    double miss_g, miss_h, real_g, real_h, node_ps;
};

template <bool k_l1>
inline __device__ LevelNode stage_level_node(hist_int_t const *cells,
                                             double const *node_sums, uint32_t p,
                                             uint32_t nb, double2 inv, double l1,
                                             double l2)
{
    double const g      = node_sums[pair_off(p)];
    double const h      = node_sums[pair_off(p) + 1];
    double const miss_g = static_cast<double>(cells[pair_off(nb - 1)]) * inv.x;
    double const miss_h = static_cast<double>(cells[pair_off(nb - 1) + 1]) * inv.y;
    return {.miss_g  = miss_g,
            .miss_h  = miss_h,
            .real_g  = g - miss_g,
            .real_h  = h - miss_h,
            .node_ps = cut_score<k_l1>(g, h, l1, l2)};
}

inline __device__ int level_missing_dirs(hist_int_t const *hists, uint32_t n_nodes,
                                         uint32_t n_sel, uint32_t f, uint32_t stride,
                                         uint32_t nb)
{
    bool all_empty = true;
    for (uint32_t base = 0; base < n_nodes; base += k_level_find_threads)
    {
        uint32_t const p     = base + threadIdx.x;
        bool           empty = true;
        if (p < n_nodes)
        {
            hist_int_t const *cells = strip_at(hists, p, n_sel, f, stride);
            empty = cells[pair_off(nb - 1)] == 0 && cells[pair_off(nb - 1) + 1] == 0;
        }
        all_empty = all_empty && (__syncthreads_and(empty ? 1 : 0) != 0);
    }
    return all_empty ? 1 : 2;
}

struct LevelPrefixTotals
{
    hist_int_t own_g, own_h, pre_g, pre_h;
};

struct CutSums
{
    hist_int_t g, h;
};

inline __device__ CutSums level_cut_sums(hist_int_t const *cells, bool load, uint32_t b,
                                         uint32_t pass,
                                         LevelPrefixTotals (&tot)[k_level_find_warps])
{
    uint32_t const tid   = threadIdx.x;
    uint32_t const warp  = tid / 32;
    uint32_t const lane  = tid % 32;
    hist_int_t     own_g = load ? cells[pair_off(b)] : 0;
    hist_int_t     own_h = load ? cells[pair_off(b) + 1] : 0;
    hist_int_t     pre_g = 0;
    hist_int_t     pre_h = 0;
    for (uint32_t j = 0; cells != nullptr && j < pass; ++j)
    {
        uint32_t const e = (j * k_level_find_threads) + tid;
        pre_g += cells[pair_off(e)];
        pre_h += cells[pair_off(e) + 1];
    }
    own_g = warp_inclusive_scan(own_g);
    own_h = warp_inclusive_scan(own_h);
    if (pass > 0)
    {
        pre_g = warp_inclusive_scan(pre_g);
        pre_h = warp_inclusive_scan(pre_h);
    }
    if (lane == 31)
    {
        tot[warp] = {.own_g = own_g, .own_h = own_h, .pre_g = pre_g, .pre_h = pre_h};
    }
    __syncthreads();
    CutSums sums = {.g = own_g, .h = own_h};
    for (uint32_t w = 0; w < k_level_find_warps; ++w)
    {
        sums.g += tot[w].pre_g + (w < warp ? tot[w].own_g : 0);
        sums.h += tot[w].pre_h + (w < warp ? tot[w].own_h : 0);
    }
    return sums;
}

inline __device__ bool feat_better(double ga, int ba, int da, int va, double gb, int bb,
                                   int db, int vb)
{
    if (va != vb)
    {
        return va > vb;
    }
    if (va == 0)
    {
        return false;
    }
    if (ga != gb)
    {
        return ga > gb;
    }
    if (ba != bb)
    {
        return ba < bb;
    }
    return da > db;
}

inline __device__ bool feat_better(FeatBest const &a, FeatBest const &b)
{
    return feat_better(a.gain, a.bin, a.dl, a.valid, b.gain, b.bin, b.dl, b.valid);
}

inline __device__ long long key_of_positive_gain(double gain)
{
    return __double_as_longlong(gain);
}

inline __device__ bool split_better(FeatBest const &a, FeatBest const &b)
{
    if (a.valid != b.valid)
    {
        return a.valid > b.valid;
    }
    if (a.valid == 0)
    {
        return false;
    }
    long long const ka = key_of_positive_gain(a.gain);
    long long const kb = key_of_positive_gain(b.gain);
    if (ka != kb)
    {
        return ka > kb;
    }
    if (a.bin != b.bin)
    {
        return a.bin < b.bin;
    }
    return a.dl > b.dl;
}

inline __device__ FeatBest warp_best_cut(FeatBest best)
{
    for (int off = 16; off > 0; off >>= 1)
    {
        double const og = __shfl_down_sync(0xffffffffU, best.gain, off);
        int const    ob = __shfl_down_sync(0xffffffffU, best.bin, off);
        int const    od = __shfl_down_sync(0xffffffffU, best.dl, off);
        int const    ov = __shfl_down_sync(0xffffffffU, best.valid, off);
        if (feat_better(og, ob, od, ov, best.gain, best.bin, best.dl, best.valid))
        {
            best.gain  = og;
            best.bin   = ob;
            best.dl    = od;
            best.valid = ov;
        }
    }
    return best;
}

// perf: An infeasible node does NOT veto the whole level candidate. It
// contributes its parent score (zero gain) and the broadcast split still
// applies to it. At depth >= 5 some frontier node is always near-empty, so
// vetoing rejected every good deep cut and GPU levelwise trailed its own CPU
// grower (and catboost) at scale.
template <bool k_l1>
inline __device__ double level_cut_score(SplitSumsDev const &s, double node_ps,
                                         double l1, double l2, double min_child_hess)
{
    bool const ok = s.hL >= min_child_hess && s.hR >= min_child_hess;
    return ok ? cut_score<k_l1>(s.gL, s.hL, l1, l2) +
                    cut_score<k_l1>(s.gR, s.hR, l1, l2)
              : node_ps;
}

struct NodeCut
{
    double miss_g, miss_h, real_g, real_h, node_score, lo, hi;
    int    mc;
};

template <bool k_l1>
inline __device__ void
score_cut_exact(double pg, double ph, uint32_t b, int dl, NodeCut const &nd, double l1,
                double l2, double min_child_hess, double min_gain, FeatBest &best)
{
    auto const s =
        split_sums_dev(pg, ph, nd.miss_g, nd.miss_h, nd.real_g, nd.real_h, dl);
    if (s.hL < min_child_hess || s.hR < min_child_hess)
    {
        return;
    }
    if (nd.mc != 0)
    {
        double const wL = bounded_leaf_weight(s.gL, s.hL, l1, l2, nd.lo, nd.hi);
        double const wR = bounded_leaf_weight(s.gR, s.hR, l1, l2, nd.lo, nd.hi);
        if (static_cast<double>(nd.mc) * (wR - wL) < 0.0)
        {
            return;
        }
    }
    double const gain = cut_score<k_l1>(s.gL, s.hL, l1, l2) +
                        cut_score<k_l1>(s.gR, s.hR, l1, l2) - nd.node_score;
    if (!(gain > 0.0 && gain >= min_gain))
    {
        return;
    }
    FeatBest const cand = {.gain  = gain,
                           .gL    = s.gL,
                           .hL    = s.hL,
                           .gR    = s.gR,
                           .hR    = s.hR,
                           .bin   = static_cast<int32_t>(b),
                           .dl    = dl,
                           .valid = 1,
                           .sel   = best.sel};
    if (split_better(cand, best))
    {
        best = cand;
    }
}

inline __device__ FeatBest warp_best_split(FeatBest best)
{
    for (int off = 16; off > 0; off >>= 1)
    {
        FeatBest const o = {.gain  = __shfl_down_sync(0xffffffffU, best.gain, off),
                            .gL    = __shfl_down_sync(0xffffffffU, best.gL, off),
                            .hL    = __shfl_down_sync(0xffffffffU, best.hL, off),
                            .gR    = __shfl_down_sync(0xffffffffU, best.gR, off),
                            .hR    = __shfl_down_sync(0xffffffffU, best.hR, off),
                            .bin   = __shfl_down_sync(0xffffffffU, best.bin, off),
                            .dl    = __shfl_down_sync(0xffffffffU, best.dl, off),
                            .valid = __shfl_down_sync(0xffffffffU, best.valid, off),
                            .sel   = best.sel};
        if (split_better(o, best))
        {
            best = o;
        }
    }
    return best;
}

constexpr float k_inf_f32 = __builtin_inff();

inline __device__ Interval interval_of(double x)
{
    return {.lo = __double2float_rd(x), .hi = __double2float_ru(x)};
}

constexpr float k_two_pow_32 = 4294967296.0F;

inline __device__ float float_below(hist_int_t q)
{
    int32_t const  hi = static_cast<int32_t>(q >> 32);
    uint32_t const lo = static_cast<uint32_t>(q);
    return __fmaf_rd(__int2float_rd(hi), k_two_pow_32, __uint2float_rd(lo));
}

inline __device__ float float_above(hist_int_t q)
{
    int32_t const  hi = static_cast<int32_t>(q >> 32);
    uint32_t const lo = static_cast<uint32_t>(q);
    return __fmaf_ru(__int2float_ru(hi), k_two_pow_32, __uint2float_ru(lo));
}

inline __device__ Interval interval_of(hist_int_t q, float inv)
{
    return {.lo = __fmul_rd(float_below(q), inv), .hi = __fmul_ru(float_above(q), inv)};
}

inline __device__ Interval operator+(Interval a, Interval b)
{
    return {.lo = __fadd_rd(a.lo, b.lo), .hi = __fadd_ru(a.hi, b.hi)};
}

inline __device__ Interval operator-(Interval a, Interval b)
{
    return {.lo = __fsub_rd(a.lo, b.hi), .hi = __fsub_ru(a.hi, b.lo)};
}

inline __device__ Interval square(Interval a)
{
    float const near = a.lo > 0.0f ? a.lo : (a.hi < 0.0f ? -a.hi : 0.0f);
    float const far  = fmaxf(-a.lo, a.hi);
    return {.lo = __fmul_rd(near, near), .hi = __fmul_ru(far, far)};
}

inline __device__ Interval thresholded(Interval g, Interval l1)
{
    float const lo = g.lo > l1.hi ? __fsub_rd(g.lo, l1.hi)
                                  : (g.lo < -l1.lo ? __fadd_rd(g.lo, l1.lo) : 0.0f);
    float const hi = g.hi > l1.lo ? __fsub_ru(g.hi, l1.lo)
                                  : (g.hi < -l1.hi ? __fadd_ru(g.hi, l1.hi) : 0.0f);
    return {.lo = lo, .hi = hi};
}

inline __device__ float quotient_above(float x, float y)
{
    float const q = x / y;
    return q < k_inf_f32 ? __int_as_float(__float_as_int(q) + 1) : q;
}

inline __device__ float quotient_below(float x, float y)
{
    float const q = x / y;
    return q > 0.0f ? __int_as_float(__float_as_int(q) - 1) : 0.0f;
}

template <bool k_l1>
inline __device__ Interval cut_score_bounds(Interval g, Interval h, Interval l1,
                                            Interval l2)
{
    Interval const t  = k_l1 ? thresholded(g, l1) : g;
    Interval const t2 = square(t);
    Interval const d  = h + l2;
    float const    hi =
        d.lo > 0.0f ? quotient_above(t2.hi, d.lo) : (d.hi > 0.0f ? k_inf_f32 : 0.0f);
    float const lo = d.lo > 0.0f ? quotient_below(t2.lo, d.hi) : 0.0f;
    return {.lo = lo, .hi = hi};
}

inline __device__ bool is_finite_dev(double x)
{
    return (__double_as_longlong(x) & 0x7fffffffffffffffLL) < 0x7ff0000000000000LL;
}

struct ScreenNode
{
    Interval real_g, real_h, miss_g, miss_h, node_score, l1, l2;
    Interval min_child_hess, min_gain;
    float    inv_g, inv_h;
    bool     on;
};

inline __device__ ScreenNode screen_node(NodeCut const &nd, NodeScreen const &ns,
                                         longlong2 miss_q, GhQuant const &q,
                                         ScreenConst const &c, bool exhaustive)
{
    bool const finite = is_finite_dev(nd.real_g) && is_finite_dev(nd.real_h) &&
                        is_finite_dev(nd.miss_g) && is_finite_dev(nd.miss_h) &&
                        is_finite_dev(nd.node_score);
    float const inv_g = q.inv_f.x;
    float const inv_h = q.inv_f.y;
    bool const  exact_inv =
        static_cast<double>(inv_g) == q.inv.x && static_cast<double>(inv_h) == q.inv.y;
    Interval const miss_g = interval_of(miss_q.x, inv_g);
    Interval const miss_h = interval_of(miss_q.y, inv_h);
    return {.real_g         = ns.sum_grad - miss_g,
            .real_h         = ns.sum_hess - miss_h,
            .miss_g         = miss_g,
            .miss_h         = miss_h,
            .node_score     = ns.score_bounds,
            .l1             = c.l1,
            .l2             = c.l2,
            .min_child_hess = c.min_child_hess,
            .min_gain       = c.min_gain,
            .inv_g          = inv_g,
            .inv_h          = inv_h,
            .on             = !exhaustive && finite && exact_inv};
}

struct CutScreen
{
    float upper, certified;
    bool  drop;
};

struct SplitBounds
{
    Interval gL, hL, gR, hR;
};

inline __device__ SplitBounds split_bounds(Interval pg, Interval ph,
                                           ScreenNode const &sn, int dl)
{
    Interval const rg = sn.real_g - pg;
    Interval const rh = sn.real_h - ph;
    return {.gL = dl != 0 ? pg + sn.miss_g : pg,
            .hL = dl != 0 ? ph + sn.miss_h : ph,
            .gR = dl != 0 ? rg : rg + sn.miss_g,
            .hR = dl != 0 ? rh : rh + sn.miss_h};
}

template <bool k_l1>
inline __device__ CutScreen screen_cut(Interval pg, Interval ph, ScreenNode const &sn,
                                       int dl, int mc)
{
    SplitBounds const b = split_bounds(pg, ph, sn, dl);
    bool const        surely_valid =
        b.hL.lo >= sn.min_child_hess.hi && b.hR.lo >= sn.min_child_hess.hi;
    bool const surely_invalid =
        b.hL.hi < sn.min_child_hess.lo || b.hR.hi < sn.min_child_hess.lo;
    Interval const gain = (cut_score_bounds<k_l1>(b.gL, b.hL, sn.l1, sn.l2) +
                           cut_score_bounds<k_l1>(b.gR, b.hR, sn.l1, sn.l2)) -
                          sn.node_score;
    bool const certified =
        surely_valid && mc == 0 && gain.lo > 0.0f && gain.lo >= sn.min_gain.hi;
    return {.upper     = gain.hi,
            .certified = certified ? gain.lo : 0.0f,
            .drop      = surely_invalid || gain.hi <= 0.0f || gain.hi < sn.min_gain.lo};
}

inline __device__ float warp_max_nonnegative(float v)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __uint_as_float(__reduce_max_sync(0xffffffffU, __float_as_uint(v)));
#else
    for (int off = 16; off > 0; off >>= 1)
    {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffU, v, off));
    }
    return v;
#endif
}

struct Survivors
{
    hist_int_t qg, qh;
    uint32_t   b;
    int        dl;
    uint32_t   n;
};

template <bool k_l1>
inline __device__ void flush_survivors(Survivors &sv, double2 inv, NodeCut const &nd,
                                       double l1, double l2, double min_child_hess,
                                       double min_gain, FeatBest &best)
{
    if (threadIdx.x < sv.n)
    {
        double const pg = static_cast<double>(sv.qg) * inv.x;
        double const ph = static_cast<double>(sv.qh) * inv.y;
        score_cut_exact<k_l1>(pg, ph, sv.b, sv.dl, nd, l1, l2, min_child_hess, min_gain,
                              best);
    }
    sv.n = 0;
}

inline __device__ void append_survivors(Survivors &sv, uint32_t mask, hist_int_t qg,
                                        hist_int_t qh, uint32_t b, int dl)
{
    uint32_t const   lane = threadIdx.x;
    uint32_t const   cnt  = __popc(mask);
    uint32_t const   nth  = lane >= sv.n ? lane - sv.n + 1 : 1;
    int const        src  = static_cast<int>(__fns(mask, 0, static_cast<int>(nth)));
    hist_int_t const g_in = __shfl_sync(0xffffffffU, qg, src);
    hist_int_t const h_in = __shfl_sync(0xffffffffU, qh, src);
    uint32_t const   b_in = __shfl_sync(0xffffffffU, b, src);
    if (lane >= sv.n && lane < sv.n + cnt)
    {
        sv.qg = g_in;
        sv.qh = h_in;
        sv.b  = b_in;
        sv.dl = dl;
    }
    sv.n += cnt;
}

struct CutPrefix
{
    hist_int_t qg, qh;
};

template <bool k_derived, bool k_store>
inline __device__ longlong2 load_cut(CutStrip const &s, uint32_t b, uint32_t n_cut)
{
    static_assert(sizeof(longlong2) == 2 * sizeof(hist_int_t));
    if (b >= n_cut)
    {
        return longlong2{0, 0};
    }
    longlong2 const p = reinterpret_cast<longlong2 const *>(s.cells)[b];
    if constexpr (!k_derived)
    {
        return p;
    }
    else
    {
        longlong2 const q = reinterpret_cast<longlong2 const *>(s.small)[b];
        longlong2 const d = {p.x - q.x, p.y - q.y};
        if constexpr (k_store)
        {
            reinterpret_cast<longlong2 *>(s.store)[b] = d;
        }
        return d;
    }
}

inline __device__ longlong2 strip_miss(CutStrip const &s, uint32_t nb)
{
    size_t const at   = pair_off(nb - 1);
    longlong2    miss = {s.cells[at], s.cells[at + 1]};
    if (s.small != nullptr)
    {
        miss.x -= s.small[at];
        miss.y -= s.small[at + 1];
    }
    return miss;
}

inline __device__ CutPrefix warp_cut_prefix(longlong2 cut, hist_int_t &carry_g,
                                            hist_int_t &carry_h)
{
    hist_int_t const sg  = warp_inclusive_scan(cut.x);
    hist_int_t const sh_ = warp_inclusive_scan(cut.y);
    CutPrefix const  pre = {.qg = carry_g + sg, .qh = carry_h + sh_};
    carry_g += __shfl_sync(0xffffffffU, sg, 31);
    carry_h += __shfl_sync(0xffffffffU, sh_, 31);
    return pre;
}

template <bool k_l1, bool k_derived, bool k_store>
inline __device__ FeatBest sweep_cuts(CutStrip const &s, uint32_t n_cut, int n_dirs,
                                      NodeCut const &nd, ScreenNode const &sn,
                                      double2 inv, double l1, double l2,
                                      double min_child_hess, double min_gain,
                                      uint32_t sel)
{
    uint32_t const lane = threadIdx.x;
    FeatBest       best = {};
    best.sel            = static_cast<int32_t>(sel);
    Survivors  sv       = {};
    float      l_star   = 0.0f;
    hist_int_t carry_g  = 0;
    hist_int_t carry_h  = 0;
    longlong2  next     = load_cut<k_derived, k_store>(s, lane, n_cut);
    for (uint32_t base = 0; base < n_cut; base += 32)
    {
        uint32_t const  b   = base + lane;
        longlong2 const cut = next;
        next                = load_cut<k_derived, k_store>(s, b + 32, n_cut);
        CutPrefix const pre = warp_cut_prefix(cut, carry_g, carry_h);
        Interval const  pg  = interval_of(pre.qg, sn.inv_g);
        Interval const  ph  = interval_of(pre.qh, sn.inv_h);
        for (int d = 0; d < n_dirs; ++d)
        {
            int const       dl = 1 - d;
            CutScreen const sc = screen_cut<k_l1>(pg, ph, sn, dl, nd.mc);
            l_star =
                fmaxf(l_star, warp_max_nonnegative(b < n_cut ? sc.certified : 0.0f));
            bool const keep =
                b < n_cut && (!sn.on || (!sc.drop && !(sc.upper < l_star)));
            uint32_t const mask = __ballot_sync(0xffffffffU, keep);
            if (mask == 0)
            {
                continue;
            }
            if (sv.n + __popc(mask) > 32)
            {
                flush_survivors<k_l1>(sv, inv, nd, l1, l2, min_child_hess, min_gain,
                                      best);
            }
            append_survivors(sv, mask, pre.qg, pre.qh, b, dl);
        }
    }
    flush_survivors<k_l1>(sv, inv, nd, l1, l2, min_child_hess, min_gain, best);
    return warp_best_split(best);
}

inline __device__ bool node_sweeps(char const *allowed, size_t oidx, uint32_t nb)
{
    return (allowed == nullptr || allowed[oidx] != 0) && nb >= 2;
}

template <bool k_l1>
inline __device__ FeatBest sweep_strip(CutStrip const &s, uint32_t n_cut,
                                       uint32_t stride, int n_dirs, NodeCut const &nd,
                                       ScreenNode const &sn, double2 inv, double l1,
                                       double l2, double min_child_hess,
                                       double min_gain, uint32_t sel)
{
    if (s.small == nullptr)
    {
        return sweep_cuts<k_l1, false, false>(s, n_cut, n_dirs, nd, sn, inv, l1, l2,
                                              min_child_hess, min_gain, sel);
    }
    if (s.store == nullptr)
    {
        return sweep_cuts<k_l1, true, false>(s, n_cut, n_dirs, nd, sn, inv, l1, l2,
                                             min_child_hess, min_gain, sel);
    }
    derive_strip_range(s, static_cast<uint32_t>(pair_off(n_cut)), stride);
    return sweep_cuts<k_l1, true, true>(s, n_cut, n_dirs, nd, sn, inv, l1, l2,
                                        min_child_hess, min_gain, sel);
}

template <bool k_l1>
inline __device__ void
find_node(hist_int_t *hists, hist_int_t const *parents, SiblingDerive const *derive,
          uint32_t const *features, uint32_t const *n_bins, double const *node_sums,
          double const *node_bounds, char const *allowed, int const *monotone,
          uint32_t n_sel, uint32_t stride, double l1, double l2, double min_child_hess,
          double min_gain, ScreenConst const &screen, NodeScreen const *node_screen,
          FeatBest *out, uint32_t const *hist_slot, GhQuant const *quant,
          bool exhaustive, uint32_t n_nodes, bool store_derived)
{
    uint32_t const node = (2 * blockIdx.y) + (blockIdx.x & 1U);
    uint32_t const sel  = blockIdx.x >> 1;
    uint32_t const lane = threadIdx.x;
    if (node >= n_nodes)
    {
        return;
    }
    uint32_t const slot  = hist_slot != nullptr ? hist_slot[node] : node;
    CutStrip const strip = open_strip(hists, parents, derive[node], slot, n_sel, sel,
                                      stride, store_derived);
    size_t const   oidx  = (static_cast<size_t>(node) * n_sel) + sel;
    if (lane == 0)
    {
        out[oidx] = FeatBest{};
    }
    uint32_t const f  = features[sel];
    uint32_t const nb = n_bins[f];
    if (!node_sweeps(allowed, oidx, nb))
    {
        if (strip.store != nullptr)
        {
            derive_strip_range(strip, 0, stride);
        }
        return;
    }
    double2 const    inv     = quant->inv;
    double const     g_total = node_sums[pair_off(node)];
    double const     h_total = node_sums[pair_off(node) + 1];
    longlong2 const  miss_q  = strip_miss(strip, nb);
    double const     miss_g  = static_cast<double>(miss_q.x) * inv.x;
    double const     miss_h  = static_cast<double>(miss_q.y) * inv.y;
    NodeScreen const ns      = node_screen[node];
    NodeCut const    nd      = {.miss_g     = miss_g,
                                .miss_h     = miss_h,
                                .real_g     = g_total - miss_g,
                                .real_h     = h_total - miss_h,
                                .node_score = ns.score,
                                .lo         = node_bounds[pair_off(node)],
                                .hi         = node_bounds[pair_off(node) + 1],
                                .mc         = monotone[f]};
    ScreenNode const sn      = screen_node(nd, ns, miss_q, *quant, screen, exhaustive);
    uint32_t const   n_cut   = nb - 2;
    int const        n_dirs  = (miss_q.x == 0 && miss_q.y == 0) ? 1 : 2;

    FeatBest const best = sweep_strip<k_l1>(strip, n_cut, stride, n_dirs, nd, sn, inv,
                                            l1, l2, min_child_hess, min_gain, sel);
    if (lane == 0 && best.valid != 0)
    {
        out[oidx] = best;
    }
}

__global__ void __launch_bounds__(32, 16)
    find_kernel(hist_int_t *hists, hist_int_t const *parents,
                SiblingDerive const *derive, uint32_t const *features,
                uint32_t const *n_bins, double const *node_sums,
                double const *node_bounds, char const *allowed, int const *monotone,
                uint32_t n_sel, uint32_t stride, double l1, double l2,
                double min_child_hess, double min_gain, ScreenConst screen,
                NodeScreen const *node_screen, FeatBest *out, uint32_t const *hist_slot,
                GhQuant const *quant, bool exhaustive, uint32_t n_nodes,
                bool store_derived)
{
    if (l1 == 0.0)
    {
        find_node<false>(hists, parents, derive, features, n_bins, node_sums,
                         node_bounds, allowed, monotone, n_sel, stride, l1, l2,
                         min_child_hess, min_gain, screen, node_screen, out, hist_slot,
                         quant, exhaustive, n_nodes, store_derived);
        return;
    }
    find_node<true>(hists, parents, derive, features, n_bins, node_sums, node_bounds,
                    allowed, monotone, n_sel, stride, l1, l2, min_child_hess, min_gain,
                    screen, node_screen, out, hist_slot, quant, exhaustive, n_nodes,
                    store_derived);
}

inline dim3 find_grid(uint32_t n_sel, uint32_t n_nodes)
{
    return {2 * n_sel, (n_nodes + 1) / 2};
}

constexpr uint32_t k_reduce_threads = 256;

inline __device__ bool first_max_better(double gain, uint32_t sel, double best_gain,
                                        uint32_t best_sel)
{
    return sel != k_not_selected &&
           (gain > best_gain || (gain == best_gain && sel < best_sel));
}

__global__ void reduce_kernel(FeatBest const *per_feat, uint32_t n_sel, FeatBest *out)
{
    __shared__ double   gains[k_reduce_threads];
    __shared__ uint32_t sels[k_reduce_threads];
    uint32_t const      node      = blockIdx.x;
    FeatBest const     *row       = per_feat + (static_cast<size_t>(node) * n_sel);
    double              best_gain = 0.0;
    uint32_t            best_sel  = k_not_selected;
    for (uint32_t s = threadIdx.x; s < n_sel; s += blockDim.x)
    {
        if (row[s].valid != 0 && row[s].gain > best_gain)
        {
            best_gain = row[s].gain;
            best_sel  = s;
        }
    }
    gains[threadIdx.x] = best_gain;
    sels[threadIdx.x]  = best_sel;
    __syncthreads();
    for (uint32_t off = blockDim.x / 2; off > 0; off >>= 1)
    {
        if (threadIdx.x < off &&
            first_max_better(gains[threadIdx.x + off], sels[threadIdx.x + off],
                             gains[threadIdx.x], sels[threadIdx.x]))
        {
            gains[threadIdx.x] = gains[threadIdx.x + off];
            sels[threadIdx.x]  = sels[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        out[node] = sels[0] == k_not_selected ? FeatBest{} : row[sels[0]];
    }
}

struct LevelPassSums
{
    double parent, cut[2];
};

struct LevelCut
{
    uint32_t b, pass;
    bool     in;
};

template <bool k_l1>
inline __device__ double level_cut_partial(LevelNode const &nd, CutSums const &sums,
                                           bool live, int dl, double2 inv, double l1,
                                           double l2, double min_child_hess)
{
    double const pg = static_cast<double>(sums.g) * inv.x;
    double const ph = static_cast<double>(sums.h) * inv.y;
    auto const   s =
        split_sums_dev(pg, ph, nd.miss_g, nd.miss_h, nd.real_g, nd.real_h, dl);
    return live ? level_cut_score<k_l1>(s, nd.node_ps, l1, l2, min_child_hess) : 0.0;
}

template <bool k_l1>
inline __device__ void
stage_level_chunk(LevelNode (&s_node)[k_level_find_threads], hist_int_t const *hists,
                  double const *node_sums, uint32_t base, uint32_t n_nodes,
                  uint32_t n_sel, uint32_t f, uint32_t stride, uint32_t nb, double2 inv,
                  double l1, double l2)
{
    uint32_t const p = base + threadIdx.x;
    __syncthreads();
    if (p < n_nodes)
    {
        s_node[threadIdx.x] = stage_level_node<k_l1>(
            strip_at(hists, p, n_sel, f, stride), node_sums, p, nb, inv, l1, l2);
    }
    __syncthreads();
}

template <bool k_l1>
inline __device__ LevelPassSums level_pass_sums(
    hist_int_t const *hists, double const *node_sums,
    LevelNode (&s_node)[k_level_find_threads],
    LevelPrefixTotals (&s_tot)[2][k_level_find_warps], uint32_t f, uint32_t n_sel,
    uint32_t n_nodes, uint32_t stride, uint32_t nb, int n_dirs, LevelCut cut,
    double2 inv, double l1, double l2, double min_child_hess)
{
    LevelPassSums  acc = {.parent = 0.0, .cut = {0.0, 0.0}};
    ShuffleTreeSum ps_tree{};
    ShuffleTreeSum cut_tree[2]{};
    for (uint32_t base = 0; base < n_nodes; base += k_level_find_threads)
    {
        stage_level_chunk<k_l1>(s_node, hists, node_sums, base, n_nodes, n_sel, f,
                                stride, nb, inv, l1, l2);
        uint32_t const n_chunk = min(k_level_find_threads, n_nodes - base);
        for (uint32_t i0 = 0; i0 < n_chunk; i0 += 32)
        {
            for (uint32_t i = 0; i < 32; ++i)
            {
                uint32_t const    local  = i0 + bit_reverse5(i);
                uint32_t const    p      = base + local;
                bool const        active = p < n_nodes;
                bool const        live   = active && cut.in;
                LevelNode const   nd     = active ? s_node[local] : LevelNode{};
                hist_int_t const *cells =
                    active ? strip_at(hists, p, n_sel, f, stride) : nullptr;
                CutSums const sums =
                    level_cut_sums(cells, live, cut.b, cut.pass, s_tot[i & 1U]);
                double const t_ps = ps_tree.push(i, nd.node_ps);
                if (i == 31)
                {
                    acc.parent += t_ps;
                }
#pragma unroll
                for (int d = 0; d < 2; ++d)
                {
                    int const    dl = 1 - d;
                    double const cs =
                        d < n_dirs ? level_cut_partial<k_l1>(nd, sums, live, dl, inv,
                                                             l1, l2, min_child_hess)
                                   : 0.0;
                    double const t = cut_tree[dl].push(i, cs);
                    if (i == 31)
                    {
                        acc.cut[dl] += t;
                    }
                }
            }
        }
    }
    return acc;
}

inline __device__ FeatBest block_best_cut(FeatBest best,
                                          FeatBest (&s_best)[k_level_find_warps])
{
    uint32_t const warp = threadIdx.x / 32;
    uint32_t const lane = threadIdx.x % 32;
    best                = warp_best_cut(best);
    if (lane == 0)
    {
        s_best[warp] = best;
    }
    __syncthreads();
    for (uint32_t w = 1; w < k_level_find_warps; ++w)
    {
        FeatBest const o = s_best[w];
        if (feat_better(o, best))
        {
            best = o;
        }
    }
    return best;
}

struct LevelFindShared
{
    LevelNode         node[k_level_find_threads];
    LevelPrefixTotals tot[2][k_level_find_warps];
    FeatBest          best[k_level_find_warps];
};

template <bool k_l1>
inline __device__ void
level_find_feature(LevelFindShared &sh, hist_int_t *hists, hist_int_t const *parents,
                   SiblingDerive const *derive, uint32_t const *features,
                   uint32_t const *n_bins, double const *node_sums, uint32_t n_sel,
                   uint32_t n_nodes, uint32_t stride, double l1, double l2,
                   double min_child_hess, double min_gain, FeatBest *out_feat,
                   GhQuant const *quant)
{
    uint32_t const f   = blockIdx.x;
    uint32_t const tid = threadIdx.x;
    uint32_t const fid = features[f];
    uint32_t const nb  = n_bins[fid];

    derive_level_strips(hists, parents, derive, n_nodes, n_sel, f, stride);
    __syncthreads();
    if (tid == 0)
    {
        out_feat[f] = FeatBest{};
    }
    if (nb < 2)
    {
        return;
    }
    uint32_t const n_cut  = nb - 2;
    double2 const  inv    = quant->inv;
    int const      n_dirs = level_missing_dirs(hists, n_nodes, n_sel, f, stride, nb);

    FeatBest best = {.gain  = 0.0,
                     .gL    = 0.0,
                     .hL    = 0.0,
                     .gR    = 0.0,
                     .hR    = 0.0,
                     .bin   = 0,
                     .dl    = 0,
                     .valid = 0,
                     .sel   = static_cast<int32_t>(f)};
    for (uint32_t pass = 0; pass * k_level_find_threads < n_cut; ++pass)
    {
        LevelCut const      cut = {.b    = (pass * k_level_find_threads) + tid,
                                   .pass = pass,
                                   .in   = (pass * k_level_find_threads) + tid < n_cut};
        LevelPassSums const sums =
            level_pass_sums<k_l1>(hists, node_sums, sh.node, sh.tot, f, n_sel, n_nodes,
                                  stride, nb, n_dirs, cut, inv, l1, l2, min_child_hess);
#pragma unroll
        for (int d = 0; d < 2; ++d)
        {
            int const    dl   = 1 - d;
            double const gain = sums.cut[dl] - sums.parent;
            if (d < n_dirs && cut.in && gain > best.gain && gain >= min_gain)
            {
                best = {.gain  = gain,
                        .gL    = 0,
                        .hL    = 0,
                        .gR    = 0,
                        .hR    = 0,
                        .bin   = static_cast<int32_t>(cut.b),
                        .dl    = dl,
                        .valid = 1,
                        .sel   = static_cast<int32_t>(f)};
            }
        }
    }
    best = block_best_cut(best, sh.best);
    if (tid == 0)
    {
        out_feat[f] = best;
    }
}

__global__ void __launch_bounds__(k_level_find_threads)
    level_find_kernel(hist_int_t *hists, hist_int_t const *parents,
                      SiblingDerive const *derive, uint32_t const *features,
                      uint32_t const *n_bins, double const *node_sums, uint32_t n_sel,
                      uint32_t n_nodes, uint32_t stride, double l1, double l2,
                      double min_child_hess, double min_gain, FeatBest *out_feat,
                      GhQuant const *quant)
{
    __shared__ LevelFindShared sh;
    if (l1 == 0.0)
    {
        level_find_feature<false>(sh, hists, parents, derive, features, n_bins,
                                  node_sums, n_sel, n_nodes, stride, l1, l2,
                                  min_child_hess, min_gain, out_feat, quant);
        return;
    }
    level_find_feature<true>(sh, hists, parents, derive, features, n_bins, node_sums,
                             n_sel, n_nodes, stride, l1, l2, min_child_hess, min_gain,
                             out_feat, quant);
}

__global__ void level_child_sums_kernel(hist_int_t const *hists,
                                        double const *node_sums, FeatBest const *winner,
                                        uint32_t const *features,
                                        uint32_t const *n_bins, uint32_t n_nodes,
                                        uint32_t n_sel, uint32_t stride, double *out4,
                                        GhQuant const *quant)
{
    uint32_t const p = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (p >= n_nodes)
    {
        return;
    }
    FeatBest const b = *winner;
    if (b.valid == 0)
    {
        out4[(4 * p) + 0] = 0.0;
        out4[(4 * p) + 1] = 0.0;
        out4[(4 * p) + 2] = 0.0;
        out4[(4 * p) + 3] = 0.0;
        return;
    }
    auto const        sel   = static_cast<uint32_t>(b.sel);
    uint32_t const    nb    = n_bins[features[sel]];
    hist_int_t const *cells = hists + ((static_cast<size_t>(p) * n_sel + sel) * stride);
    double const      g     = node_sums[pair_off(p)];
    double const      h     = node_sums[pair_off(p) + 1];
    double2 const     inv   = quant->inv;
    double const      miss_g = static_cast<double>(cells[pair_off(nb - 1)]) * inv.x;
    double const      miss_h = static_cast<double>(cells[pair_off(nb - 1) + 1]) * inv.y;
    hist_int_t        pq_g = 0, pq_h = 0;
    for (uint32_t bb = 0; bb <= static_cast<uint32_t>(b.bin); ++bb)
    {
        pq_g += cells[pair_off(bb)];
        pq_h += cells[pair_off(bb) + 1];
    }
    double const pg = static_cast<double>(pq_g) * inv.x;
    double const ph = static_cast<double>(pq_h) * inv.y;
    auto const s = split_sums_dev(pg, ph, miss_g, miss_h, g - miss_g, h - miss_h, b.dl);
    out4[(4 * p) + 0] = s.gL;
    out4[(4 * p) + 1] = s.hL;
    out4[(4 * p) + 2] = s.gR;
    out4[(4 * p) + 3] = s.hR;
}

__global__ void map_leaf_values_kernel(uint32_t const *leaf_by_row,
                                       float const *node_values, uint32_t n_values,
                                       float *values, uint32_t n)
{
    uint32_t const r = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (r >= n)
    {
        return;
    }
    uint32_t const leaf = leaf_by_row[r];
    values[r]           = leaf < n_values ? node_values[leaf] : 0.0F;
}

template <DeviceObjectiveKind Kind, bool Weighted>
__global__ void gh_from_scores_kernel(float const *scores, float const *labels,
                                      float const *weights, uint32_t n, float2 *gh)
{
    uint32_t const span = gridDim.x * blockDim.x;
    for (uint32_t r = (blockIdx.x * blockDim.x) + threadIdx.x; r < n; r += span)
    {
        float const s = scores[r];
        float const y = labels[r];
        float       g = 0.0F;
        float       h = 0.0F;
        if constexpr (Kind == DeviceObjectiveKind::mse)
        {
            g = s - y;
            h = 1.0F;
        }
        else if constexpr (Kind == DeviceObjectiveKind::logloss)
        {
            float const p = 1.0F / (1.0F + expf(-s));
            g             = p - y;
            h             = p * (1.0F - p);
        }
        else if constexpr (Kind == DeviceObjectiveKind::poisson)
        {
            float const sc = fminf(fmaxf(s, -k_poisson_max_log), k_poisson_max_log);
            float const mu = expf(sc);
            g              = mu - y;
            h              = mu;
        }
        if constexpr (Weighted)
        {
            float const w = weights[r];
            g *= w;
            h *= w;
        }
        gh[r] = {.x = g, .y = h};
    }
}

template <DeviceObjectiveKind Kind>
inline void gh_from_scores_weighted(bool weighted, float const *scores,
                                    float const *labels, float const *weights,
                                    uint32_t n, float2 *gh, dim3 grid, dim3 block)
{
    if (weighted)
    {
        gh_from_scores_kernel<Kind, true>
            <<<grid, block>>>(scores, labels, weights, n, gh);
    }
    else
    {
        gh_from_scores_kernel<Kind, false>
            <<<grid, block>>>(scores, labels, weights, n, gh);
    }
}

inline void gh_from_scores(DeviceObjectiveKind kind, bool weighted, float const *scores,
                           float const *labels, float const *weights, uint32_t n,
                           float2 *gh)
{
    dim3 const grid(std::clamp<uint32_t>(n / 256, 1, 1024));
    dim3 const block(256);
    switch (kind)
    {
    case DeviceObjectiveKind::mse:
        gh_from_scores_weighted<DeviceObjectiveKind::mse>(weighted, scores, labels,
                                                          weights, n, gh, grid, block);
        break;
    case DeviceObjectiveKind::logloss:
        gh_from_scores_weighted<DeviceObjectiveKind::logloss>(
            weighted, scores, labels, weights, n, gh, grid, block);
        break;
    case DeviceObjectiveKind::poisson:
        gh_from_scores_weighted<DeviceObjectiveKind::poisson>(
            weighted, scores, labels, weights, n, gh, grid, block);
        break;
    case DeviceObjectiveKind::none:
        break;
    }
    check(cudaGetLastError(), "gh_from_scores launch");
}

template <DeviceObjectiveKind Kind>
__global__ void eval_loss_pass1_kernel(float const *scores, float const *labels,
                                       uint32_t n, double *partial)
{
    __shared__ double sl[256];
    double            acc = 0.0;
    for (uint32_t r = (blockIdx.x * blockDim.x) + threadIdx.x; r < n;
         r += gridDim.x * blockDim.x)
    {
        float const s = scores[r];
        float const y = labels[r];
        if constexpr (Kind == DeviceObjectiveKind::mse)
        {
            double const d = static_cast<double>(s) - static_cast<double>(y);
            acc += d * d;
        }
        else if constexpr (Kind == DeviceObjectiveKind::logloss)
        {
            float const ax = fabsf(s);
            acc += fmaxf(0.0F, s) + log1pf(expf(-ax)) - (y * s);
        }
        else if constexpr (Kind == DeviceObjectiveKind::poisson)
        {
            float const f = fminf(fmaxf(s, -k_poisson_max_log), k_poisson_max_log);
            acc += static_cast<double>(expf(f)) -
                   (static_cast<double>(y) * static_cast<double>(f));
        }
    }
    sl[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1U)
    {
        if (threadIdx.x < s)
        {
            sl[threadIdx.x] += sl[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        partial[blockIdx.x] = sl[0];
    }
}

inline uint32_t eval_loss_pass1(DeviceObjectiveKind kind, float const *scores,
                                float const *labels, uint32_t n, double *partial)
{
    dim3 const grid(std::clamp<uint32_t>(n / 256, 1, 1024));
    dim3 const block(256);
    switch (kind)
    {
    case DeviceObjectiveKind::mse:
        eval_loss_pass1_kernel<DeviceObjectiveKind::mse>
            <<<grid, block>>>(scores, labels, n, partial);
        break;
    case DeviceObjectiveKind::logloss:
        eval_loss_pass1_kernel<DeviceObjectiveKind::logloss>
            <<<grid, block>>>(scores, labels, n, partial);
        break;
    case DeviceObjectiveKind::poisson:
        eval_loss_pass1_kernel<DeviceObjectiveKind::poisson>
            <<<grid, block>>>(scores, labels, n, partial);
        break;
    case DeviceObjectiveKind::none:
        break;
    }
    check(cudaGetLastError(), "eval loss pass1 launch");
    return grid.x;
}

template <typename BinT>
__global__ void
route_add_kernel(BinT const *bins, uint32_t const *n_bins, uint32_t n_rows,
                 uint32_t n_feats, uint32_t const *feature, uint32_t const *split_bin,
                 uint32_t const *left, uint32_t const *right,
                 uint32_t const *default_left, uint32_t const *is_leaf,
                 float const *value, float lr, float *scores, uint32_t n,
                 uint32_t const *bin_rows, uint32_t const *score_rows)
{
    uint32_t const k = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (k >= n)
    {
        return;
    }
    uint32_t const r   = mapped_row(bin_rows, k);
    uint32_t       idx = 0;
    while (is_leaf[idx] == 0)
    {
        uint32_t const f    = feature[idx];
        uint32_t const last = n_bins[f] - 1;
        uint32_t const b    = bins[tiled_cell(f, r, n_rows, n_feats)];
        bool const l = (b == last) ? (default_left[idx] != 0) : (b <= split_bin[idx]);
        idx          = l ? left[idx] : right[idx];
    }
    scores[mapped_row(score_rows, k)] += lr * value[idx];
}

// perf: Identity row list built on device: full-data fits
// never ship the 64MB identity permutation over the bus or build it on host.
__global__ void iota_kernel(uint32_t *out, uint32_t n)
{
    uint32_t const i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n)
    {
        out[i] = i;
    }
}

__global__ void sum_gh_pass1_kernel(float2 const *gh, uint32_t n, double2 *partial)
{
    __shared__ double sg[256];
    __shared__ double sh[256];
    double            g = 0.0;
    double            h = 0.0;
    for (uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
    {
        g += gh[i].x;
        h += gh[i].y;
    }
    sg[threadIdx.x] = g;
    sh[threadIdx.x] = h;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1U)
    {
        if (threadIdx.x < s)
        {
            sg[threadIdx.x] += sg[threadIdx.x + s];
            sh[threadIdx.x] += sh[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        partial[blockIdx.x] = {sg[0], sh[0]};
    }
}

__global__ void sum_gh_pass2_kernel(double2 const *partial, uint32_t n_blocks,
                                    double2 *out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        double g = 0.0;
        double h = 0.0;
        for (uint32_t b = 0; b < n_blocks; ++b)
        {
            g += partial[b].x;
            h += partial[b].y;
        }
        *out = {g, h};
    }
}

// NOLINTEND(bugprone-easily-swappable-parameters,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,readability-function-cognitive-complexity,readability-identifier-naming)

} // namespace
} // namespace bonsai
