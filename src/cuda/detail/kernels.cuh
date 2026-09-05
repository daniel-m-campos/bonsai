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

inline __device__ void derive_large_strip(hist_int_t *hists, hist_int_t const *parents,
                                          SiblingDerive const &d, uint32_t slot,
                                          uint32_t n_sel, uint32_t sel, uint32_t stride)
{
    if (d.parent_slot == k_not_selected)
    {
        return;
    }
    hist_int_t       *large  = strip_at(hists, slot, n_sel, sel, stride);
    hist_int_t const *parent = strip_at(parents, d.parent_slot, n_sel, sel, stride);
    hist_int_t const *small  = strip_at(hists, d.small_slot, n_sel, sel, stride);
    for (uint32_t i = threadIdx.x; i < stride; i += blockDim.x)
    {
        large[i] = parent[i] - small[i];
    }
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
    return {.gL = pg + (dl != 0 ? miss_g : 0.0),
            .hL = ph + (dl != 0 ? miss_h : 0.0),
            .gR = (real_g - pg) + (dl == 0 ? miss_g : 0.0),
            .hR = (real_h - ph) + (dl == 0 ? miss_h : 0.0)};
}

inline __device__ double warp_sum(double v)
{
    for (int o = 16; o > 0; o >>= 1)
    {
        v += __shfl_down_sync(0xffffffffU, v, o);
    }
    return __shfl_sync(0xffffffffU, v, 0);
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

__global__ void find_kernel(hist_int_t *hists, hist_int_t const *parents,
                            SiblingDerive const *derive, uint32_t const *features,
                            uint32_t const *n_bins, double const *node_sums,
                            double const *node_bounds, char const *allowed,
                            int const *monotone, uint32_t n_sel, uint32_t stride,
                            double l1, double l2, double min_child_hess,
                            double min_gain, FeatBest *out, uint32_t const *hist_slot,
                            GhQuant const *quant)
{
    uint32_t const node = blockIdx.y;
    uint32_t const sel  = blockIdx.x;
    uint32_t const lane = threadIdx.x;
    if (sel >= n_sel)
    {
        return;
    }
    uint32_t const slot = hist_slot != nullptr ? hist_slot[node] : node;
    derive_large_strip(hists, parents, derive[node], slot, n_sel, sel, stride);
    size_t const oidx = (static_cast<size_t>(node) * n_sel) + sel;
    if (lane == 0)
    {
        out[oidx] = FeatBest{};
    }
    if (allowed != nullptr && allowed[oidx] == 0)
    {
        return;
    }
    uint32_t const f  = features[sel];
    uint32_t const nb = n_bins[f];
    if (nb < 2)
    {
        return;
    }
    hist_int_t const *cells      = strip_at(hists, slot, n_sel, sel, stride);
    double2 const     inv        = quant->inv;
    double const      g_total    = node_sums[pair_off(node)];
    double const      h_total    = node_sums[pair_off(node) + 1];
    hist_int_t const  miss_qg    = cells[pair_off(nb - 1)];
    hist_int_t const  miss_qh    = cells[pair_off(nb - 1) + 1];
    double const      miss_g     = static_cast<double>(miss_qg) * inv.x;
    double const      miss_h     = static_cast<double>(miss_qh) * inv.y;
    double const      node_score = score(g_total, h_total, l1, l2);
    double const      real_grad  = g_total - miss_g;
    double const      real_hess  = h_total - miss_h;
    double const      lo         = node_bounds[pair_off(node)];
    double const      hi         = node_bounds[pair_off(node) + 1];
    int const         mc         = monotone[f];
    uint32_t const    n_cut      = nb - 2;

    int const n_dirs = (miss_qg == 0 && miss_qh == 0) ? 1 : 2;

    double  best_gain = 0.0;
    int32_t best_bin = 0, best_dl = 0, best_valid = 0;
    double  bgL = 0, bhL = 0, bgR = 0, bhR = 0;

    hist_int_t carry_g = 0;
    hist_int_t carry_h = 0;
    for (uint32_t base = 0; base < n_cut; base += 32)
    {
        uint32_t const b   = base + lane;
        hist_int_t     sg  = (b < n_cut) ? cells[pair_off(b)] : 0;
        hist_int_t     sh_ = (b < n_cut) ? cells[pair_off(b) + 1] : 0;
        for (int off = 1; off < 32; off <<= 1)
        {
            hist_int_t const ng = __shfl_up_sync(0xffffffffU, sg, off);
            hist_int_t const nh = __shfl_up_sync(0xffffffffU, sh_, off);
            if (lane >= static_cast<uint32_t>(off))
            {
                sg += ng;
                sh_ += nh;
            }
        }
        double const pg = static_cast<double>(carry_g + sg) * inv.x;
        double const ph = static_cast<double>(carry_h + sh_) * inv.y;
        carry_g += __shfl_sync(0xffffffffU, sg, 31);
        carry_h += __shfl_sync(0xffffffffU, sh_, 31);

        if (b < n_cut)
        {
            for (int d = 0; d < n_dirs; ++d)
            {
                int const  dl = 1 - d;
                auto const s =
                    split_sums_dev(pg, ph, miss_g, miss_h, real_grad, real_hess, dl);
                if (s.hL < min_child_hess || s.hR < min_child_hess)
                {
                    continue;
                }
                if (mc != 0)
                {
                    double const wL = bounded_leaf_weight(s.gL, s.hL, l1, l2, lo, hi);
                    double const wR = bounded_leaf_weight(s.gR, s.hR, l1, l2, lo, hi);
                    if (static_cast<double>(mc) * (wR - wL) < 0.0)
                    {
                        continue;
                    }
                }
                double const gain =
                    score(s.gL, s.hL, l1, l2) + score(s.gR, s.hR, l1, l2) - node_score;
                if (gain > 0.0 && gain >= min_gain &&
                    feat_better(gain, static_cast<int>(b), dl, 1, best_gain, best_bin,
                                best_dl, best_valid))
                {
                    best_gain  = gain;
                    best_bin   = static_cast<int32_t>(b);
                    best_dl    = dl;
                    best_valid = 1;
                    bgL = s.gL, bhL = s.hL, bgR = s.gR, bhR = s.hR;
                }
            }
        }
    }

    for (int off = 16; off > 0; off >>= 1)
    {
        double const og  = __shfl_down_sync(0xffffffffU, best_gain, off);
        int const    ob  = __shfl_down_sync(0xffffffffU, best_bin, off);
        int const    od  = __shfl_down_sync(0xffffffffU, best_dl, off);
        int const    ov  = __shfl_down_sync(0xffffffffU, best_valid, off);
        double const ogL = __shfl_down_sync(0xffffffffU, bgL, off);
        double const ohL = __shfl_down_sync(0xffffffffU, bhL, off);
        double const ogR = __shfl_down_sync(0xffffffffU, bgR, off);
        double const ohR = __shfl_down_sync(0xffffffffU, bhR, off);
        if (feat_better(og, ob, od, ov, best_gain, best_bin, best_dl, best_valid))
        {
            best_gain = og, best_bin = ob, best_dl = od, best_valid = ov;
            bgL = ogL, bhL = ohL, bgR = ogR, bhR = ohR;
        }
    }
    if (lane == 0 && best_valid != 0)
    {
        out[oidx] = {.gain  = best_gain,
                     .gL    = bgL,
                     .hL    = bhL,
                     .gR    = bgR,
                     .hR    = bhR,
                     .bin   = best_bin,
                     .dl    = best_dl,
                     .valid = 1,
                     .sel   = static_cast<int32_t>(sel)};
    }
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

__global__ void level_find_kernel(hist_int_t *hists, hist_int_t const *parents,
                                  SiblingDerive const *derive, uint32_t const *features,
                                  uint32_t const *n_bins, double const *node_sums,
                                  uint32_t n_sel, uint32_t n_nodes, uint32_t stride,
                                  double l1, double l2, double min_child_hess,
                                  double min_gain, double *score_scratch,
                                  FeatBest *out_feat, GhQuant const *quant)
{
    uint32_t const max_cut    = stride / 2;
    uint32_t const f          = blockIdx.x;
    uint32_t const lane       = threadIdx.x;
    uint32_t const fid        = features[f];
    uint32_t const nb         = n_bins[fid];
    double        *s_score[2] = {score_scratch + (static_cast<size_t>(f) * 2 * max_cut),
                                 score_scratch + ((static_cast<size_t>(f) * 2 + 1) * max_cut)};
    double         parent_sum = 0.0;

    derive_level_strips(hists, parents, derive, n_nodes, n_sel, f, stride);
    if (lane == 0)
    {
        out_feat[f] = FeatBest{};
    }
    if (nb < 2)
    {
        return;
    }
    uint32_t const n_cut = nb - 2;

    bool all_missing_empty = true;
    for (uint32_t base = 0; base < n_nodes; base += 32)
    {
        uint32_t const p     = base + lane;
        bool           empty = true;
        if (p < n_nodes)
        {
            hist_int_t const *cells =
                hists + ((static_cast<size_t>(p) * n_sel + f) * stride);
            empty = cells[pair_off(nb - 1)] == 0 && cells[pair_off(nb - 1) + 1] == 0;
        }
        all_missing_empty =
            all_missing_empty && (__all_sync(0xffffffffU, empty ? 1 : 0) != 0);
    }
    int const n_dirs = all_missing_empty ? 1 : 2;

    for (uint32_t b = lane; b < n_cut; b += 32)
    {
        for (int d = 0; d < n_dirs; ++d)
        {
            s_score[1 - d][b] = 0.0;
        }
    }
    __syncwarp();

    double2 const inv = quant->inv;
    for (uint32_t base = 0; base < n_nodes; base += 32)
    {
        uint32_t const    p      = base + lane;
        bool const        active = p < n_nodes;
        hist_int_t const *cells =
            active ? hists + ((static_cast<size_t>(p) * n_sel + f) * stride) : nullptr;
        double const g = active ? node_sums[pair_off(p)] : 0.0;
        double const h = active ? node_sums[pair_off(p) + 1] : 0.0;
        double const miss_g =
            active ? static_cast<double>(cells[pair_off(nb - 1)]) * inv.x : 0.0;
        double const miss_h =
            active ? static_cast<double>(cells[pair_off(nb - 1) + 1]) * inv.y : 0.0;
        double const real_g  = g - miss_g;
        double const real_h  = h - miss_h;
        double const node_ps = active ? score(g, h, l1, l2) : 0.0;
        parent_sum += warp_sum(node_ps);

        hist_int_t pq_g = 0;
        hist_int_t pq_h = 0;
        for (uint32_t b = 0; b < n_cut; ++b)
        {
            if (active)
            {
                pq_g += cells[pair_off(b)];
                pq_h += cells[pair_off(b) + 1];
            }
            double const pg = static_cast<double>(pq_g) * inv.x;
            double const ph = static_cast<double>(pq_h) * inv.y;
            for (int d = 0; d < n_dirs; ++d)
            {
                int const  dl = 1 - d;
                auto const s =
                    split_sums_dev(pg, ph, miss_g, miss_h, real_g, real_h, dl);
                // perf: An infeasible node does NOT veto the whole level
                // candidate. It contributes its parent score (zero gain) and
                // the broadcast split still applies to it. At depth >= 5 some
                // frontier node is always near-empty, so vetoing rejected every
                // good deep cut and GPU levelwise trailed its own CPU grower
                // (and catboost) at scale.
                bool const   ok = s.hL >= min_child_hess && s.hR >= min_child_hess;
                double const cs =
                    !active
                        ? 0.0
                        : (ok ? score(s.gL, s.hL, l1, l2) + score(s.gR, s.hR, l1, l2)
                              : node_ps);
                double const sum = warp_sum(cs);
                if (lane == 0)
                {
                    s_score[dl][b] += sum;
                }
            }
        }
        __syncwarp();
    }

    if (lane == 0)
    {
        FeatBest best = {};
        for (uint32_t b = 0; b < n_cut; ++b)
        {
            for (int d = 0; d < n_dirs; ++d)
            {
                int const    dl   = 1 - d;
                double const gain = s_score[dl][b] - parent_sum;
                if (gain > best.gain && gain >= min_gain)
                {
                    best = {.gain  = gain,
                            .gL    = 0,
                            .hL    = 0,
                            .gR    = 0,
                            .hR    = 0,
                            .bin   = static_cast<int32_t>(b),
                            .dl    = dl,
                            .valid = 1,
                            .sel   = static_cast<int32_t>(f)};
                }
            }
        }
        out_feat[f] = best;
    }
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
