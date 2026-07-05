#pragma once

// Device kernels and their PODs/constants, extracted from
// histogram_builder.cu for readability (docs/architecture/10-cuda.md). Still
// one translation unit: this header is included only by that .cu, inside
// bonsai's anonymous namespace. The scan math (score, bounded_leaf_weight)
// comes from split.hpp and is constexpr, hence device-callable.

#include <cstddef>
#include <cstdint>
#include <cuda.h>

#include <vector_types.h>

#include "bonsai/split.hpp"

namespace bonsai
{
namespace
{

// NOLINTBEGIN(bugprone-easily-swappable-parameters,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,readability-function-cognitive-complexity,readability-identifier-naming)

// Nodes with fewer rows than this build on the CPU: the kernel launch +
// synchronous copy-back round trip outweighs the histogram work itself
// below roughly this size (knee measured on Jetson Orin Nano).
constexpr size_t k_min_gpu_rows = 512;

// Shared-memory histogram footprint cap (stride floats, 48 KiB/block
// budget). Datasets binned past ~6k bins per feature fall back to the CPU
// builder instead of failing the kernel launch at runtime.
constexpr size_t k_max_shared_bytes = 48UL * 1024UL;

// Widened index of the first (grad) slot of pair i in a flat [grad0, hess0,
// grad1, hess1, ...] array; the hess slot is pair_off(i) + 1.
__device__ constexpr size_t pair_off(uint32_t i)
{
    return 2 * static_cast<size_t>(i);
}

// Interleaves the raw grad/hess uploads into float2 pairs on the device,
// replacing the serial per-tree host pack.
__global__ void interleave_kernel(float const *grad, float const *hess, uint32_t n,
                                  float2 *gh)
{
    uint32_t const span = gridDim.x * blockDim.x;
    for (uint32_t r = (blockIdx.x * blockDim.x) + threadIdx.x; r < n; r += span)
    {
        gh[r] = {.x = grad[r], .y = hess[r]};
    }
}

// Reorders (grad, hess) into level order once, so hist_kernel reads them
// sequentially instead of re-gathering per feature.
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

// Grid is (feature, node, row-chunk). Shared accumulation is float
// (native atomics; double atomics CAS-loop), cross-chunk merge is double —
// rounding stays bounded per <= 32k-row chunk.
template <typename BinT>
__global__ void hist_kernel(BinT const *bins, float2 const *gh_ordered,
                            uint32_t const *rows, uint32_t const *row_ofs,
                            uint32_t const *row_cnt, uint32_t const *features,
                            uint32_t const *n_bins, uint32_t n_rows, uint32_t n_sel,
                            double *out, uint32_t stride, uint32_t const *out_slot)
{
    // Two sub-histograms split by warp parity spread atomic contention.
    extern __shared__ float sh[];
    uint32_t const          f    = features[blockIdx.x];
    uint32_t const          node = blockIdx.y;
    uint32_t const          nb   = n_bins[f];
    for (uint32_t i = threadIdx.x; i < 4 * nb; i += blockDim.x)
    {
        sh[i] = 0.0F;
    }
    __syncthreads();
    float          *my  = sh + (static_cast<size_t>((threadIdx.x >> 5) & 1U) * 2 * nb);
    BinT const     *fb  = bins + (static_cast<size_t>(f) * n_rows);
    uint32_t const  ofs = row_ofs[node];
    uint32_t const *nrows = rows + ofs;
    float2 const   *ngh   = gh_ordered + ofs;
    uint32_t const  cnt   = row_cnt[node];
    uint32_t const  span  = gridDim.z * blockDim.x;
    for (uint32_t k = (blockIdx.z * blockDim.x) + threadIdx.x; k < cnt; k += span)
    {
        uint32_t const b = fb[nrows[k]];
        float2 const   v = ngh[k];
        atomicAdd(&my[pair_off(b)], v.x);
        atomicAdd(&my[pair_off(b) + 1], v.y);
    }
    __syncthreads();
    uint32_t const oslot = out_slot != nullptr ? out_slot[node] : node;
    double *o = out + (((static_cast<size_t>(oslot) * n_sel) + blockIdx.x) * stride);
    for (uint32_t i = threadIdx.x; i < 2 * nb; i += blockDim.x)
    {
        float const v = sh[i] + sh[(2 * nb) + i];
        if (v != 0.0F)
        {
            atomicAdd(&o[i], static_cast<double>(v));
        }
    }
}

// Small nodes skip the shared-memory stage: below ~512 rows the fixed
// per-(node,feature) zero+merge cost dominates, so one block per node
// accumulates row visits straight into the node's global slot in double.
template <typename BinT>
__global__ void hist_small_kernel(BinT const *bins, float2 const *gh_ordered,
                                  uint32_t const *rows, uint32_t const *row_ofs,
                                  uint32_t const *row_cnt, uint32_t const *features,
                                  uint32_t n_rows, uint32_t n_sel, double *out,
                                  uint32_t stride, uint32_t const *out_slot)
{
    uint32_t const  node = blockIdx.x;
    uint32_t const  cnt  = row_cnt[node];
    uint32_t const  ofs  = row_ofs[node];
    uint32_t const *nr   = rows + ofs;
    float2 const   *ngh  = gh_ordered + ofs;
    double         *o    = out + (static_cast<size_t>(out_slot[node]) * n_sel * stride);
    for (uint32_t k = threadIdx.x; k < cnt * n_sel; k += blockDim.x)
    {
        uint32_t const sel = k / cnt;
        uint32_t const i   = k % cnt;
        uint32_t const b = bins[(static_cast<size_t>(features[sel]) * n_rows) + nr[i]];
        float2 const   v = ngh[i];
        atomicAdd(&o[(sel * stride) + (2 * b)], static_cast<double>(v.x));
        atomicAdd(&o[(sel * stride) + (2 * b) + 1], static_cast<double>(v.y));
    }
}

// --- Stage B: device row partitioning (docs/architecture/11-gpu-resident.md).
// Rows live in ping-pong segment buffers; each split routes its parent
// segment into stable left/right child segments via count -> scan -> scatter
// (hand-rolled: no CUB, the TU stays self-contained). CHUNK rows per block,
// each thread owning ROWS_PER_THREAD consecutive rows keeps the scatter
// stable and the scans tiny.
constexpr uint32_t k_part_rows_per_thread = 16;
constexpr uint32_t k_part_block           = 256;
constexpr uint32_t k_part_chunk           = k_part_block * k_part_rows_per_thread;

// Device-side view of one PartitionOp plus its parent segment.
struct PartOpDev
{
    uint32_t ofs, cnt, fid, bin, dl;
};

inline __device__ bool goes_left_dev(uint32_t b, uint32_t last_bin, uint32_t bin,
                                     uint32_t dl)
{
    if (b == last_bin)
    {
        return dl != 0;
    }
    return b <= bin;
}

// Phase 1: per-(op, chunk) left-count; flags cached for the scatter pass.
template <typename BinT>
__global__ void route_count_kernel(BinT const *bins, uint32_t const *n_bins,
                                   uint32_t const *rows, PartOpDev const *ops,
                                   uint32_t n_rows, uint32_t max_chunks, uint8_t *flags,
                                   uint32_t *block_counts)
{
    __shared__ uint32_t sh[k_part_block];
    PartOpDev const     op    = ops[blockIdx.y];
    uint32_t const      chunk = blockIdx.x;
    uint32_t const      base  = chunk * k_part_chunk;
    BinT const         *fb    = bins + (static_cast<size_t>(op.fid) * n_rows);
    uint32_t const      last  = n_bins[op.fid] - 1;
    uint32_t            mine  = 0;
    for (uint32_t j = 0; j < k_part_rows_per_thread; ++j)
    {
        uint32_t const i = base + (threadIdx.x * k_part_rows_per_thread) + j;
        if (i < op.cnt)
        {
            bool const l = goes_left_dev(fb[rows[op.ofs + i]], last, op.bin, op.dl);
            flags[op.ofs + i] = l ? 1 : 0;
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

// Phase 2: exclusive scan of each op's chunk counts; total -> n_left[op].
__global__ void seg_scan_kernel(uint32_t *block_counts, uint32_t max_chunks,
                                uint32_t *n_left)
{
    if (threadIdx.x != 0)
    {
        return;
    }
    uint32_t *c   = block_counts + (static_cast<size_t>(blockIdx.x) * max_chunks);
    uint32_t  run = 0;
    for (uint32_t k = 0; k < max_chunks; ++k)
    {
        uint32_t const v = c[k];
        c[k]             = run;
        run += v;
    }
    n_left[blockIdx.x] = run;
}

// Phase 3: stable scatter into the other rows/gh buffers. Each thread's
// consecutive rows write in order; block and thread bases come from the
// scanned counts, so left keeps ascending order, then right.
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
        mine += (i < op.cnt && flags[op.ofs + i] != 0) ? 1U : 0U;
    }
    sh[threadIdx.x + 1] = mine;
    if (threadIdx.x == 0)
    {
        sh[0] = 0;
    }
    __syncthreads();
    // Hillis-Steele inclusive scan over thread counts -> exclusive bases.
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
    uint32_t const nl_total = n_left[blockIdx.y];
    uint32_t const block_lefts =
        block_counts[(static_cast<size_t>(blockIdx.y) * max_chunks) + chunk];
    uint32_t lefts  = block_lefts + sh[threadIdx.x];
    uint32_t before = base + (threadIdx.x * k_part_rows_per_thread);
    for (uint32_t j = 0; j < k_part_rows_per_thread; ++j)
    {
        uint32_t const i = base + (threadIdx.x * k_part_rows_per_thread) + j;
        if (i >= op.cnt)
        {
            break;
        }
        uint32_t dst = 0;
        if (flags[op.ofs + i] != 0)
        {
            dst = op.ofs + lefts;
            ++lefts;
        }
        else
        {
            dst = op.ofs + nl_total + (before + j - lefts);
        }
        rows_out[dst] = rows_in[op.ofs + i];
        gh_out[dst]   = gh_in[op.ofs + i];
    }
}

// Records each segment row's final leaf id (persistent per-row array).
__global__ void stamp_kernel(uint32_t const *rows, PartOpDev const *segs,
                             uint32_t const *node_ids, uint32_t *leaf_by_row)
{
    PartOpDev const seg = segs[blockIdx.x];
    uint32_t const  id  = node_ids[blockIdx.x];
    for (uint32_t i = threadIdx.x; i < seg.cnt; i += blockDim.x)
    {
        leaf_by_row[rows[seg.ofs + i]] = id;
    }
}

// Larger children derive on-device: child[large] = parent - child[small].
// Slot triples are (parent, small, large); slot_doubles is one slot's span.
__global__ void subtract_kernel(double const *parents, double *children,
                                uint32_t const *triples, uint32_t slot_doubles)
{
    uint32_t const *t     = triples + (3UL * blockIdx.y);
    uint32_t const  span  = gridDim.x * blockDim.x;
    double const   *par   = parents + (static_cast<size_t>(t[0]) * slot_doubles);
    double const   *small = children + (static_cast<size_t>(t[1]) * slot_doubles);
    double         *large = children + (static_cast<size_t>(t[2]) * slot_doubles);
    for (uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < slot_doubles;
         i += span)
    {
        large[i] = par[i] - small[i];
    }
}

// Per-(node, feature) best split. 56-byte POD; dl encodes default_left.
struct FeatBest
{
    double  gain, gL, hL, gR, hR;
    int32_t bin, dl, valid, sel;
};

// One thread walks one (node, selected-feature) histogram sequentially,
// replicating the CPU scan in split.cpp exactly: same prefix order, both
// default_left routings, min_child_hess, monotone feasibility, min_gain.
// The scan is <= 254 iterations; clarity beats occupancy here.
__global__ void find_kernel(double const *hists, uint32_t const *features,
                            uint32_t const *n_bins, double const *node_sums,
                            double const *node_bounds, char const *allowed,
                            int const *monotone, uint32_t n_sel, uint32_t stride,
                            double l1, double l2, double min_child_hess,
                            double min_gain, FeatBest *out)
{
    uint32_t const node = blockIdx.y;
    uint32_t const sel  = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (sel >= n_sel)
    {
        return;
    }
    FeatBest best                                  = {};
    out[(static_cast<size_t>(node) * n_sel) + sel] = best;
    if (allowed != nullptr && allowed[(static_cast<size_t>(node) * n_sel) + sel] == 0)
    {
        return;
    }
    uint32_t const f  = features[sel];
    uint32_t const nb = n_bins[f];
    if (nb < 2)
    {
        return; // no cut cells (degenerate feature)
    }
    double const *cells =
        hists + (((static_cast<size_t>(node) * n_sel) + sel) * stride);
    double const g_total    = node_sums[pair_off(node)];
    double const h_total    = node_sums[pair_off(node) + 1];
    double const miss_g     = cells[pair_off(nb - 1)];
    double const miss_h     = cells[pair_off(nb - 1) + 1];
    double const node_score = score(g_total, h_total, l1, l2);
    double const real_grad  = g_total - miss_g;
    double const real_hess  = h_total - miss_h;
    double const lo         = node_bounds[pair_off(node)];
    double const hi         = node_bounds[pair_off(node) + 1];
    int const    mc         = monotone[f];

    // Cut cells are bins [0, nb-2): the last real bin cannot split and the
    // final cell is the missing bin (mirrors Histogram::cut_cells()).
    double pg = 0.0;
    double ph = 0.0;
    for (uint32_t b = 0; b + 2 < nb; ++b)
    {
        pg += cells[pair_off(b)];
        ph += cells[pair_off(b) + 1];
        for (int dl = 1; dl >= 0; --dl)
        {
            double const gL = pg + (dl != 0 ? miss_g : 0.0);
            double const hL = ph + (dl != 0 ? miss_h : 0.0);
            double const gR = (real_grad - pg) + (dl == 0 ? miss_g : 0.0);
            double const hR = (real_hess - ph) + (dl == 0 ? miss_h : 0.0);
            if (hL < min_child_hess || hR < min_child_hess)
            {
                continue;
            }
            if (mc != 0)
            {
                double const wL = bounded_leaf_weight(gL, hL, l1, l2, lo, hi);
                double const wR = bounded_leaf_weight(gR, hR, l1, l2, lo, hi);
                if (static_cast<double>(mc) * (wR - wL) < 0.0)
                {
                    continue;
                }
            }
            double const gain =
                score(gL, hL, l1, l2) + score(gR, hR, l1, l2) - node_score;
            if (gain > best.gain && gain >= min_gain)
            {
                best = {.gain  = gain,
                        .gL    = gL,
                        .hL    = hL,
                        .gR    = gR,
                        .hR    = hR,
                        .bin   = static_cast<int32_t>(b),
                        .dl    = dl,
                        .valid = 1,
                        .sel   = static_cast<int32_t>(sel)};
            }
        }
    }
    out[(static_cast<size_t>(node) * n_sel) + sel] = best;
}

// Per-node winner in ascending selected-feature order with strict >,
// matching reduce_in_feature_order's lowest-feature-id tie-break.
__global__ void reduce_kernel(FeatBest const *per_feat, uint32_t n_sel, FeatBest *out)
{
    if (threadIdx.x != 0)
    {
        return;
    }
    uint32_t const  node = blockIdx.x;
    FeatBest        best = {};
    FeatBest const *row  = per_feat + (static_cast<size_t>(node) * n_sel);
    for (uint32_t s = 0; s < n_sel; ++s)
    {
        if (row[s].valid != 0 && row[s].gain > best.gain)
        {
            best = row[s];
        }
    }
    out[node] = best;
}

// NOLINTEND(bugprone-easily-swappable-parameters,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,readability-function-cognitive-complexity,readability-identifier-naming)

} // namespace
} // namespace bonsai
