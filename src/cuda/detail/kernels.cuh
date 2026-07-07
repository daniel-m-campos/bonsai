#pragma once

// Device kernels and their PODs/constants, extracted from
// histogram_engine.cu for readability (docs/architecture/10-cuda.md). Still
// one translation unit: this header is included only by that .cu, inside
// bonsai's anonymous namespace. The scan math (score, bounded_leaf_weight)
// comes from split.hpp and is constexpr, hence device-callable.

#include <algorithm>
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
// engine instead of failing the kernel launch at runtime.
constexpr size_t k_max_shared_bytes = 48UL * 1024UL;

// Widened index of the first (grad) slot of pair i in a flat [grad0, hess0,
// grad1, hess1, ...] array; the hess slot is pair_off(i) + 1.
constexpr __device__ size_t pair_off(uint32_t i)
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

// A 1-D "one thread per element" launch: grid covers n elements in 256-thread
// blocks, capped so tiny inputs don't over-subscribe. The two wrappers below
// borrow thrust's free-function names (interleave ~ transform, gather) so call
// sites read as algorithms; they are just the launch boilerplate for the
// kernels above kept in one place instead of re-spelled per call site.
inline void interleave(float const *grad, float const *hess, uint32_t n, float2 *gh)
{
    interleave_kernel<<<dim3(std::clamp<uint32_t>(n / 256, 1, 1024)), dim3(256)>>>(
        grad, hess, n, gh);
    check(cudaGetLastError(), "interleave launch");
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

// gh_ordered[k] = gh[rows[k]] for k in [0, n) — a device gather.
inline void gather(float2 const *gh, uint32_t const *rows, uint32_t n,
                   float2 *gh_ordered)
{
    gather_gh_kernel<<<dim3(std::clamp<uint32_t>(n / 256, 1, 512)), dim3(256)>>>(
        gh, rows, n, gh_ordered);
    check(cudaGetLastError(), "gather launch");
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
// True when candidate a beats b under the serial scan's tie-break: higher
// gain, then lower bin, then default_left first (dl 1 before 0). Ties never
// replace in the serial `gain > best` loop, so equal gains keep the earlier.
inline __device__ bool feat_better(double ga, int ba, int da, int va, double gb, int bb,
                                   int db, int vb)
{
    if (va != vb)
    {
        return va > vb; // a valid, b not -> a wins
    }
    if (va == 0)
    {
        return false; // both invalid
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

// One warp per (node, feature). The 32 lanes cooperate on the <= 254-bin cut
// scan: a warp-tiled inclusive prefix sum builds each bin's left grad/hess,
// every lane scores its own bins, and a warp reduce picks the winner with the
// same (max gain, then lowest bin, then default_left) tie-break as the serial
// CPU scan. The tiled summation order differs, so results are tolerance-equal
// (docs/architecture/11-gpu-resident.md), not bit-equal.
__global__ void find_kernel(double const *hists, uint32_t const *features,
                            uint32_t const *n_bins, double const *node_sums,
                            double const *node_bounds, char const *allowed,
                            int const *monotone, uint32_t n_sel, uint32_t stride,
                            double l1, double l2, double min_child_hess,
                            double min_gain, FeatBest *out)
{
    uint32_t const node = blockIdx.y;
    uint32_t const sel  = blockIdx.x;
    uint32_t const lane = threadIdx.x; // 0..31
    if (sel >= n_sel)
    {
        return;
    }
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
        return; // no cut cells (degenerate feature)
    }
    double const  *cells      = hists + (oidx * stride);
    double const   g_total    = node_sums[pair_off(node)];
    double const   h_total    = node_sums[pair_off(node) + 1];
    double const   miss_g     = cells[pair_off(nb - 1)];
    double const   miss_h     = cells[pair_off(nb - 1) + 1];
    double const   node_score = score(g_total, h_total, l1, l2);
    double const   real_grad  = g_total - miss_g;
    double const   real_hess  = h_total - miss_h;
    double const   lo         = node_bounds[pair_off(node)];
    double const   hi         = node_bounds[pair_off(node) + 1];
    int const      mc         = monotone[f];
    uint32_t const n_cut      = nb - 2; // cut cells are bins [0, nb-2)

    double  best_gain = 0.0;
    int32_t best_bin = 0, best_dl = 0, best_valid = 0;
    double  bgL = 0, bhL = 0, bgR = 0, bhR = 0;

    // Running inclusive prefix carried across 32-bin tiles.
    double carry_g = 0.0;
    double carry_h = 0.0;
    for (uint32_t base = 0; base < n_cut; base += 32)
    {
        uint32_t const b  = base + lane;
        double         vg = (b < n_cut) ? cells[pair_off(b)] : 0.0;
        double         vh = (b < n_cut) ? cells[pair_off(b) + 1] : 0.0;
        // Warp inclusive scan (shuffle-up) of this tile's grad/hess.
        double sg  = vg;
        double sh_ = vh;
        for (int off = 1; off < 32; off <<= 1)
        {
            double ng = __shfl_up_sync(0xffffffffU, sg, off);
            double nh = __shfl_up_sync(0xffffffffU, sh_, off);
            if (lane >= static_cast<uint32_t>(off))
            {
                sg += ng;
                sh_ += nh;
            }
        }
        double const pg = carry_g + sg; // inclusive prefix through bin b
        double const ph = carry_h + sh_;
        carry_g += __shfl_sync(0xffffffffU, sg, 31);
        carry_h += __shfl_sync(0xffffffffU, sh_, 31);

        if (b < n_cut)
        {
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
                if (gain >= min_gain &&
                    feat_better(gain, static_cast<int>(b), dl, 1, best_gain, best_bin,
                                best_dl, best_valid))
                {
                    best_gain  = gain;
                    best_bin   = static_cast<int32_t>(b);
                    best_dl    = dl;
                    best_valid = 1;
                    bgL = gL, bhL = hL, bgR = gR, bhR = hR;
                }
            }
        }
    }

    // Warp reduce to the winning candidate under the same tie-break.
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

// Oblivious level-find: one split for the whole frontier. For feature f, the
// gain of a cut (bin, default_left) is sum over ALL frontier nodes of
// score(left)+score(right) minus sum of node scores, and the cut must be
// feasible (min_child_hess) for every node. One warp per feature; each lane
// owns nodes {lane, lane+32, ...} (up to 32*MAXK), carrying their prefixes in
// registers and reducing across lanes with shuffles. Mirrors
// update_best_for_feature_for_level in split.cpp (tolerance-equal).
__global__ void level_find_kernel(double const *hists, uint32_t const *features,
                                  uint32_t const *n_bins, double const *node_sums,
                                  uint32_t n_sel, uint32_t n_nodes, uint32_t stride,
                                  double l1, double l2, double min_child_hess,
                                  double min_gain, FeatBest *out_feat)
{
    constexpr int  MAXK = 8; // 32 * 8 = up to 256 frontier nodes (depth 8)
    uint32_t const f    = blockIdx.x;
    uint32_t const lane = threadIdx.x;
    uint32_t const fid  = features[f];
    uint32_t const nb   = n_bins[fid];
    if (lane == 0)
    {
        out_feat[f] = FeatBest{};
    }
    if (nb < 2)
    {
        return;
    }
    uint32_t const n_cut = nb - 2;

    double mg[MAXK], mh[MAXK], rg[MAXK], rh[MAXK], pg[MAXK], ph[MAXK];
    int    kcnt        = 0;
    double lane_parent = 0.0;
    for (uint32_t p = lane; p < n_nodes && kcnt < MAXK; p += 32, ++kcnt)
    {
        double const *cells = hists + ((static_cast<size_t>(p) * n_sel + f) * stride);
        double const  g     = node_sums[pair_off(p)];
        double const  h     = node_sums[pair_off(p) + 1];
        mg[kcnt]            = cells[pair_off(nb - 1)];
        mh[kcnt]            = cells[pair_off(nb - 1) + 1];
        rg[kcnt]            = g - mg[kcnt];
        rh[kcnt]            = h - mh[kcnt];
        pg[kcnt]            = 0.0;
        ph[kcnt]            = 0.0;
        lane_parent += score(g, h, l1, l2);
    }
    for (int o = 16; o > 0; o >>= 1)
    {
        lane_parent += __shfl_down_sync(0xffffffffU, lane_parent, o);
    }
    double const sum_parent = __shfl_sync(0xffffffffU, lane_parent, 0);

    FeatBest best = {};
    for (uint32_t b = 0; b < n_cut; ++b)
    {
        double csL = 0.0, csR = 0.0;
        int    feasL = 1, feasR = 1;
        for (int k = 0; k < kcnt; ++k)
        {
            uint32_t const p = lane + (32U * static_cast<uint32_t>(k));
            double const  *cells =
                hists + ((static_cast<size_t>(p) * n_sel + f) * stride);
            pg[k] += cells[pair_off(b)];
            ph[k] += cells[pair_off(b) + 1];
            double const gL1 = pg[k] + mg[k], hL1 = ph[k] + mh[k];
            double const gR1 = rg[k] - pg[k], hR1 = rh[k] - ph[k];
            if (hL1 < min_child_hess || hR1 < min_child_hess)
            {
                feasL = 0;
            }
            else
            {
                csL += score(gL1, hL1, l1, l2) + score(gR1, hR1, l1, l2);
            }
            double const gL0 = pg[k], hL0 = ph[k];
            double const gR0 = (rg[k] - pg[k]) + mg[k], hR0 = (rh[k] - ph[k]) + mh[k];
            if (hL0 < min_child_hess || hR0 < min_child_hess)
            {
                feasR = 0;
            }
            else
            {
                csR += score(gL0, hL0, l1, l2) + score(gR0, hR0, l1, l2);
            }
        }
        for (int o = 16; o > 0; o >>= 1)
        {
            csL += __shfl_down_sync(0xffffffffU, csL, o);
            csR += __shfl_down_sync(0xffffffffU, csR, o);
            feasL &= __shfl_down_sync(0xffffffffU, feasL, o);
            feasR &= __shfl_down_sync(0xffffffffU, feasR, o);
        }
        if (lane == 0)
        {
            if (feasL != 0)
            {
                double const g = csL - sum_parent;
                if (g > best.gain && g >= min_gain)
                {
                    best = {.gain  = g,
                            .gL    = 0,
                            .hL    = 0,
                            .gR    = 0,
                            .hR    = 0,
                            .bin   = static_cast<int32_t>(b),
                            .dl    = 1,
                            .valid = 1,
                            .sel   = static_cast<int32_t>(f)};
                }
            }
            if (feasR != 0)
            {
                double const g = csR - sum_parent;
                if (g > best.gain && g >= min_gain)
                {
                    best = {.gain  = g,
                            .gL    = 0,
                            .hL    = 0,
                            .gR    = 0,
                            .hR    = 0,
                            .bin   = static_cast<int32_t>(b),
                            .dl    = 0,
                            .valid = 1,
                            .sel   = static_cast<int32_t>(f)};
                }
            }
        }
    }
    if (lane == 0)
    {
        out_feat[f] = best;
    }
}

// Given the oblivious level split (feature index sel, bin, default_left),
// each thread computes one node's (left, right) child sums from its device
// histogram — 4 doubles per node [gL, hL, gR, hR] — so the host can fill the
// children's SplitInput.sums (device histograms aren't host-scannable).
__global__ void level_child_sums_kernel(double const *hists, double const *node_sums,
                                        uint32_t sel, uint32_t bin, int dl,
                                        uint32_t n_nodes, uint32_t n_sel,
                                        uint32_t stride, uint32_t nb, double *out4)
{
    uint32_t const p = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (p >= n_nodes)
    {
        return;
    }
    double const *cells  = hists + ((static_cast<size_t>(p) * n_sel + sel) * stride);
    double const  g      = node_sums[pair_off(p)];
    double const  h      = node_sums[pair_off(p) + 1];
    double const  miss_g = cells[pair_off(nb - 1)];
    double const  miss_h = cells[pair_off(nb - 1) + 1];
    double        pg = 0.0, ph = 0.0;
    for (uint32_t b = 0; b <= bin; ++b)
    {
        pg += cells[pair_off(b)];
        ph += cells[pair_off(b) + 1];
    }
    double const gL   = pg + (dl != 0 ? miss_g : 0.0);
    double const hL   = ph + (dl != 0 ? miss_h : 0.0);
    double const gR   = (g - miss_g - pg) + (dl == 0 ? miss_g : 0.0);
    double const hR   = (h - miss_h - ph) + (dl == 0 ? miss_h : 0.0);
    out4[(4 * p) + 0] = gL;
    out4[(4 * p) + 1] = hL;
    out4[(4 * p) + 2] = gR;
    out4[(4 * p) + 3] = hR;
}

// NOLINTEND(bugprone-easily-swappable-parameters,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,readability-function-cognitive-complexity,readability-identifier-naming)

} // namespace
} // namespace bonsai
