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

// Default shared-memory histogram footprint cap (stride floats, 48 KiB
// static budget). The engine raises it at runtime to the device's opt-in
// limit (~99 KiB on consumer parts, 227 KiB on sm_90), moving the CPU
// fallback cliff from ~3k to ~6k+ bins per feature.
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

// The (left, right) grad/hess of a cut, given the inclusive prefix through
// its bin and the missing cell routed by default_left. Device mirror of
// split_sums_at in src/split.cpp — the single source of truth for
// missing-routing semantics; every find/child-sums kernel calls this.
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

inline __device__ int warp_all(int v)
{
    return __all_sync(0xffffffffU, v != 0) ? 1 : 0;
}

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
                // gain > 0.0 mirrors the CPU's strict `gain > best` with a
                // zero-initialized best: zero-gain cuts never become valid.
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
// gain of a cut (bin, default_left) is the sum over ALL frontier nodes of
// score(left) + score(right) minus the sum of node scores, and the cut must
// be feasible (min_child_hess) for every node. One warp per feature: nodes
// are processed in chunks of 32 (one per lane), each chunk's per-bin child
// scores and feasibility warp-reduce into shared per-bin accumulators, so any
// frontier width works. A final lane-0 scan picks the best (bin, dl). Mirrors
// update_best_for_feature_for_level in split.cpp (tolerance-equal).
__global__ void level_find_kernel(double const *hists, uint32_t const *features,
                                  uint32_t const *n_bins, double const *node_sums,
                                  uint32_t n_sel, uint32_t n_nodes, uint32_t stride,
                                  double l1, double l2, double min_child_hess,
                                  double min_gain, double *score_scratch,
                                  int *feas_scratch, FeatBest *out_feat)
{
    // Per-feature scratch slice [2][max_cut]: bin width can exceed any shared
    // budget on the u16-bin path, and this warp is the slice's only writer.
    uint32_t const max_cut    = stride / 2; // >= n_cut for every feature
    uint32_t const f          = blockIdx.x;
    uint32_t const lane       = threadIdx.x;
    uint32_t const fid        = features[f];
    uint32_t const nb         = n_bins[fid];
    double        *s_score[2] = {score_scratch + (static_cast<size_t>(f) * 2 * max_cut),
                                 score_scratch + ((static_cast<size_t>(f) * 2 + 1) * max_cut)};
    int           *s_feas[2]  = {feas_scratch + (static_cast<size_t>(f) * 2 * max_cut),
                                 feas_scratch + ((static_cast<size_t>(f) * 2 + 1) * max_cut)};
    double         parent_sum = 0.0;

    if (lane == 0)
    {
        out_feat[f] = FeatBest{};
    }
    if (nb < 2)
    {
        return;
    }
    uint32_t const n_cut = nb - 2; // cut cells are bins [0, nb-2)
    for (uint32_t b = lane; b < n_cut; b += 32)
    {
        for (int dl = 0; dl < 2; ++dl)
        {
            s_score[dl][b] = 0.0;
            s_feas[dl][b]  = 1;
        }
    }
    __syncwarp();

    for (uint32_t base = 0; base < n_nodes; base += 32)
    {
        uint32_t const p      = base + lane;
        bool const     active = p < n_nodes;
        double const  *cells =
            active ? hists + ((static_cast<size_t>(p) * n_sel + f) * stride) : nullptr;
        double const g      = active ? node_sums[pair_off(p)] : 0.0;
        double const h      = active ? node_sums[pair_off(p) + 1] : 0.0;
        double const miss_g = active ? cells[pair_off(nb - 1)] : 0.0;
        double const miss_h = active ? cells[pair_off(nb - 1) + 1] : 0.0;
        double const real_g = g - miss_g;
        double const real_h = h - miss_h;
        parent_sum += warp_sum(active ? score(g, h, l1, l2) : 0.0);

        double pg = 0.0;
        double ph = 0.0;
        for (uint32_t b = 0; b < n_cut; ++b)
        {
            if (active)
            {
                pg += cells[pair_off(b)];
                ph += cells[pair_off(b) + 1];
            }
            for (int dl = 1; dl >= 0; --dl)
            {
                auto const s =
                    split_sums_dev(pg, ph, miss_g, miss_h, real_g, real_h, dl);
                bool const ok =
                    !active || (s.hL >= min_child_hess && s.hR >= min_child_hess);
                double const cs   = (active && ok) ? score(s.gL, s.hL, l1, l2) +
                                                       score(s.gR, s.hR, l1, l2)
                                                   : 0.0;
                int const    feas = warp_all(ok ? 1 : 0);
                double const sum  = warp_sum(cs);
                if (lane == 0)
                {
                    s_score[dl][b] += sum;
                    s_feas[dl][b] &= feas;
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
            for (int dl = 1; dl >= 0; --dl)
            {
                if (s_feas[dl][b] == 0)
                {
                    continue;
                }
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

// Given the winning oblivious level split (read from the reduced FeatBest so
// no host round-trip is needed), each thread computes one node's (left,
// right) child sums from its device histogram — 4 doubles per node
// [gL, hL, gR, hR] — so the host can fill the children's SplitInput.sums
// (device histograms aren't host-scannable). Writes zeros when no split won.
__global__ void level_child_sums_kernel(double const *hists, double const *node_sums,
                                        FeatBest const *winner,
                                        uint32_t const *features,
                                        uint32_t const *n_bins, uint32_t n_nodes,
                                        uint32_t n_sel, uint32_t stride, double *out4)
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
    auto const     sel    = static_cast<uint32_t>(b.sel);
    uint32_t const nb     = n_bins[features[sel]];
    double const  *cells  = hists + ((static_cast<size_t>(p) * n_sel + sel) * stride);
    double const   g      = node_sums[pair_off(p)];
    double const   h      = node_sums[pair_off(p) + 1];
    double const   miss_g = cells[pair_off(nb - 1)];
    double const   miss_h = cells[pair_off(nb - 1) + 1];
    double         pg = 0.0, ph = 0.0;
    for (uint32_t bb = 0; bb <= static_cast<uint32_t>(b.bin); ++bb)
    {
        pg += cells[pair_off(bb)];
        ph += cells[pair_off(bb) + 1];
    }
    auto const s = split_sums_dev(pg, ph, miss_g, miss_h, g - miss_g, h - miss_h, b.dl);
    out4[(4 * p) + 0] = s.gL;
    out4[(4 * p) + 1] = s.hL;
    out4[(4 * p) + 2] = s.gR;
    out4[(4 * p) + 3] = s.hR;
}

// --- BONSAI_EXP_DEVICE_GRAD experiment kernels (decision 52 phase A probe;
// throwaway plumbing, not the production API) --------------------------------

__global__ void exp_fill_kernel(float *out, float value, uint32_t n)
{
    uint32_t const r = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (r < n)
    {
        out[r] = value;
    }
}

// Fused post-tree pass: apply the score update from the device row->leaf
// assignment and compute next-iteration MSE gradients in place. hess = 1.
__global__ void exp_update_kernel(float *scores, float const *labels,
                                  uint32_t const *leaf_by_row, float const *node_values,
                                  float lr, float *grad, float *hess, uint32_t n)
{
    uint32_t const r = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (r >= n)
    {
        return;
    }
    float const s = scores[r] + (lr * node_values[leaf_by_row[r]]);
    scores[r]     = s;
    grad[r]       = s - labels[r];
    hess[r]       = 1.0F;
}

// NOLINTEND(bugprone-easily-swappable-parameters,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,readability-function-cognitive-complexity,readability-identifier-naming)

} // namespace
} // namespace bonsai
