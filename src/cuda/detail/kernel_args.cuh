#pragma once

#include <cuda.h>

#include <vector_types.h>

#include <cstddef>
#include <cstdint>

namespace bonsai
{
namespace cuda_detail
{

// perf: Nodes with fewer rows than this take hist_small_kernel, which
// accumulates straight into the node's global slot: below roughly this size
// the per-(node, feature) shared-memory zero and merge dominates the
// histogram work. The 2026-08-17 sweep measured every cutoff above 512
// worse at every cell.
inline constexpr size_t k_min_gpu_rows = 512;

using hist_int_t = long long;

struct GhQuant
{
    float2  scale;
    double2 inv;
    float2  inv_f;
};

struct Interval
{
    float lo, hi;
};

struct CutConst
{
    double l1, l2, min_child_hess, min_gain;
};

struct ScreenConst
{
    Interval l1, l2, min_child_hess, min_gain;
};

struct NodeScreen
{
    double   score;
    Interval score_bounds, sum_grad, sum_hess;
};

struct NodeRows
{
    uint32_t const *rows;
    float2 const   *gh;
    uint32_t        count;
};

inline constexpr size_t hist_shared_bytes(size_t max_bins)
{
    return 2 * max_bins * sizeof(hist_int_t);
}

// perf: Default shared-memory histogram footprint cap (stride int64 cells,
// 48 KiB static budget). The engine raises it at runtime to the device's
// opt-in limit (~99 KiB on consumer parts, 227 KiB on sm_90), moving the bin
// count the device refuses from ~3k to ~6k+ per feature.
inline constexpr size_t   k_max_shared_bytes     = 48UL * 1024UL;
inline constexpr uint32_t k_fill_blocks_per_sm   = 4;
inline constexpr uint32_t k_derive_blocks_per_sm = 4;
inline constexpr uint32_t k_level_find_threads   = 256;
inline constexpr uint32_t k_level_find_warps     = k_level_find_threads / 32;

// perf: A 16-feature tile of 255-bin planes is 64 KiB, one block per 100 KiB
// SM, so that block carries every warp the SM holds: 1024 threads under the
// launch bound (64 registers, no spills). Same-pod RTX PRO 6000 train over
// 100 trees at 16M x 128: 3.32 s at width 8 x 512 threads, 3.09 s at width
// 16 x 512, 2.90 s at 16 x 768, 2.88 s at 16 x 1024; the wider strip costs
// the partition kernels 0.15 s of that, two rows per 32-byte sector, not four.
inline constexpr uint32_t k_tile_fill_threads  = 1024;
inline constexpr uint32_t k_small_fill_threads = 128;

inline constexpr uint32_t k_bin_tile_width   = 16;
inline constexpr uint32_t k_plane_group      = 32;
inline constexpr uint32_t k_plane_group_mask = k_plane_group - 1;
static_assert((k_bin_tile_width & (k_bin_tile_width - 1)) == 0,
              "the tile width must be a power of two: the index arithmetic divides by "
              "it on every bin read");

inline __host__ __device__ uint32_t tile_strip(uint32_t t, uint32_t n_feats)
{
    uint32_t const tail = n_feats - (t * k_bin_tile_width);
    return tail < k_bin_tile_width ? tail : k_bin_tile_width;
}

inline __host__ __device__ uint32_t tile_count(uint32_t n_feats)
{
    return (n_feats + k_bin_tile_width - 1) / k_bin_tile_width;
}

inline __host__ __device__ uint32_t tile_stride(uint32_t stride)
{
    return (stride + (2 * k_plane_group) - 1) & ~((2 * k_plane_group) - 1);
}

inline __host__ __device__ size_t tiled_cell(uint32_t f, uint32_t r, uint32_t n_rows,
                                             uint32_t n_feats)
{
    uint32_t const t = f / k_bin_tile_width;
    return (static_cast<size_t>(n_rows) * t * k_bin_tile_width) +
           (static_cast<size_t>(r) * tile_strip(t, n_feats)) + (f % k_bin_tile_width);
}

inline __host__ __device__ uint32_t mapped_row(uint32_t const *rows, uint32_t k)
{
    return rows == nullptr ? k : rows[k];
}

inline constexpr uint32_t k_not_selected = 0xFFFFFFFFU;

struct SiblingDerive
{
    uint32_t parent_slot;
    uint32_t small_slot;
};

inline constexpr SiblingDerive k_filled_slot{k_not_selected, k_not_selected};

struct PartOpDev
{
    uint32_t offset, count, fid, bin, dl;
};

struct PartTilesDev
{
    unsigned long long *status;
    uint32_t           *counter;
    uint32_t            base;
    uint32_t            epoch;
};

struct RowSeg
{
    uint32_t offset = 0;
    uint32_t count  = 0;
};

struct BuildSeg
{
    uint32_t offset;
    uint32_t count;
    uint32_t slot;
};

inline constexpr __host__ __device__ bool left_is_small(uint32_t nl, uint32_t count)
{
    return nl <= count - nl;
}

struct SmallChildDev
{
    BuildSeg *seg        = nullptr;
    uint32_t *left_count = nullptr;
    uint32_t  slot       = 0;
};

constexpr __host__ __device__ size_t pair_off(uint32_t i)
{
    return 2 * static_cast<size_t>(i);
}

struct PartOpTable
{
    PartOpDev const *ops;

    __device__ PartOpDev at(uint32_t i) const
    {
        return ops[i];
    }
};

struct PartOpValue
{
    PartOpDev op;

    __device__ PartOpDev at(uint32_t /*i*/) const
    {
        return op;
    }
};

struct FindNodesRef
{
    double const        *sums;
    double const        *bounds;
    NodeScreen const    *screens;
    SiblingDerive const *derives;
    uint32_t const      *slots;

    __device__ uint32_t slot(uint32_t node) const
    {
        return slots != nullptr ? slots[node] : node;
    }
    __device__ SiblingDerive derive(uint32_t node) const
    {
        return derives[node];
    }
    __device__ double2 sum(uint32_t node) const
    {
        return double2{sums[pair_off(node)], sums[pair_off(node) + 1]};
    }
    __device__ NodeScreen screen(uint32_t node) const
    {
        return screens[node];
    }
    __device__ double2 bound(uint32_t node) const
    {
        return double2{bounds[pair_off(node)], bounds[pair_off(node) + 1]};
    }
};

inline constexpr uint32_t k_leaf_find_nodes = 2;

template <typename T>
inline __device__ T pick_node(uint32_t node, T const (&a)[k_leaf_find_nodes])
{
    return node == 0 ? a[0] : a[1];
}

struct LeafNodeStats
{
    double2    sum;
    double2    bound;
    NodeScreen screen;
};

struct LeafRootNodes
{
    LeafNodeStats root;

    __device__ uint32_t slot(uint32_t /*node*/) const
    {
        return 0;
    }
    __device__ SiblingDerive derive(uint32_t /*node*/) const
    {
        return k_filled_slot;
    }
    __device__ double2 sum(uint32_t /*node*/) const
    {
        return root.sum;
    }
    __device__ NodeScreen screen(uint32_t /*node*/) const
    {
        return root.screen;
    }
    __device__ double2 bound(uint32_t /*node*/) const
    {
        return root.bound;
    }
};

struct LeafPairNodes
{
    LeafNodeStats   stats[k_leaf_find_nodes];
    uint32_t const *left_count;
    uint32_t        parent_slot;
    uint32_t        fresh_slot;
    uint32_t        count;

    __device__ bool small(uint32_t node) const
    {
        return (node == 0) == left_is_small(*left_count, count);
    }
    __device__ uint32_t slot(uint32_t node) const
    {
        return small(node) ? fresh_slot : parent_slot;
    }
    __device__ SiblingDerive derive(uint32_t node) const
    {
        return small(node) ? k_filled_slot : SiblingDerive{parent_slot, fresh_slot};
    }
    __device__ double2 sum(uint32_t node) const
    {
        return pick_node(node, stats).sum;
    }
    __device__ NodeScreen screen(uint32_t node) const
    {
        return pick_node(node, stats).screen;
    }
    __device__ double2 bound(uint32_t node) const
    {
        return pick_node(node, stats).bound;
    }
};

} // namespace cuda_detail
} // namespace bonsai
