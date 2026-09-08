#pragma once

#include "bonsai/row_view.hpp"
#include "bonsai/types.hpp"
#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

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

struct FeatBest
{
    double  gain, gL, hL, gR, hR;
    int32_t bin, dl, valid, sel;
};

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

struct SmallChildDev
{
    uint32_t *seg  = nullptr;
    uint32_t  slot = 0;
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

inline __device__ double2 pick_pair(uint32_t node,
                                    double const (&a)[2 * k_leaf_find_nodes])
{
    return node == 0 ? double2{a[0], a[1]} : double2{a[2], a[3]};
}

struct LeafFindNodes
{
    double        sums[2 * k_leaf_find_nodes];
    double        bounds[2 * k_leaf_find_nodes];
    NodeScreen    screens[k_leaf_find_nodes];
    SiblingDerive derives[k_leaf_find_nodes];
    uint32_t      slots[k_leaf_find_nodes];

    __device__ uint32_t slot(uint32_t node) const
    {
        return pick_node(node, slots);
    }
    __device__ SiblingDerive derive(uint32_t node) const
    {
        return pick_node(node, derives);
    }
    __device__ double2 sum(uint32_t node) const
    {
        return pick_pair(node, sums);
    }
    __device__ NodeScreen screen(uint32_t node) const
    {
        return pick_node(node, screens);
    }
    __device__ double2 bound(uint32_t node) const
    {
        return pick_pair(node, bounds);
    }
};

inline void check(cudaError_t rc, char const *what)
{
    if (rc != cudaSuccess)
    {
        throw std::runtime_error(std::string{"cuda: "} + what + ": " +
                                 cudaGetErrorString(rc));
    }
}

// perf: Stream-ordered allocation with the device mempool told to keep freed
// memory: the default release threshold of 0 returns every free to the OS
// at the next sync, and on GeForce drivers the resulting cudaMalloc/cudaFree
// churn synchronizes the whole process (the 5090's ~11-14s per-fit
// overhead). BONSAI_CUDA_SYNC_ALLOC=1 restores plain cudaMalloc.
inline bool use_async_alloc()
{
    static bool const enabled = []
    {
        if (std::getenv("BONSAI_CUDA_SYNC_ALLOC") != nullptr)
        {
            return false;
        }
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess)
        {
            return false;
        }
        int supported = 0;
        if (cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, dev) !=
                cudaSuccess ||
            supported == 0)
        {
            return false;
        }
        cudaMemPool_t pool{};
        if (cudaDeviceGetDefaultMemPool(&pool, dev) != cudaSuccess)
        {
            return false;
        }
        uint64_t threshold = UINT64_MAX;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
        return true;
    }();
    return enabled;
}

inline void *alloc_device(size_t bytes)
{
    void *p = nullptr;
    if (use_async_alloc())
    {
        check(cudaMallocAsync(&p, bytes, cudaStreamDefault), "mallocAsync");
    }
    else
    {
        check(cudaMalloc(&p, bytes), "malloc");
    }
    return p;
}

inline void free_device(void *p)
{
    if (p == nullptr)
    {
        return;
    }
    if (use_async_alloc())
    {
        cudaFreeAsync(p, cudaStreamDefault);
    }
    else
    {
        cudaFree(p);
    }
}

template <typename T> class DeviceBuffer
{
  public:
    DeviceBuffer() = default;
    ~DeviceBuffer()
    {
        free_device(ptr_);
    }
    DeviceBuffer(DeviceBuffer const &)            = delete;
    DeviceBuffer &operator=(DeviceBuffer const &) = delete;
    DeviceBuffer(DeviceBuffer &&)                 = delete;
    DeviceBuffer &operator=(DeviceBuffer &&)      = delete;

    T *data() const
    {
        return ptr_;
    }

    void reserve(size_t needed)
    {
        if (needed <= capacity_)
        {
            return;
        }
        size_t grown = capacity_ == 0 ? needed : capacity_;
        while (grown < needed)
        {
            grown *= 2;
        }
        free_device(ptr_);
        ptr_      = nullptr;
        capacity_ = 0;
        ptr_      = static_cast<T *>(alloc_device(grown * sizeof(T)));
        capacity_ = grown;
    }

    void upload(T const *host, size_t n)
    {
        reserve(n);
        check(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }

  private:
    T     *ptr_      = nullptr;
    size_t capacity_ = 0;
};

class PartTiles
{
  public:
    PartTilesDev arm(size_t status_words, uint32_t n_tiles)
    {
        if (status_words > words_)
        {
            status_.reserve(status_words);
            check(cudaMemset(status_.data(), 0,
                             status_words * sizeof(unsigned long long)),
                  "partition status zero");
            words_ = status_words;
        }
        if (!counter_ready_)
        {
            counter_.reserve(1);
            check(cudaMemset(counter_.data(), 0, sizeof(uint32_t)),
                  "partition counter zero");
            counter_ready_ = true;
        }
        ++epoch_;
        PartTilesDev const dev{status_.data(), counter_.data(), issued_, epoch_};
        issued_ += n_tiles;
        return dev;
    }

  private:
    DeviceBuffer<unsigned long long> status_;
    DeviceBuffer<uint32_t>           counter_;
    size_t                           words_         = 0;
    uint32_t                         issued_        = 0;
    uint32_t                         epoch_         = 0;
    bool                             counter_ready_ = false;
};

struct RowMap
{
    DeviceBuffer<row_id_t> ids;
    size_t                 n        = 0;
    bool                   identity = true;

    RowMap() = default;

    RowMap(std::span<row_id_t const> rows, size_t plane_rows)
    {
        stage(rows, plane_rows);
    }

    void stage(RowView const &view)
    {
        n        = view.size();
        identity = view.is_identity();
        if (!identity)
        {
            std::vector<row_id_t> const materialized = view.materialize();
            ids.upload(materialized.data(), materialized.size());
        }
    }

    void stage(std::span<row_id_t const> rows, size_t plane_rows)
    {
        identity = rows.empty();
        n        = identity ? plane_rows : rows.size();
        if (!identity)
        {
            ids.upload(rows.data(), rows.size());
        }
    }

    row_id_t const *data() const
    {
        return identity ? nullptr : ids.data();
    }
};

template <typename T> class PinnedBuffer
{
  public:
    explicit PinnedBuffer(size_t n)
    {
        check(cudaHostAlloc(&ptr_, n * sizeof(T), cudaHostAllocDefault), "hostAlloc");
    }
    ~PinnedBuffer()
    {
        cudaFreeHost(ptr_);
    }
    PinnedBuffer(PinnedBuffer const &)            = delete;
    PinnedBuffer &operator=(PinnedBuffer const &) = delete;

    T *data() const
    {
        return ptr_;
    }

  private:
    T *ptr_ = nullptr;
};

template <typename T> class MappedBuffer
{
  public:
    MappedBuffer() = default;
    ~MappedBuffer()
    {
        cudaFreeHost(host_);
    }
    MappedBuffer(MappedBuffer const &)            = delete;
    MappedBuffer &operator=(MappedBuffer const &) = delete;

    T *device(size_t n)
    {
        if (n > capacity_)
        {
            cudaFreeHost(host_);
            host_     = nullptr;
            capacity_ = 0;
            check(cudaHostAlloc(&host_, n * sizeof(T), cudaHostAllocMapped),
                  "hostAllocMapped");
            check(cudaHostGetDevicePointer(&dev_, host_, 0), "mapped device pointer");
            capacity_ = n;
        }
        return dev_;
    }
    std::span<T const> host(size_t n) const
    {
        return {host_, n};
    }

  private:
    T     *host_     = nullptr;
    T     *dev_      = nullptr;
    size_t capacity_ = 0;
};

class StreamFence
{
  public:
    StreamFence() = default;
    ~StreamFence()
    {
        if (event_ != nullptr)
        {
            cudaEventDestroy(event_);
        }
    }
    StreamFence(StreamFence const &)            = delete;
    StreamFence &operator=(StreamFence const &) = delete;

    void record()
    {
        if (event_ == nullptr)
        {
            check(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming),
                  "fence event");
        }
        check(cudaEventRecord(event_), "fence record");
    }
    void wait() const
    {
        check(cudaEventSynchronize(event_), "fence wait");
    }

  private:
    cudaEvent_t event_ = nullptr;
};

template <typename T> struct Staged
{
    std::vector<T>  host;
    DeviceBuffer<T> dev;

    void sync()
    {
        dev.upload(host.data(), host.size());
    }
    void fetch(size_t n)
    {
        host.resize(n);
        dev.reserve(n);
        check(
            cudaMemcpy(host.data(), dev.data(), n * sizeof(T), cudaMemcpyDeviceToHost),
            "fetch");
    }
    void reserve(size_t n)
    {
        dev.reserve(n);
    }
    T *device() const
    {
        return dev.data();
    }
    size_t size() const
    {
        return host.size();
    }
    bool empty() const
    {
        return host.empty();
    }
    void clear()
    {
        host.clear();
    }
};

} // namespace cuda_detail
} // namespace bonsai
