#pragma once

#include "bonsai/row_view.hpp"
#include "bonsai/types.hpp"
#include <cuda.h>

#include <cassert>
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
inline constexpr size_t   k_max_shared_bytes   = 48UL * 1024UL;
inline constexpr uint32_t k_fill_blocks_per_sm = 4;

// perf: Three 32 KiB tile blocks fit a 100 KiB SM, so 512 threads per block
// keep 1536 threads resident where 256 left 768; the 16M-row root fill
// measured 0.68 s at 256 and 0.60 s at 512 per 100 trees on an L40S.
inline constexpr uint32_t k_tile_fill_threads  = 512;
inline constexpr uint32_t k_small_fill_threads = 128;

inline constexpr uint32_t k_bin_tile_width = 8;
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

// sync: Page-locked staging paired with its device mirror: the host-to-device
// half of Staged, with the two properties a per-round staging path needs. The
// host side is pinned and the upload is a cudaMemcpyAsync, so it never
// stream-syncs the way a pageable copy does (that sync drains every kernel
// already queued, which on a plane that stages a handful of scalars per round
// is the round's whole idle). The host side stays live until the copy lands,
// so a caller may only rewrite it after a later blocking copy on the same
// stream has fenced the previous upload. Pageable Staged remains the default
// everywhere that stages per level or per tree, where the sync is amortized
// and the aliasing rule is a hazard.
template <typename T> class PinnedStaged
{
  public:
    PinnedStaged() = default;
    ~PinnedStaged()
    {
        cudaFreeHost(host_);
#ifndef NDEBUG
        if (fence_ != nullptr)
        {
            cudaEventDestroy(fence_);
        }
#endif
    }
    PinnedStaged(PinnedStaged const &)            = delete;
    PinnedStaged &operator=(PinnedStaged const &) = delete;

    T *host() const
    {
        return host_;
    }
    T *device() const
    {
        return dev_.data();
    }

    void reserve(size_t n)
    {
        assert_fenced();
        dev_.reserve(n);
        if (n <= capacity_)
        {
            return;
        }
        cudaFreeHost(host_);
        host_     = nullptr;
        capacity_ = 0;
        check(cudaHostAlloc(&host_, n * sizeof(T), cudaHostAllocDefault), "hostAlloc");
        capacity_ = n;
    }

    void sync(size_t n) const
    {
        assert_fenced();
        check(
            cudaMemcpyAsync(dev_.data(), host_, n * sizeof(T), cudaMemcpyHostToDevice),
            "pinned upload");
#ifndef NDEBUG
        if (fence_ == nullptr)
        {
            check(cudaEventCreateWithFlags(&fence_, cudaEventDisableTiming),
                  "pinned fence event");
        }
        check(cudaEventRecord(fence_), "pinned fence record");
#endif
    }

  private:
    void assert_fenced() const
    {
        assert((fence_ == nullptr || cudaEventQuery(fence_) == cudaSuccess) &&
               "PinnedStaged: host buffer rewritten before the prior upload's "
               "blocking copy fenced it");
    }

    T              *host_     = nullptr;
    size_t          capacity_ = 0;
    DeviceBuffer<T> dev_;
#ifndef NDEBUG
    mutable cudaEvent_t fence_ = nullptr;
#endif
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
