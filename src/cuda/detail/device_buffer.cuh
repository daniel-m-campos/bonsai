#pragma once

// Owning device allocation and the host/device staging pair it backs, plus the
// small POD types and tuning constants that cross the host/device boundary.
// Shared by the CUDA translation units (the single-GPU histogram_engine.cu and
// the device-context implementation TU):
// everything here has external linkage in namespace bonsai::cuda_detail and
// references no internal-linkage entity, so including it from more than one TU
// is ODR-clean. These are kernel-free RAII/utility templates and PODs; the
// kernels that consume them stay anonymous in kernels.cuh, private per TU.

#include <cuda.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace bonsai
{
namespace cuda_detail
{

// Nodes with fewer rows than this build on the CPU: the kernel launch +
// synchronous copy-back round trip outweighs the histogram work itself
// below roughly this size (knee measured on Jetson Orin Nano).
inline constexpr size_t k_min_gpu_rows = 512;

// Default shared-memory histogram footprint cap (stride floats, 48 KiB
// static budget). The engine raises it at runtime to the device's opt-in
// limit (~99 KiB on consumer parts, 227 KiB on sm_90), moving the bin count
// the device refuses from ~3k to ~6k+ per feature.
inline constexpr size_t k_max_shared_bytes = 48UL * 1024UL;
// Resident 256-thread histogram blocks one SM seats at the tiled shared
// budget; the chunk axis fills the device to sm_count times this.
inline constexpr uint32_t k_fill_blocks_per_sm = 4;

// --- The device bin plane's layout, in one place ----------------------------
// Features are grouped into tiles of k_bin_tile_width. Tile t starts at cell
// n_rows * t * k_bin_tile_width, and one row's strip inside it is
// tile_strip(t) cells wide, so the tile's bin ids for a row are adjacent and
// one memory sector serves 32 / strip rows of a node instead of one. Same
// scheme the host mirror uses (Dataset::row_major_bins), at the width shared
// memory allows. Every reader and writer of the plane goes through
// tiled_cell; nothing else may assume an index expression.
// 8, not the width the depthwise build alone would pick: one plane serves
// both growers, and at 16 a leafwise round, which histograms one node, runs a
// grid too narrow to fill the device and loses more than depthwise gains.
inline constexpr uint32_t k_bin_tile_width = 8;
static_assert((k_bin_tile_width & (k_bin_tile_width - 1)) == 0,
              "the tile width must be a power of two: the index arithmetic divides by "
              "it on every bin read");

// The tail-aware strip width of tile t: the last tile is narrow when the
// feature count is not a multiple of the width.
inline __host__ __device__ uint32_t tile_strip(uint32_t t, uint32_t n_feats)
{
    uint32_t const tail = n_feats - (t * k_bin_tile_width);
    return tail < k_bin_tile_width ? tail : k_bin_tile_width;
}

inline __host__ __device__ uint32_t tile_count(uint32_t n_feats)
{
    return (n_feats + k_bin_tile_width - 1) / k_bin_tile_width;
}

// The cell holding feature f of row r.
inline __host__ __device__ size_t tiled_cell(uint32_t f, uint32_t r, uint32_t n_rows,
                                             uint32_t n_feats)
{
    uint32_t const t = f / k_bin_tile_width;
    return (static_cast<size_t>(n_rows) * t * k_bin_tile_width) +
           (static_cast<size_t>(r) * tile_strip(t, n_feats)) + (f % k_bin_tile_width);
}

// Marks a feature the current tree did not select, in the per-feature slot
// map the tiled histogram kernel reads.
inline constexpr uint32_t k_not_selected = 0xFFFFFFFFU;

// Per-(node, feature) best split. 56-byte POD; dl encodes default_left.
struct FeatBest
{
    double  gain, gL, hL, gR, hR;
    int32_t bin, dl, valid, sel;
};

// Device-side view of one PartitionOp plus its parent segment.
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

// Stream-ordered allocation with the device mempool told to keep freed
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

// Owning device allocation, shaped after thrust::device_vector's capacity API
// (data/reserve) but deliberately grow-only: capacity never shrinks and
// contents are dropped on reallocation (callers re-upload per use), so no
// resize-time device memset is ever paid.
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

// Page-locked host staging: pinned transfers run at full PCIe rate and never
// bounce through the driver's internal staging copy.
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

// Page-locked staging paired with its device mirror: the host-to-device half of
// Staged, with the two properties a per-round staging path needs. The host side
// is pinned and the upload is asynchronous, so it never stream-syncs the way a
// pageable copy does (that sync drains every kernel already queued, which on a
// plane that stages a handful of scalars per round is the round's whole idle).
// The host side stays live until the copy lands, so a caller may only rewrite
// it after a later blocking copy on the same stream has fenced the previous
// upload. Pageable Staged remains the default everywhere that stages per level
// or per tree, where the sync is amortized and the aliasing rule is a hazard.
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

    // Host -> device, asynchronous on the null stream. reserve(n) first.
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
    // Debug-only: the aliasing contract above (host_ may only be rewritten
    // once the prior upload has fenced) is checked, never enforced in
    // release. assert() discards its argument under NDEBUG, so fence_ (also
    // debug-only) is never referenced by a release build.
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

// A host staging vector paired with its device mirror — the shape that recurs
// throughout the engine's Impl. `host` is filled (or received) on the CPU;
// sync() pushes it to the device, fetch() pulls a device result back. Mirrors
// thrust's host_vector/device_vector duo without the dependency (the backend
// stays one self-contained TU).
template <typename T> struct Staged
{
    std::vector<T>  host;
    DeviceBuffer<T> dev;

    // Host -> device: grow the mirror and upload the whole staging vector.
    void sync()
    {
        dev.upload(host.data(), host.size());
    }
    // Device -> host: size the staging vector to n and copy the result back
    // (implicitly synchronizes, like every DtoH copy in this backend).
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
