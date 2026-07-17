#pragma once

// Owning device allocation and the host/device staging pair it backs, plus the
// small POD types and tuning constants that cross the host/device boundary.
// Shared by the CUDA translation units (the single-GPU histogram_engine.cu and
// the device-context implementation TU; docs/architecture/19-multi-gpu.md):
// everything here has external linkage in namespace bonsai::cuda_detail and
// references no internal-linkage entity, so including it from more than one TU
// is ODR-clean. These are kernel-free RAII/utility templates and PODs; the
// kernels that consume them stay anonymous in kernels.cuh, private per TU.

#include <cuda.h>

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
// limit (~99 KiB on consumer parts, 227 KiB on sm_90), moving the CPU
// fallback cliff from ~3k to ~6k+ bins per feature.
inline constexpr size_t k_max_shared_bytes = 48UL * 1024UL;

// Per-(node, feature) best split. 56-byte POD; dl encodes default_left.
struct FeatBest
{
    double  gain, gL, hL, gR, hR;
    int32_t bin, dl, valid, sel;
};

// Device-side view of one PartitionOp plus its parent segment.
struct PartOpDev
{
    uint32_t ofs, cnt, fid, bin, dl;
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
// churn synchronizes the whole process (the 5090's ~11-14s per-fit overhead,
// decision 48). BONSAI_CUDA_SYNC_ALLOC=1 restores plain cudaMalloc.
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

// A host staging vector paired with its device mirror — the shape that recurs
// throughout the engine's Impl. `host` is filled (or received) on the CPU;
// sync() pushes it to the device, fetch() pulls a device result back. Mirrors
// thrust's host_vector/device_vector duo without the dependency (decision 40:
// the backend stays one self-contained TU).
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
