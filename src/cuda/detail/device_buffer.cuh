#pragma once

// Owning device allocation and the host/device staging pair it backs,
// extracted from histogram_engine.cu for readability. Injected into bonsai's
// anonymous namespace: this header is included by exactly that one TU.

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace bonsai
{
namespace
{

void check(cudaError_t rc, char const *what)
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
// bounce through the driver's internal staging copy. Grow-only like
// DeviceBuffer (contents dropped on reallocation); the sized constructor
// covers the one-shot uses.
template <typename T> class PinnedBuffer
{
  public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t n)
    {
        reserve(n);
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
        cudaFreeHost(ptr_);
        ptr_      = nullptr;
        capacity_ = 0;
        T *fresh  = nullptr;
        check(cudaHostAlloc(&fresh, grown * sizeof(T), cudaHostAllocDefault),
              "hostAlloc");
        ptr_      = fresh;
        capacity_ = grown;
    }

  private:
    T     *ptr_      = nullptr;
    size_t capacity_ = 0;
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

} // namespace
} // namespace bonsai
