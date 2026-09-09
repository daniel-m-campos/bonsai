#pragma once

#include "bonsai/row_view.hpp"
#include "bonsai/types.hpp"
#include "cut_order.hpp"
#include "kernel_args.cuh"
#include "profile.cuh"
#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bonsai
{
namespace cuda_detail
{

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

inline size_t pool_reserved_free_bytes()
{
    if (!use_async_alloc())
    {
        return 0;
    }
    int           dev = 0;
    cudaMemPool_t pool{};
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetDefaultMemPool(&pool, dev) != cudaSuccess)
    {
        return 0;
    }
    cuuint64_t reserved = 0;
    cuuint64_t used     = 0;
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
    return reserved > used ? static_cast<size_t>(reserved - used) : 0;
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
    PartTilesDev arm(uint32_t n_tiles)
    {
        if (n_tiles > words_)
        {
            status_.reserve(n_tiles);
            check(cudaMemset(status_.data(), 0, n_tiles * sizeof(unsigned long long)),
                  "partition status zero");
            words_ = n_tiles;
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
        if (n > capacity_)
        {
            throw std::logic_error("mapped buffer read past its device size");
        }
        return {host_, n};
    }

  private:
    T     *host_     = nullptr;
    T     *dev_      = nullptr;
    size_t capacity_ = 0;
};

class Once
{
  public:
    bool first()
    {
        return !std::exchange(done_, true);
    }

  private:
    bool done_ = false;
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

    void record(cudaStream_t stream = nullptr)
    {
        if (event_ == nullptr)
        {
            check(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming),
                  "fence event");
        }
        check(cudaEventRecord(event_, stream), "fence record");
    }
    void wait() const
    {
        check(cudaEventSynchronize(event_), "fence wait");
    }

  private:
    cudaEvent_t event_ = nullptr;
};

class Stream
{
  public:
    Stream()
    {
        check(cudaStreamCreate(&stream_), "stream create");
    }
    ~Stream()
    {
        cudaStreamDestroy(stream_);
    }
    Stream(Stream const &)            = delete;
    Stream &operator=(Stream const &) = delete;

    cudaStream_t get() const
    {
        return stream_;
    }

  private:
    cudaStream_t stream_ = nullptr;
};

class KernelTimer
{
  public:
    KernelTimer() = default;
    ~KernelTimer()
    {
        for (cudaEvent_t const e : events_)
        {
            if (e != nullptr)
            {
                cudaEventDestroy(e);
            }
        }
    }
    KernelTimer(KernelTimer const &)            = delete;
    KernelTimer &operator=(KernelTimer const &) = delete;

    void begin()
    {
        record(0);
    }
    void end()
    {
        record(1);
    }
    void add_elapsed(double &seconds) const
    {
        if (!enabled_)
        {
            return;
        }
        float ms = 0.0F;
        check(cudaEventElapsedTime(&ms, events_[0], events_[1]),
              "kernel timer elapsed");
        seconds += ms / 1e3;
    }

  private:
    void record(size_t i)
    {
        if (!enabled_)
        {
            return;
        }
        if (events_[i] == nullptr)
        {
            check(cudaEventCreate(&events_[i]), "kernel timer event");
        }
        check(cudaEventRecord(events_[i]), "kernel timer record");
    }

    bool        enabled_   = profile_on();
    cudaEvent_t events_[2] = {nullptr, nullptr};
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
