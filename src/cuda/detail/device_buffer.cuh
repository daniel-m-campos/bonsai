#pragma once

// Owning device allocation and the host/device staging pair it backs,
// extracted from histogram_builder.cu for readability. Injected into bonsai's
// anonymous namespace: this header is included by exactly that one TU.

#include <cuda.h>

#include <cstddef>
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
        cudaFree(ptr_);
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
        cudaFree(ptr_);
        ptr_      = nullptr;
        capacity_ = 0;
        check(cudaMalloc(&ptr_, grown * sizeof(T)), "malloc");
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

// A host staging vector paired with its device mirror — the shape that recurs
// throughout the builder's Impl. `host` is filled (or received) on the CPU;
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
