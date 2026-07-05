#pragma once

// Owning device allocation + error check, extracted from
// histogram_builder.cu for readability. Injected into bonsai's anonymous
// namespace: this header is included by exactly that one TU.

#include <cuda.h>

#include <cstddef>
#include <stdexcept>
#include <string>

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

// Owning device allocation. Grow-only, geometric; contents dropped on
// reallocation (callers re-upload per use).
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

    T *get() const
    {
        return ptr_;
    }

    void ensure(size_t needed)
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
        ensure(n);
        check(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }

  private:
    T     *ptr_      = nullptr;
    size_t capacity_ = 0;
};

} // namespace
} // namespace bonsai
