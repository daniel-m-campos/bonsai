#pragma once

#include <cstddef>
#include <memory>
#include <new>
#include <utility>

namespace bonsai::detail
{

// Zero-argument construction is a no-op, so the first write to a page homes
// it on the writer's node: only for a buffer a parallel producer writes end
// to end before any read (invariants: untouched-resize-writes-nothing).
template <typename T> struct RawAllocator
{
    using value_type = T;

    static constexpr std::align_val_t alignment{64};

    RawAllocator() = default;
    template <typename U>
    explicit RawAllocator(RawAllocator<U> const & /*other*/) noexcept
    {
    }

    T *allocate(std::size_t n)
    {
        return static_cast<T *>(::operator new(n * sizeof(T), alignment));
    }

    void deallocate(T *p, std::size_t n) noexcept
    {
        ::operator delete(p, n * sizeof(T), alignment);
    }

    template <typename U> void construct(U * /*p*/) noexcept {}

    template <typename U, typename... Args> void construct(U *p, Args &&...args)
    {
        std::construct_at(p, std::forward<Args>(args)...);
    }

    friend bool operator==(RawAllocator const &, RawAllocator const &) = default;
};

} // namespace bonsai::detail
