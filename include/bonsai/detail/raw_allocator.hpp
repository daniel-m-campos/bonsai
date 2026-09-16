#pragma once

#include <cstddef>
#include <memory>

namespace bonsai::detail
{

// Zero-argument construction is a no-op, so the first write to a page homes
// it on the writer's node: only for a buffer a parallel producer writes end
// to end before any read (invariants: untouched-resize-writes-nothing).
template <typename T> struct RawAllocator : std::allocator<T>
{
    using std::allocator<T>::allocator;
    template <typename U> struct rebind
    {
        using other = RawAllocator<U>;
    };
    void construct(T * /*p*/) {}
};

} // namespace bonsai::detail
