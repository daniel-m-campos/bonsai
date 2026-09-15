#pragma once

#include <cstddef>
#include <new>
#include <vector>

namespace bonsai::detail
{

// Zero-argument construction is a no-op, so the first write to a page homes
// it on the writer's node: only for a buffer a parallel producer writes end
// to end before any read (invariants: untouched-resize-writes-nothing).
template <typename T> struct UntouchedAllocator
{
    using value_type = T;

    UntouchedAllocator() = default;
    // Allocator rebind requires an implicit converting constructor.
    // NOLINTNEXTLINE(google-explicit-constructor)
    template <typename U> UntouchedAllocator(UntouchedAllocator<U> const & /*other*/) {}

    T *allocate(size_t n)
    {
        return static_cast<T *>(::operator new(n * sizeof(T)));
    }

    void deallocate(T *p, size_t /*n*/)
    {
        ::operator delete(p);
    }

    void construct(T * /*p*/) {}

    friend bool operator==(UntouchedAllocator const &, UntouchedAllocator const &)
    {
        return true;
    }
};

template <typename T> using untouched_vector = std::vector<T, UntouchedAllocator<T>>;

} // namespace bonsai::detail
