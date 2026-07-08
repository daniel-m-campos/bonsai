#pragma once

#include <cstddef>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

namespace bonsai::detail
{

// Size-class free-list for histogram cell blocks. populate() allocates a
// fresh n_bins block per (node, feature) and drops it when the node leaves
// the frontier; at high bin counts the first-touch page faults of those
// fresh blocks dominate the fit (bins-axis exponent, decision 47). Recycling
// keeps the pages warm. Mutex'd: allocations are per-node, not per-row.
class HistBlockPool
{
  public:
    static HistBlockPool &instance()
    {
        static HistBlockPool pool;
        return pool;
    }

    void *take(size_t bytes)
    {
        {
            std::scoped_lock lock(mu_);
            auto            &list = free_[bytes];
            if (!list.empty())
            {
                void *p = list.back();
                list.pop_back();
                return p;
            }
        }
        return ::operator new(bytes);
    }

    void give(void *p, size_t bytes)
    {
        std::scoped_lock lock(mu_);
        free_[bytes].push_back(p);
    }

    ~HistBlockPool()
    {
        for (auto &[bytes, list] : free_)
        {
            for (void *p : list)
            {
                ::operator delete(p);
            }
        }
    }

    HistBlockPool(HistBlockPool const &)            = delete;
    HistBlockPool &operator=(HistBlockPool const &) = delete;

  private:
    HistBlockPool() = default;

    std::mutex                                      mu_;
    std::unordered_map<size_t, std::vector<void *>> free_;
};

// Stateless allocator over the pool; all instances compare equal, so vector
// moves and swaps behave exactly like std::allocator.
template <typename T> struct PoolAllocator
{
    using value_type = T;

    PoolAllocator() = default;
    // Allocator rebind requires an implicit converting constructor.
    // NOLINTNEXTLINE(google-explicit-constructor)
    template <typename U> PoolAllocator(PoolAllocator<U> const & /*other*/) {}

    T *allocate(size_t n)
    {
        return static_cast<T *>(HistBlockPool::instance().take(n * sizeof(T)));
    }

    void deallocate(T *p, size_t n)
    {
        HistBlockPool::instance().give(p, n * sizeof(T));
    }

    friend bool operator==(PoolAllocator const &, PoolAllocator const &)
    {
        return true;
    }
};

} // namespace bonsai::detail
