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
        // Per-thread front cache: the level fill takes and gives hundreds of
        // node arenas per level from every worker, and a single mutex over
        // that traffic serializes the allocator. The global pool stays the
        // spill target, so memory stays bounded.
        auto &local = local_free()[bytes];
        if (!local.empty())
        {
            void *p = local.back();
            local.pop_back();
            return p;
        }
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
        auto &local = local_free()[bytes];
        if (local.size() < k_local_cap)
        {
            local.push_back(p);
            return;
        }
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

    static constexpr size_t k_local_cap = 64;

    static std::unordered_map<size_t, std::vector<void *>> &local_free()
    {
        static thread_local std::unordered_map<size_t, std::vector<void *>> lists;
        return lists;
    }

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
