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
        free_blocks(free_);
    }

    HistBlockPool(HistBlockPool const &)            = delete;
    HistBlockPool &operator=(HistBlockPool const &) = delete;

  private:
    using block_lists_t = std::unordered_map<size_t, std::vector<void *>>;

    HistBlockPool() = default;

    static constexpr size_t k_local_cap = 64;

    // Blocks come from unsized ::operator new, so unsized delete is the
    // matching form.
    static void free_blocks(block_lists_t &lists)
    {
        for (auto &[bytes, list] : lists)
        {
            for (void *p : list)
            {
                ::operator delete(p);
            }
        }
    }

    // Thread exit frees its own cached blocks rather than handing them to the
    // pool, which may already be gone: static and thread storage destroy in
    // unspecified order. Recycling is lost for at most k_local_cap blocks per
    // size class per exiting thread, and OpenMP workers live for the process.
    struct LocalCache
    {
        block_lists_t lists;

        ~LocalCache()
        {
            free_blocks(lists);
        }
    };

    static block_lists_t &local_free()
    {
        static thread_local LocalCache cache;
        return cache.lists;
    }

    std::mutex    mu_;
    block_lists_t free_;
};

// Stateless allocator over the pool; all instances compare equal, so vector
// moves and swaps behave exactly like std::allocator. Init = false drops the
// value-init a resize would do, leaving raw storage the owner must start every
// element's lifetime in: only NodeHistograms may ask for that, and only
// because its arena reset zeroes every cell across workers right after the
// resize.
template <typename T, bool Init = true> struct PoolAllocator
{
    using value_type = T;
    // A non-type parameter puts this allocator outside allocator_traits'
    // default rebind, so it spells its own.
    template <typename U> struct rebind
    {
        using other = PoolAllocator<U, Init>;
    };

    PoolAllocator() = default;
    // Allocator rebind requires an implicit converting constructor.
    // NOLINTNEXTLINE(google-explicit-constructor)
    template <typename U> PoolAllocator(PoolAllocator<U, Init> const & /*other*/) {}

    T *allocate(size_t n)
    {
        return static_cast<T *>(HistBlockPool::instance().take(n * sizeof(T)));
    }

    void deallocate(T *p, size_t n)
    {
        HistBlockPool::instance().give(p, n * sizeof(T));
    }

    // Value-init from resize() lands here and does nothing; every other
    // construction falls through to allocator_traits' placement new.
    void construct(T * /*p*/)
        requires(!Init)
    {
    }

    friend bool operator==(PoolAllocator const &, PoolAllocator const &)
    {
        return true;
    }
};

template <typename T> using RawPoolAllocator = PoolAllocator<T, false>;

} // namespace bonsai::detail
