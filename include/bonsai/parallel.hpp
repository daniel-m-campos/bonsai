#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#ifdef BONSAI_USE_OPENMP
#include <omp.h>
#endif

namespace bonsai::parallel
{

namespace internal
{
// Global worker-count knob, set once at startup from ParallelConfig.
// 0 = use all hardware threads. Serial builds ignore it.
inline int &n_threads_slot()
{
    static int n = 0;
    return n;
}
} // namespace internal

inline void set_n_threads(uint32_t n)
{
    internal::n_threads_slot() = static_cast<int>(n);
}

inline int n_threads()
{
#ifdef BONSAI_USE_OPENMP
    int const requested = internal::n_threads_slot();
    return requested > 0 ? requested : omp_get_max_threads();
#else
    return 1;
#endif
}

// Runs f(i) for i in [0, n). Iterations must be independent. Each index is
// processed by exactly one thread, so per-index outputs are bit-identical
// to a serial run at any thread count (no cross-thread reductions).
// Dynamic scheduling keeps asymmetric cores (e.g. P/E) busy; the chunk
// size scales with n so per-chunk overhead stays negligible for big loops
// while small loops still spread one index per thread.
template <typename F> void for_each_index(size_t n, F &&f)
{
#ifdef BONSAI_USE_OPENMP
    int const nt = n_threads();
    if (nt > 1 && n > 1)
    {
        auto const chunk = static_cast<int64_t>(
            std::max<size_t>(1, n / (static_cast<size_t>(nt) * 4)));
#pragma omp parallel for schedule(dynamic, chunk) num_threads(nt)
        for (int64_t i = 0; i < static_cast<int64_t>(n); ++i)
        {
            f(static_cast<size_t>(i));
        }
        return;
    }
#endif
    for (size_t i = 0; i < n; ++i)
    {
        f(i);
    }
}

} // namespace bonsai::parallel
