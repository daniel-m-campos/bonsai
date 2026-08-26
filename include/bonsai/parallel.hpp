#pragma once

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <print>
#include <string_view>
#include <utility>

#ifdef __linux__
#include <fstream>
#include <string>
#endif

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

// Whole CPUs a "QUOTA PERIOD" microsecond pair allows, the cgroup v2
// cpu.max line and the v1 file pair joined. 0 = unlimited ("max" or a
// negative quota), malformed, or empty; a sub-CPU quota still needs one
// worker.
inline int quota_cpus(std::string_view pair)
{
    char const *const begin     = pair.data();
    char const *const end       = begin + pair.size();
    int64_t           quota     = 0;
    auto const [rest, quota_ec] = std::from_chars(begin, end, quota);
    if (quota_ec != std::errc{} || quota <= 0)
    {
        return 0;
    }
    char const *cursor = rest;
    while (cursor != end && (*cursor == ' ' || *cursor == '\t'))
    {
        ++cursor;
    }
    int64_t period = 0;
    if (std::from_chars(cursor, end, period).ec != std::errc{} || period <= 0)
    {
        return 0;
    }
    return static_cast<int>(std::max<int64_t>(1, quota / period));
}

#ifdef __linux__
// First line of a file, empty when it does not open.
inline std::string first_line(char const *path)
{
    std::ifstream in{path};
    std::string   line;
    std::getline(in, line);
    return line;
}
#endif

// Whole CPUs the CPU bandwidth quota allows, 0 when there is none. Reads
// the unified cgroup v2 mount then the v1 pair, both at the paths a
// container sees for its own cgroup; a process in a non-root cgroup of the
// host namespace reads its quota as unlimited, since resolving that needs
// the relative path from /proc/self/cgroup joined to the mount point.
inline int cgroup_quota_cpus()
{
#ifdef __linux__
    if (int const unified = quota_cpus(first_line("/sys/fs/cgroup/cpu.max"));
        unified > 0)
    {
        return unified;
    }
    return quota_cpus(first_line("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") + " " +
                      first_line("/sys/fs/cgroup/cpu/cpu.cfs_period_us"));
#else
    return 0;
#endif
}

// Read once: the quota is fixed for the process lifetime.
inline int cached_quota_cpus()
{
    static int const quota = cgroup_quota_cpus();
    return quota;
}

// An explicit count above the quota is honored, not clamped, so the
// warning is the only signal. No quota (0, including an unreadable one)
// and counts within it stay silent, as does auto (0), which clamps.
inline bool should_warn(int requested, int quota)
{
    return requested > 0 && quota > 0 && requested > quota;
}

// Once per process, off the parallel path: set_n_threads is the
// configuration point, called before a fit and never inside a section.
inline void warn_if_over_quota(int requested)
{
    static std::atomic_flag warned;
    int const               quota = cached_quota_cpus();
    if (!should_warn(requested, quota) ||
        warned.test_and_set(std::memory_order_relaxed))
    {
        return;
    }
    std::println(stderr,
                 "bonsai: n_threads={} exceeds the cgroup CPU quota of {}, so "
                 "CFS throttling will slow the fit; lower n_threads to {} or "
                 "set OMP_WAIT_POLICY=passive to avoid it.",
                 requested, quota, quota);
}
} // namespace internal

inline void set_n_threads(uint32_t n)
{
    internal::n_threads_slot() = static_cast<int>(n);
    internal::warn_if_over_quota(static_cast<int>(n));
}

// Auto (n_threads = 0) caps the worker count: per-level parallel sections
// are short, so on many-core hosts OpenMP barrier spin-wait dominates
// useful work: 60 vCPU measured 10x slower than 16. Auto also clamps
// to the cgroup CPU bandwidth quota when one is set: OpenMP sizes its pool
// from the affinity mask, which a quota-limited container leaves at the
// host's core count, so an unclamped pool burns its quota early and the
// fit spends most of every period frozen by the scheduler. Explicit counts
// pass through uncapped, since the fixed-N contract keys model bytes to the
// resolved count; one over the quota draws a warning instead.
inline constexpr int auto_thread_cap = 16;

inline int n_threads()
{
#ifdef BONSAI_USE_OPENMP
    int const requested = internal::n_threads_slot();
    if (requested > 0)
    {
        return requested;
    }
    int const quota  = internal::cached_quota_cpus();
    int const capped = std::min(omp_get_max_threads(), auto_thread_cap);
    return quota > 0 ? std::min(capped, quota) : capped;
#else
    return 1;
#endif
}

// Runs f(i) for i in [0, n) on a team of `workers`. Iterations must be
// independent. Each index is processed by exactly one thread, so per-index
// OUTPUTS are bit-identical at any thread count; sites whose work
// DECOMPOSITION consults n_threads() (the fill plan) key the model bits to
// the configured count, the fixed-N contract
// (invariants: host-determinism). Dynamic scheduling keeps asymmetric
// cores (e.g. P/E) busy; the chunk size scales with n so per-chunk overhead
// stays negligible for big loops while small loops still spread one index
// per thread. A team of one runs the loop inline, entering no region.
template <typename F> void for_each_index_on(int workers, size_t n, F &&f)
{
#ifdef BONSAI_USE_OPENMP
    int const nt = std::max(1, workers);
    if (nt > 1 && n > 1)
    {
        // maybe_unused: referenced only by the pragma, which CUDA device
        // compilation passes drop even with OpenMP enabled host-side.
        [[maybe_unused]] auto const chunk = static_cast<int64_t>(
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

// The same loop on the configured team. Sites that size a team to the work
// call for_each_index_on directly; everything else spreads over n_threads().
template <typename F> void for_each_index(size_t n, F &&f)
{
    for_each_index_on(n_threads(), n, std::forward<F>(f));
}

} // namespace bonsai::parallel
