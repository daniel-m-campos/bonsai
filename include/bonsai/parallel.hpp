#pragma once

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <print>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#ifdef __linux__
#include <fstream>
#include <sched.h>
#include <string>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
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
// One CPU the process may run on, with the package and core it sits in.
struct CpuPlace
{
    int cpu;
    int package;
    int core;
};

// The order a team fills CPUs in: one CPU per core, one package before the
// next (invariants: team-packs-one-package-first).
inline std::vector<int> place_order(std::vector<CpuPlace> places)
{
    std::ranges::sort(places,
                      [](CpuPlace const &a, CpuPlace const &b)
                      {
                          return std::tie(a.package, a.core, a.cpu) <
                                 std::tie(b.package, b.core, b.cpu);
                      });
    std::vector<int> order;
    for (size_t i = 0; i < places.size(); ++i)
    {
        if (i == 0 || places[i].package != places[i - 1].package ||
            places[i].core != places[i - 1].core)
        {
            order.push_back(places[i].cpu);
        }
    }
    return order;
}

#ifdef __linux__
inline int topology_id(int cpu, char const *leaf)
{
    std::string const path =
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/" + leaf;
    int               id   = cpu;
    std::string const line = first_line(path.c_str());
    std::from_chars(line.data(), line.data() + line.size(), id);
    return id;
}

// The APIC id widths of the SMT and core levels (cpuid leaf 0xB), zero
// widths when the leaf is absent.
struct ApicLevels
{
    unsigned smt_bits  = 0;
    unsigned core_bits = 0;
    bool     present   = false;
};

#if defined(__x86_64__) || defined(__i386__)
inline ApicLevels apic_levels()
{
    ApicLevels out;
    unsigned   a = 0;
    unsigned   b = 0;
    unsigned   c = 0;
    unsigned   d = 0;
    if (__get_cpuid_max(0, nullptr) < 0xB)
    {
        return out;
    }
    for (unsigned sub = 0; sub < 4; ++sub)
    {
        __cpuid_count(0xB, sub, a, b, c, d);
        unsigned const type = (c >> 8) & 0xFF;
        if (type == 1)
        {
            out.smt_bits = a & 0x1F;
            out.present  = true;
        }
        else if (type == 2)
        {
            out.core_bits = a & 0x1F;
            out.present   = true;
        }
    }
    return out;
}

// The x2APIC id of the CPU the calling thread runs on.
inline unsigned apic_id()
{
    unsigned a = 0;
    unsigned b = 0;
    unsigned c = 0;
    unsigned d = 0;
    __cpuid_count(0xB, 0, a, b, c, d);
    return d;
}
#endif

// The place of every CPU in the mask. The runtime reads its topology from
// the CPU itself and a virtual machine's /sys can name each vCPU a core of
// its own while cpuid pairs them as siblings, so on x86 the calling thread
// visits each CPU for its APIC id and /sys is the fallback.
inline std::vector<CpuPlace> mask_places()
{
    std::vector<CpuPlace> places;
    cpu_set_t             set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0)
    {
        return places;
    }
#if defined(__x86_64__) || defined(__i386__)
    ApicLevels const levels = apic_levels();
    if (levels.present && levels.core_bits > 0)
    {
        for (int c = 0; c < CPU_SETSIZE; ++c)
        {
            if (!CPU_ISSET(c, &set))
            {
                continue;
            }
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(c, &one);
            if (sched_setaffinity(0, sizeof one, &one) != 0)
            {
                continue;
            }
            unsigned const id = apic_id();
            places.push_back({.cpu     = c,
                              .package = static_cast<int>(id >> levels.core_bits),
                              .core    = static_cast<int>(id >> levels.smt_bits)});
        }
        sched_setaffinity(0, sizeof set, &set);
        return places;
    }
#endif
    for (int c = 0; c < CPU_SETSIZE; ++c)
    {
        if (CPU_ISSET(c, &set))
        {
            places.push_back({.cpu     = c,
                              .package = topology_id(c, "physical_package_id"),
                              .core    = topology_id(c, "core_id")});
        }
    }
    return places;
}

// The process mask in team order, read once before any thread is bound.
inline std::vector<int> const &team_cpus()
{
    static std::vector<int> const cpus = place_order(mask_places());
    return cpus;
}

// Any of the runtime's placement variables, OMP_PROC_BIND=false included,
// leaves placement to the runtime.
inline bool runtime_places_threads()
{
    static bool const set = std::getenv("OMP_PROC_BIND") != nullptr ||
                            std::getenv("OMP_PLACES") != nullptr ||
                            std::getenv("KMP_AFFINITY") != nullptr ||
                            std::getenv("GOMP_CPU_AFFINITY") != nullptr;
    return set;
}

inline void run_on(int cpu)
{
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(cpu, &one);
    sched_setaffinity(0, sizeof one, &one);
}
#endif

inline std::atomic<int> &placed_callers()
{
    static std::atomic<int> n{0};
    return n;
}

#ifdef BONSAI_USE_OPENMP
// The runtime's thread count, read once and before any CallerPlace narrows
// the calling thread: the runtime sizes itself from the mask it first sees,
// so a fit with an explicit count opening the scope ahead of the first
// region would otherwise leave every later auto count at one.
inline int hardware_threads()
{
    static int const n = omp_get_max_threads();
    return n;
}
#endif

// A worker binds to the tid-th team CPU once and keeps it, only while a
// CallerPlace is open: without one the team runs wherever the scheduler
// puts it, as it did before placement existed.
inline void bind_worker(int tid, int team)
{
#ifdef __linux__
    static thread_local bool bound = false;
    if (bound || tid == 0 || placed_callers().load(std::memory_order_relaxed) == 0)
    {
        return;
    }
    std::vector<int> const &cpus = team_cpus();
    if (runtime_places_threads() || cpus.size() < static_cast<size_t>(team))
    {
        return;
    }
    bound = true;
    run_on(cpus[static_cast<size_t>(tid)]);
#else
    (void) tid;
    (void) team;
#endif
}
} // namespace internal

// The configuration point, called before a fit and never inside a parallel
// section: the fill plan keys model bytes to the resolved count, so a change
// mid-fit alters the model without an error.
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
// resolved count; one over the quota draws a warning instead (invariants:
// auto-thread-cap).
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
    int const capped = std::min(internal::hardware_threads(), auto_thread_cap);
    return quota > 0 ? std::min(capped, quota) : capped;
#else
    return 1;
#endif
}

// The placement scope, opened by a fit's entry point for the length of the
// call: the calling thread takes the team's first CPU, so everything it
// allocates between regions homes on the team's node, and its workers bind
// to the CPUs after it. The mask comes back when the scope closes, so no
// thread created outside a fit inherits a one-CPU mask. Skipped when the
// runtime's placement variables are set or the mask is narrower than the
// team (invariants: caller-mask-restored-after-fit).
class CallerPlace
{
  public:
    CallerPlace()
    {
#ifdef BONSAI_USE_OPENMP
        internal::hardware_threads();
#endif
#ifdef __linux__
        std::vector<int> const &cpus = internal::team_cpus();
        if (internal::runtime_places_threads() ||
            cpus.size() < static_cast<size_t>(n_threads()))
        {
            return;
        }
        restore_ = sched_getaffinity(0, sizeof saved_, &saved_) == 0;
        if (restore_)
        {
            internal::run_on(cpus[0]);
            internal::placed_callers().fetch_add(1, std::memory_order_relaxed);
        }
#endif
    }
    ~CallerPlace()
    {
#ifdef __linux__
        if (restore_)
        {
            internal::placed_callers().fetch_sub(1, std::memory_order_relaxed);
            sched_setaffinity(0, sizeof saved_, &saved_);
        }
#endif
    }
    CallerPlace(CallerPlace const &)            = delete;
    CallerPlace &operator=(CallerPlace const &) = delete;

  private:
#ifdef __linux__
    bool      restore_ = false;
    cpu_set_t saved_{};
#endif
};

// Runs f(i) for i in [0, n) on a team of `workers`, each worker at its
// place while a CallerPlace is open. Iterations must be independent. Each index is
// processed by exactly one thread, so per-index OUTPUTS are bit-identical at any
// thread count; sites whose work DECOMPOSITION consults n_threads() (the fill plan)
// key the model bits to the configured count, the fixed-N contract (invariants:
// host-determinism). Dynamic scheduling keeps asymmetric cores (e.g. P/E) busy; the
// chunk size scales with n so per-chunk overhead stays negligible for big loops
// while small loops still spread one index per thread. A team of one runs the loop
// inline, entering no region.
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
#pragma omp parallel num_threads(nt)
        {
            internal::bind_worker(omp_get_thread_num(), nt);
#pragma omp for schedule(dynamic, chunk)
            for (int64_t i = 0; i < static_cast<int64_t>(n); ++i)
            {
                f(static_cast<size_t>(i));
            }
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
