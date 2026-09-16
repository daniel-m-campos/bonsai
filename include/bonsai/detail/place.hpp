#pragma once

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <ranges>
#include <string>
#include <tuple>
#include <vector>

#ifdef __linux__
#include <fstream>
#include <sched.h>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#endif

namespace bonsai::parallel::internal
{

// One CPU the process may run on, with the package and core it sits in.
struct CpuPlace
{
    int cpu;
    int package;
    int core;
};

// The team order of these places: one CPU per core, one package before the
// next (invariants: team-packs-one-package-first).
inline std::vector<int> place_cpus(std::vector<CpuPlace> places)
{
    std::ranges::sort(places, {}, [](CpuPlace const &p)
                      { return std::tuple{p.package, p.core, p.cpu}; });
    auto const [dup, end] = std::ranges::unique(
        places, {}, [](CpuPlace const &p) { return std::tuple{p.package, p.core}; });
    places.erase(dup, end);
    return places | std::views::transform(&CpuPlace::cpu) |
           std::ranges::to<std::vector<int>>();
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

inline int place_id(int cpu, char const *leaf)
{
    std::string const path =
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/" + leaf;
    int               id   = cpu;
    std::string const line = first_line(path.c_str());
    std::from_chars(line.data(), line.data() + line.size(), id);
    return id;
}

// The APIC id widths of the SMT and core levels (cpuid leaf 0xB); zero
// core width when the leaf is absent.
struct ApicLevels
{
    unsigned smt_bits  = 0;
    unsigned core_bits = 0;
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
        }
        else if (type == 2)
        {
            out.core_bits = a & 0x1F;
        }
    }
    return out;
}

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

inline bool run_on(int cpu)
{
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(cpu, &one);
    return sched_setaffinity(0, sizeof one, &one) == 0;
}

// The process mask and its CPUs in team order, read once from a thread
// carrying that mask. The runtime reads its topology from the CPU itself
// and a virtual machine's /sys can name each vCPU a core of its own while
// cpuid pairs them as siblings, so on x86 the reading thread visits each
// CPU for its APIC id, with /sys as the fallback; a visit whose restore
// fails leaves the team empty, so nothing is ever placed on a mask the
// thread could not get back.
struct CpuPlaces
{
    cpu_set_t        set{};
    bool             ok = false;
    std::vector<int> team;
};

inline CpuPlaces const &cpu_places()
{
    static CpuPlaces const mask = []
    {
        CpuPlaces m;
        CPU_ZERO(&m.set);
        m.ok = sched_getaffinity(0, sizeof m.set, &m.set) == 0;
        if (!m.ok)
        {
            return m;
        }
        std::vector<CpuPlace> places;
#if defined(__x86_64__) || defined(__i386__)
        ApicLevels const levels = apic_levels();
        if (levels.core_bits > 0)
        {
            for (int c = 0; c < CPU_SETSIZE; ++c)
            {
                if (CPU_ISSET(c, &m.set) && run_on(c))
                {
                    unsigned const id = apic_id();
                    places.push_back(
                        {.cpu     = c,
                         .package = static_cast<int>(id >> levels.core_bits),
                         .core    = static_cast<int>(id >> levels.smt_bits)});
                }
            }
            if (sched_setaffinity(0, sizeof m.set, &m.set) != 0)
            {
                return m;
            }
            m.team = place_cpus(std::move(places));
            return m;
        }
#endif
        for (int c = 0; c < CPU_SETSIZE; ++c)
        {
            if (CPU_ISSET(c, &m.set))
            {
                places.push_back({.cpu     = c,
                                  .package = place_id(c, "physical_package_id"),
                                  .core    = place_id(c, "core_id")});
            }
        }
        m.team = place_cpus(std::move(places));
        return m;
    }();
    return mask;
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
#endif

// How many fits hold a place at once: one, so two fits in one process
// never pack onto the same CPUs; the second runs unplaced.
inline std::atomic<int> &placed_fits()
{
    static std::atomic<int> n{0};
    return n;
}

// Whether the calling thread's fit holds a place, read before a region
// opens and handed to its workers.
inline bool &fit_place_open_flag()
{
    static thread_local bool placed = false;
    return placed;
}

inline bool fit_place_open()
{
    return fit_place_open_flag();
}

// A worker's place for one region: while the caller is placed it runs on
// the tid-th team CPU, and a worker still carrying a pin from an earlier
// fit takes the process mask back the first time it serves an unplaced
// region, so nothing stays pinned past the fit that placed it.
inline void take_place(int tid, bool placed)
{
#ifdef __linux__
    static thread_local int on_cpu = -1;
    if (tid == 0)
    {
        return;
    }
    CpuPlaces const &mask = cpu_places();
    if (!placed)
    {
        if (on_cpu >= 0 && mask.ok &&
            sched_setaffinity(0, sizeof mask.set, &mask.set) == 0)
        {
            on_cpu = -1;
        }
        return;
    }
    if (static_cast<size_t>(tid) >= mask.team.size())
    {
        return;
    }
    int const cpu = mask.team[static_cast<size_t>(tid)];
    if (on_cpu != cpu && run_on(cpu))
    {
        on_cpu = cpu;
    }
#else
    (void) tid;
    (void) placed;
#endif
}

} // namespace bonsai::parallel::internal
