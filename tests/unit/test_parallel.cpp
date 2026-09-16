#include <catch2/catch_test_macros.hpp>

#include "bonsai/parallel.hpp"

#include <vector>

using namespace bonsai; // NOLINT

TEST_CASE("auto thread count is capped", "[parallel]")
{
    parallel::set_n_threads(0);
    int const n = parallel::n_threads();
    CHECK(n >= 1);
    CHECK(n <= parallel::auto_thread_cap);
}

// The parser is tested from strings; whether a fit on a quota-limited
// container stays unthrottled is an end-to-end check no unit test can make.
TEST_CASE("cgroup quota pairs parse to whole CPUs", "[parallel]")
{
    CHECK(parallel::internal::quota_cpus("200000 100000") == 2);
    CHECK(parallel::internal::quota_cpus("1360000 100000") == 13);
    CHECK(parallel::internal::quota_cpus("50000 100000") == 1);
    CHECK(parallel::internal::quota_cpus("max 100000") == 0);
    CHECK(parallel::internal::quota_cpus("-1 100000") == 0);
    CHECK(parallel::internal::quota_cpus("200000") == 0);
    CHECK(parallel::internal::quota_cpus("200000 0") == 0);
    CHECK(parallel::internal::quota_cpus("") == 0);
}

// The predicate is tested directly; the emission itself is one line on
// stderr, once per process, which no unit test can observe twice.
TEST_CASE("only an explicit count over a real quota warns", "[parallel]")
{
    CHECK(parallel::internal::should_warn(24, 13));
    CHECK_FALSE(parallel::internal::should_warn(13, 13));
    CHECK_FALSE(parallel::internal::should_warn(8, 13));
    CHECK_FALSE(parallel::internal::should_warn(24, 0));
    CHECK_FALSE(parallel::internal::should_warn(0, 13));
}

TEST_CASE("explicit thread count passes through uncapped", "[parallel]")
{
    parallel::set_n_threads(24);
#ifdef BONSAI_USE_OPENMP
    CHECK(parallel::n_threads() == 24);
#else
    CHECK(parallel::n_threads() == 1);
#endif
    parallel::set_n_threads(0);
}

// INVARIANT: team-packs-one-package-first
// A team fills one CPU per core, cores in id order, one package before the
// next: SMT siblings of a core never both host a worker, and a team no
// wider than a package shares its cache and memory node. Read from the
// process mask and /sys on Linux; the ordering is pinned from values here.
TEST_CASE("team order packs one package first, one CPU per core",
          "[parallel][invariant]")
{
    using parallel::internal::CpuPlace;
    std::vector<CpuPlace> const places = {
        {.cpu = 0, .package = 0, .core = 0}, {.cpu = 1, .package = 1, .core = 0},
        {.cpu = 2, .package = 0, .core = 1}, {.cpu = 3, .package = 1, .core = 1},
        {.cpu = 4, .package = 0, .core = 0}, {.cpu = 5, .package = 1, .core = 0},
        {.cpu = 6, .package = 0, .core = 1}, {.cpu = 7, .package = 1, .core = 1},
    };
    CHECK(parallel::internal::place_cpus(places) == std::vector<int>{0, 2, 1, 3});
    CHECK(parallel::internal::place_cpus({}).empty());
}

#ifdef __linux__
#include <sched.h>

// INVARIANT: caller-mask-restored-after-fit
// A FitPlace narrows the calling thread to the team's first CPU for its
// scope and hands the thread's own mask back when it closes, so a fit
// leaves the thread it ran on, and every thread created after it, exactly
// as placed as before. With the runtime's placement variables set the
// scope changes nothing. Linux only: elsewhere the scope is a no-op and
// there is nothing to restore.
TEST_CASE("the caller's place lasts its scope and the mask comes back",
          "[parallel][invariant]")
{
    cpu_set_t before;
    CPU_ZERO(&before);
    REQUIRE(sched_getaffinity(0, sizeof before, &before) == 0);
    bool const placed_by_runtime = parallel::internal::runtime_places_threads();
    parallel::set_n_threads(1);
    {
        parallel::FitPlace const placed;
        cpu_set_t                inside;
        CPU_ZERO(&inside);
        REQUIRE(sched_getaffinity(0, sizeof inside, &inside) == 0);
        auto const &team = parallel::internal::cpu_places().team;
        if (!placed_by_runtime && !team.empty())
        {
            CHECK(CPU_COUNT(&inside) == 1);
            CHECK(CPU_ISSET(team[0], &inside));
            CHECK(parallel::internal::placed_fits().load() == 1);
            CHECK(parallel::internal::fit_place_open());
        }
    }
    cpu_set_t after;
    CPU_ZERO(&after);
    REQUIRE(sched_getaffinity(0, sizeof after, &after) == 0);
    CHECK(CPU_EQUAL(&before, &after));
    CHECK(parallel::internal::placed_fits().load() == 0);
    CHECK(!parallel::internal::fit_place_open());
    parallel::set_n_threads(0);
}
#endif
