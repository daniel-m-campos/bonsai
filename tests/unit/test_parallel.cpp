#include <catch2/catch_test_macros.hpp>

#include "bonsai/parallel.hpp"

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
