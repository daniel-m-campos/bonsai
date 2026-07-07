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
