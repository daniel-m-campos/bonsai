#include <catch2/catch_test_macros.hpp>

#include "bonsai/bin_mapper.hpp"

// Stub. TODO(user): write tests as BinMapper::fit / transform are
// implemented. Cover analytical cases: small column, NaN handling,
// low-cardinality fallback, dedup, sentinel handling.

TEST_CASE("BinMapper compiles", "[bin_mapper]") {
    REQUIRE(true);
}
