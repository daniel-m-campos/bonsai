#include <catch2/catch_test_macros.hpp>

#include "bonsai/io/csv.hpp"

// Stub. TODO(user): tests for csv::parse on small committed CSVs in
// tests/data/; round-trip read_csv -> Dataset; fit_from_csv producing
// a sane BinMappers.

TEST_CASE("csv compiles", "[csv]") {
    REQUIRE(true);
}
