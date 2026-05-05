#include <catch2/catch_test_macros.hpp>

#include "bonsai/dataset.hpp"

// Stub. TODO(user): tests for Dataset::bin producing column-major
// uint16_t storage matching BinMapper transforms; labels / weights
// ownership; n_buckets per feature.

TEST_CASE("Dataset: smoke", "[dataset][smoke]") {
    REQUIRE(true);
}
