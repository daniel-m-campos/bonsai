#include <catch2/catch_test_macros.hpp>

#include "bonsai/detail/untouched.hpp"

using namespace bonsai; // NOLINT

namespace
{
struct Counted
{
    inline static int constructed = 0;
    int               value       = 7;
    Counted()
    {
        ++constructed;
    }
};
} // namespace

// INVARIANT: untouched-resize-writes-nothing
// A resize of an untouched vector allocates and constructs nothing: every
// zero-argument construction is skipped, so the first write to each page is
// the producer's and homes the page on the writer's memory node, which is
// what lets a bound team read the row mirror from its own node. A
// construction from a value runs as usual.
TEST_CASE("untouched vector resizes without constructing", "[parallel][invariant]")
{
    Counted::constructed = 0;
    detail::untouched_vector<Counted> v;
    v.resize(64);
    CHECK(v.size() == 64);
    CHECK(Counted::constructed == 0);
    Counted const probe;
    v.push_back(probe);
    CHECK(Counted::constructed == 1);
    CHECK(v.back().value == 7);
}
