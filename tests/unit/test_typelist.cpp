#include <catch2/catch_test_macros.hpp>
#include <type_traits>

#include "bonsai/typelist.hpp"

using namespace bonsai; // NOLINT

namespace
{
struct A
{
};
struct B
{
};
struct C
{
};
struct D
{
};
struct E
{
};

using L0 = TypeList<>;
using L1 = TypeList<A>;
using L2 = TypeList<A, B>;
using L3 = TypeList<C, D, E>;
} // namespace

// ---- size_v ----------------------------------------------------------------
static_assert(size_v<L0> == 0);
static_assert(size_v<L1> == 1);
static_assert(size_v<L2> == 2);
static_assert(size_v<L3> == 3);

// ---- type_at_t -------------------------------------------------------------
static_assert(std::is_same_v<type_at_t<0, L1>, A>);
static_assert(std::is_same_v<type_at_t<0, L2>, A>);
static_assert(std::is_same_v<type_at_t<1, L2>, B>);
static_assert(std::is_same_v<type_at_t<2, L3>, E>);

// ---- concat_t --------------------------------------------------------------
static_assert(std::is_same_v<concat_t<>, TypeList<>>);
static_assert(std::is_same_v<concat_t<L1>, TypeList<A>>);
static_assert(std::is_same_v<concat_t<L1, L2>, TypeList<A, A, B>>);
static_assert(size_v<concat_t<L2, L3>> == 5);
static_assert(std::is_same_v<type_at_t<4, concat_t<L2, L3>>, E>);

// ---- cartesian_product_t ---------------------------------------------------
using CP_2x2 = cartesian_product_t<L2, TypeList<C, D>>;
static_assert(size_v<CP_2x2> == 4);
// Order: (A,C) (A,D) (B,C) (B,D)
static_assert(std::is_same_v<type_at_t<0, type_at_t<0, CP_2x2>>, A>);
static_assert(std::is_same_v<type_at_t<1, type_at_t<0, CP_2x2>>, C>);
static_assert(std::is_same_v<type_at_t<0, type_at_t<1, CP_2x2>>, A>);
static_assert(std::is_same_v<type_at_t<1, type_at_t<1, CP_2x2>>, D>);
static_assert(std::is_same_v<type_at_t<0, type_at_t<2, CP_2x2>>, B>);
static_assert(std::is_same_v<type_at_t<1, type_at_t<2, CP_2x2>>, C>);
static_assert(std::is_same_v<type_at_t<0, type_at_t<3, CP_2x2>>, B>);
static_assert(std::is_same_v<type_at_t<1, type_at_t<3, CP_2x2>>, D>);

// 3D: 2 x 1 x 2 = 4
using CP_3D = cartesian_product_t<TypeList<A, B>, TypeList<C>, TypeList<D, E>>;
static_assert(size_v<CP_3D> == 4);
// First combo: (A, C, D)
static_assert(std::is_same_v<type_at_t<0, type_at_t<0, CP_3D>>, A>);
static_assert(std::is_same_v<type_at_t<1, type_at_t<0, CP_3D>>, C>);
static_assert(std::is_same_v<type_at_t<2, type_at_t<0, CP_3D>>, D>);
// Last combo: (B, C, E)
static_assert(std::is_same_v<type_at_t<0, type_at_t<3, CP_3D>>, B>);
static_assert(std::is_same_v<type_at_t<2, type_at_t<3, CP_3D>>, E>);

// Empty dim collapses the product:
static_assert(size_v<cartesian_product_t<L2, TypeList<>>> == 0);

// ---- for_each_type ---------------------------------------------------------
TEST_CASE("for_each_type iterates each type once", "[typelist][for_each_type]")
{
    int counter = 0;
    for_each_type<L3>([&counter]<typename T>() { ++counter; });
    CHECK(counter == 3);
}

TEST_CASE("for_each_type over empty list is a no-op", "[typelist][for_each_type]")
{
    int counter = 0;
    for_each_type<L0>([&counter]<typename T>() { ++counter; });
    CHECK(counter == 0);
}

TEST_CASE("typelist static_asserts compile", "[typelist]")
{
    // Pure compile-time check; runtime SUCCEED so Catch reports a case.
    SUCCEED();
}
