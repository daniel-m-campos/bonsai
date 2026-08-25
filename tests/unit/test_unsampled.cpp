#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <vector>

#include "bonsai/row_view.hpp"
#include "bonsai/types.hpp"
#include "grower_impl.hpp"

using namespace bonsai; // NOLINT
namespace gd = bonsai::grower_detail;

namespace
{

// Which rows the routing visited. A flag per row rather than a list: the walk
// is parallel and each row writes its own slot.
std::vector<char> visited(RowView const &view, std::vector<row_id_t> const &sampled,
                          size_t plane_rows)
{
    std::vector<char> seen(plane_rows, 0);
    gd::for_each_unsampled(view, sampled, [&](row_id_t r) { seen[r] = 1; });
    return seen;
}

} // namespace

TEST_CASE("for_each_unsampled: a sampled fit routes the whole complement",
          "[grower][unsampled]")
{
    // The identity view is a fit over every row of the plane, so the rows the
    // sampler dropped are exactly the rows still owed this tree's value.
    auto const seen = visited(RowView::all(10), {1, 4, 5, 9}, 10);
    CHECK(seen == std::vector<char>{1, 0, 1, 1, 0, 0, 1, 1, 1, 0});
}

TEST_CASE("for_each_unsampled: nothing is owed when the sampler kept everything",
          "[grower][unsampled]")
{
    auto const seen = visited(RowView::all(6), {0, 1, 2, 3, 4, 5}, 6);
    CHECK(seen == std::vector<char>(6, 0));
}

TEST_CASE("for_each_unsampled: a view routes nothing outside itself",
          "[grower][unsampled][view]")
{
    // The fit is about the view's rows; the rows the view left out are not a
    // complement it owes anything to.
    std::vector<row_id_t> const rows{10, 11, 12, 20, 21};
    RowView const               view = RowView::encode(rows, 64);
    auto const                  seen = visited(view, {10, 12, 21}, 64);

    std::vector<char> want(64, 0);
    want[11] = 1;
    want[20] = 1;
    CHECK(seen == want);
}

TEST_CASE("for_each_unsampled: a view whose rows are all in the fit routes nothing",
          "[grower][unsampled][view]")
{
    // The unsampled default: the row list IS the view, so the walk over the
    // rows outside it is pure waste and does not happen.
    std::vector<row_id_t> const rows{10, 11, 12, 20, 21};
    RowView const               view = RowView::encode(rows, 64);
    CHECK(visited(view, rows, 64) == std::vector<char>(64, 0));
}
