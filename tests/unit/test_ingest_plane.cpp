#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"

using namespace bonsai; // NOLINT

namespace
{

// A host-side stand-in for a backend plane: serves canned columns and
// counts materializations, so laziness is observable without a device.
struct FakePlane final : IngestPlane
{
    std::vector<std::vector<uint8_t>> columns;
    mutable int                       materialized = 0;

    void materialize(BinColumns &cols) const override
    {
        std::get<U8Columns>(cols) = columns;
        ++materialized;
    }
};

detail::ColumnBatch two_column_batch()
{
    detail::ColumnBatch batch;
    batch.features = {{0.1F, 0.9F, 0.5F, 0.3F}, {4.0F, 1.0F, 2.0F, 3.0F}};
    batch.labels   = {0.0F, 1.0F, 0.0F, 1.0F};
    return batch;
}

} // namespace

TEST_CASE("plane-backed dataset materializes host bins lazily", "[dataset]")
{
    auto const batch   = two_column_batch();
    auto const mappers = BinMappers::fit(batch, BinMapperConfig{});
    auto       plane   = std::make_shared<FakePlane>();
    // Serve exactly what the host fill would produce, via a host-binned twin.
    auto const twin = Dataset::bin(batch, mappers, DataConfig{});
    for (size_t f = 0; f < twin.n_features(); ++f)
    {
        twin.visit_bins(f, [&](auto bins)
                        { plane->columns.emplace_back(bins.begin(), bins.end()); });
    }

    auto const ds = Dataset::bin(batch, mappers, DataConfig{}, plane);
    REQUIRE(ds.ingest_plane() == plane);
    REQUIRE(ds.n_features() == 2);
    REQUIRE(ds.n_rows() == 4);
    REQUIRE(plane->materialized == 0); // metadata reads stay lazy

    SECTION("bin_at triggers one materialization and matches the host fill")
    {
        for (size_t f = 0; f < 2; ++f)
        {
            for (size_t r = 0; r < 4; ++r)
            {
                REQUIRE(ds.bin_at(f, r) == twin.bin_at(f, r));
            }
        }
        REQUIRE(plane->materialized == 1);
    }

    SECTION("visit_bins and row_major_bins agree with the host fill")
    {
        ds.visit_bins(1, [&](auto bins) { REQUIRE(bins[0] == twin.bin_at(1, 0)); });
        REQUIRE(ds.mirror().bins().size() == twin.mirror().bins().size());
        REQUIRE(plane->materialized == 1);
    }
}

namespace
{

// A plane that CAN rewrite its own columns, which is what a device backend
// does. Counts both routes so a test can prove which one ran: the whole point
// of the seam is that a backend with its own gather is never asked to bring
// the plane home.
struct GatheringPlane final : IngestPlane
{
    std::vector<std::vector<uint8_t>> columns;
    mutable int                       materialized = 0;
    mutable int                       selected     = 0;

    void materialize(BinColumns &cols) const override
    {
        std::get<U8Columns>(cols) = columns;
        ++materialized;
    }

    std::shared_ptr<IngestPlane const>
    select_columns(std::span<feature_id_t const> keep,
                   std::span<row_id_t const>     rows) const override
    {
        ++selected;
        auto out = std::make_shared<GatheringPlane>();
        for (feature_id_t const f : keep)
        {
            std::vector<uint8_t> col;
            if (rows.empty())
            {
                col = columns[f];
            }
            else
            {
                for (row_id_t const r : rows)
                {
                    col.push_back(columns[f][r]);
                }
            }
            out->columns.push_back(std::move(col));
        }
        return out;
    }
};

detail::ColumnBatch three_column_batch()
{
    detail::ColumnBatch batch;
    batch.features = {
        {0.1F, 0.9F, 0.5F, 0.3F}, {4.0F, 1.0F, 2.0F, 3.0F}, {2.0F, 2.0F, 9.0F, 0.0F}};
    batch.labels = {0.0F, 1.0F, 0.0F, 1.0F};
    return batch;
}

// The columns a host-binned twin holds, in plane order.
std::vector<std::vector<uint8_t>> host_columns(Dataset const &twin)
{
    std::vector<std::vector<uint8_t>> out;
    for (size_t f = 0; f < twin.n_features(); ++f)
    {
        twin.visit_bins(f,
                        [&](auto bins) { out.emplace_back(bins.begin(), bins.end()); });
    }
    return out;
}

} // namespace

TEST_CASE("a plane that gathers its own columns is never brought home", "[dataset]")
{
    auto const batch   = three_column_batch();
    auto const mappers = BinMappers::fit(batch, BinMapperConfig{});
    auto const twin    = Dataset::bin(batch, mappers, DataConfig{});
    auto       plane   = std::make_shared<GatheringPlane>();
    plane->columns     = host_columns(twin);

    auto const                ds   = Dataset::bin(batch, mappers, DataConfig{}, plane);
    std::vector<feature_id_t> keep = {2, 0};
    auto const                sub  = ds.select_features(keep);

    CHECK(plane->selected == 1);
    // The claim: a column rewrite of a plane-backed dataset costs no
    // materialization, which is the round trip the device path exists to skip.
    CHECK(plane->materialized == 0);
    REQUIRE(sub.n_features() == 2);
    REQUIRE(sub.n_rows() == 4);
    for (size_t r = 0; r < 4; ++r)
    {
        CHECK(sub.bin_at(0, r) == twin.bin_at(2, r));
        CHECK(sub.bin_at(1, r) == twin.bin_at(0, r));
    }
}

TEST_CASE("a gathering plane is handed the view's rows, not the plane's", "[dataset]")
{
    auto const batch   = three_column_batch();
    auto const mappers = BinMappers::fit(batch, BinMapperConfig{});
    auto const twin    = Dataset::bin(batch, mappers, DataConfig{});
    auto       plane   = std::make_shared<GatheringPlane>();
    plane->columns     = host_columns(twin);

    auto const                  ds  = Dataset::bin(batch, mappers, DataConfig{}, plane);
    std::vector<row_id_t> const ids = {3, 1};
    auto const                viewed = ds.with_rows(RowView::encode(ids, ds.n_rows()));
    std::vector<feature_id_t> keep   = {1};
    auto const                sub    = viewed.select_features(keep);

    CHECK(plane->materialized == 0);
    REQUIRE(sub.n_rows() == 2);
    REQUIRE(sub.n_features() == 1);
    // Rows renumber from zero in the order the view names them.
    CHECK(sub.bin_at(0, 0) == twin.bin_at(1, 3));
    CHECK(sub.bin_at(0, 1) == twin.bin_at(1, 1));
    CHECK(sub.labels()[0] == batch.labels[3]);
    CHECK(sub.labels()[1] == batch.labels[1]);
}

TEST_CASE("a plane that declines falls back to the host gather", "[dataset]")
{
    auto const batch   = three_column_batch();
    auto const mappers = BinMappers::fit(batch, BinMapperConfig{});
    auto const twin    = Dataset::bin(batch, mappers, DataConfig{});
    auto       plane   = std::make_shared<FakePlane>(); // no select_columns override
    plane->columns     = host_columns(twin);

    auto const                ds   = Dataset::bin(batch, mappers, DataConfig{}, plane);
    std::vector<feature_id_t> keep = {2, 0};
    auto const                sub  = ds.select_features(keep);

    // The fallback is the whole reason the base returns null rather than
    // asserting: the result must be identical, only slower.
    CHECK(plane->materialized == 1);
    REQUIRE(sub.n_features() == 2);
    for (size_t r = 0; r < 4; ++r)
    {
        CHECK(sub.bin_at(0, r) == twin.bin_at(2, r));
        CHECK(sub.bin_at(1, r) == twin.bin_at(0, r));
    }
}
