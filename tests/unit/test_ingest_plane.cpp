#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <memory>
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

    void materialize(std::vector<std::vector<uint8_t>> &u8,
                     std::vector<std::vector<uint16_t>> & /*u16*/) const override
    {
        u8 = columns;
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
        REQUIRE(ds.row_major_bins().size() == twin.row_major_bins().size());
        REQUIRE(plane->materialized == 1);
    }
}
