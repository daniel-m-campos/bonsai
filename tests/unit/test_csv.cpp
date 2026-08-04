#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/io/csv.hpp"

namespace
{

std::string tiny_csv()
{
    return std::string{BONSAI_TESTS_DATA_DIR} + "/tiny.csv";
}

bonsai::Config base_config()
{
    bonsai::Config cfg;
    cfg.data.header                = true;
    cfg.data.label_column          = 0;
    cfg.bin_mapper.max_bin         = 16;
    cfg.bin_mapper.n_samples       = 1000;
    cfg.bin_mapper.min_data_in_bin = 1;
    return cfg;
}

} // namespace

TEST_CASE("Csv: parse populates ColumnBatch with header names and labels", "[csv][fit]")
{
    bonsai::DataConfig data_cfg;
    data_cfg.header       = true;
    data_cfg.label_column = 0;

    auto const batch = bonsai::detail::csv::parse(tiny_csv(), data_cfg);

    REQUIRE(batch.features.size() == 3);
    REQUIRE(batch.feature_names.size() == 3);
    REQUIRE(batch.feature_names[0] == "f1");
    REQUIRE(batch.feature_names[1] == "f2");
    REQUIRE(batch.feature_names[2] == "f3");

    REQUIRE(batch.labels.size() == 4);
    REQUIRE_THAT(batch.labels[0], Catch::Matchers::WithinAbs(0.5F, 1e-6F));
    REQUIRE_THAT(batch.labels[3], Catch::Matchers::WithinAbs(3.5F, 1e-6F));
    REQUIRE(batch.weights.empty());

    // Each feature column has n_rows entries.
    for (auto const &col : batch.features)
    {
        REQUIRE(col.size() == 4);
    }
}

TEST_CASE("Csv: nan literals become NaN", "[csv][nan]")
{
    bonsai::DataConfig data_cfg;
    data_cfg.header       = true;
    data_cfg.label_column = 0;

    auto const batch = bonsai::detail::csv::parse(tiny_csv(), data_cfg);

    // f2 column, row 1 (0-indexed) and f1 column, row 2 hold "nan".
    REQUIRE(std::isnan(batch.features[1][1]));
    REQUIRE(std::isnan(batch.features[0][2]));
    // Other entries are finite.
    REQUIRE(std::isfinite(batch.features[0][0]));
    REQUIRE(std::isfinite(batch.features[2][3]));
}

TEST_CASE("Csv: an empty field is a parse error naming its position",
          "[csv][nan][edge]")
{
    // Missing is spelled `nan`. An empty field could be an intended gap or a
    // truncated line, so the reader refuses instead of guessing, and says
    // where to look.
    // Written to the temp dir, not the fixture dir: tests/data holds
    // committed inputs and is restored from a CI cache.
    auto const path =
        (std::filesystem::temp_directory_path() / "bonsai-empty-field.csv").string();
    {
        std::ofstream out(path);
        out << "label,f1,f2\n1.0,2.0,3.0\n2.0,,4.0\n";
    }
    bonsai::DataConfig data_cfg;
    data_cfg.header       = true;
    data_cfg.label_column = 0;

    REQUIRE_THROWS_WITH(bonsai::detail::csv::parse(path, data_cfg),
                        Catch::Matchers::ContainsSubstring("row 2") &&
                            Catch::Matchers::ContainsSubstring("column 2") &&
                            Catch::Matchers::ContainsSubstring("'nan'"));
    std::filesystem::remove(path);
}

TEST_CASE("Csv: read_csv pipes through fit_from_csv to a usable Dataset",
          "[csv][fit][smoke]")
{
    auto       cfg     = base_config();
    auto const mappers = bonsai::io::fit_from_csv(tiny_csv(), cfg);
    auto const dataset = bonsai::io::read_csv(tiny_csv(), cfg.data, mappers);

    REQUIRE(dataset.n_rows() == 4);
    REQUIRE(dataset.n_features() == 3);
    REQUIRE(dataset.labels().size() == 4);
    REQUIRE(mappers.size() == 3);
}

TEST_CASE("Csv: ignore_columns drops the column from features", "[csv][fit]")
{
    bonsai::DataConfig data_cfg;
    data_cfg.header         = true;
    data_cfg.label_column   = 0;
    data_cfg.ignore_columns = {2}; // drop f2

    auto const batch = bonsai::detail::csv::parse(tiny_csv(), data_cfg);
    REQUIRE(batch.feature_names.size() == 2);
    REQUIRE(batch.feature_names[0] == "f1");
    REQUIRE(batch.feature_names[1] == "f3");
}
