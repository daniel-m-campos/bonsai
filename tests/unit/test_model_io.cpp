#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/io/model.hpp"
#include "bonsai/registry/make_booster.hpp"
#include "bonsai/types.hpp"

using namespace bonsai; // NOLINT

namespace
{

struct RawFeatures
{
    std::vector<float> data;
    size_t n_rows;
    size_t n_features;
    features_view view() const
    {
        return features_view{data.data(), n_rows, n_features};
    }
};

RawFeatures to_raw(detail::ColumnBatch const &batch)
{
    size_t const n_features = batch.features.size();
    size_t const n_rows     = n_features == 0 ? 0 : batch.features[0].size();
    std::vector<float> data(n_rows * n_features);
    for (size_t f = 0; f < n_features; ++f)
    {
        for (size_t r = 0; r < n_rows; ++r)
        {
            data[(r * n_features) + f] = batch.features[f][r];
        }
    }
    return RawFeatures{
        .data = std::move(data), .n_rows = n_rows, .n_features = n_features};
}

struct TempPath
{
    std::filesystem::path path;
    TempPath()
        : path(std::filesystem::temp_directory_path() /
               (std::string{"bonsai_model_"} +
                std::to_string(std::hash<TempPath const *>{}(this)) + ".msgpack"))
    {
    }
    ~TempPath()
    {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    TempPath(TempPath const &)            = delete;
    TempPath &operator=(TempPath const &) = delete;
    TempPath(TempPath &&)                 = delete;
    TempPath &operator=(TempPath &&)      = delete;
    std::string str() const
    {
        return path.string();
    }
};

Config tiny_cfg()
{
    Config cfg{};
    cfg.tree_config.min_data_in_leaf = 0;
    cfg.tree_config.min_child_hess   = 0.0F;
    cfg.tree_config.max_depth        = 2;
    cfg.booster_config.learning_rate = 0.3F;
    return cfg;
}

detail::ColumnBatch tiny_batch()
{
    return detail::ColumnBatch{
        .features      = {{0.0F, 0.1F, 0.9F, 1.0F}},
        .labels        = {-1.0F, -1.0F, +1.0F, +1.0F},
        .weights       = {},
        .feature_names = {"a"},
    };
}

} // namespace

TEST_CASE("ModelIo: save -> load -> predict reproduces predictions",
          "[model_io][smoke]")
{
    auto const batch         = tiny_batch();
    BinMappers const mappers = BinMappers::fit(batch, {});
    Dataset const train      = Dataset::bin(batch, mappers, {});
    RawFeatures const raw    = to_raw(batch);

    Config const cfg = tiny_cfg();
    auto booster     = make_booster(cfg);
    for (int i = 0; i < 5; ++i)
    {
        booster->update_one_iter(train);
    }

    std::vector<float> y_before(raw.n_rows);
    booster->predict(raw.view(), y_before);

    TempPath const tmp;
    io::save_booster(*booster, tmp.str(), mappers, cfg.dispatch,
                     cfg.booster_config.learning_rate);

    auto loaded = io::load_booster(tmp.str());

    REQUIRE(loaded.booster != nullptr);
    REQUIRE(loaded.dispatch.objective_name == cfg.dispatch.objective_name);
    REQUIRE(loaded.mappers.size() == mappers.size());
    REQUIRE(loaded.learning_rate ==
            Catch::Approx(cfg.booster_config.learning_rate).epsilon(1e-6));
    REQUIRE(loaded.booster->n_iters() == 5);

    std::vector<float> y_after(raw.n_rows);
    loaded.booster->predict(raw.view(), y_after);

    for (size_t i = 0; i < y_before.size(); ++i)
    {
        // Bit-equal after round-trip; predictions are deterministic and
        // we stored every float (init_score, learning_rate, leaf values).
        REQUIRE(y_after[i] == y_before[i]);
    }
}

TEST_CASE("ModelIo: bad magic throws", "[model_io][edge]")
{
    TempPath const tmp;
    {
        FILE *fp = std::fopen(tmp.str().c_str(), "wb");
        REQUIRE(fp != nullptr);
        char const garbage[] = "not-a-bonsai-model";
        std::fwrite(garbage, 1, sizeof(garbage), fp);
        std::fclose(fp);
    }
    REQUIRE_THROWS(io::load_booster(tmp.str()));
}
