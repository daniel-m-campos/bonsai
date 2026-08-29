#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <limits>
#include <nlohmann/json.hpp>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/io/model.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/registry/make_booster.hpp"
#include "bonsai/registry/names.hpp"
#include "bonsai/registry/typelists.hpp"
#include "bonsai/typelist.hpp"
#include "bonsai/types.hpp"
#include "test_grower_helpers.hpp"

using namespace bonsai; // NOLINT

namespace
{

struct RawFeatures
{
    std::vector<float> data;
    size_t             n_rows;
    size_t             n_features;
    features_view      view() const
    {
        return features_view{data.data(), n_rows, n_features};
    }
};

RawFeatures to_raw(detail::ColumnBatch const &batch)
{
    size_t const       n_features = batch.features.size();
    size_t const       n_rows     = n_features == 0 ? 0 : batch.features[0].size();
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

std::vector<uint8_t> read_file_bytes(std::string const &path)
{
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>{std::istreambuf_iterator<char>(in),
                                std::istreambuf_iterator<char>()};
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
    TempPath   &operator=(TempPath &&)    = delete;
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

template <typename O> detail::ColumnBatch batch_for();

template <> detail::ColumnBatch batch_for<MSEObjective>()
{
    return tiny_batch();
}

template <> detail::ColumnBatch batch_for<LogLossObjective>()
{
    return detail::ColumnBatch{
        .features      = {{0.0F, 0.1F, 0.9F, 1.0F}},
        .labels        = {0.0F, 0.0F, 1.0F, 1.0F},
        .weights       = {},
        .feature_names = {"a"},
    };
}

template <> detail::ColumnBatch batch_for<MAEObjective>()
{
    return batch_for<MSEObjective>();
}

template <> detail::ColumnBatch batch_for<HuberObjective>()
{
    return batch_for<MSEObjective>();
}

template <> detail::ColumnBatch batch_for<QuantileObjective>()
{
    return batch_for<MSEObjective>();
}

template <> detail::ColumnBatch batch_for<PoissonObjective>()
{
    // Non-negative counts (the objective rejects negative labels).
    return detail::ColumnBatch{
        .features      = {{0.0F, 0.1F, 0.9F, 1.0F}},
        .labels        = {0.0F, 1.0F, 3.0F, 5.0F},
        .weights       = {},
        .feature_names = {"a"},
    };
}

template <> detail::ColumnBatch batch_for<SoftmaxObjective>()
{
    // 3-class labels over the same separable feature.
    return detail::ColumnBatch{
        .features      = {{0.0F, 0.1F, 0.5F, 0.6F, 0.9F, 1.0F}},
        .labels        = {0.0F, 0.0F, 1.0F, 1.0F, 2.0F, 2.0F},
        .weights       = {},
        .feature_names = {"a"},
    };
}

using DispatchCombos = cartesian_product_t<Objectives, Growers, Samplers>;

} // namespace

TEMPLATE_LIST_TEST_CASE("ModelIo: save -> load -> predict reproduces predictions",
                        "[model_io][smoke]", DispatchCombos)
{
    test::skip_without_cuda<type_at_t<1, TestType>>();
    using O                   = type_at_t<0, TestType>;
    using G                   = type_at_t<1, TestType>;
    using S                   = type_at_t<2, TestType>;
    auto const        batch   = batch_for<O>();
    BinMappers const  mappers = BinMappers::fit(batch, {});
    Dataset const     train   = Dataset::bin(batch, mappers, {});
    RawFeatures const raw     = to_raw(batch);

    Config cfg                  = tiny_cfg();
    cfg.dispatch.objective_name = std::string{impl_name<O>::value};
    cfg.dispatch.grower_name    = std::string{impl_name<G>::value};
    cfg.dispatch.sampler_name   = std::string{impl_name<S>::value};
    auto booster                = make_booster(cfg);
    for (int i = 0; i < 5; ++i)
    {
        booster->update_one_iter(train);
    }

    std::vector<float> y_before(raw.n_rows);
    booster->predict(raw.view(), y_before);

    TempPath const tmp;
    io::save_booster(*booster, tmp.str(), mappers, cfg);

    auto loaded = io::load_booster(tmp.str());

    REQUIRE(loaded.booster != nullptr);
    REQUIRE(loaded.cfg == cfg);
    REQUIRE(loaded.mappers.size() == mappers.size());
    REQUIRE(loaded.booster->n_iters() == 5);

    std::vector<float> y_after(raw.n_rows);
    loaded.booster->predict(raw.view(), y_after);

    for (size_t i = 0; i < y_before.size(); ++i)
    {
        // Bit-equal after round-trip; predictions are deterministic and
        // we stored every float (init_score, learning_rate, leaf values).
        REQUIRE(y_after[i] == y_before[i]);
    }

    // Split gains round-trip too: importance identical on both sides.
    auto const gain_before = booster->feature_importance(ImportanceType::gain);
    auto const gain_after  = loaded.booster->feature_importance(ImportanceType::gain);
    REQUIRE(gain_before == gain_after);
}

TEST_CASE("ModelIo: full Config round-trips via save/load", "[model_io][config]")
{
    // Exercises every leaf-type conversion the Config macros generate
    // (bool, int, unsigned, float, string, vector<int>, vector<string>,
    // optional<float>). If a new leaf type is added to Config without a
    // matching nlohmann conversion, this test fails to compile or round-trip.
    auto const       batch   = tiny_batch();
    BinMappers const mappers = BinMappers::fit(batch, {});
    Dataset const    train   = Dataset::bin(batch, mappers, {});

    Config cfg     = tiny_cfg();
    auto   booster = make_booster(cfg);
    booster->update_one_iter(train);

    // Populate every non-dispatch field with a non-default value.
    // Dispatch fields must keep matching the trained booster.
    cfg.data.train                    = "train.csv";
    cfg.data.valid                    = {"valid_a.csv", "valid_b.csv"};
    cfg.data.test                     = "test.csv";
    cfg.data.format                   = "tsv";
    cfg.data.header                   = false;
    cfg.data.label_column             = 3;
    cfg.data.weight_column            = 7;
    cfg.data.ignore_columns           = {1, 2, 5};
    cfg.bin_mapper.max_bin            = 127;
    cfg.bin_mapper.n_samples          = 50'000;
    cfg.bin_mapper.seed               = 314159;
    cfg.bin_mapper.min_data_in_bin    = 4;
    cfg.tree_config.min_child_hess    = 2.5F;
    cfg.tree_config.min_gain_to_split = 0.1F;
    cfg.tree_config.lambda_l2         = 0.5F;
    cfg.tree_config.max_depth         = 10;
    cfg.tree_config.min_data_in_leaf  = 30;
    cfg.booster_config.n_iters        = 500;
    cfg.booster_config.learning_rate  = 0.1F;
    cfg.booster_config.random_seed    = 7;
    cfg.booster_config.log_intervals  = 10;
    cfg.metrics.fit                   = {"rmse", "mae"};
    cfg.metrics.eval                  = {"rmse"};

    TempPath const tmp;
    io::save_booster(*booster, tmp.str(), mappers, cfg);
    auto const loaded = io::load_booster(tmp.str());
    REQUIRE(loaded.cfg == cfg);
}

TEST_CASE("ModelIo: committed v7 fixtures re-save byte-identically",
          "[model_io][fixture]")
{
    // Persistence byte-proof. Each fixture was written by the format-v7
    // serializer and committed before the config serializers moved to the
    // section descriptors. load -> save must reproduce the exact bytes:
    // nlohmann sorts object keys, so byte equality is key-set equality. A
    // mismatch here means the persisted JSON key set changed. The regressor
    // covers the single-booster path (init_score); the softmax model covers
    // MulticlassBooster (init_scores).
    for (auto const *name :
         {"fixture_v7_regressor.bonsai", "fixture_v7_softmax.bonsai"})
    {
        CAPTURE(name);
        auto const path     = std::string{BONSAI_TESTS_DATA_DIR} + "/" + name;
        auto const original = read_file_bytes(path);
        REQUIRE_FALSE(original.empty());

        auto loaded = io::load_booster(path);
        REQUIRE(loaded.booster != nullptr);

        TempPath const tmp;
        io::save_booster(*loaded.booster, tmp.str(), loaded.mappers, loaded.cfg);
        auto const resaved = read_file_bytes(tmp.str());

        REQUIRE(resaved == original);
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

namespace
{

// Round-trips the saved file through msgpack, hands the JSON root to
// `corrupt`, and rewrites it: the shape of every load-validation case.
template <typename CorruptF>
void rewrite_corrupted(std::string const &path, CorruptF const &corrupt)
{
    std::ifstream              in(path, std::ios::binary);
    std::vector<uint8_t> const bytes{std::istreambuf_iterator<char>(in),
                                     std::istreambuf_iterator<char>()};
    auto                       root = nlohmann::json::from_msgpack(bytes);
    corrupt(root);
    auto const    out = nlohmann::json::to_msgpack(root);
    std::ofstream o(path, std::ios::binary);
    o.write(reinterpret_cast<char const *>(out.data()),
            static_cast<std::streamsize>(out.size()));
}

} // namespace

TEST_CASE("ModelIo: corrupt tree shapes refuse to load", "[model_io][edge]")
{
    auto const       batch   = batch_for<MSEObjective>();
    BinMappers const mappers = BinMappers::fit(batch, {});
    Dataset const    train   = Dataset::bin(batch, mappers, {});
    Config const     cfg     = tiny_cfg();

    Booster<MSEObjective, DepthwiseGrower<CpuHistogramEngine>, AllRowsSampler> booster{
        cfg};
    booster.update_one_iter(train);

    SECTION("dense node link out of range")
    {
        TempPath const tmp;
        io::save_booster(booster, tmp.str(), mappers, cfg);
        rewrite_corrupted(tmp.str(),
                          [](nlohmann::json &root)
                          {
                              auto &node         = root["trees"][0]["nodes"][0];
                              node["feature_id"] = 0;
                              node["left"]       = 1U << 20;
                          });
        REQUIRE_THROWS_WITH(io::load_booster(tmp.str()),
                            Catch::Matchers::ContainsSubstring("point forward"));
    }

    SECTION("dense depth beyond the node count")
    {
        TempPath const tmp;
        io::save_booster(booster, tmp.str(), mappers, cfg);
        rewrite_corrupted(tmp.str(), [](nlohmann::json &root)
                          { root["trees"][0]["depth"] = 1U << 20; });
        REQUIRE_THROWS_WITH(io::load_booster(tmp.str()),
                            Catch::Matchers::ContainsSubstring("node count"));
    }

    SECTION("a depth disagreeing with the stored structure is refused")
    {
        TempPath const tmp;
        io::save_booster(booster, tmp.str(), mappers, cfg);
        rewrite_corrupted(tmp.str(),
                          [](nlohmann::json &root) { root["trees"][0]["depth"] = 0; });
        REQUIRE_THROWS_WITH(io::load_booster(tmp.str()),
                            Catch::Matchers::ContainsSubstring("disagrees"));
    }

    SECTION("a NaN threshold is refused")
    {
        TempPath const tmp;
        io::save_booster(booster, tmp.str(), mappers, cfg);
        rewrite_corrupted(tmp.str(),
                          [](nlohmann::json &root)
                          {
                              auto &node         = root["trees"][0]["nodes"][0];
                              node["feature_id"] = 0;
                              node["left"]       = 1;
                              node["right"]      = 2;
                              node["threshold_or_value"] =
                                  std::numeric_limits<double>::quiet_NaN();
                          });
        REQUIRE_THROWS_WITH(io::load_booster(tmp.str()),
                            Catch::Matchers::ContainsSubstring("not a number"));
    }

    SECTION("a covers array shorter than the nodes is refused")
    {
        TempPath const tmp;
        io::save_booster(booster, tmp.str(), mappers, cfg);
        rewrite_corrupted(tmp.str(), [](nlohmann::json &root)
                          { root["trees"][0]["covers"] = {1.0, 2.0}; });
        REQUIRE_THROWS_WITH(io::load_booster(tmp.str()),
                            Catch::Matchers::ContainsSubstring("covers length"));
    }

    SECTION("a backward link cannot smuggle a cycle")
    {
        TempPath const tmp;
        io::save_booster(booster, tmp.str(), mappers, cfg);
        rewrite_corrupted(tmp.str(),
                          [](nlohmann::json &root)
                          {
                              auto &node         = root["trees"][0]["nodes"][0];
                              node["feature_id"] = 0;
                              node["left"]       = 0;
                          });
        REQUIRE_THROWS_WITH(io::load_booster(tmp.str()),
                            Catch::Matchers::ContainsSubstring("point forward"));
    }

    SECTION("a split naming a missing feature is refused")
    {
        TempPath const tmp;
        io::save_booster(booster, tmp.str(), mappers, cfg);
        rewrite_corrupted(tmp.str(),
                          [](nlohmann::json &root)
                          {
                              auto &node         = root["trees"][0]["nodes"][0];
                              node["feature_id"] = 400000000;
                              node["left"]       = 1;
                              node["right"]      = 2;
                          });
        REQUIRE_THROWS_WITH(
            io::load_booster(tmp.str()),
            Catch::Matchers::ContainsSubstring("feature the model lacks"));
    }

    SECTION("oblivious leaf table disagrees with its depth")
    {
        Config lvl_cfg               = cfg;
        lvl_cfg.dispatch.grower_name = "levelwise";
        Booster<MSEObjective,
                ObliviousGrower<CpuHistogramEngine, HistogramLevelSplitFinder>,
                AllRowsSampler>
            oblivious{lvl_cfg};
        oblivious.update_one_iter(train);
        TempPath const tmp;
        io::save_booster(oblivious, tmp.str(), mappers, lvl_cfg);
        rewrite_corrupted(tmp.str(), [](nlohmann::json &root)
                          { root["trees"][0]["leaves"].erase(0); });
        REQUIRE_THROWS_WITH(io::load_booster(tmp.str()),
                            Catch::Matchers::ContainsSubstring("disagrees"));
    }
}
