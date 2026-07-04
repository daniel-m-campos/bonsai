#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "bonsai/cli/pipeline.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/io/model.hpp"

using namespace bonsai;      // NOLINT
using namespace bonsai::cli; // NOLINT

namespace
{

std::string const k_tiny_path = std::string{BONSAI_TESTS_DATA_DIR} + "/tiny.csv";

Config make_tiny_config()
{
    Config cfg;
    cfg.data.train             = k_tiny_path;
    cfg.bin_mapper.max_bin     = 8;
    cfg.bin_mapper.n_samples   = 100;
    cfg.booster_config.n_iters = 5;
    return cfg;
}

} // namespace

TEST_CASE("load_train_from_csv: tiny.csv yields N rows and non-empty mappers",
          "[cli_pipeline][load]")
{
    auto const cfg    = make_tiny_config();
    auto const loaded = load_train_from_csv(cfg, cfg.data.train);

    CHECK(loaded.train.n_rows() == 4);
    CHECK(loaded.train.n_features() > 0);
    CHECK(loaded.mappers.size() > 0);
}

TEST_CASE("train_in_memory: progress callback fires n_iters times in 1..n order",
          "[cli_pipeline][train]")
{
    auto const cfg    = make_tiny_config();
    auto const loaded = load_train_from_csv(cfg, cfg.data.train);

    std::vector<size_t> iters;
    auto booster = train_in_memory(cfg, loaded.train, [&](size_t iter, size_t /*total*/)
                                   { iters.push_back(iter); });

    REQUIRE(iters.size() == cfg.booster_config.n_iters);
    for (size_t i = 0; i < iters.size(); ++i)
    {
        CHECK(iters[i] == i + 1);
    }
    CHECK(booster->n_iters() == cfg.booster_config.n_iters);
}

TEST_CASE("score_csv: returns one raw score per row", "[cli_pipeline][score]")
{
    auto const cfg     = make_tiny_config();
    auto const loaded  = load_train_from_csv(cfg, cfg.data.train);
    auto const booster = train_in_memory(cfg, loaded.train);

    auto const scored = score_csv(*booster, cfg.data.train, cfg.data);
    CHECK(scored.raw_scores.size() == 4);
}

TEST_CASE("score_and_label_csv: labels match the CSV's label column",
          "[cli_pipeline][score]")
{
    auto const cfg     = make_tiny_config();
    auto const loaded  = load_train_from_csv(cfg, cfg.data.train);
    auto const booster = train_in_memory(cfg, loaded.train);

    auto const sl = score_and_label_csv(*booster, cfg.data.train, cfg.data);
    REQUIRE(sl.raw_scores.size() == 4);
    REQUIRE(sl.labels.size() == 4);
    CHECK(sl.labels[0] == Catch::Approx(0.5F));
    CHECK(sl.labels[1] == Catch::Approx(1.5F));
    CHECK(sl.labels[2] == Catch::Approx(2.5F));
    CHECK(sl.labels[3] == Catch::Approx(3.5F));
}

TEST_CASE("train_with_progress: early stopping truncates to the best iteration",
          "[cli_pipeline][early_stop]")
{
    // Train and validate on the same tiny file: valid loss converges after a
    // couple of trees, so a small patience must stop well before n_iters and
    // the kept model must have best_iter + 1 <= n_iters trees.
    Config cfg;
    cfg.data.train                           = k_tiny_path;
    cfg.data.valid                           = {k_tiny_path};
    cfg.bin_mapper.max_bin                   = 8;
    cfg.bin_mapper.n_samples                 = 100;
    cfg.booster_config.n_iters               = 200;
    cfg.booster_config.learning_rate         = 0.5F;
    cfg.booster_config.early_stopping_rounds = 3;
    cfg.tree_config.min_data_in_leaf         = 0;
    cfg.tree_config.min_child_hess           = 0.0F;

    auto const loaded  = load_train_and_valid_from_csv(cfg);
    auto       booster = train_with_progress(cfg, loaded, {});
    CHECK(booster->n_iters() < 200);

    Config no_es                               = cfg;
    no_es.booster_config.early_stopping_rounds = 0;
    auto const loaded2                         = load_train_and_valid_from_csv(no_es);
    auto       full = train_with_progress(no_es, loaded2, {});
    CHECK(full->n_iters() == 200);
}

TEST_CASE("train_with_progress: warm start continues a saved model",
          "[cli_pipeline][warm_start]")
{
    Config cfg;
    cfg.data.train                   = k_tiny_path;
    cfg.bin_mapper.max_bin           = 8;
    cfg.bin_mapper.n_samples         = 100;
    cfg.booster_config.n_iters       = 6;
    cfg.booster_config.learning_rate = 0.3F;
    cfg.tree_config.min_data_in_leaf = 0;
    cfg.tree_config.min_child_hess   = 0.0F;

    // Reference: 6 iterations straight through.
    auto const loaded_a = load_train_and_valid_from_csv(cfg);
    auto       straight = train_with_progress(cfg, loaded_a, {});

    // Warm start: 3 iterations, save, reload, 3 more.
    Config half                 = cfg;
    half.booster_config.n_iters = 3;
    auto const loaded_b         = load_train_and_valid_from_csv(half);
    auto       first            = train_with_progress(half, loaded_b, {});
    auto const tmp = std::string{BONSAI_TESTS_DATA_DIR} + "/_warm_start_tmp.msgpack";
    io::save_booster(*first, tmp, loaded_b.mappers, half);

    auto       reloaded = io::load_booster(tmp);
    auto const loaded_c =
        load_train_and_valid_with_mappers(half, std::move(reloaded.mappers));
    auto continued =
        train_with_progress(half, loaded_c, {}, std::move(reloaded.booster));
    std::remove(tmp.c_str());

    REQUIRE(continued->n_iters() == 6);

    // FP grouping differs when scores are rebuilt (sum-then-scale vs
    // incremental), so compare within tolerance, not bytes.
    std::vector<float> ps(loaded_a.train.features.n_rows);
    std::vector<float> pc(loaded_a.train.features.n_rows);
    straight->predict(loaded_a.train.features.view(), ps);
    continued->predict(loaded_a.train.features.view(), pc);
    for (size_t i = 0; i < ps.size(); ++i)
    {
        CHECK(pc[i] == Catch::Approx(ps[i]).margin(1e-4));
    }
}
