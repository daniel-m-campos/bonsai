#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "bonsai/config/errors.hpp"
#include "bonsai/config/toml.hpp"

TEST_CASE("Toml: parse_toml populates known sections with defaults preserved",
          "[toml][fit]")
{
    constexpr auto kText = R"(
[data]
train = "train.csv"
test = "test.csv"
label_column = 0

[tree]
max_depth = 8
lambda_l2 = 0.5

[booster]
n_iters = 250
learning_rate = 0.1
)";
    auto const     cfg   = bonsai::config::parse_toml(kText);

    REQUIRE(cfg.data.train == "train.csv");
    REQUIRE(cfg.data.test == "test.csv");
    REQUIRE(cfg.data.label_column == 0);
    REQUIRE(cfg.tree_config.max_depth == 8);
    REQUIRE_THAT(cfg.tree_config.lambda_l2, Catch::Matchers::WithinAbs(0.5F, 1e-6F));
    REQUIRE(cfg.booster_config.n_iters == 250);
    REQUIRE_THAT(cfg.booster_config.learning_rate,
                 Catch::Matchers::WithinAbs(0.1F, 1e-6F));
    // Untouched defaults survive.
    REQUIRE(cfg.dispatch.objective_name == "mse");
    REQUIRE(cfg.bin_mapper.max_bin == 255);
}

TEST_CASE("Toml: unknown section is rejected in strict mode", "[toml][edge]")
{
    constexpr auto kText = R"(
[bogus]
foo = 1
)";
    REQUIRE_THROWS_AS(bonsai::config::parse_toml(kText), bonsai::ConfigError);
}

TEST_CASE("Toml: unknown key inside a known section is rejected", "[toml][edge]")
{
    constexpr auto kText = R"(
[tree]
max_depth = 6
bogus_key = 1
)";
    REQUIRE_THROWS_AS(bonsai::config::parse_toml(kText), bonsai::ConfigError);
}

TEST_CASE("Toml: wrong type throws ConfigError", "[toml][edge]")
{
    constexpr auto kText = R"(
[tree]
max_depth = "six"
)";
    REQUIRE_THROWS_AS(bonsai::config::parse_toml(kText), bonsai::ConfigError);
}

TEST_CASE("Overrides: apply dotted keys updates the right fields", "[overrides][fit]")
{
    bonsai::Config                              cfg;
    std::vector<bonsai::config::Override> const ovs = {
        {"tree.max_depth", "10"},          {"booster.n_iters", "5"},
        {"booster.learning_rate", "0.25"}, {"dispatch.objective_name", "logloss"},
        {"data.header", "false"},
    };
    bonsai::config::apply_overrides(cfg, ovs);

    REQUIRE(cfg.tree_config.max_depth == 10);
    REQUIRE(cfg.booster_config.n_iters == 5);
    REQUIRE_THAT(cfg.booster_config.learning_rate,
                 Catch::Matchers::WithinAbs(0.25F, 1e-6F));
    REQUIRE(cfg.dispatch.objective_name == "logloss");
    REQUIRE_FALSE(cfg.data.header);
}

TEST_CASE("Overrides: malformed value throws ConfigError", "[overrides][edge]")
{
    bonsai::Config                              cfg;
    std::vector<bonsai::config::Override> const ovs = {
        {"tree.max_depth", "not_a_number"}};
    REQUIRE_THROWS_AS(bonsai::config::apply_overrides(cfg, ovs), bonsai::ConfigError);
}

TEST_CASE("Overrides: unknown key throws ConfigError", "[overrides][edge]")
{
    bonsai::Config                              cfg;
    std::vector<bonsai::config::Override> const ovs = {{"tree.bogus", "1"}};
    REQUIRE_THROWS_AS(bonsai::config::apply_overrides(cfg, ovs), bonsai::ConfigError);
}

TEST_CASE("Overrides: non-dotted key throws ConfigError", "[overrides][edge]")
{
    bonsai::Config                              cfg;
    std::vector<bonsai::config::Override> const ovs = {{"max_depth", "8"}};
    REQUIRE_THROWS_AS(bonsai::config::apply_overrides(cfg, ovs), bonsai::ConfigError);
}

TEST_CASE("Toml: [metrics] section populates fit and eval lists", "[toml][metrics]")
{
    constexpr auto kText = R"(
[metrics]
fit  = ["rmse", "mae"]
eval = ["rmse", "r2", "mae"]
)";
    auto const     cfg   = bonsai::config::parse_toml(kText);
    REQUIRE(cfg.metrics.fit.size() == 2);
    CHECK(cfg.metrics.fit[0] == "rmse");
    CHECK(cfg.metrics.fit[1] == "mae");
    REQUIRE(cfg.metrics.eval.size() == 3);
    CHECK(cfg.metrics.eval[0] == "rmse");
    CHECK(cfg.metrics.eval[1] == "r2");
    CHECK(cfg.metrics.eval[2] == "mae");
}

TEST_CASE("Toml: booster.log_intervals parses to BoosterConfig",
          "[toml][booster][log_intervals]")
{
    constexpr auto kText = R"(
[booster]
log_intervals = 5
)";
    auto const     cfg   = bonsai::config::parse_toml(kText);
    CHECK(cfg.booster_config.log_intervals == 5);
}

TEST_CASE("Overrides: comma-separated value populates vector<string>",
          "[overrides][metrics]")
{
    bonsai::Config                              cfg;
    std::vector<bonsai::config::Override> const ovs = {{"metrics.eval", "rmse,mae,r2"}};
    bonsai::config::apply_overrides(cfg, ovs);
    REQUIRE(cfg.metrics.eval.size() == 3);
    CHECK(cfg.metrics.eval[0] == "rmse");
    CHECK(cfg.metrics.eval[1] == "mae");
    CHECK(cfg.metrics.eval[2] == "r2");
}

TEST_CASE("Overrides: empty value yields empty vector", "[overrides][metrics]")
{
    bonsai::Config cfg;
    cfg.metrics.fit                                 = {"keep"};
    std::vector<bonsai::config::Override> const ovs = {{"metrics.fit", ""}};
    bonsai::config::apply_overrides(cfg, ovs);
    CHECK(cfg.metrics.fit.empty());
}

TEST_CASE("Overrides: empty token in list rejected", "[overrides][metrics][edge]")
{
    bonsai::Config                              cfg;
    std::vector<bonsai::config::Override> const ovs = {{"metrics.fit", "a,,b"}};
    REQUIRE_THROWS_AS(bonsai::config::apply_overrides(cfg, ovs), bonsai::ConfigError);
}

TEST_CASE("dump_toml: default config contains every section and key landmark",
          "[toml][dump]")
{
    auto const text = bonsai::config::dump_toml(bonsai::Config{});

    CHECK(text.find("[data]") != std::string::npos);
    CHECK(text.find("[bin_mapper]") != std::string::npos);
    CHECK(text.find("[tree]") != std::string::npos);
    CHECK(text.find("[booster]") != std::string::npos);
    CHECK(text.find("[dispatch]") != std::string::npos);
    CHECK(text.find("[metrics]") != std::string::npos);

    CHECK(text.find("n_iters = 100") != std::string::npos);
    CHECK(text.find("max_depth = 6") != std::string::npos);
    CHECK(text.find("fit = []") != std::string::npos);
    CHECK(text.find("eval = []") != std::string::npos);

    REQUIRE_NOTHROW(bonsai::config::parse_toml(text));
}

TEST_CASE("dump_toml: round-trip on default Config", "[toml][dump][roundtrip]")
{
    bonsai::Config const original{};
    auto const           text       = bonsai::config::dump_toml(original);
    auto const           round_trip = bonsai::config::parse_toml(text);

    CHECK(round_trip.data.train == original.data.train);
    CHECK(round_trip.data.test == original.data.test);
    CHECK(round_trip.data.label_column == original.data.label_column);
    CHECK(round_trip.data.header == original.data.header);
    CHECK(round_trip.data.valid == original.data.valid);
    CHECK(round_trip.data.missing_sentinel == original.data.missing_sentinel);

    CHECK(round_trip.tree_config.max_depth == original.tree_config.max_depth);
    CHECK_THAT(round_trip.tree_config.lambda_l2,
               Catch::Matchers::WithinAbs(original.tree_config.lambda_l2, 1e-7F));

    CHECK(round_trip.booster_config.n_iters == original.booster_config.n_iters);
    CHECK_THAT(
        round_trip.booster_config.learning_rate,
        Catch::Matchers::WithinAbs(original.booster_config.learning_rate, 1e-7F));
    CHECK(round_trip.booster_config.random_seed == original.booster_config.random_seed);
    CHECK(round_trip.booster_config.log_intervals ==
          original.booster_config.log_intervals);

    CHECK(round_trip.dispatch.objective_name == original.dispatch.objective_name);
    CHECK(round_trip.dispatch.grower_name == original.dispatch.grower_name);
    CHECK(round_trip.dispatch.sampler_name == original.dispatch.sampler_name);

    CHECK(round_trip.metrics.fit == original.metrics.fit);
    CHECK(round_trip.metrics.eval == original.metrics.eval);
}

TEST_CASE("dump_toml: round-trip on non-default Config exercises every codec type",
          "[toml][dump][roundtrip]")
{
    bonsai::Config original;
    original.data.train            = "train.csv";
    original.data.test             = "test.csv";
    original.data.valid            = {"valid1.csv", "valid2.csv"};
    original.data.header           = false;
    original.data.label_column     = 7;
    original.data.weight_column    = 3;
    original.data.ignore_columns   = {1, 4, 9};
    original.data.missing_sentinel = 0.5F;

    original.tree_config.max_depth        = 10;
    original.tree_config.lambda_l2        = 0.25F;
    original.booster_config.n_iters       = 500;
    original.booster_config.learning_rate = 0.01F;
    original.booster_config.log_intervals = 5;
    original.dispatch.objective_name      = "logloss";
    original.metrics.fit                  = {"logloss", "accuracy"};
    original.metrics.eval                 = {"accuracy"};

    auto const text       = bonsai::config::dump_toml(original);
    auto const round_trip = bonsai::config::parse_toml(text);

    CHECK(round_trip.data.train == original.data.train);
    CHECK(round_trip.data.test == original.data.test);
    CHECK(round_trip.data.valid == original.data.valid);
    CHECK(round_trip.data.header == original.data.header);
    CHECK(round_trip.data.label_column == original.data.label_column);
    CHECK(round_trip.data.weight_column == original.data.weight_column);
    CHECK(round_trip.data.ignore_columns == original.data.ignore_columns);
    REQUIRE(round_trip.data.missing_sentinel.has_value());
    CHECK_THAT(*round_trip.data.missing_sentinel,
               Catch::Matchers::WithinAbs(*original.data.missing_sentinel, 1e-7F));

    CHECK(round_trip.tree_config.max_depth == original.tree_config.max_depth);
    CHECK_THAT(round_trip.tree_config.lambda_l2,
               Catch::Matchers::WithinAbs(original.tree_config.lambda_l2, 1e-7F));

    CHECK(round_trip.booster_config.n_iters == original.booster_config.n_iters);
    CHECK_THAT(
        round_trip.booster_config.learning_rate,
        Catch::Matchers::WithinAbs(original.booster_config.learning_rate, 1e-7F));
    CHECK(round_trip.booster_config.log_intervals ==
          original.booster_config.log_intervals);

    CHECK(round_trip.dispatch.objective_name == original.dispatch.objective_name);
    CHECK(round_trip.metrics.fit == original.metrics.fit);
    CHECK(round_trip.metrics.eval == original.metrics.eval);
}

TEST_CASE("Overrides: parallel.device_id parses to ParallelConfig", "[overrides][fit]")
{
    bonsai::Config                              cfg;
    std::vector<bonsai::config::Override> const ovs = {{"parallel.device_id", "3"}};
    bonsai::config::apply_overrides(cfg, ovs);
    REQUIRE(cfg.parallel.device_id == 3);
    REQUIRE(bonsai::Config{}.parallel.device_id == 0);
}

TEST_CASE("Toml: parallel.device_ids parses from an integer array", "[toml][fit]")
{
    constexpr auto kText = R"(
[parallel]
device_ids = [0, 1, 3]
)";
    auto const     cfg   = bonsai::config::parse_toml(kText);
    REQUIRE(cfg.parallel.device_ids == std::vector<uint32_t>{0, 1, 3});
    // Untouched default is empty.
    REQUIRE(bonsai::Config{}.parallel.device_ids.empty());
}

TEST_CASE("Overrides: parallel.device_ids parses from a comma-separated string",
          "[overrides][fit]")
{
    bonsai::Config                              cfg;
    std::vector<bonsai::config::Override> const ovs = {{"parallel.device_ids", "0,1"}};
    bonsai::config::apply_overrides(cfg, ovs);
    REQUIRE(cfg.parallel.device_ids == std::vector<uint32_t>{0, 1});
}

TEST_CASE("Overrides: parallel.device_ids rejects a non-integer token",
          "[overrides][edge]")
{
    bonsai::Config                              cfg;
    std::vector<bonsai::config::Override> const ovs = {{"parallel.device_ids", "0,x"}};
    REQUIRE_THROWS_AS(bonsai::config::apply_overrides(cfg, ovs), bonsai::ConfigError);
}

TEST_CASE("Overrides: parallel.device_ids rejects a negative token",
          "[overrides][edge]")
{
    bonsai::Config                              cfg;
    std::vector<bonsai::config::Override> const ovs = {{"parallel.device_ids", "0,-1"}};
    REQUIRE_THROWS_AS(bonsai::config::apply_overrides(cfg, ovs), bonsai::ConfigError);
}
