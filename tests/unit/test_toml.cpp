#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "bonsai/config/errors.hpp"
#include "bonsai/config/internal/codec.hpp"
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

TEST_CASE("Toml: the retired missing_sentinel knob is an unknown key", "[toml][edge]")
{
    // Sentinel-to-NaN conversion is preprocessing, not a knob. A config that
    // still sets it must fail loudly rather than be read and ignored.
    constexpr auto kText = R"(
[data]
missing_sentinel = -999.0
)";
    REQUIRE_THROWS_AS(bonsai::config::parse_toml(kText), bonsai::ConfigError);

    bonsai::Config                              cfg;
    std::vector<bonsai::config::Override> const ovs = {
        {"data.missing_sentinel", "-999.0"}};
    REQUIRE_THROWS_AS(bonsai::config::apply_overrides(cfg, ovs), bonsai::ConfigError);
}

TEST_CASE("Toml: wrong type throws ConfigError", "[toml][edge]")
{
    constexpr auto kText = R"(
[tree]
max_depth = "six"
)";
    REQUIRE_THROWS_AS(bonsai::config::parse_toml(kText), bonsai::ConfigError);
}

TEST_CASE("Toml: a bad list names the item and the type it expected", "[toml][edge]")
{
    SECTION("an item of the wrong type")
    {
        constexpr auto kText = R"(
[tree]
monotone_constraints = [1, "a"]
)";
        REQUIRE_THROWS_WITH(bonsai::config::parse_toml(kText),
                            Catch::Matchers::ContainsSubstring(
                                "item 1: wrong type (expected integer)"));
    }
    SECTION("a scalar in place of the array")
    {
        constexpr auto kText = R"(
[tree]
monotone_constraints = "abc"
)";
        REQUIRE_THROWS_WITH(bonsai::config::parse_toml(kText),
                            Catch::Matchers::ContainsSubstring("must be an array"));
    }
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

namespace
{
char const *set_comma_decimal_locale()
{
    for (char const *name : {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8"})
    {
        if (std::setlocale(LC_NUMERIC, name) != nullptr)
        {
            return name;
        }
    }
    return nullptr;
}
} // namespace

TEST_CASE("Overrides: a float parses the same under a comma-decimal locale",
          "[overrides][edge]")
{
    std::string const previous = std::setlocale(LC_NUMERIC, nullptr);
    if (set_comma_decimal_locale() == nullptr)
    {
        SKIP("no comma-decimal locale installed");
    }
    bonsai::Config                              cfg;
    std::vector<bonsai::config::Override> const ovs = {
        {"booster.learning_rate", "0.25"}};
    auto const apply = [&] { bonsai::config::apply_overrides(cfg, ovs); };
    CHECK_NOTHROW(apply());
    std::setlocale(LC_NUMERIC, previous.c_str());
    CHECK(cfg.booster_config.learning_rate == 0.25F);
}

namespace
{
float min_gain_from(std::string const &spelling)
{
    bonsai::Config cfg;
    bonsai::config::apply_overrides(cfg, {{"tree.min_gain_to_split", spelling}});
    return cfg.tree_config.min_gain_to_split;
}
} // namespace

TEST_CASE("Overrides: a float beyond the float range reads as strtof reads it",
          "[overrides][edge]")
{
    constexpr float kInf = std::numeric_limits<float>::infinity();
    SECTION("underflow keeps the sign of zero")
    {
        CHECK(min_gain_from("1e-50") == 0.0F);
        CHECK_FALSE(std::signbit(min_gain_from("1e-50")));
        CHECK(std::signbit(min_gain_from("-1e-50")));
        CHECK(min_gain_from("1e-400") == 0.0F);
    }
    SECTION("overflow keeps the sign of infinity, which a finite key refuses")
    {
        using bonsai::config::internal::read_float_from_string;
        CHECK(read_float_from_string("1e+40").value() == kInf);
        CHECK(read_float_from_string("-1e+40").value() == -kInf);
        CHECK(read_float_from_string("1e+400").value() == kInf);
        REQUIRE_THROWS_WITH(min_gain_from("1e+40"), Catch::Matchers::ContainsSubstring(
                                                        "must be finite (got inf)"));
    }
    SECTION("a subnormal stays subnormal")
    {
        CHECK(min_gain_from("1e-45") == std::numeric_limits<float>::denorm_min());
    }
}

TEST_CASE("Overrides: a float accepts a leading sign, leading space and hex",
          "[overrides][edge]")
{
    SECTION("accepted spellings")
    {
        CHECK(min_gain_from("+0.1") == 0.1F);
        CHECK(min_gain_from(" \t0.1") == 0.1F);
        CHECK(min_gain_from("0x1p-3") == 0.125F);
        CHECK(min_gain_from("-0X1p-3") == -0.125F);
    }
    SECTION("refused spellings")
    {
        for (std::string const spelling : {"+-1", "--1", "0x-1", "0.1 ", "0x", "+"})
        {
            CAPTURE(spelling);
            CHECK_THROWS_AS(min_gain_from(spelling), bonsai::ConfigError);
        }
    }
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

TEST_CASE("Config codec: an integer past its field's width is refused",
          "[toml][overrides][edge]")
{
    using Catch::Matchers::ContainsSubstring;
    bonsai::Config cfg;
    SECTION("an override one past a u8 field")
    {
        REQUIRE_THROWS_WITH(
            bonsai::config::apply_overrides(cfg, {{"tree.min_data_in_leaf", "256"}}),
            ContainsSubstring(
                "[tree].min_data_in_leaf: must be at most 255 (got 256)"));
    }
    SECTION("an override one past a u32 field")
    {
        REQUIRE_THROWS_WITH(
            bonsai::config::apply_overrides(cfg,
                                            {{"booster.random_seed", "4294967338"}}),
            ContainsSubstring("[booster].random_seed: must be at most 4294967295"));
    }
    SECTION("a TOML value past an int field")
    {
        REQUIRE_THROWS_WITH(
            bonsai::config::parse_toml("[data]\nlabel_column = 2147483648\n"),
            ContainsSubstring("[data].label_column: must be at most 2147483647"));
    }
    SECTION("a TOML value below an int field")
    {
        REQUIRE_THROWS_WITH(
            bonsai::config::parse_toml("[data]\nweight_column = -2147483649\n"),
            ContainsSubstring("[data].weight_column: must be at least -2147483648"));
    }
    SECTION("the widest value a field holds is kept")
    {
        bonsai::config::apply_overrides(cfg, {{"tree.max_depth", "255"}});
        REQUIRE(cfg.tree_config.max_depth == 255);
    }
}

TEST_CASE("Config bounds: a value outside its key's domain is refused",
          "[toml][overrides][edge][nan]")
{
    using Catch::Matchers::ContainsSubstring;
    struct Case
    {
        char const *key;
        char const *value;
        char const *message;
    };
    auto const [key, value, message] = GENERATE(
        Case{"booster.learning_rate", "nan",
             "[booster].learning_rate: must be finite and > 0 (got nan)"},
        Case{"booster.learning_rate", "-0.5", "must be finite and > 0 (got -0.5)"},
        Case{"booster.learning_rate", "0", "must be finite and > 0 (got 0)"},
        Case{"booster.learning_rate", "1e39", "must be finite and > 0 (got inf)"},
        Case{"booster.dart_drop_rate", "1.5",
             "[booster].dart_drop_rate: must be in [0, 1] (got 1.5)"},
        Case{"tree.lambda_l2", "-0.99", "[tree].lambda_l2: must be finite and >= 0"},
        Case{"tree.lambda_l2", "nan", "[tree].lambda_l2: must be finite and >= 0"},
        Case{"tree.lambda_l1", "-500", "[tree].lambda_l1: must be finite and >= 0"},
        Case{"tree.min_gain_to_split", "nan",
             "[tree].min_gain_to_split: must be finite (got nan)"},
        Case{"tree.feature_fraction", "0",
             "[tree].feature_fraction: must be in (0, 1] (got 0)"},
        Case{"tree.feature_fraction", "2", "must be in (0, 1] (got 2)"},
        Case{"tree.feature_fraction", "nan", "must be in (0, 1] (got nan)"},
        Case{"tree.monotone_constraints", "1,0,2",
             "[tree].monotone_constraints: item 2: must be in [-1, 1] (got 2)"},
        Case{"objective.quantile_alpha", "0",
             "[objective].quantile_alpha: must be in (0, 1) (got 0)"},
        Case{"objective.quantile_alpha", "1.5", "must be in (0, 1) (got 1.5)"},
        Case{"objective.huber_delta", "0",
             "[objective].huber_delta: must be finite and > 0 (got 0)"},
        Case{"bin_mapper.max_bin", "2",
             "[bin_mapper].max_bin: must be in [3, 65535] (got 2)"},
        Case{"bin_mapper.max_bin", "-7", "must be in [3, 65535] (got -7)"},
        Case{"bin_mapper.max_bin", "65536", "must be in [3, 65535] (got 65536)"});
    CAPTURE(key, value);
    bonsai::Config cfg;
    REQUIRE_THROWS_WITH(bonsai::config::apply_overrides(cfg, {{key, value}}),
                        ContainsSubstring(message));
}

TEST_CASE("Config bounds: a TOML value outside its key's domain is refused",
          "[toml][edge]")
{
    REQUIRE_THROWS_WITH(bonsai::config::parse_toml("[booster]\nlearning_rate = -0.5\n"),
                        Catch::Matchers::ContainsSubstring(
                            "[booster].learning_rate: must be finite and > 0"));
}

TEST_CASE("Config bounds: a value on a closed edge of its domain is kept",
          "[toml][overrides]")
{
    bonsai::Config cfg;
    bonsai::config::apply_overrides(cfg, {{"tree.feature_fraction", "1"},
                                          {"tree.lambda_l2", "0"},
                                          {"tree.monotone_constraints", "-1,0,1"},
                                          {"booster.dart_drop_rate", "1"},
                                          {"bin_mapper.max_bin", "3"}});
    CHECK(cfg.tree_config.feature_fraction == 1.0F);
    CHECK(cfg.tree_config.lambda_l2 == 0.0F);
    CHECK(cfg.tree_config.monotone_constraints == std::vector<int>{-1, 0, 1});
    CHECK(cfg.booster_config.dart_drop_rate == 1.0F);
    CHECK(cfg.bin_mapper.max_bin == 3);
    bonsai::config::apply_overrides(cfg, {{"bin_mapper.max_bin", "65535"}});
    CHECK(cfg.bin_mapper.max_bin == 65535);
}

TEST_CASE("Config bounds: require_in_bounds refuses a field set by hand",
          "[toml][edge]")
{
    bonsai::Config cfg;
    REQUIRE_NOTHROW(bonsai::config::require_in_bounds(cfg));
    cfg.bin_mapper.max_bin = 1;
    REQUIRE_THROWS_WITH(bonsai::config::require_in_bounds(cfg),
                        Catch::Matchers::ContainsSubstring(
                            "[bin_mapper].max_bin: must be in [3, 65535]"));
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
    original.data.train          = "train.csv";
    original.data.test           = "test.csv";
    original.data.valid          = {"valid1.csv", "valid2.csv"};
    original.data.header         = false;
    original.data.label_column   = 7;
    original.data.weight_column  = 3;
    original.data.ignore_columns = {1, 4, 9};

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

TEST_CASE("stated_keys: names the keys both sources state, values untouched",
          "[toml][overrides]")
{
    auto const path = std::string{BONSAI_TESTS_DATA_DIR} + "/_stated_keys_tmp.toml";
    {
        std::ofstream out{path};
        out << "[objective]\nquantile_alpha = 0.5\n\n[tree]\nmax_depth = 4\n";
    }
    std::vector<bonsai::config::Override> const overrides{
        {"dispatch.objective_name", "quantile"}, {"tree.max_depth", "9"}};

    auto const keys = bonsai::config::stated_keys(path, overrides);

    CHECK(std::ranges::find(keys, "objective.quantile_alpha") != keys.end());
    CHECK(std::ranges::find(keys, "dispatch.objective_name") != keys.end());

    // Precedence is resolve's, not stated_keys': the file states max_depth and
    // so does the override, and the override still wins the value.
    auto const cfg = bonsai::config::resolve(path, overrides);
    CHECK(cfg.tree_config.max_depth == 9);
    std::remove(path.c_str());
}
