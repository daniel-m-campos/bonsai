#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <random>
#include <vector>

#include "bonsai/config/config.hpp"
#include "bonsai/config/errors.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/types.hpp"

using namespace bonsai; // NOLINT

namespace
{

Config cfg_with_subsample(float p)
{
    Config cfg{};
    cfg.sampler.subsample = p;
    return cfg;
}

} // namespace

TEST_CASE("AllRowsSampler: returns full iota regardless of grad/hess",
          "[sampler][all_rows]")
{
    AllRowsSampler s{Config{}};
    std::vector<row_id_t> out(5);
    std::mt19937 rng(0);
    size_t const n = s.sample({}, {}, rng, out);
    REQUIRE(n == out.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        CHECK(out[i] == static_cast<row_id_t>(i));
    }
}

TEST_CASE("BernoulliSampler: subsample=1.0 short-circuits to iota",
          "[sampler][bernoulli]")
{
    BernoulliSampler s{cfg_with_subsample(1.0F)};
    std::vector<row_id_t> out(8);
    std::mt19937 rng(0);
    size_t const n = s.sample({}, {}, rng, out);
    REQUIRE(n == out.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        CHECK(out[i] == static_cast<row_id_t>(i));
    }
}

TEST_CASE("BernoulliSampler: subsample=0.5 selects ~half of rows",
          "[sampler][bernoulli]")
{
    size_t constexpr n_rows = 10000;
    BernoulliSampler s{cfg_with_subsample(0.5F)};
    std::vector<row_id_t> out(n_rows);
    std::mt19937 rng(42);
    size_t const n = s.sample({}, {}, rng, out);

    // 5-sigma envelope: mean=5000, var=n*p*(1-p)=2500, sigma=50, so 5sigma=250.
    auto const lo = 5000 - 250;
    auto const hi = 5000 + 250;
    CHECK(static_cast<int>(n) >= lo);
    CHECK(static_cast<int>(n) <= hi);

    // Selected indices land at the front and are strictly increasing.
    for (size_t i = 1; i < n; ++i)
    {
        CHECK(out[i] > out[i - 1]);
    }
}

TEST_CASE("BernoulliSampler: same seed → same selection",
          "[sampler][bernoulli][determinism]")
{
    size_t constexpr n_rows = 1000;
    BernoulliSampler s1{cfg_with_subsample(0.3F)};
    BernoulliSampler s2{cfg_with_subsample(0.3F)};
    std::vector<row_id_t> out1(n_rows);
    std::vector<row_id_t> out2(n_rows);
    std::mt19937 rng1(123);
    std::mt19937 rng2(123);

    size_t const n1 = s1.sample({}, {}, rng1, out1);
    size_t const n2 = s2.sample({}, {}, rng2, out2);
    REQUIRE(n1 == n2);
    for (size_t i = 0; i < n1; ++i)
    {
        CHECK(out1[i] == out2[i]);
    }
}

TEST_CASE("BernoulliSampler: non-positive p throws ConfigError",
          "[sampler][bernoulli][error]")
{
    CHECK_THROWS_AS(BernoulliSampler{cfg_with_subsample(0.0F)}, ConfigError);
    CHECK_THROWS_AS(BernoulliSampler{cfg_with_subsample(-0.5F)}, ConfigError);
}
