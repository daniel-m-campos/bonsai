#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <random>
#include <vector>

#include "bonsai/config/config.hpp"
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

TEST_CASE("AllRowsSampler: bench 1k rows", "[bench][sampler][all_rows]")
{
    AllRowsSampler        s{Config{}};
    std::vector<row_id_t> out(1'000);
    std::mt19937          rng(42);
    BENCHMARK("all_rows: 1k")
    {
        return s.sample({}, {}, rng, out);
    };
}

TEST_CASE("AllRowsSampler: bench 100k rows", "[bench][sampler][all_rows]")
{
    AllRowsSampler        s{Config{}};
    std::vector<row_id_t> out(100'000);
    std::mt19937          rng(42);
    BENCHMARK("all_rows: 100k")
    {
        return s.sample({}, {}, rng, out);
    };
}

TEST_CASE("AllRowsSampler: bench 1M rows", "[bench][sampler][all_rows]")
{
    AllRowsSampler        s{Config{}};
    std::vector<row_id_t> out(1'000'000);
    std::mt19937          rng(42);
    BENCHMARK("all_rows: 1M")
    {
        return s.sample({}, {}, rng, out);
    };
}

TEST_CASE("BernoulliSampler p=0.5: bench 1k rows", "[bench][sampler][bernoulli]")
{
    BernoulliSampler      s{cfg_with_subsample(0.5F)};
    std::vector<row_id_t> out(1'000);
    BENCHMARK("bernoulli p=0.5: 1k")
    {
        std::mt19937 rng(42);
        return s.sample({}, {}, rng, out);
    };
}

TEST_CASE("BernoulliSampler p=0.5: bench 100k rows", "[bench][sampler][bernoulli]")
{
    BernoulliSampler      s{cfg_with_subsample(0.5F)};
    std::vector<row_id_t> out(100'000);
    BENCHMARK("bernoulli p=0.5: 100k")
    {
        std::mt19937 rng(42);
        return s.sample({}, {}, rng, out);
    };
}

TEST_CASE("BernoulliSampler p=0.5: bench 1M rows", "[bench][sampler][bernoulli]")
{
    BernoulliSampler      s{cfg_with_subsample(0.5F)};
    std::vector<row_id_t> out(1'000'000);
    BENCHMARK("bernoulli p=0.5: 1M")
    {
        std::mt19937 rng(42);
        return s.sample({}, {}, rng, out);
    };
}

TEST_CASE("BernoulliSampler p=0.8: bench 100k rows", "[bench][sampler][bernoulli]")
{
    BernoulliSampler      s{cfg_with_subsample(0.8F)};
    std::vector<row_id_t> out(100'000);
    BENCHMARK("bernoulli p=0.8: 100k")
    {
        std::mt19937 rng(42);
        return s.sample({}, {}, rng, out);
    };
}
