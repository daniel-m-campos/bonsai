// The device mapper fit reproduces the host's create_cuts bit for bit, and
// the model hash gate depends on it. Each column here targets one branch of
// the host algorithm (all-distinct, greedy, stride, tie runs, zero signs, the
// NaN and infinity tails), and the cuts compare as uint32 bit patterns, so a
// sign flip on a zero or a one-ulp midpoint difference fails loudly.
// Compiled in every build; SKIPs at runtime unless cuda_available().

#include "bonsai/bin_mapper.hpp"
#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/types.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace
{

constexpr size_t k_rows = 4096;

std::vector<uint32_t> bits_of(bonsai::floats_view cuts)
{
    std::vector<uint32_t> out(cuts.size());
    std::memcpy(out.data(), cuts.data(), cuts.size() * sizeof(float));
    return out;
}

using ColumnFill = void (*)(std::mt19937 &, std::vector<float> &);

void normal_column(std::mt19937 &rng, std::vector<float> &col)
{
    std::normal_distribution<float> d;
    for (auto &v : col)
    {
        v = d(rng);
    }
}

void few_values_column(std::mt19937 &rng, std::vector<float> &col)
{
    std::uniform_int_distribution<int> d(0, 6);
    for (auto &v : col)
    {
        v = static_cast<float>(d(rng)) * 0.25f;
    }
}

void zero_spike_column(std::mt19937 &rng, std::vector<float> &col)
{
    std::normal_distribution<float> d;
    std::bernoulli_distribution     spike(0.6);
    for (auto &v : col)
    {
        v = spike(rng) ? 0.0f : d(rng);
    }
}

void signed_zero_column(std::mt19937 &rng, std::vector<float> &col)
{
    std::normal_distribution<float> d;
    std::bernoulli_distribution     zero(0.3);
    std::bernoulli_distribution     negative(0.5);
    for (auto &v : col)
    {
        v = zero(rng) ? (negative(rng) ? -0.0f : 0.0f) : d(rng);
    }
}

void nan_sprinkled_column(std::mt19937 &rng, std::vector<float> &col)
{
    std::normal_distribution<float> d;
    std::bernoulli_distribution     missing(0.2);
    for (auto &v : col)
    {
        v = missing(rng) ? std::numeric_limits<float>::quiet_NaN() : d(rng);
    }
}

void infinities_column(std::mt19937 &rng, std::vector<float> &col)
{
    std::normal_distribution<float> d;
    for (size_t i = 0; i < col.size(); ++i)
    {
        col[i] = i % 97 == 0   ? std::numeric_limits<float>::infinity()
                 : i % 89 == 0 ? -std::numeric_limits<float>::infinity()
                 : i % 83 == 0 ? FLT_MAX
                               : d(rng);
    }
}

void constant_column(std::mt19937 & /*rng*/, std::vector<float> &col)
{
    for (auto &v : col)
    {
        v = 3.5f;
    }
}

void all_nan_column(std::mt19937 & /*rng*/, std::vector<float> &col)
{
    for (auto &v : col)
    {
        v = std::numeric_limits<float>::quiet_NaN();
    }
}

void budget_plus_one_column(std::mt19937 &rng, std::vector<float> &col)
{
    std::uniform_int_distribution<int> d(0, 253);
    for (auto &v : col)
    {
        v = static_cast<float>(d(rng));
    }
}

// Subnormal, huge, and huge-negative values side by side, so every branch of
// std::midpoint's overflow guard is reached: |a| < 2 FLT_MIN next to |b| >
// FLT_MAX / 2 and the reverse, and two huge values together.
void midpoint_guard_column(std::mt19937 &rng, std::vector<float> &col)
{
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::uniform_int_distribution<int>    kind(0, 2);
    float const                           hi = FLT_MAX / 2;
    for (auto &v : col)
    {
        switch (kind(rng))
        {
        case 0:
            v = -(hi + u(rng) * hi);
            break;
        case 1:
            v = u(rng) * FLT_MIN;
            break;
        default:
            v = hi + u(rng) * hi;
        }
    }
}

struct Matrix
{
    std::vector<float>       raw;
    size_t                   n_rows{};
    size_t                   n_features{};
    std::vector<std::string> names;

    bonsai::features_view view() const
    {
        return bonsai::features_view{raw.data(), n_rows, n_features};
    }
};

constexpr std::array<ColumnFill, 10> k_fills = {
    normal_column,          few_values_column,    zero_spike_column, signed_zero_column,
    nan_sprinkled_column,   infinities_column,    constant_column,   all_nan_column,
    budget_plus_one_column, midpoint_guard_column};

Matrix adversarial_matrix(size_t n_rows, size_t n_features = k_fills.size())
{
    Matrix m;
    m.n_rows     = n_rows;
    m.n_features = n_features;
    m.raw.resize(n_rows * n_features);
    std::mt19937       rng(7);
    std::vector<float> col(n_rows);
    for (size_t f = 0; f < n_features; ++f)
    {
        k_fills[f % k_fills.size()](rng, col);
        for (size_t r = 0; r < n_rows; ++r)
        {
            m.raw[r * m.n_features + f] = col[r];
        }
        m.names.push_back("c" + std::to_string(f));
    }
    return m;
}

void require_same_cuts(bonsai::BinMappers const &host, bonsai::BinMappers const &device)
{
    REQUIRE(device.size() == host.size());
    for (size_t f = 0; f < host.size(); ++f)
    {
        INFO("column " << f);
        REQUIRE(bits_of(device[f].cuts()) == bits_of(host[f].cuts()));
    }
}

void require_device_fit_matches(Matrix const &m, bonsai::BinMapperConfig const &cfg,
                                bonsai::BinEdges const &edges = {})
{
    auto const resident =
        bonsai::cuda_upload(m.view(), static_cast<size_t>(cfg.max_bin));
    REQUIRE(resident != nullptr);
    auto const host   = bonsai::BinMappers::fit(m.view(), m.names, cfg, edges);
    auto const device = bonsai::cuda_fit_mappers(*resident, m.names, cfg, edges);
    require_same_cuts(host, device);
    REQUIRE(std::vector<std::string>(device.feature_names().begin(),
                                     device.feature_names().end()) == m.names);
}

} // namespace

TEST_CASE("CudaMapperFit: device cuts equal the host cuts", "[cuda][fit]")
{
    if (!bonsai::cuda_available())
    {
        SKIP("no CUDA device");
    }
    bonsai::BinMapperConfig cfg;
    cfg.n_samples = 0;

    SECTION("every row, default budget")
    {
        require_device_fit_matches(adversarial_matrix(k_rows), cfg);
    }
    SECTION("a sampled subset of the rows")
    {
        cfg.n_samples = 1500;
        require_device_fit_matches(adversarial_matrix(k_rows), cfg);
    }
    SECTION("a narrow budget takes the greedy and stride paths on every column")
    {
        cfg.max_bin = 4;
        require_device_fit_matches(adversarial_matrix(k_rows), cfg);
    }
    SECTION("a wide budget crosses the u8 bin width")
    {
        cfg.max_bin = 1024;
        require_device_fit_matches(adversarial_matrix(k_rows), cfg);
    }
    SECTION("a small column sorts by std::sort on the host")
    {
        require_device_fit_matches(adversarial_matrix(1000), cfg);
    }
    SECTION("bin_edges overrides skip the device cut on that column")
    {
        bonsai::BinEdges edges;
        edges.emplace_back(0, std::vector<float>{-1.0f, 0.0f, 1.0f});
        require_device_fit_matches(adversarial_matrix(k_rows), cfg, edges);
    }
    SECTION("a matrix wider than one column chunk")
    {
        require_device_fit_matches(adversarial_matrix(256, 1100), cfg);
    }
}

TEST_CASE("CudaMapperFit: an upload declines what the fill would refuse",
          "[cuda][edge]")
{
    if (!bonsai::cuda_available())
    {
        SKIP("no CUDA device");
    }
    Matrix const m = adversarial_matrix(64);
    REQUIRE(bonsai::cuda_upload(m.view(), 255) != nullptr);
    REQUIRE(bonsai::cuda_upload(bonsai::features_view{m.raw.data(), 0, m.n_features},
                                255) == nullptr);
    REQUIRE(bonsai::cuda_upload(m.view(), size_t{1} << 20) == nullptr);
}
