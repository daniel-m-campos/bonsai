#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
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

// The candidate list a fit over a whole dataset passes: candidate i is row i,
// which is what every sampler assumed before the list was a parameter.
std::vector<row_id_t> identity(size_t n)
{
    std::vector<row_id_t> ids(n);
    std::iota(ids.begin(), ids.end(), row_id_t{0});
    return ids;
}

bool is_member(std::vector<row_id_t> const &candidates, row_id_t r)
{
    return std::ranges::find(candidates, r) != candidates.end();
}

} // namespace

TEST_CASE("AllRowsSampler: returns full iota regardless of grad/hess",
          "[sampler][all_rows]")
{
    AllRowsSampler        s{Config{}};
    auto const            candidates = identity(5);
    std::vector<row_id_t> out(5);
    std::mt19937          rng(0);
    size_t const          n = s.sample({}, {}, rng, candidates, out);
    REQUIRE(n == out.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        CHECK(out[i] == static_cast<row_id_t>(i));
    }
}

TEST_CASE("BernoulliSampler: subsample=1.0 short-circuits to iota",
          "[sampler][bernoulli]")
{
    BernoulliSampler      s{cfg_with_subsample(1.0F)};
    auto const            candidates = identity(8);
    std::vector<row_id_t> out(8);
    std::mt19937          rng(0);
    size_t const          n = s.sample({}, {}, rng, candidates, out);
    REQUIRE(n == out.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        CHECK(out[i] == static_cast<row_id_t>(i));
    }
}

TEST_CASE("BernoulliSampler: subsample=0.5 selects ~half of rows",
          "[sampler][bernoulli]")
{
    constexpr size_t      n_rows = 10000;
    BernoulliSampler      s{cfg_with_subsample(0.5F)};
    auto const            candidates = identity(n_rows);
    std::vector<row_id_t> out(n_rows);
    std::mt19937          rng(42);
    size_t const          n = s.sample({}, {}, rng, candidates, out);

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
    constexpr size_t      n_rows = 1000;
    BernoulliSampler      s1{cfg_with_subsample(0.3F)};
    BernoulliSampler      s2{cfg_with_subsample(0.3F)};
    auto const            candidates = identity(n_rows);
    std::vector<row_id_t> out1(n_rows);
    std::vector<row_id_t> out2(n_rows);
    std::mt19937          rng1(123);
    std::mt19937          rng2(123);

    size_t const n1 = s1.sample({}, {}, rng1, candidates, out1);
    size_t const n2 = s2.sample({}, {}, rng2, candidates, out2);
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

TEST_CASE("GossSampler: keeps top-|grad| rows and amplifies sampled rest",
          "[sampler][goss]")
{
    Config cfg{};
    cfg.sampler.top_rate   = 0.2F;
    cfg.sampler.other_rate = 0.2F;
    GossSampler s{cfg};

    // 10 rows; rows 3 and 7 have the largest |grad| -> always kept, unscaled.
    std::vector<float>    grad{0.1F, -0.2F, 0.1F,  5.0F, -0.1F,
                            0.2F, 0.1F,  -4.0F, 0.2F, -0.1F};
    std::vector<float>    hess(10, 1.0F);
    auto const            candidates = identity(10);
    std::vector<row_id_t> out(10);
    std::mt19937          rng(42);

    size_t const n = s.sample(grad, hess, rng, candidates, out);
    // top 2 + 2 sampled others.
    REQUIRE(n == 4);

    bool has3 = false;
    bool has7 = false;
    for (size_t i = 0; i < n; ++i)
    {
        has3 |= out[i] == 3;
        has7 |= out[i] == 7;
        if (i > 0)
        {
            CHECK(out[i] > out[i - 1]); // ascending
        }
    }
    CHECK(has3);
    CHECK(has7);
    CHECK(grad[3] == 5.0F); // top rows unscaled
    CHECK(grad[7] == -4.0F);

    // Amplification factor (1 - 0.2) / 0.2 = 4 applied to the sampled rest.
    float const amplify = 4.0F;
    for (size_t i = 0; i < n; ++i)
    {
        row_id_t const r = out[i];
        if (r != 3 && r != 7)
        {
            CHECK(hess[r] == amplify);
        }
    }
}

TEST_CASE("GossSampler: same seed → same selection and scaling",
          "[sampler][goss][determinism]")
{
    Config cfg{};
    cfg.sampler.top_rate   = 0.3F;
    cfg.sampler.other_rate = 0.3F;
    GossSampler s{cfg};

    auto run = [&]
    {
        std::vector<float>    grad{1.0F, -2.0F, 0.5F,  3.0F, -0.1F,
                                0.2F, 1.5F,  -0.4F, 0.8F, -1.2F};
        std::vector<float>    hess(10, 1.0F);
        auto const            candidates = identity(10);
        std::vector<row_id_t> out(10);
        std::mt19937          rng(7);
        size_t const          n = s.sample(grad, hess, rng, candidates, out);
        out.resize(n);
        return std::pair{out, grad};
    };
    auto const a = run();
    auto const b = run();
    CHECK(a.first == b.first);
    CHECK(a.second == b.second);
}

// The candidate list is the row universe a fit may draw from, which for a
// Dataset view is the view's rows. A sampler emits ids out of that list, never
// positions into it: grad, hess and the bins stay globally indexed.

TEST_CASE("AllRowsSampler: copies the candidate list verbatim",
          "[sampler][all_rows][candidates]")
{
    AllRowsSampler              s{Config{}};
    std::vector<row_id_t> const candidates{7, 3, 3, 900, 42};
    std::vector<row_id_t>       out(candidates.size());
    std::mt19937                rng(0);
    size_t const                n = s.sample({}, {}, rng, candidates, out);
    REQUIRE(n == candidates.size());
    CHECK(out == candidates);
}

// Square brackets in a name are tag syntax to Catch2 and corrupt ctest's
// discovered test list, so the range is spelled out.
TEST_CASE("BernoulliSampler: draws out of the candidates, not the whole range",
          "[sampler][bernoulli][candidates]")
{
    constexpr size_t      n_candidates = 10000;
    std::vector<row_id_t> candidates(n_candidates);
    // Row ids spread over a plane 7x the candidate count: an emitted position
    // would land outside the list and the membership check would catch it.
    for (size_t i = 0; i < n_candidates; ++i)
    {
        candidates[i] = static_cast<row_id_t>((i * 7) + 3);
    }
    BernoulliSampler      s{cfg_with_subsample(0.5F)};
    std::vector<row_id_t> out(n_candidates);
    std::mt19937          rng(42);
    size_t const          n = s.sample({}, {}, rng, candidates, out);

    // Same 5-sigma envelope as the identity case: the draw is per candidate.
    CHECK(static_cast<int>(n) >= 4750);
    CHECK(static_cast<int>(n) <= 5250);
    for (size_t i = 0; i < n; ++i)
    {
        CHECK(is_member(candidates, out[i]));
        if (i > 0)
        {
            CHECK(out[i] > out[i - 1]); // candidate order, which is ascending
        }
    }
}

TEST_CASE("BernoulliSampler: subsample=1.0 over candidates copies the list",
          "[sampler][bernoulli][candidates]")
{
    BernoulliSampler            s{cfg_with_subsample(1.0F)};
    std::vector<row_id_t> const candidates{4, 5, 6, 99, 100};
    std::vector<row_id_t>       out(candidates.size());
    std::mt19937                rng(0);
    size_t const                n = s.sample({}, {}, rng, candidates, out);
    REQUIRE(n == candidates.size());
    CHECK(out == candidates);
}

TEST_CASE("GossSampler: ranks and amplifies the candidate rows",
          "[sampler][goss][candidates]")
{
    Config cfg{};
    cfg.sampler.top_rate   = 0.2F;
    cfg.sampler.other_rate = 0.2F;
    GossSampler s{cfg};

    // 10 candidates scattered through a 32-row plane. Every non-candidate row
    // carries the largest |grad| in the array: a ranking that read positions as
    // row ids would pick them up, and an amplify that wrote positions would
    // scale them.
    std::vector<row_id_t> const candidates{2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    std::vector<float>          grad(32, 100.0F);
    std::vector<float>          hess(32, 1.0F);
    std::vector<float> const small{0.1F, -0.2F, 0.1F, -0.1F, 0.2F, 0.1F, 0.2F, -0.1F};
    size_t                   k = 0;
    for (row_id_t const r : candidates)
    {
        grad[r] = r == 7 ? 5.0F : (r == 19 ? -4.0F : small[k++ % small.size()]);
    }

    std::vector<row_id_t> out(candidates.size());
    std::mt19937          rng(42);
    size_t const          n = s.sample(grad, hess, rng, candidates, out);
    REQUIRE(n == 4); // top 2 + 2 sampled others

    bool has7  = false;
    bool has19 = false;
    for (size_t i = 0; i < n; ++i)
    {
        CHECK(is_member(candidates, out[i]));
        has7 |= out[i] == 7;
        has19 |= out[i] == 19;
        if (i > 0)
        {
            CHECK(out[i] > out[i - 1]);
        }
    }
    CHECK(has7);
    CHECK(has19);
    CHECK(grad[7] == 5.0F); // top rows unscaled
    CHECK(grad[19] == -4.0F);

    float const amplify = 4.0F; // (1 - 0.2) / 0.2
    for (size_t i = 0; i < n; ++i)
    {
        row_id_t const r = out[i];
        if (r != 7 && r != 19)
        {
            CHECK(hess[r] == amplify);
        }
    }
    // Rows outside the candidate list are untouched, high |grad| and all.
    for (row_id_t r = 0; r < 32; ++r)
    {
        if (!is_member(candidates, r))
        {
            CHECK(grad[r] == 100.0F);
            CHECK(hess[r] == 1.0F);
        }
    }
}

TEST_CASE("GossSampler: invalid rates throw ConfigError", "[sampler][goss][config]")
{
    Config bad{};
    bad.sampler.top_rate   = 0.9F;
    bad.sampler.other_rate = 0.5F; // sum > 1
    CHECK_THROWS_AS(GossSampler{bad}, ConfigError);

    Config zero{};
    zero.sampler.top_rate   = 0.0F;
    zero.sampler.other_rate = 0.1F;
    CHECK_THROWS_AS(GossSampler{zero}, ConfigError);
}
