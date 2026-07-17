// Device-resident MSE objective tests. Compiled in every build; each case
// SKIPs at runtime unless cuda_available(). They drive the full Booster so
// the whole resident seam runs: labels + scores uploaded once, the device
// gradient kernel replacing the gh upload, and the fused route+score-update
// replacing the finalize D2H. The escape hatch (BONSAI_HOST_OBJECTIVE=1)
// forces the host objective path for a same-process A/B.
//
// GPU histograms accumulate per-chunk in float with atomics in arbitrary
// order, so resident and host-objective models match to tolerance, not
// bit-exactly. The ship contract is a four-decimal r2 match; predictions are
// additionally required to agree closely.

#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/cuda/grower.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/types.hpp"
#include "test_grower_helpers.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace
{

using namespace bonsai;

struct RegData
{
    test::Built        built;
    std::vector<float> raw; // row-major, n_rows * n_features
    std::vector<float> y;
    size_t             n_rows;
    size_t             n_features;

    features_view view() const
    {
        return features_view{raw.data(), n_rows, n_features};
    }
};

// A linear-signal regression scenario with light noise: comfortably above the
// engine's CPU-fallback cutoff so the device path (and thus resident mode)
// really runs, and learnable enough that r2 is a meaningful sanity gate.
RegData make_regression(size_t n, size_t nf, uint32_t seed, bool binary = false)
{
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> u(0.0F, 1.0F);
    std::normal_distribution<float>       noise(0.0F, 0.1F);
    std::vector<float>                    coef(nf);
    for (float &c : coef)
    {
        c = (u(rng) * 2.0F) - 1.0F;
    }
    detail::ColumnBatch batch;
    batch.features.assign(nf, std::vector<float>(n));
    batch.feature_names.resize(nf);
    for (size_t j = 0; j < nf; ++j)
    {
        batch.feature_names[j] = "f" + std::to_string(j);
    }
    batch.labels.assign(n, 0.0F);
    std::vector<float> raw(n * nf);
    std::vector<float> signal(n);
    for (size_t r = 0; r < n; ++r)
    {
        float s = 0.0F;
        for (size_t j = 0; j < nf; ++j)
        {
            float const v        = u(rng);
            batch.features[j][r] = v;
            raw[(r * nf) + j]    = v;
            s += coef[j] * v;
        }
        signal[r] = s + noise(rng);
    }
    if (binary)
    {
        std::vector<float> sorted = signal;
        std::ranges::sort(sorted);
        float const median = sorted[n / 2];
        for (size_t r = 0; r < n; ++r)
        {
            batch.labels[r] = signal[r] > median ? 1.0F : 0.0F;
        }
    }
    else
    {
        for (size_t r = 0; r < n; ++r)
        {
            batch.labels[r] = signal[r];
        }
    }
    std::vector<float> y(batch.labels.begin(), batch.labels.end());
    return RegData{.built      = test::build(std::move(batch)),
                   .raw        = std::move(raw),
                   .y          = std::move(y),
                   .n_rows     = n,
                   .n_features = nf};
}

Config reg_cfg()
{
    Config cfg{};
    cfg.tree_config.max_depth        = 5;
    cfg.tree_config.min_data_in_leaf = 10;
    cfg.tree_config.min_child_hess   = 1.0F;
    cfg.booster_config.learning_rate = 0.1F;
    cfg.booster_config.random_seed   = 42;
    return cfg;
}

double r2_of(std::vector<float> const &pred, std::vector<float> const &y)
{
    double mean = 0.0;
    for (float const v : y)
    {
        mean += v;
    }
    mean /= static_cast<double>(y.size());
    double ss_res = 0.0;
    double ss_tot = 0.0;
    for (size_t i = 0; i < y.size(); ++i)
    {
        double const d = static_cast<double>(y[i]) - pred[i];
        ss_res += d * d;
        double const t = static_cast<double>(y[i]) - mean;
        ss_tot += t * t;
    }
    return 1.0 - (ss_res / ss_tot);
}

float max_abs_diff(std::vector<float> const &a, std::vector<float> const &b)
{
    float m = 0.0F;
    for (size_t i = 0; i < a.size(); ++i)
    {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}

// Fit `iters` rounds and predict on `data`. host_forced sets the escape hatch
// so the host objective path runs even for an otherwise-eligible booster.
template <typename BoosterT>
std::vector<float> fit_predict(Config const &cfg, RegData const &data, size_t iters,
                               bool host_forced)
{
    if (host_forced)
    {
        setenv("BONSAI_HOST_OBJECTIVE", "1", 1);
    }
    else
    {
        unsetenv("BONSAI_HOST_OBJECTIVE");
    }
    BoosterT booster{cfg};
    for (size_t i = 0; i < iters; ++i)
    {
        booster.update_one_iter(data.built.ds);
    }
    unsetenv("BONSAI_HOST_OBJECTIVE");
    std::vector<float> pred(data.n_rows);
    booster.predict(data.view(), floats_out{pred});
    return pred;
}

template <typename G> using MseBooster = Booster<MSEObjective, G, AllRowsSampler>;
template <typename G>
using MseBernoulliBooster = Booster<MSEObjective, G, BernoulliSampler>;
template <typename G>
using LogLossBooster = Booster<LogLossObjective, G, AllRowsSampler>;

} // namespace

TEST_CASE("Resident MSE matches host-objective GPU (depthwise)", "[cuda][resident]")
{
    if (!cuda_available())
    {
        SKIP("resident MSE needs a usable CUDA device");
    }
    auto const data = make_regression(8192, 6, 11);
    auto const cfg  = reg_cfg();
    auto const host = fit_predict<MseBooster<CudaDepthwiseGrower>>(cfg, data, 40, true);
    auto const res = fit_predict<MseBooster<CudaDepthwiseGrower>>(cfg, data, 40, false);

    double const r2_host = r2_of(host, data.y);
    double const r2_res  = r2_of(res, data.y);
    INFO("r2 host=" << r2_host << " resident=" << r2_res
                    << " max_abs_pred_diff=" << max_abs_diff(host, res));
    REQUIRE(r2_res > 0.9);
    REQUIRE(r2_res == Catch::Approx(r2_host).margin(1e-4));
    REQUIRE(max_abs_diff(host, res) < 5e-3F);
}

TEST_CASE("Resident MSE matches host-objective GPU (oblivious)", "[cuda][resident]")
{
    if (!cuda_available())
    {
        SKIP("resident MSE needs a usable CUDA device");
    }
    auto const data = make_regression(8192, 6, 13);
    auto const cfg  = reg_cfg();
    auto const host = fit_predict<MseBooster<CudaObliviousGrower>>(cfg, data, 40, true);
    auto const res = fit_predict<MseBooster<CudaObliviousGrower>>(cfg, data, 40, false);

    double const r2_host = r2_of(host, data.y);
    double const r2_res  = r2_of(res, data.y);
    INFO("r2 host=" << r2_host << " resident=" << r2_res
                    << " max_abs_pred_diff=" << max_abs_diff(host, res));
    REQUIRE(r2_res > 0.9);
    REQUIRE(r2_res == Catch::Approx(r2_host).margin(1e-4));
    REQUIRE(max_abs_diff(host, res) < 5e-3F);
}

TEST_CASE("Resident MSE reaches a reasonable multi-tree r2", "[cuda][resident]")
{
    if (!cuda_available())
    {
        SKIP("resident MSE needs a usable CUDA device");
    }
    auto const data = make_regression(8192, 6, 17);
    auto const cfg  = reg_cfg();
    auto const res = fit_predict<MseBooster<CudaDepthwiseGrower>>(cfg, data, 80, false);
    INFO("resident r2=" << r2_of(res, data.y));
    REQUIRE(r2_of(res, data.y) > 0.95);
}

TEST_CASE("Resident MSE warm-starts from loaded trees (depthwise)", "[cuda][resident]")
{
    if (!cuda_available())
    {
        SKIP("resident MSE needs a usable CUDA device");
    }
    auto const data = make_regression(8192, 6, 19);
    auto const cfg  = reg_cfg();

    // A base model to warm-start from; both continuations reload it and add
    // rounds, one host-forced and one resident, then must agree.
    MseBooster<CudaDepthwiseGrower> base{cfg};
    setenv("BONSAI_HOST_OBJECTIVE", "1", 1);
    for (size_t i = 0; i < 20; ++i)
    {
        base.update_one_iter(data.built.ds);
    }
    unsetenv("BONSAI_HOST_OBJECTIVE");
    auto const  trees = base.trees();
    float const init  = base.init_score();

    auto continue_from = [&](bool host_forced)
    {
        if (host_forced)
        {
            setenv("BONSAI_HOST_OBJECTIVE", "1", 1);
        }
        else
        {
            unsetenv("BONSAI_HOST_OBJECTIVE");
        }
        MseBooster<CudaDepthwiseGrower> b{cfg};
        b.load_state(trees, init); // warm start: trees present, scores rebuilt
        for (size_t i = 0; i < 15; ++i)
        {
            b.update_one_iter(data.built.ds);
        }
        unsetenv("BONSAI_HOST_OBJECTIVE");
        std::vector<float> pred(data.n_rows);
        b.predict(data.view(), floats_out{pred});
        return pred;
    };

    auto const host = continue_from(true);
    auto const res  = continue_from(false);
    INFO("warm-start max_abs_pred_diff=" << max_abs_diff(host, res));
    REQUIRE(r2_of(res, data.y) == Catch::Approx(r2_of(host, data.y)).margin(1e-4));
    REQUIRE(max_abs_diff(host, res) < 5e-3F);
}

TEST_CASE("Resident MSE matches host-objective GPU under Bernoulli sampling",
          "[cuda][resident]")
{
    if (!cuda_available())
    {
        SKIP("resident MSE needs a usable CUDA device");
    }
    auto const data       = make_regression(8192, 6, 23);
    Config     cfg        = reg_cfg();
    cfg.sampler.subsample = 0.7F; // drop ~30% of rows per tree

    auto const host =
        fit_predict<MseBernoulliBooster<CudaDepthwiseGrower>>(cfg, data, 40, true);
    auto const res =
        fit_predict<MseBernoulliBooster<CudaDepthwiseGrower>>(cfg, data, 40, false);

    double const r2_host = r2_of(host, data.y);
    double const r2_res  = r2_of(res, data.y);
    INFO("bernoulli r2 host=" << r2_host << " resident=" << r2_res
                              << " max_abs_pred_diff=" << max_abs_diff(host, res));
    REQUIRE(r2_res > 0.9);
    REQUIRE(r2_res == Catch::Approx(r2_host).margin(1e-4));
    REQUIRE(max_abs_diff(host, res) < 5e-3F);
}

TEST_CASE("Escape hatch: BONSAI_HOST_OBJECTIVE forces the host path",
          "[cuda][resident]")
{
    if (!cuda_available())
    {
        SKIP("resident MSE needs a usable CUDA device");
    }
    auto const data = make_regression(8192, 6, 29);
    auto const cfg  = reg_cfg();
    // With the hatch set the host objective path runs (resident never arms);
    // the model still trains, and matches the resident model within tolerance.
    auto const host = fit_predict<MseBooster<CudaDepthwiseGrower>>(cfg, data, 40, true);
    auto const res = fit_predict<MseBooster<CudaDepthwiseGrower>>(cfg, data, 40, false);
    INFO("escape-hatch r2=" << r2_of(host, data.y));
    REQUIRE(r2_of(host, data.y) > 0.9);
    REQUIRE(r2_of(host, data.y) == Catch::Approx(r2_of(res, data.y)).margin(1e-4));
}

TEST_CASE("LogLoss takes the host path and trains on GPU", "[cuda][resident]")
{
    if (!cuda_available())
    {
        SKIP("resident MSE needs a usable CUDA device");
    }
    auto const data = make_regression(8192, 6, 31, /*binary=*/true);
    auto const cfg  = reg_cfg();
    // device_objective_kind<LogLoss> is none, so the resident block is compiled
    // out and LogLoss trains through the host objective path on the GPU grower.
    auto const pred =
        fit_predict<LogLossBooster<CudaDepthwiseGrower>>(cfg, data, 40, false);
    size_t correct = 0;
    for (size_t i = 0; i < data.n_rows; ++i)
    {
        float const p = 1.0F / (1.0F + std::exp(-pred[i]));
        if ((p >= 0.5F ? 1.0F : 0.0F) == data.y[i])
        {
            ++correct;
        }
    }
    double const acc = static_cast<double>(correct) / static_cast<double>(data.n_rows);
    INFO("logloss accuracy=" << acc);
    REQUIRE(acc > 0.9);
}
