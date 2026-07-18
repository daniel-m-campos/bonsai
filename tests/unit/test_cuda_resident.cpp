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
#include <cstdio>
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
// `weighted` attaches non-uniform per-row weights, drawn AFTER the feature and
// label stream so the same seed yields identical features/labels with and
// without them (the "weights actually change the model" A/B relies on that).
RegData make_regression(size_t n, size_t nf, uint32_t seed, bool binary = false,
                        bool weighted = false)
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
    if (weighted)
    {
        batch.weights.assign(n, 0.0F);
        for (size_t r = 0; r < n; ++r)
        {
            batch.weights[r] = 0.25F + (3.75F * u(rng)); // in [0.25, 4.0]
        }
    }
    std::vector<float> y(batch.labels.begin(), batch.labels.end());
    return RegData{.built      = test::build(std::move(batch)),
                   .raw        = std::move(raw),
                   .y          = std::move(y),
                   .n_rows     = n,
                   .n_features = nf};
}

// Count-valued targets for Poisson: y ~ round(exp(signal)), y >= 0. `log_rate_offset`
// shifts every log-rate, so a large positive offset lands the init score and
// the per-row scores in the clamp region (|score| > k_poisson_max_log), driving
// the device kernel's fminf/fmaxf clamp on every gradient.
RegData make_counts(size_t n, size_t nf, uint32_t seed, float log_rate_offset = 0.0F,
                    bool weighted = false)
{
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> u(0.0F, 1.0F);
    std::vector<float>                    coef(nf);
    for (float &c : coef)
    {
        c = (u(rng) * 1.2F) - 0.4F;
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
        double const rate = std::exp(static_cast<double>(s + log_rate_offset));
        batch.labels[r]   = static_cast<float>(std::llround(rate));
    }
    if (weighted)
    {
        batch.weights.assign(n, 0.0F);
        for (size_t r = 0; r < n; ++r)
        {
            batch.weights[r] = 0.25F + (3.75F * u(rng));
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

// Emits the observed resident-vs-host deltas so the campaign A/B can quote
// exact numbers; printed unconditionally, not only on assertion failure.
void report(char const *tag, std::vector<float> const &host,
            std::vector<float> const &res, std::vector<float> const &y)
{
    double const rh = r2_of(host, y);
    double const rr = r2_of(res, y);
    std::fprintf(stderr,
                 "[resident] %-10s r2_host=%.6f r2_res=%.6f dr2=%.2e "
                 "max_pred_diff=%.5f\n",
                 tag, rh, rr, std::abs(rh - rr),
                 static_cast<double>(max_abs_diff(host, res)));
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
template <typename G>
using LogLossBernoulliBooster = Booster<LogLossObjective, G, BernoulliSampler>;
template <typename G>
using PoissonBooster                   = Booster<PoissonObjective, G, AllRowsSampler>;
template <typename G> using MaeBooster = Booster<MAEObjective, G, AllRowsSampler>;

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
    report("depthwise", host, res, data.y);

    // Both paths build histograms and root sums on the device (identity fit),
    // so the only divergence is atomic-add ordering: r2 holds to four decimals.
    REQUIRE(r2_of(res, data.y) > 0.9);
    REQUIRE(r2_of(res, data.y) == Catch::Approx(r2_of(host, data.y)).margin(1e-4));
    REQUIRE(max_abs_diff(host, res) < 2e-2F);
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
    report("oblivious", host, res, data.y);

    REQUIRE(r2_of(res, data.y) > 0.9);
    REQUIRE(r2_of(res, data.y) == Catch::Approx(r2_of(host, data.y)).margin(1e-4));
    REQUIRE(max_abs_diff(host, res) < 2e-2F);
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
    report("warmstart", host, res, data.y);
    REQUIRE(r2_of(res, data.y) == Catch::Approx(r2_of(host, data.y)).margin(1e-4));
    REQUIRE(max_abs_diff(host, res) < 2e-2F);
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
    report("bernoulli", host, res, data.y);

    // A row subset makes begin_root sum the root grad/hess differently on each
    // path: the host objective reduces on the CPU (serial), while resident
    // reduces the gathered subset on the device (blocked two-pass). That extra
    // reduction-order gap seeds the root split and compounds over rounds, so
    // this case tolerates a wider band than the identity fits above; both
    // models are equally accurate (the r2 gap stays in the third decimal).
    REQUIRE(r2_of(res, data.y) > 0.9);
    REQUIRE(r2_of(res, data.y) == Catch::Approx(r2_of(host, data.y)).margin(3e-3));
    REQUIRE(max_abs_diff(host, res) < 0.25F);
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
    report("hatch", host, res, data.y);
    REQUIRE(r2_of(host, data.y) > 0.9);
    REQUIRE(r2_of(host, data.y) == Catch::Approx(r2_of(res, data.y)).margin(1e-4));
}

double accuracy_of(std::vector<float> const &pred, std::vector<float> const &y)
{
    size_t correct = 0;
    for (size_t i = 0; i < y.size(); ++i)
    {
        float const p = 1.0F / (1.0F + std::exp(-pred[i]));
        if ((p >= 0.5F ? 1.0F : 0.0F) == y[i])
        {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(y.size());
}

TEST_CASE("Resident LogLoss matches host-objective GPU (depthwise)", "[cuda][resident]")
{
    if (!cuda_available())
    {
        SKIP("resident LogLoss needs a usable CUDA device");
    }
    auto const data = make_regression(8192, 6, 31, /*binary=*/true);
    auto const cfg  = reg_cfg();
    // device_objective_kind<LogLoss> is logloss, so the resident sigmoid kernel
    // derives (p - y, p(1 - p)) on device; the host arm forces the CPU objective.
    auto const host =
        fit_predict<LogLossBooster<CudaDepthwiseGrower>>(cfg, data, 40, true);
    auto const res =
        fit_predict<LogLossBooster<CudaDepthwiseGrower>>(cfg, data, 40, false);
    report("logloss", host, res, data.y);

    // Identity fit: both arms build histograms and root sums on device, so the
    // only divergence is the sigmoid's transcendental ulp plus atomic-add order.
    // The four-decimal r2 convention holds; predictions agree closely.
    REQUIRE(accuracy_of(res, data.y) > 0.9);
    REQUIRE(r2_of(res, data.y) == Catch::Approx(r2_of(host, data.y)).margin(1e-4));
    REQUIRE(max_abs_diff(host, res) < 5e-2F);
}

TEST_CASE("Resident LogLoss matches host-objective GPU (oblivious)", "[cuda][resident]")
{
    if (!cuda_available())
    {
        SKIP("resident LogLoss needs a usable CUDA device");
    }
    auto const data = make_regression(8192, 6, 37, /*binary=*/true);
    auto const cfg  = reg_cfg();
    auto const host =
        fit_predict<LogLossBooster<CudaObliviousGrower>>(cfg, data, 40, true);
    auto const res =
        fit_predict<LogLossBooster<CudaObliviousGrower>>(cfg, data, 40, false);
    report("logloss-ob", host, res, data.y);

    REQUIRE(accuracy_of(res, data.y) > 0.9);
    REQUIRE(r2_of(res, data.y) == Catch::Approx(r2_of(host, data.y)).margin(1e-4));
    REQUIRE(max_abs_diff(host, res) < 5e-2F);
}

TEST_CASE("Resident Poisson matches host-objective GPU (depthwise)", "[cuda][resident]")
{
    if (!cuda_available())
    {
        SKIP("resident Poisson needs a usable CUDA device");
    }
    auto const data = make_counts(8192, 6, 41);
    auto const cfg  = reg_cfg();
    // device_objective_kind<Poisson> is poisson: the resident kernel clamps the
    // score, exponentiates, and writes (mu - y, mu) exactly as the host does.
    auto const host =
        fit_predict<PoissonBooster<CudaDepthwiseGrower>>(cfg, data, 40, true);
    auto const res =
        fit_predict<PoissonBooster<CudaDepthwiseGrower>>(cfg, data, 40, false);
    report("poisson", host, res, data.y);

    REQUIRE(std::isfinite(r2_of(res, data.y)));
    REQUIRE(r2_of(res, data.y) == Catch::Approx(r2_of(host, data.y)).margin(1e-4));
    REQUIRE(max_abs_diff(host, res) < 5e-2F);
}

TEST_CASE("Resident Poisson clamps runaway scores, parity holds at the clamp",
          "[cuda][resident]")
{
    if (!cuda_available())
    {
        SKIP("resident Poisson needs a usable CUDA device");
    }
    // Counts around exp(31.5): the init score log(mean) > k_poisson_max_log, so
    // every row's score sits in the clamp region from round one and the device
    // kernel's fminf/fmaxf fires each gradient. Both arms clamp to the same
    // constant, so parity holds and no score turns to inf/nan.
    auto const data           = make_counts(4096, 4, 43, /*log_rate_offset=*/31.5F);
    Config     cfg            = reg_cfg();
    cfg.tree_config.max_depth = 3;
    auto const host =
        fit_predict<PoissonBooster<CudaDepthwiseGrower>>(cfg, data, 12, true);
    auto const res =
        fit_predict<PoissonBooster<CudaDepthwiseGrower>>(cfg, data, 12, false);
    report("poisson-clamp", host, res, data.y);

    for (float const p : res)
    {
        REQUIRE(std::isfinite(p));
    }
    REQUIRE(r2_of(res, data.y) == Catch::Approx(r2_of(host, data.y)).margin(1e-3));
    REQUIRE(max_abs_diff(host, res) < 1e-1F);
}

TEST_CASE(
    "Resident weighted MSE matches host-objective GPU and differs from unweighted",
    "[cuda][resident]")
{
    if (!cuda_available())
    {
        SKIP("resident MSE needs a usable CUDA device");
    }
    auto const wdata =
        make_regression(8192, 6, 47, /*binary=*/false, /*weighted=*/true);
    auto const udata =
        make_regression(8192, 6, 47, /*binary=*/false, /*weighted=*/false);
    auto const cfg = reg_cfg();

    // Weighted resident vs weighted host: the device kernel scales (grad, hess)
    // by the resident weight, the same multiply the host arm applies on the CPU.
    auto const host =
        fit_predict<MseBooster<CudaDepthwiseGrower>>(cfg, wdata, 40, true);
    auto const res =
        fit_predict<MseBooster<CudaDepthwiseGrower>>(cfg, wdata, 40, false);
    report("wmse", host, res, wdata.y);
    REQUIRE(r2_of(res, wdata.y) == Catch::Approx(r2_of(host, wdata.y)).margin(1e-4));
    REQUIRE(max_abs_diff(host, res) < 2e-2F);

    // Same features/labels, no weights: the weighted model must actually differ,
    // proving the weights reached the device gradient (not silently dropped).
    auto const unweighted =
        fit_predict<MseBooster<CudaDepthwiseGrower>>(cfg, udata, 40, false);
    REQUIRE(max_abs_diff(res, unweighted) > 1e-3F);
}

TEST_CASE("Resident weighted LogLoss under Bernoulli sampling matches host",
          "[cuda][resident]")
{
    if (!cuda_available())
    {
        SKIP("resident LogLoss needs a usable CUDA device");
    }
    auto   data = make_regression(8192, 6, 53, /*binary=*/true, /*weighted=*/true);
    Config cfg  = reg_cfg();
    cfg.sampler.subsample = 0.7F; // Bernoulli row sampling on top of weights

    auto const host =
        fit_predict<LogLossBernoulliBooster<CudaDepthwiseGrower>>(cfg, data, 40, true);
    auto const res =
        fit_predict<LogLossBernoulliBooster<CudaDepthwiseGrower>>(cfg, data, 40, false);
    report("wlogloss-bern", host, res, data.y);

    // Bernoulli subsets reduce differently on host (serial) vs device (blocked),
    // widening the band as the MSE-Bernoulli case documents; both stay accurate.
    REQUIRE(accuracy_of(res, data.y) > 0.85);
    REQUIRE(r2_of(res, data.y) == Catch::Approx(r2_of(host, data.y)).margin(3e-3));
    REQUIRE(max_abs_diff(host, res) < 0.35F);
}

TEST_CASE("MAE stays on the host path, the escape hatch is a no-op", "[cuda][resident]")
{
    if (!cuda_available())
    {
        SKIP("this MAE proxy needs a usable CUDA device");
    }
    auto const data = make_regression(8192, 6, 59);
    auto const cfg  = reg_cfg();
    // device_objective_kind<MAE> is none, so try_resident_round is compiled out
    // and MAE always trains on the host objective path. The behavioral proof:
    // BONSAI_HOST_OBJECTIVE has no path to toggle, so forcing it changes nothing
    // and the two models are byte-for-byte identical (a resident objective would
    // diverge here on the atomic-order tolerance).
    auto const def = fit_predict<MaeBooster<CudaDepthwiseGrower>>(cfg, data, 40, false);
    auto const forced =
        fit_predict<MaeBooster<CudaDepthwiseGrower>>(cfg, data, 40, true);
    REQUIRE(max_abs_diff(def, forced) == 0.0F);
}
