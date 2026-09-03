#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "bonsai/cli/pipeline.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/errors.hpp"
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

    CHECK(loaded.train.plane_n_rows() == 4);
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

    auto const scored =
        score_csv(*booster, cfg.data.train, cfg.data, loaded.mappers.size());
    CHECK(scored.raw_scores.size() == 4);
}

TEST_CASE("score_csv: a CSV of the wrong width is refused before predict",
          "[cli_pipeline][score][edge]")
{
    auto const cfg     = make_tiny_config();
    auto const loaded  = load_train_from_csv(cfg, cfg.data.train);
    auto const booster = train_in_memory(cfg, loaded.train);

    REQUIRE_THROWS_WITH(
        score_csv(*booster, cfg.data.train, cfg.data, loaded.mappers.size() + 1),
        Catch::Matchers::ContainsSubstring("columns must match"));
}

TEST_CASE("score_and_label_csv: labels match the CSV's label column",
          "[cli_pipeline][score]")
{
    auto const cfg     = make_tiny_config();
    auto const loaded  = load_train_from_csv(cfg, cfg.data.train);
    auto const booster = train_in_memory(cfg, loaded.train);

    auto const sl =
        score_and_label_csv(*booster, cfg.data.train, cfg.data, loaded.mappers.size());
    REQUIRE(sl.raw_scores.size() == 4);
    REQUIRE(sl.labels.size() == 4);
    CHECK(sl.labels[0] == Catch::Approx(0.5F));
    CHECK(sl.labels[1] == Catch::Approx(1.5F));
    CHECK(sl.labels[2] == Catch::Approx(2.5F));
    CHECK(sl.labels[3] == Catch::Approx(3.5F));
}

TEST_CASE("reconcile_warm_start: inherits the loaded dispatch, refuses a "
          "disagreement",
          "[cli_pipeline][ctor][edge]")
{
    Config loaded;
    loaded.dispatch.objective_name = "softmax";
    loaded.objective.n_classes     = 5;

    SECTION("default fields inherit the loaded model's")
    {
        auto const merged = reconcile_warm_start(Config{}, loaded);
        CHECK(merged.dispatch == loaded.dispatch);
        CHECK(merged.objective == loaded.objective);
    }
    SECTION("restating the model's own values passes")
    {
        Config restated;
        restated.dispatch.objective_name = "softmax";
        restated.objective.n_classes     = 5;
        auto const merged =
            reconcile_warm_start(restated, loaded, {"dispatch.objective_name"});
        CHECK(merged.dispatch == loaded.dispatch);
    }
    SECTION("an unnamed non-default field refuses")
    {
        Config wrong;
        wrong.dispatch.objective_name = "logloss";
        REQUIRE_THROWS_WITH(
            reconcile_warm_start(wrong, loaded),
            Catch::Matchers::ContainsSubstring("a warm start continues"));
    }
    SECTION("an explicitly named default that disagrees refuses too")
    {
        REQUIRE_THROWS_WITH(
            reconcile_warm_start(Config{}, loaded, {"dispatch.objective_name"}),
            Catch::Matchers::ContainsSubstring("dispatch.objective_name"));
    }
    SECTION("a named field that agrees leaves unnamed fields free to inherit")
    {
        Config partial;
        partial.dispatch.objective_name = "softmax";
        auto const merged =
            reconcile_warm_start(partial, loaded, {"dispatch.objective_name"});
        CHECK(merged.objective.n_classes == 5);
    }
    SECTION("an explicit disagreeing objective refuses")
    {
        Config wrong;
        wrong.dispatch.objective_name = "softmax";
        wrong.objective.n_classes     = 7;
        REQUIRE_THROWS_WITH(
            reconcile_warm_start(wrong, loaded, {"objective.n_classes"}),
            Catch::Matchers::ContainsSubstring("objective.n_classes"));
    }
}

TEST_CASE("train_with_progress: labels outside the objective's domain are "
          "refused",
          "[cli_pipeline][fit][edge]")
{
    auto       cfg    = make_tiny_config();
    auto const loaded = load_train_and_validation_from_csv(cfg);

    SECTION("logloss refuses a label that is neither 0 nor 1")
    {
        cfg.dispatch.objective_name = "logloss";
        auto train                  = loaded.train;
        train.labels                = {0.0F, 1.0F, 2.0F, 1.0F};
        REQUIRE_THROWS_WITH(
            train_with_progress(cfg, train),
            Catch::Matchers::ContainsSubstring("logloss labels must be 0 or 1"));
    }
    SECTION("softmax refuses a label at or past n_classes")
    {
        cfg.dispatch.objective_name = "softmax";
        cfg.objective.n_classes     = 3;
        auto train                  = loaded.train;
        train.labels                = {0.0F, 1.0F, 2.0F, 3.0F};
        REQUIRE_THROWS_WITH(
            train_with_progress(cfg, train),
            Catch::Matchers::ContainsSubstring("softmax labels must be integers"));
    }
    SECTION("bench's in-memory trainer is checked too")
    {
        cfg.dispatch.objective_name = "logloss";
        auto const in_memory        = load_train_from_csv(cfg, cfg.data.train);
        REQUIRE_THROWS_WITH(
            train_in_memory(cfg, in_memory.train),
            Catch::Matchers::ContainsSubstring("logloss labels must be 0 or 1"));
    }
    SECTION("softmax refuses a fractional label")
    {
        cfg.dispatch.objective_name = "softmax";
        cfg.objective.n_classes     = 3;
        auto train                  = loaded.train;
        train.labels                = {0.0F, 1.0F, 2.0F, 1.5F};
        REQUIRE_THROWS_WITH(
            train_with_progress(cfg, train),
            Catch::Matchers::ContainsSubstring("softmax labels must be integers"));
    }
}

TEST_CASE("fit: a validation CSV of the wrong width is refused before eval",
          "[cli_pipeline][load][edge]")
{
    auto const narrow = std::string{BONSAI_TESTS_DATA_DIR} + "/_narrow_valid_tmp.csv";
    {
        std::ofstream out{narrow};
        out << "f0,label\n0.1,1.0\n0.2,0.0\n";
    }
    auto cfg       = make_tiny_config();
    cfg.data.valid = {narrow};

    REQUIRE_THROWS_WITH(load_train_and_validation_from_csv(cfg),
                        Catch::Matchers::ContainsSubstring("columns must match"));
    std::remove(narrow.c_str());
}

TEST_CASE("train_with_progress: log_intervals fires the baseline, the period, "
          "and the final tick",
          "[cli_pipeline][train]")
{
    auto cfg                         = make_tiny_config();
    cfg.data.valid                   = {k_tiny_path};
    cfg.booster_config.n_iters       = 7;
    cfg.booster_config.log_intervals = 3;

    auto const          loaded = load_train_and_validation_from_csv(cfg);
    std::vector<size_t> iters;
    std::vector<size_t> valid_widths;
    auto const          on_tick = [&](FitTick const &tick)
    {
        iters.push_back(tick.iter);
        valid_widths.push_back(tick.validation_preds.size());
        CHECK(tick.n_iters == 7);
        CHECK(tick.train_preds.size() == 4);
        CHECK(tick.train_labels.size() == 4);
        CHECK(tick.validation_labels.size() == 4);
    };

    SECTION("with a sink, period is n_iters / log_intervals rounded down")
    {
        train_with_progress(cfg, loaded, on_tick);
        CHECK(iters == std::vector<size_t>{0, 2, 4, 6, 7});
        CHECK(valid_widths == std::vector<size_t>{4, 4, 4, 4, 4});
    }
    SECTION("without a validation set the validation spans are empty")
    {
        std::vector<size_t> widths;
        train_with_progress(cfg, loaded.train,
                            [&](FitTick const &tick)
                            {
                                widths.push_back(tick.validation_preds.size() +
                                                 tick.validation_labels.size());
                            });
        CHECK(widths == std::vector<size_t>{0, 0, 0, 0, 0});
    }
    SECTION("log_intervals of zero fires nothing even with a sink")
    {
        cfg.booster_config.log_intervals = 0;
        train_with_progress(cfg, loaded, on_tick);
        CHECK(iters.empty());
    }
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

    auto const loaded  = load_train_and_validation_from_csv(cfg);
    auto       booster = train_with_progress(cfg, loaded, {});
    CHECK(booster->n_iters() < 200);

    Config no_es                               = cfg;
    no_es.booster_config.early_stopping_rounds = 0;
    auto const loaded2 = load_train_and_validation_from_csv(no_es);
    auto       full    = train_with_progress(no_es, loaded2, {});
    CHECK(full->n_iters() == 200);
}

TEST_CASE("train_with_progress: a prebinned validation set routes identically",
          "[cli_pipeline][early_stop][eval]")
{
    // A validation set handed over already binned skips the gate and routes
    // in bin space from the first round. Three features at depth 6 never pay
    // for the bin pass at this round count, so the reference arm below is the
    // raw walk, and the two must agree round for round.
    Config cfg;
    cfg.data.train                           = k_tiny_path;
    cfg.data.valid                           = {k_tiny_path};
    cfg.bin_mapper.max_bin                   = 8;
    cfg.bin_mapper.n_samples                 = 100;
    cfg.booster_config.n_iters               = 40;
    cfg.booster_config.learning_rate         = 0.3F;
    cfg.booster_config.early_stopping_rounds = 3;
    cfg.tree_config.min_data_in_leaf         = 0;
    cfg.tree_config.min_child_hess           = 0.0F;

    auto const loaded = load_train_and_validation_from_csv(cfg);
    REQUIRE(loaded.validation.has_value());
    std::vector<float> raw_history;
    auto const raw = train_with_progress(cfg, loaded.train, *loaded.validation, {}, {},
                                         std::ref(raw_history));

    // The same rows, binned once with the training set's own mappers.
    LabeledData prebinned = *loaded.validation;
    prebinned.dataset     = Dataset::bin(prebinned.features.view(), prebinned.labels,
                                         loaded.mappers, cfg.data);
    REQUIRE(prebinned.dataset.bins_are_u8());
    std::vector<float> binned_history;
    auto const binned = train_with_progress(cfg, loaded.train, prebinned, {}, {},
                                            std::ref(binned_history));

    REQUIRE(!raw_history.empty());
    CHECK(binned->n_iters() == raw->n_iters());
    CHECK(binned_history == raw_history);
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
    auto const loaded_a = load_train_and_validation_from_csv(cfg);
    auto       straight = train_with_progress(cfg, loaded_a, {});

    // Warm start: 3 iterations, save, reload, 3 more.
    Config half                 = cfg;
    half.booster_config.n_iters = 3;
    auto const loaded_b         = load_train_and_validation_from_csv(half);
    auto       first            = train_with_progress(half, loaded_b, {});
    auto const tmp = std::string{BONSAI_TESTS_DATA_DIR} + "/_warm_start_tmp.msgpack";
    io::save_booster(*first, tmp, loaded_b.mappers, half);

    auto       reloaded = io::load_booster(tmp);
    auto const loaded_c =
        load_train_and_validation_with_mappers(half, std::move(reloaded.mappers));
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

TEST_CASE("train_with_progress: multiclass early stopping truncates whole rounds",
          "[cli_pipeline][early_stop][multiclass]")
{
    // Three separable classes on one feature; train == valid, so the loss
    // converges after a few rounds and patience must fire well before
    // n_iters. Truncation must keep whole rounds (n_classes trees each).
    auto const dir        = std::filesystem::temp_directory_path();
    auto const train_path = dir / "bonsai_mc_train.csv";
    auto const valid_path = dir / "bonsai_mc_valid.csv";
    {
        std::ofstream out(train_path);
        out << "label,f1\n";
        for (int rep = 0; rep < 4; ++rep)
        {
            out << "0," << 0.1 + rep * 0.01 << "\n";
            out << "1," << 1.1 + rep * 0.01 << "\n";
            out << "2," << 2.1 + rep * 0.01 << "\n";
        }
    }
    {
        // One contradictory label: as the model's margins grow, this row's
        // logloss grows with them, so the valid loss eventually rises and
        // patience can fire (a perfectly separable valid set improves
        // monotonically forever under logloss).
        std::ofstream out(valid_path);
        out << "label,f1\n0,0.1\n1,1.1\n2,2.1\n1,0.11\n";
    }
    Config cfg;
    cfg.data.train              = train_path.string();
    cfg.data.valid              = {valid_path.string()};
    cfg.dispatch.objective_name = "softmax";
    cfg.objective.n_classes     = 3;
    // Every distinct value gets its own cut, so the classes are
    // bin-separable (stride-2 cuts merge 0.13 and 1.1 into one bin).
    cfg.bin_mapper.max_bin                   = 100;
    cfg.bin_mapper.n_samples                 = 100;
    cfg.booster_config.n_iters               = 60;
    cfg.booster_config.learning_rate         = 0.5F;
    cfg.booster_config.early_stopping_rounds = 3;
    cfg.tree_config.min_data_in_leaf         = 0;
    cfg.tree_config.min_child_hess           = 0.0F;

    auto const loaded  = load_train_and_validation_from_csv(cfg);
    auto       booster = train_with_progress(cfg, loaded, {});
    CHECK(booster->n_iters() < 60);

    // Whole rounds only: the tree count must be a multiple of n_classes
    // (n_iters() reports rounds, so seeing it at all proves the division).
    // The truncated model still separates the three clean valid rows.
    std::vector<float> preds(4);
    booster->predict(loaded.validation->features.view(), preds);
    CHECK(preds[0] == Catch::Approx(0.0F));
    CHECK(preds[1] == Catch::Approx(1.0F));
    CHECK(preds[2] == Catch::Approx(2.0F));

    Config no_es                               = cfg;
    no_es.booster_config.early_stopping_rounds = 0;
    auto const loaded2 = load_train_and_validation_from_csv(no_es);
    auto       full    = train_with_progress(no_es, loaded2, {});
    CHECK(full->n_iters() == 60);
    std::filesystem::remove(train_path);
    std::filesystem::remove(valid_path);
}

TEST_CASE("train_with_progress: early stopping after warm start keeps the "
          "warm rounds",
          "[cli_pipeline][early_stop][warm_start]")
{
    // Train 5 rounds, save, continue with early stopping: the truncated
    // model must keep the 5 warm rounds the best loss was measured on top
    // of. The pre-fix truncate(best_iter + 1) kept only a session-local
    // prefix of the WARM model.
    Config cfg;
    cfg.data.train                   = k_tiny_path;
    cfg.data.valid                   = {k_tiny_path};
    cfg.bin_mapper.max_bin           = 8;
    cfg.bin_mapper.n_samples         = 100;
    cfg.booster_config.n_iters       = 5;
    cfg.booster_config.learning_rate = 0.5F;
    cfg.tree_config.min_data_in_leaf = 0;
    cfg.tree_config.min_child_hess   = 0.0F;

    auto const loaded_a = load_train_and_validation_from_csv(cfg);
    auto       first    = train_with_progress(cfg, loaded_a, {});
    auto const tmp      = std::string{BONSAI_TESTS_DATA_DIR} + "/_warm_es_tmp.msgpack";
    io::save_booster(*first, tmp, loaded_a.mappers, cfg);

    Config more                               = cfg;
    more.booster_config.n_iters               = 200;
    more.booster_config.early_stopping_rounds = 3;
    auto       reloaded                       = io::load_booster(tmp);
    auto const loaded_b =
        load_train_and_validation_with_mappers(more, std::move(reloaded.mappers));
    auto continued =
        train_with_progress(more, loaded_b, {}, std::move(reloaded.booster));
    std::remove(tmp.c_str());

    CHECK(continued->n_iters() >= 5);  // warm rounds survive truncation
    CHECK(continued->n_iters() < 205); // and early stopping did fire
}

TEST_CASE("multiclass warm start keeps the loaded class priors",
          "[cli_pipeline][warm_start][multiclass]")
{
    // Imbalanced classes make the log-prior init distinctly nonzero. After
    // save/load + one continued round, seeding the ES score matrix at the
    // warm round count must reproduce the loaded model's logloss exactly —
    // the pre-fix lazy init zeroed init_scores_, shifting every class base.
    auto const path = std::filesystem::temp_directory_path() / "bonsai_mc_prior.csv";
    {
        std::ofstream out(path);
        out << "label,f1\n";
        for (int rep = 0; rep < 8; ++rep)
        {
            out << "0," << 0.1 + rep * 0.01 << "\n";
        }
        out << "1,1.1\n2,2.1\n1,1.11\n2,2.11\n";
    }
    Config cfg;
    cfg.data.train                   = path.string();
    cfg.dispatch.objective_name      = "softmax";
    cfg.objective.n_classes          = 3;
    cfg.bin_mapper.max_bin           = 100;
    cfg.bin_mapper.n_samples         = 100;
    cfg.booster_config.n_iters       = 3;
    cfg.tree_config.min_data_in_leaf = 0;
    cfg.tree_config.min_child_hess   = 0.0F;

    auto const loaded_a = load_train_and_validation_from_csv(cfg);
    auto       first    = train_with_progress(cfg, loaded_a, {});
    auto const tmp      = std::string{BONSAI_TESTS_DATA_DIR} + "/_mc_prior_tmp.msgpack";
    io::save_booster(*first, tmp, loaded_a.mappers, cfg);

    auto        reloaded = io::load_booster(tmp);
    auto const &X        = loaded_a.train.features;
    float const loaded_loss =
        reloaded.booster->eval(X.view(), floats_view{loaded_a.train.labels});

    Config one                 = cfg;
    one.booster_config.n_iters = 1; // one continued round re-runs lazy init
    auto const loaded_b =
        load_train_and_validation_with_mappers(one, std::move(reloaded.mappers));
    auto continued =
        train_with_progress(one, loaded_b, {}, std::move(reloaded.booster));
    std::remove(path.string().c_str());
    std::remove(tmp.c_str());

    // Seed the ES matrix as of the 3 warm rounds: with priors intact this
    // is exactly the loaded model, so the losses match.
    std::vector<float> seeded(X.n_rows * continued->score_width());
    continued->seed_validation_scores(X.view(), seeded, 3);
    float const seeded_loss =
        continued->validation_loss(seeded, floats_view{loaded_a.train.labels});
    CHECK(seeded_loss == Catch::Approx(loaded_loss).margin(1e-5));
}

// INVARIANT: dart-excludes-early-stopping
// DART and early stopping are incompatible by construction and the
// combination throws rather than silently producing wrong validation
// numbers: DART rescales earlier trees every round, which invalidates the
// incrementally accumulated validation scores early stopping reads.
TEST_CASE("train_with_progress: DART and early stopping refuse to combine",
          "[cli_pipeline][train][ctor][invariant]")
{
    auto cfg          = make_tiny_config();
    cfg.data.valid    = {k_tiny_path};
    auto const loaded = load_train_and_validation_from_csv(cfg);
    REQUIRE(loaded.validation.has_value());

    cfg.booster_config.dart_drop_rate        = 0.1F;
    cfg.booster_config.early_stopping_rounds = 2;
    CHECK_THROWS_AS(train_with_progress(cfg, loaded.train, loaded.validation.value()),
                    ConfigError);

    SECTION("DART alone, with no early stopping, trains")
    {
        cfg.booster_config.early_stopping_rounds = 0;
        CHECK_NOTHROW(
            train_with_progress(cfg, loaded.train, loaded.validation.value()));
    }
}
