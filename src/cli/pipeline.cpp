#include "bonsai/cli/pipeline.hpp"

#include <format>
#include <ranges>

#include "bonsai/cuda/histogram_engine.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/cli/common.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/config/errors.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/detail/perf.hpp"
#include "bonsai/io/csv.hpp"
#include "bonsai/registry/make_booster.hpp"
#include "bonsai/registry/objective_dispatch.hpp"
#include "bonsai/types.hpp"

namespace bonsai::cli
{

LoadedTrain load_train_from_csv(Config const &cfg, std::string const &path)
{
    auto const batch   = detail::parse_input(path, cfg.data);
    auto       mappers = BinMappers::fit(batch, cfg.bin_mapper);
    select_device_for(cfg);
    auto plane = grower_runs_on_device(cfg.dispatch.grower_name)
                     ? cuda_ingest(batch, mappers)
                     : nullptr;
    auto train = Dataset::bin(batch, mappers, cfg.data, std::move(plane));
    return LoadedTrain{.mappers = std::move(mappers), .train = std::move(train)};
}

namespace
{

void check_label_domain(Config const &cfg, floats_view labels, std::string_view which)
{
    auto const &name   = cfg.dispatch.objective_name;
    auto const  refuse = [&](auto in_domain, std::string const &rule)
    {
        auto const bad = std::ranges::find_if_not(labels, in_domain);
        if (bad != labels.end())
        {
            throw std::invalid_argument(std::format(
                "{} labels must be {}; {} labels include {}", name, rule, which, *bad));
        }
    };
    if (name == "logloss")
    {
        refuse([](float v) { return v == 0.0F || v == 1.0F; }, "0 or 1");
    }
    else if (name == "softmax")
    {
        auto const k = static_cast<float>(cfg.objective.n_classes);
        refuse([k](float v) { return v >= 0.0F && v < k && v == std::floor(v); },
               std::format("integers in [0, {})", cfg.objective.n_classes));
    }
}

} // namespace

std::unique_ptr<ITrainableBooster>
train_in_memory(Config const &cfg, Dataset const &train, ProgressFn const &on_progress)
{
    check_label_domain(cfg, train.labels(), "train");
    select_device_for(cfg);
    auto       booster = make_booster(cfg);
    auto const n_iters = cfg.booster_config.n_iters;
    for (uint32_t i = 0; i < n_iters; ++i)
    {
        booster->update_one_iter(train);
        if (on_progress)
        {
            on_progress(static_cast<size_t>(i) + 1, static_cast<size_t>(n_iters));
        }
    }
    return booster;
}

namespace
{

struct ParsedFeatures
{
    detail::ColumnBatch batch;
    FeatureBuffer       buf;
};

ParsedFeatures parse_and_buffer(std::string const &path, DataConfig const &data_cfg)
{
    auto batch = detail::parse_input(path, data_cfg);
    auto buf   = to_feature_buffer(batch);
    return ParsedFeatures{.batch = std::move(batch), .buf = std::move(buf)};
}

LabeledData make_labeled(detail::ColumnBatch const &batch, DataConfig const &data_cfg,
                         BinMappers const &mappers)
{
    auto               features = to_feature_buffer(batch);
    auto               dataset  = Dataset::bin(batch, mappers, data_cfg);
    std::vector<float> labels(batch.labels.begin(), batch.labels.end());
    return LabeledData{.dataset  = std::move(dataset),
                       .features = std::move(features),
                       .labels   = std::move(labels)};
}

LabeledData load_labeled(std::string const &path, DataConfig const &data_cfg,
                         BinMappers const &mappers)
{
    return make_labeled(detail::parse_input(path, data_cfg), data_cfg, mappers);
}

LabeledData load_validation_labeled(std::string const &path, DataConfig const &data_cfg)
{
    auto               batch    = detail::parse_input(path, data_cfg);
    auto               features = to_feature_buffer(batch);
    std::vector<float> labels(batch.labels.begin(), batch.labels.end());
    return LabeledData{
        .dataset = {}, .features = std::move(features), .labels = std::move(labels)};
}

std::optional<LabeledData> load_validation_checked(Config const     &cfg,
                                                   BinMappers const &mappers)
{
    if (cfg.data.valid.empty())
    {
        return std::nullopt;
    }
    auto validation = load_validation_labeled(cfg.data.valid[0], cfg.data);
    require_n_features(validation.features.n_features, mappers.size(),
                       "'" + cfg.data.valid[0] + "'");
    return validation;
}

} // namespace

Config reconcile_warm_start(Config cfg, Config const &loaded_cfg,
                            std::vector<std::string> const &stated_keys)
{
    auto const named = [&](std::string_view key)
    { return std::ranges::find(stated_keys, key) != stated_keys.end(); };
    auto const field = [&](std::string_view key, auto &value, auto const &loaded_value,
                           auto const &default_value)
    {
        if (value == loaded_value)
        {
            return;
        }
        if (named(key) || value != default_value)
        {
            throw std::invalid_argument(std::format(
                "a warm start continues the loaded model, and {}={} disagrees "
                "with the model's {}; drop the key or restate the model's own",
                key, value, loaded_value));
        }
        value = loaded_value;
    };
    DispatchConfig const  dd;
    ObjectiveConfig const od;
    field("dispatch.objective_name", cfg.dispatch.objective_name,
          loaded_cfg.dispatch.objective_name, dd.objective_name);
    field("dispatch.grower_name", cfg.dispatch.grower_name,
          loaded_cfg.dispatch.grower_name, dd.grower_name);
    field("dispatch.sampler_name", cfg.dispatch.sampler_name,
          loaded_cfg.dispatch.sampler_name, dd.sampler_name);
    field("objective.huber_delta", cfg.objective.huber_delta,
          loaded_cfg.objective.huber_delta, od.huber_delta);
    field("objective.quantile_alpha", cfg.objective.quantile_alpha,
          loaded_cfg.objective.quantile_alpha, od.quantile_alpha);
    field("objective.n_classes", cfg.objective.n_classes,
          loaded_cfg.objective.n_classes, od.n_classes);
    return cfg;
}

LoadedTrainValidation load_train_and_validation_with_mappers(Config const &cfg,
                                                             BinMappers    mappers)
{
    auto train      = load_labeled(cfg.data.train, cfg.data, mappers);
    auto validation = load_validation_checked(cfg, mappers);
    return LoadedTrainValidation{.mappers    = std::move(mappers),
                                 .train      = std::move(train),
                                 .validation = std::move(validation)};
}

LoadedTrainValidation load_train_and_validation_from_csv(Config const &cfg)
{
    auto const train_batch = detail::parse_input(cfg.data.train, cfg.data);
    auto       mappers     = BinMappers::fit(train_batch, cfg.bin_mapper);
    auto       train       = make_labeled(train_batch, cfg.data, mappers);

    if (cfg.data.valid.size() > 1)
    {
        std::println(stderr,
                     "fit: data.valid has {} entries; only the first is used "
                     "for per-iter eval metrics",
                     cfg.data.valid.size());
    }
    auto validation = load_validation_checked(cfg, mappers);

    return LoadedTrainValidation{.mappers    = std::move(mappers),
                                 .train      = std::move(train),
                                 .validation = std::move(validation)};
}

std::unique_ptr<ITrainableBooster> train_with_progress(
    Config const &cfg, LoadedTrainValidation const &loaded, FitTickFn const &on_tick,
    std::unique_ptr<ITrainableBooster> initial, EvalHistoryRef eval_history)
{
    if (loaded.validation)
    {
        return train_with_progress(cfg, loaded.train, *loaded.validation, on_tick,
                                   std::move(initial), eval_history);
    }
    return train_with_progress(cfg, loaded.train, on_tick, std::move(initial),
                               eval_history);
}

namespace
{

constexpr size_t k_transform_bytes = 256;

bool bin_validation_pays(size_t n_features, uint8_t max_depth, uint32_t rounds)
{
    size_t const lines  = size_t{max_depth} * 64;
    size_t const saving = std::min(n_features * 4, lines) - std::min(n_features, lines);
    return rounds * saving > n_features * k_transform_bytes;
}

bool has_raw_rows(LabeledData const &data)
{
    return data.features.n_rows == 0 || !data.features.data.empty() ||
           !data.features.borrowed.empty();
}

size_t eval_row_count(LabeledData const &data)
{
    return data.dataset.plane_n_rows() > 0 ? data.dataset.view_n_rows()
                                           : data.features.n_rows;
}

Dataset bin_validation(LabeledData const &validation, Dataset const &train,
                       DataConfig const &data_cfg)
{
    detail::Phase<&detail::FitProfiler::eval_bin_s> phase;
    return Dataset::bin(validation.features.view(), validation.labels, train.mappers(),
                        data_cfg);
}

void route_last_round(ITrainableBooster const &booster, features_view X,
                      floats_out scores)
{
    detail::Phase<&detail::FitProfiler::eval_route_s> phase;
    booster.accumulate_last_round(X, scores);
}

void route_last_round_binned(ITrainableBooster const &booster, Dataset const &bins,
                             floats_out scores)
{
    detail::Phase<&detail::FitProfiler::eval_route_s> phase;
    booster.accumulate_last_round_binned(bins, scores);
}

bool route_last_round_resident(ITrainableBooster &booster, Dataset const &bins,
                               floats_out scores, std::optional<float> &loss)
{
    detail::Phase<&detail::FitProfiler::eval_route_s> phase;
    return booster.accumulate_last_round_resident(bins, scores, loss);
}

float round_validation_loss(ITrainableBooster const &booster,
                            std::span<float const> scores, floats_view labels)
{
    detail::Phase<&detail::FitProfiler::eval_loss_s> phase;
    return booster.validation_loss(scores, labels);
}

class ValidationScorer
{
  public:
    ValidationScorer(Config const &cfg, Dataset const &train, LabeledData const &valid,
                     [[maybe_unused]] bool warm_start)
        : cfg_(cfg), train_(train), valid_(valid),
          device_grower_(grower_runs_on_device(cfg.dispatch.grower_name))
    {
        if (valid.dataset.view_n_rows() > 0 && valid.dataset.bins_are_u8())
        {
            bins_ = &valid.dataset;
        }
        else
        {
            defer_binning_ = valid.dataset.plane_n_rows() == 0 && train.bins_are_u8() &&
                             has_raw_rows(valid);
        }
        assert(has_raw_rows(valid) || (bins_ != nullptr && !warm_start));
        assert(valid.dataset.row_view().is_identity() ||
               (bins_ != nullptr && !warm_start));
        if (!valid.dataset.row_view().is_identity())
        {
            narrowed_labels_ = valid.dataset.row_view().gather(valid.labels);
        }
    }

    size_t base() const
    {
        return base_;
    }

    float score_round(ITrainableBooster &booster, uint32_t i)
    {
        if (i == 0)
        {
            seed(booster);
        }
        bin_if_due(i);
        if (i == 0)
        {
            arm_device(booster);
        }
        std::optional<float> const device_loss = route(booster);
        return device_loss.has_value()
                   ? *device_loss
                   : round_validation_loss(booster, scores_, labels());
    }

  private:
    void seed(ITrainableBooster const &booster)
    {
        base_ = booster.n_iters() - 1;
        scores_.resize(eval_row_count(valid_) * booster.score_width());
        booster.seed_validation_scores(valid_.features.view(), scores_, base_);
    }

    void bin_if_due(uint32_t i)
    {
        if (!defer_binning_)
        {
            return;
        }
        if (!device_grower_ && !bin_validation_pays(valid_.features.n_features,
                                                    cfg_.tree_config.max_depth, i))
        {
            return;
        }
        binned_here_   = bin_validation(valid_, train_, cfg_.data);
        bins_          = &*binned_here_;
        defer_binning_ = false;
    }

    void arm_device(ITrainableBooster &booster)
    {
        if (!device_grower_ || bins_ == nullptr || !bins_->row_view().is_identity())
        {
            return;
        }
        detail::Phase<&detail::FitProfiler::eval_arm_s> phase;
        device_eval_ =
            booster.begin_resident_validation(*bins_, std::span<float const>{scores_});
    }

    std::optional<float> route(ITrainableBooster &booster)
    {
        std::optional<float> device_loss;
        if (bins_ == nullptr)
        {
            route_last_round(booster, valid_.features.view(), scores_);
            return device_loss;
        }
        if (device_eval_)
        {
            device_eval_ =
                route_last_round_resident(booster, *bins_, scores_, device_loss);
        }
        if (!device_eval_)
        {
            route_last_round_binned(booster, *bins_, scores_);
        }
        return device_loss;
    }

    floats_view labels() const
    {
        return valid_.dataset.row_view().is_identity() ? floats_view{valid_.labels}
                                                       : floats_view{narrowed_labels_};
    }

    Config const          &cfg_;
    Dataset const         &train_;
    LabeledData const     &valid_;
    bool const             device_grower_;
    Dataset const         *bins_          = nullptr;
    bool                   defer_binning_ = false;
    bool                   device_eval_   = false;
    std::optional<Dataset> binned_here_;
    std::vector<float>     narrowed_labels_;
    std::vector<float>     scores_;
    size_t                 base_ = 0;
};

class EarlyStopping
{
  public:
    explicit EarlyStopping(uint32_t rounds) : rounds_(rounds) {}

    bool stops_after(uint32_t i, float loss)
    {
        if (i == 0 || loss < best_loss_)
        {
            best_loss_ = loss;
            best_iter_ = i;
            return false;
        }
        return i - best_iter_ >= rounds_;
    }

    float best_loss() const
    {
        return best_loss_;
    }

    uint32_t best_iter() const
    {
        return best_iter_;
    }

  private:
    uint32_t rounds_;
    float    best_loss_ = 0.0F;
    uint32_t best_iter_ = 0;
};

using ValidationRef = std::optional<std::reference_wrapper<LabeledData const>>;

class FitTicker
{
  public:
    FitTicker(Config const &cfg, LabeledData const &train, ValidationRef validation,
              FitTickFn const &on_tick)
        : on_tick_(on_tick), train_(train), validation_(validation),
          n_iters_(cfg.booster_config.n_iters),
          enabled_(cfg.booster_config.log_intervals > 0 && static_cast<bool>(on_tick)),
          period_(enabled_ ? std::max<uint32_t>(
                                 1, n_iters_ / std::max<uint32_t>(
                                                   1, cfg.booster_config.log_intervals))
                           : n_iters_)
    {
        if (!enabled_)
        {
            return;
        }
        train_preds_.resize(train.features.n_rows);
        if (validation)
        {
            validation_preds_.resize(validation->get().features.n_rows);
        }
    }

    void baseline(IBooster const &booster)
    {
        if (enabled_)
        {
            fire(booster, 0);
        }
    }

    void after_round(IBooster const &booster, uint32_t i)
    {
        if (!enabled_)
        {
            return;
        }
        auto const one_based = i + 1;
        if (one_based % period_ == 0 || one_based == n_iters_)
        {
            fire(booster, one_based);
        }
    }

  private:
    void fire(IBooster const &booster, size_t iter)
    {
        booster.predict(train_.features.view(), train_preds_);
        floats_out  v_preds;
        floats_view v_labels;
        if (validation_)
        {
            booster.predict(validation_->get().features.view(), validation_preds_);
            v_preds  = validation_preds_;
            v_labels = validation_->get().labels;
        }
        on_tick_(FitTick{
            .iter              = iter,
            .n_iters           = static_cast<size_t>(n_iters_),
            .train_preds       = train_preds_,
            .train_labels      = train_.labels,
            .validation_preds  = v_preds,
            .validation_labels = v_labels,
        });
    }

    FitTickFn const   &on_tick_;
    LabeledData const &train_;
    ValidationRef      validation_;
    uint32_t           n_iters_;
    bool               enabled_;
    uint32_t           period_;
    std::vector<float> train_preds_;
    std::vector<float> validation_preds_;
};

struct EvalPlan
{
    bool           early_stop = false;
    ValidationRef  scored;
    EvalHistoryRef history;
};

EvalPlan eval_plan(Config const &cfg, ValidationRef validation,
                   EvalHistoryRef eval_history)
{
    EvalPlan plan;
    plan.early_stop =
        cfg.booster_config.early_stopping_rounds > 0 && validation.has_value();
    bool const track_history = eval_history.has_value() && validation.has_value() &&
                               eval_row_count(validation->get()) > 0 &&
                               cfg.booster_config.dart_drop_rate == 0.0F;
    if (plan.early_stop && eval_row_count(validation->get()) == 0)
    {
        throw ConfigError("early stopping needs a non-empty validation set");
    }
    if (plan.early_stop && cfg.booster_config.dart_drop_rate > 0.0F)
    {
        throw ConfigError("early_stopping_rounds cannot be combined with "
                          "dart_drop_rate");
    }
    if (plan.early_stop || track_history)
    {
        plan.scored = validation;
    }
    if (track_history)
    {
        plan.history = eval_history;
    }
    return plan;
}

void record_eval(std::vector<float> &history, ValidationScorer const &scorer,
                 uint32_t i, float loss)
{
    if (i == 0 && scorer.base() > 0)
    {
        history.insert(history.end(), scorer.base(),
                       std::numeric_limits<float>::quiet_NaN());
    }
    history.push_back(loss);
}

void announce_early_stop(Config const &cfg, uint32_t i, EarlyStopping const &stopper)
{
    std::println("fit: early stop at iter {} (best iter {}, valid {} loss {})", i + 1,
                 stopper.best_iter() + 1, cfg.dispatch.objective_name,
                 stopper.best_loss());
}

std::unique_ptr<ITrainableBooster>
train_impl(Config const &cfg, LabeledData const &train, ValidationRef validation,
           FitTickFn const &on_tick, std::unique_ptr<ITrainableBooster> initial,
           EvalHistoryRef eval_history)
{
    check_label_domain(cfg, train.labels, "train");
    if (validation)
    {
        check_label_domain(cfg, validation->get().labels, "validation");
    }
    select_device_for(cfg);
    [[maybe_unused]] bool const warm_start = initial != nullptr;
    auto booster = initial ? std::move(initial) : make_booster(cfg);

    FitTicker ticker(cfg, train, validation, on_tick);
    ticker.baseline(*booster);

    auto const                      plan = eval_plan(cfg, validation, eval_history);
    std::optional<ValidationScorer> scorer;
    if (plan.scored)
    {
        scorer.emplace(cfg, train.dataset, plan.scored->get(), warm_start);
    }
    EarlyStopping stopper(cfg.booster_config.early_stopping_rounds);

    for (uint32_t i = 0; i < cfg.booster_config.n_iters; ++i)
    {
        booster->update_one_iter(train.dataset);
        ticker.after_round(*booster, i);
        if (!scorer)
        {
            continue;
        }
        float const loss = scorer->score_round(*booster, i);
        if (plan.history)
        {
            record_eval(plan.history->get(), *scorer, i, loss);
        }
        if (plan.early_stop && stopper.stops_after(i, loss))
        {
            booster->truncate(scorer->base() + stopper.best_iter() + 1);
            if (on_tick)
            {
                announce_early_stop(cfg, i, stopper);
            }
            break;
        }
    }

    return booster;
}

} // namespace

std::unique_ptr<ITrainableBooster> train_with_progress(
    Config const &cfg, LabeledData const &train, FitTickFn const &on_tick,
    std::unique_ptr<ITrainableBooster> initial, EvalHistoryRef eval_history)
{
    return train_impl(cfg, train, {}, on_tick, std::move(initial), eval_history);
}

std::unique_ptr<ITrainableBooster>
train_with_progress(Config const &cfg, LabeledData const &train,
                    LabeledData const &validation, FitTickFn const &on_tick,
                    std::unique_ptr<ITrainableBooster> initial,
                    EvalHistoryRef                     eval_history)
{
    return train_impl(cfg, train, std::ref(validation), on_tick, std::move(initial),
                      eval_history);
}

ScoredBatch score_csv(IBooster const &booster, std::string const &path,
                      DataConfig const &data_cfg, size_t n_features, size_t n_trees)
{
    auto pf = parse_and_buffer(path, data_cfg);
    require_n_features(pf.buf.n_features, n_features, "'" + path + "'");
    std::vector<float> raw(pf.buf.n_rows);
    booster.predict_at(pf.buf.view(), raw, n_trees);
    return ScoredBatch{.features = std::move(pf.buf), .raw_scores = std::move(raw)};
}

ScoredAndLabeled score_and_label_csv(IBooster const &booster, std::string const &path,
                                     DataConfig const &data_cfg, size_t n_features)
{
    auto pf = parse_and_buffer(path, data_cfg);
    require_n_features(pf.buf.n_features, n_features, "'" + path + "'");
    std::vector<float> raw(pf.buf.n_rows);
    booster.predict(pf.buf.view(), raw);
    std::vector<float> labels(pf.batch.labels.begin(), pf.batch.labels.end());
    return ScoredAndLabeled{.features   = std::move(pf.buf),
                            .raw_scores = std::move(raw),
                            .labels     = std::move(labels)};
}

void write_predictions(std::FILE *out, std::vector<float> const &y_hat)
{
    std::println(out, "prediction");
    for (float const v : y_hat)
    {
        std::println(out, "{}", v);
    }
}

} // namespace bonsai::cli
