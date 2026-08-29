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

namespace
{

void select_device_for(Config const &cfg)
{
    if (cfg.dispatch.grower_name.starts_with("cuda"))
    {
        cuda_select_device(cfg.parallel.device_id);
    }
}

} // namespace

LoadedTrain load_train_from_csv(Config const &cfg, std::string const &path)
{
    auto const batch   = detail::parse_input(path, cfg.data);
    auto       mappers = BinMappers::fit(batch, cfg.bin_mapper);
    select_device_for(cfg);
    auto plane = cfg.dispatch.grower_name.starts_with("cuda")
                     ? cuda_ingest(batch, mappers)
                     : nullptr;
    auto train = Dataset::bin(batch, mappers, cfg.data, std::move(plane));
    return LoadedTrain{.mappers = std::move(mappers), .train = std::move(train)};
}

namespace
{

void check_label_domain(Config const &cfg, floats_view labels, std::string_view which)
{
    auto const &name = cfg.dispatch.objective_name;
    if (name == "logloss")
    {
        for (float const v : labels)
        {
            if (v != 0.0F && v != 1.0F)
            {
                throw std::invalid_argument(std::format(
                    "logloss labels must be 0 or 1; {} labels include {}", which, v));
            }
        }
        return;
    }
    if (name == "softmax")
    {
        auto const k = static_cast<float>(cfg.objective.n_classes);
        for (float const v : labels)
        {
            if (!(v >= 0.0F) || v >= k || v != std::floor(v))
            {
                throw std::invalid_argument(
                    std::format("softmax labels must be integers in [0, {}); {} labels "
                                "include {}",
                                cfg.objective.n_classes, which, v));
            }
        }
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

} // namespace

namespace
{

void check_csv_width(size_t given, size_t expected, std::string const &path)
{
    if (given != expected)
    {
        throw std::runtime_error(
            "this model was fit on " + std::to_string(expected) + " features and '" +
            path + "' parsed to " + std::to_string(given) +
            " columns; a tree routes on feature ids, so the columns must match");
    }
}

} // namespace

Config reconcile_warm_start(Config cfg, Config const &loaded_cfg,
                            std::vector<std::string> const &explicit_keys)
{
    auto const named = [&](std::string_view key)
    { return std::ranges::find(explicit_keys, key) != explicit_keys.end(); };
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
    auto                       train = load_labeled(cfg.data.train, cfg.data, mappers);
    std::optional<LabeledData> validation;
    if (!cfg.data.valid.empty())
    {
        validation = load_validation_labeled(cfg.data.valid[0], cfg.data);
        check_csv_width(validation->features.n_features, mappers.size(),
                        cfg.data.valid[0]);
    }
    return LoadedTrainValidation{.mappers    = std::move(mappers),
                                 .train      = std::move(train),
                                 .validation = std::move(validation)};
}

LoadedTrainValidation load_train_and_validation_from_csv(Config const &cfg)
{
    auto const train_batch = detail::parse_input(cfg.data.train, cfg.data);
    auto       mappers     = BinMappers::fit(train_batch, cfg.bin_mapper);
    auto       train       = make_labeled(train_batch, cfg.data, mappers);

    std::optional<LabeledData> validation;
    if (!cfg.data.valid.empty())
    {
        if (cfg.data.valid.size() > 1)
        {
            std::println(stderr,
                         "fit: data.valid has {} entries; only the first is used "
                         "for per-iter eval metrics",
                         cfg.data.valid.size());
        }
        validation = load_validation_labeled(cfg.data.valid[0], cfg.data);
        check_csv_width(validation->features.n_features, mappers.size(),
                        cfg.data.valid[0]);
    }

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

using ValidationRef = std::optional<std::reference_wrapper<LabeledData const>>;

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
    auto       booster = initial ? std::move(initial) : make_booster(cfg);
    auto const n_iters = cfg.booster_config.n_iters;
    auto const log_iv  = cfg.booster_config.log_intervals;

    bool const has_sink = static_cast<bool>(on_tick);

    bool const ticks_enabled = log_iv > 0 && has_sink;
    auto const period =
        ticks_enabled ? std::max<uint32_t>(1, n_iters / std::max<uint32_t>(1, log_iv))
                      : n_iters;

    std::vector<float> train_preds(train.features.n_rows);
    std::vector<float> validation_preds;
    if (validation)
    {
        validation_preds.resize(validation->get().features.n_rows);
    }

    auto fire_tick = [&](size_t iter)
    {
        booster->predict(train.features.view(), train_preds);
        floats_out  v_preds;
        floats_view v_labels;
        if (validation)
        {
            booster->predict(validation->get().features.view(), validation_preds);
            v_preds  = validation_preds;
            v_labels = validation->get().labels;
        }
        on_tick(FitTick{
            .iter              = iter,
            .n_iters           = static_cast<size_t>(n_iters),
            .train_preds       = train_preds,
            .train_labels      = train.labels,
            .validation_preds  = v_preds,
            .validation_labels = v_labels,
        });
    };

    if (ticks_enabled)
    {
        fire_tick(0);
    }

    auto const es_rounds  = cfg.booster_config.early_stopping_rounds;
    bool const es_enabled = es_rounds > 0 && validation.has_value();
    bool const track_eval = eval_history.has_value() && validation.has_value() &&
                            eval_row_count(validation->get()) > 0 &&
                            cfg.booster_config.dart_drop_rate == 0.0F;
    if (es_enabled && eval_row_count(validation->get()) == 0)
    {
        throw ConfigError("early stopping needs a non-empty validation set");
    }
    if (es_enabled && cfg.booster_config.dart_drop_rate > 0.0F)
    {
        throw ConfigError("early_stopping_rounds cannot be combined with "
                          "dart_drop_rate");
    }
    std::optional<Dataset> binned_here;
    Dataset const         *validation_bins = nullptr;
    bool                   defer_binning   = false;
    if (es_enabled || track_eval)
    {
        auto const &valid = validation->get();
        if (valid.dataset.view_n_rows() > 0 && valid.dataset.bins_are_u8())
        {
            validation_bins = &valid.dataset;
        }
        else
        {
            defer_binning = valid.dataset.plane_n_rows() == 0 &&
                            train.dataset.bins_are_u8() && has_raw_rows(valid);
        }
        assert(has_raw_rows(valid) || (validation_bins != nullptr && !warm_start));
        assert(valid.dataset.row_view().is_identity() ||
               (validation_bins != nullptr && !warm_start));
    }

    std::vector<float> narrowed_labels;
    floats_view        eval_labels;
    if (validation)
    {
        auto const &valid = validation->get();
        eval_labels       = valid.labels;
        if (!valid.dataset.row_view().is_identity())
        {
            narrowed_labels = valid.dataset.row_view().gather(valid.labels);
            eval_labels     = narrowed_labels;
        }
    }

    std::vector<float> es_scores;
    float              best_loss   = 0.0F;
    uint32_t           best_iter   = 0;
    size_t             es_base     = 0;
    bool const         cuda_grower = cfg.dispatch.grower_name.starts_with("cuda");
    bool               device_eval = false;

    for (uint32_t i = 0; i < n_iters; ++i)
    {
        booster->update_one_iter(train.dataset);
        if (ticks_enabled)
        {
            auto const one_based = static_cast<uint32_t>(i + 1);
            bool const is_period = (one_based % period) == 0;
            bool const is_final  = one_based == n_iters;
            if (is_period || is_final)
            {
                fire_tick(static_cast<size_t>(one_based));
            }
        }
        if (es_enabled || track_eval)
        {
            auto const &valid = validation->get();
            if (i == 0)
            {
                es_base = booster->n_iters() - 1;
                es_scores.resize(eval_row_count(valid) * booster->score_width());
                booster->seed_validation_scores(valid.features.view(), es_scores,
                                                es_base);
                if (track_eval && es_base > 0)
                {
                    eval_history->get().insert(eval_history->get().end(), es_base,
                                               std::numeric_limits<float>::quiet_NaN());
                }
            }
            if (defer_binning &&
                (cuda_grower || bin_validation_pays(valid.features.n_features,
                                                    cfg.tree_config.max_depth, i)))
            {
                binned_here     = bin_validation(valid, train.dataset, cfg.data);
                validation_bins = &*binned_here;
                defer_binning   = false;
            }
            std::optional<float> device_loss;
            if (i == 0 && cuda_grower && validation_bins != nullptr &&
                validation_bins->row_view().is_identity())
            {
                detail::Phase<&detail::FitProfiler::eval_arm_s> phase;
                device_eval = booster->begin_resident_validation(
                    *validation_bins, std::span<float const>{es_scores});
            }
            if (validation_bins != nullptr)
            {
                if (device_eval)
                {
                    device_eval = route_last_round_resident(*booster, *validation_bins,
                                                            es_scores, device_loss);
                }
                if (!device_eval)
                {
                    route_last_round_binned(*booster, *validation_bins, es_scores);
                }
            }
            else
            {
                route_last_round(*booster, valid.features.view(), es_scores);
            }
            float const loss =
                device_loss.has_value()
                    ? *device_loss
                    : round_validation_loss(*booster, es_scores, eval_labels);
            if (track_eval)
            {
                eval_history->get().push_back(loss);
            }
            if (!es_enabled)
            {
                continue;
            }
            if (i == 0 || loss < best_loss)
            {
                best_loss = loss;
                best_iter = i;
            }
            else if (i - best_iter >= es_rounds)
            {
                booster->truncate(es_base + best_iter + 1);
                if (has_sink)
                {
                    std::println("fit: early stop at iter {} (best iter {}, valid {} "
                                 "loss {})",
                                 i + 1, best_iter + 1, cfg.dispatch.objective_name,
                                 best_loss);
                }
                break;
            }
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
    check_csv_width(pf.buf.n_features, n_features, path);
    std::vector<float> raw(pf.buf.n_rows);
    booster.predict_at(pf.buf.view(), raw, n_trees);
    return ScoredBatch{.features = std::move(pf.buf), .raw_scores = std::move(raw)};
}

ScoredAndLabeled score_and_label_csv(IBooster const &booster, std::string const &path,
                                     DataConfig const &data_cfg, size_t n_features)
{
    auto pf = parse_and_buffer(path, data_cfg);
    check_csv_width(pf.buf.n_features, n_features, path);
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
