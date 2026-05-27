#include "bonsai/cli/pipeline.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/io/csv.hpp"
#include "bonsai/registry/make_booster.hpp"

namespace bonsai::cli
{

LoadedTrain load_train_from_csv(Config const &cfg, std::string const &path)
{
    auto mappers = io::fit_from_csv(path, cfg);
    auto train   = io::read_csv(path, cfg.data, mappers);
    return LoadedTrain{.mappers = std::move(mappers), .train = std::move(train)};
}

std::unique_ptr<IBooster> train_in_memory(Config const &cfg, Dataset const &train,
                                          ProgressFn const &on_progress)
{
    auto       booster = make_booster(cfg);
    auto const n_iters = cfg.booster_config.n_iters;
    for (uint32_t i = 0; i < n_iters; ++i)
    {
        booster->update_one_iter(train);
        if (on_progress)
        {
            on_progress(static_cast<std::size_t>(i) + 1,
                        static_cast<std::size_t>(n_iters));
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
    auto batch = detail::csv::parse(path, data_cfg);
    auto buf   = to_feature_buffer(batch);
    return ParsedFeatures{.batch = std::move(batch), .buf = std::move(buf)};
}

LabeledData load_labeled(std::string const &path, DataConfig const &data_cfg,
                         BinMappers const &mappers)
{
    auto               batch    = detail::csv::parse(path, data_cfg);
    auto               features = to_feature_buffer(batch);
    auto               dataset  = Dataset::bin(batch, mappers, data_cfg);
    std::vector<float> labels(batch.labels.begin(), batch.labels.end());
    return LabeledData{.dataset  = std::move(dataset),
                       .features = std::move(features),
                       .labels   = std::move(labels)};
}

} // namespace

LoadedTrainValid load_train_and_valid_from_csv(Config const &cfg)
{
    auto mappers = io::fit_from_csv(cfg.data.train, cfg);
    auto train   = load_labeled(cfg.data.train, cfg.data, mappers);

    std::optional<LabeledData> valid;
    if (!cfg.data.valid.empty())
    {
        if (cfg.data.valid.size() > 1)
        {
            std::println(stderr,
                         "fit: data.valid has {} entries; only the first is used "
                         "for per-iter eval metrics",
                         cfg.data.valid.size());
        }
        valid = load_labeled(cfg.data.valid[0], cfg.data, mappers);
    }

    return LoadedTrainValid{.mappers = std::move(mappers),
                            .train   = std::move(train),
                            .valid   = std::move(valid)};
}

std::unique_ptr<IBooster> train_with_progress(Config const           &cfg,
                                              LoadedTrainValid const &loaded,
                                              FitTickFn const        &on_tick)
{
    auto       booster = make_booster(cfg);
    auto const n_iters = cfg.booster_config.n_iters;
    auto const log_iv  = cfg.booster_config.log_intervals;

    // Period = max(1, n_iters / log_intervals). log_iv > n_iters -> log every iter.
    // log_iv == 0 disables ticks entirely (on_tick still ignored).
    bool const ticks_enabled = log_iv > 0 && static_cast<bool>(on_tick);
    auto const period =
        ticks_enabled ? std::max<uint32_t>(1, n_iters / std::max<uint32_t>(1, log_iv))
                      : n_iters;

    std::vector<float> train_preds(loaded.train.features.n_rows);
    std::vector<float> valid_preds;
    if (loaded.valid.has_value())
    {
        valid_preds.resize(loaded.valid->features.n_rows);
    }

    auto fire_tick = [&](std::size_t iter)
    {
        booster->predict(loaded.train.features.view(), train_preds);
        std::span<float>       v_preds;
        std::span<float const> v_labels;
        if (loaded.valid.has_value())
        {
            booster->predict(loaded.valid->features.view(), valid_preds);
            v_preds  = valid_preds;
            v_labels = loaded.valid->labels;
        }
        on_tick(FitTick{
            .iter         = iter,
            .n_iters      = static_cast<std::size_t>(n_iters),
            .train_preds  = train_preds,
            .train_labels = loaded.train.labels,
            .valid_preds  = v_preds,
            .valid_labels = v_labels,
        });
    };

    if (ticks_enabled)
    {
        // iter 0 baseline: predicts from init_score (no trees yet).
        fire_tick(0);
    }

    for (uint32_t i = 0; i < n_iters; ++i)
    {
        booster->update_one_iter(loaded.train.dataset);
        if (ticks_enabled)
        {
            auto const one_based = static_cast<uint32_t>(i + 1);
            bool const is_period = (one_based % period) == 0;
            bool const is_final  = one_based == n_iters;
            if (is_period || is_final)
            {
                fire_tick(static_cast<std::size_t>(one_based));
            }
        }
    }

    return booster;
}

ScoredBatch score_csv(IBooster const &booster, std::string const &path,
                      DataConfig const &data_cfg)
{
    auto               pf = parse_and_buffer(path, data_cfg);
    std::vector<float> raw(pf.buf.n_rows);
    booster.predict(pf.buf.view(), raw);
    return ScoredBatch{.features = std::move(pf.buf), .raw_scores = std::move(raw)};
}

ScoredAndLabeled score_and_label_csv(IBooster const &booster, std::string const &path,
                                     DataConfig const &data_cfg)
{
    auto               pf = parse_and_buffer(path, data_cfg);
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
