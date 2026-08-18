#include "bonsai/cli/pipeline.hpp"

#include "bonsai/cuda/histogram_engine.hpp"

#include <algorithm>
#include <cassert>
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

// Device placement (parallel.device_id, issue #158) happens at every entry
// that precedes device work, because cudaSetDevice is thread-local: ingest
// and training may run on different threads (the Python Dataset flow). CPU
// growers never touch it.
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
    // The ingest transaction (decision 54): cuda growers bin on the device;
    // cuda_ingest declines (nullptr) without a backend/device.
    select_device_for(cfg);
    auto plane = cfg.dispatch.grower_name.starts_with("cuda")
                     ? cuda_ingest(batch, mappers)
                     : nullptr;
    auto train = Dataset::bin(batch, mappers, cfg.data, std::move(plane));
    return LoadedTrain{.mappers = std::move(mappers), .train = std::move(train)};
}

std::unique_ptr<IBooster> train_in_memory(Config const &cfg, Dataset const &train,
                                          ProgressFn const &on_progress)
{
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

// Validation sets feed train_with_progress, which takes features and labels
// and bins them itself, once the rounds it has run have paid for the pass.
// Binning here would charge every fit for it, including the short ones that
// never get there.
LabeledData load_validation_labeled(std::string const &path, DataConfig const &data_cfg)
{
    auto               batch    = detail::parse_input(path, data_cfg);
    auto               features = to_feature_buffer(batch);
    std::vector<float> labels(batch.labels.begin(), batch.labels.end());
    return LabeledData{
        .dataset = {}, .features = std::move(features), .labels = std::move(labels)};
}

} // namespace

LoadedTrainValidation load_train_and_validation_with_mappers(Config const &cfg,
                                                             BinMappers    mappers)
{
    auto                       train = load_labeled(cfg.data.train, cfg.data, mappers);
    std::optional<LabeledData> validation;
    if (!cfg.data.valid.empty())
    {
        validation = load_validation_labeled(cfg.data.valid[0], cfg.data);
    }
    return LoadedTrainValidation{.mappers    = std::move(mappers),
                                 .train      = std::move(train),
                                 .validation = std::move(validation)};
}

LoadedTrainValidation load_train_and_validation_from_csv(Config const &cfg)
{
    // Parse the train CSV once: fit mappers and bin from the same batch.
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
    }

    return LoadedTrainValidation{.mappers    = std::move(mappers),
                                 .train      = std::move(train),
                                 .validation = std::move(validation)};
}

std::unique_ptr<IBooster> train_with_progress(Config const                &cfg,
                                              LoadedTrainValidation const &loaded,
                                              FitTickFn const             &on_tick,
                                              std::unique_ptr<IBooster>    initial,
                                              EvalHistoryRef               eval_history)
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

// Whether `rounds` of the raw walk have cost more than one bin pass would
// have. Per validation row (the row count is on both sides): a round saves
// min(4p, 64d) - min(p, 64d) bytes, a float row against a bin row with each
// capped by the lines a depth-capped walk touches, and the pass costs p
// transforms priced at k_transform_bytes of that same bandwidth.
constexpr size_t k_transform_bytes = 256;

bool bin_validation_pays(size_t n_features, uint8_t max_depth, uint32_t rounds)
{
    size_t const lines  = size_t{max_depth} * 64;
    size_t const saving = std::min(n_features * 4, lines) - std::min(n_features, lines);
    return rounds * saving > n_features * k_transform_bytes;
}

// Whether the caller's raw rows are readable. A validation set built from
// device-resident input keeps its bins and no host matrix.
bool has_raw_rows(LabeledData const &data)
{
    return data.features.n_rows == 0 || !data.features.data.empty() ||
           !data.features.borrowed.empty();
}

// The one-time bin pass, lapped so its cost shows up beside the per-round
// buckets it buys.
Dataset bin_validation(LabeledData const &validation, Dataset const &train,
                       DataConfig const &data_cfg)
{
    detail::Phase<&detail::FitProfiler::eval_bin_s> phase;
    return Dataset::bin(validation.features.view(), validation.labels, train.mappers(),
                        data_cfg);
}

// The per-round eval phases, one bucket each. The Phase covers a whole call
// the way a grower's step methods carry theirs, so the round loop below states
// what runs and the instrument boundary lives at the seam. Which of the two
// routes runs is the loop's own state, so each is its own call.
void route_last_round(IBooster const &booster, features_view X, floats_out scores)
{
    detail::Phase<&detail::FitProfiler::eval_route_s> phase;
    booster.accumulate_last_round(X, scores);
}

void route_last_round_binned(IBooster const &booster, Dataset const &bins,
                             floats_out scores)
{
    detail::Phase<&detail::FitProfiler::eval_route_s> phase;
    booster.accumulate_last_round_binned(bins, scores);
}

bool route_last_round_resident(IBooster &booster, Dataset const &bins,
                               floats_out scores)
{
    detail::Phase<&detail::FitProfiler::eval_route_s> phase;
    return booster.accumulate_last_round_resident(bins, scores);
}

float round_validation_loss(IBooster const &booster, std::span<float const> scores,
                            floats_view labels)
{
    detail::Phase<&detail::FitProfiler::eval_loss_s> phase;
    return booster.validation_loss(scores, labels);
}

// The one body behind both public forms: an engaged reference is a fit with a
// validation set, an empty one a fit without.
using ValidationRef = std::optional<std::reference_wrapper<LabeledData const>>;

std::unique_ptr<IBooster> train_impl(Config const &cfg, LabeledData const &train,
                                     ValidationRef validation, FitTickFn const &on_tick,
                                     std::unique_ptr<IBooster> initial,
                                     EvalHistoryRef            eval_history)
{
    select_device_for(cfg);
    [[maybe_unused]] bool const warm_start = initial != nullptr;
    auto       booster = initial ? std::move(initial) : make_booster(cfg);
    auto const n_iters = cfg.booster_config.n_iters;
    auto const log_iv  = cfg.booster_config.log_intervals;

    // A caller without a tick callback gets no fit-time output at all: this
    // stdout is the C runtime's, which escapes any redirection an embedder
    // installed, so the library must not write to it unsolicited.
    bool const has_sink = static_cast<bool>(on_tick);

    // Period = max(1, n_iters / log_intervals). log_iv > n_iters -> log every iter.
    // log_iv == 0 disables ticks entirely (on_tick still ignored).
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
        // iter 0 baseline: predicts from init_score (no trees yet).
        fire_tick(0);
    }

    // Early stopping: incremental validation raw scores (score_base +
    // per-tree contributions) keep the per-iteration eval O(rows), not
    // O(rows*trees).
    auto const es_rounds  = cfg.booster_config.early_stopping_rounds;
    bool const es_enabled = es_rounds > 0 && validation.has_value();
    // History shares the incremental accumulation below; DART's per-round
    // rescaling invalidates it, so no history there (es already throws).
    bool const track_eval = eval_history.has_value() && validation.has_value() &&
                            validation->get().features.n_rows > 0 &&
                            cfg.booster_config.dart_drop_rate == 0.0F;
    if (es_enabled && validation->get().features.n_rows == 0)
    {
        // A zero-row validation set makes every loss NaN, and NaN never
        // improves, so patience would fire at the first opportunity and
        // silently truncate the model.
        throw ConfigError("early stopping needs a non-empty validation set");
    }
    if (es_enabled && cfg.booster_config.dart_drop_rate > 0.0F)
    {
        // DART rescales earlier trees each iteration, which invalidates the
        // incrementally accumulated validation scores below.
        throw ConfigError("early_stopping_rounds cannot be combined with "
                          "dart_drop_rate");
    }
    // A validation set that arrives already binned is the round-1 fast lane:
    // its pass is paid, so nothing is left to price. Otherwise the fit runs
    // the raw walk and switches representation mid-run, once the rounds it has
    // actually run have cost more than the pass would have. How many rounds a
    // fit runs cannot be predicted (early stopping decides it), so counting
    // beats guessing: a short fit never pays, a long one collects. Either
    // walk routes the same rows to the same leaves, so the switch changes no
    // trace bytes. The mirror the binned walk indexes exists only at 8-bit
    // bins, which the training mappers decide for both sets alike.
    std::optional<Dataset> binned_here;
    Dataset const         *validation_bins = nullptr;
    bool                   defer_binning   = false;
    if (es_enabled || track_eval)
    {
        auto const &valid = validation->get();
        if (valid.dataset.n_rows() > 0 && valid.dataset.bins_are_u8())
        {
            validation_bins = &valid.dataset;
        }
        else
        {
            defer_binning = valid.dataset.n_rows() == 0 &&
                            train.dataset.bins_are_u8() && has_raw_rows(valid);
        }
        // The raw walk and the warm-start seed read the caller's matrix, so a
        // validation set that kept none must arrive binned and start cold.
        // The Python binding refuses the rest with a message; this is the
        // seam restating it.
        assert(has_raw_rows(valid) || (validation_bins != nullptr && !warm_start));
    }

    std::vector<float> es_scores;
    float              best_loss = 0.0F;
    uint32_t           best_iter = 0;
    size_t             es_base   = 0; // warm-start rounds present before this loop
    // A device grower bins the validation set immediately (skipping the
    // pays-off gate) so its eval plane can arm at round 0; the transform cost
    // is one upload against a per-round host walk saved.
    bool const cuda_grower = cfg.dispatch.grower_name.starts_with("cuda");
    bool       device_eval = false;

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
                // Warm start: seed with the pre-existing rounds' raw scores,
                // excluding the round just added (0 rounds = base scores).
                es_base = booster->n_iters() - 1;
                es_scores.resize(valid.features.n_rows * booster->score_width());
                booster->seed_validation_scores(valid.features.view(), es_scores,
                                                es_base);
                if (track_eval && es_base > 0)
                {
                    // NaN placeholders for the warm-start rounds keep history
                    // indices equal to absolute model rounds.
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
            if (i == 0 && cuda_grower && validation_bins != nullptr)
            {
                device_eval = booster->begin_resident_validation(
                    *validation_bins, std::span<float const>{es_scores});
            }
            if (validation_bins != nullptr)
            {
                // A device decline disarms for the rest of the fit: the plane's
                // scores stop tracking the model the moment one round routes on
                // the host instead.
                if (device_eval)
                {
                    device_eval = route_last_round_resident(*booster, *validation_bins,
                                                            es_scores);
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
            float const loss = round_validation_loss(*booster, es_scores, valid.labels);
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
                // best_iter counts THIS loop's rounds; keep the warm-start
                // rounds the best loss was measured on top of.
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

std::unique_ptr<IBooster> train_with_progress(Config const             &cfg,
                                              LabeledData const        &train,
                                              FitTickFn const          &on_tick,
                                              std::unique_ptr<IBooster> initial,
                                              EvalHistoryRef            eval_history)
{
    return train_impl(cfg, train, {}, on_tick, std::move(initial), eval_history);
}

std::unique_ptr<IBooster>
train_with_progress(Config const &cfg, LabeledData const &train,
                    LabeledData const &validation, FitTickFn const &on_tick,
                    std::unique_ptr<IBooster> initial, EvalHistoryRef eval_history)
{
    return train_impl(cfg, train, std::ref(validation), on_tick, std::move(initial),
                      eval_history);
}

ScoredBatch score_csv(IBooster const &booster, std::string const &path,
                      DataConfig const &data_cfg, size_t n_trees)
{
    auto               pf = parse_and_buffer(path, data_cfg);
    std::vector<float> raw(pf.buf.n_rows);
    booster.predict_at(pf.buf.view(), raw, n_trees);
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
