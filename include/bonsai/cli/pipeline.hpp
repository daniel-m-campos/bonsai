#pragma once

#include <cstddef>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/cli/common.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/types.hpp"

namespace bonsai::cli
{

struct LoadedTrain
{
    BinMappers mappers;
    Dataset    train;
};

// Step 1 of training (minimal): fit bin mappers from `path`, then bin the same
// file. Used by tests that don't need a row-major feature buffer.
LoadedTrain load_train_from_csv(Config const &cfg, std::string const &path);

// One side of a train/valid pair. Carries both the binned Dataset (for
// update_one_iter / labels) and the row-major FeatureBuffer (for predict).
struct LabeledData
{
    Dataset            dataset;
    FeatureBuffer      features;
    std::vector<float> labels;
};

struct LoadedTrainValid
{
    BinMappers                 mappers;
    LabeledData                train;
    std::optional<LabeledData> valid;
};

// Fit bin mappers on cfg.data.train, bin it, and capture a row-major buffer.
// If cfg.data.valid is non-empty, additionally bin valid[0] with the same
// mappers. valid[1..] are ignored with a stderr warning.
LoadedTrainValid load_train_and_valid_from_csv(Config const &cfg);

// Warm-start variant: bin train/valid with EXISTING mappers (from a loaded
// model) instead of refitting them — continuation must see the same bins.
LoadedTrainValid load_train_and_valid_with_mappers(Config const &cfg,
                                                   BinMappers    mappers);

// Progress callback: receives (iter_one_based, total_iters) after each
// boosting iteration. The helper calls it every iteration; the caller throttles
// (e.g. `fit`'s "every 10" check lives in the callback body).
using ProgressFn = std::function<void(size_t, size_t)>;

// Step 2 of training: build a booster from cfg via the registry and run
// `cfg.booster_config.n_iters` iterations of `update_one_iter` on `train`.
std::unique_ptr<IBooster> train_in_memory(Config const &cfg, Dataset const &train,
                                          ProgressFn const &on_progress = {});

// Per-tick snapshot of model performance during fit. Predictions are RAW
// scores (link not applied yet); the printer decides whether to invert and
// MAY mutate the preds spans in place to do so (they alias scratch buffers
// owned by `train_with_progress` and are overwritten next tick anyway).
// Do not retain any span past the callback call.
struct FitTick
{
    size_t      iter;        // 0-based; 0 == init_score baseline
    size_t      n_iters;     // cfg.booster_config.n_iters
    floats_out  train_preds; // mutable; link-inversion target
    floats_view train_labels;
    floats_out  valid_preds; // empty if no valid set
    floats_view valid_labels;
};

using FitTickFn = std::function<void(FitTick const &)>;

// Full training entry point used by `bonsai fit`. Runs n_iters of
// update_one_iter on train; when log_intervals > 0, invokes on_tick at iter 0
// (init_score baseline), every floor(n_iters/log_intervals) iters, and at the
// final iter. Predictions live in scratch buffers owned by this function;
// callbacks must not retain references past the call.
// `initial` continues training an existing booster (warm start) instead of
// constructing a fresh one from cfg; cfg still drives n_iters / ticks /
// early stopping for the continuation.
std::unique_ptr<IBooster> train_with_progress(Config const             &cfg,
                                              LoadedTrainValid const   &loaded,
                                              FitTickFn const          &on_tick = {},
                                              std::unique_ptr<IBooster> initial = {});

// Same, but train and valid arrive separately (valid may be null). Lets a
// caller pair a long-lived pre-binned train set with a per-call validation
// set without copying the train LabeledData (the copy would also change the
// Dataset address that keys the GPU upload-skip cache, decision 54).
std::unique_ptr<IBooster> train_with_progress(Config const             &cfg,
                                              LabeledData const        &train,
                                              LabeledData const        *valid,
                                              FitTickFn const          &on_tick = {},
                                              std::unique_ptr<IBooster> initial = {});

struct ScoredBatch
{
    FeatureBuffer      features;
    std::vector<float> raw_scores;
};

// Parse CSV at `path`, build a row-major feature buffer, predict raw scores.
// No link inverse applied — caller decides via apply_link_inverse_by_name.
ScoredBatch score_csv(IBooster const &booster, std::string const &path,
                      DataConfig const &data_cfg, size_t n_trees = 0);

struct ScoredAndLabeled
{
    FeatureBuffer      features;
    std::vector<float> raw_scores;
    std::vector<float> labels;
};

// Like score_csv but also extracts the label column for downstream eval.
ScoredAndLabeled score_and_label_csv(IBooster const &booster, std::string const &path,
                                     DataConfig const &data_cfg);

// Writes a single-column predictions CSV (header "prediction").
void write_predictions(std::FILE *out, std::vector<float> const &y_hat);

} // namespace bonsai::cli
