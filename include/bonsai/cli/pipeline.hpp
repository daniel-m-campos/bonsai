#pragma once

#include <cstddef>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/cli/common.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/dataset.hpp"

namespace bonsai::cli
{

struct LoadedTrain
{
    BinMappers mappers;
    Dataset train;
};

// Step 1 of training: fit bin mappers from `path`, then bin the same file.
LoadedTrain load_train_from_csv(Config const &cfg, std::string const &path);

// Progress callback: receives (iter_one_based, total_iters) after each
// boosting iteration. The helper calls it every iteration; the caller throttles
// (e.g. `fit`'s "every 10" check lives in the callback body).
using ProgressFn = std::function<void(std::size_t, std::size_t)>;

// Step 2 of training: build a booster from cfg via the registry and run
// `cfg.booster_config.n_iters` iterations of `update_one_iter` on `train`.
std::unique_ptr<IBooster> train_in_memory(Config const &cfg, Dataset const &train,
                                          ProgressFn const &on_progress = {});

struct ScoredBatch
{
    FeatureBuffer features;
    std::vector<float> raw_scores;
};

// Parse CSV at `path`, build a row-major feature buffer, predict raw scores.
// No link inverse applied — caller decides via apply_link_inverse_by_name.
ScoredBatch score_csv(IBooster const &booster, std::string const &path,
                      DataConfig const &data_cfg);

struct ScoredAndLabeled
{
    FeatureBuffer features;
    std::vector<float> raw_scores;
    std::vector<float> labels;
};

// Like score_csv but also extracts the label column for downstream eval.
ScoredAndLabeled score_and_label_csv(IBooster const &booster, std::string const &path,
                                     DataConfig const &data_cfg);

// Writes a single-column predictions CSV (header "prediction").
void write_predictions(std::FILE *out, std::vector<float> const &y_hat);

} // namespace bonsai::cli
