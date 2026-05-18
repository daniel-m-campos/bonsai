#include "bonsai/cli/pipeline.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <print>
#include <string>
#include <utility>
#include <vector>

#include "bonsai/booster.hpp"
#include "bonsai/cli/common.hpp"
#include "bonsai/config/config.hpp"
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
    auto booster       = make_booster(cfg);
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
    FeatureBuffer buf;
};

ParsedFeatures parse_and_buffer(std::string const &path, DataConfig const &data_cfg)
{
    auto batch = detail::csv::parse(path, data_cfg);
    auto buf   = to_feature_buffer(batch);
    return ParsedFeatures{.batch = std::move(batch), .buf = std::move(buf)};
}

} // namespace

ScoredBatch score_csv(IBooster const &booster, std::string const &path,
                      DataConfig const &data_cfg)
{
    auto pf = parse_and_buffer(path, data_cfg);
    std::vector<float> raw(pf.buf.n_rows);
    booster.predict(pf.buf.view(), raw);
    return ScoredBatch{.features = std::move(pf.buf), .raw_scores = std::move(raw)};
}

ScoredAndLabeled score_and_label_csv(IBooster const &booster, std::string const &path,
                                     DataConfig const &data_cfg)
{
    auto pf = parse_and_buffer(path, data_cfg);
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
