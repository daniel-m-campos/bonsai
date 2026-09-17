#pragma once

#include <cstdio>
#include <cstdlib>
#include <expected>
#include <print>
#include <string>
#include <string_view>
#include <utility>

#include "bonsai/cli/common.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/io/model.hpp"

namespace bonsai::cli
{

struct FitOpts
{
    CommonOpts  common;
    std::string model_path;
    std::string init_model_path; // warm start: continue training this model
};

struct PredictOpts
{
    CommonOpts  common;
    std::string model_path;
    std::string data_path;
    std::string out_path; // empty -> stdout
    bool        apply_link    = true;
    std::size_t num_iteration = 0; // predict with the first k trees; 0 = all
};

struct EvalOpts
{
    CommonOpts  common;
    std::string model_path;
    std::string data_path;
};

struct BenchOpts
{
    CommonOpts  common;
    std::string model_path; // optional output
};

struct ImportanceOpts
{
    std::string model_path;
};

struct DumpOpts
{
    std::string model_path;
};

struct ScoringInputs
{
    Config            cfg;
    io::LoadedBooster loaded;
    std::string       path;
};

inline std::expected<ScoringInputs, int> scoring_inputs(CommonOpts const  &common,
                                                        std::string const &model_path,
                                                        std::string const &data_path,
                                                        std::string_view   subcommand)
{
    auto cfg = resolve_config(common);
    if (dump_config(common, cfg))
    {
        return std::unexpected(EXIT_SUCCESS);
    }
    auto loaded = io::load_booster(model_path);
    auto path   = data_path.empty() ? cfg.data.test : data_path;
    if (path.empty())
    {
        std::println(stderr, "{}: data path is required (--data or [data].test)",
                     subcommand);
        return std::unexpected(2);
    }
    return ScoringInputs{
        .cfg = std::move(cfg), .loaded = std::move(loaded), .path = std::move(path)};
}

int run_fit(FitOpts const &opts);
int run_predict(PredictOpts const &opts);
int run_eval(EvalOpts const &opts);
int run_bench(BenchOpts const &opts);
int run_importance(ImportanceOpts const &opts);
int run_dump(DumpOpts const &opts);
int run_info();
int run_params();

} // namespace bonsai::cli
