#pragma once

#include <string>

#include "bonsai/cli/common.hpp"

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
    std::string out_path;          // empty -> stdout
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

int run_fit(FitOpts const &opts);
int run_predict(PredictOpts const &opts);
int run_eval(EvalOpts const &opts);
int run_bench(BenchOpts const &opts);
int run_importance(ImportanceOpts const &opts);
int run_dump(DumpOpts const &opts);
int run_info();
int run_params();

} // namespace bonsai::cli
