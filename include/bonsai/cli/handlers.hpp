#pragma once

#include <string>

#include "bonsai/cli/common.hpp"

namespace bonsai::cli
{

struct FitOpts
{
    CommonOpts common;
    std::string model_path;
};

struct PredictOpts
{
    CommonOpts common;
    std::string model_path;
    std::string data_path;
    std::string out_path; // empty -> stdout
    bool apply_link = true;
};

struct EvalOpts
{
    CommonOpts common;
    std::string model_path;
    std::string data_path;
};

struct BenchOpts
{
    CommonOpts common;
    std::string model_path; // optional output
};

int run_fit(FitOpts const &opts);
int run_predict(PredictOpts const &opts);
int run_eval(EvalOpts const &opts);
int run_bench(BenchOpts const &opts);
int run_info();

} // namespace bonsai::cli
