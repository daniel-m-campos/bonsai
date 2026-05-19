#include <cstdlib>
#include <exception>
#include <print>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include "bonsai/cli/common.hpp"
#include "bonsai/cli/handlers.hpp"
#include "bonsai/config/errors.hpp"

namespace
{

void add_common(CLI::App *cmd, bonsai::cli::CommonOpts &opts,
                std::vector<std::string> &set_kvs)
{
    cmd->add_option("-c,--config", opts.config_path, "Path to a TOML config file")
        ->check(CLI::ExistingFile);
    cmd->add_option("--set", set_kvs,
                    "Override a dotted config key (repeatable), e.g. "
                    "--set tree.max_depth=8")
        ->take_all();
}

void collect_overrides(std::vector<std::string> const &set_kvs,
                       bonsai::cli::CommonOpts &opts)
{
    for (auto const &kv : set_kvs)
    {
        auto const eq = kv.find('=');
        if (eq == std::string::npos)
        {
            throw bonsai::ConfigError("--set: expected key=value, got '" + kv + "'");
        }
        opts.overrides.push_back({kv.substr(0, eq), kv.substr(eq + 1)});
    }
}

} // namespace

int main(int argc, char *argv[])
{
    CLI::App app{"bonsai: a histogram gradient-boosted tree CLI"};
    app.require_subcommand(1);

    using namespace bonsai::cli;

    FitOpts fit;
    std::vector<std::string> fit_kvs;
    auto *fit_cmd = app.add_subcommand("fit", "Train a model from a CSV dataset");
    add_common(fit_cmd, fit.common, fit_kvs);
    fit_cmd->add_option("--model", fit.model_path, "Output model file (MessagePack)");

    PredictOpts predict;
    std::vector<std::string> predict_kvs;
    auto *predict_cmd = app.add_subcommand("predict", "Predict on a CSV dataset");
    add_common(predict_cmd, predict.common, predict_kvs);
    predict_cmd->add_option("--model", predict.model_path, "Input model file")
        ->required();
    predict_cmd->add_option("--data", predict.data_path,
                            "Input CSV file (overrides [data].test)");
    predict_cmd->add_option("--out", predict.out_path, "Output CSV (default: stdout)");
    predict_cmd->add_flag("!--raw-scores", predict.apply_link,
                          "Skip the link inverse for classification objectives");

    EvalOpts eval;
    std::vector<std::string> eval_kvs;
    auto *eval_cmd = app.add_subcommand("eval", "Evaluate a model on a CSV dataset");
    add_common(eval_cmd, eval.common, eval_kvs);
    eval_cmd->add_option("--model", eval.model_path, "Input model file")->required();
    eval_cmd->add_option("--data", eval.data_path,
                         "Input CSV file (overrides [data].test)");

    BenchOpts bench;
    std::vector<std::string> bench_kvs;
    auto *bench_cmd =
        app.add_subcommand("bench", "Time fit+predict for a config and dataset");
    add_common(bench_cmd, bench.common, bench_kvs);
    bench_cmd->add_option("--model", bench.model_path, "Optional output model file");

    app.add_subcommand("info", "Print available (objective, grower, sampler) combos");

    CLI11_PARSE(app, argc, argv);

    try
    {
        if (fit_cmd->parsed())
        {
            collect_overrides(fit_kvs, fit.common);
            return run_fit(fit);
        }
        if (predict_cmd->parsed())
        {
            collect_overrides(predict_kvs, predict.common);
            return run_predict(predict);
        }
        if (eval_cmd->parsed())
        {
            collect_overrides(eval_kvs, eval.common);
            return run_eval(eval);
        }
        if (bench_cmd->parsed())
        {
            collect_overrides(bench_kvs, bench.common);
            return run_bench(bench);
        }
        if (app.got_subcommand("info"))
        {
            return run_info();
        }
    }
    catch (std::exception const &e)
    {
        std::println("bonsai: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
