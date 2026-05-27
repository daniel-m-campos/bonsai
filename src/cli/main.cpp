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
    cmd->add_flag("--dump-config", opts.dump_config,
                  "Print the resolved config (after -c + --set) as TOML and exit");
}

void collect_overrides(std::vector<std::string> const &set_kvs,
                       bonsai::cli::CommonOpts        &opts)
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

constexpr auto params_footer =
    "See `bonsai params` for available --set keys (default config in TOML).";

void register_fit(CLI::App &app, bonsai::cli::FitOpts &opts,
                  std::vector<std::string> &kvs, int &rc)
{
    auto *cmd = app.add_subcommand("fit", "Train a model from a CSV dataset");
    add_common(cmd, opts.common, kvs);
    cmd->add_option("--model", opts.model_path, "Output model file (MessagePack)");
    cmd->footer(params_footer);
    cmd->callback(
        [&]
        {
            collect_overrides(kvs, opts.common);
            rc = bonsai::cli::run_fit(opts);
        });
}

void register_predict(CLI::App &app, bonsai::cli::PredictOpts &opts,
                      std::vector<std::string> &kvs, int &rc)
{
    auto *cmd = app.add_subcommand("predict", "Predict on a CSV dataset");
    add_common(cmd, opts.common, kvs);
    cmd->add_option("--model", opts.model_path, "Input model file")->required();
    cmd->add_option("--data", opts.data_path, "Input CSV file (overrides [data].test)");
    cmd->add_option("--out", opts.out_path, "Output CSV (default: stdout)");
    cmd->add_flag("!--raw-scores", opts.apply_link,
                  "Skip the link inverse for classification objectives");
    cmd->footer(params_footer);
    cmd->callback(
        [&]
        {
            collect_overrides(kvs, opts.common);
            rc = bonsai::cli::run_predict(opts);
        });
}

void register_eval(CLI::App &app, bonsai::cli::EvalOpts &opts,
                   std::vector<std::string> &kvs, int &rc)
{
    auto *cmd = app.add_subcommand("eval", "Evaluate a model on a CSV dataset");
    add_common(cmd, opts.common, kvs);
    cmd->add_option("--model", opts.model_path, "Input model file")->required();
    cmd->add_option("--data", opts.data_path, "Input CSV file (overrides [data].test)");
    cmd->footer(params_footer);
    cmd->callback(
        [&]
        {
            collect_overrides(kvs, opts.common);
            rc = bonsai::cli::run_eval(opts);
        });
}

void register_bench(CLI::App &app, bonsai::cli::BenchOpts &opts,
                    std::vector<std::string> &kvs, int &rc)
{
    auto *cmd =
        app.add_subcommand("bench", "Time fit+predict for a config and dataset");
    add_common(cmd, opts.common, kvs);
    cmd->add_option("--model", opts.model_path, "Optional output model file");
    cmd->footer(params_footer);
    cmd->callback(
        [&]
        {
            collect_overrides(kvs, opts.common);
            rc = bonsai::cli::run_bench(opts);
        });
}

void register_info(CLI::App &app, int &rc)
{
    auto *cmd = app.add_subcommand(
        "info", "Print available (objective, grower, sampler) combos");
    cmd->callback([&] { rc = bonsai::cli::run_info(); });
}

void register_params(CLI::App &app, int &rc)
{
    auto *cmd = app.add_subcommand(
        "params", "Print the default config as TOML (lists all --set keys)");
    cmd->callback([&] { rc = bonsai::cli::run_params(); });
}

} // namespace

int main(int argc, char *argv[])
{
    CLI::App app{"bonsai: a histogram gradient-boosted tree CLI"};
    app.require_subcommand(1);

    using namespace bonsai::cli;

    // Each subcommand's callback fires after CLI11 finishes parsing its
    // tokens. The closure captures the local *Opts struct by reference, runs
    // the corresponding handler, and stashes its return code in `rc`. `main`
    // returns that value below.
    int rc = EXIT_SUCCESS;

    FitOpts                  fit;
    std::vector<std::string> fit_kvs;
    register_fit(app, fit, fit_kvs, rc);

    PredictOpts              predict;
    std::vector<std::string> predict_kvs;
    register_predict(app, predict, predict_kvs, rc);

    EvalOpts                 eval;
    std::vector<std::string> eval_kvs;
    register_eval(app, eval, eval_kvs, rc);

    BenchOpts                bench;
    std::vector<std::string> bench_kvs;
    register_bench(app, bench, bench_kvs, rc);

    register_info(app, rc);
    register_params(app, rc);

    try
    {
        CLI11_PARSE(app, argc, argv);
    }
    catch (std::exception const &e)
    {
        std::println("bonsai: {}", e.what());
        return EXIT_FAILURE;
    }

    return rc;
}
