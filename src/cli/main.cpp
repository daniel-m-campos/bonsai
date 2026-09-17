
#include <cstdio>
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

template <class Opts> struct Command
{
    Opts                     opts;
    std::vector<std::string> set_kvs;
};

template <class Opts>
CLI::App *add_cmd(CLI::App &app, char const *name, char const *desc, Command<Opts> &cmd,
                  int (&run)(Opts const &), int &rc)
{
    auto *sub = app.add_subcommand(name, desc);
    add_common(sub, cmd.opts.common, cmd.set_kvs);
    sub->footer(params_footer);
    sub->callback(
        [&]
        {
            collect_overrides(cmd.set_kvs, cmd.opts.common);
            rc = run(cmd.opts);
        });
    return sub;
}

} // namespace

int main(int argc, char *argv[])
try
{
    CLI::App app{"bonsai: a histogram gradient-boosted tree CLI"};
    app.require_subcommand(1);

    using namespace bonsai::cli; // NOLINT(google-build-using-namespace)

    int rc = EXIT_SUCCESS;

    Command<FitOpts> fit;
    auto            *fit_cmd =
        add_cmd(app, "fit", "Train a model from a CSV dataset", fit, run_fit, rc);
    fit_cmd->add_option("--model", fit.opts.model_path,
                        "Output model file (MessagePack)");
    fit_cmd->add_option("--init-model", fit.opts.init_model_path,
                        "Continue training from this saved model (warm start)");

    Command<PredictOpts> predict;
    auto                *predict_cmd =
        add_cmd(app, "predict", "Predict on a CSV dataset", predict, run_predict, rc);
    predict_cmd->add_option("--model", predict.opts.model_path, "Input model file")
        ->required();
    predict_cmd->add_option("--data", predict.opts.data_path,
                            "Input CSV file (overrides [data].test)");
    predict_cmd->add_option("--out", predict.opts.out_path,
                            "Output CSV (default: stdout)");
    predict_cmd->add_flag("!--raw-scores", predict.opts.apply_link,
                          "Skip the link inverse for classification objectives");
    predict_cmd->add_option("--num-iteration", predict.opts.num_iteration,
                            "Predict with only the first K trees (0 = all)");

    Command<EvalOpts> eval;
    auto             *eval_cmd =
        add_cmd(app, "eval", "Evaluate a model on a CSV dataset", eval, run_eval, rc);
    eval_cmd->add_option("--model", eval.opts.model_path, "Input model file")
        ->required();
    eval_cmd->add_option("--data", eval.opts.data_path,
                         "Input CSV file (overrides [data].test)");

    Command<BenchOpts> bench;
    add_cmd(app, "bench", "Time fit+predict for a config and dataset", bench, run_bench,
            rc)
        ->add_option("--model", bench.opts.model_path, "Optional output model file");

    ImportanceOpts importance;
    app.add_subcommand("importance",
                       "Print per-feature gain and split-count importance")
        ->callback([&] { rc = run_importance(importance); })
        ->add_option("--model", importance.model_path, "Trained model (.msgpack)")
        ->required();

    DumpOpts dump;
    app.add_subcommand("dump", "Print every tree as indented text")
        ->callback([&] { rc = run_dump(dump); })
        ->add_option("--model", dump.model_path, "Trained model (.msgpack)")
        ->required();

    app.add_subcommand("info", "Print available (objective, grower, sampler) combos")
        ->callback([&] { rc = run_info(); });
    app.add_subcommand("params",
                       "Print the default config as TOML (lists all --set keys)")
        ->callback([&] { rc = run_params(); });

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
catch (std::exception const &e)
{
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,modernize-use-std-print)
    std::fprintf(stderr, "bonsai: %s\n", e.what());
    return EXIT_FAILURE;
}
catch (...)
{
    std::fputs("bonsai: unknown error\n", stderr);
    return EXIT_FAILURE;
}
