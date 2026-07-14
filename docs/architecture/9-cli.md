# 9. CLI

> **Status:** Done for the MVP. CLI11-based subcommand layout, all six subcommands wired (`fit`, `predict`, `eval`, `bench`, `info`, `params`).

## Layout

`src/cli/main.cpp` is the thin CLI11 root: it builds six `CLI::App` subcommands and dispatches to the per-subcommand handler in `src/cli/{fit,predict,eval,bench,info,params}.cpp`. Each handler owns one free function (`run_fit`, etc.) over a strongly-typed `*Opts` struct declared in `src/cli/handlers.hpp`.

Shared helpers live in `src/cli/common.{hpp,cpp}`:

- `CommonOpts { config_path, overrides }`: the two flags every subcommand wants.
- `resolve_config(CommonOpts)`: load TOML, apply overrides, return a `Config`.
- `to_feature_buffer(ColumnBatch)`: row-major float matrix for `IBooster::predict`.

## Subcommands

| Command | Purpose | Required | Notes |
|---|---|---|---|
| `fit`     | Train a model from CSV | `-c <config>` | optional `--model <path>` writes a MessagePack file via `bonsai::io::save_booster` |
| `predict` | Predict on a CSV | `--model`     | `--data` overrides `[data].test`; `--out` writes CSV, default stdout |
| `eval`    | Evaluate a model | `--model`     | regression → RMSE; logloss → logloss + accuracy |
| `bench`   | Time fit+predict | `-c <config>` | prints `load_seconds`, `fit_seconds`, `predict_seconds`, `rows_per_sec` |
| `info`    | List dispatch combos | —         | enumerates `available_combos()` from the registry |
| `params`  | Print default config as TOML | —    | lists every `--set` key in a parseable skeleton |

## Overrides

Every command accepts `--set <key>=<value>` (repeatable). Keys are dotted into `Config` sections (`tree.max_depth`, `booster.n_iters`, …). Underscores in field names; sections are the TOML headers. Last write wins across multiple flags and the TOML file.

Every command except `info` and `params` also accepts `--dump-config`. When set, the resolved config (defaults → TOML → `--set`) is printed as TOML and the command exits without running. Useful for debugging override precedence and as a reproducibility artifact.

Unknown keys throw `ConfigError`; unknown TOML sections also throw, and toml++ is configured in strict mode (unknown keys inside a known section are rejected too).

## Predict link inverse

For classification objectives (currently `logloss`) the CLI applies the link inverse (sigmoid) to raw scores before writing predictions, matching decision 22 (booster owns predict-path link). The flag `--raw-scores` on `predict` opts out.

## Exit codes

- `0`: success.
- `1`: runtime error (CSV parse error, missing file, IO failure, `ConfigError` from override or TOML). The error message is on stderr.
- `2`: usage error (missing required option, like `data.train` empty for `fit`).

## What's not here

Fit-time progress is text-based (`std::print` from `on_tick`), gated on `booster.log_intervals` (see [`8-config.md`](8-config.md)). No `indicators` progress bars yet.

- `--early-stopping-rounds` on `fit`: deferred until eval-during-fit is wired.
- Multi-validation (`[data].valid`): only the first path is consumed for per-iter eval metrics; additional entries log a stderr warning. Multiple simultaneous valid sets deferred.

## Cross-references

- [`8-config.md`](8-config.md): TOML schema + override syntax.
- [`6-dispatch.md`](6-dispatch.md): `info` reads from the same registry the CLI uses to instantiate boosters.
