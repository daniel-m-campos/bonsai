#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "bonsai/config/config.hpp"

namespace bonsai::config
{

// Load a Config from a TOML file. Recognized sections: [data],
// [bin_mapper], [tree], [booster], [dispatch]. Throws ConfigError with a
// key path on unknown keys, type mismatches, or bad values.
Config load_toml(std::string const &path);

// Parse from an in-memory TOML string. Used by tests.
Config parse_toml(std::string_view text);

// Serialize a Config back to TOML. Output is parseable by `parse_toml` —
// `parse_toml(dump_toml(cfg)) == cfg` for any cfg built from the codec.
// `dump_toml(Config{})` is the canonical default-config dump used by
// `bonsai params`.
std::string dump_toml(Config const &cfg);

// One parsed TOML value, widened to the codec type universe.
using OverrideValue = std::variant<bool, int64_t, double, std::string,
                                   std::vector<std::string>, std::vector<int>>;

// The dotted keys a TOML text explicitly sets, with parsed typed values, in
// registry order. This is what a sparse overrides object (Params.from_toml)
// needs — only the stated keys, never the resolved whole. Strict like
// load_toml: unknown sections or keys throw ConfigError.
std::vector<std::pair<std::string, OverrideValue>>
typed_overrides(std::string_view text);

// Apply CLI dotted-key overrides like "tree.max_depth=8" to an existing
// Config (last write wins). Throws ConfigError on unknown key or bad value.
struct Override
{
    std::string key;
    std::string value;
};

void apply_overrides(Config &cfg, std::vector<Override> const &overrides);

// The one precedence rule for every entry point (CLI and Python): optional
// TOML file, then key=value overrides on top. Callers that own the process
// thread pool (the CLI) additionally apply cfg.parallel afterwards.
Config resolve(std::string const &toml_path, std::vector<Override> const &overrides);

} // namespace bonsai::config
