#pragma once

#include <string>
#include <string_view>
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

// Whether the TOML file contains the named top-level section, regardless of
// the values it sets. Value comparison cannot distinguish an absent section
// from one that explicitly restates the defaults; callers that fix a section
// elsewhere (a prebuilt Dataset) need the structural answer.
bool toml_has_section(std::string const &path, std::string_view section);

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
