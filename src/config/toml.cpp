#include "bonsai/config/toml.hpp"

#include <string>
#include <string_view>
#include <vector>

#include <toml++/impl/parse_error.hpp>
#include <toml++/impl/parser.hpp>
#include <toml++/impl/table.hpp>
#include <toml++/toml.hpp>

#include "bonsai/config/config.hpp"
#include "bonsai/config/errors.hpp"
#include "bonsai/config/internal/dispatch.hpp"
#include "bonsai/config/sections/all.hpp"

namespace bonsai::config
{

namespace
{

Config from_root(toml::table const &root)
{
    Config cfg;
    internal::load_from_table(root, cfg, internal::all_sections);
    return cfg;
}

} // namespace

Config load_toml(std::string const &path)
{
    try
    {
        auto root = toml::parse_file(path);
        return from_root(root);
    }
    catch (toml::parse_error const &e)
    {
        throw ConfigError(std::string{"config: TOML parse error: "} + e.what());
    }
}

Config parse_toml(std::string_view text)
{
    try
    {
        auto root = toml::parse(text);
        return from_root(root);
    }
    catch (toml::parse_error const &e)
    {
        throw ConfigError(std::string{"config: TOML parse error: "} + e.what());
    }
}

void apply_overrides(Config &cfg, std::vector<Override> const &overrides)
{
    for (auto const &ov : overrides)
    {
        internal::apply_override(cfg, ov.key, ov.value, internal::all_sections);
    }
}

Config resolve(std::string const &toml_path, std::vector<Override> const &overrides)
{
    Config cfg = toml_path.empty() ? Config{} : load_toml(toml_path);
    apply_overrides(cfg, overrides);
    return cfg;
}

} // namespace bonsai::config
