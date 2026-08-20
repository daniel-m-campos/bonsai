#include "bonsai/config/toml.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>
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

namespace
{

// Widen a parsed field value into the codec type universe.
template <typename T> OverrideValue widen(T value)
{
    if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, std::string> ||
                  std::is_same_v<T, std::vector<std::string>> ||
                  std::is_same_v<T, std::vector<int>>)
    {
        return value;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return static_cast<double>(value);
    }
    else
    {
        static_assert(std::is_integral_v<T>, "unmapped config field type");
        return static_cast<int64_t>(value);
    }
}

} // namespace

std::vector<std::pair<std::string, OverrideValue>>
typed_overrides(std::string_view text)
{
    toml::table root;
    try
    {
        root = toml::parse(text);
    }
    catch (toml::parse_error const &e)
    {
        throw ConfigError(std::string{"config: TOML parse error: "} + e.what());
    }
    std::vector<std::pair<std::string, OverrideValue>> out;
    std::unordered_set<std::string>                    known;
    std::apply(
        [&](auto const &...secs)
        {
            (
                [&]
                {
                    known.insert(std::string{secs.name});
                    if (auto const *node = root.get(secs.name))
                    {
                        auto const *table = node->as_table();
                        if (table == nullptr)
                        {
                            throw ConfigError(std::string{"config: ["} +
                                              std::string{secs.name} +
                                              "] must be a table");
                        }
                        internal::visit_section(*table, secs,
                                                [&](auto const &field, auto value)
                                                {
                                                    out.emplace_back(
                                                        std::string{secs.name} + "." +
                                                            std::string{field.leaf},
                                                        widen(std::move(value)));
                                                });
                    }
                }(),
                ...);
        },
        internal::all_sections);
    for (auto const &[k, unused] : root)
    {
        if (!known.contains(std::string{k.str()}))
        {
            throw ConfigError(std::string{"config: unknown section ["} +
                              std::string{k.str()} + "]");
        }
    }
    return out;
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
