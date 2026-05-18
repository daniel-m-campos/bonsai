#pragma once

#include <format>
#include <string>
#include <string_view>
#include <toml++/impl/table.hpp>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include <toml++/toml.hpp>

#include "bonsai/config/config.hpp"
#include "bonsai/config/errors.hpp"
#include "bonsai/config/internal/codec.hpp"

namespace bonsai::config::internal
{

[[noreturn]] inline void key_error(std::string_view section, std::string_view key,
                                   std::string_view reason)
{
    throw ConfigError(std::format("config: [{}].{}: {}", section, key, reason));
}

inline std::pair<std::string_view, std::string_view> split_key(std::string_view key)
{
    auto const dot = key.find('.');
    if (dot == std::string_view::npos)
    {
        throw ConfigError("override: key must be dotted (e.g. 'tree.max_depth'): '" +
                          std::string{key} + "'");
    }
    return {key.substr(0, dot), key.substr(dot + 1)};
}

template <typename Section>
void load_section(toml::table const &table, Config &cfg, Section const &sec)
{
    std::unordered_set<std::string> seen;
    std::apply(
        [&](auto const &...fields)
        {
            (
                [&]
                {
                    using T =
                        typename std::remove_cvref_t<decltype(fields)>::member_type;
                    seen.insert(std::string{fields.leaf});
                    if (auto const *node = table.get(fields.leaf))
                    {
                        auto r = FieldCodec<T>::from_toml(*node);
                        if (!r)
                        {
                            key_error(sec.name, fields.leaf, r.error());
                        }
                        (cfg.*(sec.sub)).*(fields.member) = std::move(*r);
                    }
                }(),
                ...);
        },
        sec.fields);

    for (auto const &[k, _] : table)
    {
        if (!seen.contains(std::string{k.str()}))
        {
            key_error(sec.name, k.str(), "unknown key");
        }
    }
}

template <typename Section>
bool apply_leaf(Config &cfg, Section const &section, std::string_view leaf,
                std::string_view value)
{
    bool handled = false;
    std::apply(
        [&](auto const &...fs)
        {
            (
                [&]
                {
                    using T = typename std::remove_cvref_t<decltype(fs)>::member_type;
                    if (handled || fs.leaf != leaf)
                    {
                        return;
                    }
                    auto r = FieldCodec<T>::from_string(value);
                    if (!r)
                    {
                        key_error(section.name, leaf, r.error());
                    }
                    (cfg.*(section.sub)).*(fs.member) = std::move(*r);
                    handled                           = true;
                }(),
                ...);
        },
        section.fields);
    return handled;
}

template <typename Sections>
void load_from_table(toml::table const &root, Config &cfg, Sections const &sections)
{
    // 1. Reject unknown top-level sections.
    std::apply(
        [&](auto const &...sections)
        {
            for (auto const &[k, _] : root)
            {
                auto const name = k.str();
                bool known      = false;
                ((sections.name == name ? (known = true, void()) : void()), ...);
                if (!known)
                {
                    throw ConfigError("config: unknown section [" + std::string{name} +
                                      "]");
                }
            }
        },
        sections);

    // 2. Dispatch each present section to load_section.
    std::apply(
        [&](auto const &...sections)
        {
            (
                [&]
                {
                    if (auto const *t = root.get_as<toml::table>(sections.name))
                    {
                        load_section(*t, cfg, sections);
                    }
                }(),
                ...);
        },
        sections);
}

template <typename Sections>
void apply_override(Config &cfg, std::string_view key, std::string_view value,
                    Sections const &sections)
{
    auto const [section_name, leaf] = split_key(key);
    bool dispatched                 = false;
    std::apply(
        [&](auto const &...secs)
        {
            (
                [&]
                {
                    if (dispatched || secs.name != section_name)
                    {
                        return;
                    }
                    dispatched = true;
                    if (!apply_leaf(cfg, secs, leaf, value))
                    {
                        key_error(section_name, leaf, "unknown key");
                    }
                }(),
                ...);
        },
        sections);
    if (!dispatched)
    {
        throw ConfigError("override: unknown section '" + std::string{section_name} +
                          "'");
    }
}

} // namespace bonsai::config::internal
