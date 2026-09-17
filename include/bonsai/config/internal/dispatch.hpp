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

template <typename Tuple, typename F> void tuple_for_each(Tuple const &tuple, F &&f)
{
    std::apply([&](auto const &...item) { (f(item), ...); }, tuple);
}

template <typename Tuple, typename F> bool tuple_any_of(Tuple const &tuple, F &&f)
{
    return std::apply([&](auto const &...item) { return (f(item) || ...); }, tuple);
}

// One strict walk of a section's fields against its TOML table: every
// present key parses through its codec, unknown keys error. The sink decides
// what a parsed value becomes (assignment into a Config, a recorded
// override), so strictness is stated once for every consumer.
template <typename Section, typename Sink>
void visit_section(toml::table const &table, Section const &sec, Sink &&sink)
{
    std::unordered_set<std::string> seen;
    tuple_for_each(sec.fields,
                   [&](auto const &field)
                   {
                       using T =
                           typename std::remove_cvref_t<decltype(field)>::member_type;
                       seen.insert(std::string{field.leaf});
                       if (auto const *node = table.get(field.leaf))
                       {
                           auto r = FieldCodec<T>::from_toml(*node);
                           if (!r)
                           {
                               key_error(sec.name, field.leaf, r.error());
                           }
                           sink(field, std::move(*r));
                       }
                   });

    for (auto const &[k, _] : table)
    {
        if (!seen.contains(std::string{k.str()}))
        {
            key_error(sec.name, k.str(), "unknown key");
        }
    }
}

template <typename Section>
void load_section(toml::table const &table, Config &cfg, Section const &sec)
{
    visit_section(table, sec, [&](auto const &field, auto value)
                  { (cfg.*(sec.sub)).*(field.member) = std::move(value); });
}

template <typename Section>
bool apply_leaf(Config &cfg, Section const &section, std::string_view leaf,
                std::string_view value)
{
    return tuple_any_of(
        section.fields,
        [&](auto const &field)
        {
            using T = typename std::remove_cvref_t<decltype(field)>::member_type;
            if (field.leaf != leaf)
            {
                return false;
            }
            auto r = FieldCodec<T>::from_string(value);
            if (!r)
            {
                key_error(section.name, leaf, r.error());
            }
            (cfg.*(section.sub)).*(field.member) = std::move(*r);
            return true;
        });
}

template <typename Sections>
void require_known_sections(toml::table const &root, Sections const &sections)
{
    for (auto const &[k, _] : root)
    {
        auto const name = k.str();
        if (!tuple_any_of(sections,
                          [&](auto const &section) { return section.name == name; }))
        {
            throw ConfigError("config: unknown section [" + std::string{name} + "]");
        }
    }
}

template <typename Sections>
void load_stated_sections(toml::table const &root, Config &cfg,
                          Sections const &sections)
{
    tuple_for_each(sections,
                   [&](auto const &section)
                   {
                       if (auto const *t = root.get_as<toml::table>(section.name))
                       {
                           load_section(*t, cfg, section);
                       }
                   });
}

template <typename Sections>
void load_from_table(toml::table const &root, Config &cfg, Sections const &sections)
{
    require_known_sections(root, sections);
    load_stated_sections(root, cfg, sections);
}

template <typename Sections>
void apply_override(Config &cfg, std::string_view key, std::string_view value,
                    Sections const &sections)
{
    auto const [section_name, leaf] = split_key(key);
    bool const dispatched =
        tuple_any_of(sections,
                     [&](auto const &section)
                     {
                         if (section.name != section_name)
                         {
                             return false;
                         }
                         if (!apply_leaf(cfg, section, leaf, value))
                         {
                             key_error(section_name, leaf, "unknown key");
                         }
                         return true;
                     });
    if (!dispatched)
    {
        throw ConfigError("override: unknown section '" + std::string{section_name} +
                          "'");
    }
}

} // namespace bonsai::config::internal
