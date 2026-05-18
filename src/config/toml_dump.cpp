#include "bonsai/config/toml.hpp"

#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include <toml++/toml.hpp>
#include <toml++/impl/array.hpp>
#include <toml++/impl/table.hpp>

#include "bonsai/config/config.hpp"
#include "bonsai/config/internal/codec.hpp"
#include "bonsai/config/sections/all.hpp"

namespace bonsai::config
{

namespace
{

// Insert one field's value into a per-section toml::table. For std::optional
// fields the key is skipped when nullopt (TOML can't represent absent), so
// round-trip through parse_toml restores the constructor default.
template <typename T>
void insert_field(toml::table &tbl, std::string_view leaf, T const &value)
{
    if constexpr (requires { value.has_value(); })
    {
        if (!value.has_value())
        {
            return;
        }
        tbl.insert_or_assign(std::string{leaf},
                             internal::FieldCodec<T>::to_toml(value));
    }
    else
    {
        tbl.insert_or_assign(std::string{leaf},
                             internal::FieldCodec<T>::to_toml(value));
    }
}

} // namespace

std::string dump_toml(Config const &cfg)
{
    toml::table root;
    std::apply(
        [&](auto const &...section)
        {
            auto emit = [&](auto const &s)
            {
                toml::table tbl;
                auto const &sub = cfg.*(s.sub);
                std::apply(
                    [&](auto const &...f)
                    { (insert_field(tbl, f.leaf, sub.*(f.member)), ...); },
                    s.fields);
                root.insert_or_assign(std::string{s.name}, std::move(tbl));
            };
            (emit(section), ...);
        },
        internal::all_sections);

    std::ostringstream oss;
    oss << root;
    return oss.str();
}

} // namespace bonsai::config
