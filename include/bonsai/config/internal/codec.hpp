#pragma once

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <toml++/impl/node.hpp>
#include <type_traits>
#include <vector>

#include <toml++/toml.hpp>

namespace bonsai::config::internal
{

template <typename T> using ParseResult = std::expected<T, std::string>;

template <typename T> struct FieldCodec;

// -------------------- helpers -----------------------------------------------

template <typename U> ParseResult<U> read_uint_from_toml(toml::node const &node)
{
    auto opt = node.value<int64_t>();
    if (!opt)
    {
        return std::unexpected("wrong type (expected integer)");
    }
    if (*opt < 0)
    {
        return std::unexpected("must be non-negative");
    }
    return static_cast<U>(*opt);
}

template <typename U> ParseResult<U> read_uint_from_string(std::string_view value)
{
    int64_t v       = 0;
    auto const *beg = value.data();
    auto const *end = beg + value.size();
    auto const res  = std::from_chars(beg, end, v);
    if (res.ec != std::errc{} || res.ptr != end)
    {
        return std::unexpected("cannot parse integer from '" + std::string{value} +
                               "'");
    }
    if (v < 0)
    {
        return std::unexpected("must be non-negative");
    }
    return static_cast<U>(v);
}

inline ParseResult<int> read_int_from_string(std::string_view value)
{
    int v           = 0;
    auto const *beg = value.data();
    auto const *end = beg + value.size();
    auto const res  = std::from_chars(beg, end, v);
    if (res.ec != std::errc{} || res.ptr != end)
    {
        return std::unexpected("cannot parse integer from '" + std::string{value} +
                               "'");
    }
    return v;
}

inline ParseResult<float> read_float_from_string(std::string_view value)
{
    std::string const owned{value};
    char *end       = nullptr;
    float const v   = std::strtof(owned.c_str(), &end);
    auto const used = static_cast<size_t>(end != nullptr ? end - owned.c_str() : 0);
    if (used != owned.size())
    {
        return std::unexpected("cannot parse float from '" + owned + "'");
    }
    return v;
}

// -------------------- bool ---------------------------------------------------

template <> struct FieldCodec<bool>
{
    static ParseResult<bool> from_toml(toml::node const &node)
    {
        auto opt = node.value<bool>();
        if (!opt)
        {
            return std::unexpected("wrong type");
        }
        return *opt;
    }
    static ParseResult<bool> from_string(std::string_view value)
    {
        if (value == "true" || value == "1")
        {
            return true;
        }
        if (value == "false" || value == "0")
        {
            return false;
        }
        return std::unexpected("expected true/false, got '" + std::string{value} + "'");
    }
};

// -------------------- int ----------------------------------------------------

template <> struct FieldCodec<int>
{
    static ParseResult<int> from_toml(toml::node const &node)
    {
        auto opt = node.value<int64_t>();
        if (!opt)
        {
            return std::unexpected("wrong type");
        }
        return static_cast<int>(*opt);
    }
    static ParseResult<int> from_string(std::string_view value)
    {
        return read_int_from_string(value);
    }
};

// -------------------- unsigned ints (uint8_t / uint32_t / uint64_t / size_t) -

template <typename U>
    requires std::is_unsigned_v<U> && (!std::is_same_v<U, bool>)
struct FieldCodec<U>
{
    static ParseResult<U> from_toml(toml::node const &node)
    {
        return read_uint_from_toml<U>(node);
    }
    static ParseResult<U> from_string(std::string_view value)
    {
        return read_uint_from_string<U>(value);
    }
};

// -------------------- float --------------------------------------------------

template <> struct FieldCodec<float>
{
    static ParseResult<float> from_toml(toml::node const &node)
    {
        auto opt = node.value<double>();
        if (!opt)
        {
            return std::unexpected("wrong type (expected float)");
        }
        return static_cast<float>(*opt);
    }
    static ParseResult<float> from_string(std::string_view value)
    {
        return read_float_from_string(value);
    }
};

// -------------------- std::string --------------------------------------------

template <> struct FieldCodec<std::string>
{
    static ParseResult<std::string> from_toml(toml::node const &node)
    {
        auto opt = node.value<std::string>();
        if (!opt)
        {
            return std::unexpected("wrong type");
        }
        return *opt;
    }
    static ParseResult<std::string> from_string(std::string_view value)
    {
        return std::string{value};
    }
};

// -------------------- std::vector<std::string> -------------------------------

template <> struct FieldCodec<std::vector<std::string>>
{
    static ParseResult<std::vector<std::string>> from_toml(toml::node const &node)
    {
        auto const *arr = node.as_array();
        if (arr == nullptr)
        {
            return std::unexpected("must be an array of strings");
        }
        std::vector<std::string> out;
        out.reserve(arr->size());
        for (auto const &item : *arr)
        {
            auto v = item.value<std::string>();
            if (!v)
            {
                return std::unexpected("items must be strings");
            }
            out.push_back(*v);
        }
        return out;
    }
    static ParseResult<std::vector<std::string>> from_string(std::string_view)
    {
        return std::unexpected("cannot set list-valued key via CLI override");
    }
};

// -------------------- std::vector<int> ---------------------------------------

template <> struct FieldCodec<std::vector<int>>
{
    static ParseResult<std::vector<int>> from_toml(toml::node const &node)
    {
        auto const *arr = node.as_array();
        if (arr == nullptr)
        {
            return std::unexpected("must be an array of ints");
        }
        std::vector<int> out;
        out.reserve(arr->size());
        for (auto const &item : *arr)
        {
            auto v = item.value<int64_t>();
            if (!v)
            {
                return std::unexpected("items must be integers");
            }
            out.push_back(static_cast<int>(*v));
        }
        return out;
    }
    static ParseResult<std::vector<int>> from_string(std::string_view)
    {
        return std::unexpected("cannot set list-valued key via CLI override");
    }
};

// -------------------- std::optional<float> -----------------------------------

template <> struct FieldCodec<std::optional<float>>
{
    static ParseResult<std::optional<float>> from_toml(toml::node const &node)
    {
        auto opt = node.value<double>();
        if (!opt)
        {
            return std::unexpected("wrong type (expected float)");
        }
        return std::optional<float>{static_cast<float>(*opt)};
    }
    static ParseResult<std::optional<float>> from_string(std::string_view value)
    {
        auto r = read_float_from_string(value);
        if (!r)
        {
            return std::unexpected(r.error());
        }
        return std::optional<float>{*r};
    }
};

} // namespace bonsai::config::internal
