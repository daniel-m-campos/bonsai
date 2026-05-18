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
    static toml::value<bool> to_toml(bool v) { return toml::value{v}; }
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
    static toml::value<int64_t> to_toml(int v)
    {
        return toml::value{static_cast<int64_t>(v)};
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
    static toml::value<int64_t> to_toml(U v)
    {
        return toml::value{static_cast<int64_t>(v)};
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
    static toml::value<double> to_toml(float v)
    {
        return toml::value{static_cast<double>(v)};
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
    static toml::value<std::string> to_toml(std::string const &v)
    {
        return toml::value{v};
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
    static ParseResult<std::vector<std::string>> from_string(std::string_view value)
    {
        // Comma-separated; empty token rejected (e.g. "a,,b" or trailing ",").
        // Lets `--set metrics.fit=rmse,mae` work; bare empty string -> empty vec.
        std::vector<std::string> out;
        if (value.empty())
        {
            return out;
        }
        size_t start = 0;
        while (start <= value.size())
        {
            auto const comma = value.find(',', start);
            auto const piece = value.substr(start, comma - start);
            if (piece.empty())
            {
                return std::unexpected("empty value in comma-separated list");
            }
            out.emplace_back(piece);
            if (comma == std::string_view::npos)
            {
                break;
            }
            start = comma + 1;
        }
        return out;
    }
    static toml::array to_toml(std::vector<std::string> const &v)
    {
        toml::array a;
        for (auto const &s : v)
        {
            a.push_back(s);
        }
        return a;
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
    static toml::array to_toml(std::vector<int> const &v)
    {
        toml::array a;
        for (auto const &x : v)
        {
            a.push_back(static_cast<int64_t>(x));
        }
        return a;
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
    // Caller (dump_toml) skips the key when nullopt; precondition: has_value().
    static toml::value<double> to_toml(std::optional<float> const &v)
    {
        return toml::value{static_cast<double>(*v)};
    }
};

} // namespace bonsai::config::internal
