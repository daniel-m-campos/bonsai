#pragma once

#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace bonsai::config::internal
{

struct Bounds
{
    double lower;
    double upper;
    bool   lower_open;
    bool   upper_open;

    constexpr bool in_bounds(double v) const
    {
        bool const above = lower_open ? v > lower : v >= lower;
        bool const below = upper_open ? v < upper : v <= upper;
        return above && below;
    }
};

inline constexpr double k_unbounded = std::numeric_limits<double>::infinity();

inline constexpr Bounds k_finite{-k_unbounded, k_unbounded, true, true};
inline constexpr Bounds k_positive{0.0, k_unbounded, true, true};
inline constexpr Bounds k_non_negative{0.0, k_unbounded, false, true};
inline constexpr Bounds k_open_unit{0.0, 1.0, true, true};
inline constexpr Bounds k_fraction{0.0, 1.0, true, false};
inline constexpr Bounds k_probability{0.0, 1.0, false, false};
inline constexpr Bounds k_direction{-1.0, 1.0, false, false};

inline std::string to_string(Bounds const &b)
{
    bool const has_lower = std::isfinite(b.lower);
    bool const has_upper = std::isfinite(b.upper);
    if (has_lower && has_upper)
    {
        return std::format("in {}{}, {}{}", b.lower_open ? '(' : '[', b.lower, b.upper,
                           b.upper_open ? ')' : ']');
    }
    if (has_lower)
    {
        return std::format("finite and {} {}", b.lower_open ? ">" : ">=", b.lower);
    }
    if (has_upper)
    {
        return std::format("finite and {} {}", b.upper_open ? "<" : "<=", b.upper);
    }
    return "finite";
}

template <typename T>
concept Bounded = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template <Bounded T>
std::optional<std::string> out_of_bounds(Bounds const &b, T const &value)
{
    if (b.in_bounds(static_cast<double>(value)))
    {
        return std::nullopt;
    }
    return std::format("must be {} (got {})", to_string(b), value);
}

template <Bounded T>
std::optional<std::string> out_of_bounds(Bounds const &b, std::vector<T> const &items)
{
    for (size_t i = 0; i < items.size(); ++i)
    {
        if (auto why = out_of_bounds(b, items[i]))
        {
            return std::format("item {}: {}", i, *why);
        }
    }
    return std::nullopt;
}

} // namespace bonsai::config::internal
