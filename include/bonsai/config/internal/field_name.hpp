#pragma once

// field_name<&T::m>() names a member at compile time by parsing the
// identifier out of a consteval function's signature. The static_assert
// below pins the signature format for the toolchain we build with; a
// compiler that changes it fails the build here, loudly.

#if !defined(__GNUC__) && !defined(__clang__)
#error "bonsai::config::field_name() relies on GCC/Clang __PRETTY_FUNCTION__ format"
#endif

#include <source_location>
#include <string_view>

namespace bonsai::config::internal
{

// Parse out the identifier following the last "::" inside the signature,
// stopping at the first character that can't belong to an identifier.
// Handles both:
//   GCC:   "... [with auto MemPtr = &TreeConfig::max_depth]"
//   Clang: "... [MemPtr = &TreeConfig::max_depth]"
consteval std::string_view parse_member_identifier(std::string_view fn)
{
    auto const last = fn.rfind("::");
    if (last == std::string_view::npos)
    {
        return {};
    }
    auto const start = last + 2;
    auto       end   = start;
    while (end < fn.size())
    {
        auto const c        = fn[end];
        bool const is_ident = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                              (c >= '0' && c <= '9') || c == '_';
        if (!is_ident)
        {
            break;
        }
        ++end;
    }
    return fn.substr(start, end - start);
}

template <auto MemPtr> consteval std::string_view raw_member_name()
{
    return std::source_location::current().function_name();
}

template <auto MemPtr> consteval std::string_view field_name()
{
    constexpr auto name = parse_member_identifier(raw_member_name<MemPtr>());
    static_assert(!name.empty(),
                  "field_name(): could not extract identifier from "
                  "function signature - toolchain format may have changed");
    return name;
}

} // namespace bonsai::config::internal
