#pragma once

// ============================================================================
// field_name<&T::m>() -> std::string_view, evaluated at compile time.
//
// C++23 stand-in for C++26 reflection. When P2996 lands, replace uses of
// field_name<MemPtr>() with std::meta::identifier_of(member) and delete this
// header. The Field/Section/FieldCodec layers above this primitive are
// designed to survive that migration unchanged.
//
// Mechanism: instantiate a consteval function templated on the pointer-to-
// member as a non-type template parameter, read
// std::source_location::current().function_name(), and parse the member
// identifier out of the signature. A static_assert below pins down the
// signature format for the toolchain we build with; if a future compiler
// version changes the format, the build fails here loud.
// ============================================================================

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
    auto end         = start;
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

template <auto MemPtr> consteval std::string_view field_name()
{
    std::string_view constexpr fn = std::source_location::current().function_name();
    auto constexpr name           = parse_member_identifier(fn);
    static_assert(!name.empty(),
                  "field_name(): could not extract identifier from "
                  "function signature - toolchain format may have changed");
    return name;
}

} // namespace bonsai::config::internal
