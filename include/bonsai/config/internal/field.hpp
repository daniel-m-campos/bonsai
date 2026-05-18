#pragma once

#include <string_view>
#include <tuple>
#include <type_traits>

#include "bonsai/config/config.hpp"
#include "bonsai/config/internal/field_name.hpp"

namespace bonsai::config::internal
{

template <typename T> struct member_pointer_traits;

template <typename C, typename M> struct member_pointer_traits<M C::*>
{
    using class_type  = C;
    using member_type = M;
};

template <typename Sub, typename T> struct Field
{
    std::string_view leaf;
    T Sub::*member;
    using class_type  = Sub;
    using member_type = T;
};

template <auto MemPtr> consteval auto field()
{
    using Traits = member_pointer_traits<std::remove_cv_t<decltype(MemPtr)>>;
    using Sub    = typename Traits::class_type;
    using T      = typename Traits::member_type;
    return Field<Sub, T>{field_name<MemPtr>(), MemPtr};
}

template <typename Sub, typename Fields> struct Section
{
    std::string_view name;
    Sub Config::*sub;
    Fields fields;
    using sub_type = Sub;
};

template <typename Sub, typename... Fs>
consteval auto make_section(std::string_view name, Sub Config::*sub, Fs... fs)
{
    return Section<Sub, std::tuple<Fs...>>{name, sub, std::tuple{fs...}};
}

} // namespace bonsai::config::internal
