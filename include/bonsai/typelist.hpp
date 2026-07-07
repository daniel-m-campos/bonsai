#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

namespace bonsai
{

template <typename... Ts> struct TypeList
{
    static constexpr size_t size = sizeof...(Ts);
};

template <typename L> struct size;

template <typename... Ts>
struct size<TypeList<Ts...>> : std::integral_constant<size_t, sizeof...(Ts)>
{
};

template <typename L> inline constexpr size_t size_v = size<L>::value;

template <size_t I, typename L> struct type_at;

template <size_t I, typename Head, typename... Tail>
struct type_at<I, TypeList<Head, Tail...>> : type_at<I - 1, TypeList<Tail...>>
{
};

template <typename Head, typename... Tail> struct type_at<0, TypeList<Head, Tail...>>
{
    using type = Head;
};

template <size_t I, typename L> using type_at_t = typename type_at<I, L>::type;

template <typename... Ls> struct concat;

template <> struct concat<>
{
    using type = TypeList<>;
};

template <typename... Ts> struct concat<TypeList<Ts...>>
{
    using type = TypeList<Ts...>;
};

template <typename... As, typename... Bs, typename... Rest>
struct concat<TypeList<As...>, TypeList<Bs...>, Rest...>
    : concat<TypeList<As..., Bs...>, Rest...>
{
};

template <typename... Ls> using concat_t = typename concat<Ls...>::type;

// Cartesian product of N typelists -> TypeList of TypeList<T0, T1, ..., T(N-1)>.
// Each inner TypeList is one combination, one type per input typelist in input order.
template <typename... Ls> struct cartesian_product_impl;

template <typename First, typename... Rest>
struct cartesian_product_impl<First, Rest...>
{
  private:
    template <typename Combo, typename T> struct prepend;
    template <typename... Cs, typename T> struct prepend<TypeList<Cs...>, T>
    {
        using type = TypeList<T, Cs...>;
    };

    template <typename T, typename Combos> struct prepend_each;
    template <typename T, typename... Combos>
    struct prepend_each<T, TypeList<Combos...>>
    {
        using type = TypeList<typename prepend<Combos, T>::type...>;
    };

    template <typename... Ts> struct expand_first;
    template <> struct expand_first<>
    {
        using type = TypeList<>;
    };
    template <typename T, typename... Ts> struct expand_first<T, Ts...>
    {
        using rest_combos = typename cartesian_product_impl<Rest...>::type;
        using head_block  = typename prepend_each<T, rest_combos>::type;
        using tail_block  = typename expand_first<Ts...>::type;
        using type        = concat_t<head_block, tail_block>;
    };

    template <typename L> struct apply;
    template <typename... Ts> struct apply<TypeList<Ts...>>
    {
        using type = typename expand_first<Ts...>::type;
    };

  public:
    using type = typename apply<First>::type;
};

template <> struct cartesian_product_impl<>
{
    using type = TypeList<TypeList<>>;
};

template <typename... Ls>
using cartesian_product_t = typename cartesian_product_impl<Ls...>::type;

// for_each_type<L>(callable): compile-time iteration over a typelist.
// callable is invoked once per type T with a template lambda receiving T via
// a type-tag argument (callable.template operator()<T>()).
namespace detail
{

template <typename F, typename... Ts>
constexpr void for_each_type_impl(F &&f, TypeList<Ts...> /*unused*/)
{
    // Invoked once per type on the lvalue; forwarding in a fold would move
    // f more than once.
    (f.template operator()<Ts>(), ...);
}

} // namespace detail

template <typename L, typename F> constexpr void for_each_type(F &&f)
{
    detail::for_each_type_impl(std::forward<F>(f), L{});
}

} // namespace bonsai
