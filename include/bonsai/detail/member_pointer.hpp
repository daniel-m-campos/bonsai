#pragma once

namespace bonsai::detail
{

template <typename T> struct member_pointer_traits;

template <typename C, typename M> struct member_pointer_traits<M C::*>
{
    using class_type  = C;
    using member_type = M;
};

} // namespace bonsai::detail
