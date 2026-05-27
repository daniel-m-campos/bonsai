#pragma once

#include <type_traits>

#include "bonsai/grower.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/registry/names.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/split.hpp"
#include "bonsai/typelist.hpp"

namespace bonsai
{

using Objectives = TypeList<MSEObjective, LogLossObjective>;
using Growers    = TypeList<DepthwiseGrower<HistogramNodeSplitFinder>,
                            ObliviousGrower<HistogramLevelSplitFinder>>;
using Samplers   = TypeList<AllRowsSampler, BernoulliSampler>;

namespace detail
{
template <typename L> struct all_named;
template <typename... Ts>
struct all_named<TypeList<Ts...>> : std::bool_constant<(HasName<Ts> && ...)>
{
};
template <typename L> inline constexpr bool all_named_v = all_named<L>::value;
} // namespace detail

// Single point of truth for "every dispatchable impl has an impl_name<T>
// specialization." Fires at the typelist edit site instead of at every
// for_each_type consumer.
static_assert(detail::all_named_v<Objectives>,
              "every type in Objectives needs an impl_name<T> specialization");
static_assert(detail::all_named_v<Growers>,
              "every type in Growers needs an impl_name<T> specialization");
static_assert(detail::all_named_v<Samplers>,
              "every type in Samplers needs an impl_name<T> specialization");

} // namespace bonsai
