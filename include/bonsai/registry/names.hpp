#pragma once

#include <concepts>
#include <string_view>

#include "bonsai/grower.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/split.hpp"

namespace bonsai
{

// External name trait. Specialize per impl. Each specialization defines
// `static constexpr std::string_view value = "...";`
// Keeping the trait external (vs. a static member on each impl) avoids
// coupling impls to a dispatch convention they don't otherwise need.
template <typename T> struct impl_name;

// HasName concept: a type has a name if impl_name<T>::value is a
// constant-evaluable string_view. Used by typelist construction to
// statically reject misnamed/unnamed impls.
template <typename T>
concept HasName = requires {
    { impl_name<T>::value } -> std::convertible_to<std::string_view>;
};

template <> struct impl_name<MSEObjective>
{
    static std::string_view constexpr value = "mse";
};

template <> struct impl_name<LogLossObjective>
{
    static std::string_view constexpr value = "logloss";
};

template <> struct impl_name<DepthwiseGrower<HistogramNodeSplitFinder>>
{
    static std::string_view constexpr value = "depthwise";
};

template <> struct impl_name<AllRowsSampler>
{
    static std::string_view constexpr value = "all_rows";
};

} // namespace bonsai
