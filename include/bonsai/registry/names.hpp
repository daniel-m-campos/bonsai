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
    static constexpr std::string_view value = "mse";
};

template <> struct impl_name<LogLossObjective>
{
    static constexpr std::string_view value = "logloss";
};

template <> struct impl_name<MAEObjective>
{
    static constexpr std::string_view value = "mae";
};

template <> struct impl_name<HuberObjective>
{
    static constexpr std::string_view value = "huber";
};

template <> struct impl_name<QuantileObjective>
{
    static constexpr std::string_view value = "quantile";
};

template <> struct impl_name<SoftmaxObjective>
{
    static constexpr std::string_view value = "softmax";
};

template <> struct impl_name<DepthwiseGrower<HistogramNodeSplitFinder>>
{
    static constexpr std::string_view value = "depthwise";
};

template <> struct impl_name<ObliviousGrower<HistogramLevelSplitFinder>>
{
    static constexpr std::string_view value = "oblivious";
};

template <> struct impl_name<LeafwiseGrower<HistogramNodeSplitFinder>>
{
    static constexpr std::string_view value = "leafwise";
};

template <> struct impl_name<AllRowsSampler>
{
    static constexpr std::string_view value = "all_rows";
};

template <> struct impl_name<BernoulliSampler>
{
    static constexpr std::string_view value = "bernoulli";
};

template <> struct impl_name<GossSampler>
{
    static constexpr std::string_view value = "goss";
};

} // namespace bonsai
