#pragma once

#include <concepts>
#include <string_view>

#include "bonsai/cuda/grower.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/split.hpp"

namespace bonsai
{

// External name trait: impl_name<T>::value is the string the registry,
// the config and the model file know T by. Keeping the trait external
// (vs. a static member on each impl) avoids coupling impls to a dispatch
// convention they don't otherwise need. A name is registered by one
// name_of<T> row below; impl_name reads it.
template <typename T> struct impl_name;

// HasName concept: a type has a name if impl_name<T>::value is a
// constant-evaluable string_view. Used by typelist construction to
// statically reject misnamed/unnamed impls.
template <typename T>
concept HasName = requires {
    { impl_name<T>::value } -> std::convertible_to<std::string_view>;
};

template <typename T> inline constexpr std::string_view name_of = {};

template <> inline constexpr std::string_view name_of<MSEObjective>      = "mse";
template <> inline constexpr std::string_view name_of<LogLossObjective>  = "logloss";
template <> inline constexpr std::string_view name_of<MAEObjective>      = "mae";
template <> inline constexpr std::string_view name_of<HuberObjective>    = "huber";
template <> inline constexpr std::string_view name_of<QuantileObjective> = "quantile";
template <> inline constexpr std::string_view name_of<PoissonObjective>  = "poisson";
template <> inline constexpr std::string_view name_of<SoftmaxObjective>  = "softmax";

template <>
inline constexpr std::string_view name_of<DepthwiseGrower<CpuHistogramEngine>> =
    "depthwise";

// The tree it builds is oblivious (symmetric); the growth policy it follows is
// levelwise, and the external name states the policy like its two siblings.
template <>
inline constexpr std::string_view name_of<ObliviousGrower<CpuHistogramEngine>> =
    "levelwise";
template <>
inline constexpr std::string_view name_of<LeafwiseGrower<CpuHistogramEngine>> =
    "leafwise";
template <>
inline constexpr std::string_view name_of<CudaDepthwiseGrower> = "cuda_depthwise";
template <>
inline constexpr std::string_view name_of<CudaObliviousGrower> = "cuda_levelwise";
template <>
inline constexpr std::string_view name_of<CudaLeafwiseGrower> = "cuda_leafwise";

template <> inline constexpr std::string_view name_of<AllRowsSampler>   = "all_rows";
template <> inline constexpr std::string_view name_of<BernoulliSampler> = "bernoulli";
template <> inline constexpr std::string_view name_of<GossSampler>      = "goss";

template <typename T>
    requires(!name_of<T>.empty())
struct impl_name<T>
{
    static constexpr std::string_view value = name_of<T>;
};

} // namespace bonsai
