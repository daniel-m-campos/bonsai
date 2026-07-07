#pragma once

#include <concepts>
#include <cstdint>
#include <string_view>

namespace bonsai
{

// The kind of supervised-learning task an objective or metric applies to.
// Used to type-check metric-vs-objective compatibility (e.g. reject
// `accuracy` on a regression model).
enum class TaskKind : uint8_t
{
    regression,
    binary_classification,
    multiclass_classification,
};

// Human-readable name for error messages and logs.
std::string_view task_kind_name(TaskKind kind);

// External trait, specialized per impl in BOTH the Objectives and Metrics
// typelists. An objective and a metric belong together when their task_of<T>
// agree. Specialize alongside the impl's other traits (impl_name<T>, etc.).
template <typename T> struct task_of;

template <typename T>
concept HasTaskKind = requires {
    { task_of<T>::value } -> std::convertible_to<TaskKind>;
};

} // namespace bonsai
