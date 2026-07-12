#include "bonsai/registry/objective_dispatch.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>

#include "bonsai/objective_traits.hpp"
#include "bonsai/registry/make_booster.hpp"
#include "bonsai/registry/names.hpp"
#include "bonsai/registry/typelists.hpp"
#include "bonsai/task.hpp"
#include "bonsai/typelist.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

namespace
{

using LinkFn     = void (*)(floats_out);
using EvalFn     = float (*)(Config const &, floats_view, floats_view);
using DefaultsFn = std::span<std::string_view const> (*)();

struct LinkEntry
{
    std::string_view name;
    LinkFn           apply;
};

struct EvalEntry
{
    std::string_view name;
    EvalFn           eval;
};

struct TaskEntry
{
    std::string_view name;
    TaskKind         task;
};

struct DefaultsEntry
{
    std::string_view name;
    DefaultsFn       defaults;
};

template <typename O> void link_thunk(floats_out scores)
{
    link_inverse_of<O>::apply(scores);
}

template <typename O>
float eval_thunk(Config const &cfg, floats_view scores, floats_view labels)
{
    return O{cfg}.eval(scores, labels);
}

template <typename O> std::span<std::string_view const> defaults_thunk()
{
    return default_metrics_of<O>::value();
}

inline constexpr auto link_table = make_table<Objectives, LinkEntry>(
    []<typename O>()
    {
        static_assert(HasLinkInverse<O>,
                      "Objective needs link_inverse_of specialization");
        return LinkEntry{impl_name<O>::value, &link_thunk<O>};
    });
inline constexpr auto eval_table = make_table<Objectives, EvalEntry>(
    []<typename O>() { return EvalEntry{impl_name<O>::value, &eval_thunk<O>}; });
inline constexpr auto task_table = make_table<Objectives, TaskEntry>(
    []<typename O>()
    {
        static_assert(HasTaskKind<O>, "Objective needs task_of specialization");
        return TaskEntry{impl_name<O>::value, task_of<O>::value};
    });
inline constexpr auto defaults_table = make_table<Objectives, DefaultsEntry>(
    []<typename O>()
    {
        static_assert(HasDefaultMetricNames<O>,
                      "Objective needs default_metrics_of specialization");
        return DefaultsEntry{impl_name<O>::value, &defaults_thunk<O>};
    });

} // namespace

void apply_link_inverse_by_name(std::string_view objective_name, floats_out scores)
{
    for (auto const &e : link_table)
    {
        if (e.name == objective_name)
        {
            e.apply(scores);
            return;
        }
    }
    throw UnknownImplError("apply_link_inverse_by_name: no objective '" +
                           std::string{objective_name} + "'");
}

float eval_objective_by_name(std::string_view objective_name, Config const &cfg,
                             floats_view scores, floats_view labels)
{
    for (auto const &e : eval_table)
    {
        if (e.name == objective_name)
        {
            return e.eval(cfg, scores, labels);
        }
    }
    throw UnknownImplError("eval_objective_by_name: no objective '" +
                           std::string{objective_name} + "'");
}

TaskKind task_kind_by_name(std::string_view objective_name)
{
    for (auto const &e : task_table)
    {
        if (e.name == objective_name)
        {
            return e.task;
        }
    }
    throw UnknownImplError("task_kind_by_name: no objective '" +
                           std::string{objective_name} + "'");
}

std::span<std::string_view const>
default_metric_names_by_name(std::string_view objective_name)
{
    for (auto const &e : defaults_table)
    {
        if (e.name == objective_name)
        {
            return e.defaults();
        }
    }
    throw UnknownImplError("default_metric_names_by_name: no objective '" +
                           std::string{objective_name} + "'");
}

} // namespace bonsai
