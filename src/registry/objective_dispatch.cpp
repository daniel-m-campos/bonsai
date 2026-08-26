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
using DefaultsFn = std::span<std::string_view const> (*)();

struct LinkEntry
{
    std::string_view name;
    LinkFn           apply;
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

template <typename Table>
auto const &lookup(Table const &table, std::string_view name, char const *what)
{
    for (auto const &e : table)
    {
        if (e.name == name)
        {
            return e;
        }
    }
    throw UnknownImplError(std::string{what} + ": no objective '" + std::string{name} +
                           "'");
}

} // namespace

void apply_link_inverse_by_name(std::string_view objective_name, floats_out scores)
{
    lookup(link_table, objective_name, "apply_link_inverse_by_name").apply(scores);
}

TaskKind task_kind_by_name(std::string_view objective_name)
{
    return lookup(task_table, objective_name, "task_kind_by_name").task;
}

std::span<std::string_view const>
default_metric_names_by_name(std::string_view objective_name)
{
    return lookup(defaults_table, objective_name, "default_metric_names_by_name")
        .defaults();
}

} // namespace bonsai
