#include "bonsai/registry/objective_dispatch.hpp"

#include <algorithm>
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

using LinkFn = void (*)(floats_out);

struct ObjectiveEntry
{
    std::string_view                  name;
    LinkFn                            apply;
    TaskKind                          task;
    std::span<std::string_view const> defaults;
};

template <typename O> void link_thunk(floats_out scores)
{
    link_inverse_of<O>::apply(scores);
}

inline constexpr auto objective_table = make_table<Objectives, ObjectiveEntry>(
    []<typename O>()
    {
        static_assert(HasLinkInverse<O>,
                      "Objective needs link_inverse_of specialization");
        static_assert(HasTaskKind<O>, "Objective needs task_of specialization");
        static_assert(HasDefaultMetricNames<O>,
                      "Objective needs default_metrics_of specialization");
        return ObjectiveEntry{impl_name<O>::value, &link_thunk<O>, task_of<O>::value,
                              default_metrics_of<O>::names};
    });

ObjectiveEntry const &lookup(std::string_view name, char const *what)
{
    auto const *const entry =
        std::ranges::find(objective_table, name, &ObjectiveEntry::name);
    if (entry == objective_table.end())
    {
        throw UnknownImplError(std::string{what} + ": no objective '" +
                               std::string{name} + "'");
    }
    return *entry;
}

} // namespace

void apply_link_inverse_by_name(std::string_view objective_name, floats_out scores)
{
    lookup(objective_name, "apply_link_inverse_by_name").apply(scores);
}

TaskKind task_kind_by_name(std::string_view objective_name)
{
    return lookup(objective_name, "task_kind_by_name").task;
}

std::span<std::string_view const>
default_metric_names_by_name(std::string_view objective_name)
{
    return lookup(objective_name, "default_metric_names_by_name").defaults;
}

} // namespace bonsai
