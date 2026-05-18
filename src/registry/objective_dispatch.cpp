#include "bonsai/registry/objective_dispatch.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

#include "bonsai/objective_traits.hpp"
#include "bonsai/registry/make_booster.hpp"
#include "bonsai/registry/names.hpp"
#include "bonsai/registry/typelists.hpp"
#include "bonsai/typelist.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

namespace
{

using LinkFn = void (*)(floats_out);

struct LinkEntry
{
    std::string_view name;
    LinkFn apply;
};

template <typename O> void link_thunk(floats_out scores)
{
    link_inverse_of<O>::apply(scores);
}

auto constexpr create_link_table()
{
    std::array<LinkEntry, size_v<Objectives>> out{};
    size_t i = 0;
    for_each_type<Objectives>(
        [&i, &out]<typename O>()
        {
            static_assert(HasName<O>, "Objective needs impl_name specialization");
            static_assert(HasLinkInverse<O>,
                          "Objective needs link_inverse_of specialization");
            out[i++] = LinkEntry{impl_name<O>::value, &link_thunk<O>};
        });
    return out;
}

inline auto constexpr link_table = create_link_table();

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

} // namespace bonsai
