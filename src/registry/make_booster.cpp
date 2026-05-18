#include "bonsai/registry/make_booster.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/registry/names.hpp"
#include "bonsai/registry/typelists.hpp"
#include "bonsai/typelist.hpp"

namespace bonsai
{

namespace
{

using Configurations = cartesian_product_t<Objectives, Growers, Samplers>;

struct Entry
{
    std::string_view objective_name;
    std::string_view grower_name;
    std::string_view sampler_name;
    std::unique_ptr<IBooster> (*factory)(Config const &);
};

template <typename Configuration>
std::unique_ptr<IBooster> factory_for(Config const &cfg)
{
    using O  = type_at_t<0, Configuration>;
    using G  = type_at_t<1, Configuration>;
    using Sa = type_at_t<2, Configuration>;
    return std::make_unique<Booster<O, G, Sa>>(cfg);
}

auto constexpr create_table()
{
    std::array<Entry, size_v<Configurations>> out{};
    size_t i = 0;
    for_each_type<Configurations>(
        [&i, &out]<typename Configuration>()
        {
            using O  = type_at_t<0, Configuration>;
            using G  = type_at_t<1, Configuration>;
            using Sa = type_at_t<2, Configuration>;
            static_assert(
                HasName<O>,
                "Objective in Objectives typelist needs impl_name specialization");
            static_assert(HasName<G>,
                          "Grower in Growers typelist needs impl_name specialization");
            static_assert(
                HasName<Sa>,
                "Sampler in Samplers typelist needs impl_name specialization");
            out[i++] = Entry{
                impl_name<O>::value,
                impl_name<G>::value,
                impl_name<Sa>::value,
                &factory_for<Configuration>,
            };
        });
    return out;
}

inline auto constexpr configurations = create_table();

} // namespace

std::unique_ptr<IBooster> make_booster(Config const &config)
{
    std::string_view const obj = config.dispatch.objective_name;
    std::string_view const gr  = config.dispatch.grower_name;
    std::string_view const sa  = config.dispatch.sampler_name;

    for (Entry const &e : configurations)
    {
        if (e.objective_name == obj && e.grower_name == gr && e.sampler_name == sa)
        {
            return e.factory(config);
        }
    }

    throw UnknownImplError("make_booster: no impl for (" + std::string{obj} + ", " +
                           std::string{gr} + ", " + std::string{sa} + ")");
}

std::vector<AvailableCombo> available_combos()
{
    std::vector<AvailableCombo> out;
    out.reserve(configurations.size());
    for (Entry const &e : configurations)
    {
        out.push_back({e.objective_name, e.grower_name, e.sampler_name});
    }
    return out;
}

} // namespace bonsai
