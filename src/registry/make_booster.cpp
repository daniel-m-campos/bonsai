#include "bonsai/registry/make_booster.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <memory>
#include <print>
#include <string_view>
#include <vector>

#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/detail/perf.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/registry/configurations.hpp"
#include "bonsai/registry/names.hpp"
#include "bonsai/typelist.hpp"

namespace bonsai
{

namespace
{

struct Entry
{
    std::string_view objective_name;
    std::string_view grower_name;
    std::string_view sampler_name;
    std::unique_ptr<ITrainableBooster> (*factory)(Config const &);
};

template <typename Combo>
std::unique_ptr<ITrainableBooster> factory_for(Config const &cfg)
{
    return std::make_unique<BoosterFor<Combo>>(cfg);
}

inline constexpr auto configurations = make_table<Configurations, Entry>(
    []<typename Combo>()
    {
        using O  = type_at_t<0, Combo>;
        using G  = type_at_t<1, Combo>;
        using Sa = type_at_t<2, Combo>;
        return Entry{
            impl_name<O>::value,
            impl_name<G>::value,
            impl_name<Sa>::value,
            &factory_for<Combo>,
        };
    });

struct GrowerEntry
{
    std::string_view name;
    bool             on_device;
};

inline constexpr auto growers = make_table<Growers, GrowerEntry>(
    []<typename G>()
    { return GrowerEntry{impl_name<G>::value, GPULevelEngine<typename G::Engine>}; });

void note_leaf_budget_route(Config const &config, DispatchConfig const &routed)
{
    if (!detail::GrowProfiler::instance().enabled || routed == config.dispatch)
    {
        return;
    }
    std::println(stderr,
                 "bonsai: a leaf budget of {} under depth {} cannot bind; {} grows the "
                 "tree as {}",
                 config.tree_config.max_leaves, unsigned{config.tree_config.max_depth},
                 config.dispatch.grower_name, routed.grower_name);
}

} // namespace

std::unique_ptr<ITrainableBooster> make_booster(Config const &config)
{
    DispatchConfig const   routed = resolve_dispatch(config);
    std::string_view const obj    = routed.objective_name;
    std::string_view const gr     = routed.grower_name;
    std::string_view const sa     = routed.sampler_name;
    note_leaf_budget_route(config, routed);

    auto const matches = [&](Entry const &e)
    { return e.objective_name == obj && e.grower_name == gr && e.sampler_name == sa; };
    auto const *const entry = std::ranges::find_if(configurations, matches);
    if (entry == configurations.end())
    {
        throw UnknownImplError("make_booster", routed);
    }
    return entry->factory(config);
}

UnknownImplError::UnknownImplError(std::string_view      caller,
                                   DispatchConfig const &dispatch)
    : std::runtime_error(std::format("{}: no impl for ({}, {}, {})", caller,
                                     dispatch.objective_name, dispatch.grower_name,
                                     dispatch.sampler_name))
{
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

bool grower_runs_on_device(std::string_view grower_name)
{
    auto const *const grower =
        std::ranges::find(growers, grower_name, &GrowerEntry::name);
    return grower != growers.end() && grower->on_device;
}

void select_device_for(Config const &config)
{
    if (grower_runs_on_device(config.dispatch.grower_name))
    {
        cuda_select_device(config.parallel.device_id);
    }
}

} // namespace bonsai
