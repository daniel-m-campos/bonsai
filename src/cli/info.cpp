#include "bonsai/cli/handlers.hpp"

#include <cstdlib>
#include <print>
#include <ranges>
#include <set>
#include <string_view>

#include "bonsai/cuda/grower.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/registry/make_booster.hpp"
#include "bonsai/registry/names.hpp"

namespace bonsai::cli
{

namespace
{

bool trains_here(std::string_view grower_name)
{
    return !grower_runs_on_device(grower_name) || cuda_available();
}

} // namespace

int run_info()
{
    auto const combos = available_combos();
    auto const names  = [&](std::string_view AvailableCombo::*field)
    {
        return combos | std::views::transform(field) |
               std::ranges::to<std::set<std::string_view>>();
    };
    auto const objectives = names(&AvailableCombo::objective_name);
    auto const growers    = names(&AvailableCombo::grower_name);
    auto const samplers   = names(&AvailableCombo::sampler_name);

    std::println("bonsai");
    std::print("  objectives: ");
    for (auto const &n : objectives)
    {
        std::print("{} ", n);
    }
    std::print("\n  growers:    ");
    for (auto const &n : growers)
    {
        std::print("{}{} ", n, trains_here(n) ? "" : " (predict-only here)");
    }
    std::print("\n  samplers:   ");
    for (auto const &n : samplers)
    {
        std::print("{} ", n);
    }
    std::println("\n\nAvailable combos ({}):", combos.size());
    for (auto const &c : combos)
    {
        std::println("  {} / {} / {}", c.objective_name, c.grower_name, c.sampler_name);
    }
    return EXIT_SUCCESS;
}

} // namespace bonsai::cli
