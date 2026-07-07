#include "bonsai/cli/handlers.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <print>
#include <string>
#include <vector>

#include "bonsai/booster.hpp"
#include "bonsai/io/model.hpp"

namespace bonsai::cli
{

int run_importance(ImportanceOpts const &opts)
{
    auto const loaded = io::load_booster(opts.model_path);
    auto const names  = loaded.mappers.feature_names();

    auto pad = [&](std::vector<double> v)
    {
        v.resize(std::max(v.size(), loaded.mappers.size()), 0.0);
        return v;
    };
    auto const gain  = pad(loaded.booster->feature_importance(ImportanceType::gain));
    auto const split = pad(loaded.booster->feature_importance(ImportanceType::split));

    std::vector<size_t> order(gain.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::ranges::sort(order, [&](size_t a, size_t b) { return gain[a] > gain[b]; });

    std::println("{:<24} {:>14} {:>8}", "feature", "gain", "split");
    for (size_t const f : order)
    {
        auto const name = f < names.size() ? names[f] : "f" + std::to_string(f);
        std::println("{:<24} {:>14.2f} {:>8.0f}", name, gain[f], split[f]);
    }
    return EXIT_SUCCESS;
}

} // namespace bonsai::cli
