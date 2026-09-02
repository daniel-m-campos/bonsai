#pragma once

#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"

namespace bonsai
{

class UnknownImplError : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

// Look up (objective_name, grower_name, sampler_name) in the compile-time
// dispatch table and return the combo's booster; softmax combos build a
// MulticlassBooster, every other combo a Booster. The trainable interface
// comes back because every caller of this factory goes on to train.
// Throws UnknownImplError if the triple is not in the table.
std::unique_ptr<ITrainableBooster> make_booster(Config const &config);

struct AvailableCombo
{
    std::string_view objective_name;
    std::string_view grower_name;
    std::string_view sampler_name;
};

// Enumerate every (objective, grower, sampler) combo in the dispatch
// table, in typelist order.
std::vector<AvailableCombo> available_combos();

// Answered from the grower's engine type, never from the spelling of its
// name (invariants: device-grower-by-engine-type).
bool grower_runs_on_device(std::string_view grower_name);

// A no-op for a grower that runs on the host, so every trainer calls it.
void select_device_for(Config const &config);

} // namespace bonsai
