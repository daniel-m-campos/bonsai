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
// dispatch table and return a monomorphized Booster<O,G,Sa> as IBooster.
// Throws UnknownImplError if the triple is not in the table.
std::unique_ptr<IBooster> make_booster(Config const &config);

struct AvailableCombo
{
    std::string_view objective_name;
    std::string_view grower_name;
    std::string_view sampler_name;
};

// Enumerate every (objective, grower, sampler) combo in the dispatch
// table. Used by `bonsai info` and the Python sidecar for sanity.
std::vector<AvailableCombo> available_combos();

} // namespace bonsai
