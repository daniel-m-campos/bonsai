#pragma once

#include <memory>
#include <stdexcept>

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

} // namespace bonsai
