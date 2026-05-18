#pragma once

#include <string_view>

#include "bonsai/types.hpp"

namespace bonsai
{

// Look up an objective by name in the Objectives typelist and apply its
// inverse link function in place. Throws UnknownImplError if no match.
void apply_link_inverse_by_name(std::string_view objective_name, floats_out scores);

} // namespace bonsai
