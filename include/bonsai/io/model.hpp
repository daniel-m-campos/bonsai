#pragma once

#include <memory>
#include <string>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"

namespace bonsai::io
{

// Serialize a booster (the full training Config, init score, bin mappers,
// and every tree) to a MessagePack binary file at `path`. The booster must
// be one of the concrete Booster<O,G,Sa> instantiations registered in the
// dispatch table; otherwise std::runtime_error is thrown.
void save_booster(IBooster const &booster, std::string const &path,
                  BinMappers const &mappers, Config const &cfg);

struct LoadedBooster
{
    std::unique_ptr<IBooster> booster;
    BinMappers                mappers;
    Config                    cfg;
};

// Read back what save_booster wrote. Reconstructs the booster type from
// the on-disk dispatch triple by going through the same registry.
LoadedBooster load_booster(std::string const &path);

} // namespace bonsai::io
