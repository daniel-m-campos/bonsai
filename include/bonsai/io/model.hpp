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
// be an instantiation the dispatch table registers; otherwise
// std::runtime_error is thrown.
void save_booster(IBooster const &booster, std::string const &path,
                  BinMappers const &mappers, Config const &cfg);

struct LoadedBooster
{
    std::unique_ptr<ITrainableBooster> booster;
    BinMappers                         mappers;
    Config                             cfg;
};

// Read back what save_booster wrote. Reconstructs the booster type from
// the on-disk dispatch triple by going through the same registry.
LoadedBooster load_booster(std::string const &path);

// The same serialization without the file: save_booster is
// save_booster_bytes plus a write, load_booster a read plus
// load_booster_bytes, so bytes and file carry identical content (neither
// carries the eval history, a training artifact).
std::vector<uint8_t> save_booster_bytes(IBooster const   &booster,
                                        BinMappers const &mappers, Config const &cfg);
// `source` names the file in the bad-magic message; empty for pickle bytes,
// which have no path to name.
LoadedBooster load_booster_bytes(std::vector<uint8_t> const &bytes,
                                 std::string_view            source = {});

} // namespace bonsai::io
