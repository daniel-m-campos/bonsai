#include "bonsai/cli/handlers.hpp"

#include <cstdio>
#include <cstdlib>
#include <print>

#include "bonsai/io/model.hpp"

namespace bonsai::cli
{

int run_dump(DumpOpts const &opts)
{
    auto const loaded = io::load_booster(opts.model_path);
    std::fputs(loaded.booster->dump(loaded.mappers.feature_names()).c_str(), stdout);
    return EXIT_SUCCESS;
}

} // namespace bonsai::cli
