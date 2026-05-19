#include "bonsai/cli/handlers.hpp"

#include <cstdlib>
#include <print>

#include "bonsai/config/config.hpp"
#include "bonsai/config/toml.hpp"

namespace bonsai::cli
{

int run_params()
{
    std::println("{}", bonsai::config::dump_toml(Config{}));
    return EXIT_SUCCESS;
}

} // namespace bonsai::cli
