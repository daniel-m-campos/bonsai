#pragma once

#include <stdexcept>
#include <string>

namespace bonsai
{

class ConfigError : public std::runtime_error
{
  public:
    explicit ConfigError(std::string const &msg) : std::runtime_error(msg) {}
};

} // namespace bonsai
