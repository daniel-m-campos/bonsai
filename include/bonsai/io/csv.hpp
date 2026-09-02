#pragma once

#include <string>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"

namespace bonsai::io
{

Dataset read_csv(std::string const &path, DataConfig const &cfg,
                 BinMappers const &mappers);

BinMappers fit_from_csv(std::string const &path, Config const &cfg);

} // namespace bonsai::io

namespace bonsai::detail::csv
{

// Parses a CSV file once into a ColumnBatch.
ColumnBatch parse(std::string const &path, DataConfig const &cfg);

} // namespace bonsai::detail::csv

namespace bonsai::detail::libsvm
{

// LIBSVM sparse text (`label idx:val ...`, 1-based indices), materialized
// dense: input-format support, not sparse compute.
ColumnBatch parse(std::string const &path, DataConfig const &cfg);

} // namespace bonsai::detail::libsvm

namespace bonsai::detail
{

// Dispatch on DataConfig::format ("csv" | "libsvm").
ColumnBatch parse_input(std::string const &path, DataConfig const &cfg);

} // namespace bonsai::detail
