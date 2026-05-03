#pragma once

#include <string>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"

namespace bonsai::io
{

// Reader free functions per format. Each returns a Dataset (or, for fit,
// the BinMappers fitted on the file). Adding parquet etc. = sibling
// functions in this namespace.

Dataset read_csv(std::string const &path, DataConfig const &cfg,
                 BinMappers const &mappers);

BinMappers fit_from_csv(std::string const &path, Config const &cfg);

} // namespace bonsai::io

namespace bonsai::detail::csv
{

// Internal: parses a CSV file once into a ColumnBatch.
// Used by both io::read_csv (then bins) and io::fit_from_csv (then fits).
ColumnBatch parse(std::string const &path, DataConfig const &cfg);

} // namespace bonsai::detail::csv
