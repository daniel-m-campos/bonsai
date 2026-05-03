#include "bonsai/io/csv.hpp"

namespace bonsai::io
{

// TODO(user): implement read_csv and fit_from_csv. Both call
// detail::csv::parse, then hand the ColumnBatch to Dataset::bin or
// BinMappers::fit respectively.
// Spec: docs/architecture/1-dataset.md "Readers" section.

} // namespace bonsai::io

namespace bonsai::detail::csv
{

// TODO(user): implement parse. ~50 LOC numeric-only CSV reader.
// header row, comma-separated, configurable label and weight columns.

} // namespace bonsai::detail::csv
