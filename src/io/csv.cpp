#include "bonsai/io/csv.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/detail/perf.hpp"
#include "bonsai/parallel.hpp"

namespace bonsai::detail
{

namespace
{

std::string read_file(std::string const &path, std::string_view parser)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error(std::string{parser} + ": cannot open '" + path + "'");
    }
    in.seekg(0, std::ios::end);
    auto const size = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::string buf(size, '\0');
    in.read(buf.data(), static_cast<std::streamsize>(size));
    return buf;
}

std::string_view next_line(std::string const &buf, size_t &pos)
{
    size_t const nl   = buf.find('\n', pos);
    size_t const end  = nl == std::string::npos ? buf.size() : nl;
    auto         line = std::string_view{buf}.substr(pos, end - pos);
    pos               = end + 1;
    if (!line.empty() && line.back() == '\r')
    {
        line.remove_suffix(1);
    }
    return line;
}

std::vector<std::string> numbered_names(size_t n)
{
    std::vector<std::string> names;
    names.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        names.emplace_back("f" + std::to_string(i));
    }
    return names;
}

} // namespace

} // namespace bonsai::detail

namespace bonsai::detail::csv
{

using detail::IngestProfiler;

namespace
{

constexpr float k_nan = std::numeric_limits<float>::quiet_NaN();

std::string_view trim(std::string_view s)
{
    if (!s.empty() && s.back() == '\r')
    {
        s.remove_suffix(1);
    }
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
    {
        s.remove_prefix(1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t'))
    {
        s.remove_suffix(1);
    }
    return s;
}

bool is_nan_literal(std::string_view s)
{
    return s.size() == 3 && (s[0] == 'n' || s[0] == 'N') &&
           (s[1] == 'a' || s[1] == 'A') && (s[2] == 'n' || s[2] == 'N');
}

float parse_field(std::string_view raw, size_t row, size_t col)
{
    auto const s = trim(raw);
    if (s.empty())
    {
        throw std::runtime_error(
            "csv::parse: empty field at row " + std::to_string(row + 1) + ", column " +
            std::to_string(col + 1) + "; write missing values as 'nan'");
    }
    float val{};
    auto const [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), val);
    if (ec != std::errc{} || ptr != s.data() + s.size())
    {
        if (is_nan_literal(s))
        {
            return k_nan;
        }
        throw std::runtime_error("csv::parse: bad numeric field '" + std::string{s} +
                                 "' at row " + std::to_string(row + 1) + ", column " +
                                 std::to_string(col + 1));
    }
    return val;
}

void split_csv_line(std::string_view line, std::vector<std::string_view> &out)
{
    out.clear();
    size_t start = 0;
    for (size_t i = 0; i < line.size(); ++i)
    {
        if (line[i] == ',')
        {
            out.emplace_back(line.substr(start, i - start));
            start = i + 1;
        }
    }
    out.emplace_back(line.substr(start));
}

std::vector<std::string> header_names(std::string const &buf, size_t &pos,
                                      std::string const &path)
{
    if (buf.empty())
    {
        throw std::runtime_error("csv::parse: empty file '" + path + "'");
    }
    std::vector<std::string_view> fields;
    split_csv_line(next_line(buf, pos), fields);
    std::vector<std::string> names;
    names.reserve(fields.size());
    for (auto const &f : fields)
    {
        names.emplace_back(trim(f));
    }
    return names;
}

std::vector<std::string_view> body_lines(std::string const &buf, size_t pos)
{
    std::vector<std::string_view> lines;
    while (pos < buf.size())
    {
        auto const line = next_line(buf, pos);
        if (!line.empty())
        {
            lines.push_back(line);
        }
    }
    return lines;
}

size_t field_count(std::string_view line)
{
    std::vector<std::string_view> fields;
    split_csv_line(line, fields);
    return fields.size();
}

std::vector<size_t> resolve_feature_cols(size_t n_cols, DataConfig const &cfg)
{
    auto const lbl = cfg.label_column;
    auto const wt  = cfg.weight_column;
    if (lbl < 0 || static_cast<size_t>(lbl) >= n_cols)
    {
        throw std::runtime_error("csv::parse: label_column out of range");
    }
    if (wt >= 0 && static_cast<size_t>(wt) >= n_cols)
    {
        throw std::runtime_error("csv::parse: weight_column out of range");
    }
    std::unordered_set<int> const ignore(cfg.ignore_columns.begin(),
                                         cfg.ignore_columns.end());
    std::vector<size_t>           out;
    out.reserve(n_cols);
    for (size_t c = 0; c < n_cols; ++c)
    {
        auto const ci = static_cast<int>(c);
        if (ci == lbl || ci == wt || ignore.contains(ci))
        {
            continue;
        }
        out.push_back(c);
    }
    return out;
}

struct ColDest
{
    enum class Kind : uint8_t
    {
        feature,
        label,
        weight,
        ignore
    };
    Kind   kind = Kind::ignore;
    size_t idx  = 0;
};

std::vector<ColDest> column_destinations(size_t                     n_cols,
                                         std::vector<size_t> const &feature_cols,
                                         DataConfig const          &cfg)
{
    std::vector<ColDest> dest(n_cols);
    dest[static_cast<size_t>(cfg.label_column)] = {ColDest::Kind::label, 0};
    if (cfg.weight_column >= 0)
    {
        dest[static_cast<size_t>(cfg.weight_column)] = {ColDest::Kind::weight, 0};
    }
    for (size_t f = 0; f < feature_cols.size(); ++f)
    {
        dest[feature_cols[f]] = {ColDest::Kind::feature, f};
    }
    return dest;
}

ColumnBatch allocate_batch(std::vector<std::string> const &all_names,
                           std::vector<size_t> const &feature_cols, size_t n_rows,
                           bool has_weight)
{
    ColumnBatch batch;
    batch.features.assign(feature_cols.size(), std::vector<float>(n_rows));
    batch.labels.resize(n_rows);
    if (has_weight)
    {
        batch.weights.resize(n_rows);
    }
    batch.feature_names.reserve(feature_cols.size());
    for (auto const fc : feature_cols)
    {
        batch.feature_names.push_back(all_names[fc]);
    }
    return batch;
}

void store_field(ColDest const &dest, size_t r, float v, ColumnBatch &batch)
{
    switch (dest.kind)
    {
    case ColDest::Kind::feature:
        batch.features[dest.idx][r] = v;
        break;
    case ColDest::Kind::label:
        batch.labels[r] = v;
        break;
    case ColDest::Kind::weight:
        batch.weights[r] = v;
        break;
    case ColDest::Kind::ignore:
        break;
    }
}

void parse_row(std::string_view line, size_t r, std::vector<ColDest> const &dest,
               ColumnBatch &batch, std::string const &path)
{
    size_t const n_cols = dest.size();
    size_t       c      = 0;
    size_t       start  = 0;
    for (size_t i = 0; i <= line.size(); ++i)
    {
        if (i != line.size() && line[i] != ',')
        {
            continue;
        }
        if (c >= n_cols)
        {
            break;
        }
        store_field(dest[c], r, parse_field(line.substr(start, i - start), r, c),
                    batch);
        ++c;
        start = i + 1;
    }
    if (c != n_cols)
    {
        throw std::runtime_error("csv::parse: column count mismatch in '" + path + "'");
    }
}

template <typename ParseRow>
size_t first_failing_row(size_t n_rows, ParseRow const &parse)
{
    std::atomic<size_t> first_bad{std::numeric_limits<size_t>::max()};
    parallel::for_each_index(n_rows,
                             [&](size_t r)
                             {
                                 try
                                 {
                                     parse(r);
                                 }
                                 catch (...)
                                 {
                                     size_t seen = first_bad.load();
                                     while (r < seen &&
                                            !first_bad.compare_exchange_weak(seen, r))
                                     {
                                     }
                                 }
                             });
    return first_bad.load();
}

} // namespace

ColumnBatch parse(std::string const &path, DataConfig const &cfg)
{
    auto               &prof = IngestProfiler::instance();
    IngestProfiler::Lap lap;

    std::string const buf = read_file(path, "csv::parse");
    lap(prof.read_s);

    size_t                   pos = 0;
    std::vector<std::string> all_names =
        cfg.header ? header_names(buf, pos, path) : std::vector<std::string>{};
    auto const lines = body_lines(buf, pos);
    if (all_names.empty() && !lines.empty())
    {
        all_names = numbered_names(field_count(lines.front()));
    }
    lap(prof.index_s);

    auto const feature_cols = resolve_feature_cols(all_names.size(), cfg);
    auto const dest         = column_destinations(all_names.size(), feature_cols, cfg);
    auto       batch =
        allocate_batch(all_names, feature_cols, lines.size(), cfg.weight_column >= 0);

    auto const parse_line_into = [&](size_t r)
    { parse_row(lines[r], r, dest, batch, path); };
    if (size_t const bad = first_failing_row(lines.size(), parse_line_into);
        bad != std::numeric_limits<size_t>::max())
    {
        parse_line_into(bad);
        throw std::runtime_error("csv::parse: malformed row in '" + path + "'");
    }
    lap(prof.parse_s);

    return batch;
}

} // namespace bonsai::detail::csv

namespace bonsai::io
{

Dataset read_csv(std::string const &path, DataConfig const &cfg,
                 BinMappers const &mappers)
{
    auto const batch = detail::csv::parse(path, cfg);
    return Dataset::bin(batch, mappers, cfg);
}

BinMappers fit_from_csv(std::string const &path, Config const &cfg)
{
    auto const batch = detail::csv::parse(path, cfg.data);
    return BinMappers::fit(batch, cfg.bin_mapper);
}

} // namespace bonsai::io

namespace bonsai::detail::libsvm
{

namespace
{

struct Entry
{
    uint32_t feature;
    float    value;
};

struct SparseRow
{
    float              label{};
    std::vector<Entry> entries;
};

void skip_spaces(std::string_view line, size_t &cursor)
{
    while (cursor < line.size() && line[cursor] == ' ')
    {
        ++cursor;
    }
}

Entry parse_pair(std::string_view line, size_t &cursor, std::string const &path)
{
    size_t const colon = line.find(':', cursor);
    if (colon == std::string_view::npos)
    {
        throw std::runtime_error("libsvm::parse: malformed pair in '" + path + "'");
    }
    uint32_t idx{};
    std::from_chars(line.data() + cursor, line.data() + colon, idx);
    size_t const vend = std::min(line.find(' ', colon), line.size());
    float        val{};
    std::from_chars(line.data() + colon + 1, line.data() + vend, val);
    if (idx == 0)
    {
        throw std::runtime_error(
            "libsvm::parse: feature indices are 1-based; got 0 in '" + path + "'");
    }
    cursor = vend;
    return {.feature = idx - 1, .value = val};
}

SparseRow parse_sparse_row(std::string_view line, std::string const &path)
{
    SparseRow    row;
    size_t const sp   = line.find(' ');
    auto const   lend = sp == std::string_view::npos ? line.size() : sp;
    std::from_chars(line.data(), // NOLINT(bugprone-suspicious-stringview-data-usage)
                    line.data() + lend, row.label);

    size_t cursor = lend;
    skip_spaces(line, cursor);
    while (cursor < line.size())
    {
        row.entries.push_back(parse_pair(line, cursor, path));
        skip_spaces(line, cursor);
    }
    return row;
}

ColumnBatch dense_from_rows(std::vector<SparseRow> const &rows, DataConfig const &cfg)
{
    uint32_t max_feature = 0;
    for (auto const &row : rows)
    {
        for (auto const &entry : row.entries)
        {
            max_feature = std::max(max_feature, entry.feature);
        }
    }
    size_t const n_rows   = rows.size();
    size_t const inferred = rows.empty() ? 0 : max_feature + 1;
    size_t const n_features =
        cfg.libsvm_n_features > 0
            ? std::max(inferred, static_cast<size_t>(cfg.libsvm_n_features))
            : inferred;

    ColumnBatch batch;
    batch.labels.reserve(n_rows);
    batch.features.assign(n_features, std::vector<float>(n_rows, 0.0F));
    for (size_t r = 0; r < n_rows; ++r)
    {
        batch.labels.push_back(rows[r].label);
        for (auto const &[f, v] : rows[r].entries)
        {
            batch.features[f][r] = v;
        }
    }
    batch.feature_names = numbered_names(n_features);
    return batch;
}

} // namespace

ColumnBatch parse(std::string const &path, DataConfig const &cfg)
{
    std::string const      buf = read_file(path, "libsvm::parse");
    std::vector<SparseRow> rows;
    size_t                 pos = 0;
    while (pos < buf.size())
    {
        auto const line = next_line(buf, pos);
        if (line.empty() || line.front() == '#')
        {
            continue;
        }
        rows.push_back(parse_sparse_row(line, path));
    }
    return dense_from_rows(rows, cfg);
}

} // namespace bonsai::detail::libsvm

namespace bonsai::detail
{

ColumnBatch parse_input(std::string const &path, DataConfig const &cfg)
{
    if (cfg.format == "libsvm")
    {
        return libsvm::parse(path, cfg);
    }
    return csv::parse(path, cfg);
}

} // namespace bonsai::detail
