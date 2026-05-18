#include "bonsai/io/csv.hpp"

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"

namespace bonsai::detail::csv
{

namespace
{

float constexpr k_nan = std::numeric_limits<float>::quiet_NaN();

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

float parse_field(std::string_view raw, bool missing_nan, std::optional<float> sentinel)
{
    auto const s = trim(raw);
    if (s.empty())
    {
        if (missing_nan)
        {
            return k_nan;
        }
        throw std::runtime_error("csv::parse: empty field with missing_nan=false");
    }
    if (is_nan_literal(s))
    {
        return k_nan;
    }
    std::string const buf{s};
    char *end           = nullptr;
    float const val     = std::strtof(buf.c_str(), &end);
    auto const consumed = static_cast<size_t>(end != nullptr ? end - buf.c_str() : 0);
    if (consumed != buf.size())
    {
        throw std::runtime_error("csv::parse: bad numeric field '" + buf + "'");
    }
    if (sentinel && val == *sentinel)
    {
        return k_nan;
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

std::vector<std::string> read_header(std::ifstream &in, std::string const &path,
                                     std::vector<std::string_view> &scratch)
{
    std::string line;
    if (!std::getline(in, line))
    {
        throw std::runtime_error("csv::parse: empty file '" + path + "'");
    }
    split_csv_line(line, scratch);
    std::vector<std::string> names;
    names.reserve(scratch.size());
    for (auto const &f : scratch)
    {
        names.emplace_back(trim(f));
    }
    return names;
}

std::vector<float> parse_row(std::string_view line, DataConfig const &cfg,
                             std::vector<std::string_view> &scratch)
{
    split_csv_line(line, scratch);
    std::vector<float> row;
    row.reserve(scratch.size());
    for (auto const &f : scratch)
    {
        row.push_back(parse_field(f, cfg.missing_nan, cfg.missing_sentinel));
    }
    return row;
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
    std::vector<size_t> out;
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

ColumnBatch materialize(std::vector<std::vector<float>> const &rows,
                        std::vector<size_t> const &feature_cols,
                        std::vector<std::string> const &all_names,
                        DataConfig const &cfg)
{
    auto const n_rows = rows.size();
    ColumnBatch batch;
    batch.features.assign(feature_cols.size(), std::vector<float>(n_rows));
    batch.labels.resize(n_rows);
    if (cfg.weight_column >= 0)
    {
        batch.weights.resize(n_rows);
    }
    batch.feature_names.reserve(feature_cols.size());
    for (auto const fc : feature_cols)
    {
        batch.feature_names.push_back(all_names[fc]);
    }
    for (size_t r = 0; r < n_rows; ++r)
    {
        auto const &row = rows[r];
        batch.labels[r] = row[static_cast<size_t>(cfg.label_column)];
        if (cfg.weight_column >= 0)
        {
            batch.weights[r] = row[static_cast<size_t>(cfg.weight_column)];
        }
        for (size_t f = 0; f < feature_cols.size(); ++f)
        {
            batch.features[f][r] = row[feature_cols[f]];
        }
    }
    return batch;
}

} // namespace

ColumnBatch parse(std::string const &path, DataConfig const &cfg)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("csv::parse: cannot open '" + path + "'");
    }

    std::vector<std::string_view> scratch;
    std::vector<std::string> all_names;
    if (cfg.header)
    {
        all_names = read_header(in, path, scratch);
    }

    std::vector<std::vector<float>> rows;
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || (line.size() == 1 && line[0] == '\r'))
        {
            continue;
        }
        auto row = parse_row(line, cfg, scratch);
        if (all_names.empty())
        {
            all_names.reserve(row.size());
            for (size_t i = 0; i < row.size(); ++i)
            {
                all_names.emplace_back("f" + std::to_string(i));
            }
        }
        if (row.size() != all_names.size())
        {
            throw std::runtime_error("csv::parse: column count mismatch in '" + path +
                                     "'");
        }
        rows.push_back(std::move(row));
    }

    auto const feature_cols = resolve_feature_cols(all_names.size(), cfg);
    return materialize(rows, feature_cols, all_names, cfg);
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
