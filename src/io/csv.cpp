#include "bonsai/io/csv.hpp"

#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/parallel.hpp"

namespace bonsai::detail::csv
{

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
    float      val{};
    auto const [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), val);
    if (ec != std::errc{} || ptr != s.data() + s.size())
    {
        if (is_nan_literal(s))
        {
            return k_nan;
        }
        throw std::runtime_error("csv::parse: bad numeric field '" + std::string{s} +
                                 "'");
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

std::string read_file(std::string const &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("csv::parse: cannot open '" + path + "'");
    }
    in.seekg(0, std::ios::end);
    auto const size = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::string buf(size, '\0');
    in.read(buf.data(), static_cast<std::streamsize>(size));
    return buf;
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

// Where each CSV column's parsed value lands in the ColumnBatch.
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
    size_t idx  = 0; // feature slot when kind == feature
};

} // namespace

ColumnBatch parse(std::string const &path, DataConfig const &cfg)
{
    // Read the whole file once, index the data lines, then parse rows in
    // parallel straight into column-major storage. Each row writes only
    // its own slots, so the result is identical at any thread count.
    std::string const buf = read_file(path);

    std::vector<std::string_view> scratch;
    std::vector<std::string>      all_names;
    size_t                        pos = 0;

    auto next_line = [&](size_t &p) -> std::string_view
    {
        size_t const nl   = buf.find('\n', p);
        size_t const end  = nl == std::string::npos ? buf.size() : nl;
        auto         line = std::string_view{buf}.substr(p, end - p);
        p                 = end + 1;
        if (!line.empty() && line.back() == '\r')
        {
            line.remove_suffix(1);
        }
        return line;
    };

    if (cfg.header)
    {
        if (buf.empty())
        {
            throw std::runtime_error("csv::parse: empty file '" + path + "'");
        }
        split_csv_line(next_line(pos), scratch);
        all_names.reserve(scratch.size());
        for (auto const &f : scratch)
        {
            all_names.emplace_back(trim(f));
        }
    }

    std::vector<std::string_view> lines;
    while (pos < buf.size())
    {
        auto const line = next_line(pos);
        if (!line.empty())
        {
            lines.push_back(line);
        }
    }

    if (all_names.empty() && !lines.empty())
    {
        split_csv_line(lines.front(), scratch);
        all_names.reserve(scratch.size());
        for (size_t i = 0; i < scratch.size(); ++i)
        {
            all_names.emplace_back("f" + std::to_string(i));
        }
    }

    auto const   feature_cols = resolve_feature_cols(all_names.size(), cfg);
    size_t const n_cols       = all_names.size();
    size_t const n_rows       = lines.size();

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

    // Parses every field (including ignored columns, so bad data still
    // fails) and stores by destination. Throws on malformed rows.
    auto parse_line_into = [&](size_t r)
    {
        auto const line  = lines[r];
        size_t     c     = 0;
        size_t     start = 0;
        for (size_t i = 0; i <= line.size(); ++i)
        {
            if (i != line.size() && line[i] != ',')
            {
                continue;
            }
            if (c >= n_cols)
            {
                break; // too many fields; reported below
            }
            float const v = parse_field(line.substr(start, i - start), cfg.missing_nan,
                                        cfg.missing_sentinel);
            switch (dest[c].kind)
            {
            case ColDest::Kind::feature: batch.features[dest[c].idx][r] = v; break;
            case ColDest::Kind::label:   batch.labels[r] = v; break;
            case ColDest::Kind::weight:  batch.weights[r] = v; break;
            case ColDest::Kind::ignore:  break;
            }
            ++c;
            start = i + 1;
        }
        if (c != n_cols)
        {
            throw std::runtime_error("csv::parse: column count mismatch in '" + path +
                                     "'");
        }
    };

    // Exceptions must not escape a worker thread: record the first bad row,
    // then re-parse it serially so the original error propagates.
    std::atomic<size_t> first_bad{std::numeric_limits<size_t>::max()};
    parallel::for_each_index(n_rows,
                             [&](size_t r)
                             {
                                 try
                                 {
                                     parse_line_into(r);
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
    if (size_t const bad = first_bad.load();
        bad != std::numeric_limits<size_t>::max())
    {
        parse_line_into(bad); // throws with the field-level message
        throw std::runtime_error("csv::parse: malformed row in '" + path + "'");
    }

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
