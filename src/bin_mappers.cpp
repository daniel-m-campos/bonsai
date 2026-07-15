#include "bonsai/bin_mappers.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "bonsai/bin_mapper.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/config/errors.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/detail/perf.hpp"
#include "bonsai/parallel.hpp"

namespace bonsai
{

namespace
{

// One shared row sample for the whole matrix (decision 64): every feature's
// cuts come from the same rows, so the O(n) selection pass runs once instead of
// once per feature (mapper-fit was ~5-8s of a 16M fit). Empty result means
// "n_rows <= n_samples, use every row" — the whole-column path, unchanged and
// bit-identical for datasets that fit the sample.
std::vector<uint32_t> sample_rows(size_t n_rows, BinMapperConfig const &cfg)
{
    if (n_rows <= cfg.n_samples)
    {
        return {};
    }
    std::vector<uint32_t> picked;
    picked.reserve(cfg.n_samples);
    std::ranges::sample(std::views::iota(uint32_t{0}, static_cast<uint32_t>(n_rows)),
                        std::back_inserter(picked),
                        static_cast<std::ptrdiff_t>(cfg.n_samples),
                        std::mt19937(cfg.seed));
    return picked;
}

// Gather one feature's NaN-free values at the shared sample rows (or the whole
// column when the sample is empty), the working set BinMapper::from_sample cuts.
template <typename ColumnFn>
std::vector<float> gather(std::span<uint32_t const> rows, size_t n_rows, ColumnFn value)
{
    std::vector<float> out;
    if (rows.empty())
    {
        out.reserve(n_rows);
        for (size_t r = 0; r < n_rows; ++r)
        {
            float const v = value(r);
            if (!std::isnan(v))
            {
                out.push_back(v);
            }
        }
        return out;
    }
    out.reserve(rows.size());
    for (uint32_t const r : rows)
    {
        float const v = value(r);
        if (!std::isnan(v))
        {
            out.push_back(v);
        }
    }
    return out;
}

// Seed the override slots before the parallel loop: from_edges validates and
// throws ConfigError, which must not cross the parallel region.
void seed_edge_slots(BinEdges const &bin_edges, size_t n_features,
                     std::vector<std::optional<BinMapper>> &slots)
{
    for (auto const &[col, edges] : bin_edges)
    {
        if (col >= n_features)
        {
            throw ConfigError("bin_edges: column " + std::to_string(col) +
                              " is out of range for " + std::to_string(n_features) +
                              " features");
        }
        if (slots[col])
        {
            throw ConfigError("bin_edges: column " + std::to_string(col) +
                              " listed twice");
        }
        slots[col] = BinMapper::from_edges(edges);
    }
}

} // namespace

BinMappers BinMappers::fit(detail::ColumnBatch const &batch, BinMapperConfig const &cfg,
                           BinEdges const &bin_edges)
{
    detail::IngestProfiler::Lap lap;
    // One shared row sample, then feature-parallel gather+cut. Optional slots
    // because BinMapper has no default constructor.
    size_t const n_rows = batch.features.empty() ? 0 : batch.features[0].size();
    auto const   rows   = sample_rows(n_rows, cfg);
    std::vector<std::optional<BinMapper>> slots(batch.features.size());
    seed_edge_slots(bin_edges, batch.features.size(), slots);
    parallel::for_each_index(
        batch.features.size(),
        [&](size_t f)
        {
            if (slots[f])
            {
                return; // explicit edges: nothing to fit
            }
            auto const &col = batch.features[f];
            slots[f]        = BinMapper::from_sample(
                gather(rows, n_rows, [&](size_t r) { return col[r]; }), cfg);
        });
    lap(detail::IngestProfiler::instance().fit_s);

    BinMappers out;
    out.mappers_.reserve(slots.size());
    for (auto &s : slots)
    {
        // Every slot was filled by the loop above.
        out.mappers_.push_back(
            std::move(*s)); // NOLINT(bugprone-unchecked-optional-access)
    }
    out.feature_names_ = batch.feature_names;
    return out;
}

BinMappers BinMappers::fit(features_view X, std::vector<std::string> feature_names,
                           BinMapperConfig const &cfg, BinEdges const &bin_edges)
{
    detail::IngestProfiler::Lap lap;
    size_t const                n = X.extent(0);
    size_t const                f = X.extent(1);
    // One shared row sample, then each feature gathers only those rows straight
    // from the row-major matrix — no full-column scratch, and the O(n) sample
    // runs once for the matrix instead of once per feature (identical cuts to
    // the ColumnBatch overload for the same sample).
    auto const                            rows = sample_rows(n, cfg);
    std::vector<std::optional<BinMapper>> slots(f);
    seed_edge_slots(bin_edges, f, slots);
    parallel::for_each_index(f,
                             [&](size_t c)
                             {
                                 if (slots[c])
                                 {
                                     return; // explicit edges: nothing to fit
                                 }
                                 slots[c] = BinMapper::from_sample(
                                     gather(rows, n, [&](size_t r) { return X[r, c]; }),
                                     cfg);
                             });
    lap(detail::IngestProfiler::instance().fit_s);

    std::vector<BinMapper> mappers;
    mappers.reserve(f);
    for (auto &s : slots)
    {
        // Every slot was filled by the loop above.
        mappers.push_back(std::move(*s)); // NOLINT(bugprone-unchecked-optional-access)
    }
    return from_mappers(std::move(mappers), std::move(feature_names));
}

BinMapper const &BinMappers::operator[](size_t fid) const
{
    return mappers_[fid];
}

size_t BinMappers::size() const
{
    return mappers_.size();
}

std::span<std::string const> BinMappers::feature_names() const
{
    return feature_names_;
}

} // namespace bonsai
