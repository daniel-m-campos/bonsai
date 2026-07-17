#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/cuda/grower.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/cuda/multi_engine.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/types.hpp"

namespace bonsai::test
{

// Skips the running test case when the grower under test needs a CUDA
// device this host (or build) doesn't have. No-op for CPU growers, so
// TEMPLATE_LIST cases over the full registry stay unconditional.
template <typename GrowerT> void skip_without_cuda()
{
    if constexpr (std::same_as<typename GrowerT::Engine, CudaHistogramEngine> ||
                  std::same_as<typename GrowerT::Engine, MultiCudaHistogramEngine>)
    {
        if (!cuda_available())
        {
            SKIP("cuda grower needs a usable CUDA device");
        }
    }
}

template <typename TreeT>
inline float predict_one(TreeT const &tree, std::vector<float> row)
{
    std::array<float, 1> out{};
    tree.predict(features_view{row.data(), 1, row.size()}, floats_out{out});
    return out[0];
}

struct Built
{
    BinMappers          mappers;
    Dataset             ds;
    detail::ColumnBatch batch; // retained so ingest tests can re-bin the raw
};

inline Built build(detail::ColumnBatch batch)
{
    BinMappers mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset    ds      = Dataset::bin(batch, mappers, {});
    return Built{
        .mappers = std::move(mappers), .ds = std::move(ds), .batch = std::move(batch)};
}

inline std::vector<row_id_t> iota_rows(size_t n)
{
    std::vector<row_id_t> v(n);
    for (row_id_t i = 0; i < n; ++i)
    {
        v[i] = i;
    }
    return v;
}

struct ScenarioInputs
{
    Built                 built;
    std::vector<float>    grad;
    std::vector<float>    hess;
    std::vector<row_id_t> rows;
};

// Single feature {0.0, 0.1, 0.9, 1.0}; grad {-1,-1,+1,+1} → a midpoint
// split cleanly separates the two halves. Used by smoke / NaN / hess
// / min-data tests in both grower test files.
inline ScenarioInputs separable_4row()
{
    detail::ColumnBatch batch{
        .features      = {{0.0F, 0.1F, 0.9F, 1.0F}},
        .labels        = std::vector<float>(4, 0.0F),
        .weights       = {},
        .feature_names = {"a"},
    };
    return ScenarioInputs{
        .built = build(std::move(batch)),
        .grad  = {-1.0F, -1.0F, +1.0F, +1.0F},
        .hess  = std::vector<float>(4, 1.0F),
        .rows  = iota_rows(4),
    };
}

// Single feature {0.0, 0.5, 1.0}; grad uniform → no candidate cut has
// positive gain. Used by no-split / empty-rows tests.
inline ScenarioInputs uniform_3row()
{
    detail::ColumnBatch batch{
        .features      = {{0.0F, 0.5F, 1.0F}},
        .labels        = std::vector<float>(3, 0.0F),
        .weights       = {},
        .feature_names = {"a"},
    };
    return ScenarioInputs{
        .built = build(std::move(batch)),
        .grad  = std::vector<float>(3, 1.0F),
        .hess  = std::vector<float>(3, 1.0F),
        .rows  = iota_rows(3),
    };
}

// Two-row scaffold for max_depth=0 tests.
inline ScenarioInputs two_value_pair()
{
    detail::ColumnBatch batch{
        .features      = {{0.0F, 1.0F}},
        .labels        = std::vector<float>(2, 0.0F),
        .weights       = {},
        .feature_names = {"a"},
    };
    return ScenarioInputs{
        .built = build(std::move(batch)),
        .grad  = {-1.0F, +1.0F},
        .hess  = std::vector<float>(2, 1.0F),
        .rows  = iota_rows(2),
    };
}

} // namespace bonsai::test
