#include "bonsai/bin_mapper.hpp"
#include "bonsai/bin_mappers.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/detail/perf.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "detail/device_buffer.cuh"
#include "detail/mapper_fit_kernels.cuh"

namespace bonsai
{

using namespace cuda_detail;

namespace
{

constexpr size_t k_fit_chunk_bytes   = size_t{512} << 20;
constexpr size_t k_fit_chunk_columns = 1024;

uint32_t fit_chunk_width(size_t n_feats, size_t m)
{
    size_t const by_bytes =
        k_fit_chunk_bytes / std::max<size_t>(1, m * sizeof(uint32_t));
    size_t const width = std::min(by_bytes, k_fit_chunk_columns);
    return static_cast<uint32_t>(std::clamp<size_t>(width, 1, n_feats));
}

struct FitScratch
{
    DeviceBuffer<uint32_t> keys;
    DeviceBuffer<uint32_t> swap;
    DeviceBuffer<uint32_t> nan_counts;
    DeviceBuffer<uint8_t>  parity;
    DeviceBuffer<float>    cuts;
    DeviceBuffer<uint32_t> n_cuts;
    std::vector<float>     host_cuts;
    std::vector<uint32_t>  host_n_cuts;

    FitScratch(uint32_t width, uint32_t m, uint32_t slot)
        : host_cuts(static_cast<size_t>(width) * slot), host_n_cuts(width)
    {
        keys.reserve(static_cast<size_t>(width) * m);
        swap.reserve(static_cast<size_t>(width) * m);
        nan_counts.reserve(width);
        parity.reserve(width);
        cuts.reserve(host_cuts.size());
        n_cuts.reserve(width);
    }
};

void fit_chunk(DeviceMatrix const &X, uint32_t const *rows, uint32_t m, uint32_t col0,
               uint32_t w, uint32_t budget, FitScratch &s)
{
    check(cudaMemset(s.nan_counts.data(), 0, w * sizeof(uint32_t)), "nan count reset");
    dim3 const         grid((m + k_transpose_tile - 1) / k_transpose_tile,
                            (w + k_transpose_tile - 1) / k_transpose_tile);
    SampleMatrix const sample{X.data, static_cast<uint32_t>(X.n_feats), rows, m};
    gather_keys_kernel<<<grid, dim3(k_transpose_tile, k_transpose_row_step)>>>(
        sample, col0, w, s.keys.data(), s.nan_counts.data());
    check(cudaGetLastError(), "mapper fit gather launch");
    radix_sort_columns_kernel<<<w, k_sort_threads>>>(s.keys.data(), s.swap.data(), m,
                                                     s.parity.data());
    check(cudaGetLastError(), "mapper fit sort launch");
    column_cuts_kernel<<<(w + k_cuts_warps - 1) / k_cuts_warps, k_cuts_threads>>>(
        s.keys.data(), s.swap.data(), s.parity.data(), s.nan_counts.data(), m, w,
        budget, s.cuts.data(), s.n_cuts.data());
    check(cudaGetLastError(), "mapper fit cuts launch");
    size_t const slot = budget + 2;
    check(cudaMemcpy(s.host_cuts.data(), s.cuts.data(), w * slot * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "cuts download");
    check(cudaMemcpy(s.host_n_cuts.data(), s.n_cuts.data(), w * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost),
          "cut count download");
}

void note_device_fit(size_t n_feats, uint32_t m, uint32_t width)
{
    if (!detail::IngestProfiler::instance().enabled)
    {
        return;
    }
    std::println(
        stderr,
        "bonsai: mapper fit on the device: {} columns x {} sample rows in chunks of {}",
        n_feats, m, width);
}

void fit_empty_sample(std::vector<std::optional<BinMapper>> &slots,
                      BinMapperConfig const                 &cfg)
{
    std::vector<float> none;
    for (auto &s : slots)
    {
        if (!s)
        {
            s = BinMapper::from_sample(none, cfg);
        }
    }
}

void fit_columns(DeviceMatrix const &X, uint32_t const *rows, uint32_t m,
                 BinMapperConfig const                 &cfg,
                 std::vector<std::optional<BinMapper>> &slots)
{
    auto const     budget = static_cast<uint32_t>(cfg.max_bin - 2);
    uint32_t const slot   = budget + 2;
    uint32_t const width  = fit_chunk_width(X.n_feats, m);
    note_device_fit(X.n_feats, m, width);
    FitScratch scratch(width, m, slot);
    for (size_t col0 = 0; col0 < X.n_feats; col0 += width)
    {
        auto const w = static_cast<uint32_t>(std::min<size_t>(width, X.n_feats - col0));
        fit_chunk(X, rows, m, static_cast<uint32_t>(col0), w, budget, scratch);
        for (uint32_t j = 0; j < w; ++j)
        {
            if (slots[col0 + j])
            {
                continue;
            }
            auto const first =
                scratch.host_cuts.begin() + static_cast<size_t>(j) * slot;
            slots[col0 + j] = BinMapper::from_cuts(
                std::vector<float>(first, first + scratch.host_n_cuts[j]));
        }
    }
}

} // namespace

BinMappers cuda_fit_mappers(DeviceMatrix const      &X,
                            std::vector<std::string> feature_names,
                            BinMapperConfig const &cfg, BinEdges const &bin_edges)
{
    detail::IngestProfiler::Lap lap;
    auto                        slots = mappers_from_edges(bin_edges, X.n_feats);
    auto const                  rows  = bin_sample_rows(X.n_rows, cfg);
    auto const m = static_cast<uint32_t>(rows.empty() ? X.n_rows : rows.size());
    DeviceBuffer<uint32_t> row_ids;
    if (!rows.empty())
    {
        row_ids.upload(rows.data(), rows.size());
    }
    if (m == 0)
    {
        fit_empty_sample(slots, cfg);
    }
    else
    {
        fit_columns(X, rows.empty() ? nullptr : row_ids.data(), m, cfg, slots);
    }
    lap(detail::IngestProfiler::instance().fit_s);
    return BinMappers::from_mappers(resolve_mappers(slots), std::move(feature_names));
}

} // namespace bonsai
