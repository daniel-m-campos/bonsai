// CUDA histogram backend: clang CUDA C++, same libc++/C++23 as the rest of
// the build. Design, batching, and precision scheme:
// docs/architecture/10-cuda.md.

#include "bonsai/config/errors.hpp"
#include "bonsai/config/tree_config.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/perf.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <memory>
#include <print>
#include <span>
#include <utility>
#include <vector>
#include <vector_types.h>

#include "detail/device_buffer.cuh"
#include "detail/device_context.cuh"
#include "detail/ingest_kernels.cuh"

namespace bonsai
{

// The device buffers, ingest plane, and CudaDeviceContext now live in
// namespace bonsai::cuda_detail (external linkage, shared with the
// device-context TU); name them unqualified throughout this TU.
using namespace cuda_detail;

// Flat device/host buffers throughout this file are offset by hand (docs/
// architecture/10-cuda.md); grad/hess travel as an adjacent pair everywhere
// in this API, matching the gradient-boosting literature's convention.
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

bool cuda_available()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

void cuda_select_device(uint32_t device_id)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess)
    {
        n = 0;
    }
    if (device_id == 0)
    {
        // The default device: set it when one exists, stay a no-op on
        // GPU-less hosts so graceful degradation (begin_root declines,
        // host fallback trains) is untouched.
        if (n > 0)
        {
            cudaSetDevice(0);
        }
        return;
    }
    if (std::cmp_greater_equal(device_id, n))
    {
        throw ConfigError("parallel.device_id " + std::to_string(device_id) +
                          " is out of range: " + std::to_string(n) +
                          " CUDA device(s) visible");
    }
    if (cudaSetDevice(static_cast<int>(device_id)) != cudaSuccess)
    {
        throw ConfigError("parallel.device_id " + std::to_string(device_id) +
                          ": cudaSetDevice failed");
    }
}

namespace
{
// The multi-GPU shard set. Not thread-safe against concurrent train calls.
std::vector<uint32_t> g_selected_devices;
} // namespace

void cuda_select_devices(std::span<uint32_t const> ids)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess)
    {
        n = 0;
    }
    for (uint32_t const id : ids)
    {
        // Duplicate ids are allowed (N contexts on one device); only the
        // range is validated, id by id, mirroring cuda_select_device.
        if (std::cmp_greater_equal(id, n))
        {
            throw ConfigError("parallel.device_ids entry " + std::to_string(id) +
                              " is out of range: " + std::to_string(n) +
                              " CUDA device(s) visible");
        }
    }
    g_selected_devices.assign(ids.begin(), ids.end());
}

std::vector<uint32_t> cuda_selected_devices()
{
    return g_selected_devices;
}

// The device-resident state (CudaDeviceContext) plus the CPU fallback engine
// used when begin_root declines the resident path. The engine forwards its
// device methods to ctx and keeps the fallback branches (populate) on cpu.
struct CudaHistogramEngine::Impl
{
    CudaDeviceContext  ctx;
    CpuHistogramEngine cpu;
};

CudaHistogramEngine::CudaHistogramEngine() : impl_(std::make_unique<Impl>()) {}
CudaHistogramEngine::~CudaHistogramEngine()                               = default;
CudaHistogramEngine::CudaHistogramEngine(CudaHistogramEngine &&) noexcept = default;
CudaHistogramEngine &
CudaHistogramEngine::operator=(CudaHistogramEngine &&) noexcept = default;

void CudaHistogramEngine::begin_tree(Dataset const &ds, floats_view grad,
                                     floats_view hess)
{
    impl_->ctx.begin_tree(ds, grad, hess);
}

// Host-plane fallback: builds the node's histograms on the CPU. Runs only
// when begin_root declines the resident path (oversized max_bin) — the GPU
// copy-back path this replaced was phase-1/2 research, retired by decision 41.
void CudaHistogramEngine::populate(Dataset const &ds, floats_view grad,
                                   floats_view hess, SplitInput &split_input,
                                   std::span<feature_id_t const> selected)
{
    auto &prof_counters = impl_->ctx.prof_counters;
    auto  lap           = prof_counters.lap();
    ++prof_counters.cpu_calls;
    impl_->cpu.populate(ds, grad, hess, split_input, selected);
    lap(prof_counters.cpu_s);
}

bool CudaHistogramEngine::begin_root(Dataset const &ds, floats_view grad,
                                     floats_view hess, SplitInput &root,
                                     std::span<feature_id_t const> selected)
{
    return impl_->ctx.begin_root(ds, grad, hess, root, selected);
}

void CudaHistogramEngine::stamp_leaves(std::span<LeafStamp const> stamps)
{
    impl_->ctx.stamp_leaves(stamps);
}

void CudaHistogramEngine::partition_level(Dataset const               &ds,
                                          std::span<PartitionOp const> ops,
                                          std::span<uint32_t>          child_counts)
{
    impl_->ctx.partition_level(ds, ops, child_counts);
}

void CudaHistogramEngine::finalize_rows(std::span<node_id_t> leaf_by_row)
{
    impl_->ctx.finalize_rows(leaf_by_row);
}

void CudaHistogramEngine::finalize_tree(std::span<float const> node_values,
                                        std::span<float>       values,
                                        std::span<node_id_t>   leaf_ids)
{
    impl_->ctx.finalize_tree(node_values, values, leaf_ids);
}

void CudaHistogramEngine::advance_level(Dataset const &ds, std::span<LevelOp const> ops)
{
    impl_->ctx.advance_level(ds, ops);
}

void CudaHistogramEngine::advance_layout_only()
{
    impl_->ctx.advance_layout_only();
}

void CudaHistogramEngine::find_splits_many(Dataset const &ds, TreeConfig const &config,
                                           std::span<SplitInput const> level,
                                           std::span<SplitOutput>      out,
                                           std::span<HistCell>         child_sums)
{
    impl_->ctx.find_splits_many(ds, config, level, out, child_sums);
}

void CudaHistogramEngine::find_level_split(Dataset const &ds, TreeConfig const &config,
                                           std::span<SplitInput const> level,
                                           std::span<SplitOutput>      out,
                                           std::span<HistCell>         child_sums)
{
    impl_->ctx.find_level_split(ds, config, level, out, child_sums);
}

// ---- The ingest transaction (decision 54) -----------------------------------

namespace
{

// Concatenated per-feature cut tables + offsets (n_feats + 1), device-side.
struct CutsTable
{
    DeviceBuffer<float>    cuts;
    DeviceBuffer<uint32_t> ofs;
};

// Out-param: DeviceBuffer is deliberately pinned in place (no copy/move).
void upload_cuts(BinMappers const &mappers, CutsTable &t)
{
    std::vector<uint32_t> ofs(mappers.size() + 1, 0);
    std::vector<float>    flat;
    for (size_t f = 0; f < mappers.size(); ++f)
    {
        auto const cuts = mappers[f].cuts();
        flat.insert(flat.end(), cuts.begin(), cuts.end());
        ofs[f + 1] = static_cast<uint32_t>(flat.size());
    }
    t.cuts.upload(flat.data(), flat.size());
    t.ofs.upload(ofs.data(), ofs.size());
}

// Mirror of begin_root's resident-path gate with every feature selected
// (the hist kernel's shared budget holds ONE feature's histogram, so the
// gate is the max single-feature bins, not the sum): if grow would fall
// back to the host data plane — which wants host bins — decline device
// ingest and keep today's eager host fill.
bool ingest_would_decline(BinMappers const &mappers)
{
    size_t max_bins = 0;
    for (size_t f = 0; f < mappers.size(); ++f)
    {
        max_bins = std::max(max_bins, mappers[f].n_bins());
    }
    size_t ceiling = k_max_shared_bytes;
    int    dev     = 0;
    int    optin   = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) ==
            cudaSuccess)
    {
        ceiling = std::max(ceiling, static_cast<size_t>(optin));
    }
    return 4 * max_bins * sizeof(float) > ceiling;
}

std::shared_ptr<CudaIngestPlane> make_ingest_plane(BinMappers const &mappers,
                                                   size_t            n_rows)
{
    auto                  plane = std::make_shared<CudaIngestPlane>();
    std::vector<uint32_t> counts(mappers.size());
    bool                  u8 = true;
    for (size_t f = 0; f < mappers.size(); ++f)
    {
        counts[f] = static_cast<uint32_t>(mappers[f].n_bins());
        u8        = u8 && counts[f] <= 256;
    }
    plane->bins_are_u8 = u8;
    plane->n_rows      = n_rows;
    plane->n_feats     = mappers.size();
    plane->n_bins.upload(counts.data(), counts.size());
    size_t const cells = n_rows * mappers.size();
    if (u8)
    {
        plane->bins8.reserve(cells);
    }
    else
    {
        plane->bins16.reserve(cells);
    }
    return plane;
}

// Raw chunks stream through one device buffer, ~64MB a piece; each chunk is
// copied then binned before the next (the copy dominates and already runs
// at bus rate — dbin in ingest-profile says whether overlap is ever worth
// the staging machinery).
constexpr size_t k_ingest_chunk_bytes = 64UL * 1024UL * 1024UL;

} // namespace

std::shared_ptr<IngestPlane const> cuda_ingest(features_view     X,
                                               BinMappers const &mappers)
{
    if (!cuda_available() || X.extent(0) == 0 || mappers.size() == 0 ||
        ingest_would_decline(mappers))
    {
        return nullptr;
    }
    detail::IngestProfiler::Lap lap;
    auto const                  n_rows  = X.extent(0);
    auto const                  n_feats = mappers.size();
    auto                        plane   = make_ingest_plane(mappers, n_rows);
    CutsTable                   table;
    upload_cuts(mappers, table);

    size_t const rows_per_chunk =
        std::max<size_t>(1, k_ingest_chunk_bytes / (n_feats * sizeof(float)));
    DeviceBuffer<float> raw;
    raw.reserve(rows_per_chunk * n_feats);
    for (size_t row0 = 0; row0 < n_rows; row0 += rows_per_chunk)
    {
        auto const rows  = std::min(rows_per_chunk, n_rows - row0);
        auto const cells = static_cast<uint32_t>(rows * n_feats);
        check(cudaMemcpy(raw.data(), &X[row0, 0], cells * sizeof(float),
                         cudaMemcpyHostToDevice),
              "ingest raw upload");
        dim3 const grid((cells + 255) / 256);
        if (plane->bins_are_u8)
        {
            bin_rows_kernel<<<grid, dim3(256)>>>(
                raw.data(), static_cast<uint32_t>(rows), static_cast<uint32_t>(row0),
                static_cast<uint32_t>(n_feats), static_cast<uint32_t>(n_rows),
                table.cuts.data(), table.ofs.data(), plane->bins8.data());
        }
        else
        {
            bin_rows_kernel<<<grid, dim3(256)>>>(
                raw.data(), static_cast<uint32_t>(rows), static_cast<uint32_t>(row0),
                static_cast<uint32_t>(n_feats), static_cast<uint32_t>(n_rows),
                table.cuts.data(), table.ofs.data(), plane->bins16.data());
        }
        check(cudaGetLastError(), "ingest bin launch");
    }
    check(cudaDeviceSynchronize(), "ingest sync");
    lap(detail::IngestProfiler::instance().dbin_s);
    return plane;
}

std::shared_ptr<IngestPlane const> cuda_ingest(detail::ColumnBatch const &batch,
                                               BinMappers const          &mappers)
{
    if (!cuda_available() || batch.features.empty() || ingest_would_decline(mappers))
    {
        return nullptr;
    }
    detail::IngestProfiler::Lap lap;
    auto const                  n_rows = batch.features[0].size();
    auto                        plane  = make_ingest_plane(mappers, n_rows);
    CutsTable                   table;
    upload_cuts(mappers, table);

    size_t const rows_per_chunk =
        std::max<size_t>(1, k_ingest_chunk_bytes / sizeof(float));
    DeviceBuffer<float> raw;
    raw.reserve(std::min(rows_per_chunk, n_rows));
    for (size_t f = 0; f < mappers.size(); ++f)
    {
        auto const     n_cuts = static_cast<uint32_t>(mappers[f].n_bins());
        uint32_t const c0     = [&]
        {
            uint32_t ofs = 0;
            for (size_t g = 0; g < f; ++g)
            {
                ofs += static_cast<uint32_t>(mappers[g].n_bins());
            }
            return ofs;
        }();
        for (size_t row0 = 0; row0 < n_rows; row0 += rows_per_chunk)
        {
            auto const n =
                static_cast<uint32_t>(std::min(rows_per_chunk, n_rows - row0));
            check(cudaMemcpy(raw.data(), batch.features[f].data() + row0,
                             n * sizeof(float), cudaMemcpyHostToDevice),
                  "ingest raw upload");
            dim3 const grid((n + 255) / 256);
            if (plane->bins_are_u8)
            {
                bin_col_kernel<<<grid, dim3(256)>>>(
                    raw.data(), n, static_cast<uint32_t>(row0), table.cuts.data() + c0,
                    n_cuts, plane->bins8.data() + (f * n_rows));
            }
            else
            {
                bin_col_kernel<<<grid, dim3(256)>>>(
                    raw.data(), n, static_cast<uint32_t>(row0), table.cuts.data() + c0,
                    n_cuts, plane->bins16.data() + (f * n_rows));
            }
            check(cudaGetLastError(), "ingest bin launch");
        }
    }
    check(cudaDeviceSynchronize(), "ingest sync");
    lap(detail::IngestProfiler::instance().dbin_s);
    return plane;
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

} // namespace bonsai
