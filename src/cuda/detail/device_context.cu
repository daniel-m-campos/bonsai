
#include "bonsai/config/errors.hpp"
#include "bonsai/config/tree_config.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/dataset.hpp"
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
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <vector_types.h>

#include "device_buffer.cuh"
#include "device_context.cuh"
#include "kernels.cuh"

namespace bonsai
{
namespace cuda_detail
{

constexpr uint32_t k_sum_blocks = 64;

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

namespace
{

[[noreturn]] void refuse_hist_budget(size_t max_bins, size_t limit)
{
    throw ConfigError(
        "cuda: the widest selected feature has " + std::to_string(max_bins) +
        " bins, so one node histogram needs " +
        std::to_string(4 * max_bins * sizeof(float)) +
        " bytes of shared memory, above this device's " + std::to_string(limit) +
        "-byte limit. Lower bin_mapper.max_bin, or train on the host plane "
        "(device=\"cpu\", or a dispatch.grower_name without the cuda_ prefix).");
}

[[noreturn]] void refuse_leaf_pool(size_t max_slots, size_t n_selected, size_t max_bins)
{
    throw ConfigError(
        "cuda_leafwise: a leaf budget of " + std::to_string(max_slots) +
        " leaves over " + std::to_string(n_selected) + " features at " +
        std::to_string(max_bins) +
        " bins needs a histogram pool larger than a quarter of this device's free "
        "memory. Lower tree.max_leaves or bin_mapper.max_bin, or train on the host "
        "plane (device=\"cpu\", or a dispatch.grower_name without the cuda_ prefix).");
}

[[noreturn]] void refuse_empty_selection()
{
    throw ConfigError("cuda: this tree selected no features; raise "
                      "tree.feature_fraction, or train on the host plane "
                      "(device=\"cpu\").");
}

template <typename BinT>
__global__ void gather_cols_kernel(BinT const *src, BinT *dst, uint32_t const *keep,
                                   uint32_t const *rows, uint32_t n_rows_src,
                                   uint32_t n_feats_src, uint32_t n_rows_dst,
                                   uint32_t n_feats_dst)
{
    uint32_t const i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= n_rows_dst)
    {
        return;
    }
    uint32_t const t = blockIdx.y;
    uint32_t const w = tile_strip(t, n_feats_dst);
    uint32_t const r = mapped_row(rows, i);
    for (uint32_t j = 0; j < w; ++j)
    {
        uint32_t const f = (t * k_bin_tile_width) + j;
        dst[tiled_cell(f, i, n_rows_dst, n_feats_dst)] =
            src[tiled_cell(keep[f], r, n_rows_src, n_feats_src)];
    }
}

template <typename BinT>
void materialize_tiled(DeviceBuffer<BinT> const &bins, size_t n_rows, size_t n_feats,
                       std::vector<std::vector<BinT>> &out)
{
    out.resize(n_feats);
    for (size_t f = 0; f < n_feats; ++f)
    {
        out[f].resize(n_rows);
    }
    size_t const      chunk = std::max<size_t>(1, (1UL << 22) / k_bin_tile_width);
    std::vector<BinT> staging(chunk * k_bin_tile_width);
    auto const        tiles = tile_count(static_cast<uint32_t>(n_feats));
    for (uint32_t t = 0; t < tiles; ++t)
    {
        size_t const wt   = tile_strip(t, static_cast<uint32_t>(n_feats));
        size_t const base = n_rows * t * k_bin_tile_width;
        for (size_t r0 = 0; r0 < n_rows; r0 += chunk)
        {
            size_t const rows = std::min(chunk, n_rows - r0);
            check(cudaMemcpy(staging.data(), bins.data() + base + (r0 * wt),
                             rows * wt * sizeof(BinT), cudaMemcpyDeviceToHost),
                  "materialize bins");
            for (size_t j = 0; j < wt; ++j)
            {
                auto &col = out[(t * k_bin_tile_width) + j];
                for (size_t r = 0; r < rows; ++r)
                {
                    col[r0 + r] = staging[(r * wt) + j];
                }
            }
        }
    }
}

template <typename BinT> void stage_tiled(Dataset const &dataset, BinT *staging)
{
    size_t const     n_rows  = dataset.plane_n_rows();
    auto const       n_feats = static_cast<uint32_t>(dataset.n_features());
    constexpr size_t block   = 8192;
    parallel::for_each_index(
        (n_rows + block - 1) / block,
        [&](size_t b)
        {
            size_t const r0 = b * block;
            size_t const r1 = std::min(r0 + block, n_rows);
            for (uint32_t f = 0; f < n_feats; ++f)
            {
                dataset.visit_bins(f,
                                   [&](auto src)
                                   {
                                       uint32_t const t   = f / k_bin_tile_width;
                                       size_t const   wt  = tile_strip(t, n_feats);
                                       BinT          *dst = staging +
                                                   (n_rows * t * k_bin_tile_width) +
                                                   (f % k_bin_tile_width);
                                       for (size_t r = r0; r < r1; ++r)
                                       {
                                           dst[r * wt] = static_cast<BinT>(src[r]);
                                       }
                                   });
            }
        });
}

} // namespace

void CudaIngestPlane::materialize(BinColumns &cols) const
{
    if (bins_are_u8)
    {
        materialize_tiled(bins8, n_rows, n_feats, std::get<U8Columns>(cols));
        return;
    }
    materialize_tiled(bins16, n_rows, n_feats, std::get<U16Columns>(cols));
}

std::shared_ptr<IngestPlane const>
CudaIngestPlane::select_columns(std::span<feature_id_t const> keep,
                                std::span<row_id_t const>     rows) const
{
    if (keep.empty() || keep.size() > n_feats)
    {
        return nullptr;
    }
    for (feature_id_t const f : keep)
    {
        if (f >= n_feats)
        {
            return nullptr;
        }
    }
    RowMap const map{rows, n_rows};
    size_t const out_rows = map.n;
    if (out_rows == 0)
    {
        return nullptr;
    }

    std::vector<uint32_t> counts(n_feats);
    check(cudaMemcpy(counts.data(), n_bins.data(), n_feats * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost),
          "select_columns bin counts");
    std::vector<uint32_t> kept_counts(keep.size());
    bool                  u8 = true;
    for (size_t k = 0; k < keep.size(); ++k)
    {
        kept_counts[k] = counts[keep[k]];
        u8             = u8 && kept_counts[k] <= 256;
    }
    if (u8 != bins_are_u8)
    {
        return nullptr;
    }

    auto out         = std::make_shared<CudaIngestPlane>();
    out->bins_are_u8 = bins_are_u8;
    out->tile_w      = tile_w;
    out->n_rows      = out_rows;
    out->n_feats     = keep.size();
    out->n_bins.upload(kept_counts.data(), kept_counts.size());

    DeviceBuffer<uint32_t> d_keep;
    d_keep.upload(keep.data(), keep.size());

    auto const n_dst = static_cast<uint32_t>(out_rows);
    auto const f_dst = static_cast<uint32_t>(keep.size());
    dim3 const grid((n_dst + 255) / 256, tile_count(f_dst));
    auto const launch = [&](auto const *src, auto *dst)
    {
        gather_cols_kernel<<<grid, dim3(256)>>>(
            src, dst, d_keep.data(), map.data(), static_cast<uint32_t>(n_rows),
            static_cast<uint32_t>(n_feats), n_dst, f_dst);
    };
    size_t const cells = out_rows * keep.size();
    if (bins_are_u8)
    {
        out->bins8.reserve(cells);
        launch(bins8.data(), out->bins8.data());
    }
    else
    {
        out->bins16.reserve(cells);
        launch(bins16.data(), out->bins16.data());
    }
    check(cudaGetLastError(), "select_columns gather launch");
    check(cudaDeviceSynchronize(), "select_columns gather");
    return out;
}

void CudaDeviceContext::LevelPipeline::prof_record_begin(bool root)
{
    if (!prof_ev_ready)
    {
        for (auto &e : prof_ev)
        {
            check(cudaEventCreate(&e), "profile event create");
        }
        prof_ev_ready = true;
    }
    prof_ev_root = root;
    if (!root)
    {
        check(cudaEventRecord(prof_ev[ev_before_memset]), "profile event record");
    }
}

void CudaDeviceContext::LevelPipeline::prof_read(ProfileCounters &prof)
{
    if (!prof_ev_recorded)
    {
        return;
    }
    prof_ev_recorded = false;
    float ms         = 0.0F;
    check(cudaEventElapsedTime(&ms, prof_ev[ev_after_memset], prof_ev[ev_after_hist]),
          "profile event hist");
    (prof_ev_root ? prof.root_hist_s : prof.adv_hist_s) += ms / 1e3;
    if (prof_ev_root)
    {
        return;
    }
    check(
        cudaEventElapsedTime(&ms, prof_ev[ev_before_memset], prof_ev[ev_after_memset]),
        "profile event memset");
    prof.adv_memset_s += ms / 1e3;
    check(cudaEventElapsedTime(&ms, prof_ev[ev_after_hist], prof_ev[ev_after_subtract]),
          "profile event subtract");
    prof.adv_sub_s += ms / 1e3;
}

CudaDeviceContext::LevelPipeline::~LevelPipeline()
{
    if (prof_ev_ready)
    {
        for (auto &e : prof_ev)
        {
            cudaEventDestroy(e);
        }
    }
    if (part_ev_ready)
    {
        for (auto &e : part_ev)
        {
            cudaEventDestroy(e);
        }
    }
}

size_t CudaDeviceContext::LevelPipeline::stage_children(
    std::span<CudaHistogramEngine::LevelOp const> ops)
{
    size_t max_rows = 0;
    row_offsets.clear();
    row_counts.clear();
    slots.clear();
    small_offsets.clear();
    small_counts.clear();
    small_slots.clear();
    triples.clear();
    for (CudaHistogramEngine::LevelOp const &op : ops)
    {
        uint32_t const offset = next_offsets[op.small_slot];
        uint32_t const count  = next_counts[op.small_slot];
        if (count >= k_min_gpu_rows)
        {
            row_offsets.host.push_back(offset);
            row_counts.host.push_back(count);
            slots.host.push_back(op.small_slot);
            max_rows = std::max<size_t>(max_rows, count);
        }
        else
        {
            small_offsets.host.push_back(offset);
            small_counts.host.push_back(count);
            small_slots.host.push_back(op.small_slot);
        }
        triples.host.push_back(op.parent_slot);
        triples.host.push_back(op.small_slot);
        triples.host.push_back(op.large_slot);
    }
    if (!row_offsets.empty())
    {
        row_offsets.sync();
        row_counts.sync();
        slots.sync();
    }
    if (!small_offsets.empty())
    {
        small_offsets.sync();
        small_counts.sync();
        small_slots.sync();
    }
    triples.sync();
    return max_rows;
}

void CudaDeviceContext::LevelPipeline::layout_children(
    std::span<CudaHistogramEngine::PartitionOp const> ops,
    std::span<uint32_t>                               child_counts)
{
    size_t const n = ops.size();
    next_offsets.assign(2 * n, 0);
    next_counts.assign(2 * n, 0);
    for (size_t k = 0; k < n; ++k)
    {
        uint32_t const nl               = nl_dev.host[k];
        uint32_t const parent_offset    = part_ops.host[k].offset;
        uint32_t const parent_count     = part_ops.host[k].count;
        next_offsets[ops[k].left_slot]  = parent_offset;
        next_counts[ops[k].left_slot]   = nl;
        next_offsets[ops[k].right_slot] = parent_offset + nl;
        next_counts[ops[k].right_slot]  = parent_count - nl;
        child_counts[2 * k]             = nl;
        child_counts[(2 * k) + 1]       = parent_count - nl;
    }
}

bool CudaDeviceContext::LevelPipeline::stage_find_inputs(
    std::span<SplitInput const> level, TreeConfig const &config, Dataset const &ds)
{
    size_t const n = level.size();
    node_sums.host.resize(2 * n);
    node_bounds.host.resize(2 * n);
    bool any_mask = false;
    for (size_t i = 0; i < n; ++i)
    {
        node_sums.host[2 * i]         = level[i].sums.sum_grad;
        node_sums.host[(2 * i) + 1]   = level[i].sums.sum_hess;
        node_bounds.host[2 * i]       = level[i].lo;
        node_bounds.host[(2 * i) + 1] = level[i].hi;
        any_mask                      = any_mask || !level[i].allowed.empty();
    }
    node_sums.sync();
    node_bounds.sync();
    if (any_mask)
    {
        allowed.host.resize(n * n_selected);
        for (size_t i = 0; i < n; ++i)
        {
            for (uint32_t s = 0; s < n_selected; ++s)
            {
                allowed.host[(i * n_selected) + s] =
                    level[i].allowed.empty() ? char{1}
                                             : level[i].allowed[features.host[s]];
            }
        }
        allowed.sync();
    }
    monotone.host.resize(ds.n_features());
    for (feature_id_t f = 0; f < ds.n_features(); ++f)
    {
        monotone.host[f] = monotone_constraint_of(config, f);
    }
    monotone.sync();
    return any_mask;
}

void CudaDeviceContext::LevelPipeline::unpack_splits(std::span<SplitInput const> level,
                                                     TreeConfig const           &config,
                                                     std::span<SplitOutput>      out,
                                                     std::span<HistCell> child_sums)
{
    for (size_t i = 0; i < level.size(); ++i)
    {
        FeatBest const &b = node_best.host[i];
        bool const      eligible =
            level[i].row_count >= 2 * static_cast<size_t>(config.min_data_in_leaf);
        if (b.valid == 0 || !eligible)
        {
            out[i]                  = {};
            child_sums[2 * i]       = {};
            child_sums[(2 * i) + 1] = {};
            continue;
        }
        out[i]                  = {.gain       = b.gain,
                                   .feature_id = static_cast<feature_id_t>(
                      features.host[static_cast<size_t>(b.sel)]),
                                   .bin_id       = static_cast<bin_id_t>(b.bin),
                                   .default_left = b.dl != 0,
                                   .valid        = true};
        child_sums[2 * i]       = {.sum_grad = static_cast<float>(b.gL),
                                   .sum_hess = static_cast<float>(b.hL)};
        child_sums[(2 * i) + 1] = {.sum_grad = static_cast<float>(b.gR),
                                   .sum_hess = static_cast<float>(b.hR)};
    }
}

void CudaDeviceContext::LevelPipeline::stage_level_sums(
    std::span<SplitInput const> level)
{
    node_sums.host.resize(2 * level.size());
    for (size_t i = 0; i < level.size(); ++i)
    {
        node_sums.host[2 * i]       = level[i].sums.sum_grad;
        node_sums.host[(2 * i) + 1] = level[i].sums.sum_hess;
    }
    node_sums.sync();
}

void CudaDeviceContext::init_shared_limit()
{
    if (shared_probed)
    {
        return;
    }
    shared_probed = true;
    int dev       = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
    {
        return;
    }
    if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev) !=
        cudaSuccess)
    {
        sm_count = 0;
    }
    int optin = 0;
    if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) !=
            cudaSuccess ||
        static_cast<size_t>(optin) <= k_max_shared_bytes)
    {
        return;
    }
    if (cudaFuncSetAttribute(hist_kernel<uint8_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             optin) == cudaSuccess &&
        cudaFuncSetAttribute(hist_kernel<uint16_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             optin) == cudaSuccess)
    {
        shared_limit = static_cast<size_t>(optin);
    }
    cudaGetLastError();
}

size_t CudaDeviceContext::stage_selection(Dataset const                &ds,
                                          std::span<feature_id_t const> selected)
{
    size_t max_sel_bins = 0;
    for (feature_id_t const fid : selected)
    {
        max_sel_bins = std::max(max_sel_bins, ds.n_bins(fid));
    }
    lvl.n_selected = static_cast<uint32_t>(selected.size());
    lvl.stride     = static_cast<uint32_t>(2 * max_sel_bins);
    lvl.features.host.assign(selected.begin(), selected.end());
    lvl.features.sync();
    lvl.sel_slot.host.assign(ds.n_features(), k_not_selected);
    for (uint32_t i = 0; i < selected.size(); ++i)
    {
        lvl.sel_slot.host[selected[i]] = i;
    }
    lvl.sel_slot.sync();
    return max_sel_bins;
}

void CudaDeviceContext::require_hist_fits(size_t max_sel_bins) const
{
    if (lvl.n_selected == 0)
    {
        refuse_empty_selection();
    }
    if (!hist_budget_ok(max_sel_bins))
    {
        refuse_hist_budget(max_sel_bins, shared_limit);
    }
}

void CudaDeviceContext::launch_root_sums(float2 const *gh, uint32_t n)
{
    lvl.sum_partial.reserve(k_sum_blocks);
    lvl.sum_out.reserve(1);
    sum_gh_pass1_kernel<<<dim3(k_sum_blocks), dim3(256)>>>(gh, n,
                                                           lvl.sum_partial.data());
    check(cudaGetLastError(), "root sum pass1 launch");
    sum_gh_pass2_kernel<<<dim3(1), dim3(32)>>>(lvl.sum_partial.data(), k_sum_blocks,
                                               lvl.sum_out.data());
    check(cudaGetLastError(), "root sum pass2 launch");
}

HistCell CudaDeviceContext::fetch_root_sums()
{
    double2 sums{};
    check(
        cudaMemcpy(&sums, lvl.sum_out.data(), sizeof(double2), cudaMemcpyDeviceToHost),
        "root sums fetch");
    return {.sum_grad = static_cast<float>(sums.x),
            .sum_hess = static_cast<float>(sums.y)};
}

void CudaDeviceContext::wait_for_profile(ProfileCounters::Lap &lap)
{
    if (!prof_counters.enabled)
    {
        return;
    }
    check(cudaDeviceSynchronize(), "profile wait");
    lap(prof_counters.gpu_wait_s);
    lvl.prof_read(prof_counters);
}

void CudaDeviceContext::note_plane(bool tiled, size_t shared)
{
    if (plane_noted || !prof_counters.enabled)
    {
        return;
    }
    plane_noted = true;
    std::println(stderr,
                 "bonsai: bin plane is tile-blocked, width {}, {} cells; histogram "
                 "build is {} at {} shared bytes per block",
                 k_bin_tile_width, data.bins_are_u8 ? "u8" : "u16",
                 tiled ? "tiled" : "one feature per block", shared);
}

void CudaDeviceContext::launch_hist(uint32_t ds_rows, uint32_t ds_feats,
                                    uint32_t n_nodes, uint32_t max_rows,
                                    float2 const *gh, uint32_t const *rows,
                                    uint32_t const *offsets, uint32_t const *counts,
                                    double *out, uint32_t const *slots)
{
    size_t const tiled_shared =
        static_cast<size_t>(k_bin_tile_width) * lvl.stride * sizeof(float);
    bool const tiled = tiled_shared <= k_max_shared_bytes;
    note_plane(tiled, tiled ? tiled_shared : 2UL * lvl.stride * sizeof(float));
    uint32_t const grid_x  = tiled ? tile_count(ds_feats) : lvl.n_selected;
    uint32_t const by_rows = (max_rows + 32767) / 32768;
    uint32_t const blocks  = grid_x * n_nodes;
    uint32_t const fill =
        sm_count > 0
            ? ((static_cast<uint32_t>(sm_count) * k_fill_blocks_per_sm) + blocks - 1) /
                  blocks
            : 1;
    uint32_t const n_chunks = std::clamp<uint32_t>(std::max(by_rows, fill), 1, 64);
    if (tiled)
    {
        dim3 const grid(grid_x, n_nodes, n_chunks);
        data.dispatch_bins(
            [&](auto const *bins)
            {
                hist_tile_kernel<k_bin_tile_width><<<grid, dim3(256), tiled_shared>>>(
                    bins, gh, rows, offsets, counts, lvl.sel_slot.device(),
                    data.n_bins_ptr(), ds_rows, ds_feats, lvl.n_selected, out,
                    lvl.stride, slots);
            });
        return;
    }
    dim3 const grid(grid_x, n_nodes, n_chunks);
    data.dispatch_bins(
        [&](auto const *bins)
        {
            hist_kernel<<<grid, dim3(256), 2UL * lvl.stride * sizeof(float)>>>(
                bins, gh, rows, offsets, counts, lvl.features.device(),
                data.n_bins_ptr(), ds_rows, ds_feats, lvl.n_selected, out, lvl.stride,
                slots);
        });
}

void CudaDeviceContext::ensure_dataset(Dataset const &dataset)
{
    if (auto const &receipt = dataset.ingest_plane();
        receipt && receipt->backend_tag() == cuda_backend_tag())
    {
        auto plane = std::static_pointer_cast<CudaIngestPlane const>(receipt);
        DatasetKey const key{.store   = &dataset.store(),
                             .bins0   = plane.get(),
                             .n_rows  = plane->n_rows,
                             .n_feats = plane->n_feats};
        if (data.adopted == plane && data.key == key)
        {
            return;
        }
        data.adopted           = std::move(plane);
        data.bins_are_u8       = data.adopted->bins_are_u8;
        data.key               = key;
        lvl.root_rows_cached_n = 0;
        return;
    }
    data.adopted      = nullptr;
    void const *first = dataset.n_features() > 0
                            ? dataset.visit_bins(0, [](auto bins) -> void const *
                                                 { return bins.data(); })
                            : nullptr;
    if (data.key == DatasetKey{.store   = &dataset.store(),
                               .bins0   = first,
                               .n_rows  = dataset.plane_n_rows(),
                               .n_feats = dataset.n_features()})
    {
        return;
    }
    auto                  blap = prof_counters.lap();
    std::vector<uint32_t> counts(dataset.n_features());
    for (size_t f = 0; f < dataset.n_features(); ++f)
    {
        counts[f] = static_cast<uint32_t>(dataset.n_bins(f));
    }
    data.bins_are_u8       = dataset.bins_are_u8();
    lvl.root_rows_cached_n = 0;
    size_t const cells     = dataset.n_features() * dataset.plane_n_rows();
    if (data.bins_are_u8)
    {
        data.bins8.reserve(cells);
        PinnedBuffer<uint8_t> staging(cells);
        stage_tiled(dataset, staging.data());
        check(cudaMemcpy(data.bins8.data(), staging.data(), cells,
                         cudaMemcpyHostToDevice),
              "upload bins");
    }
    else
    {
        data.bins16.reserve(cells);
        PinnedBuffer<uint16_t> staging(cells);
        stage_tiled(dataset, staging.data());
        check(cudaMemcpy(data.bins16.data(), staging.data(), cells * sizeof(uint16_t),
                         cudaMemcpyHostToDevice),
              "upload bins");
    }
    data.n_bins.upload(counts.data(), counts.size());
    blap(prof_counters.bins_upload_s);
    data.key = {.store   = &dataset.store(),
                .bins0   = first,
                .n_rows  = dataset.plane_n_rows(),
                .n_feats = dataset.n_features()};
}

void CudaDeviceContext::begin_tree(Dataset const &ds, floats_view grad,
                                   floats_view hess)
{
    ensure_dataset(ds);
    if (resident.armed)
    {
        auto       lap = prof_counters.lap();
        auto const n   = static_cast<uint32_t>(resident.n_rows);
        grads.gh.reserve(resident.n_rows);
        gh_from_scores(resident.kind, resident.weighted, resident.scores.data(),
                       resident.labels.data(),
                       resident.weighted ? resident.weights.data() : nullptr, n,
                       grads.gh.data());
        lap(prof_counters.obj_kernel_s);
        return;
    }
    auto       lap = prof_counters.lap();
    auto const n   = static_cast<uint32_t>(grad.size());
    grads.grad_raw.upload(grad.data(), grad.size());
    grads.hess_raw.upload(hess.data(), hess.size());
    grads.gh.reserve(grad.size());
    interleave(grads.grad_raw.data(), grads.hess_raw.data(), n, grads.gh.data());
    lap(prof_counters.gh_upload_s);
}

uint32_t CudaDeviceContext::stage_root_rows(SplitInput const &root, bool identity)
{
    auto const n = static_cast<uint32_t>(identity ? root.row_count : root.rows.size());
    lvl.rows.reserve(n);
    if (static_cast<size_t>(n) == data.key.n_rows &&
        lvl.root_rows_cached(data.key.n_rows))
    {
        check(cudaMemcpy(lvl.rows.data(), lvl.root_rows.data(),
                         static_cast<size_t>(n) * sizeof(uint32_t),
                         cudaMemcpyDeviceToDevice),
              "root rows restore");
        return n;
    }
    if (identity)
    {
        iota_kernel<<<dim3((n + 255) / 256), dim3(256)>>>(lvl.rows.data(), n);
        check(cudaGetLastError(), "iota launch");
    }
    else
    {
        lvl.rows.upload(root.rows.data(), root.rows.size());
    }
    if (static_cast<size_t>(n) == data.key.n_rows)
    {
        lvl.root_rows.reserve(n);
        check(cudaMemcpy(lvl.root_rows.data(), lvl.rows.data(),
                         static_cast<size_t>(n) * sizeof(uint32_t),
                         cudaMemcpyDeviceToDevice),
              "root rows cache");
        lvl.root_rows_cached_n = data.key.n_rows;
    }
    return n;
}

void CudaDeviceContext::begin_root(Dataset const &ds, floats_view grad,
                                   floats_view hess, SplitInput &root,
                                   std::span<feature_id_t const> selected)
{
    init_shared_limit();
    require_hist_fits(stage_selection(ds, selected));

    bool const identity = root.rows.empty() && root.row_count == data.key.n_rows;
    auto const n = static_cast<uint32_t>(identity ? root.row_count : root.rows.size());

    auto root_lap = prof_counters.lap();
    lvl.cur_is_a  = true;
    lvl.cur().reserve(lvl.slot_doubles());
    check(cudaMemset(lvl.cur().data(), 0, lvl.slot_doubles() * sizeof(double)),
          "zero root slot");
    stage_root_rows(root, identity);
    lvl.row_offsets.host.assign(1, 0);
    lvl.row_offsets.sync();
    lvl.row_counts.host.assign(1, n);
    lvl.row_counts.sync();
    lvl.slots.host.assign(1, 0);
    lvl.slots.sync();
    root_lap(prof_counters.root_stage_s);
    if (identity)
    {
        launch_root_sums(grads.gh.data(), n);
    }
    lvl.gh_ordered.reserve(n);
    gather(grads.gh.data(), lvl.rows.data(), n, lvl.gh_ordered.data());
    if (prof_counters.enabled)
    {
        lvl.prof_record_begin(/*root=*/true);
        check(cudaEventRecord(lvl.prof_ev[LevelPipeline::ev_after_memset]),
              "profile event record");
    }
    launch_hist(static_cast<uint32_t>(ds.plane_n_rows()),
                static_cast<uint32_t>(ds.n_features()), 1, static_cast<uint32_t>(n),
                lvl.gh_ordered.data(), lvl.rows.data(), lvl.row_offsets.device(),
                lvl.row_counts.device(), lvl.cur().data(), lvl.slots.device());
    check(cudaGetLastError(), "root hist launch");
    if (prof_counters.enabled)
    {
        check(cudaEventRecord(lvl.prof_ev[LevelPipeline::ev_after_hist]),
              "profile event record");
        lvl.prof_ev_recorded = true;
    }

    lvl.slot_offsets.assign(1, 0);
    lvl.slot_counts.assign(1, n);
    lvl.leaf_by_row.reserve(ds.plane_n_rows());

    auto sums_lap = prof_counters.lap();
    if (identity)
    {
        root.sums      = fetch_root_sums();
        root.row_count = n;
    }
    else if (resident.armed)
    {
        launch_root_sums(lvl.gh_ordered.data(), n);
        root.sums      = fetch_root_sums();
        root.row_count = n;
    }
    else
    {
        double sg = 0.0;
        double sh = 0.0;
        for (row_id_t const r : root.rows)
        {
            sg += grad[r];
            sh += hess[r];
        }
        root.sums      = {.sum_grad = static_cast<float>(sg),
                          .sum_hess = static_cast<float>(sh)};
        root.row_count = root.rows.size();
    }
    sums_lap(prof_counters.root_sums_s);
    prof_counters.node_launched();
}

void CudaDeviceContext::launch_stamp(
    std::span<CudaHistogramEngine::LeafStamp const> stamps,
    std::span<uint32_t const> slot_offsets, std::span<uint32_t const> slot_counts,
    uint32_t const *rows, char const *label)
{
    if (stamps.empty())
    {
        return;
    }
    auto stamp_lap = prof_counters.lap();
    lvl.part_ops.clear();
    lvl.stamp_ids.clear();
    for (CudaHistogramEngine::LeafStamp const &st : stamps)
    {
        lvl.part_ops.host.push_back(
            {slot_offsets[st.slot], slot_counts[st.slot], 0, 0, 0});
        lvl.stamp_ids.host.push_back(st.node_id);
    }
    lvl.part_ops.sync();
    lvl.stamp_ids.sync();
    stamp_kernel<<<dim3(static_cast<uint32_t>(stamps.size())), dim3(256)>>>(
        rows, lvl.part_ops.device(), lvl.stamp_ids.device(), lvl.leaf_by_row.data());
    check(cudaGetLastError(), label);
    stamp_lap(prof_counters.fin_stamp_s);
}

void CudaDeviceContext::stamp_leaves(
    std::span<CudaHistogramEngine::LeafStamp const> stamps)
{
    launch_stamp(stamps, lvl.slot_offsets, lvl.slot_counts, lvl.cur_rows().data(),
                 "stamp launch");
}

void CudaDeviceContext::partition_level(
    Dataset const & /*ds*/, std::span<CudaHistogramEngine::PartitionOp const> ops,
    std::span<uint32_t> child_counts)
{
    if (ops.empty())
    {
        lvl.next_offsets.clear();
        lvl.next_counts.clear();
        return;
    }
    auto &prof = prof_counters;
    auto  lap  = prof.lap();

    size_t const n       = ops.size();
    uint32_t     max_cnt = 0;
    lvl.part_ops.clear();
    for (CudaHistogramEngine::PartitionOp const &op : ops)
    {
        uint32_t const count = lvl.slot_counts[op.parent_slot];
        lvl.part_ops.host.push_back({lvl.slot_offsets[op.parent_slot], count,
                                     op.feature_id, op.bin_id,
                                     op.default_left ? 1U : 0U});
        max_cnt = std::max(max_cnt, count);
    }
    lvl.part_ops.sync();
    uint32_t const max_chunks =
        std::max(1U, (max_cnt + k_part_chunk - 1) / k_part_chunk);
    lvl.flags.reserve(data.key.n_rows);
    lvl.block_counts.reserve(n * max_chunks);
    lvl.nl_dev.reserve(n);
    lap(prof.part_stage_s);

    if (prof.enabled && !lvl.part_ev_ready)
    {
        for (auto &e : lvl.part_ev)
        {
            check(cudaEventCreate(&e), "part event create");
        }
        lvl.part_ev_ready = true;
    }
    auto const mark = [&](int i)
    {
        if (prof.enabled)
        {
            check(cudaEventRecord(lvl.part_ev[i]), "part event record");
        }
    };
    mark(0);
    dim3 const grid(max_chunks, static_cast<uint32_t>(n));
    data.dispatch_bins(
        [&](auto const *bins)
        {
            route_count_kernel<<<grid, dim3(k_part_block)>>>(
                bins, data.n_bins_ptr(), lvl.cur_rows().data(), lvl.part_ops.device(),
                static_cast<uint32_t>(data.key.n_rows),
                static_cast<uint32_t>(data.key.n_feats), max_chunks, lvl.flags.data(),
                lvl.block_counts.data());
        });
    check(cudaGetLastError(), "route launch");
    mark(1);
    seg_scan_kernel<<<dim3(static_cast<uint32_t>(n)), dim3(k_part_block)>>>(
        lvl.block_counts.data(), max_chunks, lvl.nl_dev.device());
    check(cudaGetLastError(), "seg scan launch");
    mark(2);
    lvl.other_rows().reserve(data.key.n_rows);
    lvl.other_gh().reserve(data.key.n_rows);
    scatter_kernel<<<grid, dim3(k_part_block)>>>(
        lvl.cur_rows().data(), lvl.cur_gh().data(), lvl.flags.data(),
        lvl.part_ops.device(), lvl.block_counts.data(), lvl.nl_dev.device(), max_chunks,
        lvl.other_rows().data(), lvl.other_gh().data());
    check(cudaGetLastError(), "scatter launch");
    mark(3);
    lvl.nl_dev.fetch(n);
    if (prof.enabled)
    {
        double *const spans[3] = {&prof.part_route_s, &prof.part_scan_s,
                                  &prof.part_scatter_s};
        for (int i = 0; i < 3; ++i)
        {
            float ms = 0.0F;
            check(cudaEventElapsedTime(&ms, lvl.part_ev[i], lvl.part_ev[i + 1]),
                  "part event elapsed");
            *spans[i] += ms / 1e3;
        }
        ++prof.launches;
    }
    lap(prof.gpu_s);

    lvl.layout_children(ops, child_counts);
}

void CudaDeviceContext::finalize_tree(std::span<float const> node_values,
                                      std::span<float>       values,
                                      std::span<node_id_t>   leaf_ids)
{
    auto       map_lap = prof_counters.lap();
    auto const n       = static_cast<uint32_t>(values.size());
    lvl.epi_node_vals.upload(node_values.data(), node_values.size());
    lvl.epi_values.reserve(values.size());
    dim3 const grid((n + 255) / 256);
    map_leaf_values_kernel<<<grid, dim3(256)>>>(
        lvl.leaf_by_row.data(), lvl.epi_node_vals.data(),
        static_cast<uint32_t>(node_values.size()), lvl.epi_values.data(), n);
    check(cudaGetLastError(), "epilogue map launch");
    map_lap(prof_counters.fin_map_s);
    auto flap = prof_counters.lap();
    if (prof_counters.enabled)
    {
        check(cudaDeviceSynchronize(), "epilogue wait");
        flap(prof_counters.fin_wait_s);
        lvl.prof_read(prof_counters);
    }
    check(cudaMemcpy(leaf_ids.data(), lvl.leaf_by_row.data(),
                     leaf_ids.size() * sizeof(node_id_t), cudaMemcpyDeviceToHost),
          "epilogue leaf ids copy");
    check(cudaMemcpy(values.data(), lvl.epi_values.data(),
                     values.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "epilogue values copy");
    flap(prof_counters.fin_d2h_s);
}

void CudaDeviceContext::advance_level(Dataset const                                &ds,
                                      std::span<CudaHistogramEngine::LevelOp const> ops)
{
    if (ops.empty())
    {
        return;
    }
    auto &prof = prof_counters;
    auto  lap  = prof.lap();

    size_t const max_rows = lvl.stage_children(ops);
    lap(prof.adv_stage_s);

    size_t const child_slots = 2 * ops.size();
    lvl.other().reserve(child_slots * lvl.slot_doubles());
    if (prof.enabled)
    {
        lvl.prof_record_begin(/*root=*/false);
    }
    check(cudaMemset(lvl.other().data(), 0,
                     child_slots * lvl.slot_doubles() * sizeof(double)),
          "zero level");
    if (prof.enabled)
    {
        check(cudaEventRecord(lvl.prof_ev[LevelPipeline::ev_after_memset]),
              "profile event record");
    }
    if (!lvl.row_offsets.empty())
    {
        launch_hist(static_cast<uint32_t>(ds.plane_n_rows()),
                    static_cast<uint32_t>(ds.n_features()),
                    static_cast<uint32_t>(lvl.row_offsets.size()),
                    static_cast<uint32_t>(max_rows), lvl.other_gh().data(),
                    lvl.other_rows().data(), lvl.row_offsets.device(),
                    lvl.row_counts.device(), lvl.other().data(), lvl.slots.device());
    }
    data.dispatch_bins(
        [&](auto const *bins)
        {
            if (!lvl.small_offsets.empty())
            {
                hist_small_kernel<<<
                    dim3(static_cast<uint32_t>(lvl.small_offsets.size())), dim3(128)>>>(
                    bins, lvl.other_gh().data(), lvl.other_rows().data(),
                    lvl.small_offsets.device(), lvl.small_counts.device(),
                    lvl.features.device(), static_cast<uint32_t>(ds.plane_n_rows()),
                    static_cast<uint32_t>(ds.n_features()), lvl.n_selected,
                    lvl.other().data(), lvl.stride, lvl.small_slots.device());
            }
        });
    check(cudaGetLastError(), "level hist launch");
    if (prof.enabled)
    {
        check(cudaEventRecord(lvl.prof_ev[LevelPipeline::ev_after_hist]),
              "profile event record");
    }
    auto const sd = static_cast<uint32_t>(lvl.slot_doubles());
    subtract_kernel<<<dim3(std::clamp<uint32_t>((sd + 255) / 256, 1, 256),
                           static_cast<uint32_t>(ops.size())),
                      dim3(256)>>>(lvl.cur().data(), lvl.other().data(),
                                   lvl.triples.device(), sd);
    check(cudaGetLastError(), "subtract launch");
    if (prof.enabled)
    {
        check(cudaEventRecord(lvl.prof_ev[LevelPipeline::ev_after_subtract]),
              "profile event record");
        lvl.prof_ev_recorded = true;
    }
    lvl.cur_is_a     = !lvl.cur_is_a;
    lvl.slot_offsets = lvl.next_offsets;
    lvl.slot_counts  = lvl.next_counts;
    if (prof.enabled)
    {
        ++prof.launches;
        prof.gpu_nodes += child_slots;
    }
    lap(prof.gpu_s);
}

void CudaDeviceContext::advance_layout_only()
{
    lvl.cur_is_a     = !lvl.cur_is_a;
    lvl.slot_offsets = lvl.next_offsets;
    lvl.slot_counts  = lvl.next_counts;
}

void CudaDeviceContext::find_splits_many(Dataset const &ds, TreeConfig const &config,
                                         std::span<SplitInput const> level,
                                         std::span<SplitOutput>      out,
                                         std::span<HistCell>         child_sums)
{
    size_t const n    = level.size();
    auto        &prof = prof_counters;
    auto         lap  = prof.lap();

    wait_for_profile(lap);
    bool const any_mask = lvl.stage_find_inputs(level, config, ds);
    lap(prof.find_stage_s);

    lvl.feat_best.reserve(n * lvl.n_selected);
    lvl.node_best.reserve(n);
    find_kernel<<<dim3(lvl.n_selected, static_cast<uint32_t>(n)), dim3(32)>>>(
        lvl.cur().data(), lvl.features.device(), data.n_bins_ptr(),
        lvl.node_sums.device(), lvl.node_bounds.device(),
        any_mask ? lvl.allowed.device() : nullptr, lvl.monotone.device(),
        lvl.n_selected, lvl.stride, config.lambda_l1, config.lambda_l2,
        config.min_child_hess, config.min_gain_to_split, lvl.feat_best.data(),
        /*hist_slot=*/nullptr);
    check(cudaGetLastError(), "find launch");
    reduce_kernel<<<dim3(static_cast<uint32_t>(n)), dim3(32)>>>(
        lvl.feat_best.data(), lvl.n_selected, lvl.node_best.device());
    check(cudaGetLastError(), "reduce launch");
    if (prof.enabled)
    {
        check(cudaDeviceSynchronize(), "find kernel wait");
        lap(prof.find_kern_s);
    }
    lvl.node_best.fetch(n);
    if (prof.enabled)
    {
        ++prof.launches;
        lap(prof.find_d2h_s);
    }
    lap(prof.gpu_s);

    lvl.unpack_splits(level, config, out, child_sums);
    lap(prof.unpack_s);
}

void CudaDeviceContext::find_level_split(Dataset const & /*ds*/,
                                         TreeConfig const           &config,
                                         std::span<SplitInput const> level,
                                         std::span<SplitOutput>      out,
                                         std::span<HistCell>         child_sums)
{
    size_t const n    = level.size();
    auto        &prof = prof_counters;
    auto         lap  = prof.lap();

    wait_for_profile(lap);
    lvl.stage_level_sums(level);
    lap(prof.lfind_stage_s);

    size_t const scratch = static_cast<size_t>(lvl.n_selected) * 2 * (lvl.stride / 2);
    lvl.level_score.reserve(scratch);
    lvl.feat_best.reserve(lvl.n_selected);
    lvl.node_best.reserve(1);
    lvl.level_child.reserve(4 * n);
    level_find_kernel<<<dim3(lvl.n_selected), dim3(32)>>>(
        lvl.cur().data(), lvl.features.device(), data.n_bins_ptr(),
        lvl.node_sums.device(), lvl.n_selected, static_cast<uint32_t>(n), lvl.stride,
        config.lambda_l1, config.lambda_l2, config.min_child_hess,
        config.min_gain_to_split, lvl.level_score.data(), lvl.feat_best.data());
    check(cudaGetLastError(), "level find launch");
    reduce_kernel<<<dim3(1), dim3(32)>>>(lvl.feat_best.data(), lvl.n_selected,
                                         lvl.node_best.device());
    check(cudaGetLastError(), "level reduce launch");
    level_child_sums_kernel<<<dim3((static_cast<uint32_t>(n) + 127) / 128),
                              dim3(128)>>>(
        lvl.cur().data(), lvl.node_sums.device(), lvl.node_best.device(),
        lvl.features.device(), data.n_bins_ptr(), static_cast<uint32_t>(n),
        lvl.n_selected, lvl.stride, lvl.level_child.device());
    check(cudaGetLastError(), "level child sums launch");
    lvl.node_best.fetch(1);
    lvl.level_child.fetch(4 * n);
    if (prof.enabled)
    {
        prof.launches += 3;
    }
    lap(prof.gpu_s);

    FeatBest const &b = lvl.node_best.host[0];
    SplitOutput     split{};
    if (b.valid != 0)
    {
        split = {.gain       = b.gain,
                 .feature_id = static_cast<feature_id_t>(
                     lvl.features.host[static_cast<size_t>(b.sel)]),
                 .bin_id       = static_cast<bin_id_t>(b.bin),
                 .default_left = b.dl != 0,
                 .valid        = true};
    }
    for (size_t i = 0; i < n; ++i)
    {
        out[i]            = split;
        child_sums[2 * i] = {
            .sum_grad = static_cast<float>(lvl.level_child.host[(4 * i) + 0]),
            .sum_hess = static_cast<float>(lvl.level_child.host[(4 * i) + 1])};
        child_sums[(2 * i) + 1] = {
            .sum_grad = static_cast<float>(lvl.level_child.host[(4 * i) + 2]),
            .sum_hess = static_cast<float>(lvl.level_child.host[(4 * i) + 3])};
    }
    lap(prof.unpack_s);
}

bool CudaDeviceContext::leaf_pool_ok(size_t bytes) const
{
    size_t free_bytes = 0;
    size_t total      = 0;
    if (cudaMemGetInfo(&free_bytes, &total) != cudaSuccess)
    {
        return false;
    }
    return bytes <= free_bytes / 4;
}

namespace
{

size_t leaf_max_slots(TreeConfig const &config)
{
    if (config.max_leaves != 0)
    {
        return config.max_leaves;
    }
    return config.max_depth < 31 ? (size_t{1} << config.max_depth)
                                 : static_cast<size_t>(-1);
}

size_t widest_bins(Dataset const &ds)
{
    size_t max_bins = 0;
    for (size_t f = 0; f < ds.n_features(); ++f)
    {
        max_bins = std::max(max_bins, ds.n_bins(f));
    }
    return max_bins;
}

} // namespace

bool CudaDeviceContext::leaf_budget_ok(TreeConfig const &config, size_t n_selected,
                                       size_t max_bins) const
{
    if (n_selected == 0 || !hist_budget_ok(max_bins))
    {
        return false;
    }
    size_t const max_slots    = leaf_max_slots(config);
    size_t const slot_doubles = n_selected * 2 * max_bins;
    if (max_slots > static_cast<size_t>(-1) / std::max<size_t>(1, slot_doubles))
    {
        return false;
    }
    return leaf_pool_ok(max_slots * slot_doubles * sizeof(double));
}

void CudaDeviceContext::leaf_begin_root(Dataset const &ds, TreeConfig const &config,
                                        floats_view /*grad*/, floats_view /*hess*/,
                                        SplitInput                   &root,
                                        std::span<feature_id_t const> selected)
{
    init_shared_limit();
    size_t const max_slots    = leaf_max_slots(config);
    size_t const max_sel_bins = stage_selection(ds, selected);
    if (!resident.armed)
    {
        require_hist_fits(max_sel_bins);
        if (!leaf_budget_ok(config, selected.size(), max_sel_bins))
        {
            refuse_leaf_pool(max_slots, selected.size(), max_sel_bins);
        }
    }
    leaf.monotone.host.resize(ds.n_features());
    for (feature_id_t f = 0; f < ds.n_features(); ++f)
    {
        leaf.monotone.host[f] = monotone_constraint_of(config, f);
    }
    leaf.monotone.sync();

    auto root_lap  = prof_counters.lap();
    lvl.cur_is_a   = true;
    leaf.max_slots = static_cast<uint32_t>(max_slots);
    leaf.next_slot = 1;
    leaf.slot_offsets.assign(max_slots, 0);
    leaf.slot_counts.assign(max_slots, 0);
    leaf.pool.reserve(max_slots * lvl.slot_doubles());
    check(cudaMemset(leaf.pool.data(), 0,
                     max_slots * lvl.slot_doubles() * sizeof(double)),
          "zero leaf pool");

    bool const     identity = root.rows.empty() && root.row_count == data.key.n_rows;
    uint32_t const n        = stage_root_rows(root, identity);
    leaf.slot_counts[0]     = n;
    leaf.build_seg.reserve(3);
    leaf.build_seg.host()[0] = 0;
    leaf.build_seg.host()[1] = n;
    leaf.build_seg.host()[2] = 0;
    leaf.build_seg.sync(3);
    root_lap(prof_counters.root_stage_s);

    lvl.gh_ordered.reserve(n);
    gather(grads.gh.data(), lvl.rows.data(), n, lvl.gh_ordered.data());
    launch_root_sums(lvl.gh_ordered.data(), n);

    launch_hist(static_cast<uint32_t>(ds.plane_n_rows()),
                static_cast<uint32_t>(ds.n_features()), 1, static_cast<uint32_t>(n),
                lvl.gh_ordered.data(), lvl.rows.data(), leaf.build_seg.device(),
                leaf.build_seg.device() + 1, leaf.pool.data(),
                leaf.build_seg.device() + 2);
    check(cudaGetLastError(), "leaf root hist launch");
    lvl.leaf_by_row.reserve(ds.plane_n_rows());

    auto sums_lap  = prof_counters.lap();
    root.sums      = fetch_root_sums();
    root.row_count = n;
    sums_lap(prof_counters.root_sums_s);
    prof_counters.node_launched();
}

CudaHistogramEngine::LeafRound
CudaDeviceContext::leaf_split(Dataset const & /*ds*/,
                              CudaHistogramEngine::LeafPartOp const &op)
{
    auto &prof = prof_counters;
    auto  lap  = prof.lap();

    uint32_t const offset = leaf.slot_offsets[op.parent_slot];
    uint32_t const count  = leaf.slot_counts[op.parent_slot];
    leaf.part_op.reserve(1);
    *leaf.part_op.host() = {offset, count, op.feature_id, op.bin_id,
                            op.default_left ? 1U : 0U};
    leaf.part_op.sync(1);
    uint32_t const max_chunks = std::max(1U, (count + k_part_chunk - 1) / k_part_chunk);
    lvl.flags.reserve(data.key.n_rows);
    lvl.block_counts.reserve(max_chunks);
    lvl.nl_dev.reserve(1);
    lap(prof.part_stage_s);

    dim3 const grid(max_chunks, 1);
    data.dispatch_bins(
        [&](auto const *bins)
        {
            route_count_kernel<<<grid, dim3(k_part_block)>>>(
                bins, data.n_bins_ptr(), lvl.rows.data(), leaf.part_op.device(),
                static_cast<uint32_t>(data.key.n_rows),
                static_cast<uint32_t>(data.key.n_feats), max_chunks, lvl.flags.data(),
                lvl.block_counts.data());
        });
    check(cudaGetLastError(), "leaf route launch");
    seg_scan_kernel<<<dim3(1), dim3(k_part_block)>>>(lvl.block_counts.data(),
                                                     max_chunks, lvl.nl_dev.device());
    check(cudaGetLastError(), "leaf seg scan launch");
    lvl.rows_b.reserve(data.key.n_rows);
    lvl.gh_b.reserve(data.key.n_rows);
    scatter_kernel<<<grid, dim3(k_part_block)>>>(
        lvl.rows.data(), lvl.gh_ordered.data(), lvl.flags.data(), leaf.part_op.device(),
        lvl.block_counts.data(), lvl.nl_dev.device(), max_chunks, lvl.rows_b.data(),
        lvl.gh_b.data());
    check(cudaGetLastError(), "leaf scatter launch");
    check(cudaMemcpyAsync(lvl.rows.data() + offset, lvl.rows_b.data() + offset,
                          static_cast<size_t>(count) * sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice),
          "leaf rows copy-back");
    check(cudaMemcpyAsync(lvl.gh_ordered.data() + offset, lvl.gh_b.data() + offset,
                          static_cast<size_t>(count) * sizeof(float2),
                          cudaMemcpyDeviceToDevice),
          "leaf gh copy-back");
    lvl.nl_dev.fetch(1);
    if (prof.enabled)
    {
        ++prof.launches;
    }
    lap(prof.gpu_s);

    uint32_t const                 nl = lvl.nl_dev.host[0];
    CudaHistogramEngine::LeafRound round{};
    round.left_count  = nl;
    round.right_count = count - nl;
    if (nl == 0 || nl == count)
    {
        return round;
    }
    if (leaf.next_slot >= leaf.max_slots)
    {
        throw std::runtime_error("cuda: leaf histogram pool exhausted");
    }
    bool const     left_small           = round.left_count <= round.right_count;
    uint32_t const fresh                = leaf.next_slot++;
    round.left_slot                     = left_small ? fresh : op.parent_slot;
    round.right_slot                    = left_small ? op.parent_slot : fresh;
    leaf.slot_offsets[round.left_slot]  = offset;
    leaf.slot_counts[round.left_slot]   = round.left_count;
    leaf.slot_offsets[round.right_slot] = offset + nl;
    leaf.slot_counts[round.right_slot]  = round.right_count;
    return round;
}

void CudaDeviceContext::leaf_build(Dataset const &ds, uint32_t small_slot,
                                   uint32_t large_slot)
{
    auto &prof = prof_counters;
    auto  lap  = prof.lap();

    uint32_t const small_offset = leaf.slot_offsets[small_slot];
    uint32_t const small_count  = leaf.slot_counts[small_slot];

    leaf.build_seg.reserve(3);
    leaf.build_seg.host()[0] = small_offset;
    leaf.build_seg.host()[1] = small_count;
    leaf.build_seg.host()[2] = small_slot;
    leaf.build_seg.sync(3);
    lap(prof.adv_stage_s);

    if (small_count >= k_min_gpu_rows)
    {
        launch_hist(static_cast<uint32_t>(ds.plane_n_rows()),
                    static_cast<uint32_t>(ds.n_features()), 1, small_count,
                    lvl.gh_ordered.data(), lvl.rows.data(), leaf.build_seg.device(),
                    leaf.build_seg.device() + 1, leaf.pool.data(),
                    leaf.build_seg.device() + 2);
    }
    data.dispatch_bins(
        [&](auto const *bins)
        {
            if (small_count < k_min_gpu_rows)
            {
                hist_small_kernel<<<dim3(1), dim3(128)>>>(
                    bins, lvl.gh_ordered.data(), lvl.rows.data(),
                    leaf.build_seg.device(), leaf.build_seg.device() + 1,
                    lvl.features.device(), static_cast<uint32_t>(ds.plane_n_rows()),
                    static_cast<uint32_t>(ds.n_features()), lvl.n_selected,
                    leaf.pool.data(), lvl.stride, leaf.build_seg.device() + 2);
            }
        });
    check(cudaGetLastError(), "leaf hist launch");
    auto const sd = static_cast<uint32_t>(lvl.slot_doubles());
    subtract_inplace_kernel<<<dim3(std::clamp<uint32_t>((sd + 255) / 256, 1, 256)),
                              dim3(256)>>>(leaf.pool.data(), large_slot, small_slot,
                                           sd);
    check(cudaGetLastError(), "leaf subtract launch");
    if (prof.enabled)
    {
        ++prof.launches;
        prof.gpu_nodes += 2;
    }
    lap(prof.gpu_s);
}

bool CudaDeviceContext::leaf_stage_find(std::span<SplitInput const> nodes,
                                        std::span<uint32_t const>   slots)
{
    size_t const n = nodes.size();
    leaf.find_stats.reserve(4 * n);
    leaf.find_slots.reserve(n);
    double *stats    = leaf.find_stats.host();
    bool    any_mask = false;
    for (size_t i = 0; i < n; ++i)
    {
        stats[2 * i]                 = nodes[i].sums.sum_grad;
        stats[(2 * i) + 1]           = nodes[i].sums.sum_hess;
        stats[(2 * n) + (2 * i)]     = nodes[i].lo;
        stats[(2 * n) + (2 * i) + 1] = nodes[i].hi;
        leaf.find_slots.host()[i]    = slots[i];
        any_mask                     = any_mask || !nodes[i].allowed.empty();
    }
    leaf.find_stats.sync(4 * n);
    leaf.find_slots.sync(n);
    if (any_mask)
    {
        lvl.allowed.host.resize(n * lvl.n_selected);
        for (size_t i = 0; i < n; ++i)
        {
            for (uint32_t s = 0; s < lvl.n_selected; ++s)
            {
                lvl.allowed.host[(i * lvl.n_selected) + s] =
                    nodes[i].allowed.empty() ? char{1}
                                             : nodes[i].allowed[lvl.features.host[s]];
            }
        }
        lvl.allowed.sync();
    }
    return any_mask;
}

void CudaDeviceContext::leaf_find(Dataset const & /*ds*/, TreeConfig const &config,
                                  std::span<SplitInput const> nodes,
                                  std::span<uint32_t const>   slots,
                                  std::span<SplitOutput>      out,
                                  std::span<HistCell>         child_sums)
{
    auto const n    = static_cast<uint32_t>(nodes.size());
    auto      &prof = prof_counters;
    auto       lap  = prof.lap();

    wait_for_profile(lap);
    bool const any_mask = leaf_stage_find(nodes, slots);
    lap(prof.find_stage_s);

    lvl.feat_best.reserve(static_cast<size_t>(n) * lvl.n_selected);
    lvl.node_best.reserve(n);
    find_kernel<<<dim3(lvl.n_selected, n), dim3(32)>>>(
        leaf.pool.data(), lvl.features.device(), data.n_bins_ptr(),
        leaf.find_stats.device(), leaf.find_stats.device() + (2 * n),
        any_mask ? lvl.allowed.device() : nullptr, leaf.monotone.device(),
        lvl.n_selected, lvl.stride, config.lambda_l1, config.lambda_l2,
        config.min_child_hess, config.min_gain_to_split, lvl.feat_best.data(),
        leaf.find_slots.device());
    check(cudaGetLastError(), "leaf find launch");
    reduce_kernel<<<dim3(n), dim3(32)>>>(lvl.feat_best.data(), lvl.n_selected,
                                         lvl.node_best.device());
    check(cudaGetLastError(), "leaf reduce launch");
    lvl.node_best.fetch(n);
    if (prof.enabled)
    {
        ++prof.launches;
    }
    lap(prof.gpu_s);

    lvl.unpack_splits(nodes, config, out, child_sums);
    lap(prof.unpack_s);
}

void CudaDeviceContext::leaf_stamp(
    std::span<CudaHistogramEngine::LeafStamp const> stamps)
{
    launch_stamp(stamps, leaf.slot_offsets, leaf.slot_counts, lvl.rows.data(),
                 "leaf stamp launch");
}

bool CudaDeviceContext::resident_begin(Dataset const &ds, DeviceObjectiveKind kind,
                                       std::span<float const> initial_scores,
                                       float                  learning_rate)
{
    if (kind != DeviceObjectiveKind::mse && kind != DeviceObjectiveKind::logloss &&
        kind != DeviceObjectiveKind::poisson)
    {
        return false;
    }
    ensure_dataset(ds);
    init_shared_limit();
    if (ds.n_features() == 0 || initial_scores.size() != ds.plane_n_rows() ||
        !hist_budget_ok(widest_bins(ds)))
    {
        return false;
    }
    if (resident.labels_key != ds.labels_identity())
    {
        resident.labels.upload(ds.labels().data(), ds.labels().size());
        if (!ds.weights().empty())
        {
            resident.weights.upload(ds.weights().data(), ds.weights().size());
        }
        resident.labels_key = ds.labels_identity();
    }
    resident.scores.upload(initial_scores.data(), initial_scores.size());
    resident.kind          = kind;
    resident.weighted      = !ds.weights().empty();
    resident.n_rows        = ds.plane_n_rows();
    resident.learning_rate = learning_rate;
    resident.rows.stage(ds.row_view());
    resident.armed = true;
    return true;
}

bool CudaDeviceContext::resident_begin_leaf(Dataset const &ds, TreeConfig const &config,
                                            DeviceObjectiveKind    kind,
                                            std::span<float const> initial_scores,
                                            float                  learning_rate)
{
    ensure_dataset(ds);
    init_shared_limit();
    if (ds.n_features() == 0 ||
        !leaf_budget_ok(config, ds.n_features(), widest_bins(ds)))
    {
        return false;
    }
    return resident_begin(ds, kind, initial_scores, learning_rate);
}

void CudaDeviceContext::NodeTable::stage(
    std::span<CudaHistogramEngine::ResidentNode const> nodes)
{
    size_t const nn = nodes.size();
    feature.host.resize(nn);
    split_bin.host.resize(nn);
    left.host.resize(nn);
    right.host.resize(nn);
    default_left.host.resize(nn);
    is_leaf.host.resize(nn);
    value.host.resize(nn);
    for (size_t i = 0; i < nn; ++i)
    {
        CudaHistogramEngine::ResidentNode const &rn = nodes[i];
        feature.host[i]                             = rn.feature_id;
        split_bin.host[i]                           = rn.split_bin;
        left.host[i]                                = rn.left;
        right.host[i]                               = rn.right;
        default_left.host[i]                        = rn.default_left ? 1U : 0U;
        is_leaf.host[i]                             = rn.is_leaf ? 1U : 0U;
        value.host[i]                               = rn.value;
    }
    feature.sync();
    split_bin.sync();
    left.sync();
    right.sync();
    default_left.sync();
    is_leaf.sync();
    value.sync();
}

void CudaDeviceContext::resident_finalize(
    std::span<CudaHistogramEngine::ResidentNode const> nodes)
{
    auto lap = prof_counters.lap();
    resident.nodes.stage(nodes);

    auto const n = static_cast<uint32_t>(resident.rows.n);
    dim3 const grid((n + 255) / 256);
    data.dispatch_bins(
        [&](auto const *bins)
        {
            NodeTable const &t = resident.nodes;
            route_add_kernel<<<grid, dim3(256)>>>(
                bins, data.n_bins_ptr(), static_cast<uint32_t>(data.key.n_rows),
                static_cast<uint32_t>(data.key.n_feats), t.feature.device(),
                t.split_bin.device(), t.left.device(), t.right.device(),
                t.default_left.device(), t.is_leaf.device(), t.value.device(),
                resident.learning_rate, resident.scores.data(), n,
                resident.rows.data());
        });
    check(cudaGetLastError(), "resident route+add launch");
    lap(prof_counters.score_kernel_s);
}

void CudaDeviceContext::resident_end(std::span<float> scores_out)
{
    if (!resident.armed)
    {
        return;
    }
    check(cudaMemcpy(scores_out.data(), resident.scores.data(),
                     scores_out.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "resident scores fetch");
    resident.armed = false;
}

bool CudaDeviceContext::eval_begin(Dataset const &valid, DeviceObjectiveKind kind,
                                   std::span<float const> initial_scores)
{
    veval.armed = false;
    if (!valid.bins_are_u8() || valid.n_features() == 0 ||
        initial_scores.size() != valid.plane_n_rows())
    {
        return false;
    }
    veval.kind = valid.labels().size() == valid.plane_n_rows()
                     ? kind
                     : DeviceObjectiveKind::none;
    if (veval.kind != DeviceObjectiveKind::none)
    {
        veval.labels.upload(valid.labels().data(), valid.labels().size());
        veval.loss_partial.reserve(1024);
    }
    size_t const n_rows  = valid.plane_n_rows();
    size_t const n_feats = valid.n_features();
    // perf: Rearrange the host bins into the device plane's tile order once; the
    // per-round walk then reads the same addressing as the training plane.
    // Parallel over rows: serial, this pass costs more than the walks it
    // enables (0.5s per 134M cells measured on an EPYC draw).
    std::vector<uint8_t> tiled(n_rows * n_feats);
    parallel::for_each_index(
        n_rows,
        [&](size_t r)
        {
            for (size_t f = 0; f < n_feats; ++f)
            {
                tiled[tiled_cell(static_cast<uint32_t>(f), static_cast<uint32_t>(r),
                                 static_cast<uint32_t>(n_rows),
                                 static_cast<uint32_t>(n_feats))] =
                    static_cast<uint8_t>(valid.bin_at(f, r));
            }
        });
    std::vector<uint32_t> nb(n_feats);
    for (size_t f = 0; f < n_feats; ++f)
    {
        nb[f] = static_cast<uint32_t>(valid.n_bins(f));
    }
    veval.bins.upload(tiled.data(), tiled.size());
    veval.n_bins.upload(nb.data(), nb.size());
    veval.scores.upload(initial_scores.data(), initial_scores.size());
    veval.n_rows  = n_rows;
    veval.n_feats = n_feats;
    veval.armed   = true;
    if (prof_counters.enabled)
    {
        std::fprintf(stderr, "cuda eval plane armed: rows=%zu feats=%zu\n", n_rows,
                     n_feats);
    }
    return true;
}

std::optional<float> CudaDeviceContext::eval_accumulate(
    std::span<CudaHistogramEngine::ResidentNode const> nodes, float lr,
    std::span<float> scores_out)
{
    auto lap = prof_counters.lap();
    veval.nodes.stage(nodes);

    auto const       n = static_cast<uint32_t>(veval.n_rows);
    dim3 const       grid((n + 255) / 256);
    NodeTable const &t = veval.nodes;
    route_add_kernel<<<grid, dim3(256)>>>(
        veval.bins.data(), veval.n_bins.data(), n, static_cast<uint32_t>(veval.n_feats),
        t.feature.device(), t.split_bin.device(), t.left.device(), t.right.device(),
        t.default_left.device(), t.is_leaf.device(), t.value.device(), lr,
        veval.scores.data(), n, nullptr);
    check(cudaGetLastError(), "eval route+add launch");
    if (veval.kind != DeviceObjectiveKind::none)
    {
        uint32_t const blocks =
            eval_loss_pass1(veval.kind, veval.scores.data(), veval.labels.data(), n,
                            veval.loss_partial.device());
        veval.loss_partial.fetch(blocks);
        double total = 0.0;
        for (uint32_t b = 0; b < blocks; ++b)
        {
            total += veval.loss_partial.host[b];
        }
        lap(prof_counters.eval_kernel_s);
        return static_cast<float>(total / static_cast<double>(n));
    }
    check(cudaMemcpy(scores_out.data(), veval.scores.data(),
                     scores_out.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "eval scores fetch");
    lap(prof_counters.eval_kernel_s);
    return std::nullopt;
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

} // namespace cuda_detail
} // namespace bonsai
