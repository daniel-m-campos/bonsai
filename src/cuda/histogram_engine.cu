
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

using namespace cuda_detail;

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

bool cuda_available()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// sync: cudaSetDevice binds the calling thread only, so every entry that
// precedes device work selects again; ingest and training can run on
// different threads under the Python Dataset flow.
void cuda_select_device(uint32_t device_id)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess)
    {
        n = 0;
    }
    if (device_id == 0)
    {
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

struct CudaHistogramEngine::Impl
{
    CudaDeviceContext ctx;
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

void CudaHistogramEngine::populate(Dataset const & /*ds*/, floats_view /*grad*/,
                                   floats_view /*hess*/, SplitInput & /*split_input*/,
                                   std::span<feature_id_t const> /*selected*/)
{
    throw ConfigError("cuda: this engine has no host fill; a tree it cannot hold "
                      "is refused by begin_root");
}

void CudaHistogramEngine::begin_root(Dataset const &ds, floats_view grad,
                                     floats_view hess, SplitInput &root,
                                     std::span<feature_id_t const> selected)
{
    impl_->ctx.begin_root(ds, grad, hess, root, selected);
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
                                           std::span<NodeTotals>       child_sums)
{
    impl_->ctx.find_splits_many(ds, config, level, out, child_sums);
}

void CudaHistogramEngine::find_level_split(Dataset const &ds, TreeConfig const &config,
                                           std::span<SplitInput const> level,
                                           std::span<SplitOutput>      out,
                                           std::span<NodeTotals>       child_sums)
{
    impl_->ctx.find_level_split(ds, config, level, out, child_sums);
}

void CudaHistogramEngine::leaf_begin_root(Dataset const &ds, TreeConfig const &config,
                                          floats_view grad, floats_view hess,
                                          SplitInput                   &root,
                                          std::span<feature_id_t const> selected)
{
    impl_->ctx.leaf_begin_root(ds, config, grad, hess, root, selected);
}

CudaHistogramEngine::LeafRound CudaHistogramEngine::leaf_split(Dataset const    &ds,
                                                               LeafPartOp const &op)
{
    return impl_->ctx.leaf_split(ds, op);
}

void CudaHistogramEngine::leaf_find(Dataset const &ds, TreeConfig const &config,
                                    std::span<SplitInput const> nodes,
                                    std::span<uint32_t const>   slots,
                                    std::span<SplitOutput>      out,
                                    std::span<NodeTotals>       child_sums)
{
    impl_->ctx.leaf_find(ds, config, nodes, slots, out, child_sums);
}

void CudaHistogramEngine::leaf_stamp(std::span<LeafStamp const> stamps)
{
    impl_->ctx.leaf_stamp(stamps);
}

bool CudaHistogramEngine::resident_begin(Dataset const &ds, DeviceObjectiveKind kind,
                                         std::span<float const> initial_scores,
                                         float                  learning_rate)
{
    return impl_->ctx.resident_begin(ds, kind, initial_scores, learning_rate);
}

bool CudaHistogramEngine::resident_begin_leaf(Dataset const         &ds,
                                              TreeConfig const      &config,
                                              DeviceObjectiveKind    kind,
                                              std::span<float const> initial_scores,
                                              float                  learning_rate)
{
    return impl_->ctx.resident_begin_leaf(ds, config, kind, initial_scores,
                                          learning_rate);
}

bool CudaHistogramEngine::resident_armed() const
{
    return impl_->ctx.resident_armed();
}

void CudaHistogramEngine::resident_finalize(std::span<ResidentNode const> nodes)
{
    impl_->ctx.resident_finalize(nodes);
}

void CudaHistogramEngine::resident_end(std::span<float> scores_out)
{
    impl_->ctx.resident_end(scores_out);
}

bool CudaHistogramEngine::eval_begin(Dataset const &valid, DeviceObjectiveKind kind,
                                     std::span<float const> initial_scores)
{
    return impl_->ctx.eval_begin(valid, kind, initial_scores);
}

std::optional<float>
CudaHistogramEngine::eval_accumulate(std::span<ResidentNode const> nodes, float lr,
                                     std::span<float> scores_out)
{
    return impl_->ctx.eval_accumulate(nodes, lr, scores_out);
}

namespace
{

struct CutsTable
{
    DeviceBuffer<float>    cuts;
    DeviceBuffer<uint32_t> ofs;
};

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

bool bins_exceed_shared(size_t max_bins)
{
    size_t ceiling = k_max_shared_bytes;
    int    dev     = 0;
    int    optin   = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) ==
            cudaSuccess)
    {
        ceiling = std::max(ceiling, static_cast<size_t>(optin));
    }
    return hist_shared_bytes(max_bins) > ceiling;
}

bool ingest_would_decline(BinMappers const &mappers)
{
    size_t max_bins = 0;
    for (size_t f = 0; f < mappers.size(); ++f)
    {
        max_bins = std::max(max_bins, mappers[f].n_bins());
    }
    return bins_exceed_shared(max_bins);
}

constexpr size_t k_resident_matrix_share = 2;

bool matrix_fits_device(size_t bytes)
{
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
    {
        return false;
    }
    return bytes <= pool_reserved_free_bytes() + free_bytes / k_resident_matrix_share;
}

struct UploadedMatrix
{
    DeviceBuffer<float> raw;
    DeviceMatrix        view;
};

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
    plane->tile_w      = k_bin_tile_width;
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

constexpr size_t k_ingest_chunk_bytes = 64UL * 1024UL * 1024UL;
constexpr size_t k_ingest_ring_slots  = 3;
constexpr size_t k_ingest_copy_block  = size_t{1} << 20;

void copy_to_pinned(float const *src, float *dst, size_t cells)
{
    parallel::for_each_index((cells + k_ingest_copy_block - 1) / k_ingest_copy_block,
                             [&](size_t b)
                             {
                                 size_t const c0 = b * k_ingest_copy_block;
                                 size_t const c1 =
                                     std::min(c0 + k_ingest_copy_block, cells);
                                 std::copy_n(src + c0, c1 - c0, dst + c0);
                             });
}

struct IngestSlot
{
    PinnedBuffer<float> host;
    DeviceBuffer<float> dev;
    Stream              stream;
    StreamFence         done;
    bool                pending = false;

    explicit IngestSlot(size_t cells) : host(cells)
    {
        dev.reserve(cells);
    }
};

class IngestRing
{
  public:
    explicit IngestRing(size_t cells_per_chunk)
    {
        for (size_t i = 0; i < k_ingest_ring_slots; ++i)
        {
            slots_.push_back(std::make_unique<IngestSlot>(cells_per_chunk));
        }
        if (detail::IngestProfiler::instance().enabled)
        {
            std::println(stderr,
                         "bonsai: device ingest stages chunks of {} cells through {} "
                         "pinned slots, copy and bin overlapped",
                         cells_per_chunk, k_ingest_ring_slots);
        }
    }

    template <typename Launch>
    void stage(float const *src, size_t cells, Launch &&launch)
    {
        IngestSlot &slot = *slots_[next_];
        next_            = (next_ + 1) % slots_.size();
        if (slot.pending)
        {
            slot.done.wait();
        }
        copy_to_pinned(src, slot.host.data(), cells);
        check(cudaMemcpyAsync(slot.dev.data(), slot.host.data(), cells * sizeof(float),
                              cudaMemcpyHostToDevice, slot.stream.get()),
              "ingest raw upload");
        launch(static_cast<float const *>(slot.dev.data()), slot.stream.get());
        check(cudaGetLastError(), "ingest bin launch");
        slot.done.record(slot.stream.get());
        slot.pending = true;
    }

  private:
    std::vector<std::unique_ptr<IngestSlot>> slots_;
    size_t                                   next_ = 0;
};

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
    IngestRing ring(std::min(rows_per_chunk, n_rows) * n_feats);
    for (size_t row0 = 0; row0 < n_rows; row0 += rows_per_chunk)
    {
        auto const rows  = std::min(rows_per_chunk, n_rows - row0);
        auto const cells = static_cast<uint32_t>(rows * n_feats);
        dim3 const grid((cells + 255) / 256);
        ring.stage(&X[row0, 0], cells,
                   [&](float const *raw, cudaStream_t stream)
                   {
                       plane->with_bins(
                           [&](auto *bins)
                           {
                               bin_rows_kernel<<<grid, dim3(256), 0, stream>>>(
                                   raw, static_cast<uint32_t>(rows),
                                   static_cast<uint32_t>(row0),
                                   static_cast<uint32_t>(n_feats),
                                   static_cast<uint32_t>(n_rows), table.cuts.data(),
                                   table.ofs.data(), bins);
                           });
                   });
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
    IngestRing ring(std::min(rows_per_chunk, n_rows));
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
            dim3 const grid((n + 255) / 256);
            ring.stage(batch.features[f].data() + row0, n,
                       [&](float const *raw, cudaStream_t stream)
                       {
                           plane->with_bins(
                               [&](auto *bins)
                               {
                                   bin_col_kernel<<<grid, dim3(256), 0, stream>>>(
                                       raw, n, static_cast<uint32_t>(row0),
                                       static_cast<uint32_t>(f),
                                       static_cast<uint32_t>(n_rows),
                                       static_cast<uint32_t>(mappers.size()),
                                       table.cuts.data() + c0, n_cuts, bins);
                               });
                       });
        }
    }
    check(cudaDeviceSynchronize(), "ingest sync");
    lap(detail::IngestProfiler::instance().dbin_s);
    return plane;
}

void cuda_download(float const *src, size_t n, float *dst)
{
    check(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToHost),
          "device download");
}

std::shared_ptr<DeviceMatrix const> cuda_upload(features_view X, size_t max_bin)
{
    size_t const cells = X.extent(0) * X.extent(1);
    if (!cuda_available() || cells == 0 || bins_exceed_shared(max_bin) ||
        !matrix_fits_device(cells * sizeof(float)))
    {
        return nullptr;
    }
    detail::IngestProfiler::Lap lap;
    auto                        owner = std::make_shared<UploadedMatrix>();
    owner->raw.upload(X.data_handle(), cells);
    owner->view = DeviceMatrix{owner->raw.data(), X.extent(0), X.extent(1)};
    lap(detail::IngestProfiler::instance().upload_s);
    return std::shared_ptr<DeviceMatrix const>(owner, &owner->view);
}

std::shared_ptr<IngestPlane const> cuda_ingest_device(DeviceMatrix const &X,
                                                      BinMappers const   &mappers)
{
    detail::IngestProfiler::Lap lap;
    auto const                  n_rows  = X.n_rows;
    auto const                  n_feats = X.n_feats;
    auto                        plane   = make_ingest_plane(mappers, n_rows);
    CutsTable                   table;
    upload_cuts(mappers, table);

    size_t const rows_per_chunk =
        std::max<size_t>(1, k_ingest_chunk_bytes / (n_feats * sizeof(float)));
    for (size_t row0 = 0; row0 < n_rows; row0 += rows_per_chunk)
    {
        auto const rows  = std::min(rows_per_chunk, n_rows - row0);
        auto const cells = static_cast<uint32_t>(rows * n_feats);
        auto const chunk = X.data + (row0 * n_feats);
        dim3 const grid((cells + 255) / 256);
        plane->with_bins(
            [&](auto *bins)
            {
                bin_rows_kernel<<<grid, dim3(256)>>>(
                    chunk, static_cast<uint32_t>(rows), static_cast<uint32_t>(row0),
                    static_cast<uint32_t>(n_feats), static_cast<uint32_t>(n_rows),
                    table.cuts.data(), table.ofs.data(), bins);
            });
        check(cudaGetLastError(), "ingest bin launch");
    }
    check(cudaDeviceSynchronize(), "ingest sync");
    lap(detail::IngestProfiler::instance().dbin_s);
    return plane;
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

} // namespace bonsai
