// Data-parallel histogram backend: N CudaDeviceContexts, one per selected
// device, driven behind the same HistogramEngine + GPULevelEngine concepts as
// the single-GPU engine (docs/architecture/19-multi-gpu.md). Compiled as clang
// CUDA C++, same libc++/C++23 as the rest of the build. The per-device state
// and its kernels live in the shared CudaDeviceContext (detail/device_context);
// this TU adds only the fan-out, the reduce-to-coordinator, and the finalize
// merges, plus one small accumulate kernel kept anonymous and private here.

#include "bonsai/config/tree_config.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/cuda/multi_engine.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <memory>
#include <span>
#include <vector>

#include "detail/device_buffer.cuh"
#include "detail/device_context.cuh"

namespace bonsai
{

using namespace cuda_detail;

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

namespace
{

// scratch[i] += staging[i], grid-stride. One shard's partial level histogram
// folded into the coordinator's running reduction.
__global__ void accumulate_kernel(double *scratch, double const *staging, size_t n)
{
    size_t const span = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x; i < n;
         i += span)
    {
        scratch[i] += staging[i];
    }
}

} // namespace

struct MultiCudaHistogramEngine::Impl
{
    std::vector<uint32_t>                           devices; // one per context
    std::vector<std::unique_ptr<CudaDeviceContext>> ctxs;
    std::vector<char>  peer_ok; // vs coordinator (index 0); 1 = peer copy usable
    CpuHistogramEngine cpu;     // host-fallback fill when begin_root declines

    // Coordinator-owned (devices[0]) reduction buffers. scratch holds the
    // running global histogram, staging one shard's partial before it folds in.
    DeviceBuffer<double>                  scratch;
    DeviceBuffer<double>                  staging;
    std::unique_ptr<PinnedBuffer<double>> pinned; // host-staged bounce buffer
    size_t                                pinned_cap = 0;

    // Row sharding for N > 1. rows = the global row ids a context owns; a
    // contiguous identity range merges with one copy per shard, a sliced
    // sampled vector merges per row.
    struct Shard
    {
        std::vector<row_id_t> rows;
        size_t                lo = 0;
        size_t                hi = 0;
    };
    std::vector<Shard> shards;
    bool               contiguous = false;

    size_t n() const
    {
        return ctxs.size();
    }

    double *host_stage(size_t count)
    {
        if (count > pinned_cap)
        {
            pinned     = std::make_unique<PinnedBuffer<double>>(count);
            pinned_cap = count;
        }
        return pinned->data();
    }

    // Reduces every shard's live level histogram (total doubles, slot-indexed)
    // onto scratch on the coordinator device, device id order. Peer memcpy
    // where peer access is real, a pinned host bounce otherwise. A shard's
    // device is synchronized before its histogram is read: default-stream
    // ordering does not cross devices.
    void reduce_level(size_t total)
    {
        auto const coord = static_cast<int>(devices[0]);
        check(cudaSetDevice(coord), "multi reduce set coord");
        scratch.reserve(total);
        staging.reserve(total);
        check(cudaDeviceSynchronize(), "multi reduce coord sync");
        check(cudaMemcpy(scratch.data(), ctxs[0]->lvl.cur().data(),
                         total * sizeof(double), cudaMemcpyDeviceToDevice),
              "multi reduce coord copy");
        for (size_t k = 1; k < ctxs.size(); ++k)
        {
            double const *src = ctxs[k]->lvl.cur().data();
            check(cudaSetDevice(static_cast<int>(devices[k])),
                  "multi reduce set shard");
            check(cudaDeviceSynchronize(), "multi reduce shard sync");
            if (peer_ok[k] != 0)
            {
                check(cudaMemcpyPeer(staging.data(), coord, src,
                                     static_cast<int>(devices[k]),
                                     total * sizeof(double)),
                      "multi reduce peer copy");
            }
            else
            {
                double *host = host_stage(total);
                check(cudaMemcpy(host, src, total * sizeof(double),
                                 cudaMemcpyDeviceToHost),
                      "multi reduce d2h");
                check(cudaSetDevice(coord), "multi reduce set coord h2d");
                check(cudaMemcpy(staging.data(), host, total * sizeof(double),
                                 cudaMemcpyHostToDevice),
                      "multi reduce h2d");
            }
            check(cudaSetDevice(coord), "multi reduce set coord accum");
            auto const blocks = std::clamp<uint32_t>(
                static_cast<uint32_t>((total + 255) / 256), 1, 1024);
            accumulate_kernel<<<dim3(blocks), dim3(256)>>>(scratch.data(),
                                                           staging.data(), total);
            check(cudaGetLastError(), "multi accumulate launch");
            // Sync so the next shard's staging write cannot race this fold-in.
            check(cudaDeviceSynchronize(), "multi accumulate sync");
        }
    }

    // Writes shard k's owned entries from a full-length host scratch into the
    // caller's full-length span.
    template <typename T>
    void merge_shard(size_t k, std::vector<T> const &src, std::span<T> dst)
    {
        Shard const &sh = shards[k];
        if (contiguous)
        {
            std::copy(src.begin() + static_cast<std::ptrdiff_t>(sh.lo),
                      src.begin() + static_cast<std::ptrdiff_t>(sh.hi),
                      dst.begin() + static_cast<std::ptrdiff_t>(sh.lo));
        }
        else
        {
            for (row_id_t const r : sh.rows)
            {
                dst[r] = src[r];
            }
        }
    }
};

MultiCudaHistogramEngine::MultiCudaHistogramEngine() : impl_(std::make_unique<Impl>())
{
    std::vector<uint32_t> sel = cuda_selected_devices();
    if (sel.empty())
    {
        int dev = 0;
        cudaGetDevice(&dev);
        sel.assign(1, static_cast<uint32_t>(dev));
    }
    impl_->devices = sel;
    size_t const N = sel.size();
    impl_->ctxs.reserve(N);
    for (size_t k = 0; k < N; ++k)
    {
        impl_->ctxs.push_back(std::make_unique<CudaDeviceContext>());
    }

    // Peer probe vs the coordinator device; a duplicate id (same device) copies
    // as a plain D2D and needs no peer enable.
    auto const coord = static_cast<int>(sel[0]);
    impl_->peer_ok.assign(N, 0);
    for (size_t k = 0; k < N; ++k)
    {
        if (sel[k] == sel[0])
        {
            impl_->peer_ok[k] = 1;
            continue;
        }
        int c1 = 0;
        int c2 = 0;
        cudaDeviceCanAccessPeer(&c1, coord, static_cast<int>(sel[k]));
        cudaDeviceCanAccessPeer(&c2, static_cast<int>(sel[k]), coord);
        impl_->peer_ok[k] = (c1 != 0 && c2 != 0) ? 1 : 0;
    }
    // Enable peer access for every ordered pair of distinct devices where both
    // directions are reachable (the already-enabled return code is ignored).
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            if (sel[i] == sel[j])
            {
                continue;
            }
            int c1 = 0;
            int c2 = 0;
            cudaDeviceCanAccessPeer(&c1, static_cast<int>(sel[i]),
                                    static_cast<int>(sel[j]));
            cudaDeviceCanAccessPeer(&c2, static_cast<int>(sel[j]),
                                    static_cast<int>(sel[i]));
            if (c1 != 0 && c2 != 0)
            {
                cudaSetDevice(static_cast<int>(sel[i]));
                cudaDeviceEnablePeerAccess(static_cast<int>(sel[j]), 0);
                cudaGetLastError(); // clear the already-enabled sticky error
            }
        }
    }
    // Validation hook so the host-staged fallback is testable on any host.
    if (std::getenv("BONSAI_MULTI_HOST_STAGED") != nullptr)
    {
        std::fill(impl_->peer_ok.begin(), impl_->peer_ok.end(), char{0});
    }
    cudaSetDevice(coord);
}

MultiCudaHistogramEngine::~MultiCudaHistogramEngine() = default;
MultiCudaHistogramEngine::MultiCudaHistogramEngine(
    MultiCudaHistogramEngine &&) noexcept = default;
MultiCudaHistogramEngine &
MultiCudaHistogramEngine::operator=(MultiCudaHistogramEngine &&) noexcept = default;

void MultiCudaHistogramEngine::begin_tree(Dataset const &ds, floats_view grad,
                                          floats_view hess)
{
    for (size_t k = 0; k < impl_->n(); ++k)
    {
        check(cudaSetDevice(static_cast<int>(impl_->devices[k])),
              "multi begin_tree set");
        impl_->ctxs[k]->begin_tree(ds, grad, hess);
    }
}

void MultiCudaHistogramEngine::populate(Dataset const &ds, floats_view grad,
                                        floats_view hess, SplitInput &split_input,
                                        std::span<feature_id_t const> selected)
{
    // Host-plane fallback (begin_root declined), same wrapper as the single
    // engine: the shards are never touched on this path.
    impl_->cpu.populate(ds, grad, hess, split_input, selected);
}

bool MultiCudaHistogramEngine::begin_root(Dataset const &ds, floats_view grad,
                                          floats_view hess, SplitInput &root,
                                          std::span<feature_id_t const> selected)
{
    size_t const N = impl_->n();
    if (N == 1)
    {
        check(cudaSetDevice(static_cast<int>(impl_->devices[0])), "multi root set");
        return impl_->ctxs[0]->begin_root(ds, grad, hess, root, selected);
    }

    // Shard the ROOT's row space into N contiguous chunks. The identity
    // (empty rows, full row_count) becomes explicit per-shard ranges: the
    // shards cannot use begin_root's identity fast path (acceptable at
    // bring-up). A sampled row vector slices directly.
    impl_->shards.assign(N, {});
    bool const   identity = root.rows.empty();
    size_t const total    = identity ? root.row_count : root.rows.size();
    impl_->contiguous     = identity;
    for (size_t k = 0; k < N; ++k)
    {
        size_t const lo = (total * k) / N;
        size_t const hi = (total * (k + 1)) / N;
        Impl::Shard &sh = impl_->shards[k];
        sh.lo           = lo;
        sh.hi           = hi;
        sh.rows.resize(hi - lo);
        if (identity)
        {
            for (size_t r = lo; r < hi; ++r)
            {
                sh.rows[r - lo] = static_cast<row_id_t>(r);
            }
        }
        else
        {
            std::copy(root.rows.begin() + static_cast<std::ptrdiff_t>(lo),
                      root.rows.begin() + static_cast<std::ptrdiff_t>(hi),
                      sh.rows.begin());
        }
    }

    double grad_sum    = 0.0;
    double hess_sum    = 0.0;
    size_t total_count = 0;
    for (size_t k = 0; k < N; ++k)
    {
        check(cudaSetDevice(static_cast<int>(impl_->devices[k])),
              "multi root shard set");
        SplitInput shard;
        shard.id      = root.id;
        shard.lo      = root.lo;
        shard.hi      = root.hi;
        shard.allowed = root.allowed;
        shard.rows    = impl_->shards[k].rows;
        if (!impl_->ctxs[k]->begin_root(ds, grad, hess, shard, selected))
        {
            return false; // any decline drops the whole engine to host fallback
        }
        grad_sum += shard.sums.sum_grad;
        hess_sum += shard.sums.sum_hess;
        total_count += shard.row_count;
    }
    root.sums      = {.sum_grad = static_cast<float>(grad_sum),
                      .sum_hess = static_cast<float>(hess_sum)};
    root.row_count = total_count;
    return true;
}

void MultiCudaHistogramEngine::stamp_leaves(std::span<LeafStamp const> stamps)
{
    for (size_t k = 0; k < impl_->n(); ++k)
    {
        check(cudaSetDevice(static_cast<int>(impl_->devices[k])), "multi stamp set");
        impl_->ctxs[k]->stamp_leaves(stamps);
    }
}

void MultiCudaHistogramEngine::partition_level(Dataset const               &ds,
                                               std::span<PartitionOp const> ops,
                                               std::span<uint32_t> child_counts)
{
    size_t const N = impl_->n();
    if (N == 1)
    {
        check(cudaSetDevice(static_cast<int>(impl_->devices[0])), "multi part set");
        impl_->ctxs[0]->partition_level(ds, ops, child_counts);
        return;
    }
    std::fill(child_counts.begin(), child_counts.end(), uint32_t{0});
    std::vector<uint32_t> local(child_counts.size());
    for (size_t k = 0; k < N; ++k)
    {
        check(cudaSetDevice(static_cast<int>(impl_->devices[k])),
              "multi part shard set");
        std::fill(local.begin(), local.end(), uint32_t{0});
        impl_->ctxs[k]->partition_level(ds, ops, local);
        for (size_t i = 0; i < child_counts.size(); ++i)
        {
            child_counts[i] += local[i];
        }
    }
}

void MultiCudaHistogramEngine::advance_level(Dataset const           &ds,
                                             std::span<LevelOp const> ops)
{
    for (size_t k = 0; k < impl_->n(); ++k)
    {
        check(cudaSetDevice(static_cast<int>(impl_->devices[k])), "multi advance set");
        impl_->ctxs[k]->advance_level(ds, ops);
    }
}

void MultiCudaHistogramEngine::advance_layout_only()
{
    for (size_t k = 0; k < impl_->n(); ++k)
    {
        check(cudaSetDevice(static_cast<int>(impl_->devices[k])), "multi layout set");
        impl_->ctxs[k]->advance_layout_only();
    }
}

void MultiCudaHistogramEngine::finalize_rows(std::span<node_id_t> leaf_by_row)
{
    size_t const N = impl_->n();
    if (N == 1)
    {
        check(cudaSetDevice(static_cast<int>(impl_->devices[0])), "multi fin_rows set");
        impl_->ctxs[0]->finalize_rows(leaf_by_row);
        return;
    }
    std::vector<node_id_t> scratch(leaf_by_row.size());
    for (size_t k = 0; k < N; ++k)
    {
        check(cudaSetDevice(static_cast<int>(impl_->devices[k])),
              "multi fin_rows shard set");
        impl_->ctxs[k]->finalize_rows(scratch);
        impl_->merge_shard<node_id_t>(k, scratch, leaf_by_row);
    }
}

void MultiCudaHistogramEngine::finalize_tree(std::span<float const> node_values,
                                             std::span<float>       values,
                                             std::span<node_id_t>   leaf_ids)
{
    size_t const N = impl_->n();
    if (N == 1)
    {
        check(cudaSetDevice(static_cast<int>(impl_->devices[0])), "multi fin_tree set");
        impl_->ctxs[0]->finalize_tree(node_values, values, leaf_ids);
        return;
    }
    std::vector<float>     vscratch(values.size());
    std::vector<node_id_t> iscratch(leaf_ids.size());
    for (size_t k = 0; k < N; ++k)
    {
        check(cudaSetDevice(static_cast<int>(impl_->devices[k])),
              "multi fin_tree shard set");
        impl_->ctxs[k]->finalize_tree(node_values, vscratch, iscratch);
        impl_->merge_shard<float>(k, vscratch, values);
        impl_->merge_shard<node_id_t>(k, iscratch, leaf_ids);
    }
}

void MultiCudaHistogramEngine::find_splits_many(Dataset const              &ds,
                                                TreeConfig const           &config,
                                                std::span<SplitInput const> level,
                                                std::span<SplitOutput>      out,
                                                std::span<HistCell>         child_sums)
{
    size_t const N = impl_->n();
    if (N == 1)
    {
        check(cudaSetDevice(static_cast<int>(impl_->devices[0])), "multi find set");
        impl_->ctxs[0]->find_splits_many(ds, config, level, out, child_sums);
        return;
    }
    size_t const total = level.size() * impl_->ctxs[0]->lvl.slot_doubles();
    impl_->reduce_level(total);
    check(cudaSetDevice(static_cast<int>(impl_->devices[0])), "multi find coord set");
    impl_->ctxs[0]->find_splits_many(ds, config, level, out, child_sums,
                                     impl_->scratch.data());
}

void MultiCudaHistogramEngine::find_level_split(Dataset const              &ds,
                                                TreeConfig const           &config,
                                                std::span<SplitInput const> level,
                                                std::span<SplitOutput>      out,
                                                std::span<HistCell>         child_sums)
{
    size_t const N = impl_->n();
    if (N == 1)
    {
        check(cudaSetDevice(static_cast<int>(impl_->devices[0])), "multi lfind set");
        impl_->ctxs[0]->find_level_split(ds, config, level, out, child_sums);
        return;
    }
    size_t const total = level.size() * impl_->ctxs[0]->lvl.slot_doubles();
    impl_->reduce_level(total);
    check(cudaSetDevice(static_cast<int>(impl_->devices[0])), "multi lfind coord set");
    impl_->ctxs[0]->find_level_split(ds, config, level, out, child_sums,
                                     impl_->scratch.data());
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

} // namespace bonsai
