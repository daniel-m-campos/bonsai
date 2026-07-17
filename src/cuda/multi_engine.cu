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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <exception>
#include <memory>
#include <mutex>
#include <print>
#include <span>
#include <thread>
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

// Runs fn(k) for k in [0,N) across devices concurrently: N-1 worker threads
// plus the caller thread for the last context. Each worker binds its device
// (cudaSetDevice is thread-local) before fn. The first thrown exception is
// captured and rethrown on the caller after every thread joins. N == 1 calls
// fn(0) inline so the degenerate path carries no thread overhead.
template <typename Fn>
void fan_out(std::vector<uint32_t> const &devices, size_t N, Fn &&fn)
{
    if (N == 1)
    {
        check(cudaSetDevice(static_cast<int>(devices[0])), "multi fan set");
        fn(0);
        return;
    }
    std::mutex         mtx;
    std::exception_ptr first;
    auto               worker = [&](size_t k)
    {
        try
        {
            check(cudaSetDevice(static_cast<int>(devices[k])), "multi fan set");
            fn(k);
        }
        catch (...)
        {
            std::lock_guard<std::mutex> const lock(mtx);
            if (!first)
            {
                first = std::current_exception();
            }
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(N - 1);
    for (size_t k = 0; k + 1 < N; ++k)
    {
        pool.emplace_back(worker, k);
    }
    worker(N - 1);
    for (std::thread &t : pool)
    {
        t.join();
    }
    if (first)
    {
        std::rethrow_exception(first);
    }
}

// BONSAI_MULTI_PROFILE=1 wall-clock accumulators, printed at engine
// destruction. Cheap steady_clock laps on the caller thread around each
// fan-out join; no per-device breakdown. reduce is a subphase of find.
struct MultiProfile
{
    using clock         = std::chrono::steady_clock;
    bool   enabled      = std::getenv("BONSAI_MULTI_PROFILE") != nullptr;
    double begin_tree_s = 0, begin_root_s = 0, advance_s = 0, partition_s = 0;
    double find_s = 0, reduce_s = 0, finalize_rows_s = 0, finalize_s = 0;

    struct Lap
    {
        bool              enabled;
        clock::time_point t0 = clock::now();
        void              operator()(double &sink)
        {
            if (!enabled)
            {
                return;
            }
            auto const t1 = clock::now();
            sink += std::chrono::duration<double>(t1 - t0).count();
            t0 = t1;
        }
    };
    Lap lap()
    {
        return Lap{.enabled = enabled};
    }

    MultiProfile()                                    = default;
    MultiProfile(MultiProfile const &)                = delete;
    MultiProfile &operator=(MultiProfile const &)     = delete;
    MultiProfile(MultiProfile &&) noexcept            = delete;
    MultiProfile &operator=(MultiProfile &&) noexcept = delete;

    ~MultiProfile()
    {
        if (!enabled)
        {
            return;
        }
        try
        {
            std::println(stderr,
                         "multi-profile: begin_tree={:.3f}s begin_root={:.3f}s "
                         "advance={:.3f}s partition={:.3f}s find={:.3f}s "
                         "reduce={:.3f}s finalize_rows={:.3f}s finalize={:.3f}s",
                         begin_tree_s, begin_root_s, advance_s, partition_s, find_s,
                         reduce_s, finalize_rows_s, finalize_s);
        }
        catch (...)
        {
            std::fputs("multi-profile: failed to format profile line\n", stderr);
        }
    }
};

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

    // begin_root shard cache. The identity shards depend only on (total, N),
    // so an unchanged full-data fit reuses the row vectors and their per-context
    // SplitInputs instead of rebuilding N of them every tree. si holds the
    // per-context begin_root inputs; only the scalar/allowed fields refresh per
    // tree, the row vectors stay put on reuse.
    std::vector<SplitInput> si;
    size_t                  cached_total    = 0;
    bool                    cached_identity = false;

    MultiProfile prof;

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
    auto         lap = impl_->prof.lap();
    size_t const N   = impl_->n();
    // The full gradient stream is the same 51GB-per-fit upload on every device.
    // When the last begin_root left a contiguous identity sharding, each shard
    // needs only its own [lo, hi) of that stream: gh stays full-length so
    // begin_root's gather still indexes it by global row id, and the shard's
    // row list only ever gathers rows in [lo, hi). N == 1, a sampled sharding
    // (arbitrary global rows per shard), and the pre-first-root state upload
    // the whole stream. The sampler is fixed for a fit, so a contiguous flag
    // set by the previous tree's begin_root still describes this tree.
    if (N == 1 || !impl_->contiguous)
    {
        fan_out(impl_->devices, N,
                [&](size_t k) { impl_->ctxs[k]->begin_tree(ds, grad, hess); });
    }
    else
    {
        size_t const total = grad.size();
        fan_out(impl_->devices, N,
                [&](size_t k)
                {
                    size_t const lo = (total * k) / N;
                    size_t const hi = (total * (k + 1)) / N;
                    impl_->ctxs[k]->begin_tree(ds, grad, hess, lo, hi - lo);
                });
    }
    lap(impl_->prof.begin_tree_s);
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
    // bring-up). A sampled row vector slices directly. An identity fit whose
    // boundaries and total match the previous tree reuses the cached shard row
    // vectors and their SplitInputs; only sampled fits (or a changed total)
    // rebuild them.
    bool const   identity = root.rows.empty();
    size_t const total    = identity ? root.row_count : root.rows.size();
    impl_->contiguous     = identity;
    bool const reuse      = identity && impl_->cached_identity &&
                       impl_->cached_total == total && impl_->shards.size() == N &&
                       impl_->si.size() == N;
    if (!reuse)
    {
        impl_->shards.assign(N, {});
        impl_->si.assign(N, {});
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
            impl_->si[k].rows = sh.rows;
        }
        impl_->cached_identity = identity;
        impl_->cached_total    = total;
    }

    // Per-shard SplitInputs; the row vectors are cached above, so only the
    // per-tree scalar and allowed-mask fields refresh here. Each thread reads
    // and fills its own so the begin_root calls fan out without sharing state.
    std::vector<SplitInput> &si = impl_->si;
    for (size_t k = 0; k < N; ++k)
    {
        si[k].id      = root.id;
        si[k].lo      = root.lo;
        si[k].hi      = root.hi;
        si[k].allowed = root.allowed;
    }
    std::vector<char> ok(N, 1);
    auto              lap = impl_->prof.lap();
    fan_out(impl_->devices, N,
            [&](size_t k)
            {
                ok[k] = impl_->ctxs[k]->begin_root(ds, grad, hess, si[k], selected)
                            ? char{1}
                            : char{0};
            });
    lap(impl_->prof.begin_root_s);

    // Any decline drops the whole engine to host fallback.
    for (size_t k = 0; k < N; ++k)
    {
        if (ok[k] == 0)
        {
            return false;
        }
    }
    double grad_sum    = 0.0;
    double hess_sum    = 0.0;
    size_t total_count = 0;
    for (size_t k = 0; k < N; ++k)
    {
        grad_sum += si[k].sums.sum_grad;
        hess_sum += si[k].sums.sum_hess;
        total_count += si[k].row_count;
    }
    root.sums      = {.sum_grad = static_cast<float>(grad_sum),
                      .sum_hess = static_cast<float>(hess_sum)};
    root.row_count = total_count;
    return true;
}

void MultiCudaHistogramEngine::stamp_leaves(std::span<LeafStamp const> stamps)
{
    fan_out(impl_->devices, impl_->n(),
            [&](size_t k) { impl_->ctxs[k]->stamp_leaves(stamps); });
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
    // Per-shard counts buffers so the fan-out writes never collide; summed on
    // the caller after join.
    std::vector<std::vector<uint32_t>> locals(
        N, std::vector<uint32_t>(child_counts.size()));
    auto lap = impl_->prof.lap();
    fan_out(impl_->devices, N,
            [&](size_t k) { impl_->ctxs[k]->partition_level(ds, ops, locals[k]); });
    lap(impl_->prof.partition_s);
    for (size_t k = 0; k < N; ++k)
    {
        for (size_t i = 0; i < child_counts.size(); ++i)
        {
            child_counts[i] += locals[k][i];
        }
    }
}

void MultiCudaHistogramEngine::advance_level(Dataset const           &ds,
                                             std::span<LevelOp const> ops)
{
    auto lap = impl_->prof.lap();
    fan_out(impl_->devices, impl_->n(),
            [&](size_t k) { impl_->ctxs[k]->advance_level(ds, ops); });
    lap(impl_->prof.advance_s);
}

void MultiCudaHistogramEngine::advance_layout_only()
{
    fan_out(impl_->devices, impl_->n(),
            [&](size_t k) { impl_->ctxs[k]->advance_layout_only(); });
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
    auto lap = impl_->prof.lap();
    if (impl_->contiguous)
    {
        // Disjoint [lo, hi) destinations: each context copies only its shard's
        // slice of its device leaf_by_row straight into the caller's span. No
        // host scratch, no merge; the fan-out writes never collide.
        fan_out(impl_->devices, N,
                [&](size_t k)
                {
                    Impl::Shard const &sh = impl_->shards[k];
                    check(cudaMemcpy(leaf_by_row.data() + sh.lo,
                                     impl_->ctxs[k]->lvl.leaf_by_row.data() + sh.lo,
                                     (sh.hi - sh.lo) * sizeof(node_id_t),
                                     cudaMemcpyDeviceToHost),
                          "multi fin_rows slice d2h");
                });
    }
    else
    {
        // Sampled: rows are non-contiguous, keep the per-shard scratch + merge.
        std::vector<std::vector<node_id_t>> scratch(
            N, std::vector<node_id_t>(leaf_by_row.size()));
        fan_out(impl_->devices, N,
                [&](size_t k) { impl_->ctxs[k]->finalize_rows(scratch[k]); });
        for (size_t k = 0; k < N; ++k)
        {
            impl_->merge_shard<node_id_t>(k, scratch[k], leaf_by_row);
        }
    }
    lap(impl_->prof.finalize_rows_s);
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
    auto lap = impl_->prof.lap();
    if (impl_->contiguous)
    {
        // Each context maps its device values (full range, cheap) then downloads
        // only its shard's [lo, hi) slice into the caller's spans. Disjoint
        // offsets make the shared-output writes collision-free; no host scratch.
        fan_out(impl_->devices, N,
                [&](size_t k)
                {
                    Impl::Shard const &sh = impl_->shards[k];
                    impl_->ctxs[k]->finalize_tree(node_values, values, leaf_ids, sh.lo,
                                                  sh.hi - sh.lo);
                });
    }
    else
    {
        // Sampled: rows are non-contiguous, keep the per-shard scratch + merge.
        std::vector<std::vector<float>> vscratch(N, std::vector<float>(values.size()));
        std::vector<std::vector<node_id_t>> iscratch(
            N, std::vector<node_id_t>(leaf_ids.size()));
        fan_out(
            impl_->devices, N, [&](size_t k)
            { impl_->ctxs[k]->finalize_tree(node_values, vscratch[k], iscratch[k]); });
        for (size_t k = 0; k < N; ++k)
        {
            impl_->merge_shard<float>(k, vscratch[k], values);
            impl_->merge_shard<node_id_t>(k, iscratch[k], leaf_ids);
        }
    }
    lap(impl_->prof.finalize_s);
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
    auto         find_lap   = impl_->prof.lap();
    size_t const total      = level.size() * impl_->ctxs[0]->lvl.slot_doubles();
    auto         reduce_lap = impl_->prof.lap();
    impl_->reduce_level(total);
    reduce_lap(impl_->prof.reduce_s);
    check(cudaSetDevice(static_cast<int>(impl_->devices[0])), "multi find coord set");
    impl_->ctxs[0]->find_splits_many(ds, config, level, out, child_sums,
                                     impl_->scratch.data());
    find_lap(impl_->prof.find_s);
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
    auto         find_lap   = impl_->prof.lap();
    size_t const total      = level.size() * impl_->ctxs[0]->lvl.slot_doubles();
    auto         reduce_lap = impl_->prof.lap();
    impl_->reduce_level(total);
    reduce_lap(impl_->prof.reduce_s);
    check(cudaSetDevice(static_cast<int>(impl_->devices[0])), "multi lfind coord set");
    impl_->ctxs[0]->find_level_split(ds, config, level, out, child_sums,
                                     impl_->scratch.data());
    find_lap(impl_->prof.find_s);
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

} // namespace bonsai
