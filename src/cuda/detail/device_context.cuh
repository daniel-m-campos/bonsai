#pragma once

// Per-device state and operations shared by CudaHistogramEngine (one context)
// and, in P3, MultiCudaHistogramEngine (N contexts); see
// docs/architecture/19-multi-gpu.md. Injected into bonsai's anonymous
// namespace: this header is included by exactly the histogram_engine.cu TU.

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
#include <print>
#include <span>
#include <utility>
#include <vector>
#include <vector_types.h>

#include "device_buffer.cuh"
#include "kernels.cuh"

namespace bonsai
{
namespace
{

// Flat device/host buffers throughout this file are offset by hand (docs/
// architecture/10-cuda.md); grad/hess travel as an adjacent pair everywhere
// in this API, matching the gradient-boosting literature's convention.
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

// BONSAI_CUDA_PROFILE=1 accumulators, printed at engine destruction.
struct ProfileCounters
{
    using clock     = std::chrono::steady_clock;
    bool   enabled  = std::getenv("BONSAI_CUDA_PROFILE") != nullptr;
    double upload_s = 0, gpu_s = 0, unpack_s = 0, cpu_s = 0;
    double part_stage_s = 0, adv_stage_s = 0, find_stage_s = 0, lfind_stage_s = 0;
    double gh_upload_s = 0, root_stage_s = 0, gpu_wait_s = 0;
    double bins_upload_s = 0, fin_wait_s = 0, fin_d2h_s = 0;
    double find_kern_s = 0, find_d2h_s = 0;
    // Marginal-round decomposition (decision 71 campaign): device-side spans
    // from event pairs read at the next profile sync, plus the begin_root
    // host reduction, so every millisecond of the round has a name.
    double root_sums_s = 0, adv_memset_s = 0, adv_hist_s = 0, adv_sub_s = 0;
    double root_hist_s = 0, fin_stamp_s = 0, fin_map_s = 0;
    size_t launches = 0, gpu_nodes = 0, cpu_calls = 0;

    ProfileCounters()                                       = default;
    ProfileCounters(ProfileCounters const &)                = delete;
    ProfileCounters &operator=(ProfileCounters const &)     = delete;
    ProfileCounters(ProfileCounters &&) noexcept            = delete;
    ProfileCounters &operator=(ProfileCounters &&) noexcept = delete;

    // Running stopwatch: lap(sink) adds the time since the previous lap into
    // sink, a no-op when profiling is off. One per method call marks off its
    // upload / gpu / unpack phases against the shared accumulators.
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

    ~ProfileCounters()
    {
        if (!enabled || (gpu_s == 0 && cpu_s == 0))
        {
            return;
        }
        try
        {
            std::println(stderr,
                         "cuda-profile: upload={:.2f}s gpu={:.2f}s unpack={:.2f}s "
                         "cpu_fallback={:.2f}s | {} launches covering {} nodes, {} "
                         "cpu-fallback nodes",
                         part_stage_s + adv_stage_s + find_stage_s + lfind_stage_s,
                         gpu_s, unpack_s, cpu_s, launches, gpu_nodes, cpu_calls);
            std::println(stderr,
                         "cuda-upload-decomp: gh={:.2f}s root_stage={:.2f}s "
                         "part_stage={:.2f}s adv_stage={:.2f}s find_stage={:.2f}s "
                         "lfind_stage={:.2f}s gpu_wait={:.2f}s legacy={:.2f}s "
                         "bins_upload={:.2f}s fin_wait={:.2f}s fin_d2h={:.2f}s "
                         "find_kern={:.2f}s find_d2h={:.2f}s",
                         gh_upload_s, root_stage_s, part_stage_s, adv_stage_s,
                         find_stage_s, lfind_stage_s, gpu_wait_s, upload_s,
                         bins_upload_s, fin_wait_s, fin_d2h_s, find_kern_s, find_d2h_s);
            std::println(stderr,
                         "cuda-round-decomp: root_sums={:.2f}s root_hist={:.2f}s "
                         "adv_memset={:.2f}s adv_hist={:.2f}s adv_sub={:.2f}s "
                         "fin_stamp={:.2f}s fin_map={:.2f}s",
                         root_sums_s, root_hist_s, adv_memset_s, adv_hist_s, adv_sub_s,
                         fin_stamp_s, fin_map_s);
        }
        catch (...)
        {
            std::fputs("cuda-profile: failed to format profile line\n", stderr);
        }
    }
};

// The CUDA ingest transaction's product (decision 54, doc 15): the
// feature-major binned matrix, resident from birth. Dataset carries it as
// an opaque receipt; ensure_dataset adopts it instead of uploading host
// columns; materialize() pulls host columns home once for the host
// consumers (fallback decline, route_unsampled under row sampling).
// The problem this solves: ensure_dataset receives a shared_ptr<IngestPlane>
// (the base type) and must prove it is really a CudaIngestPlane before
// downcasting, without RTTI. Every plane carries an opaque tag pointer, and
// this function is the only source of this backend's tag (the address of a
// function-local static, unique process-wide), so tag equality proves the
// concrete type and makes the static_cast sound.
inline void const *cuda_backend_tag()
{
    static char const anchor = 0;
    return &anchor;
}

class CudaIngestPlane final : public IngestPlane
{
  public:
    CudaIngestPlane() : IngestPlane(cuda_backend_tag()) {}

    DeviceBuffer<uint8_t>  bins8;
    DeviceBuffer<uint16_t> bins16;
    DeviceBuffer<uint32_t> n_bins; // per-feature bin counts
    bool                   bins_are_u8 = false;
    size_t                 n_rows      = 0;
    size_t                 n_feats     = 0;

    void materialize(std::vector<std::vector<uint8_t>>  &u8,
                     std::vector<std::vector<uint16_t>> &u16) const override
    {
        if (bins_are_u8)
        {
            u8.resize(n_feats);
            for (size_t f = 0; f < n_feats; ++f)
            {
                u8[f].resize(n_rows);
                check(cudaMemcpy(u8[f].data(), bins8.data() + (f * n_rows), n_rows,
                                 cudaMemcpyDeviceToHost),
                      "materialize bins");
            }
            return;
        }
        u16.resize(n_feats);
        for (size_t f = 0; f < n_feats; ++f)
        {
            u16[f].resize(n_rows);
            check(cudaMemcpy(u16[f].data(), bins16.data() + (f * n_rows),
                             n_rows * sizeof(uint16_t), cudaMemcpyDeviceToHost),
                  "materialize bins");
        }
    }
};

// Per-device state and its operations, extracted from CudaHistogramEngine::Impl
// (decision 53's device-resident cluster). CudaHistogramEngine owns exactly one
// and forwards through it; the CPU fallback engine stays in the wrapper.
struct CudaDeviceContext
{
    // Dataset-resident plane (decision 53): the binned matrix and its
    // identity, uploaded once per ensure_dataset and read by every launch.
    struct DeviceData
    {
        // One of bins8/bins16 per dataset (uint8 iff every feature fits 256
        // bins); feature-major, n_features * n_rows.
        DeviceBuffer<uint8_t>  bins8;
        DeviceBuffer<uint16_t> bins16;
        bool                   bins_are_u8 = false;
        DeviceBuffer<uint32_t> n_bins; // per-feature bin counts

        // Adopted ingest plane: when the dataset was device-binned, the
        // matrix already lives in the plane and the local buffers stay
        // empty. Accessors below pick the live storage.
        std::shared_ptr<CudaIngestPlane const> adopted;

        // Uploaded-dataset identity cookies: compared by address to skip a
        // redundant re-upload of the same Dataset, never dereferenced. A
        // mismatch is harmless, the matrix just re-uploads.
        Dataset const *ds      = nullptr;
        void const    *bins0   = nullptr;
        size_t         n_rows  = 0;
        size_t         n_feats = 0;

        uint32_t const *n_bins_ptr() const
        {
            return adopted ? adopted->n_bins.data() : n_bins.data();
        }

        // Calls fn with the active binned-matrix pointer (uint8 when every feature
        // fits 256 bins, else uint16) — the one branch every histogram and
        // partition launch shares.
        template <typename F> void dispatch_bins(F &&fn)
        {
            if (bins_are_u8)
            {
                std::forward<F>(fn)(adopted ? adopted->bins8.data() : bins8.data());
            }
            else
            {
                std::forward<F>(fn)(adopted ? adopted->bins16.data() : bins16.data());
            }
        }
    };

    // Per-tree gradient plane: raw uploads interleaved into (grad, hess)
    // pairs once per tree by populate.
    struct GradientPlane
    {
        DeviceBuffer<float>  grad_raw; // per-tree raw uploads
        DeviceBuffer<float>  hess_raw;
        DeviceBuffer<float2> gh; // interleaved (grad, hess) per row
    };

    // Per-level pipeline (decision 53): resident rows, gathered gradients,
    // and histograms ping-pong between parent and child sides; the Staged<>
    // buffers feed each level's find, partition, and stamp launches.
    // gh_ordered lives here rather than in the gradient plane because it is
    // the level-row-ordered gather and ping-pongs with gh_b.
    struct LevelPipeline
    {
        DeviceBuffer<uint32_t> rows;       // concatenated node row lists
        DeviceBuffer<float2>   gh_ordered; // gathered into level row order
        Staged<uint32_t>       row_ofs;    // per batched node: offset into rows
        Staged<uint32_t>       row_cnt;    // per batched node: row count
        Staged<uint32_t>       features;

        // Resident level state (phase 3): ping-pong per-level histogram
        // buffers, slot-indexed [slot][sel][2 * max_sel_bins] like `out`.
        // cur() holds the frontier the next find reads; advance_level writes
        // children into other() and swaps.
        DeviceBuffer<double> level_a;
        DeviceBuffer<double> level_b;
        bool                 cur_is_a   = true;
        uint32_t             n_selected = 0;
        uint32_t             stride = 0;  // doubles per (slot, feature): 2*max_sel_bins
        Staged<uint32_t>     slots;       // hist out_slot per batched small
        Staged<uint32_t>     triples;     // (parent, small, large) per op
        Staged<double>       node_sums;   // 2 per frontier node
        Staged<double>       node_bounds; // lo, hi per frontier node
        Staged<char>         allowed;     // n_nodes * n_sel, only when constrained
        Staged<int>          monotone;    // per feature
        DeviceBuffer<FeatBest> feat_best;
        Staged<FeatBest>       node_best;
        Staged<double>         level_child; // oblivious: 4 per node [gL,hL,gR,hR]
        DeviceBuffer<double>   level_score; // oblivious: per (feature, dl, bin) scores
        Staged<uint32_t>       sofs;        // small-node subset: ofs/cnt/slot
        Staged<uint32_t>       scnt;
        Staged<uint32_t>       sslot;

        // Stage B: resident rows. rows/gh_ordered are the "a" side; children
        // scatter into the "b" side and the pair swaps with the hist buffers.
        DeviceBuffer<uint32_t> rows_b;
        DeviceBuffer<float2>   gh_b;
        DeviceBuffer<uint8_t>  flags; // per-row route flag, scatter reuse
        Staged<PartOpDev>      part_ops;
        DeviceBuffer<uint32_t> block_counts; // per (op, chunk), scanned in place
        Staged<uint32_t>       nl_dev;       // per op: total left count
        Staged<uint32_t>       stamp_ids;
        DeviceBuffer<uint32_t> leaf_by_row; // per row id: final leaf node
        // Pristine root row list for full-data fits: partitioning ping-pongs
        // the working rows buffer, so the identity permutation is cached once
        // and restored device-to-device per tree instead of re-uploaded
        // (decision 53 step 2). 0 = invalid; only ever the identity, which is
        // what every sampler returns when its size equals n_rows.
        DeviceBuffer<uint32_t> root_rows;
        size_t                 root_rows_cached_n = 0;
        DeviceBuffer<double2>  sum_partial;   // root-sum pass-1 block partials
        DeviceBuffer<double2>  sum_out;       // root-sum result (1 element)
        DeviceBuffer<float>    epi_node_vals; // per-tree epilogue value table
        DeviceBuffer<float>    epi_values;    // per-row mapped values
        std::vector<uint32_t>  slot_ofs;      // current level's segment layout
        std::vector<uint32_t>  slot_cnt;
        std::vector<uint32_t>  next_ofs; // children layout, live after partition
        std::vector<uint32_t>  next_cnt;

        // Profile-only (decision 71 campaign): event pairs bracketing the
        // async histogram-build phases, recorded at launch and read at the
        // next profile sync so measuring never serializes the pipeline.
        // ev[0..3]: memset start, memset end / hist start, hist end /
        // subtract start, subtract end. Root builds record ev[1]..ev[2] only.
        cudaEvent_t prof_ev[4]       = {};
        bool        prof_ev_ready    = false; // events created
        bool        prof_ev_recorded = false; // a build awaits reading
        bool        prof_ev_root     = false; // recorded span is a root build

        void prof_record_begin(bool root)
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
                check(cudaEventRecord(prof_ev[0]), "profile event record");
            }
        }

        // Call only after a sync that guarantees the recorded events are past.
        void prof_read(ProfileCounters &prof)
        {
            if (!prof_ev_recorded)
            {
                return;
            }
            prof_ev_recorded = false;
            float ms         = 0.0F;
            check(cudaEventElapsedTime(&ms, prof_ev[1], prof_ev[2]),
                  "profile event hist");
            (prof_ev_root ? prof.root_hist_s : prof.adv_hist_s) += ms / 1e3;
            if (prof_ev_root)
            {
                return;
            }
            check(cudaEventElapsedTime(&ms, prof_ev[0], prof_ev[1]),
                  "profile event memset");
            prof.adv_memset_s += ms / 1e3;
            check(cudaEventElapsedTime(&ms, prof_ev[2], prof_ev[3]),
                  "profile event subtract");
            prof.adv_sub_s += ms / 1e3;
        }

        ~LevelPipeline()
        {
            if (prof_ev_ready)
            {
                for (auto &e : prof_ev)
                {
                    cudaEventDestroy(e);
                }
            }
        }

        DeviceBuffer<uint32_t> &cur_rows()
        {
            return cur_is_a ? rows : rows_b;
        }
        DeviceBuffer<uint32_t> &other_rows()
        {
            return cur_is_a ? rows_b : rows;
        }
        DeviceBuffer<float2> &cur_gh()
        {
            return cur_is_a ? gh_ordered : gh_b;
        }
        DeviceBuffer<float2> &other_gh()
        {
            return cur_is_a ? gh_b : gh_ordered;
        }
        DeviceBuffer<double> &cur()
        {
            return cur_is_a ? level_a : level_b;
        }
        DeviceBuffer<double> &other()
        {
            return cur_is_a ? level_b : level_a;
        }
        size_t slot_doubles() const
        {
            return static_cast<size_t>(n_selected) * stride;
        }

        // Buckets each child op by the smaller child's size: nodes above the row
        // cutoff stage into the shared-memory kernel's (row_ofs, row_cnt, slots),
        // the rest into the direct-global kernel's (sofs, scnt, sslot); every op's
        // (parent, small, large) triple stages for the subtract. Returns the
        // largest small-child row count (sizes the shared kernel's chunk grid).
        size_t stage_children(std::span<CudaHistogramEngine::LevelOp const> ops)
        {
            size_t max_rows = 0;
            row_ofs.clear();
            row_cnt.clear();
            slots.clear();
            sofs.clear();
            scnt.clear();
            sslot.clear();
            triples.clear();
            for (CudaHistogramEngine::LevelOp const &op : ops)
            {
                uint32_t const ofs = next_ofs[op.small_slot];
                uint32_t const cnt = next_cnt[op.small_slot];
                if (cnt >= k_min_gpu_rows)
                {
                    row_ofs.host.push_back(ofs);
                    row_cnt.host.push_back(cnt);
                    slots.host.push_back(op.small_slot);
                    max_rows = std::max<size_t>(max_rows, cnt);
                }
                else
                {
                    sofs.host.push_back(ofs);
                    scnt.host.push_back(cnt);
                    sslot.host.push_back(op.small_slot);
                }
                triples.host.push_back(op.parent_slot);
                triples.host.push_back(op.small_slot);
                triples.host.push_back(op.large_slot);
            }
            if (!row_ofs.empty())
            {
                row_ofs.sync();
                row_cnt.sync();
                slots.sync();
            }
            if (!sofs.empty())
            {
                sofs.sync();
                scnt.sync();
                sslot.sync();
            }
            triples.sync();
            return max_rows;
        }

        // Fills the child slots' (offset, count) layout from the device left-counts
        // and echoes (n_left, n_right) per op back to the caller.
        void layout_children(std::span<CudaHistogramEngine::PartitionOp const> ops,
                             std::span<uint32_t> child_counts)
        {
            size_t const n = ops.size();
            next_ofs.assign(2 * n, 0);
            next_cnt.assign(2 * n, 0);
            for (size_t k = 0; k < n; ++k)
            {
                uint32_t const nl           = nl_dev.host[k];
                uint32_t const parent_ofs   = part_ops.host[k].ofs;
                uint32_t const parent_cnt   = part_ops.host[k].cnt;
                next_ofs[ops[k].left_slot]  = parent_ofs;
                next_cnt[ops[k].left_slot]  = nl;
                next_ofs[ops[k].right_slot] = parent_ofs + nl;
                next_cnt[ops[k].right_slot] = parent_cnt - nl;
                child_counts[2 * k]         = nl;
                child_counts[(2 * k) + 1]   = parent_cnt - nl;
            }
        }

        // Stages the per-node totals, monotone-bound box, optional allowed-feature
        // mask, and per-feature monotone directions the find kernel reads. Returns
        // whether any node carried a mask (the kernel gets nullptr otherwise).
        bool stage_find_inputs(std::span<SplitInput const> level,
                               TreeConfig const &config, Dataset const &ds)
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
                            level[i].allowed.empty()
                                ? char{1}
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

        // Translates each node's device-side best split into a host SplitOutput and
        // its (left, right) child sums; a node with no valid split or too few rows
        // to split emits an empty output.
        void unpack_splits(std::span<SplitInput const> level, TreeConfig const &config,
                           std::span<SplitOutput> out, std::span<HistCell> child_sums)
        {
            for (size_t i = 0; i < level.size(); ++i)
            {
                FeatBest const &b        = node_best.host[i];
                bool const      eligible = level[i].row_count >=
                                      2 * static_cast<size_t>(config.min_data_in_leaf);
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

        // Oblivious level-find staging: node_sums only. The level kernel reads no
        // bounds/monotone/interaction state (the oblivious grower rejects those
        // constraints at construction), so the full stage_find_inputs is waste.
        void stage_level_sums(std::span<SplitInput const> level)
        {
            node_sums.host.resize(2 * level.size());
            for (size_t i = 0; i < level.size(); ++i)
            {
                node_sums.host[2 * i]       = level[i].sums.sum_grad;
                node_sums.host[(2 * i) + 1] = level[i].sums.sum_hess;
            }
            node_sums.sync();
        }
    };

    DeviceData      data;
    GradientPlane   grads;
    LevelPipeline   lvl;
    ProfileCounters prof_counters;

    // Runtime shared-memory ceiling for the hist kernels: the opt-in limit
    // when the device grants one (both BinT instantiations opted in), else
    // the 48 KiB static budget. Resolved lazily on first use so engine
    // construction never touches the CUDA runtime.
    size_t shared_limit  = k_max_shared_bytes;
    bool   shared_probed = false;

    void init_shared_limit()
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
        int optin = 0;
        if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                   dev) != cudaSuccess ||
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
        cudaGetLastError(); // clear any sticky attribute error
    }

    void ensure_dataset(Dataset const &dataset)
    {
        // Device-binned dataset: adopt its plane — the matrix is already
        // resident; nothing crosses the bus. The plane pointer is the
        // identity; the backend tag proves the concrete type without RTTI.
        if (auto const &receipt = dataset.ingest_plane();
            receipt && receipt->backend_tag() == cuda_backend_tag())
        {
            auto plane = std::static_pointer_cast<CudaIngestPlane const>(receipt);
            if (data.adopted == plane)
            {
                return;
            }
            data.adopted           = std::move(plane);
            data.bins_are_u8       = data.adopted->bins_are_u8;
            data.ds                = &dataset;
            data.bins0             = data.adopted.get();
            data.n_rows            = data.adopted->n_rows;
            data.n_feats           = data.adopted->n_feats;
            lvl.root_rows_cached_n = 0;
            return;
        }
        data.adopted      = nullptr;
        void const *first = dataset.n_features() > 0
                                ? dataset.visit_bins(0, [](auto bins) -> void const *
                                                     { return bins.data(); })
                                : nullptr;
        bool const  same  = data.ds == &dataset && data.bins0 == first &&
                          data.n_rows == dataset.n_rows() &&
                          data.n_feats == dataset.n_features();
        if (same)
        {
            return;
        }
        auto                  blap = prof_counters.lap();
        std::vector<uint32_t> counts(dataset.n_features());
        for (size_t f = 0; f < dataset.n_features(); ++f)
        {
            counts[f] = static_cast<uint32_t>(dataset.n_bins(f));
        }
        // The Dataset stores u8 exactly when every feature fits 256 bins,
        // the same criterion the kernels dispatch on — no narrowing pass.
        data.bins_are_u8       = dataset.bins_are_u8();
        lvl.root_rows_cached_n = 0;
        // One pinned staging buffer + one memcpy per matrix: pageable
        // per-feature copies serialize on GeForce drivers (decision 48),
        // and pinned transfers run at full PCIe rate.
        size_t const cells = dataset.n_features() * dataset.n_rows();
        if (data.bins_are_u8)
        {
            data.bins8.reserve(cells);
            PinnedBuffer<uint8_t> staging(cells);
            parallel::for_each_index(dataset.n_features(),
                                     [&](size_t f)
                                     {
                                         dataset.visit_bins(
                                             f,
                                             [&](auto src)
                                             {
                                                 std::copy(src.begin(), src.end(),
                                                           staging.data() +
                                                               (f * dataset.n_rows()));
                                             });
                                     });
            check(cudaMemcpy(data.bins8.data(), staging.data(), cells,
                             cudaMemcpyHostToDevice),
                  "upload bins");
        }
        else
        {
            data.bins16.reserve(cells);
            PinnedBuffer<uint16_t> staging(cells);
            parallel::for_each_index(dataset.n_features(),
                                     [&](size_t f)
                                     {
                                         dataset.visit_bins(
                                             f,
                                             [&](auto src)
                                             {
                                                 std::copy(src.begin(), src.end(),
                                                           staging.data() +
                                                               (f * dataset.n_rows()));
                                             });
                                     });
            check(cudaMemcpy(data.bins16.data(), staging.data(),
                             cells * sizeof(uint16_t), cudaMemcpyHostToDevice),
                  "upload bins");
        }
        data.n_bins.upload(counts.data(), counts.size());
        blap(prof_counters.bins_upload_s);
        data.ds      = &dataset;
        data.bins0   = first;
        data.n_rows  = dataset.n_rows();
        data.n_feats = dataset.n_features();
    }

    void begin_tree(Dataset const &ds, floats_view grad, floats_view hess)
    {
        ensure_dataset(ds);
        auto       lap = prof_counters.lap();
        auto const n   = static_cast<uint32_t>(grad.size());
        grads.grad_raw.upload(grad.data(), grad.size());
        grads.hess_raw.upload(hess.data(), hess.size());
        grads.gh.reserve(grad.size());
        interleave(grads.grad_raw.data(), grads.hess_raw.data(), n, grads.gh.data());
        lap(prof_counters.gh_upload_s);
    }

    bool begin_root(Dataset const &ds, floats_view grad, floats_view hess,
                    SplitInput &root, std::span<feature_id_t const> selected)
    {
        size_t max_sel_bins = 0;
        for (feature_id_t const fid : selected)
        {
            max_sel_bins = std::max(max_sel_bins, ds.n_bins(fid));
        }
        init_shared_limit();
        if (selected.empty() || 4 * max_sel_bins * sizeof(float) > shared_limit)
        {
            return false; // the LevelStep falls back to host histogram building
        }
        lvl.n_selected = static_cast<uint32_t>(selected.size());
        lvl.stride     = static_cast<uint32_t>(2 * max_sel_bins);
        lvl.features.host.assign(selected.begin(), selected.end());
        lvl.features.sync();

        // Identity contract (decision 71 campaign): a full-data fit passes empty
        // rows + row_count == n_rows; the identity never touches the host or the
        // bus (built by iota_kernel once, cached, restored D2D per tree).
        bool const identity = root.rows.empty() && root.row_count == data.n_rows;
        auto const n =
            static_cast<uint32_t>(identity ? root.row_count : root.rows.size());

        auto root_lap = prof_counters.lap();
        lvl.cur_is_a  = true;
        lvl.cur().reserve(lvl.slot_doubles());
        check(cudaMemset(lvl.cur().data(), 0, lvl.slot_doubles() * sizeof(double)),
              "zero root slot");
        lvl.rows.reserve(n);
        if (static_cast<size_t>(n) == data.n_rows &&
            lvl.root_rows_cached_n == data.n_rows)
        {
            check(cudaMemcpy(lvl.rows.data(), lvl.root_rows.data(),
                             static_cast<size_t>(n) * sizeof(uint32_t),
                             cudaMemcpyDeviceToDevice),
                  "root rows restore");
        }
        else
        {
            if (identity)
            {
                iota_kernel<<<dim3((n + 255) / 256), dim3(256)>>>(lvl.rows.data(), n);
                check(cudaGetLastError(), "iota launch");
            }
            else
            {
                lvl.rows.upload(root.rows.data(), root.rows.size());
            }
            if (static_cast<size_t>(n) == data.n_rows)
            {
                lvl.root_rows.reserve(n);
                check(cudaMemcpy(lvl.root_rows.data(), lvl.rows.data(),
                                 static_cast<size_t>(n) * sizeof(uint32_t),
                                 cudaMemcpyDeviceToDevice),
                      "root rows cache");
                lvl.root_rows_cached_n = data.n_rows;
            }
        }
        lvl.row_ofs.host.assign(1, 0);
        lvl.row_ofs.sync();
        lvl.row_cnt.host.assign(1, n);
        lvl.row_cnt.sync();
        lvl.slots.host.assign(1, 0);
        lvl.slots.sync();
        root_lap(prof_counters.root_stage_s);
        if (identity)
        {
            // Deterministic two-pass device reduce over the uploaded gh buffer
            // replaces the 16M-row host loop (decision 71 campaign); queued
            // before the histogram build so the later 16B fetch drains only
            // these two small kernels.
            constexpr uint32_t k_sum_blocks = 64;
            lvl.sum_partial.reserve(k_sum_blocks);
            lvl.sum_out.reserve(1);
            sum_gh_pass1_kernel<<<dim3(k_sum_blocks), dim3(256)>>>(
                grads.gh.data(), n, lvl.sum_partial.data());
            check(cudaGetLastError(), "root sum pass1 launch");
            sum_gh_pass2_kernel<<<dim3(1), dim3(32)>>>(
                lvl.sum_partial.data(), k_sum_blocks, lvl.sum_out.data());
            check(cudaGetLastError(), "root sum pass2 launch");
        }
        lvl.gh_ordered.reserve(n);
        gather(grads.gh.data(), lvl.rows.data(), n, lvl.gh_ordered.data());
        auto const n_chunks = std::clamp<uint32_t>((n + 32767) / 32768, 1, 64);
        dim3 const grid(lvl.n_selected, 1, n_chunks);
        if (prof_counters.enabled)
        {
            lvl.prof_record_begin(/*root=*/true);
            check(cudaEventRecord(lvl.prof_ev[1]), "profile event record");
        }
        data.dispatch_bins(
            [&](auto const *bins)
            {
                hist_kernel<<<grid, dim3(256), 2UL * lvl.stride * sizeof(float)>>>(
                    bins, lvl.gh_ordered.data(), lvl.rows.data(), lvl.row_ofs.device(),
                    lvl.row_cnt.device(), lvl.features.device(), data.n_bins_ptr(),
                    static_cast<uint32_t>(ds.n_rows()), lvl.n_selected,
                    lvl.cur().data(), lvl.stride, lvl.slots.device());
            });
        check(cudaGetLastError(), "root hist launch");
        if (prof_counters.enabled)
        {
            check(cudaEventRecord(lvl.prof_ev[2]), "profile event record");
            lvl.prof_ev_recorded = true;
        }

        lvl.slot_ofs.assign(1, 0);
        lvl.slot_cnt.assign(1, n);
        lvl.leaf_by_row.reserve(ds.n_rows());

        auto sums_lap = prof_counters.lap();
        if (identity)
        {
            // The sum kernels were queued ahead of the root histogram build, so
            // this 16B fetch waits only on them, not on the hist kernel.
            double2 sums{};
            check(cudaMemcpy(&sums, lvl.sum_out.data(), sizeof(double2),
                             cudaMemcpyDeviceToHost),
                  "root sums fetch");
            root.sums      = {.sum_grad = static_cast<float>(sums.x),
                              .sum_hess = static_cast<float>(sums.y)};
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
        if (prof_counters.enabled)
        {
            ++prof_counters.launches;
            ++prof_counters.gpu_nodes;
        }
        return true;
    }

    void stamp_leaves(std::span<CudaHistogramEngine::LeafStamp const> stamps)
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
                {lvl.slot_ofs[st.slot], lvl.slot_cnt[st.slot], 0, 0, 0});
            lvl.stamp_ids.host.push_back(st.node_id);
        }
        lvl.part_ops.sync();
        lvl.stamp_ids.sync();
        stamp_kernel<<<dim3(static_cast<uint32_t>(stamps.size())), dim3(256)>>>(
            lvl.cur_rows().data(), lvl.part_ops.device(), lvl.stamp_ids.device(),
            lvl.leaf_by_row.data());
        check(cudaGetLastError(), "stamp launch");
        stamp_lap(prof_counters.fin_stamp_s);
    }

    void partition_level(Dataset const & /*ds*/,
                         std::span<CudaHistogramEngine::PartitionOp const> ops,
                         std::span<uint32_t>                               child_counts)
    {
        if (ops.empty())
        {
            lvl.next_ofs.clear();
            lvl.next_cnt.clear();
            return;
        }
        auto &prof = prof_counters;
        auto  lap  = prof.lap();

        size_t const n       = ops.size();
        uint32_t     max_cnt = 0;
        lvl.part_ops.clear();
        for (CudaHistogramEngine::PartitionOp const &op : ops)
        {
            uint32_t const cnt = lvl.slot_cnt[op.parent_slot];
            lvl.part_ops.host.push_back({lvl.slot_ofs[op.parent_slot], cnt,
                                         op.feature_id, op.bin_id,
                                         op.default_left ? 1U : 0U});
            max_cnt = std::max(max_cnt, cnt);
        }
        lvl.part_ops.sync();
        uint32_t const max_chunks =
            std::max(1U, (max_cnt + k_part_chunk - 1) / k_part_chunk);
        lvl.flags.reserve(data.n_rows);
        lvl.block_counts.reserve(n * max_chunks);
        lvl.nl_dev.reserve(n);
        lap(prof.part_stage_s);

        dim3 const grid(max_chunks, static_cast<uint32_t>(n));
        data.dispatch_bins(
            [&](auto const *bins)
            {
                route_count_kernel<<<grid, dim3(k_part_block)>>>(
                    bins, data.n_bins_ptr(), lvl.cur_rows().data(),
                    lvl.part_ops.device(), static_cast<uint32_t>(data.n_rows),
                    max_chunks, lvl.flags.data(), lvl.block_counts.data());
            });
        check(cudaGetLastError(), "route launch");
        seg_scan_kernel<<<dim3(static_cast<uint32_t>(n)), dim3(32)>>>(
            lvl.block_counts.data(), max_chunks, lvl.nl_dev.device());
        check(cudaGetLastError(), "seg scan launch");
        lvl.other_rows().reserve(data.n_rows);
        lvl.other_gh().reserve(data.n_rows);
        scatter_kernel<<<grid, dim3(k_part_block)>>>(
            lvl.cur_rows().data(), lvl.cur_gh().data(), lvl.flags.data(),
            lvl.part_ops.device(), lvl.block_counts.data(), lvl.nl_dev.device(),
            max_chunks, lvl.other_rows().data(), lvl.other_gh().data());
        check(cudaGetLastError(), "scatter launch");
        lvl.nl_dev.fetch(n); // DtoH, implicit sync
        if (prof.enabled)
        {
            ++prof.launches;
        }
        lap(prof.gpu_s);

        lvl.layout_children(ops, child_counts);
    }

    void finalize_rows(std::span<node_id_t> leaf_by_row)
    {
        check(cudaMemcpy(leaf_by_row.data(), lvl.leaf_by_row.data(),
                         leaf_by_row.size() * sizeof(node_id_t),
                         cudaMemcpyDeviceToHost),
              "leaf ids copy");
    }

    void finalize_tree(std::span<float const> node_values, std::span<float> values,
                       std::span<node_id_t> leaf_ids)
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

    void advance_level(Dataset const                                &ds,
                       std::span<CudaHistogramEngine::LevelOp const> ops)
    {
        if (ops.empty())
        {
            return;
        }
        auto &prof = prof_counters;
        auto  lap  = prof.lap();

        // Rows are already device-resident; only the per-child layout stages here.
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
            check(cudaEventRecord(lvl.prof_ev[1]), "profile event record");
        }
        data.dispatch_bins(
            [&](auto const *bins)
            {
                if (!lvl.row_ofs.empty())
                {
                    auto const n_chunks = std::clamp<uint32_t>(
                        (static_cast<uint32_t>(max_rows) + 32767) / 32768, 1, 64);
                    dim3 const grid(lvl.n_selected,
                                    static_cast<uint32_t>(lvl.row_ofs.size()),
                                    n_chunks);
                    hist_kernel<<<grid, dim3(256), 2UL * lvl.stride * sizeof(float)>>>(
                        bins, lvl.other_gh().data(), lvl.other_rows().data(),
                        lvl.row_ofs.device(), lvl.row_cnt.device(),
                        lvl.features.device(), data.n_bins_ptr(),
                        static_cast<uint32_t>(ds.n_rows()), lvl.n_selected,
                        lvl.other().data(), lvl.stride, lvl.slots.device());
                }
                if (!lvl.sofs.empty())
                {
                    hist_small_kernel<<<dim3(static_cast<uint32_t>(lvl.sofs.size())),
                                        dim3(128)>>>(
                        bins, lvl.other_gh().data(), lvl.other_rows().data(),
                        lvl.sofs.device(), lvl.scnt.device(), lvl.features.device(),
                        static_cast<uint32_t>(ds.n_rows()), lvl.n_selected,
                        lvl.other().data(), lvl.stride, lvl.sslot.device());
                }
            });
        check(cudaGetLastError(), "level hist launch");
        if (prof.enabled)
        {
            check(cudaEventRecord(lvl.prof_ev[2]), "profile event record");
        }
        auto const sd = static_cast<uint32_t>(lvl.slot_doubles());
        subtract_kernel<<<dim3(std::clamp<uint32_t>((sd + 255) / 256, 1, 256),
                               static_cast<uint32_t>(ops.size())),
                          dim3(256)>>>(lvl.cur().data(), lvl.other().data(),
                                       lvl.triples.device(), sd);
        check(cudaGetLastError(), "subtract launch");
        if (prof.enabled)
        {
            check(cudaEventRecord(lvl.prof_ev[3]), "profile event record");
            lvl.prof_ev_recorded = true;
        }
        lvl.cur_is_a = !lvl.cur_is_a;
        lvl.slot_ofs = lvl.next_ofs;
        lvl.slot_cnt = lvl.next_cnt;
        if (prof.enabled)
        {
            ++prof.launches;
            prof.gpu_nodes += child_slots;
        }
        lap(prof.gpu_s);
    }

    // Final-level advance (decision 71 campaign): the children of the last
    // level are leaves, so their histograms are never read by any find; only
    // the layout flip survives so stamping sees the final segments.
    void advance_layout_only()
    {
        lvl.cur_is_a = !lvl.cur_is_a;
        lvl.slot_ofs = lvl.next_ofs;
        lvl.slot_cnt = lvl.next_cnt;
    }

    void find_splits_many(Dataset const &ds, TreeConfig const &config,
                          std::span<SplitInput const> level, std::span<SplitOutput> out,
                          std::span<HistCell> child_sums)
    {
        size_t const n    = level.size();
        auto        &prof = prof_counters;
        auto         lap  = prof.lap();

        if (prof.enabled)
        {
            // Separate awaited async kernel time from true staging cost: the
            // first Staged sync below otherwise absorbs the previous level's
            // in-flight kernels into find_stage.
            check(cudaDeviceSynchronize(), "profile wait");
            lap(prof.gpu_wait_s);
            lvl.prof_read(prof);
        }
        bool const any_mask = lvl.stage_find_inputs(level, config, ds);
        lap(prof.find_stage_s);

        lvl.feat_best.reserve(n * lvl.n_selected);
        lvl.node_best.reserve(n);
        find_kernel<<<dim3(lvl.n_selected, static_cast<uint32_t>(n)), dim3(32)>>>(
            lvl.cur().data(), lvl.features.device(), data.n_bins_ptr(),
            lvl.node_sums.device(), lvl.node_bounds.device(),
            any_mask ? lvl.allowed.device() : nullptr, lvl.monotone.device(),
            lvl.n_selected, lvl.stride, config.lambda_l1, config.lambda_l2,
            config.min_child_hess, config.min_gain_to_split, lvl.feat_best.data());
        check(cudaGetLastError(), "find launch");
        reduce_kernel<<<dim3(static_cast<uint32_t>(n)), dim3(32)>>>(
            lvl.feat_best.data(), lvl.n_selected, lvl.node_best.device());
        check(cudaGetLastError(), "reduce launch");
        if (prof.enabled)
        {
            // Peel the awaited kernel+reduce compute from the node_best D2H so a
            // slow find lap can be attributed to FP64 scan time vs transfer.
            check(cudaDeviceSynchronize(), "find kernel wait");
            lap(prof.find_kern_s);
        }
        lvl.node_best.fetch(n); // DtoH, implicit sync
        if (prof.enabled)
        {
            ++prof.launches;
            lap(prof.find_d2h_s);
        }
        lap(prof.gpu_s);

        lvl.unpack_splits(level, config, out, child_sums);
        lap(prof.unpack_s);
    }

    void find_level_split(Dataset const & /*ds*/, TreeConfig const &config,
                          std::span<SplitInput const> level, std::span<SplitOutput> out,
                          std::span<HistCell> child_sums)
    {
        size_t const n    = level.size();
        auto        &prof = prof_counters;
        auto         lap  = prof.lap();

        if (prof.enabled)
        {
            // Same peel as find_splits_many: without it, the first Staged sync
            // below absorbs the previous level's in-flight histogram kernels
            // into lfind_stage, misattributing device compute as staging (the
            // decision-62 lesson, replayed on the oblivious path).
            check(cudaDeviceSynchronize(), "profile wait");
            lap(prof.gpu_wait_s);
            lvl.prof_read(prof);
        }
        lvl.stage_level_sums(level);
        lap(prof.lfind_stage_s);

        // find -> reduce -> child-sums queue back to back (the child kernel reads
        // the reduced winner on-device), so the level pays one sync at the fetch.
        size_t const scratch =
            static_cast<size_t>(lvl.n_selected) * 2 * (lvl.stride / 2);
        lvl.level_score.reserve(scratch);
        lvl.feat_best.reserve(lvl.n_selected);
        lvl.node_best.reserve(1);
        lvl.level_child.reserve(4 * n);
        level_find_kernel<<<dim3(lvl.n_selected), dim3(32)>>>(
            lvl.cur().data(), lvl.features.device(), data.n_bins_ptr(),
            lvl.node_sums.device(), lvl.n_selected, static_cast<uint32_t>(n),
            lvl.stride, config.lambda_l1, config.lambda_l2, config.min_child_hess,
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
        lvl.node_best.fetch(1); // DtoH, implicit sync
        lvl.level_child.fetch(4 * n);
        if (prof.enabled)
        {
            prof.launches += 3;
        }
        lap(prof.gpu_s);

        // One split for the whole frontier, broadcast to every node; each node's
        // (left, right) sums seed the children's SplitInput.sums for the next
        // level's find (their device histograms are not host-scannable).
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
};

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

} // namespace
} // namespace bonsai
