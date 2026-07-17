#pragma once

// Per-device state and operations shared by CudaHistogramEngine (one context)
// and, in P3, MultiCudaHistogramEngine (N contexts); see
// docs/architecture/19-multi-gpu.md. A real header: the types live in
// namespace bonsai::cuda_detail with external linkage and the heavy method
// bodies compile once in device_context.cu, so a second translation unit can
// include this without an ODR clash. Every launch and every cuda_runtime call
// stays in the .cu; this header carries only declarations, data members, and
// small member-only accessors, and names no entity from kernels.cuh.

#include "bonsai/config/tree_config.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <driver_types.h>
#include <memory>
#include <print>
#include <span>
#include <utility>
#include <vector>
#include <vector_types.h>

#include "device_buffer.cuh"

namespace bonsai
{
namespace cuda_detail
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
// concrete type and makes the static_cast sound. The inline function's local
// static is guaranteed to be one object across every translation unit that
// calls it, so the identity holds process-wide even now that two TUs link
// against this header.
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
                     std::vector<std::vector<uint16_t>> &u16) const override;
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

        void prof_record_begin(bool root);
        // Call only after a sync that guarantees the recorded events are past.
        void prof_read(ProfileCounters &prof);
        ~LevelPipeline();

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

    void init_shared_limit();
    void ensure_dataset(Dataset const &dataset);
    // offset/count slice the per-tree gradient upload: only grad/hess
    // [offset, offset + count) is uploaded and interleaved into the same
    // positions of the full-length gh buffer; the defaults upload the whole
    // stream so single-GPU stays byte-for-byte. gh keeps global row indexing,
    // so the rows outside the slice hold stale values no shard row list ever
    // gathers. MultiCudaHistogramEngine passes each identity shard's
    // [lo, hi - lo) (docs/architecture/19-multi-gpu.md).
    void begin_tree(Dataset const &ds, floats_view grad, floats_view hess,
                    size_t offset = 0, size_t count = SIZE_MAX);
    bool begin_root(Dataset const &ds, floats_view grad, floats_view hess,
                    SplitInput &root, std::span<feature_id_t const> selected);
    void stamp_leaves(std::span<CudaHistogramEngine::LeafStamp const> stamps);
    void partition_level(Dataset const                                    &ds,
                         std::span<CudaHistogramEngine::PartitionOp const> ops,
                         std::span<uint32_t> child_counts);
    void finalize_rows(std::span<node_id_t> leaf_by_row);
    // offset/count slice the two D2H downloads into the caller spans; the
    // defaults copy the whole span so single-GPU stays byte-for-byte. The map
    // kernel always runs the full range. MultiCudaHistogramEngine passes each
    // shard's [offset, offset + count) so every context writes only its own
    // rows into the shared output (docs/architecture/19-multi-gpu.md).
    void finalize_tree(std::span<float const> node_values, std::span<float> values,
                       std::span<node_id_t> leaf_ids, size_t offset = 0,
                       size_t count = SIZE_MAX);
    void advance_level(Dataset const                                &ds,
                       std::span<CudaHistogramEngine::LevelOp const> ops);
    // Final-level advance (decision 71 campaign): the children of the last
    // level are leaves, so their histograms are never read by any find; only
    // the layout flip survives so stamping sees the final segments.
    void advance_layout_only();
    // hists_override, when non-null, replaces lvl.cur() as the level histogram
    // source the find kernels scan (same slot-indexed layout, slot_doubles()
    // doubles per slot). MultiCudaHistogramEngine passes the coordinator's
    // reduced-across-shards buffer so find runs on the global histogram
    // (docs/architecture/19-multi-gpu.md); the default keeps the single-GPU
    // path byte-for-byte unchanged.
    void find_splits_many(Dataset const &ds, TreeConfig const &config,
                          std::span<SplitInput const> level, std::span<SplitOutput> out,
                          std::span<HistCell> child_sums,
                          double const       *hists_override = nullptr);
    void find_level_split(Dataset const &ds, TreeConfig const &config,
                          std::span<SplitInput const> level, std::span<SplitOutput> out,
                          std::span<HistCell> child_sums,
                          double const       *hists_override = nullptr);
};

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

} // namespace cuda_detail
} // namespace bonsai
