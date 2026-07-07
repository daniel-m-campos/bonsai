// CUDA histogram backend: clang CUDA C++, same libc++/C++23 as the rest of
// the build. Design, batching, and precision scheme:
// docs/architecture/10-cuda.md.

#include "bonsai/config/tree_config.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <functional>
#include <memory>
#include <print>
#include <span>
#include <utility>
#include <vector>
#include <vector_types.h>

#include "detail/device_buffer.cuh"
#include "detail/kernels.cuh"

namespace bonsai
{

// Flat device/host buffers throughout this file are offset by hand (docs/
// architecture/10-cuda.md); grad/hess travel as an adjacent pair everywhere
// in this API, matching the gradient-boosting literature's convention.
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

bool cuda_available()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// BONSAI_CUDA_PROFILE=1 accumulators, printed at builder destruction.
struct ProfileCounters
{
    using clock     = std::chrono::steady_clock;
    bool   enabled  = std::getenv("BONSAI_CUDA_PROFILE") != nullptr;
    double upload_s = 0, gpu_s = 0, unpack_s = 0, cpu_s = 0;
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
                         upload_s, gpu_s, unpack_s, cpu_s, launches, gpu_nodes,
                         cpu_calls);
        }
        catch (...)
        {
            std::fputs("cuda-profile: failed to format profile line\n", stderr);
        }
    }
};

struct CudaHistogramEngine::Impl
{
    // One of bins8/bins16 per dataset (uint8 iff every feature fits 256
    // bins); feature-major, n_features * n_rows.
    DeviceBuffer<uint8_t>  bins8;
    DeviceBuffer<uint16_t> bins16;
    bool                   bins_are_u8 = false;
    DeviceBuffer<uint32_t> n_bins;   // per-feature bin counts
    DeviceBuffer<float>    grad_raw; // per-tree raw uploads, interleaved below
    DeviceBuffer<float>    hess_raw;
    DeviceBuffer<float2>   gh;         // interleaved (grad, hess) per row
    DeviceBuffer<float2>   gh_ordered; // gathered into level row order
    DeviceBuffer<uint32_t> rows;       // concatenated node row lists
    Staged<uint32_t>       row_ofs;    // per batched node: offset into rows
    Staged<uint32_t>       row_cnt;    // per batched node: row count
    Staged<uint32_t>       features;
    Staged<double>         out;
    CpuHistogramEngine     cpu;
    ProfileCounters        prof_counters;

    // Uploaded-dataset identity heuristic; any mismatch just re-uploads.
    Dataset const *ds      = nullptr;
    void const    *bins0   = nullptr;
    size_t         n_rows  = 0;
    size_t         n_feats = 0;

    // Host staging for `rows` only: its device side ping-pongs with rows_b, so
    // it is not a Staged pair like the other buffers.
    std::vector<uint32_t> host_rows;

    // Resident level state (phase 3): ping-pong per-level histogram buffers,
    // slot-indexed [slot][sel][2 * max_sel_bins] like `out`. cur() holds the
    // frontier the next find reads; advance_level writes children into
    // other() and swaps.
    DeviceBuffer<double>   level_a;
    DeviceBuffer<double>   level_b;
    bool                   cur_is_a   = true;
    uint32_t               n_selected = 0;
    uint32_t               stride = 0;  // doubles per (slot, feature): 2*max_sel_bins
    Staged<uint32_t>       slots;       // hist out_slot per batched small
    Staged<uint32_t>       triples;     // (parent, small, large) per op
    Staged<double>         node_sums;   // 2 per frontier node
    Staged<double>         node_bounds; // lo, hi per frontier node
    Staged<char>           allowed;     // n_nodes * n_sel, only when constrained
    Staged<int>            monotone;    // per feature
    DeviceBuffer<FeatBest> feat_best;
    Staged<FeatBest>       node_best;
    Staged<uint32_t>       sofs; // small-node subset: ofs/cnt/slot
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
    std::vector<uint32_t>  slot_ofs;    // current level's segment layout
    std::vector<uint32_t>  slot_cnt;
    std::vector<uint32_t>  next_ofs; // children layout, live after partition
    std::vector<uint32_t>  next_cnt;

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

    // Calls fn with the active binned-matrix pointer (uint8 when every feature
    // fits 256 bins, else uint16) — the one branch every histogram and
    // partition launch shares.
    template <typename F> void dispatch_bins(F &&fn)
    {
        if (bins_are_u8)
        {
            std::forward<F>(fn)(bins8.data());
        }
        else
        {
            std::forward<F>(fn)(bins16.data());
        }
    }

    // Concatenates the batched nodes' row lists into one device upload; the
    // per-node offsets index it in-kernel. Returns {total_rows, max_rows}.
    std::pair<size_t, size_t>
    stage_level_rows(std::vector<std::reference_wrapper<SplitInput>> const &batched,
                     std::span<feature_id_t const>                          selected)
    {
        size_t total_rows = 0;
        size_t max_rows   = 0;
        row_ofs.clear();
        row_cnt.clear();
        for (SplitInput const &node : batched)
        {
            row_ofs.host.push_back(static_cast<uint32_t>(total_rows));
            row_cnt.host.push_back(static_cast<uint32_t>(node.rows.size()));
            total_rows += node.rows.size();
            max_rows = std::max(max_rows, node.rows.size());
        }
        host_rows.resize(total_rows);
        for (size_t n = 0; n < batched.size(); ++n)
        {
            std::ranges::copy(batched[n].get().rows,
                              host_rows.begin() + row_ofs.host[n]);
        }
        features.host.assign(selected.begin(), selected.end());
        rows.upload(host_rows.data(), total_rows);
        row_ofs.sync();
        row_cnt.sync();
        features.sync();
        return {total_rows, max_rows};
    }

    // Copies the device histogram cells back into each batched node's
    // per-feature host Histogram.
    void
    unpack_histograms(std::vector<std::reference_wrapper<SplitInput>> const &batched,
                      std::span<feature_id_t const> selected, uint32_t stride)
    {
        for (size_t n = 0; n < batched.size(); ++n)
        {
            for (size_t s = 0; s < selected.size(); ++s)
            {
                Histogram    &h = batched[n].get().hists[selected[s]];
                double const *cells =
                    out.host.data() + (((n * selected.size()) + s) * stride);
                for (size_t b = 0; b < h.size(); ++b)
                {
                    h.add(static_cast<bin_id_t>(b), cells[2 * b], cells[(2 * b) + 1]);
                }
            }
        }
    }

    // Buckets each child op by the smaller child's size: nodes above the row
    // cutoff stage into the shared-memory kernel's (row_ofs, row_cnt, slots),
    // the rest into the direct-global kernel's (sofs, scnt, sslot); every op's
    // (parent, small, large) triple stages for the subtract. Returns the
    // largest small-child row count (sizes the shared kernel's chunk grid).
    size_t stage_children(std::span<LevelOp const> ops)
    {
        size_t max_rows = 0;
        row_ofs.clear();
        row_cnt.clear();
        slots.clear();
        sofs.clear();
        scnt.clear();
        sslot.clear();
        triples.clear();
        for (LevelOp const &op : ops)
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
    void layout_children(std::span<PartitionOp const> ops,
                         std::span<uint32_t>          child_counts)
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
    bool stage_find_inputs(std::span<SplitInput const> level, TreeConfig const &config,
                           Dataset const &ds)
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

    // Translates each node's device-side best split into a host SplitOutput and
    // its (left, right) child sums; a node with no valid split or too few rows
    // to split emits an empty output.
    void unpack_splits(std::span<SplitInput const> level, TreeConfig const &config,
                       std::span<SplitOutput> out, std::span<HistCell> child_sums)
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
            child_sums[2 * i]       = {.sum_grad = b.gL, .sum_hess = b.hL};
            child_sums[(2 * i) + 1] = {.sum_grad = b.gR, .sum_hess = b.hR};
        }
    }

    void ensure_dataset(Dataset const &dataset)
    {
        void const *first =
            dataset.n_features() > 0
                ? static_cast<void const *>(dataset.feature_bins(0).data())
                : nullptr;
        bool const same = ds == &dataset && bins0 == first &&
                          n_rows == dataset.n_rows() && n_feats == dataset.n_features();
        if (same)
        {
            return;
        }
        std::vector<uint32_t> counts(dataset.n_features());
        bool                  all_u8 = true;
        for (size_t f = 0; f < dataset.n_features(); ++f)
        {
            counts[f] = static_cast<uint32_t>(dataset.n_bins(f));
            all_u8    = all_u8 && counts[f] <= 256;
        }
        bins_are_u8 = all_u8;
        if (all_u8)
        {
            bins8.reserve(dataset.n_features() * dataset.n_rows());
            std::vector<uint8_t> narrow(dataset.n_rows());
            for (size_t f = 0; f < dataset.n_features(); ++f)
            {
                auto const src = dataset.feature_bins(f);
                for (size_t r = 0; r < src.size(); ++r)
                {
                    narrow[r] = static_cast<uint8_t>(src[r]);
                }
                check(cudaMemcpy(bins8.data() + (f * dataset.n_rows()), narrow.data(),
                                 narrow.size(), cudaMemcpyHostToDevice),
                      "upload bins");
            }
        }
        else
        {
            bins16.reserve(dataset.n_features() * dataset.n_rows());
            for (size_t f = 0; f < dataset.n_features(); ++f)
            {
                check(cudaMemcpy(bins16.data() + (f * dataset.n_rows()),
                                 dataset.feature_bins(f).data(),
                                 dataset.n_rows() * sizeof(uint16_t),
                                 cudaMemcpyHostToDevice),
                      "upload bins");
            }
        }
        n_bins.upload(counts.data(), counts.size());
        ds      = &dataset;
        bins0   = first;
        n_rows  = dataset.n_rows();
        n_feats = dataset.n_features();
    }
};

CudaHistogramEngine::CudaHistogramEngine() : impl_(std::make_unique<Impl>()) {}
CudaHistogramEngine::~CudaHistogramEngine()                               = default;
CudaHistogramEngine::CudaHistogramEngine(CudaHistogramEngine &&) noexcept = default;
CudaHistogramEngine &
CudaHistogramEngine::operator=(CudaHistogramEngine &&) noexcept = default;

void CudaHistogramEngine::begin_tree(Dataset const &ds, floats_view grad,
                                     floats_view hess)
{
    impl_->ensure_dataset(ds);
    auto const n = static_cast<uint32_t>(grad.size());
    impl_->grad_raw.upload(grad.data(), grad.size());
    impl_->hess_raw.upload(hess.data(), hess.size());
    impl_->gh.reserve(grad.size());
    interleave(impl_->grad_raw.data(), impl_->hess_raw.data(), n, impl_->gh.data());
}

void CudaHistogramEngine::populate(Dataset const &ds, floats_view grad,
                                   floats_view hess, SplitInput &split_input,
                                   std::span<feature_id_t const> selected)
{
    std::array const one = {std::ref(split_input)};
    populate_many(ds, grad, hess, one, selected);
}

// Splits a level's nodes: sub-cutoff nodes (or any node when a feature's
// histogram would overflow shared memory) go to the CPU builder; the rest
// batch into one device launch.
static void
split_gpu_cpu_nodes(split_input_refs nodes, bool shared_fits,
                    std::vector<std::reference_wrapper<SplitInput>> &cpu_nodes,
                    std::vector<std::reference_wrapper<SplitInput>> &batched)
{
    batched.reserve(nodes.size());
    for (SplitInput &node : nodes)
    {
        if (node.rows.size() < k_min_gpu_rows || !shared_fits)
        {
            cpu_nodes.emplace_back(node);
        }
        else
        {
            batched.emplace_back(node);
        }
    }
}

// CPU-builder shape contract: every feature gets a Histogram, zero-binned where
// unselected so the finders skip it.
static void reserve_placeholder_hists(
    std::vector<std::reference_wrapper<SplitInput>> const &batched, Dataset const &ds,
    std::span<feature_id_t const> selected)
{
    for (SplitInput &node : batched)
    {
        node.hists.reserve(ds.n_features());
        size_t j = 0;
        for (feature_id_t fid = 0; fid < ds.n_features(); ++fid)
        {
            bool const sel = j < selected.size() && selected[j] == fid;
            node.hists.emplace_back(sel ? ds.n_bins(fid) : 0);
            j += sel ? 1 : 0;
        }
    }
}

void CudaHistogramEngine::populate_many(Dataset const &ds, floats_view grad,
                                        floats_view hess, split_input_refs nodes,
                                        std::span<feature_id_t const> selected)
{
    size_t max_selected_bins = 0;
    for (feature_id_t const fid : selected)
    {
        max_selected_bins = std::max(max_selected_bins, ds.n_bins(fid));
    }
    bool const shared_fits =
        4 * max_selected_bins * sizeof(float) <= k_max_shared_bytes; // 2 sub-hists

    auto &prof_counters = impl_->prof_counters;
    auto  lap           = prof_counters.lap();

    std::vector<std::reference_wrapper<SplitInput>> cpu_nodes;
    std::vector<std::reference_wrapper<SplitInput>> batched;
    split_gpu_cpu_nodes(nodes, shared_fits, cpu_nodes, batched);
    // One worker per node; the CPU builder's inner parallel loops degrade to a
    // team of one inside this region, so results stay bit-identical.
    prof_counters.cpu_calls += cpu_nodes.size();
    parallel::for_each_index(
        cpu_nodes.size(),
        [&](size_t i) { impl_->cpu.populate(ds, grad, hess, cpu_nodes[i], selected); });
    lap(prof_counters.cpu_s);
    if (batched.empty())
    {
        return;
    }

    reserve_placeholder_hists(batched, ds, selected);
    if (selected.empty())
    {
        return;
    }

    auto const [total_rows, max_rows] = impl_->stage_level_rows(batched, selected);
    lap(prof_counters.upload_s);

    auto const   stride = static_cast<uint32_t>(2 * max_selected_bins);
    size_t const out_doubles =
        static_cast<size_t>(stride) * selected.size() * batched.size();
    impl_->out.reserve(out_doubles);
    check(cudaMemset(impl_->out.device(), 0, out_doubles * sizeof(double)), "zero out");

    // Chunk count sized by the level's largest node, capped at 64.
    uint32_t const chunk    = 32768;
    auto const     n_chunks = std::clamp<uint32_t>(
        (static_cast<uint32_t>(max_rows) + chunk - 1) / chunk, 1, 64);
    dim3 const grid(static_cast<uint32_t>(selected.size()),
                    static_cast<uint32_t>(batched.size()), n_chunks);
    dim3 const block(256);
    impl_->gh_ordered.reserve(total_rows);
    gather(impl_->gh.data(), impl_->rows.data(), static_cast<uint32_t>(total_rows),
           impl_->gh_ordered.data());
    impl_->dispatch_bins(
        [&](auto const *bins)
        {
            hist_kernel<<<grid, block,
                          2 * static_cast<size_t>(stride) * sizeof(float)>>>(
                bins, impl_->gh_ordered.data(), impl_->rows.data(),
                impl_->row_ofs.device(), impl_->row_cnt.device(),
                impl_->features.device(), impl_->n_bins.data(),
                static_cast<uint32_t>(ds.n_rows()),
                static_cast<uint32_t>(selected.size()), impl_->out.device(), stride,
                nullptr);
        });
    check(cudaGetLastError(), "launch");

    impl_->out.fetch(out_doubles); // DtoH, implicit sync
    if (prof_counters.enabled)
    {
        ++prof_counters.launches;
        prof_counters.gpu_nodes += batched.size();
    }
    lap(prof_counters.gpu_s);

    impl_->unpack_histograms(batched, selected, stride);
    lap(prof_counters.unpack_s);
}

bool CudaHistogramEngine::begin_root(Dataset const &ds, floats_view grad,
                                     floats_view hess, SplitInput &root,
                                     std::span<feature_id_t const> selected)
{
    Impl  &im           = *impl_;
    size_t max_sel_bins = 0;
    for (feature_id_t const fid : selected)
    {
        max_sel_bins = std::max(max_sel_bins, ds.n_bins(fid));
    }
    if (selected.empty() || 4 * max_sel_bins * sizeof(float) > k_max_shared_bytes)
    {
        return false; // the LevelStep falls back to host histogram building
    }
    im.n_selected = static_cast<uint32_t>(selected.size());
    im.stride     = static_cast<uint32_t>(2 * max_sel_bins);
    im.features.host.assign(selected.begin(), selected.end());
    im.features.sync();

    im.cur_is_a = true;
    im.cur().reserve(im.slot_doubles());
    check(cudaMemset(im.cur().data(), 0, im.slot_doubles() * sizeof(double)),
          "zero root slot");
    auto const n = static_cast<uint32_t>(root.rows.size());
    im.rows.upload(root.rows.data(), root.rows.size());
    im.row_ofs.host.assign(1, 0);
    im.row_ofs.sync();
    im.row_cnt.host.assign(1, n);
    im.row_cnt.sync();
    im.slots.host.assign(1, 0);
    im.slots.sync();
    im.gh_ordered.reserve(root.rows.size());
    gather(im.gh.data(), im.rows.data(), n, im.gh_ordered.data());
    auto const n_chunks = std::clamp<uint32_t>((n + 32767) / 32768, 1, 64);
    dim3 const grid(im.n_selected, 1, n_chunks);
    im.dispatch_bins(
        [&](auto const *bins)
        {
            hist_kernel<<<grid, dim3(256), 2UL * im.stride * sizeof(float)>>>(
                bins, im.gh_ordered.data(), im.rows.data(), im.row_ofs.device(),
                im.row_cnt.device(), im.features.device(), im.n_bins.data(),
                static_cast<uint32_t>(ds.n_rows()), im.n_selected, im.cur().data(),
                im.stride, im.slots.device());
        });
    check(cudaGetLastError(), "root hist launch");

    im.slot_ofs.assign(1, 0);
    im.slot_cnt.assign(1, n);
    im.leaf_by_row.reserve(ds.n_rows());

    double sg = 0.0;
    double sh = 0.0;
    for (row_id_t const r : root.rows)
    {
        sg += grad[r];
        sh += hess[r];
    }
    root.sums      = {.sum_grad = sg, .sum_hess = sh};
    root.row_count = root.rows.size();
    if (im.prof_counters.enabled)
    {
        ++im.prof_counters.launches;
        ++im.prof_counters.gpu_nodes;
    }
    return true;
}

void CudaHistogramEngine::stamp_leaves(std::span<LeafStamp const> stamps)
{
    Impl &im = *impl_;
    if (stamps.empty())
    {
        return;
    }
    im.part_ops.clear();
    im.stamp_ids.clear();
    for (LeafStamp const &st : stamps)
    {
        im.part_ops.host.push_back(
            {im.slot_ofs[st.slot], im.slot_cnt[st.slot], 0, 0, 0});
        im.stamp_ids.host.push_back(st.node_id);
    }
    im.part_ops.sync();
    im.stamp_ids.sync();
    stamp_kernel<<<dim3(static_cast<uint32_t>(stamps.size())), dim3(256)>>>(
        im.cur_rows().data(), im.part_ops.device(), im.stamp_ids.device(),
        im.leaf_by_row.data());
    check(cudaGetLastError(), "stamp launch");
}

void CudaHistogramEngine::partition_level(Dataset const & /*ds*/,
                                          std::span<PartitionOp const> ops,
                                          std::span<uint32_t>          child_counts)
{
    Impl &im = *impl_;
    if (ops.empty())
    {
        im.next_ofs.clear();
        im.next_cnt.clear();
        return;
    }
    auto &prof = im.prof_counters;
    auto  lap  = prof.lap();

    size_t const n       = ops.size();
    uint32_t     max_cnt = 0;
    im.part_ops.clear();
    for (PartitionOp const &op : ops)
    {
        uint32_t const cnt = im.slot_cnt[op.parent_slot];
        im.part_ops.host.push_back({im.slot_ofs[op.parent_slot], cnt, op.feature_id,
                                    op.bin_id, op.default_left ? 1U : 0U});
        max_cnt = std::max(max_cnt, cnt);
    }
    im.part_ops.sync();
    uint32_t const max_chunks =
        std::max(1U, (max_cnt + k_part_chunk - 1) / k_part_chunk);
    im.flags.reserve(im.n_rows);
    im.block_counts.reserve(n * max_chunks);
    im.nl_dev.reserve(n);
    lap(prof.upload_s);

    dim3 const grid(max_chunks, static_cast<uint32_t>(n));
    im.dispatch_bins(
        [&](auto const *bins)
        {
            route_count_kernel<<<grid, dim3(k_part_block)>>>(
                bins, im.n_bins.data(), im.cur_rows().data(), im.part_ops.device(),
                static_cast<uint32_t>(im.n_rows), max_chunks, im.flags.data(),
                im.block_counts.data());
        });
    check(cudaGetLastError(), "route launch");
    seg_scan_kernel<<<dim3(static_cast<uint32_t>(n)), dim3(32)>>>(
        im.block_counts.data(), max_chunks, im.nl_dev.device());
    check(cudaGetLastError(), "seg scan launch");
    im.other_rows().reserve(im.n_rows);
    im.other_gh().reserve(im.n_rows);
    scatter_kernel<<<grid, dim3(k_part_block)>>>(
        im.cur_rows().data(), im.cur_gh().data(), im.flags.data(), im.part_ops.device(),
        im.block_counts.data(), im.nl_dev.device(), max_chunks, im.other_rows().data(),
        im.other_gh().data());
    check(cudaGetLastError(), "scatter launch");
    im.nl_dev.fetch(n); // DtoH, implicit sync
    if (prof.enabled)
    {
        ++prof.launches;
    }
    lap(prof.gpu_s);

    im.layout_children(ops, child_counts);
}

void CudaHistogramEngine::finalize_rows(std::span<node_id_t> leaf_by_row)
{
    Impl &im = *impl_;
    check(cudaMemcpy(leaf_by_row.data(), im.leaf_by_row.data(),
                     leaf_by_row.size() * sizeof(node_id_t), cudaMemcpyDeviceToHost),
          "leaf ids copy");
}

void CudaHistogramEngine::advance_level(Dataset const &ds, std::span<LevelOp const> ops)
{
    Impl &im = *impl_;
    if (ops.empty())
    {
        return;
    }
    auto &prof = im.prof_counters;
    auto  lap  = prof.lap();

    // Rows are already device-resident; only the per-child layout stages here.
    size_t const max_rows = im.stage_children(ops);
    lap(prof.upload_s);

    size_t const child_slots = 2 * ops.size();
    im.other().reserve(child_slots * im.slot_doubles());
    check(cudaMemset(im.other().data(), 0,
                     child_slots * im.slot_doubles() * sizeof(double)),
          "zero level");
    im.dispatch_bins(
        [&](auto const *bins)
        {
            if (!im.row_ofs.empty())
            {
                auto const n_chunks = std::clamp<uint32_t>(
                    (static_cast<uint32_t>(max_rows) + 32767) / 32768, 1, 64);
                dim3 const grid(im.n_selected, static_cast<uint32_t>(im.row_ofs.size()),
                                n_chunks);
                hist_kernel<<<grid, dim3(256), 2UL * im.stride * sizeof(float)>>>(
                    bins, im.other_gh().data(), im.other_rows().data(),
                    im.row_ofs.device(), im.row_cnt.device(), im.features.device(),
                    im.n_bins.data(), static_cast<uint32_t>(ds.n_rows()), im.n_selected,
                    im.other().data(), im.stride, im.slots.device());
            }
            if (!im.sofs.empty())
            {
                hist_small_kernel<<<dim3(static_cast<uint32_t>(im.sofs.size())),
                                    dim3(128)>>>(
                    bins, im.other_gh().data(), im.other_rows().data(),
                    im.sofs.device(), im.scnt.device(), im.features.device(),
                    static_cast<uint32_t>(ds.n_rows()), im.n_selected,
                    im.other().data(), im.stride, im.sslot.device());
            }
        });
    check(cudaGetLastError(), "level hist launch");
    auto const sd = static_cast<uint32_t>(im.slot_doubles());
    subtract_kernel<<<dim3(std::clamp<uint32_t>((sd + 255) / 256, 1, 256),
                           static_cast<uint32_t>(ops.size())),
                      dim3(256)>>>(im.cur().data(), im.other().data(),
                                   im.triples.device(), sd);
    check(cudaGetLastError(), "subtract launch");
    im.cur_is_a = !im.cur_is_a;
    im.slot_ofs = im.next_ofs;
    im.slot_cnt = im.next_cnt;
    if (prof.enabled)
    {
        ++prof.launches;
        prof.gpu_nodes += child_slots;
    }
    lap(prof.gpu_s);
}

void CudaHistogramEngine::find_splits_many(Dataset const &ds, TreeConfig const &config,
                                           std::span<SplitInput const> level,
                                           std::span<SplitOutput>      out,
                                           std::span<HistCell>         child_sums)
{
    Impl        &im   = *impl_;
    size_t const n    = level.size();
    auto        &prof = im.prof_counters;
    auto         lap  = prof.lap();

    bool const any_mask = im.stage_find_inputs(level, config, ds);
    lap(prof.upload_s);

    im.feat_best.reserve(n * im.n_selected);
    im.node_best.reserve(n);
    find_kernel<<<dim3((im.n_selected + 31) / 32, static_cast<uint32_t>(n)),
                  dim3(32)>>>(im.cur().data(), im.features.device(), im.n_bins.data(),
                              im.node_sums.device(), im.node_bounds.device(),
                              any_mask ? im.allowed.device() : nullptr,
                              im.monotone.device(), im.n_selected, im.stride,
                              config.lambda_l1, config.lambda_l2, config.min_child_hess,
                              config.min_gain_to_split, im.feat_best.data());
    check(cudaGetLastError(), "find launch");
    reduce_kernel<<<dim3(static_cast<uint32_t>(n)), dim3(32)>>>(
        im.feat_best.data(), im.n_selected, im.node_best.device());
    check(cudaGetLastError(), "reduce launch");
    im.node_best.fetch(n); // DtoH, implicit sync
    if (prof.enabled)
    {
        ++prof.launches;
    }
    lap(prof.gpu_s);

    im.unpack_splits(level, config, out, child_sums);
    lap(prof.unpack_s);
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

} // namespace bonsai
