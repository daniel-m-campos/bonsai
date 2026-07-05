// CUDA histogram backend: clang CUDA C++, same libc++/C++23 as the rest of
// the build. Design, batching, and precision scheme:
// docs/architecture/10-cuda.md.

#include "bonsai/cuda/histogram_builder.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/types.hpp"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "detail/device_buffer.cuh"
#include "detail/kernels.cuh"

namespace bonsai
{

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

    ~ProfileCounters()
    {
        if (!enabled || (gpu_s == 0 && cpu_s == 0))
        {
            return;
        }
        std::fprintf(stderr,
                     "cuda-profile: upload=%.2fs gpu=%.2fs unpack=%.2fs "
                     "cpu_fallback=%.2fs | %zu launches covering %zu nodes, "
                     "%zu cpu-fallback nodes\n",
                     upload_s, gpu_s, unpack_s, cpu_s, launches, gpu_nodes, cpu_calls);
    }
};

struct CudaHistogramBuilder::Impl
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
    DeviceBuffer<uint32_t> row_ofs;    // per batched node: offset into rows
    DeviceBuffer<uint32_t> row_cnt;    // per batched node: row count
    DeviceBuffer<uint32_t> features;
    DeviceBuffer<double>   out;
    CpuHistogramBuilder    cpu;
    ProfileCounters        prof;

    // Uploaded-dataset identity heuristic; any mismatch just re-uploads.
    Dataset const *ds      = nullptr;
    void const    *bins0   = nullptr;
    size_t         n_rows  = 0;
    size_t         n_feats = 0;

    // Host staging reused across levels.
    std::vector<double>   host_out;
    std::vector<uint32_t> host_features;
    std::vector<uint32_t> host_rows;
    std::vector<uint32_t> host_ofs;
    std::vector<uint32_t> host_cnt;

    // Resident level state (phase 3): ping-pong per-level histogram buffers,
    // slot-indexed [slot][sel][2 * max_sel_bins] like `out`. cur() holds the
    // frontier the next find reads; advance_level writes children into
    // other() and swaps.
    DeviceBuffer<double>   level_a;
    DeviceBuffer<double>   level_b;
    bool                   cur_is_a = true;
    bool                   resident = false;
    uint32_t               n_sel    = 0;
    uint32_t               stride   = 0; // doubles per (slot, feature): 2*max_sel_bins
    DeviceBuffer<uint32_t> slots;        // hist out_slot per batched small
    DeviceBuffer<uint32_t> triples;      // (parent, small, large) per op
    DeviceBuffer<double>   node_sums;    // 2 per frontier node
    DeviceBuffer<double>   node_bounds;  // lo, hi per frontier node
    DeviceBuffer<char>     allowed;      // n_nodes * n_sel, only when constrained
    DeviceBuffer<int>      monotone;     // per feature
    DeviceBuffer<FeatBest> feat_best;
    DeviceBuffer<FeatBest> node_best;
    std::vector<uint32_t>  host_slots;
    std::vector<uint32_t>  host_triples;
    std::vector<uint32_t>  host_sofs; // small-node subset: ofs/cnt/slot
    std::vector<uint32_t>  host_scnt;
    std::vector<uint32_t>  host_sslot;
    DeviceBuffer<uint32_t> sofs;
    DeviceBuffer<uint32_t> scnt;
    DeviceBuffer<uint32_t> sslot;

    // Stage B: resident rows. rows/gh_ordered are the "a" side; children
    // scatter into the "b" side and the pair swaps with the hist buffers.
    DeviceBuffer<uint32_t>  rows_b;
    DeviceBuffer<float2>    gh_b;
    DeviceBuffer<uint8_t>   flags; // per-row route flag, scatter reuse
    DeviceBuffer<PartOpDev> part_ops;
    DeviceBuffer<uint32_t>  block_counts; // per (op, chunk), scanned in place
    DeviceBuffer<uint32_t>  nl_dev;       // per op: total left count
    DeviceBuffer<uint32_t>  stamp_ids;
    DeviceBuffer<uint32_t>  leaf_by_row; // per row id: final leaf node
    std::vector<uint32_t>   slot_ofs;    // current level's segment layout
    std::vector<uint32_t>   slot_cnt;
    std::vector<uint32_t>   next_ofs; // children layout, live after partition
    std::vector<uint32_t>   next_cnt;
    std::vector<PartOpDev>  host_part;
    std::vector<uint32_t>   host_nl;
    std::vector<uint32_t>   host_ids;

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
    std::vector<double>   host_nsums;
    std::vector<double>   host_bounds;
    std::vector<char>     host_allowed;
    std::vector<int>      host_mono;
    std::vector<FeatBest> host_best;

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
        return static_cast<size_t>(n_sel) * stride;
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
            bins8.ensure(dataset.n_features() * dataset.n_rows());
            std::vector<uint8_t> narrow(dataset.n_rows());
            for (size_t f = 0; f < dataset.n_features(); ++f)
            {
                auto const src = dataset.feature_bins(f);
                for (size_t r = 0; r < src.size(); ++r)
                {
                    narrow[r] = static_cast<uint8_t>(src[r]);
                }
                check(cudaMemcpy(bins8.get() + (f * dataset.n_rows()), narrow.data(),
                                 narrow.size(), cudaMemcpyHostToDevice),
                      "upload bins");
            }
        }
        else
        {
            bins16.ensure(dataset.n_features() * dataset.n_rows());
            for (size_t f = 0; f < dataset.n_features(); ++f)
            {
                check(cudaMemcpy(bins16.get() + (f * dataset.n_rows()),
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

CudaHistogramBuilder::CudaHistogramBuilder() : impl_(std::make_unique<Impl>()) {}
CudaHistogramBuilder::~CudaHistogramBuilder()                                = default;
CudaHistogramBuilder::CudaHistogramBuilder(CudaHistogramBuilder &&) noexcept = default;
CudaHistogramBuilder &
CudaHistogramBuilder::operator=(CudaHistogramBuilder &&) noexcept = default;

void CudaHistogramBuilder::begin_tree(Dataset const &ds, floats_view grad,
                                      floats_view hess)
{
    impl_->ensure_dataset(ds);
    impl_->resident = false;
    auto const n    = static_cast<uint32_t>(grad.size());
    impl_->grad_raw.upload(grad.data(), grad.size());
    impl_->hess_raw.upload(hess.data(), hess.size());
    impl_->gh.ensure(grad.size());
    interleave_kernel<<<dim3(std::clamp<uint32_t>(n / 256, 1, 1024)), dim3(256)>>>(
        impl_->grad_raw.get(), impl_->hess_raw.get(), n, impl_->gh.get());
    check(cudaGetLastError(), "interleave launch");
}

void CudaHistogramBuilder::populate(Dataset const &ds, floats_view grad,
                                    floats_view hess, SplitInput &split_input,
                                    std::span<feature_id_t const> selected)
{
    std::array const one = {std::ref(split_input)};
    populate_many(ds, grad, hess, one, selected);
}

void CudaHistogramBuilder::populate_many(Dataset const &ds, floats_view grad,
                                         floats_view hess, split_input_refs nodes,
                                         std::span<feature_id_t const> selected)
{
    size_t max_sel_bins = 0;
    for (feature_id_t const fid : selected)
    {
        max_sel_bins = std::max(max_sel_bins, ds.n_bins(fid));
    }
    bool const shared_fits =
        4 * max_sel_bins * sizeof(float) <= k_max_shared_bytes; // 2 sub-hists

    auto &prof = impl_->prof;
    auto  t0   = ProfileCounters::clock::now();
    auto  lap  = [&](double &sink)
    {
        if (!prof.enabled)
        {
            return;
        }
        auto const t1 = ProfileCounters::clock::now();
        sink += std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
    };

    // Sub-cutoff nodes go to the CPU builder; the rest share one launch.
    std::vector<std::reference_wrapper<SplitInput>> cpu_nodes;
    std::vector<std::reference_wrapper<SplitInput>> batched;
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
    // One worker per node; the CPU builder's inner parallel loops degrade
    // to a team of one inside this region, so results stay bit-identical.
    prof.cpu_calls += cpu_nodes.size();
    parallel::for_each_index(
        cpu_nodes.size(),
        [&](size_t i) { impl_->cpu.populate(ds, grad, hess, cpu_nodes[i], selected); });
    lap(prof.cpu_s);
    if (batched.empty())
    {
        return;
    }

    // CPU-builder shape contract: zero-binned placeholders when unselected.
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
    if (selected.empty())
    {
        return;
    }

    // One concatenated row upload; per-node offsets index it in-kernel.
    size_t total_rows = 0;
    size_t max_rows   = 0;
    impl_->host_ofs.clear();
    impl_->host_cnt.clear();
    for (SplitInput const &node : batched)
    {
        impl_->host_ofs.push_back(static_cast<uint32_t>(total_rows));
        impl_->host_cnt.push_back(static_cast<uint32_t>(node.rows.size()));
        total_rows += node.rows.size();
        max_rows = std::max(max_rows, node.rows.size());
    }
    impl_->host_rows.resize(total_rows);
    for (size_t n = 0; n < batched.size(); ++n)
    {
        SplitInput const &node = batched[n];
        std::ranges::copy(node.rows, impl_->host_rows.begin() + impl_->host_ofs[n]);
    }
    impl_->host_features.assign(selected.begin(), selected.end());
    impl_->rows.upload(impl_->host_rows.data(), total_rows);
    impl_->row_ofs.upload(impl_->host_ofs.data(), batched.size());
    impl_->row_cnt.upload(impl_->host_cnt.data(), batched.size());
    impl_->features.upload(impl_->host_features.data(), selected.size());
    lap(prof.upload_s);

    auto const   stride = static_cast<uint32_t>(2 * max_sel_bins);
    size_t const out_doubles =
        static_cast<size_t>(stride) * selected.size() * batched.size();
    impl_->out.ensure(out_doubles);
    check(cudaMemset(impl_->out.get(), 0, out_doubles * sizeof(double)), "zero out");

    // Chunk count sized by the level's largest node, capped at 64.
    uint32_t const chunk    = 32768;
    auto const     n_chunks = std::clamp<uint32_t>(
        (static_cast<uint32_t>(max_rows) + chunk - 1) / chunk, 1, 64);
    dim3 const grid(static_cast<uint32_t>(selected.size()),
                    static_cast<uint32_t>(batched.size()), n_chunks);
    dim3 const block(256);
    impl_->gh_ordered.ensure(total_rows);
    gather_gh_kernel<<<
        dim3(std::clamp<uint32_t>(static_cast<uint32_t>(total_rows / 256), 1, 512)),
        block>>>(impl_->gh.get(), impl_->rows.get(), static_cast<uint32_t>(total_rows),
                 impl_->gh_ordered.get());
    check(cudaGetLastError(), "gather launch");
    auto const launch = [&](auto const *bins)
    {
        hist_kernel<<<grid, block, 2 * stride * sizeof(float)>>>(
            bins, impl_->gh_ordered.get(), impl_->rows.get(), impl_->row_ofs.get(),
            impl_->row_cnt.get(), impl_->features.get(), impl_->n_bins.get(),
            static_cast<uint32_t>(ds.n_rows()), static_cast<uint32_t>(selected.size()),
            impl_->out.get(), stride, nullptr);
    };
    if (impl_->bins_are_u8)
    {
        launch(impl_->bins8.get());
    }
    else
    {
        launch(impl_->bins16.get());
    }
    check(cudaGetLastError(), "launch");

    impl_->host_out.resize(out_doubles);
    check(cudaMemcpy(impl_->host_out.data(), impl_->out.get(),
                     out_doubles * sizeof(double),
                     cudaMemcpyDeviceToHost), // implicit sync
          "copy back");
    if (prof.enabled)
    {
        ++prof.launches;
        prof.gpu_nodes += batched.size();
    }
    lap(prof.gpu_s);

    for (size_t n = 0; n < batched.size(); ++n)
    {
        for (size_t s = 0; s < selected.size(); ++s)
        {
            Histogram    &h = batched[n].get().hists[selected[s]];
            double const *cells =
                impl_->host_out.data() + (((n * selected.size()) + s) * stride);
            for (size_t b = 0; b < h.size(); ++b)
            {
                h.add(static_cast<bin_id_t>(b), cells[2 * b], cells[(2 * b) + 1]);
            }
        }
    }
    lap(prof.unpack_s);
}

bool CudaHistogramBuilder::resident() const
{
    return impl_->resident;
}

bool CudaHistogramBuilder::begin_root(Dataset const &ds, floats_view grad,
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
        return false; // caller degrades to the copy-back path
    }
    im.resident = true;
    im.n_sel    = static_cast<uint32_t>(selected.size());
    im.stride   = static_cast<uint32_t>(2 * max_sel_bins);
    im.host_features.assign(selected.begin(), selected.end());
    im.features.upload(im.host_features.data(), im.host_features.size());

    im.cur_is_a = true;
    im.cur().ensure(im.slot_doubles());
    check(cudaMemset(im.cur().get(), 0, im.slot_doubles() * sizeof(double)),
          "zero root slot");
    auto const n = static_cast<uint32_t>(root.rows.size());
    im.rows.upload(root.rows.data(), root.rows.size());
    uint32_t const zero = 0;
    im.row_ofs.upload(&zero, 1);
    im.row_cnt.upload(&n, 1);
    im.slots.upload(&zero, 1);
    im.gh_ordered.ensure(root.rows.size());
    gather_gh_kernel<<<dim3(std::clamp<uint32_t>(n / 256, 1, 512)), dim3(256)>>>(
        im.gh.get(), im.rows.get(), n, im.gh_ordered.get());
    check(cudaGetLastError(), "gather launch");
    auto const n_chunks = std::clamp<uint32_t>((n + 32767) / 32768, 1, 64);
    dim3 const grid(im.n_sel, 1, n_chunks);
    auto const launch = [&](auto const *bins)
    {
        hist_kernel<<<grid, dim3(256), 2UL * im.stride * sizeof(float)>>>(
            bins, im.gh_ordered.get(), im.rows.get(), im.row_ofs.get(),
            im.row_cnt.get(), im.features.get(), im.n_bins.get(),
            static_cast<uint32_t>(ds.n_rows()), im.n_sel, im.cur().get(), im.stride,
            im.slots.get());
    };
    if (im.bins_are_u8)
    {
        launch(im.bins8.get());
    }
    else
    {
        launch(im.bins16.get());
    }
    check(cudaGetLastError(), "root hist launch");

    im.slot_ofs.assign(1, 0);
    im.slot_cnt.assign(1, n);
    im.leaf_by_row.ensure(ds.n_rows());

    double sg = 0.0;
    double sh = 0.0;
    for (row_id_t const r : root.rows)
    {
        sg += grad[r];
        sh += hess[r];
    }
    root.sums      = {sg, sh};
    root.row_count = root.rows.size();
    if (im.prof.enabled)
    {
        ++im.prof.launches;
        ++im.prof.gpu_nodes;
    }
    return true;
}

void CudaHistogramBuilder::stamp_leaves(std::span<LeafStamp const> stamps)
{
    Impl &im = *impl_;
    if (stamps.empty())
    {
        return;
    }
    im.host_part.clear();
    im.host_ids.clear();
    for (LeafStamp const &st : stamps)
    {
        im.host_part.push_back({im.slot_ofs[st.slot], im.slot_cnt[st.slot], 0, 0, 0});
        im.host_ids.push_back(st.node_id);
    }
    im.part_ops.upload(im.host_part.data(), im.host_part.size());
    im.stamp_ids.upload(im.host_ids.data(), im.host_ids.size());
    stamp_kernel<<<dim3(static_cast<uint32_t>(stamps.size())), dim3(256)>>>(
        im.cur_rows().get(), im.part_ops.get(), im.stamp_ids.get(),
        im.leaf_by_row.get());
    check(cudaGetLastError(), "stamp launch");
}

void CudaHistogramBuilder::partition_level(Dataset const & /*ds*/,
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
    auto &prof = im.prof;
    auto  t0   = ProfileCounters::clock::now();
    auto  lap  = [&](double &sink)
    {
        if (!prof.enabled)
        {
            return;
        }
        auto const t1 = ProfileCounters::clock::now();
        sink += std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
    };

    size_t const n       = ops.size();
    uint32_t     max_cnt = 0;
    im.host_part.clear();
    for (PartitionOp const &op : ops)
    {
        uint32_t const cnt = im.slot_cnt[op.parent_slot];
        im.host_part.push_back({im.slot_ofs[op.parent_slot], cnt, op.feature_id,
                                op.bin_id, op.default_left ? 1U : 0U});
        max_cnt = std::max(max_cnt, cnt);
    }
    im.part_ops.upload(im.host_part.data(), n);
    uint32_t const max_chunks =
        std::max(1U, (max_cnt + k_part_chunk - 1) / k_part_chunk);
    im.flags.ensure(im.n_rows);
    im.block_counts.ensure(n * max_chunks);
    im.nl_dev.ensure(n);
    lap(prof.upload_s);

    dim3 const grid(max_chunks, static_cast<uint32_t>(n));
    auto const route = [&](auto const *bins)
    {
        route_count_kernel<<<grid, dim3(k_part_block)>>>(
            bins, im.n_bins.get(), im.cur_rows().get(), im.part_ops.get(),
            static_cast<uint32_t>(im.n_rows), max_chunks, im.flags.get(),
            im.block_counts.get());
    };
    if (im.bins_are_u8)
    {
        route(im.bins8.get());
    }
    else
    {
        route(im.bins16.get());
    }
    check(cudaGetLastError(), "route launch");
    seg_scan_kernel<<<dim3(static_cast<uint32_t>(n)), dim3(32)>>>(
        im.block_counts.get(), max_chunks, im.nl_dev.get());
    check(cudaGetLastError(), "seg scan launch");
    im.other_rows().ensure(im.n_rows);
    im.other_gh().ensure(im.n_rows);
    scatter_kernel<<<grid, dim3(k_part_block)>>>(
        im.cur_rows().get(), im.cur_gh().get(), im.flags.get(), im.part_ops.get(),
        im.block_counts.get(), im.nl_dev.get(), max_chunks, im.other_rows().get(),
        im.other_gh().get());
    check(cudaGetLastError(), "scatter launch");
    im.host_nl.resize(n);
    check(cudaMemcpy(im.host_nl.data(), im.nl_dev.get(), n * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost), // implicit sync
          "partition counts");
    if (prof.enabled)
    {
        ++prof.launches;
    }
    lap(prof.gpu_s);

    im.next_ofs.assign(2 * n, 0);
    im.next_cnt.assign(2 * n, 0);
    for (size_t k = 0; k < n; ++k)
    {
        uint32_t const nl              = im.host_nl[k];
        uint32_t const parent_ofs      = im.host_part[k].ofs;
        uint32_t const parent_cnt      = im.host_part[k].cnt;
        im.next_ofs[ops[k].left_slot]  = parent_ofs;
        im.next_cnt[ops[k].left_slot]  = nl;
        im.next_ofs[ops[k].right_slot] = parent_ofs + nl;
        im.next_cnt[ops[k].right_slot] = parent_cnt - nl;
        child_counts[2 * k]            = nl;
        child_counts[(2 * k) + 1]      = parent_cnt - nl;
    }
}

void CudaHistogramBuilder::finalize_rows(std::span<node_id_t> leaf_by_row)
{
    Impl &im = *impl_;
    check(cudaMemcpy(leaf_by_row.data(), im.leaf_by_row.get(),
                     leaf_by_row.size() * sizeof(node_id_t), cudaMemcpyDeviceToHost),
          "leaf ids copy");
}

void CudaHistogramBuilder::advance_level(Dataset const           &ds,
                                         std::span<LevelOp const> ops)
{
    Impl &im = *impl_;
    if (ops.empty())
    {
        return;
    }
    auto &prof = im.prof;
    auto  t0   = ProfileCounters::clock::now();
    auto  lap  = [&](double &sink)
    {
        if (!prof.enabled)
        {
            return;
        }
        auto const t1 = ProfileCounters::clock::now();
        sink += std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
    };

    // Smalls route to the shared-memory kernel above the row cutoff and to
    // the direct-global kernel below it; rows are already device-resident.
    size_t max_rows = 0;
    im.host_ofs.clear();
    im.host_cnt.clear();
    im.host_slots.clear();
    im.host_sofs.clear();
    im.host_scnt.clear();
    im.host_sslot.clear();
    im.host_triples.clear();
    for (LevelOp const &op : ops)
    {
        uint32_t const ofs = im.next_ofs[op.small_slot];
        uint32_t const cnt = im.next_cnt[op.small_slot];
        if (cnt >= k_min_gpu_rows)
        {
            im.host_ofs.push_back(ofs);
            im.host_cnt.push_back(cnt);
            im.host_slots.push_back(op.small_slot);
            max_rows = std::max<size_t>(max_rows, cnt);
        }
        else
        {
            im.host_sofs.push_back(ofs);
            im.host_scnt.push_back(cnt);
            im.host_sslot.push_back(op.small_slot);
        }
        im.host_triples.push_back(op.parent_slot);
        im.host_triples.push_back(op.small_slot);
        im.host_triples.push_back(op.large_slot);
    }
    if (!im.host_ofs.empty())
    {
        im.row_ofs.upload(im.host_ofs.data(), im.host_ofs.size());
        im.row_cnt.upload(im.host_cnt.data(), im.host_cnt.size());
        im.slots.upload(im.host_slots.data(), im.host_slots.size());
    }
    if (!im.host_sofs.empty())
    {
        im.sofs.upload(im.host_sofs.data(), im.host_sofs.size());
        im.scnt.upload(im.host_scnt.data(), im.host_scnt.size());
        im.sslot.upload(im.host_sslot.data(), im.host_sslot.size());
    }
    im.triples.upload(im.host_triples.data(), im.host_triples.size());
    lap(prof.upload_s);

    size_t const child_slots = 2 * ops.size();
    im.other().ensure(child_slots * im.slot_doubles());
    check(cudaMemset(im.other().get(), 0,
                     child_slots * im.slot_doubles() * sizeof(double)),
          "zero level");
    auto const launch = [&](auto const *bins)
    {
        if (!im.host_ofs.empty())
        {
            auto const n_chunks = std::clamp<uint32_t>(
                (static_cast<uint32_t>(max_rows) + 32767) / 32768, 1, 64);
            dim3 const grid(im.n_sel, static_cast<uint32_t>(im.host_ofs.size()),
                            n_chunks);
            hist_kernel<<<grid, dim3(256), 2UL * im.stride * sizeof(float)>>>(
                bins, im.other_gh().get(), im.other_rows().get(), im.row_ofs.get(),
                im.row_cnt.get(), im.features.get(), im.n_bins.get(),
                static_cast<uint32_t>(ds.n_rows()), im.n_sel, im.other().get(),
                im.stride, im.slots.get());
        }
        if (!im.host_sofs.empty())
        {
            hist_small_kernel<<<dim3(static_cast<uint32_t>(im.host_sofs.size())),
                                dim3(128)>>>(
                bins, im.other_gh().get(), im.other_rows().get(), im.sofs.get(),
                im.scnt.get(), im.features.get(), static_cast<uint32_t>(ds.n_rows()),
                im.n_sel, im.other().get(), im.stride, im.sslot.get());
        }
    };
    if (im.bins_are_u8)
    {
        launch(im.bins8.get());
    }
    else
    {
        launch(im.bins16.get());
    }
    check(cudaGetLastError(), "level hist launch");
    auto const sd = static_cast<uint32_t>(im.slot_doubles());
    subtract_kernel<<<dim3(std::clamp<uint32_t>((sd + 255) / 256, 1, 256),
                           static_cast<uint32_t>(ops.size())),
                      dim3(256)>>>(im.cur().get(), im.other().get(), im.triples.get(),
                                   sd);
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

void CudaHistogramBuilder::find_splits_many(Dataset const &ds, TreeConfig const &config,
                                            std::span<SplitInput const> level,
                                            std::span<SplitOutput>      out,
                                            std::span<HistCell>         child_sums)
{
    Impl        &im   = *impl_;
    size_t const n    = level.size();
    auto        &prof = im.prof;
    auto         t0   = ProfileCounters::clock::now();
    auto         lap  = [&](double &sink)
    {
        if (!prof.enabled)
        {
            return;
        }
        auto const t1 = ProfileCounters::clock::now();
        sink += std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
    };

    im.host_nsums.resize(2 * n);
    im.host_bounds.resize(2 * n);
    bool any_mask = false;
    for (size_t i = 0; i < n; ++i)
    {
        im.host_nsums[2 * i]        = level[i].sums.sum_grad;
        im.host_nsums[(2 * i) + 1]  = level[i].sums.sum_hess;
        im.host_bounds[2 * i]       = level[i].lo;
        im.host_bounds[(2 * i) + 1] = level[i].hi;
        any_mask                    = any_mask || !level[i].allowed.empty();
    }
    im.node_sums.upload(im.host_nsums.data(), 2 * n);
    im.node_bounds.upload(im.host_bounds.data(), 2 * n);
    if (any_mask)
    {
        im.host_allowed.resize(n * im.n_sel);
        for (size_t i = 0; i < n; ++i)
        {
            for (uint32_t s = 0; s < im.n_sel; ++s)
            {
                im.host_allowed[(i * im.n_sel) + s] =
                    level[i].allowed.empty() ? char{1}
                                             : level[i].allowed[im.host_features[s]];
            }
        }
        im.allowed.upload(im.host_allowed.data(), im.host_allowed.size());
    }
    im.host_mono.resize(ds.n_features());
    for (feature_id_t f = 0; f < ds.n_features(); ++f)
    {
        im.host_mono[f] = monotone_constraint_of(config, f);
    }
    im.monotone.upload(im.host_mono.data(), im.host_mono.size());
    lap(prof.upload_s);

    im.feat_best.ensure(n * im.n_sel);
    im.node_best.ensure(n);
    find_kernel<<<dim3((im.n_sel + 31) / 32, static_cast<uint32_t>(n)), dim3(32)>>>(
        im.cur().get(), im.features.get(), im.n_bins.get(), im.node_sums.get(),
        im.node_bounds.get(), any_mask ? im.allowed.get() : nullptr, im.monotone.get(),
        im.n_sel, im.stride, config.lambda_l1, config.lambda_l2, config.min_child_hess,
        config.min_gain_to_split, im.feat_best.get());
    check(cudaGetLastError(), "find launch");
    reduce_kernel<<<dim3(static_cast<uint32_t>(n)), dim3(32)>>>(
        im.feat_best.get(), im.n_sel, im.node_best.get());
    check(cudaGetLastError(), "reduce launch");
    im.host_best.resize(n);
    check(cudaMemcpy(im.host_best.data(), im.node_best.get(), n * sizeof(FeatBest),
                     cudaMemcpyDeviceToHost), // implicit sync
          "find copy back");
    if (prof.enabled)
    {
        ++prof.launches;
    }
    lap(prof.gpu_s);

    for (size_t i = 0; i < n; ++i)
    {
        FeatBest const &b = im.host_best[i];
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
                      im.host_features[static_cast<size_t>(b.sel)]),
                                   .bin_id       = static_cast<bin_id_t>(b.bin),
                                   .default_left = b.dl != 0,
                                   .valid        = true};
        child_sums[2 * i]       = {b.gL, b.hL};
        child_sums[(2 * i) + 1] = {b.gR, b.hR};
    }
    lap(prof.unpack_s);
}

} // namespace bonsai
