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

namespace bonsai
{

bool cuda_available()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

namespace
{

// Nodes with fewer rows than this build on the CPU: the kernel launch +
// synchronous copy-back round trip outweighs the histogram work itself
// below roughly this size (knee measured on Jetson Orin Nano).
constexpr size_t k_min_gpu_rows = 512;

// Shared-memory histogram footprint cap (stride floats, 48 KiB/block
// budget). Datasets binned past ~6k bins per feature fall back to the CPU
// builder instead of failing the kernel launch at runtime.
constexpr size_t k_max_shared_bytes = 48UL * 1024UL;

void check(cudaError_t rc, char const *what)
{
    if (rc != cudaSuccess)
    {
        throw std::runtime_error(std::string{"cuda: "} + what + ": " +
                                 cudaGetErrorString(rc));
    }
}

// Owning device allocation. Grow-only, geometric; contents dropped on
// reallocation (callers re-upload per use).
template <typename T> class DeviceBuffer
{
  public:
    DeviceBuffer() = default;
    ~DeviceBuffer()
    {
        cudaFree(ptr_);
    }
    DeviceBuffer(DeviceBuffer const &)            = delete;
    DeviceBuffer &operator=(DeviceBuffer const &) = delete;

    T *get() const
    {
        return ptr_;
    }

    void ensure(size_t needed)
    {
        if (needed <= capacity_)
        {
            return;
        }
        size_t grown = capacity_ == 0 ? needed : capacity_;
        while (grown < needed)
        {
            grown *= 2;
        }
        cudaFree(ptr_);
        ptr_      = nullptr;
        capacity_ = 0;
        check(cudaMalloc(&ptr_, grown * sizeof(T)), "malloc");
        capacity_ = grown;
    }

    void upload(T const *host, size_t n)
    {
        ensure(n);
        check(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }

  private:
    T     *ptr_      = nullptr;
    size_t capacity_ = 0;
};

// Reorders (grad, hess) into level order once, so hist_kernel reads them
// sequentially instead of re-gathering per feature.
__global__ void gather_gh_kernel(float2 const *gh, uint32_t const *rows,
                                 uint32_t total_rows, float2 *gh_ordered)
{
    uint32_t const span = gridDim.x * blockDim.x;
    for (uint32_t k = (blockIdx.x * blockDim.x) + threadIdx.x; k < total_rows;
         k += span)
    {
        gh_ordered[k] = gh[rows[k]];
    }
}

// Grid is (feature, node, row-chunk). Shared accumulation is float
// (native atomics; double atomics CAS-loop), cross-chunk merge is double —
// rounding stays bounded per <= 32k-row chunk.
template <typename BinT>
__global__ void hist_kernel(BinT const *bins, float2 const *gh_ordered,
                            uint32_t const *rows, uint32_t const *row_ofs,
                            uint32_t const *row_cnt, uint32_t const *features,
                            uint32_t const *n_bins, uint32_t n_rows, uint32_t n_sel,
                            double *out, uint32_t stride, uint32_t const *out_slot)
{
    // Two sub-histograms split by warp parity spread atomic contention.
    extern __shared__ float sh[];
    uint32_t const          f    = features[blockIdx.x];
    uint32_t const          node = blockIdx.y;
    uint32_t const          nb   = n_bins[f];
    for (uint32_t i = threadIdx.x; i < 4 * nb; i += blockDim.x)
    {
        sh[i] = 0.0F;
    }
    __syncthreads();
    float          *my    = sh + (((threadIdx.x >> 5) & 1U) * 2 * nb);
    BinT const     *fb    = bins + (static_cast<size_t>(f) * n_rows);
    uint32_t const  ofs   = row_ofs[node];
    uint32_t const *nrows = rows + ofs;
    float2 const   *ngh   = gh_ordered + ofs;
    uint32_t const  cnt   = row_cnt[node];
    uint32_t const  span  = gridDim.z * blockDim.x;
    for (uint32_t k = (blockIdx.z * blockDim.x) + threadIdx.x; k < cnt; k += span)
    {
        uint32_t const b = fb[nrows[k]];
        float2 const   v = ngh[k];
        atomicAdd(&my[2 * b], v.x);
        atomicAdd(&my[(2 * b) + 1], v.y);
    }
    __syncthreads();
    uint32_t const oslot = out_slot != nullptr ? out_slot[node] : node;
    double *o = out + (((static_cast<size_t>(oslot) * n_sel) + blockIdx.x) * stride);
    for (uint32_t i = threadIdx.x; i < 2 * nb; i += blockDim.x)
    {
        float const v = sh[i] + sh[(2 * nb) + i];
        if (v != 0.0F)
        {
            atomicAdd(&o[i], static_cast<double>(v));
        }
    }
}

// Small nodes skip the shared-memory stage: below ~512 rows the fixed
// per-(node,feature) zero+merge cost dominates, so one block per node
// accumulates row visits straight into the node's global slot in double.
template <typename BinT>
__global__ void hist_small_kernel(BinT const *bins, float2 const *gh_ordered,
                                  uint32_t const *rows, uint32_t const *row_ofs,
                                  uint32_t const *row_cnt, uint32_t const *features,
                                  uint32_t n_rows, uint32_t n_sel, double *out,
                                  uint32_t stride, uint32_t const *out_slot)
{
    uint32_t const  node = blockIdx.x;
    uint32_t const  cnt  = row_cnt[node];
    uint32_t const  ofs  = row_ofs[node];
    uint32_t const *nr   = rows + ofs;
    float2 const   *ngh  = gh_ordered + ofs;
    double *o = out + (static_cast<size_t>(out_slot[node]) * n_sel * stride);
    for (uint32_t k = threadIdx.x; k < cnt * n_sel; k += blockDim.x)
    {
        uint32_t const sel = k / cnt;
        uint32_t const i   = k % cnt;
        uint32_t const b   = bins[(static_cast<size_t>(features[sel]) * n_rows) + nr[i]];
        float2 const   v   = ngh[i];
        atomicAdd(&o[(sel * stride) + (2 * b)], static_cast<double>(v.x));
        atomicAdd(&o[(sel * stride) + (2 * b) + 1], static_cast<double>(v.y));
    }
}

// Larger children derive on-device: child[large] = parent - child[small].
// Slot triples are (parent, small, large); slot_doubles is one slot's span.
__global__ void subtract_kernel(double const *parents, double *children,
                                uint32_t const *triples, uint32_t slot_doubles)
{
    uint32_t const *t     = triples + (3UL * blockIdx.y);
    uint32_t const  span  = gridDim.x * blockDim.x;
    double const   *par   = parents + (static_cast<size_t>(t[0]) * slot_doubles);
    double const   *small = children + (static_cast<size_t>(t[1]) * slot_doubles);
    double         *large = children + (static_cast<size_t>(t[2]) * slot_doubles);
    for (uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < slot_doubles;
         i += span)
    {
        large[i] = par[i] - small[i];
    }
}

// Per-(node, feature) best split. 56-byte POD; dl encodes default_left.
struct FeatBest
{
    double  gain, gL, hL, gR, hR;
    int32_t bin, dl, valid, sel;
};

// One thread walks one (node, selected-feature) histogram sequentially,
// replicating the CPU scan in split.cpp exactly: same prefix order, both
// default_left routings, min_child_hess, monotone feasibility, min_gain.
// The scan is <= 254 iterations; clarity beats occupancy here.
__global__ void find_kernel(double const *hists, uint32_t const *features,
                            uint32_t const *n_bins, double const *node_sums,
                            double const *node_bounds, char const *allowed,
                            int const *monotone, uint32_t n_sel, uint32_t stride,
                            double l1, double l2, double min_child_hess,
                            double min_gain, FeatBest *out)
{
    if (threadIdx.x != 0)
    {
        return;
    }
    uint32_t const node = blockIdx.y;
    uint32_t const sel  = blockIdx.x;
    FeatBest       best = {};
    out[(static_cast<size_t>(node) * n_sel) + sel] = best;
    if (allowed != nullptr && allowed[(static_cast<size_t>(node) * n_sel) + sel] == 0)
    {
        return;
    }
    uint32_t const f  = features[sel];
    uint32_t const nb = n_bins[f];
    if (nb < 2)
    {
        return; // no cut cells (degenerate feature)
    }
    double const *cells =
        hists + (((static_cast<size_t>(node) * n_sel) + sel) * stride);
    double const g_total = node_sums[2 * node];
    double const h_total = node_sums[(2 * node) + 1];
    double const miss_g  = cells[2 * (nb - 1)];
    double const miss_h  = cells[(2 * (nb - 1)) + 1];
    double const node_score = score(g_total, h_total, l1, l2);
    double const real_grad  = g_total - miss_g;
    double const real_hess  = h_total - miss_h;
    double const lo         = node_bounds[2 * node];
    double const hi         = node_bounds[(2 * node) + 1];
    int const    mc         = monotone[f];

    // Cut cells are bins [0, nb-2): the last real bin cannot split and the
    // final cell is the missing bin (mirrors Histogram::cut_cells()).
    double pg = 0.0;
    double ph = 0.0;
    for (uint32_t b = 0; b + 2 < nb; ++b)
    {
        pg += cells[2 * b];
        ph += cells[(2 * b) + 1];
        for (int dl = 1; dl >= 0; --dl)
        {
            double const gL = pg + (dl != 0 ? miss_g : 0.0);
            double const hL = ph + (dl != 0 ? miss_h : 0.0);
            double const gR = (real_grad - pg) + (dl == 0 ? miss_g : 0.0);
            double const hR = (real_hess - ph) + (dl == 0 ? miss_h : 0.0);
            if (hL < min_child_hess || hR < min_child_hess)
            {
                continue;
            }
            if (mc != 0)
            {
                double const wL = bounded_leaf_weight(gL, hL, l1, l2, lo, hi);
                double const wR = bounded_leaf_weight(gR, hR, l1, l2, lo, hi);
                if (static_cast<double>(mc) * (wR - wL) < 0.0)
                {
                    continue;
                }
            }
            double const gain = score(gL, hL, l1, l2) + score(gR, hR, l1, l2) -
                                node_score;
            if (gain > best.gain && gain >= min_gain)
            {
                best = {gain, gL, hL, gR, hR, static_cast<int32_t>(b), dl, 1,
                        static_cast<int32_t>(sel)};
            }
        }
    }
    out[(static_cast<size_t>(node) * n_sel) + sel] = best;
}

// Per-node winner in ascending selected-feature order with strict >,
// matching reduce_in_feature_order's lowest-feature-id tie-break.
__global__ void reduce_kernel(FeatBest const *per_feat, uint32_t n_sel,
                              FeatBest *out)
{
    if (threadIdx.x != 0)
    {
        return;
    }
    uint32_t const  node = blockIdx.x;
    FeatBest        best = {};
    FeatBest const *row  = per_feat + (static_cast<size_t>(node) * n_sel);
    for (uint32_t s = 0; s < n_sel; ++s)
    {
        if (row[s].valid != 0 && row[s].gain > best.gain)
        {
            best = row[s];
        }
    }
    out[node] = best;
}

} // namespace

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
    DeviceBuffer<uint32_t> n_bins;     // per-feature bin counts
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
    std::vector<float2>   host_gh;

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
    std::vector<double>    host_nsums;
    std::vector<double>    host_bounds;
    std::vector<char>      host_allowed;
    std::vector<int>       host_mono;
    std::vector<FeatBest>  host_best;

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
    impl_->host_gh.resize(grad.size());
    for (size_t r = 0; r < grad.size(); ++r)
    {
        impl_->host_gh[r] = {grad[r], hess[r]};
    }
    impl_->gh.upload(impl_->host_gh.data(), impl_->host_gh.size());
}

void CudaHistogramBuilder::populate(Dataset const &ds, floats_view grad,
                                    floats_view hess, SplitInput &node,
                                    std::span<feature_id_t const> selected)
{
    std::array const one = {std::ref(node)};
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

void CudaHistogramBuilder::advance_level(Dataset const &ds,
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

    // One concatenated row upload; nodes route to the shared-memory kernel
    // above the row cutoff and to the direct-global kernel below it.
    size_t total_rows = 0;
    size_t max_rows   = 0;
    im.host_ofs.clear();
    im.host_cnt.clear();
    im.host_slots.clear();
    im.host_sofs.clear();
    im.host_scnt.clear();
    im.host_sslot.clear();
    im.host_triples.clear();
    std::vector<uint32_t> all_ofs;
    all_ofs.reserve(ops.size());
    for (LevelOp const &op : ops)
    {
        auto const ofs = static_cast<uint32_t>(total_rows);
        auto const cnt = static_cast<uint32_t>(op.small_rows.size());
        all_ofs.push_back(ofs);
        if (cnt >= k_min_gpu_rows)
        {
            im.host_ofs.push_back(ofs);
            im.host_cnt.push_back(cnt);
            im.host_slots.push_back(op.small_slot);
            max_rows = std::max(max_rows, op.small_rows.size());
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
        total_rows += op.small_rows.size();
    }
    im.host_rows.resize(total_rows);
    for (size_t k = 0; k < ops.size(); ++k)
    {
        std::ranges::copy(ops[k].small_rows, im.host_rows.begin() + all_ofs[k]);
    }
    im.rows.upload(im.host_rows.data(), std::max<size_t>(total_rows, 1));
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
    im.gh_ordered.ensure(std::max<size_t>(total_rows, 1));
    if (total_rows > 0)
    {
        gather_gh_kernel<<<dim3(std::clamp<uint32_t>(
                               static_cast<uint32_t>(total_rows / 256), 1, 512)),
                           dim3(256)>>>(im.gh.get(), im.rows.get(),
                                        static_cast<uint32_t>(total_rows),
                                        im.gh_ordered.get());
        check(cudaGetLastError(), "gather launch");
        auto const launch = [&](auto const *bins)
        {
            if (!im.host_ofs.empty())
            {
                auto const n_chunks = std::clamp<uint32_t>(
                    (static_cast<uint32_t>(max_rows) + 32767) / 32768, 1, 64);
                dim3 const grid(im.n_sel,
                                static_cast<uint32_t>(im.host_ofs.size()), n_chunks);
                hist_kernel<<<grid, dim3(256), 2UL * im.stride * sizeof(float)>>>(
                    bins, im.gh_ordered.get(), im.rows.get(), im.row_ofs.get(),
                    im.row_cnt.get(), im.features.get(), im.n_bins.get(),
                    static_cast<uint32_t>(ds.n_rows()), im.n_sel, im.other().get(),
                    im.stride, im.slots.get());
            }
            if (!im.host_sofs.empty())
            {
                hist_small_kernel<<<dim3(static_cast<uint32_t>(im.host_sofs.size())),
                                    dim3(128)>>>(
                    bins, im.gh_ordered.get(), im.rows.get(), im.sofs.get(),
                    im.scnt.get(), im.features.get(),
                    static_cast<uint32_t>(ds.n_rows()), im.n_sel, im.other().get(),
                    im.stride, im.sslot.get());
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
    }
    auto const sd = static_cast<uint32_t>(im.slot_doubles());
    subtract_kernel<<<dim3(std::clamp<uint32_t>((sd + 255) / 256, 1, 256),
                           static_cast<uint32_t>(ops.size())),
                      dim3(256)>>>(im.cur().get(), im.other().get(),
                                   im.triples.get(), sd);
    check(cudaGetLastError(), "subtract launch");
    im.cur_is_a = !im.cur_is_a;
    if (prof.enabled)
    {
        ++prof.launches;
        prof.gpu_nodes += child_slots;
    }
    lap(prof.gpu_s);
}

void CudaHistogramBuilder::find_splits_many(Dataset const &ds,
                                            TreeConfig const &config,
                                            std::span<SplitInput const> level,
                                            std::span<SplitOutput>      out,
                                            std::span<HistCell>         child_sums)
{
    Impl        &im = *impl_;
    size_t const n  = level.size();
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
        im.host_nsums[2 * i]       = level[i].sums.sum_grad;
        im.host_nsums[(2 * i) + 1] = level[i].sums.sum_hess;
        im.host_bounds[2 * i]      = level[i].lo;
        im.host_bounds[(2 * i) + 1] = level[i].hi;
        any_mask = any_mask || !level[i].allowed.empty();
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
                    level[i].allowed.empty()
                        ? char{1}
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
    find_kernel<<<dim3(im.n_sel, static_cast<uint32_t>(n)), dim3(32)>>>(
        im.cur().get(), im.features.get(), im.n_bins.get(), im.node_sums.get(),
        im.node_bounds.get(), any_mask ? im.allowed.get() : nullptr,
        im.monotone.get(), im.n_sel, im.stride, config.lambda_l1, config.lambda_l2,
        config.min_child_hess, config.min_gain_to_split, im.feat_best.get());
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
        FeatBest const &b        = im.host_best[i];
        bool const      eligible = level[i].row_count >=
                              2 * static_cast<size_t>(config.min_data_in_leaf);
        if (b.valid == 0 || !eligible)
        {
            out[i]                 = {};
            child_sums[2 * i]      = {};
            child_sums[(2 * i) + 1] = {};
            continue;
        }
        out[i] = {.gain         = b.gain,
                  .feature_id   = static_cast<feature_id_t>(
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
