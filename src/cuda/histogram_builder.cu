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
                            double *out, uint32_t stride)
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
    double *o = out + (((static_cast<size_t>(node) * n_sel) + blockIdx.x) * stride);
    for (uint32_t i = threadIdx.x; i < 2 * nb; i += blockDim.x)
    {
        atomicAdd(&o[i],
                  static_cast<double>(sh[i]) + static_cast<double>(sh[(2 * nb) + i]));
    }
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
            impl_->out.get(), stride);
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

} // namespace bonsai
