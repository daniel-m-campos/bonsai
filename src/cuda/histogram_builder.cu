// CUDA histogram backend, compiled as CUDA C++ by the same clang/libc++
// toolchain as the rest of bonsai (no nvcc, no second standard library) —
// so it uses bonsai types directly. One block per (selected feature,
// row-chunk): threads stride the chunk, accumulate (grad, hess) into a
// shared-memory float histogram with native atomics, then merge into the
// per-feature double output with global atomics (out is pre-zeroed).
// Float-per-chunk keeps rounding bounded per <= 32k-row chunk while the
// cross-chunk merge stays double.

#include "bonsai/cuda/histogram_builder.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/types.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace bonsai
{

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
        check(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice),
              "upload");
    }

  private:
    T     *ptr_      = nullptr;
    size_t capacity_ = 0;
};

__global__ void hist_kernel(uint16_t const *bins, float const *grad,
                            float const *hess, uint32_t const *rows,
                            uint32_t n_node_rows, uint32_t const *features,
                            uint32_t const *n_bins, uint32_t n_rows, double *out,
                            uint32_t stride)
{
    extern __shared__ float sh[];
    uint32_t const f  = features[blockIdx.x];
    uint32_t const nb = n_bins[f];
    for (uint32_t i = threadIdx.x; i < 2 * nb; i += blockDim.x)
    {
        sh[i] = 0.0F;
    }
    __syncthreads();
    uint16_t const *fb   = bins + (static_cast<size_t>(f) * n_rows);
    uint32_t const  span = gridDim.y * blockDim.x;
    for (uint32_t k = (blockIdx.y * blockDim.x) + threadIdx.x; k < n_node_rows;
         k += span)
    {
        uint32_t const r = rows[k];
        uint32_t const b = fb[r];
        atomicAdd(&sh[2 * b], grad[r]);
        atomicAdd(&sh[(2 * b) + 1], hess[r]);
    }
    __syncthreads();
    double *o = out + (static_cast<size_t>(blockIdx.x) * stride);
    for (uint32_t i = threadIdx.x; i < 2 * nb; i += blockDim.x)
    {
        atomicAdd(&o[i], static_cast<double>(sh[i]));
    }
}

} // namespace

struct CudaHistogramBuilder::Impl
{
    DeviceBuffer<uint16_t> bins;     // n_features * n_rows, feature-major
    DeviceBuffer<uint32_t> n_bins;   // per-feature bin counts
    DeviceBuffer<float>    grad;
    DeviceBuffer<float>    hess;
    DeviceBuffer<uint32_t> rows;
    DeviceBuffer<uint32_t> features;
    DeviceBuffer<double>   out;
    CpuHistogramBuilder    cpu;

    // Identity of the uploaded dataset. The booster reuses one live Dataset
    // for a whole fit, so a mismatch on any field just re-uploads.
    Dataset const *ds      = nullptr;
    void const    *bins0   = nullptr;
    size_t         n_rows  = 0;
    size_t         n_feats = 0;

    // Host staging reused across nodes.
    std::vector<double>   host_out;
    std::vector<uint32_t> host_features;

    void ensure_dataset(Dataset const &dataset)
    {
        void const *first =
            dataset.n_features() > 0
                ? static_cast<void const *>(dataset.feature_bins(0).data())
                : nullptr;
        bool const same = ds == &dataset && bins0 == first &&
                          n_rows == dataset.n_rows() &&
                          n_feats == dataset.n_features();
        if (same)
        {
            return;
        }
        bins.ensure(dataset.n_features() * dataset.n_rows());
        std::vector<uint32_t> counts(dataset.n_features());
        for (size_t f = 0; f < dataset.n_features(); ++f)
        {
            check(cudaMemcpy(bins.get() + (f * dataset.n_rows()),
                             dataset.feature_bins(f).data(),
                             dataset.n_rows() * sizeof(uint16_t),
                             cudaMemcpyHostToDevice),
                  "upload bins");
            counts[f] = static_cast<uint32_t>(dataset.n_bins(f));
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
    impl_->grad.upload(grad.data(), grad.size());
    impl_->hess.upload(hess.data(), hess.size());
}

void CudaHistogramBuilder::populate(Dataset const &ds, floats_view grad,
                                    floats_view hess, SplitInput &node,
                                    std::span<feature_id_t const> selected)
{
    size_t max_sel_bins = 0;
    for (feature_id_t const fid : selected)
    {
        max_sel_bins = std::max(max_sel_bins, ds.n_bins(fid));
    }
    if (node.rows.size() < k_min_gpu_rows ||
        2 * max_sel_bins * sizeof(float) > k_max_shared_bytes)
    {
        impl_->cpu.populate(ds, grad, hess, node, selected);
        return;
    }
    // Same shape contract as the CPU builder: unselected features get
    // zero-binned placeholder histograms the split finders skip.
    node.hists.reserve(ds.n_features());
    size_t j = 0;
    for (feature_id_t fid = 0; fid < ds.n_features(); ++fid)
    {
        bool const sel = j < selected.size() && selected[j] == fid;
        node.hists.emplace_back(sel ? ds.n_bins(fid) : 0);
        j += sel ? 1 : 0;
    }
    if (selected.empty())
    {
        return;
    }

    auto const   stride      = static_cast<uint32_t>(2 * max_sel_bins);
    size_t const out_doubles = static_cast<size_t>(stride) * selected.size();

    impl_->host_features.assign(selected.begin(), selected.end());
    impl_->rows.upload(node.rows.data(), node.rows.size());
    impl_->features.upload(impl_->host_features.data(), selected.size());
    impl_->out.ensure(out_doubles);
    check(cudaMemset(impl_->out.get(), 0, out_doubles * sizeof(double)),
          "zero out");

    // Enough row chunks to occupy the device when few features are
    // selected; capped so tiny nodes don't spawn empty blocks.
    uint32_t const chunk = 32768;
    auto const     n_chunks = std::clamp<uint32_t>(
        (static_cast<uint32_t>(node.rows.size()) + chunk - 1) / chunk, 1, 64);
    dim3 const grid(static_cast<uint32_t>(selected.size()), n_chunks);
    dim3 const block(256);
    hist_kernel<<<grid, block, stride * sizeof(float)>>>(
        impl_->bins.get(), impl_->grad.get(), impl_->hess.get(),
        impl_->rows.get(), static_cast<uint32_t>(node.rows.size()),
        impl_->features.get(), impl_->n_bins.get(),
        static_cast<uint32_t>(ds.n_rows()), impl_->out.get(), stride);
    check(cudaGetLastError(), "launch");

    impl_->host_out.resize(out_doubles);
    check(cudaMemcpy(impl_->host_out.data(), impl_->out.get(),
                     out_doubles * sizeof(double),
                     cudaMemcpyDeviceToHost), // implicit sync
          "copy back");

    for (size_t s = 0; s < selected.size(); ++s)
    {
        Histogram    &h     = node.hists[selected[s]];
        double const *cells = impl_->host_out.data() + (s * stride);
        for (size_t b = 0; b < h.size(); ++b)
        {
            h.add(static_cast<bin_id_t>(b), cells[2 * b], cells[(2 * b) + 1]);
        }
    }
}

} // namespace bonsai
