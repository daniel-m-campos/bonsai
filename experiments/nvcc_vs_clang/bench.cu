// Isolated kernel microbenchmark: times bonsai's two FP64-bound device
// kernels (hist_kernel's double-atomic cross-chunk merge, find_kernel's
// one-lane double prefix scan) so the SAME source can be compiled by clang
// (-x cuda) and nvcc and compared. See README.md.
//
// The kernels are the REAL ones, #included verbatim from src/cuda/detail;
// shim/bonsai/split.hpp supplies just the constexpr gain math they call.
// Representative MSD-level dimensions (464k rows, 90 features, 256 bins).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include <cuda.h>

#include "../../src/cuda/detail/device_buffer.cuh"
#include "../../src/cuda/detail/kernels.cuh"

namespace
{

constexpr uint32_t N_ROWS  = 463715;
constexpr uint32_t N_FEAT  = 90;
constexpr uint32_t NB      = 256;     // bins including the missing bin
constexpr uint32_t STRIDE  = 2 * NB;  // doubles per (node, feature) slot
constexpr uint32_t N_NODES = 32;      // frontier nodes at a mid tree level
constexpr int      WARMUP  = 5;
constexpr int      ITERS   = 50;

void ck(cudaError_t rc, char const *what)
{
    if (rc != cudaSuccess)
    {
        std::fprintf(stderr, "cuda %s: %s\n", what, cudaGetErrorString(rc));
        std::exit(1);
    }
}

template <typename T> T *dalloc(size_t n)
{
    T *p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    return p;
}

template <typename T> T *dcopy(std::vector<T> const &h)
{
    T *p = dalloc<T>(h.size());
    ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "H2D");
    return p;
}

// Mean milliseconds per launch over ITERS, after WARMUP.
template <typename Launch> double bench(Launch launch)
{
    cudaEvent_t a = nullptr;
    cudaEvent_t b = nullptr;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < WARMUP; ++i)
    {
        launch();
    }
    ck(cudaDeviceSynchronize(), "warmup sync");
    cudaEventRecord(a);
    for (int i = 0; i < ITERS; ++i)
    {
        launch();
    }
    cudaEventRecord(b);
    ck(cudaEventSynchronize(b), "bench sync");
    float ms = 0.0F;
    cudaEventElapsedTime(&ms, a, b);
    ck(cudaGetLastError(), "launch");
    return static_cast<double>(ms) / ITERS;
}

} // namespace

int main()
{
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    std::printf("device: %s (sm_%d%d), CUDA runtime %d\n", prop.name, prop.major,
                prop.minor, CUDART_VERSION);

    uint32_t const rows_per_node = N_ROWS / N_NODES;

    // --- hist_kernel inputs (uint8 bins: the max_bin<=256 MSD path) ---
    std::vector<uint8_t> h_bins(static_cast<size_t>(N_FEAT) * N_ROWS);
    for (size_t i = 0; i < h_bins.size(); ++i)
    {
        h_bins[i] = static_cast<uint8_t>((i * 131 + 7) % NB);
    }
    std::vector<float2> h_gh(N_ROWS);
    for (uint32_t i = 0; i < N_ROWS; ++i)
    {
        h_gh[i] = make_float2(static_cast<float>((i % 7)) * 0.1F - 0.3F,
                              static_cast<float>((i % 5)) * 0.01F + 0.01F);
    }
    std::vector<uint32_t> h_rows(N_ROWS);
    for (uint32_t i = 0; i < N_ROWS; ++i)
    {
        h_rows[i] = i;
    }
    std::vector<uint32_t> h_ofs(N_NODES);
    std::vector<uint32_t> h_cnt(N_NODES);
    for (uint32_t n = 0; n < N_NODES; ++n)
    {
        h_ofs[n] = n * rows_per_node;
        h_cnt[n] = (n + 1 == N_NODES) ? (N_ROWS - h_ofs[n]) : rows_per_node;
    }
    std::vector<uint32_t> h_feat(N_FEAT);
    for (uint32_t f = 0; f < N_FEAT; ++f)
    {
        h_feat[f] = f;
    }
    std::vector<uint32_t> h_nbins(N_FEAT, NB);

    auto *d_bins  = dcopy(h_bins);
    auto *d_gh    = dcopy(h_gh);
    auto *d_rows  = dcopy(h_rows);
    auto *d_ofs   = dcopy(h_ofs);
    auto *d_cnt   = dcopy(h_cnt);
    auto *d_feat  = dcopy(h_feat);
    auto *d_nbins = dcopy(h_nbins);
    auto *d_out   = dalloc<double>(static_cast<size_t>(N_NODES) * N_FEAT * STRIDE);
    ck(cudaMemset(d_out, 0, static_cast<size_t>(N_NODES) * N_FEAT * STRIDE * sizeof(double)),
       "zero out");

    dim3 const   hist_grid(N_FEAT, N_NODES, 1);
    size_t const shmem = 2UL * STRIDE * sizeof(float);
    double const hist_ms = bench(
        [&]
        {
            bonsai::hist_kernel<uint8_t><<<hist_grid, 256, shmem>>>(
                d_bins, d_gh, d_rows, d_ofs, d_cnt, d_feat, d_nbins, N_ROWS, N_FEAT,
                d_out, STRIDE, nullptr);
        });

    // --- find_kernel inputs (device-resident level histograms) ---
    std::vector<double> h_hists(static_cast<size_t>(N_NODES) * N_FEAT * STRIDE);
    for (size_t i = 0; i < h_hists.size(); ++i)
    {
        h_hists[i] = static_cast<double>((i * 2654435761U) % 1000) * 0.001 + 0.001;
    }
    std::vector<double> h_nsums(2 * N_NODES);
    for (uint32_t n = 0; n < N_NODES; ++n)
    {
        h_nsums[2 * n]     = 10.0 + n;   // grad
        h_nsums[2 * n + 1] = 100.0 + n;  // hess
    }
    std::vector<double> h_nbounds(2 * N_NODES);
    for (uint32_t n = 0; n < N_NODES; ++n)
    {
        h_nbounds[2 * n]     = -std::numeric_limits<double>::infinity();
        h_nbounds[2 * n + 1] = std::numeric_limits<double>::infinity();
    }
    std::vector<int> h_mono(N_FEAT, 0);

    auto *d_hists   = dcopy(h_hists);
    auto *d_nsums   = dcopy(h_nsums);
    auto *d_nbounds = dcopy(h_nbounds);
    auto *d_mono    = dcopy(h_mono);
    auto *d_fbest   = dalloc<bonsai::FeatBest>(static_cast<size_t>(N_NODES) * N_FEAT);

    dim3 const   find_grid((N_FEAT + 31) / 32, N_NODES, 1);
    double const find_ms = bench(
        [&]
        {
            bonsai::find_kernel<<<find_grid, 32>>>(d_hists, d_feat, d_nbins, d_nsums,
                                                   d_nbounds, nullptr, d_mono, N_FEAT,
                                                   STRIDE, 0.0, 1.0, 1e-3, 0.0, d_fbest);
        });

    std::printf("hist_kernel  %8.4f ms/launch   (grid %ux%ux1, %u-bin double merge)\n",
                hist_ms, N_FEAT, N_NODES, NB);
    std::printf("find_kernel  %8.4f ms/launch   (grid %ux%u, one-lane double scan)\n",
                find_ms, (N_FEAT + 31) / 32, N_NODES);
    return 0;
}
